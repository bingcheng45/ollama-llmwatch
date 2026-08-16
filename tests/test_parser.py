"""Tests for llmwatch's parser and tracker.

Fixtures are real, sanitised llama-server output captured from Ollama 0.32.13 on
macOS. They exist because the log format is an internal detail with no stability
guarantee -- these tests are the early-warning system for format drift.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    CacheInfo, GenDone, GenTick, PrefillDone, PrefillTick, RequestEnd, RequestStart,
    Tracker, fmt_bar, fmt_duration, parse_line,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def load(name):
    with open(os.path.join(FIXTURES, name), "r", errors="replace") as fh:
        return fh.readlines()


def events(name):
    return [ev for ev in (parse_line(l) for l in load(name)) if ev is not None]


def outputs(name):
    tracker = Tracker()
    out = []
    for line in load(name):
        out.extend(tracker.feed(parse_line(line)))
    return out


def data_of(outs, event_name):
    for out in outs:
        if out.data.get("event") == event_name:
            return out.data
    return None


class TestLineParsing(unittest.TestCase):

    def test_request_start(self):
        ev = parse_line(
            "slot   operator(): id  0 | task 49 | new prompt, n_ctx_slot = 131072, "
            "n_keep = 4, task.n_tokens = 14518")
        self.assertIsInstance(ev, RequestStart)
        self.assertEqual((ev.slot, ev.task, ev.prompt_tokens, ev.ctx), (0, 49, 14518, 131072))

    def test_prefill_tick(self):
        ev = parse_line(
            "slot print_timing: id  0 | task 49 | prompt processing, n_tokens =   1536, "
            "progress = 0.11, t =  13.38 s / 114.81 tokens per second")
        self.assertIsInstance(ev, PrefillTick)
        self.assertEqual(ev.processed, 1536)
        self.assertAlmostEqual(ev.progress, 0.11)
        self.assertAlmostEqual(ev.elapsed, 13.38)
        self.assertAlmostEqual(ev.rate, 114.81)

    def test_prompt_eval_is_prefill_not_generation(self):
        """The single most dangerous ambiguity: 'prompt eval time' also matches
        the generation pattern. Misclassifying it reports prefill's ~100 tok/s
        as the generation rate."""
        ev = parse_line(
            "slot print_timing: id  0 | task 49 | prompt eval time =  143870.98 ms / "
            "14518 tokens (    9.91 ms per token,   100.91 tokens per second)")
        self.assertIsInstance(ev, PrefillDone)
        self.assertEqual(ev.tokens, 14518)
        self.assertAlmostEqual(ev.rate, 100.91)

    def test_generation_done(self):
        ev = parse_line(
            "slot print_timing: id  0 | task 49 |        eval time =   18764.56 ms /   140 "
            "tokens (  134.03 ms per token,     7.46 tokens per second)")
        self.assertIsInstance(ev, GenDone)
        self.assertEqual(ev.tokens, 140)
        self.assertAlmostEqual(ev.rate, 7.46)

    def test_gen_tick_and_cache_and_total(self):
        ev = parse_line("slot print_timing: id  0 | task 49 | n_decoded =    104, "
                        "tg =   7.39 t/s, tg_3s =   7.39 t/s")
        self.assertIsInstance(ev, GenTick)
        self.assertEqual((ev.decoded, ev.rate, ev.rate_3s), (104, 7.39, 7.39))

        ev = parse_line("slot   operator(): id  0 | task 1040 | cached n_tokens = 32112, "
                        "memory_seq_rm [32112, end)")
        self.assertIsInstance(ev, CacheInfo)
        self.assertEqual(ev.cached, 32112)

        ev = parse_line("slot print_timing: id  0 | task 49 |       total time =  "
                        "162635.53 ms / 14658 tokens")
        self.assertIsInstance(ev, RequestEnd)
        self.assertAlmostEqual(ev.ms, 162635.53)

    def test_noise_returns_none(self):
        for line in ["", "some unrelated log line\n",
                     "srv  update_slots: all slots are idle\n",
                     'time=2026-08-17T00:47:28+08:00 level=INFO msg="something"\n']:
            self.assertIsNone(parse_line(line), line)


class TestNormalRequest(unittest.TestCase):
    """request-with-progress.log: a 14,518-token prompt, no cache reuse."""

    def test_phases_reported_with_correct_totals(self):
        outs = outputs("request-with-progress.log")

        prefill = data_of(outs, "prefill_done")
        self.assertEqual(prefill["tokens"], 14518)
        self.assertAlmostEqual(prefill["rate"], 100.91)
        self.assertAlmostEqual(prefill["seconds"], 143.87098)

        gen = data_of(outs, "generate_done")
        self.assertEqual(gen["tokens"], 140)
        self.assertAlmostEqual(gen["rate"], 7.46)

        end = data_of(outs, "request_end")
        self.assertAlmostEqual(end["seconds"], 162.63553)
        # This is the number that justifies the tool existing.
        self.assertGreater(end["prefill_share_pct"], 85)

    def test_summaries_emitted_in_real_order(self):
        """llama-server prints its timing block after generation streams, so a
        naive implementation reports PREFILL after GENERATE."""
        order = [o.data["event"] for o in outputs("request-with-progress.log")
                 if o.data.get("event") in ("prefill_done", "generate_done", "request_end")]
        self.assertEqual(order, ["prefill_done", "generate_done", "request_end"])

    def test_progress_advances_monotonically(self):
        fractions = [o.data["fraction"] for o in outputs("request-with-progress.log")
                     if o.data.get("event") == "prefill_tick"]
        self.assertTrue(fractions)
        self.assertEqual(fractions, sorted(fractions))
        self.assertLessEqual(max(fractions), 1.0)


class TestCacheHit(unittest.TestCase):
    """cache-hit.log: 32,952-token prompt, 32,112 already cached.

    Only 840 tokens need computing. A tool that trusts llama.cpp's raw `progress`
    field reports 98% done on the first tick and an ETA of ~0.1s, when ~10s of
    work remains. This is precisely when an honest ETA matters most.
    """

    def test_only_uncached_tokens_are_counted(self):
        outs = outputs("cache-hit.log")
        ticks = [o.data for o in outs if o.data.get("event") == "prefill_tick"]
        self.assertTrue(ticks)
        first = ticks[0]
        self.assertEqual(first["cached"], 32112)
        self.assertEqual(first["to_process"], 840)      # 32952 - 32112
        self.assertEqual(first["processed"], 328)

    def test_fraction_ignores_the_misleading_raw_progress(self):
        first = [o.data for o in outputs("cache-hit.log")
                 if o.data.get("event") == "prefill_tick"][0]
        # Raw llama.cpp progress here is 0.98; the honest figure is 328/840.
        self.assertAlmostEqual(first["fraction"], 328 / 840.0, places=4)
        self.assertLess(first["fraction"], 0.5)

    def test_eta_is_realistic(self):
        """Actual prefill took 16.9s total; at 6.37s elapsed, ~10.5s remained.
        The naive ETA from raw progress would be ~0.1s."""
        first = [o.data for o in outputs("cache-hit.log")
                 if o.data.get("event") == "prefill_tick"][0]
        self.assertGreater(first["eta_seconds"], 5)
        self.assertLess(first["eta_seconds"], 15)

    def test_summary_reports_processed_and_cached(self):
        prefill = data_of(outputs("cache-hit.log"), "prefill_done")
        self.assertEqual(prefill["tokens"], 840)
        self.assertEqual(prefill["cached"], 32112)


class TestConcurrency(unittest.TestCase):
    """Two slots interleaved must not contaminate each other -- the failure that
    made concurrent benchmarking unreadable before task-id keying."""

    def test_interleaved_tasks_tracked_separately(self):
        lines = [
            "slot   operator(): id  0 | task 1 | new prompt, n_ctx_slot = 131072, "
            "n_keep = 4, task.n_tokens = 1000",
            "slot   operator(): id  1 | task 2 | new prompt, n_ctx_slot = 131072, "
            "n_keep = 4, task.n_tokens = 2000",
            "slot print_timing: id  0 | task 1 | prompt processing, n_tokens =   500, "
            "progress = 0.50, t =   5.00 s / 100.00 tokens per second",
            "slot print_timing: id  1 | task 2 | prompt processing, n_tokens =   500, "
            "progress = 0.25, t =   5.00 s / 100.00 tokens per second",
        ]
        tracker = Tracker()
        ticks = []
        for line in lines:
            for out in tracker.feed(parse_line(line)):
                if out.data.get("event") == "prefill_tick":
                    ticks.append(out.data)
        self.assertEqual(ticks[0]["to_process"], 1000)
        self.assertEqual(ticks[1]["to_process"], 2000)
        self.assertAlmostEqual(ticks[0]["fraction"], 0.5)
        self.assertAlmostEqual(ticks[1]["fraction"], 0.25)


class TestRobustness(unittest.TestCase):

    def test_out_of_order_events_do_not_raise(self):
        """Starting mid-stream is the normal case: llmwatch attaches to a log
        that is already running."""
        tracker = Tracker()
        for line in ["slot print_timing: id  0 | task 7 | n_decoded =  10, tg = 5.0 t/s, "
                     "tg_3s = 5.0 t/s",
                     "slot print_timing: id  0 | task 7 |        eval time = 1000.00 ms / "
                     "10 tokens (100.00 ms per token, 10.00 tokens per second)",
                     "slot print_timing: id  0 | task 7 |       total time = 1000.00 ms / "
                     "10 tokens"]:
            tracker.feed(parse_line(line))  # must not raise

    def test_model_load_fixture_parses(self):
        evs = events("model-load.log")
        self.assertTrue(evs, "model-load.log produced no events")

    def test_formatters(self):
        self.assertEqual(fmt_duration(5.25), "5.2s")
        self.assertEqual(fmt_duration(147), "2m27s")
        self.assertEqual(fmt_duration(3700), "1h01m")
        self.assertEqual(fmt_duration(-1), "?")
        self.assertEqual(fmt_bar(0.5, width=10), "[#####-----]")
        self.assertEqual(fmt_bar(2.0, width=4), "[####]")   # clamps
        self.assertEqual(fmt_bar(-1, width=4), "[----]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
