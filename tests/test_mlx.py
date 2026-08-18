"""Tests for the MLX runner dialect.

Ollama serves GGUF models through llama-server and `-mlx` models through its own
MLX runner, which logs nothing the llama.cpp patterns match: no slots, no
`tokens per second`, and a request's timings split across lines that share no
id. Before this was handled an MLX user saw a model name in the header and
`0 req` forever, with no error to explain it.

The fixture is real, sanitised output from Ollama 0.32.9 with gemma4:26b-mlx on
macOS. Like the llama-server fixtures it is an early-warning system for format
drift, and it deliberately contains both orderings of the generation-stats race
described in TestGenerationStatsRace.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    MlxGenStats, MlxPrefillTick, MlxRequestEnd, MlxRequestStart, MlxRunnerReady,
    MlxRunnerStart, Tracker, parse_line, parse_mlx_timestamp,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def load(name):
    with open(os.path.join(FIXTURES, name), "r", errors="replace") as fh:
        return fh.readlines()


def outputs(name):
    tracker = Tracker()
    out = []
    for line in load(name):
        out.extend(tracker.feed(parse_line(line)))
    return out


def events_named(outs, name):
    return [o.data for o in outs if o.data.get("event") == name]


class TestMlxLineParsing(unittest.TestCase):

    def test_cache_miss_starts_a_request(self):
        ev = parse_line(
            'time=2026-08-18T13:17:06.834+08:00 level=INFO source=prefix_cache.go:124 '
            'msg="cache miss" total=337 matched=0 cached=0 left=337')
        self.assertIsInstance(ev, MlxRequestStart)
        self.assertEqual((ev.prompt_tokens, ev.cached, ev.miss), (337, 0, True))

    def test_cache_hit_reports_what_was_reused(self):
        ev = parse_line(
            'time=2026-08-18T13:19:06.388+08:00 level=INFO source=prefix_cache.go:124 '
            'msg="cache hit" total=633 matched=68 cached=68 left=565')
        self.assertIsInstance(ev, MlxRequestStart)
        self.assertEqual((ev.prompt_tokens, ev.cached, ev.miss), (633, 68, False))

    def test_prefill_tick(self):
        ev = parse_line(
            'time=2026-08-18T13:17:07.785+08:00 level=INFO source=pipeline.go:200 '
            'msg="Prompt processing progress" processed=336 total=337')
        self.assertIsInstance(ev, MlxPrefillTick)
        self.assertEqual((ev.processed, ev.total), (336, 337))

    def test_speculative_decode_stats(self):
        ev = parse_line(
            'time=2026-08-18T13:17:14.130+08:00 level=INFO source=speculate_stats.go:62 '
            'msg="speculative decode stats" iterations=117 drafted=115 accepted=84 '
            'acceptance=0.73 avg_draft=0.98 max_draft=4 avg_accepted=0.72')
        self.assertIsInstance(ev, MlxGenStats)
        self.assertEqual((ev.iterations, ev.drafted, ev.accepted), (117, 115, 84))
        self.assertAlmostEqual(ev.acceptance, 0.73)

    def test_runner_lifecycle(self):
        start = parse_line(
            'time=2026-08-18T13:16:56.626+08:00 level=INFO source=client.go:391 '
            'msg="starting mlx runner subprocess" model=gemma4:26b-mlx port=56223')
        self.assertIsInstance(start, MlxRunnerStart)
        self.assertEqual(start.model, "gemma4:26b-mlx")
        self.assertIsInstance(
            parse_line('time=2026-08-18T13:17:06.831+08:00 level=INFO source=client.go:106 '
                       'msg="mlx runner is ready" port=56223'),
            MlxRunnerReady)


class TestRequestEndDurations(unittest.TestCase):
    """Go prints durations in whichever unit keeps them readable, so all of
    them have to be understood -- reading `166.792µs` as seconds would put a
    two-minute request on the board."""

    def _took(self, value):
        ev = parse_line(
            'time=2026-08-18T13:17:14.131+08:00 level=INFO source=server.go:225 '
            'msg=ServeHTTP method=POST path=/v1/completions took=%s status="200 OK"' % value)
        self.assertIsInstance(ev, MlxRequestEnd)
        return ev.seconds

    def test_seconds(self):
        self.assertAlmostEqual(self._took("7.298789917s"), 7.298789917)

    def test_milliseconds(self):
        self.assertAlmostEqual(self._took("50.717208ms"), 0.050717208)

    def test_microseconds(self):
        self.assertAlmostEqual(self._took("166.792µs"), 0.000166792)

    def test_compound_durations(self):
        """Go concatenates units past a minute. Reading `1m30.5s` as one minute
        would drop half a minute off exactly the slow request someone is
        watching this tool to understand."""
        self.assertAlmostEqual(self._took("1m30.5s"), 90.5)
        self.assertAlmostEqual(self._took("2h30m0s"), 9000.0)

    def test_nanoseconds(self):
        self.assertAlmostEqual(self._took("847ns"), 847e-9)

    def test_status_requests_are_not_requests(self):
        """Only POST completions is a model request. The status polling on the
        same handler runs every 15s and would otherwise end the real one."""
        self.assertIsNone(parse_line(
            'time=2026-08-18T13:17:06.831+08:00 level=INFO source=server.go:225 '
            'msg=ServeHTTP method=GET path=/v1/status took=166.792µs status="200 OK"'))

    def test_outer_gin_line_is_ignored(self):
        """The `[GIN] POST /api/chat` line includes model load time. Counting it
        as the request would blame a 10s weight load on the prompt."""
        self.assertIsNone(parse_line(
            '[GIN] 2026/08/18 - 13:17:14 | 200 | 17.630247208s |       '
            '127.0.0.1 | POST     "/api/chat"'))


class TestTimestamps(unittest.TestCase):
    """MLX prints no rate and no elapsed, so every duration is reconstructed
    from the gap between two log timestamps."""

    def test_difference_across_offset(self):
        a = parse_mlx_timestamp("time=2026-08-18T13:17:06.834+08:00 msg=x")
        b = parse_mlx_timestamp("time=2026-08-18T13:17:07.785+08:00 msg=x")
        self.assertAlmostEqual(b - a, 0.951, places=3)

    def test_utc_and_offset_agree(self):
        local = parse_mlx_timestamp("time=2026-08-18T13:00:00.000+08:00 msg=x")
        utc = parse_mlx_timestamp("time=2026-08-18T05:00:00.000Z msg=x")
        self.assertAlmostEqual(local, utc)

    def test_unparseable_is_none_not_an_exception(self):
        self.assertIsNone(parse_mlx_timestamp("no timestamp here"))
        self.assertIsNone(parse_mlx_timestamp("time=2026-13-99T99:99:99 msg=x"))


class TestTrackedRequests(unittest.TestCase):

    def setUp(self):
        self.outs = outputs("mlx-request.log")

    def test_both_requests_are_seen(self):
        starts = events_named(self.outs, "request_start")
        self.assertEqual([s["prompt_tokens"] for s in starts], [337, 633])

    def test_model_name_survives_the_registry_prefix(self):
        self.assertEqual(events_named(self.outs, "request_start")[0]["model"],
                         "gemma4:26b-mlx")

    def test_weight_load_is_timed(self):
        started = events_named(self.outs, "server_started")
        self.assertEqual(len(started), 1)
        self.assertAlmostEqual(started[0]["seconds"], 10.205, places=2)

    def test_prefill_rate_is_measured_not_guessed(self):
        prefill = events_named(self.outs, "prefill_done")[0]
        self.assertEqual(prefill["tokens"], 336)
        self.assertAlmostEqual(prefill["seconds"], 0.951, places=2)
        self.assertAlmostEqual(prefill["rate"], 336 / 0.951, places=1)

    def test_cache_reuse_is_reported(self):
        self.assertTrue(events_named(self.outs, "cache_miss"))
        prefill = events_named(self.outs, "prefill_done")[1]
        self.assertEqual(prefill["cached"], 68)

    def test_generation_tokens_are_reconstructed_from_speculation(self):
        """MLX never prints a token count. Each iteration commits one token from
        the target model plus the drafted tokens accepted that round."""
        generated = events_named(self.outs, "generate_done")
        self.assertEqual([g["tokens"] for g in generated], [117 + 84, 189 + 341])

    def test_request_totals(self):
        ends = events_named(self.outs, "request_end")
        self.assertAlmostEqual(ends[0]["seconds"], 7.298789917)
        self.assertAlmostEqual(ends[1]["seconds"], 24.143523833)


CACHE_MISS = ('time=2026-08-18T13:17:06.834+08:00 source=prefix_cache.go:124 '
              'msg="cache miss" total=337 matched=0 cached=0 left=337')
PREFILL = ('time=2026-08-18T13:17:07.785+08:00 source=pipeline.go:200 '
           'msg="Prompt processing progress" processed=336 total=337')
COMPLETED = ('time=2026-08-18T13:17:14.131+08:00 source=server.go:225 msg=ServeHTTP '
             'method=POST path=/v1/completions took=7.298789917s status="200 OK"')
STATS = ('time=2026-08-18T13:17:14.130+08:00 source=speculate_stats.go:62 '
         'msg="speculative decode stats" iterations=117 drafted=115 accepted=84 '
         'acceptance=0.73 avg_draft=0.98')
PEAK = ('time=2026-08-18T13:17:14.383+08:00 source=pipeline.go:91 '
        'msg="peak memory" size="16.37 GiB"')


def feed_lines(lines):
    tracker = Tracker()
    out = []
    for line in lines:
        out.extend(tracker.feed(parse_line(line)))
    return out


class TestGenerationStatsRace(unittest.TestCase):
    """The runner writes its stats line and its ServeHTTP line from different
    places, so either can reach the log first, and both orders appear in the
    fixture. Closing a request on the ServeHTTP line would therefore drop the
    generation rate from every request that happened to lose the race.

    `peak memory` is what makes this tractable: across a full session it appears
    exactly once per request and always after both, so it is the one line whose
    position can be relied on.
    """

    def test_no_phantom_abandoned_requests(self):
        self.assertEqual(events_named(outputs("mlx-request.log"),
                                      "request_abandoned"), [])

    def test_generation_recorded_in_either_order(self):
        """Request 1's stats arrive before its completion line, request 2's
        after. Both report a rate, and both report it in time to be summarised
        with the request rather than stranded after it."""
        outs = outputs("mlx-request.log")
        self.assertEqual(len(events_named(outs, "generate_done")), 2)
        order = [o.data["event"] for o in outs
                 if o.data.get("event") in ("generate_done", "request_end")]
        self.assertEqual(order, ["generate_done", "request_end"] * 2)

    def test_stats_after_completion_still_reach_the_summary(self):
        outs = feed_lines([CACHE_MISS, PREFILL, COMPLETED, STATS, PEAK])
        self.assertEqual(events_named(outs, "generate_done")[0]["tokens"], 117 + 84)

    def test_stats_before_completion_still_reach_the_summary(self):
        outs = feed_lines([CACHE_MISS, PREFILL, STATS, COMPLETED, PEAK])
        self.assertEqual(events_named(outs, "generate_done")[0]["tokens"], 117 + 84)

    def test_the_request_stays_open_until_the_closing_line(self):
        """Nothing is summarised early: until `peak memory` lands, a stats line
        can still arrive and change what the summary says."""
        self.assertEqual(events_named(feed_lines([CACHE_MISS, PREFILL, COMPLETED]),
                                      "request_end"), [])

    def test_a_missing_closing_line_cannot_strand_a_request(self):
        """If a build ever stops printing it, the next request still closes the
        previous one rather than leaving it on the board forever."""
        outs = feed_lines([CACHE_MISS, PREFILL, STATS, COMPLETED, CACHE_MISS])
        self.assertEqual(len(events_named(outs, "request_end")), 1)
        self.assertEqual(events_named(outs, "request_abandoned"), [])


class TestEnginesCoexist(unittest.TestCase):
    """Which engine runs is a property of the model, not the install, so one log
    can hold both across a restart. Nothing selects between them."""

    def test_llama_cpp_lines_still_parse(self):
        ev = parse_line(
            "slot   operator(): id  0 | task 49 | new prompt, n_ctx_slot = 131072, "
            "n_keep = 4, task.n_tokens = 14518")
        self.assertEqual(ev.prompt_tokens, 14518)

    def test_both_dialects_through_one_tracker(self):
        tracker = Tracker()
        out = []
        for line in load("mlx-request.log") + load("request-with-progress.log"):
            out.extend(tracker.feed(parse_line(line)))
        self.assertGreaterEqual(len(events_named(out, "request_end")), 3)


if __name__ == "__main__":
    unittest.main()
