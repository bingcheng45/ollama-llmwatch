"""Tests for llmwatch's live rendering and between-tick projection.

The bug these guard against: llama-server writes a progress line only once per
512-token batch, which at typical rates is every 5-10 seconds. A display that
repaints only when a line arrives sits visibly frozen in between, and users
reasonably conclude the tool (or the model) has hung.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    SPARK_UNICODE, Stats, Style, Tracker, compose_frame, fmt_bar, parse_line, project,
    render_board, render_help, render_idle, render_live, render_recent, render_summary,
    sparkline, spinner_frame,
)

PLAIN = Style(color=False, unicode_ok=False, width=100)
FANCY = Style(color=True, unicode_ok=True, width=120)


def strip_ansi(text):
    out, i = [], 0
    while i < len(text):
        if text[i] == "\033":
            while i < len(text) and text[i] != "m":
                i += 1
            i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


class TestProjection(unittest.TestCase):

    def test_extrapolates_from_last_known_rate(self):
        self.assertEqual(project(100, 10.0, 5.0, None), 150)

    def test_never_overruns_the_total(self):
        """The bar must not claim work that hasn't happened, however stale the
        last tick is."""
        self.assertEqual(project(100, 10.0, 999.0, 120), 120)

    def test_no_movement_without_rate_or_time(self):
        self.assertEqual(project(100, 0.0, 5.0, None), 100)
        self.assertEqual(project(100, 10.0, 0.0, None), 100)
        self.assertEqual(project(100, 10.0, -3.0, None), 100)


class TestLiveLineMoves(unittest.TestCase):
    """The regression test for 'nothing happens for a long time'."""

    PREFILL = {"event": "prefill_tick", "task": 1, "model": "m", "processed": 1000,
               "to_process": 10000, "cached": 0, "fraction": 0.1, "rate": 100.0,
               "elapsed": 10.0, "eta_seconds": 90.0}

    def test_display_changes_between_log_ticks(self):
        early = strip_ansi(render_live(self.PREFILL, age=0.0, style=PLAIN, now=0))
        later = strip_ansi(render_live(self.PREFILL, age=5.0, style=PLAIN, now=0))
        self.assertNotEqual(early, later,
                            "live line must advance even with no new log data")

    def test_projected_tokens_and_percent_advance(self):
        later = strip_ansi(render_live(self.PREFILL, age=5.0, style=PLAIN, now=0))
        self.assertIn("1,500/10,000 tok", later)   # 1000 + 100 tok/s * 5s
        self.assertIn("15%", later)

    def test_eta_counts_down(self):
        mid = strip_ansi(render_live(self.PREFILL, age=30.0, style=PLAIN, now=0))
        self.assertIn("eta 1m00s", mid)            # 90 - 30

    def test_eta_never_goes_negative(self):
        """An ETA that overshoots must clamp at zero, not count backwards. Slow
        rate keeps this below 100% so it stays in the bar state."""
        slow = dict(self.PREFILL, rate=1.0, eta_seconds=5.0)
        line = strip_ansi(render_live(slow, age=50.0, style=PLAIN, now=0))
        self.assertIn("eta 0.0s", line)
        self.assertNotIn("-", line.split("eta")[1])

    def test_elapsed_advances(self):
        line = strip_ansi(render_live(self.PREFILL, age=20.0, style=PLAIN, now=0))
        self.assertIn("elapsed 30.0s", line)       # 10 measured + 20 since

    def test_spinner_animates_over_time(self):
        frames = {spinner_frame(t / 10.0, PLAIN) for t in range(0, 20)}
        self.assertGreater(len(frames), 1, "spinner must have more than one frame")

    def test_pre_first_tick_state_still_shows_a_clock(self):
        """Before the first 512-token batch completes -- or forever, if the prompt
        was cached -- there are no progress ticks. That state must still move."""
        data = {"event": "request_start", "task": 1, "model": "m",
                "prompt_tokens": 6637, "started": 0}
        early = strip_ansi(render_live(data, age=0.5, style=PLAIN, now=0))
        later = strip_ansi(render_live(data, age=12.0, style=PLAIN, now=0))
        self.assertIn("6,637 tok", early)
        self.assertIn("elapsed 12.0s", later)
        self.assertNotEqual(early, later)

    def test_generation_line_advances_too(self):
        data = {"event": "generate_tick", "task": 1, "model": "m", "decoded": 100,
                "rate": 10.0, "rate_3s": 12.0, "elapsed": 10.0}
        later = strip_ansi(render_live(data, age=3.0, style=PLAIN, now=0))
        self.assertIn("130 tok", later)            # 100 + 10 tok/s * 3s


class TestPresentation(unittest.TestCase):

    def test_cached_tokens_are_surfaced(self):
        data = {"event": "prefill_tick", "task": 1, "model": "m", "processed": 300,
                "to_process": 840, "cached": 32112, "fraction": 0.357, "rate": 50.0,
                "elapsed": 6.0, "eta_seconds": 10.0}
        line = strip_ansi(render_live(data, age=0.0, style=PLAIN, now=0))
        self.assertIn("+32,112 cached", line)
        self.assertIn("/840 tok", line)

    def test_summary_reports_the_prefill_share(self):
        lines = render_summary(
            {"tokens": 47174, "cached": 0, "seconds": 480.0, "rate": 98.0},
            {"tokens": 300, "seconds": 21.0, "rate": 14.2},
            {"seconds": 501.0, "prefill_share_pct": 95.8}, PLAIN)
        blob = strip_ansi("\n".join(lines))
        self.assertIn("PREFILL", blob)
        self.assertIn("47,174 tok", blob)
        self.assertIn("GENERATE", blob)
        self.assertIn("96% was prefill", blob)

    def test_summary_survives_a_missing_phase(self):
        """Attaching mid-request means the prefill block may never be seen."""
        lines = render_summary(None, {"tokens": 10, "seconds": 1.0, "rate": 10.0},
                               {"seconds": 1.0, "prefill_share_pct": None}, PLAIN)
        self.assertTrue(any("TOTAL" in strip_ansi(l) for l in lines))

    def test_no_ansi_when_color_disabled(self):
        line = render_live(TestLiveLineMoves.PREFILL, age=1.0, style=PLAIN, now=0)
        self.assertNotIn("\033", line)

    def test_ansi_present_when_color_enabled(self):
        line = render_live(TestLiveLineMoves.PREFILL, age=1.0, style=FANCY, now=0)
        self.assertIn("\033", line)

    def test_bar_styles(self):
        self.assertEqual(fmt_bar(0.5, 10, PLAIN), "[#####-----]")
        self.assertEqual(fmt_bar(0.5, 10, FANCY), "█████░░░░░")

    def test_idle_line_is_rendered(self):
        self.assertIn("waiting", strip_ansi(render_idle(3.0, PLAIN)))

    def test_unknown_event_renders_empty(self):
        self.assertEqual(render_live({"event": "nope"}, 1.0, PLAIN, now=0), "")
        self.assertEqual(render_live(None, 1.0, PLAIN, now=0), "")


class TestPercentTokenConsistency(unittest.TestCase):
    """Regression: the live line once showed '96%  3,584/39,528 tok'.

    Both numbers were individually true and mutually contradictory: 96% came from
    llama.cpp's raw progress (which counts cached tokens) while the token pair
    counted only computed tokens against the FULL prompt. Whatever is displayed,
    the percentage and the token ratio must agree.
    """

    @staticmethod
    def displayed(line):
        pct = re.search(r"(\d+)%", line)
        pair = re.search(r"([\d,]+)/([\d,]+) tok", line)
        if not pct or not pair:
            return None
        to_int = lambda s: int(s.replace(",", ""))
        return int(pct.group(1)), to_int(pair.group(1)), to_int(pair.group(2))

    def render_from_log(self, prompt_tokens, cached, processed, raw_progress):
        tracker = Tracker()
        lines = [
            "slot   operator(): id  0 | task 9 | new prompt, n_ctx_slot = 131072, "
            "n_keep = 4, task.n_tokens = %d" % prompt_tokens,
            "slot   operator(): id  0 | task 9 | cached n_tokens = %d, "
            "memory_seq_rm [%d, end)" % (cached, cached),
            "slot print_timing: id  0 | task 9 | prompt processing, n_tokens = %d, "
            "progress = %.2f, t =  10.00 s / 68.00 tokens per second"
            % (processed, raw_progress),
        ]
        data = None
        for line in lines:
            for out in tracker.feed(parse_line(line)):
                if out.data.get("event") == "prefill_tick":
                    data = out.data
        return strip_ansi(render_live(data, age=0.0, style=PLAIN, now=0))

    def test_the_exact_reported_case_no_longer_contradicts_itself(self):
        """Reported as '96%  3,584/39,528 tok'. All 3,584 uncached tokens were in
        fact done, so the honest rendering is the prompt-read state -- and above
        all it must never pair a percentage with a ratio that means something
        different."""
        line = self.render_from_log(prompt_tokens=39528, cached=35944,
                                    processed=3584, raw_progress=0.96)
        self.assertIn("waiting for first token", line)
        self.assertIn("+35,944 cached", line)
        self.assertNotIn("96%", line)
        self.assertIsNone(self.displayed(line),
                          "no percentage/ratio pair should be shown in this state")

    def test_mid_progress_with_heavy_cache_is_self_consistent(self):
        """Same shape as the report, caught halfway: 1,792 of 3,584 uncached."""
        line = self.render_from_log(prompt_tokens=39528, cached=35944,
                                    processed=1792, raw_progress=0.95)
        pct, seen, total = self.displayed(line)
        self.assertEqual((seen, total), (1792, 3584))
        self.assertAlmostEqual(pct, 50, delta=1)

    def test_percentage_always_matches_the_token_ratio(self):
        cases = [
            (10000, 0, 2500, 0.25),       # no cache
            (32952, 32112, 328, 0.98),    # the fixture case
            (5000, 4000, 500, 0.90),      # half of the uncached work done
        ]
        for prompt, cached, processed, raw in cases:
            line = self.render_from_log(prompt, cached, processed, raw)
            pct, seen, total = self.displayed(line)
            expected = 100.0 * seen / total
            self.assertAlmostEqual(
                pct, expected, delta=1,
                msg="%d%% shown against %d/%d in: %s" % (pct, seen, total, line))


class TestPrefillCompleteState(unittest.TestCase):
    """At 100% the prompt is read, but the request is not done: the server still
    builds logits, validates the KV cache and produces the first token, none of
    which llama-server logs. Reported symptom: '100% ... elapsed 28.0s', which
    reads like a stall.
    """

    DATA = {"event": "prefill_tick", "task": 1, "model": "m", "processed": 244,
            "to_process": 244, "cached": 41009, "fraction": 1.0, "rate": 57.0,
            "elapsed": 4.0, "eta_seconds": 0.0}

    def test_hundred_percent_names_the_wait_instead_of_showing_a_full_bar(self):
        line = strip_ansi(render_live(self.DATA, age=24.0, style=PLAIN, now=0))
        self.assertIn("waiting for first token", line)
        self.assertNotIn("100%", line)

    def test_clock_keeps_running_in_that_state(self):
        early = strip_ansi(render_live(self.DATA, age=1.0, style=PLAIN, now=0))
        later = strip_ansi(render_live(self.DATA, age=24.0, style=PLAIN, now=0))
        self.assertIn("elapsed 28.0s", later)     # 4 measured + 24 since
        self.assertNotEqual(early, later)

    def test_cached_context_is_still_reported(self):
        line = strip_ansi(render_live(self.DATA, age=1.0, style=PLAIN, now=0))
        self.assertIn("+41,009 cached", line)

    def test_below_the_threshold_still_shows_a_bar(self):
        data = dict(self.DATA, fraction=0.5, processed=122)
        self.assertIn("%", strip_ansi(render_live(data, age=0.0, style=PLAIN, now=0)))


class TestSparkline(unittest.TestCase):

    def test_empty_and_single(self):
        self.assertEqual(sparkline([], PLAIN), "")
        self.assertEqual(sparkline(None, PLAIN), "")
        self.assertEqual(len(sparkline([5.0], PLAIN)), 1)

    def test_monotonic_input_is_non_decreasing(self):
        spark = sparkline([1.0, 2.0, 3.0, 4.0, 5.0], FANCY)
        idx = [SPARK_UNICODE.index(c) for c in spark]
        self.assertEqual(idx, sorted(idx))
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], len(SPARK_UNICODE) - 1)

    def test_flat_input_does_not_divide_by_zero(self):
        spark = sparkline([7.0, 7.0, 7.0], FANCY)
        self.assertEqual(len(spark), 3)
        self.assertEqual(len(set(spark)), 1)

    def test_ascii_fallback(self):
        spark = sparkline([1.0, 9.0], PLAIN)
        self.assertTrue(all(c in ".:-=+*#" for c in spark), spark)

    def test_none_values_are_skipped(self):
        self.assertEqual(len(sparkline([1.0, None, 3.0], PLAIN)), 2)


class TestBoard(unittest.TestCase):

    def snap(self):
        stats = Stats(clock=lambda: 100.0)
        for i in range(3):
            prefill = {"tokens": 10000, "cached": 2000, "seconds": 100.0, "rate": 100.0 + i}
            generation = {"tokens": 200, "seconds": 20.0, "rate": 10.0 + i}
            end = {"task": i, "seconds": 120.0, "prefill_share_pct": 83.0}
            stats.record("m", prefill, generation, end)
        return stats.snapshot("m")

    def test_board_shows_all_headline_stats(self):
        blob = strip_ansi("\n".join(render_board(self.snap(), PLAIN, 100)))
        for label in ("PREFILL", "GENERATE", "CACHE", "TTFT", "WAIT"):
            self.assertIn(label, blob)
        self.assertIn("peak", blob)
        self.assertIn("low", blob)

    def test_board_has_no_ansi_when_colour_disabled(self):
        self.assertNotIn("\033", "\n".join(render_board(self.snap(), PLAIN, 100)))

    def test_compact_mode_drops_sparklines(self):
        full = render_board(self.snap(), FANCY, 100, compact=False)
        compact = render_board(self.snap(), FANCY, 100, compact=True)
        self.assertLess(len(compact), len(full))

    def test_empty_snapshot_renders_without_crashing(self):
        snap = Stats(clock=lambda: 0.0).snapshot("nothing-yet")
        blob = strip_ansi("\n".join(render_board(snap, PLAIN, 80)))
        self.assertIn("PREFILL", blob)

    def test_frame_fits_the_terminal_and_keeps_the_live_line(self):
        for rows in (10, 14, 24, 40):
            frame = compose_frame(self.snap(), "LIVE-MARKER", PLAIN, 80, rows)
            self.assertLessEqual(len(frame[:rows - 1]), rows - 1)
            joined = strip_ansi("\n".join(frame[:rows - 1]))
            self.assertIn("LIVE-MARKER", joined,
                          "the live line must never be dropped (rows=%d)" % rows)


class TestRecentHeaderAndHelp(unittest.TestCase):

    ROWS = [{"task": 596, "tokens": 3103, "seconds": 36.1, "rate": 88.2, "share": 98.0},
            {"task": 550, "tokens": 244, "seconds": 45.9, "rate": 48.1, "share": 11.0}]

    def test_recent_pane_labels_its_columns(self):
        """Four bare numbers per row is a guessing game."""
        lines = render_recent(self.ROWS, PLAIN, 100)
        header = strip_ansi(lines[0])
        for label in ("task", "prompt", "total", "prefill speed", "share of wait"):
            self.assertIn(label, header)
        self.assertIn("596", strip_ansi(lines[1]))

    def test_header_present_even_with_one_row(self):
        self.assertEqual(len(render_recent(self.ROWS[:1], PLAIN, 100)), 2)

    def test_help_explains_the_things_people_ask_about(self):
        blob = strip_ansi("\n".join(render_help(PLAIN, 100, 60)))
        for topic in ("PREFILL", "GENERATE", "cached", "TTFT", "token-weighted",
                      "waiting for", "sparkline"):
            self.assertIn(topic, blob)

    def test_help_fits_the_terminal(self):
        self.assertLessEqual(len(render_help(PLAIN, 80, 12)), 11)

    def test_help_replaces_the_board_but_keeps_the_live_line(self):
        snap = Stats(clock=lambda: 0.0).snapshot("m")
        frame = strip_ansi("\n".join(
            compose_frame(snap, "LIVE-MARKER", PLAIN, 100, 40, help_visible=True)))
        self.assertIn("how to read this", frame)
        self.assertIn("LIVE-MARKER", frame)
        # "WAIT" appears in the help text itself (explaining the row), so check
        # for board-only content instead.
        self.assertNotIn("of session spent in prefill", frame)
        self.assertNotIn("recent", frame)

    def test_colour_does_not_change_layout(self):
        """ANSI escapes have zero width on screen but count in %-12s. Padding a
        already-coloured string silently wrecks column alignment -- which is
        exactly what happened to the help screen's label column.
        """
        wide_plain = Style(color=False, unicode_ok=True, width=120)
        wide_fancy = Style(color=True, unicode_ok=True, width=120)
        for fn in (lambda s: render_help(s, 120, 60),
                   lambda s: render_recent(self.ROWS, s, 120)):
            plain_lines = fn(wide_plain)
            fancy_lines = [strip_ansi(l) for l in fn(wide_fancy)]
            self.assertEqual(plain_lines, fancy_lines)

    def test_board_colour_does_not_change_layout(self):
        snap = Stats(clock=lambda: 0.0)
        snap.record("m", {"tokens": 5000, "cached": 100, "seconds": 50.0, "rate": 100.0},
                    {"tokens": 200, "seconds": 20.0, "rate": 10.0},
                    {"task": 1, "seconds": 70.0, "prefill_share_pct": 71.0})
        s = snap.snapshot("m")
        plain = render_board(s, Style(color=False, unicode_ok=True, width=120), 120)
        fancy = [strip_ansi(l) for l in
                 render_board(s, Style(color=True, unicode_ok=True, width=120), 120)]
        self.assertEqual(plain, fancy)

    def test_hint_mentions_both_keys(self):
        snap = Stats(clock=lambda: 0.0).snapshot("m")
        frame = strip_ansi("\n".join(compose_frame(snap, "x", PLAIN, 100, 40)))
        self.assertIn("h help", frame)
        self.assertIn("ctrl-c quit", frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
