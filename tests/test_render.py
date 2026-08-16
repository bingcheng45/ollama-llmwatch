"""Tests for llmwatch's live rendering and between-tick projection.

The bug these guard against: llama-server writes a progress line only once per
512-token batch, which at typical rates is every 5-10 seconds. A display that
repaints only when a line arrives sits visibly frozen in between, and users
reasonably conclude the tool (or the model) has hung.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    Style, fmt_bar, project, render_idle, render_live, render_summary, spinner_frame,
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

    def test_eta_counts_down_and_never_goes_negative(self):
        mid = strip_ansi(render_live(self.PREFILL, age=30.0, style=PLAIN, now=0))
        self.assertIn("eta 1m00s", mid)            # 90 - 30
        past = strip_ansi(render_live(self.PREFILL, age=500.0, style=PLAIN, now=0))
        self.assertNotIn("-", past.split("eta")[1])

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
