"""Tests that no rendered line can exceed the terminal width.

The bug these guard against, as reported from a real session:

    2,537/14,517 tok +50,688 ca─red     53 tok/s8 cached     38 tok/s
    ! 5 cancels in acin a row - client keeps timing out

Two frames overlaid. One line was longer than the terminal, so it wrapped; every
row below shifted down; the next repaint started from cursor-home and landed on
the wrong rows. Clipping height alone is not enough -- width must be clipped too.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    Screen, Stats, Style, compose_frame, render_live, render_live_detail,
    truncate_visible, visible_len,
)

FANCY = Style(color=True, unicode_ok=True, width=110)


class FakeStream:
    def __init__(self):
        self.written = []

    def write(self, text):
        self.written.append(text)

    def flush(self):
        pass


class TestVisibleLength(unittest.TestCase):

    def test_ansi_escapes_take_no_columns(self):
        self.assertEqual(visible_len("\033[1mbold\033[0m"), 4)
        self.assertEqual(visible_len("plain"), 5)
        self.assertEqual(visible_len(""), 0)

    def test_truncate_counts_visible_characters_only(self):
        text = "\033[32m" + "x" * 50 + "\033[0m"
        self.assertEqual(visible_len(truncate_visible(text, 10)), 10)

    def test_short_text_is_untouched(self):
        text = "\033[1mshort\033[0m"
        self.assertEqual(truncate_visible(text, 80), text)

    def test_colour_is_reset_so_it_cannot_bleed(self):
        """Cutting mid-colour without a reset leaves the rest of the terminal
        tinted."""
        cut = truncate_visible("\033[31m" + "y" * 40, 5)
        self.assertTrue(cut.endswith("\033[0m"))

    def test_zero_and_negative_widths_are_safe(self):
        self.assertEqual(truncate_visible("anything", 0), "")
        self.assertEqual(truncate_visible("anything", -5), "")


class TestFrameFitsTheTerminal(unittest.TestCase):

    def stats(self):
        stats = Stats(clock=lambda: 0.0)
        for i in range(4):
            stats.record("qwen3.8:27b-mtp-128k",
                         {"tokens": 14517, "cached": 50688, "seconds": 140.0, "rate": 100.0},
                         {"tokens": 200, "seconds": 20.0, "rate": 10.0},
                         {"task": i, "seconds": 160.0, "prefill_share_pct": 90.0,
                          "draft": (0.53, 462, 872, 3.1)})
        return stats.snapshot("qwen3.8:27b-mtp-128k")

    def reported_case(self):
        """The exact state from the report: long model name, big cached count,
        two findings and a projection all at once."""
        data = {"event": "prefill_tick", "task": 1, "model": "qwen3.8:27b-mtp-128k",
                "processed": 2537, "to_process": 14517, "cached": 50688,
                "fraction": 0.17, "rate": 38.0, "elapsed": 67.0, "eta_seconds": 318.0,
                "cache_miss": False, "status": None, "draft": None}
        snap = dict(self.stats(), looping=True, repeat_count=5, recent_cancels=5)
        return data, snap

    def test_every_line_fits_after_drawing(self):
        data, snap = self.reported_case()
        for cols in (80, 100, 110, 120, 200):
            style = Style(color=True, unicode_ok=True, width=cols)
            frame = compose_frame(
                snap, render_live(data, 0.0, style), style, cols, 40,
                live_detail=render_live_detail(data, snap, style, 0.0),
                system={"contention": ["swapping 3.0/4.0 GB"]})
            screen = Screen(stream=FakeStream())
            screen.active = True
            screen.stream = FakeStream()
            screen.draw(frame, 40, cols)
            written = "".join(screen.stream.written)
            body = written.replace("\033[H", "").replace("\033[J", "")
            for line in body.split("\033[K\n"):
                self.assertLessEqual(
                    visible_len(line.replace("\033[K", "")), cols - 1,
                    "line exceeds %d cols: %r" % (cols, line[:120]))

    def test_each_finding_gets_its_own_line(self):
        """Joined with spaces, two findings plus a projection ran past 120
        columns -- a run-on that wrapped and read as a wall of text."""
        data, snap = self.reported_case()
        style = Style(color=False, unicode_ok=True, width=110)
        lines = render_live_detail(data, snap, style, 0.0)
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(visible_len(line), 100, line)

    def test_projection_is_its_own_line_and_always_present(self):
        data, snap = self.reported_case()
        for cols in (80, 110, 200):
            style = Style(color=False, unicode_ok=True, width=cols)
            lines = render_live_detail(data, snap, style, 0.0)
            self.assertTrue(any("answer ready" in l for l in lines),
                            "projection lost at %d cols" % cols)

    def test_draw_without_cols_still_works(self):
        """Older call sites pass rows only; must not crash."""
        screen = Screen(stream=FakeStream())
        screen.draw(["a line"], 10)
        self.assertTrue(screen.stream.written)


if __name__ == "__main__":
    unittest.main(verbosity=2)
