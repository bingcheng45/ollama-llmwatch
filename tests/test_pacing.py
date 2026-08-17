"""Tests for frame pacing and cancelled-request detection.

Pacing has two failure modes that pull in opposite directions: repaint only when
the log speaks and the display looks frozen for 5-10s at a time (the v0.2.0 bug);
repaint on every line and a burst becomes flicker plus wasted CPU.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    FRAME_ACTIVE, FRAME_IDLE, MIN_FRAME_GAP, Tracker, parse_line, plan_frame,
)


class TestPacing(unittest.TestCase):

    def test_new_data_repaints_immediately(self):
        """The whole point: a log line must not wait for the next tick."""
        paint, _ = plan_frame(now=1.0, last_paint=0.0, next_frame=99.0,
                              got_data=True, active=True)
        self.assertTrue(paint)

    def test_frames_continue_with_no_data_at_all(self):
        """v0.2.0's anti-freeze guarantee must not regress."""
        paint, _ = plan_frame(now=5.0, last_paint=4.0, next_frame=4.5,
                              got_data=False, active=True)
        self.assertTrue(paint)

    def test_nothing_due_means_no_paint(self):
        paint, nxt = plan_frame(now=1.0, last_paint=0.95, next_frame=2.0,
                                got_data=False, active=True)
        self.assertFalse(paint)
        self.assertEqual(nxt, 2.0)

    def test_burst_cannot_exceed_the_frame_ceiling(self):
        """50 lines arriving together must not cause 50 repaints."""
        now, last_paint, next_frame = 1.0, 1.0, 1.0
        paints = 0
        for _ in range(50):
            now += 0.0001          # a burst lands within a fraction of a frame
            paint, next_frame = plan_frame(now, last_paint, next_frame,
                                           got_data=True, active=True)
            if paint:
                paints += 1
                last_paint = now
        self.assertEqual(paints, 0, "burst repainted faster than the ceiling")

    def test_deferred_paint_is_rescheduled_not_dropped(self):
        paint, nxt = plan_frame(now=1.0, last_paint=1.0, next_frame=1.0,
                                got_data=True, active=True)
        self.assertFalse(paint)
        self.assertAlmostEqual(nxt, 1.0 + MIN_FRAME_GAP)

    def test_idle_uses_the_slower_floor(self):
        _, nxt_active = plan_frame(2.0, 0.0, 0.0, False, active=True)
        _, nxt_idle = plan_frame(2.0, 0.0, 0.0, False, active=False)
        self.assertAlmostEqual(nxt_active, 2.0 + FRAME_ACTIVE)
        self.assertAlmostEqual(nxt_idle, 2.0 + FRAME_IDLE)
        self.assertGreater(nxt_idle, nxt_active)

    def test_active_floor_is_fluid_enough_to_animate(self):
        """A 1s floor would visibly stutter the spinner and jump the ETA."""
        self.assertLessEqual(FRAME_ACTIVE, 0.2)


class TestCancelledRequests(unittest.TestCase):
    """A slot serves one request at a time. If a new one starts while another is
    open, the old one was cancelled -- the client disconnected (a Codex timeout)
    and llama-server never wrote its `total time` line. Previously this left a
    header with nothing under it, which read like llmwatch had lost track.
    """

    START = ("slot   operator(): id  0 | task %d | new prompt, n_ctx_slot = 131072, "
             "n_keep = 4, task.n_tokens = 1000")

    def test_new_request_on_same_slot_marks_the_old_one_cancelled(self):
        tracker = Tracker()
        events = []
        for task in (1, 2):
            for out in tracker.feed(parse_line(self.START % task)):
                events.append(out.data)
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds, ["request_start", "request_abandoned", "request_start"])
        self.assertEqual(events[1]["task"], 1)

    def test_completed_request_is_not_marked_cancelled(self):
        tracker = Tracker()
        lines = [
            self.START % 1,
            "slot print_timing: id  0 | task 1 | prompt eval time = 1000.00 ms / 500 "
            "tokens (2.00 ms per token, 500.00 tokens per second)",
            "slot print_timing: id  0 | task 1 |        eval time = 1000.00 ms / 10 "
            "tokens (100.00 ms per token, 10.00 tokens per second)",
            "slot print_timing: id  0 | task 1 |       total time = 2000.00 ms / 510 tokens",
            self.START % 2,
        ]
        kinds = []
        for line in lines:
            for out in tracker.feed(parse_line(line)):
                kinds.append(out.data["event"])
        self.assertNotIn("request_abandoned", kinds)

    def test_other_slots_are_untouched(self):
        tracker = Tracker()
        tracker.feed(parse_line(self.START % 1))
        events = []
        for out in tracker.feed(parse_line(
                "slot   operator(): id  1 | task 2 | new prompt, n_ctx_slot = 131072, "
                "n_keep = 4, task.n_tokens = 1000")):
            events.append(out.data["event"])
        self.assertNotIn("request_abandoned", events,
                         "a request on a different slot is still running")


if __name__ == "__main__":
    unittest.main(verbosity=2)
