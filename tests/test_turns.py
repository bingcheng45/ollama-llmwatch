"""Tests for whole-turn timing: prompt submitted to final answer.

Every other number in llmwatch is about one model request. This one is about the
thing a person actually waits through, which on an agent is many requests plus a
lot of tool time. Three properties matter and are easy to lose:

- the duration comes from Codex's own clock, because llmwatch attaches to the
  session file at its end and routinely never sees a turn begin
- reasoning effort is recorded with it, because a high-effort turn and a
  low-effort one on the same model are not the same workload
- the message text sitting next to those timings in the session file is never
  stored, matching the property the request history already has
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    CodexTail, History, MIN_COMPARE_TURNS, Style, render_codex, render_turns,
    short_model_name, show_turns, export_history,
)

PLAIN = Style(color=False, unicode_ok=False, width=100)

FINAL_TEXT = "Here is the summary of the refactor you asked for, with secrets in it."


def started(turn_id="t1", at=1_700_000_000):
    return {"payload": {"type": "task_started", "turn_id": turn_id, "started_at": at}}


def context(turn_id="t1", model="ollama-local/qwen3:27b", effort="high"):
    return {"type": "turn_context",
            "payload": {"turn_id": turn_id, "model": model,
                        "collaboration_mode": {"mode": "default",
                                               "settings": {"reasoning_effort": effort}}}}


def complete(turn_id="t1", at=1_700_000_000, duration_ms=740_000, ttft_ms=165_000):
    return {"payload": {"type": "task_complete", "turn_id": turn_id,
                        "last_agent_message": FINAL_TEXT,
                        "started_at": at, "completed_at": at + duration_ms // 1000,
                        "duration_ms": duration_ms,
                        "time_to_first_token_ms": ttft_ms}}


def aborted(turn_id="t1", at=1_700_000_000, duration_ms=34_000):
    return {"payload": {"type": "turn_aborted", "turn_id": turn_id, "reason": "interrupted",
                        "started_at": at, "completed_at": at + duration_ms // 1000,
                        "duration_ms": duration_ms}}


class TailTestCase(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "rollout-test.jsonl")
        open(self.path, "w").close()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, records):
        with open(self.path, "a") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

    def tail(self):
        tail = CodexTail(path=self.path)
        tail.poll()                    # attach at EOF, like the real thing
        return tail

    def turns(self, records):
        tail = self.tail()
        self.write(records)
        tail.poll()
        return tail, tail.drain()


class TestTurnCapture(TailTestCase):

    def test_a_finished_turn_reports_its_total_time(self):
        _tail, turns = self.turns([started(), context(), complete()])
        self.assertEqual(len(turns), 1)
        self.assertAlmostEqual(turns[0]["seconds"], 740.0)
        self.assertAlmostEqual(turns[0]["ttft_seconds"], 165.0)
        self.assertTrue(turns[0]["completed"])

    def test_effort_is_recorded_with_the_turn(self):
        _tail, turns = self.turns([started(), context(effort="low"), complete()])
        self.assertEqual(turns[0]["effort"], "low")

    def test_effort_falls_back_to_the_thread_default(self):
        """A turn with no turn_context of its own still ran at some effort."""
        tail = self.tail()
        self.write([{"payload": {"type": "thread_settings_applied",
                                 "thread_settings": {"reasoning_effort": "medium",
                                                     "model": "ollama-local/qwen3:27b"}}}])
        tail.poll()
        self.write([started(), complete()])
        tail.poll()
        self.assertEqual(tail.drain()[0]["effort"], "medium")

    def test_a_turn_specific_effort_beats_the_thread_default(self):
        """The thread default can be changed between turns; the turn's own
        context is what it actually ran at."""
        tail = self.tail()
        self.write([{"payload": {"type": "thread_settings_applied",
                                 "thread_settings": {"reasoning_effort": "low"}}}])
        tail.poll()
        self.write([started(), context(effort="high"), complete()])
        tail.poll()
        self.assertEqual(tail.drain()[0]["effort"], "high")

    def test_an_unknown_effort_string_is_dropped_not_stored(self):
        """An unrecognised value would become its own history bucket and split
        the samples for nothing."""
        _tail, turns = self.turns([started(), context(effort="turbo"), complete()])
        self.assertIsNone(turns[0]["effort"])

    def test_model_name_is_normalised_to_match_the_ollama_log(self):
        """Codex writes `ollama-local/qwen3:27b`; the log writes a file path.
        History keys on the last segment or the two never line up."""
        _tail, turns = self.turns([started(), context(model="ollama-local/qwen3:27b"),
                                   complete()])
        self.assertEqual(turns[0]["model"], "qwen3:27b")
        self.assertEqual(short_model_name("/var/models/qwen3-27b.gguf"), "qwen3-27b.gguf")

    def test_a_turn_started_before_we_attached_is_still_timed(self):
        """llmwatch attaches at the END of the session file, so the common case
        is that task_started was written before it was watching. Codex stamps the
        start on the completion record too, which is why this works at all."""
        tail = self.tail()                 # never sees task_started
        self.write([complete(duration_ms=900_000)])
        tail.poll()
        self.assertAlmostEqual(tail.drain()[0]["seconds"], 900.0)

    def test_an_interrupted_turn_is_recorded_but_marked(self):
        _tail, turns = self.turns([started(), context(), aborted()])
        self.assertFalse(turns[0]["completed"])
        self.assertEqual(turns[0]["reason"], "interrupted")

    def test_tool_calls_are_counted_for_the_turn(self):
        call = {"payload": {"type": "function_call", "name": "shell",
                            "arguments": json.dumps({"cmd": "ls"})}}
        _tail, turns = self.turns([started(), context(), call, call, complete()])
        self.assertEqual(turns[0]["tool_calls"], 2)

    def test_drain_hands_each_turn_over_exactly_once(self):
        tail, turns = self.turns([started(), context(), complete()])
        self.assertEqual(len(turns), 1)
        self.assertEqual(tail.drain(), [])

    def test_an_absurd_duration_is_refused_rather_than_reported(self):
        """A clock change or a misread field would otherwise print '37h' as a
        turn time, which is worse than printing nothing."""
        _tail, turns = self.turns([started(), complete(duration_ms=48 * 3600 * 1000)])
        self.assertEqual(turns, [])

    def test_a_negative_duration_is_refused(self):
        _tail, turns = self.turns([started(), complete(duration_ms=-5000)])
        self.assertEqual(turns, [])

    def test_a_completion_with_no_timings_at_all_is_refused(self):
        """An older or truncated record must not become a turn timed from
        whenever llmwatch happened to attach."""
        tail = self.tail()
        self.write([{"payload": {"type": "task_complete", "turn_id": "t9"}}])
        tail.poll()
        self.assertEqual(tail.drain(), [])

    def test_a_running_turn_reports_elapsed_time_live(self):
        tail = self.tail()
        self.write([started(at=int(time.time()) - 30), context()])
        tail.poll()
        state = tail.state()
        self.assertGreaterEqual(state["turn_seconds"], 29)
        self.assertEqual(state["effort"], "high")

    def test_the_clock_stops_when_the_turn_ends(self):
        tail, _turns = self.turns([started(), context(), complete()])
        self.assertIsNone(tail.state()["turn_seconds"])
        self.assertAlmostEqual(tail.state()["last_turn"]["seconds"], 740.0)


class TestTurnPrivacy(TailTestCase):
    """The session file contains the agent's actual output. The request history
    deliberately cannot hold prompt content; the turn history must not either."""

    def test_the_final_message_never_leaves_the_tail(self):
        _tail, turns = self.turns([started(), context(), complete()])
        self.assertNotIn(FINAL_TEXT, json.dumps(turns))

    def test_the_final_message_is_not_written_to_the_database(self):
        db = os.path.join(self.dir, "history.db")
        history = History(path=db)
        _tail, turns = self.turns([started(), context(), complete()])
        history.record_turn(turns[0])
        history.close()
        with open(db, "rb") as fh:
            blob = fh.read()
        self.assertNotIn(b"refactor you asked for", blob)

    def test_the_turns_table_has_no_column_that_could_hold_content(self):
        db = os.path.join(self.dir, "history.db")
        History(path=db).close()
        cur = sqlite3.connect(db).execute("PRAGMA table_info(turns)")
        columns = {row[1] for row in cur.fetchall()}
        for forbidden in ("prompt", "text", "content", "message", "response", "command"):
            self.assertNotIn(forbidden, columns)
        self.assertIn("seconds", columns)
        self.assertIn("effort", columns)


class Args(object):
    def __init__(self, **kw):
        self.days = 30
        self.model = None
        self.turns = True
        self.export = None
        for key, value in kw.items():
            setattr(self, key, value)


class TestTurnHistory(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.history = History(path=os.path.join(self.dir, "history.db"))
        self.now = 1_700_000_000.0

    def tearDown(self):
        self.history.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def add(self, model, seconds, effort="high", n=1, completed=True, tool_calls=3,
            age_days=0.0):
        for i in range(n):
            self.history.record_turn({
                "model": model, "effort": effort, "seconds": seconds,
                "ttft_seconds": seconds / 4.0, "tool_calls": tool_calls,
                "completed": completed,
                "ended_at": self.now - age_days * 86400 - i})

    def test_a_turn_round_trips(self):
        self.add("mtp", 740.0)
        rows = self.history.turn_rows(days=30, now=self.now)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["effort"], "high")
        self.assertAlmostEqual(rows[0]["seconds"], 740.0)

    def test_profile_splits_by_effort(self):
        self.add("mtp", 60.0, effort="low", n=4)
        self.add("mtp", 900.0, effort="high", n=4)
        profile = self.history.turn_profile("mtp", days=30, now=self.now)
        self.assertEqual(profile["turns"], 8)
        self.assertAlmostEqual(profile["efforts"]["low"]["median_seconds"], 60.0)
        self.assertAlmostEqual(profile["efforts"]["high"]["median_seconds"], 900.0)

    def test_interrupted_turns_are_counted_but_never_timed(self):
        """Their duration measures how long you waited before giving up."""
        self.add("mtp", 100.0, n=3)
        self.add("mtp", 9000.0, n=2, completed=False)
        profile = self.history.turn_profile("mtp", days=30, now=self.now)
        self.assertEqual(profile["turns"], 3)
        self.assertEqual(profile["interrupted"], 2)
        self.assertAlmostEqual(profile["median_seconds"], 100.0)

    def test_the_median_is_not_dragged_by_one_abandoned_turn(self):
        """One turn where you walked away must not set the expectation for
        every turn after it."""
        self.add("mtp", 60.0, n=6)
        self.add("mtp", 7200.0, n=1)
        profile = self.history.turn_profile("mtp", days=30, now=self.now)
        self.assertLess(profile["median_seconds"], 120.0)

    def test_old_turns_fall_outside_the_window(self):
        self.add("mtp", 60.0, n=2, age_days=40)
        self.add("mtp", 90.0, n=2, age_days=1)
        self.assertEqual(self.history.turn_profile("mtp", days=30,
                                                   now=self.now)["turns"], 2)

    def test_a_turn_with_no_duration_is_refused(self):
        self.assertFalse(self.history.record_turn({"model": "mtp", "seconds": None}))

    def test_the_existing_request_history_still_opens(self):
        """The turns table arrives in an already-created database."""
        self.history.record("mtp", {"tokens": 10, "cached": 0, "seconds": 1.0, "rate": 10.0},
                            {"tokens": 5, "seconds": 1.0, "rate": 5.0},
                            {"seconds": 2.0}, now=self.now)
        self.add("mtp", 60.0)
        self.assertEqual(len(self.history.all_rows(now=self.now)), 1)
        self.assertEqual(len(self.history.turn_rows(now=self.now)), 1)

    def render(self, fn, args):
        import io
        buf = io.StringIO()
        code = fn(args, out=buf, history=self.history, now=self.now)
        return code, buf.getvalue()

    def test_the_turns_table_answers_how_long_this_takes(self):
        self.add("mtp", 60.0, effort="low", n=3)
        self.add("mtp", 1500.0, effort="high", n=3)
        _code, text = self.render(show_turns, Args())
        self.assertIn("1m00s", text)
        self.assertIn("25m00s", text)
        self.assertIn("low", text)
        self.assertIn("high", text)

    def test_turns_view_says_so_when_there_is_nothing_recorded(self):
        code, text = self.render(show_turns, Args())
        self.assertEqual(code, 1)
        self.assertIn("--codex", text)

    def test_export_can_dump_turns(self):
        self.add("mtp", 60.0)
        _code, text = self.render(export_history, Args(export="json", turns=True))
        self.assertIn('"effort": "high"', text)
        _code, text = self.render(export_history, Args(export="csv", turns=True))
        self.assertIn("effort", text.splitlines()[0])

    def test_export_without_turns_still_dumps_requests(self):
        self.history.record("mtp", {"tokens": 10, "cached": 0, "seconds": 1.0, "rate": 10.0},
                            {"tokens": 5, "seconds": 1.0, "rate": 5.0},
                            {"seconds": 2.0}, now=self.now)
        _code, text = self.render(export_history, Args(export="json", turns=False))
        self.assertIn('"prefill_rate"', text)


def profile(turns=6, median=600.0, efforts=None, interrupted=0):
    return {"model": "m", "turns": turns, "interrupted": interrupted,
            "median_seconds": median, "median_ttft": 60.0, "median_tool_calls": 4,
            "total_seconds": turns * median, "efforts": efforts or {}}


def effort(turns=4, median=600.0):
    return {"turns": turns, "median_seconds": median, "median_ttft": 60.0}


class TestTurnComparison(unittest.TestCase):

    def render(self, a, b):
        return "\n".join(render_turns(a, b, PLAIN))

    def test_nothing_is_shown_when_neither_side_has_turns(self):
        self.assertEqual(render_turns(profile(turns=0), profile(turns=0), PLAIN), [])

    def test_a_shared_effort_level_gets_a_verdict(self):
        blob = self.render(
            profile(efforts={"high": effort(median=300.0)}),
            profile(efforts={"high": effort(median=600.0)}))
        self.assertIn("x quicker", blob)

    def test_the_pooled_total_never_claims_a_winner(self):
        """A model run mostly on low effort against one run mostly on high is
        not a comparison, and the pooled row must not pretend otherwise."""
        blob = self.render(
            profile(median=379.0, efforts={"low": effort(turns=6, median=57.0),
                                           "high": effort(turns=5, median=2009.0)}),
            profile(median=776.0, efforts={"high": effort(turns=4, median=776.0)}))
        pooled = [l for l in blob.splitlines() if "TURN" in l][0]
        self.assertNotIn("quicker", pooled)
        self.assertIn("all efforts pooled", pooled)

    def test_thin_data_reports_counts_instead_of_a_ratio(self):
        blob = self.render(
            profile(efforts={"high": effort(turns=MIN_COMPARE_TURNS - 1)}),
            profile(efforts={"high": effort(turns=MIN_COMPARE_TURNS - 1, median=900.0)}))
        self.assertIn("need %d each" % MIN_COMPARE_TURNS, blob)
        self.assertNotIn("x quicker", blob)

    def test_an_effort_only_one_side_ran_is_labelled_not_compared(self):
        blob = self.render(profile(efforts={"low": effort()}),
                           profile(efforts={"high": effort()}))
        self.assertEqual(blob.count("one side only"), 2)
        self.assertNotIn("x quicker", blob)

    def test_interrupted_turns_are_disclosed(self):
        blob = self.render(profile(interrupted=3), profile())
        self.assertIn("3 interrupted", blob)


class TestTurnDisplay(unittest.TestCase):

    def test_a_running_turn_shows_the_clock_and_the_effort(self):
        blob = "\n".join(render_codex(
            {"action": "shell", "detail": "pytest", "calls": 6, "turn_seconds": 740.0,
             "effort": "high", "last_turn": None}, PLAIN, 100))
        self.assertIn("12m20s so far", blob)
        self.assertIn("effort high", blob)

    def test_a_finished_turn_shows_its_total(self):
        blob = "\n".join(render_codex(
            {"action": "done", "calls": 14, "turn_seconds": None, "effort": "high",
             "last_turn": {"seconds": 740.0, "effort": "high", "tool_calls": 14,
                           "completed": True}}, PLAIN, 100))
        self.assertIn("last turn", blob)
        self.assertIn("12m20s", blob)

    def test_a_finished_turn_does_not_leave_a_stale_this_turn_line(self):
        blob = "\n".join(render_codex(
            {"action": "done", "calls": 14, "turn_seconds": None,
             "last_turn": {"seconds": 740.0, "completed": True}}, PLAIN, 100))
        self.assertNotIn("this turn", blob)

    def test_a_slow_turn_is_called_slow_against_your_own_history(self):
        blob = "\n".join(render_codex(
            {"action": "done", "turn_seconds": None,
             "last_turn": {"seconds": 740.0, "completed": True,
                           "typical_seconds": 360.0}}, PLAIN, 100))
        self.assertIn("2.1x your usual 6m00s", blob)

    def test_a_normal_turn_is_not_dressed_up_as_a_finding(self):
        blob = "\n".join(render_codex(
            {"action": "done", "turn_seconds": None,
             "last_turn": {"seconds": 610.0, "completed": True,
                           "typical_seconds": 600.0}}, PLAIN, 100))
        self.assertIn("about your usual", blob)

    def test_an_interrupted_turn_says_so(self):
        blob = "\n".join(render_codex(
            {"action": "interrupted", "turn_seconds": None,
             "last_turn": {"seconds": 34.0, "completed": False,
                           "reason": "interrupted"}}, PLAIN, 100))
        self.assertIn("interrupted", blob)

    def test_a_codex_state_without_turn_fields_still_renders(self):
        """State from an older session, or one where no turn has begun."""
        blob = "\n".join(render_codex(
            {"action": "shell", "detail": "ls", "calls": 2}, PLAIN, 100))
        self.assertIn("2 tool calls", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
