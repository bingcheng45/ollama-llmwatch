"""Tests for the 'why is this slow' signals and the diagnosis line.

These exist so llmwatch can answer the only question that matters mid-wait:
keep waiting, or kill it? Each diagnosis must be short, plain, and actionable --
and above all correct, since a wrong warning is worse than no warning.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    CacheMiss, CheckpointCreated, CheckpointRestored, DraftAcceptance, LOOP_REPEATS,
    Stats, Style, Tracker, diagnose, parse_line, project_completion, render_board,
)

PLAIN = Style(color=False, unicode_ok=False, width=100)


def strip_ansi(text):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestNewLineParsing(unittest.TestCase):

    def test_cache_miss(self):
        ev = parse_line(
            "slot   operator(): id  0 | task 49 | forcing full prompt re-processing due "
            "to lack of cache data (likely due to SWA or hybrid/recurrent memory)")
        self.assertIsInstance(ev, CacheMiss)
        self.assertEqual((ev.slot, ev.task), (0, 49))

    def test_checkpoint_restored(self):
        ev = parse_line(
            "slot   operator(): id  0 | task 1040 | restored context checkpoint "
            "(pos_min = 42986, pos_max = 42986, n_tokens = 42987, n_past = 42987, "
            "size = 318.3 MiB)")
        self.assertIsInstance(ev, CheckpointRestored)
        self.assertEqual(ev.tokens, 42987)

    def test_checkpoint_created(self):
        ev = parse_line(
            "slot create_check: id  0 | task 1040 | created context checkpoint 3 of 32 "
            "(pos_min = 43444, pos_max = 43444, n_tokens = 43445, size = 3.1 MiB)")
        self.assertIsInstance(ev, CheckpointCreated)
        self.assertEqual((ev.index, ev.total), (3, 32))

    def test_draft_acceptance(self):
        ev = parse_line(
            "slot print_timing: id  0 | task 969 | draft acceptance = 0.52982 "
            "(  462 accepted /   872 generated), mean len =  3.12")
        self.assertIsInstance(ev, DraftAcceptance)
        self.assertAlmostEqual(ev.rate, 0.52982)
        self.assertEqual((ev.accepted, ev.generated), (462, 872))
        self.assertAlmostEqual(ev.mean_len, 3.12)

    def test_new_lines_do_not_shadow_existing_ones(self):
        """These lines contain 'n_tokens =' too; they must not be mistaken for a
        request start or a cache-size line."""
        from llmwatch import RequestStart
        ev = parse_line("slot   operator(): id  0 | task 7 | new prompt, n_ctx_slot = "
                        "131072, n_keep = 4, task.n_tokens = 500")
        self.assertIsInstance(ev, RequestStart)


class TestTrackerContext(unittest.TestCase):

    START = ("slot   operator(): id  0 | task 5 | new prompt, n_ctx_slot = 131072, "
             "n_keep = 4, task.n_tokens = 20000")
    TICK = ("slot print_timing: id  0 | task 5 | prompt processing, n_tokens = 5000, "
            "progress = 0.25, t = 50.00 s / 100.00 tokens per second")
    MISS = ("slot   operator(): id  0 | task 5 | forcing full prompt re-processing due "
            "to lack of cache data")

    def feed(self, lines):
        tracker = Tracker()
        outs = []
        for line in lines:
            outs.extend(tracker.feed(parse_line(line)))
        return outs

    def test_cache_miss_reaches_the_live_payload(self):
        outs = self.feed([self.START, self.TICK, self.MISS])
        live = [o.data for o in outs if o.data.get("event") == "prefill_tick"]
        self.assertTrue(live[-1]["cache_miss"])

    def test_cache_miss_is_also_committed_as_its_own_event(self):
        outs = self.feed([self.START, self.MISS])
        self.assertIn("cache_miss", [o.data.get("event") for o in outs])

    def test_state_change_re_emits_the_live_view_immediately(self):
        """Otherwise the warning waits for the next 512-token batch, which at
        100 tok/s is five seconds of the user not knowing."""
        outs = self.feed([self.START, self.TICK, self.MISS])
        ticks = [o for o in outs if o.data.get("event") == "prefill_tick"]
        self.assertEqual(len(ticks), 2, "expected a re-emitted live payload")

    def test_checkpoint_restore_sets_a_status(self):
        outs = self.feed([self.START, self.TICK,
                          "slot   operator(): id  0 | task 5 | restored context "
                          "checkpoint (pos_min = 1, pos_max = 1, n_tokens = 4242)"])
        live = [o.data for o in outs if o.data.get("event") == "prefill_tick"]
        self.assertIn("4,242", live[-1]["status"])


class TestDiagnosis(unittest.TestCase):

    BASE = {"event": "prefill_tick", "task": 1, "to_process": 39528, "cached": 0,
            "rate": 100.0, "elapsed": 10.0, "eta_seconds": 300.0}

    def test_cache_miss_states_the_cost_not_just_the_fact(self):
        """'cache miss' alone isn't actionable; 'rereading 39,528 tok (~6m35s)' is."""
        line = " ".join(strip_ansi(x) for x in
                        diagnose(dict(self.BASE, cache_miss=True), {}, PLAIN))
        self.assertIn("cache gone", line)
        self.assertIn("39,528", line)
        self.assertIn("~", line)

    def test_loop_detection(self):
        snap = {"looping": True, "repeat_count": 4}
        line = " ".join(strip_ansi(x) for x in diagnose(self.BASE, snap, PLAIN))
        self.assertIn("loop", line)
        self.assertIn("4x", line)

    def test_repeated_cancels(self):
        line = " ".join(strip_ansi(x) for x in
                        diagnose(self.BASE, {"recent_cancels": 3}, PLAIN))
        self.assertIn("timing out", line)

    def test_long_context_suggests_compacting(self):
        data = dict(self.BASE, to_process=45000)
        line = " ".join(strip_ansi(x) for x in diagnose(data, {}, PLAIN))
        self.assertIn("long chat", line)
        self.assertIn("compact", line)

    def test_big_conversation_that_is_fully_cached_is_not_warned_about(self):
        """41k of context costs nothing to continue if 41k is cached. Warning
        about total context size rather than tokens actually read is both wrong
        and needlessly alarming."""
        data = dict(self.BASE, to_process=244, cached=41009)
        line = " ".join(strip_ansi(x) for x in diagnose(data, {}, PLAIN))
        self.assertNotIn("long chat", line)
        self.assertNotIn("compact", line)

    def test_healthy_cached_request_says_so(self):
        data = dict(self.BASE, to_process=244, cached=41009)
        line = " ".join(strip_ansi(x) for x in diagnose(data, {}, PLAIN))
        self.assertIn("cache working", line)

    def test_low_draft_acceptance_is_called_out(self):
        data = {"event": "generate_tick", "decoded": 100, "rate": 10.0,
                "draft": (0.22, 22, 100, 1.4)}
        line = " ".join(strip_ansi(x) for x in diagnose(data, {}, PLAIN))
        self.assertIn("22%", line)

    def test_good_draft_acceptance_is_not_nagged_about(self):
        data = {"event": "generate_tick", "decoded": 100, "rate": 10.0,
                "draft": (0.71, 71, 100, 3.4)}
        line = " ".join(strip_ansi(x) for x in diagnose(data, {}, PLAIN))
        self.assertNotIn("slowing", line)

    def test_at_most_two_findings_so_the_line_stays_readable(self):
        snap = {"looping": True, "repeat_count": 5, "recent_cancels": 4}
        self.assertLessEqual(len(diagnose(dict(self.BASE, cache_miss=True), snap, PLAIN)), 2)

    def test_findings_are_short(self):
        snap = {"looping": True, "repeat_count": 5}
        for finding in diagnose(dict(self.BASE, cache_miss=True), snap, PLAIN):
            self.assertLessEqual(len(strip_ansi(finding)), 70, finding)


class TestCompletionProjection(unittest.TestCase):

    SNAP = {"generation": {"avg": 10.0}, "avg_output_tokens": 200.0}

    def test_prefill_projection_includes_the_writing_time(self):
        data = {"event": "prefill_tick", "eta_seconds": 60.0}
        # 60s left reading + 200 tokens / 10 tok/s = 80s
        self.assertAlmostEqual(project_completion(self.SNAP, data, 0.0), 80.0)

    def test_projection_shrinks_as_time_passes(self):
        data = {"event": "prefill_tick", "eta_seconds": 60.0}
        self.assertLess(project_completion(self.SNAP, data, 30.0),
                        project_completion(self.SNAP, data, 0.0))

    def test_no_history_means_no_guess(self):
        data = {"event": "prefill_tick", "eta_seconds": 60.0}
        self.assertIsNone(project_completion({}, data, 0.0))

    def test_generation_projection_counts_down(self):
        data = {"event": "generate_tick", "decoded": 100, "rate": 10.0}
        self.assertAlmostEqual(project_completion(self.SNAP, data, 0.0), 10.0)

    def test_projection_never_negative(self):
        data = {"event": "generate_tick", "decoded": 500, "rate": 10.0}
        self.assertEqual(project_completion(self.SNAP, data, 0.0), 0.0)


class TestBoardExtras(unittest.TestCase):

    def snap_with(self, **end_extra):
        stats = Stats(clock=lambda: 0.0)
        end = {"task": 1, "seconds": 100.0, "prefill_share_pct": 90.0}
        end.update(end_extra)
        stats.record("m", {"tokens": 1000, "cached": 500, "seconds": 90.0, "rate": 11.0},
                     {"tokens": 100, "seconds": 10.0, "rate": 10.0}, end)
        return stats.snapshot("m")

    def test_draft_acceptance_on_the_board(self):
        snap = self.snap_with(draft=(0.53, 462, 872, 3.12))
        blob = strip_ansi("\n".join(render_board(snap, PLAIN, 100)))
        self.assertIn("DRAFT", blob)
        self.assertIn("53%", blob)

    def test_cache_misses_counted_on_the_board(self):
        blob = strip_ansi("\n".join(render_board(self.snap_with(cache_miss=True),
                                                 PLAIN, 100)))
        self.assertIn("full reread", blob)

    def test_no_draft_line_for_non_mtp_models(self):
        blob = strip_ansi("\n".join(render_board(self.snap_with(), PLAIN, 100)))
        self.assertNotIn("DRAFT", blob)


class TestCodexPane(unittest.TestCase):

    def state(self, detail):
        return {"action": "exec_command", "detail": detail, "calls": 10,
                "waiting_since": 149.0}

    def test_long_commands_are_truncated_to_stay_glanceable(self):
        from llmwatch import render_codex
        long_cmd = ("cd /Users/user/ai_projects/memory-chess && check() { hits=$(rg -lF "
                    "\"$1\" src --glob '!**/__tests__/**' | head -20); echo $hits; }")
        lines = render_codex(self.state(long_cmd), PLAIN, 120)
        detail_line = strip_ansi(lines[1])
        self.assertLessEqual(len(detail_line), 90, detail_line)
        self.assertIn("...", detail_line)

    def test_short_commands_are_shown_whole(self):
        from llmwatch import render_codex
        lines = render_codex(self.state("git status"), PLAIN, 120)
        self.assertIn("git status", strip_ansi(lines[1]))
        self.assertNotIn("...", strip_ansi(lines[1]))

    def test_pane_reports_the_wait_and_call_count(self):
        from llmwatch import render_codex
        blob = strip_ansi("\n".join(render_codex(self.state("ls"), PLAIN, 120)))
        self.assertIn("10 tool calls", blob)
        self.assertIn("2m29s", blob)

    def test_no_pane_without_state(self):
        from llmwatch import render_codex
        self.assertEqual(render_codex(None, PLAIN, 120), [])

    def test_argument_summary_picks_the_meaningful_field(self):
        from llmwatch import CodexTail
        self.assertEqual(CodexTail._summarise('{"cmd":"rg -n foo src"}'), "rg -n foo src")
        self.assertEqual(CodexTail._summarise('{"file_path":"/a/b.py"}'), "/a/b.py")
        self.assertIsNone(CodexTail._summarise(None))
        self.assertIn("x", CodexTail._summarise('{"unknown":"x"}'))

    def test_malformed_json_does_not_raise(self):
        from llmwatch import CodexTail
        tail = CodexTail()
        tail._consume("not json at all\n")     # must be ignored silently
        self.assertIsNone(tail.action)


class TestLoopAndCancelTracking(unittest.TestCase):

    def test_identical_prompt_sizes_flag_a_loop(self):
        stats = Stats(clock=lambda: 0.0)
        for _ in range(LOOP_REPEATS):
            stats.note_start("m", 47174)
        self.assertTrue(stats.snapshot("m")["looping"])

    def test_varied_prompt_sizes_do_not(self):
        stats = Stats(clock=lambda: 0.0)
        for size in (1000, 2000, 3000):
            stats.note_start("m", size)
        self.assertFalse(stats.snapshot("m")["looping"])

    def test_cancel_streak_clears_on_a_completed_request(self):
        stats = Stats(clock=lambda: 0.0)
        stats.note_cancel("m")
        stats.note_cancel("m")
        self.assertEqual(stats.snapshot("m")["recent_cancels"], 2)
        stats.record("m", None, None, {"task": 1, "seconds": 1.0})
        self.assertEqual(stats.snapshot("m")["recent_cancels"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
