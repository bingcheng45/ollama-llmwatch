"""Failure-mode tests: things that break in the real world, not in the happy path.

llmwatch runs for hours beside an agent while Ollama restarts, logs rotate, tools
fail and clients disconnect. It must degrade quietly rather than lie or crash.

Note on networking: llmwatch makes exactly one network call, the daily update
check, and it is written to fail silent. A local model needs no internet either
-- only pulling models does. So "network down" is not a failure mode for either;
what actually breaks is the local server, the log file, or the agent's tools,
which is what these cover.
"""

import ast
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    CodexTail, Stats, Style, Tracker, diagnose, find_log, parse_line, render_codex,
)

PLAIN = Style(color=False, unicode_ok=False, width=100)


def strip_ansi(text):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestNoNetworkDependency(unittest.TestCase):
    """This started as "no network client, at all", and the shape of the rule
    has survived even though the list has grown to two.

    The exceptions are the update check and --proxy. A proxy cannot avoid an
    HTTP stack -- being one is the entire feature -- but it can avoid making
    every other run pay for it. So the rule is no longer "one import" but
    "nothing at module scope, and only in the functions that cannot work
    without it": starting llmwatch to watch a local model still loads no HTTP
    stack at all.
    """

    # Both are lazy, and both are checked below for being lazy.
    HTTP_HOLDERS = ["fetch_latest_version", "_proxy_server_class"]

    def source(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "llmwatch.py")
        with open(path) as fh:
            return fh.read()

    def test_no_third_party_network_client(self):
        """stdlib or nothing. A tool that watches a LOCAL model has no business
        pulling an HTTP library in as a dependency."""
        for module in ("import requests", "import httpx", "from urllib3",
                       "import aiohttp", "import socket"):
            self.assertNotIn(module, self.source(), module)

    def test_http_is_imported_only_where_it_cannot_be_avoided(self):
        tree = ast.parse(self.source())
        holders = [node.name for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and any(isinstance(inner, (ast.Import, ast.ImportFrom))
                           and ("urllib" in ast.dump(inner)
                                or "http" in ast.dump(inner))
                           for inner in ast.walk(node))]
        self.assertEqual(sorted(holders), sorted(self.HTTP_HOLDERS))

    def test_it_is_not_imported_at_module_scope(self):
        """Keeping the imports inside those functions means starting llmwatch
        never loads an HTTP stack unless the check or the proxy actually runs."""
        tree = ast.parse(self.source())
        top_level = [ast.dump(node) for node in tree.body
                     if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual([d for d in top_level if "urllib" in d], [])

    def test_the_url_carries_nothing_about_the_user(self):
        """The same URL for everyone: no query string, no version, no model."""
        from llmwatch import UPDATE_URL
        self.assertEqual(UPDATE_URL, "https://pypi.org/pypi/ollama-llmwatch/json")
        self.assertNotIn("?", UPDATE_URL)


class TestBrokenLogFile(unittest.TestCase):

    def test_missing_log_is_reported_not_crashed(self):
        os.environ["LLMWATCH_LOG"] = "/nonexistent/definitely/not/here.log"
        try:
            kind, target = find_log()
            self.assertEqual(kind, "file")     # honours the override, fails later cleanly
        finally:
            del os.environ["LLMWATCH_LOG"]

    def test_garbage_lines_never_raise(self):
        tracker = Tracker()
        for junk in ("", "\n", "\x00\x01binary", "slot print_timing: id | task |",
                     "prompt processing, n_tokens = notanumber, progress = x",
                     "a" * 10000):
            self.assertIsNone(parse_line(junk), junk[:40])
            tracker.feed(parse_line(junk))

    def test_truncated_numbers_do_not_produce_absurd_state(self):
        """A half-written line (the reader can catch a partial flush) must not
        register as a completed request."""
        tracker = Tracker()
        outs = tracker.feed(parse_line(
            "slot print_timing: id  0 | task 5 | prompt eval time =   1000.00 ms /"))
        self.assertEqual(outs, [])


class TestCodexFileFailures(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "rollout-test.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, records, mode="a"):
        with open(self.path, mode) as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

    def tail(self):
        return CodexTail(path=self.path)

    def call(self, name="exec_command", cmd="ls"):
        return {"payload": {"type": "function_call", "name": name,
                            "arguments": json.dumps({"cmd": cmd})}}

    def output(self, text):
        return {"payload": {"type": "function_call_output", "output": text}}

    def test_truncated_file_is_re_read_instead_of_going_silent(self):
        """A shrinking file would otherwise leave the offset past EOF, and the
        pane would show stale state forever."""
        tail = self.tail()
        tail.poll()                                      # attach: skips existing content
        self.write([self.call(cmd="a much longer original command here to grow the file")])
        tail.poll()
        self.assertIn("much longer", tail.detail)

        self.write([self.call(cmd="short")], mode="w")   # truncate to something smaller
        tail.poll()
        self.assertIn("short", tail.detail)

    def test_replaced_file_is_detected_by_inode(self):
        """Rotating a new file into place keeps the name but changes the inode;
        size alone cannot see it."""
        tail = self.tail()
        tail.poll()                                      # attach: skips existing content
        self.write([self.call(cmd="original")])
        tail.poll()
        self.assertIn("original", tail.detail)

        other = self.path + ".new"
        with open(other, "w") as fh:
            fh.write(json.dumps(self.call(cmd="replacement")) + "\n")
        os.replace(other, self.path)
        tail.poll()
        self.assertIn("replacement", tail.detail)

    def test_deleted_file_does_not_crash(self):
        tail = self.tail()
        self.write([self.call()])
        tail.poll()
        os.remove(self.path)
        tail.poll()          # must not raise

    def test_partial_line_at_end_is_ignored(self):
        tail = self.tail()
        with open(self.path, "w") as fh:
            fh.write('{"payload": {"type": "function_call", "name": "exec_comm')
        tail.poll()
        self.assertIsNone(tail.action)

    def test_no_codex_directory_at_all(self):
        tail = CodexTail(sessions_dir="/nonexistent/codex/sessions")
        tail.poll()
        self.assertIsNone(tail.state())


class TestBrokenToolDetection(unittest.TestCase):

    def test_real_codex_harness_error_is_detected(self):
        self.assertTrue(CodexTail.looks_like_failure(
            'failed to parse function arguments: invalid type: string "15000", '
            'expected u64 at line 1 column 215'))

    def test_shell_failures_detected(self):
        for text in ("bash: rg: command not found",
                     "cat: /nope.txt: No such file or directory",
                     "Permission denied",
                     "Process exited with code 2"):
            self.assertTrue(CodexTail.looks_like_failure(text), text)

    def test_successful_output_is_not_flagged(self):
        """The false-positive cases that a naive keyword match gets wrong."""
        for text in ("Process exited with code 0 Output: src/types/game.ts",
                     "src/errors/handler.ts\nsrc/error_codes.ts",
                     "3 files changed",
                     "",
                     None):
            self.assertFalse(CodexTail.looks_like_failure(text), repr(text))

    def test_repeated_identical_failure_is_counted(self):
        tail = CodexTail()
        for _ in range(3):
            tail._consume(json.dumps({"payload": {
                "type": "function_call_output", "output": "failed to parse arguments"}}))
        self.assertEqual(tail.error_repeats, 3)

    def test_a_success_clears_the_failure_streak(self):
        tail = CodexTail()
        tail._consume(json.dumps({"payload": {"type": "function_call_output",
                                              "output": "failed to parse"}}))
        tail._consume(json.dumps({"payload": {"type": "function_call_output",
                                              "output": "ok, 3 files"}}))
        self.assertIsNone(tail.error)
        self.assertEqual(tail.error_repeats, 0)

    def test_repeating_failure_is_diagnosed_as_the_cause(self):
        """A broken tool is WHY the agent loops; it should rank above the loop
        symptom, because it tells you what to go and fix."""
        data = {"event": "prefill_tick", "to_process": 1000, "rate": 100.0}
        codex = {"error": "failed to parse function arguments", "error_repeats": 3}
        findings = [strip_ansi(f) for f in
                    diagnose(data, {"looping": True, "repeat_count": 3}, PLAIN, codex)]
        self.assertIn("broken call", " ".join(findings))
        self.assertLess(" ".join(findings).index("broken call"),
                        len(" ".join(findings)))

    def test_single_failure_is_not_escalated(self):
        data = {"event": "prefill_tick", "to_process": 1000, "rate": 100.0}
        codex = {"error": "failed once", "error_repeats": 1}
        findings = " ".join(strip_ansi(f) for f in diagnose(data, {}, PLAIN, codex))
        self.assertNotIn("broken call", findings)

    def test_pane_shows_the_failure(self):
        state = {"action": "exec_command", "detail": "rg foo", "calls": 2,
                 "error": "failed to parse function arguments", "error_repeats": 3}
        blob = strip_ansi("\n".join(render_codex(state, PLAIN, 100)))
        self.assertIn("failing 3x", blob)
        self.assertIn("failed to parse", blob)


class TestDiagnoseEdges(unittest.TestCase):
    """Branches the coverage audit found untested."""

    def test_no_data_yields_no_findings(self):
        self.assertEqual(diagnose(None, {}, PLAIN), [])
        self.assertEqual(diagnose({}, {}, PLAIN), [])

    def test_status_is_the_fallback_when_nothing_notable(self):
        data = {"event": "prefill_tick", "to_process": 100, "cached": 0,
                "rate": 50.0, "status": "saving cache checkpoint 3/32"}
        findings = " ".join(strip_ansi(f) for f in diagnose(data, {}, PLAIN))
        self.assertIn("checkpoint 3/32", findings)

    def test_one_cancel_is_not_worth_warning_about(self):
        data = {"event": "prefill_tick", "to_process": 100, "rate": 50.0}
        findings = " ".join(strip_ansi(f) for f in
                            diagnose(data, {"recent_cancels": 1}, PLAIN))
        self.assertNotIn("timing out", findings)

    def test_two_cancels_is(self):
        data = {"event": "prefill_tick", "to_process": 100, "rate": 50.0}
        findings = " ".join(strip_ansi(f) for f in
                            diagnose(data, {"recent_cancels": 2}, PLAIN))
        self.assertIn("timing out", findings)

    def test_missing_rate_does_not_produce_a_bogus_cost_estimate(self):
        data = {"event": "prefill_tick", "to_process": 39528, "cache_miss": True}
        findings = " ".join(strip_ansi(f) for f in diagnose(data, {}, PLAIN))
        self.assertIn("cache gone", findings)
        self.assertNotIn("~", findings)      # no rate known, so no time claimed

    def test_empty_stats_snapshot_is_safe(self):
        snap = Stats(clock=lambda: 0.0).snapshot("unseen")
        data = {"event": "prefill_tick", "to_process": 100, "rate": 10.0}
        diagnose(data, snap, PLAIN)          # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestCodexAttachDoesNotReplayHistory(unittest.TestCase):
    """Attaching set offset=0, so a long-running session was replayed in full:
    the largest session file on the development machine was 5.97 MB, every
    historical tool call was counted, and an hours-old action displayed as if it
    were current."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "rollout-existing.jsonl")
        with open(self.path, "w") as fh:
            for i in range(500):
                fh.write(json.dumps({"payload": {
                    "type": "function_call", "name": "exec_command",
                    "arguments": json.dumps({"cmd": "historical command %d" % i})}}) + "\n")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_history_is_not_replayed_on_attach(self):
        tail = CodexTail(path=self.path)
        tail.poll()
        self.assertEqual(tail.calls, 0, "attaching replayed %d historical calls" % tail.calls)
        self.assertIsNone(tail.action)

    def test_new_activity_after_attach_is_still_picked_up(self):
        tail = CodexTail(path=self.path)
        tail.poll()
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"payload": {
                "type": "function_call", "name": "exec_command",
                "arguments": json.dumps({"cmd": "the current command"})}}) + "\n")
        tail.poll()
        self.assertEqual(tail.calls, 1)
        self.assertIn("current command", tail.detail)

    def test_attach_to_a_missing_file_is_safe(self):
        tail = CodexTail(path=os.path.join(self.dir, "nope.jsonl"))
        tail.poll()
        self.assertEqual(tail.offset, 0)
