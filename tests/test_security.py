"""Tests that untrusted text cannot reach the terminal as escape sequences.

Two things on screen are not authored by llmwatch: model names (from the log)
and agent tool-call arguments (from a Codex session file). Both can carry
control characters. `\\x1b[2J` clears the screen, `\\x1b]0;...\\x07` rewrites the
terminal title, and some terminals honour worse.

The realistic path is not a hand-crafted model name. An LLM writes tool-call
arguments; prompt injection from any page or file the agent reads can shape
them; Codex records them verbatim; llmwatch prints them.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    CodexTail, Style, Tracker, parse_line, render_codex, render_header, safe_text,
)

PLAIN = Style(color=False, unicode_ok=False, width=100)
FANCY = Style(color=True, unicode_ok=True, width=100)

PAYLOAD = "EVIL\x1b[2J\x1b[31mHACKED\x1b]0;pwned\x07"


class TestSafeText(unittest.TestCase):

    def test_strips_escape_sequences(self):
        cleaned = safe_text(PAYLOAD)
        self.assertNotIn("\x1b", cleaned)
        self.assertNotIn("\x07", cleaned)
        self.assertIn("EVIL", cleaned)      # the readable part survives

    def test_strips_newlines_and_tabs(self):
        """A newline in a one-line field would break the frame layout."""
        cleaned = safe_text("a\nb\tc\rd")
        self.assertEqual(cleaned, "abcd")

    def test_strips_c1_controls(self):
        self.assertEqual(safe_text("a\x9bb"), "ab")

    def test_leaves_ordinary_text_alone(self):
        self.assertEqual(safe_text("qwen3.8:27b-mtp-128k"), "qwen3.8:27b-mtp-128k")
        self.assertEqual(safe_text("rg -n 'foo bar' src/"), "rg -n 'foo bar' src/")

    def test_none_passes_through(self):
        self.assertIsNone(safe_text(None))

    def test_length_is_bounded(self):
        """An enormous field should not be able to blow up a frame."""
        self.assertLessEqual(len(safe_text("x" * 10000, limit=50)), 50)


class TestModelNameBoundary(unittest.TestCase):

    def test_malicious_model_name_is_sanitised_at_parse(self):
        ev = parse_line('time=x msg="template selection" model=registry/%s selected=y'
                        % PAYLOAD)
        self.assertIsNotNone(ev)
        self.assertNotIn("\x1b", ev.model)

    def test_it_cannot_reach_the_header(self):
        tracker = Tracker()
        tracker.feed(parse_line('msg="template selection" model=registry/%s x=1' % PAYLOAD))
        outs = tracker.feed(parse_line(
            "slot   operator(): id  0 | task 1 | new prompt, n_ctx_slot = 100, "
            "n_keep = 4, task.n_tokens = 50"))
        data = [o.data for o in outs if o.data.get("event") == "request_start"][0]
        self.assertNotIn("\x1b", render_header(data, PLAIN))

    def test_our_own_colour_still_works(self):
        """Sanitisation must not strip the escapes llmwatch itself adds."""
        self.assertIn("\x1b", FANCY.bold("styled"))


class TestToolCallBoundary(unittest.TestCase):
    """The path that a prompt injection could actually travel."""

    def tail_with(self, arguments):
        tail = CodexTail()
        tail._consume(json.dumps({"payload": {
            "type": "function_call", "name": "exec_command",
            "arguments": json.dumps({"cmd": arguments})}}))
        return tail

    def test_escape_in_tool_arguments_is_stripped(self):
        tail = self.tail_with("rm -rf / %s" % PAYLOAD)
        self.assertNotIn("\x1b", tail.detail)
        self.assertIn("rm -rf /", tail.detail)

    def test_escape_in_tool_name_is_stripped(self):
        tail = CodexTail()
        tail._consume(json.dumps({"payload": {
            "type": "function_call", "name": "exec%s" % PAYLOAD, "arguments": "{}"}}))
        self.assertNotIn("\x1b", tail.action)

    def test_escape_in_tool_error_is_stripped(self):
        tail = CodexTail()
        tail._consume(json.dumps({"payload": {
            "type": "function_call_output",
            "output": "failed to parse %s" % PAYLOAD}}))
        self.assertNotIn("\x1b", tail.error or "")

    def test_nothing_untrusted_survives_into_the_rendered_pane(self):
        tail = self.tail_with(PAYLOAD)
        rendered = "\n".join(render_codex(tail.state(), PLAIN, 100))
        self.assertNotIn("\x1b", rendered)


class TestNoShellInvocation(unittest.TestCase):

    def test_no_shell_true_anywhere(self):
        """A fixed command string is safe today, but shell=True is how a variable
        gets interpolated into a shell six months from now."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "llmwatch.py")
        with open(path) as fh:
            self.assertNotIn("shell=True", fh.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
