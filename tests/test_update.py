"""Tests for the update check.

This is the only network call in the program, added because 0.8.0 shipped to
GitHub and never reached PyPI, and nobody running an older copy could tell.
Every test here injects the clock, the cache path and the fetch, so the suite
still needs no network and touches no real home directory.

What these are really guarding is restraint: at most one call a day, nothing
about the user in it, silence on every failure, and never a word in --json.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    UPDATE_INTERVAL, Style, UIState, check_for_update, compose_frame,
    fetch_latest_version, handle_key, render_update, start_update_check,
    update_check_disabled, update_is_newer, upgrade_command, upgrade_plan,
    version_tuple,
)

PLAIN = Style(color=False, unicode_ok=False, width=100)


class TestVersionComparison(unittest.TestCase):

    def test_newer_is_newer(self):
        self.assertTrue(update_is_newer("0.9.0", "0.10.0"))
        self.assertTrue(update_is_newer("0.9.0", "1.0.0"))
        self.assertTrue(update_is_newer("0.9.0", "0.9.1"))

    def test_same_or_older_is_not(self):
        self.assertFalse(update_is_newer("0.9.0", "0.9.0"))
        self.assertFalse(update_is_newer("0.9.0", "0.8.9"))

    def test_numeric_not_lexicographic(self):
        """The bug this exists to prevent: "0.10.0" sorts before "0.9.0" as a
        string, so a string compare tells everyone to downgrade at 0.10."""
        self.assertTrue(update_is_newer("0.9.0", "0.10.0"))
        self.assertFalse(update_is_newer("0.10.0", "0.9.0"))

    def test_prereleases_are_refused_rather_than_guessed(self):
        """Ordering "1.0.0rc1" against "1.0.0" by guesswork is how an update
        prompt ends up recommending something older than what you have."""
        self.assertIsNone(version_tuple("1.0.0rc1"))
        self.assertIsNone(version_tuple("1.0.0.dev3"))
        self.assertFalse(update_is_newer("0.9.0", "1.0.0rc1"))

    def test_junk_is_never_an_update(self):
        for value in (None, "", "latest", "0.9.x", "not a version"):
            self.assertFalse(update_is_newer("0.9.0", value), repr(value))
        self.assertFalse(update_is_newer(None, "1.0.0"))


class TestOnlyHttpsIsFetched(unittest.TestCase):
    """urlopen follows file:, ftp: and custom schemes. The URL is a constant
    today, but the function takes one as a parameter, and a version check that
    can be pointed at a local path is a file read wearing a hat.
    """

    def test_non_https_schemes_are_refused_without_opening_anything(self):
        for url in ("file:///etc/passwd", "ftp://example.com/x",
                    "http://pypi.org/pypi/ollama-llmwatch/json", "", None):
            self.assertIsNone(fetch_latest_version(url=url, timeout=0.01), repr(url))


class TestCachingAndFailure(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "update-check.json")

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def fetch(self, value, counter):
        def call():
            counter.append(1)
            return value
        return call

    def test_first_run_fetches_and_reports(self):
        calls = []
        found = check_for_update("0.9.0", now=1000.0, path=self.path,
                                 fetch=self.fetch("0.10.0", calls))
        self.assertEqual(found, "0.10.0")
        self.assertEqual(len(calls), 1)

    def test_second_run_the_same_day_does_not_call_out(self):
        calls = []
        check_for_update("0.9.0", now=1000.0, path=self.path,
                         fetch=self.fetch("0.10.0", calls))
        found = check_for_update("0.9.0", now=1000.0 + UPDATE_INTERVAL - 1,
                                 path=self.path, fetch=self.fetch("0.10.0", calls))
        self.assertEqual(len(calls), 1, "cache should have served the second run")
        self.assertEqual(found, "0.10.0")

    def test_the_next_day_it_checks_again(self):
        calls = []
        check_for_update("0.9.0", now=1000.0, path=self.path,
                         fetch=self.fetch("0.10.0", calls))
        check_for_update("0.9.0", now=1000.0 + UPDATE_INTERVAL + 1,
                         path=self.path, fetch=self.fetch("0.10.0", calls))
        self.assertEqual(len(calls), 2)

    def test_a_failed_check_is_cached_too(self):
        """Otherwise a machine with no route out retries on every single launch,
        paying the timeout each time for an answer that will not change."""
        calls = []
        self.assertIsNone(check_for_update("0.9.0", now=1000.0, path=self.path,
                                           fetch=self.fetch(None, calls)))
        self.assertIsNone(check_for_update("0.9.0", now=1000.5, path=self.path,
                                           fetch=self.fetch(None, calls)))
        self.assertEqual(len(calls), 1)

    def test_a_raising_fetch_never_escapes(self):
        def explode():
            raise IOError("no route to host")
        box = {}
        thread = start_update_check(box, current="0.9.0", path=self.path, fetch=explode)
        if thread:
            thread.join(timeout=5)
        self.assertEqual(box, {})

    def test_a_corrupt_cache_is_survivable(self):
        with open(self.path, "w") as fh:
            fh.write("{not json at all")
        calls = []
        found = check_for_update("0.9.0", now=1000.0, path=self.path,
                                 fetch=self.fetch("0.10.0", calls))
        self.assertEqual(found, "0.10.0")

    def test_nothing_about_the_user_is_written_down(self):
        check_for_update("0.9.0", now=1000.0, path=self.path,
                         fetch=self.fetch("0.10.0", []))
        with open(self.path) as fh:
            stored = json.load(fh)
        self.assertEqual(set(stored), {"latest", "checked"})


class TestOptOut(unittest.TestCase):

    def setUp(self):
        self.previous = os.environ.pop("LLMWATCH_NO_UPDATE_CHECK", None)

    def tearDown(self):
        os.environ.pop("LLMWATCH_NO_UPDATE_CHECK", None)
        if self.previous is not None:
            os.environ["LLMWATCH_NO_UPDATE_CHECK"] = self.previous

    def test_on_by_default(self):
        self.assertFalse(update_check_disabled())

    def test_any_truthy_value_turns_it_off(self):
        for value in ("1", "true", "yes", "anything"):
            os.environ["LLMWATCH_NO_UPDATE_CHECK"] = value
            self.assertTrue(update_check_disabled(), value)

    def test_explicit_zero_leaves_it_on(self):
        for value in ("0", "false", "no", ""):
            os.environ["LLMWATCH_NO_UPDATE_CHECK"] = value
            self.assertFalse(update_check_disabled(), value)

    def test_opting_out_starts_no_thread_at_all(self):
        os.environ["LLMWATCH_NO_UPDATE_CHECK"] = "1"
        box = {}

        def fetch():
            raise AssertionError("opted out, but the check ran anyway")

        self.assertIsNone(start_update_check(box, fetch=fetch))
        self.assertEqual(box, {})


class TestUpgradeCommand(unittest.TestCase):
    """Four supported installs, four different upgrade commands.

    Naming the wrong one is worse than naming none: it fails in front of the
    user, and the next notice gets ignored.
    """

    def test_uv_tool(self):
        self.assertEqual(
            upgrade_command("/home/x/.local/share/uv/tools/ollama-llmwatch/"
                            "lib/python3.12/site-packages/llmwatch.py"),
            "uv tool upgrade ollama-llmwatch")

    def test_pipx(self):
        self.assertEqual(
            upgrade_command("/home/x/.local/pipx/venvs/ollama-llmwatch/"
                            "lib/python3.12/site-packages/llmwatch.py"),
            "pipx upgrade ollama-llmwatch")

    def test_plain_pip(self):
        self.assertEqual(
            upgrade_command("/usr/lib/python3/dist-packages/llmwatch.py"),
            "pip install --upgrade ollama-llmwatch")
        self.assertEqual(
            upgrade_command("/home/x/venv/lib/python3.12/site-packages/llmwatch.py"),
            "pip install --upgrade ollama-llmwatch")

    def test_uv_wins_over_the_site_packages_inside_it(self):
        """A uv tool install contains a site-packages directory, so the more
        specific marker has to be tested first or every uv user is told to
        run pip, which will not upgrade the tool they are running."""
        path = ("/home/x/.local/share/uv/tools/ollama-llmwatch/"
                "lib/python3.12/site-packages/llmwatch.py")
        self.assertIn("uv tool upgrade", upgrade_command(path))

    def test_a_single_downloaded_file_is_told_to_fetch_it_again(self):
        directory = tempfile.mkdtemp()
        try:
            command = upgrade_command(os.path.join(directory, "llmwatch.py"))
            self.assertIn("curl", command)
            self.assertIn("llmwatch.py", command)
        finally:
            os.rmdir(directory)

    def test_a_git_checkout_is_told_to_pull(self):
        directory = tempfile.mkdtemp()
        try:
            os.mkdir(os.path.join(directory, ".git"))
            self.assertEqual(upgrade_command(os.path.join(directory, "llmwatch.py")),
                             "git pull")
        finally:
            os.rmdir(os.path.join(directory, ".git"))
            os.rmdir(directory)


class TestUpgradePlan(unittest.TestCase):
    """What `u` would actually run, and when it refuses to run anything.

    Refusing is the feature. Pulling over somebody's uncommitted work, or
    upgrading into an interpreter they are not running, fails in a way that is
    far harder to understand than a sentence saying why it stopped.
    """

    HAVE = staticmethod(lambda tool: "/usr/bin/" + tool)
    MISSING = staticmethod(lambda tool: None)

    def test_uv_and_pipx_are_argument_lists_never_shell_strings(self):
        for path, expected in [
            ("/x/.local/share/uv/tools/ollama-llmwatch/lib/p/site-packages/llmwatch.py",
             ["uv", "tool", "upgrade", "ollama-llmwatch"]),
            ("/x/.local/pipx/venvs/ollama-llmwatch/lib/p/site-packages/llmwatch.py",
             ["pipx", "upgrade", "ollama-llmwatch"]),
        ]:
            argv, blocker = upgrade_plan(path, which=self.HAVE)
            self.assertIsNone(blocker)
            self.assertEqual(argv, expected)

    def test_pip_upgrades_the_interpreter_that_is_running(self):
        """A bare `pip` may belong to a different interpreter entirely, and
        upgrading into one you are not running is the most confusing possible
        outcome: the notice never goes away."""
        argv, blocker = upgrade_plan("/x/venv/lib/python3.12/site-packages/llmwatch.py",
                                     which=self.HAVE)
        self.assertIsNone(blocker)
        self.assertEqual(argv[:3], [sys.executable, "-m", "pip"])

    def test_a_missing_tool_blocks_rather_than_failing_later(self):
        argv, blocker = upgrade_plan(
            "/x/.local/share/uv/tools/ollama-llmwatch/lib/p/site-packages/llmwatch.py",
            which=self.MISSING)
        self.assertIsNone(argv)
        self.assertIn("uv", blocker)

    def test_a_dirty_checkout_is_never_pulled_over(self):
        directory = tempfile.mkdtemp()
        try:
            os.mkdir(os.path.join(directory, ".git"))
            argv, blocker = upgrade_plan(os.path.join(directory, "llmwatch.py"),
                                         which=self.HAVE)
            # Not a real repository, so git cannot report its state, and the
            # answer to "I do not know" is the same as to "yes, dirty".
            self.assertIsNone(argv)
            self.assertTrue(blocker)
        finally:
            os.rmdir(os.path.join(directory, ".git"))
            os.rmdir(directory)

    def test_a_single_file_is_replaced_where_it_lives(self):
        """`curl -O` writes to the current directory, which is where you
        happened to be standing, not where the program is."""
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "llmwatch.py")
            argv, blocker = upgrade_plan(path, which=self.HAVE)
            self.assertIsNone(blocker)
            self.assertIn("-o", argv)
            self.assertEqual(argv[argv.index("-o") + 1], path)
        finally:
            os.rmdir(directory)

    def test_nothing_in_the_command_comes_from_outside(self):
        """Every argument is a constant or a path this program already knows.
        Nothing from a log line, a model name or a version reaches it."""
        for path in ("/x/.local/share/uv/tools/ollama-llmwatch/lib/p/site-packages/llmwatch.py",
                     "/x/.local/pipx/venvs/ollama-llmwatch/lib/p/site-packages/llmwatch.py",
                     "/x/venv/lib/python3.12/site-packages/llmwatch.py"):
            argv, _ = upgrade_plan(path, which=self.HAVE)
            for argument in argv:
                self.assertNotIn(";", argument)
                self.assertNotIn("&", argument)
                self.assertNotIn("|", argument)


class TestUpgradeKeyRequiresConfirmation(unittest.TestCase):

    def test_u_does_nothing_when_there_is_no_update(self):
        """Otherwise the key invites people to reinstall what they already
        have, and the pane has nothing true to put in its heading."""
        state = UIState()
        self.assertFalse(handle_key(state, "u", [], update=None))
        self.assertEqual(state.view, "live")

    def test_u_opens_the_confirmation_rather_than_upgrading(self):
        state = UIState()
        self.assertTrue(handle_key(state, "u", [], update="999.0.0"))
        self.assertEqual(state.view, "upgrade")
        self.assertFalse(state.upgrade_requested)

    def test_the_same_key_cannot_open_and_confirm(self):
        """One keystroke opens, a different one commits. Nobody should change
        what is installed by leaning on the keyboard."""
        state = UIState()
        handle_key(state, "u", [], update="999.0.0")
        handle_key(state, "u", [], update="999.0.0")
        self.assertFalse(state.upgrade_requested)
        self.assertEqual(state.view, "live")

    def test_y_confirms(self):
        state = UIState()
        handle_key(state, "u", [], update="999.0.0")
        handle_key(state, "y", [], update="999.0.0")
        self.assertTrue(state.upgrade_requested)

    def test_escape_backs_out(self):
        for key in ("ESC", "n"):
            state = UIState()
            handle_key(state, "u", [], update="999.0.0")
            handle_key(state, key, [], update="999.0.0")
            self.assertFalse(state.upgrade_requested, key)
            self.assertEqual(state.view, "live", key)

    def test_quit_still_works_from_the_confirmation(self):
        state = UIState()
        handle_key(state, "u", [], update="999.0.0")
        with self.assertRaises(KeyboardInterrupt):
            handle_key(state, "q", [], update="999.0.0")


class TestDisplay(unittest.TestCase):

    def snapshot(self):
        phase = {"peak": 100.0, "avg": 90.0, "low": 80.0,
                 "tokens": 1000, "seconds": 10.0, "recent": []}
        return {"model": "m", "requests": 1, "session_seconds": 10.0,
                "prefill": dict(phase), "generation": dict(phase), "recent": []}

    def test_the_notice_names_the_versions_and_the_command(self):
        """Not a hardcoded command: which one is right depends on how this copy
        was installed, and the suite runs from a checkout where it is `git
        pull` rather than any package manager."""
        line = render_update("999.0.0", PLAIN)
        self.assertIn("999.0.0", line)
        self.assertIn(upgrade_command(), line)

    def test_it_refuses_to_advertise_a_downgrade(self):
        """Belt and braces against a future caller passing the raw fetch result
        instead of the checked one. Telling someone on 0.9.0 to "upgrade" to
        0.7.0 is worse than never mentioning updates at all."""
        self.assertIsNone(render_update("0.0.1", PLAIN))
        self.assertIsNone(render_update("not a version", PLAIN))

    def test_nothing_is_drawn_without_an_update(self):
        self.assertIsNone(render_update(None, PLAIN))
        frame = "\n".join(compose_frame(self.snapshot(), "waiting", PLAIN, 90, 30))
        self.assertNotIn("update available", frame)

    def test_the_frame_shows_it_when_there_is_one(self):
        frame = "\n".join(compose_frame(self.snapshot(), "waiting", PLAIN, 90, 30,
                                        update="999.0.0"))
        self.assertIn("update available: 999.0.0", frame)

    def test_it_never_costs_the_live_line(self):
        """On a terminal too short for both, the thing you are watching wins."""
        frame = compose_frame(self.snapshot(), "LIVE-MARKER", PLAIN, 90, 6,
                              update="999.0.0")
        blob = "\n".join(frame)
        self.assertIn("LIVE-MARKER", blob)
        self.assertNotIn("update available", blob)


if __name__ == "__main__":
    unittest.main()
