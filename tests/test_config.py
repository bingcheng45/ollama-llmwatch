"""Tests for the saved settings file.

The point of the file is that `ollama-llmwatch` on its own does what you
configured, so the thing that has to be right is precedence. Anything typed now
beats anything decided earlier, or the file silently overrules the command and
the flag looks broken.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    DEFAULT_PROXY_PORT, DEFAULT_UPSTREAM, config_path, effective_settings,
    read_config, write_config,
)


class TestConfigPath(unittest.TestCase):

    def test_it_follows_xdg_like_the_history_file_does(self):
        old = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = "/tmp/somewhere"
        try:
            self.assertEqual(
                config_path(),
                "/tmp/somewhere/ollama-llmwatch/config.json")
        finally:
            if old is None:
                os.environ.pop("XDG_CONFIG_HOME", None)
            else:
                os.environ["XDG_CONFIG_HOME"] = old

    def test_it_falls_back_to_dot_config(self):
        old = os.environ.pop("XDG_CONFIG_HOME", None)
        try:
            self.assertTrue(config_path().endswith(
                "/.config/ollama-llmwatch/config.json"))
        finally:
            if old is not None:
                os.environ["XDG_CONFIG_HOME"] = old


class TestReadingAndWriting(unittest.TestCase):

    def path(self):
        d = tempfile.mkdtemp()
        return os.path.join(d, "config.json")

    def test_a_missing_file_is_empty_settings_not_an_error(self):
        """Nobody has configured anything on a fresh install, which is the
        normal case rather than a problem."""
        self.assertEqual(read_config("/nonexistent/nope.json"), {})

    def test_junk_is_ignored_rather_than_fatal(self):
        """A settings file must never be the reason the tool will not start."""
        p = self.path()
        with open(p, "w") as fh:
            fh.write("{not json at all")
        self.assertEqual(read_config(p), {})

    def test_a_non_object_is_ignored(self):
        p = self.path()
        with open(p, "w") as fh:
            json.dump([1, 2, 3], fh)
        self.assertEqual(read_config(p), {})

    def test_a_round_trip(self):
        p = self.path()
        self.assertTrue(write_config({"watch": "proxy", "proxy_port": 8099}, p))
        self.assertEqual(read_config(p),
                         {"watch": "proxy", "proxy_port": 8099})

    def test_writing_creates_the_directory(self):
        p = os.path.join(tempfile.mkdtemp(), "nested", "deep", "config.json")
        self.assertTrue(write_config({"watch": "ollama"}, p))
        self.assertEqual(read_config(p), {"watch": "ollama"})

    def test_an_unwritable_path_reports_failure_rather_than_raising(self):
        self.assertFalse(write_config({"watch": "ollama"}, "/proc/x/y.json"))

    def test_unknown_keys_are_dropped(self):
        """The file is edited by hand as often as not, and a typo should not
        become a setting that silently does nothing forever."""
        p = self.path()
        with open(p, "w") as fh:
            json.dump({"watch": "proxy", "notAThing": 1}, fh)
        self.assertEqual(read_config(p), {"watch": "proxy"})

    def test_an_invalid_watch_mode_is_dropped(self):
        p = self.path()
        with open(p, "w") as fh:
            json.dump({"watch": "telepathy"}, fh)
        self.assertEqual(read_config(p), {})


class Args:
    def __init__(self, proxy=None, log=None, upstream=None):
        self.proxy = proxy
        self.log = log
        self.upstream = upstream


class TestPrecedence(unittest.TestCase):
    """Typed now beats set once. In order: flag, environment, saved file,
    built-in default."""

    def resolve(self, args=None, env=None, config=None):
        return effective_settings(args or Args(), env or {}, config or {})

    def test_nothing_configured_watches_ollama(self):
        s = self.resolve()
        self.assertEqual(s["watch"][0], "ollama")
        self.assertEqual(s["watch"][1], "default")

    def test_the_saved_file_is_enough_to_proxy(self):
        """The whole point: no flag, no variable, and it proxies."""
        s = self.resolve(config={"watch": "proxy"})
        self.assertEqual(s["watch"], ("proxy", "config"))

    def test_the_environment_beats_the_file(self):
        s = self.resolve(env={"LLMWATCH_PROXY": "8081"},
                         config={"watch": "ollama"})
        self.assertEqual(s["watch"], ("proxy", "env"))

    def test_a_flag_beats_the_environment_and_the_file(self):
        s = self.resolve(args=Args(proxy=""),
                         env={"LLMWATCH_PROXY": "9999"},
                         config={"watch": "log", "log": "/tmp/x.log"})
        self.assertEqual(s["watch"], ("proxy", "flag"))

    def test_an_explicit_log_flag_beats_a_configured_proxy(self):
        """Same rule that stopped an exported LLMWATCH_PROXY overruling --log:
        a saved setting is just as ambient as a variable."""
        s = self.resolve(args=Args(log="/tmp/llama.log"),
                         config={"watch": "proxy"})
        self.assertEqual(s["watch"], ("log", "flag"))

    def test_a_configured_log_path_is_used_without_any_flag(self):
        s = self.resolve(config={"watch": "log", "log": "/tmp/llama.log"})
        self.assertEqual(s["watch"], ("log", "config"))
        self.assertEqual(s["log"], ("/tmp/llama.log", "config"))

    def test_the_upstream_and_port_come_from_the_file_too(self):
        s = self.resolve(config={"watch": "proxy", "proxy_port": 8099,
                                 "upstream": "http://127.0.0.1:1234"})
        self.assertEqual(s["proxy_port"], (8099, "config"))
        self.assertEqual(s["upstream"], ("http://127.0.0.1:1234", "config"))

    def test_defaults_fill_the_rest(self):
        s = self.resolve(config={"watch": "proxy"})
        self.assertEqual(s["proxy_port"], (DEFAULT_PROXY_PORT, "default"))
        self.assertEqual(s["upstream"], (DEFAULT_UPSTREAM, "default"))

    def test_every_setting_reports_where_it_came_from(self):
        """The settings pane shows this, because "why is it doing that" is the
        question a settings file creates."""
        s = self.resolve(config={"watch": "proxy"})
        for key, (_value, source) in s.items():
            self.assertIn(source, ("flag", "env", "config", "default"), key)



from llmwatch import (  # noqa: E402
    Style, render_settings, settings_config, settings_key, settings_open,
)

PLAIN = Style(color=False, unicode_ok=False, width=100)


def opened(**over):
    base = {"watch": ("ollama", "default"),
            "proxy_port": (8081, "default"),
            "upstream": ("http://127.0.0.1:8080", "default"),
            "log": (None, "default")}
    base.update({k: v for k, v in over.items()})
    return settings_open(base)


class TestSettingsPane(unittest.TestCase):
    """Flags are for scripts and agents and still win. This is for the person
    who should not have to learn them."""

    def press(self, state, *keys):
        action = None
        for key in keys:
            state, action = settings_key(state, key)
        return state, action

    def test_a_number_picks_the_mode(self):
        state, _ = self.press(opened(), "2")
        self.assertEqual(state["mode"], "proxy")
        self.assertTrue(state["dirty"])

    def test_escape_closes_and_w_saves(self):
        self.assertEqual(self.press(opened(), "\x1b")[1], "close")
        self.assertEqual(self.press(opened(), "w")[1], "save")

    def test_enter_on_a_text_row_starts_editing_prefilled(self):
        state, _ = self.press(opened(), "j", "\r")
        self.assertEqual(state["editing"], "upstream")
        self.assertEqual(state["buffer"], "http://127.0.0.1:8080")

    def test_typing_while_editing_is_not_taken_as_shortcuts(self):
        """`w` is save and `1` picks a mode, and a path contains both. Editing
        has to be modal or a path cannot be typed at all."""
        state, action = self.press(opened(), "j", "j", "\r", "w", "1")
        self.assertIsNone(action)
        self.assertTrue(state["buffer"].endswith("w1"))
        self.assertEqual(state["mode"], "ollama")

    def test_enter_keeps_the_edit_and_escape_drops_it(self):
        state, _ = self.press(opened(), "j", "j", "\r")
        state["buffer"] = "/tmp/a.log"
        state, _ = self.press(state, "\r")
        self.assertEqual(state["log"], "/tmp/a.log")

        state, _ = self.press(state, "\r")
        state["buffer"] = "/tmp/discarded.log"
        state, _ = self.press(state, "\x1b")
        self.assertEqual(state["log"], "/tmp/a.log")
        self.assertIsNone(state["editing"])

    def test_backspace_deletes(self):
        state, _ = self.press(opened(), "j", "j", "\r")
        state["buffer"] = "abc"
        state, _ = self.press(state, "\x7f")
        self.assertEqual(state["buffer"], "ab")

    def test_a_port_that_is_not_a_number_is_refused_not_stored(self):
        state = opened()
        state["editing"], state["buffer"] = "proxy_port", "not-a-port"
        state, _ = self.press(state, "\r")
        self.assertEqual(state["proxy_port"], 8081)
        self.assertIn("number", state["message"])

    def test_what_gets_saved_is_only_what_the_pane_sets(self):
        state, _ = self.press(opened(), "2")
        data = settings_config(state)
        self.assertEqual(data["watch"], "proxy")
        self.assertIn("proxy_port", data)
        self.assertNotIn("cursor", data)
        self.assertNotIn("editing", data)

    def test_a_saved_pane_round_trips_through_the_file(self):
        state, _ = self.press(opened(), "2")
        path = os.path.join(tempfile.mkdtemp(), "config.json")
        self.assertTrue(write_config(settings_config(state), path))
        self.assertEqual(read_config(path)["watch"], "proxy")


class TestSettingsRendering(unittest.TestCase):

    def test_it_says_which_mode_is_in_force_and_where_it_came_from(self):
        text = "\n".join(render_settings(
            opened(watch=("proxy", "config")), PLAIN))
        self.assertIn("An OpenAI server", text)
        self.assertIn("config", text)

    def test_it_offers_what_it_found_running(self):
        text = "\n".join(render_settings(
            opened(), PLAIN, detected=[":8080 llama.cpp"]))
        self.assertIn(":8080 llama.cpp", text)

    def test_unsaved_changes_are_called_out(self):
        state, _ = settings_key(opened(), "2")
        text = "\n".join(render_settings(state, PLAIN))
        self.assertIn("unsaved", text)

    def test_a_rejected_port_is_shown_rather_than_swallowed(self):
        state = opened()
        state["editing"], state["buffer"] = "proxy_port", "abc"
        state, _ = settings_key(state, "\r")
        self.assertIn("number", "\n".join(render_settings(state, PLAIN)))


class TestTheSourceColumnStaysHonest(unittest.TestCase):
    """It exists to answer "why is it doing that", so it cannot go on saying
    `default` about a value typed a moment ago."""

    def test_a_typed_value_says_typed(self):
        state = opened()
        state["editing"], state["buffer"] = "log", "/tmp/a.log"
        state, _ = settings_key(state, "\r")
        text = "\n".join(render_settings(state, PLAIN))
        self.assertIn("typed", text)

    def test_an_untouched_value_still_reports_its_real_source(self):
        state = opened(upstream=("http://127.0.0.1:1234", "config"))
        text = "\n".join(render_settings(state, PLAIN))
        self.assertIn("config", text)

if __name__ == "__main__":
    unittest.main()
