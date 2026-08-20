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
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    DEFAULT_PROXY_PORT, DEFAULT_UPSTREAM, LOG_SEARCH_MAX_AGE, config_path,
    effective_settings,
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
    """Arrows, enter, space, escape. Nothing else.

    A letter cannot be both a shortcut and half a filename, and a settings
    screen that needs its own key chart has failed at the one job it has.
    """

    def press(self, state, *keys):
        action = None
        for key in keys:
            state, action = settings_key(state, key)
        return state, action

    def test_it_opens_on_a_menu_not_a_wall_of_fields(self):
        state = opened()
        self.assertEqual(state["level"], "top")
        text = "\n".join(render_settings(state, PLAIN))
        self.assertIn("What you are running", text)
        self.assertIn("Where to find it", text)
        self.assertNotIn("upstream", text)

    def test_arrows_move_and_stop_at_the_ends(self):
        state, _ = self.press(opened(), "UP")
        self.assertEqual(state["cursor"], 0)
        state, _ = self.press(state, "DOWN", "DOWN", "DOWN")
        self.assertEqual(state["cursor"], 1)

    def test_enter_opens_a_category_and_escape_comes_back(self):
        state, _ = self.press(opened(), "\r")
        self.assertEqual(state["level"], "watch")
        state, _ = self.press(state, "ESC")
        self.assertEqual(state["level"], "top")

    def test_escape_at_the_top_closes(self):
        self.assertEqual(self.press(opened(), "ESC")[1], "close")

    def test_space_works_wherever_enter_does(self):
        state, _ = self.press(opened(), " ")
        self.assertEqual(state["level"], "watch")

    def test_choosing_what_you_run_sets_everything_that_implies(self):
        """The point of a preset: one choice, not three."""
        state, action = self.press(opened(), "\r", "DOWN", "\r")
        self.assertEqual(state["preset"], "mlx")
        self.assertEqual(state["mode"], "proxy")
        self.assertEqual(state["upstream"], "http://127.0.0.1:8080")
        self.assertEqual(action, "save")
        self.assertEqual(state["level"], "top")

    def test_lm_studio_brings_its_own_port(self):
        """Nobody should have to know LM Studio is 1234."""
        state = opened()
        state["cursor"] = 0
        state, _ = self.press(state, "\r")
        state["cursor"] = 3
        state, _ = self.press(state, "\r")
        self.assertEqual(state["upstream"], "http://127.0.0.1:1234")

    def test_choosing_saves_without_a_separate_step(self):
        _state, action = self.press(opened(), "\r", "\r")
        self.assertEqual(action, "save")

    def test_no_letter_does_anything_at_the_menu(self):
        for key in ("w", "a", "1", "2", "3", "j", "k", "q", "s"):
            state, action = self.press(opened(), key)
            self.assertIsNone(action, key)
            self.assertEqual(state["level"], "top", key)
            self.assertEqual(state["cursor"], 0, key)

    def test_a_path_is_typed_only_after_choosing_to_edit_it(self):
        state, _ = self.press(opened(), "DOWN", "\r")
        self.assertEqual(state["level"], "where")
        state, _ = self.press(state, "DOWN", "DOWN", "DOWN", "\r")
        self.assertEqual(state["editing"], "log")
        for ch in "/tmp/a.log":
            state, _ = self.press(state, ch)
        state, action = self.press(state, "\r")
        self.assertEqual(state["log"], "/tmp/a.log")
        self.assertEqual(action, "save")

    def test_letters_typed_into_a_field_stay_in_the_field(self):
        state, _ = self.press(opened(), "DOWN", "\r", "DOWN", "DOWN", "DOWN", "\r")
        state, action = self.press(state, "w", "a", "1")
        self.assertIsNone(action)
        self.assertTrue(state["buffer"].endswith("wa1"))

    def test_escape_while_typing_drops_the_edit_not_the_screen(self):
        state, _ = self.press(opened(), "DOWN", "\r", "DOWN", "DOWN", "DOWN", "\r")
        state["buffer"] = "/tmp/nope.log"
        state, action = self.press(state, "ESC")
        self.assertIsNone(action)
        self.assertIsNone(state["editing"])
        self.assertEqual(state["log"], "")

    def test_a_port_that_is_not_a_number_is_refused(self):
        state = opened()
        state["level"], state["editing"], state["buffer"] = (
            "where", "proxy_port", "not-a-port")
        state, action = self.press(state, "\r")
        self.assertEqual(state["proxy_port"], 8081)
        self.assertIsNone(action)
        self.assertIn("number", state["message"])

    def test_the_menu_shows_the_current_answer_in_the_same_words(self):
        state = opened(watch=("proxy", "config"),
                       upstream=("http://127.0.0.1:1234", "config"))
        text = "\n".join(render_settings(state, PLAIN))
        self.assertIn("LM Studio", text)

    def test_what_gets_saved_is_only_what_the_pane_sets(self):
        state, _ = self.press(opened(), "\r", "DOWN", "\r")
        data = settings_config(state)
        self.assertEqual(data["watch"], "proxy")
        self.assertNotIn("cursor", data)
        self.assertNotIn("level", data)

    def test_a_saved_pane_round_trips_through_the_file(self):
        state, _ = self.press(opened(), "\r", "DOWN", "\r")
        path = os.path.join(tempfile.mkdtemp(), "config.json")
        self.assertTrue(write_config(settings_config(state), path))
        self.assertEqual(read_config(path)["watch"], "proxy")


class TestSettingsRendering(unittest.TestCase):

    def test_each_level_names_only_the_keys_that_work_there(self):
        state = opened()
        self.assertIn("enter open", "\n".join(render_settings(state, PLAIN)))
        state, _ = settings_key(state, "\r")
        self.assertIn("enter choose", "\n".join(render_settings(state, PLAIN)))
        state, _ = settings_key(state, "ESC")
        state, _ = settings_key(state, "DOWN")
        state, _ = settings_key(state, "\r")
        self.assertIn("enter edit", "\n".join(render_settings(state, PLAIN)))

    def test_the_chosen_option_is_marked(self):
        state = opened()
        state, _ = settings_key(state, "\r")
        self.assertIn("* Ollama", "\n".join(render_settings(state, PLAIN)))

    def test_it_offers_what_it_found_running(self):
        text = "\n".join(render_settings(opened(), PLAIN,
                                         detected=[":8080 llama.cpp"]))
        self.assertIn(":8080 llama.cpp", text)

    def test_a_typed_value_says_typed(self):
        state = opened()
        state["level"] = "where"
        state["editing"], state["buffer"] = "log", "/tmp/a.log"
        state, _ = settings_key(state, "\r")
        self.assertIn("typed", "\n".join(render_settings(state, PLAIN)))

    def test_speculative_variants_are_not_offered_as_a_choice(self):
        """MTP and the rest are not a way of connecting, and there is nothing
        to pick: llmwatch reads acceptance off whichever server is in front."""
        state, _ = settings_key(opened(), "\r")
        text = "\n".join(render_settings(state, PLAIN)).lower()
        for word in ("mtp", "eagle", "dflash", "speculative"):
            self.assertNotIn(word, text)


from llmwatch import backend_suggestion  # noqa: E402


class TestNoticingTheWrongBackend(unittest.TestCase):
    """Picking the backend is the one thing that cannot be got wrong quietly,
    because every wrong choice looks identical: an empty board. So when the
    chosen one has nothing to show and another plainly does, say so.

    Only ever a suggestion. Switching under someone silently would be worse
    than the empty board, since the numbers would change meaning without
    anything on screen admitting it.
    """

    def test_nothing_to_say_when_the_current_backend_is_working(self):
        self.assertIsNone(backend_suggestion(
            "ollama", {"models_loaded": 1, "oai_port": 8080}, busy=True))

    def test_ollama_with_nothing_loaded_and_a_server_running_suggests_it(self):
        got = backend_suggestion(
            "ollama", {"models_loaded": 0, "oai_port": 8080}, busy=False)
        self.assertEqual(got["mode"], "proxy")
        self.assertIn("8080", got["why"])

    def test_a_proxy_that_cannot_reach_its_server_suggests_ollama(self):
        got = backend_suggestion(
            "proxy", {"upstream_ok": False, "models_loaded": 2}, busy=False)
        self.assertEqual(got["mode"], "ollama")

    def test_a_proxy_with_nothing_else_running_says_nothing(self):
        """No suggestion is better than a bad one: if Ollama is empty too,
        switching solves nothing."""
        self.assertIsNone(backend_suggestion(
            "proxy", {"upstream_ok": False, "models_loaded": 0}, busy=False))

    def test_a_stale_log_with_a_server_running_suggests_the_proxy(self):
        got = backend_suggestion(
            "log", {"oai_port": 8080}, busy=False)
        self.assertEqual(got["mode"], "proxy")

    def test_a_busy_backend_is_never_second_guessed(self):
        """Traffic is arriving. Whatever else is running is not the point."""
        for mode in ("ollama", "proxy", "log"):
            self.assertIsNone(backend_suggestion(
                mode, {"models_loaded": 0, "oai_port": 8080,
                       "upstream_ok": False}, busy=True), mode)

    def test_it_survives_an_empty_system_snapshot(self):
        self.assertIsNone(backend_suggestion("ollama", None, busy=False))
        self.assertIsNone(backend_suggestion("ollama", {}, busy=False))


class TestTheSuggestionReachesTheUser(unittest.TestCase):

    def test_the_idle_line_offers_it_and_says_which_key(self):
        from llmwatch import render_idle
        line = render_idle(90.0, PLAIN, {
            "models_loaded": 0,
            "suggestion": {"mode": "proxy",
                           "why": "an OpenAI server is answering on :8080"}})
        self.assertIn("8080", line)
        self.assertIn("press s", line)

    def test_the_pane_shows_it_without_inventing_a_shortcut_for_it(self):
        """It is shown on the menu, and taken by choosing it like anything
        else. A one-key accept would be the only letter in the interface, and
        the reason for that letter would have to be remembered."""
        state = opened()
        state["suggestion"] = {"mode": "proxy", "why": "server on :8080"}
        text = "\n".join(render_settings(state, PLAIN))
        self.assertIn("server on :8080", text)
        self.assertNotIn("accept", text)

    def test_no_letter_takes_the_suggestion(self):
        state = opened()
        state["suggestion"] = {"mode": "proxy", "why": "x"}
        for key in ("a", "y", "s"):
            after, action = settings_key(state, key)
            self.assertIsNone(action, key)
            self.assertEqual(after["mode"], "ollama", key)

    def test_the_suggestion_only_shows_on_the_menu(self):
        """Deep in a list of options it would be noise: the choice is already
        on screen."""
        state = opened()
        state["suggestion"] = {"mode": "proxy", "why": "server on :8080"}
        state, _ = settings_key(state, "\r")
        self.assertNotIn("server on :8080",
                         "\n".join(render_settings(state, PLAIN)))


from llmwatch import (  # noqa: E402
    describe_backend, discover_backends, identify_log,
)


class TestFindingItForYou(unittest.TestCase):
    """"Where is it" assumes you know a port or a path. For anyone who does
    not, that is the end of the road, so the first row offers to look."""

    def write(self, name, text):
        d = tempfile.mkdtemp()
        path = os.path.join(d, name)
        with open(path, "w") as fh:
            fh.write(text)
        return d, path

    def test_a_log_is_identified_by_reading_it_not_by_its_name(self):
        """The name is whatever the person redirecting stderr felt like
        typing, so it says nothing."""
        _d, path = self.write("anything-at-all.log",
            "x I slot print_timing: id  3 | task 44 | draft acceptance = 0.57818"
            " (  159 accepted /   275 generated), mean len =  4.97\n")
        self.assertEqual(identify_log(path), "llama.cpp")

    def test_an_unrelated_log_is_not_claimed(self):
        _d, path = self.write("nginx.log", "127.0.0.1 - - [x] GET / 200\n")
        self.assertIsNone(identify_log(path))

    def test_a_missing_or_unreadable_file_is_not_an_error(self):
        self.assertIsNone(identify_log("/nonexistent/nope.log"))
        self.assertIsNone(identify_log("/"))

    def test_binary_rubbish_does_not_crash_the_scan(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "x.log")
        with open(path, "wb") as fh:
            fh.write(bytes(range(256)) * 4)
        self.assertIsNone(identify_log(path))

    def test_a_stale_log_is_not_offered(self):
        """A log from last week is not what is running now."""
        d, path = self.write("old.log",
            "x I slot print_timing: id  1 | task 1 | draft acceptance = 0.5"
            " (  1 accepted /   2 generated), mean len =  1.00\n")
        old = os.path.getmtime(path) + LOG_SEARCH_MAX_AGE + 60
        found = discover_backends(dirs=(d,), ports=(), now=old)
        self.assertEqual(found, [])

    def test_the_freshest_log_comes_first(self):
        d = tempfile.mkdtemp()
        line = ("x I slot print_timing: id  1 | task 1 | draft acceptance = 0.5"
                " (  1 accepted /   2 generated), mean len =  1.00\n")
        for name, when in (("old.log", 5000), ("new.log", 10)):
            path = os.path.join(d, name)
            with open(path, "w") as fh:
                fh.write(line)
            os.utime(path, (time.time() - when, time.time() - when))
        found = discover_backends(dirs=(d,), ports=())
        self.assertTrue(found[0]["path"].endswith("new.log"))

    def test_each_result_carries_what_to_do_about_it(self):
        """The pane applies `apply` and has no opinions of its own."""
        _d, path = self.write("a.log",
            "x I slot print_timing: id  1 | task 1 | draft acceptance = 0.5"
            " (  1 accepted /   2 generated), mean len =  1.00\n")
        row = discover_backends(dirs=(os.path.dirname(path),), ports=())[0]
        self.assertEqual(row["apply"], {"watch": "log", "log": path})
        self.assertIn("llama.cpp", describe_backend(row))


class TestTheScanDoesNotFindItself(unittest.TestCase):

    def test_our_own_listening_port_is_skipped(self):
        """llmwatch's proxy answers /v1/models and relays the upstream's Server
        header, so it looks exactly like whatever is behind it. Offering it
        would point llmwatch at itself."""
        import http.server
        import threading

        body = json.dumps({"object": "list", "data": [{"id": "m"}]}).encode()

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        port = srv.server_address[1]

        self.assertTrue(discover_backends(dirs=(), ports=(port,)))
        self.assertEqual(
            discover_backends(dirs=(), ports=(port,), skip_ports=(port,)), [])


class TestFindingItFromThePane(unittest.TestCase):

    def press(self, state, *keys):
        action = None
        for key in keys:
            state, action = settings_key(state, key)
        return state, action

    def test_the_first_row_offers_to_look(self):
        state, _ = self.press(opened(), "DOWN", "\r")
        text = "\n".join(render_settings(state, PLAIN))
        self.assertIn("I do not know", text)

    def test_choosing_it_asks_the_loop_to_scan(self):
        """The pane opens no sockets and reads no files; it says what it wants
        and the loop does it."""
        state, action = self.press(opened(), "DOWN", "\r", "\r")
        self.assertEqual(action, "discover")

    def test_picking_a_result_applies_everything_it_implies(self):
        state = opened()
        state["level"], state["cursor"] = "found", 0
        state["found"] = [{"kind": "log", "engine": "llama.cpp",
                           "path": "/tmp/x.log", "age": 5.0,
                           "apply": {"watch": "log", "log": "/tmp/x.log"}}]
        state, action = self.press(state, "\r")
        self.assertEqual(state["mode"], "log")
        self.assertEqual(state["log"], "/tmp/x.log")
        self.assertEqual(action, "save")
        self.assertEqual(state["level"], "top")

    def test_finding_nothing_says_so_rather_than_showing_an_empty_box(self):
        state = opened()
        state["level"], state["found"] = "found", []
        text = "\n".join(render_settings(state, PLAIN))
        self.assertIn("nothing found", text)

    def test_escape_from_the_results_goes_back_not_out(self):
        state = opened()
        state["level"], state["found"] = "found", []
        state, action = self.press(state, "ESC")
        self.assertIsNone(action)
        self.assertEqual(state["level"], "top")


class TestDiscoveredNamesCannotDriveTheTerminal(unittest.TestCase):
    """A filename is chosen by whoever can write to the directory, and one of
    the directories searched is world-writable. `\\x1b[2J` clears the screen and
    `\\x1b]0;...\\x07` rewrites the title, so a name is untrusted input and has
    to cross the same boundary every other untrusted string does.
    """

    def test_an_escape_sequence_in_a_filename_does_not_reach_the_terminal(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "pwn\x1b[2J\x1b]0;hijacked\x07.log")
        with open(path, "w") as fh:
            fh.write("x I slot print_timing: id  1 | task 1 | draft acceptance"
                     " = 0.5 (  1 accepted /   2 generated), mean len =  1.00\n")
        row = discover_backends(dirs=(d,), ports=())[0]
        self.assertNotIn("\x1b", describe_backend(row))

    def test_a_hostile_model_id_from_a_server_is_defused_too(self):
        row = {"kind": "server", "engine": "llama.cpp", "port": 8080,
               "models": ["evil\x1b[2Jname"], "apply": {}}
        self.assertNotIn("\x1b", describe_backend(row))

    def test_the_idle_line_defuses_them_as_well(self):
        from llmwatch import render_idle
        line = render_idle(5.0, PLAIN, {
            "proxying": True, "upstream": "http://127.0.0.1:8080",
            "upstream_models": ["a\x1b[2Jb"]})
        self.assertNotIn("\x1b", line)

if __name__ == "__main__":
    unittest.main()
