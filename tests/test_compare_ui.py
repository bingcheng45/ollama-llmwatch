"""Tests for the interactive model comparison (`c`).

Three things here are easy to get wrong and expensive to ship wrong: arrow keys
arriving as three separate bytes, a picker that hides models you have never
measured, and a comparison that prints a confident ratio from two samples.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    MIN_COMPARE_SAMPLES, Style, UIState, compose_frame, decode_keys, handle_key,
    median_request_time, render_compare, render_median_request, render_picker, visible_len,
)

PLAIN = Style(color=False, unicode_ok=False, width=100)
FANCY = Style(color=True, unicode_ok=True, width=100)


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


MODELS = [
    {"model": "qwen3.8:27b-mtp-128k", "requests": 48, "gen_rate": 14.3, "last_seen": 7200.0},
    {"model": "qwen3.8:27b-128k", "requests": 44, "gen_rate": 10.0, "last_seen": 86400.0},
    {"model": "qwen3.8:27b-q4_K_M", "requests": 0, "gen_rate": None, "last_seen": None},
]


def profile(name, requests=48, gen=14.3, prefill=101.0, ttft=112.0, cache=68.0,
            draft=None, prompt_med=12000.0, gen_med=200.0):
    return {"model": name, "requests": requests, "last_seen": 7200.0, "gen_rate": gen,
            "prefill_rate": prefill, "ttft": ttft, "cache_pct": cache, "draft_pct": draft,
            "median_prompt_tokens": prompt_med, "median_gen_tokens": gen_med}


class TestKeyDecoding(unittest.TestCase):
    """An arrow is three bytes and a lone Esc is a prefix of every arrow.

    A live run caught what unit tests with a synthetic reader could not: pressing
    Down CLOSED the menu, because sys.stdin.read(1) buffers everything available
    and the following select() then reports no data, making an arrow look exactly
    like a bare Esc. Hence reading the raw fd and decoding a buffer.
    """

    def test_arrows_in_one_read(self):
        self.assertEqual(decode_keys("\x1b[A")[0], ["UP"])
        self.assertEqual(decode_keys("\x1b[B")[0], ["DOWN"])
        self.assertEqual(decode_keys("\x1b[C")[0], ["RIGHT"])
        self.assertEqual(decode_keys("\x1b[D")[0], ["LEFT"])

    def test_a_sequence_split_across_reads_is_held_then_completed(self):
        keys, rest = decode_keys("\x1b")
        self.assertEqual(keys, [])
        self.assertEqual(rest, "\x1b")          # held, not misread as Esc
        keys, rest = decode_keys(rest + "[B")
        self.assertEqual(keys, ["DOWN"])
        self.assertEqual(rest, "")

    def test_lone_escape_emerges_on_flush(self):
        self.assertEqual(decode_keys("\x1b")[0], [])
        self.assertEqual(decode_keys("\x1b", flush=True)[0], ["ESC"])

    def test_ordinary_characters_pass_through(self):
        self.assertEqual(decode_keys("chj")[0], ["c", "h", "j"])

    def test_mixed_input_in_one_chunk(self):
        self.assertEqual(decode_keys("c\x1b[Bx")[0], ["c", "DOWN", "x"])

    def test_unknown_sequence_falls_back_to_escape(self):
        self.assertEqual(decode_keys("\x1b[Z")[0], ["ESC"])

    def test_enter_and_digits(self):
        self.assertEqual(decode_keys("\r3")[0], ["\r", "3"])


class TestTransitions(unittest.TestCase):

    def setUp(self):
        self.state = UIState()

    def press(self, *keys):
        for key in keys:
            handle_key(self.state, key, MODELS)
        return self.state

    def test_c_opens_and_closes_the_picker(self):
        self.assertEqual(self.press("c").view, "picker")
        self.assertEqual(self.press("c").view, "live")

    def test_cursor_moves_and_clamps(self):
        self.press("c")
        self.assertEqual(self.press("UP").cursor, 0)          # already at the top
        self.assertEqual(self.press("DOWN").cursor, 1)
        self.assertEqual(self.press("DOWN", "DOWN", "DOWN").cursor, len(MODELS) - 1)

    def test_jk_work_as_well_as_arrows(self):
        self.press("c", "j")
        self.assertEqual(self.state.cursor, 1)
        self.press("k")
        self.assertEqual(self.state.cursor, 0)

    def test_digits_jump(self):
        self.press("c", "3")
        self.assertEqual(self.state.cursor, 2)

    def test_digit_beyond_the_list_is_ignored(self):
        self.press("c", "9")
        self.assertEqual(self.state.cursor, 0)

    def test_a_unicode_digit_that_is_not_a_number_does_not_crash(self):
        """str.isdigit() is True for superscript two, but int() rejects it.

        It is a dedicated key on French AZERTY, and the reader passes raw
        decoded bytes straight through, so the ValueError escaped follow() as
        a traceback with the terminal still in raw mode.
        """
        self.press("c")
        self.assertFalse(handle_key(self.state, "\N{SUPERSCRIPT TWO}", MODELS))
        self.assertEqual(self.state.cursor, 0)

    def test_two_enters_select_a_pair_and_show_the_comparison(self):
        self.press("c", "\r", "DOWN", "\r")
        self.assertEqual(self.state.model_a, MODELS[0]["model"])
        self.assertEqual(self.state.model_b, MODELS[1]["model"])
        self.assertEqual(self.state.view, "compare")

    def test_escape_backs_out_one_level_at_a_time(self):
        self.press("c", "\r", "DOWN", "\r")
        self.assertEqual(self.press("ESC").view, "picker")
        self.assertIsNone(self.state.model_b)
        self.assertEqual(self.press("ESC").view, "live")

    def test_help_is_reachable_from_anywhere(self):
        self.press("c")
        self.assertEqual(self.press("h").view, "help")
        self.assertEqual(self.press("h").view, "live")

    def test_q_quits_from_any_view(self):
        self.press("c")
        with self.assertRaises(KeyboardInterrupt):
            handle_key(self.state, "q", MODELS)

    def test_reopening_the_picker_clears_a_stale_selection(self):
        self.press("c", "\r", "c", "c")
        self.assertIsNone(self.state.model_a)

    def test_enter_with_no_models_does_not_crash(self):
        self.press("c")
        self.assertFalse(handle_key(self.state, "\r", []))


class TestPicker(unittest.TestCase):

    def test_lists_models_with_counts_and_rates(self):
        blob = strip_ansi("\n".join(render_picker(MODELS, UIState(), PLAIN)))
        self.assertIn("qwen3.8:27b-mtp-128k", blob)
        self.assertIn("48", blob)
        self.assertIn("14.3", blob)

    def test_unmeasured_models_are_shown_not_hidden(self):
        """Today's default: 4 models installed, 1 with history. Hiding the rest
        would leave someone wondering where their model went."""
        blob = strip_ansi("\n".join(render_picker(MODELS, UIState(), PLAIN)))
        self.assertIn("qwen3.8:27b-q4_K_M", blob)
        self.assertIn("no data yet", blob)

    def test_cursor_and_selection_are_visible(self):
        state = UIState()
        state.cursor = 1
        state.model_a = MODELS[0]["model"]
        blob = strip_ansi("\n".join(render_picker(MODELS, state, PLAIN)))
        self.assertIn("(A)", blob)
        self.assertIn("now pick a second model", blob)

    def test_keys_are_documented_on_screen(self):
        blob = strip_ansi("\n".join(render_picker(MODELS, UIState(), PLAIN)))
        for hint in ("up/down", "enter", "esc"):
            self.assertIn(hint, blob)


class TestNoDataPath(unittest.TestCase):

    def test_names_the_exact_command_to_get_data(self):
        empty = {"model": "qwen3.8:27b-q4_K_M", "requests": 0}
        blob = strip_ansi("\n".join(render_compare(profile("mtp"), empty, [], PLAIN)))
        self.assertIn("no recorded requests yet", blob)
        self.assertIn('ollama run qwen3.8:27b-q4_K_M "hello"', blob)
        self.assertIn(str(MIN_COMPARE_SAMPLES), blob)

    def test_says_what_the_other_side_already_has(self):
        empty = {"model": "unused", "requests": 0}
        blob = strip_ansi("\n".join(render_compare(profile("mtp"), empty, [], PLAIN)))
        self.assertIn("already has 48 requests", blob)

    def test_both_sides_empty(self):
        empty_a = {"model": "a", "requests": 0}
        empty_b = {"model": "b", "requests": 0}
        blob = strip_ansi("\n".join(render_compare(empty_a, empty_b, [], PLAIN)))
        self.assertIn("a has no recorded requests", blob)
        self.assertIn("b has no recorded requests", blob)


class TestCompareView(unittest.TestCase):

    def buckets(self, a_n=48, b_n=44, a_rate=14.3, b_rate=10.0):
        enough = a_n >= MIN_COMPARE_SAMPLES and b_n >= MIN_COMPARE_SAMPLES
        return [{"band": "large", "cache_miss": False, "a_n": a_n, "b_n": b_n,
                 "a_rate": a_rate, "b_rate": b_rate,
                 "ratio": (a_rate / b_rate) if enough else None, "enough": enough}]

    def render(self, a=None, b=None, buckets=None):
        return strip_ansi("\n".join(render_compare(
            a or profile("mtp"), b or profile("base", gen=10.0, ttft=123.0, cache=64.0),
            buckets if buckets is not None else self.buckets(), PLAIN)))

    def test_headline_ratio(self):
        self.assertIn("A 1.43x faster", self.render())

    def test_a_small_difference_reads_as_about_the_same(self):
        blob = self.render(b=profile("base", gen=14.0))     # ~2% apart
        self.assertIn("about the same", blob)
        self.assertNotIn("1.02x", blob)

    def test_thin_buckets_never_print_a_ratio(self):
        blob = self.render(buckets=self.buckets(a_n=2, b_n=2))
        self.assertIn("need %d each" % MIN_COMPARE_SAMPLES, blob)
        self.assertIn("n=2", blob)

    def test_ttft_and_cache_are_shown(self):
        blob = self.render()
        self.assertIn("TTFT", blob)
        self.assertIn("CACHE", blob)

    def test_identical_ttft_is_not_reported_as_a_win(self):
        blob = self.render(b=profile("base", gen=10.0, ttft=112.0))
        self.assertIn("about the same", blob)
        self.assertNotIn("0.0s sooner", blob)

    def test_non_speculative_build_is_named_not_blank(self):
        blob = self.render(a=profile("mtp", draft=53.0),
                           b=profile("base", gen=10.0, draft=None))
        self.assertIn("53%", blob)
        self.assertIn("not a speculative build", blob)

    def test_every_line_fits_the_terminal(self):
        for cols in (80, 100, 140):
            style = Style(color=True, unicode_ok=True, width=cols)
            lines = render_compare(profile("mtp"), profile("base", gen=10.0),
                                   self.buckets(), style, cols)
            for line in lines:
                self.assertLessEqual(visible_len(line), cols - 1, line)


class TestMedianRequest(unittest.TestCase):
    """Rates are abstract. Seconds saved on the work you actually do are not."""

    def test_arithmetic(self):
        # 12,000 tok / 100 tok/s = 120s reading, 200 tok / 10 tok/s = 20s writing
        self.assertAlmostEqual(
            median_request_time(profile("m", prefill=100.0, gen=10.0), 12000, 200), 140.0)

    def test_reports_the_saving(self):
        blob = strip_ansi("\n".join(render_median_request(
            profile("mtp", prefill=100.0, gen=14.0),
            profile("base", prefill=100.0, gen=10.0), PLAIN)))
        self.assertIn("median request", blob)
        self.assertIn("saves", blob)

    def test_no_saving_claimed_when_they_match(self):
        blob = strip_ansi("\n".join(render_median_request(
            profile("a", prefill=100.0, gen=10.0),
            profile("b", prefill=100.0, gen=10.0), PLAIN)))
        self.assertIn("no meaningful difference", blob)

    def test_missing_history_produces_nothing_rather_than_a_guess(self):
        self.assertEqual(render_median_request({"model": "a"}, {"model": "b"}, PLAIN), [])


class TestFrameIntegration(unittest.TestCase):

    def test_picker_view_keeps_the_live_line(self):
        state = UIState()
        state.view = "picker"
        frame = strip_ansi("\n".join(compose_frame(
            {"model": "m", "prefill": {}, "generation": {}, "recent": []},
            "LIVE-MARKER", PLAIN, 100, 40, ui=state, picker=MODELS)))
        self.assertIn("pick two models", frame)
        self.assertIn("LIVE-MARKER", frame)

    def test_compare_view_keeps_the_live_line(self):
        state = UIState()
        state.view = "compare"
        state.model_a, state.model_b = "mtp", "base"
        frame = strip_ansi("\n".join(compose_frame(
            {"model": "m", "prefill": {}, "generation": {}, "recent": []},
            "LIVE-MARKER", PLAIN, 100, 40, ui=state, compare=["   GENERATE  A 14.3"])))
        self.assertIn("mtp", frame)
        self.assertIn("LIVE-MARKER", frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
