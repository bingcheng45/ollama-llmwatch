"""Tests for the persistent request history.

Everything llmwatch measures used to die at exit, which is why "is this model
build actually faster?" kept being answered with a fresh hand-written benchmark.
These tests pin the three things that make the recorded answer trustworthy:
token-weighted rates, like-for-like bucketing, and a refusal to draw conclusions
from thin data.
"""

import io
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    History, MIN_COMPARE_SAMPLES, export_history, show_compare, show_history, size_band,
)


class Args(object):
    def __init__(self, **kw):
        self.days = 7
        self.model = None
        self.compare = None
        self.export = None
        for key, value in kw.items():
            setattr(self, key, value)


class HistoryTestCase(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "history.db")
        self.history = History(path=self.db)
        self.now = 1_700_000_000.0

    def tearDown(self):
        self.history.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def add(self, model, n=1, gen_rate=10.0, prefill_tokens=12000, cached=0,
            cache_miss=False, age_days=0.0):
        for i in range(n):
            self.history.record(
                model,
                {"tokens": prefill_tokens, "cached": cached,
                 "seconds": prefill_tokens / 100.0, "rate": 100.0},
                {"tokens": 200, "seconds": 200.0 / gen_rate, "rate": gen_rate},
                {"seconds": 300.0, "prefill_share_pct": 90.0, "cache_miss": cache_miss},
                now=self.now - age_days * 86400 - i)


class TestStorage(HistoryTestCase):

    def test_a_request_round_trips(self):
        self.add("mtp")
        rows = self.history.all_rows(now=self.now)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "mtp")

    def test_schema_cannot_hold_prompt_content(self):
        """The same privacy property the Ollama log has: timings only."""
        cur = sqlite3.connect(self.db).execute("PRAGMA table_info(requests)")
        columns = {row[1] for row in cur.fetchall()}
        for forbidden in ("prompt", "text", "content", "message", "response", "command"):
            self.assertNotIn(forbidden, columns)
        self.assertIn("prefill_rate", columns)

    def test_an_unusable_database_disables_history_instead_of_crashing(self):
        """A monitor that dies because its logbook is locked is worse than one
        with no logbook."""
        blocked = os.path.join(self.dir, "not-a-dir")
        with open(blocked, "w") as fh:
            fh.write("x")
        history = History(path=os.path.join(blocked, "nested", "history.db"))
        self.assertFalse(history.enabled)
        self.assertFalse(history.record("m", {}, {}, {"seconds": 1.0}))
        self.assertEqual(history.rollup(now=self.now), [])
        self.assertEqual(history.compare("a", "b", now=self.now), [])

    def test_recording_survives_missing_phases(self):
        self.assertTrue(self.history.record("m", None, None, {"seconds": 1.0},
                                            now=self.now))


class TestRollup(HistoryTestCase):

    def test_rates_are_token_weighted(self):
        """One tiny fast request must not outvote a big slow one, the same rule
        the live stats follow."""
        self.history.record("m", {"tokens": 10, "cached": 0, "seconds": 1.0, "rate": 10.0},
                            {"tokens": 10000, "seconds": 1000.0, "rate": 10.0},
                            {"seconds": 1001.0}, now=self.now - 10)
        self.history.record("m", {"tokens": 10, "cached": 0, "seconds": 1.0, "rate": 10.0},
                            {"tokens": 10, "seconds": 0.01, "rate": 1000.0},
                            {"seconds": 1.0}, now=self.now - 5)
        rate = self.history.rollup(now=self.now)[0]["gen_rate"]
        self.assertLess(rate, 20.0)          # not the ~505 a mean of rates gives

    def test_window_excludes_older_requests(self):
        self.add("m", n=3)
        self.add("m", n=4, age_days=30)
        self.assertEqual(self.history.rollup(days=7, now=self.now)[0]["requests"], 3)

    def test_change_against_the_previous_window(self):
        self.add("m", n=5, gen_rate=10.0)                 # this week
        self.add("m", n=5, gen_rate=20.0, age_days=8)     # last week
        row = self.history.rollup(days=7, now=self.now)[0]
        self.assertLess(row["gen_change_pct"], -40)       # got about half as fast

    def test_no_prior_data_is_reported_as_such_not_as_zero_change(self):
        self.add("m", n=3)
        row = self.history.rollup(days=7, now=self.now)[0]
        self.assertIsNone(row["gen_change_pct"])
        self.assertEqual(row["previous_requests"], 0)

    def test_models_are_kept_separate(self):
        self.add("mtp", n=3, gen_rate=14.0)
        self.add("base", n=3, gen_rate=10.0)
        rows = {r["model"]: r for r in self.history.rollup(now=self.now)}
        self.assertAlmostEqual(rows["mtp"]["gen_rate"], 14.0, places=1)
        self.assertAlmostEqual(rows["base"]["gen_rate"], 10.0, places=1)

    def test_model_filter(self):
        self.add("mtp", n=3)
        self.add("base", n=3)
        rows = self.history.rollup(now=self.now, model="mtp")
        self.assertEqual([r["model"] for r in rows], ["mtp"])


class TestCompare(HistoryTestCase):

    def test_like_for_like_bucketing(self):
        """Sizes are bucketed so a cached 244-token request is never weighed
        against an uncached 47k one."""
        self.add("mtp", n=6, gen_rate=14.0, prefill_tokens=12000)
        self.add("base", n=6, gen_rate=10.0, prefill_tokens=12000)
        self.add("mtp", n=6, gen_rate=30.0, prefill_tokens=500)
        self.add("base", n=6, gen_rate=28.0, prefill_tokens=500)
        rows = {r["band"]: r for r in self.history.compare("mtp", "base", now=self.now)}
        self.assertIn("large", rows)
        self.assertIn("tiny", rows)
        self.assertAlmostEqual(rows["large"]["ratio"], 1.4, places=1)
        self.assertLess(rows["tiny"]["ratio"], 1.2)

    def test_cache_state_is_part_of_the_bucket(self):
        self.add("mtp", n=6, cache_miss=True)
        self.add("base", n=6, cache_miss=True)
        self.add("mtp", n=6, cache_miss=False)
        self.add("base", n=6, cache_miss=False)
        states = {r["cache_miss"] for r in self.history.compare("mtp", "base", now=self.now)}
        self.assertEqual(states, {True, False})

    def test_thin_data_reports_counts_instead_of_a_ratio(self):
        """Two samples is not evidence. Printing '1.8x faster' from it is worse
        than printing nothing."""
        self.add("mtp", n=2, gen_rate=14.0)
        self.add("base", n=2, gen_rate=10.0)
        row = self.history.compare("mtp", "base", now=self.now)[0]
        self.assertIsNone(row["ratio"])
        self.assertFalse(row["enough"])
        self.assertEqual(row["a_n"], 2)

    def test_threshold_boundary(self):
        self.add("mtp", n=MIN_COMPARE_SAMPLES)
        self.add("base", n=MIN_COMPARE_SAMPLES)
        self.assertTrue(self.history.compare("mtp", "base", now=self.now)[0]["enough"])

    def test_one_sided_data_is_not_a_comparison(self):
        self.add("mtp", n=10)
        row = self.history.compare("mtp", "base", now=self.now)[0]
        self.assertIsNone(row["ratio"])
        self.assertEqual(row["b_n"], 0)

    def test_other_models_are_ignored(self):
        self.add("mtp", n=6)
        self.add("base", n=6)
        self.add("unrelated", n=6)
        for row in self.history.compare("mtp", "base", now=self.now):
            self.assertEqual(row["a_n"] + row["b_n"], 12)


class TestSizeBands(unittest.TestCase):

    def test_bands(self):
        self.assertEqual(size_band(500), "tiny")
        self.assertEqual(size_band(5000), "small")
        self.assertEqual(size_band(20000), "large")
        self.assertEqual(size_band(60000), "huge")
        self.assertEqual(size_band(None), "unknown")


class TestOutput(HistoryTestCase):

    def render(self, fn, args):
        out = io.StringIO()
        code = fn(args, out=out, history=self.history, now=self.now)
        return code, out.getvalue()

    def test_history_output(self):
        self.add("mtp", n=4, gen_rate=14.0)
        code, text = self.render(show_history, Args())
        self.assertEqual(code, 0)
        self.assertIn("mtp", text)
        self.assertIn("14.0 tok/s", text)

    def test_empty_history_explains_itself(self):
        code, text = self.render(show_history, Args())
        self.assertEqual(code, 1)
        self.assertIn("no history yet", text)

    def test_compare_output_states_the_bucketing(self):
        self.add("mtp", n=6, gen_rate=14.0)
        self.add("base", n=6, gen_rate=10.0)
        code, text = self.render(show_compare, Args(compare=["mtp", "base"]))
        self.assertEqual(code, 0)
        self.assertIn("faster", text)
        self.assertIn("not the same workload", text)

    def test_compare_says_when_there_is_not_enough_data(self):
        """Wording belongs to the renderer; what matters is that thin data
        reports counts and never a ratio."""
        self.add("mtp", n=2)
        self.add("base", n=2)
        _code, text = self.render(show_compare, Args(compare=["mtp", "base"], no_color=True))
        self.assertIn("need %d each" % MIN_COMPARE_SAMPLES, text)
        self.assertIn("n=2", text)
        self.assertNotIn("x faster", text)

    def test_export_json_and_csv(self):
        self.add("mtp", n=2)
        _code, text = self.render(export_history, Args(export="json"))
        self.assertIn('"model": "mtp"', text)
        _code, text = self.render(export_history, Args(export="csv"))
        self.assertIn("model,", text.splitlines()[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
