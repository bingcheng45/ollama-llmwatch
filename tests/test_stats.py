"""Tests for the per-model statistics accumulator.

A stats board is only useful if its numbers are defensible. These pin the three
decisions that are easy to get subtly wrong: how averages are weighted, which
requests may set the extremes, and that two models never contaminate each other.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import MIN_TOKENS_FOR_EXTREMES, PhaseStats, Stats  # noqa: E402


def req(task=1, pre_tokens=1000, pre_seconds=10.0, pre_rate=100.0, cached=0,
        gen_tokens=100, gen_seconds=10.0, gen_rate=10.0, total=20.0, share=50.0,
        model="m"):
    prefill = {"event": "prefill_done", "task": task, "tokens": pre_tokens,
               "cached": cached, "seconds": pre_seconds, "rate": pre_rate}
    generation = {"event": "generate_done", "task": task, "tokens": gen_tokens,
                  "seconds": gen_seconds, "rate": gen_rate}
    end = {"event": "request_end", "task": task, "seconds": total,
           "prefill_share_pct": share, "model": model}
    return prefill, generation, end


class TestAveraging(unittest.TestCase):

    def test_average_is_token_weighted_not_mean_of_rates(self):
        """One big slow request and one tiny fast one.

        Mean of rates = (10 + 1000)/2 = 505 tok/s, which describes nobody's
        experience. Token-weighted = 10,100 tok / 1000.1 s ~= 10.1 tok/s, which
        is what actually happened.
        """
        phase = PhaseStats()
        phase.record(10000, 1000.0, 10.0)
        phase.record(100, 0.1, 1000.0)
        self.assertAlmostEqual(phase.average, 10100 / 1000.1, places=6)
        self.assertLess(phase.average, 20.0)
        mean_of_rates = (10.0 + 1000.0) / 2
        self.assertNotAlmostEqual(phase.average, mean_of_rates, places=1)

    def test_average_is_none_before_any_data(self):
        self.assertIsNone(PhaseStats().average)


class TestExtremes(unittest.TestCase):

    def test_tiny_request_cannot_set_the_low(self):
        """The real case: a cached prompt computes 4 tokens at a meaningless
        rate. Before the floor, that number became the session low forever."""
        phase = PhaseStats()
        phase.record(10000, 100.0, 100.0)
        phase.record(4, 0.31, 12.9)            # cache hit
        self.assertEqual(phase.low, 100.0)
        self.assertEqual(phase.peak, 100.0)

    def test_request_at_the_floor_counts(self):
        phase = PhaseStats()
        phase.record(MIN_TOKENS_FOR_EXTREMES, 1.0, 55.0)
        self.assertEqual(phase.low, 55.0)

    def test_extremes_track_real_requests(self):
        phase = PhaseStats()
        for rate in (50.0, 120.0, 80.0):
            phase.record(1000, 1000.0 / rate, rate)
        self.assertEqual(phase.peak, 120.0)
        self.assertEqual(phase.low, 50.0)

    def test_totals_include_tiny_requests_even_though_extremes_do_not(self):
        """Excluded from peak/low, but they still consumed real time."""
        phase = PhaseStats()
        phase.record(10000, 100.0, 100.0)
        phase.record(4, 0.5, 8.0)
        self.assertEqual(phase.tokens, 10004)
        self.assertAlmostEqual(phase.seconds, 100.5)
        self.assertEqual(phase.count, 2)


class TestPerModelIsolation(unittest.TestCase):

    def test_models_do_not_contaminate_each_other(self):
        """MTP and base differ ~1.34x on code; pooling them describes neither."""
        stats = Stats(clock=lambda: 0.0)
        stats.record("mtp", *req(pre_rate=120.0, pre_tokens=1000, pre_seconds=1000 / 120.0))
        stats.record("base", *req(pre_rate=60.0, pre_tokens=1000, pre_seconds=1000 / 60.0))

        mtp = stats.snapshot("mtp")
        base = stats.snapshot("base")
        self.assertAlmostEqual(mtp["prefill"]["peak"], 120.0)
        self.assertAlmostEqual(base["prefill"]["peak"], 60.0)
        self.assertEqual(mtp["requests"], 1)
        self.assertEqual(base["requests"], 1)
        self.assertEqual(mtp["models_seen"], 2)

    def test_unknown_model_yields_empty_snapshot_not_a_crash(self):
        snap = Stats(clock=lambda: 0.0).snapshot("never-seen")
        self.assertEqual(snap["requests"], 0)
        self.assertIsNone(snap["prefill"]["avg"])


class TestDerivedFigures(unittest.TestCase):

    def test_cache_rate(self):
        stats = Stats(clock=lambda: 0.0)
        stats.record("m", *req(pre_tokens=1000, cached=3000))
        snap = stats.snapshot("m")
        self.assertAlmostEqual(snap["cache_pct"], 75.0)     # 3000 / 4000
        self.assertEqual(snap["cached_tokens"], 3000)

    def test_ttft_min_avg_max(self):
        stats = Stats(clock=lambda: 0.0)
        for seconds in (1.0, 5.0, 9.0):
            stats.record("m", *req(pre_seconds=seconds))
        ttft = stats.snapshot("m")["ttft"]
        self.assertAlmostEqual(ttft["min"], 1.0)
        self.assertAlmostEqual(ttft["max"], 9.0)
        self.assertAlmostEqual(ttft["avg"], 5.0)

    def test_prefill_share_of_wall_clock(self):
        stats = Stats(clock=lambda: 0.0)
        stats.record("m", *req(pre_seconds=90.0, gen_seconds=10.0, total=100.0))
        self.assertAlmostEqual(stats.snapshot("m")["prefill_share_pct"], 90.0)

    def test_recent_is_newest_first_and_bounded(self):
        stats = Stats(clock=lambda: 0.0)
        for i in range(30):
            stats.record("m", *req(task=i))
        recent = stats.snapshot("m")["recent"]
        self.assertLessEqual(len(recent), 20)
        self.assertEqual(recent[0]["task"], 29)

    def test_request_without_generation_still_recorded(self):
        """Cancelled or generation-less requests must not break the board."""
        prefill, _generation, end = req()
        stats = Stats(clock=lambda: 0.0)
        stats.record("m", prefill, None, end)
        snap = stats.snapshot("m")
        self.assertEqual(snap["requests"], 1)
        self.assertEqual(snap["generation"]["count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
