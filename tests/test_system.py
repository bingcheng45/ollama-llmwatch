"""Tests for contention detection: noticing that the model is running slow, and
saying what is competing with it.

Design note these tests encode: llmwatch does NOT display CPU% or GPU%. On Apple
Silicon inference is memory-bandwidth bound, so the GPU sits near 100% and the
CPU near idle whether throughput is 13 tok/s or 8 -- neither number moves when
performance does. Slowdowns are detected from llmwatch's own rate measurements,
and system signals are used only to explain them.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    PhaseStats, SLOWDOWN_RATIO, Stats, Style, SystemProbe, detect_slowdown, diagnose,
    render_board, render_system,
)

PLAIN = Style(color=False, unicode_ok=False, width=100)


def strip_ansi(text):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestBaseline(unittest.TestCase):

    def test_median_ignores_a_single_contended_outlier(self):
        """A mean baseline would be dragged down by one bad request, and then
        the next bad request would look normal."""
        phase = PhaseStats()
        for rate in (13.0, 13.5, 12.8, 13.2, 2.0):
            phase.record(1000, 1000.0 / rate, rate)
        self.assertGreater(phase.median, 12.0)

    def test_median_of_empty_is_none(self):
        self.assertIsNone(PhaseStats().median)

    def test_median_of_even_count(self):
        phase = PhaseStats()
        for rate in (10.0, 20.0):
            phase.record(1000, 1.0, rate)
        self.assertAlmostEqual(phase.median, 15.0)


class TestSlowdownDetection(unittest.TestCase):

    def snap(self, rates, phase="prefill"):
        stats = Stats(clock=lambda: 0.0)
        for rate in rates:
            prefill = {"tokens": 1000, "cached": 0, "seconds": 1000.0 / rate, "rate": rate}
            gen = {"tokens": 100, "seconds": 10.0, "rate": rate}
            stats.record("m", prefill, gen,
                         {"task": 1, "seconds": 20.0, "prefill_share_pct": 50.0})
        return stats.snapshot("m")

    def test_detects_a_real_slowdown(self):
        snap = self.snap([100.0, 102.0, 98.0, 101.0])
        data = {"event": "prefill_tick", "rate": 50.0}
        current, typical = detect_slowdown(data, snap)
        self.assertEqual(current, 50.0)
        self.assertGreater(typical, 95.0)

    def test_normal_variation_is_not_flagged(self):
        snap = self.snap([100.0, 102.0, 98.0, 101.0])
        data = {"event": "prefill_tick", "rate": 92.0}
        self.assertIsNone(detect_slowdown(data, snap))

    def test_boundary_is_respected(self):
        snap = self.snap([100.0, 100.0, 100.0])
        just_under = {"event": "prefill_tick", "rate": 100.0 * SLOWDOWN_RATIO - 0.1}
        just_over = {"event": "prefill_tick", "rate": 100.0 * SLOWDOWN_RATIO + 0.1}
        self.assertIsNotNone(detect_slowdown(just_under, snap))
        self.assertIsNone(detect_slowdown(just_over, snap))

    def test_no_baseline_means_no_accusation(self):
        """Two samples is not a baseline; calling the third one 'slow' would be
        noise dressed as insight."""
        snap = self.snap([100.0, 100.0])
        data = {"event": "prefill_tick", "rate": 10.0}
        self.assertIsNone(detect_slowdown(data, snap))

    def test_phases_use_their_own_baselines(self):
        """Generation runs ~7x slower than prefill; comparing across them would
        flag every single request as slow."""
        snap = self.snap([100.0, 100.0, 100.0])
        gen_data = {"event": "generate_tick", "rate": 100.0}
        self.assertIsNone(detect_slowdown(gen_data, snap))

    def test_unknown_event_is_ignored(self):
        self.assertIsNone(detect_slowdown({"event": "request_start"}, self.snap([1.0])))
        self.assertIsNone(detect_slowdown(None, None))


class TestContentionReporting(unittest.TestCase):

    def probe_with(self, **values):
        probe = SystemProbe(clock=lambda: 0.0)
        for key, value in values.items():
            setattr(probe, key, value)
        return probe

    def test_second_model_is_the_headline_cause(self):
        """Measured at ~28% loss on this hardware -- the biggest single cause."""
        probe = self.probe_with(models_loaded=2, models_gb=34.0)
        self.assertIn("2 models loaded (34 GB)", probe.contention())

    def test_one_model_is_not_contention(self):
        self.assertEqual(self.probe_with(models_loaded=1).contention(), [])

    def test_swapping_is_reported_only_when_substantial(self):
        self.assertTrue(self.probe_with(swap_used_gb=3.0, swap_total_gb=4.0).contention())
        self.assertEqual(self.probe_with(swap_used_gb=0.2, swap_total_gb=4.0).contention(), [])

    def test_low_memory_and_high_load(self):
        self.assertTrue(self.probe_with(memory_free_pct=10).contention())
        self.assertEqual(self.probe_with(memory_free_pct=60).contention(), [])
        self.assertTrue(self.probe_with(load1=12.0).contention())
        self.assertEqual(self.probe_with(load1=2.0).contention(), [])

    def test_quiet_machine_reports_nothing(self):
        probe = self.probe_with(models_loaded=1, swap_used_gb=0.1, swap_total_gb=4.0,
                                memory_free_pct=60, load1=1.0)
        self.assertEqual(probe.contention(), [])

    def test_slowdown_names_the_cause_not_just_the_symptom(self):
        stats = Stats(clock=lambda: 0.0)
        for _ in range(4):
            stats.record("m", {"tokens": 1000, "cached": 0, "seconds": 10.0, "rate": 100.0},
                         {"tokens": 100, "seconds": 10.0, "rate": 10.0},
                         {"task": 1, "seconds": 20.0, "prefill_share_pct": 50.0})
        snap = stats.snapshot("m")
        data = {"event": "prefill_tick", "rate": 40.0, "to_process": 1000}
        system = self.probe_with(models_loaded=2, models_gb=34.0).snapshot()
        findings = " ".join(strip_ansi(f) for f in
                            diagnose(data, snap, PLAIN, None, system))
        self.assertIn("40 vs 100 tok/s", findings)
        self.assertIn("2 models loaded", findings)


class TestExpensiveProbesAreRateLimited(unittest.TestCase):
    """`ps -Ao pcpu,comm -r` costs ~46ms, and its caller is the render loop.

    diagnose() runs from paint() at ~10 fps, so an unthrottled read forked `ps`
    ten times a second for as long as the slowdown lasted -- roughly 460ms of
    forking per second, spent precisely when the machine is already struggling
    and the user is watching to find out why.
    """

    def test_repeated_asks_within_the_interval_probe_once(self):
        now = [1000.0]
        calls = []
        probe = SystemProbe(clock=lambda: now[0],
                            busiest_fn=lambda: (calls.append(1), "python 98%")[1])
        for _ in range(10):                     # one second of frames
            self.assertEqual(probe.busiest(), "python 98%")
            now[0] += 0.1
        self.assertEqual(len(calls), 1, "forked ps %d times in one second" % len(calls))

    def test_it_does_refresh_once_the_interval_passes(self):
        """Throttling must not freeze the answer: the busiest process changes."""
        now = [1000.0]
        calls = []
        probe = SystemProbe(clock=lambda: now[0],
                            busiest_fn=lambda: (calls.append(1),
                                                "p%d" % len(calls))[1])
        first = probe.busiest()
        now[0] += SystemProbe.BUSIEST_INTERVAL + 0.1
        second = probe.busiest()
        self.assertEqual((first, second), ("p1", "p2"))

    def test_diagnose_uses_the_injected_reader_not_the_raw_one(self):
        """The wiring, not just the throttle: if diagnose ignored the callable
        the throttle would exist and never be reached."""
        stats = Stats(clock=lambda: 0.0)
        for _ in range(4):
            stats.record("m", {"tokens": 1000, "cached": 0, "seconds": 10.0, "rate": 100.0},
                         {"tokens": 100, "seconds": 10.0, "rate": 10.0},
                         {"task": 1, "seconds": 20.0, "prefill_share_pct": 50.0})
        data = {"event": "prefill_tick", "rate": 40.0, "to_process": 1000}
        findings = " ".join(strip_ansi(f) for f in
                            diagnose(data, stats.snapshot("m"), PLAIN, None,
                                     {"contention": []}, lambda: "ffmpeg 190%"))
        self.assertIn("ffmpeg 190%", findings)


class TestSystemLine(unittest.TestCase):

    def test_no_cpu_or_gpu_percentages_anywhere(self):
        """Locks in the design decision: those numbers don't move when
        performance does, and GPU% needs sudo on macOS anyway."""
        system = {"models_loaded": 1, "swap_used_gb": 0.2, "load1": 2.0,
                  "contention": []}
        blob = strip_ansi("\n".join(render_system(system, PLAIN)))
        for banned in ("CPU", "GPU", "%cpu", "%gpu"):
            self.assertNotIn(banned, blob)

    def test_contention_is_surfaced_prominently(self):
        system = {"contention": ["2 models loaded (34 GB)", "swapping 3.0/4.0 GB"]}
        blob = strip_ansi("\n".join(render_system(system, PLAIN)))
        self.assertIn("!", blob)
        self.assertIn("2 models loaded", blob)

    def test_quiet_machine_says_clear(self):
        system = {"models_loaded": 1, "swap_used_gb": 0.2, "load1": 2.0, "contention": []}
        self.assertIn("clear", strip_ansi("\n".join(render_system(system, PLAIN))))

    def test_absent_probe_renders_nothing(self):
        self.assertEqual(render_system(None, PLAIN), [])
        self.assertEqual(render_system({}, PLAIN), [])

    def test_board_includes_the_system_line(self):
        snap = Stats(clock=lambda: 0.0).snapshot("m")
        blob = strip_ansi("\n".join(render_board(
            snap, PLAIN, 100, system={"contention": ["swapping 3.0/4.0 GB"]})))
        self.assertIn("SYSTEM", blob)

    def test_board_without_a_probe_is_unchanged(self):
        snap = Stats(clock=lambda: 0.0).snapshot("m")
        self.assertNotIn("SYSTEM", strip_ansi("\n".join(render_board(snap, PLAIN, 100))))


class TestProbeMechanics(unittest.TestCase):

    def test_expensive_probes_are_rate_limited(self):
        """`ollama ps` costs ~27ms; polling it every frame at 10fps would burn
        more CPU than the thing it is measuring."""
        self.assertGreaterEqual(SystemProbe.MODELS_INTERVAL, 10.0)
        self.assertGreaterEqual(SystemProbe.CHEAP_INTERVAL, 1.0)

    def test_probe_survives_missing_commands(self):
        probe = SystemProbe(clock=lambda: 0.0)
        self.assertIsNone(probe._run(["definitely-not-a-real-command-xyz"]))

    def test_snapshot_is_safe_before_any_poll(self):
        snap = SystemProbe(clock=lambda: 0.0).snapshot()
        self.assertEqual(snap["contention"], [])
        self.assertIsNone(snap["load1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestServerDownDetection(unittest.TestCase):
    """Idle is ambiguous: nothing has happened yet, or nothing can happen.

    With Ollama stopped the log file still exists, so llmwatch starts fine and
    waits forever. A newcomer cannot tell whether the tool is broken or the
    server is.
    """

    def test_stopped_server_is_named(self):
        from llmwatch import render_idle
        line = strip_ansi(render_idle(30.0, PLAIN,
                                      {"server_ok": False,
                                       "server_problem": "Ollama is not running"}))
        self.assertIn("Ollama is not running", line)
        self.assertIn("pick up automatically", line)

    def test_missing_binary_is_named_differently(self):
        from llmwatch import render_idle
        line = strip_ansi(render_idle(1.0, PLAIN,
                                      {"server_ok": False,
                                       "server_problem": "ollama not found on PATH"}))
        self.assertIn("not found on PATH", line)

    def test_running_but_no_model_warns_about_the_load(self):
        from llmwatch import render_idle
        line = strip_ansi(render_idle(5.0, PLAIN,
                                      {"server_ok": True, "models_loaded": 0}))
        self.assertIn("no model loaded", line)
        self.assertIn("first request", line)

    def test_healthy_idle_is_the_plain_message(self):
        from llmwatch import render_idle
        line = strip_ansi(render_idle(5.0, PLAIN,
                                      {"server_ok": True, "models_loaded": 1}))
        self.assertIn("waiting for a request", line)
        self.assertNotIn("not running", line)

    def test_unknown_state_does_not_accuse(self):
        """Before the first probe completes, saying 'not running' would be a lie."""
        from llmwatch import render_idle
        self.assertIn("waiting for a request", strip_ansi(render_idle(1.0, PLAIN, None)))
        self.assertIn("waiting for a request",
                      strip_ansi(render_idle(1.0, PLAIN, {"server_ok": None})))

    def test_first_poll_probes_immediately(self):
        """time.monotonic() can start near zero, in which case an interval check
        against 0.0 silently skips the first poll -- delaying "Ollama is not
        running" by 15s, exactly when someone is staring at the screen."""
        probe = SystemProbe(clock=lambda: 0.05)
        calls = []
        probe._read_load = lambda: calls.append("load")
        probe._read_swap = lambda: calls.append("swap")
        probe._read_memory = lambda: calls.append("memory")
        probe._read_models = lambda: calls.append("models")
        probe.poll()
        self.assertEqual(sorted(calls), ["load", "memory", "models", "swap"])

    def test_second_poll_respects_the_interval(self):
        probe = SystemProbe(clock=lambda: 0.05)
        probe.poll()
        calls = []
        probe._read_models = lambda: calls.append("models")
        probe.poll()
        self.assertEqual(calls, [], "should not re-probe within the interval")
