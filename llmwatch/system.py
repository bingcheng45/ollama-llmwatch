"""What else is competing for the machine.

A model that suddenly halves in speed is usually not the model's fault, and
this is what lets the board say so instead of just reporting a smaller number.
"""
import os
import re
import subprocess
import time

from .oai import detect_openai_server, list_upstream_models
from .clients import clients_elsewhere, find_clients


class SystemProbe:
    """Cheap, sudo-free signals that explain a measured slowdown.

    Deliberately NOT CPU% or GPU%. On Apple Silicon inference is memory-bandwidth
    bound: the GPU sits near 100% and the CPU near idle whether you are getting
    13 tok/s or 8, so neither number predicts anything. GPU utilisation would
    need powermetrics, which requires sudo.

    What does predict it -- measured on an M1 Max: background apps cost ~20%, a
    second loaded model ~28%, another process hammering the GPU ~44%. All of
    those show up as memory pressure, swap, load, or a second resident model.
    """

    CHEAP_INTERVAL = 5.0     # swap / memory / load
    MODELS_INTERVAL = 15.0   # `ollama ps` costs ~27ms, so poll it rarely
    BUSIEST_INTERVAL = 10.0  # `ps -Ao pcpu,comm -r` costs ~46ms
    # Rare on purpose. It only feeds an idle-screen hint, and it is the one
    # probe that opens a socket, so it must not become a thing llmwatch does
    # continuously to somebody else's server.
    OAI_INTERVAL = 30.0
    # Long enough that a slow first request is not mistaken for a wrong
    # backend: a 27B pays about ten seconds just to load.
    QUIET_BEFORE_LOOKING = 60.0
    # Reading a handful of small config files. Rare because there is no hurry:
    # a misrouted client stays misrouted until somebody changes it.
    CLIENTS_INTERVAL = 30.0

    def __init__(self, clock=time.monotonic, look_for_oai=False,
                 upstream=None, busiest_fn=None):
        self._clock = clock
        # Injected for the same reason as the clock: the throttling below is
        # the whole point of this field, and a test that has to reach into a
        # module to check it is testing the layout rather than the behaviour.
        self._busiest_fn = busiest_fn or busiest_process
        # -inf, not 0: time.monotonic() can start near zero, in which case
        # `now - 0 < INTERVAL` and the very first poll probes nothing. That would
        # delay "Ollama is not running" by 15s -- precisely when the user is
        # staring at the screen wondering why nothing is happening.
        self._cheap_at = float("-inf")
        self._models_at = float("-inf")
        self._busiest_at = float("-inf")
        self._busiest = None
        self.swap_used_gb = None
        self.swap_total_gb = None
        self.memory_free_pct = None
        self.load1 = None
        self.models_loaded = None
        self.model_names = []
        self.models_gb = None
        self.server_ok = None          # None = not checked yet
        self.server_problem = None
        self._look_for_oai = look_for_oai
        self._oai_at = float("-inf")
        self.oai_port = None
        # Set only under --proxy. The idle screen then describes this server
        # rather than Ollama, which is not what is being watched.
        self.upstream = upstream
        # How long the watched backend has shown nothing. Set by the loop,
        # which is the only thing that knows.
        self.idle_seconds = None
        self.listen_url = None
        self._clients_at = float("-inf")
        self.clients_elsewhere = []
        self._upstream_at = float("-inf")
        self.upstream_ok = None
        self.upstream_models = None

    def poll(self):
        now = self._clock()
        if now - self._cheap_at >= self.CHEAP_INTERVAL:
            self._cheap_at = now
            self._read_load()
            self._read_swap()
            self._read_memory()
        if now - self._models_at >= self.MODELS_INTERVAL:
            self._models_at = now
            self._read_models()
        # Only when the Ollama side has nothing to show. A working Ollama setup
        # never probes anything, so this cannot cost or surprise the people it
        # has nothing to tell.
        # Widened from "Ollama has nothing loaded" to "the backend being
        # watched has gone quiet", because the backend that is wrong is exactly
        # the one that never shows anything, whichever it is.
        quiet = (self.idle_seconds is not None
                 and self.idle_seconds >= self.QUIET_BEFORE_LOOKING)
        if (self._look_for_oai
                and (self.server_ok is False or self.models_loaded == 0 or quiet)
                and now - self._oai_at >= self.OAI_INTERVAL):
            self._oai_at = now
            self.oai_port = detect_openai_server()
        # Same rate limit and the same reason: this opens a socket, so it is
        # not something to do on every frame.
        if self.listen_url and now - self._clients_at >= self.CLIENTS_INTERVAL:
            self._clients_at = now
            self.clients_elsewhere = clients_elsewhere(find_clients(),
                                                       self.listen_url)
        if self.upstream and now - self._upstream_at >= self.OAI_INTERVAL:
            self._upstream_at = now
            models = list_upstream_models(self.upstream)
            self.upstream_ok = models is not None
            self.upstream_models = models

    def _read_load(self):
        try:
            self.load1 = os.getloadavg()[0]
        except (OSError, AttributeError):
            self.load1 = None

    def _read_swap(self):
        out = self._run(["sysctl", "-n", "vm.swapusage"])
        if not out:
            return
        used = re.search(r"used\s*=\s*([\d.]+)M", out)
        total = re.search(r"total\s*=\s*([\d.]+)M", out)
        self.swap_used_gb = float(used.group(1)) / 1024 if used else None
        self.swap_total_gb = float(total.group(1)) / 1024 if total else None

    def _read_memory(self):
        out = self._run(["memory_pressure"])
        match = re.search(r"free percentage:\s*(\d+)", out or "")
        self.memory_free_pct = int(match.group(1)) if match else None

    def _read_models(self):
        """Also establishes whether the server is up at all.

        `ollama ps` exits non-zero when the daemon isn't running, which is the
        difference between "nothing has happened yet" and "nothing CAN happen".
        Without this, a stopped Ollama looks identical to an idle one, and the
        first thing a new user sees is a spinner that never resolves.
        """
        try:
            result = subprocess.run(["ollama", "ps"], capture_output=True,
                                    text=True, timeout=5)
        except FileNotFoundError:
            self.server_ok = False
            self.server_problem = "ollama not found on PATH"
            return
        except (OSError, subprocess.SubprocessError):
            return                      # transient; leave the last known state
        if result.returncode != 0:
            self.server_ok = False
            self.server_problem = "Ollama is not running"
            self.models_loaded = None
            return
        self.server_ok = True
        self.server_problem = None
        out = result.stdout
        rows = [r for r in out.splitlines()[1:] if r.strip()]
        self.models_loaded = len(rows)
        # First column is the model name. Read here rather than shelling out
        # again, so filling in a missing name costs nothing extra.
        self.model_names = [r.split()[0] for r in rows if r.split()]
        total = 0.0
        for row in rows:
            match = re.search(r"([\d.]+)\s*GB", row)
            if match:
                total += float(match.group(1))
        self.models_gb = total or None

    @staticmethod
    def _run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return None

    def busiest(self):
        """The biggest CPU consumer, at most once every BUSIEST_INTERVAL.

        Deliberately not part of poll(): this is asked for only once a slowdown
        has already been detected and nothing cheaper explained it, so paying
        for it on the ordinary path would be pure waste.

        The throttle matters because the caller is the render loop. diagnose()
        runs from paint() at ~10 fps, so an unthrottled read forked `ps` ten
        times a second for as long as the slowdown lasted, which is exactly
        when the machine could least afford it.
        """
        now = self._clock()
        if now - self._busiest_at >= self.BUSIEST_INTERVAL:
            self._busiest_at = now
            self._busiest = self._busiest_fn()
        return self._busiest

    def contention(self):
        """Short phrases naming what is competing, most impactful first."""
        out = []
        if (self.models_loaded or 0) > 1:
            size = (" (%.0f GB)" % self.models_gb) if self.models_gb else ""
            out.append("%d models loaded%s" % (self.models_loaded, size))
        if self.swap_used_gb and self.swap_total_gb and \
                self.swap_used_gb > 0.5 * self.swap_total_gb:
            out.append("swapping %.1f/%.1f GB" % (self.swap_used_gb, self.swap_total_gb))
        if self.memory_free_pct is not None and self.memory_free_pct < 25:
            out.append("%d%% memory free" % self.memory_free_pct)
        if self.load1 and self.load1 > 8:
            out.append("load %.0f" % self.load1)
        return out

    def snapshot(self):
        return {"server_ok": self.server_ok, "server_problem": self.server_problem,
                "swap_used_gb": self.swap_used_gb, "swap_total_gb": self.swap_total_gb,
                "memory_free_pct": self.memory_free_pct, "load1": self.load1,
                "models_loaded": self.models_loaded, "models_gb": self.models_gb,
                "models_running": self.model_names,
                "oai_port": self.oai_port,
                "proxying": bool(self.upstream), "upstream": self.upstream,
                "upstream_ok": self.upstream_ok,
                "upstream_models": self.upstream_models,
                "clients_elsewhere": self.clients_elsewhere,
                "contention": self.contention()}


def busiest_process():
    """Name the biggest CPU consumer. Only called when a slowdown is detected --
    it costs ~46ms, too much to poll continuously."""
    try:
        # No shell: the pipe to `head` was doing nothing that a slice cannot.
        out = subprocess.run(["ps", "-Ao", "pcpu,comm", "-r"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [r.strip() for r in out.splitlines()[1:3] if r.strip()]
    if not rows:
        return None
    parts = rows[0].split(None, 1)
    if len(parts) != 2:
        return None
    pct, name = parts
    return "%s %s%%" % (os.path.basename(name.strip()), pct)
