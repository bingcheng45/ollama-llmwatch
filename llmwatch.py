#!/usr/bin/env python3
"""llmwatch - see what your local Ollama model is actually doing.

Local LLMs spend most of a request READING your prompt (prefill), not writing the
answer (generation). On an M1 Max with a dense 27B Q4 model, a 47k-token prompt
takes ~8 minutes of prefill before the first character appears, while the answer
itself takes ~20 seconds. Roughly 96% of the wait is invisible: your coding agent
shows a spinner and nothing else.

Ollama's HTTP API cannot tell you about that window -- it emits its first chunk
only after prefill finishes. llama.cpp added a `prompt_progress` field for exactly
this (ggml-org/llama.cpp#14685) but Ollama does not pass it through. So the server
log is currently the only place the information exists.

llmwatch tails that log and answers the question the spinner can't:
    "is it still reading my prompt, or is it actually writing -- and how long
     until it starts?"

Structure: four layers, the first three pure and testable.
    parse_line(line) -> Event        pure, no I/O
    Tracker.feed(event) -> [Output]  pure state machine
    render_live(...) / render_*      pure formatting
    follow()                         all the I/O, incl. the repaint thread

Requires Python 3.9+. Standard library only, on purpose.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import namedtuple

try:
    import queue
except ImportError:  # pragma: no cover - py2 only, never hit
    queue = None

__version__ = "0.2.0"

# How often the live line repaints. llama-server logs progress only once per
# 512-token batch (5-10s apart at typical rates), so without an independent
# repaint clock the display looks frozen for many seconds at a time.
REPAINT_HZ = 10.0

# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

ServerStarted = namedtuple("ServerStarted", "seconds")
ModelLoaded = namedtuple("ModelLoaded", "model")
RequestStart = namedtuple("RequestStart", "slot task prompt_tokens ctx")
CacheInfo = namedtuple("CacheInfo", "slot task cached")
PrefillTick = namedtuple("PrefillTick", "slot task processed progress elapsed rate")
PrefillDone = namedtuple("PrefillDone", "slot task ms tokens rate")
GenTick = namedtuple("GenTick", "slot task decoded rate rate_3s")
GenDone = namedtuple("GenDone", "slot task ms tokens rate")
RequestEnd = namedtuple("RequestEnd", "slot task ms tokens")

# --------------------------------------------------------------------------
# Layer 1: parsing
#
# These match llama-server's internal log lines, which carry no stability
# guarantee across versions. Everything here fails soft: an unrecognised line
# returns None rather than raising. Run with --debug-unparsed to find lines that
# look like timings but did not match, and please open an issue with a sample.
# --------------------------------------------------------------------------

_SLOT = r"id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|"

RE_SERVER_STARTED = re.compile(r'llama-server started in ([\d.]+) seconds')
RE_MODEL = re.compile(r"template selection.*?model=(\S+)")
RE_REQ_START = re.compile(
    _SLOT + r".*?new prompt, n_ctx_slot = (\d+), n_keep = (-?\d+), task\.n_tokens = (\d+)")
RE_CACHED = re.compile(_SLOT + r".*?cached n_tokens = (\d+)")
RE_PREFILL_TICK = re.compile(
    _SLOT + r".*?prompt processing, n_tokens =\s*(\d+), progress =\s*([\d.]+),"
            r"\s*t =\s*([\d.]+) s / ([\d.]+) tokens per second")
RE_PREFILL_DONE = re.compile(
    _SLOT + r".*?prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
RE_GEN_TICK = re.compile(
    _SLOT + r".*?n_decoded =\s*(\d+), tg =\s*([\d.]+) t/s, tg_3s =\s*([\d.]+) t/s")
RE_GEN_DONE = re.compile(
    _SLOT + r".*?\beval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
RE_TOTAL = re.compile(_SLOT + r".*?total time =\s*([\d.]+) ms /\s*(\d+) tokens")


def parse_line(line):
    """Turn one log line into an Event, or None if it isn't one we care about.

    Order matters: 'prompt eval time' also matches the generation pattern
    (\\beval time), so prefill must be tested first.
    """
    m = RE_PREFILL_TICK.search(line)
    if m:
        return PrefillTick(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                           float(m.group(4)), float(m.group(5)), float(m.group(6)))

    m = RE_GEN_TICK.search(line)
    if m:
        return GenTick(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                       float(m.group(4)), float(m.group(5)))

    m = RE_PREFILL_DONE.search(line)
    if m:
        return PrefillDone(int(m.group(1)), int(m.group(2)), float(m.group(3)),
                           int(m.group(4)), float(m.group(5)))

    # Only reachable when the line is NOT a prompt-eval line, since that was
    # matched above; guard anyway so ordering bugs can't silently corrupt data.
    if "prompt eval time" not in line:
        m = RE_GEN_DONE.search(line)
        if m:
            return GenDone(int(m.group(1)), int(m.group(2)), float(m.group(3)),
                           int(m.group(4)), float(m.group(5)))

    m = RE_TOTAL.search(line)
    if m:
        return RequestEnd(int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4)))

    m = RE_REQ_START.search(line)
    if m:
        return RequestStart(int(m.group(1)), int(m.group(2)), int(m.group(5)), int(m.group(3)))

    m = RE_CACHED.search(line)
    if m:
        return CacheInfo(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = RE_SERVER_STARTED.search(line)
    if m:
        return ServerStarted(float(m.group(1)))

    m = RE_MODEL.search(line)
    if m:
        return ModelLoaded(m.group(1).rsplit("/", 1)[-1].strip('"'))

    return None


def looks_like_timing(line):
    """Did this line look like something we should have parsed? Used by
    --debug-unparsed to surface format drift instead of hiding it."""
    return "print_timing" in line or "new prompt" in line


# --------------------------------------------------------------------------
# Layer 2: tracking
# --------------------------------------------------------------------------

Output = namedtuple("Output", "kind text data")  # kind: "live" | "line"


def fmt_duration(seconds):
    """45.2s / 2m27s / 1h04m."""
    if seconds is None or seconds < 0:
        return "?"
    if seconds < 60:
        return "%.1fs" % seconds
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return "%dm%02ds" % (minutes, secs)
    hours, minutes = divmod(minutes, 60)
    return "%dh%02dm" % (hours, minutes)


def fmt_bar(fraction, width=20, style=None):
    """Progress bar. Unicode blocks when the terminal can take them, else ASCII."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(width * fraction)
    if style and style.unicode:
        return "█" * filled + "░" * (width - filled)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class Request:
    """One model request, tracked from prompt to final timings."""

    def __init__(self, slot, task, model, prompt_tokens=None, ctx=None):
        self.slot = slot
        self.task = task
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.ctx = ctx
        self.cached = 0
        self.started = time.time()
        # llama-server prints its timing block AFTER generation has streamed, so
        # phase summaries are buffered and emitted in true order at request end.
        self.prefill = None   # (ms, tokens, rate)
        self.generation = None

    @property
    def to_process(self):
        """Tokens that actually need computing -- total minus what the prompt
        cache already holds. This is the number that governs the wait."""
        if self.prompt_tokens is None:
            return None
        return max(0, self.prompt_tokens - self.cached)


class Tracker:
    """Consumes events, emits Outputs. No I/O, no terminal escapes: a caller
    decides whether 'live' means an animated line or a JSON record."""

    def __init__(self):
        self.model = "?"
        self.requests = {}

    def _key(self, ev):
        return (ev.slot, ev.task)

    def _get(self, ev):
        key = self._key(ev)
        if key not in self.requests:
            self.requests[key] = Request(ev.slot, ev.task, self.model)
        return self.requests[key]

    def feed(self, ev):
        """Returns a list of Output. Never raises on unexpected ordering."""
        if ev is None:
            return []

        if isinstance(ev, ModelLoaded):
            self.model = ev.model
            return [Output("line", "model loaded: %s" % ev.model,
                           {"event": "model_loaded", "model": ev.model})]

        if isinstance(ev, ServerStarted):
            return [Output("line", "weights loaded into GPU in %.1fs" % ev.seconds,
                           {"event": "server_started", "seconds": ev.seconds})]

        if isinstance(ev, RequestStart):
            req = Request(ev.slot, ev.task, self.model, ev.prompt_tokens, ev.ctx)
            self.requests[self._key(ev)] = req
            return [Output("line", "",
                           {"event": "request_start", "task": ev.task,
                            "prompt_tokens": ev.prompt_tokens, "model": req.model,
                            "started": req.started})]

        if isinstance(ev, CacheInfo):
            req = self._get(ev)
            # Several cache lines arrive per request as checkpoints advance; the
            # first is the one that describes what was reused up front.
            if not req.cached:
                req.cached = ev.cached
            return []

        if isinstance(ev, PrefillTick):
            req = self._get(ev)
            total = req.to_process
            # Fall back to llama.cpp's own fraction if we never saw the prompt size.
            if total:
                fraction = min(1.0, ev.processed / float(total))
            else:
                fraction = ev.progress
                total = int(ev.processed / ev.progress) if ev.progress else ev.processed
            eta = (ev.elapsed / fraction - ev.elapsed) if fraction > 0 else None
            return [Output("live", "",
                           {"event": "prefill_tick", "task": ev.task,
                            "model": req.model,
                            "processed": ev.processed, "to_process": total,
                            "cached": req.cached, "fraction": fraction,
                            "rate": ev.rate, "elapsed": ev.elapsed, "eta_seconds": eta})]

        if isinstance(ev, PrefillDone):
            req = self._get(ev)
            req.prefill = (ev.ms, ev.tokens, ev.rate)
            return []

        if isinstance(ev, GenTick):
            req = self._get(ev)
            elapsed = ev.decoded / ev.rate if ev.rate else 0.0
            return [Output("live", "",
                           {"event": "generate_tick", "task": ev.task,
                            "model": req.model, "decoded": ev.decoded,
                            "rate": ev.rate, "rate_3s": ev.rate_3s, "elapsed": elapsed})]

        if isinstance(ev, GenDone):
            req = self._get(ev)
            req.generation = (ev.ms, ev.tokens, ev.rate)
            return []

        if isinstance(ev, RequestEnd):
            return self._finish(ev)

        return []

    def _finish(self, ev):
        req = self.requests.pop(self._key(ev), None)
        total_s = ev.ms / 1000.0
        prefill = req.prefill if req else None
        generation = req.generation if req else None
        share = None
        if prefill and total_s > 0:
            share = prefill[0] / 1000.0 / total_s * 100

        out = []
        if prefill:
            out.append(Output("line", "", {
                "event": "prefill_done", "task": ev.task, "tokens": prefill[1],
                "cached": req.cached, "seconds": prefill[0] / 1000.0, "rate": prefill[2]}))
        if generation:
            out.append(Output("line", "", {
                "event": "generate_done", "task": ev.task, "tokens": generation[1],
                "seconds": generation[0] / 1000.0, "rate": generation[2]}))
        out.append(Output("line", "", {
            "event": "request_end", "task": ev.task, "seconds": total_s,
            "model": req.model if req else "?",
            "prefill_share_pct": share,
            "started": req.started if req else None}))
        return out


# --------------------------------------------------------------------------
# Layer 3: rendering (pure)
# --------------------------------------------------------------------------

SPINNER_UNICODE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_ASCII = "|/-\\"


class Style:
    """Terminal capabilities + colours, resolved once."""

    def __init__(self, color=True, unicode_ok=True, width=80):
        self.color = color
        self.unicode = unicode_ok
        self.width = width

    def paint(self, text, code):
        if not self.color:
            return text
        return "\033[%sm%s\033[0m" % (code, text)

    def dim(self, t):
        return self.paint(t, "2")

    def bold(self, t):
        return self.paint(t, "1")

    def cyan(self, t):
        return self.paint(t, "36")

    def green(self, t):
        return self.paint(t, "32")

    def yellow(self, t):
        return self.paint(t, "33")

    @classmethod
    def detect(cls, no_color=False):
        is_tty = sys.stdout.isatty()
        color = is_tty and not no_color and not os.environ.get("NO_COLOR")
        enc = (getattr(sys.stdout, "encoding", "") or "").lower()
        return cls(color=color, unicode_ok="utf" in enc,
                   width=shutil.get_terminal_size((80, 24)).columns)


def spinner_frame(seconds, style):
    frames = SPINNER_UNICODE if style.unicode else SPINNER_ASCII
    return frames[int(seconds * 10) % len(frames)]


def project(processed, rate, age, total):
    """Estimated position between log ticks.

    llama-server reports progress once per 512-token batch. Freezing the display
    between those reports is what made the tool feel dead; extrapolating from the
    last measured rate keeps it honest and moving. Never runs past the total, so
    the bar cannot claim completion that hasn't happened.
    """
    if rate <= 0 or age <= 0:
        return processed
    if total is None:
        return processed + rate * age
    return min(total, processed + rate * age)


def render_live(data, age, style, now=None):
    """One live status line. `age` is seconds since this data arrived."""
    if data is None:
        return ""
    event = data.get("event")
    spin = spinner_frame((now if now is not None else time.time()), style)

    if event == "prefill_tick":
        total = data.get("to_process") or 0
        seen = project(data["processed"], data.get("rate", 0), age, total or None)
        fraction = (seen / float(total)) if total else data.get("fraction", 0.0)
        elapsed = data.get("elapsed", 0) + age
        eta = data.get("eta_seconds")
        eta = max(0.0, eta - age) if eta is not None else None
        bar_width = 24 if style.width >= 100 else 16
        cached = data.get("cached") or 0
        cached_note = (" +%s cached" % format(cached, ",")) if cached else ""
        return "%s %s %s %s %s  %s  %s" % (
            spin,
            style.bold("PREFILL "),
            style.cyan(fmt_bar(fraction, bar_width, style)),
            style.bold("%3.0f%%" % (fraction * 100)),
            style.dim("%s/%s tok%s" % (format(int(seen), ","), format(total, ","), cached_note)),
            "%5.0f tok/s" % data.get("rate", 0),
            style.dim("elapsed %s | eta %s" % (fmt_duration(elapsed), fmt_duration(eta))),
        )

    if event == "generate_tick":
        decoded = int(project(data["decoded"], data.get("rate", 0), age, None))
        elapsed = data.get("elapsed", 0) + age
        return "%s %s %s  %s  %s" % (
            spin,
            style.bold("GENERATE"),
            style.green("%s tok" % format(decoded, ",")),
            "%.1f tok/s" % data.get("rate", 0),
            style.dim("now %.1f | elapsed %s"
                      % (data.get("rate_3s", 0), fmt_duration(elapsed))),
        )

    if event == "request_start":
        # No progress ticks exist yet -- llama-server emits the first only after a
        # full 512-token batch, and a cache hit may mean none ever arrive. Show a
        # running clock so this state never looks stalled.
        total = data.get("prompt_tokens")
        what = ("reading prompt, %s tok" % format(total, ",")) if total else "starting"
        return "%s %s %s" % (spin, style.bold("PREFILL "),
                             style.dim("%s | elapsed %s" % (what, fmt_duration(age))))
    return ""


def render_idle(age, style):
    spin = spinner_frame(time.time(), style)
    return style.dim("%s waiting for a request  (idle %s)" % (spin, fmt_duration(age)))


def render_header(data, style, show_time=True):
    """`show_time` is False when replaying history: llama-server's timing lines
    carry no timestamps, so during replay the only clock available is 'now',
    which would be a fabricated request time."""
    model = data.get("model", "?")
    if show_time:
        stamp = time.strftime("%H:%M:%S", time.localtime(data.get("started") or time.time()))
        label = "%s  %s  task %s" % (stamp, model, data.get("task"))
    else:
        label = "%s  task %s  (most recent)" % (model, data.get("task"))
    rule = "─" if style.unicode else "-"
    pad = max(0, min(style.width, 78) - len(label) - 4)
    return style.dim(rule * 2 + " ") + style.bold(label) + style.dim(" " + rule * pad)


def render_summary(prefill, generation, end, style):
    """The committed block printed when a request finishes."""
    lines = []
    tee = ("├", "└") if style.unicode else ("|", "`")

    if prefill:
        cached = prefill.get("cached") or 0
        note = (" +%s cached" % format(cached, ",")) if cached else ""
        lines.append("  %s %s %s %s %s" % (
            tee[0],
            style.bold("PREFILL "),
            ("%9s tok%s" % (format(prefill["tokens"], ","), note)).ljust(26),
            fmt_duration(prefill["seconds"]).rjust(8),
            style.dim("  avg %6.1f tok/s" % prefill["rate"])))

    if generation:
        lines.append("  %s %s %s %s %s" % (
            tee[0],
            style.bold("GENERATE"),
            ("%9s tok" % format(generation["tokens"], ",")).ljust(26),
            fmt_duration(generation["seconds"]).rjust(8),
            style.dim("  avg %6.1f tok/s" % generation["rate"])))

    share = end.get("prefill_share_pct")
    tail = ""
    if share is not None:
        # A split bar: how much of the wall clock was spent before the first
        # token appeared. This is the whole point of the tool, so it gets a bar.
        width = 20
        filled = int(round(width * share / 100.0))
        if style.unicode:
            bar = "█" * filled + "▓" * (width - filled)
        else:
            bar = "#" * filled + "=" * (width - filled)
        tail = "  %s %s" % (style.yellow(bar),
                            style.dim("%.0f%% was prefill" % share))
    lines.append("  %s %s %s %s%s" % (
        tee[1],
        style.bold("TOTAL   "),
        " " * 26,
        fmt_duration(end["seconds"]).rjust(8),
        tail))
    return lines


# --------------------------------------------------------------------------
# Layer 4: I/O
# --------------------------------------------------------------------------

MACOS_HOMEBREW = ["/opt/homebrew/var/log/ollama.log", "/usr/local/var/log/ollama.log"]
MACOS_APP = ["~/.ollama/logs/server.log", "~/Library/Logs/Ollama/server.log"]
LINUX_PATHS = ["/var/log/ollama.log", "~/.ollama/logs/server.log"]


def candidate_paths():
    return [os.path.expanduser(p) for p in MACOS_HOMEBREW + MACOS_APP + LINUX_PATHS]


def find_log():
    """Return (kind, target). kind is 'file' or 'journalctl'."""
    env = os.environ.get("LLMWATCH_LOG")
    if env:
        return ("file", os.path.expanduser(env))
    for path in candidate_paths():
        if os.path.isfile(path):
            return ("file", path)
    # Linux systemd: no file, but the journal has the same lines. Experimental --
    # unverified by the author, please report issues.
    if _has_journal():
        return ("journalctl", "ollama")
    return (None, None)


def _has_journal():
    try:
        r = subprocess.run(["journalctl", "--version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def tail_command(kind, target, from_start=False):
    if kind == "journalctl":
        return ["journalctl", "-fu", target, "-n", "0" if not from_start else "2000"]
    return ["tail", "-n", "0" if not from_start else "+1", "-F", target]


def current_model():
    """The log only names a model when one is LOADED, so an already-warm model
    would otherwise display as '?'."""
    try:
        rows = subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                              timeout=5).stdout.splitlines()
        if len(rows) > 1 and rows[1].strip():
            return rows[1].split()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "?"


def _reader(proc, q, stop):
    """Log lines arrive on their own schedule; the UI must not wait for them."""
    try:
        for line in proc.stdout:
            if stop.is_set():
                break
            q.put(line)
    except (ValueError, OSError):
        pass
    finally:
        q.put(None)


def follow(args):
    kind, target = find_log()
    if not kind:
        sys.stderr.write(
            "llmwatch: could not find an Ollama log. Tried:\n  " +
            "\n  ".join(candidate_paths()) +
            "\n\nPass one explicitly with --log PATH or set LLMWATCH_LOG.\n"
            "Note: llmwatch needs LOCAL log access; a remote Ollama server won't work.\n")
        return 2

    tracker = Tracker()
    tracker.model = current_model()
    style = Style.detect(no_color=args.no_color)
    interactive = sys.stdout.isatty() and not args.json

    if not args.json:
        where = target if kind == "file" else "journalctl -u %s" % target
        sys.stderr.write("llmwatch %s  watching %s\n" % (__version__, where))

    proc = subprocess.Popen(tail_command(kind, target), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    q = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(target=_reader, args=(proc, q, stop))
    thread.daemon = True
    thread.start()

    live_data = None
    live_at = time.time()
    idle_since = time.time()
    pending = {}          # task -> {"prefill":..., "generation":...}
    dirty = False

    def clear():
        if interactive and dirty:
            sys.stdout.write("\r\033[K")

    def commit(text):
        clear()
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    try:
        while True:
            # 1. drain whatever the log produced since the last repaint
            drained = False
            while True:
                try:
                    line = q.get_nowait()
                except Exception:
                    break
                if line is None:
                    raise KeyboardInterrupt
                drained = True
                ev = parse_line(line)
                if ev is None:
                    if args.debug_unparsed and looks_like_timing(line):
                        sys.stderr.write("UNPARSED: %s" % line)
                    continue

                for out in tracker.feed(ev):
                    kind_, data = out.kind, out.data
                    name = data.get("event")

                    if args.json:
                        sys.stdout.write(json.dumps(data) + "\n")
                        sys.stdout.flush()
                        continue

                    if kind_ == "live":
                        live_data, live_at = data, time.time()
                        continue

                    if name == "request_start":
                        commit("")
                        commit(render_header(data, style))
                        live_data, live_at = data, time.time()
                    elif name in ("prefill_done", "generate_done"):
                        slot = pending.setdefault(data["task"], {})
                        slot["prefill" if name == "prefill_done" else "generation"] = data
                    elif name == "request_end":
                        slot = pending.pop(data["task"], {})
                        for text in render_summary(slot.get("prefill"),
                                                   slot.get("generation"), data, style):
                            commit(text)
                        live_data = None
                        idle_since = time.time()
                    elif name in ("model_loaded", "server_started"):
                        commit("  " + style.dim(out.text))

            if drained:
                idle_since = time.time() if live_data is None else idle_since

            # 2. repaint on our own clock, not the log's
            if interactive:
                if live_data is not None:
                    text = render_live(live_data, time.time() - live_at, style)
                else:
                    text = render_idle(time.time() - idle_since, style)
                if text:
                    sys.stdout.write("\r\033[K  " + text)
                    sys.stdout.flush()
                    dirty = True

            time.sleep(1.0 / REPAINT_HZ)
    except KeyboardInterrupt:
        clear()
        sys.stdout.write("\n")
    finally:
        stop.set()
        proc.terminate()
    return 0


def summarise_last(args):
    """Replay a chunk of the log and print the most recent completed request."""
    kind, target = find_log()
    if kind != "file":
        sys.stderr.write("llmwatch --last needs a log file (journalctl not supported yet)\n")
        return 2
    try:
        with open(target, "r", errors="replace") as fh:
            lines = fh.readlines()[-8000:]
    except OSError as exc:
        sys.stderr.write("llmwatch: cannot read %s: %s\n" % (target, exc))
        return 2

    tracker = Tracker()
    tracker.model = current_model()
    style = Style.detect(no_color=args.no_color)
    groups = []
    pending = {}
    for line in lines:
        for out in tracker.feed(parse_line(line)):
            name = out.data.get("event")
            if name in ("prefill_done", "generate_done"):
                slot = pending.setdefault(out.data["task"], {})
                slot["prefill" if name == "prefill_done" else "generation"] = out.data
            elif name == "request_end":
                slot = pending.pop(out.data["task"], {})
                groups.append((slot.get("prefill"), slot.get("generation"), out.data))

    if not groups:
        sys.stderr.write("llmwatch: no completed request found in the recent log\n")
        return 1

    prefill, generation, end = groups[-1]
    if args.json:
        for part in (prefill, generation, end):
            if part:
                sys.stdout.write(json.dumps(part) + "\n")
        return 0

    sys.stdout.write(render_header(end, style, show_time=False) + "\n")
    for text in render_summary(prefill, generation, end, style):
        sys.stdout.write(text + "\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llmwatch",
        description="Live prefill progress and tok/s for a local Ollama model.")
    parser.add_argument("--last", action="store_true",
                        help="summarise the most recent completed request and exit")
    parser.add_argument("--json", action="store_true",
                        help="emit one JSON object per event (for status bars/scripts)")
    parser.add_argument("--log", metavar="PATH", help="path to the Ollama log")
    parser.add_argument("--no-color", action="store_true", help="disable colour output")
    parser.add_argument("--debug-unparsed", action="store_true",
                        help="print timing-ish lines that failed to parse (bug reports)")
    parser.add_argument("--version", action="version", version="llmwatch " + __version__)
    args = parser.parse_args(argv)

    if args.log:
        os.environ["LLMWATCH_LOG"] = args.log
    return summarise_last(args) if args.last else follow(args)


if __name__ == "__main__":
    sys.exit(main())
