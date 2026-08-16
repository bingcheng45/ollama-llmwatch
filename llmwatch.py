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

Structure: three layers, each testable in isolation.
    parse_line(line) -> Event      pure, no I/O
    Tracker.feed(event) -> Output  pure state machine
    main()                         all the I/O

Requires Python 3.9+. Standard library only, on purpose.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import namedtuple

__version__ = "0.1.0"

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

RE_SLOT = re.compile(_SLOT)
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


def fmt_bar(fraction, width=20):
    filled = int(width * max(0.0, min(1.0, fraction)))
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
    decides whether 'live' means an overwritten line or a JSON record."""

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
            return [Output("line", "   model loaded: %s" % ev.model,
                           {"event": "model_loaded", "model": ev.model})]

        if isinstance(ev, ServerStarted):
            return [Output("line", "   weights loaded into GPU in %.1fs" % ev.seconds,
                           {"event": "server_started", "seconds": ev.seconds})]

        if isinstance(ev, RequestStart):
            req = Request(ev.slot, ev.task, self.model, ev.prompt_tokens, ev.ctx)
            self.requests[self._key(ev)] = req
            stamp = time.strftime("%H:%M:%S", time.localtime(req.started))
            header = "-- %s  %s  (task %d)" % (stamp, req.model, ev.task)
            return [Output("line", header,
                           {"event": "request_start", "task": ev.task,
                            "prompt_tokens": ev.prompt_tokens, "model": req.model})]

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
            cached_note = ""
            if req.cached:
                cached_note = "  (%s cached)" % format(req.cached, ",")
            text = ("PREFILL  %s %3.0f%%  %s/%s tok%s  %5.0f tok/s  eta %s"
                    % (fmt_bar(fraction), fraction * 100, format(ev.processed, ","),
                       format(total, ","), cached_note, ev.rate, fmt_duration(eta)))
            return [Output("live", text,
                           {"event": "prefill_tick", "task": ev.task,
                            "processed": ev.processed, "to_process": total,
                            "cached": req.cached, "fraction": fraction,
                            "rate": ev.rate, "eta_seconds": eta})]

        if isinstance(ev, PrefillDone):
            req = self._get(ev)
            req.prefill = (ev.ms, ev.tokens, ev.rate)
            return []

        if isinstance(ev, GenTick):
            text = ("GENERATE %s tok   %.1f tok/s (now %.1f)"
                    % (format(ev.decoded, ","), ev.rate, ev.rate_3s))
            return [Output("live", text,
                           {"event": "generate_tick", "task": ev.task,
                            "decoded": ev.decoded, "rate": ev.rate, "rate_3s": ev.rate_3s})]

        if isinstance(ev, GenDone):
            req = self._get(ev)
            req.generation = (ev.ms, ev.tokens, ev.rate)
            return []

        if isinstance(ev, RequestEnd):
            return self._finish(ev)

        return []

    def _finish(self, ev):
        req = self.requests.pop(self._key(ev), None)
        out = []
        total_s = ev.ms / 1000.0

        if req and req.prefill:
            ms, tokens, rate = req.prefill
            cached_note = ""
            if req.cached:
                cached_note = " (+%s cached)" % format(req.cached, ",")
            out.append(Output("line",
                              " v PREFILL  %s tok%s  in %7s   avg %6.1f tok/s"
                              % (format(tokens, ","), cached_note, fmt_duration(ms / 1000.0), rate),
                              {"event": "prefill_done", "task": ev.task, "tokens": tokens,
                               "cached": req.cached, "seconds": ms / 1000.0, "rate": rate}))
        if req and req.generation:
            ms, tokens, rate = req.generation
            out.append(Output("line",
                              " v GENERATE %s tok  in %7s   avg %6.1f tok/s"
                              % (format(tokens, ","), fmt_duration(ms / 1000.0), rate),
                              {"event": "generate_done", "task": ev.task, "tokens": tokens,
                               "seconds": ms / 1000.0, "rate": rate}))

        note = ""
        share = None
        if req and req.prefill and total_s > 0:
            share = req.prefill[0] / 1000.0 / total_s * 100
            note = ("   (first token after %s = %.0f%% of the wait)"
                    % (fmt_duration(req.prefill[0] / 1000.0), share))
        out.append(Output("line", " = TOTAL    %7s%s" % (fmt_duration(total_s), note),
                          {"event": "request_end", "task": ev.task,
                           "seconds": total_s, "prefill_share_pct": share}))
        out.append(Output("line", "", {"event": "blank"}))
        return out


# --------------------------------------------------------------------------
# Layer 3: I/O
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


class Console:
    """Renders Outputs to a terminal, overwriting 'live' lines in place."""

    def __init__(self, use_color=True):
        self.use_color = use_color and sys.stdout.isatty()
        self.dirty = False

    def emit(self, out):
        if out.kind == "live":
            if sys.stdout.isatty():
                sys.stdout.write("\r\033[K   " + out.text)
                self.dirty = True
            else:
                return  # progress is meaningless when piped; --json covers that
        else:
            if self.dirty and sys.stdout.isatty():
                sys.stdout.write("\r\033[K")
            sys.stdout.write(out.text + "\n")
            self.dirty = False
        sys.stdout.flush()


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
    console = Console(use_color=not args.no_color)

    if not args.json:
        where = target if kind == "file" else "journalctl -u %s" % target
        sys.stderr.write("llmwatch %s - watching %s\n" % (__version__, where))

    proc = subprocess.Popen(tail_command(kind, target), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            ev = parse_line(line)
            if ev is None:
                if args.debug_unparsed and looks_like_timing(line):
                    sys.stderr.write("UNPARSED: %s" % line)
                continue
            for out in tracker.feed(ev):
                if args.json:
                    if out.data.get("event") != "blank":
                        sys.stdout.write(json.dumps(out.data) + "\n")
                        sys.stdout.flush()
                else:
                    console.emit(out)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
    finally:
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
    finished = []
    for line in lines:
        outs = tracker.feed(parse_line(line))
        for out in outs:
            if out.data.get("event") in ("prefill_done", "generate_done", "request_end"):
                finished.append(out)

    if not finished:
        sys.stderr.write("llmwatch: no completed request found in the recent log\n")
        return 1

    # Walk back to the last request_end and print its group.
    tail = []
    for out in reversed(finished):
        tail.append(out)
        if out.data.get("event") == "prefill_done":
            break
    for out in reversed(tail):
        if args.json:
            sys.stdout.write(json.dumps(out.data) + "\n")
        else:
            sys.stdout.write(out.text + "\n")
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
