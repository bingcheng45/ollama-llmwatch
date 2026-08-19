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

Ollama serves GGUF models with llama-server and `-mlx` models with its own MLX
runner. The two log nothing alike, so both dialects are parsed, into one set of
events; nothing above the parser knows which engine ran.

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
import atexit
import calendar
import datetime
import json
import os
import re
import select
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import namedtuple

try:                       # POSIX only; the TUI degrades to no-keyboard elsewhere
    import termios
    import tty
except ImportError:        # pragma: no cover - Windows
    termios = None
    tty = None

try:
    import queue
except ImportError:  # pragma: no cover - py2 only, never hit
    queue = None

__version__ = "0.9.1"

# Frame pacing. The loop wakes on new log data OR on these deadlines, whichever
# comes first, so fresh data appears immediately while animation (spinner,
# elapsed, ETA countdown, projected position) keeps moving when the log is
# silent -- llama-server logs progress only once per 512-token batch, 5-10s
# apart at typical rates.
FRAME_ACTIVE = 0.1     # 10 fps while a request is in flight
# Also 10 fps when nothing is running. This was 2 fps on the reasoning that an
# idle spinner needs no more, which was true of the spinner and wrong about the
# tool: a board that updates twice a second reads as laggy, and "is this thing
# even alive?" is the doubt llmwatch exists to remove. A frame is one write of
# at most a screenful, so the cost of the faster floor is a few KB/s and ten
# wakeups a second while you are watching it; MIN_FRAME_GAP still caps the rest.
FRAME_IDLE = 0.1
MIN_FRAME_GAP = 1 / 30.0   # hard ceiling so a log flood can't spin the CPU

# Requests smaller than this are excluded from peak/low. A cached prompt can
# process 4 tokens at a meaningless rate; without this floor that number becomes
# the session "low" forever and the board quietly lies.
MIN_TOKENS_FOR_EXTREMES = 64

RECENT_LIMIT = 20      # requests kept for the sparkline and the recent pane

# The update check. This is the only thing in llmwatch that touches the network,
# and it stays this small on purpose: one GET to PyPI's public JSON, at most
# once a day, on a background thread, with every failure swallowed. It sends
# nothing about you or your models -- the request body is empty and the URL is
# the same one for every user of the package.
#
# It exists because the alternative was worse. 0.8.0 shipped to GitHub and never
# reached PyPI, and nobody running an older copy had any way to find out.
UPDATE_URL = "https://pypi.org/pypi/ollama-llmwatch/json"
UPDATE_INTERVAL = 24 * 3600
UPDATE_TIMEOUT = 2.0   # a slow network must never delay the first frame

# "That took 12m20s" is a fact; "which is 2x your usual" is the useful part. Both
# need enough history behind them not to be noise.
TURN_TYPICAL_DAYS = 30
MIN_TURNS_FOR_TYPICAL = 3
# Below this, a phase was not measured so much as glimpsed. Rates are divisions
# by this number, so a window of a millisecond turns a handful of tokens into a
# five-figure tok/s that then outranks every real sample on the board. A sanity
# floor, not a claim about how fast a model can be.
MIN_MEASURABLE_SECONDS = 0.02

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

# Events that explain WHY a wait is long -- the difference between "20 seconds"
# and "8 minutes" is usually whether the prompt cache survived.
CacheMiss = namedtuple("CacheMiss", "slot task")
CheckpointRestored = namedtuple("CheckpointRestored", "slot task tokens")
CheckpointCreated = namedtuple("CheckpointCreated", "slot task index total")
CheckpointErased = namedtuple("CheckpointErased", "slot task tokens")
# Speculative decoding (the -mtp builds): how many drafted tokens survived.
DraftAcceptance = namedtuple("DraftAcceptance", "slot task rate accepted generated mean_len")

# Ollama runs GGUF models through llama-server but `-mlx` models through its own
# MLX runner, which logs a different dialect entirely. These carry the raw facts
# off one MLX line each; the Tracker turns them into the events above, so only
# the parser and one Tracker branch know MLX exists. Each keeps the log's own
# timestamp because MLX splits a timing across lines and never prints a rate --
# elapsed has to be reconstructed from the gap, and using wall-clock time here
# would silently produce nonsense when replaying an old log with --last.
MlxRunnerStart = namedtuple("MlxRunnerStart", "model ts")
MlxRunnerReady = namedtuple("MlxRunnerReady", "ts")
MlxRequestStart = namedtuple("MlxRequestStart", "prompt_tokens cached miss ts")
MlxPrefillTick = namedtuple("MlxPrefillTick", "processed total ts")
MlxGenStats = namedtuple("MlxGenStats", "iterations drafted accepted acceptance avg_draft ts")
MlxRequestEnd = namedtuple("MlxRequestEnd", "seconds ts")
# The last line of a request, and the only one whose position is fixed: the
# completion and stats lines are written from different places and either can
# land first, but `peak memory` always follows both. It is what makes a request
# safe to close, so a rate is never missing just because the log raced.
MlxPeakMemory = namedtuple("MlxPeakMemory", "ts")

MLX_EVENTS = (MlxRunnerStart, MlxRunnerReady, MlxRequestStart, MlxPrefillTick,
              MlxGenStats, MlxRequestEnd, MlxPeakMemory)

# A server speaking the OpenAI API (mlx_lm.server, LM Studio, llama-server, vLLM)
# is watched from the outside, by proxying it, because the numbers simply are not
# in its log: mlx_lm.server prints prefill progress and nothing else -- no
# generation token count, no rate. `usage` in the response body is the only place
# a completion size exists, and it arrives last, so every rate here is computed
# from timestamps the proxy took itself.
#
# The exception is OaiPrefillTick, which does come from a log. Prefill on a 28k
# prompt is minutes of silence on the wire (the first byte is only sent once it
# finishes), so the progress bar has to come from somewhere else or not exist.
OaiRequestStart = namedtuple("OaiRequestStart", "model ts")
OaiPrefillTick = namedtuple("OaiPrefillTick", "processed total ts")
OaiGenTick = namedtuple("OaiGenTick", "decoded ts")
# `draft` is (rate, accepted, generated, mean_len) or None, matching what the
# log backend puts on a Request -- so history, the comparison pane and the
# drafts-often-rejected hint read one shape whichever backend filled it in.
# It defaults to None because every other OpenAI server sends no timings block
# at all, and most llama-server runs have no drafter loaded.
OaiRequestEnd = namedtuple(
    "OaiRequestEnd",
    "prompt_tokens cached_tokens completion_tokens started first_token last_token "
    "ts draft")
OaiRequestEnd.__new__.__defaults__ = (None,)
# The client hung up, or the upstream died, before the response completed. No
# usage block was ever sent, so there is nothing to time -- but the request has
# to leave the board rather than sit there open forever.
OaiRequestAborted = namedtuple("OaiRequestAborted", "ts")

OAI_EVENTS = (OaiRequestStart, OaiPrefillTick, OaiGenTick, OaiRequestEnd,
              OaiRequestAborted)

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
RE_CACHE_MISS = re.compile(_SLOT + r".*?forcing full prompt re-processing")
RE_CKPT_RESTORED = re.compile(
    _SLOT + r".*?restored context checkpoint \(.*?n_tokens = (\d+)")
RE_CKPT_CREATED = re.compile(_SLOT + r".*?created context checkpoint (\d+) of (\d+)")
RE_CKPT_ERASED = re.compile(
    _SLOT + r".*?erased invalidated context checkpoint \(.*?n_tokens = (\d+)")
RE_DRAFT = re.compile(
    _SLOT + r".*?draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\),"
            r"\s*mean len =\s*([\d.]+)")

# ---- MLX engine (Apple Silicon) -------------------------------------------
# Same fail-soft contract as above: these are internal Ollama log lines with no
# stability guarantee, and an unrecognised one returns None rather than raising.
RE_MLX_TS = re.compile(
    r"\btime=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:\d{2})?")
RE_MLX_RUNNER_START = re.compile(r'msg="starting mlx runner subprocess" model=(\S+)')
RE_MLX_RUNNER_READY = re.compile(r'msg="mlx runner is ready"')
RE_MLX_CACHE = re.compile(r'msg="cache (miss|hit)" total=(\d+) matched=(\d+)')
RE_MLX_PREFILL = re.compile(r'msg="Prompt processing progress" processed=(\d+) total=(\d+)')
RE_MLX_PEAK = re.compile(r'msg="peak memory"')
RE_MLX_SPEC = re.compile(
    r'msg="speculative decode stats" iterations=(\d+) drafted=(\d+) accepted=(\d+)'
    r' acceptance=([\d.]+) avg_draft=([\d.]+)')
# The runner's own completion timing. Deliberately not the outer `[GIN] POST
# "/api/chat"` line, which also includes however long the model took to load --
# attributing a 10s weight load to the request would wreck every rate on screen.
RE_MLX_DONE = re.compile(
    r'msg=ServeHTTP method=POST path=/v1/(?:completions|chat/completions)'
    r' took=([\d.]+(?:ns|µs|us|ms|h|m|s)(?:[\d.]+(?:ns|µs|us|ms|h|m|s))*)')

# Go prints a duration in whichever units keep it readable and concatenates them
# past a minute, so the one number on screen can arrive as `7.29s`, `166.792µs`
# or `1m30.5s`. Longest units first: `ms` has to win over `m` followed by `s`.
RE_GO_DURATION = re.compile(r"([\d.]+)(ns|µs|us|ms|h|m|s)")
_GO_UNITS = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3,
             "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_go_duration(text):
    """Seconds from a Go duration string, or None if none of it parses.

    Reading `1m30.5s` as one minute would quietly drop half a minute off a slow
    request, which is exactly the request someone is watching this tool to
    understand.
    """
    total, seen = 0.0, False
    for value, unit in RE_GO_DURATION.findall(text):
        try:
            total += float(value) * _GO_UNITS[unit]
        except ValueError:
            continue
        seen = True
    return total if seen else None


def parse_mlx_timestamp(line):
    """Epoch seconds from an Ollama structured-log line, or None.

    Only ever used for differences between two lines of the same log, but the
    UTC offset is honoured anyway so a machine that crosses a DST boundary
    mid-session cannot produce a negative prefill.
    """
    m = RE_MLX_TS.search(line)
    if not m:
        return None
    stamp, offset = m.group(1), m.group(2)
    frac = 0.0
    if "." in stamp:
        stamp, digits = stamp.split(".", 1)
        try:
            frac = float("0." + digits)
        except ValueError:
            frac = 0.0
    try:
        parsed = datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    seconds = calendar.timegm(parsed.timetuple()) + frac
    if offset and offset != "Z":
        sign = 1 if offset[0] == "+" else -1
        seconds -= sign * (int(offset[1:3]) * 3600 + int(offset[4:6]) * 60)
    return seconds


def parse_mlx_line(line):
    """MLX-runner counterpart to parse_line. Returns an Mlx* event or None."""
    m = RE_MLX_PREFILL.search(line)
    if m:
        return MlxPrefillTick(int(m.group(1)), int(m.group(2)), parse_mlx_timestamp(line))

    m = RE_MLX_CACHE.search(line)
    if m:
        total, matched = int(m.group(2)), int(m.group(3))
        return MlxRequestStart(total, matched, m.group(1) == "miss", parse_mlx_timestamp(line))

    m = RE_MLX_SPEC.search(line)
    if m:
        return MlxGenStats(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                           float(m.group(4)), float(m.group(5)), parse_mlx_timestamp(line))

    m = RE_MLX_DONE.search(line)
    if m:
        seconds = parse_go_duration(m.group(1))
        if seconds is not None:
            return MlxRequestEnd(seconds, parse_mlx_timestamp(line))

    m = RE_MLX_RUNNER_START.search(line)
    if m:
        return MlxRunnerStart(short_model_name(m.group(1)), parse_mlx_timestamp(line))

    if RE_MLX_PEAK.search(line):
        return MlxPeakMemory(parse_mlx_timestamp(line))

    if RE_MLX_RUNNER_READY.search(line):
        return MlxRunnerReady(parse_mlx_timestamp(line))

    return None


# ---- standalone mlx_lm.server ---------------------------------------------
# Ollama's MLX runner (above) and the mlx_lm.server you run yourself share a
# name and nothing else: this one is plain Python `logging` output, so none of
# the RE_MLX_* patterns above match it. They are deliberately not loosened to
# try -- widening a pattern until it catches both dialects is how a parser
# starts reporting one engine's numbers under the other's name.
#
# Only the prefill line is useful. Everything else it prints is either cache
# bookkeeping or a BaseHTTPServer access line with one-second resolution, which
# is too coarse to time anything a person is waiting on.
RE_MLXS_PREFILL = re.compile(r"Prompt processing progress: (\d+)/(\d+)")
RE_MLXS_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ")


def parse_mlx_server_timestamp(line):
    """Epoch seconds from a Python logging prefix, or None.

    Unlike Ollama's UTC line this one is local time with no offset printed, so
    it is resolved as local -- which is what makes it comparable with the
    time.time() the proxy stamps its own events with.
    """
    m = RE_MLXS_TS.match(line)
    if not m:
        return None
    try:
        return datetime.datetime.strptime(
            m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
    except (ValueError, OSError, OverflowError):
        return None


def parse_mlx_server_line(line):
    """Prefill progress from a standalone mlx_lm.server log, or None.

    Returns an Oai* event, not an Mlx* one: the request it belongs to is owned
    by the proxy, and routing it to the MLX adapter would attach it to a
    different slot's request.
    """
    m = RE_MLXS_PREFILL.search(line)
    if m:
        return OaiPrefillTick(int(m.group(1)), int(m.group(2)),
                              parse_mlx_server_timestamp(line))
    return None


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

    m = RE_DRAFT.search(line)
    if m:
        return DraftAcceptance(int(m.group(1)), int(m.group(2)), float(m.group(3)),
                               int(m.group(4)), int(m.group(5)), float(m.group(6)))

    m = RE_CACHE_MISS.search(line)
    if m:
        return CacheMiss(int(m.group(1)), int(m.group(2)))

    m = RE_CKPT_RESTORED.search(line)
    if m:
        return CheckpointRestored(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = RE_CKPT_CREATED.search(line)
    if m:
        return CheckpointCreated(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                 int(m.group(4)))

    m = RE_CKPT_ERASED.search(line)
    if m:
        return CheckpointErased(int(m.group(1)), int(m.group(2)), int(m.group(3)))

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
        # Untrusted: a model name can carry anything. Sanitise before it can
        # reach a terminal (see safe_text).
        return ModelLoaded(short_model_name(m.group(1)))

    # Last, and unconditionally: which engine is running is a property of the
    # model, not of the install, and a single log can hold both across a restart
    # (`gemma3:27b` then `gemma4:26b-mlx`). Sniffing the engine once and locking
    # it in would go blind halfway through the file, so both dialects are simply
    # always understood and no flag or setting selects between them. The
    # standalone mlx_lm.server dialect joins on the same terms.
    ev = parse_mlx_line(line)
    if ev is not None:
        return ev
    return parse_mlx_server_line(line)


def looks_like_timing(line):
    """Did this line look like something we should have parsed? Used by
    --debug-unparsed to surface format drift instead of hiding it."""
    return ("print_timing" in line or "new prompt" in line
            or "Prompt processing progress" in line or "decode stats" in line
            or "Prompt Cache:" in line)


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
        # Why this request is slow, in the server's own words.
        self.cache_miss = False
        self.restored_tokens = 0
        self.checkpoints = None      # (index, total)
        self.draft = None            # (rate, accepted, generated, mean_len)
        self.status = None           # short human phrase for the live line
        self.last_live = None        # last emitted live payload, for re-emission

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

    # The MLX runner serves one request at a time and numbers nothing, so it
    # gets a fixed synthetic slot and a counter standing in for llama.cpp's
    # task id. Slot 0 is what a single-slot llama-server uses too, and the two
    # engines never appear in the same request, so they cannot collide.
    MLX_SLOT = 0
    # The proxied OpenAI backend gets its own slot. The two adapters cannot run
    # against the same server, but a stale key from one must never be able to
    # collide with a live request from the other.
    OAI_SLOT = 1

    def __init__(self):
        self.model = "?"
        self.requests = {}
        self._mlx_task = 0
        self._mlx_started = None    # ts of the current request's first line
        self._mlx_prefilled = None  # ts prefill finished == generation began
        self._mlx_loading = None     # ts the runner subprocess was spawned
        self._mlx_pending_end = None  # request duration, held until it can close
        self._oai_task = 0
        self._oai_started = None    # wall clock the proxy sent the request
        self._oai_first = None      # wall clock the first token came back
        self._oai_open = False      # is there a request the log ticks belong to?

    def _key(self, ev):
        return (ev.slot, ev.task)

    def _get(self, ev):
        key = self._key(ev)
        if key not in self.requests:
            self.requests[key] = Request(ev.slot, ev.task, self.model)
        return self.requests[key]

    @staticmethod
    def _context(req):
        """The 'why is this slow' fields, attached to every live payload."""
        return {"cache_miss": req.cache_miss, "status": req.status,
                "restored_tokens": req.restored_tokens,
                "checkpoints": req.checkpoints, "draft": req.draft}

    def feed(self, ev):
        """Returns a list of Output. Never raises on unexpected ordering."""
        if ev is None:
            return []

        if isinstance(ev, MLX_EVENTS):
            return self._feed_mlx(ev)

        if isinstance(ev, OAI_EVENTS):
            return self._feed_oai(ev)

        if isinstance(ev, ModelLoaded):
            self.model = ev.model
            return [Output("line", "model loaded: %s" % ev.model,
                           {"event": "model_loaded", "model": ev.model})]

        if isinstance(ev, ServerStarted):
            return [Output("line", "weights loaded into GPU in %.1fs" % ev.seconds,
                           {"event": "server_started", "seconds": ev.seconds})]

        if isinstance(ev, RequestStart):
            outs = []
            # A slot handles one request at a time, so anything still open on this
            # slot was cancelled -- the client disconnected (a Codex timeout, say)
            # and llama-server never wrote its `total time` line. Without this the
            # display just shows a header with nothing under it, which reads like
            # the tool lost track.
            for key in [k for k in self.requests if k[0] == ev.slot]:
                old = self.requests.pop(key)
                outs.append(Output("line", "", {
                    "event": "request_abandoned", "task": old.task,
                    "model": old.model, "prompt_tokens": old.prompt_tokens}))

            req = Request(ev.slot, ev.task, self.model, ev.prompt_tokens, ev.ctx)
            self.requests[self._key(ev)] = req
            outs.append(Output("line", "",
                               {"event": "request_start", "task": ev.task,
                                "prompt_tokens": ev.prompt_tokens, "model": req.model,
                                "started": req.started}))
            return outs

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
            payload = {"event": "prefill_tick", "task": ev.task,
                       "model": req.model,
                       "processed": ev.processed, "to_process": total,
                       "cached": req.cached, "fraction": fraction,
                       "rate": ev.rate, "elapsed": ev.elapsed, "eta_seconds": eta}
            payload.update(self._context(req))
            req.last_live = payload
            return [Output("live", "", payload)]

        # ---- state that explains the wait ---------------------------------
        if isinstance(ev, (CacheMiss, CheckpointRestored, CheckpointCreated,
                           CheckpointErased, DraftAcceptance)):
            req = self._get(ev)
            outs = []
            if isinstance(ev, CacheMiss):
                req.cache_miss = True
                req.status = "cache miss - reprocessing the whole prompt"
                # Committed as well as shown live: this is the single most useful
                # thing to know when deciding whether to keep waiting.
                outs.append(Output("line", "", {
                    "event": "cache_miss", "task": ev.task,
                    "prompt_tokens": req.prompt_tokens}))
            elif isinstance(ev, CheckpointRestored):
                req.restored_tokens = ev.tokens
                req.status = "restored %s cached tokens" % format(ev.tokens, ",")
            elif isinstance(ev, CheckpointCreated):
                req.checkpoints = (ev.index, ev.total)
                req.status = "saving cache checkpoint %d/%d" % (ev.index, ev.total)
            elif isinstance(ev, CheckpointErased):
                req.status = "discarding stale cache (%s tok)" % format(ev.tokens, ",")
            else:
                req.draft = (ev.rate, ev.accepted, ev.generated, ev.mean_len)

            # Re-emit the current live view so the change shows immediately
            # instead of waiting for the next 512-token batch.
            if req.last_live:
                payload = dict(req.last_live)
                payload.update(self._context(req))
                req.last_live = payload
                outs.append(Output("live", "", payload))
            return outs

        if isinstance(ev, PrefillDone):
            req = self._get(ev)
            req.prefill = (ev.ms, ev.tokens, ev.rate)
            return []

        if isinstance(ev, GenTick):
            req = self._get(ev)
            elapsed = ev.decoded / ev.rate if ev.rate else 0.0
            payload = {"event": "generate_tick", "task": ev.task,
                       "model": req.model, "decoded": ev.decoded,
                       "rate": ev.rate, "rate_3s": ev.rate_3s, "elapsed": elapsed}
            payload.update(self._context(req))
            req.last_live = payload
            return [Output("live", "", payload)]

        if isinstance(ev, GenDone):
            req = self._get(ev)
            req.generation = (ev.ms, ev.tokens, ev.rate)
            return []

        if isinstance(ev, RequestEnd):
            return self._finish(ev)

        return []

    def _feed_mlx(self, ev):
        """Translate one MLX event into the llama.cpp events the rest of this
        class already handles.

        Everything downstream -- rendering, stats, history, compare -- then sees
        one shape of request regardless of which engine produced it. The cost of
        that is here: MLX splits a request across lines that share no id, and
        prints progress without a rate, so both have to be reconstructed.
        """
        if isinstance(ev, MlxRunnerStart):
            self._mlx_loading = ev.ts
            return self.feed(ModelLoaded(ev.model))

        if isinstance(ev, MlxRunnerReady):
            # "Loading" here is the weight load, the same thing llama-server
            # reports as `llama-server started in Ns` -- worth showing, because
            # on a 26B model it is ~10s of the first request's apparent latency.
            if self._mlx_loading is None or ev.ts is None:
                return []
            seconds = max(0.0, ev.ts - self._mlx_loading)
            self._mlx_loading = None
            return self.feed(ServerStarted(seconds))

        if isinstance(ev, MlxRequestStart):
            # Belt and braces: if a build ever stops printing the closing line,
            # the previous request still closes here instead of hanging on the
            # board forever.
            outs = self._mlx_close()
            self._mlx_task += 1
            self._mlx_started = ev.ts
            self._mlx_prefilled = None
            task = self._mlx_task
            # MLX gives no context size; None keeps the ctx field honestly empty
            # rather than inventing a window the renderer would then draw.
            outs.extend(self.feed(RequestStart(self.MLX_SLOT, task, ev.prompt_tokens, None)))
            outs.extend(self.feed(CacheInfo(self.MLX_SLOT, task, ev.cached)))
            if ev.miss:
                outs.extend(self.feed(CacheMiss(self.MLX_SLOT, task)))
            return outs

        if isinstance(ev, MlxPrefillTick):
            elapsed = self._mlx_elapsed_since(self._mlx_started, ev.ts)
            self._mlx_prefilled = ev.ts
            rate = (ev.processed / elapsed) if elapsed > 0 else 0.0
            fraction = (ev.processed / float(ev.total)) if ev.total else 0.0
            return self.feed(PrefillTick(self.MLX_SLOT, self._mlx_task, ev.processed,
                                         fraction, elapsed, rate))

        if isinstance(ev, MlxGenStats):
            # MLX never prints a generation token count. With speculative
            # decoding each iteration commits one token from the target model
            # plus whichever drafted tokens were accepted, which reconstructs it
            # exactly rather than approximately.
            tokens = ev.iterations + ev.accepted
            # Generation began when prefill ended, except on a prompt the cache
            # served whole: MLX prints no progress line then, so the request
            # start is the only honest mark.
            began = self._mlx_prefilled or self._mlx_started
            seconds = self._mlx_elapsed_since(began, ev.ts)
            outs = self.feed(DraftAcceptance(self.MLX_SLOT, self._mlx_task, ev.acceptance,
                                             ev.accepted, ev.drafted, ev.avg_draft))
            # No duration means no rate. Reporting 0 tok/s here would not just
            # look odd on one request: it passes the small-request noise guard,
            # so it lands in `low`, drags the token-weighted average up by
            # adding tokens with no seconds, and skews the median that slowdown
            # detection reads. One unmeasurable request would recolour the board.
            if seconds <= 0 or tokens <= 0:
                return outs
            outs.extend(self.feed(GenDone(self.MLX_SLOT, self._mlx_task,
                                          seconds * 1000.0, tokens, tokens / seconds)))
            return outs

        if isinstance(ev, MlxRequestEnd):
            # Held, not emitted. The stats line carrying the generation rate is
            # written from somewhere else in the runner and lands either side of
            # this one; closing here would drop that rate from the summary on
            # every request that happened to lose the race.
            self._mlx_pending_end = ev.seconds
            return []

        if isinstance(ev, MlxPeakMemory):
            return self._mlx_close()

        return []

    def _mlx_close(self):
        """Emit the held request end, once nothing more can arrive for it."""
        if self._mlx_pending_end is None:
            return []
        seconds, self._mlx_pending_end = self._mlx_pending_end, None
        req = self.requests.get((self.MLX_SLOT, self._mlx_task))
        if req is None:
            # Attaching to a log mid-request: the tail of one is visible but its
            # start never was, so there is no model, no prompt size and no phase
            # to report. A summary of nothing but a duration, under a model
            # named "?", reads as a bug rather than as the partial view it is.
            return []
        # The prefill summary llama-server prints as its own line has to be
        # derived here, from where generation was seen to start.
        if req is not None and req.prefill is None and req.last_live:
            elapsed = self._mlx_elapsed_since(self._mlx_started, self._mlx_prefilled)
            processed = req.last_live.get("processed") or 0
            if elapsed > 0 and processed:
                req.prefill = (elapsed * 1000.0, processed, processed / elapsed)
        tokens = (req.generation[1] if req and req.generation else 0)
        self._mlx_started = self._mlx_prefilled = None
        return self.feed(RequestEnd(self.MLX_SLOT, self._mlx_task,
                                    seconds * 1000.0, tokens))

    @staticmethod
    def _mlx_elapsed_since(start, now):
        """Seconds between two log timestamps, or 0.0 if either is missing.

        Never negative: a clock step backwards should cost the display a rate,
        not turn it into a number that reads as real.
        """
        if start is None or now is None:
            return 0.0
        return max(0.0, now - start)

    def _feed_oai(self, ev):
        """Translate one proxied-request event into the llama.cpp events the
        rest of this class already handles.

        Same contract as _feed_mlx, but the numbers come from a different place:
        nothing here was printed by the server, so a request that never sent a
        `usage` block closes with no rate at all rather than an estimate.
        """
        if isinstance(ev, OaiRequestStart):
            outs = []
            if ev.model and ev.model != self.model:
                outs.extend(self.feed(ModelLoaded(ev.model)))
            self._oai_task += 1
            self._oai_started = ev.ts
            self._oai_first = None
            self._oai_open = True
            outs.extend(self.feed(
                # The prompt size is in the usage block, which arrives last, so
                # it is genuinely unknown here. None leaves the field empty
                # rather than seeding it with a number the ticks would contradict.
                RequestStart(self.OAI_SLOT, self._oai_task, None, None)))
            return outs

        if isinstance(ev, OaiPrefillTick):
            # The log keeps producing these between requests -- another client,
            # or a warm-up. With no request open they belong to nothing.
            if not self._oai_open:
                return []
            elapsed = self._mlx_elapsed_since(self._oai_started, ev.ts)
            if elapsed <= 0:
                return []
            fraction = (ev.processed / float(ev.total)) if ev.total else 0.0
            return self.feed(PrefillTick(self.OAI_SLOT, self._oai_task, ev.processed,
                                         fraction, elapsed, ev.processed / elapsed))

        if isinstance(ev, OaiGenTick):
            if not self._oai_open:
                return []
            if self._oai_first is None:
                self._oai_first = ev.ts
            elapsed = self._mlx_elapsed_since(self._oai_first, ev.ts)
            if elapsed <= 0:
                return []
            rate = ev.decoded / elapsed
            # No 3s window is tracked here; the same rate in both slots keeps the
            # renderer honest rather than inventing a second series.
            return self.feed(GenTick(self.OAI_SLOT, self._oai_task, ev.decoded, rate, rate))

        if isinstance(ev, OaiRequestAborted):
            return self._oai_abandon()

        if isinstance(ev, OaiRequestEnd):
            if not self._oai_open:
                return []
            self._oai_open = False
            outs = []
            cached = ev.cached_tokens or 0
            if cached:
                outs.extend(self.feed(CacheInfo(self.OAI_SLOT, self._oai_task, cached)))

            prefill_s = self._mlx_elapsed_since(ev.started, ev.first_token)
            # First token to last, not to the end of the response: the trailing
            # usage and [DONE] frames are protocol, not generation.
            gen_s = self._mlx_elapsed_since(ev.first_token, ev.last_token)
            total_s = self._mlx_elapsed_since(ev.started, ev.ts)

            # usage counts the whole prompt including whatever the cache served.
            # Only the remainder was actually computed, and dividing by the full
            # figure would report a prefill rate the machine never achieved.
            fresh = max(0, (ev.prompt_tokens or 0) - cached)
            if prefill_s >= MIN_MEASURABLE_SECONDS and fresh > 0:
                outs.extend(self.feed(PrefillDone(self.OAI_SLOT, self._oai_task,
                                                  prefill_s * 1000.0, fresh,
                                                  fresh / prefill_s)))
            # Same guard as the MLX path: a phase with no duration or no tokens
            # contributes no rate, because a zero would pass the noise filter and
            # then drag down `low`, the weighted average and the slowdown median.
            gen_tokens = ev.completion_tokens or 0
            if gen_s >= MIN_MEASURABLE_SECONDS and gen_tokens > 0:
                outs.extend(self.feed(GenDone(self.OAI_SLOT, self._oai_task,
                                              gen_s * 1000.0, gen_tokens,
                                              gen_tokens / gen_s)))
            # Set on the Request rather than passed through RequestEnd, because
            # that is where the log backend puts it and where _finish reads it
            # from -- so history and the hints stay backend-agnostic.
            if ev.draft is not None:
                req = self.requests.get((self.OAI_SLOT, self._oai_task))
                if req is not None:
                    req.draft = ev.draft
            outs.extend(self.feed(RequestEnd(self.OAI_SLOT, self._oai_task,
                                             total_s * 1000.0, gen_tokens)))
            self._oai_started = self._oai_first = None
            return outs

        return []

    def _oai_abandon(self):
        """Drop the open proxied request without recording anything.

        A cancelled request has real timings but a truncated token count, and
        agents cancel constantly -- opencode abandons a stream the moment a tool
        result makes it stale. Banking those as completed requests would fill the
        history with short, fast-looking rows that never happened.
        """
        self._oai_open = False
        self._oai_started = self._oai_first = None
        req = self.requests.pop((self.OAI_SLOT, self._oai_task), None)
        if req is None:
            return []
        return [Output("line", "", {
            "event": "request_abandoned", "task": req.task,
            "model": req.model, "prompt_tokens": req.prompt_tokens})]

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
            "cache_miss": bool(req and req.cache_miss),
            "draft": req.draft if req else None,
            "started": req.started if req else None}))
        return out


class PhaseStats:
    """Rates for one phase (prefill or generation) of one model."""

    def __init__(self):
        self.tokens = 0
        self.seconds = 0.0
        self.count = 0
        self.peak = None
        self.low = None
        self.recent = []      # per-request rates, oldest first

    def record(self, tokens, seconds, rate):
        self.count += 1
        self.tokens += tokens
        self.seconds += seconds
        self.recent.append(rate)
        if len(self.recent) > RECENT_LIMIT:
            del self.recent[0]
        # Tiny requests are real, but their rates are noise -- see the constant.
        if tokens >= MIN_TOKENS_FOR_EXTREMES:
            self.peak = rate if self.peak is None else max(self.peak, rate)
            self.low = rate if self.low is None else min(self.low, rate)

    @property
    def average(self):
        """Token-weighted: total tokens / total seconds.

        Deliberately NOT the mean of per-request rates. Averaging rates gives a
        4-token request the same weight as a 47,000-token one, which produces a
        number that matches no experience anybody actually had.
        """
        return (self.tokens / self.seconds) if self.seconds > 0 else None

    @property
    def median(self):
        """Typical recent rate. Median, not mean: one contended outlier shouldn't
        move the baseline that slowdown detection compares against."""
        values = sorted(self.recent)
        if not values:
            return None
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0

    def snapshot(self):
        return {"tokens": self.tokens, "seconds": self.seconds, "count": self.count,
                "peak": self.peak, "low": self.low, "avg": self.average,
                "median": self.median, "recent": list(self.recent)}


class ModelStats:
    """Everything tracked for a single model."""

    def __init__(self):
        self.prefill = PhaseStats()
        self.generation = PhaseStats()
        self.cached_tokens = 0
        self.requests = 0
        self.wall_seconds = 0.0
        self.ttfts = []
        self.recent = []
        self.cache_misses = 0
        self.draft_accepted = 0
        self.draft_generated = 0
        self.prompt_sizes = []
        self.recent_cancels = 0

    def note_start(self, prompt_tokens):
        if prompt_tokens:
            self.prompt_sizes.append(prompt_tokens)
            if len(self.prompt_sizes) > RECENT_LIMIT:
                del self.prompt_sizes[0]

    def note_cancel(self):
        self.recent_cancels += 1

    def repeat_count(self):
        """How many times in a row the prompt size was identical.

        A retry loop (client times out, resends the same context) looks exactly
        like this, and is otherwise invisible -- you just see it feel slow.
        """
        if not self.prompt_sizes:
            return 0
        last = self.prompt_sizes[-1]
        count = 0
        for size in reversed(self.prompt_sizes):
            if size != last:
                break
            count += 1
        return count

    def record(self, prefill, generation, end):
        self.requests += 1
        self.wall_seconds += end.get("seconds") or 0.0
        self.recent_cancels = 0            # a completion clears the streak
        if end.get("cache_miss"):
            self.cache_misses += 1
        draft = end.get("draft")
        if draft:
            _rate_unused, accepted, generated, _mean = draft
            self.draft_accepted += accepted
            self.draft_generated += generated
        if prefill:
            self.prefill.record(prefill["tokens"], prefill["seconds"], prefill["rate"])
            self.cached_tokens += prefill.get("cached") or 0
            # The log has no queue-admission timestamp, so prefill duration is the
            # closest honest proxy for time-to-first-token.
            self.ttfts.append(prefill["seconds"])
        if generation:
            self.generation.record(generation["tokens"], generation["seconds"],
                                   generation["rate"])
        self.recent.append({
            "task": end.get("task"),
            "tokens": (prefill or {}).get("tokens", 0),
            "seconds": end.get("seconds") or 0.0,
            "rate": (prefill or {}).get("rate"),
            "share": end.get("prefill_share_pct"),
        })
        if len(self.recent) > RECENT_LIMIT:
            del self.recent[0]

    def snapshot(self):
        total_prompt = self.prefill.tokens + self.cached_tokens
        cache_rate = (self.cached_tokens / float(total_prompt) * 100) if total_prompt else None
        share = None
        if self.wall_seconds > 0:
            share = self.prefill.seconds / self.wall_seconds * 100
        ttft = None
        if self.ttfts:
            ttft = {"min": min(self.ttfts), "max": max(self.ttfts),
                    "avg": sum(self.ttfts) / len(self.ttfts)}
        draft_pct = None
        if self.draft_generated:
            draft_pct = self.draft_accepted / float(self.draft_generated) * 100
        # Mean output tokens per request, used to project when an answer will
        # actually be finished rather than just started.
        avg_out = (self.generation.tokens / float(self.generation.count)
                   if self.generation.count else None)
        avg_prefill_s = (self.prefill.seconds / float(self.prefill.count)
                         if self.prefill.count else None)
        return {
            "requests": self.requests,
            "prefill": self.prefill.snapshot(),
            "generation": self.generation.snapshot(),
            "cache_pct": cache_rate,
            "cached_tokens": self.cached_tokens,
            "cache_misses": self.cache_misses,
            "draft_pct": draft_pct,
            "repeat_count": self.repeat_count(),
            "looping": self.repeat_count() >= LOOP_REPEATS,
            "recent_cancels": self.recent_cancels,
            "avg_output_tokens": avg_out,
            "avg_prefill_seconds": avg_prefill_s,
            "ttft": ttft,
            "prefill_share_pct": share,
            "recent": list(reversed(self.recent)),
        }


class Stats:
    """Per-model session statistics.

    Scoped by model on purpose: the MTP and base builds of the same model differ
    by ~1.34x on code, so pooling them produces an average that describes neither.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.started = clock()
        self.by_model = {}

    def _model(self, model):
        return self.by_model.setdefault(model or "?", ModelStats())

    def record(self, model, prefill, generation, end):
        self._model(model).record(prefill, generation, end)

    def note_start(self, model, prompt_tokens):
        self._model(model).note_start(prompt_tokens)

    def note_cancel(self, model):
        self._model(model).note_cancel()

    def snapshot(self, model):
        data = self.by_model.get(model)
        snap = data.snapshot() if data else ModelStats().snapshot()
        snap["model"] = model or "?"
        snap["session_seconds"] = self._clock() - self.started
        snap["models_seen"] = len(self.by_model)
        return snap


# --------------------------------------------------------------------------
# Layer 3: rendering (pure)
# --------------------------------------------------------------------------

SPINNER_UNICODE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_ASCII = "|/-\\"

RE_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

# Anything we did not generate ourselves gets stripped of control characters
# before it can reach a terminal.
RE_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def safe_text(value, limit=300):
    """Make untrusted text safe to print.

    Two things end up on screen that llmwatch did not author: model names (from
    the log) and agent tool-call arguments (from a Codex session file). Both can
    carry escape sequences. `\\x1b[2J` clears the screen, `\\x1b]0;...\\x07`
    rewrites the terminal title, and some terminals honour considerably worse.

    The realistic path is not a hand-crafted model name: an LLM writes tool-call
    arguments, prompt injection from any page or file the agent reads can shape
    them, Codex records them verbatim, and llmwatch prints them.

    Sanitise at the boundary where untrusted data enters, so no renderer has to
    remember. Note this must happen BEFORE styling: truncate_visible()
    deliberately preserves ANSI, because llmwatch's own colour goes through it.
    """
    if value is None:
        return None
    return RE_CONTROL.sub("", str(value))[:limit]


def short_model_name(value):
    """The last path segment of a model name, sanitised.

    Two sources name the same model differently: the Ollama log writes a file
    path (`/models/qwen3.8-27b.gguf`), Codex writes a provider-qualified id
    (`ollama-local/qwen3.8:27b`). History keys on this form so a turn recorded
    from Codex lines up with the requests recorded from the log.
    """
    if value is None:
        return None
    return safe_text(str(value).rsplit("/", 1)[-1].strip('"'), limit=80)


def visible_len(text):
    """Length as the terminal sees it. ANSI escapes occupy no columns."""
    return len(RE_ANSI.sub("", text))


def truncate_visible(text, width):
    """Cut to `width` visible columns, keeping escape sequences intact.

    Without this a long line wraps onto the next row, which pushes every
    subsequent row down while the next repaint still starts from cursor-home --
    so frames overlay each other and the display turns to garbage. Clipping is
    the safety net; renderers also shorten their content when space is tight.
    """
    if width <= 0:
        return ""
    if visible_len(text) <= width:
        return text
    out, count, i, saw_escape = [], 0, 0, False
    while i < len(text) and count < width:
        match = RE_ANSI.match(text, i)
        if match:
            out.append(match.group())
            i = match.end()
            saw_escape = True
            continue
        out.append(text[i])
        i += 1
        count += 1
    # Reset, or the colour of the cut-off text bleeds into the rest of the line.
    return "".join(out) + ("\033[0m" if saw_escape else "")


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


PREFILL_BATCH = 512      # llama-server reports prefill progress every 512 tokens
GEN_TICK_TOKENS = 50     # and generation roughly every 50


def project(processed, rate, age, total, cap_ahead=None):
    """Estimated position between log ticks.

    llama-server reports progress once per 512-token batch. Freezing the display
    between those reports is what made the tool feel dead; extrapolating from the
    last measured rate keeps it honest and moving. Never runs past the total, so
    the bar cannot claim completion that hasn't happened.
    """
    if rate <= 0 or age <= 0:
        return processed
    projected = processed + rate * age
    # Never run more than one reporting interval ahead. A tick arrives every
    # batch, so a projection beyond that is certainly ahead of reality -- and
    # once it is, the monotonic clamp holds the bar there until reality catches
    # up, which looks like a stall.
    if cap_ahead is not None:
        projected = min(projected, processed + cap_ahead)
    if total is None:
        return projected
    return min(total, projected)


class ProgressFloor:
    """Keeps displayed progress monotonic.

    Between log ticks the position is projected from the last measured rate. When
    the real rate dips, the next tick lands BEHIND the projection and the bar
    visibly rewinds (reported as 47% -> 46% -> 47%). Tokens only ever move
    forward, so the display should too: never show less than was already shown
    for this request.

    Held outside render_live so that function stays deterministic for its inputs;
    pass None and no clamping happens.

    Keyed by phase as well as request. Prefill and generate count different
    things -- tokens read versus tokens written -- and share one task id, so a
    floor keyed on the task alone carries the prompt size into the generate
    line: a 47k prompt made GENERATE open at "47,000 tok" and stay there,
    because every real count was smaller and got clamped up.
    """

    def __init__(self):
        self.key = None
        self.value = 0

    def clamp(self, key, value):
        if key != self.key:              # new request, or new phase of one
            self.key, self.value = key, value
        self.value = max(self.value, value)
        return self.value


def render_live(data, age, style, now=None, floor=None):
    """One live status line. `age` is seconds since this data arrived."""
    if data is None:
        return ""
    event = data.get("event")
    spin = spinner_frame((now if now is not None else time.time()), style)

    if event == "prefill_tick":
        total = data.get("to_process") or 0
        seen = project(data["processed"], data.get("rate", 0), age, total or None,
                       cap_ahead=PREFILL_BATCH)
        if floor is not None:
            seen = floor.clamp((data.get("task"), event), seen)
        fraction = (seen / float(total)) if total else data.get("fraction", 0.0)
        elapsed = data.get("elapsed", 0) + age
        eta = data.get("eta_seconds")
        eta = max(0.0, eta - age) if eta is not None else None
        bar_width = 24 if style.width >= 100 else 16
        cached = data.get("cached") or 0
        cached_note = (" +%s cached" % format(cached, ",")) if cached else ""

        # Prompt batches are done, but the request isn't: the server still has to
        # build logits, validate/restore the KV cache and produce the first token,
        # and llama-server logs NONE of that. Showing "PREFILL 100%" with a clock
        # ticking up reads like a stall, so name the state honestly instead.
        if fraction >= 0.999:
            return "%s %s %s" % (
                spin, style.bold("PREFILL "),
                style.dim("prompt read (%s tok%s) - waiting for first token | elapsed %s"
                          % (format(total, ","), cached_note, fmt_duration(elapsed))))
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
        decoded = int(project(data["decoded"], data.get("rate", 0), age, None,
                              cap_ahead=GEN_TICK_TOKENS))
        if floor is not None:
            decoded = int(floor.clamp((data.get("task"), event), decoded))
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


def project_completion(snap, data, age):
    """Seconds until the ANSWER is finished, not just started.

    Prefill ETA only tells you when text will begin appearing. What people
    actually want to know is when they can read the result, which needs the
    session's measured generation rate and typical output length.
    """
    if not data:
        return None
    gen_rate = (snap.get("generation") or {}).get("avg")
    avg_out = snap.get("avg_output_tokens")
    if not gen_rate or not avg_out:
        return None            # no history yet: don't invent a number
    if data.get("event") == "prefill_tick":
        eta = data.get("eta_seconds")
        if eta is None:
            return None
        remaining = eta - age
        if remaining <= 0:
            # Past the estimate. Clamping to zero made the total a CONSTANT, so
            # the line froze and stopped being feedback at all. Say we're overdue
            # instead and let the caller show a live clock.
            return None
        return remaining + avg_out / gen_rate
    if data.get("event") == "generate_tick":
        done = data.get("decoded", 0) + data.get("rate", 0) * age
        return max(0.0, avg_out - done) / gen_rate
    return None


LONG_CONTEXT_TOKENS = 30000     # beyond this, re-reading dominates every turn
LOW_DRAFT_ACCEPT = 0.35         # below this, speculative decoding is losing
LOOP_REPEATS = 3                # identical prompt sizes in a row = likely retry loop
LOOP_ESCALATE = 5               # by here it is not going to resolve on its own


SLOWDOWN_RATIO = 0.7        # below 70% of the usual rate is worth explaining
SLOWDOWN_MIN_SAMPLES = 3    # need a baseline before calling anything "slow"


def detect_slowdown(data, snap):
    """Is this request measurably slower than usual, and by how much?

    Compares against llmwatch's own measurements rather than guessing from system
    metrics -- the tool already knows the only number that matters. Returns
    (current, typical) or None.
    """
    if not data or not snap:
        return None
    if data.get("event") == "prefill_tick":
        phase = snap.get("prefill") or {}
    elif data.get("event") == "generate_tick":
        phase = snap.get("generation") or {}
    else:
        return None
    typical, current = phase.get("median"), data.get("rate")
    if not typical or not current or phase.get("count", 0) < SLOWDOWN_MIN_SAMPLES:
        return None
    if current >= typical * SLOWDOWN_RATIO:
        return None
    return current, typical


def diagnose(data, snap, style, codex=None, system=None, busiest=None):
    """Short, plain-English reads on what's happening, worth acting on.

    Deliberately terse: this line exists to help someone decide "keep waiting or
    kill it?" in about a second. Raw telemetry belongs on the board, not here.
    Returns at most two findings, most important first.

    `busiest` is the callable that names the biggest CPU consumer. The live loop
    passes SystemProbe.busiest, which rate-limits it; the default is the raw
    read, which is fine for one-shot callers but must not be used per frame.
    """
    if not data:
        return []
    snap = snap or {}
    out = []
    rate = (snap.get("prefill") or {}).get("avg") or data.get("rate")
    total = data.get("to_process") or data.get("prompt_tokens") or 0
    cached = data.get("cached") or 0

    def cost(tokens):
        return (" (~%s)" % fmt_duration(tokens / rate)) if rate and tokens else ""

    if data.get("cache_miss"):
        out.append(style.yellow("! cache gone - rereading all %s tok%s"
                                % (format(total, ","), cost(total))))

    # A measured slowdown, named. "GPU is busy" is not actionable; "2 models
    # loaded (34 GB)" tells you exactly what to close.
    slow = detect_slowdown(data, snap)
    if slow:
        current, typical = slow
        causes = list((system or {}).get("contention") or [])
        if not causes:
            # Nothing obvious in memory or model state: name whatever is eating
            # the machine instead. Costs ~46ms, so only once we already know
            # something is slow, and rate-limited by the caller on top of that.
            worst = (busiest or busiest_process)()
            if worst:
                causes = [worst]
        why = (" - " + ", ".join(causes[:2])) if causes else ""
        out.append(style.yellow("! slow: %.0f vs %.0f tok/s usual%s"
                                % (current, typical, why)))

    # A repeating tool failure is the CAUSE that loop detection sees the symptom
    # of, so it ranks above it: it tells you what to actually go and fix.
    if codex and (codex.get("error_repeats") or 0) >= 2:
        out.append(style.yellow("! same tool error %dx - agent stuck on a broken call"
                                % codex["error_repeats"]))

    if snap.get("looping"):
        repeats = snap.get("repeat_count", LOOP_REPEATS)
        # Quantify it. "May be stuck" is advice; "5x, ~11m spent" is a decision.
        spent = snap.get("avg_prefill_seconds")
        cost = (" ~%s spent" % fmt_duration(spent * repeats)) if spent else ""
        verdict = "likely stuck - interrupt" if repeats >= LOOP_ESCALATE else \
                  "agent may be stuck in a loop"
        out.append(style.yellow("! same prompt %dx - %s%s" % (repeats, verdict, cost)))

    if snap.get("recent_cancels", 0) >= 2:
        out.append(style.yellow("! %d cancels in a row - client keeps timing out"
                                % snap["recent_cancels"]))

    # Key this on tokens actually being READ, not on total context size. A 41k
    # conversation with 41k cached costs nothing to continue; warning about it
    # would be both wrong and alarming.
    if not out and total >= LONG_CONTEXT_TOKENS:
        out.append(style.dim("long chat: reading %s tok this turn%s - consider compacting"
                             % (format(total, ","), cost(total))))

    draft = data.get("draft")
    if draft and data.get("event") == "generate_tick" and draft[0] < LOW_DRAFT_ACCEPT:
        out.append(style.dim("drafts %.0f%% accepted - MTP is slowing this one down"
                             % (draft[0] * 100)))

    if not out and cached and total:
        out.append(style.dim("cache working: only %s of %s tok to read"
                             % (format(total, ","), format(total + cached, ","))))

    if not out and data.get("status"):
        out.append(style.dim(data["status"]))
    return out[:2]


def render_live_detail(data, snap, style, age=0.0, codex=None, system=None,
                       busiest=None):
    """Second live line: why this is slow, and when the answer will be ready."""
    # One finding per LINE, not joined with spaces. Concatenated, two findings
    # plus a projection ran past 120 columns -- a run-on that both wrapped (and
    # corrupted the frame) and read as a wall of text.
    lines = list(diagnose(data, snap, style, codex, system, busiest))
    finish = project_completion(snap or {}, data, age)
    if finish is not None:
        lines.append(style.dim("answer ready ~%s" % fmt_duration(finish)))
    elif data and data.get("event") in ("prefill_tick", "generate_tick"):
        # No estimate available, or the estimate has been overrun. Either way a
        # live clock is better than a frozen number or an empty line.
        waited = (data.get("elapsed") or 0) + age
        lines.append(style.dim("running past estimate - %s so far" % fmt_duration(waited)))
    return lines


def render_idle(age, style, system=None):
    """Idle is ambiguous: nothing has happened yet, or nothing CAN happen.
    Saying which is the difference between waiting patiently and waiting
    pointlessly."""
    spin = spinner_frame(time.time(), style)
    system = system or {}
    if system.get("server_ok") is False:
        problem = system.get("server_problem") or "Ollama is not running"
        return style.yellow("%s %s - start it and this will pick up automatically"
                            % (spin, problem))
    if system.get("models_loaded") == 0:
        return style.dim("%s no model loaded - the first request pays a load "
                         "(~10s for a 27B)  (idle %s)" % (spin, fmt_duration(age)))
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


SPARK_UNICODE = "▁▂▃▄▅▆▇█"
SPARK_ASCII = ".:-=+*#"


def sparkline(values, style):
    """Trend of recent per-request rates.

    Scaled between the observed min and max rather than from zero: the point is
    to see change (throttling, contention, a slower model) and a zero-based
    scale flattens exactly that.
    """
    values = [v for v in (values or []) if v is not None]
    if not values:
        return ""
    chars = SPARK_UNICODE if style.unicode else SPARK_ASCII
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return chars[len(chars) // 2] * len(values)
    span = high - low
    return "".join(chars[min(len(chars) - 1,
                             int((v - low) / span * (len(chars) - 1) + 0.5))]
                   for v in values)


def _rate(value):
    return ("%6.1f" % value) if value is not None else "     -"


def render_system(system, style):
    """One interpretive line, not a wall of gauges.

    No CPU% or GPU%: during inference the GPU is pinned near 100% and the CPU
    near idle whether throughput is good or bad, so those numbers never change
    and never help. What changes is contention for memory bandwidth.
    """
    if not system:
        return []
    causes = system.get("contention") or []
    if causes:
        return ["%s %s" % (style.bold("SYSTEM  "),
                           style.yellow("! " + ", ".join(causes)))]
    bits = []
    if system.get("models_loaded") is not None:
        bits.append("%d model%s" % (system["models_loaded"],
                                    "" if system["models_loaded"] == 1 else "s"))
    if system.get("swap_used_gb") is not None:
        bits.append("swap %.1f GB" % system["swap_used_gb"])
    if system.get("load1") is not None:
        bits.append("load %.1f" % system["load1"])
    if not bits:
        return []
    return ["%s %s" % (style.bold("SYSTEM  "),
                       style.dim("clear - " + ", ".join(bits)))]


def render_board(snap, style, width=80, compact=False, system=None):
    """The stats board. Returns a list of lines, no trailing newlines."""
    lines = []
    pre, gen = snap["prefill"], snap["generation"]

    lines.append("%s peak %s   avg %s   low %s tok/s   %s" % (
        style.bold("PREFILL "), style.green(_rate(pre["peak"])), _rate(pre["avg"]),
        style.yellow(_rate(pre["low"])),
        style.dim("%s tok - %s" % (format(pre["tokens"], ","), fmt_duration(pre["seconds"])))))
    if not compact:
        spark = sparkline(pre["recent"], style)
        if spark:
            lines.append("         %s %s" % (style.cyan(spark),
                                             style.dim("last %d" % len(pre["recent"]))))

    lines.append("%s peak %s   avg %s   low %s tok/s   %s" % (
        style.bold("GENERATE"), style.green(_rate(gen["peak"])), _rate(gen["avg"]),
        style.yellow(_rate(gen["low"])),
        style.dim("%s tok - %s" % (format(gen["tokens"], ","), fmt_duration(gen["seconds"])))))
    if not compact:
        spark = sparkline(gen["recent"], style)
        if spark:
            lines.append("         %s %s" % (style.cyan(spark),
                                             style.dim("last %d" % len(gen["recent"]))))

    if snap.get("cache_pct") is not None:
        misses = snap.get("cache_misses") or 0
        note = ("%s tok never recomputed" % format(snap["cached_tokens"], ","))
        if misses:
            note += " - %d full reread%s" % (misses, "" if misses == 1 else "s")
        lines.append("%s %s reused - %s" % (
            style.bold("CACHE   "), "%4.0f%%" % snap["cache_pct"], style.dim(note)))

    if snap.get("draft_pct") is not None:
        pct = snap["draft_pct"]
        verdict = "speculative decoding is paying off" if pct >= 50 else \
                  "drafts often rejected - base build may be faster"
        lines.append("%s %s accepted - %s" % (
            style.bold("DRAFT   "), "%4.0f%%" % pct, style.dim(verdict)))

    ttft = snap.get("ttft")
    if ttft:
        # Approximate: prefill duration. The log has no queue-admission time, so
        # true client-send-to-first-token cannot be derived from it.
        lines.append("%s min %s - avg %s - max %s %s" % (
            style.bold("TTFT    "), fmt_duration(ttft["min"]), fmt_duration(ttft["avg"]),
            fmt_duration(ttft["max"]), style.dim("(approx: prefill time)")))

    share = snap.get("prefill_share_pct")
    if share is not None:
        bar_w = 20 if width >= 70 else 10
        filled = int(round(bar_w * share / 100.0))
        bar = ("█" * filled + "░" * (bar_w - filled)) if style.unicode else \
              ("#" * filled + "-" * (bar_w - filled))
        lines.append("%s %s %s" % (style.bold("WAIT    "), style.yellow(bar),
                                   style.dim("%.0f%% of session spent in prefill" % share)))
    lines.extend(render_system(system, style))
    return lines


def render_codex(state, style, width=80):
    """What the agent itself is doing. Opt-in (--codex): this reads your Codex
    session file, which contains command text and file paths."""
    if not state:
        return []
    out = []
    action = state.get("action")
    if action:
        out.append("%s %s" % (style.bold("%-12s" % "last action"),
                                style.cyan(action)))
        if state.get("detail"):
            # Keep it glanceable. The full command lives in your agent's window;
            # here it only has to be recognisable.
            limit = max(30, min(72, width - 20))
            detail = state["detail"]
            if len(detail) > limit:
                detail = detail[:limit - 1] + "…" if style.unicode else detail[:limit - 3] + "..."
            out.append("%-12s %s" % ("", style.dim(detail)))
    if state.get("error"):
        repeats = state.get("error_repeats") or 1
        label = "tool failed" if repeats < 2 else "failing %dx" % repeats
        limit = max(30, min(72, width - 20))
        out.append("%s %s" % (style.bold("%-12s" % label),
                                style.yellow(state["error"][:limit])))
    # The turn clock: how long since you pressed enter. Every other number here
    # is about one model request; this is the one you are actually waiting on.
    elapsed = state.get("turn_seconds")
    running = elapsed is not None
    # Once a turn has ended its tool count belongs to "last turn", not to a
    # "this turn" line that is no longer about anything in flight.
    if running or not state.get("last_turn"):
        facts = []
        if running:
            facts.append("%s so far" % fmt_duration(elapsed))
        if state.get("calls"):
            facts.append("%d tool calls" % state["calls"])
        if running and state.get("effort"):
            facts.append("effort %s" % state["effort"])
        if facts:
            out.append("%-12s %s" % ("this turn", style.dim(" - ".join(facts))))
    if state.get("waiting_since") is not None:
        out.append("%-12s %s" % ("waiting on", style.dim(
            "model for %s" % fmt_duration(state["waiting_since"]))))
    out.extend(render_last_turn(state.get("last_turn"), style))
    return out


def render_last_turn(turn, style):
    """What the turn that just finished cost, and whether that was normal.

    The number people want after an agent goes quiet for twenty minutes is not a
    token rate: it is "how long did that take, and is that what this usually
    takes?".
    """
    if not turn or turn.get("seconds") is None:
        return []
    parts = [fmt_duration(turn["seconds"])]
    if turn.get("effort"):
        parts.append("effort %s" % turn["effort"])
    if turn.get("tool_calls"):
        parts.append("%d tool calls" % turn["tool_calls"])
    if not turn.get("completed"):
        parts.append(turn.get("reason") or "interrupted")
    line = "%-12s %s" % ("last turn", style.bold(parts[0]))
    if len(parts) > 1:
        line += style.dim(" - " + " - ".join(parts[1:]))

    typical = turn.get("typical_seconds")
    out = [line]
    if typical:
        # Only worth saying when it is far enough from typical to act on.
        ratio = turn["seconds"] / typical if typical else 1.0
        if ratio >= 1.0 + SAME_WITHIN:
            note = "%.1fx your usual %s" % (ratio, fmt_duration(typical))
        elif ratio <= 1.0 - SAME_WITHIN:
            note = "faster than your usual %s" % fmt_duration(typical)
        else:
            note = "about your usual %s" % fmt_duration(typical)
        out.append("%-12s %s" % ("", style.dim(note)))
    return out


def render_recent(rows, style, width=80, limit=6):
    """Recent requests, newest first. Carries a header row: without one the
    columns are four unlabelled numbers and the reader has to guess."""
    out = [style.dim("%-6s %12s %9s %14s   %s"
                     % ("task", "prompt", "total", "prefill speed", "share of wait"))]
    for row in rows[:limit]:
        share = ("%3.0f%% reading" % row["share"]) if row.get("share") is not None else ""
        rate = ("%6.1f tok/s" % row["rate"]) if row.get("rate") is not None else ""
        out.append("%-6s %12s %9s %14s   %s" % (
            row.get("task", "?"), format(row.get("tokens", 0), ",") + " tok",
            fmt_duration(row.get("seconds")), rate, style.dim(share)))
    return out


HELP_TEXT = [
    ("", "llmwatch shows what your local model is doing right now."),
    ("", ""),
    ("PREFILL", "The model is READING your prompt. Silent in your agent, and usually"),
    ("", "the long part: a 47,000-token agent prompt can take minutes."),
    ("GENERATE", "The model is WRITING its answer. This is the text you actually see."),
    ("", ""),
    ("the bar", "How much of the prompt still needs computing, with an ETA. It moves"),
    ("", "smoothly because position is projected from the last measured rate --"),
    ("", "the server only reports progress once every 512 tokens."),
    ("+N cached", "Tokens reused from a previous request, so they cost nothing. Only"),
    ("", "tokens that actually need computing are counted in the bar."),
    ("waiting for", "Prompt batches are done, but the server is still building logits and"),
    ("first token", "checking its cache. It logs nothing here, so the clock keeps running."),
    ("", ""),
    ("peak/avg/low", "Per request, for the current model only. avg is token-weighted"),
    ("", "(total tokens / total seconds), not an average of rates. Requests under"),
    ("", "64 tokens are excluded from peak/low so cache hits can't skew them."),
    ("CACHE", "Share of all prompt tokens this session that were reused, not recomputed."),
    ("TTFT", "Time to first token, approximated by prefill duration -- the log has no"),
    ("", "record of when your client sent the request."),
    ("WAIT", "Share of total wall clock spent reading rather than writing. On a coding"),
    ("", "agent this is often above 90%, which is the point of the tool."),
    ("sparkline", "The last 20 request rates. A downward trend means throttling,"),
    ("", "memory pressure, or something else competing for the GPU."),
    ("DRAFT", "Speculative decoding acceptance (-mtp builds). Above ~50% it is"),
    ("", "winning; well below, the plain build is likely faster."),
    ("", ""),
    ("warnings", "The line under the live bar tells you whether to keep waiting:"),
    ("cache gone", "Nothing was reused - the whole prompt is being reread. Slowest case."),
    ("looping", "Same prompt size several times running: your client may be"),
    ("", "retrying, so the wait may never end. Worth interrupting."),
    ("cancels", "Your client keeps disconnecting - usually a timeout set too low."),
    ("long chat", "You are re-reading a lot every turn. Compacting will help more"),
    ("", "than any speed tuning."),
    ("answer ready", "Estimated finish for the whole answer, from your own measured"),
    ("", "rates -- not just when text starts appearing."),
    ("", ""),
    ("codex pane", "With --codex: what your agent last did, and how long it has been"),
    ("", "waiting. Reads your Codex session file, so it is opt-in."),
    ("this turn", "Wall clock since you pressed enter, with the reasoning effort the"),
    ("last turn", "turn is running at. When a turn ends its total is kept, so the next"),
    ("", "one can be called normal or slow against your own history. Tool time"),
    ("", "is included: it is time you waited. Durations only are stored, never"),
    ("", "the message text sitting next to them in the session file."),
    ("", ""),
    ("compare", "Press c to compare two models on your own recorded requests:"),
    ("", "speed, time-to-first-token, cache, what it saves per request, and"),
    ("", "median turn time split by effort level. See also --turns."),
    ("", ""),
    ("keys", "h help   -   c compare   -   u upgrade, when one is available"
             "   -   q or ctrl-c quit"),
]


SAME_WITHIN = 0.05          # closer than this is "about the same", not "1.03x"


class UIState:
    """Which view is showing, and what the picker has selected.

    Kept separate from rendering so every transition can be tested without a
    terminal.
    """

    def __init__(self):
        self.view = "live"          # live | help | picker | compare | upgrade
        self.cursor = 0
        self.model_a = None
        self.model_b = None
        # Set only by an explicit confirmation in the upgrade view. The loop
        # watches it; nothing else in here runs a command.
        self.upgrade_requested = False
        # Filled in by the loop on the first frame the pane is visible, and
        # cleared when it closes. Working the plan out shells out to `git
        # status` and scans PATH, which is not something the render loop can
        # afford to redo ten times a second. Resolution lives in the loop, not
        # here: every transition in this class stays testable without a
        # terminal, a checkout, or a subprocess.
        self.upgrade_plan = None

    def reset_selection(self):
        self.cursor = 0
        self.model_a = None
        self.model_b = None


def handle_key(state, key, models, update=None):
    """Apply a keypress. Returns True if anything changed (so we repaint now)."""
    names = [m["model"] for m in models]

    if key in ("q", "Q"):
        raise KeyboardInterrupt

    # Confirmation, handled before the general keys so `u` cannot both open this
    # pane and confirm it. One keystroke opens, a different one commits.
    if state.view == "upgrade":
        if key in ("y", "Y"):
            state.upgrade_requested = True
            return True
        if key in ("ESC", "n", "N", "u", "U"):
            state.view = "live"
            state.upgrade_plan = None   # do not answer tomorrow with today's tree
            return True
        return False

    if key in ("u", "U"):
        # Offered only when there is something to upgrade to. Otherwise the key
        # would invite people to reinstall the version they already have.
        if not update:
            return False
        state.view = "upgrade"
        return True

    if key in ("h", "H", "?"):
        state.view = "live" if state.view == "help" else "help"
        return True

    if key in ("c", "C"):
        if state.view == "picker":
            state.view = "live"
        else:
            state.view = "picker"
            state.reset_selection()
        return True

    if state.view == "picker":
        if key in ("UP", "k"):
            state.cursor = max(0, state.cursor - 1)
            return True
        if key in ("DOWN", "j"):
            state.cursor = min(max(0, len(names) - 1), state.cursor + 1)
            return True
        # isdecimal, not isdigit: isdigit admits superscripts and other Unicode
        # digits that int() then rejects, and superscript two is a dedicated key
        # on French AZERTY. The reader hands us raw decoded bytes, so that
        # ValueError escapes follow() as a traceback, in raw mode, mid-render.
        if key and key.isdecimal() and key != "0":
            index = int(key) - 1
            if index < len(names):
                state.cursor = index
                return True
            return False
        if key in ("\r", "\n"):
            if not names:
                return False
            chosen = names[state.cursor]
            if state.model_a is None:
                state.model_a = chosen
            else:
                state.model_b = chosen
                state.view = "compare"
            return True
        if key == "ESC":
            state.view = "live"
            return True
        return False

    if state.view == "compare" and key == "ESC":
        state.view = "picker"
        state.model_b = None
        return True
    if state.view == "help" and key == "ESC":
        state.view = "live"
        return True
    return False


def render_picker(models, state, style, cols=80):
    """The model list. Models with no recorded requests stay selectable: telling
    someone how to get data is more useful than a greyed-out row that explains
    nothing."""
    lines = [style.dim("   #  %-28s %6s %11s   %s"
                       % ("model", "req", "gen tok/s", "last seen"))]
    for index, entry in enumerate(models[:9]):
        marker = style.cyan(">") if index == state.cursor else " "
        chosen = " (A)" if entry["model"] == state.model_a else ""
        if entry["requests"]:
            seen = "%s ago" % fmt_duration(entry["last_seen"] or 0)
            rate = "%.1f" % entry["gen_rate"] if entry["gen_rate"] else "-"
            tail = "%6d %11s   %s" % (entry["requests"], rate, seen)
        else:
            tail = "%6d %11s   %s" % (0, "-", style.dim("no data yet"))
        name = (entry["model"] + chosen)[:28]
        row = " %s %d  %-28s %s" % (marker, index + 1, name, tail)
        lines.append(style.bold(row) if index == state.cursor else row)

    lines.append("")
    if state.model_a:
        lines.append("  A: %s%s" % (style.cyan(state.model_a),
                                    style.dim("      now pick a second model")))
    else:
        lines.append(style.dim("  pick the first model"))
    lines.append(style.dim("  up/down move   1-9 jump   enter pick   esc back"))
    return lines


def _verdict(a_value, b_value, label_a="A", label_b="B", unit="x faster"):
    if not a_value or not b_value:
        return "no comparison"
    ratio = a_value / b_value
    if abs(ratio - 1.0) <= SAME_WITHIN:
        return "about the same"
    if ratio > 1:
        return "%s %.2f%s" % (label_a, ratio, unit)
    return "%s %.2f%s" % (label_b, 1 / ratio, unit)


def _pair_bars(label, a_value, b_value, style, suffix="tok/s", width=20):
    """Two bars normalised to the larger value: the comparison should read
    visually before it reads numerically."""
    top = max(a_value or 0, b_value or 0) or 1
    out = []
    for tag, value in (("A", a_value), ("B", b_value)):
        filled = int(round(width * (value or 0) / top))
        bar = ("█" * filled + "░" * (width - filled)) if style.unicode else \
              ("#" * filled + "-" * (width - filled))
        out.append("   %-10s %s %s %s" % (label if tag == "A" else "", tag,
                                          style.cyan(bar),
                                          ("%6.1f %s" % (value, suffix)) if value
                                          else "     - "))
    return out


def render_turns(turns_a, turns_b, style):
    """Turn time in the comparison: the only row here measured in the unit you
    actually wait in.

    The verdict lives on the per-effort rows, never on the total. A model you ran
    mostly on low effort and one you ran mostly on high are not comparable, and
    pooling them produces exactly the kind of confident-but-meaningless ratio the
    prompt-size buckets exist to prevent -- 'A 2.05x quicker' sitting directly
    above a row showing A losing at every effort level both sides share.
    """
    turns_a, turns_b = turns_a or {}, turns_b or {}
    if not (turns_a.get("turns") or turns_b.get("turns")):
        return []

    def cell(profile):
        if not profile.get("turns"):
            return "no turns yet"
        return "%s (n=%d)" % (fmt_duration(profile["median_seconds"]), profile["turns"])

    lines = ["", style.dim("   median agent turn: your prompt to the final answer, "
                           "tool time included")]
    lines.append("   %-10s A %-18s B %-18s %s" % (
        "TURN", cell(turns_a), cell(turns_b),
        style.dim("all efforts pooled")))

    efforts_a = turns_a.get("efforts") or {}
    efforts_b = turns_b.get("efforts") or {}
    for name in sorted(set(efforts_a) | set(efforts_b)):
        side_a, side_b = efforts_a.get(name) or {}, efforts_b.get(name) or {}
        thin = min(side_a.get("turns", 0), side_b.get("turns", 0)) < MIN_COMPARE_TURNS
        if not (side_a.get("turns") and side_b.get("turns")):
            result = style.dim("one side only")
        elif thin:
            result = style.dim("need %d each" % MIN_COMPARE_TURNS)
        else:
            result = style.bold(_verdict(side_b.get("median_seconds"),
                                         side_a.get("median_seconds"), unit="x quicker"))
        lines.append("   %-10s A %-18s B %-18s %s" % (
            "  " + name, cell(side_a), cell(side_b), result))

    interrupted = (turns_a.get("interrupted") or 0) + (turns_b.get("interrupted") or 0)
    if interrupted:
        lines.append(style.dim("     %d interrupted turn%s excluded: they measure your "
                               "patience, not the model"
                               % (interrupted, "" if interrupted == 1 else "s")))
    return lines


def render_compare(profile_a, profile_b, buckets, style, cols=80, days=30,
                   turns_a=None, turns_b=None):
    """The comparison. Also used by `--compare` so the CLI and the TUI cannot
    drift apart."""
    if not profile_a or not profile_b:
        return [style.dim("  pick two models to compare")]

    missing = [p for p in (profile_a, profile_b) if not p.get("requests")]
    if missing:
        lines = []
        for profile in missing:
            lines.append("  %s has no recorded requests yet." % style.bold(profile["model"]))
        lines.append("")
        lines.append("  Run anything against it and it will appear here:")
        lines.append(style.cyan('      ollama run %s "hello"' % missing[0]["model"]))
        lines.append(style.dim("  or point your agent at it for a turn. A comparison needs "
                               "about %d" % MIN_COMPARE_SAMPLES))
        lines.append(style.dim("  requests per side before it will report a ratio."))
        have = [p for p in (profile_a, profile_b) if p.get("requests")]
        if have:
            lines.append("")
            for profile in have:
                lines.append(style.dim("  %s already has %d requests recorded."
                                       % (profile["model"], profile["requests"])))
        return lines

    lines = ["   %s      %s" % (
        style.dim("%d requests, last %s ago" % (profile_a["requests"],
                                                fmt_duration(profile_a["last_seen"] or 0))),
        style.dim("%d requests, last %s ago" % (profile_b["requests"],
                                                fmt_duration(profile_b["last_seen"] or 0)))), ""]

    gen = _pair_bars("GENERATE", profile_a.get("gen_rate"), profile_b.get("gen_rate"), style)
    gen[0] += "  " + style.bold(_verdict(profile_a.get("gen_rate"), profile_b.get("gen_rate")))
    lines.extend(gen)
    lines.extend(_pair_bars("PREFILL", profile_a.get("prefill_rate"),
                            profile_b.get("prefill_rate"), style))

    if profile_a.get("ttft") and profile_b.get("ttft"):
        sooner = abs(profile_a["ttft"] - profile_b["ttft"])
        who = "A" if profile_a["ttft"] < profile_b["ttft"] else "B"
        # "B 0.0s sooner" is noise dressed as a finding.
        note = ("%s %s sooner" % (who, fmt_duration(sooner))) if sooner >= 1 \
            else "about the same"
        lines.append("   %-10s A %-10s B %-10s %s" % (
            "TTFT", fmt_duration(profile_a["ttft"]), fmt_duration(profile_b["ttft"]),
            style.dim(note)))
    if profile_a.get("cache_pct") is not None or profile_b.get("cache_pct") is not None:
        lines.append("   %-10s A %-10s B %-10s" % (
            "CACHE", "%.0f%%" % (profile_a.get("cache_pct") or 0),
            "%.0f%%" % (profile_b.get("cache_pct") or 0)))
    if profile_a.get("draft_pct") is not None or profile_b.get("draft_pct") is not None:
        fmt = lambda p: ("%.0f%%" % p["draft_pct"]) if p.get("draft_pct") is not None \
            else "not a speculative build"
        lines.append("   %-10s A %-10s B %s" % ("DRAFT", fmt(profile_a), fmt(profile_b)))

    if buckets:
        lines.append("")
        lines.append(style.dim("   by prompt size          A            B          result"))
        for row in buckets:
            if row["ratio"]:
                result = _verdict(row["a_rate"], row["b_rate"])
            elif row["enough"]:
                result = "about the same"
            else:
                result = style.dim("need %d each" % MIN_COMPARE_SAMPLES)
            lines.append("     %-7s %-5s %11s %12s      %s" % (
                row["band"], "miss" if row["cache_miss"] else "hit",
                "%.1f (n=%d)" % (row["a_rate"], row["a_n"]) if row["a_rate"]
                else "n=%d" % row["a_n"],
                "%.1f (n=%d)" % (row["b_rate"], row["b_n"]) if row["b_rate"]
                else "n=%d" % row["b_n"],
                result))

    lines.extend(render_median_request(profile_a, profile_b, style))
    lines.extend(render_turns(turns_a, turns_b, style))
    return lines


def median_request_time(profile, prompt_tokens, gen_tokens):
    """Seconds to finish a request of this shape: read the prompt, write the
    answer."""
    prefill_rate, gen_rate = profile.get("prefill_rate"), profile.get("gen_rate")
    if not prefill_rate or not gen_rate:
        return None
    return prompt_tokens / prefill_rate + gen_tokens / gen_rate


def render_median_request(profile_a, profile_b, style):
    """Rates are abstract. Seconds saved on the work you actually do are not."""
    prompts = [p.get("median_prompt_tokens") for p in (profile_a, profile_b)
               if p.get("median_prompt_tokens")]
    answers = [p.get("median_gen_tokens") for p in (profile_a, profile_b)
               if p.get("median_gen_tokens")]
    if not prompts or not answers:
        return []
    prompt_tokens = sum(prompts) / len(prompts)
    gen_tokens = sum(answers) / len(answers)
    a_time = median_request_time(profile_a, prompt_tokens, gen_tokens)
    b_time = median_request_time(profile_b, prompt_tokens, gen_tokens)
    if not a_time or not b_time:
        return []

    out = ["", style.dim("   on your median request (%s tok prompt, %s tok answer)"
                         % (format(int(prompt_tokens), ","), format(int(gen_tokens), ",")))]
    top = max(a_time, b_time)
    for tag, value in (("A", a_time), ("B", b_time)):
        filled = int(round(20 * value / top))
        bar = ("█" * filled) if style.unicode else ("#" * filled)
        out.append("     %s  %-8s %s" % (tag, fmt_duration(value), style.cyan(bar)))
    saved = abs(a_time - b_time)
    who = "A" if a_time < b_time else "B"
    if saved >= 1:
        out.append("     " + style.bold("%s saves %s per request" % (who, fmt_duration(saved))))
    else:
        out.append("     " + style.dim("no meaningful difference per request"))
    return out


def installed_models():
    """Models on disk. The picker shows these too, so a model you have never
    measured is visible with an explanation rather than simply absent."""
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    names = []
    for row in out.splitlines()[1:]:
        row = row.strip()
        if row:
            names.append(safe_text(row.split()[0], limit=80))
    return names


def pickable_models(history):
    """Recorded models first (they can actually be compared), then installed
    ones with no data."""
    recorded = history.models() if history else []
    seen = {entry["model"] for entry in recorded}
    unmeasured = [{"model": name, "requests": 0, "gen_rate": None, "last_seen": None}
                  for name in installed_models() if name not in seen]
    return recorded + sorted(unmeasured, key=lambda entry: entry["model"])


def build_compare(history, model_a, model_b, style, cols=100, days=30, now=None):
    """One entry point for both the TUI view and `--compare`, so they cannot
    drift apart."""
    if not history or not model_a or not model_b:
        return [style.dim("  pick two models to compare")]
    return render_compare(history.profile(model_a, days=days, now=now),
                          history.profile(model_b, days=days, now=now),
                          history.compare(model_a, model_b, days=days, now=now),
                          style, cols, days,
                          turns_a=history.turn_profile(model_a, days=days, now=now),
                          turns_b=history.turn_profile(model_b, days=days, now=now))


def render_help(style, cols, rows):
    out = [style.bold("llmwatch %s - how to read this" % __version__), ""]
    for label, text in HELP_TEXT:
        # Pad BEFORE colouring: ANSI escapes have no width on screen but do count
        # in %-12s, which silently destroys column alignment.
        padded = "%-12s" % label
        out.append("  " + (style.cyan(padded) if label else padded) +
                   " " + (text if label else style.dim(text)))
    return out[:max(1, rows - 1)]


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


# --------------------------------------------------------------------------
# Layer 4b: the proxy
#
# For a server speaking the OpenAI API there is no log worth tailing -- see the
# Oai* events for why -- so llmwatch stands between the client and the server
# and reads the numbers off the wire instead.
#
# This sits in the middle of somebody's coding loop, which sets the rules:
#
#   * bytes are relayed the moment they arrive and never accumulated, because
#     any buffering here is felt as lag on every token
#   * every failure path still proxies. A monitor that takes the model down
#     with it is worse than one that misses a request, so an unreadable chunk
#     costs a measurement and nothing else
#   * message content is never read, kept or written. Only `usage`, the model
#     name and arrival times leave this class -- the same property History
#     documents about itself
# --------------------------------------------------------------------------

DEFAULT_UPSTREAM = "http://127.0.0.1:8080"
# urlopen will happily open file:, ftp: or a custom scheme, and --upstream is
# the one part of the proxy's configuration that comes from outside. Checked
# where it is accepted rather than trusted at the call site.
UPSTREAM_SCHEMES = ("http://", "https://")
DEFAULT_PROXY_PORT = 8081
LOOPBACK_HOSTS = frozenset(["127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"])

INSTRUMENTED_PATHS = ("/v1/chat/completions", "/v1/completions")

# Connection-level headers describe one hop and must not be copied onto the
# next: forwarding an upstream `Transfer-Encoding: chunked` while writing the
# decoded body back is how a proxy corrupts a response.
HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
])

# A prefill on a long prompt is minutes of silence before the first byte, so
# this cannot be tight. It exists only so a vanished upstream cannot pin a
# thread forever.
PROXY_TIMEOUT = 900.0
# Cap on a non-streaming body held in memory to read its usage block. Past this
# the response is relayed unmeasured rather than buffered.
MAX_BUFFERED_BODY = 32 * 1024 * 1024


def _sse_events(buffer, chunk):
    """Split streamed bytes into complete `data:` payloads.

    Returns (leftover, payloads). Incremental by construction: a chunk that
    ends mid-line leaves the remainder in `leftover` for the next call, so
    nothing waits on the whole response.
    """
    buffer += chunk
    payloads = []
    while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        line = line.strip()
        if line.startswith(b"data:"):
            payloads.append(line[5:].strip())
    return buffer, payloads


def draft_counts(timings):
    """(accepted, generated) drafted tokens from a `timings` block, or None.

    llama-server reports speculative decoding here; every other OpenAI server
    omits the block entirely. Ordered accepted-first to match what the log
    backend already hands DraftAcceptance, so there is one convention rather
    than one per source.

    None means "no measurement", which is not the same as zero: a server with
    no drafter, a run that drafted nothing, and a garbled block all have no
    acceptance rate to show, and printing 0% for them would read as a drafter
    performing terribly rather than as a drafter that was never there.
    """
    if not isinstance(timings, dict):
        return None
    if "draft_n" not in timings or "draft_n_accepted" not in timings:
        return None
    try:
        generated = int(timings["draft_n"])
        accepted = int(timings["draft_n_accepted"])
    except (TypeError, ValueError):
        return None
    if generated <= 0 or accepted < 0:
        return None
    # Above 1.0 would print as "140% accepted". Whatever produced it, it is not
    # a measurement.
    if accepted > generated:
        return None
    return (accepted, generated)


def read_stream_usage(payload):
    """Pull what is measurable out of one SSE payload.

    Returns (produced_content, usage_or_None, timings_or_None). Deliberately
    narrow: the delta is tested for emptiness and thrown away without being
    stored or returned, so no caller can accidentally end up holding message
    text.
    """
    if not payload or payload == b"[DONE]":
        return False, None, None
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return False, None, None
    if not isinstance(obj, dict):
        return False, None, None
    produced = False
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        # `text` is the /v1/completions spelling of content. tool_calls count
        # too, and must: a coding agent's turn is frequently nothing but a tool
        # call, and treating those as producing no output leaves the busiest
        # half of an agent session invisible.
        produced = (bool(delta.get("content")) or bool(first.get("text"))
                    or bool(delta.get("tool_calls")))
    usage = obj.get("usage")
    timings = obj.get("timings")
    return (produced, usage if isinstance(usage, dict) else None,
            timings if isinstance(timings, dict) else None)


def usage_counts(usage):
    """(prompt, cached, completion) from a usage block, missing fields as 0."""
    if not isinstance(usage, dict):
        return (0, 0, 0)
    details = usage.get("prompt_tokens_details")
    cached = 0
    if isinstance(details, dict):
        try:
            cached = int(details.get("cached_tokens") or 0)
        except (TypeError, ValueError):
            cached = 0
    def _int(value):
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
    return (_int(usage.get("prompt_tokens")), cached,
            _int(usage.get("completion_tokens")))


def prepare_body(raw):
    """(body_to_forward, model, is_stream) for an instrumented request.

    Streaming responses only carry a usage block when the client asked for one,
    so llmwatch asks on the client's behalf. mlx_lm.server reads
    stream_options["include_usage"] with a bare subscript, so a client that
    sends stream_options without that key crashes the upstream -- normalising
    the dict here fixes that for free rather than passing the crash through.

    On anything unparseable the body is returned untouched: a request that
    cannot be measured must still be a request that works.
    """
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return raw, None, False
    except (ValueError, TypeError):
        return raw, None, False

    model = short_model_name(obj.get("model"))
    is_stream = bool(obj.get("stream"))
    if not is_stream:
        return raw, model, False

    options = obj.get("stream_options")
    if not isinstance(options, dict):
        options = {}
    if options.get("include_usage") is True:
        return raw, model, True
    options = dict(options)
    options["include_usage"] = True
    obj = dict(obj)
    obj["stream_options"] = options
    try:
        return json.dumps(obj).encode("utf-8"), model, True
    except (TypeError, ValueError):
        return raw, model, True


_PROXY_SERVER_CLASS = None


def _proxy_server_class():
    """Build the proxy's server class, importing an HTTP stack only now.

    llmwatch watches a local model and deliberately does not load an HTTP stack
    to start up -- see TestNoNetworkDependency, and the update check, which
    imports urllib inside the one function that needs it for the same reason.
    A proxy cannot avoid the dependency, but it can avoid paying for it in
    every run that never proxies anything, so the classes are built here, on
    first use, rather than at import.
    """
    global _PROXY_SERVER_CLASS
    if _PROXY_SERVER_CLASS is not None:
        return _PROXY_SERVER_CLASS

    import http.server
    import urllib.error
    import urllib.request

    class _ProxyHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            """Silence. The base class writes to stderr, which is the terminal the
            TUI is drawing on."""

        # All verbs go through one path; only POSTs to a completions endpoint are
        # measured, everything else is a plain relay so `GET /v1/models` and any
        # endpoint added upstream keep working without llmwatch knowing about them.
        def do_GET(self):
            self._relay()

        def do_POST(self):
            self._relay()

        def do_DELETE(self):
            self._relay()

        def do_OPTIONS(self):
            self._relay()

        def _emit(self, ev):
            try:
                self.server.emit(ev)
            except Exception:
                pass

        def _relay(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""

            # Before anything else is done with it: the target decides which
            # host this request goes to, and only a plain rooted path may.
            path = safe_request_path(self.path)
            if path is None:
                self._send_bad_target()
                return

            measured = self.command == "POST" and path in INSTRUMENTED_PATHS
            model, is_stream = None, False
            if measured:
                raw, model, is_stream = prepare_body(raw)

            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP}
            req = urllib.request.Request(
                self.server.upstream.rstrip("/") + path,
                data=raw if raw else None, headers=headers, method=self.command)

            started = time.time()
            if measured:
                self._emit(OaiRequestStart(model, started))
            try:
                # B310 is silenced because the scheme is checked, not ignored:
                # _ProxyServer refuses to start on anything but http/https, so
                # this URL cannot be a file: or a custom scheme.
                upstream = urllib.request.urlopen(  # nosec B310
                    req, timeout=PROXY_TIMEOUT)
            except urllib.error.HTTPError as err:
                # A real HTTP error is a real answer: pass it through verbatim so
                # the client sees the server's own message, not llmwatch's.
                self._send_error_response(err, measured)
                return
            except (urllib.error.URLError, OSError, ValueError) as err:
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                self._send_gateway_error(err)
                return

            with upstream:
                self._pump(upstream, measured, is_stream, started)

        def _send_bad_target(self):
            """Refused before any connection is made, so a rejected target
            cannot be distinguished from a rejected one by timing either."""
            body = json.dumps({"error": {
                "message": "llmwatch: refusing to forward this request target",
                "type": "invalid_request_error"}}).encode("utf-8")
            try:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, BrokenPipeError):
                pass

        def _send_error_response(self, err, measured):
            try:
                body = err.read()
            except Exception:
                body = b""
            if measured:
                self._emit(OaiRequestAborted(time.time()))
            try:
                self.send_response(err.code)
                for key, value in (err.headers or {}).items():
                    if key.lower() not in HOP_BY_HOP:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, BrokenPipeError):
                pass

        def _send_gateway_error(self, err):
            """The upstream is unreachable. The client gets a clean, immediate
            error in the shape it already parses, rather than a hang."""
            body = json.dumps({"error": {
                "message": "llmwatch: upstream %s unreachable (%s)" % (
                    self.server.upstream, safe_text(str(err), limit=120)),
                "type": "upstream_unavailable"}}).encode("utf-8")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, BrokenPipeError):
                pass

        def _pump(self, upstream, measured, is_stream, started):
            """Relay the response, measuring it on the way past if asked."""
            status = getattr(upstream, "status", None) or upstream.getcode()
            out_headers = [(k, v) for k, v in upstream.headers.items()
                           if k.lower() not in HOP_BY_HOP]
            length = upstream.headers.get("Content-Length")

            # A response with a known length is not a stream; buffering it adds no
            # latency because the client cannot act on a partial body anyway.
            if length is not None or not is_stream:
                self._pump_whole(upstream, status, out_headers, length,
                                 measured, started)
                return

            try:
                self.send_response(status)
                for key, value in out_headers:
                    self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
            except (OSError, BrokenPipeError):
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                return

            buffer = b""
            decoded, first_token, usage, timings = 0, None, None, None
            # A rate needs an interval, and an interval needs two arrivals. A
            # short answer often lands in a single TCP read, where every token
            # appears to arrive at the same instant -- which divides out to tens
            # of thousands of tokens per second and, once banked, permanently
            # poisons `peak` and the token-weighted average.
            last_token, arrivals = None, 0
            last_tick = 0.0
            try:
                while True:
                    chunk = upstream.read(4096)
                    if not chunk:
                        break
                    # Relayed before it is inspected: measurement must never be in
                    # front of the bytes the client is waiting for.
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                    if not measured:
                        continue
                    produced_here = False
                    try:
                        buffer, payloads = _sse_events(buffer, chunk)
                        for payload in payloads:
                            produced, found, found_timings = read_stream_usage(payload)
                            if produced:
                                decoded += 1
                                produced_here = True
                            if found:
                                usage = found
                            # llama-server fills timings on the final chunk, the
                            # same one that carries usage. Kept separately so a
                            # server that sends one without the other still
                            # yields whichever it did send.
                            if found_timings:
                                timings = found_timings
                    except Exception:
                        # Format drift costs the numbers for this request, not the
                        # request itself.
                        buffer = b""
                    now = time.time()
                    if produced_here:
                        arrivals += 1
                        if first_token is None:
                            first_token = now
                        last_token = now
                    if measured and decoded and now - last_tick >= 0.25:
                        last_tick = now
                        self._emit(OaiGenTick(decoded, now))
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (OSError, BrokenPipeError):
                # The client hung up or the upstream died mid-stream. Either way the
                # token count is truncated, so nothing is banked.
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                return

            if measured:
                # One arrival is not a measurement of anything, so the window is
                # withheld and the request is recorded without a generation rate.
                self._finish_measure(usage, timings, first_token,
                                     last_token if arrivals >= 2 else None, started)

        def _pump_whole(self, upstream, status, out_headers, length, measured, started):
            try:
                declared = int(length) if length is not None else -1
            except (TypeError, ValueError):
                declared = -1
            if declared > MAX_BUFFERED_BODY:
                measured = False

            try:
                body = upstream.read()
            except OSError:
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                return
            try:
                self.send_response(status)
                for key, value in out_headers:
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
            except (OSError, BrokenPipeError):
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                return

            if not measured:
                return
            usage, timings = None, None
            try:
                obj = json.loads(body)
                if isinstance(obj, dict):
                    if isinstance(obj.get("usage"), dict):
                        usage = obj["usage"]
                    if isinstance(obj.get("timings"), dict):
                        timings = obj["timings"]
            except (ValueError, TypeError):
                usage, timings = None, None
            # No first-token mark exists for a non-streamed response, so the split
            # between prefill and generation is genuinely unknown. The request is
            # still recorded with its duration and token counts; the adapter simply
            # publishes no rate for it, which beats splitting it on a guess.
            self._finish_measure(usage, timings, None, None, started)

        def _finish_measure(self, usage, timings, first_token, last_token, started):
            prompt_tokens, cached, completion = usage_counts(usage)
            if usage is None:
                # Without a usage block there is no honest token count. Streamed
                # deltas were counted, but a delta is not reliably one token, so
                # they animate the live pane and are not banked as a measurement.
                self._emit(OaiRequestAborted(time.time()))
                return
            # Acceptance is a ratio, not a rate, so it needs no timing window and
            # survives even the requests whose generation window was withheld.
            counts = draft_counts(timings)
            draft = None
            if counts is not None:
                accepted, generated = counts
                # mean accepted-run length is not in the JSON (it needs the
                # verification-step count, which only the log reports), so the
                # slot is left empty rather than filled with a derived guess.
                draft = (accepted / float(generated), accepted, generated, None)
            self._emit(OaiRequestEnd(prompt_tokens, cached, completion,
                                     started, first_token, last_token, time.time(),
                                     draft))


    class _ProxyServer(http.server.ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

        def __init__(self, address, upstream, emit):
            # Checked here, the one place every request must pass through, so
            # no future caller can introduce a file: upstream by another route.
            if not valid_upstream(upstream):
                raise ValueError(
                    "upstream must be http:// or https://, got %r" % (upstream,))
            http.server.ThreadingHTTPServer.__init__(self, address, _ProxyHandler)
            self.upstream = upstream
            self.emit = emit

        def handle_error(self, request, client_address):
            """A broken client connection is routine here, not something to dump a
            traceback about onto the screen the TUI owns."""

    _PROXY_SERVER_CLASS = _ProxyServer
    return _PROXY_SERVER_CLASS


def safe_request_path(path):
    """The request target to forward, or None if it must be refused.

    The upstream URL is built by joining this onto the configured base, so a
    target that can reach past the first `/` chooses the host instead of the
    path. `@evil.com/` is the one that matters: joined on, it turns the
    configured `http://127.0.0.1:8080` into userinfo and sends the request,
    headers and all, to evil.com. `//evil.com/x` is the protocol-relative
    spelling of the same trick.

    Refused rather than rewritten. There is no legitimate OpenAI endpoint that
    needs any of these spellings, so a client sending one is not a client this
    should be trying to satisfy.
    """
    if not path or not path.startswith("/"):
        return None
    if path.startswith("//") or path.startswith("/\\"):
        return None
    # A backslash is a path separator to enough HTTP stacks that it is not
    # worth reasoning about which one is on the other end.
    if "\\" in path:
        return None
    return path


def valid_upstream(url):
    """Is this something the proxy may forward to?

    Only http and https. A `file:` upstream would turn every request the client
    makes into a local file read, and the reply would be relayed back to it.
    """
    return bool(url) and str(url).startswith(UPSTREAM_SCHEMES)


def parse_listen(text, default_port=DEFAULT_PROXY_PORT):
    """'8081' / '127.0.0.1:8081' / ':8081' -> (host, port)."""
    text = (text or "").strip()
    if not text:
        return ("127.0.0.1", default_port)
    if ":" in text:
        host, _, port = text.rpartition(":")
        host = host or "127.0.0.1"
    else:
        # A bare number is a port; a bare word is a host.
        if text.isdigit():
            return ("127.0.0.1", int(text))
        return (text, default_port)
    try:
        return (host, int(port))
    except ValueError:
        return (host, default_port)


def start_proxy(listen, upstream, emit):
    """Serve until the process exits. Returns the bound (host, port).

    Raises OSError if the port is taken -- which must be fatal rather than
    silent: a proxy that failed to bind means the client is still talking
    straight to the server, and llmwatch would sit there showing an idle
    screen that looks like nothing is happening.
    """
    host, port = listen
    server = _proxy_server_class()((host, port), upstream, emit)
    thread = threading.Thread(target=server.serve_forever, name="llmwatch-proxy")
    thread.daemon = True
    thread.start()
    return server.server_address[:2]


# --------------------------------------------------------------------------
# The update check
#
# Everything here fails silent. A tool that watches something else must never
# be the thing that breaks, and least of all over its own version number.
# --------------------------------------------------------------------------

def version_tuple(text):
    """Comparable form of a plain X.Y.Z version, or None for anything else.

    Deliberately refuses suffixes (rc, dev, post). Guessing how a pre-release
    orders against a release is how an update prompt ends up telling someone to
    "upgrade" to something older than what they already have.
    """
    parts = (text or "").strip().split(".")
    if not 1 <= len(parts) <= 4 or not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def update_is_newer(current, latest):
    """Is `latest` a release worth telling someone about? Unknown means no."""
    here, there = version_tuple(current), version_tuple(latest)
    if here is None or there is None:
        return False
    return there > here


def update_cache_path():
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "ollama-llmwatch", "update-check.json")


def read_update_cache(now, path=None):
    """(latest, fresh) from the cache. `fresh` decides whether to skip the call.

    A failed check caches too, as an empty result: an unreachable network should
    cost one attempt a day, not one on every launch.
    """
    try:
        with open(path or update_cache_path()) as fh:
            data = json.load(fh)
        checked = float(data.get("checked") or 0)
        return data.get("latest"), (now - checked) < UPDATE_INTERVAL
    except (OSError, ValueError, AttributeError):
        return None, False


def write_update_cache(latest, now, path=None):
    target = path or update_cache_path()
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as fh:
            json.dump({"latest": latest, "checked": now}, fh)
    except (OSError, ValueError):
        pass          # a cache we cannot write just means we check again tomorrow


def fetch_latest_version(url=UPDATE_URL, timeout=UPDATE_TIMEOUT):
    """The newest version on PyPI, or None if anything at all goes wrong.

    The single network call in this program. It is a GET of a public JSON
    document, identical for every user, with no body and nothing appended to the
    URL: PyPI learns that somebody asked, which it already knew from the install.
    """
    import urllib.request          # local: nothing imports a network client
                                   # unless this function is actually called
    # urlopen will happily follow file:, ftp: or a custom scheme, which turns a
    # url that ever becomes caller-controlled into a file read. It is a constant
    # today; this keeps that true no matter who calls it later.
    if not (url or "").startswith("https://"):
        return None
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ollama-llmwatch/%s" % __version__})
        # B310 is silenced because the scheme is checked above, not ignored.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            if getattr(response, "status", 200) != 200:
                return None
            payload = json.loads(response.read(64 * 1024).decode("utf-8", "replace"))
        return (payload.get("info") or {}).get("version")
    except Exception:              # noqa: BLE001 - DNS, TLS, timeout, junk JSON
        return None                # are all the same answer here: say nothing


def update_check_disabled():
    """Off by env var, and off for --json, which must stay machine-readable and
    is the mode most likely to be running unattended in a script."""
    value = (os.environ.get("LLMWATCH_NO_UPDATE_CHECK") or "").strip().lower()
    return value not in ("", "0", "false", "no")


def check_for_update(current=None, now=None, path=None, fetch=None):
    """The whole check, start to finish. Returns a version string or None.

    Pure enough to test: the clock, the cache location and the fetch are all
    injectable, so no test needs a network or a real home directory.
    """
    current = current or __version__
    now = time.time() if now is None else now
    latest, fresh = read_update_cache(now, path)
    if not fresh:
        latest = (fetch or fetch_latest_version)()
        write_update_cache(latest, now, path)
    return latest if update_is_newer(current, latest) else None


def start_update_check(box, current=None, path=None, fetch=None):
    """Run the check on a daemon thread and leave the answer in `box`.

    On a thread because a stalled DNS lookup must not delay the first frame by
    even the timeout: the board is on screen before this has finished, and the
    notice appears whenever it arrives.
    """
    if update_check_disabled():
        return None

    def run():
        try:
            found = check_for_update(current=current, path=path, fetch=fetch)
            if found:
                box["latest"] = found
        except Exception:          # noqa: BLE001 - never take the tool down
            pass

    thread = threading.Thread(target=run)
    thread.daemon = True           # never hold up exit waiting on the network
    thread.start()
    return thread


def upgrade_command(path=None):
    """The command that upgrades *this* copy, worked out from where it lives.

    There are four supported ways to install this and they upgrade differently.
    Telling a pipx user to run uv, or someone who curled a single file to run a
    package manager they never used, is worse than saying nothing: the advice
    fails in front of them and they learn to ignore the line.
    """
    path = os.path.abspath(path or __file__).replace("\\", "/")
    lowered = path.lower()
    if "/uv/tools/" in lowered:
        return "uv tool upgrade ollama-llmwatch"
    if "/pipx/" in lowered:
        return "pipx upgrade ollama-llmwatch"
    if "site-packages" in lowered or "dist-packages" in lowered:
        return "pip install --upgrade ollama-llmwatch"
    if os.path.isdir(os.path.join(os.path.dirname(path), ".git")):
        return "git pull"      # a checkout, which is how contributors run it
    # A single file someone downloaded: there is no package to upgrade, so the
    # only honest instruction is to fetch it again.
    return ("curl -O https://raw.githubusercontent.com/"
            "bingcheng45/ollama-llmwatch/main/llmwatch.py")


class _UpgradeRequested(Exception):
    """Leave the loop and upgrade. Not an error: the frame cannot keep drawing
    a program that is being replaced underneath it."""


def run_upgrade(style, plan=None):
    """Run the upgrade on the normal screen, then stop.

    Deliberately does not restart llmwatch afterwards. The file it would
    re-exec has just been overwritten, and a program that relaunches itself
    from something it just replaced is a good way to turn a failed download
    into a confusing crash. Telling someone to run one word is fine.
    """
    argv, blocker = plan if plan is not None else upgrade_plan()
    if blocker:
        sys.stderr.write("llmwatch: cannot upgrade automatically: %s\n"
                         "Run this instead:\n  %s\n" % (blocker, upgrade_command()))
        return 1
    sys.stdout.write("\n%s\n\n" % style.dim("$ " + " ".join(argv)))
    sys.stdout.flush()
    try:
        code = subprocess.call(argv)
    except (OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write("llmwatch: upgrade failed to start: %s\n" % exc)
        return 1
    if code != 0:
        sys.stderr.write("\nllmwatch: upgrade exited %d. Nothing was changed by "
                         "llmwatch itself.\n" % code)
        return code
    sys.stdout.write("\n%s\n" % style.bold("upgraded. start llmwatch again."))
    return 0


def upgrade_plan(path=None, which=None):
    """(argv, blocker) for upgrading this copy in place.

    argv is always an argument list, never a shell string, and never contains
    anything derived from the log, a model name or a version: the whole command
    comes from a fixed set chosen by where this file lives. Nothing an attacker
    can reach gets to influence what runs.

    A blocker means "do not run this, and say why". Refusing is the right answer
    more often than it looks: pulling over somebody's uncommitted work, or
    writing a file the user cannot write, fails in a way that is much harder to
    understand than a sentence explaining it up front.
    """
    path = os.path.abspath(path or __file__)
    directory = os.path.dirname(path)
    lowered = path.replace("\\", "/").lower()
    which = which or shutil.which

    def needs(tool, argv):
        if not which(tool):
            return None, "%s is not on PATH" % tool
        return argv, None

    if "/uv/tools/" in lowered:
        return needs("uv", ["uv", "tool", "upgrade", "ollama-llmwatch"])
    if "/pipx/" in lowered:
        return needs("pipx", ["pipx", "upgrade", "ollama-llmwatch"])
    if "site-packages" in lowered or "dist-packages" in lowered:
        # sys.executable, not a bare `pip`: the pip first on PATH may belong to
        # an entirely different interpreter, and "upgraded" into one you are not
        # running is the most confusing possible outcome.
        return [sys.executable, "-m", "pip", "install", "--upgrade",
                "ollama-llmwatch"], None
    if os.path.isdir(os.path.join(directory, ".git")):
        if not which("git"):
            return None, "git is not on PATH"
        # --ff-only, and only on a clean tree. This is a contributor's checkout;
        # merging into their work uninvited is not an upgrade, it is a mess.
        dirty = _git_is_dirty(directory)
        if dirty is None:
            return None, "cannot read the state of this checkout"
        if dirty:
            return None, "this checkout has uncommitted changes"
        return ["git", "-C", directory, "pull", "--ff-only"], None

    # A single downloaded file: replace it where it actually lives, rather than
    # dropping a second copy into whatever directory you happened to be in.
    if not which("curl"):
        return None, "curl is not on PATH"
    if not os.access(directory, os.W_OK):
        return None, "%s is not writable" % directory
    return ["curl", "-fsSL", "-o", path,
            "https://raw.githubusercontent.com/bingcheng45/"
            "ollama-llmwatch/main/llmwatch.py"], None


def _git_is_dirty(directory):
    """True, False, or None when git cannot tell us."""
    try:
        result = subprocess.run(
            ["git", "-C", directory, "status", "--porcelain"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def render_upgrade_confirm(latest, style, cols=80, plan=None):
    """The confirmation pane. Shows the exact command before it runs.

    An upgrade is one keystroke away, so the keystroke that starts it must not
    be the same one that opened this. Nobody should be able to change what is
    installed by leaning on the keyboard.
    """
    argv, blocker = plan if plan is not None else upgrade_plan()
    out = ["", style.bold("upgrade to %s" % latest) if latest else
           style.bold("upgrade"), ""]
    if blocker:
        out.append(style.yellow("cannot upgrade automatically: %s" % blocker))
        out.append("")
        out.append("run this yourself instead:")
        out.append(style.cyan("  " + upgrade_command()))
        out.append("")
        out.append(style.dim("esc  back"))
        return out
    out.append("this will run:")
    out.append(style.cyan("  " + " ".join(argv)))
    out.append("")
    out.append(style.dim("llmwatch quits afterwards, because the program it is "
                         "running from is what changes."))
    out.append("")
    out.append(style.dim("y  do it     esc  back"))
    return out


def render_update(latest, style):
    """One dim line. An update is worth mentioning, not worth interrupting for.

    Re-checks that the version really is newer, even though check_for_update
    already did. Handed a stale or lower version by a future caller, the honest
    failure is to say nothing; the alternative is telling someone running 0.9.0
    to "upgrade" to 0.7.0, which is worse than never mentioning it.
    """
    if not update_is_newer(__version__, latest):
        return None
    return style.dim("update available: %s (you have %s) - press u, or run %s"
                     % (latest, __version__, upgrade_command()))


class Screen:
    """Full-screen output on the alternate buffer, with guaranteed restore.

    A wedged terminal is worse than any missing feature, so the restore sequence
    is wired three ways: the caller's finally block, an atexit hook, and a
    SIGTERM handler (atexit does not run when the process is signalled).
    """

    ENTER = "\033[?1049h\033[?25l"
    LEAVE = "\033[?25h\033[?1049l"

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.active = False
        self.resized = threading.Event()
        self._prev_term = None

    def enter(self):
        if self.active:
            return
        self.stream.write(self.ENTER)
        self.stream.flush()
        self.active = True
        atexit.register(self.leave)
        # Restore at SIGNAL time, not at exit time. Waiting for the finally block
        # means the restore sequence is written microseconds before the process
        # dies, and a pty can discard buffered data when the slave closes -- so
        # ctrl-c could leave the terminal on the alternate screen.
        self._prev_term = signal.getsignal(signal.SIGTERM)
        self._prev_int = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, self._on_term)
        signal.signal(signal.SIGINT, self._on_int)
        if hasattr(signal, "SIGWINCH"):
            signal.signal(signal.SIGWINCH, lambda *a: self.resized.set())

    def _on_term(self, *_args):
        self.leave()
        raise SystemExit(0)          # unwinds finally blocks, unlike a bare kill

    def _on_int(self, *_args):
        self.leave()
        raise KeyboardInterrupt

    def leave(self):
        if not self.active:
            return
        self.active = False
        try:
            self.stream.write(self.LEAVE)
            self.stream.flush()
        except (ValueError, OSError):
            pass

    def size(self):
        self.resized.clear()
        return shutil.get_terminal_size((80, 24))

    def draw(self, lines, rows, cols=None):
        """One write per frame. Several writes at 10 fps flicker visibly.

        Clipped in BOTH dimensions. Height alone is not enough: one over-long
        line wraps, every row below it shifts down, and the next cursor-home
        repaint lands on the wrong rows -- the frame overlays itself and the
        screen fills with fragments of two frames at once.
        """
        clipped = lines[:max(1, rows - 1)]
        if cols:
            # cols - 1: writing exactly to the last column triggers auto-wrap on
            # some terminals, which is the very thing being avoided.
            clipped = [truncate_visible(line, max(1, cols - 1)) for line in clipped]
        body = "\033[K\n".join(clipped)
        self.stream.write("\033[H" + body + "\033[K\033[J")
        self.stream.flush()


FRAME_MARGIN = 2       # columns of breathing room between the frame and the edge


def compose_frame(snap, live_text, style, cols, rows, hint=True, help_visible=False,
                  live_detail=None, codex=None, system=None,
                  ui=None, picker=None, compare=None, update=None):
    """Build the whole TUI frame, inset from the terminal edge.

    One margin applied here rather than an indent baked into each renderer:
    every line then shares a single left edge, including content under a
    divider, and no pane can drift out of alignment with the others. The width
    the renderers are given shrinks to match, so the inset costs no content.
    """
    pad = " " * FRAME_MARGIN
    return [pad + line if line else line
            for line in frame_lines(snap, live_text, style,
                                    max(20, cols - FRAME_MARGIN), rows, hint,
                                    help_visible, live_detail, codex, system,
                                    ui, picker, compare, update)]


def frame_lines(snap, live_text, style, cols, rows, hint=True, help_visible=False,
                live_detail=None, codex=None, system=None,
                ui=None, picker=None, compare=None, update=None):
    """The frame itself, written as if it started at column 0.

    Panes drop in priority order when the terminal is short -- sparklines first,
    then the recent list -- but the live line is never dropped, because that is
    the thing the tool exists to show.
    """
    rule = "─" if style.unicode else "-"

    def divider(label):
        text = "%s %s " % (rule * 2, label)
        return style.dim(text + rule * max(0, min(cols, 100) - len(text) - 1))

    budget = max(1, rows - 1)

    def with_live(body, label):
        """Every modal keeps the live line visible: you should not lose sight of
        a running request because you opened a menu."""
        tail = ["", divider(label), live_text or style.dim("idle")]
        for detail in (live_detail or []):
            tail.append(detail)
        return body[:max(1, budget - len(tail))] + tail

    if ui is not None and ui.view == "picker":
        return with_live([divider("compare: pick two models"), ""]
                         + render_picker(picker or [], ui, style, cols), "live")

    if ui is not None and ui.view == "compare":
        head = divider("%s  vs  %s" % (ui.model_a or "?", ui.model_b or "?"))
        return with_live([head, ""] + (compare or []), "live")

    if ui is not None and ui.view == "upgrade":
        return with_live([divider("upgrade")]
                         + render_upgrade_confirm(update, style, cols,
                                                  plan=ui.upgrade_plan), "live")

    if help_visible:
        # Help replaces the board but keeps the live line: you should never lose
        # sight of the running request just because you asked what a column means.
        help_lines = render_help(style, cols, rows - 4)
        tail = ["", divider("live"), live_text or style.dim("idle")]
        for detail in (live_detail or []):
            tail.append(detail)
        return help_lines + tail

    # Reserved first and never trimmed: everything else is context, but this is
    # the answer to "is it still reading my prompt, or is it writing?".
    live_block = ["", divider("live"), live_text or style.dim("idle")]
    for detail in (live_detail or []):
        live_block.append(detail)
    if len(live_block) >= budget:
        return live_block[-budget:]

    title = "llmwatch %s   %s" % (__version__, style.bold(snap.get("model", "?")))
    meta = "%d req - %s" % (snap.get("requests", 0),
                            fmt_duration(snap.get("session_seconds", 0)))
    if snap.get("models_seen", 0) > 1:
        meta += " - %d models seen" % snap["models_seen"]

    compact = rows < 22
    recent = snap.get("recent") or []
    recent_block = []
    if recent:
        recent_block = ["", divider("recent")] + render_recent(recent, style, cols, limit=6)

    # Highest priority first; a section is included only if it fits whole,
    # so the layout degrades in steps instead of tearing mid-pane.
    codex_block = []
    if codex:
        rows_ = render_codex(codex, style, cols)
        if rows_:
            codex_block = ["", divider("codex")] + rows_

    head = []
    for block in ([title + "   " + style.dim(meta)],
                  render_board(snap, style, cols, compact=compact, system=system),
                  codex_block,
                  recent_block,
                  # Last of the optional blocks, so a short terminal drops the
                  # version notice before it drops anything you are watching.
                  ["", render_update(update, style)] if update else [],
                  ["", style.dim("h help   -   c compare   -   ctrl-c quit")] if hint else []):
        if block and len(head) + len(block) + len(live_block) <= budget:
            head.extend(block)
    return head + live_block


def plan_frame(now, last_paint, next_frame, got_data, active):
    """Decide whether to repaint now, and when the next frame is due.

    Returns (should_paint, next_frame). Pure, so the pacing guarantees can be
    tested without threads or a terminal:

    - new data repaints immediately rather than waiting for the next tick
    - with no data at all, frames still arrive at the floor rate (the v0.2.0
      anti-freeze guarantee)
    - never faster than MIN_FRAME_GAP, so a log flood cannot spin the CPU
    """
    due = got_data or now >= next_frame
    if not due:
        return False, next_frame
    if (now - last_paint) < MIN_FRAME_GAP:
        return False, last_paint + MIN_FRAME_GAP      # defer briefly, don't drop
    return True, now + (FRAME_ACTIVE if active else FRAME_IDLE)


MIN_COMPARE_SAMPLES = 5      # below this, report counts rather than a ratio

# Turns are far rarer than requests -- a busy afternoon is a few dozen, not a few
# thousand -- so the floor for claiming a turn-time ratio is lower. It is not
# zero: two turns a side is an anecdote.
MIN_COMPARE_TURNS = 3

# Prompt-size bands. Comparing a cached 244-token request against an uncached
# 47,000-token one produces a number that means nothing; bucketing keeps the
# comparison honest.
SIZE_BANDS = ((0, 1000, "tiny"), (1000, 10000, "small"),
              (10000, 50000, "large"), (50000, None, "huge"))


def size_band(tokens):
    for low, high, name in SIZE_BANDS:
        if tokens is None:
            return "unknown"
        if tokens >= low and (high is None or tokens < high):
            return name
    return "unknown"


class History:
    """A local record of every completed request.

    Without this, everything llmwatch measures dies at exit -- which is why the
    same "is this model build actually faster?" question ends up being answered
    with a hand-written benchmark script every time.

    Two tables: one request as llama-server sees it, and one agent turn as you
    experience it (prompt submitted to final text, tool calls and all).

    Stores timings, model names and reasoning effort only. There is deliberately
    no column that could hold prompt content, matching the property the Ollama
    log itself has -- and which the Codex session file does NOT, which is why
    only durations are lifted out of it.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS requests (
        id              INTEGER PRIMARY KEY,
        ts              REAL NOT NULL,
        model           TEXT NOT NULL,
        prompt_tokens   INTEGER,
        cached_tokens   INTEGER,
        prefill_tokens  INTEGER,
        prefill_seconds REAL,
        prefill_rate    REAL,
        gen_tokens      INTEGER,
        gen_seconds     REAL,
        gen_rate        REAL,
        total_seconds   REAL,
        prefill_share   REAL,
        cache_miss      INTEGER,
        draft_rate      REAL
    );
    CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
    CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model, ts);

    CREATE TABLE IF NOT EXISTS turns (
        id            INTEGER PRIMARY KEY,
        ts            REAL NOT NULL,
        model         TEXT NOT NULL,
        effort        TEXT,
        seconds       REAL NOT NULL,
        ttft_seconds  REAL,
        tool_calls    INTEGER,
        completed     INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_turns_model ON turns(model, ts);
    """

    def __init__(self, path=None):
        self.path = path or self.default_path()
        self.conn = None
        self.enabled = True
        self._connect()

    @staticmethod
    def default_path():
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        return os.path.join(base, "ollama-llmwatch", "history.db")

    def _connect(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.conn = sqlite3.connect(self.path, timeout=1.0)
            self.conn.executescript(self.SCHEMA)
            self.conn.commit()
        except (sqlite3.Error, OSError):
            # A monitor that dies because its logbook is locked is worse than one
            # with no logbook. Carry on without history.
            self.conn = None
            self.enabled = False

    def record(self, model, prefill, generation, end, now=None):
        if not self.conn:
            return False
        prefill = prefill or {}
        generation = generation or {}
        try:
            self.conn.execute(
                "INSERT INTO requests (ts, model, prompt_tokens, cached_tokens, "
                "prefill_tokens, prefill_seconds, prefill_rate, gen_tokens, "
                "gen_seconds, gen_rate, total_seconds, prefill_share, cache_miss, "
                "draft_rate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (now if now is not None else time.time(), model or "?",
                 (prefill.get("tokens") or 0) + (prefill.get("cached") or 0),
                 prefill.get("cached") or 0,
                 prefill.get("tokens"), prefill.get("seconds"), prefill.get("rate"),
                 generation.get("tokens"), generation.get("seconds"),
                 generation.get("rate"), end.get("seconds"),
                 end.get("prefill_share_pct"), 1 if end.get("cache_miss") else 0,
                 (end.get("draft") or [None])[0]))
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False

    def record_turn(self, turn, now=None):
        """One finished agent turn: prompt submitted to final text.

        Kept in its own table rather than as a column on `requests`, because a
        turn contains many requests -- and, on an agent, a good deal of time that
        is not a model request at all.
        """
        if not self.conn or not turn or turn.get("seconds") is None:
            return False
        try:
            self.conn.execute(
                "INSERT INTO turns (ts, model, effort, seconds, ttft_seconds, "
                "tool_calls, completed) VALUES (?,?,?,?,?,?,?)",
                (turn.get("ended_at") or (now if now is not None else time.time()),
                 turn.get("model") or "?", turn.get("effort"),
                 float(turn["seconds"]), turn.get("ttft_seconds"),
                 turn.get("tool_calls"), 1 if turn.get("completed") else 0))
            self.conn.commit()
            return True
        except (sqlite3.Error, TypeError, ValueError):
            return False

    def turn_rows(self, days=None, model=None, now=None):
        if not self.conn:
            return []
        now = now if now is not None else time.time()
        query, args = "SELECT * FROM turns WHERE ts >= ?", [
            0 if days is None else now - days * 86400]
        if model:
            query += " AND model = ?"
            args.append(model)
        try:
            cur = self.conn.execute(query + " ORDER BY ts", args)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    def turn_profile(self, model, days=30, now=None):
        """How long a turn takes on this model, split by reasoning effort.

        Medians, not means: one turn where you walked away and left the agent
        blocked on an approval prompt would otherwise set the expectation for
        every turn after it. Interrupted turns are counted but never timed --
        their duration measures your patience, not the model.
        """
        if not self.conn:
            return None
        rows = self.turn_rows(days=days, model=model, now=now)
        done = [r for r in rows if r.get("completed")]
        by_effort = {}
        for row in done:
            by_effort.setdefault(row.get("effort") or "unknown", []).append(row)
        return {
            "model": model,
            "turns": len(done),
            "interrupted": len(rows) - len(done),
            "median_seconds": self._median([r.get("seconds") for r in done]),
            "median_ttft": self._median([r.get("ttft_seconds") for r in done]),
            "median_tool_calls": self._median([r.get("tool_calls") for r in done]),
            "total_seconds": sum(r.get("seconds") or 0 for r in done),
            "efforts": {
                name: {"turns": len(group),
                       "median_seconds": self._median([r.get("seconds") for r in group]),
                       "median_ttft": self._median([r.get("ttft_seconds") for r in group])}
                for name, group in sorted(by_effort.items())
            },
        }

    def turn_rollup(self, days=7, model=None, now=None):
        """One row per model and effort level, for `--turns`."""
        if not self.conn:
            return []
        rows = self.turn_rows(days=days, model=model, now=now)
        groups = {}
        for row in rows:
            groups.setdefault((row["model"], row.get("effort") or "unknown"), []).append(row)
        out = []
        for (name, effort), group in sorted(groups.items()):
            done = [r for r in group if r.get("completed")]
            out.append({
                "model": name, "effort": effort,
                "turns": len(done), "interrupted": len(group) - len(done),
                "median_seconds": self._median([r.get("seconds") for r in done]),
                "longest_seconds": max([r.get("seconds") or 0 for r in done] or [0]) or None,
                "median_tool_calls": self._median([r.get("tool_calls") for r in done]),
            })
        return out

    def _rows(self, since, until=None, model=None):
        query = "SELECT * FROM requests WHERE ts >= ?"
        args = [since]
        if until is not None:
            query += " AND ts < ?"
            args.append(until)
        if model:
            query += " AND model = ?"
            args.append(model)
        try:
            cur = self.conn.execute(query, args)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    @staticmethod
    def _weighted(rows, tokens_key, seconds_key):
        """Token-weighted rate, for the same reason the live stats use one: a
        mean of rates lets a 4-token request outvote a 47,000-token one."""
        tokens = sum(r.get(tokens_key) or 0 for r in rows)
        seconds = sum(r.get(seconds_key) or 0 for r in rows)
        return (tokens / seconds) if seconds > 0 else None

    def rollup(self, days=7, model=None, now=None):
        """Per-model summary for a window, with the change against the window
        immediately before it (so "slower than last week" is answerable)."""
        if not self.conn:
            return []
        now = now if now is not None else time.time()
        span = days * 86400
        # No upper bound on the current window: a request logged at exactly `now`
        # belongs to it, and `ts < now` quietly dropped it.
        current = self._rows(now - span, None, model)
        previous = self._rows(now - 2 * span, now - span, model)

        out = []
        for name in sorted({r["model"] for r in current}):
            mine = [r for r in current if r["model"] == name]
            before = [r for r in previous if r["model"] == name]
            gen = self._weighted(mine, "gen_tokens", "gen_seconds")
            gen_before = self._weighted(before, "gen_tokens", "gen_seconds")
            change = None
            if gen and gen_before:
                change = (gen - gen_before) / gen_before * 100
            cached = sum(r.get("cached_tokens") or 0 for r in mine)
            prompt = sum(r.get("prompt_tokens") or 0 for r in mine)
            out.append({
                "model": name, "requests": len(mine),
                "prefill_rate": self._weighted(mine, "prefill_tokens", "prefill_seconds"),
                "gen_rate": gen, "gen_change_pct": change,
                "previous_requests": len(before),
                "cache_pct": (cached / prompt * 100) if prompt else None,
                "cache_misses": sum(1 for r in mine if r.get("cache_miss")),
            })
        return out

    def compare(self, model_a, model_b, days=30, now=None):
        """Compare two models on LIKE-FOR-LIKE requests.

        Bucketed by prompt size and cache state, with the sample count reported
        for each. An unbucketed comparison is how a 1.8x result turns out to have
        been two different workloads.
        """
        if not self.conn:
            return []
        now = now if now is not None else time.time()
        # No upper bound, for the same reason as rollup: `ts < now` dropped the
        # most recent request, which is the one most likely to matter.
        rows = self._rows(now - days * 86400, None)
        buckets = {}
        for row in rows:
            if row["model"] not in (model_a, model_b):
                continue
            key = (size_band(row.get("prefill_tokens")), bool(row.get("cache_miss")))
            buckets.setdefault(key, {model_a: [], model_b: []})[row["model"]].append(row)

        out = []
        for (band, miss), sides in sorted(buckets.items()):
            a_rows, b_rows = sides[model_a], sides[model_b]
            a_rate = self._weighted(a_rows, "gen_tokens", "gen_seconds")
            b_rate = self._weighted(b_rows, "gen_tokens", "gen_seconds")
            enough = (len(a_rows) >= MIN_COMPARE_SAMPLES
                      and len(b_rows) >= MIN_COMPARE_SAMPLES)
            out.append({
                "band": band, "cache_miss": miss,
                "a_n": len(a_rows), "b_n": len(b_rows),
                "a_rate": a_rate, "b_rate": b_rate,
                # Only claim a ratio when both sides have enough samples to mean
                # something. Otherwise report the counts and say so.
                "ratio": (a_rate / b_rate) if (enough and a_rate and b_rate) else None,
                "enough": enough,
            })
        return out

    def models(self, now=None):
        """Every model we have ever recorded, most-used first."""
        if not self.conn:
            return []
        try:
            cur = self.conn.execute(
                "SELECT model, COUNT(*), MAX(ts), SUM(gen_tokens), SUM(gen_seconds) "
                "FROM requests GROUP BY model")
        except sqlite3.Error:
            return []
        now = now if now is not None else time.time()
        out = []
        for name, count, last, tokens, seconds in cur.fetchall():
            out.append({"model": name, "requests": count,
                        "last_seen": (now - last) if last else None,
                        "gen_rate": (tokens / seconds) if seconds else None})
        return sorted(out, key=lambda r: -r["requests"])

    @staticmethod
    def _median(values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return None
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0

    def profile(self, model, days=30, now=None):
        """Everything the compare view shows for one model."""
        if not self.conn:
            return None
        now = now if now is not None else time.time()
        rows = self._rows(now - days * 86400, None, model)
        if not rows:
            return {"model": model, "requests": 0}
        cached = sum(r.get("cached_tokens") or 0 for r in rows)
        prompt = sum(r.get("prompt_tokens") or 0 for r in rows)
        drafts = [r["draft_rate"] for r in rows if r.get("draft_rate") is not None]
        return {
            "model": model,
            "requests": len(rows),
            "last_seen": now - max(r["ts"] for r in rows),
            "gen_rate": self._weighted(rows, "gen_tokens", "gen_seconds"),
            "prefill_rate": self._weighted(rows, "prefill_tokens", "prefill_seconds"),
            "ttft": self._median([r.get("prefill_seconds") for r in rows]),
            "cache_pct": (cached / prompt * 100) if prompt else None,
            "draft_pct": (sum(drafts) / len(drafts) * 100) if drafts else None,
            "median_prompt_tokens": self._median([r.get("prefill_tokens") for r in rows]),
            "median_gen_tokens": self._median([r.get("gen_tokens") for r in rows]),
        }

    def all_rows(self, days=None, now=None):
        if not self.conn:
            return []
        now = now if now is not None else time.time()
        since = 0 if days is None else now - days * 86400
        return self._rows(since)

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
            self.conn = None


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

    def __init__(self, clock=time.monotonic):
        self._clock = clock
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
        self.models_gb = None
        self.server_ok = None          # None = not checked yet
        self.server_problem = None

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
            self._busiest = busiest_process()
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


class CodexTail:
    """Reads Codex's own session rollout file to show what the agent is doing.

    Opt-in, because unlike the Ollama log this file contains real content --
    commands, file paths, message text. Correlation with a given model request is
    by time only: no shared request id exists between Codex and Ollama.
    """

    SESSIONS = "~/.codex/sessions"

    # Codex records no success/failure flag on tool results -- only free text --
    # so failure detection is heuristic and deliberately conservative. A grep hit
    # containing the word "error" must NOT be reported as a failed tool.
    FAILURE_PREFIXES = ("failed to", "error:", "traceback (most recent call last)")
    FAILURE_PHRASES = ("command not found", "no such file or directory",
                       "permission denied", "is not recognized as an internal")
    RE_EXIT_CODE = re.compile(r"process exited with code (\d+)", re.I)

    # A turn longer than this is a misread field or a clock that moved, not a
    # real wait. Reporting "37h" as a turn time is worse than reporting nothing.
    MAX_TURN_SECONDS = 12 * 3600

    def __init__(self, path=None, sessions_dir=None):
        # An explicit path pins one session; otherwise follow whichever is newest.
        self.fixed_path = path
        self.sessions_dir = sessions_dir or self.SESSIONS
        self.path = path
        self.offset = 0
        self.inode = None
        self._attached = False
        self.action = None
        self.detail = None
        self.calls = 0
        self.last_call_at = None
        self.error = None
        self.error_repeats = 0
        # ---- turn tracking ------------------------------------------------
        # A turn is one user prompt: submitted, then thinking, tool calls and
        # output, until the agent stops. That whole span is the number a user
        # actually feels, and no part of it is visible in the Ollama log, which
        # only ever sees the individual model requests inside it.
        self.model = None            # thread-level model, last seen
        self.effort = None           # thread-level reasoning effort, last seen
        self.turn_id = None
        self.turn_started = None     # our own clock, only used as a fallback
        self.turn_model = None
        self.turn_effort = None
        self.last_turn = None        # the most recently finished turn
        self.finished = []           # finished turns not yet handed over

    def newest_session(self):
        root = os.path.expanduser(self.sessions_dir)
        newest, newest_mtime = None, 0
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not (name.startswith("rollout-") and name.endswith(".jsonl")):
                    continue
                full = os.path.join(dirpath, name)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    continue
                if mtime > newest_mtime:
                    newest, newest_mtime = full, mtime
        return newest

    def poll(self):
        """Read whatever is new. Cheap enough to call every frame."""
        newest = self.fixed_path or self.newest_session()
        # `not self._attached` matters as much as the path check: with a path
        # supplied up front, `newest != self.path` is false on the very first
        # poll, so a change-only test would still replay the file.
        if newest and (newest != self.path or not self._attached):
            # Start at the END, like `tail -n 0`. Starting at 0 meant attaching
            # to a long-running session replayed the whole thing: the largest
            # session file on the development machine was 5.97 MB, every
            # historical tool call counted, and an hours-old action displayed as
            # if it were current.
            self.path, self.calls, self._attached = newest, 0, True
            try:
                stat = os.stat(newest)
                self.offset, self.inode = stat.st_size, stat.st_ino
            except OSError:
                self.offset, self.inode = 0, None
        if not self.path:
            return
        try:
            # Two ways the file can restart under us: replaced (new inode, e.g.
            # rotated into place) or truncated (smaller than where we are). Both
            # would otherwise leave the offset past EOF and freeze the pane on
            # stale state. A same-size in-place rewrite is undetectable by either
            # signal -- Codex only appends, so that case does not arise in practice.
            stat = os.stat(self.path)
            if stat.st_ino != self.inode or stat.st_size < self.offset:
                self.offset = 0
            self.inode = stat.st_ino
            with open(self.path, "r", errors="replace") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return

        # The file is being appended to live, so the last line may be half
        # written. Consume only complete lines and leave the remainder for the
        # next poll -- otherwise a partial record is parsed as garbage, skipped,
        # and its real content never seen.
        parts = chunk.split("\n")
        remainder = parts.pop()
        for line in parts:
            self._consume(line)
        self.offset += len(chunk.encode("utf-8")) - len(remainder.encode("utf-8"))

    @classmethod
    def looks_like_failure(cls, output):
        """Conservative: only obvious failures. A false 'tool failed' is worse
        than a missed one, because it sends you debugging the wrong thing."""
        if not output:
            return False
        low = " ".join(str(output).split()).lower()
        if low.startswith(cls.FAILURE_PREFIXES):
            return True
        if any(phrase in low for phrase in cls.FAILURE_PHRASES):
            return True
        match = cls.RE_EXIT_CODE.search(low)
        return bool(match and match.group(1) != "0")

    def _consume(self, line):
        try:
            record = json.loads(line)
        except ValueError:
            return
        payload = record.get("payload") or {}
        kind = payload.get("type") or record.get("type")
        if kind == "function_call":
            self.calls += 1
            self.action = safe_text(payload.get("name"), limit=60) or "tool call"
            self.detail = safe_text(self._summarise(payload.get("arguments")))
            self.last_call_at = time.time()
        elif kind == "function_call_output":
            output = payload.get("output")
            if self.looks_like_failure(output):
                short = safe_text(" ".join(str(output).split()), limit=90)
                # The same failure repeating is the interesting case: the agent
                # is retrying something that cannot work, and will keep burning
                # full prompt reads until someone stops it.
                self.error_repeats = self.error_repeats + 1 if short == self.error else 1
                self.error = short
            else:
                self.error = None
                self.error_repeats = 0
        elif kind == "task_started":
            self.calls = 0
            self.action = "thinking"
            self.detail = None
            self.last_call_at = time.time()
            self._start_turn(payload)
        elif kind == "turn_context":
            # Per-turn settings, and the authoritative effort for this turn: a
            # thread default can be changed between turns.
            settings = (payload.get("collaboration_mode") or {}).get("settings") or {}
            self.turn_model = short_model_name(payload.get("model")) or self.turn_model
            effort = self._effort(settings.get("reasoning_effort"))
            if effort:
                self.turn_effort = effort
        elif kind == "thread_settings_applied":
            settings = payload.get("thread_settings") or {}
            self.model = short_model_name(settings.get("model")) or self.model
            self.effort = self._effort(settings.get("reasoning_effort")) or self.effort
        elif kind == "task_complete":
            self.action = "done"
            self.detail = None
            self.last_call_at = None
            self._finish_turn(payload, completed=True)
        elif kind == "turn_aborted":
            self.action = "interrupted"
            self.detail = None
            self.last_call_at = None
            self._finish_turn(payload, completed=False)

    # ---- turns ------------------------------------------------------------

    EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

    @classmethod
    def _effort(cls, value):
        """Only record an effort level we recognise. An unknown string would
        become its own bucket in the history and split the samples for no gain."""
        if value is None:
            return None
        name = safe_text(str(value), limit=16).strip().lower()
        return name if name in cls.EFFORTS else None

    @staticmethod
    def _number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def _start_turn(self, payload):
        self.turn_id = safe_text(payload.get("turn_id"), limit=64)
        started = self._number(payload.get("started_at"))
        self.turn_started = started if started else time.time()
        self.turn_model = self.model
        # Inherit the thread default until this turn's own context arrives.
        self.turn_effort = self.effort

    def _turn_seconds(self, payload):
        """Codex's own clock first: it knows when the prompt was submitted, and
        llmwatch attaches at the end of the file, so it may never have seen the
        start of a turn that is already running."""
        seconds = self._number(payload.get("duration_ms"))
        seconds = seconds / 1000.0 if seconds is not None else None
        if seconds is None:
            started = self._number(payload.get("started_at"))
            ended = self._number(payload.get("completed_at"))
            if started and ended:
                seconds = ended - started
        if seconds is None and self.turn_started:
            seconds = time.time() - self.turn_started
        if seconds is None or seconds < 0 or seconds > self.MAX_TURN_SECONDS:
            return None
        return seconds

    def _finish_turn(self, payload, completed):
        """One finished turn, as a row.

        Content is dropped here and never travels further: `task_complete`
        carries `last_agent_message`, the agent's final text, which is exactly
        the kind of thing the rest of this tool refuses to keep.
        """
        seconds = self._turn_seconds(payload)
        ttft = self._number(payload.get("time_to_first_token_ms"))
        ended = self._number(payload.get("completed_at"))
        turn = {
            "turn_id": safe_text(payload.get("turn_id"), limit=64) or self.turn_id,
            "model": self.turn_model or self.model,
            "effort": self.turn_effort or self.effort,
            "seconds": seconds,
            "ttft_seconds": (ttft / 1000.0) if ttft is not None and ttft >= 0 else None,
            "tool_calls": self.calls,
            "completed": bool(completed),
            "reason": None if completed else (safe_text(payload.get("reason"), limit=40)
                                              or "interrupted"),
            "ended_at": ended if ended else time.time(),
        }
        self.turn_id = None
        self.turn_started = None
        if seconds is None:
            return          # nothing trustworthy to show or store
        self.last_turn = turn
        self.finished.append(turn)

    def drain(self):
        """Hand over finished turns exactly once, so a caller can record them
        without having to track what it has already seen."""
        out, self.finished = self.finished, []
        return out

    @staticmethod
    def _summarise(arguments):
        """One short line out of a JSON argument blob."""
        if not arguments:
            return None
        try:
            args = json.loads(arguments)
        except (ValueError, TypeError):
            return str(arguments)[:80]
        for key in ("cmd", "command", "path", "file_path", "query", "pattern"):
            if key in args:
                value = args[key]
                if isinstance(value, list):
                    value = " ".join(str(v) for v in value)
                return " ".join(str(value).split())[:100]
        return " ".join(json.dumps(args).split())[:100]

    def state(self):
        if not self.path:
            return None
        waiting = (time.time() - self.last_call_at) if self.last_call_at else None
        elapsed = (time.time() - self.turn_started) if self.turn_started else None
        return {"action": self.action, "detail": self.detail,
                "calls": self.calls, "waiting_since": waiting,
                "error": self.error, "error_repeats": self.error_repeats,
                "turn_seconds": elapsed,
                "effort": self.turn_effort or self.effort,
                "model": self.turn_model or self.model,
                "last_turn": self.last_turn}


class Keyboard:
    """Single-key input without Enter.

    Restored the same three ways as the screen: a keyboard left in cbreak mode is
    almost as annoying as a terminal left on the alternate buffer.
    """

    def __init__(self):
        self.fd = None
        self.saved = None

    def enter(self):
        if termios is None or not sys.stdin.isatty():
            return False
        try:
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except (termios.error, ValueError, OSError):
            self.saved = None
            return False
        atexit.register(self.leave)
        return True

    def leave(self):
        if self.saved is None:
            return
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        except (termios.error, ValueError, OSError):
            pass
        self.saved = None


def _reader(proc, q, stop):
    """Log lines arrive on their own schedule; the UI must not wait for them."""
    try:
        for line in proc.stdout:
            if stop.is_set():
                break
            q.put(("log", line))
    except (ValueError, OSError):
        pass
    finally:
        q.put(("log", None))


ARROW_KEYS = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
# How long to wait for the rest of an escape sequence before deciding a bare Esc
# was pressed. 50ms proved too tight in practice: the reader thread competes with
# the render loop for the GIL, so the follow-up bytes of an arrow key could miss
# the window and arrive as three separate keys - pressing Down closed the menu.
# 150ms is still imperceptible when actually pressing Esc.
ESC_TIMEOUT = 0.15


def decode_keys(buffer, flush=False):
    """Turn raw input bytes into named keys. Returns (keys, leftover).

    An arrow is three bytes (\x1b[A). Two things make this fiddly:

    - A lone Esc is a prefix of every arrow, so a partial sequence must be held
      back until either the rest arrives or the read times out (`flush`).
    - Reads must come from the raw fd. sys.stdin.read(1) pulls everything
      available into Python's buffer and returns one character, after which
      select() on the descriptor reports "no data" -- so a real arrow looks
      exactly like a lone Esc, and pressing Down closes the menu.
    """
    keys, index = [], 0
    while index < len(buffer):
        char = buffer[index]
        if char != "\x1b":
            keys.append(char)
            index += 1
            continue
        rest = buffer[index + 1:]
        if rest.startswith("["):
            if len(rest) >= 2:
                keys.append(ARROW_KEYS.get(rest[1], "ESC"))
                index += 3
                continue
            if not flush:
                break                    # final byte still in flight
            keys.append("ESC")
            index = len(buffer)
            continue
        if not rest and not flush:
            break                        # might yet become an arrow
        keys.append("ESC")
        index += 1
    return keys, buffer[index:]


def _key_reader(q, stop):
    """Keys go through the same queue as log lines, so a keypress wakes the main
    loop immediately instead of waiting for the next frame."""
    try:
        fd = sys.stdin.fileno()
    except (ValueError, OSError):
        return
    pending = ""
    while not stop.is_set():
        try:
            ready, _, _ = select.select([fd], [], [], ESC_TIMEOUT if pending else 0.25)
            if ready:
                chunk = os.read(fd, 64).decode("utf-8", "replace")
                if not chunk:
                    return
                pending += chunk
                keys, pending = decode_keys(pending)
            elif pending:
                keys, pending = decode_keys(pending, flush=True)
            else:
                continue
            for key in keys:
                q.put(("key", key))
        except (ValueError, OSError):
            return


def follow(args):
    # `--proxy` with no value yields "", which is still a request to proxy.
    proxying = args.proxy is not None or bool(os.environ.get("LLMWATCH_PROXY"))
    kind, target = find_log()
    if not kind and not proxying:
        sys.stderr.write(
            "llmwatch: could not find an Ollama log. Tried:\n  " +
            "\n  ".join(candidate_paths()) +
            "\n\nPass one explicitly with --log PATH or set LLMWATCH_LOG.\n"
            "Watching an OpenAI-compatible server (mlx_lm.server, LM Studio) instead?\n"
            "Use --proxy, which reads the numbers off the wire rather than from a log.\n"
            "Note: llmwatch needs LOCAL log access; a remote Ollama server won't work.\n")
        return 2

    tracker = Tracker()
    # `ollama ps` is the wrong question when the server is not Ollama; in proxy
    # mode the model names itself on the first request instead.
    if not proxying:
        tracker.model = current_model()
    style = Style.detect(no_color=args.no_color)
    interactive = sys.stdout.isatty() and not args.json
    # Below ~10 rows the panes cannot be laid out usefully, so fall back to the
    # single-line renderer rather than draw something corrupt.
    tui = (interactive and not args.plain
           and shutil.get_terminal_size((80, 24)).lines >= 10)

    q = queue.Queue()
    stop = threading.Event()

    proxy_at, upstream = None, None
    if proxying:
        listen = parse_listen(args.proxy or os.environ.get("LLMWATCH_PROXY"))
        upstream = (args.upstream or os.environ.get("LLMWATCH_UPSTREAM")
                    or DEFAULT_UPSTREAM)
        if not valid_upstream(upstream):
            sys.stderr.write(
                "llmwatch: --upstream must be http:// or https://, got %r\n"
                % (upstream,))
            return 2
        if listen[0] not in LOOPBACK_HOSTS and not args.proxy_allow_remote:
            sys.stderr.write(
                "llmwatch: refusing to listen on %s.\n"
                "A proxy on a non-loopback address relays whatever it is sent, "
                "and this one authenticates nobody.\n"
                "Pass --proxy-allow-remote if that is genuinely what you want.\n"
                % listen[0])
            return 2
        try:
            proxy_at = start_proxy(listen, upstream, lambda ev: q.put(("event", ev)))
        except OSError as err:
            # Fatal on purpose: if the bind failed the client is still talking
            # straight to the server, and llmwatch would show an idle screen
            # that reads as "nothing is happening" rather than "not connected".
            sys.stderr.write(
                "llmwatch: cannot listen on %s:%s (%s)\n" % (listen[0], listen[1], err))
            return 2

    if not args.json:
        if proxy_at:
            sys.stderr.write("llmwatch %s  proxying http://%s:%d -> %s\n"
                             % (__version__, proxy_at[0], proxy_at[1], upstream))
            if kind:
                where = target if kind == "file" else "journalctl -u %s" % target
                sys.stderr.write("             prefill progress from %s\n" % where)
        else:
            where = target if kind == "file" else "journalctl -u %s" % target
            sys.stderr.write("llmwatch %s  watching %s\n" % (__version__, where))

    # Started before the log is even open, and never waited on. --json is
    # excluded in update_check_disabled()'s caller below: that output is parsed
    # by other programs and usually runs unattended.
    update_box = {}
    upgrading = False
    if not args.json:
        start_update_check(update_box)

    # In proxy mode the log is an optional extra: it supplies the prefill
    # progress bar and nothing else, so its absence costs an animation rather
    # than a measurement.
    proc = None
    if kind:
        proc = subprocess.Popen(tail_command(kind, target), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        thread = threading.Thread(target=_reader, args=(proc, q, stop))
        thread.daemon = True
        thread.start()

    stats = Stats()
    screen = Screen() if tui else None
    keyboard = Keyboard() if tui else None
    # Not gated on `tui`: turn timings are worth recording in --plain and --json
    # runs too, and they are the one thing here that outlives the session.
    codex_tail = CodexTail() if args.codex else None
    probe = SystemProbe()          # cheap and rate-limited; useful in both modes
    history = None if args.no_history else History()
    progress_floor = ProgressFloor()   # displayed progress must never rewind
    ui = UIState()
    picker_models = []
    compare_lines = []
    live_data = None
    live_at = time.time()
    idle_since = time.time()
    pending = {}          # task -> {"prefill":..., "generation":...}
    dirty = False

    def clear():
        if interactive and dirty:
            sys.stdout.write("\r\033[K")

    def commit(text):
        if tui:
            return                    # history lives in the recent pane instead
        clear()
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def handle(line):
        """Returns True if this line changed anything worth repainting."""
        ev = parse_line(line)
        if ev is None:
            if args.debug_unparsed and looks_like_timing(line):
                sys.stderr.write("UNPARSED: %s" % line)
            return False
        return dispatch(ev)

    def dispatch(ev):
        """Feed one event to the tracker and act on what comes back.

        Split out from handle() because the proxy produces events directly --
        it never has a log line to parse -- and both sources have to land in
        exactly the same place or the two backends drift apart.
        """
        nonlocal live_data, live_at, idle_since
        changed = False
        for out in tracker.feed(ev):
            data = out.data
            name = data.get("event")
            changed = True

            if args.json:
                sys.stdout.write(json.dumps(data) + "\n")
                sys.stdout.flush()
                continue

            if out.kind == "live":
                live_data, live_at = data, time.time()
                continue

            if name == "request_start":
                stats.note_start(data.get("model") or tracker.model,
                                 data.get("prompt_tokens"))
                commit("")
                commit(render_header(data, style))
                live_data, live_at = data, time.time()
            elif name == "cache_miss":
                commit("  " + style.yellow(
                    "! cache gone - rereading the whole %s-token prompt"
                    % format(data.get("prompt_tokens") or 0, ",")))
            elif name in ("prefill_done", "generate_done"):
                slot = pending.setdefault(data["task"], {})
                slot["prefill" if name == "prefill_done" else "generation"] = data
            elif name == "request_end":
                slot = pending.pop(data["task"], {})
                stats.record(data.get("model") or tracker.model,
                             slot.get("prefill"), slot.get("generation"), data)
                if history:
                    history.record(data.get("model") or tracker.model,
                                   slot.get("prefill"), slot.get("generation"), data)
                for text in render_summary(slot.get("prefill"),
                                           slot.get("generation"), data, style):
                    commit(text)
                live_data = None
                idle_since = time.time()
            elif name == "request_abandoned":
                pending.pop(data["task"], None)
                stats.note_cancel(data.get("model") or tracker.model)
                commit("  " + style.dim("x task %s cancelled - client disconnected before "
                                        "completion" % data["task"]))
                if live_data is not None and live_data.get("task") == data["task"]:
                    live_data = None
                    idle_since = time.time()
            elif name in ("model_loaded", "server_started"):
                commit("  " + style.dim(out.text))
        return changed

    def poll_codex():
        """Read the agent's session file and bank any turn that just finished.

        Separate from paint() because a --json or --plain run never paints, and
        the turn record is the part worth keeping either way.
        """
        if not codex_tail:
            return
        codex_tail.poll()
        for turn in codex_tail.drain():
            model = turn.get("model") or tracker.model
            turn["model"] = model
            if history:
                # Compare against what this model USUALLY takes before adding
                # this turn to the pile, or a slow turn quietly raises the bar
                # it is being measured against.
                profile = history.turn_profile(model, days=TURN_TYPICAL_DAYS)
                if profile and profile.get("turns") >= MIN_TURNS_FOR_TYPICAL:
                    turn["typical_seconds"] = profile["median_seconds"]
                history.record_turn(turn)
            if args.json:
                sys.stdout.write(json.dumps(dict(turn, event="turn_end")) + "\n")
                sys.stdout.flush()
            elif not tui:
                effort = (" at %s effort" % turn["effort"]) if turn.get("effort") else ""
                verb = "turn finished" if turn.get("completed") else "turn interrupted"
                commit("  " + style.cyan("%s in %s%s"
                                         % (verb, fmt_duration(turn["seconds"]), effort)))

    def live_text():
        if live_data is not None:
            return render_live(live_data, time.time() - live_at, style,
                               floor=progress_floor)
        return render_idle(time.time() - idle_since, style,
                           probe.snapshot() if probe else None)

    def paint():
        if not interactive:
            return
        # Plain mode scrolls, so the notice is committed once as a line of its
        # own; the TUI redraws it every frame instead, from the same box.
        if not tui and update_box.get("latest") and not update_box.get("announced"):
            update_box["announced"] = True
            commit(render_update(update_box["latest"], style))
        if probe:
            probe.poll()               # self-rate-limited: 5s cheap, 15s models
        if tui:
            if ui.view == "upgrade" and ui.upgrade_plan is None:
                # Once per opening of the pane, not once per frame. Deciding
                # how this copy upgrades runs `git status` in a checkout and
                # scans PATH otherwise; at 10 fps that stalled the render loop
                # and backed up keyboard input.
                ui.upgrade_plan = upgrade_plan()
            cols, rows = screen.size()
            snap = stats.snapshot(tracker.model)
            age = time.time() - live_at
            codex_state = codex_tail.state() if codex_tail else None
            system_state = probe.snapshot() if probe else None
            style.width = cols          # keep renderers in step with resizes
            screen.draw(compose_frame(
                snap, live_text(), style, cols, rows,
                help_visible=(ui.view == "help"),
                ui=ui, picker=picker_models, compare=compare_lines,
                live_detail=render_live_detail(live_data, snap, style, age,
                                              codex_state, system_state,
                                              probe.busiest if probe else None),
                codex=codex_state, system=system_state,
                update=update_box.get("latest")), rows, cols)
        else:
            text = live_text()
            if text:
                width = shutil.get_terminal_size((80, 24)).columns
                sys.stdout.write("\r\033[K  " + truncate_visible(text, max(1, width - 3)))
                sys.stdout.flush()
    # `dirty` is only meaningful for the plain single-line renderer.

    if tui:
        screen.enter()
        if keyboard.enter():
            key_thread = threading.Thread(target=_key_reader, args=(q, stop))
            key_thread.daemon = True
            key_thread.start()

    last_paint = 0.0
    next_frame = time.monotonic()
    try:
        while True:
            # Wake on new data OR the frame deadline, whichever comes first.
            timeout = max(0.0, next_frame - time.monotonic())
            batch = []
            try:
                batch.append(q.get(timeout=timeout))
            except Exception:
                pass
            # Drain the rest of a burst so 50 lines cause one repaint, not 50.
            while True:
                try:
                    batch.append(q.get_nowait())
                except Exception:
                    break

            got_data = False
            for kind_, payload in batch:
                if kind_ == "key":
                    was = ui.view
                    if handle_key(ui, payload, picker_models,
                                  update=update_box.get("latest")):
                        got_data = True          # repaint at once, don't wait
                    if ui.upgrade_requested:
                        # Confirmed. Everything after this is teardown, so it
                        # leaves the loop rather than trying to keep drawing a
                        # program that is being replaced underneath it.
                        raise _UpgradeRequested()
                    if ui.view == "picker" and was != "picker":
                        picker_models = pickable_models(history)
                    if ui.view == "compare" and was != "compare":
                        compare_lines = build_compare(history, ui.model_a, ui.model_b, style)
                    continue
                if kind_ == "event":
                    if dispatch(payload):
                        got_data = True
                    continue
                if payload is None:
                    raise KeyboardInterrupt
                if handle(payload):
                    got_data = True

            now = time.monotonic()
            trigger = got_data or bool(screen and screen.resized.is_set())
            do_paint, next_frame = plan_frame(now, last_paint, next_frame, trigger,
                                              live_data is not None)
            if do_paint:
                # On the frame tick rather than every wakeup: a burst of log
                # lines must not turn into a burst of stat() calls on the
                # session file.
                poll_codex()
                if not tui:
                    dirty = True
                paint()
                last_paint = now
    except _UpgradeRequested:
        upgrading = True
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        stop.set()
        if proc:
            proc.terminate()
        if history:
            history.close()
        if keyboard:
            keyboard.leave()
        if screen:
            screen.leave()
        elif interactive:
            clear()
            sys.stdout.write("\n")

    # After the screen is restored, so the package manager's own output is
    # visible on the normal terminal rather than painted over by a frame.
    if upgrading:
        return run_upgrade(style)

    # The alternate buffer takes the whole session with it when it closes, so
    # hand the numbers back on the normal screen before exiting.
    if tui and stats.by_model:
        sys.stdout.write("\nllmwatch session summary\n")
        for model in sorted(stats.by_model):
            snap = stats.snapshot(model)
            sys.stdout.write("\n  %s  (%d requests)\n" % (model, snap["requests"]))
            plain = Style(color=style.color, unicode_ok=style.unicode, width=style.width)
            for text in render_board(snap, plain, style.width, compact=True):
                sys.stdout.write("  " + text + "\n")
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


def show_history(args, out=sys.stdout, history=None, now=None):
    hist = history or History()
    rows = hist.rollup(days=args.days, model=args.model, now=now)
    if not rows:
        out.write("no history yet for that window - run llmwatch alongside your agent "
                  "and it will fill in\n")
        return 1
    out.write("last %d day%s\n\n" % (args.days, "" if args.days == 1 else "s"))
    out.write("  %-26s %6s %12s %12s %10s %s\n"
              % ("model", "req", "prefill", "generate", "cache", "vs previous"))
    for row in rows:
        change = ""
        if row["gen_change_pct"] is not None:
            arrow = "up" if row["gen_change_pct"] >= 0 else "down"
            change = "%s %.0f%%" % (arrow, abs(row["gen_change_pct"]))
        elif row["previous_requests"] == 0:
            change = "no prior data"
        out.write("  %-26s %6d %9s %11s %9s   %s\n" % (
            row["model"][:26], row["requests"],
            ("%.1f tok/s" % row["prefill_rate"]) if row["prefill_rate"] else "-",
            ("%.1f tok/s" % row["gen_rate"]) if row["gen_rate"] else "-",
            ("%.0f%%" % row["cache_pct"]) if row["cache_pct"] is not None else "-",
            change))
    return 0


def show_compare(args, out=sys.stdout, history=None, now=None):
    """Renders through build_compare, the same path the in-TUI view uses."""
    model_a, model_b = args.compare
    hist = history or History()
    style = Style.detect(no_color=getattr(args, "no_color", False))
    out.write("A  %s\nB  %s\n" % (model_a, model_b))
    out.write("compared within prompt-size and cache buckets, because a cached "
              "244-token\nrequest and an uncached 47k one are not the same workload\n\n")
    lines = build_compare(hist, model_a, model_b, style, days=args.days, now=now)
    for line in lines:
        out.write(line + "\n")
    return 0


def show_turns(args, out=sys.stdout, history=None, now=None):
    """How long a turn takes, per model and effort level.

    The question this answers is the one you ask before sending a prompt, not
    after: "if I send this to that model at high effort, am I waiting one minute
    or twenty?"
    """
    hist = history or History()
    rows = hist.turn_rollup(days=args.days, model=args.model, now=now)
    if not rows:
        out.write("no turns recorded yet - run llmwatch --codex alongside your agent "
                  "and it will fill in\n")
        return 1
    out.write("last %d day%s\n\n" % (args.days, "" if args.days == 1 else "s"))
    out.write("  %-26s %8s %6s %10s %10s %s\n"
              % ("model", "effort", "turns", "typical", "longest", "tool calls"))
    for row in rows:
        note = ""
        if row["interrupted"]:
            note = "   (%d interrupted, not timed)" % row["interrupted"]
        out.write("  %-26s %8s %6d %10s %10s %10s%s\n" % (
            row["model"][:26], row["effort"], row["turns"],
            fmt_duration(row["median_seconds"]) if row["median_seconds"] else "-",
            fmt_duration(row["longest_seconds"]) if row["longest_seconds"] else "-",
            # A turn that ran no tools is a real answer, so 0 must print as 0.
            ("%.0f" % row["median_tool_calls"])
            if row["median_tool_calls"] is not None else "-",
            note))
    out.write("\nwall clock from your prompt to the final answer, tool time included\n")
    return 0


def export_history(args, out=sys.stdout, history=None, now=None):
    hist = history or History()
    if getattr(args, "turns", False):
        rows = hist.turn_rows(days=args.days, now=now)
    else:
        rows = hist.all_rows(days=args.days, now=now)
    if args.export == "json":
        out.write(json.dumps(rows, indent=2) + "\n")
        return 0
    if not rows:
        return 1
    columns = list(rows[0].keys())
    out.write(",".join(columns) + "\n")
    for row in rows:
        out.write(",".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llmwatch",
        description="Live prefill progress and tok/s for a local Ollama model.")
    parser.add_argument("--last", action="store_true",
                        help="summarise the most recent completed request and exit")
    parser.add_argument("--json", action="store_true",
                        help="emit one JSON object per event (for status bars/scripts)")
    parser.add_argument("--log", metavar="PATH",
                        help="path to the Ollama log (or an mlx_lm.server log, "
                             "which adds prefill progress under --proxy)")
    parser.add_argument("--proxy", nargs="?", const="", metavar="[HOST:]PORT",
                        help="watch an OpenAI-compatible server (mlx_lm.server, "
                             "LM Studio, llama-server, vLLM) by proxying it. "
                             "Point your client at this address instead. "
                             "Default 127.0.0.1:%d" % DEFAULT_PROXY_PORT)
    parser.add_argument("--upstream", metavar="URL",
                        help="server --proxy forwards to (default %s)" % DEFAULT_UPSTREAM)
    parser.add_argument("--proxy-allow-remote", action="store_true",
                        help="permit --proxy to bind a non-loopback address")
    parser.add_argument("--history", action="store_true",
                        help="per-model summary from your recorded history")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                        help="compare two models on like-for-like recorded requests")
    parser.add_argument("--turns", action="store_true",
                        help="how long a whole agent turn takes, by model and "
                             "reasoning effort (needs --codex to have been recording)")
    parser.add_argument("--export", choices=("json", "csv"),
                        help="dump recorded history (add --turns for turn records)")
    parser.add_argument("--days", type=int, default=7,
                        help="window for --history/--turns/--compare/--export (default 7)")
    parser.add_argument("--model", help="restrict --history/--turns to one model")
    parser.add_argument("--no-history", action="store_true",
                        help="do not record this session (timings only are ever stored)")
    parser.add_argument("--codex", action="store_true",
                        help="show what Codex is doing, read from its session file "
                             "(contains commands and file paths)")
    parser.add_argument("--plain", action="store_true",
                        help="scrolling single-line output instead of the full-screen board")
    parser.add_argument("--no-color", action="store_true", help="disable colour output")
    parser.add_argument("--debug-unparsed", action="store_true",
                        help="print timing-ish lines that failed to parse (bug reports)")
    parser.add_argument("--version", action="version", version="llmwatch " + __version__)
    args = parser.parse_args(argv)

    if args.log:
        os.environ["LLMWATCH_LOG"] = args.log
    if args.export:
        return export_history(args)        # --turns selects which table
    if args.history:
        return show_history(args)
    if args.turns:
        return show_turns(args)
    if args.compare:
        return show_compare(args)
    return summarise_last(args) if args.last else follow(args)


if __name__ == "__main__":
    sys.exit(main())
