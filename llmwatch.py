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
import atexit
import json
import os
import re
import select
import shutil
import signal
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

__version__ = "0.4.0"

# Frame pacing. The loop wakes on new log data OR on these deadlines, whichever
# comes first, so fresh data appears immediately while animation (spinner,
# elapsed, ETA countdown, projected position) keeps moving when the log is
# silent -- llama-server logs progress only once per 512-token batch, 5-10s
# apart at typical rates.
FRAME_ACTIVE = 0.1     # 10 fps while a request is in flight
FRAME_IDLE = 0.5       # 2 fps when nothing is running: a spinner needs no more
MIN_FRAME_GAP = 1 / 30.0   # hard ceiling so a log flood can't spin the CPU

# Requests smaller than this are excluded from peak/low. A cached prompt can
# process 4 tokens at a meaningless rate; without this floor that number becomes
# the session "low" forever and the board quietly lies.
MIN_TOKENS_FOR_EXTREMES = 64

RECENT_LIMIT = 20      # requests kept for the sparkline and the recent pane

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

    def snapshot(self):
        return {"tokens": self.tokens, "seconds": self.seconds, "count": self.count,
                "peak": self.peak, "low": self.low, "avg": self.average,
                "recent": list(self.recent)}


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
        return max(0.0, eta - age) + avg_out / gen_rate
    if data.get("event") == "generate_tick":
        done = data.get("decoded", 0) + data.get("rate", 0) * age
        return max(0.0, avg_out - done) / gen_rate
    return None


LONG_CONTEXT_TOKENS = 30000     # beyond this, re-reading dominates every turn
LOW_DRAFT_ACCEPT = 0.35         # below this, speculative decoding is losing
LOOP_REPEATS = 3                # identical prompt sizes in a row = likely retry loop


def diagnose(data, snap, style):
    """Short, plain-English reads on what's happening, worth acting on.

    Deliberately terse: this line exists to help someone decide "keep waiting or
    kill it?" in about a second. Raw telemetry belongs on the board, not here.
    Returns at most two findings, most important first.
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

    if snap.get("looping"):
        out.append(style.yellow("! same prompt %dx - agent may be stuck in a loop"
                                % snap.get("repeat_count", LOOP_REPEATS)))

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


def render_live_detail(data, snap, style, age=0.0):
    """Second live line: why this is slow, and when the answer will be ready."""
    bits = diagnose(data, snap, style)
    finish = project_completion(snap or {}, data, age)
    if finish is not None:
        bits.append(style.dim("answer ready ~%s" % fmt_duration(finish)))
    return "   ".join(bits)


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


def render_board(snap, style, width=80, compact=False):
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
    return lines


def render_codex(state, style, width=80):
    """What the agent itself is doing. Opt-in (--codex): this reads your Codex
    session file, which contains command text and file paths."""
    if not state:
        return []
    out = []
    action = state.get("action")
    if action:
        out.append("  %s %s" % (style.bold("%-12s" % "last action"),
                                style.cyan(action)))
        if state.get("detail"):
            # Keep it glanceable. The full command lives in your agent's window;
            # here it only has to be recognisable.
            limit = max(30, min(72, width - 20))
            detail = state["detail"]
            if len(detail) > limit:
                detail = detail[:limit - 1] + "…" if style.unicode else detail[:limit - 3] + "..."
            out.append("  %-12s %s" % ("", style.dim(detail)))
    if state.get("calls"):
        out.append("  %-12s %s" % ("this turn", style.dim("%d tool calls" % state["calls"])))
    if state.get("waiting_since") is not None:
        out.append("  %-12s %s" % ("waiting on", style.dim(
            "model for %s" % fmt_duration(state["waiting_since"]))))
    return out


def render_recent(rows, style, width=80, limit=6):
    """Recent requests, newest first. Carries a header row: without one the
    columns are four unlabelled numbers and the reader has to guess."""
    out = [style.dim("  %-6s %12s %9s %14s   %s"
                     % ("task", "prompt", "total", "prefill speed", "share of wait"))]
    for row in rows[:limit]:
        share = ("%3.0f%% reading" % row["share"]) if row.get("share") is not None else ""
        rate = ("%6.1f tok/s" % row["rate"]) if row.get("rate") is not None else ""
        out.append("  %-6s %12s %9s %14s   %s" % (
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
    ("", ""),
    ("keys", "h close help   -   ctrl-c quit"),
]


def render_help(style, cols, rows):
    out = [style.bold(" llmwatch %s - how to read this" % __version__), ""]
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

    def draw(self, lines, rows):
        """One write per frame. Several writes at 10 fps flicker visibly."""
        clipped = lines[:max(1, rows - 1)]
        body = "\033[K\n".join(clipped)
        self.stream.write("\033[H" + body + "\033[K\033[J")
        self.stream.flush()


def compose_frame(snap, live_text, style, cols, rows, hint=True, help_visible=False,
                  live_detail="", codex=None):
    """Build the whole TUI frame. Pure: takes a snapshot, returns lines.

    Panes drop in priority order when the terminal is short -- sparklines first,
    then the recent list -- but the live line is never dropped, because that is
    the thing the tool exists to show.
    """
    rule = "─" if style.unicode else "-"

    def divider(label):
        text = "%s %s " % (rule * 2, label)
        return style.dim(text + rule * max(0, min(cols, 100) - len(text) - 1))

    budget = max(1, rows - 1)

    if help_visible:
        # Help replaces the board but keeps the live line: you should never lose
        # sight of the running request just because you asked what a column means.
        help_lines = render_help(style, cols, rows - 4)
        tail = ["", divider("live"), "  " + (live_text or style.dim("idle"))]
        if live_detail:
            tail.append("    " + live_detail)
        return help_lines + tail

    # Reserved first and never trimmed: everything else is context, but this is
    # the answer to "is it still reading my prompt, or is it writing?".
    live_block = ["", divider("live"), "  " + (live_text or style.dim("idle"))]
    if live_detail:
        live_block.append("    " + live_detail)
    if len(live_block) >= budget:
        return live_block[-budget:]

    title = " llmwatch %s   %s" % (__version__, style.bold(snap.get("model", "?")))
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
                  render_board(snap, style, cols, compact=compact),
                  codex_block,
                  recent_block,
                  ["", style.dim("  h help   -   ctrl-c quit")] if hint else []):
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


class CodexTail:
    """Reads Codex's own session rollout file to show what the agent is doing.

    Opt-in, because unlike the Ollama log this file contains real content --
    commands, file paths, message text. Correlation with a given model request is
    by time only: no shared request id exists between Codex and Ollama.
    """

    SESSIONS = "~/.codex/sessions"

    def __init__(self):
        self.path = None
        self.offset = 0
        self.action = None
        self.detail = None
        self.calls = 0
        self.last_call_at = None

    @classmethod
    def newest_session(cls):
        root = os.path.expanduser(cls.SESSIONS)
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
        newest = self.newest_session()
        if newest != self.path:
            self.path, self.offset = newest, 0
            self.calls = 0
        if not self.path:
            return
        try:
            with open(self.path, "r", errors="replace") as fh:
                fh.seek(self.offset)
                for line in fh:
                    self._consume(line)
                self.offset = fh.tell()
        except OSError:
            return

    def _consume(self, line):
        try:
            record = json.loads(line)
        except ValueError:
            return
        payload = record.get("payload") or {}
        kind = payload.get("type") or record.get("type")
        if kind == "function_call":
            self.calls += 1
            self.action = payload.get("name") or "tool call"
            self.detail = self._summarise(payload.get("arguments"))
            self.last_call_at = time.time()
        elif kind == "task_started":
            self.calls = 0
            self.action = "thinking"
            self.detail = None
            self.last_call_at = time.time()
        elif kind == "task_complete":
            self.action = "done"
            self.detail = None
            self.last_call_at = None

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
        return {"action": self.action, "detail": self.detail,
                "calls": self.calls, "waiting_since": waiting}


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


def _key_reader(q, stop):
    """Keys go through the same queue as log lines, so a keypress wakes the main
    loop immediately instead of waiting for the next frame."""
    while not stop.is_set():
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.25)
            if ready:
                ch = sys.stdin.read(1)
                if ch:
                    q.put(("key", ch))
        except (ValueError, OSError):
            return


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
    # Below ~10 rows the panes cannot be laid out usefully, so fall back to the
    # single-line renderer rather than draw something corrupt.
    tui = (interactive and not args.plain
           and shutil.get_terminal_size((80, 24)).lines >= 10)

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

    stats = Stats()
    screen = Screen() if tui else None
    keyboard = Keyboard() if tui else None
    codex_tail = CodexTail() if (tui and args.codex) else None
    help_visible = False
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
        nonlocal live_data, live_at, idle_since
        ev = parse_line(line)
        if ev is None:
            if args.debug_unparsed and looks_like_timing(line):
                sys.stderr.write("UNPARSED: %s" % line)
            return False
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

    def live_text():
        if live_data is not None:
            return render_live(live_data, time.time() - live_at, style)
        return render_idle(time.time() - idle_since, style)

    def paint():
        if not interactive:
            return
        if tui:
            cols, rows = screen.size()
            snap = stats.snapshot(tracker.model)
            age = time.time() - live_at
            if codex_tail:
                codex_tail.poll()
            screen.draw(compose_frame(
                snap, live_text(), style, cols, rows,
                help_visible=help_visible,
                live_detail=render_live_detail(live_data, snap, style, age),
                codex=codex_tail.state() if codex_tail else None), rows)
        else:
            text = live_text()
            if text:
                sys.stdout.write("\r\033[K  " + text)
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
                    if payload in ("h", "H", "?"):
                        help_visible = not help_visible
                        got_data = True          # repaint at once, don't wait
                    elif payload in ("q", "Q"):
                        raise KeyboardInterrupt
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
                if not tui:
                    dirty = True
                paint()
                last_paint = now
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        stop.set()
        proc.terminate()
        if keyboard:
            keyboard.leave()
        if screen:
            screen.leave()
        elif interactive:
            clear()
            sys.stdout.write("\n")

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


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llmwatch",
        description="Live prefill progress and tok/s for a local Ollama model.")
    parser.add_argument("--last", action="store_true",
                        help="summarise the most recent completed request and exit")
    parser.add_argument("--json", action="store_true",
                        help="emit one JSON object per event (for status bars/scripts)")
    parser.add_argument("--log", metavar="PATH", help="path to the Ollama log")
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
    return summarise_last(args) if args.last else follow(args)


if __name__ == "__main__":
    sys.exit(main())
