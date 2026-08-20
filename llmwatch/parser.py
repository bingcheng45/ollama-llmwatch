"""Layer 1: log line in, event out. Pure, no I/O.

Both of Ollama's engines are parsed here, into the one vocabulary in events.py:
llama-server for GGUF models, and the MLX runner for -mlx models on Apple
Silicon, plus the standalone mlx_lm.server people run themselves.

The contract that matters is fail-soft. These are internal log lines with no
stability guarantee, so an unrecognised one returns None rather than raising:
a format change degrades llmwatch, it does not crash it.
"""
import calendar
import datetime
import re

from .events import (
    CacheInfo, CacheMiss, CheckpointCreated, CheckpointErased, CheckpointRestored,
    DraftAcceptance, GenDone, GenTick, MlxGenStats, MlxPeakMemory, MlxPrefillTick,
    MlxRequestEnd, MlxRequestStart, MlxRunnerReady, MlxRunnerStart, ModelLoaded,
    OaiPrefillTick, PrefillDone, PrefillTick, RequestEnd, RequestStart, ServerStarted)
from .text import short_model_name

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
