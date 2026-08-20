"""Layer 3: state in, strings out. Pure, and the bulk of the UI.

Nothing here touches a terminal, a clock it was not handed, or a file. Every
function takes data and returns lines, which is why almost the whole interface
can be tested without drawing anything.
"""
import time

from .constants import (
    DEFAULT_PROXY_PORT, GEN_TICK_TOKENS, LONG_CONTEXT_TOKENS, LOOP_ESCALATE,
    LOOP_REPEATS, LOW_DRAFT_ACCEPT, PREFILL_BATCH, SAME_WITHIN, SLOWDOWN_MIN_SAMPLES,
    SLOWDOWN_RATIO, __version__)
from .text import (
    SPARK_ASCII, SPARK_UNICODE, fmt_bar, fmt_duration, safe_text, spinner_frame)
from .system import busiest_process


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


def render_board_title(snap, style):
    """`llmwatch 0.9.1   qwen3.8-27b-4bit (MLX)`.

    The engine is appended only when the server identified itself. A model name
    does not imply one, so an unrecognised server gets the name alone rather
    than a guess.
    """
    title = "llmwatch %s   %s" % (__version__, style.bold(snap.get("model", "?")))
    engine = snap.get("engine")
    if engine:
        title += style.dim(" (%s)" % engine)
    return title


def render_idle(age, style, system=None):
    """Idle is ambiguous: nothing has happened yet, or nothing CAN happen.
    Saying which is the difference between waiting patiently and waiting
    pointlessly."""
    spin = spinner_frame(time.time(), style)
    system = system or {}

    # Under --proxy, Ollama is not the thing being watched and is frequently
    # not running at all. Reporting its state here described a program the
    # user had not asked about, and read as a detection failure: someone
    # proxying a loaded llama.cpp server was told "no model loaded".
    if system.get("proxying"):
        where = system.get("upstream") or "the upstream"
        if system.get("upstream_ok") is False:
            return style.yellow(
                "%s cannot reach %s - the proxy is up, the server behind it is not"
                % (spin, where))
        line = "%s waiting for a request  (idle %s)  |  forwarding to %s" % (
            spin, fmt_duration(age), where)
        # The silent failure this exists for: a client asking for a model id
        # the server does not serve gets nothing back, and nothing says why.
        models = system.get("upstream_models")
        if models:
            line += "  |  serving: %s" % ", ".join(
                safe_text(name, limit=60) for name in models[:3])
            if len(models) > 3:
                line += " (+%d)" % (len(models) - 3)
        elif models == []:
            line += "  |  the server reports no models"
        if system.get("suggestion"):
            line += "  |  %s - press s" % system["suggestion"]["why"]
        # The likeliest reason a board is empty while a model is plainly busy.
        astray = system.get("clients_elsewhere") or []
        if astray:
            names = safe_text(astray[0]["name"], limit=30)
            if len(astray) > 1:
                names += " +%d" % (len(astray) - 1)
            line += ("  |  %s not measured: pointed elsewhere, press s"
                     % names)
        return style.dim(line)

    # An OpenAI server was found while the Ollama side had nothing to show.
    # Appended rather than substituted: whatever is wrong with Ollama is still
    # the more specific thing to say, and this is the likelier explanation of
    # why the screen is empty.
    # A suggestion knows more than the bare detection does, so it wins the
    # one line available.
    suggestion = system.get("suggestion")
    hint = ""
    if suggestion:
        hint = "  |  %s - press s to switch" % suggestion["why"]
    elif system.get("oai_port"):
        port = system["oai_port"]
        hint = ("  |  OpenAI server on :%d - watch it with --proxy %d, and "
                "point your client at :%d"
                % (port, DEFAULT_PROXY_PORT, DEFAULT_PROXY_PORT))
    if system.get("server_ok") is False:
        problem = system.get("server_problem") or "Ollama is not running"
        return style.yellow("%s %s - start it and this will pick up automatically%s"
                            % (spin, problem, hint))
    if system.get("models_loaded") == 0:
        return style.dim("%s no model loaded - the first request pays a load "
                         "(~10s for a 27B)  (idle %s)%s"
                         % (spin, fmt_duration(age), hint))
    return style.dim("%s waiting for a request  (idle %s)%s"
                     % (spin, fmt_duration(age), hint))


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
