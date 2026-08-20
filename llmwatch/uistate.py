"""Which screen is showing, and what a keystroke does to it.

Above the settings pane because a keystroke can open it, and below the frame
composer, which asks this what to draw.
"""

from .constants import __version__
from .text import fmt_duration
from .settings import settings_config, settings_key


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
    ("keys", "h help   -   s settings   -   c compare   -   u upgrade, when one is available"
             "   -   q or ctrl-c quit"),
]


class UIState:
    """Which view is showing, and what the picker has selected.

    Kept separate from rendering so every transition can be tested without a
    terminal.
    """

    def __init__(self):
        self.view = "live"          # live | help | picker | compare | upgrade | settings
        self.cursor = 0
        # Pane state, built by the loop when the pane opens because it needs
        # the settings actually in force, which this class has no business
        # resolving. Cleared on close so it is never stale.
        self.settings = None
        self.settings_saved = None
        # Set by the pane, cleared by the loop, which is the only thing here
        # allowed to open sockets and read files.
        self.settings_discover = False
        # "clients" to list them, "point" to repoint the selected one. Same
        # reason as the scan: this reads and writes files, which the key
        # handler must not.
        self.settings_clients = None
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

    if state.view == "settings":
        # Routed wholesale: the pane has a text field, and a path contains the
        # same letters the board uses as shortcuts.
        state.settings, action = settings_key(state.settings, key)
        if action == "close":
            state.view = "live"
            state.settings = None
        elif action == "save":
            state.settings_saved = settings_config(state.settings)
        elif action == "discover":
            state.settings_discover = True
        elif action in ("clients", "point"):
            state.settings_clients = action
        return True

    if key in ("s", "S"):
        state.view = "settings"
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


def render_help(style, cols, rows):
    out = [style.bold("llmwatch %s - how to read this" % __version__), ""]
    for label, text in HELP_TEXT:
        # Pad BEFORE colouring: ANSI escapes have no width on screen but do count
        # in %-12s, which silently destroys column alignment.
        padded = "%-12s" % label
        out.append("  " + (style.cyan(padded) if label else padded) +
                   " " + (text if label else style.dim(text)))
    return out[:max(1, rows - 1)]
