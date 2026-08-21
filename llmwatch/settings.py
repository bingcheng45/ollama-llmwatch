"""Layer 3b: the settings pane, and the state machine behind it.

The pane is a small keyboard-driven form. Its state is plain data and its
rendering is pure, so the whole thing is testable without a terminal.
"""

from .constants import SETTINGS_PRESETS
from .text import safe_text
from .discovery import describe_backend, preset_for

# Two levels: pick the thing to change, then change it. One flat list of every
# setting is how a settings screen becomes a wall.
SETTINGS_TOP = (("watch", "What you are running"),
                ("where", "Where to find it"),
                ("clients", "Point my apps at llmwatch"))

# First, because it is the answer for anyone who would otherwise be stuck:
# the other three rows all assume you know a port or a path.
SETTINGS_WHERE = (("discover", "I do not know, find it for me"),
                  ("proxy_port", "Port your apps use"),
                  ("upstream", "Server address"),
                  ("log", "Log file"))

_ENTER = ("\r", "\n", "ENTER", " ")

_BACK = ("ESC", "LEFT")


def settings_open(settings):
    """Pane state from the settings actually in force."""
    mode = settings["watch"][0]
    upstream = settings["upstream"][0]
    return {
        "mode": mode,
        "proxy_port": settings["proxy_port"][0],
        "upstream": upstream,
        "log": settings["log"][0] or "",
        "preset": preset_for(mode, upstream),
        "sources": {k: v[1] for k, v in settings.items()},
        "level": "top",
        "cursor": 0,
        "editing": None,
        "buffer": "",
        "touched": set(),
        "message": "",
        "suggestion": None,
    }


def settings_config(state):
    """The subset worth saving. Only what this pane can set."""
    data = {"watch": state["mode"]}
    if state.get("proxy_port"):
        data["proxy_port"] = int(state["proxy_port"])
    if state.get("upstream"):
        data["upstream"] = state["upstream"]
    if state.get("log"):
        data["log"] = state["log"]
    return data


def _settings_rows(state):
    if state["level"] == "top":
        return SETTINGS_TOP
    if state["level"] == "watch":
        return [(p[0], p[1]) for p in SETTINGS_PRESETS]
    if state["level"] == "found":
        return [(str(i), describe_backend(row))
                for i, row in enumerate(state.get("found") or [])] or [("none", "")]
    if state["level"] == "clients":
        return [(str(i), row["name"])
                for i, row in enumerate(state.get("clients") or [])] or [("none", "")]
    return SETTINGS_WHERE


def settings_key(state, key):
    """One keypress. Returns (state, action), action in None/close/save.

    Arrows, enter, space and escape. No letter shortcuts: a letter is a
    shortcut on one screen and half a filename on the next, and there is no
    reading of the interface that makes both true at once.

    Choosing saves. There is no separate save step to forget, and nothing here
    is destructive enough to need confirming.
    """
    state = dict(state)
    state["message"] = ""
    rows = _settings_rows(state)

    if state["editing"]:
        field = state["editing"]
        if key in ("ESC",):
            state["editing"], state["buffer"] = None, ""
            return state, None
        if key in ("\r", "\n", "ENTER"):
            value = state["buffer"].strip()
            if field == "proxy_port":
                try:
                    value = int(value)
                except ValueError:
                    state["message"] = "a port has to be a number"
                    return state, None
            state[field] = value
            state["editing"], state["buffer"] = None, ""
            state["touched"] = set(state["touched"]) | {field}
            return state, "save"
        if key in ("\x7f", "\b", "BACKSPACE"):
            state["buffer"] = state["buffer"][:-1]
            return state, None
        if len(key) == 1 and key.isprintable():
            state["buffer"] += key
        return state, None

    if key == "UP":
        state["cursor"] = max(0, state["cursor"] - 1)
        return state, None
    if key == "DOWN":
        state["cursor"] = min(len(rows) - 1, state["cursor"] + 1)
        return state, None

    if key in _BACK:
        if state["level"] != "top":
            state["level"], state["cursor"] = "top", 0
            return state, None
        return state, "close"

    if key in _ENTER:
        name = rows[state["cursor"]][0]
        if state["level"] == "top":
            state["level"], state["cursor"] = name, 0
            # The list is read off disk, so the loop fetches it.
            if name == "clients":
                return state, "clients"
            return state, None
        if state["level"] == "clients":
            if state.get("clients"):
                state["point_index"] = state["cursor"]
                return state, "point"
            return state, None

        if state["level"] == "found":
            rows_found = state.get("found") or []
            if rows_found:
                chosen = rows_found[state["cursor"]]
                for key, value in chosen["apply"].items():
                    state["mode" if key == "watch" else key] = value
                state["preset"] = preset_for(state["mode"], state["upstream"])
                state["touched"] = set(state["touched"]) | set(
                    k for k in chosen["apply"] if k != "watch")
                state["message"] = "saved"
            state["level"], state["cursor"] = "top", 0
            return state, "save" if rows_found else None

        if state["level"] == "watch":
            values = dict(SETTINGS_PRESETS[state["cursor"]][3])
            state["preset"] = name
            state["mode"] = values["watch"]
            if "upstream" in values:
                state["upstream"] = values["upstream"]
            state["level"], state["cursor"] = "top", 0
            state["message"] = "saved"
            return state, "save"
        if name == "discover":
            # The scan opens files and sockets, so the loop runs it. This stays
            # a pure function of keypresses.
            state["message"] = "looking..."
            return state, "discover"
        state["editing"] = name
        state["buffer"] = str(state.get(name) or "")
        return state, None

    return state, None


def _settings_summary(state, name):
    """What the top level shows beside each row, in the same words as the
    screen it opens."""
    if name == "watch":
        for key, title, _why, _values in SETTINGS_PRESETS:
            if key == state["preset"]:
                return title
        return "custom"
    if state["mode"] == "ollama":
        # Nothing to find: the log is located automatically. Showing a server
        # address here would imply it mattered.
        return "found automatically"
    if state["mode"] == "log":
        return state.get("log") or "not set yet"
    return state.get("upstream") or "not set yet"


def render_settings(state, style, detected=None, width=100):
    """The pane. Two levels, and nothing on screen that is not either the
    current answer or a way to change it."""
    keys = {
        "top": "up down move    enter open    esc close",
        "watch": "up down move    enter choose    esc back",
        "where": "up down move    enter edit    esc back",
        "found": "up down move    enter use    esc back",
        "clients": "up down move    enter point it here    esc back",
    }[state["level"]]
    if state["editing"]:
        keys = "type it in    enter keep    esc cancel"

    title = {"top": "SETTINGS",
             "watch": "WHAT YOU ARE RUNNING",
             "where": "WHERE TO FIND IT",
             "found": "WHAT I FOUND",
             "clients": "YOUR APPS"}[state["level"]]
    lines = [style.bold("  " + title) + style.dim("      " + keys), ""]

    if state["level"] == "top":
        for index, (name, label) in enumerate(SETTINGS_TOP):
            mark = ">" if index == state["cursor"] else " "
            row = "  %s %-24s %s" % (mark, label, _settings_summary(state, name))
            lines.append(style.bold(row) if mark == ">" else style.dim(row))

    elif state["level"] == "watch":
        for index, (name, label, why, _values) in enumerate(SETTINGS_PRESETS):
            mark = ">" if index == state["cursor"] else " "
            chosen = "*" if name == state["preset"] else " "
            row = "  %s %s %-18s %s" % (mark, chosen, label, why)
            lines.append(style.bold(row) if mark == ">" else style.dim(row))

    elif state["level"] == "clients":
        here = state.get("listen_url") or ""
        rows_c = state.get("clients") or []
        if not rows_c:
            lines.append(style.dim("  no apps found that llmwatch knows how "
                                   "to configure"))
        for index, row in enumerate(rows_c):
            mark = ">" if index == state["cursor"] else " "
            pointed = "* here" if row["url"] == here else "      "
            text = "  %s %s %-12s %s" % (mark, pointed, row["name"],
                                         safe_text(row["url"], limit=60))
            lines.append(style.bold(text) if mark == ">" else style.dim(text))
        if rows_c:
            lines.append("")
            lines.append(style.dim("  a backup is kept beside each file"))

    elif state["level"] == "found":
        rows_found = state.get("found") or []
        if not rows_found:
            lines.append(style.dim("  nothing found. Is the model server "
                                   "running?"))
        for index, row in enumerate(rows_found):
            mark = ">" if index == state["cursor"] else " "
            text = "  %s %s" % (mark, describe_backend(row))
            lines.append(style.bold(text) if mark == ">" else style.dim(text))

    else:
        for index, (name, label) in enumerate(SETTINGS_WHERE):
            if name == "discover":
                mark = ">" if index == state["cursor"] else " "
                text = "  %s %s" % (mark, label)
                lines.append(style.bold(text) if mark == ">"
                             else style.dim(text))
                continue
            mark = ">" if index == state["cursor"] else " "
            if state["editing"] == name:
                shown, source = state["buffer"] + "_", ""
            elif name in state.get("touched", ()):
                shown, source = str(state.get(name) or "-"), "typed"
            else:
                shown = str(state.get(name) or "-")
                source = state["sources"].get(name, "default")
            row = "  %s %-22s %-32s %s" % (mark, label, shown, source)
            lines.append(style.bold(row) if mark == ">" else style.dim(row))

    if state.get("suggestion") and state["level"] == "top":
        lines.append("")
        lines.append(style.yellow("  looks like: " + state["suggestion"]["why"]))
    elif detected and state["level"] in ("top", "watch"):
        lines.append("")
        lines.append(style.dim("  found running: " + ", ".join(detected)))

    if state["message"]:
        lines.append("")
        lines.append(style.yellow("  " + state["message"]))
    return lines
