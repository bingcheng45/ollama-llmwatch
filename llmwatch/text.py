"""Turning values into display strings, and untrusted bytes into safe ones.

The bottom of the rendering stack, and deliberately below the parser and the
proxy too: both of those handle strings that came from a model or a log and
must be sanitised before anything else touches them.
"""
import os
import re
import shutil
import sys


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


SPARK_UNICODE = "▁▂▃▄▅▆▇█"

SPARK_ASCII = ".:-=+*#"
