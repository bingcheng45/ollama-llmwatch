"""The terminal itself, and the composition of one frame.

Screen owns the alternate buffer and guarantees the restore; the frame
functions decide what a screenful looks like at a given size.
"""
import atexit
import shutil
import signal
import sys
import threading

from .constants import FRAME_ACTIVE, FRAME_IDLE, MIN_FRAME_GAP
from .text import fmt_duration, truncate_visible
from .render import render_board, render_board_title, render_codex, render_recent
from .settings import render_settings
from .uistate import render_help, render_picker
from .update import render_update, render_upgrade_confirm


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

    if ui is not None and getattr(ui, "view", None) == "settings" and ui.settings:
        # Same rule as help: the board goes, the live line stays, because
        # changing a setting is no reason to lose sight of a running request.
        found = []
        if (system or {}).get("oai_port"):
            found.append(":%d" % system["oai_port"])
        pane = render_settings(ui.settings, style, detected=found, width=cols)
        tail = ["", divider("live"), live_text or style.dim("idle")]
        for detail in (live_detail or []):
            tail.append(detail)
        return pane + tail

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

    title = render_board_title(snap, style)
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
                  ["", style.dim("h help   -   s settings   -   c compare   -   ctrl-c quit")] if hint else []):
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
