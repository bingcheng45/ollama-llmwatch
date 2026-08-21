"""Raw-mode key reading, and decoding escape sequences into key names.

POSIX only. Everything degrades to a no-keyboard board elsewhere rather than
failing.
"""
import atexit
import os
import select
import sys

try:                       # POSIX only; the TUI degrades to no-keyboard elsewhere
    import termios
    import tty
except ImportError:        # pragma: no cover - Windows
    termios = None
    tty = None


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
