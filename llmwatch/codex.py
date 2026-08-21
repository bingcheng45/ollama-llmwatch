"""Reading Codex's rollout files to see what the agent is doing.

llmwatch can see the model working but not why. This is the other half: the
tool calls the agent is making while the model is busy.
"""
import json
import os
import re
import time

from .text import safe_text, short_model_name


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
