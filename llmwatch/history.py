"""The SQLite store: finished requests and whole agent turns.

Kept so that "that took 12m20s" can become "which is 2x your usual", which is
the part that is actually useful.
"""
import os
import sqlite3
import time

from .constants import MIN_COMPARE_SAMPLES

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
