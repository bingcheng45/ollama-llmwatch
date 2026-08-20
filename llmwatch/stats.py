"""Session aggregates: per-phase, per-model, and overall.

Kept apart from the tracker because these answer "how has it been going", not
"what is happening now", and they are what the history and comparison screens
are built out of.
"""
import time

from .constants import LOOP_REPEATS, MIN_TOKENS_FOR_EXTREMES, RECENT_LIMIT


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

    @property
    def median(self):
        """Typical recent rate. Median, not mean: one contended outlier shouldn't
        move the baseline that slowdown detection compares against."""
        values = sorted(self.recent)
        if not values:
            return None
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0

    def snapshot(self):
        return {"tokens": self.tokens, "seconds": self.seconds, "count": self.count,
                "peak": self.peak, "low": self.low, "avg": self.average,
                "median": self.median, "recent": list(self.recent)}


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
        avg_prefill_s = (self.prefill.seconds / float(self.prefill.count)
                         if self.prefill.count else None)
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
            "avg_prefill_seconds": avg_prefill_s,
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
