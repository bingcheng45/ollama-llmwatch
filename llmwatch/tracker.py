"""Layer 2: events in, state out. A pure state machine, no I/O.

Tracker.feed(event) -> [Output] is the whole interface. Everything about what a
request is currently doing lives here, which is why it is testable without a
terminal, a log file or a running model.
"""
import time

from .constants import MIN_MEASURABLE_SECONDS
from .events import (
    CacheInfo, CacheMiss, CheckpointCreated, CheckpointErased, CheckpointRestored,
    DraftAcceptance, GenDone, GenTick, MLX_EVENTS, MlxGenStats, MlxPeakMemory,
    MlxPrefillTick, MlxRequestEnd, MlxRequestStart, MlxRunnerReady, MlxRunnerStart,
    ModelLoaded, OAI_EVENTS, OaiEngine, OaiGenTick, OaiPrefillTick, OaiRequestAborted,
    OaiRequestEnd, OaiRequestStart, Output, PrefillDone, PrefillTick, RequestEnd,
    RequestStart, ServerStarted)


class Request:
    """One model request, tracked from prompt to final timings."""

    def __init__(self, slot, task, model, prompt_tokens=None, ctx=None):
        self.slot = slot
        self.task = task
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.ctx = ctx
        self.cached = 0
        self.started = time.time()
        # llama-server prints its timing block AFTER generation has streamed, so
        # phase summaries are buffered and emitted in true order at request end.
        self.prefill = None   # (ms, tokens, rate)
        self.generation = None
        # Why this request is slow, in the server's own words.
        self.cache_miss = False
        self.restored_tokens = 0
        self.checkpoints = None      # (index, total)
        self.draft = None            # (rate, accepted, generated, mean_len)
        self.status = None           # short human phrase for the live line
        self.last_live = None        # last emitted live payload, for re-emission

    @property
    def to_process(self):
        """Tokens that actually need computing -- total minus what the prompt
        cache already holds. This is the number that governs the wait."""
        if self.prompt_tokens is None:
            return None
        return max(0, self.prompt_tokens - self.cached)


class Tracker:
    """Consumes events, emits Outputs. No I/O, no terminal escapes: a caller
    decides whether 'live' means an animated line or a JSON record."""

    # The MLX runner serves one request at a time and numbers nothing, so it
    # gets a fixed synthetic slot and a counter standing in for llama.cpp's
    # task id. Slot 0 is what a single-slot llama-server uses too, and the two
    # engines never appear in the same request, so they cannot collide.
    MLX_SLOT = 0
    # The proxied OpenAI backend gets its own slot. The two adapters cannot run
    # against the same server, but a stale key from one must never be able to
    # collide with a live request from the other.
    OAI_SLOT = 1

    def __init__(self):
        self.model = "?"
        # Which engine is behind the numbers, once the server has said so.
        # None until then, and never inferred from the model name.
        self.engine = None
        self.requests = {}
        self._mlx_task = 0
        self._mlx_started = None    # ts of the current request's first line
        self._mlx_prefilled = None  # ts prefill finished == generation began
        self._mlx_loading = None     # ts the runner subprocess was spawned
        self._mlx_pending_end = None  # request duration, held until it can close
        self._oai_task = 0
        self._oai_started = None    # wall clock the proxy sent the request
        self._oai_first = None      # wall clock the first token came back
        self._oai_open = False      # is there a request the log ticks belong to?

    def _key(self, ev):
        return (ev.slot, ev.task)

    def _get(self, ev):
        key = self._key(ev)
        if key not in self.requests:
            self.requests[key] = Request(ev.slot, ev.task, self.model)
        return self.requests[key]

    @staticmethod
    def _context(req):
        """The 'why is this slow' fields, attached to every live payload."""
        return {"cache_miss": req.cache_miss, "status": req.status,
                "restored_tokens": req.restored_tokens,
                "checkpoints": req.checkpoints, "draft": req.draft}

    def feed(self, ev):
        """Returns a list of Output. Never raises on unexpected ordering."""
        if ev is None:
            return []

        if isinstance(ev, MLX_EVENTS):
            return self._feed_mlx(ev)

        if isinstance(ev, OAI_EVENTS):
            return self._feed_oai(ev)

        if isinstance(ev, ModelLoaded):
            self.model = ev.model
            return [Output("line", "model loaded: %s" % ev.model,
                           {"event": "model_loaded", "model": ev.model})]

        if isinstance(ev, ServerStarted):
            return [Output("line", "weights loaded into GPU in %.1fs" % ev.seconds,
                           {"event": "server_started", "seconds": ev.seconds})]

        if isinstance(ev, RequestStart):
            outs = []
            # A slot handles one request at a time, so anything still open on this
            # slot was cancelled -- the client disconnected (a Codex timeout, say)
            # and llama-server never wrote its `total time` line. Without this the
            # display just shows a header with nothing under it, which reads like
            # the tool lost track.
            for key in [k for k in self.requests if k[0] == ev.slot]:
                old = self.requests.pop(key)
                outs.append(Output("line", "", {
                    "event": "request_abandoned", "task": old.task,
                    "model": old.model, "prompt_tokens": old.prompt_tokens}))

            req = Request(ev.slot, ev.task, self.model, ev.prompt_tokens, ev.ctx)
            self.requests[self._key(ev)] = req
            outs.append(Output("line", "",
                               {"event": "request_start", "task": ev.task,
                                "prompt_tokens": ev.prompt_tokens, "model": req.model,
                                "started": req.started}))
            return outs

        if isinstance(ev, CacheInfo):
            req = self._get(ev)
            # Several cache lines arrive per request as checkpoints advance; the
            # first is the one that describes what was reused up front.
            if not req.cached:
                req.cached = ev.cached
            return []

        if isinstance(ev, PrefillTick):
            req = self._get(ev)
            total = req.to_process
            # Fall back to llama.cpp's own fraction if we never saw the prompt size.
            if total:
                fraction = min(1.0, ev.processed / float(total))
            else:
                fraction = ev.progress
                total = int(ev.processed / ev.progress) if ev.progress else ev.processed
            eta = (ev.elapsed / fraction - ev.elapsed) if fraction > 0 else None
            payload = {"event": "prefill_tick", "task": ev.task,
                       "model": req.model,
                       "processed": ev.processed, "to_process": total,
                       "cached": req.cached, "fraction": fraction,
                       "rate": ev.rate, "elapsed": ev.elapsed, "eta_seconds": eta}
            payload.update(self._context(req))
            req.last_live = payload
            return [Output("live", "", payload)]

        # ---- state that explains the wait ---------------------------------
        if isinstance(ev, (CacheMiss, CheckpointRestored, CheckpointCreated,
                           CheckpointErased, DraftAcceptance)):
            req = self._get(ev)
            outs = []
            if isinstance(ev, CacheMiss):
                req.cache_miss = True
                req.status = "cache miss - reprocessing the whole prompt"
                # Committed as well as shown live: this is the single most useful
                # thing to know when deciding whether to keep waiting.
                outs.append(Output("line", "", {
                    "event": "cache_miss", "task": ev.task,
                    "prompt_tokens": req.prompt_tokens}))
            elif isinstance(ev, CheckpointRestored):
                req.restored_tokens = ev.tokens
                req.status = "restored %s cached tokens" % format(ev.tokens, ",")
            elif isinstance(ev, CheckpointCreated):
                req.checkpoints = (ev.index, ev.total)
                req.status = "saving cache checkpoint %d/%d" % (ev.index, ev.total)
            elif isinstance(ev, CheckpointErased):
                req.status = "discarding stale cache (%s tok)" % format(ev.tokens, ",")
            else:
                req.draft = (ev.rate, ev.accepted, ev.generated, ev.mean_len)

            # Re-emit the current live view so the change shows immediately
            # instead of waiting for the next 512-token batch.
            if req.last_live:
                payload = dict(req.last_live)
                payload.update(self._context(req))
                req.last_live = payload
                outs.append(Output("live", "", payload))
            return outs

        if isinstance(ev, PrefillDone):
            req = self._get(ev)
            req.prefill = (ev.ms, ev.tokens, ev.rate)
            return []

        if isinstance(ev, GenTick):
            req = self._get(ev)
            elapsed = ev.decoded / ev.rate if ev.rate else 0.0
            payload = {"event": "generate_tick", "task": ev.task,
                       "model": req.model, "decoded": ev.decoded,
                       "rate": ev.rate, "rate_3s": ev.rate_3s, "elapsed": elapsed}
            payload.update(self._context(req))
            req.last_live = payload
            return [Output("live", "", payload)]

        if isinstance(ev, GenDone):
            req = self._get(ev)
            req.generation = (ev.ms, ev.tokens, ev.rate)
            return []

        if isinstance(ev, RequestEnd):
            return self._finish(ev)

        return []

    def _feed_mlx(self, ev):
        """Translate one MLX event into the llama.cpp events the rest of this
        class already handles.

        Everything downstream -- rendering, stats, history, compare -- then sees
        one shape of request regardless of which engine produced it. The cost of
        that is here: MLX splits a request across lines that share no id, and
        prints progress without a rate, so both have to be reconstructed.
        """
        if isinstance(ev, MlxRunnerStart):
            self._mlx_loading = ev.ts
            return self.feed(ModelLoaded(ev.model))

        if isinstance(ev, MlxRunnerReady):
            # "Loading" here is the weight load, the same thing llama-server
            # reports as `llama-server started in Ns` -- worth showing, because
            # on a 26B model it is ~10s of the first request's apparent latency.
            if self._mlx_loading is None or ev.ts is None:
                return []
            seconds = max(0.0, ev.ts - self._mlx_loading)
            self._mlx_loading = None
            return self.feed(ServerStarted(seconds))

        if isinstance(ev, MlxRequestStart):
            # Belt and braces: if a build ever stops printing the closing line,
            # the previous request still closes here instead of hanging on the
            # board forever.
            outs = self._mlx_close()
            self._mlx_task += 1
            self._mlx_started = ev.ts
            self._mlx_prefilled = None
            task = self._mlx_task
            # MLX gives no context size; None keeps the ctx field honestly empty
            # rather than inventing a window the renderer would then draw.
            # The MLX runner and llama-server write different logs and are
            # read by different parsers, so on this path the engine is known
            # for certain rather than detected.
            self.engine = "MLX"
            outs.extend(self.feed(RequestStart(self.MLX_SLOT, task, ev.prompt_tokens, None)))
            outs.extend(self.feed(CacheInfo(self.MLX_SLOT, task, ev.cached)))
            if ev.miss:
                outs.extend(self.feed(CacheMiss(self.MLX_SLOT, task)))
            return outs

        if isinstance(ev, MlxPrefillTick):
            elapsed = self._mlx_elapsed_since(self._mlx_started, ev.ts)
            self._mlx_prefilled = ev.ts
            rate = (ev.processed / elapsed) if elapsed > 0 else 0.0
            fraction = (ev.processed / float(ev.total)) if ev.total else 0.0
            return self.feed(PrefillTick(self.MLX_SLOT, self._mlx_task, ev.processed,
                                         fraction, elapsed, rate))

        if isinstance(ev, MlxGenStats):
            # MLX never prints a generation token count. With speculative
            # decoding each iteration commits one token from the target model
            # plus whichever drafted tokens were accepted, which reconstructs it
            # exactly rather than approximately.
            tokens = ev.iterations + ev.accepted
            # Generation began when prefill ended, except on a prompt the cache
            # served whole: MLX prints no progress line then, so the request
            # start is the only honest mark.
            began = self._mlx_prefilled or self._mlx_started
            seconds = self._mlx_elapsed_since(began, ev.ts)
            outs = self.feed(DraftAcceptance(self.MLX_SLOT, self._mlx_task, ev.acceptance,
                                             ev.accepted, ev.drafted, ev.avg_draft))
            # No duration means no rate. Reporting 0 tok/s here would not just
            # look odd on one request: it passes the small-request noise guard,
            # so it lands in `low`, drags the token-weighted average up by
            # adding tokens with no seconds, and skews the median that slowdown
            # detection reads. One unmeasurable request would recolour the board.
            if seconds <= 0 or tokens <= 0:
                return outs
            outs.extend(self.feed(GenDone(self.MLX_SLOT, self._mlx_task,
                                          seconds * 1000.0, tokens, tokens / seconds)))
            return outs

        if isinstance(ev, MlxRequestEnd):
            # Held, not emitted. The stats line carrying the generation rate is
            # written from somewhere else in the runner and lands either side of
            # this one; closing here would drop that rate from the summary on
            # every request that happened to lose the race.
            self._mlx_pending_end = ev.seconds
            return []

        if isinstance(ev, MlxPeakMemory):
            return self._mlx_close()

        return []

    def _mlx_close(self):
        """Emit the held request end, once nothing more can arrive for it."""
        if self._mlx_pending_end is None:
            return []
        seconds, self._mlx_pending_end = self._mlx_pending_end, None
        req = self.requests.get((self.MLX_SLOT, self._mlx_task))
        if req is None:
            # Attaching to a log mid-request: the tail of one is visible but its
            # start never was, so there is no model, no prompt size and no phase
            # to report. A summary of nothing but a duration, under a model
            # named "?", reads as a bug rather than as the partial view it is.
            return []
        # The prefill summary llama-server prints as its own line has to be
        # derived here, from where generation was seen to start.
        if req is not None and req.prefill is None and req.last_live:
            elapsed = self._mlx_elapsed_since(self._mlx_started, self._mlx_prefilled)
            processed = req.last_live.get("processed") or 0
            if elapsed > 0 and processed:
                req.prefill = (elapsed * 1000.0, processed, processed / elapsed)
        tokens = (req.generation[1] if req and req.generation else 0)
        self._mlx_started = self._mlx_prefilled = None
        return self.feed(RequestEnd(self.MLX_SLOT, self._mlx_task,
                                    seconds * 1000.0, tokens))

    @staticmethod
    def _mlx_elapsed_since(start, now):
        """Seconds between two log timestamps, or 0.0 if either is missing.

        Never negative: a clock step backwards should cost the display a rate,
        not turn it into a number that reads as real.
        """
        if start is None or now is None:
            return 0.0
        return max(0.0, now - start)

    def _feed_oai(self, ev):
        """Translate one proxied-request event into the llama.cpp events the
        rest of this class already handles.

        Same contract as _feed_mlx, but the numbers come from a different place:
        nothing here was printed by the server, so a request that never sent a
        `usage` block closes with no rate at all rather than an estimate.
        """
        if isinstance(ev, OaiEngine):
            self.engine = ev.engine
            return []

        if isinstance(ev, OaiRequestStart):
            outs = []
            if ev.model and ev.model != self.model:
                outs.extend(self.feed(ModelLoaded(ev.model)))
            self._oai_task += 1
            self._oai_started = ev.ts
            self._oai_first = None
            self._oai_open = True
            outs.extend(self.feed(
                # The prompt size is in the usage block, which arrives last, so
                # it is genuinely unknown here. None leaves the field empty
                # rather than seeding it with a number the ticks would contradict.
                RequestStart(self.OAI_SLOT, self._oai_task, None, None)))
            return outs

        if isinstance(ev, OaiPrefillTick):
            # The log keeps producing these between requests -- another client,
            # or a warm-up. With no request open they belong to nothing.
            if not self._oai_open:
                return []
            elapsed = self._mlx_elapsed_since(self._oai_started, ev.ts)
            if elapsed <= 0:
                return []
            fraction = (ev.processed / float(ev.total)) if ev.total else 0.0
            return self.feed(PrefillTick(self.OAI_SLOT, self._oai_task, ev.processed,
                                         fraction, elapsed, ev.processed / elapsed))

        if isinstance(ev, OaiGenTick):
            if not self._oai_open:
                return []
            if self._oai_first is None:
                self._oai_first = ev.ts
            elapsed = self._mlx_elapsed_since(self._oai_first, ev.ts)
            if elapsed <= 0:
                return []
            rate = ev.decoded / elapsed
            # No 3s window is tracked here; the same rate in both slots keeps the
            # renderer honest rather than inventing a second series.
            return self.feed(GenTick(self.OAI_SLOT, self._oai_task, ev.decoded, rate, rate))

        if isinstance(ev, OaiRequestAborted):
            return self._oai_abandon()

        if isinstance(ev, OaiRequestEnd):
            if not self._oai_open:
                return []
            self._oai_open = False
            outs = []
            cached = ev.cached_tokens or 0
            if cached:
                outs.extend(self.feed(CacheInfo(self.OAI_SLOT, self._oai_task, cached)))

            prefill_s = self._mlx_elapsed_since(ev.started, ev.first_token)
            # First token to last, not to the end of the response: the trailing
            # usage and [DONE] frames are protocol, not generation.
            gen_s = self._mlx_elapsed_since(ev.first_token, ev.last_token)
            total_s = self._mlx_elapsed_since(ev.started, ev.ts)

            # usage counts the whole prompt including whatever the cache served.
            # Only the remainder was actually computed, and dividing by the full
            # figure would report a prefill rate the machine never achieved.
            fresh = max(0, (ev.prompt_tokens or 0) - cached)
            if prefill_s >= MIN_MEASURABLE_SECONDS and fresh > 0:
                outs.extend(self.feed(PrefillDone(self.OAI_SLOT, self._oai_task,
                                                  prefill_s * 1000.0, fresh,
                                                  fresh / prefill_s)))
            # Same guard as the MLX path: a phase with no duration or no tokens
            # contributes no rate, because a zero would pass the noise filter and
            # then drag down `low`, the weighted average and the slowdown median.
            gen_tokens = ev.completion_tokens or 0
            if gen_s >= MIN_MEASURABLE_SECONDS and gen_tokens > 0:
                outs.extend(self.feed(GenDone(self.OAI_SLOT, self._oai_task,
                                              gen_s * 1000.0, gen_tokens,
                                              gen_tokens / gen_s)))
            # Set on the Request rather than passed through RequestEnd, because
            # that is where the log backend puts it and where _finish reads it
            # from -- so history and the hints stay backend-agnostic.
            if ev.draft is not None:
                req = self.requests.get((self.OAI_SLOT, self._oai_task))
                if req is not None:
                    req.draft = ev.draft
            outs.extend(self.feed(RequestEnd(self.OAI_SLOT, self._oai_task,
                                             total_s * 1000.0, gen_tokens)))
            self._oai_started = self._oai_first = None
            return outs

        return []

    def _oai_abandon(self):
        """Drop the open proxied request without recording anything.

        A cancelled request has real timings but a truncated token count, and
        agents cancel constantly -- opencode abandons a stream the moment a tool
        result makes it stale. Banking those as completed requests would fill the
        history with short, fast-looking rows that never happened.
        """
        self._oai_open = False
        self._oai_started = self._oai_first = None
        req = self.requests.pop((self.OAI_SLOT, self._oai_task), None)
        if req is None:
            return []
        return [Output("line", "", {
            "event": "request_abandoned", "task": req.task,
            "model": req.model, "prompt_tokens": req.prompt_tokens})]

    def _finish(self, ev):
        req = self.requests.pop(self._key(ev), None)
        total_s = ev.ms / 1000.0
        prefill = req.prefill if req else None
        generation = req.generation if req else None
        share = None
        if prefill and total_s > 0:
            share = prefill[0] / 1000.0 / total_s * 100

        out = []
        if prefill:
            out.append(Output("line", "", {
                "event": "prefill_done", "task": ev.task, "tokens": prefill[1],
                "cached": req.cached, "seconds": prefill[0] / 1000.0, "rate": prefill[2]}))
        if generation:
            out.append(Output("line", "", {
                "event": "generate_done", "task": ev.task, "tokens": generation[1],
                "seconds": generation[0] / 1000.0, "rate": generation[2]}))
        out.append(Output("line", "", {
            "event": "request_end", "task": ev.task, "seconds": total_s,
            "model": req.model if req else "?",
            "prefill_share_pct": share,
            "cache_miss": bool(req and req.cache_miss),
            "draft": req.draft if req else None,
            "started": req.started if req else None}))
        return out
