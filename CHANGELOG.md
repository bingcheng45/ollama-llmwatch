# Changelog

## 0.3.0 — unreleased

**Full-screen stats board.** Peak / average / low rates for both phases, cache hit rate, TTFT,
session totals, % of the session spent in prefill, recent-request list, and trend sparklines.

- Stats are kept **per model**. The MTP and base builds of the same model differ ~1.34x on
  code; pooling them produces an average that describes neither.
- Averages are **token-weighted** (total tokens / total seconds), not the mean of per-request
  rates — a mean gives a 4-token request the same weight as a 47,000-token one.
- Peak/low **ignore requests under 64 tokens**, so a cache hit computing 4 tokens can't become
  the session "low" forever.
- `--plain` keeps the v0.2.0 scrolling output, and is used automatically when stdout isn't a
  TTY or the terminal is under ~10 rows. `--json` is unchanged.
- The board is cleared on quit (alternate screen), so a plain-text session summary is printed
  afterwards. Terminal restore is wired to `finally`, `atexit`, SIGTERM **and** SIGINT — the
  last one restores at signal time, because waiting for exit can lose the sequence to pty
  buffering.

**More responsive.** The loop now blocks on new data *or* the frame deadline, whichever comes
first: new log data repaints immediately instead of waiting up to 100 ms, while animation still
runs at 10 fps with no data at all. Bursts are coalesced into one repaint, there's a 30 fps
ceiling, and the idle floor drops to 2 fps so it costs almost nothing when idle.

**Readability**

- The recent pane has **column headers** (task / prompt / total / prefill speed / share of
  wait) — four unlabelled numbers per row was a guessing game.
- Press **`h`** for an in-app help screen explaining every phase, column and number, including
  why avg is token-weighted and why TTFT is approximate. The live line stays visible while help
  is open. `q` quits alongside ctrl-c.
- A full README covering the problem in plain English, setup, flow diagrams of how requests are
  tracked, and an honest limitations list.

**Fixes**

- **Colour no longer breaks column alignment.** Labels were padded *after* being wrapped in
  ANSI escapes, which have zero screen width but do count in `%-12s`. A test now asserts that
  stripping colour from any rendered block yields exactly the uncoloured layout.
- **"96% — 3,584/39,528 tok"**: the percentage counted cached tokens while the token ratio
  didn't. Both were individually true and mutually contradictory. A test now asserts the
  displayed percentage always matches the displayed ratio.
- **"100% ... elapsed 28.0s"**: after the last prompt batch, the server builds logits and
  validates the KV cache while logging nothing. That state now reads
  `prompt read - waiting for first token` with a running clock instead of a full bar that looks
  stalled.
- **Cancelled requests** (client disconnected, e.g. a Codex timeout) left a header with nothing
  under it. A new request on the same slot now marks the previous one cancelled explicitly.
- `--last` printed the *current* time as the request time; llama-server's timing lines have no
  timestamps, so it now says `(most recent)` rather than inventing one.

Tests: 35 → 76.

## 0.2.0 — unreleased

**The display no longer looks frozen.** llama-server logs prefill progress only once per
512-token batch — 5–10 seconds apart at typical rates — and the previous version repainted
only when a line arrived, so it sat visibly static in between and looked hung.

- Reading and rendering are now separate: a background thread consumes the log while the UI
  repaints 10x/second on its own clock.
- Position between ticks is projected from the last measured rate, clamped so the bar can
  never claim work that hasn't happened. Measured 1,594 rendered frames from ~29 log ticks.
- Elapsed counts up and ETA counts down continuously; a spinner proves liveness.
- The pre-first-tick state ("reading prompt, 6,637 tok") now carries a clock too — with a
  cached prompt, progress ticks may never arrive at all.

**Visual redesign.** Unicode block bars with ASCII fallback, colour (respecting `NO_COLOR`
and non-TTY output), a per-request header, and a tree-style summary ending in a bar showing
what share of the wall clock was prefill — the number the whole tool exists to surface.

- Idle state shows "waiting for a request" rather than a blank screen.
- Known cosmetic issue: when cache information arrives after projection has begun, the token
  total adjusts mid-flight and the numbers visibly snap once. Self-correcting.

## 0.1.0 — unreleased

First release.

- Live prefill progress with a bar, token counts, rate, and ETA — the phase no other
  Ollama monitor can see, because the HTTP API emits nothing until prefill finishes.
- Per-phase summaries on completion: tokens, wall time, average tok/s, and what share
  of the total wait was prefill.
- **Cache-aware progress.** Ollama's raw `progress` field jumps to ~98% when a prompt
  prefix is cached, while real work remains; llmwatch counts only tokens that actually
  need processing.
- Concurrent requests tracked separately by `slot id` / `task id`, so two loaded models
  don't interleave into unreadable output.
- `--last`, `--json`, `--log`, `--debug-unparsed`.
- Log discovery for macOS Homebrew and app installs; Linux journald and Docker paths
  included but **experimental and unverified**.
- Tested against real captured llama-server output from Ollama 0.32.13 (macOS, M1 Max).
