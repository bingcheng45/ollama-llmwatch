# Changelog

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
