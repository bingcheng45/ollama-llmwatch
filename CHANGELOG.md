# Changelog

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
