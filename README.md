# llmwatch

**Your local model isn't stuck — it's still reading your prompt.** `llmwatch` shows you that,
live, with a progress bar and an ETA.

```
── 01:41:41  qwen3.8:27b-mtp-128k  task 2313
  ⠹ PREFILL  ████░░░░░░░░░░░░  27% 4,096/14,906 tok    101 tok/s  elapsed 40.5s | eta 1m47s
```

...and when it finishes:

```
── 01:41:41  qwen3.8:27b-mtp-128k  task 2313
  ├ PREFILL     14,906 tok +512 cached     2m47s   avg   89.1 tok/s
  ├ GENERATE        15 tok                  2.7s   avg    5.6 tok/s
  └ TOTAL                                  2m49s  ████████████████████ 98% was prefill
```

The live line repaints 10x per second even though the server logs progress only once
per 512-token batch (5-10 seconds apart) — position is projected from the last measured
rate, so the display never sits frozen while you wonder if anything is happening.

## The problem

A local LLM request has two phases, and they perform completely differently:

| phase | what it does | bottleneck | measured on an M1 Max (27B Q4) |
|---|---|---|---|
| **prefill** | reads your prompt | GPU compute — tokens are batched, weights read once per batch | ~100 tok/s |
| **generation** | writes the answer | memory bandwidth — all 16 GB of weights re-read **per token** | ~13–17 tok/s |

Generation looks slower per token, but prefill processes *far* more tokens. A coding agent
sends 30k–55k tokens of system prompt and tool definitions on **every turn**, so a real
request looks like this:

```
prefill:     47,174 tok  ~8 minutes     <- silent, no output at all
generation:     300 tok  ~20 seconds
```

**~96% of the wait happens before the first character appears.** Your agent shows a spinner.
You have no idea whether it's 10% done or 90% done, or whether anything is happening at all.

## Why no other tool shows this

Ollama's HTTP API emits its first chunk only *after* prefill completes, so anything built on
the API — including every existing Ollama monitor — is blind to that window. llama.cpp added a
`prompt_progress` field for exactly this ([#14685](https://github.com/ggml-org/llama.cpp/issues/14685)),
but Ollama doesn't pass it through.

The server log is currently the only place the information exists. `llmwatch` tails it.

## Install

```bash
uv tool install ollama-llmwatch      # or: pipx install ollama-llmwatch
```

The installed command is `llmwatch`. The distribution is named `ollama-llmwatch` because
the PyPI name `llmwatch` belongs to an unrelated LLM cost-tracking package.

Or just take the file — it's a single script with no dependencies:

```bash
curl -O https://raw.githubusercontent.com/bingcheng45/llmwatch/main/llmwatch.py
chmod +x llmwatch.py && ./llmwatch.py
```

## Use

```bash
llmwatch                  # follow live, in a terminal beside your agent
llmwatch --last           # summarise the most recent request
llmwatch --json           # one JSON object per event, for status bars
llmwatch --log PATH       # explicit log location
llmwatch --debug-unparsed # show timing lines that failed to parse (bug reports)
```

| line | meaning |
|---|---|
| `PREFILL [####----] 41%` | still reading your prompt, with an honest ETA |
| `GENERATE 217 tok  13.2 tok/s (now 14.7)` | writing; cumulative rate and a 3-second window |
| `v PREFILL` / `v GENERATE` | phase finished: tokens, wall time, average rate |
| `= TOTAL` | whole request, and what share of it was prefill |

## Cached prompts are handled honestly

When Ollama reuses a cached prefix, its raw `progress` field jumps straight to ~98% while real
work remains. A tool that trusts it reports "98%, eta 0.1s" when ten seconds of computing are
left. `llmwatch` counts only the tokens that actually need processing:

```
 v PREFILL  840 tok (+32,112 cached)  in   16.9s   avg   49.6 tok/s
```

That case is pinned by a test against a real captured log (`tests/fixtures/cache-hit.log`).

## Related tools

These are good, and they solve a different half of the problem — they read the HTTP stream, so
they show generation but cannot see prefill:

- [ollama-token-monitor (otop)](https://github.com/TiniLLM/ollama-token-monitor) — full dashboard: tok/s, GPU, VRAM, loaded models
- [howfast](https://github.com/spinualexandru/howfast) — per-response token metrics

Use otop if you want a system dashboard. Use `llmwatch` if you want to know why nothing has
happened for four minutes.

## Known limitations

- **Needs local log access.** A remote Ollama server won't work — the log has to be readable.
- **The log format is an internal llama.cpp detail** with no stability guarantee. It may change
  between Ollama versions. Tests run against real captured logs to catch drift; if output goes
  quiet after an upgrade, run `llmwatch --debug-unparsed` and please open an issue with a sample.
- **Ollama only** in v1. The parser targets Ollama's bundled `llama-server`.
- **Linux (journald) and Docker paths are experimental** — developed on macOS/Homebrew and not
  yet verified elsewhere. Reports welcome.
- Progress ticks only appear for prompts larger than the batch size (512 tokens); shorter
  prompts jump straight to the summary, which is fine — they're fast anyway.

## Development

```bash
python3 -m unittest discover tests -v
```

Fixtures in `tests/fixtures/` are real, sanitised llama-server output. If you hit a format the
parser mishandles, a fixture plus an expected-values test is the most useful possible PR.

## License

MIT
