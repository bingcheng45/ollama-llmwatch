# ollama-llmwatch

**Your local model isn't frozen — it's still reading your prompt.** This shows you that, live,
with a progress bar, an ETA, and a plain-English answer to *"should I keep waiting?"*

```
 ollama-llmwatch 0.5.2   qwen3.8:27b-mtp-128k   12 req - 18m04s
PREFILL  peak  114.8   avg   92.3   low   47.2 tok/s   218,442 tok - 39m12s
         ▁▂▃▅▇█▇▆▅▃▂▁▂▃▄▅ last 16
GENERATE peak   17.8   avg   13.1   low    2.7 tok/s     3,110 tok - 3m58s
         ▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃ last 16
CACHE      68% reused - 149,204 tok never recomputed
TTFT     min 0.3s - avg 1m52s - max 8m11s (approx: prefill time)
WAIT     ██████████████████░░ 91% of session spent in prefill
SYSTEM   ! swapping 2.5/4.0 GB
── recent ──────────────────────────────────────────────────────────────
  task     prompt      total   prefill speed   share of wait
  2313   14,906 tok    2m47s      89.1 tok/s    98% reading
  2288    6,633 tok    1m26s      77.1 tok/s    97% reading
── live ────────────────────────────────────────────────────────────────
  ⠹ PREFILL ████████░░░░░░░░ 41% 6,144/14,906 tok  101 tok/s  elapsed 1m01s | eta 1m27s
    cache working: only 6,144 of 21,050 tok to read   answer ready ~1m45s
```

## Install

```bash
uv tool install ollama-llmwatch      # or: pipx install ollama-llmwatch
ollama-llmwatch
```

That's it. Python 3.9+, no dependencies, works with your existing Ollama install. Run it in a
terminal next to whatever is using the model.

Prefer a single file? It's one script with no dependencies:

```bash
curl -O https://raw.githubusercontent.com/bingcheng45/ollama-llmwatch/main/llmwatch.py
chmod +x llmwatch.py && ./llmwatch.py
```

Two commands are installed: `ollama-llmwatch` and the shorter `llmwatch`. Same program.

## The problem

A local model request has two steps, and they behave completely differently:

**1. It reads your prompt.** Silent. Nothing appears. This is usually the long part.
**2. It writes the answer.** Now you see text.

A coding agent sends 30,000–55,000 tokens of instructions *every turn*. On an M1 Max running a
27B model that's roughly **eight minutes of silent reading** before a single character appears,
while the answer itself takes about twenty seconds.

So you stare at a spinner with no idea whether it's working, stuck, or nearly done. Ollama's API
sends nothing during that window, so every other monitor is blind to it too. The server log is
the only place the information exists — that's what this reads.

## Reading the screen

| what you see | what it means |
|---|---|
| `PREFILL` | reading your prompt — the silent part, with a real ETA |
| `GENERATE` | writing the answer |
| `+41,009 cached` | reused from a previous request, costing nothing |
| `WAIT ███░ 91%` | share of your session spent reading rather than writing |
| `SYSTEM !` | something is competing for your machine |
| `answer ready ~7m10s` | when the *whole answer* will be done, from your measured rates |

And the line that helps you decide whether to wait it out:

```
! cache gone - rereading all 39,528 tok (~6m35s)
! same prompt 4x - agent may be stuck in a loop
! 3 cancels in a row - client keeps timing out
! slow: 40 vs 100 tok/s usual - 2 models loaded (34 GB), swapping 3.0/4.0 GB
long chat: reading 45,000 tok this turn (~7m30s) - consider compacting
cache working: only 244 of 41,253 tok to read
```

`cache gone` is the big one: nothing was reused, you're paying full price to re-read everything,
and that's usually the moment to interrupt rather than wait.

## Usage

```bash
ollama-llmwatch              # full-screen board
ollama-llmwatch --plain      # scrolling output, keeps your shell scrollback
ollama-llmwatch --last       # summarise the most recent request and exit
ollama-llmwatch --json       # one JSON object per event, for status bars
ollama-llmwatch --codex      # also show what Codex is doing (opt-in, see FAQ)
ollama-llmwatch --log PATH   # if your log isn't auto-detected
```

Press **`h`** for in-app help explaining every number. **`q`** or **ctrl-c** quits.

---

## FAQ

**Does this need an internet connection?**
No. Neither does your model — local inference is entirely offline; the internet is only needed
to *download* models. ollama-llmwatch makes no network calls at all: it reads a local log file.
There's a test that fails if anyone adds a network client.

**Does it read my prompts or send anything anywhere?**
No. Ollama's log contains only timings and bookkeeping — no prompt text, no file names, no
responses. Nothing leaves your machine. The one exception is opt-in: `--codex` reads your Codex
session file, which *does* contain commands and file paths, which is exactly why it's off by
default.

**Will it slow down my model?**
No. It reads a file and repaints a terminal. The expensive checks are rate-limited (`ollama ps`
every 15 seconds, a process lookup only when a slowdown is already detected).

**Why is my local model so slow?**
Usually not the reason people assume. Generation is limited by memory bandwidth: your machine
must read the entire model from memory *for every single token*. A 16 GB model on an M1 Max
(400 GB/s) caps out around 25 tok/s no matter what. But the bigger cost is usually prefill —
re-reading a huge agent prompt every turn. Watch the `WAIT` line: if it says 90%+, your problem
is prompt size, not model speed.

**What's a good tok/s?**
Depends entirely on model size, because it's bandwidth-bound. Rough figures for an M1 Max:
a 27B at Q4 gives 10–17 tok/s; a 14B roughly 25–30; a mixture-of-experts model like Qwen3-30B-A3B
far more, since it only reads a fraction of its weights per token. If you want speed, a smaller
or MoE model beats any amount of tuning.

**How do I actually make things faster?**
In the order that pays off: cut your agent's prompt size (fewer plugins/tools loaded), compact
long conversations, close things competing for memory bandwidth, then consider a smaller or MoE
model. Tuning flags is the least effective lever.

**Nothing shows up / the board stays empty.**
Run `ollama-llmwatch --debug-unparsed` and see whether log lines are arriving but not being
understood. The parser reads an internal llama.cpp format that can change between Ollama
versions — if you see `UNPARSED:` lines, please
[open an issue](https://github.com/bingcheng45/ollama-llmwatch/issues) with a sample. If nothing
appears at all, check the log path with `--log`.

**Does it work with Claude Code / open-webui / my own script?**
Yes. It watches the Ollama *server*, so it doesn't care which client is talking to it. The only
Codex-specific part is the optional `--codex` pane.

**Does it work with LM Studio, llama.cpp directly, or vLLM?**
Not yet — the parser targets Ollama's bundled `llama-server`. llama.cpp's own server uses a
similar format, so support is plausible; open an issue if you'd use it.

**Linux? Windows?**
Developed and verified on macOS. Linux (journald) and Docker paths are written but unverified —
reports very welcome. Windows isn't supported.

**Why is there no CPU or GPU percentage?**
Because they don't move when performance does. During inference the GPU sits pinned near 100%
and the CPU near idle whether you're getting 13 tok/s or 8 — the bottleneck is memory bandwidth,
not compute. (GPU utilisation on macOS also requires sudo.) Instead, slowdowns are detected from
actual measured throughput, and cheap signals like loaded models and swap are used to explain
them.

**Why two commands?**
`llmwatch` alone was taken on PyPI by an unrelated project, so the package is
`ollama-llmwatch`. Both commands are installed; use whichever you prefer.

---

## How it works

It never talks to Ollama's API. It tails the log that Ollama's inference engine already writes:

```
  your agent  ──HTTP──►  Ollama  ──►  llama-server  ──writes──►  ollama.log
                                                                     │
                                                                tail -F
                                                                     ▼
                                                    parse ─► track by slot+task ─► screen
```

Every log line is tagged with a slot and task id:

```
slot print_timing: id  0 | task 2313 | prompt processing, n_tokens = 4096, progress = 0.27
                   ^^^^^   ^^^^^^^^^
```

Keying on `(slot, task)` is what keeps things straight when two models are loaded and their
output interleaves. A request moves through: *reading* → *waiting for first token* → *writing* →
*done*, or *cancelled* if the client disconnects.

Two details that cause most confusion:

- **Caching.** If part of your prompt is already cached, only the rest is computed.
  ollama-llmwatch counts *only tokens that need work* and shows the cached amount separately.
  Ollama's own progress number counts cached tokens too, which is why a naive reading says
  "96% done" when ten seconds of work remain.
- **The gap after reading.** Between the last prompt batch and the first output token, the server
  builds logits and validates its cache while logging nothing. That shows as
  `waiting for first token` with a running clock, rather than a full bar that looks stalled.

The server writes a progress line only every 512 tokens — 5–10 seconds apart — so position is
projected from the last measured rate and repainted 10x/second, clamped so the bar can never
claim work that hasn't happened.

## Limitations

- **It can't show what your agent is doing** (file names, tool calls) — that isn't in Ollama's
  log. `--codex` reads Codex's own session file to fill that gap.
- **Needs local log access.** A remote Ollama server won't work.
- **Depends on an internal log format** with no stability guarantee; it may change between
  Ollama versions. Tests run against real captured logs to catch drift.
- **TTFT is approximate** — measured as prefill duration, since the log has no record of when
  your client sent the request.
- **The full-screen board clears on quit** (that's how alternate-screen apps work); a text
  summary is printed afterwards, and `--plain` keeps normal scrollback.
- **Diagnosis thresholds are calibrated on an M1 Max with 27B models.** A 7B on a 4090 has very
  different ideas about what counts as slow.

## Contributing

Issues and PRs welcome — including "this number looks wrong". Several fixes so far came from
exactly that.

The most useful bug report includes your Ollama version (`ollama --version`), your OS, and a few
lines from `ollama-llmwatch --debug-unparsed`.

```bash
git clone https://github.com/bingcheng45/ollama-llmwatch
cd ollama-llmwatch
python3 -m unittest discover tests -v
```

One file, standard library only. The parsing, tracking, stats and rendering functions are pure —
they take data and return data — so almost everything is testable without a terminal or a running
model.

## License

MIT
