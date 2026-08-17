# ollama-llmwatch

**Your local model isn't frozen - it's still reading your prompt.** This shows you that, live,
with a progress bar, an ETA, and a plain-English answer to *"should I keep waiting?"*

```
 ollama-llmwatch 0.6.0   qwen3.8:27b-mtp-128k   12 req - 18m04s
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
    cache working: only 6,144 of 21,050 tok to read
    answer ready ~1m45s
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

A coding agent sends 30,000-55,000 tokens of instructions *every turn*. On an M1 Max running a
27B model that's roughly **eight minutes of silent reading** before a single character appears,
while the answer itself takes about twenty seconds.

So you stare at a spinner with no idea whether it's working, stuck, or nearly done. Ollama's API
sends nothing during that window, so every other monitor is blind to it too. The server log is
the only place the information exists - that's what this reads.

## Reading the screen

| what you see | what it means |
|---|---|
| `PREFILL` | reading your prompt - the silent part, with a real ETA |
| `GENERATE` | writing the answer |
| `+41,009 cached` | reused from a previous request, costing nothing |
| `WAIT ███░ 91%` | share of your session spent reading rather than writing |
| `SYSTEM !` | something is competing for your machine |
| `answer ready ~7m10s` | when the *whole answer* will be done, from your measured rates |

And the line that helps you decide whether to wait it out:

```
! cache gone - rereading all 39,528 tok (~6m35s)
! same prompt 5x - likely stuck - interrupt ~11m00s spent
! 3 cancels in a row - client keeps timing out
! slow: 40 vs 100 tok/s usual - 2 models loaded (34 GB), swapping 3.0/4.0 GB
long chat: reading 45,000 tok this turn (~7m30s) - consider compacting
cache working: only 244 of 41,253 tok to read
```

Each appears on its own line, and at most two at once, so the display stays glanceable.

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

### Looking back

Every completed request is recorded locally, so questions that used to need a
benchmark script are just a command:

```bash
ollama-llmwatch --history --days 7        # per-model summary, and the change vs last week
ollama-llmwatch --compare MODEL_A MODEL_B # which build is faster, on your real workload
ollama-llmwatch --export csv              # hand the raw numbers to a spreadsheet
ollama-llmwatch --no-history              # record nothing this session
```

```
$ ollama-llmwatch --compare qwen3.8:27b-mtp-128k qwen3.8:27b-128k
compared within prompt-size and cache buckets, because a cached 244-token
request and an uncached 47k one are not the same workload

  size     cache          mtp-128k         128k   result
  large    hit       14.3 (n=48)   10.0 (n=44)    1.43x faster
  tiny     hit       30.1 (n=12)   28.4 (n=9)     not enough data (need 5 each)
```

Comparisons are bucketed by prompt size and cache state, and report how many samples
each side had. An unbucketed comparison is how a "1.8x faster" result turns out to
have been two different workloads.

Press **`h`** for in-app help explaining every number. **`q`** or **ctrl-c** quits.

---

## FAQ

### Does this need an internet connection?

No - and neither does your model. Local inference is entirely offline; the internet is only
needed to *download* models in the first place.

ollama-llmwatch itself makes **no network calls at all**. It reads a local log file. There's a
test that fails if anyone adds a network client.

### Does it read my prompts, or send anything anywhere?

No. Ollama's log contains only timings and bookkeeping - no prompt text, no file names, no
responses. Nothing leaves your machine.

One exception, and it's opt-in: `--codex` reads your Codex session file, which *does* contain
commands and file paths. That's exactly why it's off by default.

### Will it slow down my model?

No. It reads a file and repaints a terminal. The costlier checks are rate-limited - `ollama ps`
every 15 seconds, and a process lookup only once a slowdown has already been detected.

### How do I know if Ollama isn't running?

It tells you. The idle line distinguishes three states:

```
⠹ Ollama is not running - start it and this will pick up automatically
⠹ no model loaded - the first request pays a load (~10s for a 27B)
⠹ waiting for a request  (idle 12.4s)
```

### Why is my local model so slow?

Usually not for the reason people assume.

**Generation** is limited by memory bandwidth - your machine reads the entire model from memory
*for every single token*. A 16 GB model on an M1 Max (400 GB/s) caps out around 25 tok/s no
matter what you tune.

**Prefill** is usually the bigger cost: re-reading a huge agent prompt every turn. Watch the
`WAIT` line - if it says 90%+, your problem is prompt size, not model speed.

### What counts as a good tok/s?

It depends almost entirely on model size, because it's bandwidth-bound. Rough figures for an
M1 Max:

| model | tok/s |
|---|---|
| 27B at Q4 | 10-17 |
| 14B at Q4 | 25-30 |
| MoE (e.g. Qwen3-30B-A3B) | much higher - only a fraction of weights are read per token |

### How do I actually make things faster?

In the order that pays off:

1. **Cut your agent's prompt size** - fewer plugins and tools loaded
2. **Compact long conversations**
3. **Close things competing for memory bandwidth**
4. **Use a smaller or MoE model**

Tuning flags is the least effective lever.

### Nothing shows up, or the board stays empty

Run `ollama-llmwatch --debug-unparsed`.

- **`UNPARSED:` lines appear** - the parser reads an internal llama.cpp format that changes
  between Ollama versions. Please [open an issue](https://github.com/bingcheng45/ollama-llmwatch/issues)
  with a sample line.
- **Nothing at all appears** - check the log path with `--log`.

### Does it work with Claude Code, open-webui, or my own script?

Yes. It watches the Ollama *server*, so it doesn't care which client is talking to it. The only
Codex-specific part is the optional `--codex` pane.

### Does it work with LM Studio, llama.cpp directly, or vLLM?

Not yet - the parser targets Ollama's bundled `llama-server`. llama.cpp's own server uses a
similar format, so support is plausible. Open an issue if you'd use it.

### Linux? Windows?

Developed and verified on **macOS**. Linux (journald) and Docker paths are written but
unverified - reports very welcome. Windows isn't supported.

### Why is there no CPU or GPU percentage?

Because those numbers don't move when performance does. During inference the GPU sits pinned
near 100% and the CPU near idle whether you're getting 13 tok/s or 8 - the bottleneck is memory
bandwidth, not compute. (GPU utilisation on macOS also requires sudo.)

Instead, slowdowns are detected from actual measured throughput, and cheap signals like loaded
models and swap are used to explain them.

### What does the history file store?

Timings and model names. There is deliberately no column that could hold prompt content,
matching the property the Ollama log itself has. It lives at
`~/.local/share/ollama-llmwatch/history.db` (or `$XDG_DATA_HOME`), it is plain SQLite, and
you can delete it at any time. `--no-history` skips recording entirely.

### Why are there two commands?

`llmwatch` alone was already taken on PyPI by an unrelated project, so the package is
`ollama-llmwatch`. Both commands are installed - use whichever you prefer.

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

The server writes a progress line only every 512 tokens - 5-10 seconds apart - so position is
projected from the last measured rate and repainted 10x/second, clamped so the bar can never
claim work that hasn't happened.

## Limitations

- **It can't show what your agent is doing** (file names, tool calls) - that isn't in Ollama's
  log. `--codex` reads Codex's own session file to fill that gap.
- **Needs local log access.** A remote Ollama server won't work.
- **Depends on an internal log format** with no stability guarantee; it may change between
  Ollama versions. Tests run against real captured logs to catch drift.
- **TTFT is approximate** - measured as prefill duration, since the log has no record of when
  your client sent the request.
- **The full-screen board clears on quit** (that's how alternate-screen apps work); a text
  summary is printed afterwards, and `--plain` keeps normal scrollback.
- **Diagnosis thresholds are calibrated on an M1 Max with 27B models.** A 7B on a 4090 has very
  different ideas about what counts as slow.

## Contributing

Issues and PRs welcome - including "this number looks wrong". Several fixes so far came from
exactly that.

The most useful bug report includes your Ollama version (`ollama --version`), your OS, and a few
lines from `ollama-llmwatch --debug-unparsed`.

```bash
git clone https://github.com/bingcheng45/ollama-llmwatch
cd ollama-llmwatch
python3 -m unittest discover tests -v
```

One file, standard library only. The parsing, tracking, stats and rendering functions are pure  - 
they take data and return data - so almost everything is testable without a terminal or a running
model.

## License

MIT
