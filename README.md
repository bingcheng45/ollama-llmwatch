# ollama-llmwatch

[![tests](https://github.com/bingcheng45/ollama-llmwatch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bingcheng45/ollama-llmwatch/actions/workflows/ci.yml)
[![security](https://github.com/bingcheng45/ollama-llmwatch/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/bingcheng45/ollama-llmwatch/actions/workflows/security.yml)
[![PyPI](https://img.shields.io/pypi/v/ollama-llmwatch)](https://pypi.org/project/ollama-llmwatch/)
[![Python](https://img.shields.io/pypi/pyversions/ollama-llmwatch)](https://pypi.org/project/ollama-llmwatch/)

**Your local model isn't frozen - it's still reading your prompt.** This shows you that, live,
with a progress bar, an ETA, and a plain-English answer to *"should I keep waiting?"*

And when it finishes: how long that whole thing actually took, at what reasoning effort, and
whether that was normal. See [How long does a turn take?](#how-long-does-a-turn-take)

```
  ollama-llmwatch 0.9.1   qwen3.8:27b-mtp-128k   12 req - 18m04s
  PREFILL  peak  114.8   avg   92.3   low   47.2 tok/s   218,442 tok - 39m12s
           ▁▂▃▅▇█▇▆▅▃▂▁▂▃▄▅ last 16
  GENERATE peak   17.8   avg   13.1   low    2.7 tok/s     3,110 tok - 3m58s
           ▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃ last 16
  CACHE      68% reused - 149,204 tok never recomputed
  TTFT     min 0.3s - avg 1m52s - max 8m11s (approx: prefill time)
  WAIT     ██████████████████░░ 91% of session spent in prefill
  SYSTEM   ! swapping 2.5/4.0 GB
  ── recent ────────────────────────────────────────────────────────────
  task     prompt      total   prefill speed   share of wait
  2313   14,906 tok    2m47s      89.1 tok/s    98% reading
  2288    6,633 tok    1m26s      77.1 tok/s    97% reading
  ── live ──────────────────────────────────────────────────────────────
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

Both of Ollama's engines are read automatically: `llama-server` for GGUF models, and the MLX
runner for `-mlx` models on Apple Silicon. Nothing to pick or configure.

Prefer a single file? It's one script with no dependencies:

```bash
curl -O https://raw.githubusercontent.com/bingcheng45/ollama-llmwatch/main/llmwatch.py
chmod +x llmwatch.py && ./llmwatch.py
```

Two commands are installed: `ollama-llmwatch` and the shorter `llmwatch`. Same program.

### Upgrading

**Press `u`.** When a newer version exists llmwatch says so, and `u` upgrades the copy you are
running:

```
update available: 0.10.0 (you have 0.9.1) - press u, or run uv tool upgrade ollama-llmwatch
```

It shows the exact command first and waits for `y`, so the key that opens it is never the key
that runs it. It quits afterwards, because the program it is running from is what just changed.

It refuses rather than guessing when running it would be a bad idea, and tells you why: a
checkout with uncommitted changes is never pulled over, and a missing tool is named rather than
discovered halfway through. Then you run it yourself:

```bash
uv tool upgrade ollama-llmwatch        # installed with uv
pipx upgrade ollama-llmwatch           # installed with pipx
pip install --upgrade ollama-llmwatch  # installed with pip
git pull                               # running from a checkout
curl -O https://raw.githubusercontent.com/bingcheng45/ollama-llmwatch/main/llmwatch.py
```

Nothing in that command comes from the log, a model name or a version string. It is chosen from
the list above by where the file lives, and run as an argument list rather than through a shell.

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
ollama-llmwatch --codex      # what Codex is doing, and how long this turn has taken (opt-in)
ollama-llmwatch --log PATH   # if your log isn't auto-detected
```

### Comparing models (press `c`)

Press `c` and pick two models with the arrow keys:

```
  ── compare: pick two models ──────────────────────────────────────────
     #  model                        req    gen tok/s   last seen
   > 1  qwen3.8:27b-mtp-128k          48       14.3     2h ago
     2  qwen3.8:27b-128k              44       10.0     1d ago
     3  qwen3.8:27b-q4_K_M             0          -     no data yet

    up/down move   1-9 jump   enter pick   esc back
```

```
  ── qwen3.8:27b-mtp-128k  vs  qwen3.8:27b-128k ────────────────────────
     48 requests, last 2h ago      44 requests, last 1d ago

     GENERATE   A ████████████████████   14.3 tok/s  A 1.43x faster
                B ██████████████░░░░░░   10.0 tok/s
     PREFILL    A ████████████████████  101.7 tok/s
                B ███████████████████░   98.4 tok/s
     TTFT       A 1m58s      B 2m02s      A 4.0s sooner
     CACHE      A 33%        B 31%
     DRAFT      A 53%        B not a speculative build

     by prompt size          A            B          result
       large   hit   14.3 (n=48)  10.0 (n=44)      A 1.43x faster

     on your median request (12,000 tok prompt, 207 tok answer)
       A  2m12s    ███████████████████
       B  2m22s    ████████████████████
       A saves 10.2s per request

     median agent turn: your prompt to the final answer, tool time included
     TURN       A 6m19s (n=11)       B 12m56s (n=9)       all efforts pooled
       low      A 57.3s (n=6)        B 1m44s (n=4)        A 1.81x quicker
       high     A 33m29s (n=5)       B 41m02s (n=5)       A 1.23x quicker
```

That third block is the point. Generation is 1.43x faster, but on a real request that is only
10 seconds, because reading the prompt dominates. Rates flatter; seconds do not.

The last block is the number you actually waited through. See
[How long does a turn take?](#how-long-does-a-turn-take) for where it comes from and why the
pooled row refuses to pick a winner.

Models you have never measured still appear in the list, and picking one tells you exactly how
to get data for it rather than silently doing nothing.

### How long does a turn take?

A request is one call to the model. A **turn** is one thing you asked for: submit a prompt, wait
through however many requests and tool calls it takes, read the final answer. That is the number
people actually mean by "how long does this take", and none of the rest of this tool could see
it, because the Ollama log has no idea your twelve requests were one question.

Run with `--codex` and llmwatch reads the turn boundaries out of the Codex session file. While a
turn runs you get a clock; when it ends you get the total, and whether that was normal:

```
  ── codex ─────────────────────────────────────────────────────────────
  last action  shell
               pytest tests/
  this turn    12m32s so far - 6 tool calls - effort high
  waiting on   model for 41.0s
```

```
  last turn    12m20s - effort high - 14 tool calls
               2.1x your usual 6m00s
```

Reasoning effort is recorded with it, because it is usually the largest single factor and the
one you control:

```
$ ollama-llmwatch --turns --days 30

  model                        effort  turns    typical    longest tool calls
  qwen3.8:27b-mtp-128k           high      5     33m29s      3h21m         16   (2 interrupted, not timed)
  qwen3.8:27b-mtp-128k            low      6      57.3s     34m29s          0
  gpt-5.6-sol                    high      2     10m04s     17m09s          0

wall clock from your prompt to the final answer, tool time included
```

Same model, same machine: 57 seconds on low effort, 33 minutes on high. That is the kind of
thing worth knowing before you send the prompt rather than after.

Notes on how to read it:

- **Tool time is included**, because it is time you waited. A turn that spent nine minutes
  running your test suite is a nine-minute turn.
- **Medians, not means.** One turn where you walked away and left the agent blocked would
  otherwise set the expectation for every turn after it.
- **Interrupted turns are counted but never timed.** How long you waited before pressing escape
  measures your patience, not the model.
- **The pooled TURN row never names a winner.** A model you mostly ran on low effort against one
  you mostly ran on high is not a comparison. The verdict lives on the per-effort rows, and only
  where both sides have at least 3 turns.
- Durations, model names and effort levels are stored. The message text sitting next to them in
  the session file is not, and never reaches the database. See
  [What does the history file store?](#what-does-the-history-file-store)

### Looking back

Every completed request is recorded locally, so questions that used to need a
benchmark script are just a command:

```bash
ollama-llmwatch --history --days 7        # per-model summary, and the change vs last week
ollama-llmwatch --turns --days 30         # how long a whole turn takes, by model and effort
ollama-llmwatch --compare MODEL_A MODEL_B # which build is faster, on your real workload
ollama-llmwatch --export csv              # hand the raw numbers to a spreadsheet
ollama-llmwatch --export csv --turns      # the same, for turn records
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
needed to *download* models in the first place. Everything llmwatch actually does works with the
network unplugged: it reads a local log file.

There is exactly one network call, and it is not part of the job: **once a day it asks PyPI
whether there is a newer version.** It runs on a background thread, times out after two seconds,
and stays silent if anything at all goes wrong. Turn it off and nothing else changes:

```bash
export LLMWATCH_NO_UPDATE_CHECK=1
```

It is off automatically under `--json`, which is usually running unattended in a script.

This used to say "no network calls at all", and that was true until 0.9.1. It changed because
0.8.0 shipped to GitHub and never reached PyPI, and nobody running an older copy had any way to
find out. Tests keep it to the one call: a single import, inside a single function, with no
second network client allowed anywhere in the file.

### Does it read my prompts, or send anything anywhere?

No. Ollama's log contains only timings and bookkeeping - no prompt text, no file names, no
responses. **Nothing about you or your models ever leaves your machine.**

The daily update check is the only thing that goes out, and it carries nothing: an empty GET to
the same public URL every user of the package requests, `https://pypi.org/pypi/ollama-llmwatch/json`.
No query string, no model names, no identifier, no usage data. PyPI learns that somebody asked,
which it already knew when you installed. Disable it with `LLMWATCH_NO_UPDATE_CHECK=1`.

One exception, and it's opt-in: `--codex` reads your Codex session file, which *does* contain
commands and file paths. That's exactly why it's off by default. Of what it reads, only turn
durations, model names and effort levels are ever written to disk; the message text is used for
nothing and stored nowhere.

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

- **`UNPARSED:` lines appear** - the parser reads an internal engine format that changes
  between Ollama versions. Please [open an issue](https://github.com/bingcheng45/ollama-llmwatch/issues)
  with a sample line.
- **Nothing at all appears** - check the log path with `--log`.

If the header shows your model name but the counter sits at `0 req`, the log is being read
fine and the requests are not being recognised. On versions before 0.9.0 that was what an
`-mlx` model looked like; upgrading fixes it.

### Does it work with MLX models?

Yes, since 0.9.0, with nothing to configure. Ollama runs GGUF models through `llama-server`
and `-mlx` models through its own MLX runner, and the two write completely different logs.
Both are read automatically, including in the same log across a restart.

One number differs in how it is obtained. MLX never prints a generation token count, so it is
reconstructed from the speculative decoding stats: each iteration commits one token from the
target model plus whichever drafted tokens were accepted, which is the exact count rather than
an approximation. Prefill and total times are measured, as with any other model.

### Does it work with Claude Code, open-webui, or my own script?

Yes. It watches the Ollama *server*, so it doesn't care which client is talking to it. The only
Codex-specific part is the optional `--codex` pane.

### Does it work with LM Studio, llama.cpp directly, or vLLM?

Not yet - the parser targets the two engines Ollama bundles, `llama-server` and the MLX runner.
llama.cpp's own server uses a similar format to the first, so support is plausible. Open an
issue if you'd use it.

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

Timings, model names, and reasoning effort. There is deliberately no column that could hold
prompt content, matching the property the Ollama log itself has. It lives at
`~/.local/share/ollama-llmwatch/history.db` (or `$XDG_DATA_HOME`), it is plain SQLite, and
you can delete it at any time. `--no-history` skips recording entirely.

Two tables: `requests`, one row per model request, and `turns`, one row per agent turn when
you run with `--codex`. The Codex session file that turn timings come from *does* contain your
agent's output; the duration, the model and the effort level are lifted out of it and nothing
else is, which is why `turns` has no column that could hold text either.

### Why are there two commands?

`llmwatch` alone was already taken on PyPI by an unrelated project, so the package is
`ollama-llmwatch`. Both commands are installed - use whichever you prefer.

---

## How it works

It never talks to Ollama's API. It tails the log that Ollama's inference engine already writes:

```
                                    ┌─►  llama-server  ─┐   GGUF
  your agent  ──HTTP──►  Ollama  ───┤                   ├──writes──►  ollama.log
                                    └─►  MLX runner   ──┘   -mlx          │
                                                                     tail -F
                                                                          ▼
                                                    parse ─► track by slot+task ─► screen
```

Which engine serves a request depends on the model, and the two log nothing alike. Both are
parsed into the same shape, so everything downstream of `parse` is engine-agnostic.

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
- **On MLX, generation tokens are reconstructed** from the speculative decoding stats rather
  than read directly, because that runner never prints a count. Prefill and total times are
  measured normally. A model running without speculative decoding reports no generation rate.
- **Turn times need `--codex`**, and only Codex. The Ollama log cannot tell that twelve requests
  were one question, so turn boundaries have to come from the agent. Other agents record the
  same thing in their own formats; support for them is not written yet.
- **A turn time is wall clock, not model time.** It includes tool execution, approval prompts,
  and any time the agent spent waiting on you. That is deliberate, but it means a turn time is
  not a measurement of the model alone.
- **The full-screen board clears on quit** (that's how alternate-screen apps work); a text
  summary is printed afterwards, and `--plain` keeps normal scrollback.
- **Diagnosis thresholds are calibrated on an M1 Max with 27B models.** A 7B on a 4090 has very
  different ideas about what counts as slow.

## Contributing

Issues and PRs welcome - including "this number looks wrong". Several fixes so far came from
exactly that, and a wrong number is the worst bug this project can have.

The most useful bug report includes your Ollama version (`ollama --version`), your OS, and a few
lines from `ollama-llmwatch --debug-unparsed`.

```bash
git clone https://github.com/bingcheng45/ollama-llmwatch
cd ollama-llmwatch
python3 -m unittest discover tests -v
```

That is the whole setup: one file, standard library only, no build step. The parsing, tracking,
stats and rendering functions are pure  -  they take data and return data  -  so almost
everything is testable without a terminal or a running model.

[CONTRIBUTING.md](CONTRIBUTING.md) has the rest: how the four layers fit together, what the tests
are really guarding against, the style rules, and how a release goes out. Security problems go
through [SECURITY.md](SECURITY.md) rather than a public issue. Everyone here is expected to
follow the [Code of Conduct](CODE_OF_CONDUCT.md), which is two paragraphs of "be decent" and some
detail underneath.

Every push and PR runs the tests on Python 3.9 to 3.13, on Linux and macOS, plus Bandit, CodeQL
and a dependency audit. The security scans also run weekly, because advisories appear after the
last commit does.

## License

MIT
