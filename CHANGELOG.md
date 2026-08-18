# Changelog

## 0.9.1

**You will now hear about new versions.** 0.8.0 shipped to GitHub and never reached PyPI, and
nobody running an older copy had any way to find out. Releases are automated now, and llmwatch
tells you when it is behind.

- A tag builds, tests and publishes to PyPI through Trusted Publishing, so there is no API token
  stored anywhere. It refuses to publish when the tag, `__version__` and `pyproject.toml`
  disagree: a version on PyPI can never be reused or corrected, so the number has to be right
  before it goes out rather than after.
- Bandit, a dependency audit and CodeQL run on every push and pull request, and weekly, because
  advisories appear after the last commit does.
- **One network call, and a note about a promise that changed.** This README used to say llmwatch
  makes no network calls at all. It now makes exactly one: once a day it asks PyPI for the latest
  version, on a background thread, with a two second timeout, silent on any failure. It sends
  nothing about you - an empty GET to the same public URL every user requests, with no query
  string, no model names and no identifier. It is off under `--json`, and
  `LLMWATCH_NO_UPDATE_CHECK=1` turns it off everywhere.
- The test that forbade a network client is narrowed rather than deleted, because it was doing
  its job: one import, inside one function, and no second network client anywhere in the file.
  Deleting it would have left nothing to stop the next one.
- The version comparison is numeric, not lexicographic. `"0.10.0" < "0.9.0"` as strings, which
  would have told everyone to downgrade the day 0.10 shipped. Pre-releases are refused outright
  rather than ordered by guesswork.
- **The notice names the command that upgrades your copy**, not one command for everybody. There
  are four supported ways to install this and they upgrade differently; telling a pipx user to
  run `uv`, or someone who downloaded a single file to run a package manager they never used,
  fails in front of them and teaches them to ignore the line. It reads its own location and says
  `uv tool upgrade`, `pipx upgrade`, `pip install --upgrade`, `git pull` or `curl` accordingly.

**Community files.** CONTRIBUTING, SECURITY, a code of conduct and a pull request template, none
of which existed. The security policy routes reports to a private advisory rather than a public
issue, and says what is in scope for a tool whose attack surface is log content it prints to your
terminal. A test now fails if any of these files goes missing, since GitHub's checklist notices
quietly and nobody else does.

## 0.9.0

**MLX models are no longer invisible.** Ollama runs GGUF models through `llama-server` but any
`-mlx` model through its own MLX runner, which logs a completely different dialect. llmwatch
read only the first, so on an Apple Silicon machine running something like `gemma4:26b-mlx` the
board showed the model name and then `0 req`, forever, with nothing on screen to say why. That
was the entire experience for MLX users: a tool that looked broken while the model worked fine.

- Both dialects are now understood, automatically. Nothing selects between them and there is no
  flag or setting to find: which engine runs is a property of the model, and a single log can
  hold both across a restart, so sniffing once and locking it in would go blind halfway through
  the file.
- Full parity for what matters: prompt size, cache hits and misses, live prefill progress and
  ETA, prefill and generation rates, total time, and speculative decoding acceptance.
- **Generation token counts are reconstructed, because MLX never prints one.** With speculative
  decoding each iteration commits one token from the target model plus whichever drafted tokens
  were accepted that round, so `iterations + accepted` is the exact count rather than an
  estimate standing in for one.
- Durations come from the log's own timestamps, not the wall clock, so `--last` on a log from
  yesterday reports what actually happened instead of measuring how long ago it happened.
- The weight load is timed separately (~10s for a 26B model) and kept out of the request, where
  it would otherwise be charged to your prompt. For the same reason the outer
  `[GIN] POST "/api/chat"` line is ignored in favour of the runner's own completion timing.
- The MLX runner writes its generation stats and its completion line from different places, so
  either can reach the log first, and both orders occur within one session. A request is
  therefore closed on `peak memory`, which across a full session appears exactly once per
  request and always after both. Closing on the completion line instead would drop the
  generation rate from the summary of every request that happened to lose the race, which is
  roughly half of them. If a build ever stops printing that line, the next request still closes
  the previous one rather than stranding it on the board.

**Alignment.** The frame has two indent levels: column 0 for lines that head it, two spaces for
content under a divider. The title and the key hints each carried a stray extra space, so the
one line naming your model sat out of line with every row beneath it. Both now start at column
0, with a test that fails on a stray indent anywhere in the frame.

## 0.8.0

**How long did that whole thing take?** Everything llmwatch measured until now was one model
request. A person waiting on a coding agent is waiting on a turn: one prompt, however many
requests and tool calls it takes, and the final answer at the end. With `--codex`, that span is
now timed, recorded, and comparable.

- A clock while the turn runs (`this turn 12m32s so far - 6 tool calls - effort high`), and the
  total when it ends, next to whether that was normal for this model: `last turn 12m20s`,
  `2.1x your usual 6m00s`.
- **Reasoning effort is recorded with it.** On the development machine the same model on the
  same hardware medians 57 seconds on low effort and 33 minutes on high. That is not a footnote,
  it is the largest thing you control, and it was invisible before.
- `--turns` prints the table: turns, typical, longest and tool calls per model and effort level.
  This is the question people ask *before* sending a prompt, not after.
- The compare view (`c`, or `--compare`) gains a turn block, split by effort level for the same
  reason the rate rows are split by prompt size. The pooled row deliberately never names a
  winner: a model you mostly ran on low effort against one you mostly ran on high is not a
  comparison, and a confident ratio sitting above per-effort rows that contradict it is exactly
  the failure the prompt-size buckets were added to prevent.
- Medians, not means, so one turn where you walked away does not become the expectation. Turns
  you interrupted are counted but never timed: that duration measures your patience.
- Durations come from Codex's own start and end stamps rather than llmwatch's clock, so a turn
  that was already running when you started llmwatch is still timed correctly. It attaches at
  the end of the session file and normally never sees a turn begin.
- Timings only, as everywhere else: `task_complete` carries the agent's final message, and it is
  dropped where the file is read. The new `turns` table has no column that could hold it.
- `--codex` now also works under `--plain` and `--json`, which previously ignored it.

## 0.7.0

**Press `c` to compare two models.** Pick them from a list with the arrow keys instead of
retyping model names into `--compare`.

- The picker shows models from your history *and* models installed but never measured. Today
  that is the common case: 4 models installed here, 1 with recorded requests. Picking an
  unmeasured one names the exact `ollama run` command to fix that, which beats a greyed-out row
  that explains nothing.
- The comparison shows generation and prefill side by side with bars normalised to the larger
  value, TTFT, cache rate, draft acceptance, a per-prompt-size breakdown with sample counts, and
  the time each model takes on your median request.
- That last line matters most: generation may be 1.43x faster while saving only 10 seconds on a
  real request, because reading the prompt dominates. Rates flatter; seconds do not.
- Differences under 5% read as "about the same" rather than a spurious 1.03x, and a bucket with
  fewer than 5 samples a side reports counts instead of a ratio.
- `--compare` on the command line renders through the same code, so the two cannot drift apart.

**Fixes found while building it**

- Arrow keys did not work: `sys.stdin.read(1)` buffers everything available, after which
  `select()` on the descriptor reports no data, so an arrow was indistinguishable from a bare
  Esc and pressing Down closed the menu. Input now reads the raw fd and decodes a buffer.
- Three top-level definitions were duplicated by earlier scripted edits, and for `_key_reader`
  the copies differed, so the stale byte-at-a-time reader silently shadowed the new one and
  arrow keys stayed broken while the fix sat unreachable with passing unit tests. A test now
  fails on any shadowed definition.
- A second guard fails on any function defined but never referenced. It immediately found
  `busiest_process`, written to name the process competing for the machine and never wired in;
  it now reports the culprit when a slowdown has no other explanation.

Tests: 238 -> 278.

## 0.6.0

### Security

**Untrusted text could carry terminal escape sequences.** Model names (from the log) and agent
tool-call arguments (from a Codex session file) were printed unsanitised. `\x1b[2J` clears the
screen, `\x1b]0;...\x07` rewrites the title, and some terminals honour worse.

The realistic path is not a hand-crafted model name: an LLM writes tool-call arguments, prompt
injection from any page or file the agent reads can shape them, Codex records them verbatim, and
llmwatch printed them. Both boundaries now sanitise on the way in, before styling.

Also removed the only `shell=True` in the codebase. The command was fixed so there was no
injection today, but it spawned a shell for a pipe that a list slice does just as well.

### Fixes

- **Progress no longer rewinds.** Reported as "47% then 46% then 47%". Position is projected
  from the last measured rate between ticks; when the real rate dipped, the next tick landed
  behind the projection. Displayed progress is now monotonic, and the projection cannot run more
  than one reporting interval ahead, so it does not race off and then stall.
- **The completion estimate no longer freezes.** Clamping remaining prefill at zero made the
  total a constant, so "answer ready" stopped moving and stopped being feedback. It now counts
  down every frame, and once overrun says `running past estimate - 2m00s so far` with a live
  clock.
- **Attaching no longer replays the whole Codex session.** The offset started at zero, so
  attaching to a long-running session parsed all of it (5.97 MB on the development machine),
  counted every historical tool call, and showed an hours-old action as current. It now starts at
  the end, like `tail -n 0`.

### Persistent history

Every completed request is recorded to a local SQLite file, so questions that previously needed a
hand-written benchmark are a command:

```
ollama-llmwatch --history --days 7
ollama-llmwatch --compare MODEL_A MODEL_B
ollama-llmwatch --export csv
```

Comparisons are bucketed by prompt size and cache state and report the sample count per side,
refusing to print a ratio below 5 samples each. An unbucketed comparison is how a "1.8x faster"
result turns out to have been two different workloads. Rates are token-weighted for the same
reason the live stats are.

The database stores timings and model names only; there is deliberately no column that could
hold prompt content. `--no-history` disables recording, and a database that cannot be opened
disables history rather than taking the UI down with it.

### Docs

No em dashes in any published text, with a test so it does not regress. Added an FAQ entry on
what the history file stores.

Tests: 187 -> 238.

## 0.5.3

**Fixes a display corruption bug**, reported from a real session:

```
2,537/14,517 tok +50,688 ca─red     53 tok/s8 cached     38 tok/s
! 5 cancels in acin a row - client keeps timing out
```

Frames were clipped to the terminal's height but never its width. A line longer than the
terminal wrapped, every row below shifted down, and the next cursor-home repaint landed on the
wrong rows - so two frames overlaid each other. Lines are now truncated to the visible width
(ANSI-aware, so escape sequences don't count toward it and colour can't bleed past the cut), and
the diagnosis line drops to a single finding on narrower terminals rather than relying on the
cut.

**Detects when Ollama isn't running.** Previously a stopped server looked identical to an idle
one - the log file still exists, so llmwatch waited forever with no hint that nothing could
happen. The idle line now distinguishes "not running", "no model loaded", and genuinely idle.

Found while implementing that: `SystemProbe` initialised its rate-limit timestamps to `0.0`, but
`time.monotonic()` can start near zero, so the *first* poll probed nothing and the warning was
delayed by 15 seconds - exactly when someone is staring at the screen wondering what's wrong.

**Loop warnings now quantify the cost.** `! same prompt 5x - likely stuck - interrupt ~11m00s
spent` rather than a vague "may be stuck". Escalates at 5 repeats, and stays silent about time
when there's no history to base it on.

**README:** the FAQ is now one heading per question instead of a wall of bold paragraphs.

Tests: 167 → 187.

## 0.5.2

First public release on PyPI: `uv tool install ollama-llmwatch`.

- **README rewritten for people who have never seen this before.** Install command is now in the
  first screen instead of halfway down; added a "reading the screen" table and an FAQ covering
  the questions that actually come up - does it need internet, does it read my prompts, why is my
  model slow, what's a good tok/s, why no CPU/GPU percentage, does it work with other clients.
- **Diagrams redrawn in ASCII.** The mermaid versions rendered on GitHub but appeared as raw code
  blocks on PyPI, where most people will first see this.
- Bumped `actions/checkout` and `actions/setup-python`, which were running on deprecated Node 20.

## 0.5.1 - unreleased

Renamed to **ollama-llmwatch** everywhere - PyPI, GitHub and the command - so the project has
one name instead of three. The bare `llmwatch` was already taken on PyPI by an unrelated
cost-tracking package, and having the repo, package and command disagree was going to confuse
anyone trying to find or install it.

- GitHub repo is now `bingcheng45/ollama-llmwatch` (GitHub redirects the old URL, so existing
  clones keep working).
- Installing provides two commands: `ollama-llmwatch` (canonical) and `llmwatch` (short alias).
  The alias is the only name that could collide with the other PyPI package, which is exactly
  why it is the alias and not the canonical name.

## 0.5.0 - unreleased

**Notices when the model is running slow, and says what's competing.**

Slowdowns are detected from llmwatch's own rate measurements against a rolling median baseline
(median, not mean, so one contended request doesn't move the bar it's measured against). Only
once something is genuinely slow does it consult cheap system signals to explain it:

```
! slow: 40 vs 100 tok/s usual - 2 models loaded (34 GB), swapping 3.0/4.0 GB
```

A `SYSTEM` line on the board shows contention when present, `clear` otherwise.

**No CPU or GPU percentage, on purpose.** On Apple Silicon inference is memory-bandwidth bound:
the GPU is pinned near 100% and the CPU near idle whether throughput is good or bad, so neither
number moves when performance does. GPU utilisation also requires sudo via `powermetrics`. What
does predict a slowdown - measured here - is a second loaded model (~28%), background apps
(~20%) or another process on the GPU (~44%), all of which surface as loaded models, swap,
memory pressure or load average. A test fails if CPU%/GPU% is ever added to that line.

Probes are rate-limited: cheap signals every 5s, `ollama ps` every 15s (it costs ~27ms, too
much for a 10fps loop), and the ~46ms process lookup only when a slowdown is already detected.

Tests: 143 → 167.

## 0.4.0 - unreleased

**Answers "should I keep waiting, or kill it?"** A one-line diagnosis under the progress bar,
built from log lines llmwatch previously ignored:

- `! cache gone - rereading all 39,528 tok (~6m35s)` - nothing was reused, you are paying full
  price. Usually the moment to interrupt.
- `! same prompt 4x - agent may be stuck in a loop` - identical prompt sizes in a row, which is
  what a client retry loop looks like from the server side and is otherwise invisible.
- `! 3 cancels in a row - client keeps timing out`
- `long chat: reading 45,000 tok this turn (~7m30s) - consider compacting`
- `cache working: only 244 of 41,253 tok to read`
- `drafts 22% accepted - MTP is slowing this one down`

At most two findings at a time, each under 70 characters: this line exists to be read in about
a second, not to dump telemetry.

**`answer ready ~7m10s`** - projected finish for the whole answer, not just for the moment text
starts appearing, using the session's measured generation rate and typical output length. Shown
only once there is history to base it on; it never invents a number.

**MTP draft acceptance**, live and on the board - `DRAFT 53% accepted`, with a verdict on
whether speculative decoding is earning its keep on this workload.

**`--codex`** (opt-in) shows what your agent is doing: last tool call, argument summary, tool
calls this turn, and how long it has been waiting on the model. Reads Codex's own session file,
which contains commands and file paths - hence opt-in. Codex-specific, and correlated by time
rather than request id, since no shared identifier exists.

New parsing: cache misses, context checkpoint create/restore/erase, and draft acceptance. A
state change now re-emits the live view immediately rather than waiting for the next 512-token
batch, so warnings appear at once.

**Fix:** the long-chat warning keyed on total context size, so a 41k conversation that was
fully cached - costing nothing - was warned about as "41,253 tok reread each turn (~6m52s)".
It now keys on tokens actually being read.

Tests: 84 → 120.

## 0.3.0 - unreleased

**Full-screen stats board.** Peak / average / low rates for both phases, cache hit rate, TTFT,
session totals, % of the session spent in prefill, recent-request list, and trend sparklines.

- Stats are kept **per model**. The MTP and base builds of the same model differ ~1.34x on
  code; pooling them produces an average that describes neither.
- Averages are **token-weighted** (total tokens / total seconds), not the mean of per-request
  rates - a mean gives a 4-token request the same weight as a 47,000-token one.
- Peak/low **ignore requests under 64 tokens**, so a cache hit computing 4 tokens can't become
  the session "low" forever.
- `--plain` keeps the v0.2.0 scrolling output, and is used automatically when stdout isn't a
  TTY or the terminal is under ~10 rows. `--json` is unchanged.
- The board is cleared on quit (alternate screen), so a plain-text session summary is printed
  afterwards. Terminal restore is wired to `finally`, `atexit`, SIGTERM **and** SIGINT - the
  last one restores at signal time, because waiting for exit can lose the sequence to pty
  buffering.

**More responsive.** The loop now blocks on new data *or* the frame deadline, whichever comes
first: new log data repaints immediately instead of waiting up to 100 ms, while animation still
runs at 10 fps with no data at all. Bursts are coalesced into one repaint, there's a 30 fps
ceiling, and the idle floor drops to 2 fps so it costs almost nothing when idle.

**Readability**

- The recent pane has **column headers** (task / prompt / total / prefill speed / share of
  wait) - four unlabelled numbers per row was a guessing game.
- Press **`h`** for an in-app help screen explaining every phase, column and number, including
  why avg is token-weighted and why TTFT is approximate. The live line stays visible while help
  is open. `q` quits alongside ctrl-c.
- A full README covering the problem in plain English, setup, flow diagrams of how requests are
  tracked, and an honest limitations list.

**Fixes**

- **Colour no longer breaks column alignment.** Labels were padded *after* being wrapped in
  ANSI escapes, which have zero screen width but do count in `%-12s`. A test now asserts that
  stripping colour from any rendered block yields exactly the uncoloured layout.
- **"96% - 3,584/39,528 tok"**: the percentage counted cached tokens while the token ratio
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

## 0.2.0 - unreleased

**The display no longer looks frozen.** llama-server logs prefill progress only once per
512-token batch - 5-10 seconds apart at typical rates - and the previous version repainted
only when a line arrived, so it sat visibly static in between and looked hung.

- Reading and rendering are now separate: a background thread consumes the log while the UI
  repaints 10x/second on its own clock.
- Position between ticks is projected from the last measured rate, clamped so the bar can
  never claim work that hasn't happened. Measured 1,594 rendered frames from ~29 log ticks.
- Elapsed counts up and ETA counts down continuously; a spinner proves liveness.
- The pre-first-tick state ("reading prompt, 6,637 tok") now carries a clock too - with a
  cached prompt, progress ticks may never arrive at all.

**Visual redesign.** Unicode block bars with ASCII fallback, colour (respecting `NO_COLOR`
and non-TTY output), a per-request header, and a tree-style summary ending in a bar showing
what share of the wall clock was prefill - the number the whole tool exists to surface.

- Idle state shows "waiting for a request" rather than a blank screen.
- Known cosmetic issue: when cache information arrives after projection has begun, the token
  total adjusts mid-flight and the numbers visibly snap once. Self-correcting.

## 0.1.0 - unreleased

First release.

- Live prefill progress with a bar, token counts, rate, and ETA - the phase no other
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
