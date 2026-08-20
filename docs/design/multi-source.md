# Watching more than one engine at once

Status: proposed, nothing implemented.

Line references below were checked against `c157a59`. They rot; the function
names beside them do not, so search for those if a number has drifted.

## What is being asked for

Run Ollama, a proxied MLX server and a llama.cpp log at the same time, and see
all three on one board, instead of choosing one at startup.

## Why it does not work today

Three things are single-source, and the first of them is not a gap but an
active hazard.

**Requests are keyed `(slot, task)`.** `Tracker._key` at `llmwatch.py:561`.
There is no source in the key, so two backends numbering their slots from zero
collide. This is not hypothetical: `MLX_SLOT = 0` and `OAI_SLOT = 1` are
reserved slot numbers standing in for a source, and llama.cpp hands out slots
from 0 upward, so `OAI_SLOT` and llama.cpp slot 1 are the same key.

Demonstrated against the current code. A proxied request and a llama.cpp
request, nothing to do with each other, land on one key and one overwrites the
other:

```
OAI_SLOT = 1
after a proxied request starts : {(1, 1)}
llama.cpp event               : RequestStart slot 1 task 1
after the llama.cpp request   : {(1, 1)}      <- still one entry
```

The same defect is what made `--proxy --log` record every request twice, one
request producing two `request_end` events with different durations.

**Stats are keyed by model name.** `SessionStats._model` at `llmwatch.py:1145`.
The same model served by two engines averages into one set of rates that
describes neither. The class docstring already argues this point for the MTP
and base builds of one model; two engines is the same problem, larger.

**One log is tailed.** A single `subprocess.Popen(tail_command(kind, target))`
at `llmwatch.py:5535`.

The failure mode throughout is that wrong answers look plausible. Nothing
errors, the board fills, and the numbers are averages of things that should
never have been averaged.

## Phases

Each phase is separately shippable and separately revertible.

### 1. Make the source explicit

Replace reserved-slot-as-source with a real source tag.

- `_key(ev)` becomes `(source, slot, task)`.
- `MLX_SLOT` / `OAI_SLOT` stop carrying meaning they were never suited to.
- Nine call sites, all listed by `grep -n "self.requests\[\|self.requests.pop\|self.requests.get\|_key(ev)"`.

No behaviour change and no new capability: still one source at a time. The
point is to remove the collision before anything is built on top of it.

Done when: the existing suite passes untouched, plus a test that a llama.cpp
slot 1 request and a proxied request no longer share a key. That test fails
today.

### 2. Scope stats by source

- `by_model` keyed by `(source, model)`.
- Everything reading a model name for display or history has to say which
  engine it meant.
- `history.requests` gains `engine TEXT`, nullable, with existing rows left
  NULL rather than guessed at. Migration must be additive: the file predates
  this and is the user's own data.

Done when: the same model on two engines produces two rows in `--history`, and
old history still opens and reads correctly.

### 3. Run several sources

- `follow()` supervises N tails plus optionally the proxy, each tagging its
  events.
- Failure of one source must not take the others down: a log that disappears is
  one source going quiet, not an exit.
- The settings pane moves from picking one mode to enabling several.

Done when: Ollama and a llama.cpp log feed one board simultaneously, each
request attributed to the right engine.

### 4. Show it

- `render_recent` (`llmwatch.py:1905`) gains an engine column. Columns today are
  task, prompt, total, prefill speed, share of wait.
- The board either sections by source or labels every row. Sectioning reads
  better with two, labelling scales further.
- `--compare` becomes more useful than it is now, since comparing the same model
  across two engines is the thing people actually want and cannot currently do
  without two runs.

## What could go wrong

**Silent merging.** Every defect in this area so far has produced believable
numbers rather than an error, and this change multiplies the places two things
can quietly become one. Every phase needs a test that two sources stay apart,
not only that one source still works.

**History migration.** Existing files are real user data and cannot be
rewritten on a hunch. Additive column, NULL for anything recorded before the
engine was known, and no backfill.

**Cost.** Several tails plus a proxy is several more file descriptors and
threads. It should stay opt-in rather than becoming what everyone pays for.

**Scope.** Phase 1 alone is worth shipping: it removes a live collision. If the
rest is never built, nothing is left half-done.

## Not in scope

Switching backends live, without a restart. That means tearing down a log tail
or a bound socket underneath a running tracker, and it is a separate piece of
work with its own failure modes.
