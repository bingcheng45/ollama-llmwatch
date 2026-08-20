# Contributing

Issues and pull requests are welcome - including "this number looks wrong". Several fixes so far
came from exactly that, and a wrong number is the worst bug this project can have: a monitor
nobody trusts is worse than no monitor.

## The most useful bug report

llmwatch parses internal log formats that carry no stability guarantee, so the reports that lead
to a fix nearly always include a **sample log line**.

```bash
ollama --version
ollama-llmwatch --debug-unparsed     # prints lines that looked like timings but did not parse
```

Include the Ollama version, your OS, and a few of those lines. There are issue templates for
[parser drift](.github/ISSUE_TEMPLATE/parser_drift.md),
[a wrong number](.github/ISSUE_TEMPLATE/wrong_number.md), and
[anything else](.github/ISSUE_TEMPLATE/feedback.md).

Security problems go through [SECURITY.md](SECURITY.md) instead, not a public issue.

## Getting set up

```bash
git clone https://github.com/bingcheng45/ollama-llmwatch
cd ollama-llmwatch
python3 -m unittest discover tests -v
```

That is the whole setup. Python 3.9 or newer, no dependencies, no virtualenv needed, no build
step. The tests need neither a network connection nor a running model.

To try your change against a real model, run the package from the checkout:

```bash
python3 -m llmwatch
```

Use `-m llmwatch`, not `python3 llmwatch.py`. The second one runs the generated single file,
which still holds whatever was there before your edit until you rebuild it (see below), so it is
the one way to test a change and watch it appear to do nothing.

## How this project is built

**A package you edit, a single file you ship.** The program lives in `llmwatch/`, one module per
layer. The `llmwatch.py` at the root is **generated**: `tools/bundle.py` concatenates the modules
into it, and that is what a `curl` install downloads. Edit the package, then run:

```bash
python3 tools/bundle.py
```

Do not edit `llmwatch.py` by hand. CI runs `tools/bundle.py --check` on every push, so an edit
made only there fails the build rather than quietly disappearing at the next release.

This arrangement exists because both properties are worth keeping. Splitting the file made it
navigable: a parser change is 300 lines of parser instead of one file in six thousand. Generating
the single file kept the promise that you can `curl` one script onto a machine with nothing
installed and run it. A pull request adding a dependency still needs to argue for itself; usually
there is a way to do it with what Python already ships.

**Four layers, the first three pure.** This is the reason almost everything is testable without a
terminal or a model, and it is worth preserving:

```
parse_line(line)       -> Event        no I/O            parser.py
Tracker.feed(event)    -> [Output]     state machine     tracker.py
render_*(...)          -> lines        formatting only   render.py and friends
follow()                               all the I/O       cli.py
```

**Imports point one way only.** The modules are ordered, roughly `constants` and `events` at the
bottom up to `cli` at the top, and an import may only point downwards. Nothing inside the package
imports the package root. This is not a style preference: the bundler works out its concatenation
order from the imports themselves, so a cycle has no valid order and stops the build. If you need
something from a layer above, the answer is almost always to move the shared definition down
rather than to reach up for it. `tests/test_bundle.py` will tell you which two modules are
involved.

If you find yourself wanting to read a file or check the clock inside a parser, that is usually a
sign the value should be passed in instead. Timestamps come from the log, not from
`time.time()`, so replaying an old log with `--last` reports what happened rather than how long
ago it happened.

**Fail soft, always.** A tool that watches something else must never be the thing that breaks. An
unrecognised line returns `None`. A locked history database disables history and carries on. A
failed update check says nothing. When in doubt, degrade rather than raise.

**Never invent a number.** If a rate cannot be measured, it is absent, not zero. A zero rate
lands in `low`, drags the average, and skews the median that slowdown detection compares against
- one unmeasurable request can recolour the whole board. The same goes for progress: show what
the log said, not what would look tidier.

## Tests

Every behaviour change needs a test, and the suite is the early-warning system for format drift
rather than a coverage exercise.

- **Fixtures are real output**, captured and sanitised, in `tests/fixtures/`. If you teach the
  parser a new line, add the line you actually saw, not one you wrote from memory.
- **Name the failure, not the function.** `test_prompt_eval_is_prefill_not_generation` says what
  goes wrong when it breaks; `test_parse_line_2` does not.
- **Docstrings explain why the test exists**, ideally naming the bug it caught. Several tests in
  here are gravestones for real defects, which is what stops someone "simplifying" them away.

```bash
python3 -m unittest discover tests -v          # everything
python3 -m unittest tests.test_mlx -v          # one module
```

## Style

Match the file you are editing. Beyond that:

- **120 column lines**, 4 space indent, no formatter to run.
- **Comments explain why, not what.** The code says what it does. A comment earns its place by
  recording a decision, a trade-off, or a bug that came back. If a comment only makes sense to
  someone who lived the debugging session, it needs rewriting or deleting.
- **No em dashes or en dashes in documentation.** A test enforces this, because they render
  inconsistently across terminals and PyPI.
- **Prose in the interface too.** Error messages and screen text are written for someone deciding
  whether to keep waiting, not for someone reading a stack trace.

## Pull requests

- One change per pull request. "While I was in there" belongs in its own.
- Say what the change does and, more usefully, **what went wrong without it**.
- CI runs the tests on Python 3.9 to 3.13 across Linux and macOS, plus Bandit, CodeQL and a
  dependency audit. All of it should be green before review.
- Update `README.md` and `CHANGELOG.md` when behaviour changes. The README is also the PyPI
  landing page, so a screenshot or sample board that no longer matches the program is a bug in
  its own right.

## Releasing

For maintainers. Bump the version in **both** `llmwatch/constants.py` and `pyproject.toml`, run
`python3 tools/bundle.py` so the generated file carries the new number, put that version at the
top of `CHANGELOG.md`, then:

```bash
git tag v0.9.1 && git push origin v0.9.1
```

The tag builds, tests and publishes to PyPI through Trusted Publishing. It refuses to publish if
the tag, `__version__` and `pyproject.toml` disagree: a version on PyPI can never be reused or
corrected, so the number has to be right before it goes out rather than after.
