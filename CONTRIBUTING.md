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

To try your change against a real model, run it from the checkout:

```bash
python3 llmwatch.py
```

## How this project is built

**One file, standard library only.** `llmwatch.py` is the entire program, and it is meant to stay
something you can read in an afternoon and `curl` onto a machine with nothing installed. A pull
request adding a dependency needs to argue for itself; usually there is a way to do it with what
Python already ships.

**Four layers, the first three pure.** This is the reason almost everything is testable without a
terminal or a model, and it is worth preserving:

```
parse_line(line)       -> Event        no I/O
Tracker.feed(event)    -> [Output]     state machine, no I/O
render_*(...)          -> lines        formatting only
follow()                               all the I/O lives here
```

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

For maintainers. Bump the version in **both** `llmwatch.py` and `pyproject.toml`, put that
version at the top of `CHANGELOG.md`, then:

```bash
git tag v0.9.1 && git push origin v0.9.1
```

The tag builds, tests and publishes to PyPI through Trusted Publishing. It refuses to publish if
the tag, `__version__` and `pyproject.toml` disagree: a version on PyPI can never be reused or
corrected, so the number has to be right before it goes out rather than after.
