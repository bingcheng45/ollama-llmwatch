# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub:
[**Report a vulnerability**](https://github.com/bingcheng45/ollama-llmwatch/security/advisories/new).
That opens a private advisory only you and the maintainer can see.

You should get a first response within **7 days**. If a report is confirmed, the fix ships in the
next release and the advisory is published with credit, unless you would rather not be named.

If you do not hear back in 7 days, please open a public issue saying only that you are waiting on
a security report - no details - so the silence itself is visible.

## What is worth reporting

llmwatch reads logs written by other programs and prints them to your terminal, so the
interesting attacks come through that data rather than through a network service. Reports that
are especially welcome:

- **Terminal escape injection.** Anything reaching the screen from a log, a model name, a Codex
  session file or a command line without passing through `safe_text`. A crafted model name that
  can move the cursor, rewrite earlier output, or set the window title is a real finding.
- **Command injection** through a log path, a model name, or anything else that reaches the
  `tail`/`journalctl` invocation.
- **Path traversal** in log discovery, the history database, or the update cache.
- **Anything that makes llmwatch write outside** `$XDG_DATA_HOME/ollama-llmwatch`.
- **Data leaving the machine** other than the single documented update check, or that check
  carrying anything about you.
- **A crash reachable from log content.** A monitor that dies on malformed input is a bug; one
  that dies on *attacker-chosen* input beside your agent is a security bug.

## What is out of scope

- **Findings in Ollama, llama.cpp, MLX or your models.** Report those upstream.
- **Bandit low-severity findings**, which CI reports but does not fail on. `subprocess` is called
  with an argument list and never a shell string, and the bare `except` clauses are the
  documented fail-soft contract. If you can show one of them is actually exploitable, that is a
  finding and very much in scope - the classification is the thing being disputed, not the code.
- **The update check existing.** It is documented, it sends nothing about you, and
  `LLMWATCH_NO_UPDATE_CHECK=1` turns it off. A demonstration that it sends more than documented,
  or can be pointed somewhere else, is in scope.
- **Reading `~/.codex/sessions`**, which is what `--codex` is for and why it is opt-in.

## Supported versions

Only the latest release on PyPI is supported. This is a single file with no dependencies, so
upgrading is the fix for everything.

| Version | Supported |
|---|---|
| latest release | yes |
| anything older | no - please upgrade first and confirm it still reproduces |

## What llmwatch touches

Useful context when judging whether something is a real finding:

- **Reads:** the Ollama log (from `LLMWATCH_LOG`, or the usual locations), and with `--codex`,
  the newest file in `~/.codex/sessions`.
- **Writes:** `$XDG_DATA_HOME/ollama-llmwatch/history.db` (timings only, never prompt text - a
  test asserts the schema has no column that could hold it) and `update-check.json`.
- **Runs:** `tail -F`, or `journalctl` on Linux, always as an argument list, never through a
  shell. A test asserts no `shell=True` anywhere in the file.
- **Network:** one GET to `https://pypi.org/pypi/ollama-llmwatch/json`, at most daily. Nothing
  else, enforced by tests that allow exactly one network import in one function.
