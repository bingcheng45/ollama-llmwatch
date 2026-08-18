<!--
Nothing here is mandatory. Delete what does not apply - a one line typo fix does
not need a test plan, and pretending otherwise just adds noise.
-->

## What this changes

<!-- One or two sentences. -->

## What went wrong without it

<!--
The most useful part. A bug report with a real log line, a number that read
wrong on screen, or the thing you expected to see and did not.

If this is a new feature rather than a fix, say what was awkward or invisible
before instead.
-->

## Log lines, if the parser changed

<!--
Paste the real lines, with your Ollama version. Parser changes are guesses
without them, and the fixtures in tests/fixtures/ are the early-warning system
for the next format change.

    ollama --version
    ollama-llmwatch --debug-unparsed
-->

## Checklist

- [ ] `python3 -m unittest discover tests` passes
- [ ] A test covers the change, and would fail without it
- [ ] `README.md` and `CHANGELOG.md` updated if behaviour changed
- [ ] Sample boards in the README still match what the program prints
- [ ] No new dependency, or the PR explains why one is unavoidable

<!--
Reporting a security problem? Please close this and use
https://github.com/bingcheng45/ollama-llmwatch/security/advisories/new instead,
so a fix can go out before the details are public.
-->
