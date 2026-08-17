---
name: "A number looks wrong"
about: Percentages, rates, ETAs or totals that don't add up. These reports are genuinely useful.
labels: correctness
---

**What llmwatch showed**
Paste the line or board. For example, a real past bug:
`PREFILL [###############-] 96% 3,584/39,528 tok` — the percentage counted cached tokens while
the ratio didn't.

**What you expected instead**
e.g. "96% and 3,584/39,528 can't both be right"

**Was a cached prompt involved?**
If the line mentions `+N cached`, say so — caching is behind most of these.

**Ollama version and OS**
```
ollama --version
```

**Anything from `--last`**
```
llmwatch --last
```

Numbers that disagree with each other are always a bug, even when each one is individually
correct. Don't worry about diagnosing it — just show what you saw.
