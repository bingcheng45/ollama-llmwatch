---
name: "Feedback, idea, or platform report"
about: Feature ideas, confusing output, or "it works / doesn't work on my setup"
labels: feedback
---

**What would you like?**
An idea, something that confused you, or a report that llmwatch does (or doesn't) work on your
platform.

**Your setup**
- OS:
- Ollama install (Homebrew / official app / Docker / Linux package):
- Model:
- Which agent you use it beside (Codex, Claude Code, open-webui, plain `ollama run`, ...):

**Especially wanted**

- **Linux (journald) and Docker reports.** Those code paths are written but unverified - a
  simple "it worked" or the error you hit is valuable either way.
- **Confusing output.** If a line made you stop and think, that's a design bug worth fixing.

Known-out-of-scope: showing *what* your agent is doing (file names, tool calls). Ollama's log
contains timings only - no prompt content - so llmwatch cannot see it. Happy to discuss
alternatives in an issue if you need that.
