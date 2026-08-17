---
name: "llmwatch shows nothing / stopped working after an Ollama upgrade"
about: The most likely failure. llmwatch parses an internal log format that can change.
labels: parser
---

**What you see**
e.g. "the board stays empty while a request is clearly running"

**Ollama version**
```
ollama --version
```

**OS / install method**
e.g. macOS 15, Ollama via Homebrew / Linux, Docker

**Unparsed lines** - this is the important part:
```
llmwatch --debug-unparsed
```
Paste a few `UNPARSED:` lines here. They tell us exactly how the format changed.

**A log sample (optional but ideal)**
A handful of lines covering one request, from `new prompt` through `total time`. Please remove
any file paths or usernames you'd rather not share - llmwatch never reads prompt content, so
the timing lines themselves contain no private data.
