"""Working out what is running, when the user does not know.

Scans a short list of log directories and loopback ports for something that
looks like a local model server, so "I do not know, find it for me" is a real
answer in the settings pane.
"""
import os
import time

from .constants import OAI_PROBE_TIMEOUT, SETTINGS_PRESETS
from .text import fmt_duration, safe_text
from .parser import RE_MLX_RUNNER_START, parse_line, parse_mlx_line
from .logs import candidate_paths
from .oai import _fetch_openai_models, detect_engine, engine_from_log_event


# Where a log plausibly is, when someone does not know. Bounded on every axis:
# a few directories, only *.log, only files touched recently, and only the tail
# of each is read. This runs because somebody asked it to, so it may take a
# moment, but it may not go wandering.
# nosec B108 -- these are read, never written. bandit flags /tmp because a
# world-writable directory is the wrong place to *create* a file; nothing here
# creates one. It is however the right place to look, because it is where
# people redirect a server's output. What that world-writability does mean is
# that the names found here are attacker-controlled, so everything discovered
# goes through safe_text before it can reach a terminal.
LOG_SEARCH_DIRS = ("/tmp", "~/Library/Logs", "/var/log",  # nosec B108
                   "/opt/homebrew/var/log")

LOG_SEARCH_MAX_AGE = 24 * 3600.0

LOG_SEARCH_TAIL = 64 * 1024

LOG_SEARCH_FILES = 60

# Wider than the idle-screen probe: that one runs unattended and often, this
# one runs once because it was asked to.
DISCOVER_PORTS = (8080, 1234, 8000, 8081, 5000, 11434)

# Ollama answers the OpenAI API too, but proxying it would be the wrong answer:
# it writes a log with more in it than the wire carries.
OLLAMA_PORT = 11434


def identify_log(path, tail=LOG_SEARCH_TAIL):
    """Which engine wrote this file, by reading it, or None.

    Read rather than guessed from the name, because the name is whatever the
    person redirecting stderr felt like typing. The parsers already know what
    each engine's lines look like, so a line that parses is the identification.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > tail:
                fh.seek(size - tail)
            blob = fh.read(tail)
    except (OSError, ValueError):
        return None
    try:
        text = blob.decode("utf-8", "replace")
    except (UnicodeError, AttributeError):
        return None

    saw_mlx = False
    for line in text.splitlines()[-400:]:
        line = line + "\n"
        if RE_MLX_RUNNER_START.search(line) or parse_mlx_line(line) is not None:
            saw_mlx = True
        ev = parse_line(line)
        if ev is None:
            continue
        engine = engine_from_log_event(ev)
        if engine == "llama.cpp":
            return "llama.cpp"
        if engine == "MLX":
            saw_mlx = True
    return "MLX" if saw_mlx else None


def _discover_logs(dirs=LOG_SEARCH_DIRS, now=None, known=None):
    """`known` is the list of paths to check regardless of `dirs`, which is
    where Ollama's own log lives. Passed in rather than always consulted, so
    restricting `dirs` actually restricts the search: it did not, which made
    this depend on whether the machine running it happened to have a fresh
    Ollama log.
    """
    now = time.time() if now is None else now
    found, seen = [], set()
    if known is None:
        known = candidate_paths()
    paths = [os.path.expanduser(p) for p in known]
    for directory in dirs:
        directory = os.path.expanduser(directory)
        try:
            names = sorted(os.listdir(directory))[:LOG_SEARCH_FILES]
        except OSError:
            continue
        paths.extend(os.path.join(directory, n) for n in names
                     if n.endswith(".log"))
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            age = now - os.path.getmtime(path)
        except OSError:
            continue
        if age > LOG_SEARCH_MAX_AGE:
            continue
        engine = identify_log(path)
        if engine:
            found.append({"kind": "log", "engine": engine, "path": path,
                          "age": age,
                          "apply": {"watch": "log", "log": path}})
    found.sort(key=lambda row: row["age"])
    return found


def _discover_servers(ports=DISCOVER_PORTS, timeout=None, skip_ports=()):
    # Resolved here rather than as a default, so this can be defined
    # beside the rest of the settings code instead of beside the probe.
    timeout = OAI_PROBE_TIMEOUT if timeout is None else timeout
    found = []
    for port in ports:
        # Our own proxy answers, and relays the upstream's Server header while
        # it is at it, so it looks exactly like whatever is behind it. Offering
        # it would point llmwatch at itself.
        if port in skip_ports:
            continue
        url = "http://127.0.0.1:%d" % port
        obj, headers = _fetch_openai_models(url, timeout, with_headers=True)
        if obj is None or not ("data" in obj or obj.get("object") == "list"):
            continue
        models = [row.get("id") for row in (obj.get("data") or [])
                  if isinstance(row, dict) and isinstance(row.get("id"), str)]
        if port == OLLAMA_PORT:
            found.append({"kind": "ollama", "engine": "Ollama", "port": port,
                          "models": models, "apply": {"watch": "ollama"}})
            continue
        found.append({"kind": "server", "engine": detect_engine(headers, obj),
                      "port": port, "models": models,
                      "apply": {"watch": "proxy", "upstream": url}})
    return found


def discover_backends(dirs=LOG_SEARCH_DIRS, ports=DISCOVER_PORTS, now=None,
                      skip_ports=(), known=None):
    """Everything locally that looks watchable, best first.

    Logs come before servers for the same engine, because a log carries the
    server's own rates and its prefill progress where the wire carries neither.
    """
    logs = _discover_logs(dirs, now=now, known=known)
    servers = _discover_servers(ports, skip_ports=skip_ports)
    logged_engines = {row["engine"] for row in logs}
    ranked = list(logs)
    for row in servers:
        # A llama.cpp server whose log was already found is the same thing
        # twice, and the log is the better half.
        if row["engine"] in logged_engines:
            continue
        ranked.append(row)
    return ranked


def describe_backend(row):
    """One line for the pane, in the words the rest of the pane uses."""
    # Everything interpolated here came from outside: a filename chosen by
    # whoever could write to the directory, or a model id from a server's
    # response. A name is allowed to contain an escape sequence, and one of
    # these directories is world-writable.
    if row["kind"] == "log":
        return "%s log   %s   (%s old)" % (
            row["engine"], safe_text(row["path"], limit=120),
            fmt_duration(row["age"]))
    if row["kind"] == "ollama":
        return "Ollama   %d model%s installed" % (
            len(row.get("models") or []),
            "" if len(row.get("models") or []) == 1 else "s")
    models = row.get("models") or []
    what = (safe_text(models[0], limit=80) if len(models) == 1
            else "%d models" % len(models))
    engine = (row.get("engine") or "OpenAI") + " server"
    return "%s on :%d   %s" % (engine, row["port"], what or "no models")


def backend_suggestion(mode, system, busy):
    """A better backend than the one running, or None.

    Choosing the backend is the one setting that cannot be got wrong loudly:
    every wrong choice looks the same, an empty board, and nothing on it says
    the mistake was in the choice rather than in the model.

    Never acted on automatically. Switching underneath someone would be worse
    than the empty board, because the numbers would quietly start meaning
    something else and the screen would not admit it. Suggested, and taken by
    choosing it like any other option.

    `busy` is the veto. Traffic is arriving through the current backend, so it
    is the right one whatever else happens to be running.
    """
    if busy or not system:
        return None
    port = system.get("oai_port")
    loaded = system.get("models_loaded") or 0

    if mode == "ollama" and not loaded and port:
        return {"mode": "proxy",
                "why": "an OpenAI server is answering on :%d and Ollama has "
                       "nothing loaded" % port}
    if mode == "proxy" and system.get("upstream_ok") is False and loaded:
        return {"mode": "ollama",
                "why": "the server being proxied is unreachable and Ollama has "
                       "%d model%s loaded" % (loaded, "" if loaded == 1 else "s")}
    if mode == "log" and port:
        return {"mode": "proxy",
                "why": "nothing is arriving in the log, and an OpenAI server "
                       "is answering on :%d" % port}
    return None


def preset_for(mode, upstream):
    """Which preset a set of settings looks like, or None if it matches none.

    So the pane can open with the current choice marked rather than making
    someone work out which line describes what is already running.
    """
    for key, _title, _why, values in SETTINGS_PRESETS:
        if values.get("watch") != mode:
            continue
        if "upstream" in values and values["upstream"] != upstream:
            continue
        return key
    return None
