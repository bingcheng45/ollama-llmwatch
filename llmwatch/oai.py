"""Talking to an OpenAI-compatible server, and working out what it is.

Extracted from config and proxy, which each used to reach into the other for
these. That was the one genuine import cycle in the single-file version.

The /v1/models fetch is one of only three places in llmwatch that opens a
socket, and it is bounded on every axis: loopback or the configured upstream,
a 0.3s timeout, and only while there is nothing else to show.
"""
import json

from .constants import (
    DEFAULT_PROXY_PORT, OAI_PROBE_PORTS, OAI_PROBE_TIMEOUT, UPSTREAM_SCHEMES)
from .events import MLX_EVENTS, OAI_EVENTS


def _fetch_openai_models(url, timeout, with_headers=False):
    """The /v1/models payload from `url`, or None if it could not be had.

    None means "could not ask", which the callers keep distinct from "asked,
    and it serves nothing".
    """
    import urllib.error
    import urllib.request

    try:
        # nosec B310 -- callers pass a validated http(s) upstream or a literal
        # loopback URL built from a constant port.
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models",  # nosec B310
                                    timeout=timeout) as resp:
            body = resp.read(65536)
            headers = dict(resp.headers)
    except (urllib.error.URLError, OSError, ValueError):
        return (None, {}) if with_headers else None
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return (None, {}) if with_headers else None
    obj = obj if isinstance(obj, dict) else None
    return (obj, headers) if with_headers else obj


def list_upstream_models(url, timeout=OAI_PROBE_TIMEOUT):
    """Model ids the upstream says it serves, or None if it could not be asked.

    Shown on the idle screen because the failure it explains is silent: a
    client configured with a model id the server does not serve gets nothing
    back, and nothing anywhere says why. Swapping one local server for another
    is enough to cause it, since the id is usually the file path or repo name
    and changes with the server.
    """
    obj = _fetch_openai_models(url, timeout)
    if obj is None:
        return None
    data = obj.get("data")
    if not isinstance(data, list):
        return None
    ids = [row.get("id") for row in data
           if isinstance(row, dict) and isinstance(row.get("id"), str)]
    return ids


def engine_from_log_event(ev):
    """Which engine wrote this log event, or None if it did not come from a log.

    Read from the event type rather than detected, because the two runners
    write different logs and are parsed by different functions: whichever one
    produced the event already knew.

    Only for events that arrived from a log. The Tracker re-feeds some of these
    to itself while translating an MLX request, and reading the type inside
    feed() would flip MLX to llama.cpp halfway through its own request. Proxied
    events return None: that path detects its own engine from the wire, and a
    guess here must not overwrite it.
    """
    if isinstance(ev, MLX_EVENTS):
        return "MLX"
    if isinstance(ev, OAI_EVENTS):
        return None
    return "llama.cpp"


def model_name_for_board(tracked, running):
    """The model name to show, filling in from `ollama ps` when the log did not
    say.

    The log names the model once, on the line that selects it. llmwatch started
    against an already-loaded model never sees that line and shows `?` until the
    next model load, which can be hours.

    Only when exactly one model is resident. With two there is no way to tell
    which served the request, and naming the wrong one is worse than naming
    none, because every rate on the board is scoped by model. A name that came
    from the log always wins: it saw the model actually serve something, where
    `ollama ps` only knows what is loaded.
    """
    if tracked and tracked != "?":
        return tracked
    if running and len(running) == 1:
        return running[0]
    return tracked or "?"


def detect_engine(headers, obj):
    """Which engine served this response, or None if it did not say.

    The board shows a model name, and a model name is not an engine: the same
    `qwen3.8-27b-4bit` is served by MLX on one machine and llama.cpp on another,
    and telling those apart is usually the reason for looking. So this reads
    what the server states about itself and nothing else. Never the model name.

    Unrecognised means unlabelled. A wrong engine label is worse than none,
    because the label exists precisely to be compared against.

    The tempting non-signal is `Server: BaseHTTP/0.6 Python/x.y`, which
    mlx_lm.server does send -- as stdlib's default, along with every other
    Python HTTP server including llmwatch's own proxy. Matching it would label
    everything MLX.
    """
    headers = headers or {}
    try:
        server = str(headers.get("Server") or headers.get("server") or "")
    except AttributeError:
        server = ""
    # llama.cpp names itself outright (tools/server/server-http.cpp).
    if "llama.cpp" in server.lower():
        return "llama.cpp"
    if isinstance(obj, dict):
        # `timings` is llama.cpp's own extension to the OpenAI response, so it
        # identifies the engine even behind something that rewrote Server.
        if isinstance(obj.get("timings"), dict):
            return "llama.cpp"
        # mlx_lm builds system_fingerprint as
        # {mlx_lm_version}-{mlx_version}-{platform}-{gpu_arch}, and gpu_arch
        # comes from mx.device_info(), which is applegpu_* wherever MLX runs.
        fingerprint = obj.get("system_fingerprint")
        if isinstance(fingerprint, str) and "applegpu" in fingerprint:
            return "MLX"
    return None


def detect_openai_server(ports=OAI_PROBE_PORTS, timeout=OAI_PROBE_TIMEOUT):
    """The first loopback port answering /v1/models like an OpenAI server.

    Exists for one message. mlx_lm.server on :8080 with llmwatch started on the
    Ollama backend shows `no model loaded`, which is true and useless: there is
    no Ollama model, and the thing the user is watching is not visible from
    that backend at all. The numbers for an OpenAI server are not in its log to
    be found -- mlx_lm.server logs prompt progress and nothing else -- so
    --proxy cannot be inferred away, only suggested.

    A listener is not enough to suggest it. The response has to look like the
    endpoint, because pointing --proxy at some unrelated local service would
    route the user's traffic through it.
    """
    for port in ports:
        obj = _fetch_openai_models("http://127.0.0.1:%d" % port, timeout)
        if obj is not None and ("data" in obj or obj.get("object") == "list"):
            return port
    return None


def valid_upstream(url):
    """Is this something the proxy may forward to?

    Only http and https. A `file:` upstream would turn every request the client
    makes into a local file read, and the reply would be relayed back to it.
    """
    return bool(url) and str(url).startswith(UPSTREAM_SCHEMES)


def parse_listen(text, default_port=DEFAULT_PROXY_PORT):
    """'8081' / '127.0.0.1:8081' / ':8081' / '[::1]:8081' -> (host, port)."""
    text = (text or "").strip()
    if not text:
        return ("127.0.0.1", default_port)

    # An IPv6 address is mostly colons, so rpartition cannot be let near one
    # unbracketed: '::1' would split into the host ':' on port 1, which then
    # fails the loopback check and reads as a policy refusal rather than a
    # parsing bug. Bracketed is the form that carries a port; bare is all host.
    if text.startswith("["):
        host, sep, rest = text[1:].partition("]")
        if not sep:
            return (text, default_port)
        port = rest[1:] if rest.startswith(":") else ""
        try:
            return (host, int(port))
        except ValueError:
            return (host, default_port)
    if text.count(":") > 1:
        return (text, default_port)

    if ":" in text:
        host, _, port = text.rpartition(":")
        host = host or "127.0.0.1"
    else:
        # A bare number is a port; a bare word is a host.
        if text.isdigit():
            return ("127.0.0.1", int(text))
        return (text, default_port)
    try:
        return (host, int(port))
    except ValueError:
        return (host, default_port)
