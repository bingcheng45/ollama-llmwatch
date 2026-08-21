"""Layer 4b: the pass-through proxy.

Sitting between an agent and an OpenAI-compatible server is the only way to see
timings for a backend that is not Ollama. It forwards everything unchanged and
reads the token counts out of the stream on the way past.

This is the only part of llmwatch that has to own an HTTP stack, and the import
stays inside _proxy_server_class so that watching a local model never pays for
it.
"""
import json
import threading
import time

from .events import (
    OaiEngine, OaiGenTick, OaiRequestAborted, OaiRequestEnd, OaiRequestStart)
from .text import safe_text, short_model_name
from .oai import detect_engine, valid_upstream

INSTRUMENTED_PATHS = ("/v1/chat/completions", "/v1/completions")

# Connection-level headers describe one hop and must not be copied onto the
# next: forwarding an upstream `Transfer-Encoding: chunked` while writing the
# decoded body back is how a proxy corrupts a response.
HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
])

# A prefill on a long prompt is minutes of silence before the first byte, so
# this cannot be tight. It exists only so a vanished upstream cannot pin a
# thread forever.
PROXY_TIMEOUT = 900.0

# Cap on a non-streaming body held in memory to read its usage block. Past this
# the response is relayed unmeasured rather than buffered.
MAX_BUFFERED_BODY = 32 * 1024 * 1024


def _sse_events(buffer, chunk):
    """Split streamed bytes into complete `data:` payloads.

    Returns (leftover, payloads). Incremental by construction: a chunk that
    ends mid-line leaves the remainder in `leftover` for the next call, so
    nothing waits on the whole response.
    """
    buffer += chunk
    payloads = []
    while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        line = line.strip()
        if line.startswith(b"data:"):
            payloads.append(line[5:].strip())
    return buffer, payloads


def draft_counts(timings):
    """(accepted, generated) drafted tokens from a `timings` block, or None.

    llama-server reports speculative decoding here; every other OpenAI server
    omits the block entirely. Ordered accepted-first to match what the log
    backend already hands DraftAcceptance, so there is one convention rather
    than one per source.

    None means "no measurement", which is not the same as zero: a server with
    no drafter, a run that drafted nothing, and a garbled block all have no
    acceptance rate to show, and printing 0% for them would read as a drafter
    performing terribly rather than as a drafter that was never there.
    """
    if not isinstance(timings, dict):
        return None
    if "draft_n" not in timings or "draft_n_accepted" not in timings:
        return None
    try:
        generated = int(timings["draft_n"])
        accepted = int(timings["draft_n_accepted"])
    except (TypeError, ValueError):
        return None
    if generated <= 0 or accepted < 0:
        return None
    # Above 1.0 would print as "140% accepted". Whatever produced it, it is not
    # a measurement.
    if accepted > generated:
        return None
    return (accepted, generated)


def read_stream_usage(payload):
    """Pull what is measurable out of one SSE payload.

    Returns (produced_content, usage_or_None, timings_or_None). Deliberately
    narrow: the delta is tested for emptiness and thrown away without being
    stored or returned, so no caller can accidentally end up holding message
    text.
    """
    if not payload or payload == b"[DONE]":
        return False, None, None
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return False, None, None
    if not isinstance(obj, dict):
        return False, None, None
    produced = False
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        # `text` is the /v1/completions spelling of content. tool_calls count
        # too, and must: a coding agent's turn is frequently nothing but a tool
        # call, and treating those as producing no output leaves the busiest
        # half of an agent session invisible.
        produced = (bool(delta.get("content")) or bool(first.get("text"))
                    or bool(delta.get("tool_calls")))
    usage = obj.get("usage")
    timings = obj.get("timings")
    return (produced, usage if isinstance(usage, dict) else None,
            timings if isinstance(timings, dict) else None)


def read_stream_fingerprint(payload):
    """Just `system_fingerprint`, for engine detection.

    Deliberately not folded into read_stream_usage. That function is the one
    with a test asserting no message content can escape it, and the way to keep
    that true is to not give it more to return.
    """
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    fingerprint = obj.get("system_fingerprint")
    return fingerprint if isinstance(fingerprint, str) else None


def usage_counts(usage):
    """(prompt, cached, completion) from a usage block, missing fields as 0."""
    if not isinstance(usage, dict):
        return (0, 0, 0)
    details = usage.get("prompt_tokens_details")
    cached = 0
    if isinstance(details, dict):
        try:
            cached = int(details.get("cached_tokens") or 0)
        except (TypeError, ValueError):
            cached = 0
    def _int(value):
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
    return (_int(usage.get("prompt_tokens")), cached,
            _int(usage.get("completion_tokens")))


def prepare_body(raw):
    """(body_to_forward, model, is_stream) for an instrumented request.

    Streaming responses only carry a usage block when the client asked for one,
    so llmwatch asks on the client's behalf. mlx_lm.server reads
    stream_options["include_usage"] with a bare subscript, so a client that
    sends stream_options without that key crashes the upstream -- normalising
    the dict here fixes that for free rather than passing the crash through.

    On anything unparseable the body is returned untouched: a request that
    cannot be measured must still be a request that works.
    """
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return raw, None, False
    except (ValueError, TypeError):
        return raw, None, False

    model = short_model_name(obj.get("model"))
    is_stream = bool(obj.get("stream"))
    if not is_stream:
        return raw, model, False

    options = obj.get("stream_options")
    if not isinstance(options, dict):
        options = {}
    if options.get("include_usage") is True:
        return raw, model, True
    options = dict(options)
    options["include_usage"] = True
    obj = dict(obj)
    obj["stream_options"] = options
    try:
        return json.dumps(obj).encode("utf-8"), model, True
    except (TypeError, ValueError):
        return raw, model, True


_PROXY_SERVER_CLASS = None


def _proxy_server_class():
    """Build the proxy's server class, importing an HTTP stack only now.

    llmwatch watches a local model and deliberately does not load an HTTP stack
    to start up -- see TestNoNetworkDependency, and the update check, which
    imports urllib inside the one function that needs it for the same reason.
    A proxy cannot avoid the dependency, but it can avoid paying for it in
    every run that never proxies anything, so the classes are built here, on
    first use, rather than at import.
    """
    global _PROXY_SERVER_CLASS
    if _PROXY_SERVER_CLASS is not None:
        return _PROXY_SERVER_CLASS

    import http.server
    import urllib.error
    import urllib.parse
    import urllib.request

    class _NoRedirects(urllib.request.HTTPRedirectHandler):
        """Relay a 3xx rather than following it.

        urlopen follows redirects by default, and carries the request headers
        across hosts while it does. An upstream answering 302 with a Location
        elsewhere would therefore take the Authorization header there -- the
        exact outcome safe_request_path and the origin check exist to prevent,
        reached from the other side, and past both of them because the second
        request is made inside urlopen where neither can see it.

        Refusing to follow is also just what a proxy should do. The client
        asked for this URL; if it moved, that is an answer, and the client is
        entitled to see it and decide.
        """

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    # Returning None above leaves the 3xx unhandled, so it surfaces as an
    # HTTPError and takes the same path as any other upstream error response:
    # relayed verbatim, Location header and all.
    _opener = urllib.request.build_opener(_NoRedirects)

    class _ProxyHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            """Silence. The base class writes to stderr, which is the terminal the
            TUI is drawing on."""

        # All verbs go through one path; only POSTs to a completions endpoint are
        # measured, everything else is a plain relay so `GET /v1/models` and any
        # endpoint added upstream keep working without llmwatch knowing about them.
        def do_GET(self):
            self._relay()

        def do_POST(self):
            self._relay()

        def do_DELETE(self):
            self._relay()

        def do_OPTIONS(self):
            self._relay()

        def _note_engine(self, headers, obj):
            """Announce the engine, and again if it ever changes.

            Not latched. Stopping MLX and starting llama.cpp on the same port
            is routine, and a label that was learned once would then name the
            server that had been replaced, which is worse than naming none
            because it is confidently wrong.

            Cheap enough to repeat: the header check is a dict lookup, and the
            body is only looked at once per request by the caller.
            """
            found = detect_engine(headers, obj)
            if found and found != self.server.engine:
                self.server.engine = found
                self._emit(OaiEngine(found, time.time()))

        def _emit(self, ev):
            try:
                self.server.emit(ev)
            except Exception:
                pass

        def _relay(self):
            # Before the body is read, let alone forwarded: the target decides
            # which host this request goes to, and only a plain rooted path
            # may. Reading first would mean doing attacker-sized work on behalf
            # of a request already known to be refused.
            path = safe_request_path(self.path)
            if path is None:
                self._send_bad_target()
                return

            # safe_request_path rejects the spellings that are known to reach
            # past the first slash; this re-derives the origin from the joined
            # URL and refuses anything that did not land on the configured one
            # anyway. Two independent checks, because the cost of the blocklist
            # being incomplete is the whole request -- API key included -- going
            # to a host the client picked. Both run before the body is read.
            target = self.server.upstream.rstrip("/") + path
            if urllib.parse.urlsplit(target)[:2] != self.server.upstream_origin:
                self._send_bad_target()
                return

            try:
                length = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""

            # The query string is part of self.path and is forwarded as-is, but
            # it is not part of the route: matching it exactly meant a client
            # appending one silently stopped the request being measured, while
            # leaving it working, which is the hardest kind of gap to notice.
            route = path.split("?", 1)[0]
            measured = self.command == "POST" and route in INSTRUMENTED_PATHS
            model, is_stream = None, False
            if measured:
                raw, model, is_stream = prepare_body(raw)

            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP}

            req = urllib.request.Request(
                target,
                data=raw if raw else None, headers=headers, method=self.command)

            started = time.time()
            if measured:
                self._emit(OaiRequestStart(model, started))
            try:
                # B310 is silenced because the scheme is checked, not ignored:
                # _ProxyServer refuses to start on anything but http/https, so
                # this URL cannot be a file: or a custom scheme. _opener rather
                # than urlopen so a redirect is relayed, not followed.
                upstream = _opener.open(  # nosec B310
                    req, timeout=PROXY_TIMEOUT)
            except urllib.error.HTTPError as err:
                # A real HTTP error is a real answer: pass it through verbatim so
                # the client sees the server's own message, not llmwatch's.
                self._send_error_response(err, measured)
                return
            except (urllib.error.URLError, OSError, ValueError) as err:
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                self._send_gateway_error(err)
                return

            with upstream:
                self._pump(upstream, measured, is_stream, started)

        def _send_bad_target(self):
            """Refused before any connection is made, so a rejected target
            cannot be distinguished from a rejected one by timing either.

            The connection is closed rather than kept alive: the body was
            deliberately not read, so whatever the client already sent is still
            in the socket, and on a kept-alive connection the next read would
            parse it as a request line.
            """
            body = json.dumps({"error": {
                "message": "llmwatch: refusing to forward this request target",
                "type": "invalid_request_error"}}).encode("utf-8")
            self.close_connection = True
            try:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, BrokenPipeError):
                pass

        def _send_error_response(self, err, measured):
            try:
                body = err.read()
            except Exception:
                body = b""
            if measured:
                self._emit(OaiRequestAborted(time.time()))
            try:
                self.send_response(err.code)
                for key, value in (err.headers or {}).items():
                    if key.lower() not in HOP_BY_HOP:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, BrokenPipeError):
                pass

        def _send_gateway_error(self, err):
            """The upstream is unreachable. The client gets a clean, immediate
            error in the shape it already parses, rather than a hang."""
            body = json.dumps({"error": {
                "message": "llmwatch: upstream %s unreachable (%s)" % (
                    self.server.upstream, safe_text(str(err), limit=120)),
                "type": "upstream_unavailable"}}).encode("utf-8")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, BrokenPipeError):
                pass

        def _pump(self, upstream, measured, is_stream, started):
            """Relay the response, measuring it on the way past if asked."""
            status = getattr(upstream, "status", None) or upstream.getcode()
            out_headers = [(k, v) for k, v in upstream.headers.items()
                           if k.lower() not in HOP_BY_HOP]
            length = upstream.headers.get("Content-Length")
            self._note_engine(upstream.headers, None)

            # A response with a known length is not a stream; buffering it adds no
            # latency because the client cannot act on a partial body anyway.
            if length is not None or not is_stream:
                self._pump_whole(upstream, status, out_headers, length,
                                 measured, started)
                return

            try:
                self.send_response(status)
                for key, value in out_headers:
                    self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
            except (OSError, BrokenPipeError):
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                return

            buffer = b""
            decoded, first_token, usage, timings = 0, None, None, None
            # A rate needs an interval, and an interval needs two arrivals. A
            # short answer often lands in a single TCP read, where every token
            # appears to arrive at the same instant -- which divides out to tens
            # of thousands of tokens per second and, once banked, permanently
            # poisons `peak` and the token-weighted average.
            last_token, arrivals = None, 0
            last_tick = 0.0
            # One body inspection per request: enough to notice a swapped
            # server, cheap enough not to matter.
            engine_checked = False
            try:
                while True:
                    chunk = upstream.read(4096)
                    if not chunk:
                        break
                    # Relayed before it is inspected: measurement must never be in
                    # front of the bytes the client is waiting for.
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                    if not measured:
                        continue
                    produced_here = False
                    try:
                        buffer, payloads = _sse_events(buffer, chunk)
                        for payload in payloads:
                            produced, found, found_timings = read_stream_usage(payload)
                            if produced:
                                decoded += 1
                                produced_here = True
                            if found:
                                usage = found
                            # llama-server fills timings on the final chunk, the
                            # same one that carries usage. Kept separately so a
                            # server that sends one without the other still
                            # yields whichever it did send.
                            if found_timings:
                                timings = found_timings
                            # llama.cpp puts timings on the final chunk and
                            # MLX puts its fingerprint on every one, so the
                            # free check runs always and the parse runs once.
                            if found_timings:
                                self._note_engine(None,
                                                  {"timings": found_timings})
                            elif not engine_checked:
                                engine_checked = True
                                self._note_engine(None, {
                                    "system_fingerprint":
                                        read_stream_fingerprint(payload)})
                    except Exception:
                        # Format drift costs the numbers for this request, not the
                        # request itself.
                        buffer = b""
                    now = time.time()
                    if produced_here:
                        arrivals += 1
                        if first_token is None:
                            first_token = now
                        last_token = now
                    # `measured` is already true here; anything else was
                    # skipped by the continue above.
                    if decoded and now - last_tick >= 0.25:
                        last_tick = now
                        self._emit(OaiGenTick(decoded, now))
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (OSError, BrokenPipeError):
                # The client hung up or the upstream died mid-stream. Either way the
                # token count is truncated, so nothing is banked.
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                return

            if measured:
                # One arrival is not a measurement of anything, so the window is
                # withheld and the request is recorded without a generation rate.
                self._finish_measure(usage, timings, first_token,
                                     last_token if arrivals >= 2 else None, started)

        def _pump_whole(self, upstream, status, out_headers, length, measured, started):
            try:
                declared = int(length) if length is not None else -1
            except (TypeError, ValueError):
                declared = -1
            if declared > MAX_BUFFERED_BODY:
                measured = False

            try:
                body = upstream.read()
            except OSError:
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                return
            try:
                self.send_response(status)
                for key, value in out_headers:
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
            except (OSError, BrokenPipeError):
                if measured:
                    self._emit(OaiRequestAborted(time.time()))
                return

            if not measured:
                return
            usage, timings = None, None
            try:
                obj = json.loads(body)
                if isinstance(obj, dict):
                    if isinstance(obj.get("usage"), dict):
                        usage = obj["usage"]
                    if isinstance(obj.get("timings"), dict):
                        timings = obj["timings"]
                    self._note_engine(None, obj)
            except (ValueError, TypeError):
                usage, timings = None, None
            # No first-token mark exists for a non-streamed response, so the split
            # between prefill and generation is genuinely unknown. The request is
            # still recorded with its duration and token counts; the adapter simply
            # publishes no rate for it, which beats splitting it on a guess.
            self._finish_measure(usage, timings, None, None, started)

        def _finish_measure(self, usage, timings, first_token, last_token, started):
            prompt_tokens, cached, completion = usage_counts(usage)
            if usage is None:
                # Without a usage block there is no honest token count. Streamed
                # deltas were counted, but a delta is not reliably one token, so
                # they animate the live pane and are not banked as a measurement.
                self._emit(OaiRequestAborted(time.time()))
                return
            # Acceptance is a ratio, not a rate, so it needs no timing window and
            # survives even the requests whose generation window was withheld.
            counts = draft_counts(timings)
            draft = None
            if counts is not None:
                accepted, generated = counts
                # mean accepted-run length is not in the JSON (it needs the
                # verification-step count, which only the log reports), so the
                # slot is left empty rather than filled with a derived guess.
                draft = (accepted / float(generated), accepted, generated, None)
            self._emit(OaiRequestEnd(prompt_tokens, cached, completion,
                                     started, first_token, last_token, time.time(),
                                     draft))


    class _ProxyServer(http.server.ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

        def __init__(self, address, upstream, emit):
            # Checked here, the one place every request must pass through, so
            # no future caller can introduce a file: upstream by another route.
            if not valid_upstream(upstream):
                raise ValueError(
                    "upstream must be http:// or https://, got %r" % (upstream,))
            # An IPv6 literal needs an IPv6 socket. The default family is IPv4,
            # which would fail to bind ::1 with an error about the address
            # rather than about the family, so parsing it correctly is only
            # half the job. Set on the instance, before bind happens in
            # __init__, so the class default is left alone.
            if ":" in address[0]:
                self.address_family = http.server.socket.AF_INET6
            http.server.ThreadingHTTPServer.__init__(self, address, _ProxyHandler)
            self.upstream = upstream
            # The origin every forwarded URL must still resolve to. Computed
            # once from the operator's own configuration, so the per-request
            # check below compares against something no client can influence.
            self.upstream_origin = urllib.parse.urlsplit(
                upstream.rstrip("/"))[:2]
            self.engine = None
            self.emit = emit

        def handle_error(self, request, client_address):
            """A broken client connection is routine here, not something to dump a
            traceback about onto the screen the TUI owns."""

    _PROXY_SERVER_CLASS = _ProxyServer
    return _PROXY_SERVER_CLASS


def safe_request_path(path):
    """The request target to forward, or None if it must be refused.

    The upstream URL is built by joining this onto the configured base, so a
    target that can reach past the first `/` chooses the host instead of the
    path. `@evil.com/` is the one that matters: joined on, it turns the
    configured `http://127.0.0.1:8080` into userinfo and sends the request,
    headers and all, to evil.com. `//evil.com/x` is the protocol-relative
    spelling of the same trick.

    Refused rather than rewritten. There is no legitimate OpenAI endpoint that
    needs any of these spellings, so a client sending one is not a client this
    should be trying to satisfy.
    """
    if not path or not path.startswith("/"):
        return None
    if path.startswith("//") or path.startswith("/\\"):
        return None
    # A backslash is a path separator to enough HTTP stacks that it is not
    # worth reasoning about which one is on the other end.
    if "\\" in path:
        return None
    return path


def start_proxy(listen, upstream, emit):
    """Serve until the process exits. Returns the bound (host, port).

    Raises OSError if the port is taken -- which must be fatal rather than
    silent: a proxy that failed to bind means the client is still talking
    straight to the server, and llmwatch would sit there showing an idle
    screen that looks like nothing is happening.
    """
    host, port = listen
    server = _proxy_server_class()((host, port), upstream, emit)
    thread = threading.Thread(target=server.serve_forever, name="llmwatch-proxy")
    thread.daemon = True
    thread.start()
    return server.server_address[:2]
