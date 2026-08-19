"""Tests for the OpenAI-compatible backend (--proxy).

Three things here are easy to get wrong and expensive to ship wrong: a proxy
that breaks the request it is measuring, a rate computed from a token count
that was never actually reported, and message content leaking out of a class
that promises to only ever read `usage`.
"""

import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmwatch import (  # noqa: E402
    DEFAULT_PROXY_PORT, OaiGenTick, OaiPrefillTick, OaiRequestAborted,
    LOOPBACK_HOSTS, OaiRequestEnd, OaiRequestStart, Style, Tracker,
    _sse_events, detect_openai_server, draft_counts, render_idle,
    parse_line, parse_listen, prepare_body, read_stream_usage, start_proxy,
    safe_request_path, usage_counts, valid_upstream,
)


def chunk(**payload):
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def delta(text):
    return chunk(choices=[{"delta": {"content": text}}])


USAGE_CHUNK = chunk(choices=[], usage={
    "prompt_tokens": 1000, "completion_tokens": 200,
    "prompt_tokens_details": {"cached_tokens": 400}})

# llama-server adds its own `timings` block beside `usage`, and populates
# draft_n/draft_n_accepted only while a speculative drafter (DFlash, EAGLE,
# MTP) is loaded. Same shape whichever drafter produced it.
SPEC_USAGE_CHUNK = chunk(choices=[], usage={
    "prompt_tokens": 1000, "completion_tokens": 200,
    "prompt_tokens_details": {"cached_tokens": 400}},
    timings={"predicted_per_second": 41.5, "draft_n": 350,
             "draft_n_accepted": 260})


class TestPrepareBody(unittest.TestCase):
    """The request is rewritten on its way past, so this is the one place that
    can break somebody's coding session outright."""

    def test_streaming_request_asks_for_usage(self):
        raw, _, is_stream = prepare_body(json.dumps(
            {"model": "m", "stream": True}).encode())
        self.assertTrue(is_stream)
        self.assertTrue(json.loads(raw)["stream_options"]["include_usage"])

    def test_stream_options_without_include_usage_is_normalised(self):
        """mlx_lm.server reads stream_options["include_usage"] with a bare
        subscript, so forwarding this verbatim takes the upstream down."""
        raw, _, _ = prepare_body(json.dumps(
            {"model": "m", "stream": True,
             "stream_options": {"something_else": 1}}).encode())
        options = json.loads(raw)["stream_options"]
        self.assertTrue(options["include_usage"])
        self.assertEqual(options["something_else"], 1)

    def test_a_client_that_already_asked_is_left_alone(self):
        original = json.dumps({"model": "m", "stream": True,
                               "stream_options": {"include_usage": True}}).encode()
        raw, _, _ = prepare_body(original)
        self.assertEqual(raw, original)

    def test_non_streaming_body_is_untouched(self):
        original = json.dumps({"model": "m"}).encode()
        raw, model, is_stream = prepare_body(original)
        self.assertEqual(raw, original)
        self.assertFalse(is_stream)
        self.assertEqual(model, "m")

    def test_model_name_is_shortened_like_every_other_source(self):
        _, model, _ = prepare_body(json.dumps(
            {"model": "/Users/me/models/mlx/qwen3.8-27b-4bit"}).encode())
        self.assertEqual(model, "qwen3.8-27b-4bit")

    def test_unparseable_body_is_forwarded_verbatim(self):
        """A request llmwatch cannot read is still a request that must work."""
        raw, model, is_stream = prepare_body(b"not json at all")
        self.assertEqual(raw, b"not json at all")
        self.assertIsNone(model)
        self.assertFalse(is_stream)

    def test_a_json_array_body_is_not_treated_as_a_request(self):
        raw, model, _ = prepare_body(b"[1,2,3]")
        self.assertEqual(raw, b"[1,2,3]")
        self.assertIsNone(model)


class TestStreamReading(unittest.TestCase):

    def test_events_split_across_a_chunk_boundary(self):
        """Nothing may wait for a whole response, so a payload cut in half by
        the network has to survive being reassembled."""
        buffer, payloads = _sse_events(b"", b'data: {"a":')
        self.assertEqual(payloads, [])
        buffer, payloads = _sse_events(buffer, b'1}\n')
        self.assertEqual(payloads, [b'{"a":1}'])

    def test_content_delta_counts_but_is_not_returned(self):
        produced, usage, _ = read_stream_usage(
            json.dumps({"choices": [{"delta": {"content": "hello"}}]}).encode())
        self.assertTrue(produced)
        self.assertIsNone(usage)

    def test_completions_style_text_also_counts(self):
        produced, _, _ = read_stream_usage(
            json.dumps({"choices": [{"text": "hi"}]}).encode())
        self.assertTrue(produced)

    def test_a_tool_call_delta_counts_as_output(self):
        """A coding agent's turn is often nothing but a tool call. Treating
        those as producing no output left half of a real opencode session
        recorded with no tokens and no rates at all."""
        produced, _, _ = read_stream_usage(json.dumps({"choices": [{"delta": {
            "tool_calls": [{"index": 0, "function": {"name": "write"}}]}}]}).encode())
        self.assertTrue(produced)

    def test_empty_delta_does_not_count_as_a_token(self):
        """The role-only opening chunk carries no content and must not be timed
        as the first token."""
        produced, _, _ = read_stream_usage(
            json.dumps({"choices": [{"delta": {"role": "assistant"}}]}).encode())
        self.assertFalse(produced)

    def test_done_and_malformed_payloads_are_ignored(self):
        for payload in (b"[DONE]", b"", b"{oh dear", b"null", b"[]"):
            produced, usage, _ = read_stream_usage(payload)
            self.assertFalse(produced, payload)
            self.assertIsNone(usage, payload)

    def test_usage_is_picked_up(self):
        _, usage, _ = read_stream_usage(USAGE_CHUNK[6:].strip())
        self.assertEqual(usage_counts(usage), (1000, 400, 200))

    def test_usage_counts_survive_missing_and_junk_fields(self):
        self.assertEqual(usage_counts(None), (0, 0, 0))
        self.assertEqual(usage_counts({}), (0, 0, 0))
        self.assertEqual(usage_counts({"prompt_tokens": "x"}), (0, 0, 0))
        self.assertEqual(
            usage_counts({"prompt_tokens": 5, "prompt_tokens_details": "junk"}),
            (5, 0, 0))


class TestSpeculativeDraftStats(unittest.TestCase):
    """Acceptance rate is the number that decides whether a drafter is worth
    running at all: below roughly half, verification costs more than the
    accepted tokens save, and the base build is faster. The log backend has
    reported it since the -mtp builds; over the wire it arrives in
    llama-server's `timings` block instead.
    """

    def test_counts_come_back_accepted_first(self):
        """Ordered like the log backend's (accepted, generated), not like the
        JSON's (draft_n, draft_n_accepted) -- one caller, one convention."""
        self.assertEqual(
            draft_counts({"draft_n": 350, "draft_n_accepted": 260}), (260, 350))

    def test_absent_when_nothing_was_drafted(self):
        """A server with no drafter omits both keys, and a run that drafted
        nothing has no ratio to report. Neither is a zero-percent acceptance."""
        self.assertIsNone(draft_counts({"predicted_per_second": 12.0}))
        self.assertIsNone(draft_counts({"draft_n": 0, "draft_n_accepted": 0}))

    def test_survives_missing_and_junk_fields(self):
        for timings in (None, "junk", {}, {"draft_n": "x", "draft_n_accepted": 1},
                        {"draft_n": 10}):
            self.assertIsNone(draft_counts(timings), timings)

    def test_more_accepted_than_drafted_is_rejected(self):
        """A ratio above 1.0 would print as '140% accepted'. Whatever produced
        it, it is not a measurement."""
        self.assertIsNone(draft_counts({"draft_n": 10, "draft_n_accepted": 14}))

    def test_timings_are_read_off_a_stream_payload(self):
        _, _, timings = read_stream_usage(SPEC_USAGE_CHUNK[6:].strip())
        self.assertEqual(draft_counts(timings), (260, 350))

    def test_a_response_without_timings_reports_no_draft(self):
        _, _, timings = read_stream_usage(USAGE_CHUNK[6:].strip())
        self.assertIsNone(timings)

    def test_the_tracker_publishes_acceptance_on_request_end(self):
        """Downstream (history's draft_rate column, the drafts-often-rejected
        hint) reads `draft` off request_end and does not care which backend
        filled it in."""
        t = Tracker()
        t.feed(OaiRequestStart("qwen3.8-27b", 100.0))
        outs = t.feed(OaiRequestEnd(1000, 0, 200, 100.0, 110.0, 120.0, 121.0,
                                    (260 / 350.0, 260, 350, None)))
        end = {o.data.get("event"): o.data for o in outs}["request_end"]
        rate, accepted, generated, _mean = end["draft"]
        self.assertEqual((accepted, generated), (260, 350))
        self.assertAlmostEqual(rate, 260 / 350.0)

    def test_draft_defaults_to_none_for_servers_that_send_no_timings(self):
        """Every other OpenAI server (mlx_lm.server, LM Studio, vLLM) sends no
        timings at all, and must keep working with the field absent."""
        t = Tracker()
        t.feed(OaiRequestStart("some-model", 100.0))
        outs = t.feed(OaiRequestEnd(1000, 0, 200, 100.0, 110.0, 120.0, 121.0))
        end = {o.data.get("event"): o.data for o in outs}["request_end"]
        self.assertIsNone(end["draft"])


class TestTrackerAdapter(unittest.TestCase):
    """The adapter turns proxy events into the same shape every other backend
    produces, so the rest of the program never learns a second dialect."""

    def events(self, outs):
        return [o.data.get("event") for o in outs]

    def by_event(self, outs):
        return {o.data.get("event"): o.data for o in outs}

    def test_a_whole_request(self):
        t = Tracker()
        t.feed(OaiRequestStart("qwen3.8-27b-4bit", 100.0))
        outs = t.feed(OaiRequestEnd(1000, 0, 200, 100.0, 110.0, 120.0, 121.0))
        data = self.by_event(outs)
        self.assertEqual(data["prefill_done"]["tokens"], 1000)
        self.assertAlmostEqual(data["prefill_done"]["rate"], 100.0)
        self.assertEqual(data["generate_done"]["tokens"], 200)
        # 200 tokens over first-token..last-token, not over the whole request:
        # the trailing usage and [DONE] frames are protocol, not generation.
        self.assertAlmostEqual(data["generate_done"]["rate"], 20.0)
        self.assertAlmostEqual(data["generate_done"]["seconds"], 10.0)
        self.assertAlmostEqual(data["request_end"]["seconds"], 21.0)

    def test_cached_tokens_do_not_inflate_the_prefill_rate(self):
        """usage counts the whole prompt, cache included. Dividing by that
        would report a speed the machine never reached."""
        t = Tracker()
        t.feed(OaiRequestStart("m", 0.0))
        outs = t.feed(OaiRequestEnd(1000, 900, 10, 0.0, 1.0, 2.0, 2.0))
        data = self.by_event(outs)
        self.assertEqual(data["prefill_done"]["tokens"], 100)
        self.assertAlmostEqual(data["prefill_done"]["rate"], 100.0)
        self.assertEqual(data["prefill_done"]["cached"], 900)

    def test_a_fully_cached_prompt_reports_no_prefill_rate(self):
        t = Tracker()
        t.feed(OaiRequestStart("m", 0.0))
        outs = t.feed(OaiRequestEnd(500, 500, 10, 0.0, 1.0, 2.0, 2.0))
        self.assertNotIn("prefill_done", self.events(outs))
        self.assertIn("request_end", self.events(outs))

    def test_a_phase_with_no_duration_contributes_no_rate(self):
        """A zero would pass the small-request noise guard and then drag down
        `low`, the weighted average, and the median slowdown reads."""
        t = Tracker()
        t.feed(OaiRequestStart("m", 5.0))
        outs = t.feed(OaiRequestEnd(100, 0, 10, 5.0, 5.0, 5.0, 5.0))
        self.assertEqual(self.events(outs), ["request_end"])

    def test_a_generation_too_short_to_time_reports_no_rate(self):
        """Observed for real: 8 tokens arriving in one TCP read produced a
        window of 0.0002s and a rate of 38,260 tok/s, which would have outranked
        every genuine sample on the board for as long as the history survived."""
        t = Tracker()
        t.feed(OaiRequestStart("m", 0.0))
        outs = t.feed(OaiRequestEnd(100, 0, 8, 0.0, 5.0, 5.0002, 5.001))
        self.assertNotIn("generate_done", self.events(outs))
        self.assertIn("request_end", self.events(outs))

    def test_a_single_arrival_is_not_a_measurement(self):
        """The proxy withholds last_token when it only ever saw one arrival,
        because a rate needs an interval and one instant is not one."""
        t = Tracker()
        t.feed(OaiRequestStart("m", 0.0))
        outs = t.feed(OaiRequestEnd(100, 0, 8, 0.0, 5.0, None, 6.0))
        self.assertNotIn("generate_done", self.events(outs))

    def test_a_non_streamed_response_records_no_rates(self):
        """With no first-token mark the prefill/generation split is unknown,
        and a guess would be indistinguishable on screen from a measurement."""
        t = Tracker()
        t.feed(OaiRequestStart("m", 0.0))
        outs = t.feed(OaiRequestEnd(100, 0, 50, 0.0, None, None, 10.0))
        self.assertEqual(self.events(outs), ["request_end"])
        self.assertAlmostEqual(self.by_event(outs)["request_end"]["seconds"], 10.0)

    def test_prefill_ticks_animate_the_live_pane(self):
        t = Tracker()
        t.feed(OaiRequestStart("m", 0.0))
        outs = t.feed(OaiPrefillTick(5000, 28192, 10.0))
        self.assertEqual(outs[0].kind, "live")
        self.assertAlmostEqual(outs[0].data["rate"], 500.0)
        self.assertEqual(outs[0].data["to_process"], 28192)

    def test_log_ticks_with_no_open_request_belong_to_nothing(self):
        """The log keeps producing these between requests -- another client, or
        a warm-up -- and they must not open a phantom request."""
        t = Tracker()
        self.assertEqual(t.feed(OaiPrefillTick(10, 100, 1.0)), [])
        self.assertEqual(t.feed(OaiGenTick(5, 1.0)), [])

    def test_generate_ticks_report_a_live_rate(self):
        t = Tracker()
        t.feed(OaiRequestStart("m", 0.0))
        t.feed(OaiGenTick(1, 10.0))
        outs = t.feed(OaiGenTick(21, 12.0))
        self.assertAlmostEqual(outs[0].data["rate"], 10.5)

    def test_an_aborted_request_banks_nothing(self):
        """Agents cancel constantly. Recording a truncated stream as a finished
        request would fill the history with fast-looking rows that never were."""
        t = Tracker()
        t.feed(OaiRequestStart("m", 0.0))
        outs = t.feed(OaiRequestAborted(5.0))
        self.assertEqual(self.events(outs), ["request_abandoned"])
        self.assertEqual(t.feed(OaiRequestEnd(1, 0, 1, 0.0, 1.0, 2.0, 2.0)), [])

    def test_a_new_request_closes_a_stale_one(self):
        t = Tracker()
        t.feed(OaiRequestStart("m", 0.0))
        outs = t.feed(OaiRequestStart("m", 10.0))
        self.assertIn("request_abandoned", self.events(outs))
        self.assertIn("request_start", self.events(outs))

    def test_the_model_names_itself_on_the_first_request(self):
        t = Tracker()
        outs = t.feed(OaiRequestStart("qwen3.8-27b-4bit", 0.0))
        self.assertIn("model_loaded", self.events(outs))
        self.assertEqual(t.model, "qwen3.8-27b-4bit")

    def test_the_two_backends_cannot_collide(self):
        self.assertNotEqual(Tracker.MLX_SLOT, Tracker.OAI_SLOT)


class TestStandaloneMlxServerLog(unittest.TestCase):
    """mlx_lm.server's log is plain Python logging, sharing nothing with the
    Ollama MLX-runner dialect but the words in one message."""

    def test_prefill_progress_is_parsed(self):
        ev = parse_line(
            "2026-08-19 14:46:31,875 - INFO - Prompt processing progress: 5000/28192")
        self.assertIsInstance(ev, OaiPrefillTick)
        self.assertEqual((ev.processed, ev.total), (5000, 28192))
        self.assertIsNotNone(ev.ts)

    def test_the_ollama_dialect_still_wins_its_own_lines(self):
        """Both dialects are always understood, and neither may swallow the
        other's lines."""
        ev = parse_line(
            'time=2026-08-19T14:46:31.875Z level=INFO '
            'msg="Prompt processing progress" processed=100 total=200')
        self.assertEqual(type(ev).__name__, "MlxPrefillTick")

    def test_other_lines_are_not_events(self):
        for line in ("2026-08-19 14:46:48,894 - INFO - Prompt Cache: 3 sequences, 0.48 GB",
                     ('127.0.0.1 - - [19/Aug/2026 14:46:48] '
                      '"POST /v1/chat/completions HTTP/1.1" 200 -'),
                     "2026-08-19 14:46:25,881 - INFO - Starting httpd at 127.0.0.1 on port 8080..."):
            self.assertIsNone(parse_line(line), line)

    def test_a_timestamp_that_will_not_parse_costs_the_tick_not_the_run(self):
        ev = parse_line("garbled - INFO - Prompt processing progress: 1/2")
        self.assertIsInstance(ev, OaiPrefillTick)
        self.assertIsNone(ev.ts)


class TestParseListen(unittest.TestCase):

    def test_forms(self):
        self.assertEqual(parse_listen(""), ("127.0.0.1", DEFAULT_PROXY_PORT))
        self.assertEqual(parse_listen(None), ("127.0.0.1", DEFAULT_PROXY_PORT))
        self.assertEqual(parse_listen("9000"), ("127.0.0.1", 9000))
        self.assertEqual(parse_listen(":9000"), ("127.0.0.1", 9000))
        self.assertEqual(parse_listen("0.0.0.0:9000"), ("0.0.0.0", 9000))
        self.assertEqual(parse_listen("example"), ("example", DEFAULT_PROXY_PORT))

    def test_a_junk_port_falls_back_rather_than_crashing(self):
        self.assertEqual(parse_listen("host:junk"), ("host", DEFAULT_PROXY_PORT))

    def test_ipv6_literals(self):
        """rpartition(':') cannot split an address that is mostly colons. It
        used to turn '::1' into the host ':' on port 1, which then failed the
        loopback check and refused to start -- reading as a policy decision
        rather than the parsing bug it was."""
        self.assertEqual(parse_listen("[::1]:9000"), ("::1", 9000))
        self.assertEqual(parse_listen("[::1]"), ("::1", DEFAULT_PROXY_PORT))
        self.assertEqual(parse_listen("::1"), ("::1", DEFAULT_PROXY_PORT))
        self.assertEqual(parse_listen("::"), ("::", DEFAULT_PROXY_PORT))
        self.assertEqual(parse_listen("[::]:9000"), ("::", 9000))

    def test_an_ipv6_loopback_is_recognised_as_loopback(self):
        """The point of parsing it correctly: ::1 is loopback, so it must not
        need --proxy-allow-remote."""
        self.assertIn(parse_listen("[::1]:9000")[0], LOOPBACK_HOSTS)
        self.assertIn(parse_listen("::1")[0], LOOPBACK_HOSTS)


class TestRequestTargetCannotChooseTheHost(unittest.TestCase):
    """The upstream URL is the configured base joined to the request target,
    so the target must not be able to reach past the first slash. Caught by
    CodeQL as py/partial-ssrf before this shipped.
    """

    def test_ordinary_paths_are_forwarded(self):
        for path in ("/v1/chat/completions", "/v1/models?limit=1",
                     "/v1/files/abc@example.com", "/"):
            self.assertEqual(safe_request_path(path), path)

    def test_userinfo_smuggling_is_refused(self):
        """`@evil.com/` joined onto `http://127.0.0.1:8080` makes the configured
        host into userinfo and sends the request to evil.com instead."""
        self.assertIsNone(safe_request_path("@evil.com/"))

    def test_protocol_relative_and_absolute_targets_are_refused(self):
        for path in ("//evil.com/x", "http://evil.com/x", "https://evil.com/x",
                     "/\\evil.com/x", "/v1\\..\\x", "", None):
            self.assertIsNone(safe_request_path(path), path)

    def test_the_proxy_refuses_it_on_the_wire(self):
        """End to end: the smuggled target must never reach the upstream, and
        the client must be told rather than quietly served something else."""
        import urllib.error
        import urllib.request
        upstream = StubUpstream(b'{"ok": true}', stream=False)
        self.addCleanup(upstream.close)
        seen = []
        _, port = start_proxy(("127.0.0.1", 0), upstream.url, seen.append)

        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % port,
            data=b"{}", headers={"Content-Type": "application/json"},
            method="POST")
        # Rewrite the request line to the smuggled target, which urllib would
        # otherwise normalise away before it ever reached the proxy.
        req.selector = "@evil.com/"
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(upstream.requests, [])
        self.assertEqual(seen, [])

    def refuse(self, selector, upstream_url):
        """POST `selector` as the raw request target. Returns (code, reached)."""
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            upstream_url, data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        req.selector = selector
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, True
        except urllib.error.HTTPError as err:
            return err.code, False

    def test_the_origin_is_rechecked_after_the_url_is_built(self):
        """Defence in depth, and the reason it is not redundant: the blocklist
        enumerates spellings, and this does not. Whatever a target is spelled
        like, if the joined URL does not still point at the configured origin
        it is refused before a connection is opened.
        """
        upstream = StubUpstream(b'{"ok": true}', stream=False)
        self.addCleanup(upstream.close)
        seen = []
        _, port = start_proxy(("127.0.0.1", 0), upstream.url, seen.append)
        mine = "http://127.0.0.1:%d/v1/chat/completions" % port

        # "//evil.com/x" is deliberately absent: http.server collapses the
        # doubled leading slash before self.path is set, so it arrives as the
        # ordinary path /evil.com/x on the configured host and is forwarded,
        # which is correct. safe_request_path still refuses that spelling for
        # the stacks that do not collapse it.
        for selector in ("@evil.com/", "https://evil.com/v1/chat/completions",
                         "/\\evil.com", "http://evil.com/"):
            code, reached = self.refuse(selector, mine)
            self.assertEqual(code, 400, selector)
            self.assertFalse(reached, selector)

        # Nothing got through to the upstream, and nothing was recorded as a
        # request that happened.
        self.assertEqual(upstream.requests, [])
        self.assertEqual(seen, [])

    def test_an_ordinary_target_still_reaches_the_upstream(self):
        """The guard has to refuse the smuggled spellings without breaking the
        ordinary ones, which is the failure that would be noticed last."""
        upstream = StubUpstream(b'{"ok": true}', stream=False)
        self.addCleanup(upstream.close)
        _, port = start_proxy(("127.0.0.1", 0), upstream.url, lambda ev: None)
        code, reached = self.refuse(
            "/v1/models?limit=2",
            "http://127.0.0.1:%d/v1/models" % port)
        self.assertEqual(code, 200)
        self.assertTrue(reached)
        self.assertEqual(len(upstream.requests), 1)


class TestUpstreamScheme(unittest.TestCase):
    """--upstream is the one piece of the proxy's configuration that comes from
    outside, and urlopen will open far more than http."""

    def test_http_and_https_are_allowed(self):
        self.assertTrue(valid_upstream("http://127.0.0.1:8080"))
        self.assertTrue(valid_upstream("https://example.com"))

    def test_everything_else_is_refused(self):
        """A file: upstream would turn every request the client makes into a
        local file read, with the contents relayed straight back to it."""
        for url in ("file:///etc/passwd", "ftp://host/x", "gopher://host",
                    "127.0.0.1:8080", "", None):
            self.assertFalse(valid_upstream(url), url)

    def test_the_server_refuses_to_start_on_one(self):
        """Enforced at the choke point every request passes through, not only
        in the argument parser, so no other caller can slip past it."""
        with self.assertRaises(ValueError):
            start_proxy(("127.0.0.1", 0), "file:///etc/passwd", lambda ev: None)


def _has_ipv6_loopback():
    import socket
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        s.bind(("::1", 0))
        return True
    except OSError:
        return False
    finally:
        s.close()


class TestIpv6Listen(unittest.TestCase):
    """Parsing '[::1]:9000' correctly is only half of it: the default socket
    family is IPv4, so the bind would still fail, and the error would talk
    about the address rather than the family."""

    @unittest.skipUnless(_has_ipv6_loopback(), "no IPv6 loopback on this host")
    def test_the_proxy_actually_binds_an_ipv6_loopback(self):
        host, port = parse_listen("[::1]:0")
        self.assertEqual(host, "::1")
        bound_host, bound_port = start_proxy(
            (host, port), "http://127.0.0.1:1", lambda ev: None)
        self.assertEqual(bound_host, "::1")
        self.assertTrue(bound_port > 0)


class TestRedirectsAreRelayedNotFollowed(unittest.TestCase):
    """The other direction of the same problem the target checks address.

    Those stop the client choosing the host. This stops the upstream choosing
    it: urlopen follows redirects by default and carries the request headers
    across hosts while doing so, so a 302 pointing elsewhere would deliver the
    Authorization header there -- past both target checks, because the second
    request happens inside urlopen where neither can see it.
    """

    def start(self, location):
        """Proxy in front of an upstream that answers 302 -> `location`."""
        import http.server
        import threading

        class Redirector(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()

        up = http.server.HTTPServer(("127.0.0.1", 0), Redirector)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        self.addCleanup(up.server_close)
        self.addCleanup(up.shutdown)
        _, port = start_proxy(
            ("127.0.0.1", 0), "http://127.0.0.1:%d" % up.server_address[1],
            lambda ev: None)
        return port

    def post(self, port):
        """The test client must not follow the redirect either, or it -- not
        the proxy -- is what ends up at the target, and the assertion would be
        measuring urllib rather than llmwatch."""
        import urllib.error
        import urllib.request

        class NoFollow(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        opener = urllib.request.build_opener(NoFollow)
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % port,
            data=b'{"model":"m"}',
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer SECRET"},
            method="POST")
        try:
            with opener.open(req, timeout=10) as resp:
                return resp.status, dict(resp.headers)
        except urllib.error.HTTPError as err:
            return err.code, dict(err.headers or {})

    def test_the_client_is_told_it_moved_rather_than_being_followed(self):
        target = "http://127.0.0.1:1/steal"
        code, headers = self.post(self.start(target))
        self.assertEqual(code, 302)
        # Relayed intact, so the client can decide for itself.
        self.assertEqual(headers.get("Location"), target)

    def test_the_credential_does_not_reach_the_redirect_target(self):
        """The failure this exists to prevent, asserted on the receiving end:
        a host the operator never configured must see nothing at all."""
        import http.server
        import threading

        seen = []

        class Sink(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                seen.append(dict(self.headers))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        sink = http.server.HTTPServer(("127.0.0.1", 0), Sink)
        threading.Thread(target=sink.serve_forever, daemon=True).start()
        self.addCleanup(sink.server_close)
        self.addCleanup(sink.shutdown)

        port = self.start("http://127.0.0.1:%d/steal" % sink.server_address[1])
        code, _ = self.post(port)
        self.assertEqual(code, 302)
        self.assertEqual(seen, [])


class StubUpstream:
    """A minimal OpenAI-shaped server, so the proxy is tested against a socket
    rather than against a mock of one."""

    def __init__(self, body, stream=True, status=200):
        import http.server

        self.requests = []
        self.paths = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                outer.paths.append(self.path)
                outer.requests.append(self.rfile.read(length))
                self.send_response(status)
                if stream:
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for part in body:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TestProxyEndToEnd(unittest.TestCase):
    """Through a real socket, because the failure this feature cannot have is
    'llmwatch broke the model', and that only shows up on the wire."""

    def post(self, port, payload, path="/v1/chat/completions"):
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (port, path),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()

    def run_proxy(self, upstream_url):
        """Returns (seen, port, finished).

        `finished` matters: the proxy relays the last byte before it records
        anything, precisely so measurement never delays the client -- which
        means a test that reads `seen` the instant its request returns is
        racing the thread that fills it.
        """
        seen = []
        finished = threading.Event()
        terminal = ("OaiRequestEnd", "OaiRequestAborted")

        def emit(ev):
            seen.append(ev)
            if type(ev).__name__ in terminal:
                finished.set()

        _, port = start_proxy(("127.0.0.1", 0), upstream_url, emit)
        return seen, port, finished

    def test_a_streamed_response_is_relayed_and_measured(self):
        upstream = StubUpstream([delta("a"), delta("b"), USAGE_CHUNK, b"data: [DONE]\n\n"])
        self.addCleanup(upstream.close)
        seen, port, finished = self.run_proxy(upstream.url)

        status, body = self.post(port, {"model": "m", "stream": True})
        self.assertEqual(status, 200)
        self.assertTrue(finished.wait(10))
        # The client must receive every byte the upstream sent.
        self.assertIn(b"[DONE]", body)
        self.assertIn(b'"content": "a"', body)

        kinds = [type(e).__name__ for e in seen]
        self.assertEqual(kinds[0], "OaiRequestStart")
        self.assertEqual(kinds[-1], "OaiRequestEnd")
        end = seen[-1]
        self.assertEqual(end.prompt_tokens, 1000)
        self.assertEqual(end.cached_tokens, 400)
        self.assertEqual(end.completion_tokens, 200)
        self.assertIsNotNone(end.first_token)

    def test_the_upstream_is_asked_for_usage_on_the_clients_behalf(self):
        upstream = StubUpstream([USAGE_CHUNK])
        self.addCleanup(upstream.close)
        _, port, finished = self.run_proxy(upstream.url)
        self.post(port, {"model": "m", "stream": True})
        self.assertTrue(finished.wait(10))
        self.assertTrue(json.loads(upstream.requests[0])
                        ["stream_options"]["include_usage"])

    def test_draft_acceptance_survives_the_round_trip(self):
        """End to end against a socket: a llama-server running DFlash/MTP puts
        draft counts in `timings`, and they have to reach the event that the
        history and the hints read."""
        upstream = StubUpstream(
            [delta("a"), delta("b"), SPEC_USAGE_CHUNK, b"data: [DONE]\n\n"])
        self.addCleanup(upstream.close)
        seen, port, finished = self.run_proxy(upstream.url)
        self.post(port, {"model": "m", "stream": True})
        self.assertTrue(finished.wait(10))
        end = seen[-1]
        self.assertEqual(type(end).__name__, "OaiRequestEnd")
        rate, accepted, generated, _mean = end.draft
        self.assertEqual((accepted, generated), (260, 350))
        self.assertAlmostEqual(rate, 260 / 350.0)

    def test_a_non_streamed_response_also_carries_draft_counts(self):
        """The whole-body path parses its own JSON and would otherwise be the
        one place acceptance silently vanished."""
        body = json.dumps({
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "timings": {"draft_n": 20, "draft_n_accepted": 15}}).encode()
        upstream = StubUpstream(body, stream=False)
        self.addCleanup(upstream.close)
        seen, port, finished = self.run_proxy(upstream.url)
        self.post(port, {"model": "m"})
        self.assertTrue(finished.wait(10))
        end = seen[-1]
        self.assertEqual(type(end).__name__, "OaiRequestEnd")
        self.assertEqual(end.draft[1:3], (15, 20))

    def test_a_server_without_timings_still_measures_cleanly(self):
        """The regression that matters for everyone not running a drafter."""
        upstream = StubUpstream([delta("a"), USAGE_CHUNK, b"data: [DONE]\n\n"])
        self.addCleanup(upstream.close)
        seen, port, finished = self.run_proxy(upstream.url)
        self.post(port, {"model": "m", "stream": True})
        self.assertTrue(finished.wait(10))
        end = seen[-1]
        self.assertEqual(end.completion_tokens, 200)
        self.assertIsNone(end.draft)

    def test_a_query_string_does_not_disable_measurement(self):
        """self.path carries the query, so an exact match against
        INSTRUMENTED_PATHS silently stopped measuring the moment a client
        appended one. The request still worked, which is what made it easy to
        miss: the numbers just quietly stopped arriving."""
        upstream = StubUpstream([delta("a"), delta("b"), USAGE_CHUNK,
                                 b"data: [DONE]\n\n"])
        self.addCleanup(upstream.close)
        seen, port, finished = self.run_proxy(upstream.url)
        self.post(port, {"model": "m", "stream": True},
                  path="/v1/chat/completions?foo=bar")
        self.assertTrue(finished.wait(10))
        end = seen[-1]
        self.assertEqual(type(end).__name__, "OaiRequestEnd")
        self.assertEqual(end.completion_tokens, 200)
        # and the query reached the upstream untouched
        self.assertTrue(upstream.paths[-1].endswith("?foo=bar"),
                        upstream.paths[-1])

    def test_a_refused_target_does_not_read_the_body_first(self):
        """The body is attacker-sized and the target is already known to be
        bad, so reading it is work done on behalf of a request that will not
        be forwarded. Refuse first, and close rather than leaving an unread
        body to be parsed as the next request on a kept-alive connection."""
        import socket
        upstream = StubUpstream(b'{"ok": true}', stream=False)
        self.addCleanup(upstream.close)
        seen, port, _ = self.run_proxy(upstream.url)

        conn = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.addCleanup(conn.close)
        # Declares a body far larger than it sends. If the handler reads before
        # validating, it blocks here until the timeout instead of answering.
        conn.sendall(b"POST @evil.com/ HTTP/1.1\r\nHost: x\r\n"
                     b"Content-Length: 100000000\r\n\r\nshort")
        conn.settimeout(10)
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = conn.recv(4096)
            if not chunk:
                break
            head += chunk
        self.assertIn(b"400", head.split(b"\r\n")[0])
        self.assertIn(b"close", head.lower())
        self.assertEqual(upstream.requests, [])
        self.assertEqual(seen, [])

    def test_a_tool_call_only_turn_is_still_measured(self):
        """Through the proxy, end to end: opencode's tool-calling turns must
        arrive with a first-token mark and real token counts."""
        tool = chunk(choices=[{"delta": {
            "tool_calls": [{"index": 0, "function": {"name": "write"}}]}}])
        upstream = StubUpstream([tool, tool, USAGE_CHUNK, b"data: [DONE]\n\n"])
        self.addCleanup(upstream.close)
        seen, port, finished = self.run_proxy(upstream.url)
        self.post(port, {"model": "m", "stream": True})
        self.assertTrue(finished.wait(10))
        end = seen[-1]
        self.assertEqual(type(end).__name__, "OaiRequestEnd")
        self.assertIsNotNone(end.first_token)
        self.assertEqual(end.completion_tokens, 200)

    def test_a_response_with_no_usage_is_not_banked(self):
        """Streamed deltas were counted, but a delta is not reliably one token,
        so they animate the screen and never become a measurement."""
        upstream = StubUpstream([delta("a"), b"data: [DONE]\n\n"])
        self.addCleanup(upstream.close)
        seen, port, finished = self.run_proxy(upstream.url)
        self.post(port, {"model": "m", "stream": True})
        self.assertTrue(finished.wait(10))
        self.assertEqual(type(seen[-1]).__name__, "OaiRequestAborted")

    def test_a_non_streamed_response_is_relayed_and_measured(self):
        payload = json.dumps({"choices": [{"message": {"content": "hi"}}],
                              "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        upstream = StubUpstream(payload.encode(), stream=False)
        self.addCleanup(upstream.close)
        seen, port, finished = self.run_proxy(upstream.url)
        status, body = self.post(port, {"model": "m"})
        self.assertEqual(status, 200)
        self.assertTrue(finished.wait(10))
        self.assertEqual(json.loads(body)["usage"]["prompt_tokens"], 7)
        self.assertEqual(seen[-1].completion_tokens, 3)
        self.assertIsNone(seen[-1].first_token)

    def test_an_uninstrumented_path_is_relayed_without_events(self):
        upstream = StubUpstream(b'{"data": []}', stream=False)
        self.addCleanup(upstream.close)
        seen, port, finished = self.run_proxy(upstream.url)
        status, body = self.post(port, {}, path="/v1/embeddings")
        # Nothing terminal should ever arrive for a path that is not measured.
        self.assertFalse(finished.wait(0.5))
        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"data": []}')
        self.assertEqual(seen, [])

    def test_a_dead_upstream_answers_immediately_instead_of_hanging(self):
        """The client is in somebody's editor. It gets an error it can parse,
        not a socket that never closes."""
        import urllib.error
        seen, port, finished = self.run_proxy("http://127.0.0.1:1")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(port, {"model": "m", "stream": True})
        self.assertTrue(finished.wait(10))
        self.assertEqual(caught.exception.code, 502)
        body = json.loads(caught.exception.read())
        self.assertEqual(body["error"]["type"], "upstream_unavailable")
        self.assertEqual(type(seen[-1]).__name__, "OaiRequestAborted")

    def test_an_upstream_error_reaches_the_client_unchanged(self):
        upstream = StubUpstream(b'{"error": "model not found"}', stream=False,
                                status=404)
        self.addCleanup(upstream.close)
        import urllib.error
        seen, port, finished = self.run_proxy(upstream.url)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(port, {"model": "m"})
        self.assertTrue(finished.wait(10))
        self.assertEqual(caught.exception.code, 404)
        self.assertEqual(json.loads(caught.exception.read())["error"],
                         "model not found")
        # A rejected request produced no tokens, so it is dropped rather than
        # banked as a very fast one.
        self.assertEqual(type(seen[-1]).__name__, "OaiRequestAborted")


class TestProxyKeepsNoContent(unittest.TestCase):
    """History documents that no column could hold prompt content. The proxy
    sees all of it, so that property has to be defended here too."""

    def test_no_event_carries_message_text(self):
        secret = "the launch code is 1234"
        upstream = StubUpstream([delta(secret), USAGE_CHUNK, b"data: [DONE]\n\n"])
        self.addCleanup(upstream.close)
        seen = []
        finished = threading.Event()

        def emit(ev):
            seen.append(ev)
            if type(ev).__name__ in ("OaiRequestEnd", "OaiRequestAborted"):
                finished.set()

        _, port = start_proxy(("127.0.0.1", 0), upstream.url, emit)

        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % port,
            data=json.dumps({"model": "m", "stream": True,
                             "messages": [{"role": "user", "content": secret}]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        self.assertTrue(finished.wait(10))

        for event in seen:
            self.assertNotIn(secret, repr(event))

    def test_the_reader_never_hands_back_the_delta(self):
        produced, usage, _ = read_stream_usage(
            json.dumps({"choices": [{"delta": {"content": "secret"}}]}).encode())
        self.assertNotIn("secret", repr(produced))
        self.assertNotIn("secret", repr(usage))


if __name__ == "__main__":
    unittest.main()


class TestOpenAiServerHint(unittest.TestCase):
    """The gap this closes, from a real session: mlx_lm.server was serving
    Qwen on :8080 and llmwatch, started with no arguments, sat on `no model
    loaded`. Both statements were true -- there was no Ollama model -- and
    together they read as a detection failure rather than the mode mismatch it
    was. The numbers for an OpenAI server are not in its log to be found
    (mlx_lm.server logs prompt progress and nothing else), so --proxy is not a
    shortcut that could be automated away. Saying so is.
    """

    def serve(self, body, path_ok="/v1/models"):
        """A stub that answers `path_ok` the way an OpenAI server would."""
        import http.server
        import threading

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path != path_ok:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv.server_address[1]

    def test_an_openai_server_is_recognised(self):
        port = self.serve(json.dumps(
            {"object": "list", "data": [{"id": "qwen3.8-27b-4bit"}]}).encode())
        self.assertEqual(detect_openai_server([port]), port)

    def test_a_closed_port_is_not(self):
        self.assertIsNone(detect_openai_server([1]))

    def test_something_that_is_not_an_openai_server_is_not(self):
        """A listener is not an endorsement. Suggesting --proxy at some
        unrelated service would send the user's traffic through it."""
        port = self.serve(b"<html>hello</html>", path_ok="/")
        self.assertIsNone(detect_openai_server([port]))

    def test_the_first_match_wins_and_the_rest_are_left_alone(self):
        port = self.serve(json.dumps({"object": "list", "data": []}).encode())
        self.assertEqual(detect_openai_server([1, port, 2]), port)

    def test_the_idle_pane_names_the_port_and_the_flag(self):
        line = render_idle(3.0, Style(color=False, unicode_ok=False, width=100),
                           {"models_loaded": 0, "oai_port": 8080})
        self.assertIn("8080", line)
        self.assertIn("--proxy", line)

    def test_without_a_detection_the_old_wording_is_unchanged(self):
        line = render_idle(3.0, Style(color=False, unicode_ok=False, width=100), {"models_loaded": 0})
        self.assertIn("no model loaded", line)
        self.assertNotIn("--proxy", line)

    def test_a_stopped_ollama_still_leads_with_that(self):
        """server_ok False is the more specific problem and keeps priority;
        the hint is appended rather than replacing it."""
        line = render_idle(3.0, Style(color=False, unicode_ok=False, width=100),
                           {"server_ok": False, "oai_port": 8080})
        self.assertIn("Ollama", line)
        self.assertIn("8080", line)
