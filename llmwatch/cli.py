"""Layer 4: argument parsing, the follow loop, and all the I/O.

follow() is where the threads live: the log reader, the repaint clock and the
keyboard. Everything it calls above this line is pure.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

try:
    import queue
except ImportError:  # pragma: no cover - py2 only, never hit
    queue = None

from .constants import (
    DEFAULT_PROXY_PORT, DEFAULT_UPSTREAM, LOOPBACK_HOSTS, MIN_TURNS_FOR_TYPICAL,
    TURN_TYPICAL_DAYS, __version__)
from .text import Style, fmt_duration, truncate_visible
from .parser import looks_like_timing, parse_line
from .tracker import Tracker
from .stats import Stats
from .logs import candidate_paths, current_model, find_log, tail_command
from .oai import (
    engine_from_log_event, model_name_for_board, parse_listen, valid_upstream)
from .config import (
    config_path, effective_settings, log_event_allowed, read_config, write_config)
from .proxy import start_proxy
from .clients import find_clients, point_client_at, undo_client
from .system import SystemProbe
from .render import (
    ProgressFloor, render_board, render_header, render_idle, render_live,
    render_live_detail, render_summary)
from .discovery import backend_suggestion, discover_backends
from .settings import settings_open
from .uistate import UIState, handle_key
from .history import History
from .compare import build_compare, pickable_models
from .update import (
    _UpgradeRequested, render_update, run_upgrade, start_update_check, upgrade_plan)
from .screen import Screen, compose_frame, plan_frame
from .codex import CodexTail
from .keyboard import Keyboard, _key_reader, _reader


def follow(args):
    # `--proxy` with no value yields "", which is still a request to proxy.
    # Saved settings are consulted last of all, so `ollama-llmwatch` on its own
    # does what was configured while anything typed now still wins.
    settings = effective_settings(args, os.environ, read_config())
    proxying = settings["watch"][0] == "proxy"
    if settings["watch"][0] == "log" and settings["log"][0]:
        # find_log reads this, so a configured path arrives the same way an
        # explicit --log does.
        os.environ["LLMWATCH_LOG"] = settings["log"][0]
    kind, target = find_log()
    if not kind and not proxying:
        sys.stderr.write(
            "llmwatch: could not find an Ollama log. Tried:\n  " +
            "\n  ".join(candidate_paths()) +
            "\n\nPass one explicitly with --log PATH or set LLMWATCH_LOG.\n"
            "Watching an OpenAI-compatible server (mlx_lm.server, LM Studio) instead?\n"
            "Use --proxy, which reads the numbers off the wire rather than from a log.\n"
            "Note: llmwatch needs LOCAL log access; a remote Ollama server won't work.\n")
        return 2

    tracker = Tracker()
    # `ollama ps` is the wrong question when the server is not Ollama; in proxy
    # mode the model names itself on the first request instead.
    if not proxying:
        tracker.model = current_model()
    style = Style.detect(no_color=args.no_color)
    interactive = sys.stdout.isatty() and not args.json
    # Below ~10 rows the panes cannot be laid out usefully, so fall back to the
    # single-line renderer rather than draw something corrupt.
    tui = (interactive and not args.plain
           and shutil.get_terminal_size((80, 24)).lines >= 10)

    q = queue.Queue()
    stop = threading.Event()

    proxy_at, upstream = None, None
    if proxying:
        listen = ("127.0.0.1", settings["proxy_port"][0])
        if args.proxy:
            listen = parse_listen(args.proxy)
        elif os.environ.get("LLMWATCH_PROXY"):
            listen = parse_listen(os.environ["LLMWATCH_PROXY"])
        upstream = settings["upstream"][0]
        if not valid_upstream(upstream):
            sys.stderr.write(
                "llmwatch: --upstream must be http:// or https://, got %r\n"
                % (upstream,))
            return 2
        if listen[0] not in LOOPBACK_HOSTS and not args.proxy_allow_remote:
            sys.stderr.write(
                "llmwatch: refusing to listen on %s.\n"
                "A proxy on a non-loopback address relays whatever it is sent, "
                "and this one authenticates nobody.\n"
                "Pass --proxy-allow-remote if that is genuinely what you want.\n"
                % listen[0])
            return 2
        try:
            proxy_at = start_proxy(listen, upstream, lambda ev: q.put(("event", ev)))
        except OSError as err:
            # Fatal on purpose: if the bind failed the client is still talking
            # straight to the server, and llmwatch would show an idle screen
            # that reads as "nothing is happening" rather than "not connected".
            sys.stderr.write(
                "llmwatch: cannot listen on %s:%s (%s)\n" % (listen[0], listen[1], err))
            return 2

    if not args.json:
        if proxy_at:
            sys.stderr.write("llmwatch %s  proxying http://%s:%d -> %s\n"
                             % (__version__, proxy_at[0], proxy_at[1], upstream))
            if kind:
                where = target if kind == "file" else "journalctl -u %s" % target
                sys.stderr.write("             prefill progress from %s\n" % where)
        else:
            where = target if kind == "file" else "journalctl -u %s" % target
            sys.stderr.write("llmwatch %s  watching %s\n" % (__version__, where))

    # Started before the log is even open, and never waited on. --json is
    # excluded in update_check_disabled()'s caller below: that output is parsed
    # by other programs and usually runs unattended.
    update_box = {}
    upgrading = False
    if not args.json:
        start_update_check(update_box)

    # In proxy mode the log is an optional extra: it supplies the prefill
    # progress bar and nothing else, so its absence costs an animation rather
    # than a measurement.
    proc = None
    if kind:
        proc = subprocess.Popen(tail_command(kind, target), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        thread = threading.Thread(target=_reader, args=(proc, q, stop))
        thread.daemon = True
        thread.start()

    stats = Stats()
    screen = Screen() if tui else None
    keyboard = Keyboard() if tui else None
    # Not gated on `tui`: turn timings are worth recording in --plain and --json
    # runs too, and they are the one thing here that outlives the session.
    codex_tail = CodexTail() if args.codex else None
    # The OpenAI probe is only useful when not already proxying: if --proxy is
    # set, the server is already being watched and there is nothing to suggest.
    probe = SystemProbe(look_for_oai=not proxying,
                        upstream=upstream if proxying else None)
    history = None if args.no_history else History()
    progress_floor = ProgressFloor()   # displayed progress must never rewind
    ui = UIState()
    picker_models = []
    compare_lines = []
    live_data = None
    live_at = time.time()
    idle_since = time.time()
    pending = {}          # task -> {"prefill":..., "generation":...}
    dirty = False

    def clear():
        if interactive and dirty:
            sys.stdout.write("\r\033[K")

    def commit(text):
        if tui:
            return                    # history lives in the recent pane instead
        clear()
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def handle(line):
        """Returns True if this line changed anything worth repainting."""
        ev = parse_line(line)
        if ev is None:
            if args.debug_unparsed and looks_like_timing(line):
                sys.stderr.write("UNPARSED: %s" % line)
            return False
        if not log_event_allowed(ev, proxying):
            return False
        # Servers get swapped on the same machine between runs of llmwatch and
        # during them, so this is re-read per event rather than latched.
        engine = engine_from_log_event(ev)
        if engine:
            tracker.engine = engine
        return dispatch(ev)

    def dispatch(ev):
        """Feed one event to the tracker and act on what comes back.

        Split out from handle() because the proxy produces events directly --
        it never has a log line to parse -- and both sources have to land in
        exactly the same place or the two backends drift apart.
        """
        nonlocal live_data, live_at, idle_since
        changed = False
        for out in tracker.feed(ev):
            data = out.data
            name = data.get("event")
            changed = True

            if args.json:
                sys.stdout.write(json.dumps(data) + "\n")
                sys.stdout.flush()
                continue

            if out.kind == "live":
                live_data, live_at = data, time.time()
                continue

            if name == "request_start":
                stats.note_start(data.get("model") or tracker.model,
                                 data.get("prompt_tokens"))
                commit("")
                commit(render_header(data, style))
                live_data, live_at = data, time.time()
            elif name == "cache_miss":
                commit("  " + style.yellow(
                    "! cache gone - rereading the whole %s-token prompt"
                    % format(data.get("prompt_tokens") or 0, ",")))
            elif name in ("prefill_done", "generate_done"):
                slot = pending.setdefault(data["task"], {})
                slot["prefill" if name == "prefill_done" else "generation"] = data
            elif name == "request_end":
                slot = pending.pop(data["task"], {})
                stats.record(data.get("model") or tracker.model,
                             slot.get("prefill"), slot.get("generation"), data)
                if history:
                    history.record(data.get("model") or tracker.model,
                                   slot.get("prefill"), slot.get("generation"), data)
                for text in render_summary(slot.get("prefill"),
                                           slot.get("generation"), data, style):
                    commit(text)
                live_data = None
                idle_since = time.time()
            elif name == "request_abandoned":
                pending.pop(data["task"], None)
                stats.note_cancel(data.get("model") or tracker.model)
                commit("  " + style.dim("x task %s cancelled - client disconnected before "
                                        "completion" % data["task"]))
                if live_data is not None and live_data.get("task") == data["task"]:
                    live_data = None
                    idle_since = time.time()
            elif name in ("model_loaded", "server_started"):
                commit("  " + style.dim(out.text))
        return changed

    def poll_codex():
        """Read the agent's session file and bank any turn that just finished.

        Separate from paint() because a --json or --plain run never paints, and
        the turn record is the part worth keeping either way.
        """
        if not codex_tail:
            return
        codex_tail.poll()
        for turn in codex_tail.drain():
            model = turn.get("model") or tracker.model
            turn["model"] = model
            if history:
                # Compare against what this model USUALLY takes before adding
                # this turn to the pile, or a slow turn quietly raises the bar
                # it is being measured against.
                profile = history.turn_profile(model, days=TURN_TYPICAL_DAYS)
                if profile and profile.get("turns") >= MIN_TURNS_FOR_TYPICAL:
                    turn["typical_seconds"] = profile["median_seconds"]
                history.record_turn(turn)
            if args.json:
                sys.stdout.write(json.dumps(dict(turn, event="turn_end")) + "\n")
                sys.stdout.flush()
            elif not tui:
                effort = (" at %s effort" % turn["effort"]) if turn.get("effort") else ""
                verb = "turn finished" if turn.get("completed") else "turn interrupted"
                commit("  " + style.cyan("%s in %s%s"
                                         % (verb, fmt_duration(turn["seconds"]), effort)))

    def live_text():
        if live_data is not None:
            return render_live(live_data, time.time() - live_at, style,
                               floor=progress_floor)
        return render_idle(time.time() - idle_since, style,
                           probe.snapshot() if probe else None)

    def paint():
        if not interactive:
            return
        # Plain mode scrolls, so the notice is committed once as a line of its
        # own; the TUI redraws it every frame instead, from the same box.
        if not tui and update_box.get("latest") and not update_box.get("announced"):
            update_box["announced"] = True
            commit(render_update(update_box["latest"], style))
        if probe:
            probe.idle_seconds = time.time() - idle_since
            if proxy_at:
                probe.listen_url = "http://127.0.0.1:%d/v1" % proxy_at[1]
            probe.poll()               # self-rate-limited: 5s cheap, 15s models
        if tui:
            if ui.view == "upgrade" and ui.upgrade_plan is None:
                # Once per opening of the pane, not once per frame. Deciding
                # how this copy upgrades runs `git status` in a checkout and
                # scans PATH otherwise; at 10 fps that stalled the render loop
                # and backed up keyboard input.
                ui.upgrade_plan = upgrade_plan()
            cols, rows = screen.size()
            snap = stats.snapshot(tracker.model)
            snap["engine"] = tracker.engine
            age = time.time() - live_at
            codex_state = codex_tail.state() if codex_tail else None
            system_state = probe.snapshot() if probe else None
            if system_state is not None:
                system_state["suggestion"] = backend_suggestion(
                    settings["watch"][0], system_state,
                    busy=(time.time() - idle_since) < 5.0)
            snap["model"] = model_name_for_board(
                snap.get("model"),
                (system_state or {}).get("models_running"))
            if ui.view == "settings" and ui.settings is None:
                # Carried in so the pane can offer it, rather than working it
                # out again from a snapshot it does not have.
                # Built here, not in UIState: it needs the settings in force,
                # which means the flags, the environment and the file.
                ui.settings = settings_open(settings)
                ui.settings["suggestion"] = (system_state or {}).get("suggestion")
            if ui.settings_clients:
                what, ui.settings_clients = ui.settings_clients, None
                here = "http://127.0.0.1:%d/v1" % (
                    proxy_at[1] if proxy_at else settings["proxy_port"][0])
                ui.settings["listen_url"] = here
                if what == "point":
                    rows_c = ui.settings.get("clients") or []
                    index = ui.settings.get("point_index", 0)
                    if 0 <= index < len(rows_c):
                        entry = rows_c[index]
                        if entry["url"] == here:
                            ok, msg = undo_client(entry)
                        else:
                            ok, msg = point_client_at(entry, here)
                        ui.settings["message"] = msg
                ui.settings["clients"] = find_clients()
                ui.settings["cursor"] = min(
                    ui.settings.get("cursor", 0),
                    max(0, len(ui.settings["clients"]) - 1))
            if ui.settings_discover:
                ui.settings_discover = False
                # Skip our own listening port: the proxy answers, and relays
                # the upstream's Server header, so it looks like whatever is
                # behind it and would point llmwatch at itself.
                mine = (proxy_at[1],) if proxy_at else ()
                ui.settings["found"] = discover_backends(skip_ports=mine)
                ui.settings["level"] = "found"
                ui.settings["cursor"] = 0
                ui.settings["message"] = ""
            if ui.settings_saved is not None:
                ok = write_config(ui.settings_saved)
                ui.settings_saved = None
                if ui.settings:
                    ui.settings["dirty"] = not ok
                    ui.settings["message"] = (
                        "saved to %s, restart to apply" % config_path() if ok
                        else "could not write %s" % config_path())
            style.width = cols          # keep renderers in step with resizes
            screen.draw(compose_frame(
                snap, live_text(), style, cols, rows,
                help_visible=(ui.view == "help"),
                ui=ui, picker=picker_models, compare=compare_lines,
                live_detail=render_live_detail(live_data, snap, style, age,
                                              codex_state, system_state,
                                              probe.busiest if probe else None),
                codex=codex_state, system=system_state,
                update=update_box.get("latest")), rows, cols)
        else:
            text = live_text()
            if text:
                width = shutil.get_terminal_size((80, 24)).columns
                sys.stdout.write("\r\033[K  " + truncate_visible(text, max(1, width - 3)))
                sys.stdout.flush()
    # `dirty` is only meaningful for the plain single-line renderer.

    if tui:
        screen.enter()
        if keyboard.enter():
            key_thread = threading.Thread(target=_key_reader, args=(q, stop))
            key_thread.daemon = True
            key_thread.start()

    last_paint = 0.0
    next_frame = time.monotonic()
    try:
        while True:
            # Wake on new data OR the frame deadline, whichever comes first.
            timeout = max(0.0, next_frame - time.monotonic())
            batch = []
            try:
                batch.append(q.get(timeout=timeout))
            except Exception:
                pass
            # Drain the rest of a burst so 50 lines cause one repaint, not 50.
            while True:
                try:
                    batch.append(q.get_nowait())
                except Exception:
                    break

            got_data = False
            for kind_, payload in batch:
                if kind_ == "key":
                    was = ui.view
                    if handle_key(ui, payload, picker_models,
                                  update=update_box.get("latest")):
                        got_data = True          # repaint at once, don't wait
                    if ui.upgrade_requested:
                        # Confirmed. Everything after this is teardown, so it
                        # leaves the loop rather than trying to keep drawing a
                        # program that is being replaced underneath it.
                        raise _UpgradeRequested()
                    if ui.view == "picker" and was != "picker":
                        picker_models = pickable_models(history)
                    if ui.view == "compare" and was != "compare":
                        compare_lines = build_compare(history, ui.model_a, ui.model_b, style)
                    continue
                if kind_ == "event":
                    if dispatch(payload):
                        got_data = True
                    continue
                if payload is None:
                    raise KeyboardInterrupt
                if handle(payload):
                    got_data = True

            now = time.monotonic()
            trigger = got_data or bool(screen and screen.resized.is_set())
            do_paint, next_frame = plan_frame(now, last_paint, next_frame, trigger,
                                              live_data is not None)
            if do_paint:
                # On the frame tick rather than every wakeup: a burst of log
                # lines must not turn into a burst of stat() calls on the
                # session file.
                poll_codex()
                if not tui:
                    dirty = True
                paint()
                last_paint = now
    except _UpgradeRequested:
        upgrading = True
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        stop.set()
        if proc:
            proc.terminate()
        if history:
            history.close()
        if keyboard:
            keyboard.leave()
        if screen:
            screen.leave()
        elif interactive:
            clear()
            sys.stdout.write("\n")

    # After the screen is restored, so the package manager's own output is
    # visible on the normal terminal rather than painted over by a frame.
    if upgrading:
        return run_upgrade(style)

    # The alternate buffer takes the whole session with it when it closes, so
    # hand the numbers back on the normal screen before exiting.
    if tui and stats.by_model:
        sys.stdout.write("\nllmwatch session summary\n")
        for model in sorted(stats.by_model):
            snap = stats.snapshot(model)
            sys.stdout.write("\n  %s  (%d requests)\n" % (model, snap["requests"]))
            plain = Style(color=style.color, unicode_ok=style.unicode, width=style.width)
            for text in render_board(snap, plain, style.width, compact=True):
                sys.stdout.write("  " + text + "\n")
    return 0


def summarise_last(args):
    """Replay a chunk of the log and print the most recent completed request."""
    kind, target = find_log()
    if kind != "file":
        sys.stderr.write("llmwatch --last needs a log file (journalctl not supported yet)\n")
        return 2
    try:
        with open(target, "r", errors="replace") as fh:
            lines = fh.readlines()[-8000:]
    except OSError as exc:
        sys.stderr.write("llmwatch: cannot read %s: %s\n" % (target, exc))
        return 2

    tracker = Tracker()
    tracker.model = current_model()
    style = Style.detect(no_color=args.no_color)
    groups = []
    pending = {}
    for line in lines:
        for out in tracker.feed(parse_line(line)):
            name = out.data.get("event")
            if name in ("prefill_done", "generate_done"):
                slot = pending.setdefault(out.data["task"], {})
                slot["prefill" if name == "prefill_done" else "generation"] = out.data
            elif name == "request_end":
                slot = pending.pop(out.data["task"], {})
                groups.append((slot.get("prefill"), slot.get("generation"), out.data))

    if not groups:
        sys.stderr.write("llmwatch: no completed request found in the recent log\n")
        return 1

    prefill, generation, end = groups[-1]
    if args.json:
        for part in (prefill, generation, end):
            if part:
                sys.stdout.write(json.dumps(part) + "\n")
        return 0

    sys.stdout.write(render_header(end, style, show_time=False) + "\n")
    for text in render_summary(prefill, generation, end, style):
        sys.stdout.write(text + "\n")
    return 0


def show_history(args, out=sys.stdout, history=None, now=None):
    hist = history or History()
    rows = hist.rollup(days=args.days, model=args.model, now=now)
    if not rows:
        out.write("no history yet for that window - run llmwatch alongside your agent "
                  "and it will fill in\n")
        return 1
    out.write("last %d day%s\n\n" % (args.days, "" if args.days == 1 else "s"))
    out.write("  %-26s %6s %12s %12s %10s %s\n"
              % ("model", "req", "prefill", "generate", "cache", "vs previous"))
    for row in rows:
        change = ""
        if row["gen_change_pct"] is not None:
            arrow = "up" if row["gen_change_pct"] >= 0 else "down"
            change = "%s %.0f%%" % (arrow, abs(row["gen_change_pct"]))
        elif row["previous_requests"] == 0:
            change = "no prior data"
        out.write("  %-26s %6d %9s %11s %9s   %s\n" % (
            row["model"][:26], row["requests"],
            ("%.1f tok/s" % row["prefill_rate"]) if row["prefill_rate"] else "-",
            ("%.1f tok/s" % row["gen_rate"]) if row["gen_rate"] else "-",
            ("%.0f%%" % row["cache_pct"]) if row["cache_pct"] is not None else "-",
            change))
    return 0


def show_compare(args, out=sys.stdout, history=None, now=None):
    """Renders through build_compare, the same path the in-TUI view uses."""
    model_a, model_b = args.compare
    hist = history or History()
    style = Style.detect(no_color=getattr(args, "no_color", False))
    out.write("A  %s\nB  %s\n" % (model_a, model_b))
    out.write("compared within prompt-size and cache buckets, because a cached "
              "244-token\nrequest and an uncached 47k one are not the same workload\n\n")
    lines = build_compare(hist, model_a, model_b, style, days=args.days, now=now)
    for line in lines:
        out.write(line + "\n")
    return 0


def show_turns(args, out=sys.stdout, history=None, now=None):
    """How long a turn takes, per model and effort level.

    The question this answers is the one you ask before sending a prompt, not
    after: "if I send this to that model at high effort, am I waiting one minute
    or twenty?"
    """
    hist = history or History()
    rows = hist.turn_rollup(days=args.days, model=args.model, now=now)
    if not rows:
        out.write("no turns recorded yet - run llmwatch --codex alongside your agent "
                  "and it will fill in\n")
        return 1
    out.write("last %d day%s\n\n" % (args.days, "" if args.days == 1 else "s"))
    out.write("  %-26s %8s %6s %10s %10s %s\n"
              % ("model", "effort", "turns", "typical", "longest", "tool calls"))
    for row in rows:
        note = ""
        if row["interrupted"]:
            note = "   (%d interrupted, not timed)" % row["interrupted"]
        out.write("  %-26s %8s %6d %10s %10s %10s%s\n" % (
            row["model"][:26], row["effort"], row["turns"],
            fmt_duration(row["median_seconds"]) if row["median_seconds"] else "-",
            fmt_duration(row["longest_seconds"]) if row["longest_seconds"] else "-",
            # A turn that ran no tools is a real answer, so 0 must print as 0.
            ("%.0f" % row["median_tool_calls"])
            if row["median_tool_calls"] is not None else "-",
            note))
    out.write("\nwall clock from your prompt to the final answer, tool time included\n")
    return 0


def export_history(args, out=sys.stdout, history=None, now=None):
    hist = history or History()
    if getattr(args, "turns", False):
        rows = hist.turn_rows(days=args.days, now=now)
    else:
        rows = hist.all_rows(days=args.days, now=now)
    if args.export == "json":
        out.write(json.dumps(rows, indent=2) + "\n")
        return 0
    if not rows:
        return 1
    columns = list(rows[0].keys())
    out.write(",".join(columns) + "\n")
    for row in rows:
        out.write(",".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llmwatch",
        description="Live prefill progress and tok/s for a local Ollama model.")
    parser.add_argument("--last", action="store_true",
                        help="summarise the most recent completed request and exit")
    parser.add_argument("--json", action="store_true",
                        help="emit one JSON object per event (for status bars/scripts)")
    parser.add_argument("--log", metavar="PATH",
                        help="path to the Ollama log (or an mlx_lm.server log, "
                             "which adds prefill progress under --proxy)")
    parser.add_argument("--proxy", nargs="?", const="", metavar="[HOST:]PORT",
                        help="watch an OpenAI-compatible server (mlx_lm.server, "
                             "LM Studio, llama-server, vLLM) by proxying it. "
                             "Point your client at this address instead. "
                             "Default 127.0.0.1:%d" % DEFAULT_PROXY_PORT)
    parser.add_argument("--upstream", metavar="URL",
                        help="server --proxy forwards to (default %s)" % DEFAULT_UPSTREAM)
    parser.add_argument("--proxy-allow-remote", action="store_true",
                        help="permit --proxy to bind a non-loopback address")
    parser.add_argument("--history", action="store_true",
                        help="per-model summary from your recorded history")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                        help="compare two models on like-for-like recorded requests")
    parser.add_argument("--turns", action="store_true",
                        help="how long a whole agent turn takes, by model and "
                             "reasoning effort (needs --codex to have been recording)")
    parser.add_argument("--export", choices=("json", "csv"),
                        help="dump recorded history (add --turns for turn records)")
    parser.add_argument("--days", type=int, default=7,
                        help="window for --history/--turns/--compare/--export (default 7)")
    parser.add_argument("--model", help="restrict --history/--turns to one model")
    parser.add_argument("--no-history", action="store_true",
                        help="do not record this session (timings only are ever stored)")
    parser.add_argument("--codex", action="store_true",
                        help="show what Codex is doing, read from its session file "
                             "(contains commands and file paths)")
    parser.add_argument("--plain", action="store_true",
                        help="scrolling single-line output instead of the full-screen board")
    parser.add_argument("--no-color", action="store_true", help="disable colour output")
    parser.add_argument("--debug-unparsed", action="store_true",
                        help="print timing-ish lines that failed to parse (bug reports)")
    parser.add_argument("--version", action="version", version="llmwatch " + __version__)
    args = parser.parse_args(argv)

    if args.log:
        os.environ["LLMWATCH_LOG"] = args.log
    if args.export:
        return export_history(args)        # --turns selects which table
    if args.history:
        return show_history(args)
    if args.turns:
        return show_turns(args)
    if args.compare:
        return show_compare(args)
    return summarise_last(args) if args.last else follow(args)
