"""llmwatch - see what your local Ollama model is actually doing.

Local LLMs spend most of a request READING your prompt (prefill), not writing the
answer (generation). On an M1 Max with a dense 27B Q4 model, a 47k-token prompt
takes ~8 minutes of prefill before the first character appears, while the answer
itself takes ~20 seconds. Roughly 96% of the wait is invisible: your coding agent
shows a spinner and nothing else.

Ollama's HTTP API cannot tell you about that window -- it emits its first chunk
only after prefill finishes. llama.cpp added a `prompt_progress` field for exactly
this (ggml-org/llama.cpp#14685) but Ollama does not pass it through. So the server
log is currently the only place the information exists.

Ollama serves GGUF models with llama-server and `-mlx` models with its own MLX
runner. The two log nothing alike, so both dialects are parsed, into one set of
events; nothing above the parser knows which engine ran.

llmwatch tails that log and answers the question the spinner can't:
    "is it still reading my prompt, or is it actually writing -- and how long
     until it starts?"

Structure: four layers, the first three pure and testable.
    parse_line(line) -> Event        pure, no I/O      parser.py
    Tracker.feed(event) -> [Output]  pure state machine tracker.py
    render_live(...) / render_*      pure formatting    render.py and friends
    follow()                         all the I/O        cli.py

Those layers used to be four labelled sections of one 6,000-line file. They are
now modules, ordered so that imports only ever point downwards: constants and
events at the bottom, cli at the top. Nothing here imports the package root, so
there is exactly one direction of travel and no cycles.

Every name is re-exported below, so `from llmwatch import parse_line` means the
same thing it always did; the split is an internal reorganisation, not a change
of API.

llmwatch also ships as a single generated file. tools/bundle.py concatenates
these modules, in the order their imports imply, into the llmwatch.py at the
repository root, which is what

    curl -O .../llmwatch.py && ./llmwatch.py

downloads. Both builds are tested on every push. Edit the package, then run
`python tools/bundle.py`.

Requires Python 3.9+. Standard library only, on purpose."""

from .constants import __version__  # noqa: F401

# Re-exported so `from llmwatch import parse_line` keeps working:
# the split is an internal reorganisation, not a change of API.
from .constants import (  # noqa: F401
    DEFAULT_PROXY_PORT, DEFAULT_UPSTREAM, FRAME_ACTIVE, FRAME_IDLE, GEN_TICK_TOKENS,
    LONG_CONTEXT_TOKENS, LOOPBACK_HOSTS, LOOP_ESCALATE, LOOP_REPEATS,
    LOW_DRAFT_ACCEPT, MIN_COMPARE_SAMPLES, MIN_COMPARE_TURNS, MIN_FRAME_GAP,
    MIN_MEASURABLE_SECONDS, MIN_TOKENS_FOR_EXTREMES, MIN_TURNS_FOR_TYPICAL,
    OAI_PROBE_PORTS, OAI_PROBE_TIMEOUT, PREFILL_BATCH, RECENT_LIMIT, SAME_WITHIN,
    SETTINGS_PRESETS, SLOWDOWN_MIN_SAMPLES, SLOWDOWN_RATIO, TURN_TYPICAL_DAYS,
    UPDATE_INTERVAL, UPDATE_TIMEOUT, UPDATE_URL, UPSTREAM_SCHEMES,)
from .events import (  # noqa: F401
    CacheInfo, CacheMiss, CheckpointCreated, CheckpointErased, CheckpointRestored,
    DraftAcceptance, GenDone, GenTick, MLX_EVENTS, MlxGenStats, MlxPeakMemory,
    MlxPrefillTick, MlxRequestEnd, MlxRequestStart, MlxRunnerReady, MlxRunnerStart,
    ModelLoaded, OAI_EVENTS, OaiEngine, OaiGenTick, OaiPrefillTick, OaiRequestAborted,
    OaiRequestEnd, OaiRequestStart, Output, PrefillDone, PrefillTick, RequestEnd,
    RequestStart, ServerStarted,)
from .text import (  # noqa: F401
    RE_ANSI, RE_CONTROL, SPARK_ASCII, SPARK_UNICODE, SPINNER_ASCII, SPINNER_UNICODE,
    Style, fmt_bar, fmt_duration, safe_text, short_model_name, spinner_frame,
    truncate_visible, visible_len,)
from .parser import (  # noqa: F401
    RE_CACHED, RE_CACHE_MISS, RE_CKPT_CREATED, RE_CKPT_ERASED, RE_CKPT_RESTORED,
    RE_DRAFT, RE_GEN_DONE, RE_GEN_TICK, RE_GO_DURATION, RE_MLXS_PREFILL, RE_MLXS_TS,
    RE_MLX_CACHE, RE_MLX_DONE, RE_MLX_PEAK, RE_MLX_PREFILL, RE_MLX_RUNNER_READY,
    RE_MLX_RUNNER_START, RE_MLX_SPEC, RE_MLX_TS, RE_MODEL, RE_PREFILL_DONE,
    RE_PREFILL_TICK, RE_REQ_START, RE_SERVER_STARTED, RE_TOTAL, _GO_UNITS, _SLOT,
    looks_like_timing, parse_go_duration, parse_line, parse_mlx_line,
    parse_mlx_server_line, parse_mlx_server_timestamp, parse_mlx_timestamp,)
from .tracker import (  # noqa: F401
    Request, Tracker,)
from .stats import (  # noqa: F401
    ModelStats, PhaseStats, Stats,)
from .logs import (  # noqa: F401
    LINUX_PATHS, MACOS_APP, MACOS_HOMEBREW, _has_journal, candidate_paths,
    current_model, find_log, tail_command,)
from .oai import (  # noqa: F401
    _fetch_openai_models, detect_engine, detect_openai_server, engine_from_log_event,
    list_upstream_models, model_name_for_board, parse_listen, valid_upstream,)
from .config import (  # noqa: F401
    CONFIG_KEYS, WATCH_MODES, config_path, effective_settings, log_event_allowed,
    read_config, write_config,)
from .proxy import (  # noqa: F401
    HOP_BY_HOP, INSTRUMENTED_PATHS, MAX_BUFFERED_BODY, PROXY_TIMEOUT,
    _PROXY_SERVER_CLASS, _proxy_server_class, _sse_events, draft_counts, prepare_body,
    read_stream_fingerprint, read_stream_usage, safe_request_path, start_proxy,
    usage_counts,)
from .clients import (  # noqa: F401
    BACKUP_SUFFIX, KNOWN_CLIENTS, _url_pattern, clients_elsewhere, find_clients,
    point_client_at, read_client_url, undo_client,)
from .system import (  # noqa: F401
    SystemProbe, busiest_process,)
from .render import (  # noqa: F401
    ProgressFloor, _rate, detect_slowdown, diagnose, project, project_completion,
    render_board, render_board_title, render_codex, render_header, render_idle,
    render_last_turn, render_live, render_live_detail, render_recent, render_summary,
    render_system, sparkline,)
from .discovery import (  # noqa: F401
    DISCOVER_PORTS, LOG_SEARCH_DIRS, LOG_SEARCH_FILES, LOG_SEARCH_MAX_AGE,
    LOG_SEARCH_TAIL, OLLAMA_PORT, _discover_logs, _discover_servers,
    backend_suggestion, describe_backend, discover_backends, identify_log, preset_for,)
from .settings import (  # noqa: F401
    SETTINGS_TOP, SETTINGS_WHERE, _BACK, _ENTER, _settings_rows, _settings_summary,
    render_settings, settings_config, settings_key, settings_open,)
from .uistate import (  # noqa: F401
    HELP_TEXT, UIState, handle_key, render_help, render_picker,)
from .history import (  # noqa: F401
    History, SIZE_BANDS, size_band,)
from .compare import (  # noqa: F401
    _pair_bars, _verdict, build_compare, installed_models, median_request_time,
    pickable_models, render_compare, render_median_request, render_turns,)
from .update import (  # noqa: F401
    _UpgradeRequested, _git_is_dirty, check_for_update, fetch_latest_version,
    install_path, is_single_file, read_update_cache, render_update,
    render_upgrade_confirm, run_upgrade, start_update_check, update_cache_path,
    update_check_disabled, update_is_newer,
    upgrade_command, upgrade_plan, version_tuple, write_update_cache,)
from .screen import (  # noqa: F401
    FRAME_MARGIN, Screen, compose_frame, frame_lines, plan_frame,)
from .codex import (  # noqa: F401
    CodexTail,)
from .keyboard import (  # noqa: F401
    ARROW_KEYS, ESC_TIMEOUT, Keyboard, _key_reader, _reader, decode_keys,)
from .cli import (  # noqa: F401
    export_history, follow, main, show_compare, show_history, show_turns,
    summarise_last,)

__all__ = [
    "DEFAULT_PROXY_PORT", "DEFAULT_UPSTREAM", "FRAME_ACTIVE", "FRAME_IDLE",
    "GEN_TICK_TOKENS", "LONG_CONTEXT_TOKENS", "LOOPBACK_HOSTS", "LOOP_ESCALATE",
    "LOOP_REPEATS", "LOW_DRAFT_ACCEPT", "MIN_COMPARE_SAMPLES", "MIN_COMPARE_TURNS",
    "MIN_FRAME_GAP", "MIN_MEASURABLE_SECONDS", "MIN_TOKENS_FOR_EXTREMES",
    "MIN_TURNS_FOR_TYPICAL", "OAI_PROBE_PORTS", "OAI_PROBE_TIMEOUT", "PREFILL_BATCH",
    "RECENT_LIMIT", "SAME_WITHIN", "SETTINGS_PRESETS", "SLOWDOWN_MIN_SAMPLES",
    "SLOWDOWN_RATIO", "TURN_TYPICAL_DAYS", "UPDATE_INTERVAL", "UPDATE_TIMEOUT",
    "UPDATE_URL", "UPSTREAM_SCHEMES", "CacheInfo", "CacheMiss", "CheckpointCreated",
    "CheckpointErased", "CheckpointRestored", "DraftAcceptance", "GenDone", "GenTick",
    "MLX_EVENTS", "MlxGenStats", "MlxPeakMemory", "MlxPrefillTick", "MlxRequestEnd",
    "MlxRequestStart", "MlxRunnerReady", "MlxRunnerStart", "ModelLoaded",
    "OAI_EVENTS", "OaiEngine", "OaiGenTick", "OaiPrefillTick", "OaiRequestAborted",
    "OaiRequestEnd", "OaiRequestStart", "Output", "PrefillDone", "PrefillTick",
    "RequestEnd", "RequestStart", "ServerStarted", "RE_ANSI", "RE_CONTROL",
    "SPARK_ASCII", "SPARK_UNICODE", "SPINNER_ASCII", "SPINNER_UNICODE", "Style",
    "fmt_bar", "fmt_duration", "safe_text", "short_model_name", "spinner_frame",
    "truncate_visible", "visible_len", "RE_CACHED", "RE_CACHE_MISS",
    "RE_CKPT_CREATED", "RE_CKPT_ERASED", "RE_CKPT_RESTORED", "RE_DRAFT",
    "RE_GEN_DONE", "RE_GEN_TICK", "RE_GO_DURATION", "RE_MLXS_PREFILL", "RE_MLXS_TS",
    "RE_MLX_CACHE", "RE_MLX_DONE", "RE_MLX_PEAK", "RE_MLX_PREFILL",
    "RE_MLX_RUNNER_READY", "RE_MLX_RUNNER_START", "RE_MLX_SPEC", "RE_MLX_TS",
    "RE_MODEL", "RE_PREFILL_DONE", "RE_PREFILL_TICK", "RE_REQ_START",
    "RE_SERVER_STARTED", "RE_TOTAL", "_GO_UNITS", "_SLOT", "looks_like_timing",
    "parse_go_duration", "parse_line", "parse_mlx_line", "parse_mlx_server_line",
    "parse_mlx_server_timestamp", "parse_mlx_timestamp", "Request", "Tracker",
    "ModelStats", "PhaseStats", "Stats", "LINUX_PATHS", "MACOS_APP", "MACOS_HOMEBREW",
    "_has_journal", "candidate_paths", "current_model", "find_log", "tail_command",
    "_fetch_openai_models", "detect_engine", "detect_openai_server",
    "engine_from_log_event", "list_upstream_models", "model_name_for_board",
    "parse_listen", "valid_upstream", "CONFIG_KEYS", "WATCH_MODES", "config_path",
    "effective_settings", "log_event_allowed", "read_config", "write_config",
    "HOP_BY_HOP", "INSTRUMENTED_PATHS", "MAX_BUFFERED_BODY", "PROXY_TIMEOUT",
    "_PROXY_SERVER_CLASS", "_proxy_server_class", "_sse_events", "draft_counts",
    "prepare_body", "read_stream_fingerprint", "read_stream_usage",
    "safe_request_path", "start_proxy", "usage_counts", "BACKUP_SUFFIX",
    "KNOWN_CLIENTS", "_url_pattern", "clients_elsewhere", "find_clients",
    "point_client_at", "read_client_url", "undo_client", "SystemProbe",
    "busiest_process", "ProgressFloor", "_rate", "detect_slowdown", "diagnose",
    "project", "project_completion", "render_board", "render_board_title",
    "render_codex", "render_header", "render_idle", "render_last_turn", "render_live",
    "render_live_detail", "render_recent", "render_summary", "render_system",
    "sparkline", "DISCOVER_PORTS", "LOG_SEARCH_DIRS", "LOG_SEARCH_FILES",
    "LOG_SEARCH_MAX_AGE", "LOG_SEARCH_TAIL", "OLLAMA_PORT", "_discover_logs",
    "_discover_servers", "backend_suggestion", "describe_backend",
    "discover_backends", "identify_log", "preset_for", "SETTINGS_TOP",
    "SETTINGS_WHERE", "_BACK", "_ENTER", "_settings_rows", "_settings_summary",
    "render_settings", "settings_config", "settings_key", "settings_open",
    "HELP_TEXT", "UIState", "handle_key", "render_help", "render_picker", "History",
    "SIZE_BANDS", "size_band", "_pair_bars", "_verdict", "build_compare",
    "installed_models", "median_request_time", "pickable_models", "render_compare",
    "render_median_request", "render_turns", "_UpgradeRequested", "_git_is_dirty",
    "check_for_update", "fetch_latest_version", "read_update_cache", "render_update",
    "render_upgrade_confirm", "run_upgrade", "start_update_check",
    "update_cache_path", "update_check_disabled", "update_is_newer",
    "install_path", "is_single_file",
    "upgrade_command", "upgrade_plan", "version_tuple", "write_update_cache",
    "FRAME_MARGIN", "Screen", "compose_frame", "frame_lines", "plan_frame",
    "CodexTail", "ARROW_KEYS", "ESC_TIMEOUT", "Keyboard", "_key_reader", "_reader",
    "decode_keys", "export_history", "follow", "main", "show_compare", "show_history",
    "show_turns", "summarise_last",
]
