"""The settings file: reading it, writing it, and merging it with argv.

effective_settings is the one place that decides what llmwatch is actually
going to watch, given a config file, the environment and the command line.
"""
import json
import os

from .constants import DEFAULT_PROXY_PORT, DEFAULT_UPSTREAM
from .events import OaiPrefillTick
from .oai import parse_listen


# --------------------------------------------------------------------------
# Saved settings
#
# So that `ollama-llmwatch` on its own does what you configured, rather than
# needing the same flags typed every time. The file is the least authoritative
# source there is: anything typed now beats it, because a setting saved months
# ago must never silently overrule the command in front of you.
# --------------------------------------------------------------------------

WATCH_MODES = ("ollama", "proxy", "log")

CONFIG_KEYS = ("watch", "proxy_port", "upstream", "log")


def config_path():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "ollama-llmwatch", "config.json")


def read_config(path=None):
    """Saved settings, or {} if there are none that can be used.

    Never raises and never blocks startup. A settings file that cannot be read
    is a reason to fall back to defaults, not a reason to refuse to run, and an
    unrecognised key is dropped rather than kept: the file is hand-edited as
    often as not, and a typo should not become a setting that does nothing
    forever without saying so.
    """
    try:
        with open(path or config_path()) as fh:
            obj = json.load(fh)
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(obj, dict):
        return {}
    clean = {k: v for k, v in obj.items() if k in CONFIG_KEYS}
    if clean.get("watch") not in WATCH_MODES:
        clean.pop("watch", None)
    return clean


def write_config(data, path=None):
    """Save settings. True if it worked.

    Returns rather than raises for the same reason as above: failing to save a
    preference must not take the screen down with it.
    """
    target = path or config_path()
    try:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w") as fh:
            json.dump({k: v for k, v in data.items() if k in CONFIG_KEYS},
                      fh, indent=2, sort_keys=True)
            fh.write("\n")
        return True
    except (OSError, ValueError, TypeError):
        return False


def effective_settings(args, env, config):
    """{setting: (value, source)} with source in flag/env/config/default.

    The source is carried because a settings file invites exactly one question
    the moment anything surprises you, which is where the value came from. The
    settings pane answers it without anyone having to guess.
    """
    log_arg = getattr(args, "log", None)
    proxy_arg = getattr(args, "proxy", None)
    upstream_arg = getattr(args, "upstream", None)

    if proxy_arg is not None:
        watch = ("proxy", "flag")
    elif log_arg:
        watch = ("log", "flag")
    elif env.get("LLMWATCH_PROXY"):
        watch = ("proxy", "env")
    elif env.get("LLMWATCH_LOG"):
        watch = ("log", "env")
    elif config.get("watch"):
        watch = (config["watch"], "config")
    else:
        watch = ("ollama", "default")

    if proxy_arg:
        port = (parse_listen(proxy_arg)[1], "flag")
    elif env.get("LLMWATCH_PROXY"):
        port = (parse_listen(env["LLMWATCH_PROXY"])[1], "env")
    elif config.get("proxy_port"):
        port = (config["proxy_port"], "config")
    else:
        port = (DEFAULT_PROXY_PORT, "default")

    if upstream_arg:
        upstream = (upstream_arg, "flag")
    elif env.get("LLMWATCH_UPSTREAM"):
        upstream = (env["LLMWATCH_UPSTREAM"], "env")
    elif config.get("upstream"):
        upstream = (config["upstream"], "config")
    else:
        upstream = (DEFAULT_UPSTREAM, "default")

    if log_arg:
        log = (log_arg, "flag")
    elif env.get("LLMWATCH_LOG"):
        log = (env["LLMWATCH_LOG"], "env")
    elif config.get("log"):
        log = (config["log"], "config")
    else:
        log = (None, "default")

    return {"watch": watch, "proxy_port": port,
            "upstream": upstream, "log": log}


def log_event_allowed(ev, proxying):
    """May this log event be fed to the tracker?

    Yes, unless --proxy is measuring the same requests from the wire. Both
    sources describe the same traffic, so counting both records every request
    twice, and the slot ids collide as well: OAI_SLOT is 1 and llama-server
    hands out slots from 1 upward, so they also fight over the same live
    request.

    The wire wins because it is the source that is definitely present: --proxy
    is used precisely when the log cannot be relied on. The one thing kept from
    the log is prefill progress from a standalone mlx_lm.server, which the wire
    genuinely cannot show -- the whole prefill happens before the first byte is
    sent, so without this the bar sits at "starting" for minutes.

    For llama.cpp the log is richer than the wire, carrying exact prefill and
    generation rates and per-slot detail. Watch it with --log on its own rather
    than both: this returns the wire's numbers, not the better ones.
    """
    if not proxying:
        return True
    return isinstance(ev, OaiPrefillTick)
