"""Finding the Ollama log, and the command that tails it.

Small and dependency-free on purpose: everything above it needs to locate a log
before anything else can start.
"""
import os
import subprocess


# --------------------------------------------------------------------------
# Layer 4: I/O
# --------------------------------------------------------------------------

MACOS_HOMEBREW = ["/opt/homebrew/var/log/ollama.log", "/usr/local/var/log/ollama.log"]

MACOS_APP = ["~/.ollama/logs/server.log", "~/Library/Logs/Ollama/server.log"]

LINUX_PATHS = ["/var/log/ollama.log", "~/.ollama/logs/server.log"]


def candidate_paths():
    return [os.path.expanduser(p) for p in MACOS_HOMEBREW + MACOS_APP + LINUX_PATHS]


def find_log():
    """Return (kind, target). kind is 'file' or 'journalctl'."""
    env = os.environ.get("LLMWATCH_LOG")
    if env:
        return ("file", os.path.expanduser(env))
    for path in candidate_paths():
        if os.path.isfile(path):
            return ("file", path)
    # Linux systemd: no file, but the journal has the same lines. Experimental --
    # unverified by the author, please report issues.
    if _has_journal():
        return ("journalctl", "ollama")
    return (None, None)


def _has_journal():
    try:
        r = subprocess.run(["journalctl", "--version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def tail_command(kind, target, from_start=False):
    if kind == "journalctl":
        return ["journalctl", "-fu", target, "-n", "0" if not from_start else "2000"]
    return ["tail", "-n", "0" if not from_start else "+1", "-F", target]


def current_model():
    """The log only names a model when one is LOADED, so an already-warm model
    would otherwise display as '?'."""
    try:
        rows = subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                              timeout=5).stdout.splitlines()
        if len(rows) > 1 and rows[1].strip():
            return rows[1].split()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "?"
