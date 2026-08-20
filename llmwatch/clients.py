"""Pointing other people's apps at llmwatch, and putting them back.

Editing someone else's config file is the most destructive thing llmwatch does,
so every write is backed up first and every one of them is reversible.
"""
import os
import re


# --------------------------------------------------------------------------
# Pointing other people's apps at llmwatch
#
# The proxy only sees what is routed through it, and a client left pointing at
# the model server bypasses it silently: everything works, nothing is measured,
# and the board sits empty looking like llmwatch is broken. That has to be
# fixable without knowing what a base URL is.
#
# Their files, so: never automatic, one at a time, backed up first, and edited
# by replacing the one string rather than rewriting the file, which would
# reformat it and drop comments.
# --------------------------------------------------------------------------

KNOWN_CLIENTS = (
    ("opencode", "~/.config/opencode/opencode.json", "baseURL"),
    ("Codex", "~/.codex/config.toml", "openai_base_url"),
    ("Continue", "~/.continue/config.json", "apiBase"),
    ("Aider", "~/.aider.conf.yml", "openai-api-base"),
)

BACKUP_SUFFIX = ".llmwatch-backup"


def _url_pattern(key):
    """Matches `key: "url"` and `key = "url"`, quoted key or not.

    One pattern for JSON, TOML and YAML because all three spell this the same
    way once the value is a quoted string, and the alternative is three parsers
    and three writers that each reformat somebody's file.
    """
    return re.compile(r'("?%s"?\s*[:=]\s*")([^"]*)(")' % re.escape(key))


def read_client_url(path, key):
    """What this client currently points at, or None."""
    try:
        with open(os.path.expanduser(path)) as fh:
            text = fh.read()
    except (OSError, ValueError):
        return None
    match = _url_pattern(key).search(text)
    return match.group(2) if match else None


def find_clients(clients=KNOWN_CLIENTS):
    """Installed clients and where each is currently pointed.

    Only ones that exist and that a URL could actually be read from: a config
    llmwatch cannot understand is one it must not offer to edit.
    """
    found = []
    for name, path, key in clients:
        full = os.path.expanduser(path)
        if not os.path.isfile(full):
            continue
        url = read_client_url(full, key)
        if url is None:
            continue
        found.append({"name": name, "path": full, "key": key, "url": url})
    return found


def point_client_at(entry, url):
    """Repoint one client. Returns (ok, message).

    Backed up first, and the backup is what `undo` restores. Only the URL is
    replaced, so comments, key order and formatting survive: this is somebody's
    editor config, not ours to tidy.
    """
    path = entry["path"]
    try:
        with open(path) as fh:
            text = fh.read()
    except (OSError, ValueError) as err:
        return False, "could not read %s (%s)" % (path, err)

    pattern = _url_pattern(entry["key"])
    if not pattern.search(text):
        return False, "no %s found in %s" % (entry["key"], path)
    updated = pattern.sub(lambda m: m.group(1) + url + m.group(3), text, count=1)
    if updated == text:
        return True, "%s already points at %s" % (entry["name"], url)

    try:
        with open(path + BACKUP_SUFFIX, "w") as fh:
            fh.write(text)
        with open(path, "w") as fh:
            fh.write(updated)
    except (OSError, ValueError) as err:
        return False, "could not write %s (%s)" % (path, err)
    return True, "%s now points at %s (restart it)" % (entry["name"], url)


def clients_elsewhere(clients, listen_url):
    """Known clients configured to talk to something other than llmwatch.

    llmwatch cannot see traffic that does not pass through it: a client talking
    to another port is a conversation between two other processes, and nothing
    short of packet capture would show it. The configuration is visible though,
    and it is the better signal anyway, because it can be said before the
    request rather than discovered after it.
    """
    if not listen_url:
        return []
    here = listen_url.rstrip("/")
    return [row for row in (clients or [])
            if (row.get("url") or "").rstrip("/") != here]


def undo_client(entry):
    """Put back what was there before. Returns (ok, message)."""
    backup = entry["path"] + BACKUP_SUFFIX
    try:
        with open(backup) as fh:
            text = fh.read()
        with open(entry["path"], "w") as fh:
            fh.write(text)
        os.remove(backup)
    except (OSError, ValueError) as err:
        return False, "nothing to undo for %s (%s)" % (entry["name"], err)
    return True, "%s restored" % entry["name"]
