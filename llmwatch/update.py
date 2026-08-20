"""The daily update check, and upgrading in place.

The only thing in llmwatch that touches the network unprompted: one GET to
PyPI's public JSON, at most once a day, on a background thread, with every
failure swallowed.

The upgrade path is worked out from where this copy actually lives, because
telling a pipx user to run uv is worse than saying nothing.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from .constants import UPDATE_INTERVAL, UPDATE_TIMEOUT, UPDATE_URL, __version__


# --------------------------------------------------------------------------
# The update check
#
# Everything here fails silent. A tool that watches something else must never
# be the thing that breaks, and least of all over its own version number.
# --------------------------------------------------------------------------

def version_tuple(text):
    """Comparable form of a plain X.Y.Z version, or None for anything else.

    Deliberately refuses suffixes (rc, dev, post). Guessing how a pre-release
    orders against a release is how an update prompt ends up telling someone to
    "upgrade" to something older than what they already have.
    """
    parts = (text or "").strip().split(".")
    if not 1 <= len(parts) <= 4 or not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def update_is_newer(current, latest):
    """Is `latest` a release worth telling someone about? Unknown means no."""
    here, there = version_tuple(current), version_tuple(latest)
    if here is None or there is None:
        return False
    return there > here


def update_cache_path():
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "ollama-llmwatch", "update-check.json")


def read_update_cache(now, path=None):
    """(latest, fresh) from the cache. `fresh` decides whether to skip the call.

    A failed check caches too, as an empty result: an unreachable network should
    cost one attempt a day, not one on every launch.
    """
    try:
        with open(path or update_cache_path()) as fh:
            data = json.load(fh)
        checked = float(data.get("checked") or 0)
        return data.get("latest"), (now - checked) < UPDATE_INTERVAL
    except (OSError, ValueError, AttributeError):
        return None, False


def write_update_cache(latest, now, path=None):
    target = path or update_cache_path()
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as fh:
            json.dump({"latest": latest, "checked": now}, fh)
    except (OSError, ValueError):
        pass          # a cache we cannot write just means we check again tomorrow


def fetch_latest_version(url=UPDATE_URL, timeout=UPDATE_TIMEOUT):
    """The newest version on PyPI, or None if anything at all goes wrong.

    The single network call in this program. It is a GET of a public JSON
    document, identical for every user, with no body and nothing appended to the
    URL: PyPI learns that somebody asked, which it already knew from the install.
    """
    import urllib.request          # local: nothing imports a network client
                                   # unless this function is actually called
    # urlopen will happily follow file:, ftp: or a custom scheme, which turns a
    # url that ever becomes caller-controlled into a file read. It is a constant
    # today; this keeps that true no matter who calls it later.
    if not (url or "").startswith("https://"):
        return None
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ollama-llmwatch/%s" % __version__})
        # B310 is silenced because the scheme is checked above, not ignored.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            if getattr(response, "status", 200) != 200:
                return None
            payload = json.loads(response.read(64 * 1024).decode("utf-8", "replace"))
        return (payload.get("info") or {}).get("version")
    except Exception:              # noqa: BLE001 - DNS, TLS, timeout, junk JSON
        return None                # are all the same answer here: say nothing


def update_check_disabled():
    """Off by env var, and off for --json, which must stay machine-readable and
    is the mode most likely to be running unattended in a script."""
    value = (os.environ.get("LLMWATCH_NO_UPDATE_CHECK") or "").strip().lower()
    return value not in ("", "0", "false", "no")


def check_for_update(current=None, now=None, path=None, fetch=None):
    """The whole check, start to finish. Returns a version string or None.

    Pure enough to test: the clock, the cache location and the fetch are all
    injectable, so no test needs a network or a real home directory.
    """
    current = current or __version__
    now = time.time() if now is None else now
    latest, fresh = read_update_cache(now, path)
    if not fresh:
        latest = (fetch or fetch_latest_version)()
        write_update_cache(latest, now, path)
    return latest if update_is_newer(current, latest) else None


def start_update_check(box, current=None, path=None, fetch=None):
    """Run the check on a daemon thread and leave the answer in `box`.

    On a thread because a stalled DNS lookup must not delay the first frame by
    even the timeout: the board is on screen before this has finished, and the
    notice appears whenever it arrives.
    """
    if update_check_disabled():
        return None

    def run():
        try:
            found = check_for_update(current=current, path=path, fetch=fetch)
            if found:
                box["latest"] = found
        except Exception:          # noqa: BLE001 - never take the tool down
            pass

    thread = threading.Thread(target=run)
    thread.daemon = True           # never hold up exit waiting on the network
    thread.start()
    return thread


def install_path():
    """The path that identifies this copy of llmwatch.

    llmwatch ships as two builds of the same program: the llmwatch/ package, and
    the single llmwatch.py that tools/bundle.py generates for people who curl it.
    `__file__` answers a different question in each -- in the package it names
    whichever module happened to ask, which is one directory too deep -- so the
    checks below would look for a checkout's .git inside llmwatch/ and never
    find it.

    So: the package directory when running as a package, the script itself when
    running as one file. Both are "the thing that was installed".
    """
    here = os.path.abspath(__file__)
    if __package__:
        return os.path.dirname(here)        # .../llmwatch
    return here                             # .../llmwatch.py


def is_single_file(path):
    """True when this copy is the curled script rather than an install tree.

    A directory is never the single-file case, and neither is anything a
    package manager put in place; those are handled before this is reached.
    """
    return not os.path.isdir(path)


def upgrade_command(path=None):
    """The command that upgrades *this* copy, worked out from where it lives.

    There are four supported ways to install this and they upgrade differently.
    Telling a pipx user to run uv, or someone who curled a single file to run a
    package manager they never used, is worse than saying nothing: the advice
    fails in front of them and they learn to ignore the line.
    """
    path = os.path.abspath(path or install_path()).replace("\\", "/")
    lowered = path.lower()
    if "/uv/tools/" in lowered:
        return "uv tool upgrade ollama-llmwatch"
    if "/pipx/" in lowered:
        return "pipx upgrade ollama-llmwatch"
    if "site-packages" in lowered or "dist-packages" in lowered:
        return "pip install --upgrade ollama-llmwatch"
    if os.path.isdir(os.path.join(os.path.dirname(path), ".git")):
        return "git pull"      # a checkout, which is how contributors run it
    # A single file someone downloaded: there is no package to upgrade, so the
    # only honest instruction is to fetch it again.
    return ("curl -O https://raw.githubusercontent.com/"
            "bingcheng45/ollama-llmwatch/main/llmwatch.py")


class _UpgradeRequested(Exception):
    """Leave the loop and upgrade. Not an error: the frame cannot keep drawing
    a program that is being replaced underneath it."""


def run_upgrade(style, plan=None):
    """Run the upgrade on the normal screen, then stop.

    Deliberately does not restart llmwatch afterwards. The file it would
    re-exec has just been overwritten, and a program that relaunches itself
    from something it just replaced is a good way to turn a failed download
    into a confusing crash. Telling someone to run one word is fine.
    """
    argv, blocker = plan if plan is not None else upgrade_plan()
    if blocker:
        sys.stderr.write("llmwatch: cannot upgrade automatically: %s\n"
                         "Run this instead:\n  %s\n" % (blocker, upgrade_command()))
        return 1
    sys.stdout.write("\n%s\n\n" % style.dim("$ " + " ".join(argv)))
    sys.stdout.flush()
    try:
        code = subprocess.call(argv)
    except (OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write("llmwatch: upgrade failed to start: %s\n" % exc)
        return 1
    if code != 0:
        sys.stderr.write("\nllmwatch: upgrade exited %d. Nothing was changed by "
                         "llmwatch itself.\n" % code)
        return code
    sys.stdout.write("\n%s\n" % style.bold("upgraded. start llmwatch again."))
    return 0


def upgrade_plan(path=None, which=None):
    """(argv, blocker) for upgrading this copy in place.

    argv is always an argument list, never a shell string, and never contains
    anything derived from the log, a model name or a version: the whole command
    comes from a fixed set chosen by where this file lives. Nothing an attacker
    can reach gets to influence what runs.

    A blocker means "do not run this, and say why". Refusing is the right answer
    more often than it looks: pulling over somebody's uncommitted work, or
    writing a file the user cannot write, fails in a way that is much harder to
    understand than a sentence explaining it up front.
    """
    path = os.path.abspath(path or install_path())
    directory = os.path.dirname(path)
    lowered = path.replace("\\", "/").lower()
    which = which or shutil.which

    def needs(tool, argv):
        if not which(tool):
            return None, "%s is not on PATH" % tool
        return argv, None

    if "/uv/tools/" in lowered:
        return needs("uv", ["uv", "tool", "upgrade", "ollama-llmwatch"])
    if "/pipx/" in lowered:
        return needs("pipx", ["pipx", "upgrade", "ollama-llmwatch"])
    if "site-packages" in lowered or "dist-packages" in lowered:
        # sys.executable, not a bare `pip`: the pip first on PATH may belong to
        # an entirely different interpreter, and "upgraded" into one you are not
        # running is the most confusing possible outcome.
        return [sys.executable, "-m", "pip", "install", "--upgrade",
                "ollama-llmwatch"], None
    if os.path.isdir(os.path.join(directory, ".git")):
        if not which("git"):
            return None, "git is not on PATH"
        # --ff-only, and only on a clean tree. This is a contributor's checkout;
        # merging into their work uninvited is not an upgrade, it is a mess.
        dirty = _git_is_dirty(directory)
        if dirty is None:
            return None, "cannot read the state of this checkout"
        if dirty:
            return None, "this checkout has uncommitted changes"
        return ["git", "-C", directory, "pull", "--ff-only"], None

    # Everything below replaces one downloaded file. A source tree that no
    # package manager owns and git does not track is not that, and curling a
    # script over the top of the package directory would destroy it, so the
    # honest answer is to refuse and say what this copy looks like.
    if not is_single_file(path):
        return None, ("this is a source tree, not an installed copy: upgrade it "
                      "the way you obtained it")
    # A single downloaded file: replace it where it actually lives, rather than
    # dropping a second copy into whatever directory you happened to be in.
    if not which("curl"):
        return None, "curl is not on PATH"
    if not os.access(directory, os.W_OK):
        return None, "%s is not writable" % directory
    return ["curl", "-fsSL", "-o", path,
            "https://raw.githubusercontent.com/bingcheng45/"
            "ollama-llmwatch/main/llmwatch.py"], None


def _git_is_dirty(directory):
    """True, False, or None when git cannot tell us."""
    try:
        result = subprocess.run(
            ["git", "-C", directory, "status", "--porcelain"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def render_upgrade_confirm(latest, style, cols=80, plan=None):
    """The confirmation pane. Shows the exact command before it runs.

    An upgrade is one keystroke away, so the keystroke that starts it must not
    be the same one that opened this. Nobody should be able to change what is
    installed by leaning on the keyboard.
    """
    argv, blocker = plan if plan is not None else upgrade_plan()
    out = ["", style.bold("upgrade to %s" % latest) if latest else
           style.bold("upgrade"), ""]
    if blocker:
        out.append(style.yellow("cannot upgrade automatically: %s" % blocker))
        out.append("")
        out.append("run this yourself instead:")
        out.append(style.cyan("  " + upgrade_command()))
        out.append("")
        out.append(style.dim("esc  back"))
        return out
    out.append("this will run:")
    out.append(style.cyan("  " + " ".join(argv)))
    out.append("")
    out.append(style.dim("llmwatch quits afterwards, because the program it is "
                         "running from is what changes."))
    out.append("")
    out.append(style.dim("y  do it     esc  back"))
    return out


def render_update(latest, style):
    """One dim line. An update is worth mentioning, not worth interrupting for.

    Re-checks that the version really is newer, even though check_for_update
    already did. Handed a stale or lower version by a future caller, the honest
    failure is to say nothing; the alternative is telling someone running 0.9.0
    to "upgrade" to 0.7.0, which is worse than never mentioning it.
    """
    if not update_is_newer(__version__, latest):
        return None
    return style.dim("update available: %s (you have %s) - press u, or run %s"
                     % (latest, __version__, upgrade_command()))
