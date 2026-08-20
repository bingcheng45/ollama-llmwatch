"""Tuning numbers, in one place.

Every value here is a judgement call rather than a fact, so each carries the
reasoning that produced it. They live together because a constant declared
beside its first caller quietly becomes that caller's property, and the next
module that needs it has to import backwards through the layering to get it.
"""

__version__ = "0.9.2"

# Frame pacing. The loop wakes on new log data OR on these deadlines, whichever
# comes first, so fresh data appears immediately while animation (spinner,
# elapsed, ETA countdown, projected position) keeps moving when the log is
# silent -- llama-server logs progress only once per 512-token batch, 5-10s
# apart at typical rates.
FRAME_ACTIVE = 0.1     # 10 fps while a request is in flight

# Also 10 fps when nothing is running. This was 2 fps on the reasoning that an
# idle spinner needs no more, which was true of the spinner and wrong about the
# tool: a board that updates twice a second reads as laggy, and "is this thing
# even alive?" is the doubt llmwatch exists to remove. A frame is one write of
# at most a screenful, so the cost of the faster floor is a few KB/s and ten
# wakeups a second while you are watching it; MIN_FRAME_GAP still caps the rest.
FRAME_IDLE = 0.1

MIN_FRAME_GAP = 1 / 30.0   # hard ceiling so a log flood can't spin the CPU

# Requests smaller than this are excluded from peak/low. A cached prompt can
# process 4 tokens at a meaningless rate; without this floor that number becomes
# the session "low" forever and the board quietly lies.
MIN_TOKENS_FOR_EXTREMES = 64

RECENT_LIMIT = 20      # requests kept for the sparkline and the recent pane

# The update check. This is the only thing in llmwatch that touches the network,
# and it stays this small on purpose: one GET to PyPI's public JSON, at most
# once a day, on a background thread, with every failure swallowed. It sends
# nothing about you or your models -- the request body is empty and the URL is
# the same one for every user of the package.
#
# It exists because the alternative was worse. 0.8.0 shipped to GitHub and never
# reached PyPI, and nobody running an older copy had any way to find out.
UPDATE_URL = "https://pypi.org/pypi/ollama-llmwatch/json"

UPDATE_INTERVAL = 24 * 3600

UPDATE_TIMEOUT = 2.0   # a slow network must never delay the first frame

# "That took 12m20s" is a fact; "which is 2x your usual" is the useful part. Both
# need enough history behind them not to be noise.
TURN_TYPICAL_DAYS = 30

MIN_TURNS_FOR_TYPICAL = 3

# Below this, a phase was not measured so much as glimpsed. Rates are divisions
# by this number, so a window of a millisecond turns a handful of tokens into a
# five-figure tok/s that then outranks every real sample on the board. A sanity
# floor, not a claim about how fast a model can be.
MIN_MEASURABLE_SECONDS = 0.02


PREFILL_BATCH = 512      # llama-server reports prefill progress every 512 tokens

GEN_TICK_TOKENS = 50     # and generation roughly every 50


LONG_CONTEXT_TOKENS = 30000     # beyond this, re-reading dominates every turn

LOW_DRAFT_ACCEPT = 0.35         # below this, speculative decoding is losing

LOOP_REPEATS = 3                # identical prompt sizes in a row = likely retry loop

LOOP_ESCALATE = 5               # by here it is not going to resolve on its own


SLOWDOWN_RATIO = 0.7        # below 70% of the usual rate is worth explaining

SLOWDOWN_MIN_SAMPLES = 3    # need a baseline before calling anything "slow"


SAME_WITHIN = 0.05          # closer than this is "about the same", not "1.03x"


# --------------------------------------------------------------------------
# Layer 3b: the settings pane
#
# Flags stay: they are what a script or an agent uses, and they always win.
# This is for the person who does not want to learn them, and it exists because
# every confusion this tool has produced has been the same one -- which mode am
# I in, and what is it watching. The pane answers that first and only then lets
# anything be changed.
#
# Pure state in, lines out. The terminal only supplies keypresses.
# --------------------------------------------------------------------------

# Only the typed fields. The mode is picked with 1/2/3, so it is not a
# cursor row and the cursor never points at something undrawn.
# Named for what someone is running, not for how llmwatch connects to it.
# "proxy" and "log" are true and useless to anyone who has not read the
# README; "an MLX model" is what they would say out loud. Each preset carries
# the settings that choice implies, so choosing is the whole interaction.
#
# Speculative variants (MTP, EAGLE, DFlash) are deliberately absent. They are
# not a way of connecting and there is nothing to choose: whichever server is
# in front, llmwatch reads the acceptance rate off it and says so.
SETTINGS_PRESETS = (
    ("ollama", "Ollama", "anything pulled with `ollama run`",
     {"watch": "ollama"}),
    ("mlx", "An MLX model", "served by mlx_lm.server",
     {"watch": "proxy", "upstream": "http://127.0.0.1:8080"}),
    ("gguf", "A GGUF model", "served by llama.cpp",
     {"watch": "log"}),
    ("lmstudio", "LM Studio", "its own local server",
     {"watch": "proxy", "upstream": "http://127.0.0.1:1234"}),
    ("other", "Something else", "anything speaking the OpenAI API",
     {"watch": "proxy"}),
)


# --------------------------------------------------------------------------
# Layer 4b: the proxy
#
# For a server speaking the OpenAI API there is no log worth tailing -- see the
# Oai* events for why -- so llmwatch stands between the client and the server
# and reads the numbers off the wire instead.
#
# This sits in the middle of somebody's coding loop, which sets the rules:
#
#   * bytes are relayed the moment they arrive and never accumulated, because
#     any buffering here is felt as lag on every token
#   * every failure path still proxies. A monitor that takes the model down
#     with it is worse than one that misses a request, so an unreadable chunk
#     costs a measurement and nothing else
#   * message content is never read, kept or written. Only `usage`, the model
#     name and arrival times leave this class -- the same property History
#     documents about itself
# --------------------------------------------------------------------------

DEFAULT_UPSTREAM = "http://127.0.0.1:8080"

# urlopen will happily open file:, ftp: or a custom scheme, and --upstream is
# the one part of the proxy's configuration that comes from outside. Checked
# where it is accepted rather than trusted at the call site.
UPSTREAM_SCHEMES = ("http://", "https://")

DEFAULT_PROXY_PORT = 8081

LOOPBACK_HOSTS = frozenset(["127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"])


# Where an OpenAI-compatible server is likely to be if one is running: 8080 is
# llama-server's and mlx_lm.server's default, 1234 LM Studio's, 8000 vLLM's.
# Only consulted when the Ollama side has nothing to show, and only to name the
# flag that would watch it.
OAI_PROBE_PORTS = (8080, 1234, 8000)

OAI_PROBE_TIMEOUT = 0.3


MIN_COMPARE_SAMPLES = 5      # below this, report counts rather than a ratio

# Turns are far rarer than requests -- a busy afternoon is a few dozen, not a few
# thousand -- so the floor for claiming a turn-time ratio is lower. It is not
# zero: two turns a side is an anecdote.
MIN_COMPARE_TURNS = 3
