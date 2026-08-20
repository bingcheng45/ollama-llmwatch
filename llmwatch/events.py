"""What the parser produces: one flat vocabulary of facts.

Ollama runs two engines that log nothing alike, and the point of this module is
that nothing above the parser has to know which one ran. Both dialects are
translated into these namedtuples, and the tracker only ever sees these.
"""
from collections import namedtuple

# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

ServerStarted = namedtuple("ServerStarted", "seconds")

ModelLoaded = namedtuple("ModelLoaded", "model")

RequestStart = namedtuple("RequestStart", "slot task prompt_tokens ctx")

CacheInfo = namedtuple("CacheInfo", "slot task cached")

PrefillTick = namedtuple("PrefillTick", "slot task processed progress elapsed rate")

PrefillDone = namedtuple("PrefillDone", "slot task ms tokens rate")

GenTick = namedtuple("GenTick", "slot task decoded rate rate_3s")

GenDone = namedtuple("GenDone", "slot task ms tokens rate")

RequestEnd = namedtuple("RequestEnd", "slot task ms tokens")

# Events that explain WHY a wait is long -- the difference between "20 seconds"
# and "8 minutes" is usually whether the prompt cache survived.
CacheMiss = namedtuple("CacheMiss", "slot task")

CheckpointRestored = namedtuple("CheckpointRestored", "slot task tokens")

CheckpointCreated = namedtuple("CheckpointCreated", "slot task index total")

CheckpointErased = namedtuple("CheckpointErased", "slot task tokens")

# Speculative decoding (the -mtp builds): how many drafted tokens survived.
DraftAcceptance = namedtuple("DraftAcceptance", "slot task rate accepted generated mean_len")

# Ollama runs GGUF models through llama-server but `-mlx` models through its own
# MLX runner, which logs a different dialect entirely. These carry the raw facts
# off one MLX line each; the Tracker turns them into the events above, so only
# the parser and one Tracker branch know MLX exists. Each keeps the log's own
# timestamp because MLX splits a timing across lines and never prints a rate --
# elapsed has to be reconstructed from the gap, and using wall-clock time here
# would silently produce nonsense when replaying an old log with --last.
MlxRunnerStart = namedtuple("MlxRunnerStart", "model ts")

MlxRunnerReady = namedtuple("MlxRunnerReady", "ts")

MlxRequestStart = namedtuple("MlxRequestStart", "prompt_tokens cached miss ts")

MlxPrefillTick = namedtuple("MlxPrefillTick", "processed total ts")

MlxGenStats = namedtuple("MlxGenStats", "iterations drafted accepted acceptance avg_draft ts")

MlxRequestEnd = namedtuple("MlxRequestEnd", "seconds ts")

# The last line of a request, and the only one whose position is fixed: the
# completion and stats lines are written from different places and either can
# land first, but `peak memory` always follows both. It is what makes a request
# safe to close, so a rate is never missing just because the log raced.
MlxPeakMemory = namedtuple("MlxPeakMemory", "ts")

MLX_EVENTS = (MlxRunnerStart, MlxRunnerReady, MlxRequestStart, MlxPrefillTick,
              MlxGenStats, MlxRequestEnd, MlxPeakMemory)

# A server speaking the OpenAI API (mlx_lm.server, LM Studio, llama-server, vLLM)
# is watched from the outside, by proxying it, because the numbers simply are not
# in its log: mlx_lm.server prints prefill progress and nothing else -- no
# generation token count, no rate. `usage` in the response body is the only place
# a completion size exists, and it arrives last, so every rate here is computed
# from timestamps the proxy took itself.
#
# The exception is OaiPrefillTick, which does come from a log. Prefill on a 28k
# prompt is minutes of silence on the wire (the first byte is only sent once it
# finishes), so the progress bar has to come from somewhere else or not exist.
OaiRequestStart = namedtuple("OaiRequestStart", "model ts")

OaiPrefillTick = namedtuple("OaiPrefillTick", "processed total ts")

OaiGenTick = namedtuple("OaiGenTick", "decoded ts")

# `draft` is (rate, accepted, generated, mean_len) or None, matching what the
# log backend puts on a Request -- so history, the comparison pane and the
# drafts-often-rejected hint read one shape whichever backend filled it in.
# It defaults to None because every other OpenAI server sends no timings block
# at all, and most llama-server runs have no drafter loaded.
OaiRequestEnd = namedtuple(
    "OaiRequestEnd",
    "prompt_tokens cached_tokens completion_tokens started first_token last_token "
    "ts draft")

OaiRequestEnd.__new__.__defaults__ = (None,)

# The client hung up, or the upstream died, before the response completed. No
# usage block was ever sent, so there is nothing to time -- but the request has
# to leave the board rather than sit there open forever.
OaiRequestAborted = namedtuple("OaiRequestAborted", "ts")

# Emitted once, the first time the upstream identifies itself. The engine is a
# property of the server, not of a request, so it is not carried on every event.
OaiEngine = namedtuple("OaiEngine", "engine ts")

OAI_EVENTS = (OaiRequestStart, OaiPrefillTick, OaiGenTick, OaiRequestEnd,
              OaiRequestAborted, OaiEngine)


# --------------------------------------------------------------------------
# Layer 2: tracking
# --------------------------------------------------------------------------

Output = namedtuple("Output", "kind text data")  # kind: "live" | "line"
