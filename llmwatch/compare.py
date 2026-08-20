"""The A/B screens: two models, same question.

Deliberately refuses to produce a ratio until there are enough samples behind
it, and reports counts instead. A confident number from three requests is worse
than no number.
"""
import subprocess

from .constants import MIN_COMPARE_SAMPLES, MIN_COMPARE_TURNS, SAME_WITHIN
from .text import fmt_duration, safe_text


def _verdict(a_value, b_value, label_a="A", label_b="B", unit="x faster"):
    if not a_value or not b_value:
        return "no comparison"
    ratio = a_value / b_value
    if abs(ratio - 1.0) <= SAME_WITHIN:
        return "about the same"
    if ratio > 1:
        return "%s %.2f%s" % (label_a, ratio, unit)
    return "%s %.2f%s" % (label_b, 1 / ratio, unit)


def _pair_bars(label, a_value, b_value, style, suffix="tok/s", width=20):
    """Two bars normalised to the larger value: the comparison should read
    visually before it reads numerically."""
    top = max(a_value or 0, b_value or 0) or 1
    out = []
    for tag, value in (("A", a_value), ("B", b_value)):
        filled = int(round(width * (value or 0) / top))
        bar = ("█" * filled + "░" * (width - filled)) if style.unicode else \
              ("#" * filled + "-" * (width - filled))
        out.append("   %-10s %s %s %s" % (label if tag == "A" else "", tag,
                                          style.cyan(bar),
                                          ("%6.1f %s" % (value, suffix)) if value
                                          else "     - "))
    return out


def render_turns(turns_a, turns_b, style):
    """Turn time in the comparison: the only row here measured in the unit you
    actually wait in.

    The verdict lives on the per-effort rows, never on the total. A model you ran
    mostly on low effort and one you ran mostly on high are not comparable, and
    pooling them produces exactly the kind of confident-but-meaningless ratio the
    prompt-size buckets exist to prevent -- 'A 2.05x quicker' sitting directly
    above a row showing A losing at every effort level both sides share.
    """
    turns_a, turns_b = turns_a or {}, turns_b or {}
    if not (turns_a.get("turns") or turns_b.get("turns")):
        return []

    def cell(profile):
        if not profile.get("turns"):
            return "no turns yet"
        return "%s (n=%d)" % (fmt_duration(profile["median_seconds"]), profile["turns"])

    lines = ["", style.dim("   median agent turn: your prompt to the final answer, "
                           "tool time included")]
    lines.append("   %-10s A %-18s B %-18s %s" % (
        "TURN", cell(turns_a), cell(turns_b),
        style.dim("all efforts pooled")))

    efforts_a = turns_a.get("efforts") or {}
    efforts_b = turns_b.get("efforts") or {}
    for name in sorted(set(efforts_a) | set(efforts_b)):
        side_a, side_b = efforts_a.get(name) or {}, efforts_b.get(name) or {}
        thin = min(side_a.get("turns", 0), side_b.get("turns", 0)) < MIN_COMPARE_TURNS
        if not (side_a.get("turns") and side_b.get("turns")):
            result = style.dim("one side only")
        elif thin:
            result = style.dim("need %d each" % MIN_COMPARE_TURNS)
        else:
            result = style.bold(_verdict(side_b.get("median_seconds"),
                                         side_a.get("median_seconds"), unit="x quicker"))
        lines.append("   %-10s A %-18s B %-18s %s" % (
            "  " + name, cell(side_a), cell(side_b), result))

    interrupted = (turns_a.get("interrupted") or 0) + (turns_b.get("interrupted") or 0)
    if interrupted:
        lines.append(style.dim("     %d interrupted turn%s excluded: they measure your "
                               "patience, not the model"
                               % (interrupted, "" if interrupted == 1 else "s")))
    return lines


def render_compare(profile_a, profile_b, buckets, style, cols=80, days=30,
                   turns_a=None, turns_b=None):
    """The comparison. Also used by `--compare` so the CLI and the TUI cannot
    drift apart."""
    if not profile_a or not profile_b:
        return [style.dim("  pick two models to compare")]

    missing = [p for p in (profile_a, profile_b) if not p.get("requests")]
    if missing:
        lines = []
        for profile in missing:
            lines.append("  %s has no recorded requests yet." % style.bold(profile["model"]))
        lines.append("")
        lines.append("  Run anything against it and it will appear here:")
        lines.append(style.cyan('      ollama run %s "hello"' % missing[0]["model"]))
        lines.append(style.dim("  or point your agent at it for a turn. A comparison needs "
                               "about %d" % MIN_COMPARE_SAMPLES))
        lines.append(style.dim("  requests per side before it will report a ratio."))
        have = [p for p in (profile_a, profile_b) if p.get("requests")]
        if have:
            lines.append("")
            for profile in have:
                lines.append(style.dim("  %s already has %d requests recorded."
                                       % (profile["model"], profile["requests"])))
        return lines

    lines = ["   %s      %s" % (
        style.dim("%d requests, last %s ago" % (profile_a["requests"],
                                                fmt_duration(profile_a["last_seen"] or 0))),
        style.dim("%d requests, last %s ago" % (profile_b["requests"],
                                                fmt_duration(profile_b["last_seen"] or 0)))), ""]

    gen = _pair_bars("GENERATE", profile_a.get("gen_rate"), profile_b.get("gen_rate"), style)
    gen[0] += "  " + style.bold(_verdict(profile_a.get("gen_rate"), profile_b.get("gen_rate")))
    lines.extend(gen)
    lines.extend(_pair_bars("PREFILL", profile_a.get("prefill_rate"),
                            profile_b.get("prefill_rate"), style))

    if profile_a.get("ttft") and profile_b.get("ttft"):
        sooner = abs(profile_a["ttft"] - profile_b["ttft"])
        who = "A" if profile_a["ttft"] < profile_b["ttft"] else "B"
        # "B 0.0s sooner" is noise dressed as a finding.
        note = ("%s %s sooner" % (who, fmt_duration(sooner))) if sooner >= 1 \
            else "about the same"
        lines.append("   %-10s A %-10s B %-10s %s" % (
            "TTFT", fmt_duration(profile_a["ttft"]), fmt_duration(profile_b["ttft"]),
            style.dim(note)))
    if profile_a.get("cache_pct") is not None or profile_b.get("cache_pct") is not None:
        lines.append("   %-10s A %-10s B %-10s" % (
            "CACHE", "%.0f%%" % (profile_a.get("cache_pct") or 0),
            "%.0f%%" % (profile_b.get("cache_pct") or 0)))
    if profile_a.get("draft_pct") is not None or profile_b.get("draft_pct") is not None:
        fmt = lambda p: ("%.0f%%" % p["draft_pct"]) if p.get("draft_pct") is not None \
            else "not a speculative build"
        lines.append("   %-10s A %-10s B %s" % ("DRAFT", fmt(profile_a), fmt(profile_b)))

    if buckets:
        lines.append("")
        lines.append(style.dim("   by prompt size          A            B          result"))
        for row in buckets:
            if row["ratio"]:
                result = _verdict(row["a_rate"], row["b_rate"])
            elif row["enough"]:
                result = "about the same"
            else:
                result = style.dim("need %d each" % MIN_COMPARE_SAMPLES)
            lines.append("     %-7s %-5s %11s %12s      %s" % (
                row["band"], "miss" if row["cache_miss"] else "hit",
                "%.1f (n=%d)" % (row["a_rate"], row["a_n"]) if row["a_rate"]
                else "n=%d" % row["a_n"],
                "%.1f (n=%d)" % (row["b_rate"], row["b_n"]) if row["b_rate"]
                else "n=%d" % row["b_n"],
                result))

    lines.extend(render_median_request(profile_a, profile_b, style))
    lines.extend(render_turns(turns_a, turns_b, style))
    return lines


def median_request_time(profile, prompt_tokens, gen_tokens):
    """Seconds to finish a request of this shape: read the prompt, write the
    answer."""
    prefill_rate, gen_rate = profile.get("prefill_rate"), profile.get("gen_rate")
    if not prefill_rate or not gen_rate:
        return None
    return prompt_tokens / prefill_rate + gen_tokens / gen_rate


def render_median_request(profile_a, profile_b, style):
    """Rates are abstract. Seconds saved on the work you actually do are not."""
    prompts = [p.get("median_prompt_tokens") for p in (profile_a, profile_b)
               if p.get("median_prompt_tokens")]
    answers = [p.get("median_gen_tokens") for p in (profile_a, profile_b)
               if p.get("median_gen_tokens")]
    if not prompts or not answers:
        return []
    prompt_tokens = sum(prompts) / len(prompts)
    gen_tokens = sum(answers) / len(answers)
    a_time = median_request_time(profile_a, prompt_tokens, gen_tokens)
    b_time = median_request_time(profile_b, prompt_tokens, gen_tokens)
    if not a_time or not b_time:
        return []

    out = ["", style.dim("   on your median request (%s tok prompt, %s tok answer)"
                         % (format(int(prompt_tokens), ","), format(int(gen_tokens), ",")))]
    top = max(a_time, b_time)
    for tag, value in (("A", a_time), ("B", b_time)):
        filled = int(round(20 * value / top))
        bar = ("█" * filled) if style.unicode else ("#" * filled)
        out.append("     %s  %-8s %s" % (tag, fmt_duration(value), style.cyan(bar)))
    saved = abs(a_time - b_time)
    who = "A" if a_time < b_time else "B"
    if saved >= 1:
        out.append("     " + style.bold("%s saves %s per request" % (who, fmt_duration(saved))))
    else:
        out.append("     " + style.dim("no meaningful difference per request"))
    return out


def installed_models():
    """Models on disk. The picker shows these too, so a model you have never
    measured is visible with an explanation rather than simply absent."""
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    names = []
    for row in out.splitlines()[1:]:
        row = row.strip()
        if row:
            names.append(safe_text(row.split()[0], limit=80))
    return names


def pickable_models(history):
    """Recorded models first (they can actually be compared), then installed
    ones with no data."""
    recorded = history.models() if history else []
    seen = {entry["model"] for entry in recorded}
    unmeasured = [{"model": name, "requests": 0, "gen_rate": None, "last_seen": None}
                  for name in installed_models() if name not in seen]
    return recorded + sorted(unmeasured, key=lambda entry: entry["model"])


def build_compare(history, model_a, model_b, style, cols=100, days=30, now=None):
    """One entry point for both the TUI view and `--compare`, so they cannot
    drift apart."""
    if not history or not model_a or not model_b:
        return [style.dim("  pick two models to compare")]
    return render_compare(history.profile(model_a, days=days, now=now),
                          history.profile(model_b, days=days, now=now),
                          history.compare(model_a, model_b, days=days, now=now),
                          style, cols, days,
                          turns_a=history.turn_profile(model_a, days=days, now=now),
                          turns_b=history.turn_profile(model_b, days=days, now=now))
