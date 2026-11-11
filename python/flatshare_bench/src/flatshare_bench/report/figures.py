"""The report's figures, docs/figures/report-*.svg, one function each, from the values :mod:`.data` reads.

    python -m flatshare_bench.report.figures OUT_DIR

Each figure is 1120 units wide, as the README's are, and shows at about 880 CSS pixels; a figure with a phone layout
has it at 420 units. No figure draws a title or a subtitle: the report's caption under it states its claim, its key
and its scope, a panel's title names what it plots, and its axis's title the quantity and unit as figure_style's
grammar has it. Every number a figure prints is a record's, as docs/BENCHMARKS.md's tables give it, and latency and
throughput scales start at zero unless the axis says it is logarithmic. Each figure has one axis a dimension: none
draws a second scale beside its first. A threshold is named once: a horizontal one at the plot's right end, outside it,
on its line (THRESHOLD_GAP), and a vertical one over the plot's top, centred on it.
"""

from __future__ import annotations

import datetime
import math
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

from flatshare_bench.e2.record import REPLY_TIMEOUT_MS
from flatshare_bench.figure_style import LABEL, NARROW, NOTE, VALUE, WIDTH, Box, fmt, measure, wrap
from flatshare_bench.labels import mark as config_mark
from flatshare_bench.report import data
from flatshare_bench.report.charts import (
    SPREAD_MIN,
    STAGES,
    Chart,
    Key,
    Scale,
    band_grid,
    cut_breaks,
    ecdf,
    hatch_key,
    legend,
    legend_width,
    line_key,
    linear_ticks,
    mark_key,
    pct,
    seconds,
    stage_key,
    x_axis,
    y_axis,
)


def _sentence(name: str) -> str:
    """A key's entry in sentence case: its first letter upper case, unless it opens with a version ("v1.14's")."""
    return name if re.match(r"v\d", name) else name[0].upper() + name[1:]


THRESHOLD_GAP = 10.0
"""Between a plot's right end and the name of a horizontal threshold across it, which stands outside it on the line."""


def _h_threshold(s: Chart, x0: float, x1: float, y: float, text: str) -> Box:
    """A horizontal threshold (an SLO, a requirement) across a plot from ``x0`` to ``x1`` at ``y``: a dashed rule,
    named once at the plot's right end, outside it, on its line."""
    s.rule(x0, x1, y, "slo")
    return s.text(x1 + THRESHOLD_GAP, y + 6, text, size=NOTE, cls="red tint")


def _v_threshold(s: Chart, x: float, top: float, bottom: float, text: str) -> Box:
    """A vertical threshold across a plot from ``top`` to ``bottom`` at ``x``: a dashed rule, named once over the
    plot's top, centred on it."""
    s.vrule(x, top, bottom, "slo")
    return s.text(x, top - 8, text, size=NOTE, anchor="middle", cls="red tint", halo=True)


STUB = 17.0
"""How far a phone layout's threshold reaches above and below a row's band: past its bar, clear of the rows' names."""


def _threshold_stub(s: Chart, x: float, band: float, below: float = 0.0) -> None:
    """A threshold in a phone layout's row, at ``x``: the dashed rule through the row's band alone (to ``below`` it,
    where the row holds a pair), past its bars at both ends, since the rows' names stand between the bands."""
    s.vrule(x, band - STUB, band + below + STUB, "slo")


def _threshold_key(s: Chart, x: float, baseline: float, text: str) -> Box:
    """A phone layout's threshold named once, as a key's entry from ``x``: a sample of its dashed stub, then its
    name. Returns the name's box."""
    s.vrule(x + 9, baseline - 15, baseline + 4, "slo")
    return s.text(x + 26, baseline, text, size=NOTE, cls="red tint")


# The voice turn by stage ------------------------------------------------------------------------------------------

STAGE_NAMES = {
    "endpointing": "Endpointing",
    "wait": "Wait for the LLM's first token",
    "decode": "Decode of the first sentence",
    "tts": "TTS first audio",
    "transport": "Transport and playout",
}
INSIDE = ("wait", "decode")
"""The stages whose seconds a bar prints inside the segment, where they fit: the two the scheduler sets."""


def _stage_description(cells: Sequence[data.Stages]) -> str:
    rows = "; ".join(
        f"{data.in_prose(data.label(c.point)[0])}, {seconds(c.v2v)} s, {seconds(c.wait)} s of it waiting for the "
        f"LLM's first token and {seconds(c.decode)} s decoding the first sentence"
        for c in cells
    )
    return (
        "The voice turn's mean voice-to-voice time stacked by stage from the end of the caller's speech, beside 24 "
        f"agent sessions but for voice alone: {rows}."
    )


STAGES_TITLE = "Where the voice turn's seconds go"


def stages(out: Path) -> list[Path]:
    """report-stages.svg and its phone layout: where the voice turn's seconds go, a stacked bar a configuration of
    its stages' means, its length the mean V2V."""
    cells = data.stages()
    description = _stage_description(cells)
    return [
        _stages_wide(cells, out / "report-stages.svg", description, STAGES_TITLE),
        _stages_narrow(cells, out / "report-stages-narrow.svg", description, STAGES_TITLE),
    ]


def deployable_stages(out: Path) -> list[Path]:
    """report-deployable-stages.svg and its phone layout: the voice turn's stages for W4's configurations
    (data.W4_STAGES), from W4's records."""
    cells = data.stages(data.w4_rows(), data.W4_STAGES)
    description = _stage_description(cells)
    title = "Where the voice turn's seconds go on what an operator can deploy"
    return [
        _stages_wide(cells, out / "report-deployable-stages.svg", description, title),
        _stages_narrow(cells, out / "report-deployable-stages-narrow.svg", description, title),
    ]


def _stage_bar(s: Chart, cells: data.Stages, scale: Scale, y: float, thick: float) -> None:
    """A configuration's stacked bar at ``y``, its stages' seconds inside the scheduler's two where both fit (a row
    names both or neither), and its mean V2V at its end."""
    parts = [(scale(v) - scale(0), stage) for v, stage in zip(cells.parts, STAGES, strict=True)]
    starts = s.stack(scale.start, y, parts, thick)
    inside = [
        (start + length / 2, length, f"{seconds(value)} s")
        for (length, stage), start, value in zip(parts, starts, cells.parts, strict=True)
        if stage in INSIDE
    ]
    if all(measure(text, NOTE) + 16 <= length for _, length, text in inside):
        for middle, _, text in inside:
            s.text(middle, y + thick / 2 + 6, text, size=NOTE, anchor="middle")
    s.text(
        scale(cells.v2v) + VALUE_GAP,
        y + thick / 2 + 7,
        f"{seconds(cells.v2v)} s",
        size=VALUE,
        bold=_featured(cells.point),
        halo=True,
    )


def _stages_wide(cells: Sequence[data.Stages], path: Path, description: str, title: str) -> Path:
    s = Chart(WIDTH, 400, title, description)
    entries = [(stage_key(stage), STAGE_NAMES[stage]) for stage in STAGES]
    left = (WIDTH - legend_width(entries)) / 2
    legend(s, left, 30, entries, width=WIDTH - 2 * left + 1)
    right, x0, length = LABELS_END, PLOT_START, 760
    marks, labels = linear_ticks(6, 1)
    scale = Scale(0, marks[-1] * 1000, x0, x0 + length)
    top, pitch, thick = 62, 56, 26
    rows = [top + (pitch - thick) / 2 + pitch * k for k in range(len(cells))]
    bottom = rows[-1] + thick + (pitch - thick) / 2
    last = x_axis(
        s,
        scale,
        [m * 1000 for m in marks],
        labels,
        top,
        bottom,
        title="Mean time from the end of the caller's speech (s)",
    )
    for y, c in zip(rows, cells, strict=True):
        name, note = data.label(c.point)
        middle = y + thick / 2
        s.text(
            right,
            middle - 3 if note else middle + 7,
            name,
            size=VALUE,
            anchor="end",
            bold=_featured(c.point),
        )
        if note:
            s.text(right, middle + 19, note, size=NOTE, anchor="end", cls="soft")
        _stage_bar(s, c, scale, y, thick)
    s.height = last + 14
    return s.save(path)


def _stages_narrow(cells: Sequence[data.Stages], path: Path, description: str, title: str) -> Path:
    s = Chart(NARROW, 400, title, description)
    x0, length = 18, 330
    entries = [(stage_key(stage), STAGE_NAMES[stage]) for stage in STAGES]
    y = legend(s, x0, 28, entries, width=NARROW - 2 * x0, pitch=25) + 10
    marks, labels = linear_ticks(6, 1)
    scale = Scale(0, marks[-1] * 1000, x0, x0 + length)
    thick, pitch = 24, 74
    y += 12
    for c in cells:
        name, note = data.label(c.point)
        box = s.text(x0, y + 18, name, size=VALUE, bold=_featured(c.point))
        if note:
            s.text(box.x1 + 8, y + 18, note, size=NOTE, cls="soft")
        band_grid(s, scale, [m * 1000 for m in marks], y + 24, y + 28 + thick + 4)
        _stage_bar(s, c, scale, y + 28, thick)
        y += pitch
    bottom = y - pitch + 28 + thick + 4
    last = x_axis(s, scale, [m * 1000 for m in marks], labels, None, bottom)
    for line in wrap("Mean time from the end of the caller's speech (s)", NARROW - 36, NOTE):
        last += 26
        s.text(NARROW / 2, last, line, size=NOTE, anchor="middle", cls="soft")
    s.height = last + 14
    return s.save(path)


# The KV pool over time --------------------------------------------------------------------------------------------

KV_TICKS = [0, 5, 10, 15, 20, 25]
"""Seconds since the KV case's agent requests were sent, past the timeline's last read at 24.5 s."""
KV_AGENTS = ("agent A", "agent B")
"""The two agent requests, in the order the record counts their tokens: vLLM preempts the later arrival, agent B."""
KV_POOL = ("gray wash", "red wash")
"""The pool's fill under its line: the gray's wash, and the red's while a voice turn would wait for a block."""
KV_TOKENS = [0, 400, 800]
"""Panel b's ticks, to the 900 output tokens its axis ends at."""


def _kv_description(samples: Sequence[data.PoolSample], m: data.PoolMoments) -> str:
    last = samples[-1]
    return (
        f"In panel a, the share of the KV pool's {data.POOL_BLOCKS:,} blocks in use on stock vLLM while the KV case's "
        "two agent requests of 18k prompt tokens prefill and decode, read about every 0.55 s: "
        f"{m.full.usage:.1%} at {m.full.t:g} s, {m.send.free_blocks} blocks free at {m.send.t:g} s, where the KV case "
        f"sends its voice turn, none freed until {m.before.t:g} s, and {m.after.usage:.1%} at {m.after.t:g} s, once "
        "vLLM has preempted agent B so that agent A can grow; in panel b, each request's output tokens, both decoding "
        f"into the last free blocks until agent B stops at {m.after.tokens[1]} tokens while agent A reaches "
        f"{last.tokens[0]} by {last.t:g} s."
    )


def _decoding(samples: Sequence[data.PoolSample], k: int) -> list[data.PoolSample]:
    """Agent request ``k``'s samples from the last before its first output token: its line starts as its prefill
    ends."""
    first = next(i for i, sample in enumerate(samples) if sample.tokens[k] > 0)
    return list(samples[max(first - 1, 0) :])


def kv_timeline(out: Path) -> list[Path]:
    """report-kv-timeline.svg and its phone layout: why a waiting request finds no room on vLLM 0.32. (a) The KV
    pool's blocks in use as the KV case's two agent requests prefill and decode, a filled area: the moment the KV
    case would send its voice turn pinned in green, the span from it to the last read before the preemption in red
    with its length, and the preemption named with the share of the pool it frees. (b) Each request's output tokens
    on the same time axis, the same span shaded, agent B's line dashed once it is preempted."""
    samples = data.kv_timeline()
    m = data.kv_moments(samples)
    description = _kv_description(samples, m)
    return [
        _kv_layout(samples, m, out / "report-kv-timeline.svg", description, WIDTH),
        _kv_layout(samples, m, out / "report-kv-timeline-narrow.svg", description, NARROW),
    ]


def _kv_area(s: Chart, x: Scale, y: Scale, samples: Sequence[data.PoolSample], cls: str) -> None:
    """The pool's use over ``samples`` as an area from the axis up to its line."""
    points = [(x(p.t), y(p.usage)) for p in samples]
    d = f"M{fmt(points[0][0])} {fmt(y.start)}"
    d += "".join(f"L{fmt(px)} {fmt(py)}" for px, py in points)
    s.path(d + f"L{fmt(points[-1][0])} {fmt(y.start)}Z", cls)


def _kv_layout(
    samples: Sequence[data.PoolSample],
    m: data.PoolMoments,
    path: Path,
    description: str,
    width: float,
) -> Path:
    wide = width == WIDTH
    s = Chart(width, 700, "The KV pool over time", description)
    x0 = 24 if wide else 18
    x = (
        Scale(KV_TICKS[0], KV_TICKS[-1], 96, 950)
        if wide
        else Scale(KV_TICKS[0], KV_TICKS[-1], 64, width - 22)
    )
    titles = (
        f"(a) KV pool in use, % of its {data.POOL_BLOCKS:,} blocks",
        "(b) Output tokens of each agent request",
    )
    # (a) under its title and the line of the moment's labels; (b) under it on the same time axis.
    s.text(x0, 30, titles[0], bold=True)
    a = Scale(0, 1, 82 + (132 if wide else 150), 82)
    b_title = a.start + 46
    b_top = b_title + (
        18 if wide else 30
    )  # a phone's plot holds agent A's name over its line's end
    b = Scale(0, 900, b_top + (104 if wide else 160), b_top)
    s.text(x0, b_title, titles[1], bold=True)
    sent, full = x(m.send.t), x(m.before.t)
    # The span in which nothing frees a block, behind (b)'s grid; in (a) it is the pool's own area that turns red.
    s.rect(sent, b.end, full - sent, b.start - b.end, "red area")
    y_axis(s, a, [0, 0.5, 1], [pct(v) for v in (0, 0.5, 1)], x.start, x.end)
    y_axis(s, b, KV_TOKENS, [f"{v:,}" for v in KV_TOKENS], x.start, x.end)
    for t in KV_TICKS[1:-1]:
        for scale in (a, b):
            s.vrule(x(t), scale.end, scale.start, "grid")
    title = "Time since the two agent requests were sent (s)"
    if wide:
        last = x_axis(s, x, KV_TICKS, [str(t) for t in KV_TICKS], None, b.start, title=title)
    else:
        last = x_axis(s, x, KV_TICKS, [str(t) for t in KV_TICKS], None, b.start)
        for line in wrap(title, width - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
    # (a): the pool as a filled quantity under its line, red from the voice turn's moment to the last full read.
    waiting = [p for p in samples if m.send.t <= p.t <= m.before.t]
    _kv_area(s, x, a, samples, KV_POOL[0])
    _kv_area(s, x, a, waiting, KV_POOL[1])
    s.trace([(x(p.t), a(p.usage)) for p in samples], "gray trace")
    # The moment the KV case sends its voice turn, a green pin over the plot, and what follows it, in red.
    pin = a.end - 16
    s.ref((sent, pin), (sent, a.start), "green ref")
    s.mark(sent, pin, "green", name="the voice turn's moment")
    s.text(sent - 11, pin + 6, "voice turn sent", size=NOTE, anchor="end", cls="green tint")
    s.text(
        sent + 11,
        pin + 6,
        f"no block freed for {m.before.t - m.send.t:g} s",
        size=NOTE,
        cls="red tint",
    )
    # The preemption, in the room it frees over the pool's line.
    freed = pct(m.before.usage - m.after.usage)
    at = x(m.after.t) + 14
    s.text(at, a.end + 24, "agent B preempted", size=NOTE, halo=True)
    s.text(
        at,
        a.end + 46,
        f"{freed} of the pool freed" if wide else f"{freed} freed",
        size=NOTE,
        cls="soft",
        halo=True,
    )
    # (b): each request's output tokens, named at its line's end with where it stands; in the red span the two
    # run as one, and agent B's is dashed from its preemption on, flat.
    last_read = samples[-1]
    first = min(_decoding(samples, k)[0].t for k in range(len(KV_AGENTS)))
    s.text(
        (x.start + x(first)) / 2,
        b(KV_TOKENS[1]) + 6,
        "both still prefilling" if wide else "prefilling",
        size=NOTE,
        anchor="middle",
        cls="soft",
        halo=True,
    )
    for k, name in enumerate(KV_AGENTS):
        run = _decoding(samples, k)
        if k == 1:
            live = [p for p in run if p.t <= m.after.t]
            s.trace([(x(p.t), b(p.tokens[k])) for p in live], "gray trace")
            s.trace(
                [(x(p.t), b(p.tokens[k])) for p in run if p.t >= m.after.t],
                "gray trace dashed",
            )
        else:
            s.trace([(x(p.t), b(p.tokens[k])) for p in run], "gray trace")
        end = (x(last_read.t), b(last_read.tokens[k]))
        value = f"{last_read.tokens[k]:,} tokens"
        if wide:
            box = s.text(end[0] + 10, end[1] + 6, name, size=NOTE, halo=True)
            s.text(box.x1 + 7, end[1] + 6, value, size=NOTE, cls="soft", halo=True)
        else:  # inside the plot: over agent A's rising end, under agent B's level one, clear of agent A's line
            baseline = end[1] + (20 if k else -10)
            box = s.text(x.end, baseline, value, size=NOTE, anchor="end", cls="soft", halo=True)
            s.text(box.x0 - 7, baseline, name, size=NOTE, anchor="end", halo=True)
    s.height = last + 14
    return s.save(path)


# The two waits in isolation ---------------------------------------------------------------------------------------

PROBE_ROWS = {"stock": "stock", "a": "kv-preemption", "ab": "change"}
"""Each probe block's row: its configuration's key in data.LABELS, whose name and mark (labels.MARKS) it takes."""
PROBE_FEATURED = "ab"
"""The probe block the figure is drawn for, the change's: the one configuration of the change here, so its bars take
the featured hue, and its name and its values the semibold face."""
PROBE_STEPS = 15
"""The tokens a prefill run's reply decodes from its first to its 16th: its gap a token is that time over these."""


def _probe_hue(engine: str) -> str:
    """A probe block's bars' hue: its configuration's (labels.MARKS), and the featured one for the figure's own."""
    return FEATURED if engine == PROBE_FEATURED else config_mark(PROBE_ROWS[engine])[0]


def _token_gaps(runs: Sequence[data.Probe]) -> tuple[float, float]:
    """The least and the greatest of the runs' time a token from the first to the 16th, in ms."""
    gaps = [(r.t16 - r.ttft) / PROBE_STEPS for r in runs if r.t16 is not None]
    return min(gaps), max(gaps)


def _gap_text(runs: Sequence[data.Probe]) -> str:
    lo, hi = _token_gaps(runs)
    return f"{lo:.0f}\N{EN DASH}{hi:.0f} ms/token"


def _probe_description(
    kv: dict[str, list[data.Probe]], prefill: dict[str, list[data.Probe]]
) -> str:
    def stated(values: list[int]) -> str:
        """The runs' median, the number its row prints, and their range."""
        return f"{statistics.median(values):,.0f} ms, {min(values):,}\N{EN DASH}{max(values):,}"

    names = {engine: data.in_prose(data.label(PROBE_ROWS[engine])[0]) for engine in data.ENGINES}
    medians = {e: statistics.median(r.ttft for r in runs) for e, runs in kv.items()}
    sooner = {"a": f", {medians['stock'] / medians['a']:.0f} times sooner"}
    first = "; ".join(
        f"{names[e]} {stated([r.ttft for r in runs])}{sooner.get(e, '')}" for e, runs in kv.items()
    )
    sixteenth = "; ".join(
        f"{names[e]} {stated([r.t16 for r in runs if r.t16])}, {_gap_text(runs)}"
        for e, runs in prefill.items()
    )
    return (
        f"Four probe runs on the RTX 5060 Ti for each of {names['stock']}, {names['a']} and {names['ab']} at its "
        "16-token window, each time the runs' median and then their range: in panel a, the KV case, the voice turn's "
        f"first token while two agent requests of 18k tokens hold the pool ({first}); in panel b, the prefill case, "
        "its first token and then its 16th while three agent prefills of 10k tokens run, the 16th and the time a "
        f"token between the two at {sixteenth}."
    )


def isolation(out: Path) -> list[Path]:
    """report-isolation.svg and its phone layout: the two waits in isolation, a bar a configuration at the median of
    its four probe runs, their spread a capped line, the change's bars in the featured blue. (a) The KV case: the turn's first token, sent into a pool two
    agent requests fill, with how many times sooner the KV preemption brings it. (b) The prefill case: one bar to the
    first token and on to the 16th, sent beside three agent prefills, with the 16th's time and the time a token
    between the two."""
    kv, prefill = data.probes("kv"), data.probes("prefill")
    # The description states what each case's turn was sent beside: every run's line must say it.
    for case, runs_by_engine, beside in (("kv", kv, (2, 17999)), ("prefill", prefill, (3, 10000))):
        sent = {(r.agents, r.agent_tokens) for runs in runs_by_engine.values() for r in runs}
        if sent != {beside}:
            raise ValueError(f"the {case} case's runs were sent beside {sent}, not {beside}")
    for engine, probes in prefill.items():
        if any(r.t16 is None for r in probes):
            raise ValueError(f"a prefill run of {engine} ended before its 16th token")
    description = _probe_description(kv, prefill)
    return [
        _isolation_wide(kv, prefill, out / "report-isolation.svg", description),
        _isolation_narrow(kv, prefill, out / "report-isolation-narrow.svg", description),
    ]


ISOLATION_PANELS = (
    ("(a) The KV case", "Time to the first token (ms)"),
    ("(b) The prefill case", "Time from the request (ms)"),
)
"""Each panel's title and its axis's title; the caption says what each case sends its turn beside."""
ISOLATION_TOPS = (5000, 1600)
"""Each panel's axis's end, in ms: the first tick past its slowest run."""
ISOLATION_PARTS = ("first token", "16th token")
"""What panel b's bar reaches at its seam and at its end."""


def _kv_row(s: Chart, a: Scale, y: float, engine: str, kv: dict[str, list[data.Probe]]) -> None:
    """Panel a's row: the median first token as a bar with its runs' spread, its value, and once, on the KV
    preemption's row, how many times sooner than stock vLLM's it is."""
    key = PROBE_ROWS[engine]
    firsts = [r.ttft for r in kv[engine]]
    middle = statistics.median(firsts)
    spread = firsts if a(max(firsts)) - a(min(firsts)) > SPREAD_MIN else [middle]
    end = _value_bar(s, a, y, middle, spread, _probe_hue(engine), data.label(key)[0])
    box = s.text(
        end + VALUE_GAP,
        y + 7,
        f"{middle:,.0f} ms",
        size=VALUE,
        halo=True,
        bold=engine == PROBE_FEATURED,
    )
    if engine == "a":
        stock = statistics.median(r.ttft for r in kv["stock"])
        s.text(
            box.x1 + 8,
            y + 7,
            f"{stock / middle:.0f}\N{MULTIPLICATION SIGN} sooner",
            size=NOTE,
            halo=True,
        )


def _prefill_row(
    s: Chart, b: Scale, y: float, engine: str, prefill: dict[str, list[data.Probe]]
) -> tuple[float, float]:
    """Panel b's row: one bar to the median first token and on to the median 16th, each with its runs' spread, the
    16th's value and the time a token between the two. Returns where its seam and its end are."""
    key = PROBE_ROWS[engine]
    firsts = [r.ttft for r in prefill[engine]]
    lasts = [r.t16 for r in prefill[engine] if r.t16 is not None]
    first, last = statistics.median(firsts), statistics.median(lasts)
    end = _extended_bar(s, b, y, first, last, _probe_hue(engine), data.label(key)[0])
    for values in (firsts, lasts):
        lo, hi = b(min(values)), b(max(values))
        if hi - lo > SPREAD_MIN:
            s.span(lo, hi, y)
            end = max(end, hi)
    box = s.text(
        end + VALUE_GAP,
        y + 7,
        f"{last:,.0f} ms",
        size=VALUE,
        halo=True,
        bold=engine == PROBE_FEATURED,
    )
    s.text(box.x1 + 8, y + 7, _gap_text(prefill[engine]), size=NOTE, cls="soft", halo=True)
    return b(first), b(last)


def _isolation_wide(
    kv: dict[str, list[data.Probe]],
    prefill: dict[str, list[data.Probe]],
    path: Path,
    description: str,
) -> Path:
    s = Chart(WIDTH, 400, "The two waits in isolation", description)
    a = Scale(0, ISOLATION_TOPS[0], PLOT_START, 500)
    b = Scale(0, ISOLATION_TOPS[1], 660, 880)
    for scale, (title, _) in zip((a, b), ISOLATION_PANELS, strict=True):
        s.text(scale.start, 30, title, bold=True)
    top, pitch = 76, 56
    ys = [top + pitch / 2 + pitch * k for k in range(len(data.ENGINES))]
    bottom = ys[-1] + pitch / 2
    marks_a, labels_a = linear_ticks(ISOLATION_TOPS[0], 2500)
    marks_b, labels_b = linear_ticks(ISOLATION_TOPS[1], 400)
    last = x_axis(s, a, marks_a, labels_a, top, bottom, title=ISOLATION_PANELS[0][1])
    x_axis(s, b, marks_b, labels_b, top, bottom, title=ISOLATION_PANELS[1][1])
    for y, engine in zip(ys, data.ENGINES, strict=True):
        name, note = data.label(PROBE_ROWS[engine])
        s.text(LABELS_END, y - 3, name, size=VALUE, anchor="end", bold=engine == PROBE_FEATURED)
        s.text(LABELS_END, y + 19, note, size=NOTE, anchor="end", cls="soft")
        _kv_row(s, a, y, engine, kv)
        _prefill_row(s, b, y, engine, prefill)
    legend(s, b.start, top - 12, _part_keys(), width=WIDTH - b.start - 24)
    s.height = last + 14
    return s.save(path)


def _part_key(fill: str) -> Key:
    """A part of panel b's bar in its key: a short stretch of its fill."""

    def draw(s: Chart, x: float, baseline: float) -> None:
        s.segments(x, baseline - 0.66 * NOTE, [(22, fill)], 0.7 * NOTE)

    return Key(22, draw)


def _part_keys() -> list[tuple[Key, str]]:
    """Panel b's key, over its rows: the two parts of a bar, in the first row's fills."""
    fills = EXTENSIONS[config_mark(PROBE_ROWS["stock"])[0]]
    return [
        (_part_key(fill), f"To the {part}")
        for fill, part in zip(fills, ISOLATION_PARTS, strict=True)
    ]


def _isolation_narrow(
    kv: dict[str, list[data.Probe]],
    prefill: dict[str, list[data.Probe]],
    path: Path,
    description: str,
) -> Path:
    """The phone layout: the two cases one above the other, each row's name over its bar, and panel b's two parts in
    a key under its title, where a phone's rows leave no room over the first bar."""
    s = Chart(NARROW, 1200, "The two waits in isolation", description)
    x0 = 18
    scales = (Scale(0, ISOLATION_TOPS[0], x0, 278), Scale(0, ISOLATION_TOPS[1], x0, 188))
    ticks = (linear_ticks(ISOLATION_TOPS[0], 2500), linear_ticks(ISOLATION_TOPS[1], 800))
    y = 30.0
    last = y
    for k, ((title, axis), scale, (marks, labels)) in enumerate(
        zip(ISOLATION_PANELS, scales, ticks, strict=True)
    ):
        s.text(x0, y, title, bold=True)
        y += 12
        if k == 1:
            y = legend(s, x0, y + 20, _part_keys(), width=NARROW - 2 * x0) + 8
        for engine in data.ENGINES:
            name, note = data.label(PROBE_ROWS[engine])
            box = s.text(x0, y + 20, name, size=VALUE, bold=engine == PROBE_FEATURED)
            s.text(box.x1 + 8, y + 20, note, size=NOTE, cls="soft")
            band = y + 46
            band_grid(s, scale, marks, band - 14, band + 14)
            if k == 0:
                _kv_row(s, scale, band, engine, kv)
            else:
                _prefill_row(s, scale, band, engine, prefill)
            y = band + 18
        last = x_axis(s, scale, marks, labels, None, y)
        for line in wrap(axis, NARROW - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
        y = last + 48
    s.height = last + 14
    return s.save(path)


# Both waits beside closed-loop agents -------------------------------------------------------------------------------

TAIL_LINES = {
    "stock": ("stock", "on", "trace"),
    "a": ("kv-preemption", "with", "trace dashed"),
    "ab": ("change", "with", "trace thick"),
}
"""Each probe block's line: its configuration's key in data.LABELS, whose name and hue (labels.MARKS) it takes, the
word the description puts before its name, and its stroke: stock vLLM's gray, the KV preemption's dashed in the
change's sibling blue, and the change's, the figure's own (TAIL_FEATURED), thick in the featured blue."""
TAIL_FEATURED = "ab"
"""The line the figure is drawn for, the change's: the one configuration of the change here, so it takes the featured
hue, as report-cdf's line does, and its p95 is printed in bold."""
TAIL_NAMES = {
    "fresh": {"stock": 0.3, "a": 0.68, "ab": 0.5},
    "grow": {"stock": 0.3, "a": 0.8, "ab": 0.5},
}
"""The share of its turns at which the wide figure names each line, right of it: where the next line leaves room."""


def _tail_name(engine: str) -> str:
    return data.label(TAIL_LINES[engine][0])[0]


def _tail_hue(engine: str) -> str:
    """A probe block's line's hue: its configuration's (labels.MARKS), and the featured one for the figure's own."""
    return FEATURED if engine == TAIL_FEATURED else config_mark(TAIL_LINES[engine][0])[0]


def _tail_cls(engine: str) -> str:
    """A probe block's line's classes: its hue and its stroke."""
    return f"{_tail_hue(engine)} {TAIL_LINES[engine][2]}"


TAIL_PANELS = {
    "fresh": ("(a) Fresh agent prompts", "with the agents' prompts fresh each turn"),
    "grow": ("(b) Growing agent prompts", "with each agent's prompt grown by a tool result"),
}
TAIL_AXIS = (40, 15000)
"""A first token's log axis, in ms: from under the fastest turn (43 ms) to past the slowest (11.8 s)."""
TAIL_TICKS = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
TAIL_TITLE = "Time to a call turn's first token (s, log scale)"
TAIL_SHARE = "Share of the calls' turns"
TAIL_SHARES = [0.0, 0.5, 1.0]
TAIL_GUIDE = 95
"""The percentile the guide marks: the tail the report's text and tables give."""


def _tail_description(tails: dict[str, dict[str, data.Tail]]) -> str:
    panels = []
    for letter, (mode, (_, prompts)) in zip("ab", TAIL_PANELS.items(), strict=True):
        cells = ", ".join(
            f"{t.percentile(50):,.0f} and {t.percentile(95):,.0f} ms {TAIL_LINES[engine][1]} "
            f"{data.in_prose(_tail_name(engine))} ({len(t.turns)} turns)"
            for engine, t in tails[mode].items()
        )
        panels.append(f"in panel {letter}, {prompts}, the p50 and p95 at {cells}")
    return (
        "The cumulative share of two calls' turns by their time to first token beside 12 closed-loop agent sessions, "
        "pooled over two 240 s runs of each scheduler, on a log scale with the 95th percentile marked: "
        + "; ".join(panels)
        + "."
    )


def tail(out: Path) -> list[Path]:
    """report-tail.svg and its phone layout: both waits at once, beside closed-loop agents. The calls' time to first
    token as ECDFs, each pooled over a scheduler's two runs, on one log axis: (a) with the agents' prompts fresh each
    turn, (b) under it, with each grown by a tool result; stock vLLM, the KV preemption and the change at its
    16-token window, a dot where each crosses the 95th percentile and that time over the plot."""
    tails = data.tails()
    description = _tail_description(tails)
    return [
        _tail_wide(tails, out / "report-tail.svg", description),
        _tail_narrow(tails, out / "report-tail-narrow.svg", description),
    ]


def _apart(centres: Sequence[float], widths: Sequence[float], gap: float) -> list[float]:
    """Labels' centres, given in order along a line, each pair moved apart by the least that leaves ``gap`` between
    the two labels."""
    xs = list(centres)
    for _ in xs:
        for i in range(len(xs) - 1):
            short = (widths[i] + widths[i + 1]) / 2 + gap - (xs[i + 1] - xs[i])
            if short > 0:
                xs[i] -= short / 2
                xs[i + 1] += short / 2
    return xs


def _tail_lines(s: Chart, tails: dict[str, data.Tail], x: Scale, y: Scale) -> None:
    """A panel's lines and its p95 guide: each scheduler's ECDF, a dot where it crosses the guide, and over the plot's
    top, above its dot, the p95 itself, the change's in bold."""
    guide = y(TAIL_GUIDE / 100)
    s.rule(x.start, x.end, guide, "rule")
    s.text(x.start + 10, guide + 22, f"p{TAIL_GUIDE}", size=NOTE, cls="soft", halo=True)
    for engine in TAIL_LINES:
        s.trace(ecdf(tails[engine].turns, x, y, cap=data.TAIL_TIMEOUT_MS), _tail_cls(engine))
    crossings = sorted((tails[engine].percentile(TAIL_GUIDE), engine) for engine in TAIL_LINES)
    texts = [f"{seconds(ms)} s" for ms, _ in crossings]
    featured = [engine == TAIL_FEATURED for _, engine in crossings]
    widths = [measure(t, VALUE, bold=b) for t, b in zip(texts, featured, strict=True)]
    centres = _apart([x(ms) for ms, _ in crossings], widths, 12)
    for (ms, engine), text, bold, at in zip(crossings, texts, featured, centres, strict=True):
        s.mark(x(ms), guide, _tail_hue(engine), name=f"{_tail_name(engine)}'s p{TAIL_GUIDE}")
        s.text(at, y.end - 9, text, size=VALUE, anchor="middle", bold=bold)


def _tail_wide(tails: dict[str, dict[str, data.Tail]], path: Path, description: str) -> Path:
    """The two panels one above the other on one time axis, so that a line's tail in (a) stands over its tail in (b),
    and wide enough that each line's name fits right of it."""
    s = Chart(WIDTH, 700, "The calls' first tokens beside agents", description)
    x = Scale(*TAIL_AXIS, 96, 1080, log=True)
    labels = [f"{t / 1000:g}" for t in TAIL_TICKS]
    top, height, last = 76.0, 170.0, 0.0
    for k, (mode, (title, _)) in enumerate(TAIL_PANELS.items()):
        box = s.text(24, top - 46, title, bold=True)
        s.text(box.x1 + 16, top - 46, TAIL_SHARE, size=NOTE, cls="soft")
        y = Scale(0, 1, top + height, top)
        y_axis(s, y, TAIL_SHARES, [pct(v) for v in TAIL_SHARES], x.start, x.end)
        for tick in TAIL_TICKS:
            s.vrule(x(tick), y.end, y.start, "grid")
        if k == len(TAIL_PANELS) - 1:
            last = x_axis(s, x, TAIL_TICKS, labels, None, y.start, title=TAIL_TITLE)
        _tail_lines(s, tails[mode], x, y)
        for engine, share in TAIL_NAMES[mode].items():
            at = x(_quantile(tails[mode][engine].turns, share + 0.07))
            s.text(at + 12, y(share) + 6, _tail_name(engine), size=NOTE, halo=True)
        top = y.start + 96
    s.height = last + 14
    return s.save(path)


TAIL_NARROW_TICKS = [100, 1000, 10000]
"""The phone layout's ticks, a decade apart: the wide figure's would crowd its narrower axis."""


def _tail_narrow(tails: dict[str, dict[str, data.Tail]], path: Path, description: str) -> Path:
    """The phone layout: the key over the two panels, one above the other, on the same scales."""
    s = Chart(NARROW, 1100, "The calls' first tokens beside agents", description)
    x0 = 18
    last = legend(
        s,
        x0,
        30,
        [(line_key(_tail_cls(engine)), _tail_name(engine)) for engine in TAIL_LINES],
        width=NARROW - 2 * x0,
        pitch=25,
    )
    x = Scale(*TAIL_AXIS, 64, NARROW - 22, log=True)
    for mode, (title, _) in TAIL_PANELS.items():
        top = last + 46
        s.text(x0, top, title, bold=True)
        s.text(x0, top + 26, TAIL_SHARE, size=NOTE, cls="soft")
        y = Scale(0, 1, top + 68 + 200, top + 68)
        y_axis(s, y, TAIL_SHARES, [pct(v) for v in TAIL_SHARES], x.start, x.end)
        for tick in TAIL_TICKS:
            s.vrule(x(tick), y.end, y.start, "grid")
        last = x_axis(
            s, x, TAIL_NARROW_TICKS, [f"{t / 1000:g}" for t in TAIL_NARROW_TICKS], None, y.start
        )
        for line in wrap(TAIL_TITLE, NARROW - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
        _tail_lines(s, tails[mode], x, y)
    s.height = last + 14
    return s.save(path)


# Sizing the window ------------------------------------------------------------------------------------------------

WINDOWS = {16: "change", 48: "profile"}
"""The two windows the figure marks in every panel, and the configuration (data.LABELS) each is the window of, whose
hue (labels.MARKS) its rule and its dots take: the change's own 16 tokens and the serving profile's 48."""
WINDOW_AXIS = (12, 68)
"""The sweep's window axis, in tokens: half a step past its least window (16) and its greatest (64)."""
WINDOW_TOPS = (4000, 4000)
"""Panel b's axis's end, in ms, and panel c's, in prompt tokens a second: the first tick past each one's greatest."""


def _window_description(tokens: list[int], runs: list[data.KneeRun]) -> str:
    t40 = data.knee_medians(runs, "t40_p95")
    first = data.knee_medians(runs, "ttft_p95")
    prompt = data.knee_medians(runs, "prompt_tps")
    sweep = "; ".join(f"{w} tokens {seconds(t40[w])} s" for w in t40)
    rates = ", ".join(f"{prompt[w]:,.0f} at {w}" for w in prompt)
    return (
        f"In panel a, the cumulative share of {len(tokens):,} voice replies by their output tokens, "
        f"{statistics.median(tokens):.0f} at the median and {max(tokens)} at most, "
        f"{data.share_within(tokens, 16):.0%} ending within a 16-token window and all but "
        f"{sum(t > 48 for t in tokens)} within a 48-token one; in panel b, the p95 of a 40-token reply's last token "
        f"by the window, the median of two runs ({sweep}), its first token's staying between "
        f"{min(first.values()):.0f} and {max(first.values()):.0f} ms; in panel c, the agents' prompt tokens a "
        f"second by the window's tokens, {rates}, {1 - prompt[48] / prompt[16]:.0%} less at 48 than at 16."
    )


def window(out: Path) -> list[Path]:
    """report-window.svg and its phone layout: why the window is 48 tokens. (a) The ECDF of the voice replies' output
    tokens, with its median and its end, and the 16- and 48-token windows as rules with the share of replies each
    covers. (b) The window sweep: a 40-token reply's last token at p95, each window's median of two runs a dot with
    its value, and its first token's range. (c) The agents' prompt tokens a second beside it, each a dot with its
    value. The 16- and 48-token windows are marked the same way in all three: a rule in the hue of the configuration
    whose window it is, and a dot of that hue where it meets the line."""
    tokens = data.reply_tokens()
    runs = data.knee()
    description = _window_description(tokens, runs)
    return [
        _window_layout(tokens, runs, out / "report-window.svg", description, WIDTH),
        _window_layout(tokens, runs, out / "report-window-narrow.svg", description, NARROW),
    ]


def _window_hue(size: int) -> str:
    """A window's hue: that of the configuration whose window it is, and the change's for the sweep's others."""
    return config_mark(WINDOWS.get(size, "change"))[0]


def _window_replies(s: Chart, tokens: Sequence[int], x: Scale, y: Scale, wide: bool) -> None:
    """Panel a's marks: the replies' ECDF with its median and its end named, and each marked window's rule, named
    over the plot, with the share of replies that end within it beside the dot where the two meet."""
    within = {size: data.share_within(tokens, size) for size in WINDOWS}
    for size in WINDOWS:
        hue = _window_hue(size)
        s.ref((x(size), y.start), (x(size), y.end - 12), f"{hue} ref")
        s.text(
            x(size),
            y.end - 20,
            f"{size}-token window",
            size=NOTE,
            anchor="middle",
            bold=_featured(WINDOWS[size]),
        )
    s.trace(ecdf(tokens, x, y), "green trace")
    for size in WINDOWS:
        s.mark(x(size), y(within[size]), _window_hue(size), name=f"the {size}-token window's share")
    middle = statistics.median(tokens)
    s.mark(x(middle), y(0.5), "green", name="the replies' median")
    s.text(x(middle) + 12, y(0.5) + 6, f"median {middle:.0f}", size=NOTE, cls="soft", halo=True)
    longest = max(tokens)
    if (
        wide
    ):  # after the line's end; a phone's plot ends too soon after it, and names it under its end
        s.text(x(longest) + 10, y(1) + 6, f"max {longest}", size=NOTE, cls="soft", halo=True)
    else:
        s.text(x(max(WINDOWS)) + 9, y(1) + 28, f"max {longest}", size=NOTE, cls="soft", halo=True)
    first, second = sorted(WINDOWS)
    s.text(
        x(first) - 10, y(0.42) + 6, f"{within[first]:.0%} end", size=NOTE, anchor="end", halo=True
    )
    s.text(x(first) - 10, y(0.42) + 30, "within it", size=NOTE, anchor="end", halo=True)
    beyond = sum(t > second for t in tokens)
    at = y(0.26)
    s.text(
        x(second) - 10,
        at + 6,
        f"all but {beyond} of {len(tokens):,}",
        size=NOTE,
        anchor="end",
        halo=True,
    )
    s.text(x(second) - 10, at + 30, "end within it", size=NOTE, anchor="end", halo=True)


def _window_sweep(
    s: Chart,
    w: Scale,
    y: Scale,
    medians: dict[int, float],
    cls: str,
    texts: dict[int, str],
    *,
    falling: bool = False,
) -> dict[int, Box]:
    """A sweep's line through each window's median, a dot at each in its window's hue with ``texts``'s value over
    it, the featured window's in bold, and each marked window's rule from the axis up to its dot. A ``falling``
    line's values start over their dots, clear of the line that comes down to each. Returns each value's box."""
    for size in WINDOWS:
        if size in medians:
            s.ref((w(size), y.start), (w(size), y(medians[size])), f"{_window_hue(size)} ref")
    s.trace([(w(k), y(v)) for k, v in medians.items()], cls)
    boxes = {}
    for k, v in medians.items():
        s.mark(w(k), y(v), _window_hue(k), name=f"the {k}-token window")
        if k in texts:
            boxes[k] = s.text(
                w(k) - 4 if falling else w(k),
                y(v) - 12,
                texts[k],
                size=NOTE,
                anchor="start" if falling else "middle",
                halo=True,
                bold=_featured(WINDOWS.get(k, "change")),
            )
    return boxes


def _window_layout(
    tokens: Sequence[int], runs: list[data.KneeRun], path: Path, description: str, width: float
) -> Path:
    wide = width == WIDTH
    s = Chart(width, 1100, "Why the window is 48 tokens", description)
    x0 = 24 if wide else 18

    # (a) The replies' lengths and the two windows.
    x = Scale(0, 60, 96, 480) if wide else Scale(0, 60, 64, width - 22)
    y = Scale(0, 1, 366, 86) if wide else Scale(0, 1, 340, 100)
    s.text(x0, 30, f"(a) Share of the {len(tokens):,} voice replies", bold=True)
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_axis(s, y, marks, [pct(m) for m in marks], x.start, x.end)
    ticks = list(range(0, 61, 10))
    for mark in ticks[1:]:
        s.vrule(x(mark), y.end, y.start, "grid")
    last = x_axis(
        s, x, ticks, [str(m) for m in ticks], None, y.start, title="Output tokens of the reply"
    )
    _window_replies(s, tokens, x, y, wide)

    # (b) and (c), the sweep, on one window axis: beside (a), or on a phone under it.
    windows = sorted({run.window for run in runs})
    if wide:
        w = Scale(*WINDOW_AXIS, 624, 1076)
        b_title, b = 30, Scale(0, WINDOW_TOPS[0], 216, 86)
        c_title, c = 266, Scale(0, WINDOW_TOPS[1], 366, 302)
        left = 560
    else:
        w = Scale(*WINDOW_AXIS, 64, width - 22)
        b_title = last + 50
        b = Scale(0, WINDOW_TOPS[0], b_title + 44 + 200, b_title + 44)
        c_title = b.start + 46
        c = Scale(0, WINDOW_TOPS[1], c_title + 40 + 70, c_title + 40)
        left = x0
    s.text(left, b_title, "(b) A 40-token reply, p95 (s)", bold=True)
    marks_b, labels_b = linear_ticks(WINDOW_TOPS[0] / 1000, 2)
    y_axis(s, b, [m * 1000 for m in marks_b], labels_b, w.start, w.end)
    s.text(left, c_title, "(c) Agents' prompt tokens/s", bold=True)
    marks_c, labels_c = linear_ticks(WINDOW_TOPS[1], 2000)
    y_axis(s, c, marks_c, labels_c, w.start, w.end)
    for k in windows:
        if k not in WINDOWS:
            s.vrule(w(k), b.end, b.start, "grid")
            s.vrule(w(k), c.end, c.start, "grid")
    bottom = x_axis(
        s, w, windows, [str(k) for k in windows], None, c.start, title="The window (tokens)"
    )
    # (b): the first token under the 40th, named with its range over its line's start; the 40th's dots carry their
    # values, and its name stands over its end's.
    first = data.knee_medians(runs, "ttft_p95")
    _window_sweep(s, w, b, first, f"{_window_hue(0)} trace", {})
    at = b(max(first.values())) - 13
    box = s.text(w(windows[0]) + 10, at, "first token", size=NOTE, halo=True)
    s.text(
        box.x1 + 7,
        at,
        f"{seconds(min(first.values()))}\N{EN DASH}{seconds(max(first.values()))} s",
        size=NOTE,
        cls="soft",
        halo=True,
    )
    t40 = data.knee_medians(runs, "t40_p95")
    values = {k: f"{seconds(v)} s" for k, v in t40.items()}
    boxes = _window_sweep(s, w, b, t40, f"{_window_hue(0)} trace thick", values, falling=True)
    s.text(
        boxes[windows[-1]].x1,
        b(t40[windows[-1]]) - 36,
        "40th token",
        size=NOTE,
        anchor="end",
        halo=True,
    )
    # (c): the agents' prompt tokens a second, and what the 48-token window costs them from the 16-token one.
    prompt = data.knee_medians(runs, "prompt_tps")
    _window_sweep(
        s, w, c, prompt, f"{_window_hue(0)} trace", {k: f"{v:,.0f}" for k, v in prompt.items()}
    )
    small, large = sorted(WINDOWS)
    drop = 1 - prompt[large] / prompt[small]
    s.text(
        w(large) + 10,
        c(prompt[large]) + 26,
        f"\N{MINUS SIGN}{drop:.0%} from {small}",
        size=NOTE,
        halo=True,
    )
    s.height = (last if wide else bottom) + 14
    return s.save(path)


# The rules' cost per step -----------------------------------------------------------------------------------------

DECODE_STEP_MS = (16, 20)
"""A voice reply's decode step on the RTX 5060 Ti beside agent prefill that the budget cap keeps back, from the gaps between
its tokens (the prefill case's 16-20 ms a token, docs/BENCHMARKS.md): one of the box's steps, between a reply's step
alone (W6) and the agents' at 24 agent sessions (the rounds), which the figure's band of the box's steps spans."""
STEP_RANGE = (50, 30000)
"""The host's axis's ends, in µs: the last tick under the fastest call, and past the band of the box's steps."""
STEP_TICKS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
"""The host's axis, in µs, labelled in ms."""
STEP_NARROW_TICKS = [100, 1000, 10000]
"""The phone layout's ticks on the host's axis, its decades: the wide figure's nine would crowd it."""


def _step_row(cost: data.StepCost) -> str:
    return f"{cost.running} running" + (f", {cost.variant}" if cost.variant else "")


STEP_SETS = {
    False: "In the pull requests' cost tables",
    True: "In the serving profile",
}
"""The two sets of states, each under its heading: the PRs' tables' steady states, and the box's configuration (its
256-token steps scheduled asynchronously, which the caption says)."""
STEP_AXIS = "Time of a scheduler call on the host (ms, log scale)"


def _box_steps(box: float) -> tuple[float, float]:
    """The band of the box's steps on the host's axis, in µs: from a voice reply's step alone to the agents' at 24
    agent sessions, each divided by how many times as long the box's CPU takes, so that a scheduler call stands
    against them as it would on the box."""
    return data.voice_step_ms() * 1000 / box, data.agents_step_ms() * 1000 / box


BAND = "gray wash"
"""The box's band of steps: the gray's cell fill, which stands off the page in either theme (1.3:1 on the light page and
1.7:1 on the dark), where its area tint all but vanished."""
BAND_NAME = "The box's steps, scaled to the host"
"""The band's name in the key; the caption gives its ends and the factor it is divided by."""


def _signed(us: float) -> str:
    """A change to a call in µs with its sign, a true minus for a saving."""
    return f"{us:+,.0f}".replace("-", "\N{MINUS SIGN}")


def _step_off(costs: Sequence[data.StepCost]) -> int:
    """How near the change with its switches off stays to stock vLLM's call in every state: the greatest of the
    differences between the two, which the baseline gives to 0.1 µs, rounded up to a µs so that "within" holds for
    each."""
    return math.ceil(round(max(abs(c.off - c.stock) for c in costs), 1))


def _step_note(costs: Sequence[data.StepCost]) -> str:
    """What the figure says in words of the third scheduler it measures, the change with its switches off, whose
    call would land on stock vLLM's dot in every row."""
    stock = data.in_prose(data.label("stock")[0])
    return (
        f"With its switches off, the change is within {_step_off(costs)} µs of {stock} in every row"
    )


def _step_description(costs: Sequence[data.StepCost]) -> str:
    """The figure in one sentence: its states' cost, the switches off against stock vLLM, and what the configuration to
    deploy adds in each scenario of each set, the least to the most."""
    worst = max(abs(c.off_ratio) for c in costs)
    sets = []
    for serving in (False, True):
        scenarios: dict[str, list[data.StepCost]] = {}
        for c in costs:
            if c.serving == serving:
                scenarios.setdefault(c.scenario, []).append(c)
        cells = []
        for scenario, states in scenarios.items():
            least, most = min(c.added for c in states), max(c.added for c in states)
            span = (
                f"{_signed(least)} µs"
                if len(states) == 1
                else f"{_signed(least)} to {_signed(most)} µs"
            )
            cells.append(f"{data.SCENARIOS[scenario]} ({span})")
        where = "in the serving profile" if serving else "in the pull requests' cost tables"
        sets.append(f"{where}: {', '.join(cells[:-1])} and {cells[-1]}")
    lo, hi = DECODE_STEP_MS
    edge = _box_steps(data.box_cpu())[0]
    slowest = max(c.on for c in costs)
    return (
        f"The wall time of one call of vLLM 0.32.0's Scheduler.schedule() on the host's CPU in {len(costs)} states, on "
        f"a log scale, beside a band of the box's steps from a voice reply's alone ({data.voice_step_ms():.0f} ms; "
        f"beside held agent prefill, {lo} to {hi} ms) to the agents' at 24 agent sessions "
        f"({data.agents_step_ms():.0f} ms), each divided by {data.box_cpu():.2f}, how many times as long the box's CPU "
        f"takes: the change with its switches off within {_step_off(costs)} µs and {worst:.1%} of stock vLLM in "
        "every state, and the "
        f"serving profile's change to the call in each scenario, {'; '.join(sets)}, its slowest call "
        f"{slowest:,.0f} µs against the band's lower edge at {edge:,.0f} µs."
    )


def _band_span(box: float) -> str:
    """What the band spans, under its name: the box's steps as measured, and where they stand on the host's axis."""
    lo, hi = (v / 1000 for v in _box_steps(box))
    return (
        f"{data.voice_step_ms():.0f} to {data.agents_step_ms():.0f} ms \N{DIVISION SIGN} {box:.2f}: "
        f"{lo:.1f} to {hi:.1f} ms here"
    )


def _band_key() -> Key:
    """The box's band of steps in a legend: a short stretch of it."""

    def draw(s: Chart, x: float, baseline: float) -> None:
        s.rect(x, baseline - 0.75 * NOTE, 22, 0.9 * NOTE, BAND)

    return Key(22, draw)


CLEAR = 6.0
"""How near a label may come to a reference rule beside it, in units."""


def _beside(
    s: Chart, marks: tuple[float, float], baseline: float, label: str, rules: Sequence[float]
) -> None:
    """``label`` right of a row's marks (their leftmost and rightmost x), or left of them where it would reach one of
    the vertical ``rules``: a label never crosses a reference rule."""
    width = measure(label, NOTE)
    for x0, anchor in ((marks[1] + 12, "start"), (marks[0] - 12, "end")):
        left = x0 if anchor == "start" else x0 - width
        if not any(left - CLEAR <= r <= left + width + CLEAR for r in rules):
            s.text(x0, baseline, label, size=NOTE, anchor=anchor, halo=True)
            return
    raise ValueError(f"{label!r} reaches a rule on either side of its marks")


def step_cost(out: Path) -> list[Path]:
    """report-step-cost.svg and its phone layout: the rules' cost per scheduler step on the host, a row per state under
    its scenario, the PRs' tables' steady states and then the serving profile's: stock vLLM's call a gray dot and the
    serving profile's a blue dot joined to it, with what the profile's values add printed after the pair, on one log
    scale of the host's time. The change's call with its switches off would land on stock vLLM's dot in every row, so
    a line under the key says how near it stays. A band stands for the box's steps, from a voice reply's alone to
    the agents' at 24 agent sessions, each divided by bench_host.py's factor between the box's CPU and the host's,
    so that a call stands against them as it would on the box."""
    costs = data.step_costs()
    paths = [
        _step_cost_wide(costs, out / "report-step-cost.svg"),
        _step_cost_narrow(costs, out / "report-step-cost-narrow.svg"),
    ]
    return paths


STEP_KEYS = ("stock", "profile")
"""The configurations a state's row marks, by their keys in data.LABELS: stock vLLM's dot and the serving
profile's."""


def _step_keys() -> list[tuple[Key, str]]:
    """The two marks' key; the band of the box's steps is named beside it, or in a phone's key."""
    return [
        (mark_key(hue, shape, filled=filled), data.label(key)[0])
        for key in STEP_KEYS
        for hue, shape, filled in [config_mark(key)]
    ]


def _step_marks(
    s: Chart, x: Scale, cost: data.StepCost, row_y: float, rules: Sequence[float]
) -> None:
    """A state's pair: stock vLLM's dot and the serving profile's, joined, with what the profile's values add to the
    call after the pair."""
    ends = (cost.stock, cost.on)
    s.trace([(x(v), row_y) for v in ends], "blue range", wire=False)
    for key, value in zip(STEP_KEYS, ends, strict=True):
        hue, shape, filled = config_mark(key)
        s.mark(x(value), row_y, hue, shape, filled=filled, name=f"{data.label(key)[0]}'s call")
    # What the profile adds, right of the pair, or left of it where the band stands in the way: the budget cap's is
    # a saving, the serving profile's dot left of stock vLLM's.
    _beside(s, (x(min(ends)), x(max(ends))), row_y + 6, f"{_signed(cost.added)} µs", rules)


def _step_band(s: Chart, x: Scale, top: float, bottom: float, box: float) -> list[float]:
    """The band of the box's steps from ``top`` to ``bottom``; returns its edges, which no value label crosses."""
    lo, hi = (x(v) for v in _box_steps(box))
    s.rect(lo, top, hi - lo, bottom - top, BAND)
    return [lo, hi]


def _step_heading(scenario: str) -> str:
    """A scenario's heading over its states: its name in data.SCENARIOS, as a sentence starts."""
    return data.SCENARIOS[scenario][0].upper() + data.SCENARIOS[scenario][1:]


def _step_cost_narrow(costs: Sequence[data.StepCost], path: Path) -> Path:
    """The phone layout: the key with the band's name and the line on the switches off, the axis's ticks over the
    column as under it, then each
    state's running requests left of its marks, what else sets it apart under them, under its set's and its
    scenario's headings."""
    box = data.box_cpu()
    s = Chart(NARROW, 2600, "The rules' cost per scheduler step", _step_description(costs))
    x0 = 18
    names = max(measure(f"{c.running} running", VALUE) for c in costs)
    x = Scale(*STEP_RANGE, x0 + names + 16, NARROW - 22, log=True)
    keys = [*_step_keys(), (_band_key(), BAND_NAME)]
    last = legend(s, x0, 30, keys, width=NARROW - 2 * x0, pitch=25)
    for line in wrap(_step_note(costs), NARROW - 2 * x0, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    # The axis's ticks over the column too, under the key: on a phone its foot is a screen below.
    ticks_y = last + 38
    labels = [f"{t / 1000:g}" for t in STEP_NARROW_TICKS]
    for t, label in zip(STEP_NARROW_TICKS, labels, strict=True):
        s.text(x(t), ticks_y, label, size=NOTE, anchor="middle", cls="mute")
    top = ticks_y + 12
    layout = []
    y = top
    scenario = serving = None
    # The headings end short of the band of the box's steps, which runs the whole column.
    room = x(_box_steps(box)[0]) - 10 - x0
    for cost in costs:
        opens = cost.serving != serving
        if opens:
            serving = cost.serving
            y += 18 if layout else 0
            for line in wrap(STEP_SETS[serving], room, LABEL):
                s.text(x0, y + 22, line, bold=True)
                y += 26
            y += 2
        if cost.scenario != scenario:
            scenario = cost.scenario
            y += 0 if opens else 8  # air over a heading that follows another scenario's states
            for line in wrap(_step_heading(scenario), room, NOTE):
                s.text(x0, y + 19, line, size=NOTE, cls="soft")
                y += 22
            y += 6
        layout.append((y + 15, cost))
        y += 30 + (22 if cost.variant else 0)
    bottom = y + 8
    rules = _step_band(s, x, top, bottom, box)
    for row_y, cost in layout:
        band_grid(s, x, STEP_NARROW_TICKS, row_y - 14, row_y + 14)
        s.text(x0, row_y + 7, f"{cost.running} running", size=VALUE)
        if cost.variant:
            s.text(x0, row_y + 29, cost.variant, size=NOTE, cls="soft", halo=True)
        _step_marks(s, x, cost, row_y, rules)
    last = x_axis(s, x, STEP_NARROW_TICKS, labels, None, bottom)
    for line in wrap(STEP_AXIS, NARROW - 2 * x0, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


def _step_cost_wide(costs: Sequence[data.StepCost], path: Path) -> Path:
    box = data.box_cpu()
    s = Chart(WIDTH, 1300, "The rules' cost per scheduler step", _step_description(costs))
    x0 = 24
    names = max(measure(_step_row(c), VALUE) for c in costs)
    right = x0 + names + 12
    x = Scale(*STEP_RANGE, right + 16, 1090, log=True)
    last = legend(s, x.start, 30, _step_keys(), width=x.end - x.start)
    last += 26
    s.text(x.start, last, _step_note(costs), size=NOTE, cls="soft")
    top = last + 24
    layout = []
    y = top
    scenario = serving = None
    for cost in costs:
        opens = cost.serving != serving
        if opens:
            serving = cost.serving
            y += 18 if layout else 0
            s.text(x0, y + 22, STEP_SETS[serving], bold=True)
            y += 28
        if cost.scenario != scenario:
            scenario = cost.scenario
            y += 0 if opens else 8  # air over a heading that follows another scenario's states
            for line in wrap(_step_heading(scenario), right - x0, NOTE):
                s.text(x0, y + 19, line, size=NOTE, cls="soft")
                y += 22
            y += 6
        layout.append((y + 15, cost))
        y += 30
    bottom = y + 8
    rules = _step_band(s, x, top, bottom, box)
    last = x_axis(
        s, x, STEP_TICKS, [f"{t / 1000:g}" for t in STEP_TICKS], top, bottom, title=STEP_AXIS
    )
    # The band's name left of its top, level with the first set's heading, and what it spans under it.
    s.text(rules[0] - 12, top + 22, BAND_NAME, size=NOTE, anchor="end", halo=True)
    s.text(
        rules[0] - 12,
        top + 47,
        _band_span(box),
        size=NOTE,
        anchor="end",
        cls="soft",
        halo=True,
    )
    for row_y, cost in layout:
        s.text(right, row_y + 7, _step_row(cost), size=VALUE, anchor="end")
        _step_marks(s, x, cost, row_y, rules)
    s.height = last + 14
    return s.save(path)


# The headline -----------------------------------------------------------------------------------------------------
#
# The headline's configurations, each labelled as data.LABELS names it and marked as labels.MARKS does, voice alone,
# the reference, first: figure.toml orders them.


def _runs_of(configuration: data.Configuration) -> str:
    return _runs_of_count(len(configuration.runs))


def _runs_of_count(count: int) -> str:
    return f"{count} run{'' if count == 1 else 's'}"


def _headline_description(configurations: Sequence[data.Configuration], slo: float) -> str:
    base = configurations[1].tokens
    rows = []
    for c in configurations:
        name, note = data.label(c.point)
        p95s = [run.p95 for run in c.runs]
        text = (
            f"{data.in_prose(name)}, {note}: {seconds(c.p95)} s at the median of {_runs_of(c)} "
            f"({seconds(min(p95s))} to {seconds(max(p95s))} s)"
        )
        if _featured(c.point):
            text += f", {_from_slo(c.p95, slo)}"
        if c.tokens is not None and base:
            text += (
                f", its agents at {c.tokens:,.0f} tokens/s, {c.tokens / base:.0%} of tuned vLLM's"
            )
        rows.append(text)
    return (
        f"Each configuration's voice-to-voice p95 as a bar from zero against the SLO of {seconds(slo)} s, and its "
        "agents' tokens a second over the calls' first 240 s, each at the median of its runs with their spread: "
        + "; ".join(rows[:-1])
        + f"; and {rows[-1]}."
    )


def headline(out: Path) -> list[Path]:
    """report-headline.svg and its phone layout: the headline configurations beside 24 agent sessions on the
    RTX 5060 Ti, with voice alone on top, a bar a configuration at the median of its runs and a thin line over the
    bar's end from the least run to the greatest. (a) The V2V p95 from zero, against the SLO. (b) The agents' tokens a
    second over the calls' first 240 s."""
    rows = data.w5_rows()
    configurations = data.headline(rows)
    slo = data.slo_ms(rows)
    description = _headline_description(configurations, slo)
    return [
        _headline_wide(configurations, slo, out / "report-headline.svg", description),
        _headline_narrow(configurations, slo, out / "report-headline-narrow.svg", description),
    ]


LABELS_END = 264.0
"""Where a wide figure's row names end, right-aligned: the same column in every figure whose rows are the report's
configurations, so that the figures line up down a page."""
PLOT_START = 280.0
"""And where their plots start."""
BAR = 24.0
"""A value bar's thickness: enough to stand as a row's one mark, thinner than the row's two lines of label."""
VALUE_GAP = 9.0
"""Between a bar's end, or its runs' spread, and its value."""
FEATURED = "blue"
"""The hue of the configuration a figure is drawn for (labels.MARKS): its name and its value are set in semibold."""


def _featured(point: str) -> bool:
    return config_mark(point)[0] == FEATURED


def _value_bar(
    s: Chart, scale: Scale, y: float, value: float, spread: Sequence[float], hue: str, name: str
) -> float:
    """A configuration's value as a bar from the scale's start, centred on ``y``, with its runs' ``spread`` as a
    capped line over the bar's end where it is at least SPREAD_MIN wide. Returns where the marks end."""
    s.bar(scale.start, y - BAR / 2, scale(value) - scale.start, BAR, hue, name=name)
    lo, hi = scale(min(spread)), scale(max(spread))
    if hi - lo <= SPREAD_MIN:
        return scale(value)
    s.span(lo, hi, y)
    return max(hi, scale(value))


EXTENSIONS = {
    "blue": ("hi", "soft"),
    "sky": ("soft", "blue-pale"),
    "gray": ("lo", "pale"),
    "ink": ("mid", "lo"),
}
"""A bar's fill and its extension's, by hue: the extension is the next softer tone of the same hue."""


def _extended_bar(
    s: Chart, scale: Scale, y: float, first: float, second: float, hue: str, name: str
) -> float:
    """Two nested values of one row as one bar, centred on ``y``: from the scale's start to ``first`` in the hue's
    fill, then on to ``second`` in its softer tone, parted by a seam (a median and its 95th percentile, a reply's
    first token and its 16th). Returns where the bar ends."""
    x0, x1, x2 = scale.start, scale(first), scale(second)
    fill, more = EXTENSIONS[hue]
    s.segments(x0, y - BAR / 2, [(x1 - x0, fill), (x2 - x1, more)], BAR)
    s.marks.append(Box(x0, y - BAR / 2, x2, y + BAR / 2, name))
    return x2


def _v2v_row(s: Chart, c: data.Configuration, a: Scale, y: float, slo: float) -> None:
    """A configuration's V2V p95 at ``y``: its bar, and its value after the bar, or past the SLO's rule where the
    value would cross it. The featured row's value is followed by how far it stands from the ``slo`` (ms), the
    figure's point in its own words."""
    name = data.label(c.point)[0]
    end = _value_bar(s, a, y, c.p95, [r.p95 for r in c.runs], config_mark(c.point)[0], name)
    box = _after_bar(s, end, y, f"{seconds(c.p95)} s", _featured(c.point), a(slo))
    if _featured(c.point):
        _slo_note(s, box, y, c.p95, slo)


def _from_slo(p95: float, slo: float) -> str:
    """Where a V2V p95 stands against the SLO (both ms), from the unrounded values."""
    return f"{seconds(p95 - slo)} s over the SLO" if p95 > slo else "within the SLO"


def _slo_note(s: Chart, value: Box, y: float, p95: float, slo: float) -> None:
    """After a p95's ``value`` in the row at ``y``, where it stands against the SLO: in the limit's red where it is
    over, as time the voice loses, and in the soft ink where it is within."""
    cls = "red tint" if p95 > slo else "soft"
    s.text(value.x1 + 10, y + 7, _from_slo(p95, slo), size=NOTE, cls=cls, halo=True)


def _tokens_row(s: Chart, c: data.Configuration, b: Scale, y: float, base: float) -> None:
    """A configuration's agents' tokens a second at ``y``: its bar, its value, and its share of tuned vLLM's."""
    assert c.tokens is not None
    name = data.label(c.point)[0]
    spread = [r.tokens for r in c.runs if r.tokens is not None]
    end = _value_bar(s, b, y, c.tokens, spread, config_mark(c.point)[0], name)
    bold = _featured(c.point)
    box = s.text(end + VALUE_GAP, y + 7, f"{c.tokens:,.0f}", size=VALUE, bold=bold, halo=True)
    s.text(box.x1 + 8, y + 7, f"{c.tokens / base:.0%}", size=NOTE, cls="soft", halo=True)


V2V_TITLE = "Time (s)"
"""A V2V p95 axis's title under a panel titled "Voice-to-voice p95" (figure_style's grammar: the axis names its kind
where the panel names the measure)."""
TOKENS_TITLE = "Tokens/s over the calls' first 240 s"
"""The agents' token rate's axis title, over the window every figure of the report gives it in."""
HEADLINE_PANELS = ("(a) Voice-to-voice p95", "(b) Agents' tokens/s")
HEADLINE_PITCH = 56.0
"""A row's pitch where the rows stand without headings: the headline's, and the deployable figure's."""


def _v2v_top(configurations: Sequence[data.Configuration]) -> int:
    """The V2V axis's end: the even second at or past every run."""
    worst = max(r.p95 for c in configurations for r in c.runs)
    return 2000 * math.ceil(worst / 2000)


def _headline_wide(
    configurations: Sequence[data.Configuration], slo: float, path: Path, description: str
) -> Path:
    s = Chart(WIDTH, 500, "The voice tail and the agents' throughput", description)
    right = LABELS_END
    top_ms = _v2v_top(configurations)
    a = Scale(0, top_ms, PLOT_START, 728)
    b = Scale(0, 4000, 818, 998)
    s.text(a.start, 30, HEADLINE_PANELS[0], bold=True)
    s.text(b.start, 30, HEADLINE_PANELS[1], bold=True)
    top, pitch = 76, HEADLINE_PITCH
    ys = [top + pitch / 2 + pitch * k for k in range(len(configurations))]
    bottom = ys[-1] + pitch / 2
    ticks = list(range(0, top_ms + 1, 2000))
    last = x_axis(s, a, ticks, [f"{t // 1000}" for t in ticks], top, bottom, title=V2V_TITLE)
    marks_b, labels_b = linear_ticks(4000, 2000)
    x_axis(s, b, marks_b, labels_b, top, bottom, title=TOKENS_TITLE)
    base = configurations[1].tokens
    for y, c in zip(ys, configurations, strict=True):
        name, note = data.label(c.point)
        s.text(right, y - 3, name, size=VALUE, anchor="end", bold=_featured(c.point))
        s.text(right, y + 19, note, size=NOTE, anchor="end", cls="soft")
        _v2v_row(s, c, a, y, slo)
        if c.tokens is not None:
            _tokens_row(s, c, b, y, base)
    _v_threshold(s, a(slo), top, bottom, f"SLO {seconds(slo)} s")
    s.height = last + 14
    return s.save(path)


def _headline_narrow(
    configurations: Sequence[data.Configuration], slo: float, path: Path, description: str
) -> Path:
    s = Chart(NARROW, 900, "The voice tail and the agents' throughput", description)
    x0 = 18
    top_ms = _v2v_top(configurations)
    a = Scale(0, top_ms, x0, 322)
    b = Scale(0, 4000, x0, 300)
    ticks = list(range(0, top_ms + 1, 4000))
    base = configurations[1].tokens
    y = 30
    for panel in "ab":
        scale = a if panel == "a" else b
        s.text(x0, y, HEADLINE_PANELS[panel == "b"], bold=True)
        y += 12
        if panel == "a":
            y += 30
            _threshold_key(s, x0, y - 8, f"SLO {seconds(slo)} s")
        marks = ticks if panel == "a" else [0, 2000, 4000]
        for c in configurations:
            if panel == "b" and c.tokens is None:
                continue
            name, note = data.label(c.point)
            box = s.text(x0, y + 20, name, size=VALUE, bold=_featured(c.point))
            s.text(box.x1 + 8, y + 20, note, size=NOTE, cls="soft")
            band = y + 46
            band_grid(s, scale, marks, band - 14, band + 14)
            if panel == "a":
                _v2v_row(s, c, a, band, slo)
                _threshold_stub(s, a(slo), band)
            else:
                _tokens_row(s, c, b, band, base)
            y = band + 18
        labels = [f"{t // 1000}" for t in ticks] if panel == "a" else linear_ticks(4000, 2000)[1]
        last = x_axis(s, scale, marks, labels, None, y)
        for line in wrap(V2V_TITLE if panel == "a" else TOKENS_TITLE, NARROW - 36, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
        y = last + 48
    s.height = y - 48 + 14
    return s.save(path)


# Every turn -------------------------------------------------------------------------------------------------------

CDF_LINES = {
    data.VOICE_ALONE: "ink trace dotted",
    data.TUNED: "gray trace",
    data.ROUTER: "gray trace dashed",
    data.CHANGE: "blue trace thick",
}
"""Each configuration's line, named as data.LABELS names it: its stroke."""
CDF_TICKS = [1000, 1500, 2000, 3000, 5000, 10000, 20000]
CDF_START = CDF_TICKS[0]
"""The ECDF's log axis, from 1 s, under every turn measured (the fastest took 1.38 s) with room left of the first
line for its name, to 20 s, past the slowest (18.62 s); a turn that got no reply, which counts at the reply timeout,
would pass its end, and draws no figure."""


CDF_PANELS = ("(a) Cumulative share of replies", "(b) Replies within the SLO")


def _within(samples: Sequence[float], slo: float) -> float:
    """The share of ``samples`` at or under the SLO: the replies that began within it."""
    return sum(v <= slo for v in samples) / len(samples)


def _cdf_description(configurations: Sequence[data.Configuration], slo: float) -> str:
    rows = []
    for c in configurations:
        samples = c.samples()
        rows.append(
            f"{data.in_prose(data.label(c.point)[0])}, {len(samples)} replies over {_runs_of(c)}: "
            f"{_within(samples, slo):.0%} within the SLO, the slowest at {seconds(samples[-1])} s"
        )
    return (
        "The cumulative share of every measured turn's voice-to-voice time, pooled over each configuration's runs, on "
        f"a log scale from {CDF_START / 1000:g} to {CDF_TICKS[-1] / 1000:g} s with the SLO of {seconds(slo)} s and "
        "the 95th percentile marked, and beside it the share of each configuration's replies that began within the "
        "SLO as a bar: " + "; ".join(rows) + "."
    )


def cdf(out: Path) -> list[Path]:
    """report-cdf.svg and its phone layout, for voice alone, tuned vLLM, preempt-resubmit and the configuration to
    deploy. (a) Every measured turn's V2V as ECDFs, pooled over each configuration's runs (the warm-up turns left
    out, a turn that got no reply counted at 30 s), a dot where each line crosses the SLO. (b) What the dots read: the
    share of each configuration's replies within the SLO as a bar, its runs' shares as the thin line over its end."""
    rows = data.w5_rows()
    configurations = [c for c in data.headline(rows) if c.point in CDF_LINES]
    slo = data.slo_ms(rows)
    slowest = max(v for c in configurations for v in c.samples())
    if slowest > CDF_TICKS[-1]:
        raise ValueError(f"a turn took {slowest:,.0f} ms, past the ECDF's axis: extend CDF_TICKS")
    description = _cdf_description(configurations, slo)
    return [
        _cdf_layout(configurations, slo, out / "report-cdf.svg", description, WIDTH),
        _cdf_layout(configurations, slo, out / "report-cdf-narrow.svg", description, NARROW),
    ]


WITHIN_MARKS = (0.0, 0.5, 1.0)
"""Panel b's axis's marks: shares of the replies."""
WITHIN_TITLE = "Share of the replies"


def _within_row(s: Chart, c: data.Configuration, b: Scale, top: float, slo: float) -> float:
    """A configuration's share of replies within the SLO as a row under ``top``: its name over its bar, the pooled
    share printed after the bar, and its runs' shares as the spread. Returns the row's bottom."""
    name = data.label(c.point)[0]
    bold = _featured(c.point)
    s.text(b.start, top + 20, name, size=VALUE, bold=bold)
    band = top + 46
    band_grid(s, b, WITHIN_MARKS, band - 14, band + 14)
    share = _within(c.samples(), slo)
    spread = [_within(run.samples, slo) for run in c.runs]
    end = _value_bar(s, b, band, share, spread, config_mark(c.point)[0], name)
    s.text(end + VALUE_GAP, band + 7, pct(share), size=VALUE, bold=bold, halo=True)
    return band + 18


def _cdf_layout(
    configurations: Sequence[data.Configuration],
    slo: float,
    path: Path,
    description: str,
    width: float,
) -> Path:
    wide = width == WIDTH
    s = Chart(width, 900, "Every turn's voice-to-voice time", description)
    title = "Voice-to-voice time of a reply (s, log scale)"
    if wide:  # the lines named beside themselves; the shares in a column of rows after the plot, as tall as it
        x = Scale(CDF_START, CDF_TICKS[-1], 96, 716, log=True)
        y = Scale(0, 1, 436, 86)
        s.text(x.start, 30, CDF_PANELS[0], bold=True)
    else:  # the lines named in a key over the plot, where a phone's plot has no room beside them
        s.text(18, 30, CDF_PANELS[0], bold=True)
        keys = [(line_key(cls), data.label(point)[0]) for point, cls in CDF_LINES.items()]
        last_key = legend(s, 18, 64, keys, width=width - 36, pitch=25)
        x = Scale(CDF_START, CDF_TICKS[-1], 64, width - 22, log=True)
        y = Scale(0, 1, last_key + 52 + 300, last_key + 52)
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_axis(s, y, marks, [pct(m) for m in marks], x.start, x.end)
    for tick in CDF_TICKS[1:]:
        s.vrule(x(tick), y.end, y.start, "grid")
    labels = [f"{t / 1000:g}" for t in CDF_TICKS]
    if wide:
        last = x_axis(s, x, CDF_TICKS, labels, None, y.start, title=title)
    else:
        last = x_axis(s, x, CDF_TICKS, labels, None, y.start)
        for line in wrap(title, width - 36, NOTE):
            last += 25
            s.text(18, last, line, size=NOTE, cls="soft")
    s.rule(x.start, x.end, y(0.95), "rule")
    s.text(x.start + 10, y(0.95) + 22, "p95", size=NOTE, cls="soft", halo=True)
    # The SLO's rule rises a little over the plot, so that its name clears the dot of a line that crosses it at the top.
    _v_threshold(s, x(slo), y.end - 8, y.start, f"SLO {seconds(slo)} s")
    lines = {}
    for c in configurations:
        name, cls = data.label(c.point)[0], CDF_LINES[c.point]
        points = ecdf(c.samples(), x, y, cap=REPLY_TIMEOUT_MS)
        s.trace(points, cls)
        lines[c.point] = (name, points)
    for c in configurations:  # where each line crosses the SLO: the share panel (b) prints
        s.mark(
            x(slo),
            y(_within(c.samples(), slo)),
            CDF_LINES[c.point].split()[0],
            name=f"{data.label(c.point)[0]}'s share within the SLO",
        )
    if wide:
        # The labels: the change's after its tail's end, the router's beside its line where it parts from the
        # change's, voice alone's in the space left of its line, tuned vLLM's under its line's middle.
        name, points = lines[data.CHANGE]
        s.text(points[-1][0] + 10, points[-1][1] + 6, name, size=NOTE, halo=True)
        by_point = {c.point: c for c in configurations}
        at = x(_quantile(by_point[data.ROUTER].samples(), 0.88))
        s.text(at + 12, y(0.88) + 6, lines[data.ROUTER][0], size=NOTE, halo=True)
        at = x(_quantile(by_point[data.VOICE_ALONE].samples(), 0.8))
        s.text(at - 12, y(0.8) + 6, lines[data.VOICE_ALONE][0], size=NOTE, anchor="end", halo=True)
        at = x(_quantile(by_point[data.TUNED].samples(), 0.45))
        s.text(at + 16, y(0.45) + 12, lines[data.TUNED][0], size=NOTE, halo=True)
        b = Scale(0, 1, 812, 1032)
        s.text(b.start, 30, CDF_PANELS[1], bold=True)
        pitch = (y.start - y.end) / len(configurations)
        for k, c in enumerate(configurations):
            _within_row(s, c, b, y.end - 6 + pitch * k, slo)
        marks = [pct(m) for m in WITHIN_MARKS]
        x_axis(s, b, WITHIN_MARKS, marks, None, y.start, title=WITHIN_TITLE)
        s.height = last + 14
    else:
        b = Scale(0, 1, 18, 300)
        top = last + 48
        s.text(18, top, CDF_PANELS[1], bold=True)
        top += 12
        for c in configurations:
            top = _within_row(s, c, b, top, slo)
        last = x_axis(s, b, WITHIN_MARKS, [pct(m) for m in WITHIN_MARKS], None, top)
        s.text(18, last + 25, WITHIN_TITLE, size=NOTE, cls="soft")
        s.height = last + 25 + 14
    return s.save(path)


def _quantile(samples: Sequence[float], share: float) -> float:
    """The smallest sample at or above ``share`` of the samples: where the ECDF reaches it."""
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))]


# What the agents pay ----------------------------------------------------------------------------------------------

AGENT_PANELS = (
    ("(a) Agents' TTFT p99", "Time (s)"),
    ("(b) Steps held", "Share of the engine's steps"),
    ("(c) KV preemptions", "Preemptions in a run"),
)
"""Each panel's title and its axis's title: the agents' first-token tail, and what each rule did, the budget cap's
steps held and the KV preemption's preemptions as make-room counts them."""
AGENT_NONE = ("no budget cap", "no KV preemption")
"""What panels b and c say, at the axis, for a configuration that runs neither rule."""
AGENT_STEPS = (50.0, 0.2, 25.0)
"""Each panel's ticks' step: its axis ends at the first tick at or past its largest run."""
AGENT_SPANS = ((PLOT_START, 540.0), (680.0, 840.0), (930.0, 1060.0))
"""Where each panel's plot starts and ends in the wide layout: each as wide as its bars and their values need, the
first from where every figure's plot starts beside the configurations' names."""


def _agent_values(cells: Sequence[data.AgentRun], k: int) -> list[float]:
    """Panel ``k``'s measure in each of a configuration's runs that has it: the agents' TTFT p99 in seconds (0), the
    share of the engine's steps held (1) or the preemptions (2), neither of which a run without the rule counts."""
    if k == 0:
        return [r.ttft_p99 / 1000 for r in cells]
    values = [r.held_share if k == 1 else r.preemptions for r in cells]
    return [v for v in values if v is not None]


def _agent_range(values: Sequence[float], k: int) -> str:
    """The least and the greatest of a configuration's runs, as the description gives them after a median."""
    low, high = min(values), max(values)
    return f"{low:.0%} to {high:.0%}" if k == 1 else f"{low:.0f} to {high:.0f}"


def _agents_description(runs: dict[str, list[data.AgentRun]]) -> str:
    base = statistics.median(_agent_values(runs[data.TUNED], 0))
    rows = []
    for point, cells in runs.items():
        name, note = data.label(point)
        p99s, shares, preemptions = (_agent_values(cells, k) for k in range(3))
        text = (
            f"{data.in_prose(name)} ({note}, {_runs_of_count(len(cells))}): TTFT p99 "
            f"{statistics.median(p99s):.0f} s ({_agent_range(p99s, 0)})"
        )
        if point != data.TUNED:
            text += f", {statistics.median(p99s) / base:.0%} of tuned vLLM's"
        text += (
            f", agent prefill held in {statistics.median(shares):.0%} of the steps ({_agent_range(shares, 1)})"
            if shares
            else f", {AGENT_NONE[0]}"
        )
        text += (
            f", {statistics.median(preemptions):.0f} preemptions by make-room ({_agent_range(preemptions, 2)})"
            if preemptions
            else f", {AGENT_NONE[1]}"
        )
        rows.append(text)
    return (
        "For each configuration beside 24 agent sessions, the median of its runs and their range for the agents' "
        "TTFT p99, for the share of the engine's steps over the calls in which the budget cap kept agent prefill "
        "out, and for make-room's own preemptions over the calls: "
        + "; ".join(rows[:-1])
        + f"; and {rows[-1]}."
    )


def agents(out: Path) -> list[Path]:
    """report-agents.svg and its phone layout: what the agents pay for the headline's configurations that run agents,
    in its rows, as three panels of value bars. (a) The agents' TTFT p99, with its share of tuned vLLM's. (b) The
    share of the engine's steps in which the budget cap kept agent prefill out. (c) make-room's own preemptions. Each
    bar the median of the configuration's runs, with a line over their range and its value; a configuration that
    runs neither rule says so at the axis."""
    runs = data.agent_runs()
    description = _agents_description(runs)
    return [
        _agents_wide(runs, out / "report-agents.svg", description),
        _agents_narrow(runs, out / "report-agents-narrow.svg", description),
    ]


def _agent_axes(runs: dict[str, list[data.AgentRun]]) -> list[tuple[list[float], list[str]]]:
    """Each panel's ticks and their labels, from zero to the first tick at or past its largest run."""
    axes = []
    for k, step in enumerate(AGENT_STEPS):
        worst = max(v for cells in runs.values() for v in _agent_values(cells, k))
        marks, labels = linear_ticks(step * math.ceil(worst / step), step)
        axes.append((marks, [pct(m) for m in marks] if k == 1 else labels))
    return axes


def _agent_text(k: int, value: float) -> str:
    """A median as panel ``k`` prints it: seconds, a share of the steps, or a count."""
    return (f"{value:.0f} s", pct(value), f"{value:.0f}")[k]


def _agent_cell(
    s: Chart,
    k: int,
    scale: Scale,
    y: float,
    point: str,
    cells: Sequence[data.AgentRun],
    base: float,
) -> None:
    """Panel ``k``'s bar of a configuration's runs at ``y``: their median, a line over their range, and its value,
    in panel a with its share of ``base``, tuned vLLM's; or, without the rule the panel counts, what it has not, at
    the axis. A median of zero is its value at the axis and no bar."""
    values = _agent_values(cells, k)
    if not values:
        s.text(scale.start + VALUE_GAP, y + 6, AGENT_NONE[k - 1], size=NOTE, cls="soft", halo=True)
        return
    median = statistics.median(values)
    end = scale.start
    if max(values) > 0:
        end = _value_bar(s, scale, y, median, values, config_mark(point)[0], data.label(point)[0])
    box = s.text(
        end + VALUE_GAP, y + 7, _agent_text(k, median), size=VALUE, bold=_featured(point), halo=True
    )
    if k == 0:
        s.text(box.x1 + 8, y + 7, f"{median / base:.0%}", size=NOTE, cls="soft", halo=True)


def _name_line(
    s: Chart, x: float, baseline: float, name: str, note: str, bold: bool = False
) -> float:
    """A phone layout's row label, the featured row's name in semibold: the name at ``baseline``, and its note
    after it where both fit the canvas, else on a line of its own under the name. Returns how far below
    ``baseline`` the label's last line is: 0, or the note's line."""
    box = s.text(x, baseline, name, size=VALUE, bold=bold)
    if box.x1 + 8 + measure(note, NOTE) <= NARROW - 10:
        s.text(box.x1 + 8, baseline, note, size=NOTE, cls="soft")
        return 0.0
    s.text(x, baseline + 22, note, size=NOTE, cls="soft")
    return 22.0


def _agents_wide(runs: dict[str, list[data.AgentRun]], path: Path, description: str) -> Path:
    s = Chart(WIDTH, 440, "What the agents pay", description)
    axes = _agent_axes(runs)
    scales = [
        Scale(0, marks[-1], start, end)
        for (marks, _), (start, end) in zip(axes, AGENT_SPANS, strict=True)
    ]
    top, pitch = 52, 56
    ys = [top + pitch / 2 + pitch * k for k in range(len(runs))]
    bottom = ys[-1] + pitch / 2
    last = 0.0
    for (title, axis_title), scale, (marks, labels) in zip(AGENT_PANELS, scales, axes, strict=True):
        s.text(scale.start, 30, title, bold=True)
        last = x_axis(s, scale, marks, labels, top, bottom, title=axis_title)
    base = statistics.median(_agent_values(runs[data.TUNED], 0))
    for y, (point, cells) in zip(ys, runs.items(), strict=True):
        name, note = data.label(point)
        s.text(LABELS_END, y - 3, name, size=VALUE, anchor="end", bold=_featured(point))
        s.text(
            LABELS_END,
            y + 19,
            f"{note}, {_runs_of_count(len(cells))}",
            size=NOTE,
            anchor="end",
            cls="soft",
        )
        for k, scale in enumerate(scales):
            _agent_cell(s, k, scale, y, point, cells, base)
    s.height = last + 14
    return s.save(path)


def _agents_narrow(runs: dict[str, list[data.AgentRun]], path: Path, description: str) -> Path:
    """The phone layout: the three panels one above the other, each configuration's name over its bar; a
    configuration without a panel's rule is one line there, its name and what it has not."""
    s = Chart(NARROW, 2200, "What the agents pay", description)
    x0 = 18
    axes = _agent_axes(runs)
    base = statistics.median(_agent_values(runs[data.TUNED], 0))
    y, last = 30.0, 0.0
    for k, ((title, axis_title), (marks, labels)) in enumerate(
        zip(AGENT_PANELS, axes, strict=True)
    ):
        scale = Scale(0, marks[-1], x0, 300)
        s.text(x0, y, title, bold=True)
        y += 12
        for point, cells in runs.items():
            name, note = data.label(point)
            if not _agent_values(cells, k):
                _name_line(s, x0, y + 20, name, AGENT_NONE[k - 1])
                y += 32
                continue
            detail = f"{note}, {_runs_of_count(len(cells))}"
            band = y + 46 + _name_line(s, x0, y + 20, name, detail, _featured(point))
            band_grid(s, scale, marks, band - 14, band + 14)
            _agent_cell(s, k, scale, band, point, cells, base)
            y = band + 18
        last = x_axis(s, scale, marks, labels, None, y) + 25
        s.text(x0, last, axis_title, size=NOTE, cls="soft")
        y = last + 48
    s.height = last + 14
    return s.save(path)


# Every configuration ---------------------------------------------------------------------------------------------

FRONTIER_HEADINGS = (
    ("vLLM tuned by its own settings", ("tuned",)),
    ("The operator's static answers", ("operator",)),
    ("What engines ship", ("engine",)),
    ("Routers", ("router", "resubmit")),
    ("Timed by the conversation", ("lifecycle", "lifecycle-change")),
    ("The change", ("change",)),
)
"""The figure's headings, each over the families (data.FAMILIES) whose configurations it names."""
FRONTIER_NEAR = {False: 1000.0, True: 200.0}
"""How near the SLO the description counts a configuration's p95, in ms, as its figure's caption does: within a
second for Figure 11, and within 0.2 s for what an operator can deploy, every other configuration there 0.5 s or more
over it."""
SLO_NEAR = 1000.0
"""A p95 this near the SLO, in ms, has its distance from it written after its value: on an axis that also holds a
10 s bar, three bars that end within 0.6 s of the rule end at the same place to the eye."""
FRONTIER_PITCH = 52.0
"""From a row to the next in the wide layout: a little over two bars' thickness, so that a name and its note read as
one label beside their bar and stand clear of the next row's."""
GROUP_HEAD, GROUP_GAP = 30.0, 8.0
"""A group's heading's line over its rows, and the gap that parts a group from the heading of the next."""


def _frontier_name(p: data.FrontierPoint, points: Sequence[data.FrontierPoint]) -> str:
    """A configuration's name in prose: with its note after it where the note goes on from the name ("the lifecycle
    policy over make-room"), in parentheses where another configuration of ``points`` has the same name ("a static cap
    (2 agent requests in flight)"), and alone otherwise."""
    name, note = data.label(p.point)
    if note.split(" ", 1)[0] in ("over", "on"):
        return f"{data.in_prose(name)} {note}"
    if any(data.label(q.point)[0] == name for q in points if q.point != p.point):
        return f"{data.in_prose(name)} ({note})"
    return data.in_prose(name)


def _frontier_description(
    points: Sequence[data.FrontierPoint], slo: float, calls: bool = False
) -> str:
    """The figure in one sentence: which configurations come near the SLO, as its caption counts them
    (FRONTIER_NEAR), and every configuration's median p95 and agents' tokens a second."""
    over = "the calls" if calls else "the calls' first 240 s"
    near = FRONTIER_NEAR[calls]
    close = [p for p in points if not p.failed_turns and p.p95 - slo < near]
    if calls and min(p.p95 - slo for p in points if p not in close) < 500:
        raise ValueError(
            "a deployable configuration's p95 is within 0.5 s of the SLO: the caption says otherwise"
        )
    named = [_frontier_name(p, points) for p in close]
    names = ", ".join(named[:-1]) + f" and {named[-1]}" if len(named) > 1 else named[0]
    parts = []
    for p in points:
        p95 = (
            f"at least {REPLY_TIMEOUT_MS / 1000:.0f} s, {p.failed_turns} turns failed,"
            if p.failed_turns
            else f"{seconds(p.p95)} s"
        )
        if _near_slo(p, slo):
            p95 += f", {_from_slo(p.p95, slo)},"
        tokens = p.calls_tokens if calls else p.tokens
        parts.append(f"{_frontier_name(p, points)} {p95} at {tokens:,.0f} tokens/s")
    rest = ", every other 0.5 s or more over it" if calls else ""
    return (
        f"For each of {len(points)} configurations beside 24 agent sessions of the traces with the largest contexts, "
        f"at the median of its runs, its voice-to-voice p95 against the SLO of {seconds(slo)} s and the agents' "
        f"tokens a second over {over}, only {names} within {near / 1000:g} s of the SLO{rest}: "
        + "; ".join(parts[:-1])
        + f"; and {parts[-1]}."
    )


def frontier(out: Path) -> list[Path]:
    """report-frontier.svg and its phone layout: every configuration of W4's discriminating scenario but arm 9's
    ablations, a row under its family's heading, on the headline's axes. (a) Its V2V p95 as a bar from zero, the
    median of its runs with a line over their range, in its configuration's hue, against the SLO; a configuration
    whose turns failed is a bar cut at the scale's end. (b) The agents' tokens a second over the calls' first 240 s,
    the median of its runs."""
    points = data.frontier()
    slo = data.slo_ms(data.w4_rows())
    description = _frontier_description(points, slo)
    title = "Every configuration beside 24 agent sessions"
    return [
        _frontier_wide(points, slo, out / "report-frontier.svg", description, title),
        _frontier_narrow(points, slo, out / "report-frontier-narrow.svg", description, title),
    ]


def deployable(out: Path) -> list[Path]:
    """report-deployable.svg and its phone layout: Figure 11 for W4's configurations an operator could deploy
    (data.DEPLOYABLE), each with all its runs, its panel b the agents' tokens a second over the whole calls, as
    docs/BENCHMARKS.md's W4 table under it gives them. Seven rows of six families: the rows stand as the headline's
    do, without a heading over each."""
    points = [p for p in data.frontier() if p.point in data.DEPLOYABLE]
    if {p.point for p in points} != set(data.DEPLOYABLE):
        raise ValueError(
            f"W4's records lack {sorted(set(data.DEPLOYABLE) - {p.point for p in points})}"
        )
    slo = data.slo_ms(data.w4_rows())
    description = _frontier_description(points, slo, calls=True)
    title = "What an operator can deploy, beside 24 agent sessions"
    return [
        _frontier_wide(
            points, slo, out / "report-deployable.svg", description, title, calls=True, heads=False
        ),
        _frontier_narrow(
            points,
            slo,
            out / "report-deployable-narrow.svg",
            description,
            title,
            calls=True,
            heads=False,
        ),
    ]


def _near_slo(p: data.FrontierPoint, slo: float) -> bool:
    """Whether a configuration's p95 stands near enough to the ``slo`` (ms) for its distance to be written."""
    return not p.failed_turns and abs(p.p95 - slo) < SLO_NEAR


def _frontier_note(p: data.FrontierPoint) -> str:
    """A configuration's note, with its failed turns where it has any."""
    note = data.label(p.point)[1]
    return f"{note}, {p.failed_turns} turns failed" if p.failed_turns else note


def _frontier_rows(
    points: Sequence[data.FrontierPoint],
) -> list[tuple[str, list[data.FrontierPoint]]]:
    """The configurations under their headings, in the order of E2's report within each."""
    rows = []
    for heading, families in FRONTIER_HEADINGS:
        members = [p for p in points if p.family in families]
        if members:
            rows.append((heading, members))
    placed = {p.point for _, members in rows for p in members}
    if placed != {p.point for p in points}:
        raise ValueError(f"no heading over {sorted({p.point for p in points} - placed)}")
    return rows


def _frontier_top(points: Sequence[data.FrontierPoint]) -> int:
    """Panel a's axis's end, as the headline's (``_v2v_top``): the even second at or past every run of every
    configuration that answered all its turns. One whose turns failed is cut there."""
    worst = max(max(p.p95s) for p in points if not p.failed_turns)
    return 2000 * math.ceil(worst / 2000)


def _frontier_rates(p: data.FrontierPoint, calls: bool) -> tuple[float, Sequence[float]]:
    """A configuration's agents' tokens a second, the median of its runs and each run's: over the calls' first 240 s,
    or with ``calls`` over the whole calls."""
    return (p.calls_tokens, p.calls_tokens_per_run) if calls else (p.tokens, p.tokens_per_run)


def _rates_top(points: Sequence[data.FrontierPoint], calls: bool) -> int:
    """Panel b's axis's end, from zero as every tokens/s axis is: the first tick at or past every run."""
    worst = max(max(_frontier_rates(p, calls)[1]) for p in points)
    return 2000 * math.ceil(worst / 2000)


def _after_bar(s: Chart, end: float, y: float, text: str, bold: bool, slo_x: float) -> Box:
    """A p95's value after its bar's marks, which end at ``end``: past the SLO's rule where the value would cross
    it. Returns the value's box, for a note set after it."""
    x = end + VALUE_GAP
    if x - 5 <= slo_x <= x + measure(text, VALUE, bold=bold) + 5:
        x = slo_x + VALUE_GAP
    return s.text(x, y + 7, text, size=VALUE, bold=bold, halo=True)


def _cut_bar(s: Chart, scale: Scale, y: float, hue: str, name: str) -> None:
    """A bar at ``y`` that goes on past the scale's end: square-ended there, as the kernel cuts a bar, and parted
    just before its end by two slanted breaks of the page's color, the sign that it is not drawn to its length. Its
    value says how far it goes. The breaks reach a unit past the bar's edges, so that no line of it is left across
    them."""
    top, bottom = y - BAR / 2, y + BAR / 2
    s.hbar(scale.start, top, scale.end - scale.start, BAR, EXTENSIONS[hue][0], cut=True)
    cut_breaks(s, scale.end, top, bottom)
    s.marks.append(Box(scale.start, top, scale.end, bottom, name))


def _frontier_p95(s: Chart, a: Scale, y: float, p: data.FrontierPoint, slo: float) -> None:
    """Panel a's bar of a configuration at ``y``: its median p95, a line over its runs' range, and its value, with
    how far it stands from the ``slo`` (ms) after a value near it (``_near_slo``). A configuration whose turns
    failed is a bar cut at the scale's end (``_cut_bar``), and its value says how far it goes: at least the reply
    timeout."""
    hue, bold = config_mark(p.point)[0], _featured(p.point)
    name = data.label(p.point)[0]
    if not p.failed_turns:
        end = _value_bar(s, a, y, p.p95, p.p95s, hue, name)
        box = _after_bar(s, end, y, f"{seconds(p.p95)} s", bold, a(slo))
        if _near_slo(p, slo):
            _slo_note(s, box, y, p.p95, slo)
        return
    _cut_bar(s, a, y, hue, name)
    text = f"\N{GREATER-THAN OR EQUAL TO}{REPLY_TIMEOUT_MS / 1000:.0f} s"
    s.text(a.end + VALUE_GAP, y + 7, text, size=VALUE, bold=bold, halo=True)


def _frontier_rate(s: Chart, b: Scale, y: float, p: data.FrontierPoint, calls: bool) -> None:
    """Panel b's bar of a configuration at ``y``: its agents' median tokens a second, a line over its runs' range,
    and its value."""
    tokens, spread = _frontier_rates(p, calls)
    name = f"{data.label(p.point)[0]}'s tokens"
    end = _value_bar(s, b, y, tokens, spread, config_mark(p.point)[0], name)
    s.text(end + VALUE_GAP, y + 7, f"{tokens:,.0f}", size=VALUE, bold=_featured(p.point), halo=True)


FRONTIER_CALLS_TITLE = "Tokens/s over the whole calls"
"""Panel b's axis title in what an operator can deploy, over the window docs/BENCHMARKS.md's W4 table gives."""


def _frontier_wide(
    points: Sequence[data.FrontierPoint],
    slo: float,
    path: Path,
    description: str,
    title: str,
    *,
    calls: bool = False,
    heads: bool = True,
) -> Path:
    """Figure 11's wide layout on the headline's panels: the configurations' names in its column, panel a from where
    its plot starts, and panel b beside it, over the calls' first 240 s or with ``calls`` over the whole calls.
    Without ``heads`` the rows stand at the headline's pitch with no family's heading over them."""
    s = Chart(WIDTH, 2400, title, description)
    top_ms, top_rate = _frontier_top(points), _rates_top(points, calls)
    a = Scale(0, top_ms, PLOT_START, 728)
    b = Scale(0, top_rate, 818, 1040)
    s.text(a.start, 30, HEADLINE_PANELS[0], bold=True)
    s.text(b.start, 30, HEADLINE_PANELS[1], bold=True)
    top = 76  # the SLO's name over the plot, clear of the panels' titles, as in the headline
    y = float(top)
    placed = []
    pitch = FRONTIER_PITCH if heads else HEADLINE_PITCH
    for heading, members in _frontier_rows(points):
        if heads:
            s.text(24, y + 21, heading, size=NOTE, cls="soft")
            y += GROUP_HEAD
        for p in members:
            placed.append((y + pitch / 2, p))
            y += pitch
        y += GROUP_GAP if heads else 0.0
    bottom = y - GROUP_GAP + 2 if heads else y
    ticks = list(range(0, top_ms + 1, 2000))
    last = x_axis(s, a, ticks, [f"{t // 1000}" for t in ticks], top, bottom, title=V2V_TITLE)
    marks_b, labels_b = linear_ticks(top_rate, 2000)
    x_axis(
        s, b, marks_b, labels_b, top, bottom, title=FRONTIER_CALLS_TITLE if calls else TOKENS_TITLE
    )
    for row_y, p in placed:
        s.text(
            LABELS_END,
            row_y - 3,
            data.label(p.point)[0],
            size=VALUE,
            anchor="end",
            bold=_featured(p.point),
        )
        s.text(LABELS_END, row_y + 19, _frontier_note(p), size=NOTE, anchor="end", cls="soft")
        _frontier_p95(s, a, row_y, p, slo)
        _frontier_rate(s, b, row_y, p, calls)
    _v_threshold(s, a(slo), top, bottom, f"SLO {seconds(slo)} s")
    s.height = last + 14
    return s.save(path)


def _frontier_narrow(
    points: Sequence[data.FrontierPoint],
    slo: float,
    path: Path,
    description: str,
    title: str,
    *,
    calls: bool = False,
    heads: bool = True,
) -> Path:
    """The phone layout: each configuration's name over its p95's bar on panel a's scale, and its agents' tokens a
    second a number at the row's right end, in panel b's column. The two panels' titles stand on one line, panel b's
    short over its column with whose tokens and over which window under it, and the SLO's name on a line of its
    own under both."""
    s = Chart(NARROW, 3000, title, description)
    x0, right = 18, NARROW - 18
    top_ms = _frontier_top(points)
    a = Scale(0, top_ms, x0, 270)
    ticks = list(range(0, top_ms + 1, 6000))
    rate, _, window = (FRONTIER_CALLS_TITLE if calls else TOKENS_TITLE).partition(" ")
    s.text(x0, 30, HEADLINE_PANELS[0], bold=True)
    s.text(right, 30, f"(b) {rate}", anchor="end", bold=True)
    s.text(right, 54, f"the agents', {window}", size=NOTE, anchor="end", cls="soft")
    _threshold_key(s, x0, 78, f"SLO {seconds(slo)} s")
    y = 82.0
    for heading, members in _frontier_rows(points):
        if heads:
            s.text(x0, y + 24, heading, size=NOTE, cls="soft")
            y += 32
        for p in members:
            bold = _featured(p.point)
            name = data.label(p.point)[0]
            band = y + 46 + _name_line(s, x0, y + 20, name, _frontier_note(p), bold)
            band_grid(s, a, ticks, band - 14, band + 14)
            _frontier_p95(s, a, band, p, slo)
            _threshold_stub(s, a(slo), band)
            tokens = _frontier_rates(p, calls)[0]
            s.text(right, band + 7, f"{tokens:,.0f}", size=VALUE, anchor="end", bold=bold)
            y = band + 18
        y += GROUP_GAP if heads else 0.0
    end = y - GROUP_GAP if heads else y
    last = x_axis(s, a, ticks, [f"{t // 1000}" for t in ticks], None, end) + 25
    s.text(x0, last, V2V_TITLE, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


# What each part of the lifecycle policy over make-room does ----------------------------------------------------------

ABLATION_GROUPS = (
    ("The lifecycle policy over make-room", (("9", "With all its parts"),)),
    (
        "Its gateway policy",
        (
            ("9-no-throttle", "Without the critical-window throttle"),
            ("9-no-headroom", "Without the headroom"),
            ("9-trigger-request", "The headroom armed by the request"),
            ("9-budget0", "The throttle's budget at 0 ms"),
            ("9-budget500", "The throttle's budget at 500 ms"),
            ("9-no-events", "Without the lifecycle events"),
            ("9-no-degrade", "Without degrade"),
            ("9-window48", "Without degrade, with a 48-token window"),
            ("9-engine-only", "Without the gateway policy"),
        ),
    ),
    (
        "The engine under it",
        (
            ("9-no-kv-makeroom", "Without the KV preemption"),
            ("9-no-reserve", "Without the reserve"),
            ("9-no-budget-cap", "Without the budget cap"),
            ("9-gateway-only", "Without the change"),
        ),
    ),
)
"""Arm 9 and its ablations, each under the heading of the part it removes or changes, in docs/BENCHMARKS.md's order and
named as its arm-9 table names them: each its arm and its row's name."""
ABLATION_WHOLE = "9"
"""The arm with all its parts, the row the figure is drawn for: its bar is the saturated blue (FEATURED) and its name
and its value are set in semibold, while each row that removes or changes a part keeps its configuration's own hue."""
ABLATION_MARKS = {"9-engine-only": "change", "9-gateway-only": "lifecycle"}
"""The ablations marked as another configuration of labels.MARKS, by what they leave: the engine alone is the change's
(make-room at its 16-token window), and the gateway alone the lifecycle policy alone on tuned vLLM; every other row is
the lifecycle policy over make-room's."""
ABLATION_LABELS_END, ABLATION_PLOT_START = 396.0, 412.0
"""Where this figure's row names end and its plot starts in the wide layout: its rows are the policy's parts, named in
full as docs/BENCHMARKS.md's arm-9 table names them, and the longest does not fit the configurations' column
(LABELS_END)."""
ABLATION_PITCH = 42.0
"""From a row to the next in the wide layout: a bar and three quarters of one between two."""


def _ablation_mark(arm: str) -> tuple[str, str, bool]:
    return config_mark(ABLATION_MARKS.get(arm, "lifecycle-reserve"))


def _ablation_rows(
    found: dict[str, data.Ablation],
) -> list[tuple[str, list[tuple[data.Ablation, str]]]]:
    """The groups' rows with their records' points, refusing an ablation the records have and the table does not."""
    named = {arm for _, rows in ABLATION_GROUPS for arm, _ in rows}
    if named != set(found):
        raise ValueError(
            f"arm 9's ablations differ from the figure's: {sorted(named ^ set(found))}"
        )
    return [
        (heading, [(found[arm], name) for arm, name in rows]) for heading, rows in ABLATION_GROUPS
    ]


def _ablations_description(found: dict[str, data.Ablation], slo: float) -> str:
    full = found[ABLATION_WHOLE]
    groups = []
    for heading, rows in _ablation_rows(found)[1:]:
        cells = ", ".join(
            f"{name[0].lower() + name[1:]} {seconds(a.p95)} s ({seconds(min(a.p95s))} to {seconds(max(a.p95s))} s)"
            for a, name in rows
        )
        groups.append(f"in {heading[0].lower() + heading[1:]}, {cells}")
    return (
        f"The voice-to-voice p95 against the SLO of {seconds(slo)} s, beside 24 agent sessions of the traces with "
        "the largest contexts, of the lifecycle policy "
        f"over make-room with all its parts, {seconds(full.p95)} s at the median of {_runs_of_count(len(full.p95s))} "
        f"({seconds(min(full.p95s))} to {seconds(max(full.p95s))} s), and with one part removed or changed, the median "
        f"of three runs and their range: {'; '.join(groups)}."
    )


def ablations(out: Path) -> list[Path]:
    """report-ablations.svg and its phone layout, docs/BENCHMARKS.md's: what each part of the lifecycle policy over
    make-room (W4's arm 9) does, its V2V p95 with all its parts and with each removed or changed in turn, each a bar
    from zero, the median of the runs with a line over their range, against the SLO."""
    found = data.ablations()
    slo = data.slo_ms(data.w4_rows())
    description = _ablations_description(found, slo)
    return [
        _ablations_layout(found, slo, out / "report-ablations.svg", description, WIDTH),
        _ablations_layout(found, slo, out / "report-ablations-narrow.svg", description, NARROW),
    ]


def _ablation_bar(s: Chart, a: Scale, y: float, ablation: data.Ablation, slo_x: float) -> None:
    """A row's bar at ``y``: the median p95 of its runs, a line over their range, and its value."""
    whole = ablation.arm == ABLATION_WHOLE
    hue = FEATURED if whole else _ablation_mark(ablation.arm)[0]
    end = _value_bar(s, a, y, ablation.p95, ablation.p95s, hue, ablation.point)
    _after_bar(s, end, y, f"{seconds(ablation.p95)} s", whole, slo_x)


ABLATION_AXIS = "Voice-to-voice p95 (s)"
"""The figure's one axis, which names its measure: no panel's title does."""


def _ablations_layout(
    found: dict[str, data.Ablation], slo: float, path: Path, description: str, width: float
) -> Path:
    """Each row under its part's heading, the rows' names right of a column (wide) or over their bars (a phone)."""
    s = Chart(
        width, 1600, "What each part of the lifecycle policy over make-room does", description
    )
    groups = _ablation_rows(found)
    top_ms = 2000 * math.ceil(max(max(a.p95s) for a in found.values()) / 2000)
    slo_name = f"SLO {seconds(slo)} s"
    if width == WIDTH:
        a = Scale(0, top_ms, ABLATION_PLOT_START, 1010)
        ticks = list(range(0, top_ms + 1, 2000))
        top = 40
        y = float(top)
        placed = []
        for heading, rows in groups:
            s.text(24, y + 21, heading, size=NOTE, cls="soft")
            y += GROUP_HEAD
            for ablation, name in rows:
                placed.append((y + ABLATION_PITCH / 2, ablation, name))
                y += ABLATION_PITCH
            y += GROUP_GAP
        bottom = y - GROUP_GAP + 2
        last = x_axis(
            s, a, ticks, [f"{t // 1000}" for t in ticks], top, bottom, title=ABLATION_AXIS
        )
        for row_y, ablation, name in placed:
            s.text(
                ABLATION_LABELS_END,
                row_y + 7,
                name,
                size=VALUE,
                anchor="end",
                bold=ablation.arm == ABLATION_WHOLE,
            )
            _ablation_bar(s, a, row_y, ablation, a(slo))
        _v_threshold(s, a(slo), top, bottom, slo_name)
        s.height = last + 14
        return s.save(path)
    x0 = 18
    a = Scale(0, top_ms, x0, 322)
    ticks = list(range(0, top_ms + 1, 4000))
    _threshold_key(s, x0, 28, slo_name)
    y = 34.0
    for heading, rows in groups:
        s.text(x0, y + 24, heading, size=NOTE, cls="soft")
        y += 32
        for ablation, name in rows:
            s.text(x0, y + 20, name, size=VALUE, bold=ablation.arm == ABLATION_WHOLE)
            band = y + 46
            band_grid(s, a, ticks, band - 14, band + 14)
            _ablation_bar(s, a, band, ablation, a(slo))
            _threshold_stub(s, a(slo), band)
            y = band + 18
        y += GROUP_GAP
    last = x_axis(s, a, ticks, [f"{t // 1000}" for t in ticks], None, y - GROUP_GAP) + 25
    s.text(x0, last, ABLATION_AXIS, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


PAIR_KEYS = ("tuned", "deploy")
"""A panel of rounds' two configurations, by their keys in data.LABELS: tuned vLLM, the comparator, and the
configuration to deploy."""
PAIR = 26.0
"""Between a row's bars, centre to centre, where it holds a bar for each configuration: the bars all but touch, as a
grouped bar chart's do, so that the row reads as one."""
ROUND_GAP = 26.0
"""Between a row's last bar and the next row's first, edge to edge: a bar's thickness and more, which parts the
rows."""
CALLS_PITCH = 56.0
"""A row whose name has a note under it, as the headline's rows are."""
RowBar = tuple[float, Sequence[float], str, str]
"""A configuration's bar in a row: its value, its runs' values (the value alone where it is one run's), its hue, and
the value in words."""
Round = tuple[str, Sequence[RowBar]]
"""A round's row: its name, and each configuration's bar in it, from the comparator's to the featured
configuration's, the order they stand in from the top."""
ROUNDS_END = NARROW - 60.0
"""Where a phone layout's panel of rounds ends its scale: its longest bar's value stands after the bar."""
ON_FILL, ON_FILL_CAP = 2.0, 3.5
"""A range's line where it lies on the saturated bar, drawn again in the page's color: how thick, and half its cap's
height, which is ``Chart.span``'s."""


def _row_pitch(count: int) -> float:
    """From a row of ``count`` bars to the next."""
    return PAIR * (count - 1) + BAR + ROUND_GAP


def _spread_bar(
    s: Chart, scale: Scale, y: float, value: float, spread: Sequence[float], hue: str, name: str
) -> float:
    """``_value_bar``, and on the saturated bar, where the ink of its runs' range does not read, the part of the
    range that lies on the bar again in the page's color, cap and all, clear of the axis. Returns where the marks
    end."""
    end = _value_bar(s, scale, y, value, spread, hue, name)
    lo, hi, tip = scale(min(spread)), scale(max(spread)), scale(value)
    if hue == FEATURED and hi - lo > SPREAD_MIN and lo < tip:
        cap = max(lo - ON_FILL / 2, scale.start + ON_FILL / 2)
        s.rect(cap, y - ON_FILL_CAP, ON_FILL, 2 * ON_FILL_CAP, "page")
        s.rect(cap, y - ON_FILL / 2, min(hi, tip) - cap, ON_FILL, "page")
    return end


def _row_bars(
    s: Chart, scale: Scale, y: float, bars: Sequence[RowBar], rule: float | None = None
) -> None:
    """A row's bars, one under the other from ``y``, the first one's centre: each from the scale's start to its
    value in its configuration's fill, with its runs' range, and its value after it, the featured configuration's in
    semibold, past the threshold's rule at ``rule`` where the value would cross it."""
    for k, (value, spread, hue, text) in enumerate(bars):
        at = y + PAIR * k
        end = _spread_bar(s, scale, at, value, spread, hue, f"the bar of {text}")
        _after_bar(s, end, at, text, hue == FEATURED, -math.inf if rule is None else rule)


SHARE_MARKS = [0.0, 0.25, 0.5, 0.75, 1.0]
"""The ticks of an axis of a share, from none to all."""


def _share_ticks() -> tuple[list[float], list[str]]:
    """The ticks of an axis of a share from none to all, and their labels."""
    return SHARE_MARKS, [pct(m) for m in SHARE_MARKS]


def _rate_bar(
    s: Chart,
    scale: Scale,
    y: float,
    value: float,
    spread: Sequence[float],
    hue: str,
    bold: bool,
    note: str = "",
) -> None:
    """The agents' tokens a second at ``y``: a bar to the median of the runs with their range, its value, and after
    the value a note, what the row pays or gains against another."""
    end = _spread_bar(s, scale, y, value, spread, hue, "the agents' tokens")
    box = s.text(end + VALUE_GAP, y + 7, f"{value:,.0f}", size=VALUE, bold=bold, halo=True)
    if note:
        s.text(box.x1 + 10, y + 7, note, size=NOTE, halo=True)


def _round_key(s: Chart, x: float, baseline: float, entries: Sequence[tuple[str, str]]) -> None:
    """The key of a panel of rounds, on one line from ``x``: each configuration's fill, by its hue, and its name."""
    for hue, name in entries:
        x = s.swatch(x, baseline, EXTENSIONS[hue][0], name).x1 + 24


def _round_key_narrow(
    s: Chart, x: float, baseline: float, entries: Sequence[tuple[str, str]]
) -> float:
    """The key on a phone, an entry a line. Returns the last line's baseline."""
    for k, (hue, name) in enumerate(entries):
        s.swatch(x, baseline + 26 * k, EXTENSIONS[hue][0], name)
    return baseline + 26 * (len(entries) - 1)


def _slo_rule(slo: float) -> tuple[float, str]:
    """The SLO as a panel of rounds' threshold: where it stands, and its name."""
    return slo, f"SLO {seconds(slo)} s"


def _pair_entries() -> list[tuple[str, str]]:
    """The key's entries where a round's row holds tuned vLLM's bar and the configuration to deploy's, in the order
    the bars stand in: tuned vLLM's first."""
    return [(config_mark(key)[0], data.label(key)[0]) for key in PAIR_KEYS]


def _seconds_ticks(top: float) -> tuple[list[float], list[str]]:
    """The ticks of an axis of ms from zero to ``top``, one a second, labelled in seconds."""
    marks, labels = linear_ticks(top / 1000, 1)
    return [m * 1000 for m in marks], labels


def _p95_rounds(rounds: Sequence[tuple[int, float, float]]) -> list[Round]:
    """Rounds of tuned vLLM's p95 and the configuration to deploy's, in ms, as rows."""
    hues = [config_mark(key)[0] for key in PAIR_KEYS]
    return [
        (
            f"Round {k}",
            [(p95, (p95,), hue, f"{seconds(p95)} s") for p95, hue in zip(pair, hues, strict=True)],
        )
        for k, *pair in rounds
    ]


PAIR_V2V = "Voice-to-voice p95 (s)"
"""A panel of rounds of p95's axis, as Figures 8 and 11 name the measure."""


def _rounds_wide(
    s: Chart,
    scale: Scale,
    top: float,
    rounds: Sequence[Round],
    threshold: tuple[float, str],
    ticks: tuple[list[float], list[str]],
    title: str,
) -> float:
    """A panel of rounds in the wide layout, under ``top``: a row a round, its name in the configurations' column
    beside its bars, one a configuration, each with its value after it (``_row_bars``), over the axis. The
    threshold's name stands over the plot's top as ``_v_threshold`` sets it, but no further left than the plot
    starts, where the rule is near the axis. Returns the axis's last baseline."""
    count = len(rounds[0][1])
    below = PAIR * (count - 1)
    bottom = top + _row_pitch(count) * len(rounds)
    at = scale(threshold[0])
    last = x_axis(s, scale, *ticks, top, bottom, title=title)
    for k, (name, bars) in enumerate(rounds):
        y = top + (ROUND_GAP + BAR) / 2 + _row_pitch(count) * k
        s.text(LABELS_END, y + below / 2 + 7, name, size=VALUE, anchor="end")
        _row_bars(s, scale, y, bars, at)
    s.vrule(at, top, bottom, "slo")
    half = measure(threshold[1], NOTE) / 2
    s.text(
        max(at, scale.start + half),
        top - 8,
        threshold[1],
        size=NOTE,
        anchor="middle",
        cls="red tint",
        halo=True,
    )
    return last


def _rounds_narrow(
    s: Chart,
    scale: Scale,
    y: float,
    rounds: Sequence[Round],
    threshold: tuple[float, str],
    ticks: tuple[list[float], list[str]],
    title: str,
) -> float:
    """A panel of rounds in the phone layout, from ``y``, where the threshold's name stands: each round's name, and
    under it its bars, one a configuration, each with its value after it. Returns the axis's title's last
    baseline."""
    x0, at = scale.start, scale(threshold[0])
    below = PAIR * (len(rounds[0][1]) - 1)
    _threshold_key(s, x0, y - 8, threshold[1])
    for name, bars in rounds:
        s.text(x0, y + 20, name, size=VALUE)
        band = y + 46
        band_grid(s, scale, ticks[0], band - 14, band + below + 14)
        _row_bars(s, scale, band, bars, at)
        _threshold_stub(s, at, band, below)
        y = band + below + 18
    last = x_axis(s, scale, *ticks, None, y)
    for line in wrap(title, NARROW - 2 * x0, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    return last


# Where the pool has room, and as calls overlap --------------------------------------------------------------------

MARGIN_TITLE = "Where the pool has room, and as calls overlap"


def _margin_description(
    rounds: list[tuple[int, float, float]], calls: list[data.Calls], slo: float
) -> str:
    paired = ", ".join(f"{seconds(t)} and {seconds(c)} s" for _, t, c in rounds)
    counts = "; ".join(
        f"{c.callers} callers, {statistics.median(c.over):+,.0f} ms over their SLO of {seconds(c.slo)} s at the "
        f"median of {len(c.over)} runs ({min(c.over):+,.0f} to {max(c.over):+,.0f}), agent prefill held in "
        f"{statistics.median(c.held):.0%} of the steps, the agents at {statistics.median(c.tokens):,.0f} tokens/s"
        for c in calls
    )
    deploy = data.in_prose(data.label("deploy")[0])
    return (
        "In panel a, with 16 agent sessions of the whole corpus, where the pool has room, tuned vLLM's V2V p95 and "
        f"{deploy}'s in each of five rounds against the SLO of {seconds(slo)} s ({paired}); in panels b to d, "
        f"{deploy} with two, four and six callers beside 24 agent sessions, on the same three samples of the agents, "
        "the median of the runs' p95 over that many callers' SLO, of the share of the engine's steps in which the "
        "budget cap held agent prefill, and of the agents' tokens a second over the calls' first 240 s, each with its "
        f"runs' range ({counts})."
    )


MARGIN_TOP = 4000
"""Figure 12a's axis of p95 in ms, from zero, as every latency scale is, past the slowest round's p95."""
MARGIN_A = "(a) 16 sessions of the whole corpus"
"""Panel a's title: what its rounds ran beside; the caption says that the pool has room there."""
MARGIN_BESIDE = "Beside 24 agent sessions"
"""What panels b to d's rows of callers ran beside."""
MARGIN_PANELS = (
    ("(b) Voice p95 over the SLO", "Time over the SLO (ms)", "over", 400, 200, "{:+,.0f} ms"),
    ("(c) Steps held", "Share of the engine's steps", "held", 1, 0.5, "{:.0%}"),
    ("(d) Agents' throughput", TOKENS_TITLE, "tokens", 4000, 2000, "{:,.0f}"),
)
"""Panels b to d: each one's title, its axis's title, the field of data.Calls it draws, its scale's top and its ticks'
step, and its values' format. Panel c's steps held are the budget cap's, as Figure 10's are, a share of the engine's
steps."""
MARGIN_SPANS = ((PLOT_START, 480.0), (600.0, 780.0), (880.0, 1040.0))
"""Panels b to d's plots in the wide layout, each with room after it for its longest bar's value."""
MARGIN_END = 1096.0
"""Where the three end in the wide layout, the last one's value with them: the figure's right margin."""


def margin(out: Path) -> list[Path]:
    """report-margin.svg: where the pool has room, and as calls overlap. (a) Where the pool has room: the standard
    scenario's rounds, a row each: tuned vLLM's p95 and under it the configuration to deploy's, a bar each with its
    value.
    (b) to (d) As calls overlap: with 2, 4 and 6 callers on the same three samples of the agents, a row each, the
    configuration to deploy's p95 over that many callers' SLO, the share of steps the budget cap held agent prefill
    in, and the agents' tokens a second, each a bar to the median of its runs with their range."""
    rows = data.w5_rows()
    rounds = data.standard_rounds(rows)
    calls = data.calls(rows)
    slo = data.slo_ms(rows)
    description = _margin_description(rounds, calls, slo)
    return [
        _margin_wide(rounds, calls, slo, out / "report-margin.svg", description),
        _margin_narrow(rounds, calls, slo, out / "report-margin-narrow.svg", description),
    ]


def _margin_axis(top: float, step: float, field: str) -> tuple[list[float], list[str]]:
    """A panel of b to d's ticks and their labels: a share as a percentage."""
    marks, labels = linear_ticks(top, step)
    return marks, [pct(m) for m in marks] if field == "held" else labels


def _margin_cell(s: Chart, scale: Scale, y: float, c: data.Calls, field: str, text: str) -> None:
    """A caller count's bar in one of panels b to d: to the median of its runs, with their range, and its value,
    in semibold as the configuration to deploy's are in panel a. In panel b the bar starts at the count's own SLO,
    the rule its p95 is over."""
    values = list(getattr(c, field))
    middle = statistics.median(values)
    end = _spread_bar(s, scale, y, middle, values, FEATURED, f"{c.callers} callers' {field}")
    s.text(end + VALUE_GAP, y + 7, text.format(middle), size=VALUE, bold=True, halo=True)


def _calls_label(c: data.Calls, slo: bool = True) -> tuple[str, str]:
    """A caller count's row: its name, and its note, its runs, after its own SLO where the row is set against it."""
    counted = _runs_of_count(len(c.over))
    return f"{c.callers} callers", f"SLO {seconds(c.slo)} s, {counted}" if slo else counted


def _margin_wide(rounds, calls, slo: float, path: Path, description: str) -> Path:
    s = Chart(WIDTH, 900, MARGIN_TITLE, description)
    # (a) The standard scenario, a row a round.
    a = Scale(0, MARGIN_TOP, PLOT_START, 840)
    s.text(a.start, 30, MARGIN_A, bold=True)
    _round_key(s, a.start, 60, _pair_entries())
    last = _rounds_wide(
        s, a, 100.0, _p95_rounds(rounds), _slo_rule(slo), _seconds_ticks(MARGIN_TOP), PAIR_V2V
    )
    # (b) to (d) As calls overlap, a row a caller count: what the three ran beside names them, over a bracket.
    head = last + 96
    top = head + 20
    ys = [top + CALLS_PITCH / 2 + CALLS_PITCH * k for k in range(len(calls))]
    bottom = ys[-1] + CALLS_PITCH / 2
    # A Chart's check reads each rule's place and kind, which Figure.bracket does not record for its own.
    s.painted.append(len(s.items))
    s.kinds.append("bracket")
    s.bracket(MARGIN_SPANS[0][0], MARGIN_END, head - 30, MARGIN_BESIDE, below=False)
    for y, c in zip(ys, calls, strict=True):
        name, note = _calls_label(c)
        s.text(LABELS_END, y - 3, name, size=VALUE, anchor="end")
        s.text(LABELS_END, y + 19, note, size=NOTE, anchor="end", cls="soft")
    for (title, axis, field, hi, step, text), span in zip(MARGIN_PANELS, MARGIN_SPANS, strict=True):
        scale = Scale(0, hi, *span)
        s.text(scale.start, head, title, bold=True)
        marks, labels = _margin_axis(hi, step, field)
        if field == "over":
            # Panel b's zero is the SLO: its dashed rule stands for the axis there, named where the zero would be.
            last = max(last, x_axis(s, scale, marks[1:], labels[1:], top, bottom, title=axis))
            s.vrule(scale(0), top, bottom, "slo")
            s.text(scale(0), bottom + 23, "SLO", size=NOTE, anchor="middle", cls="red tint")
        else:
            last = max(last, x_axis(s, scale, marks, labels, top, bottom, title=axis))
        for y, c in zip(ys, calls, strict=True):
            _margin_cell(s, scale, y, c, field, text)
    s.height = last + 14
    return s.save(path)


def _margin_narrow(rounds, calls, slo: float, path: Path, description: str) -> Path:
    """The phone layout: panel a's rounds, each one's name over its two bars, then b to d one above the other, each
    caller count's name over its bar."""
    s = Chart(NARROW, 2400, MARGIN_TITLE, description)
    x0 = 18
    a = Scale(0, MARGIN_TOP, x0, ROUNDS_END)
    s.text(x0, 30, MARGIN_A, bold=True)
    y = _round_key_narrow(s, x0, 60, _pair_entries()) + 44
    last = _rounds_narrow(
        s, a, y, _p95_rounds(rounds), _slo_rule(slo), _seconds_ticks(MARGIN_TOP), PAIR_V2V
    )
    y = last + 48
    s.text(x0, y, MARGIN_BESIDE, size=NOTE, cls="soft")
    y += 34
    for title, axis, field, hi, step, text in MARGIN_PANELS:
        scale = Scale(0, hi, x0, 300)
        marks, labels = _margin_axis(hi, step, field)
        s.text(x0, y, title, bold=True)
        y += 12
        if field == "over":
            y += 30
            _threshold_key(s, x0, y - 8, "SLO")
        for c in calls:
            band = y + 46 + _name_line(s, x0, y + 20, *_calls_label(c, field == "over"))
            band_grid(s, scale, marks, band - 14, band + 14)
            if field == "over":
                _threshold_stub(s, scale(0), band)
            _margin_cell(s, scale, band, c, field, text)
            y = band + 18
        last = x_axis(s, scale, marks, labels, None, y)
        for line in wrap(axis, NARROW - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
        y = last + 48
    s.height = last + 14
    return s.save(path)


# The A100 ---------------------------------------------------------------------------------------------------------

A100_TITLE = "The rules on an A100"
A100_CAP = 512
"""The step cap panel a's rounds ran at, the RTX 5060 Ti's: the configuration to deploy at it is the one the figure is
drawn for, and the other step cap's values are set against its."""
A100_TOP = 6000
"""Figure 13's axes of p95 in ms, from zero, past the slowest round's p95."""
A100_TOKENS = 16000
"""Panel b's axis of the agents' tokens a second, past the fastest run's."""
A100_A = f"(a) Each round, at {A100_CAP}-token steps"
A100_B = "(b) Both step caps: the voice tail and the agents' throughput"
A100_B_LINES = ("(b) Both step caps: the voice tail", "and the agents' throughput")
"""Panel b's title on a phone, broken between its two measures: the even break would part "the voice" from "tail"."""
A100_MEASURES = ("Voice-to-voice p95", "Agents' tokens/s")
"""Panel b's two measures, as the headline's panels name them: on a phone each one's rows stand under its name."""
A100_SPANS = ((PLOT_START, 650.0), (740.0, 900.0))
"""Panel b's two plots in the wide layout, each with room after it for its longest bar's value and the trade."""


def _a100_rounds(
    pairs: dict[int, tuple[data.Configuration, data.Configuration]],
) -> list[tuple[int, float, float]]:
    """Panel a's rounds: each one's number, tuned vLLM's p95 in it and the configuration to deploy's."""
    tuned, change = pairs[A100_CAP]
    rounds = []
    for t, c in zip(tuned.runs, change.runs, strict=True):
        if t.round != c.round:
            raise ValueError(
                f"round {t.round} of tuned vLLM pairs with round {c.round} of the configuration to deploy"
            )
        rounds.append((t.round, t.p95, c.p95))
    return rounds


def _a100_trade(
    pairs: dict[int, tuple[data.Configuration, data.Configuration]],
) -> dict[int, tuple[str, str]]:
    """By each step cap but panel a's, what the configuration to deploy pays and gains there against itself at panel
    a's, in words: its p95's change in seconds, and its agents' tokens a second's as a share."""
    base = pairs[A100_CAP][1]
    assert base.tokens is not None
    trade = {}
    for cap, (_, c) in pairs.items():
        if cap == A100_CAP:
            continue
        assert c.tokens is not None
        more = c.p95 - base.p95
        sign = "+" if more >= 0 else "\N{MINUS SIGN}"
        share = f"{c.tokens / base.tokens - 1:+.0%}".replace("-", "\N{MINUS SIGN}")
        trade[cap] = (f"{sign}{seconds(abs(more))} s", share)
    return trade


def _a100_description(
    pairs: dict[int, tuple[data.Configuration, data.Configuration]], slo: float
) -> str:
    by_round = ", ".join(f"{seconds(t)} and {seconds(c)} s" for _, t, c in _a100_rounds(pairs))
    deploy = data.in_prose(data.label("deploy")[0])
    caps = "; ".join(
        f"at {cap:,}-token steps tuned vLLM {seconds(t.p95)} s at {t.tokens:,.0f} tokens/s, {deploy} "
        f"{seconds(c.p95)} s at {c.tokens:,.0f} tokens/s, medians of {len(t.runs)} runs each"
        for cap, (t, c) in pairs.items()
    )
    trade = "; ".join(
        f"from {A100_CAP:,}- to {cap:,}-token steps {deploy}'s p95 changes by {p95} and its agents' tokens a "
        f"second by {tokens}"
        for cap, (p95, tokens) in _a100_trade(pairs).items()
    )
    return (
        f"On the A100 beside 24 agent sessions: in panel a, at the RTX 5060 Ti's {A100_CAP}-token steps, tuned "
        f"vLLM's V2V p95 and {deploy}'s in each of five rounds against the card's SLO of {seconds(slo)} s "
        f"({by_round}); in panel b, at each step cap, each one's V2V p95 and its agents' tokens a second over the "
        f"calls' first 240 s, the median of its runs with their range ({caps}; {trade})."
    )


def a100(out: Path) -> list[Path]:
    """report-a100.svg: the A100. (a) At 512-token steps, a row a round: tuned vLLM's p95 and under it the
    configuration to deploy's, a bar each with its value, against the card's SLO. (b) The trade at both step caps, a row a
    configuration under its step cap: its p95 and its agents' tokens a second, each a bar to the median of its runs
    with their range, and beside the configuration to deploy's at the other step cap what it pays and gains against
    itself at panel a's."""
    rows = data.a100_rows()
    pairs = data.a100(rows)
    slo = data.slo_ms(rows)
    description = _a100_description(pairs, slo)
    return [
        _a100_wide(pairs, slo, out / "report-a100.svg", description),
        _a100_narrow(pairs, slo, out / "report-a100-narrow.svg", description),
    ]


def _a100_groups(
    pairs: dict[int, tuple[data.Configuration, data.Configuration]],
) -> list[tuple[str, list[tuple[data.Configuration, str, str, str, bool, tuple[str, str]]]]]:
    """Panel b's rows under their step caps' headings: each configuration with its name and its runs, its hue,
    whether it is the featured one, and the trade to say beside its two values. The configuration to deploy is
    the saturated blue at panel a's step cap and a softer blue at the other."""
    trade = _a100_trade(pairs)
    groups = []
    for cap, configurations in pairs.items():
        members = []
        for key, c in zip(PAIR_KEYS, configurations, strict=True):
            hue = config_mark(key)[0]
            change = hue == FEATURED
            featured = change and cap == A100_CAP
            notes = trade.get(cap, ("", "")) if change else ("", "")
            members.append(
                (
                    c,
                    data.label(key)[0],
                    _runs_of_count(len(c.runs)),
                    hue if featured or not change else "sky",
                    featured,
                    tuple(f"{note} from {A100_CAP:,}" if note else "" for note in notes),
                )
            )
        groups.append((f"{cap:,}-token steps", members))
    return groups


def _a100_p95(
    s: Chart,
    scale: Scale,
    y: float,
    c: data.Configuration,
    hue: str,
    bold: bool,
    note: str,
    slo_x: float,
) -> None:
    """A configuration's p95 at ``y``: its bar to the median of its runs with their range, its value after it, or
    past the SLO's rule where the value would cross it (as ``_after_bar`` sets it), and after the value the trade."""
    end = _spread_bar(s, scale, y, c.p95, [r.p95 for r in c.runs], hue, "a p95")
    text = f"{seconds(c.p95)} s"
    x = end + VALUE_GAP
    if x - 5 <= slo_x <= x + measure(text, VALUE, bold=bold) + 5:
        x = slo_x + VALUE_GAP
    box = s.text(x, y + 7, text, size=VALUE, bold=bold, halo=True)
    if note:
        s.text(box.x1 + 10, y + 7, note, size=NOTE, halo=True)


def _a100_rate(
    s: Chart, scale: Scale, y: float, c: data.Configuration, hue: str, bold: bool, note: str
) -> None:
    """A configuration's agents' tokens a second at ``y``: its bar to the median of its runs with their range, its
    value, and after the value the trade."""
    assert c.tokens is not None
    spread = [r.tokens for r in c.runs if r.tokens is not None]
    _rate_bar(s, scale, y, c.tokens, spread, hue, bold, note)


def _a100_wide(pairs, slo: float, path: Path, description: str) -> Path:
    s = Chart(WIDTH, 1200, A100_TITLE, description)
    ticks = _seconds_ticks(A100_TOP)
    # (a) The rounds at panel a's step cap, a row a round.
    a = Scale(0, A100_TOP, PLOT_START, 840)
    s.text(a.start, 30, A100_A, bold=True)
    _round_key(s, a.start, 60, _pair_entries())
    last = _rounds_wide(
        s, a, 100.0, _p95_rounds(_a100_rounds(pairs)), _slo_rule(slo), ticks, PAIR_V2V
    )
    # (b) Both step caps, a row a configuration: its p95, and its agents' tokens a second.
    head = last + 54
    top = head + 46
    p95, rate = Scale(0, A100_TOP, *A100_SPANS[0]), Scale(0, A100_TOKENS, *A100_SPANS[1])
    s.text(p95.start, head, A100_B, bold=True)
    y = top
    placed = []
    for heading, members in _a100_groups(pairs):
        s.text(24, y + 21, heading, size=NOTE, cls="soft")
        y += GROUP_HEAD
        for member in members:
            placed.append((y + CALLS_PITCH / 2, member))
            y += CALLS_PITCH
        y += GROUP_GAP
    bottom = y - GROUP_GAP + 2
    last = x_axis(s, p95, *ticks, top, bottom, title=PAIR_V2V)
    x_axis(s, rate, *linear_ticks(A100_TOKENS, A100_TOKENS / 2), top, bottom, title=TOKENS_TITLE)
    for y, (c, name, runs_of, hue, bold, notes) in placed:
        s.text(LABELS_END, y - 3, name, size=VALUE, anchor="end", bold=bold)
        s.text(LABELS_END, y + 19, runs_of, size=NOTE, anchor="end", cls="soft")
        _a100_p95(s, p95, y, c, hue, bold, notes[0], p95(slo))
        _a100_rate(s, rate, y, c, hue, bold, notes[1])
    _v_threshold(s, p95(slo), top, bottom, f"SLO {seconds(slo)} s")
    s.height = last + 14
    return s.save(path)


def _a100_narrow(pairs, slo: float, path: Path, description: str) -> Path:
    """The phone layout: panel a's rounds, each one's name over its two bars, then panel b's two measures one above
    the other, each configuration's name over its bar."""
    s = Chart(NARROW, 2600, A100_TITLE, description)
    x0 = 18
    ticks = _seconds_ticks(A100_TOP)
    a = Scale(0, A100_TOP, x0, ROUNDS_END)
    s.text(x0, 30, A100_A, bold=True)
    y = _round_key_narrow(s, x0, 60, _pair_entries()) + 44
    last = _rounds_narrow(
        s, a, y, _p95_rounds(_a100_rounds(pairs)), _slo_rule(slo), ticks, PAIR_V2V
    )
    y = last + 48
    for line in A100_B_LINES:
        s.text(x0, y, line, bold=True)
        y += 26
    y += 8
    groups = _a100_groups(pairs)
    measures = (
        (Scale(0, A100_TOP, x0, 300), ticks, V2V_TITLE),
        (Scale(0, A100_TOKENS, x0, 210), linear_ticks(A100_TOKENS, A100_TOKENS / 2), TOKENS_TITLE),
    )
    for k, (measure_name, (scale, marks, axis)) in enumerate(
        zip(A100_MEASURES, measures, strict=True)
    ):
        s.text(x0, y, measure_name)
        y += 8
        if not k:
            y += 30
            _threshold_key(s, x0, y - 8, f"SLO {seconds(slo)} s")
        for heading, members in groups:
            s.text(x0, y + 24, heading, size=NOTE, cls="soft")
            y += 32
            for c, name, runs_of, hue, bold, notes in members:
                band = y + 46 + _name_line(s, x0, y + 20, name, runs_of, bold)
                band_grid(s, scale, marks[0], band - 14, band + 14)
                if k:
                    _a100_rate(s, scale, band, c, hue, bold, notes[1])
                else:
                    _a100_p95(s, scale, band, c, hue, bold, notes[0], scale(slo))
                    _threshold_stub(s, scale(slo), band)
                y = band + 18
            y += GROUP_GAP
        last = x_axis(s, scale, *marks, None, y - GROUP_GAP) + 25
        s.text(x0, last, axis, size=NOTE, cls="soft")
        y = last + 44
    s.height = last + 14
    return s.save(path)


# Two cards --------------------------------------------------------------------------------------------------------

CARDS_TITLE = "What sharing two cards buys"
CARD_KEYS = {"stock": "tuned", "change": "deploy"}
"""Each engine's configuration, by its key in data.LABELS: the rounds' stock engine runs the frozen box's flags, tuned
vLLM, and the change runs the configuration to deploy."""
CARD_ENGINES = {
    engine: (data.label(key)[0], config_mark(key)[0]) for engine, key in CARD_KEYS.items()
}
"""Each engine's name and hue, as data.LABELS and labels.MARKS give its configuration's."""
CARD_KINDS = {
    "dedicated": "Dedicated roles",
    "consolidation": "Consolidation",
    "pooling": "Pooling",
}
"""Each kind of arrangement's heading over its rows, in the figure's order."""
CARD_TOKENS = Scale(0, 15000, PLOT_START, 840)
"""Panel b's agents' tokens a second, past the most any round did (14,352)."""
CARDS_A = "(a) The consolidated card's replies over the SLO, by its agent sessions"
CARDS_B = "(b) Agents' tokens/s over both cards"
"""Panel b's title: an arrangement's agents' tokens a second, summed over its two cards; the caption says that a bar is
the median of the rounds, and that panel a pools the consolidated card's rounds."""
CARDS_BEST = "the best dedicated"
"""The best qualifying dedicated arrangement, as the hurdle's name and the arrangements' notes call it."""
CARDS_GAIN = f"+%: over {CARDS_BEST}"
"""What a bar's signed share is of, said once in the phone layout's key, whose rows have no room to say it."""


def _cards_and(items: Sequence[str]) -> str:
    """Items in prose: "a, b and c"."""
    return ", ".join(items[:-1]) + f" and {items[-1]}" if len(items) > 1 else items[0]


def _card_sessions(a: data.Arrangement) -> str:
    return f"{a.sessions[0]} + {a.sessions[1]} sessions"


def _card_row(a: data.Arrangement) -> str:
    """An arrangement's name under its kind's heading."""
    if a.kind == "dedicated":
        return a.key.removeprefix("Dedicated roles, ")
    return "+ a warm spare" if a.spare else CARD_ENGINES[a.engine][0]


def _card_gain(c: data.TwoCards, a: data.Arrangement) -> str:
    """How much more background work than the best dedicated arrangement a qualifying shared one does, the measure
    the hurdle is set in, as a signed share; nothing for a dedicated arrangement or one that misses the
    requirement, which the gate does not weigh."""
    if a.kind == "dedicated" or not a.qualifies:
        return ""
    return f"{a.tokens / c.best.tokens - 1:+.1%}".replace("-", "\N{MINUS SIGN}")


def _cards_featured(c: data.TwoCards) -> data.Arrangement:
    """The arrangement the figure is drawn for: of the change's that qualify, the one that does the most background
    work."""
    return max(
        (a for a in c.arrangements if a.qualifies and a.engine == "change"), key=lambda a: a.tokens
    )


def _cards_description(c: data.TwoCards, sessions: Sequence[int]) -> str:
    shares = " and ".join(
        _cards_and([f"{cell.share:.1%}" for cell in cells])
        + f" on {data.in_prose(CARD_ENGINES[engine][0])}"
        for engine, cells in c.consolidated.items()
    )
    rows = []
    for a in c.arrangements:
        if a.kind == "dedicated":
            what = f"dedicated roles, {_card_row(a)}"
        elif a.spare:
            what = f"consolidation on {data.in_prose(CARD_ENGINES['change'][0])} with a warm spare"
        else:
            what = f"{a.kind} on {data.in_prose(CARD_ENGINES[a.engine][0])}"
        gain = _card_gain(c, a)
        more = f", {gain} over {CARDS_BEST}" if gain else ""
        miss = "" if a.qualifies else ", missing the requirement"
        rows.append(f"{what}, {_card_sessions(a)}: {a.tokens:,.0f}{more}{miss}")
    rounds = len(c.rounds)
    return (
        f"In panel a, the share of the consolidated card's replies over their round's SLO, the {rounds} rounds "
        "pooled, by its agent sessions beside both calls, against the requirement of at most "
        f"{c.requirement:.0%}: at {_cards_and([str(n) for n in sessions])} sessions, {shares}; in panel b, each "
        "arrangement's agents' tokens a second over the calls' first 240 s summed over both cards, at the median of "
        f"the {rounds} rounds with their range, against a floor of at least {c.threshold:,.0f}, {c.hurdle:.0%} over "
        "the best dedicated arrangement: " + "; ".join(rows) + "."
    )


def cards(out: Path) -> list[Path]:
    """report-cards.svg: two cards. (a) The consolidated card's replies over their round's SLO, the rounds pooled, a
    row a number of agent sessions beside both calls: tuned vLLM's share and under it the configuration to deploy's,
    a bar each with its value, against the requirement. (b) Each arrangement's agents' tokens a second over the
    calls' first 240 s summed over both cards, under its kind's heading: a bar to the median of the rounds with their
    range, hatched where the arrangement misses the requirement, against the hurdle over the best dedicated
    arrangement, and beside each qualifying shared one how far over that arrangement it is."""
    c = data.two_cards()
    sessions = [cell.sessions for cell in c.consolidated["stock"]]
    if any([cell.sessions for cell in cells] != sessions for cells in c.consolidated.values()):
        raise ValueError("the engines' consolidated cards ran different numbers of sessions")
    description = _cards_description(c, sessions)
    return [
        _cards_wide(c, out / "report-cards.svg", description),
        _cards_narrow(c, out / "report-cards-narrow.svg", description),
    ]


def _card_rounds(c: data.TwoCards) -> list[Round]:
    """Panel a's rows, one a number of sessions: each engine's share of the consolidated card's replies over the SLO
    at it, tuned vLLM's then the configuration to deploy's, as a panel of rounds draws a row."""
    engines = [(c.consolidated[engine], CARD_ENGINES[engine][1]) for engine in CARD_KEYS]
    rows = []
    for k, cell in enumerate(engines[0][0]):
        shares = [(cells[k].share, hue) for cells, hue in engines]
        bars = [(share, (share,), hue, f"{share:.1%}") for share, hue in shares]
        rows.append((f"{cell.sessions} sessions", bars))
    return rows


def _card_rule(c: data.TwoCards) -> tuple[float, str]:
    """The requirement as panel a's threshold: where it stands, and its name."""
    return c.requirement, f"at most {c.requirement:.0%}"


def _cards_share_title(c: data.TwoCards) -> str:
    return f"Share of the card's replies over the SLO, the {len(c.rounds)} rounds pooled"


def _cards_hurdle(c: data.TwoCards) -> str:
    """The hurdle's name, its direction in its words as "at most" is in the requirement's."""
    return f"at least {c.threshold:,.0f}: {c.hurdle:.0%} over {CARDS_BEST}"


FLOOR = "soft-ink trace dashed"
"""A floor's rule, a level an arrangement must pass and not a limit to stay under: the kernel's dashed reference line
in the soft ink, so that a red dashed rule stays a limit everywhere."""


def _floor_rule(s: Chart, x: float, top: float, bottom: float, text: str) -> Box:
    """A floor across a plot from ``top`` to ``bottom`` at ``x``, as ``_v_threshold`` draws a limit: its dashed rule,
    named once over the plot's top, centred on it, in the soft ink."""
    s.ref((x, top), (x, bottom), FLOOR)
    return s.text(x, top - 8, text, size=NOTE, anchor="middle", cls="soft", halo=True)


def _floor_stub(s: Chart, x: float, band: float) -> None:
    """A floor in a phone layout's row, as ``_threshold_stub`` draws a limit's: through the row's band alone."""
    s.ref((x, band - STUB), (x, band + STUB), FLOOR)


def _floor_key(s: Chart, x: float, baseline: float, text: str) -> Box:
    """A phone layout's floor named once, as ``_threshold_key`` names a limit: a sample of its stub, then its name."""
    s.path(f"M{fmt(x + 9)} {fmt(baseline - 15)}V{fmt(baseline + 4)}", FLOOR)
    return s.text(x + 26, baseline, text, size=NOTE, cls="soft")


def _cards_key(s: Chart, c: data.TwoCards, x: float, baseline: float) -> None:
    """Panel b's key under its title: a hatched bar is an arrangement that misses the requirement."""
    entry = [(hatch_key(), f"Misses the {c.requirement:.0%} requirement")]
    legend(s, x, baseline, entry, width=legend_width(entry))


def _cards_bar(
    s: Chart,
    c: data.TwoCards,
    a: data.Arrangement,
    b: Scale,
    y: float,
    featured: data.Arrangement,
    words: bool,
) -> None:
    """Panel b's bar of an arrangement at ``y``: to the median of its rounds with their range, as ``_value_bar``
    draws one, but hatched where the arrangement misses the requirement; its value after it, or past the hurdle's
    rule where the value would cross it; and after the value, of a qualifying shared arrangement, how far over the
    best dedicated one it is. Tuned vLLM's are gray; of the change's, the featured one is the saturated blue and in
    semibold, one that misses the requirement is hatched in that blue, and the others are a softer blue. With
    ``words`` the featured arrangement's note says what it is over, and the best dedicated arrangement is named."""
    hue = CARD_ENGINES[a.engine][1]
    bold = a is featured
    if a.qualifies:
        if hue == FEATURED and not bold:
            hue = "sky"
        end = _spread_bar(s, b, y, a.tokens, a.rounds, hue, a.name)
    else:
        s.bar(b.start, y - BAR / 2, b(a.tokens) - b.start, BAR, hue, hatched=True, name=a.name)
        lo, hi = b(min(a.rounds)), b(max(a.rounds))
        if hi - lo > SPREAD_MIN:
            s.span(lo, hi, y)
        else:
            hi = b(a.tokens)
        end = max(hi, b(a.tokens))
    text = f"{a.tokens:,.0f}"
    x, rule = end + VALUE_GAP, b(c.threshold)
    if x - 5 <= rule <= x + measure(text, VALUE, bold=bold) + 5:
        x = rule + VALUE_GAP
    box = s.text(x, y + 7, text, size=VALUE, bold=bold, halo=True)
    note = _card_gain(c, a)
    if words and bold:
        note = f"{note} over {CARDS_BEST}"
    elif words and a is c.best:
        note = CARDS_BEST
    if note:
        s.text(box.x1 + 10, y + 7, note, size=NOTE, cls="" if bold else "soft", halo=True)


def _cards_wide(c: data.TwoCards, path: Path, description: str) -> Path:
    s = Chart(WIDTH, 1300, CARDS_TITLE, description)
    featured = _cards_featured(c)
    # (a) The consolidated card by its sessions, a row a number of them.
    b = CARD_TOKENS
    a = Scale(0, 1, b.start, b.end)
    s.text(a.start, 30, CARDS_A, bold=True)
    _round_key(s, a.start, 60, _pair_entries())
    ticks = _share_ticks()
    last = _rounds_wide(s, a, 100.0, _card_rounds(c), _card_rule(c), ticks, _cards_share_title(c))
    # (b) Each arrangement's agents' tokens a second, against the hurdle, under its kind's heading.
    head = last + 54
    top = head + 70
    s.text(b.start, head, CARDS_B, bold=True)
    _cards_key(s, c, b.start, head + 30)
    y = top
    placed = []
    for kind, heading in CARD_KINDS.items():
        members = [m for m in c.arrangements if m.kind == kind]
        if not members:
            continue
        s.text(24, y + 21, heading, size=NOTE, cls="soft")
        y += GROUP_HEAD
        for m in members:
            placed.append((y + FRONTIER_PITCH / 2, m))
            y += FRONTIER_PITCH
        y += GROUP_GAP
    bottom = y - GROUP_GAP + 2
    last = x_axis(s, b, *linear_ticks(b.hi, 5000), top, bottom, title=TOKENS_TITLE)
    for y, m in placed:
        name = _sentence(_card_row(m))
        s.text(LABELS_END, y - 3, name, size=VALUE, anchor="end", bold=m is featured)
        s.text(LABELS_END, y + 19, _card_sessions(m), size=NOTE, anchor="end", cls="soft")
        _cards_bar(s, c, m, b, y, featured, True)
    _floor_rule(s, b(c.threshold), top, bottom, _cards_hurdle(c))
    s.height = last + 14
    return s.save(path)


def _cards_narrow(c: data.TwoCards, path: Path, description: str) -> Path:
    """The phone layout: panel a's rows, each one's name over its two bars, then panel b, each arrangement's name
    over its bar."""
    s = Chart(NARROW, 2600, CARDS_TITLE, description)
    x0 = 18
    featured = _cards_featured(c)
    y = 30
    for line in wrap(CARDS_A, NARROW - 2 * x0, LABEL):
        s.text(x0, y, line, bold=True)
        y += 26
    y = _round_key_narrow(s, x0, y + 4, _pair_entries()) + 44
    a = Scale(0, 1, x0, ROUNDS_END)
    ticks = _share_ticks()
    last = _rounds_narrow(s, a, y, _card_rounds(c), _card_rule(c), ticks, _cards_share_title(c))
    # (b) Each arrangement's agents' tokens a second, against the hurdle, under its kind's heading.
    y = last + 48
    s.text(x0, y, CARDS_B, bold=True)
    _cards_key(s, c, x0, y + 30)
    b = Scale(CARD_TOKENS.lo, CARD_TOKENS.hi, x0, NARROW - 90)
    y += 72
    _floor_key(s, x0, y - 8, _cards_hurdle(c))
    y += 26
    s.text(x0, y - 8, CARDS_GAIN, size=NOTE, cls="soft")
    marks, labels = linear_ticks(b.hi, 5000)
    for kind, heading in CARD_KINDS.items():
        members = [m for m in c.arrangements if m.kind == kind]
        if not members:
            continue
        s.text(x0, y + 24, heading, size=NOTE, cls="soft")
        y += 32
        for m in members:
            name = _sentence(_card_row(m))
            band = y + 46 + _name_line(s, x0, y + 20, name, _card_sessions(m), m is featured)
            band_grid(s, b, marks, band - 14, band + 14)
            _cards_bar(s, c, m, b, band, featured, False)
            _floor_stub(s, b(c.threshold), band)
            y = band + 18
        y += GROUP_GAP
    last = x_axis(s, b, marks, labels, None, y - GROUP_GAP)
    for line in wrap(TOKENS_TITLE, NARROW - 2 * x0, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


# The step cap's five rounds ----------------------------------------------------------------------------------------

STEP_CAP_TITLE = "The step cap that ships"
STEP_CAP_KEYS = {("tuned", 512): "tuned", ("change", 512): "deploy", ("change", 256): "deploy"}
"""Each configuration of the step cap's comparison by (engine, step cap): its key in data.LABELS."""
STEP_CAP_SHIPS = 256
"""The step cap the serving profile ships: the configuration to deploy at it is the one the figure is drawn for."""
STEP_CAP_A = "(a) Replies over the SLO, each round"
STEP_CAP_B = "(b) Agents' tokens/s"
STEP_CAP_SHARE = "Share of the round's replies"
"""Panel a's axis title."""
STEP_CAP_TOKENS = 4000
"""Panel b's axis of the agents' tokens a second, past the fastest round's."""


def _step_cap_row(c: data.StepCap) -> tuple[str, str, str, bool]:
    """A configuration's name (data.LABELS), its step cap as the note under it, its hue, and whether it is the
    featured one: tuned vLLM is gray, and the configuration to deploy the saturated blue at the step cap that ships
    and a softer blue at the other."""
    key = STEP_CAP_KEYS[(c.engine, c.cap)]
    hue = config_mark(key)[0]
    featured = hue == FEATURED and c.cap == STEP_CAP_SHIPS
    if hue == FEATURED and not featured:
        hue = "sky"
    return data.label(key)[0], f"{c.cap}-token steps", hue, featured


REQUIREMENT = 0.05
"""The share of a run's replies the SLO allows over its threshold (docs/SLO.md, the two-card gate's requirement): the
line the step cap's rounds and the soak's hours are read against."""
ALLOWED = f"at most {REQUIREMENT:.0%}"


def _step_cap_name(c: data.StepCap) -> str:
    name, note, _, _ = _step_cap_row(c)
    return f"{name}, {note}"


def _step_cap_ordered(caps: Sequence[data.StepCap]) -> list[data.StepCap]:
    """The configurations from the comparator to the featured one, which is the last."""
    return sorted(caps, key=lambda c: _step_cap_row(c)[3])


def _step_cap_trade(caps: Sequence[data.StepCap]) -> tuple[int, str]:
    """What the smaller steps cost the agents: the other step cap the configuration to deploy ran at, and the featured
    one's tokens a second against its own there, as a share."""
    *others, featured = _step_cap_ordered(caps)
    [other] = [c for c in others if c.engine == featured.engine]
    share = f"{featured.tokens / other.tokens - 1:+.0%}".replace("-", "\N{MINUS SIGN}")
    return other.cap, share


def _step_cap_description(caps: Sequence[data.StepCap]) -> str:
    rows = "; ".join(
        f"{data.in_prose(_step_cap_name(c))}, "
        f"{', '.join(str(r.over) for r in c.rounds)} of {c.rounds[0].replies} by round, {c.over} of {c.replies} "
        f"in all, at {c.tokens:,.0f} agents' tokens a second ({min(r.tokens for r in c.rounds):,.0f} to "
        f"{max(r.tokens for r in c.rounds):,.0f})"
        for c in caps
    )
    featured = _step_cap_ordered(caps)[-1]
    other, share = _step_cap_trade(caps)
    deploy = data.in_prose(_step_cap_row(featured)[0])
    return (
        "In panel a, in each of five rounds beside 24 agent sessions, each configuration's share of the round's "
        f"replies over that round's SLO, against the requirement of {ALLOWED}, and in panel b each one's agents' "
        f"tokens a second over the calls' first 240 s, the median of its rounds with their range: {rows}; from "
        f"{other}- to {featured.cap}-token steps {deploy}'s agents' tokens a second change by {share}."
    )


def step_cap(out: Path) -> list[Path]:
    """report-step-cap.svg and its phone layout: the step cap that ships. (a) A row a round: tuned vLLM's share of the
    round's replies over its SLO, the configuration to deploy's at 512-token steps and its share at 256, a bar each
    with its value, one under the other, against the requirement. (b) A row a configuration: its agents' tokens a second, a bar
    to the median of its rounds with their range, and beside the featured one's what the smaller steps cost."""
    caps = data.step_caps()
    description = _step_cap_description(caps)
    return [
        _step_cap_wide(caps, out / "report-step-cap.svg", description),
        _step_cap_narrow(caps, out / "report-step-cap-narrow.svg", description),
    ]


def _step_cap_rounds(caps: Sequence[data.StepCap]) -> list[Round]:
    """Panel a's rows, a round each: every configuration's share of the round's replies over its SLO as a bar, from
    the comparator's to the featured configuration's."""
    ordered = _step_cap_ordered(caps)
    rows = []
    for cells in zip(*(c.rounds for c in ordered), strict=True):
        [number] = {r.round for r in cells}
        bars = [
            (r.share, (r.share,), _step_cap_row(c)[2], f"{r.share:.1%}")
            for c, r in zip(ordered, cells, strict=True)
        ]
        rows.append((f"Round {number}", bars))
    return rows


def _step_cap_entries(caps: Sequence[data.StepCap]) -> list[tuple[str, str]]:
    """Panel a's key, in the order a round's bars stand in: each configuration's hue and its name with its step
    cap, the featured one's last."""
    return [(_step_cap_row(c)[2], _step_cap_name(c)) for c in _step_cap_ordered(caps)]


def _step_cap_tokens(s: Chart, b: Scale, y: float, c: data.StepCap, trade: tuple[int, str]) -> None:
    """A configuration's agents' tokens a second at ``y``, and after the featured one's value what it gives up
    against itself at the other step cap."""
    _, _, hue, featured = _step_cap_row(c)
    note = f"{trade[1]} from {trade[0]}" if featured else ""
    _rate_bar(s, b, y, c.tokens, [r.tokens for r in c.rounds], hue, featured, note)


def _step_cap_wide(caps: Sequence[data.StepCap], path: Path, description: str) -> Path:
    s = Chart(WIDTH, 1000, STEP_CAP_TITLE, description)
    # (a) The rounds, a row each. The key takes two lines: tuned vLLM, then the configuration to deploy's two step
    # caps.
    a = Scale(0, 1, PLOT_START, 840)
    s.text(a.start, 30, STEP_CAP_A, bold=True)
    entries = _step_cap_entries(caps)
    _round_key(s, a.start, 60, entries[:1])
    _round_key(s, a.start, 86, entries[1:])
    last = _rounds_wide(
        s, a, 126.0, _step_cap_rounds(caps), (REQUIREMENT, ALLOWED), _share_ticks(), STEP_CAP_SHARE
    )
    # (b) The agents' tokens a second, a row a configuration.
    head = last + 54
    top = head + 20
    b = Scale(0, STEP_CAP_TOKENS, PLOT_START, 840)
    s.text(b.start, head, STEP_CAP_B, bold=True)
    ys = [top + CALLS_PITCH / 2 + CALLS_PITCH * k for k in range(len(caps))]
    bottom = ys[-1] + CALLS_PITCH / 2
    last = x_axis(s, b, *linear_ticks(STEP_CAP_TOKENS, 1000), top, bottom, title=TOKENS_TITLE)
    trade = _step_cap_trade(caps)
    for y, c in zip(ys, caps, strict=True):
        name, note, _, featured = _step_cap_row(c)
        s.text(LABELS_END, y - 3, name, size=VALUE, anchor="end", bold=featured)
        s.text(LABELS_END, y + 19, note, size=NOTE, anchor="end", cls="soft")
        _step_cap_tokens(s, b, y, c, trade)
    s.height = last + 14
    return s.save(path)


def _step_cap_narrow(caps: Sequence[data.StepCap], path: Path, description: str) -> Path:
    """The phone layout: panel a's rounds, each one's name over its three bars, then panel b's configurations, each
    one's name over its bar."""
    s = Chart(NARROW, 2400, STEP_CAP_TITLE, description)
    x0 = 18
    a = Scale(0, 1, x0, ROUNDS_END)
    s.text(x0, 30, STEP_CAP_A, bold=True)
    y = _round_key_narrow(s, x0, 60, _step_cap_entries(caps)) + 44
    last = _rounds_narrow(
        s, a, y, _step_cap_rounds(caps), (REQUIREMENT, ALLOWED), _share_ticks(), STEP_CAP_SHARE
    )
    y = last + 48
    s.text(x0, y, STEP_CAP_B, bold=True)
    y += 8
    b = Scale(0, STEP_CAP_TOKENS, x0, 240)
    marks, labels = linear_ticks(STEP_CAP_TOKENS, STEP_CAP_TOKENS / 2)
    trade = _step_cap_trade(caps)
    for c in caps:
        name, note, _, featured = _step_cap_row(c)
        band = y + 46 + _name_line(s, x0, y + 20, name, note, featured)
        band_grid(s, b, marks, band - 14, band + 14)
        _step_cap_tokens(s, b, band, c, trade)
        y = band + 18
    last = x_axis(s, b, marks, labels, None, y)
    for line in wrap(TOKENS_TITLE, NARROW - 2 * x0, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


# The optimization rounds -------------------------------------------------------------------------------------------

OPT_TITLE = "The serving engine's additions in five rounds"
OPT_ROWS = (
    ("v1.14's configuration to deploy", "sky"),
    ("the same with the first four additions", FEATURED),
)
"""The optimization rounds' two configurations, a scenario's two bars in order: v1.14's configuration to deploy, and
the same with the first four of version 1.15's additions to the engine (the fifth, the decode hold, is the follow-up
rounds'); each its name and its hue. Both are the configuration to deploy, in its blue (labels.MARKS): the one with
the additions is the saturated blue, the featured one, and v1.14's a softer blue."""
OPT_A = "(a) Replies over the SLO"
OPT_B = "(b) Agents' tokens/s"
OPT_SHARE_STEP = 0.05
"""Panel a's scale ends at the first multiple of this past the worst round, and at 25% or more."""
OPT_SHARE_TICKS = 0.1
"""Panel a's ticks."""
OPT_TOKENS_STEP = 5000
"""Panel b's ticks; its scale ends at the first tick past the busiest round, since the 16 sessions' samples of the
whole corpus differ far more from round to round than the 24 sessions' do."""
OPT_SHARE_TITLE = "Share of the replies, the 5 rounds pooled"
"""Panel a's axis title."""
OPT_SPANS = ((PLOT_START, 680.0), (800.0, 1050.0))
"""The two panels' plots in the wide layout, each with room after it for its longest range's value."""


def _opt_pooled(rounds: Sequence[data.StepCapRound]) -> tuple[int, int]:
    return sum(r.over for r in rounds), sum(r.replies for r in rounds)


def _opt_lines(name: str) -> tuple[str, str]:
    """A scenario's name as a row's two lines: its agent sessions, then the rest (the corpus, the callers)."""
    agents, callers = name.rsplit(", ", 1)
    head, _, corpus = agents.partition(" of ")
    return head, f"{corpus}, {callers}" if corpus else callers


def _opt_share_top(scenarios: Sequence[data.OptScenario]) -> float:
    worst = max(r.share for c in scenarios for rounds in (c.before, c.after) for r in rounds)
    return max(0.25, math.ceil(worst / OPT_SHARE_STEP + 1e-9) * OPT_SHARE_STEP)


def _opt_share_ticks(top: float) -> tuple[list[float], list[str]]:
    marks = [k * OPT_SHARE_TICKS for k in range(int(top / OPT_SHARE_TICKS + 1e-9) + 1)]
    return marks, [pct(m) for m in marks]


def _opt_tokens_top(scenarios: Sequence[data.OptScenario]) -> float:
    busiest = max(r.tokens for c in scenarios for rounds in (c.before, c.after) for r in rounds)
    return math.ceil(busiest / OPT_TOKENS_STEP + 1e-9) * OPT_TOKENS_STEP


def _opt_description(scenarios: Sequence[data.OptScenario]) -> str:
    rows = []
    for c in scenarios:
        cells = []
        for (name, _), rounds in zip(OPT_ROWS, (c.before, c.after), strict=True):
            over, replies = _opt_pooled(rounds)
            tokens = [r.tokens for r in rounds]
            cells.append(
                f"{name} {', '.join(str(r.over) for r in rounds)} by round, {over} of {replies} in all, at "
                f"{data._median(tokens):,.0f} agents' tokens a second ({min(tokens):,.0f} to {max(tokens):,.0f})"
            )
        rows.append(f"{c.name}, {', '.join(cells)}")
    return (
        f"In panel a, each scenario's replies over their round's SLO in {len(scenarios[0].after)} rounds, v1.14's "
        "configuration to deploy and the same with the first four of version 1.15's additions to the engine, the "
        f"rounds pooled with the rounds' range, against the requirement of {ALLOWED}, and in panel b the agents' "
        "tokens a second over the calls' first 240 s, the median of the rounds with their range: "
        f"{'; '.join(rows)}."
    )


def opt_rounds(out: Path) -> list[Path]:
    """report-opt-rounds.svg and its phone layout: v1.14's configuration to deploy against the same with the first
    four of version 1.15's additions, in five rounds, a row a scenario with a bar for each. (a) The share of the
    scenario's replies over their round's SLO, the rounds pooled, with the rounds' range, against the requirement.
    (b) The agents' tokens a second, the median of the rounds, with their range."""
    scenarios = data.opt_rounds()
    description = _opt_description(scenarios)
    return [
        _opt_wide(scenarios, out / "report-opt-rounds.svg", description),
        _opt_narrow(scenarios, out / "report-opt-rounds-narrow.svg", description),
    ]


def _opt_bars(c: data.OptScenario) -> tuple[list[RowBar], list[RowBar]]:
    """A scenario's row in each panel, a bar a configuration: (a) its replies over the SLO, the rounds' pooled share
    with the rounds' range, and (b) its agents' tokens a second, the rounds' median with their range."""
    shares, rates = [], []
    for (_, hue), rounds in zip(OPT_ROWS, (c.before, c.after), strict=True):
        over, replies = _opt_pooled(rounds)
        shares.append((over / replies, [r.share for r in rounds], hue, f"{over / replies:.1%}"))
        tokens = [r.tokens for r in rounds]
        rates.append((data._median(tokens), tokens, hue, f"{data._median(tokens):,.0f}"))
    return shares, rates


def _opt_entries() -> list[tuple[str, str]]:
    """The key's entries, in the order a scenario's bars stand: each configuration's hue and its name in sentence
    case."""
    return [(hue, _sentence(name)) for name, hue in OPT_ROWS]


def _opt_wide(scenarios: Sequence[data.OptScenario], path: Path, description: str) -> Path:
    """A row a scenario, its name in the configurations' column and its two bars in each panel, which the key
    names."""
    s = Chart(WIDTH, 900, OPT_TITLE, description)
    a = Scale(0, _opt_share_top(scenarios), *OPT_SPANS[0])
    b = Scale(0, _opt_tokens_top(scenarios), *OPT_SPANS[1])
    s.text(a.start, 30, OPT_A, bold=True)
    s.text(b.start, 30, OPT_B, bold=True)
    _round_key(s, a.start, 60, _opt_entries())
    top, pitch = 100.0, _row_pitch(len(OPT_ROWS))
    bottom = top + pitch * len(scenarios)
    last = x_axis(s, a, *_opt_share_ticks(a.hi), top, bottom, title=OPT_SHARE_TITLE)
    x_axis(s, b, *linear_ticks(b.hi, OPT_TOKENS_STEP), top, bottom, title=TOKENS_TITLE)
    for k, c in enumerate(scenarios):
        y = top + pitch * (k + 0.5)
        head, note = _opt_lines(c.name)
        s.text(LABELS_END, y - 3, head, size=VALUE, anchor="end")
        s.text(LABELS_END, y + 19, note, size=NOTE, anchor="end", cls="soft")
        shares, rates = _opt_bars(c)
        _row_bars(s, a, y - PAIR / 2, shares, a(REQUIREMENT))
        _row_bars(s, b, y - PAIR / 2, rates)
    _v_threshold(s, a(REQUIREMENT), top, bottom, ALLOWED)
    s.height = last + 14
    return s.save(path)


def _opt_narrow(scenarios: Sequence[data.OptScenario], path: Path, description: str) -> Path:
    """The phone layout: the two panels one above the other, each scenario's name over its two bars."""
    s = Chart(NARROW, 2400, OPT_TITLE, description)
    x0 = 18
    a = Scale(0, _opt_share_top(scenarios), x0, NARROW - 90)
    b = Scale(0, _opt_tokens_top(scenarios), x0, NARROW - 90)
    panels = (
        (OPT_A, a, _opt_share_ticks(a.hi), OPT_SHARE_TITLE),
        (OPT_B, b, linear_ticks(b.hi, OPT_TOKENS_STEP), TOKENS_TITLE),
    )
    y = 30.0
    for k, (title, scale, (marks, labels), axis) in enumerate(panels):
        s.text(x0, y, title, bold=True)
        if k:
            y += 8
        else:
            # The key over both panels, and the requirement named once, over the scenarios.
            y = _round_key_narrow(s, x0, y + 30, _opt_entries()) + 44
            _threshold_key(s, x0, y - 8, ALLOWED)
        for c in scenarios:
            band = y + 46 + _name_line(s, x0, y + 20, *_opt_lines(c.name))
            band_grid(s, scale, marks, band - 14, band + PAIR + 14)
            _row_bars(s, scale, band, _opt_bars(c)[k], None if k else scale(REQUIREMENT))
            if not k:
                _threshold_stub(s, scale(REQUIREMENT), band, PAIR)
            y = band + PAIR + 18
        last = x_axis(s, scale, marks, labels, None, y)
        for line in wrap(axis, NARROW - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
        y = last + 48
    s.height = last + 14
    return s.save(path)


# The serving engine's additions inside the engine ---------------------------------------------------------------------

ADDITIONS_TITLE = "The serving engine's additions inside the engine"
ADDITION_PANELS = (
    ("(a) A call's first token", "first_p50", "first_p95", 400, 100),
    ("(b) Its 16th token", "sixteenth_p50", "sixteenth_p95", 600, 200),
)
"""Each panel: its title, the fields of data.EnginePoint its bar reaches at its seam, the p50, and at its end, the p95,
its scale's top in ms, the first tick past the slowest point, and its ticks' step."""
ADDITIONS_AXIS = "Time (ms)"
ADDITION_PARTS = ("p50", "p95")
"""What a point's bar reaches at its seam and at its end, named in the key."""
ADDITION_UNIT = 0.445
"""A millisecond's length in the wide layout, the same in both panels, so that a first token's bar and a 16th
token's compare."""
ADDITION_ROOM = 122.0
"""Between a panel's scale and the next panel's, for the two values after a bar that reaches the scale's end."""
ADDITION_NAMES = {
    "a0-grow": "v1.14's serving profile",
    "r0-grow": "+ the calls' prefixes kept on the GPU",
    "r0j-grow": "+ the next batch just in time",
    "r0h-grow": "or each turn announced",
    "r0hj-grow": "+ both",
    "r0hjt-grow": "+ the agents' reservations taken",
    "r0hjtd-grow": "+ their decodes held for 16 tokens",
    "r0hjtd48-grow": "+ held for 48 tokens: the profile",
}
"""Each adopted point's row, by its arm in the record's table.py, named as docs/REPORT.md's table of the engine points
names it, a chain read down: each row adds to the one before it, but the one that begins "or", which adds the announced
turn to retention in place of just in time, and the next, which has both."""


def _addition_name(point: data.EnginePoint) -> str:
    """A point's row's name (ADDITION_NAMES)."""
    if point.arm not in ADDITION_NAMES:
        raise ValueError(
            f"no row name for {point.arm}: add the report's name for it to ADDITION_NAMES"
        )
    return ADDITION_NAMES[point.arm]


def _additions_description(points: Sequence[data.EnginePoint]) -> str:
    rows = "; ".join(
        f"{_addition_name(p)}, {_runs_of_count(p.runs)}: the first token {p.first_p50:,.0f} and "
        f"{p.first_p95:,.0f} ms, the 16th {p.sixteenth_p50:,.0f} and {p.sixteenth_p95:,.0f} ms"
        for p in points
    )
    return (
        "On the box, two calls beside 12 closed-loop agents, the calls' first token and their 16th token at p50 and "
        "p95, each the median of the runs, as the serving profile gained each addition it adopted, each row adding to "
        "the one before it but the announced turn's, which adds it to the kept prefixes in place of just in time, and "
        f"the next, which has both: {rows}."
    )


def additions(out: Path) -> list[Path]:
    """report-engine-points.svg and its phone layout: the serving engine's additions inside the engine, on the box, in
    the order the serving profile gained them (the engine points record's table.py, its "adopted" table), a ladder
    whose last row is the profile: each point's bar reaches its p50 and goes on to its p95, of (a) the calls' first
    token and (b) their 16th token, the medians of its runs, with the two values after it."""
    points = data.engine_points()
    for _, _, high, top_ms, _ in ADDITION_PANELS:
        if max(getattr(p, high) for p in points) > top_ms:
            raise ValueError(f"a point's {high} is past its panel's scale of {top_ms} ms")
    description = _additions_description(points)
    return [
        _additions_wide(points, out / "report-engine-points.svg", description),
        _additions_narrow(points, out / "report-engine-points-narrow.svg", description),
    ]


def _addition_row(
    s: Chart,
    scale: Scale,
    y: float,
    point: data.EnginePoint,
    fields: tuple[str, str],
    featured: bool,
) -> tuple[float, float]:
    """A point's bar in one panel, to its p50 and on to its p95, and after it the two values as the report's table
    sets them, the p50 and then the p95. Returns where the bar's seam and its end are."""
    p50, p95 = (getattr(point, field) for field in fields)
    hue = FEATURED if featured else "sky"
    end = _extended_bar(s, scale, y, p50, p95, hue, _addition_name(point))
    box = s.text(end + VALUE_GAP, y + 7, f"{p50:,.0f}", size=VALUE, bold=featured, halo=True)
    box = s.text(box.x1 + 5, y + 7, "/", size=VALUE, cls="soft", halo=True)
    s.text(box.x1 + 5, y + 7, f"{p95:,.0f}", size=VALUE, bold=featured, halo=True)
    return scale(p50), end


def _addition_keys() -> list[tuple[Key, str]]:
    """The two parts of a point's bar (ADDITION_PARTS), in the fills of every row but the profile's."""
    return [
        (_part_key(fill), f"To the {part}")
        for fill, part in zip(EXTENSIONS["sky"], ADDITION_PARTS, strict=True)
    ]


def _additions_wide(points: Sequence[data.EnginePoint], path: Path, description: str) -> Path:
    s = Chart(WIDTH, 800, ADDITIONS_TITLE, description)
    profile = points[-1]
    right = 24 + max(measure(_addition_name(p), VALUE, bold=p is profile) for p in points)
    start = right + PLOT_START - LABELS_END
    top = 50.0
    ys = [top + ABLATION_PITCH / 2 + ABLATION_PITCH * k for k in range(len(points))]
    bottom = ys[-1] + ABLATION_PITCH / 2
    last = bottom
    keys = _addition_keys()
    legend(s, right - legend_width(keys), 30, keys, width=right)
    for title, low, high, top_ms, step in ADDITION_PANELS:
        scale = Scale(0, top_ms, start, start + ADDITION_UNIT * top_ms)
        s.text(scale.start, 30, title, bold=True)
        marks, labels = linear_ticks(top_ms, step)
        last = max(last, x_axis(s, scale, marks, labels, top, bottom, title=ADDITIONS_AXIS))
        for y, point in zip(ys, points, strict=True):
            _addition_row(s, scale, y, point, (low, high), point is profile)
        start = scale.end + ADDITION_ROOM
    for y, point in zip(ys, points, strict=True):
        s.text(right, y + 7, _addition_name(point), size=VALUE, anchor="end", bold=point is profile)
    s.height = last + 14
    return s.save(path)


def _additions_narrow(points: Sequence[data.EnginePoint], path: Path, description: str) -> Path:
    """The phone layout: the two panels one above the other on one scale of time, each row's name over its bar."""
    s = Chart(NARROW, 2400, ADDITIONS_TITLE, description)
    x0 = 18
    profile = points[-1]
    unit = (NARROW - 118 - x0) / max(top_ms for _, _, _, top_ms, _ in ADDITION_PANELS)
    y = 30.0
    last = y
    for title, low, high, top_ms, step in ADDITION_PANELS:
        scale = Scale(0, top_ms, x0, x0 + unit * top_ms)
        marks, labels = linear_ticks(top_ms, step)
        s.text(x0, y, title, bold=True)
        y = legend(s, x0, y + 28, _addition_keys(), width=NARROW - 2 * x0) + 12
        for point in points:
            s.text(x0, y + 20, _addition_name(point), size=VALUE, bold=point is profile)
            band = y + 46
            band_grid(s, scale, marks, band - 14, band + 14)
            _addition_row(s, scale, band, point, (low, high), point is profile)
            y = band + 18
        last = x_axis(s, scale, marks, labels, None, y) + 25
        s.text(x0, last, ADDITIONS_AXIS, size=NOTE, cls="soft")
        y = last + 48
    s.height = last + 14
    return s.save(path)


# The CPU tier's two failures -----------------------------------------------------------------------------------------

TIER_TITLE = "The CPU tier's two failures"
TIER_PANELS = (
    "(a) Where the prompt tokens came from",
    "(b) A step beside one 1.46 GB load from the CPU tier",
)
SOURCES = (
    ("found in the GPU pool", "pale", "blue-pale"),
    ("loaded from the CPU tier", "lo", "soft"),
    ("computed", "mid", "hi"),
)
"""Panel a's sources of the prompt tokens, in their order from the axis: each its name in the key, then its segment's
fill (figure_style.SEGMENTS) in a row of first come first served, a gray, and in the queue's, the change's blue. The
tones run from the lightest to the strongest along a bar in either hue, and the strongest is what was computed, which
the panel is drawn for: the darkest gray, and in the queue's row the saturated blue."""
HALO_FILLS = ("mid", "hi")
"""The fills ink does not clear 4.5:1 on in both themes, the strongest gray and the saturated blue: a share's value
inside a segment of one stands on a halo of the page's color, the kernel's one way to set a label on a fill."""
TIER_BLUES = "The queue's row"
"""The key's last entry: the three tones again in the change's blues, as the queue's row draws them."""
TIER_NAMES = ("First come first served", "First come first served", "The queue")
"""Panel a's rows' names, data.TIER_POINTS' in order: v1.23's waiting queue, in arrival order within a priority, at
its 20 GiB tier and at 30 GiB, and the queue by what each request has left; each row's note names its tier."""
TIER_CHANGE = "The queue"
"""The row of panel a that is the change's: its bar in blue, its name and its values in semibold."""
LOAD_ROWS = (
    ("The step alone", "its median before a load", None, "ink", None),
    (
        "The copy engine",
        "one batch, 32 KiB pages",
        "one batch, 32 KiB",
        "gray",
        "on the copy engine in one batch of 32 KiB pages",
    ),
    (
        "The copy engine",
        "one batch, 2 MiB pages",
        "one batch, 2 MiB",
        "gray",
        "in one of 2 MiB pages",
    ),
    ("The copy engine", "23 chained jobs of 64 MB", "jobs of 64 MB", "gray", "in 23 jobs of 64 MB"),
    (
        "The copy engine",
        "92 chained jobs of 16 MB",
        "jobs of 16 MB",
        "gray",
        "in 92 jobs of 16 MB on chained streams",
    ),
    (
        "The SM copy",
        "Triton kernel, 4 programs",
        "SM copy, 4 programs",
        FEATURED,
        "through vLLM's Triton kernel with 4 programs",
    ),
)
"""Panel b's rows from the top: each its name and its note, its load (a key of data.STEP_LOADS; none for the step
alone), its bar's hue and the load in the description's prose: the reference's gray for the step alone, the
comparator's for a load on the copy engine, which version 1.24 leaves, and the saturated blue for the SM copy it
serves, as panel a draws first come first served in grays and the queue in blues."""
LOADS_STEP = 50
"""Between panel b's marks, in ms: its axis ends at the first one at or past the longest step."""
LOADS_TITLE = "Time (ms)"


def _ms(value: float) -> str:
    """A step's time as panel b prints it and the record's README states it: whole ms from 100 up, else to a tenth."""
    return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"


def _tier_note(src: data.TokenSources) -> str:
    """A panel a row's note: its CPU tier's size."""
    return f"a {src.tier_gib:.0f} GiB CPU tier"


def _tier_description(sources: Sequence[data.TokenSources], loads: data.StepLoads) -> str:
    rows = []
    for k, (name, src) in enumerate(zip(TIER_NAMES, sources, strict=True)):
        gpu, cpu, computed = (f"{share:.1%}" for share in src.shares)
        what = f"{data.in_prose(name)} over {_tier_note(src)}"
        rows.append(
            f"{what}, {gpu} found in the GPU pool, {cpu} loaded from the CPU tier and {computed} computed, at "
            f"{src.tokens:,.0f} agents' tokens a second and a median time to first token of {src.ttft_p50:.1f} s"
            if k == 0
            else f"{what}, {gpu}, {cpu} and {computed}, at {src.tokens:,.0f} and {src.ttft_p50:.1f} s"
        )
    steps = [f"{_ms(loads.loads[key].longest)} {prose}" for _, _, key, _, prose in LOAD_ROWS if key]
    return (
        "In panel a, where vLLM found the prompt tokens at their first schedule beside 24 agent sessions on the "
        "second GPU node, as "
        "shares of them, with the agents' tokens a second over the calls' first 240 s and their median time to first "
        f"token: {'; '.join(rows[:-1])}; and {rows[-1]}; in panel b, a step's time beside one 1.46 GB load from the "
        f"CPU tier on that node's bare card, {_ms(loads.alone)} alone at the median of the steps before each load, "
        "and the "
        f"longest during it {', '.join(steps[:-1])}, and {steps[-1]}."
    )


def tier(out: Path) -> list[Path]:
    """report-tier.svg and its phone layout: the CPU tier's two failures beside 24 agent sessions. (a) Where the prompt
    tokens came from at their first schedule, under first come first served at the 20 GiB tier and at 30 GiB and under
    the queue, beside each one's agents' tokens a second and median time to first token. (b) A step's time beside one
    1.46 GB load from the tier on that node's bare card, by the load's path: the copy engine, however it is cut, and
    vLLM's Triton kernel with 4 programs. The figure names no host: the second GPU node, Gray, ran both."""
    sources, loads = data.token_sources(), data.step_loads()
    description = _tier_description(sources, loads)
    return [
        _tier_wide(sources, loads, out / "report-tier.svg", description),
        _tier_narrow(sources, loads, out / "report-tier-narrow.svg", description),
    ]


def _tone_key(fills: Sequence[str], each: float) -> Key:
    """A key of segments' fills side by side, each ``each`` wide: one source's tone, or a row's tones in its order."""

    def draw(s: Chart, x: float, baseline: float) -> None:
        s.segments(x, baseline - 0.66 * NOTE, [(each, fill) for fill in fills], 0.7 * NOTE)

    return Key(each * len(fills), draw)


def _source_keys() -> list[tuple[Key, str]]:
    """Panel a's key, read once in a bar's order: each source's tone in the grays, then the three tones in the blues
    of the queue's row."""
    grays = [(_tone_key([gray], 18), _sentence(name)) for name, gray, _ in SOURCES]
    return [*grays, (_tone_key([blue for _, _, blue in SOURCES], 12), TIER_BLUES)]


def _shares(s: Chart, x: Scale, y: float, src: data.TokenSources, featured: bool) -> None:
    """A row's bar of its prompt tokens by source, centred on ``y``, in the grays or, ``featured``, in the blues:
    each share's value inside its segment, centred, where it fits there with clearance, on a halo where the fill is
    one ink does not read on (HALO_FILLS); else over the bar, centred on its segment but right of the value before it
    and of the axis. The featured row's computed share, which the panel is drawn for, is in semibold."""
    top = y - BAR / 2
    fills = [blue if featured else gray for _, gray, blue in SOURCES]
    parts = [(x(share) - x(0), fill) for share, fill in zip(src.shares, fills, strict=True)]
    starts = s.segments(x.start, top, parts, BAR)
    painted = len(s.items) - 1
    over = x.start + CLEAR
    for x0, (length, fill), (source, *_), share in zip(
        starts, parts, SOURCES, src.shares, strict=True
    ):
        s.covers.append((Box(x0, top, x0 + length, top + BAR, source), painted))
        text = f"{share:.1%}"
        bold = featured and source == SOURCES[-1][0]
        width = measure(text, NOTE, bold=bold)
        if width + 16 <= length:
            inside = (x0 + length / 2, y + 6)
            s.text(*inside, text, size=NOTE, anchor="middle", bold=bold, halo=fill in HALO_FILLS)
            continue
        left = max(x0 + length / 2 - width / 2, over)
        s.text(left, top - 8, text, size=NOTE, bold=bold, halo=True)
        over = left + width + 10


def _load_value(loads: data.StepLoads, key: str | None) -> float:
    return loads.alone if key is None else loads.loads[key].longest


def _loads_top(loads: data.StepLoads) -> int:
    """Panel b's axis's end: the first mark at or past the longest step beside a load."""
    worst = max(_load_value(loads, key) for _, _, key, _, _ in LOAD_ROWS)
    return LOADS_STEP * math.ceil(worst / LOADS_STEP)


def _slowest(loads: data.StepLoads) -> tuple[str, str]:
    """The load the longest step stood beside, and that step in the step alone's: the one ratio panel b states."""
    key = max((key for _, _, key, _, _ in LOAD_ROWS if key), key=lambda k: loads.loads[k].longest)
    return (
        key,
        f"{loads.loads[key].longest / loads.alone:.0f}\N{MULTIPLICATION SIGN} the step alone",
    )


def _load_bar(s: Chart, scale: Scale, y: float, value: float, hue: str, name: str) -> float:
    """A step's time as a bar from the scale's start, centred on ``y``, and where it ends."""
    s.bar(scale.start, y - BAR / 2, scale(value) - scale.start, BAR, hue, name=name)
    return scale(value)


def _tier_wide(
    sources: Sequence[data.TokenSources], loads: data.StepLoads, path: Path, description: str
) -> Path:
    s = Chart(WIDTH, 1000, TIER_TITLE, description)
    # (a) The prompt tokens by source, a bar a point, its agents' tokens a second and TTFT beside it.
    a = Scale(0, 1, PLOT_START, 780)
    s.text(a.start, 30, TIER_PANELS[0], bold=True)
    legend(s, a.start, 64, _source_keys(), width=WIDTH - 24 - a.start)
    top, pitch = 84, 64
    ys = [top + 48 + pitch * k for k in range(len(sources))]
    bottom = ys[-1] + 26
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    last = x_axis(
        s, a, marks, [pct(m) for m in marks], top, bottom, title="Share of the prompt tokens"
    )
    tokens_x, ttft_x = 925, WIDTH - 24
    s.text(tokens_x, ys[0] - 28, "Agents' tokens/s", size=NOTE, anchor="end", cls="soft")
    s.text(ttft_x, ys[0] - 28, "Agents' TTFT p50", size=NOTE, anchor="end", cls="soft")
    for y, name, src in zip(ys, TIER_NAMES, sources, strict=True):
        bold = name == TIER_CHANGE
        s.text(LABELS_END, y - 3, name, size=VALUE, anchor="end", bold=bold)
        s.text(LABELS_END, y + 19, _tier_note(src), size=NOTE, anchor="end", cls="soft")
        _shares(s, a, y, src, bold)
        s.text(tokens_x, y + 7, f"{src.tokens:,.0f}", size=VALUE, anchor="end", bold=bold)
        s.text(ttft_x, y + 7, f"{src.ttft_p50:.1f} s", size=VALUE, anchor="end", bold=bold)
    # (b) A step beside a load, by the load's path.
    b = Scale(0, _loads_top(loads), PLOT_START, 1000)
    yb = last + 60
    s.text(b.start, yb, TIER_PANELS[1], bold=True)
    top_b, pitch_b = yb + 22, HEADLINE_PITCH
    ys_b = [top_b + pitch_b / 2 + pitch_b * k for k in range(len(LOAD_ROWS))]
    ticks, labels = linear_ticks(b.hi, LOADS_STEP)
    last = x_axis(s, b, ticks, labels, top_b, ys_b[-1] + pitch_b / 2, title=LOADS_TITLE)
    slowest, ratio = _slowest(loads)
    for y, (name, note, key, hue, _) in zip(ys_b, LOAD_ROWS, strict=True):
        value, bold = _load_value(loads, key), hue == FEATURED
        s.text(LABELS_END, y - 3, name, size=VALUE, anchor="end", bold=bold)
        s.text(LABELS_END, y + 19, note, size=NOTE, anchor="end", cls="soft")
        end = _load_bar(s, b, y, value, hue, f"{name}, {note}")
        box = s.text(end + VALUE_GAP, y + 7, _ms(value), size=VALUE, bold=bold, halo=True)
        if key == slowest:
            s.text(box.x1 + 8, y + 7, ratio, size=NOTE, cls="soft", halo=True)
    s.height = last + 14
    return s.save(path)


def _tier_narrow(
    sources: Sequence[data.TokenSources], loads: data.StepLoads, path: Path, description: str
) -> Path:
    s = Chart(NARROW, 2000, TIER_TITLE, description)
    x0 = 18
    # (a)
    s.text(x0, 30, TIER_PANELS[0], bold=True)
    y = legend(s, x0, 62, _source_keys(), width=NARROW - 2 * x0, pitch=25) + 18
    a = Scale(0, 1, x0, NARROW - 26)
    marks = [0.0, 0.5, 1.0]
    for name, src in zip(TIER_NAMES, sources, strict=True):
        bold = name == TIER_CHANGE
        band = y + 68 + _name_line(s, x0, y + 20, name, _tier_note(src), bold)
        band_grid(s, a, marks, band - 14, band + 14)
        _shares(s, a, band, src, bold)
        s.text(
            x0,
            band + 36,
            f"{src.tokens:,.0f} agents' tokens/s, their TTFT p50 {src.ttft_p50:.1f} s",
            size=NOTE,
            cls="soft",
        )
        y = band + 48
    last = x_axis(s, a, marks, [pct(m) for m in marks], None, y)
    for line in wrap("Share of the prompt tokens", NARROW - 2 * x0, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    # (b)
    y = last + 48
    for line in ("(b) A step beside one 1.46 GB load", "from the CPU tier"):
        s.text(x0, y, line, bold=True)
        y += 26
    y -= 14
    b = Scale(0, _loads_top(loads), x0, 300)
    ticks = list(range(0, int(b.hi) + 1, 2 * LOADS_STEP))
    slowest, ratio = _slowest(loads)
    for name, note, key, hue, _ in LOAD_ROWS:
        value, bold = _load_value(loads, key), hue == FEATURED
        band = y + 46 + _name_line(s, x0, y + 20, name, note, bold)
        band_grid(s, b, ticks, band - 14, band + 14)
        end = _load_bar(s, b, band, value, hue, f"{name}, {note}")
        box = s.text(end + VALUE_GAP, band + 7, _ms(value), size=VALUE, bold=bold, halo=True)
        y = band + 18
        if (
            key == slowest
        ):  # the ratio after the value where it fits the canvas, else under the value, as its own
            if box.x1 + 8 + measure(ratio, NOTE) <= NARROW - 10:
                s.text(box.x1 + 8, band + 7, ratio, size=NOTE, cls="soft", halo=True)
            else:
                s.text(box.x1, band + 31, ratio, size=NOTE, anchor="end", cls="soft")
                y = band + 44
    last = x_axis(s, b, ticks, [str(t) for t in ticks], None, y)
    for line in wrap(LOADS_TITLE, NARROW - 2 * x0, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


# The card's envelope beside both calls --------------------------------------------------------------------------------

ENVELOPE_TITLE = "The card's envelope beside both calls"
ENVELOPE_ROWS = {
    ("The serving profile through v1.23", 4): "The serving profile through v1.23",
    ("The queue and the SM copy", 12): "+ the queue and the copy",
    ("The queue and the SM copy", 24): "+ the queue and the copy",
}
"""Each configuration of the envelope's gate by its name in table.py's GATE and its agent sessions: its name as Table 10
gives it. A row is named by its sessions, which the envelope counts, over its configuration."""
ENVELOPE_FEATURED = ("The queue and the SM copy", 24)
"""The row the figure is drawn for, in the saturated blue with its name and its values in semibold: version 1.24's
configuration at the envelope it carries. The other two, each a version of the configuration to deploy, are the soft
blue."""
ENVELOPE_PANELS = (
    ("share", "(a) Replies over the SLO", 0.02, "{:.1%}"),
    ("tokens", "(b) Agents' tokens/s", 5000, "{:,.0f}"),
    ("ttft_p99", "(c) Agents' TTFT p99", 100, "{:.0f} s"),
)
"""Each panel: the field of data.Envelope and its rounds it plots, its title, its scale's step and its value's format,
a latency's with its unit as every chart prints one. A scale ends at the first mark at or past the greatest round."""
ENVELOPE_TITLES = {
    "share": "Share of the replies, the {rounds} rounds pooled",
    "tokens": TOKENS_TITLE,
    "ttft_p99": "Time (s)",
}
"""Each panel's axis title, under a panel title that names its measure. The share's says what its bar is, the rounds'
replies pooled and no one round's, with the rounds' count from the record (_envelope_title)."""
ENVELOPE_SPANS = ((PLOT_START, 520.0), (624.0, 790.0), (890.0, 1024.0))
"""The wide layout's three plots, each with room after it for a row's value past its rounds' range."""


def _envelope_note(c: data.Envelope) -> str:
    """A row's note: its configuration, as a sentence's part."""
    name = ENVELOPE_ROWS[(c.name, c.sessions)]
    return name if name.startswith("+") else data.in_prose(name).removeprefix("the ")


def _envelope_title(rows: Sequence[data.Envelope], field: str) -> str:
    """A panel's axis title, the share's with the count of the rounds its bar pools."""
    return ENVELOPE_TITLES[field].format(rounds=len(rows[0].rounds))


def _envelope_description(rows: Sequence[data.Envelope]) -> str:
    cells = []
    for k, c in enumerate(rows):
        name = ENVELOPE_ROWS[(c.name, c.sessions)]
        named = k == 0 or name != ENVELOPE_ROWS[(rows[k - 1].name, rows[k - 1].sessions)]
        what = (
            f"{data.in_prose(name.removeprefix('+ '))} beside {c.sessions} agent sessions"
            if named
            else f"beside {c.sessions}"
        )
        tokens, ttft = ([r.tokens for r in c.rounds], [r.ttft_p99 for r in c.rounds])
        cells.append(
            f"{what}, {', '.join(str(r.over) for r in c.rounds)} by round, {c.over} of {c.replies} in all, its agents "
            f"at {c.tokens:,.0f} tokens a second ({min(tokens):,.0f} to {max(tokens):,.0f}) and a p99 of "
            f"{c.ttft_p99:.0f} s ({min(ttft):.0f} to {max(ttft):.0f})"
        )
    return (
        f"In panel a, each configuration's replies over their round's SLO in version 1.24's {len(rows[0].rounds)} "
        "envelope rounds, by round and as a share of the rounds' replies pooled, against the requirement of "
        f"{ALLOWED}, in panel b the agents' tokens a second over the calls' first 240 s and in panel c their time to "
        f"first token at p99, each the median of the rounds with their range: {'; '.join(cells[:-1])}; and "
        f"{cells[-1]}."
    )


def envelope(out: Path) -> list[Path]:
    """report-envelope.svg and its phone layout: the card's envelope beside both calls in version 1.24's three rounds,
    the serving profile through v1.23 at its envelope of 4 agent sessions and the queue and the copy at 12 and 24,
    three panels of value bars that share the rows. (a) Each configuration's replies over their round's SLO, pooled,
    with the rounds' range, against the requirement. (b) The agents' tokens a second, the rounds' median and their
    range. (c) The agents' time to first token at p99, likewise."""
    rows = data.envelope()
    description = _envelope_description(rows)
    return [
        _envelope_wide(rows, out / "report-envelope.svg", description),
        _envelope_narrow(rows, out / "report-envelope-narrow.svg", description),
    ]


def _envelope_axis(
    rows: Sequence[data.Envelope], field: str, step: float
) -> tuple[list[float], list[str]]:
    """A panel's marks and their labels: to the first mark at or past the greatest round."""
    worst = max(getattr(r, field) for c in rows for r in c.rounds)
    marks, labels = linear_ticks(step * math.ceil(worst / step), step)
    return marks, [pct(m) for m in marks] if field == "share" else labels


def _envelope_cell(
    s: Chart,
    rows: Sequence[data.Envelope],
    c: data.Envelope,
    field: str,
    value: str,
    scale: Scale,
    y: float,
) -> None:
    """A configuration's value in one panel at ``y``: a bar to the rounds pooled (its share over the SLO) or to their
    median, the rounds' range over its end, and its value after them; a token rate's with what it is of the first
    row's."""
    featured = (c.name, c.sessions) == ENVELOPE_FEATURED
    spread = [getattr(r, field) for r in c.rounds]
    name = f"{c.sessions} agent sessions"
    end = _value_bar(s, scale, y, getattr(c, field), spread, FEATURED if featured else "sky", name)
    text = value.format(getattr(c, field))
    if field == "share":
        _after_bar(s, end, y, text, featured, scale(REQUIREMENT))
        return
    box = s.text(end + VALUE_GAP, y + 7, text, size=VALUE, bold=featured, halo=True)
    if field == "tokens" and c is not rows[0]:
        times = f"{c.tokens / rows[0].tokens:.1f}\N{MULTIPLICATION SIGN}"
        s.text(box.x1 + 8, y + 7, times, size=NOTE, cls="soft", halo=True)


def _envelope_wide(rows: Sequence[data.Envelope], path: Path, description: str) -> Path:
    s = Chart(WIDTH, 600, ENVELOPE_TITLE, description)
    top, pitch = 76, HEADLINE_PITCH
    ys = [top + pitch / 2 + pitch * k for k in range(len(rows))]
    bottom = ys[-1] + pitch / 2
    for y, c in zip(ys, rows, strict=True):
        bold = (c.name, c.sessions) == ENVELOPE_FEATURED
        s.text(
            LABELS_END, y - 3, f"{c.sessions} agent sessions", size=VALUE, anchor="end", bold=bold
        )
        s.text(LABELS_END, y + 19, _envelope_note(c), size=NOTE, anchor="end", cls="soft")
    last = bottom
    for (field, title, step, value), span in zip(ENVELOPE_PANELS, ENVELOPE_SPANS, strict=True):
        marks, labels = _envelope_axis(rows, field, step)
        scale = Scale(0, marks[-1], *span)
        s.text(scale.start, 30, title, bold=True)
        axis = _envelope_title(rows, field)
        last = max(last, x_axis(s, scale, marks, labels, top, bottom, title=axis))
        for y, c in zip(ys, rows, strict=True):
            _envelope_cell(s, rows, c, field, value, scale, y)
        if field == "share":
            _v_threshold(s, scale(REQUIREMENT), top, bottom, ALLOWED)
    s.height = last + 14
    return s.save(path)


def _envelope_narrow(rows: Sequence[data.Envelope], path: Path, description: str) -> Path:
    s = Chart(NARROW, 2000, ENVELOPE_TITLE, description)
    x0 = 18
    y = 30.0
    for field, title, step, value in ENVELOPE_PANELS:
        marks, labels = _envelope_axis(rows, field, step)
        scale = Scale(0, marks[-1], x0, 290 if field == "tokens" else 300)
        s.text(x0, y, title, bold=True)
        y += 12
        if field == "share":  # the requirement named once, over the rows its rule runs down
            y += 30
            _threshold_key(s, x0, y - 8, ALLOWED)
        for c in rows:
            bold = (c.name, c.sessions) == ENVELOPE_FEATURED
            name = f"{c.sessions} agent sessions"
            band = y + 46 + _name_line(s, x0, y + 20, name, _envelope_note(c), bold)
            band_grid(s, scale, marks, band - 14, band + 14)
            _envelope_cell(s, rows, c, field, value, scale, band)
            if field == "share":
                _threshold_stub(s, scale(REQUIREMENT), band)
            y = band + 18
        last = x_axis(s, scale, marks, labels, None, y)
        for line in wrap(_envelope_title(rows, field), NARROW - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
        y = last + 48
    s.height = last + 14
    return s.save(path)


# The 24-hour soak ---------------------------------------------------------------------------------------------------

SOAK_TITLE = "The 24-hour soak"
SOAK_PANELS = (
    (
        "(a) Share of each hour's replies over the SLO",
        "(a) Share of each hour's replies",
        "over the SLO",
    ),
    ("(b) Agents' tokens/s in each hour",),
)
"""Each panel's title, and after it the lines a phone breaks it into where it is too long for one."""
SOAK_STEPS = (0.02, 5000)
"""Between each panel's marks: its scale ends at the first one at or past its greatest hour."""
SOAK_HOURS = (0, 6, 12, 18, 23)
"""The hours the time axis names."""
SOAK_FILLS = ("soft", "red")
"""An hour's column, its fill (figure_style.SEGMENTS): the change's blue in its soft tone, 24 hours of one
configuration, and for an hour over the requirement the red of what the voice loses, the figure's one such mark."""
SOAK_PLOT = 190.0
"""A panel's plot, its height."""


def _soak_description(soak: data.Soak) -> str:
    worst = max(soak.hours, key=lambda h: h.share)
    first, rest = soak.hours[0], soak.hours[1:]
    return (
        f"The soak's {len(soak.hours)} hours: in panel a, the share of each hour's replies over the SLO of "
        f"{soak.slo:,.0f} ms against the requirement of {ALLOWED}, {soak.over} of {soak.replies:,} replies "
        f"({soak.share:.1%}) over the day, at most {worst.over} of {worst.replies} ({worst.share:.1%}) in hour "
        f"{worst.hour}; in panel b, the agents' tokens a second, {soak.tokens:,.0f} over the day, "
        f"{first.tokens:,.0f} in hour 0 and {min(h.tokens for h in rest):,.0f} to "
        f"{max(h.tokens for h in rest):,.0f} in each hour after."
    )


def _columns(
    s: Chart, hours: Sequence[float], x: Scale, y: Scale, width: float, limit: float | None = None
) -> None:
    """Each hour's value as a column from the scale's start, in SOAK_FILLS: an hour over ``limit`` in the red."""
    for hour, value in enumerate(hours):
        left, top = x(hour) - width / 2, y(value)
        if top >= y.start:
            continue
        over = limit is not None and value > limit
        s.rect(left, top, width, y.start - top, f"s-{SOAK_FILLS[over]}")
        s.marks.append(Box(left, top, left + width, y.start, f"hour {hour}'s column"))


def _day_rule(s: Chart, x: Scale, y: Scale, value: float, text: str, *, wide: bool) -> None:
    """The whole day's value as a dotted rule across the hours, named right of the plot, or on a phone in a key over
    the plot's top right."""
    s.ref((x.start, y(value)), (x.end, y(value)), "ink trace dotted")
    if wide:
        s.text(x.end + THRESHOLD_GAP, y(value) + 6, text, size=NOTE, cls="soft")
        return
    entry = [(line_key("ink trace dotted", 24), text)]
    legend(s, x.end - legend_width(entry), y.end - 16, entry, width=x.end - x.start)


def _soak_scale(values: Sequence[float], step: float, top: float) -> tuple[Scale, list[float]]:
    """A panel's value scale, its plot's top at ``top``, and its marks: to the first one at or past the greatest
    hour."""
    marks, _ = linear_ticks(step * math.ceil(max(values) / step), step)
    return Scale(0, marks[-1], top + SOAK_PLOT, top), marks


def soak(out: Path) -> list[Path]:
    """report-soak.svg and its phone layout: the 24-hour soak by hour. (a) Each hour's replies over the SLO, against
    the requirement, the one hour over it in red with its share, beside the day's share. (b) The agents' tokens a
    second, beside the day's."""
    s_data = data.soak()
    description = _soak_description(s_data)
    return [
        _soak_layout(s_data, out / "report-soak.svg", description, WIDTH),
        _soak_layout(s_data, out / "report-soak-narrow.svg", description, NARROW),
    ]


def _soak_layout(soak: data.Soak, path: Path, description: str, width: float) -> Path:
    wide = width == WIDTH
    s = Chart(width, 900, SOAK_TITLE, description)
    x0 = 24 if wide else 18
    x = Scale(-0.5, 23.5, 96, 950) if wide else Scale(-0.5, 23.5, 64, width - 22)
    column = min(BAR, (x(1) - x(0)) * 0.68)
    hours = [str(h) for h in SOAK_HOURS]
    worst = max(soak.hours, key=lambda h: h.share)
    first = soak.hours[0]
    # (a) The replies over the SLO, by hour: the hour over the requirement in red, named with its share.
    y = 30.0
    for line in SOAK_PANELS[0][:1] if wide else SOAK_PANELS[0][1:]:
        s.text(x0, y, line, bold=True)
        y += 26
    shares = [h.share for h in soak.hours]
    a, marks = _soak_scale(shares, SOAK_STEPS[0], y + (8 if wide else 38))
    y_axis(s, a, marks, [pct(v) for v in marks], x.start, x.end)
    _columns(s, shares, x, a, column, REQUIREMENT)
    if wide:
        _h_threshold(s, x.start, x.end, a(REQUIREMENT), ALLOWED)
    else:  # a phone's plot runs the page's width: the requirement's name over its rule's end, clear of the hours
        s.rule(x.start, x.end, a(REQUIREMENT), "slo")
        s.text(x.end, a(REQUIREMENT) - 7, ALLOWED, size=NOTE, anchor="end", cls="red tint")
    box = s.text(
        x(worst.hour) + column / 2 + 7,
        a(worst.share) + 2,
        f"{worst.share:.1%}",
        size=VALUE,
        cls="red tint",
        bold=True,
        halo=True,
    )
    s.text(box.x1 + 8, a(worst.share) + 2, f"hour {worst.hour}", size=NOTE, cls="soft", halo=True)
    _day_rule(s, x, a, soak.share, f"the day, {soak.share:.1%}", wide=wide)
    last = x_axis(s, x, SOAK_HOURS, hours, None, a.start)
    # (b) The agents' tokens a second, by hour.
    y = last + 46
    s.text(x0, y, SOAK_PANELS[1][0], bold=True)
    tokens = [h.tokens for h in soak.hours]
    b, marks = _soak_scale(tokens, SOAK_STEPS[1], y + (34 if wide else 64))
    y_axis(s, b, marks, [f"{v:,.0f}" for v in marks], x.start, x.end)
    _columns(s, tokens, x, b, column)
    s.text(
        x(first.hour) + column / 2 + 7,
        b(first.tokens) + 16,
        f"{first.tokens:,.0f}",
        size=VALUE,
        halo=True,
    )
    _day_rule(s, x, b, soak.tokens, f"the day, {soak.tokens:,.0f}", wide=wide)
    last = x_axis(s, x, SOAK_HOURS, hours, None, b.start, title="Time since the soak began (h)")
    s.height = last + 14
    return s.save(path)


# The engine's starts and the kill drill ------------------------------------------------------------------------------

STARTS_TITLE = "The engine's starts and its recovery"
PHASES = (
    ("down", "Down until restarted", "s-red"),
    ("load", "Process, weights", "s-mid"),
    ("kv", "Compile, KV cache", "s-pale"),
    ("graphs", "Kernels, CUDA graphs", "gray tint"),
    ("serve", "Until it serves", "s-lo"),
)
"""The phases of a start's bar, each its key, its name in the legend and its fill's classes. Before a restart's, the
killed engine down: its unit's teardown and systemd's restart delay, the figure's one red mark, which its row's note
times. Then, from the unit's start (flatshare_bench.drills.START_PHASES): the engine's process until its weights are
loaded, torch.compile and the profiling run until its KV cache is allocated, the kernels' warmup and JIT builds until
its CUDA graphs are captured, and its server until it serves: its unit ready (the health check), or after the kill a
request answered. A start's four take the gray's ladder (PHASE_LADDER) two steps apart or more from each phase to the
next, so that no two neighbours are near tones in either theme, and the longest of a cold start in its strongest."""
PHASE_LADDER = ("s-pale", "s-lo", "s-mid", "gray tint")
"""The gray's tones a start's phases are drawn in, from its wash to its line."""
PHASE_ENDS = {
    "load": "the weights are loaded",
    "kv": "the KV cache is allocated",
    "graphs": "the CUDA graphs are captured",
}
"""The start phase (drills.START_PHASES) each of the bar's phases ends at."""
READY = "the unit is ready (its health check)"
STARTS_STEP = 60
"""Between the time axis's marks, in seconds: it ends at the first one at or past the longest start."""
STARTS_AXIS = "Time from the kill, or from the unit's start (s)"
PHASE_LINE = 22.0
"""Between the lines of a phase's label under its segment: its seconds, then its name."""

StartRow = tuple[str, str, list[tuple[str, float, float]], str]
"""A bar: its name and its note, its spans (a phase's key, from and to in seconds) and what its end is."""


def _phase_key(cls: str) -> Key:
    def draw(s: Chart, x: float, baseline: float) -> None:
        s._hpath(x, baseline - 0.66 * NOTE, 22, 0.7 * NOTE, cls, 2.0)

    return Key(22, draw)


def _start_spans(start: data.Start, offset: float, end: float) -> list[tuple[str, float, float]]:
    """A start's phases from ``offset`` seconds, the last until ``end``."""
    spans, at = [], offset
    for key, phase in PHASE_ENDS.items():
        to = offset + start.at(phase)
        spans.append((key, at, to))
        at = to
    return [*spans, ("serve", at, end)]


def _starts_rows(warm: data.Start, cold: data.Start, kill: data.Kill) -> list[StartRow]:
    """Each bar: its name and note, its spans, and what its end is. The cold start is the last, for the wide layout
    names its phases under it."""
    return [
        (
            "The engine killed",
            f"down {kill.restarted:.1f} s, then a warm start",
            [
                ("down", 0.0, kill.restarted),
                *_start_spans(kill.start, kill.restarted, kill.first_success),
            ],
            "a request answered",
        ),
        (
            "A warm start",
            "compile and JIT caches full",
            _start_spans(warm, 0.0, warm.at(READY)),
            "ready",
        ),
        (
            "A cold start",
            "empty caches, as on a new host",
            _start_spans(cold, 0.0, cold.at(READY)),
            "ready",
        ),
    ]


def _starts_description(warm: data.Start, cold: data.Start, kill: data.Kill) -> str:
    weights, kv, graphs = (cold.at(PHASE_ENDS[key]) for key in ("load", "kv", "graphs"))
    return (
        "The engine's phases in seconds after a kill and in two starts: killed mid-decode, it was down "
        f"{kill.restarted:.1f} s, its unit's last process gone {kill.gone:.1f} s after the kill and systemd's restart "
        f"{kill.restarted - kill.gone:.1f} s later, and its warm start listened "
        f"{kill.start.at('the server listens'):.1f} s after the restart, a request answered "
        f"{kill.first_success:.1f} s after the kill; a warm start was ready at {warm.at(READY):.1f} s, its weights "
        f"loaded at {warm.at(PHASE_ENDS['load']):.1f} s and its CUDA graphs captured at "
        f"{warm.at(PHASE_ENDS['graphs']):.1f} s; and a cold start, with empty caches, was ready at "
        f"{cold.at(READY):.1f} s, its weights at {weights:.1f} s, its KV cache at {kv:.1f} s after compiling and "
        f"profiling and its CUDA graphs at {graphs:.1f} s after the kernels' warmup and JIT builds."
    )


def starts(out: Path) -> list[Path]:
    """report-starts.svg and its phone layout: the engine's recovery from a kill, its warm start and its start on empty
    caches, each a bar of its phases on one time axis, its seconds after it. The wide layout names the cold start's
    phases under their own segments, each with its seconds, and keys only the phase that start has not; the phone
    layout, whose segments are shorter than their names, keys all five."""
    found = data.starts()
    warm, cold, kill = found["warm"], found["cold"], data.kill()
    rows = _starts_rows(warm, cold, kill)
    description = _starts_description(warm, cold, kill)
    return [
        _starts_wide(rows, out / "report-starts.svg", description),
        _starts_narrow(rows, out / "report-starts-narrow.svg", description),
    ]


def _start_bar(
    s: Chart, x: Scale, y: float, thick: float, spans: Sequence[tuple[str, float, float]]
) -> float:
    """A bar of ``spans`` centred on ``y``, its phases parted by seams and its free end rounded, one mark no label
    may cover. Returns where it ends."""
    fills = {key: cls for key, _, cls in PHASES}
    for k, (key, start, end) in enumerate(spans):
        s._segment(x(start), y - thick / 2, x(end) - x(start), thick, fills[key], k, len(spans))
    tip = x(spans[-1][2])
    s.marks.append(Box(x(spans[0][1]), y - thick / 2, tip, y + thick / 2, "a start's phases"))
    return tip


def _start_end(s: Chart, tip: float, y: float, seconds: float, what: str) -> None:
    """After a bar's end at ``tip``: its seconds, then what its end is."""
    box = s.text(tip + VALUE_GAP, y + 7, f"{seconds:.1f} s", size=VALUE, halo=True)
    s.text(box.x1 + 8, y + 7, what, size=NOTE, cls="soft", halo=True)


def _start_room(seconds: float, what: str) -> float:
    """The room ``_start_end`` takes after a bar's end."""
    return VALUE_GAP + measure(f"{seconds:.1f} s", VALUE) + 8 + measure(what, NOTE)


def _starts_top(rows: list[StartRow]) -> int:
    """Where the time axis ends, in seconds: its first mark at or past the longest start."""
    return STARTS_STEP * math.ceil(max(spans[-1][2] for _, _, spans, _ in rows) / STARTS_STEP)


def _phase_labels(
    x: Scale, spans: Sequence[tuple[str, float, float]]
) -> list[tuple[float, str, list[str]]]:
    """Each of a bar's phases as its label under the bar: its segment's middle, its seconds, and its name's lines, a
    name longer than its segment parted at its comma."""
    names = {key: name for key, name, _ in PHASES}
    labels = []
    for key, start, end in spans:
        first, comma, rest = names[key].partition(", ")
        whole = not comma or measure(names[key], NOTE) <= x(end) - x(start)
        lines = [names[key]] if whole else [first + ",", rest]
        labels.append(((x(start) + x(end)) / 2, f"{end - start:.1f} s", lines))
    return labels


def _starts_wide(rows: list[StartRow], path: Path, description: str) -> Path:
    s = Chart(WIDTH, 800, STARTS_TITLE, description)
    top_s = _starts_top(rows)
    # The last start's phases are named under their own segments, so the key holds only the phase it has not.
    named = {key for key, _, _ in rows[-1][2]}
    entries = [(_phase_key(cls), name) for key, name, cls in PHASES if key not in named]
    legend(s, PLOT_START, 30, entries, width=WIDTH - PLOT_START)
    x = Scale(0, top_s, PLOT_START, 990)
    top, pitch = 46, HEADLINE_PITCH
    ys = [top + pitch / 2 + pitch * k for k in range(len(rows))]
    labels = _phase_labels(x, rows[-1][2])
    under = ys[-1] + BAR / 2 + 21
    bottom = under + PHASE_LINE * max(len(lines) for _, _, lines in labels) + 12
    ticks = list(range(0, top_s + 1, STARTS_STEP))
    last = x_axis(s, x, ticks, [str(t) for t in ticks], top, bottom, title=STARTS_AXIS)
    for y, (name, note, spans, what) in zip(ys, rows, strict=True):
        s.text(LABELS_END, y - 3, name, size=VALUE, anchor="end")
        s.text(LABELS_END, y + 19, note, size=NOTE, anchor="end", cls="soft")
        _start_end(s, _start_bar(s, x, y, BAR, spans), y, spans[-1][2], what)
    for middle, took, lines in labels:
        s.text(middle, under, took, size=NOTE, anchor="middle", halo=True)
        for k, line in enumerate(lines, 1):
            s.text(
                middle,
                under + PHASE_LINE * k,
                line,
                size=NOTE,
                anchor="middle",
                cls="soft",
                halo=True,
            )
    s.height = last + 14
    return s.save(path)


def _starts_narrow(rows: list[StartRow], path: Path, description: str) -> Path:
    s = Chart(NARROW, 800, STARTS_TITLE, description)
    top_s = _starts_top(rows)
    x0 = 18.0
    entries = [(_phase_key(cls), name) for _, name, cls in PHASES]
    y = legend(s, x0, 28, entries, width=NARROW - 2 * x0, pitch=25) + 20
    # The time scale ends where every bar still has room after it for its seconds and what its end is.
    plot = min(
        (NARROW - 10 - x0 - _start_room(spans[-1][2], what)) * top_s / spans[-1][2]
        for _, _, spans, what in rows
    )
    x = Scale(0, top_s, x0, x0 + math.floor(plot))
    ticks = list(range(0, top_s + 1, 2 * STARTS_STEP))
    for name, note, spans, what in rows:
        band = y + 46 + _name_line(s, x0, y + 20, name, note)
        band_grid(s, x, ticks, band - 14, band + 14)
        _start_end(s, _start_bar(s, x, band, BAR, spans), band, spans[-1][2], what)
        y = band + 18
    last = x_axis(s, x, ticks, [str(t) for t in ticks], None, y)
    for line in wrap(STARTS_AXIS, NARROW - 2 * x0, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


# The campaign map, EXPERIMENTS' table of the campaigns: its rows' words ------------------------------------------------


def _at(t: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(t, datetime.UTC)


def _when(t: float) -> tuple[str, str]:
    """A moment as its day ("23 October") and its time of day ("08:59"), UTC."""
    at = _at(t)
    return f"{at.day} {at:%B}", at.strftime("%H:%M")


def _map_span(found: Sequence[data.Campaign]) -> str:
    """The campaigns' days in prose, from the first one's first to the last one's last: "23 October to 8 November
    2026"."""
    first, last = _at(found[0].start), _at(found[-1].end)
    return f"{first.day} {first:%B} to {last.day} {last:%B} {last.year}"


def _span(c: data.Campaign) -> str:
    """A campaign's span in prose: "23 October 00:13 to 08:59", or across days, "23 October 10:57 to 24 October 01:35"."""
    (d0, t0), (d1, t1) = _when(c.start), _when(c.end)
    return f"{d0} {t0} to {t1}" if d0 == d1 else f"{d0} {t0} to {d1} {t1}"


def _count(n: int, one: str, many: str) -> str:
    return f"one {one}" if n == 1 else f"{n} {many}"


def _rounds_and_samples(c: data.Campaign) -> str:
    """A campaign's rounds and its samples of the agents' traces, or a soak's runs."""
    if c.hours:
        alone, serving = c.hours
        return f"{alone:.0f} hours of voice alone, then {serving:.0f} hours serving"
    if c.rounds:
        runs = _count(c.rounds, "round", "rounds") + (
            " and its arms" if c.code == "The A100 run" else ""
        )
    else:
        runs = "one run a setting" if c.code == "SW" else "one run a point"
    return f"{runs}, {_count(len(c.seeds), 'sample', 'samples')}"


def campaign_row(c: data.Campaign) -> tuple[str, str, str, str, str, str]:
    """A campaign's row in EXPERIMENTS' campaign map: its code, what it measured, when it ran, its complete points,
    its rounds and samples, and the release its code was (BENCHMARKS and the records name the exact revision)."""
    points = "—" if c.hours else f"{c.points:,}"
    return (
        c.code,
        c.question,
        _span(c),
        points,
        _rounds_and_samples(c),
        " and ".join(c.releases),
    )


# Every figure -----------------------------------------------------------------------------------------------------

FIGURES = (
    stages,
    deployable_stages,
    kv_timeline,
    isolation,
    tail,
    window,
    step_cost,
    headline,
    cdf,
    agents,
    frontier,
    deployable,
    ablations,
    margin,
    a100,
    cards,
    step_cap,
    additions,
    opt_rounds,
    tier,
    envelope,
    soak,
    starts,
)


def main(out_dir: Path) -> list[Path]:
    """Every report figure in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return [path for figure in FIGURES for path in figure(out_dir)]


if __name__ == "__main__":
    for written in main(Path(sys.argv[1])):
        print(written)
