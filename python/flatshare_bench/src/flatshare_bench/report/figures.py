"""The report's figures, docs/figures/report-*.svg, one function each, from the values :mod:`.data` reads.

    python -m flatshare_bench.report.figures OUT_DIR

Each figure is 1120 units wide, as the README's are, and shows at about 840 CSS pixels; a figure with a phone layout
has it at 420 units. No figure draws a title: the report's caption under it states its claim and its key. Every
number a figure prints is a record's, as docs/BENCHMARKS.md's tables give it, and latency scales start at zero unless
the axis says it is logarithmic.
"""

from __future__ import annotations

import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

from flatshare_bench.e2.record import REPLY_TIMEOUT_MS
from flatshare_bench.figure_style import LABEL, NARROW, NOTE, VALUE, WIDTH, Box, fmt, measure, wrap
from flatshare_bench.report import data
from flatshare_bench.report.charts import (
    STAGES,
    Chart,
    Key,
    Scale,
    band_grid,
    column_rug,
    ecdf,
    hatch_key,
    legend,
    legend_columns,
    legend_width,
    line_key,
    linear_ticks,
    mark_key,
    pct,
    rug,
    runs,
    seconds,
    stage_key,
    x_axis,
    y_axis,
)


def _row_label(
    s: Chart, x: float, baseline: float, name: str, note: str, right: float, gap: float = 8
) -> float:
    """A phone layout's row label: its name at ``baseline``, and its note after the name where both end before
    ``right``, else on a line of its own under the name. Returns how far below ``baseline`` the label's last line is:
    0, or the note's line."""
    box = s.text(x, baseline, name, size=VALUE)
    if box.x1 + gap + measure(note, NOTE) <= right:
        s.text(box.x1 + gap, baseline, note, size=NOTE, cls="soft")
        return 0.0
    s.text(x, baseline + 22, note, size=NOTE, cls="soft")
    return 22.0


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
    """report-deployable-stages.svg and its phone layout: the same stages for W4's configurations of
    docs/BENCHMARKS.md's stage table (data.W4_STAGES), from W4's records."""
    cells = data.stages(data.w4_rows(), data.W4_STAGES)
    description = _stage_description(cells)
    title = "Where the voice turn's seconds go on what an operator can deploy"
    return [
        _stages_wide(cells, out / "report-deployable-stages.svg", description, title),
        _stages_narrow(cells, out / "report-deployable-stages-narrow.svg", description, title),
    ]


def _stage_bar(s: Chart, cells: data.Stages, scale: Scale, y: float, thick: float) -> None:
    """A configuration's stacked bar at ``y``, its stages' seconds inside the scheduler's two where they fit, and
    its mean V2V at its end."""
    parts = [(scale(v) - scale(0), stage) for v, stage in zip(cells.parts, STAGES, strict=True)]
    starts = s.stack(scale.start, y, parts, thick)
    for (length, stage), start, value in zip(parts, starts, cells.parts, strict=True):
        text = f"{seconds(value)} s"
        if stage in INSIDE and measure(text, NOTE) + 16 <= length:
            s.text(start + length / 2, y + thick / 2 + 6, text, size=NOTE, anchor="middle")
    s.text(scale(cells.v2v) + 8, y + thick / 2 + 6, f"{seconds(cells.v2v)} s", size=NOTE, halo=True)


def _stages_wide(cells: Sequence[data.Stages], path: Path, description: str, title: str) -> Path:
    s = Chart(WIDTH, 400, title, description)
    entries = [(stage_key(stage), STAGE_NAMES[stage]) for stage in STAGES]
    left = (WIDTH - legend_width(entries)) / 2
    legend(s, left, 30, entries, width=WIDTH - 2 * left + 1)
    right, x0, length = 300, 324, 736
    marks, labels = linear_ticks(6, 1)
    scale = Scale(0, marks[-1] * 1000, x0, x0 + length)
    top, pitch, thick = 62, 64, 28
    rows = [top + 18 + pitch * k for k in range(len(cells))]
    bottom = rows[-1] + thick + 16
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
        runs = f"{c.runs} runs"
        s.text(right, y + 12, name, size=VALUE, anchor="end")
        s.text(
            right, y + 35, f"{note}, {runs}" if note else runs, size=NOTE, anchor="end", cls="soft"
        )
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
        runs = f"{note + ', ' if note else ''}{c.runs} runs"
        y += _row_label(s, x0, y + 18, name, runs, NARROW - x0, gap=7)
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
KV_LINES = (("agent A", "gray trace"), ("agent B", "gray trace dashed"))
"""Each agent request's line in panel b: its name and its stroke."""


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
    pool's blocks in use as the KV case's two agent requests prefill and decode, the moments the pool is full, the KV
    case would send its voice turn and vLLM preempts one request, the wait between the last two shaded. (b) Each
    request's output tokens on the same time axis, the wait shaded there too."""
    samples = data.kv_timeline()
    m = data.kv_moments(samples)
    description = _kv_description(samples, m)
    return [
        _kv_layout(samples, m, out / "report-kv-timeline.svg", description, WIDTH),
        _kv_layout(samples, m, out / "report-kv-timeline-narrow.svg", description, NARROW),
    ]


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
    titles = [
        wrap(
            f"(a) The KV pool in use, % of its {data.POOL_BLOCKS:,} blocks", width - 2 * x0, LABEL
        ),
        wrap("(b) Each agent request's output tokens", width - 2 * x0, LABEL),
    ]
    if wide:
        x = Scale(KV_TICKS[0], KV_TICKS[-1], 100, 1000)
        a_top = 82
    else:
        x = Scale(KV_TICKS[0], KV_TICKS[-1], 64, width - 22)
        a_top = 30 + 26 * len(titles[0]) + 56  # the moments' two lines over the plot
    a = Scale(0, 1, a_top + (140 if wide else 190), a_top)
    b_title = a.start + (44 if wide else 50)
    b_top = b_title + 26 * (len(titles[1]) - 1) + 30
    b = Scale(0, 900, b_top + 90, b_top)
    # The wait for room, under both panels' grids.
    for scale in (a, b):
        s.rect(
            x(m.send.t),
            scale.end,
            x(m.before.t) - x(m.send.t),
            scale.start - scale.end,
            "orange area",
        )
    for k, line in enumerate(titles[0]):
        s.text(x0, 30 + 26 * k, line, size=LABEL)
    y_axis(s, a, [0, 0.5, 1], [pct(v) for v in (0, 0.5, 1)], x.start, x.end)
    for k, line in enumerate(titles[1]):
        s.text(x0, b_title + 26 * k, line, size=LABEL)
    marks_b, labels_b = linear_ticks(800, 400)
    y_axis(s, b, marks_b, labels_b, x.start, x.end)
    for t in KV_TICKS[1:-1]:
        for scale in (a, b):
            s.vrule(x(t), scale.end, scale.start, "grid")
    title = "Seconds since the two agent requests were sent"
    if wide:
        last = x_axis(s, x, KV_TICKS, [str(t) for t in KV_TICKS], None, b.start, title=title)
    else:
        last = x_axis(s, x, KV_TICKS, [str(t) for t in KV_TICKS], None, b.start)
        for line in wrap(title, width - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
    # (a): the pool, its moments above it, the wait and the preemption inside it.
    s.trace([(x(p.t), a(p.usage)) for p in samples], "gray trace")
    sends = (
        ["the KV case sends its voice turn"] if wide else ["the KV case sends", "its voice turn"]
    )
    for moment, lines, anchor in ((m.full, ["the pool full"], "end"), (m.send, sends, "start")):
        at = x(moment.t)
        first = a.end - 20 - 22 * (len(lines) - 1)
        s.trace([(at, a.end - 12), (at, a(moment.usage) - 4)], "hair")
        for k, line in enumerate(lines):
            s.text(
                at + (-6 if anchor == "end" else 6), first + 22 * k, line, size=NOTE, anchor=anchor
            )
    middle = (x(m.send.t) + x(m.before.t)) / 2
    # The wait's label, under the full pool's line; on a phone, where the shaded wait is narrower than the label,
    # higher, clear of the pool's line on either side of it.
    low = a(0) - 34 if wide else a(0.3)
    s.text(middle, low, "no room for", size=NOTE, anchor="middle", halo=True)
    s.text(middle, low + 22, "a waiting request", size=NOTE, anchor="middle", halo=True)
    if wide:
        s.text(
            x(m.after.t) + 14,
            a(0.75) + 6,
            "agent B preempted so that agent A can grow",
            size=NOTE,
            halo=True,
        )
    else:  # over the pool's line, which runs at half the pool after the preemption
        for k, line in enumerate(("agent B preempted", "so that agent A", "can grow")):
            s.text(x(m.after.t) + 20, a(0.9) + 6 + 22 * k, line, size=NOTE, halo=True)
    # (b): each request's output tokens, named at their ends; in the wait the two lines run as one.
    s.text(middle, b(600) + 5, "both decoding", size=NOTE, anchor="middle", halo=True)
    for k, (name, cls) in enumerate(KV_LINES):
        points = [(x(p.t), b(p.tokens[k])) for p in _decoding(samples, k)]
        s.trace(points, cls)
        if wide:
            s.text(points[-1][0] + 10, points[-1][1] + 6, name, size=NOTE, halo=True)
        else:  # over its line's end, inside the plot
            s.text(points[-1][0], points[-1][1] - 10, name, size=NOTE, anchor="end", halo=True)
    s.height = last + 14
    return s.save(path)


# The two waits in isolation ---------------------------------------------------------------------------------------

PROBE_ROWS = {"stock": ("stock", "gray"), "a": ("kv-preemption", "sky"), "ab": ("change", "blue")}
"""Each probe block's row: its configuration's label (data.LABELS) and its hue."""
RUN_PITCH = 11.0
"""Between a row's runs, each on a line of its own in the order they ran."""


def _run_offsets(count: int) -> list[float]:
    return [(k - (count - 1) / 2) * RUN_PITCH for k in range(count)]


def _probe_description(
    kv: dict[str, list[data.Probe]], prefill: dict[str, list[data.Probe]]
) -> str:
    def spread(values: list[int]) -> str:
        return f"{min(values):,}\N{EN DASH}{max(values):,} ms"

    names = {engine: data.in_prose(data.label(PROBE_ROWS[engine][0])[0]) for engine in data.ENGINES}
    first = "; ".join(f"{names[e]} {spread([r.ttft for r in runs])}" for e, runs in kv.items())
    sixteenth = "; ".join(
        f"{names[e]} {spread([r.t16 for r in runs if r.t16])}" for e, runs in prefill.items()
    )
    return (
        f"Four probe runs on the RTX 5060 Ti for each of {names['stock']}, {names['a']} and {names['ab']} at its "
        "16-token window: in panel a, the KV case, the voice turn's first token while two agent requests of 18k tokens hold the "
        f"pool ({first}); in panel b, the prefill case, its first token to its 16th while three agent prefills of 10k "
        f"tokens run, the 16th at {sixteenth}."
    )


def isolation(out: Path) -> list[Path]:
    """report-isolation.svg and its phone layout: the two waits in isolation, the probe runs of each case by
    configuration. (a) The KV case: the turn's first token, sent into a pool two agent requests fill. (b) The prefill
    case: each run's first token to its 16th, sent beside three agent prefills."""
    kv, prefill = data.probes("kv"), data.probes("prefill")
    # The panels' subtitles state what each case's turn was sent beside: every run's line must say it.
    for case, runs_by_engine, beside in (("kv", kv, (2, 17999)), ("prefill", prefill, (3, 10000))):
        sent = {(r.agents, r.agent_tokens) for runs in runs_by_engine.values() for r in runs}
        if sent != {beside}:
            raise ValueError(f"the {case} case's runs were sent beside {sent}, not {beside}")
    description = _probe_description(kv, prefill)
    return [
        _isolation_wide(kv, prefill, out / "report-isolation.svg", description),
        _isolation_narrow(kv, prefill, out / "report-isolation-narrow.svg", description),
    ]


ISOLATION_PANELS = (
    (
        "(a) The KV case",
        "two agent requests of 18k tokens fill the pool",
        "Time to the first token (ms)",
    ),
    (
        "(b) The prefill case",
        "three agent prefills of 10k tokens run",
        "Time from the request (ms)",
    ),
)


def _kv_runs(s: Chart, a: Scale, y: float, runs: Sequence[data.Probe], hue: str) -> None:
    """Panel a's row: each run's first token over their median, labeled right of the last."""
    middle = statistics.median(r.ttft for r in runs)
    s.median(a(middle), y, hue, half=24)
    for offset, run in zip(_run_offsets(len(runs)), runs, strict=True):
        s.mark(a(run.ttft), y + offset, hue)
    edge = a(max(r.ttft for r in runs)) + 14
    s.text(edge, y + 6, f"{middle:,.0f} ms", size=NOTE, halo=True)


def _prefill_runs(
    s: Chart, b: Scale, y: float, engine: str, runs: Sequence[data.Probe], hue: str
) -> None:
    """Panel b's row: each run from its first token to its 16th, over the 16th's median."""
    middle = statistics.median(r.t16 for r in runs if r.t16)
    s.median(b(middle), y, hue, half=24)
    for offset, run in zip(_run_offsets(len(runs)), runs, strict=True):
        if run.t16 is None:
            raise ValueError(f"a prefill run of {engine} ended before its 16th token")
        s.trace([(b(run.ttft), y + offset), (b(run.t16), y + offset)], f"{hue} trace", wire=False)
        s.mark(b(run.ttft), y + offset, hue)
        s.mark(b(run.t16), y + offset, hue, filled=True)
    edge = b(max(r.t16 for r in runs if r.t16)) + 14
    s.text(edge, y + 6, f"{middle:,.0f} ms", size=NOTE, halo=True)


ISOLATION_KEY = [
    (mark_key("gray"), "a run's first token"),
    (mark_key("gray", filled=True), "its 16th token"),
]


def _isolation_wide(
    kv: dict[str, list[data.Probe]],
    prefill: dict[str, list[data.Probe]],
    path: Path,
    description: str,
) -> Path:
    s = Chart(WIDTH, 400, "The two waits in isolation", description)
    right = 236
    a = Scale(0, 6000, 262, 640)
    b = Scale(0, 2000, 692, 1070)
    for scale, (title, note, _) in zip((a, b), ISOLATION_PANELS, strict=True):
        s.text(scale.start, 30, title, size=LABEL)
        s.text(scale.start, 54, note, size=NOTE, cls="soft")
    top, pitch = 80, 64
    rows = [top + 34 + pitch * k for k in range(len(data.ENGINES))]
    bottom = rows[-1] + 34
    marks_a, labels_a = linear_ticks(6000, 2000)
    marks_b, labels_b = linear_ticks(2000, 500)
    last = x_axis(s, a, marks_a, labels_a, top, bottom, title=ISOLATION_PANELS[0][2])
    x_axis(s, b, marks_b, labels_b, top, bottom, title=ISOLATION_PANELS[1][2])
    for y, engine in zip(rows, data.ENGINES, strict=True):
        key, hue = PROBE_ROWS[engine]
        name, note = data.label(key)
        s.text(right, y - 4, name, size=VALUE, anchor="end")
        s.text(right, y + 19, note, size=NOTE, anchor="end", cls="soft")
        _kv_runs(s, a, y, kv[engine], hue)
        _prefill_runs(s, b, y, engine, prefill[engine], hue)
    key_y = last + 34
    legend(s, b.start, key_y, ISOLATION_KEY, width=b.end - b.start + 40)
    s.height = key_y + 14
    return s.save(path)


def _isolation_narrow(
    kv: dict[str, list[data.Probe]],
    prefill: dict[str, list[data.Probe]],
    path: Path,
    description: str,
) -> Path:
    """The phone layout: the two cases one above the other, each row's name over its runs."""
    s = Chart(NARROW, 1200, "The two waits in isolation", description)
    x0 = 18
    y = legend(s, x0, 30, ISOLATION_KEY, width=NARROW - 2 * x0) + 30
    scales = (Scale(0, 6000, x0, NARROW - 62), Scale(0, 2000, x0, NARROW - 62))
    ticks = (linear_ticks(6000, 2000), linear_ticks(2000, 1000))
    last = y
    for k, ((title, note, axis), scale, (marks, labels)) in enumerate(
        zip(ISOLATION_PANELS, scales, ticks, strict=True)
    ):
        y = last + (48 if k else 8)
        s.text(x0, y, title, size=LABEL)
        for line in wrap(note, NARROW - 2 * x0, NOTE):
            y += 24
            s.text(x0, y, line, size=NOTE, cls="soft")
        y += 12
        for engine in data.ENGINES:
            key, hue = PROBE_ROWS[engine]
            name, row_note = data.label(key)
            band = y + 58 + _row_label(s, x0, y + 20, name, row_note, NARROW - x0)
            band_grid(s, scale, marks, band - 26, band + 26)
            if k == 0:
                _kv_runs(s, scale, band, kv[engine], hue)
            else:
                _prefill_runs(s, scale, band, engine, prefill[engine], hue)
            y = band + 30
        last = x_axis(s, scale, marks, labels, None, y)
        for line in wrap(axis, NARROW - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


# Both waits beside closed-loop agents -------------------------------------------------------------------------------

TAIL_LINES = {
    "stock": ("stock", "on", "gray trace"),
    "a": ("kv-preemption", "with", "sky trace"),
    "ab": ("change", "with", "blue trace thick"),
}
"""Each probe block's line: its configuration's label (data.LABELS), the word the description puts before its name,
and its stroke."""


def _tail_name(engine: str) -> str:
    return data.label(TAIL_LINES[engine][0])[0]


TAIL_PANELS = {
    "fresh": ("(a) Fresh agent prompts", "with the agents' prompts fresh each turn"),
    "grow": ("(b) Growing agent prompts", "with each agent's prompt grown by a tool result"),
}
TAIL_TICKS = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
"""A first token's log axis from 40 ms, under the fastest turn (43 ms), to 15 s, past the slowest (11.8 s)."""
TAIL_TITLE = "Time to a call turn's first token (s, log scale)"


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
    token as ECDFs, each pooled over a scheduler's two runs, on a log scale: (a) with the agents' prompts fresh each
    turn, (b) with each grown by a tool result; stock vLLM, the KV preemption and the change at its 16-token window."""
    tails = data.tails()
    description = _tail_description(tails)
    return [
        _tail_wide(tails, out / "report-tail.svg", description),
        _tail_narrow(tails, out / "report-tail-narrow.svg", description),
    ]


def _tail_lines(s: Chart, tails, mode: str, x: Scale, y: Scale) -> None:
    """A panel's p95 rule and each scheduler's ECDF."""
    s.rule(x.start, x.end, y(0.95), "rule")
    s.text(x.start + 10, y(0.95) + 22, "p95", size=NOTE, cls="soft", halo=True)
    for engine, (_, _, cls) in TAIL_LINES.items():
        s.trace(ecdf(tails[mode][engine].turns, x, y, cap=data.TAIL_TIMEOUT_MS), cls)


def _tail_wide(tails, path: Path, description: str) -> Path:
    s = Chart(WIDTH, 460, "The calls' first tokens beside agents", description)
    panels = {
        "fresh": Scale(40, 15000, 96, 520, log=True),
        "grow": Scale(40, 15000, 656, 1080, log=True),
    }
    y = Scale(0, 1, 380, 120)
    legend(
        s,
        24,
        30,
        [(line_key(cls), _tail_name(engine)) for engine, (_, _, cls) in TAIL_LINES.items()],
        width=WIDTH - 48,
    )
    s.text(24, 102, "Share of the calls' turns", size=NOTE, cls="soft")
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    last = 0.0
    for mode, x in panels.items():
        s.text(24 if mode == "fresh" else x.start, 72, TAIL_PANELS[mode][0], size=LABEL)
        if mode == "fresh":
            y_axis(s, y, marks, [pct(v) for v in marks], x.start, x.end)
        else:  # the same scale as panel a's, its labels left to it
            for v in marks:
                s.rule(x.start, x.end, y(v), "axis" if v == 0 else "grid")
            s.vrule(x.start, y.end, y.start)
        for tick in TAIL_TICKS:
            s.vrule(x(tick), y.end, y.start, "grid")
        last = x_axis(
            s,
            x,
            TAIL_TICKS,
            [f"{t / 1000:g}" for t in TAIL_TICKS],
            None,
            y.start,
            title=TAIL_TITLE,
        )
        _tail_lines(s, tails, mode, x, y)
    s.height = last + 14
    return s.save(path)


TAIL_NARROW_TICKS = [100, 1000, 10000]
"""The phone layout's ticks, a decade apart: the wide figure's would crowd its narrower axis."""


def _tail_narrow(tails, path: Path, description: str) -> Path:
    """The phone layout: the key over the two panels, one above the other, on the same scales."""
    s = Chart(NARROW, 1100, "The calls' first tokens beside agents", description)
    x0 = 18
    last = legend(
        s,
        x0,
        30,
        [(line_key(cls), _tail_name(engine)) for engine, (_, _, cls) in TAIL_LINES.items()],
        width=NARROW - 2 * x0,
        pitch=25,
    )
    x = Scale(40, 15000, 64, NARROW - 22, log=True)
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for mode in TAIL_PANELS:
        top = last + 46
        s.text(x0, top, TAIL_PANELS[mode][0], size=LABEL)
        s.text(x0, top + 26, "Share of the calls' turns", size=NOTE, cls="soft")
        y = Scale(0, 1, top + 52 + 220, top + 52)
        y_axis(s, y, marks, [pct(v) for v in marks], x.start, x.end)
        for tick in TAIL_TICKS:
            s.vrule(x(tick), y.end, y.start, "grid")
        last = x_axis(
            s, x, TAIL_NARROW_TICKS, [f"{t / 1000:g}" for t in TAIL_NARROW_TICKS], None, y.start
        )
        for line in wrap(TAIL_TITLE, NARROW - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
        _tail_lines(s, tails, mode, x, y)
    s.height = last + 14
    return s.save(path)


# Sizing the window ------------------------------------------------------------------------------------------------


def _window_description(tokens: list[int], runs: list[data.KneeRun]) -> str:
    t40 = data.knee_medians(runs, "t40_p95")
    first = data.knee_medians(runs, "ttft_p95")
    prompt = data.knee_medians(runs, "prompt_tps")
    sweep = "; ".join(f"{w} tokens {seconds(t40[w])} s" for w in t40)
    return (
        f"In panel a, the cumulative share of {len(tokens):,} voice replies by their output tokens, "
        f"{statistics.median(tokens):.0f} at the median and {max(tokens)} at most, "
        f"{data.share_within(tokens, 16):.0%} ending within a 16-token window and all but "
        f"{sum(t > 48 for t in tokens)} within a 48-token one; in panel b, the p95 of a 40-token reply's last token "
        f"by the window, the median of two runs ({sweep}), its first token's staying between "
        f"{min(first.values()):.0f} and {max(first.values()):.0f} ms; in panel c, the agents' prompt tokens a "
        f"second, {prompt[16]:,.0f} at 16 tokens and {prompt[48]:,.0f} at 48, {1 - prompt[48] / prompt[16]:.0%} less."
    )


def window(out: Path) -> list[Path]:
    """report-window.svg and its phone layout: why the window is 48 tokens. (a) The ECDF of the voice replies' output
    tokens, with the 16- and 48-token windows and the share of replies each covers. (b) The window sweep: a 40-token
    reply's last token at p95, and its first token, by the window. (c) The agents' prompt tokens a second beside it."""
    tokens = data.reply_tokens()
    runs = data.knee()
    description = _window_description(tokens, runs)
    return [
        _window_layout(tokens, runs, out / "report-window.svg", description, WIDTH),
        _window_layout(tokens, runs, out / "report-window-narrow.svg", description, NARROW),
    ]


WINDOW_LINES = (
    ("t40_p95", "blue trace", "40th token"),
    ("ttft_p95", "blue trace dashed", "first token"),
)
"""Panel b's lines: the field each draws, its stroke and its name."""


def _window_layout(tokens: Sequence[int], runs, path: Path, description: str, width: float) -> Path:
    wide = width == WIDTH
    s = Chart(width, 1000, "Why the window is 48 tokens", description)
    x0 = 24 if wide else 18

    # (a) The replies' lengths, the two windows labelled at their tops.
    x = Scale(0, 60, 96, 540) if wide else Scale(0, 60, 64, width - 22)
    y = Scale(0, 1, 400, 90) if wide else Scale(0, 1, 360, 100)
    s.text(x0, 30, "(a) Share of the voice replies", size=LABEL)
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_axis(s, y, marks, [pct(m) for m in marks], x.start, x.end)
    ticks = list(range(0, 61, 10))
    for mark in ticks[1:]:
        s.vrule(x(mark), y.end, y.start, "grid")
    last = x_axis(
        s, x, ticks, [str(m) for m in ticks], None, y.start, title="Output tokens of the reply"
    )
    within16 = data.share_within(tokens, 16)
    beyond48 = sum(t > 48 for t in tokens)
    for size, hue in ((16, "sky"), (48, "blue")):
        s.trace([(x(size), y.start), (x(size), y.end - 12)], f"{hue} ref", wire=False)
        s.text(x(size), y.end - 20, f"{size}-token window", size=NOTE, anchor="middle")
    s.trace(ecdf(tokens, x, y), "ink trace")
    s.mark(x(16), y(within16), "sky", filled=True)
    s.text(x(16) - 10, y(0.42) + 6, f"{within16:.0%} end", size=NOTE, anchor="end", halo=True)
    s.text(x(16) - 10, y(0.42) + 30, "within it", size=NOTE, anchor="end", halo=True)
    # 48's share, left of its rule: on a phone, under the curve's rise, which crosses the middle there.
    at = y(0.5) if wide else y(0.18)
    s.text(
        x(48) - 10,
        at + 6,
        f"all but {beyond48} of {len(tokens):,}",
        size=NOTE,
        anchor="end",
        halo=True,
    )
    s.text(x(48) - 10, at + 30, "end within it", size=NOTE, anchor="end", halo=True)

    # (b) and (c), the sweep, on one window axis: beside (a), or on a phone under it.
    windows = sorted({run.window for run in runs})
    if wide:
        w = Scale(8, 72, 700, 1004)
        b_title, b = 30, Scale(0, 4000, 250, 90)
        c_title, c = 296, Scale(0, 4000, 400, 320)
        left = 620
    else:
        w = Scale(8, 72, 64, width - 22)
        b_title = last + 50
        keys = [(line_key(cls), name) for _, cls, name in WINDOW_LINES]
        key_y = legend(s, x0, b_title + 30, keys, width=width - 2 * x0)
        b = Scale(0, 4000, key_y + 30 + 160, key_y + 30)
        c_title = b.start + 46
        c = Scale(0, 4000, c_title + 30 + 80, c_title + 30)
        left = x0
    s.text(left, b_title, "(b) A 40-token reply, p95 (s)", size=LABEL)
    marks_b, labels_b = linear_ticks(4, 1)
    y_axis(s, b, [m * 1000 for m in marks_b], labels_b, w.start, w.end)
    s.text(left, c_title, "(c) Agents' prompt tokens/s", size=LABEL)
    marks_c, labels_c = linear_ticks(4000, 2000)
    y_axis(s, c, marks_c, labels_c, w.start, w.end)
    for k in windows:
        s.vrule(w(k), b.end, b.start, "grid")
        s.vrule(w(k), c.end, c.start, "grid")
    bottom = x_axis(
        s, w, windows, [str(k) for k in windows], None, c.start, title="The window (tokens)"
    )
    for field, cls, name in WINDOW_LINES:
        medians = data.knee_medians(runs, field)
        s.trace([(w(k), b(v)) for k, v in medians.items()], cls)
        for run in runs:
            s.mark(w(run.window), b(getattr(run, field)), "blue")
        if wide:
            s.text(w(windows[-1]) + 14, b(medians[windows[-1]]) + 6, name, size=NOTE, halo=True)
    prompt = data.knee_medians(runs, "prompt_tps")
    s.trace([(w(k), c(v)) for k, v in prompt.items()], "blue trace")
    for run in runs:
        s.mark(w(run.window), c(run.prompt_tps), "blue")
    drop = 1 - prompt[48] / prompt[16]
    s.text(
        w(48),
        c(prompt[48]) + 26,
        f"\N{MINUS SIGN}{drop:.0%} from 16",
        size=NOTE,
        anchor="middle",
        halo=True,
    )
    s.height = (last if wide else bottom) + 14
    return s.save(path)


# The rules' cost per step -----------------------------------------------------------------------------------------

DECODE_STEP_MS = (16, 20)
"""A voice reply's decode step on the RTX 5060 Ti beside agent prefill that the hold keeps back, from the gaps between
its tokens (the prefill case's 16-20 ms a token, docs/BENCHMARKS.md): one of the box's steps against which the figure
sets the scheduler's cost, with a reply's step alone (W6) and the agents' at 24 session trees (the rounds)."""
BOX_TICKS = (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
"""The box's CPU's axis, in µs."""


def _step_row(cost: data.StepCost) -> str:
    return f"{cost.running} running" + (f", {cost.variant}" if cost.variant else "")


STEP_SETS = {
    False: "In the pull requests' cost tables",
    True: "In the serving profile: 256-token steps, scheduled asynchronously",
}
"""The two sets of states, each under its heading: the PRs' tables' steady states, and the box's configuration."""


def _step_description(costs: Sequence[data.StepCost]) -> str:
    worst = max(costs, key=lambda c: c.off_ratio)
    rows = "; ".join(
        f"{'in the serving profile, ' if c.serving else ''}{data.SCENARIOS[c.scenario]}, {_step_row(c)}: "
        f"{c.stock:,.0f}, {c.off:,.0f} and {c.on:,.0f} µs"
        for c in costs
    )
    least, most = (
        f"{f(c.added for c in costs):+,.0f}".replace("-", "\N{MINUS SIGN}") for f in (min, max)
    )
    lo, hi = DECODE_STEP_MS
    return (
        "The wall time of one call of vLLM 0.32.0's Scheduler.schedule() on the host's CPU in each state, on a "
        f"log scale whose second axis is the box's CPU, {data.box_cpu():.2f} times as long, beside the box's steps "
        f"(a voice reply's alone, {data.voice_step_ms():.0f} ms; beside held agent prefill, {lo} to {hi} ms; the "
        f"agents' at 24 session trees, {data.agents_step_ms():.0f} ms), for stock vLLM, the change with its switches "
        f"off (within {worst.off_ratio:.1%} of stock's in every state) and the configuration to deploy (its change to "
        f"the step from {least} to {most} µs): {rows}."
    )


def _band_key() -> Key:
    """The box's band of steps in a legend: a short stretch of it."""

    def draw(s: Chart, x: float, baseline: float) -> None:
        s.rect(x, baseline - 0.75 * NOTE, 22, 0.9 * NOTE, "gray band")

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
    """report-step-cost.svg: the rules' cost per scheduler step on the host, a row per state under its scenario, the
    PRs' tables' steady states and then the serving profile's: stock vLLM's step, the change's with its switches off
    (on stock's, within a few percent), and the configuration to deploy's, with what it adds, on a log scale that also
    holds a decode step on the RTX 5060 Ti. Its second axis is the box's CPU, the host's times bench_host.py's factor
    between them, on which the box's own steps stand: a voice reply's alone and beside held agent prefill, and the
    agents' at 24 session trees."""
    costs = data.step_costs()
    paths = [
        _step_cost_wide(costs, out / "report-step-cost.svg"),
        _step_cost_narrow(costs, out / "report-step-cost-narrow.svg"),
    ]
    return paths


def _step_keys(alone: float, agents: float) -> list[tuple[Key, str]]:
    lo, hi = DECODE_STEP_MS
    return [
        (mark_key("gray"), data.label("stock")[0]),
        (mark_key("gray", "dot", filled=True), data.label("switches-off")[0]),
        (mark_key("blue", filled=True), data.label("deploy")[0]),
        (line_key("gray trace dotted"), f"a reply's step on the box alone, {alone:.0f} ms"),
        (_band_key(), f"beside held agent prefill, {lo}\N{EN DASH}{hi} ms"),
        (line_key("gray trace dashed"), f"the agents' step at 24 session trees, {agents:.0f} ms"),
    ]


def _step_marks(s: Chart, x: Scale, cost, row_y: float, rules: Sequence[float]) -> None:
    """A state's marks: stock vLLM's ring, the change's with its switches off and the configuration to deploy's,
    joined, and what it adds beside them."""
    s.trace([(x(cost.off), row_y), (x(cost.on), row_y)], "blue trace", wire=False)
    s.mark(x(cost.stock), row_y, "gray", r=6.5)
    s.mark(x(cost.off), row_y, "gray", "dot", filled=True)
    s.mark(x(cost.on), row_y, "blue", filled=True)
    # What the change adds, right of the row's marks: the hold's is a saving, the deployed mark left of stock's.
    sign = "+" if cost.added >= 0 else "\N{MINUS SIGN}"
    label = f"{sign}{abs(cost.added):,.0f} µs"
    marks = (x(min(cost.stock, cost.off, cost.on)), x(max(cost.stock, cost.off, cost.on)))
    _beside(s, marks, row_y + 6, label, rules)


STEP_NARROW_TICKS = [20, 100, 1000, 10000]
"""The phone layout's ticks on the host's axis, its decades: the wide figure's ten would crowd it."""


def _step_cost_narrow(costs, path: Path) -> Path:
    """The phone layout: the key, the box's axis, then each state's name over its marks under its set's and its
    scenario's headings, and the host's axis."""
    box = data.box_cpu()
    alone, agents = data.voice_step_ms(), data.agents_step_ms()
    lo, hi = DECODE_STEP_MS
    s = Chart(NARROW, 2600, "The rules' cost per scheduler step", _step_description(costs))
    x0 = 18
    x = Scale(
        20, 30000, x0 + 10, NARROW - 22, log=True
    )  # its first tick's label clear of the canvas's edge
    last = legend(s, x0, 30, _step_keys(alone, agents), width=NARROW - 2 * x0, pitch=25)
    title = f"the box's CPU (ms), scaled: the host's time \N{MULTIPLICATION SIGN} {box:.2f}"
    y = last + 16
    for line in wrap(title, NARROW - 2 * x0, NOTE):
        y += 25
        s.text(x0, y, line, size=NOTE, cls="soft")
    top = y + 44
    for tick in (100, 1000, 10000):
        if x.lo <= tick / box <= x.hi:
            s.vrule(x(tick / box), top - 5, top, "axis")
            s.text(
                x(tick / box), top - 12, f"{tick / 1000:g}", size=NOTE, anchor="middle", cls="mute"
            )
    s.rule(x.start, x.end, top)
    layout = []
    y = top + 4
    scenario = serving = None
    # The headings and the states' names end short of the box's reference lines, which run the whole column.
    room = x(alone * 1000 / box) - 10 - x0
    for cost in costs:
        if cost.serving != serving:
            serving = cost.serving
            y += 14 if layout else 0
            for line in wrap(STEP_SETS[serving], room, NOTE):
                s.text(x0, y + 20, line, size=NOTE)
                y += 22
            y += 8
        if cost.scenario != scenario:
            scenario = cost.scenario
            text = data.SCENARIOS[scenario][0].upper() + data.SCENARIOS[scenario][1:]
            y += 6 if layout else 0
            for line in wrap(text, room, NOTE):
                s.text(x0, y + 20, line, size=NOTE, cls="mute")
                y += 22
            y += 8
        lines = wrap(_step_row(cost), room, VALUE)
        for k, line in enumerate(lines):
            s.text(x0, y + 20 + 24 * k, line, size=VALUE)
        y += 24 * (len(lines) - 1)
        layout.append((y + 44, cost))
        y += 66
    bottom = y + 6
    s.rect(
        x(lo * 1000 / box), top, x(hi * 1000 / box) - x(lo * 1000 / box), bottom - top, "gray band"
    )
    s.vrule(x(alone * 1000 / box), top, bottom, "gray trace dotted")
    s.vrule(x(agents * 1000 / box), top, bottom, "gray trace dashed")
    rules = [x(alone * 1000 / box), x(agents * 1000 / box)]
    for row_y, cost in layout:
        band_grid(s, x, STEP_NARROW_TICKS, row_y - 14, row_y + 14)
        _step_marks(s, x, cost, row_y, rules)
    last = x_axis(
        s, x, STEP_NARROW_TICKS, [f"{t / 1000:g}" for t in STEP_NARROW_TICKS], None, bottom
    )
    for line in wrap(
        "Scheduler.schedule()'s wall time a step on the host (ms, log scale)", NARROW - 2 * x0, NOTE
    ):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


def _step_cost_wide(costs, path: Path) -> Path:
    box = data.box_cpu()
    alone, agents = data.voice_step_ms(), data.agents_step_ms()
    lo, hi = DECODE_STEP_MS
    s = Chart(WIDTH, 600, "The rules' cost per scheduler step", _step_description(costs))
    right = 372
    x = Scale(20, 30000, 400, 1090, log=True)
    last = legend(s, x.start, 30, _step_keys(alone, agents), width=x.end - x.start)
    top = last + 80
    for tick in BOX_TICKS:
        if x.lo <= tick / box <= x.hi:
            s.vrule(x(tick / box), top - 5, top, "axis")
            s.text(
                x(tick / box), top - 12, f"{tick / 1000:g}", size=NOTE, anchor="middle", cls="mute"
            )
    s.text(
        (x.start + x.end) / 2,
        top - 40,
        f"the box's CPU (ms), scaled: the host's time \N{MULTIPLICATION SIGN} {box:.2f}",
        size=NOTE,
        anchor="middle",
        cls="soft",
    )
    s.rule(x.start, x.end, top)
    layout = []
    y = top + 4
    scenario = serving = None
    for cost in costs:
        if cost.serving != serving:
            serving = cost.serving
            y += 14 if layout else 0
            lines = wrap(STEP_SETS[serving], right - 24, NOTE)
            for k, line in enumerate(lines):
                s.text(24, y + 20 + 22 * k, line, size=NOTE)
            y += 22 * len(lines) + 8
        if cost.scenario != scenario:
            scenario = cost.scenario
            lines = wrap(
                data.SCENARIOS[scenario][0].upper() + data.SCENARIOS[scenario][1:], right - 24, NOTE
            )
            y += 6 if layout else 0
            for k, line in enumerate(lines):
                s.text(24, y + 20 + 22 * k, line, size=NOTE, cls="mute")
            y += 22 * len(lines) + 8
        layout.append((y + 14, cost))
        y += 30
    bottom = y + 6
    s.rect(
        x(lo * 1000 / box), top, x(hi * 1000 / box) - x(lo * 1000 / box), bottom - top, "gray band"
    )
    s.vrule(x(alone * 1000 / box), top, bottom, "gray trace dotted")
    s.vrule(x(agents * 1000 / box), top, bottom, "gray trace dashed")
    ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    last = x_axis(
        s,
        x,
        ticks,
        [f"{t / 1000:g}" for t in ticks],
        top,
        bottom,
        title="Scheduler.schedule()'s wall time a step on the host (ms, log scale)",
    )
    for row_y, cost in layout:
        s.text(right, row_y + 6, _step_row(cost), size=VALUE, anchor="end")
        _step_marks(s, x, cost, row_y, [x(alone * 1000 / box), x(agents * 1000 / box)])
    s.height = last + 14
    return s.save(path)


# The headline -----------------------------------------------------------------------------------------------------

HEADLINE_ROWS = {
    data.VOICE_ALONE: ("ink", "dot"),
    data.TUNED: ("gray", "circle"),
    data.ROUTER: ("gray", "square"),
    data.MAKE_ROOM: ("sky", "circle"),
    data.HOLD_16: ("sky", "circle"),
    data.CHANGE: ("blue", "circle"),
}
"""Each configuration's row, labelled as data.LABELS names it: its hue and its runs' marker. figure.toml orders them,
voice alone, the reference, first."""
V2V_TICKS = [1000, 2000, 5000, 10000, 20000]
"""A V2V p95 axis's ticks on a log scale of 1 to 20 s, wide enough for every run of tuned vLLM's tail."""


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
            f"{name}, {note}: V2V p95 {seconds(c.p95)} s at the median of {_runs_of(c)} "
            f"({seconds(min(p95s))} to {seconds(max(p95s))} s)"
        )
        if c.tokens is not None and base:
            text += (
                f", the agents' {c.tokens:,.0f} tokens/s ({c.tokens / base:.0%} of tuned vLLM's)"
            )
        rows.append(text)
    return (
        f"Each run's voice-to-voice p95 on a log scale against the SLO of {seconds(slo)} s, and the agents' tokens a "
        "second over the calls' first 240 s, with each configuration's median: "
        + "; ".join(rows)
        + "."
    )


def headline(out: Path) -> list[Path]:
    """report-headline.svg and its phone layout: each run of the headline configurations, beside 24 agent sessions
    on the RTX 5060 Ti, with voice alone on top. (a) The V2V p95 against the SLO, on a log scale. (b) The agents'
    tokens a second over the calls' first 240 s."""
    rows = data.w5_rows()
    configurations = data.headline(rows)
    slo = data.slo_ms(rows)
    description = _headline_description(configurations, slo)
    return [
        _headline_wide(configurations, slo, out / "report-headline.svg", description),
        _headline_narrow(configurations, slo, out / "report-headline-narrow.svg", description),
    ]


RUN_HALF = 9.0
"""Half a run's tick across a headline row: fifteen runs within 0.3 s stay apart as a rug, which markers would not
without a swarm several rows tall."""


def _place_value(
    s: Chart, xs: Sequence[float], y: float, text: str, *, slo: float, floor: float
) -> None:
    """A row's value beside its runs: after the rightmost; where that would cross the SLO's rule, before the
    leftmost, or past the rule where the axis leaves no room before them (``floor``, the axis's start)."""
    after = max(xs) + 12
    width = measure(text, NOTE)
    if not after - 4 <= slo <= after + width + 4:
        s.text(after, y + 6, text, size=NOTE, halo=True)
    elif min(xs) - 12 - width >= floor + 10:
        s.text(min(xs) - 12, y + 6, text, size=NOTE, anchor="end", halo=True)
    else:
        s.text(slo + 10, y + 6, text, size=NOTE, halo=True)


def _v2v_row(s: Chart, c: data.Configuration, a: Scale, y: float, slo_x: float) -> None:
    """A configuration's runs' V2V p95 at ``y`` as a rug under their median's marker, the median's value beside
    them."""
    hue, shape = HEADLINE_ROWS[c.point]
    xs = [a(r.p95) for r in c.runs]
    rug(s, xs, y, hue, shape, median=a(c.p95), half=RUN_HALF)
    _place_value(s, xs, y, f"{seconds(c.p95)} s", slo=slo_x, floor=a.start)


def _tokens_row(s: Chart, c: data.Configuration, b: Scale, y: float, base: float) -> None:
    hue, shape = HEADLINE_ROWS[c.point]
    assert c.tokens is not None
    xs = [b(r.tokens) for r in c.runs if r.tokens is not None]
    rug(s, xs, y, hue, shape, median=b(c.tokens), half=RUN_HALF)
    box = s.text(max(xs) + 12, y + 6, f"{c.tokens:,.0f}", size=NOTE, halo=True)
    s.text(box.x1 + 7, y + 6, f"{c.tokens / base:.0%}", size=NOTE, cls="soft", halo=True)


def _slo_rule(s: Chart, x: float, top: float, bottom: float, slo: float) -> None:
    s.vrule(x, top, bottom, "slo")
    s.text(
        x,
        top - 8,
        f"SLO {seconds(slo)} s",
        size=NOTE,
        anchor="middle",
        cls="orange tint",
        halo=True,
    )


V2V_TITLE = "seconds, log scale \N{BULLET} lower is better"
TOKENS_TITLE = "first 240 s of the calls \N{BULLET} higher is better"


def _headline_wide(
    configurations: Sequence[data.Configuration], slo: float, path: Path, description: str
) -> Path:
    s = Chart(WIDTH, 500, "The voice tail and the agents' throughput", description)
    right = 276
    a = Scale(1000, 20000, 300, 780, log=True)
    b = Scale(0, 4000, 820, 1000)
    s.text(a.start, 30, "(a) Voice-to-voice p95", size=LABEL)
    s.text(b.start, 30, "(b) Agents' tokens/s", size=LABEL)
    top, pitch = 82, 60
    ys = [top + 32 + pitch * k for k in range(len(configurations))]
    bottom = ys[-1] + 32
    last = x_axis(
        s, a, V2V_TICKS, [f"{t // 1000}" for t in V2V_TICKS], top, bottom, title=V2V_TITLE
    )
    marks_b, labels_b = linear_ticks(4000, 2000)
    x_axis(s, b, marks_b, labels_b, top, bottom, title=TOKENS_TITLE)
    _slo_rule(s, a(slo), top, bottom, slo)
    base = configurations[1].tokens
    for y, c in zip(ys, configurations, strict=True):
        name, note = data.label(c.point)
        s.text(right, y - 4, name, size=VALUE, anchor="end")
        s.text(right, y + 19, f"{note}, {_runs_of(c)}", size=NOTE, anchor="end", cls="soft")
        _v2v_row(s, c, a, y, a(slo))
        if c.tokens is not None:
            _tokens_row(s, c, b, y, base)
    s.height = last + 14
    return s.save(path)


def _headline_narrow(
    configurations: Sequence[data.Configuration], slo: float, path: Path, description: str
) -> Path:
    s = Chart(NARROW, 900, "The voice tail and the agents' throughput", description)
    x0 = 18
    # From 0.8 s, so that voice alone's value fits before its runs on the phone's narrower scale.
    a = Scale(800, 20000, x0, 330, log=True)
    b = Scale(0, 4000, x0, 300)
    base = configurations[1].tokens
    y = 30
    for panel in "ab":
        scale = a if panel == "a" else b
        s.text(
            x0, y, "(a) Voice-to-voice p95" if panel == "a" else "(b) Agents' tokens/s", size=LABEL
        )
        y += 12
        if panel == "a":
            y += 30
            s.text(
                a(slo),
                y - 8,
                f"SLO {seconds(slo)} s",
                size=NOTE,
                anchor="middle",
                cls="orange tint",
            )
        marks = V2V_TICKS if panel == "a" else [0, 2000, 4000]
        for c in configurations:
            if panel == "b" and c.tokens is None:
                continue
            name, note = data.label(c.point)
            band = y + 40 + _row_label(s, x0, y + 18, name, note, NARROW - x0, gap=7)
            band_grid(s, scale, marks, band - 12, band + 12)
            if panel == "a":
                s.vrule(a(slo), band - 12, band + 12, "slo")
                _v2v_row(s, c, a, band, a(slo))
            else:
                _tokens_row(s, c, b, band, base)
            y = band + 16
        labels = (
            [f"{t // 1000}" for t in V2V_TICKS] if panel == "a" else linear_ticks(4000, 2000)[1]
        )
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
CDF_TICKS = [1000, 2000, 5000, 10000, 20000, 30000]
"""The ECDF's log axis, from 1 s, under every turn measured (the fastest took 1.38 s), to the reply timeout."""


def _cdf_description(configurations: Sequence[data.Configuration], slo: float) -> str:
    rows = []
    for c in configurations:
        samples = c.samples()
        within = sum(v <= slo for v in samples) / len(samples)
        rows.append(
            f"{data.in_prose(data.label(c.point)[0])}, {len(samples)} replies over {_runs_of(c)}: {within:.0%} "
            "within the SLO, "
            f"the slowest at {seconds(samples[-1])} s"
        )
    return (
        "The cumulative share of every measured turn's voice-to-voice time, pooled over each configuration's runs, on "
        f"a log scale from 1 to 30 s with the SLO of {seconds(slo)} s and the 95th percentile marked: "
        + "; ".join(rows)
        + "."
    )


def cdf(out: Path) -> list[Path]:
    """report-cdf.svg and its phone layout: every measured turn's V2V as ECDFs, pooled over each configuration's runs
    (the warm-up turns left out, a turn that got no reply counted at 30 s), for voice alone, tuned vLLM, the
    abort-and-resubmit router and the change."""
    rows = data.w5_rows()
    configurations = [c for c in data.headline(rows) if c.point in CDF_LINES]
    slo = data.slo_ms(rows)
    description = _cdf_description(configurations, slo)
    return [
        _cdf_layout(configurations, slo, out / "report-cdf.svg", description, WIDTH),
        _cdf_layout(configurations, slo, out / "report-cdf-narrow.svg", description, NARROW),
    ]


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
    if wide:
        x = Scale(CDF_TICKS[0], CDF_TICKS[-1], 96, 1040, log=True)
        y = Scale(0, 1, 420, 70)
        s.text(24, 34, "Share of replies", size=NOTE, cls="soft")
    else:  # the lines named in a key over the plot, where a phone's plot has no room after their ends
        keys = [(line_key(cls), data.label(point)[0]) for point, cls in CDF_LINES.items()]
        last_key = legend(s, 18, 30, keys, width=width - 36, pitch=25)
        s.text(18, last_key + 40, "Share of replies", size=NOTE, cls="soft")
        x = Scale(CDF_TICKS[0], CDF_TICKS[-1], 64, width - 22, log=True)
        y = Scale(0, 1, last_key + 80 + 300, last_key + 80)
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
    s.text(x.start + 10, y(0.95) + 22, "p95", size=NOTE, cls="soft")
    s.vrule(x(slo), y.start, y.end, "slo")
    s.text(
        x(slo), y.end - 10, f"SLO {seconds(slo)} s", size=NOTE, anchor="middle", cls="orange tint"
    )
    lines = {}
    for c in configurations:
        name, cls = data.label(c.point)[0], CDF_LINES[c.point]
        points = ecdf(c.samples(), x, y, cap=REPLY_TIMEOUT_MS)
        s.trace(points, cls)
        lines[c.point] = (name, points)
    if wide:
        # The labels: the change's and tuned vLLM's after their tails' ends, the router's beside its line where it
        # parts from the change's, voice alone's in the space left of its line.
        for point in (data.CHANGE, data.TUNED):
            name, points = lines[point]
            end = points[-1]
            s.text(end[0] + 10, end[1] + 6, name, size=NOTE, halo=True)
        router = next(c for c in configurations if c.point == data.ROUTER)
        at = x(_quantile(router.samples(), 0.88))
        s.text(at + 12, y(0.88) + 6, lines[data.ROUTER][0], size=NOTE, halo=True)
        alone = next(c for c in configurations if c.point == data.VOICE_ALONE)
        at = x(_quantile(alone.samples(), 0.8))
        s.text(at - 12, y(0.8) + 6, lines[data.VOICE_ALONE][0], size=NOTE, anchor="end", halo=True)
    s.height = last + 14
    return s.save(path)


def _quantile(samples: Sequence[float], share: float) -> float:
    """The smallest sample at or above ``share`` of the samples: where the ECDF reaches it."""
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))]


# What the agents pay ----------------------------------------------------------------------------------------------

AGENT_PANELS = (
    ("(a) Agents' TTFT p99", "seconds \N{BULLET} lower is better"),
    ("(b) Steps held", "of the engine's steps"),
    ("(c) Preemptions", "make-room's own count"),
)
"""Each panel's title and its axis's title."""


def _agents_description(runs: dict[str, list[data.AgentRun]]) -> str:
    rows = []
    for point, cells in runs.items():
        name, note = data.label(point)
        p99s = [r.ttft_p99 / 1000 for r in cells]
        shares = [r.held_share for r in cells if r.held_share is not None]
        preemptions = [r.preemptions for r in cells if r.preemptions is not None]
        text = (
            f"{name} ({note}, {_runs_of_count(len(cells))}): TTFT p99 {statistics.median(p99s):.0f} s "
            f"({min(p99s):.0f} to {max(p99s):.0f})"
        )
        text += (
            f", agent prefill held in {statistics.median(shares):.0%} of the steps"
            if shares
            else ", no hold"
        )
        text += (
            f", {statistics.median(preemptions):.0f} preemptions by make-room"
            if preemptions
            else ", no make-room"
        )
        rows.append(text)
    return (
        "For each configuration beside 24 agent sessions, each run and the median of the agents' TTFT p99, of the "
        "share of the engine's steps over the calls in which the prefill hold kept agent prefill out, and of "
        "make-room's own preemptions over the calls: " + "; ".join(rows) + "."
    )


def _agent_row(
    s: Chart,
    values: Sequence[float],
    scale: Scale,
    y: float,
    point: str,
    value: str,
) -> None:
    """A configuration's runs at ``y`` as a rug under their median's marker, the median's ``value`` after them."""
    hue, shape = HEADLINE_ROWS[point]
    xs = [scale(v) for v in values]
    rug(s, xs, y, hue, shape, median=scale(statistics.median(values)), half=RUN_HALF)
    s.text(max(xs) + 12, y + 6, value, size=NOTE, halo=True)


def agents(out: Path) -> list[Path]:
    """report-agents.svg and its phone layout: what the agents pay, run by run, for the headline's configurations that
    run agents, in its rows. (a) The agents' TTFT p99. (b) The share of the engine's steps in which the prefill hold
    kept agent prefill out. (c) make-room's own preemptions. Each run a tick, the median a filled marker with its
    value."""
    runs = data.agent_runs()
    description = _agents_description(runs)
    return [
        _agents_wide(runs, out / "report-agents.svg", description),
        _agents_narrow(runs, out / "report-agents-narrow.svg", description),
    ]


AGENT_TICKS = (
    ([0, 100, 200], ["0", "100", "200"]),
    ([0, 0.25, 0.5], [pct(v) for v in (0, 0.25, 0.5)]),
    ([0, 25, 50], ["0", "25", "50"]),
)
AGENT_TOPS = (250, 0.5, 50)
"""Each panel's scale's end: the slowest run's TTFT p99 (230 s), the most steps held (38%), the most preemptions
(46)."""


def _agent_cell(s: Chart, k: int, scale: Scale, y: float, point: str, cells) -> None:
    """Panel ``k``'s mark of a configuration's runs at ``y``, or what it has not."""
    if k == 0:
        p99s = [r.ttft_p99 / 1000 for r in cells]
        _agent_row(s, p99s, scale, y, point, f"{statistics.median(p99s):.0f} s")
    elif k == 1:
        shares = [r.held_share for r in cells if r.held_share is not None]
        if shares:
            _agent_row(s, shares, scale, y, point, pct(statistics.median(shares)))
        else:
            s.text(scale.start + 10, y + 6, "no hold", size=NOTE, cls="mute", halo=True)
    else:
        preemptions = [r.preemptions for r in cells if r.preemptions is not None]
        if preemptions:
            _agent_row(s, preemptions, scale, y, point, f"{statistics.median(preemptions):.0f}")
        else:
            s.text(scale.start + 10, y + 6, "no make-room", size=NOTE, cls="mute", halo=True)


def _agents_wide(runs: dict[str, list[data.AgentRun]], path: Path, description: str) -> Path:
    s = Chart(WIDTH, 440, "What the agents pay", description)
    right = 276
    scales = [Scale(0, top, x, x + 220) for top, x in zip(AGENT_TOPS, (300, 580, 860), strict=True)]
    top, pitch = 52, 60
    ys = [top + 32 + pitch * k for k in range(len(runs))]
    bottom = ys[-1] + 32
    last = 0.0
    for (title, axis_title), scale, (marks, labels) in zip(
        AGENT_PANELS, scales, AGENT_TICKS, strict=True
    ):
        s.text(scale.start, 30, title, size=LABEL)
        last = x_axis(s, scale, marks, labels, top, bottom, title=axis_title)
    for y, (point, cells) in zip(ys, runs.items(), strict=True):
        name, note = data.label(point)
        s.text(right, y - 4, name, size=VALUE, anchor="end")
        s.text(
            right,
            y + 19,
            f"{note}, {_runs_of_count(len(cells))}",
            size=NOTE,
            anchor="end",
            cls="soft",
        )
        for k, scale in enumerate(scales):
            _agent_cell(s, k, scale, y, point, cells)
    s.height = last + 14
    return s.save(path)


def _agents_narrow(runs: dict[str, list[data.AgentRun]], path: Path, description: str) -> Path:
    """The phone layout: the three panels one above the other, each configuration's name over its runs."""
    s = Chart(NARROW, 2200, "What the agents pay", description)
    x0 = 18
    last = 0.0
    for k, ((title, axis_title), top_value, (marks, labels)) in enumerate(
        zip(AGENT_PANELS, AGENT_TOPS, AGENT_TICKS, strict=True)
    ):
        scale = Scale(0, top_value, x0, NARROW - 70)
        y = last + (48 if k else 30)
        s.text(x0, y, title, size=LABEL)
        y += 10
        for point, cells in runs.items():
            name, note = data.label(point)
            detail = f"{note}, {_runs_of_count(len(cells))}"
            band = y + 50 + _row_label(s, x0, y + 20, name, detail, NARROW - x0)
            band_grid(s, scale, marks, band - 18, band + 18)
            _agent_cell(s, k, scale, band, point, cells)
            y = band + 22
        last = x_axis(s, scale, marks, labels, None, y)
        last += 25
        s.text(x0, last, axis_title, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


# Every configuration ---------------------------------------------------------------------------------------------

FAMILY_MARKS = {
    "tuned": ("gray", "circle", True),
    "operator": ("gray", "diamond", True),
    "engine": ("gray", "cross", True),
    "router": ("gray", "triangle", True),
    "resubmit": ("gray", "square", True),
    "lifecycle": ("gray", "diamond", False),
    "change": ("blue", "circle", True),
    "lifecycle-change": ("sky", "diamond", False),
}
"""Each family's marker (data.FAMILIES): its hue, its shape and whether it is filled. Every configuration's row is
labelled as data.LABELS names it."""
SIBLINGS = {"9-pr-a", "9-pr-ab"}
"""The change's configurations other than the one to deploy, in the pale blue: the KV preemption alone, and the change
at its 16-token window."""
FRONTIER_HEADINGS = (
    ("vLLM tuned by its own settings", ("tuned",)),
    ("The operator's static answers", ("operator",)),
    ("What engines ship", ("engine",)),
    ("Routers", ("router", "resubmit")),
    ("Timed by the conversation", ("lifecycle", "lifecycle-change")),
    ("The change", ("change",)),
)
"""The figure's headings, each over the families (data.FAMILIES) whose configurations it names."""
FRONTIER_TOKENS = (3000, 6000)
"""Panel b's scale, past the most any configuration did (5,584 tokens a second)."""
FRONTIER_MARK = 5.6


def _frontier_hue(point: data.FrontierPoint) -> str:
    return "sky" if point.arm in SIBLINGS else FAMILY_MARKS[point.family][0]


def _frontier_description(
    points: Sequence[data.FrontierPoint], slo: float, calls: bool = False
) -> str:
    over = "the calls" if calls else "the calls' first 240 s"
    parts = []
    for p in points:
        name, note = data.label(p.point)
        p95 = (
            f"at least {REPLY_TIMEOUT_MS / 1000:.0f} s, {p.failed_turns} turns failed"
            if p.failed_turns
            else (f"{seconds(p.p95)} s")
        )
        tokens = p.calls_tokens if calls else p.tokens
        parts.append(f"{data.in_prose(name)}, {note}, {p95} at {tokens:,.0f} tokens/s")
    return (
        f"Each of {len(points)} configurations beside 24 agent sessions of the traces with the largest contexts, a row "
        "under its family, in panel a its voice-to-voice p95 on a log scale against the SLO of "
        f"{seconds(slo)} s, the median of its runs and their range, and in panel b the agents' tokens a second over "
        f"{over}, only the engine change and preempt-resubmit coming "
        "near the SLO: " + "; ".join(parts) + "."
    )


def frontier(out: Path) -> list[Path]:
    """report-frontier.svg and its phone layout: every configuration of W4's discriminating scenario but arm 9's
    ablations, a row under its family's heading. (a) Its V2V p95 on a log scale, the median of its runs with a thin
    bar over their range, its marker by its family, against the SLO. (b) The agents' tokens a second over the calls'
    first 240 s, the median of its runs."""
    points = data.frontier()
    slo = data.slo_ms(data.w4_rows())
    description = _frontier_description(points, slo)
    return [
        _frontier_layout(points, slo, out / "report-frontier.svg", description, WIDTH),
        _frontier_layout(points, slo, out / "report-frontier-narrow.svg", description, NARROW),
    ]


def deployable(out: Path) -> list[Path]:
    """report-deployable.svg and its phone layout: Figure 11 for W4's configurations an operator could deploy
    (data.DEPLOYABLE), each with all its runs, its panel b the agents' tokens a second over the whole calls, as
    docs/METHODS.md's table beside it gives them."""
    points = [p for p in data.frontier() if p.point in data.DEPLOYABLE]
    if {p.point for p in points} != set(data.DEPLOYABLE):
        raise ValueError(
            f"W4's records lack {sorted(set(data.DEPLOYABLE) - {p.point for p in points})}"
        )
    slo = data.slo_ms(data.w4_rows())
    description = _frontier_description(points, slo, calls=True)
    title = "What an operator can deploy, beside 24 agent sessions"
    return [
        _frontier_layout(
            points, slo, out / "report-deployable.svg", description, WIDTH, title, calls=True
        ),
        _frontier_layout(
            points,
            slo,
            out / "report-deployable-narrow.svg",
            description,
            NARROW,
            title,
            calls=True,
        ),
    ]


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


def _frontier_p95(s: Chart, a: Scale, y: float, p: data.FrontierPoint) -> None:
    """Panel a's marks of a configuration: its runs' range and its median p95 (at the reply timeout where turns
    failed), with its value right of them, or left where it would pass the scale's end."""
    _, shape, filled = FAMILY_MARKS[p.family]
    hue = _frontier_hue(p)
    p95 = min(p.p95, REPLY_TIMEOUT_MS)
    low, high = min(p.p95s), min(max(p.p95s), REPLY_TIMEOUT_MS)
    if len(p.p95s) > 1:
        s.path(f"M{fmt(a(low))} {fmt(y)}H{fmt(a(high))}", f"{hue} range")
    s.mark(a(p95), y, hue, shape, filled=filled, r=FRONTIER_MARK, name=p.point)
    text = (
        f"\N{GREATER-THAN OR EQUAL TO}{p95 / 1000:.0f} s"
        if p.failed_turns
        else f"{seconds(p.p95)} s"
    )
    right = max(a(high), a(p95) + FRONTIER_MARK) + 10
    if right + measure(text, NOTE) <= a.end + 36:
        s.text(right, y + 6, text, size=NOTE, halo=True)
    else:
        s.text(
            min(a(low), a(p95) - FRONTIER_MARK) - 10,
            y + 6,
            text,
            size=NOTE,
            anchor="end",
            halo=True,
        )


def _frontier_tokens(
    s: Chart, b: Scale, y: float, p: data.FrontierPoint, calls: bool = False
) -> None:
    """Panel b's mark of a configuration: its median tokens a second, over the calls' first 240 s or over the whole
    calls, with its value."""
    _, shape, filled = FAMILY_MARKS[p.family]
    tokens = p.calls_tokens if calls else p.tokens
    s.mark(
        b(tokens),
        y,
        _frontier_hue(p),
        shape,
        filled=filled,
        r=FRONTIER_MARK,
        name=f"{p.point} tokens",
    )
    s.text(b(tokens) + 12, y + 6, f"{tokens:,.0f}", size=NOTE, halo=True)


FRONTIER_TICKS = [1500, 2000, 3000, 5000, 10000, 20000, 30000]
"""Panel a's log axis of p95, from 1.5 s, under every configuration's fastest run, to the reply timeout."""


def _frontier_layout(
    points: Sequence[data.FrontierPoint],
    slo: float,
    path: Path,
    description: str,
    width: float,
    title: str = "Every configuration beside 24 agent sessions",
    *,
    calls: bool = False,
) -> Path:
    """Figure 11's layout, its panel b over the calls' first 240 s, or with ``calls`` over the whole calls."""
    wide = width == WIDTH
    s = Chart(width, 2400, title, description)
    b_title = (
        "over the calls" if calls else "first 240 s of the calls"
    ) + " \N{BULLET} higher is better"
    rows = _frontier_rows(points)
    if wide:
        right = 330
        a = Scale(1500, 30000, 352, 760, log=True)
        b = Scale(*FRONTIER_TOKENS, 830, 1040)
        heads = (a.start, b.start)
        s.text(heads[0], 30, "(a) Voice-to-voice p95 (s, log scale)", size=LABEL)
        s.text(heads[1], 30, "(b) Agents' tokens/s", size=LABEL)
        top = 56
    else:
        # The scales start in from the canvas's edge, so that their first ticks' labels stay on it.
        a = Scale(1500, 30000, 34, width - 64, log=True)
        b = Scale(*FRONTIER_TOKENS, 34, width - 64)
    marks_b, labels_b = linear_ticks(FRONTIER_TOKENS[1], 1000)
    marks_b, labels_b = marks_b[3:], labels_b[3:]
    labels_a = [f"{t / 1000:g}" for t in FRONTIER_TICKS]
    if wide:
        y = top + 14
        placed = []
        for heading, members in rows:
            s.text(24, y + 16, heading, size=NOTE, cls="mute")
            y += 26
            for p in members:
                name, note = data.label(p.point)[0], _frontier_note(p)
                s.text(right, y + 14, name, size=VALUE, anchor="end")
                s.text(right, y + 36, note, size=NOTE, anchor="end", cls="soft")
                placed.append((y + 18, p))
                y += 50
        bottom = y + 2
        last = x_axis(s, a, FRONTIER_TICKS, labels_a, top, bottom, title="lower is better")
        x_axis(
            s,
            b,
            marks_b,
            labels_b,
            top,
            bottom,
            title=b_title,
        )
        s.vrule(a(slo), top, bottom, "slo")
        s.text(
            a(slo),
            top - 6,
            f"SLO {seconds(slo)} s",
            size=NOTE,
            anchor="middle",
            cls="orange tint",
            halo=True,
        )
        for row_y, p in placed:
            _frontier_p95(s, a, row_y, p)
            _frontier_tokens(s, b, row_y, p, calls)
        s.height = last + 14
        return s.save(path)
    # The phone layout: panel a, then panel b, each configuration's name over its marks under its heading.
    last = 0.0
    for k, (head, scale, ticks, labels, axis_title) in enumerate(
        (
            (
                "(a) Voice-to-voice p95 (s, log scale)",
                a,
                FRONTIER_TICKS,
                labels_a,
                "lower is better",
            ),
            ("(b) Agents' tokens/s", b, marks_b, labels_b, b_title),
        )
    ):
        y = last + (48 if k else 30)
        s.text(18, y, head, size=LABEL)
        if k == 0:
            y += 26
            s.text(
                a(slo), y, f"SLO {seconds(slo)} s", size=NOTE, anchor="middle", cls="orange tint"
            )
        y += 8
        for heading, members in rows:
            s.text(18, y + 18, heading, size=NOTE, cls="mute")
            y += 26
            for p in members:
                name, note = data.label(p.point)[0], _frontier_note(p)
                band = y + 46 + _row_label(s, 18, y + 20, name, note, width - 18)
                band_grid(s, scale, ticks, band - 14, band + 14)
                if k == 0:
                    s.vrule(a(slo), band - 14, band + 14, "slo")
                    _frontier_p95(s, a, band, p)
                else:
                    _frontier_tokens(s, b, band, p, calls)
                y = band + 20
        last = x_axis(s, scale, ticks, labels, None, y)
        last += 25
        s.text(18, last, axis_title, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


def _pair_key(s: Chart, x: float, baseline: float, width: float, *, filled: bool = False) -> float:
    """The key of a figure that sets tuned vLLM beside the configuration to deploy: each one's marker, in its hue, and
    its name, in rows that break where ``width`` runs out. Returns the last row's baseline."""
    entries = [
        (mark_key(hue, filled=filled), data.label(key)[0])
        for key, hue in (("tuned", "gray"), ("deploy", "blue"))
    ]
    return legend(s, x, baseline, entries, width=width)


# Where the pool has room, and as calls overlap --------------------------------------------------------------------


def _margin_description(
    rounds: list[tuple[int, float, float]], calls: list[data.Calls], slo: float
) -> str:
    paired = ", ".join(f"{seconds(t)} and {seconds(c)} s" for _, t, c in rounds)
    counts = "; ".join(
        f"{c.callers} callers, {statistics.median(c.over):+,.0f} ms over their SLO of {seconds(c.slo)} s at the "
        f"median of {len(c.over)} runs ({min(c.over):+,.0f} to {max(c.over):+,.0f}), the hold in "
        f"{statistics.median(c.held):.0%} of the steps, the agents at {statistics.median(c.tokens):,.0f} tokens/s"
        for c in calls
    )
    return (
        "In panel a, with 16 agent sessions of the whole corpus, where the pool has room, tuned vLLM's V2V p95 and "
        f"the change's in each of five rounds against the SLO of {seconds(slo)} s ({paired}); in panels b to d, the "
        "change with two, four and six calls beside 24 agent sessions, on the same three samples of the agents, each "
        "run's p95 over that many callers' SLO, "
        "the share of the engine's steps in which the prefill hold held agent prefill, and the agents' tokens a "
        f"second over the calls' first 240 s ({counts})."
    )


MARGIN_FLOOR = 1500
"""Where Figure 12a's axis of p95 starts, in ms: its dumbbells encode position, and every round's two p95s lie between
2.0 and 3.9 s, so an axis from zero would leave half the panel empty and squeeze the gaps at the SLO."""


def margin(out: Path) -> list[Path]:
    """report-margin.svg: where the pool has room, and as calls overlap. (a) Where the pool has room: the standard
    scenario's rounds, tuned vLLM's p95 and the change's, joined per round. (b) to (d) As calls overlap: with 2, 4 and
    6 calls on the same three samples of the agents, the change's p95 over that many callers' SLO, the share of steps
    the hold held agent prefill in, and the agents' tokens a second; each run an open marker, the median a tick."""
    rows = data.w5_rows()
    rounds = data.standard_rounds(rows)
    calls = data.calls(rows)
    slo = data.slo_ms(rows)
    description = _margin_description(rounds, calls, slo)
    return [
        _margin_wide(rounds, calls, slo, out / "report-margin.svg", description),
        _margin_narrow(rounds, calls, slo, out / "report-margin-narrow.svg", description),
    ]


MARGIN_PANELS = (
    ("(b) p95 over the SLO (ms)", (0, 400), [0, 200, 400], "over", "{:+,.0f}"),
    ("(c) Steps holding agent prefill", (0, 1), [0, 0.5, 1], "held", "{:.0%}"),
    ("(d) Agents' tokens/s, first 240 s", (0, 4000), [0, 2000, 4000], "tokens", "{:,.0f}"),
)
"""Panels b to d: each one's title, its scale's range, its ticks, the field it draws and its values' format."""


def _margin_round(s: Chart, a_x: Scale, a_y: Scale, rounds, slo: float) -> None:
    """Panel a's marks: the SLO's rule, and each round's two p95s joined, the key over the panel naming them."""
    s.rule(a_x.start, a_x.end, a_y(slo), "slo")
    s.text(
        a_x.start + 6,
        a_y(slo) + 22,
        f"SLO {seconds(slo)} s",
        size=NOTE,
        cls="orange tint",
        halo=True,
    )
    for k, tuned, change in rounds:
        s.trace([(a_x(k), a_y(tuned)), (a_x(k), a_y(change))], "hair", wire=False)
        s.mark(a_x(k), a_y(tuned), "gray")
        s.mark(a_x(k), a_y(change), "blue")


def _margin_calls(s: Chart, calls, x: Scale, y: Scale, field: str, text: str) -> None:
    """A panel of b to d: each caller count's runs and their median, with its value; in b each count's SLO."""
    if field == "over":
        s.rule(x.start, x.end, y(0), "slo")
    for c in calls:
        values = list(getattr(c, field))
        middle = statistics.median(values)
        column_rug(s, x(c.callers), [y(v) for v in values], "blue", median=y(middle))
        s.text(x(c.callers) + 18, y(middle) + 6, text.format(middle), size=NOTE, halo=True)
        if field == "over":
            s.text(
                x(c.callers) + 18,
                y(0) - 7,
                f"SLO {seconds(c.slo)} s",
                size=NOTE,
                cls="orange tint",
                halo=True,
            )


def _margin_narrow(rounds, calls, slo: float, path: Path, description: str) -> Path:
    """The phone layout: panel a, then b to d one above the other on the same axis of callers."""
    s = Chart(NARROW, 1600, "Where the pool has room, and as calls overlap", description)
    x0 = 18
    s.text(x0, 30, "(a) Where the pool has room", size=LABEL)
    s.text(x0, 56, "16 agent sessions of the whole corpus", size=NOTE, cls="soft")
    key = _pair_key(s, x0, 86, NARROW - 2 * x0)
    s.text(x0, key + 40, "V2V p95 (s)", size=NOTE, cls="soft")
    a_x = Scale(0.5, len(rounds) + 0.5, 64, NARROW - 24)
    a_y = Scale(MARGIN_FLOOR, 4000, key + 344, key + 74)
    marks = [m / 2 for m in range(int(MARGIN_FLOOR / 500), 9)]
    y_axis(s, a_y, [m * 1000 for m in marks], [f"{m:g}" for m in marks], a_x.start, a_x.end)
    last = x_axis(
        s,
        a_x,
        [r for r, _, _ in rounds],
        [str(r) for r, _, _ in rounds],
        None,
        a_y.start,
        title="Round",
    )
    _margin_round(s, a_x, a_y, rounds, slo)
    x = Scale(1, 7.8, 54, NARROW - 30)  # room right of six for its labels
    callers = [c.callers for c in calls]
    y = last + 30
    for title, (lo, hi), ticks, field, text in MARGIN_PANELS:
        y += 40
        s.text(x0, y, title, size=LABEL)
        scale = Scale(lo, hi, y + 140, y + 36)
        labels = [f"{t:.0%}" if field == "held" else f"{t:,.0f}" for t in ticks]
        y_axis(s, scale, ticks, labels, x.start, x.end, axis=field != "over")
        _margin_calls(s, calls, x, scale, field, text)
        y = scale.start
    last = x_axis(
        s, x, callers, [str(n) for n in callers], None, y, title="Calls beside 24 agent sessions"
    )
    s.height = last + 14
    return s.save(path)


def _margin_wide(rounds, calls, slo: float, path: Path, description: str) -> Path:
    s = Chart(
        WIDTH,
        620,
        "Where the pool has room, and as calls overlap",
        description,
    )

    # (a) The standard scenario, round by round.
    a_x = Scale(0.5, len(rounds) + 0.5, 96, 440)
    a_y = Scale(MARGIN_FLOOR, 4000, 548, 136)
    s.text(24, 30, "(a) Where the pool has room", size=LABEL)
    s.text(24, 54, "16 agent sessions of the whole corpus", size=NOTE, cls="soft")
    _pair_key(s, 24, 82, 416)
    s.text(24, 110, "V2V p95 (s)", size=NOTE, cls="soft")
    marks = [m / 2 for m in range(int(MARGIN_FLOOR / 500), 9)]
    y_axis(s, a_y, [m * 1000 for m in marks], [f"{m:g}" for m in marks], a_x.start, a_x.end)
    x_axis(
        s,
        a_x,
        [r for r, _, _ in rounds],
        [str(r) for r, _, _ in rounds],
        None,
        a_y.start,
        title="Round",
    )
    _margin_round(s, a_x, a_y, rounds, slo)

    # (b) to (d) As calls overlap, on one axis of callers, with room right of six for its labels.
    x = Scale(1, 7.5, 640, 1100)
    callers = [c.callers for c in calls]
    panels = [
        (title, Scale(lo, hi, start, start - 110), ticks, field, text)
        for (title, (lo, hi), ticks, field, text), start in zip(
            MARGIN_PANELS, (220, 384, 548), strict=True
        )
    ]
    for title, y, ticks, field, text in panels:
        s.text(560, y.end - 22, title, size=LABEL)
        labels = [f"{t:.0%}" if field == "held" else f"{t:,.0f}" for t in ticks]
        y_axis(s, y, ticks, labels, x.start, x.end, axis=field != "over")
        _margin_calls(s, calls, x, y, field, text)
    last = x_axis(
        s,
        x,
        callers,
        [str(n) for n in callers],
        None,
        panels[-1][1].start,
        title="Calls beside 24 agent sessions",
    )
    s.height = last + 14
    return s.save(path)


# The A100 ---------------------------------------------------------------------------------------------------------


def _a100_description(
    pairs: dict[int, tuple[data.Configuration, data.Configuration]], slo: float
) -> str:
    tuned, change = pairs[512]
    by_round = ", ".join(
        f"{seconds(t.p95)} and {seconds(c.p95)} s"
        for t, c in zip(tuned.runs, change.runs, strict=True)
    )
    caps = "; ".join(
        f"at {cap:,}-token steps tuned vLLM {seconds(t.p95)} s at {t.tokens:,.0f} tokens/s, the change "
        f"{seconds(c.p95)} s at {c.tokens:,.0f} tokens/s, medians of {len(t.runs)} runs each"
        for cap, (t, c) in pairs.items()
    )
    return (
        "On the A100 beside 24 agent sessions: in panel a, at the RTX 5060 Ti's 512-token steps, tuned vLLM's V2V "
        f"p95 and the change's in each of five rounds against the card's SLO of {seconds(slo)} s ({by_round}); in "
        "panel b, the agents' tokens a second over the calls' first 240 s against the V2V p95, each run and the "
        f"medians, an arrow from tuned vLLM's median to the change's at the same step cap ({caps})."
    )


def a100(out: Path) -> list[Path]:
    """report-a100.svg: the A100. (a) At 512-token steps, tuned vLLM's p95 and the change's by round, joined per
    round, against the card's SLO. (b) The trade-off at both step caps: the agents' tokens a second against the
    p95, each run open, the medians filled, an arrow from tuned vLLM's median to the change's."""
    rows = data.a100_rows()
    pairs = data.a100(rows)
    slo = data.slo_ms(rows)
    description = _a100_description(pairs, slo)
    return [
        _a100_layout(pairs, slo, out / "report-a100.svg", description, WIDTH),
        _a100_layout(pairs, slo, out / "report-a100-narrow.svg", description, NARROW),
    ]


def _a100_layout(pairs, slo: float, path: Path, description: str, width: float) -> Path:
    """Panels a and b side by side, or on a phone one above the other."""
    wide = width == WIDTH
    s = Chart(width, 1000, "The rules on an A100", description)
    y_top = 6000

    # (a) By round, at 512-token steps.
    tuned, change = pairs[512]
    x0 = 24 if wide else 18
    s.text(x0, 30, "(a) Each round, 512-token steps", size=LABEL)
    # The key under the title, and on a phone, where it takes two rows, the plot under it.
    shift = max(0.0, _pair_key(s, x0, 58, 452 if wide else width - 2 * x0) + 26 - 84)
    s.text(x0, 84 + shift, "V2V p95 (s)", size=NOTE, cls="soft")
    a_x = Scale(0.5, len(tuned.runs) + 0.5, 96, 400) if wide else Scale(0.5, 5.5, 64, width - 24)
    y = (
        Scale(0, y_top, 450 + shift, 104 + shift)
        if wide
        else Scale(0, y_top, 400 + shift, 104 + shift)
    )
    marks, labels = linear_ticks(y_top / 1000, 1)
    y_axis(s, y, [m * 1000 for m in marks], labels, a_x.start, a_x.end)
    rounds = [run.round for run in tuned.runs]
    a_bottom = x_axis(s, a_x, rounds, [str(r) for r in rounds], None, y.start, title="Round")
    s.rule(a_x.start, a_x.end, y(slo), "slo")
    s.text(
        a_x.start + 6, y(slo) + 22, f"SLO {seconds(slo)} s", size=NOTE, cls="orange tint", halo=True
    )
    for t, c in zip(tuned.runs, change.runs, strict=True):
        if t.round != c.round:
            raise ValueError(
                f"round {t.round} of tuned vLLM pairs with round {c.round} of the change"
            )
        s.trace([(a_x(t.round), y(t.p95)), (a_x(c.round), y(c.p95))], "hair", wire=False)
        s.mark(a_x(t.round), y(t.p95), "gray")
        s.mark(a_x(c.round), y(c.p95), "blue")

    # (b) The trade-off at both step caps: beside (a), or on a phone under it, on the same scale of p95.
    if wide:
        b_x = Scale(0, 16000, 580, 1040)
        s.text(500, 30, "(b) The agents' throughput against the voice tail", size=LABEL)
    else:
        b_x = Scale(0, 16000, 64, width - 32)  # its last tick's label on the canvas
        lines = wrap("(b) The agents' throughput against the voice tail", width - 36, LABEL)
        for k, line in enumerate(lines):
            s.text(18, a_bottom + 56 + 26 * k, line, size=LABEL)
        # Its own key under its title: on a phone it sits far under panel a's.
        top = _pair_key(s, 18, a_bottom + 56 + 26 * len(lines) + 4, width - 36) + 34
        s.text(18, top - 4, "V2V p95 (s)", size=NOTE, cls="soft")
        y = Scale(0, y_top, top + 20 + 346, top + 20)
    y_axis(s, y, [m * 1000 for m in marks], labels, b_x.start, b_x.end)
    ticks = [0, 4000, 8000, 12000, 16000]
    for t in ticks[1:]:
        s.vrule(b_x(t), y.end, y.start, "grid")
    bottom = x_axis(
        s,
        b_x,
        ticks,
        [f"{t:,}" for t in ticks],
        None,
        y.start,
        title="Agents' tokens/s, first 240 s of the calls",
    )
    s.rule(b_x.start, b_x.end, y(slo), "slo")
    s.text(
        b_x.start + 6, y(slo) + 22, f"SLO {seconds(slo)} s", size=NOTE, cls="orange tint", halo=True
    )
    for cap, (t, c) in pairs.items():
        for config, hue in ((t, "gray"), (c, "blue")):
            for run in config.runs:
                s.mark(b_x(run.tokens), y(run.p95), hue)
        start, end = (b_x(t.tokens), y(t.p95)), (b_x(c.tokens), y(c.p95))
        s.arrow(start, end, "blue")
        s.mark(*start, "gray", filled=True, r=5.6)
        s.mark(*end, "blue", filled=True, r=5.6)
        centre = statistics.mean(b_x(run.tokens) for run in (*t.runs, *c.runs))
        name = f"{cap:,}-token steps"
        if wide:
            s.text(centre, y(slo) + 40, name, size=NOTE, anchor="middle", halo=True)
        else:  # each under its cluster, kept on the canvas, the second a line lower
            half = measure(name, NOTE) / 2
            at = min(max(centre, b_x.start + half), b_x.end - half)
            s.text(
                at,
                y(slo) + 40 + (22 if cap != min(pairs) else 0),
                name,
                size=NOTE,
                anchor="middle",
                halo=True,
            )
    s.height = bottom + 14
    return s.save(path)


# Two cards --------------------------------------------------------------------------------------------------------

CARD_ENGINES = {
    "stock": (data.label("tuned")[0], "gray"),
    "change": (data.label("deploy")[0], "blue"),
}
"""Each engine's name and hue: the rounds' stock engine runs the frozen box's flags, tuned vLLM, and the change runs
the configuration to deploy."""
CARD_KINDS = {
    "dedicated": "Dedicated roles",
    "consolidation": "Consolidation",
    "pooling": "Pooling",
}
"""Each kind of arrangement's heading over its rows, in the figure's order."""
CARD_DODGE = {"stock": -6.0, "change": 6.0}
"""Each engine's offset from its number of sessions in panel a, so that two shares a percentage point apart, as at
two sessions, stay apart."""
CARD_TOKENS = Scale(0, 15000, 712, 1040)
"""Panel b's agents' tokens a second, past the most any round did (14,352)."""


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
        miss = "" if a.qualifies else ", missing the requirement"
        rows.append(f"{what}, {_card_sessions(a)}: {a.tokens:,.0f}{miss}")
    rounds = len(c.rounds)
    return (
        f"In panel a, the share of the consolidated card's replies over their round's SLO, the {rounds} rounds "
        "pooled, by its agent sessions beside both calls, against the requirement of at most "
        f"{c.requirement:.0%}: at {_cards_and([str(n) for n in sessions])} sessions, {shares}; in panel b, each "
        "arrangement's agents' tokens a second over the calls' first 240 s summed over both cards, at the median of "
        f"the {rounds} rounds, against {c.threshold:,.0f}, {c.hurdle:.0%} over the best dedicated arrangement: "
        + "; ".join(rows)
        + "."
    )


def _cards_key(s: Chart, c: data.TwoCards, x: float, baseline: float) -> None:
    """Panel b's key under its subtitle: a hatched bar is an arrangement that misses the requirement."""
    entry = [(hatch_key(), f"misses the {c.requirement:.0%} requirement")]
    legend(s, x, baseline, entry, width=legend_width(entry))


def _at_end(s: Chart, x: float, baseline: float, label: str, rule: float) -> None:
    """A bar's value at its end, from ``x``, or right of the vertical ``rule`` where it would cross it; on a halo, as
    the gridlines run behind it."""
    width = measure(label, NOTE)
    if x - CLEAR <= rule <= x + width + CLEAR:
        x = rule + CLEAR + 2
    s.text(x, baseline, label, size=NOTE, halo=True)


def cards(out: Path) -> list[Path]:
    """report-cards.svg: two cards. (a) The consolidated card's replies over their round's SLO by its agent sessions
    beside both calls, the rounds pooled, on stock vLLM and with the change, against the requirement. (b) Each
    arrangement's agents' tokens a second over the calls' first 240 s summed over both cards: a bar at the median of
    the rounds and a tick for each round, hatched where the arrangement misses the requirement, against the hurdle
    over the best dedicated arrangement."""
    c = data.two_cards()
    sessions = [cell.sessions for cell in c.consolidated["stock"]]
    if any([cell.sessions for cell in cells] != sessions for cells in c.consolidated.values()):
        raise ValueError("the engines' consolidated cards ran different numbers of sessions")
    description = _cards_description(c, sessions)
    return [
        _cards_wide(c, sessions, out / "report-cards.svg", description),
        _cards_narrow(c, sessions, out / "report-cards-narrow.svg", description),
    ]


def _cards_lines(s: Chart, c: data.TwoCards, a_x: Scale, a_y: Scale) -> None:
    """Panel a's lines: each engine's consolidated card by its sessions, the key over the panel naming them."""
    for engine, cells in c.consolidated.items():
        _, hue = CARD_ENGINES[engine]
        points = [(a_x(k) + CARD_DODGE[engine], a_y(cell.share)) for k, cell in enumerate(cells)]
        s.trace(points, f"{hue} trace")
        for px, py in points:
            s.mark(px, py, hue, filled=True)


def _cards_bar(
    s: Chart, c: data.TwoCards, a: data.Arrangement, b: Scale, y: float, thick: float
) -> None:
    """Panel b's bar of an arrangement: the median of its rounds, each round's tick, the value at its end."""
    _, hue = CARD_ENGINES[a.engine]
    length = b(a.tokens) - b.start
    s.bar(b.start, y - thick / 2, length, thick, hue, hatched=not a.qualifies, name=a.name)
    runs(s, [b(value) for value in a.rounds], y, hue, half=thick / 2 + 5)
    end = max(b(v) for v in (a.tokens, *a.rounds)) + 10
    _at_end(s, end, y + 6, f"{a.tokens:,.0f}", b(c.threshold))


def _cards_narrow(c: data.TwoCards, sessions: Sequence[int], path: Path, description: str) -> Path:
    """The phone layout: panel a over panel b, each arrangement's name over its bar."""
    s = Chart(NARROW, 1600, "What sharing two cards buys", description)
    x0 = 18
    s.text(x0, 30, "(a) Replies over the SLO", size=LABEL)
    s.text(x0, 54, f"the consolidated card, {len(c.rounds)} rounds pooled", size=NOTE, cls="soft")
    shift = _pair_key(s, x0, 80, NARROW - 2 * x0, filled=True) - 65
    a_x = Scale(-0.5, len(sessions) - 0.5, 64, NARROW - 112)
    a_y = Scale(0, 1, 340 + shift, 90 + shift)
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_axis(s, a_y, marks, [pct(m) for m in marks], a_x.start, a_x.end)
    last = x_axis(
        s,
        a_x,
        list(range(len(sessions))),
        [str(n) for n in sessions],
        None,
        a_y.start,
        title="Agent sessions beside both calls",
    )
    s.rule(a_x.start, a_x.end, a_y(c.requirement), "slo")
    s.text(
        a_x.end + 10,
        a_y(c.requirement) + 6,
        f"at most {c.requirement:.0%}",
        size=NOTE,
        cls="orange tint",
    )
    _cards_lines(s, c, a_x, a_y)
    # (b) Each arrangement's agents' tokens a second, against the hurdle, under its kind's heading.
    y = last + 56
    s.text(x0, y, "(b) Agents' tokens/s", size=LABEL)
    s.text(
        x0,
        y + 24,
        f"summed over both cards, median of {len(c.rounds)} rounds",
        size=NOTE,
        cls="soft",
    )
    _cards_key(s, c, x0, y + 50)
    b = Scale(CARD_TOKENS.lo, CARD_TOKENS.hi, x0, NARROW - 80)
    top = y + 96
    y = top + 4
    thick = 18
    ticks, labels = linear_ticks(b.hi, 5000)
    for kind, heading in CARD_KINDS.items():
        members = [a for a in c.arrangements if a.kind == kind]
        if not members:
            continue
        s.text(x0, y + 18, heading, size=NOTE, cls="mute")
        y += 26
        for a in members:
            band = y + 46 + _row_label(s, x0, y + 20, _card_row(a), _card_sessions(a), NARROW - x0)
            band_grid(s, b, ticks, band - 16, band + 16)
            _cards_bar(s, c, a, b, band, thick)
            # The hurdle's rule across each band, over its bar as in the wide layout, clear of the names over them.
            s.vrule(b(c.threshold), band - 16, band + 16, "slo")
            y = band + 22
    bottom = y + 4
    hurdle = b(c.threshold)
    s.text(
        hurdle,
        top - 10,
        f"{c.hurdle:.0%} over the best dedicated, {c.threshold:,.0f}",
        size=NOTE,
        anchor="middle",
        cls="orange tint",
    )
    last = x_axis(s, b, ticks, labels, None, bottom)
    for line in wrap(TOKENS_TITLE, NARROW - 2 * x0, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


def _cards_wide(c: data.TwoCards, sessions: Sequence[int], path: Path, description: str) -> Path:
    s = Chart(WIDTH, 660, "What sharing two cards buys", description)
    b = CARD_TOKENS
    left, right = 498, b.start - 16  # panel b's titles and headings, and its rows' names
    top, heading, pitch, thick = 92, 28, 50, 20

    # (b)'s rows, under their kinds' headings, which set the plot's height: a name that would pass the headings' edge
    # breaks over two lines, its row a line taller.
    rows = []
    y = top + 4
    for kind, name in CARD_KINDS.items():
        members = [a for a in c.arrangements if a.kind == kind]
        if not members:
            continue
        y += 8 if rows else 0
        s.text(left, y + 16, name, size=NOTE, cls="mute")
        y += heading
        for a in members:
            lines = wrap(_card_row(a), right - left, VALUE)
            extra = 22 * (len(lines) - 1)
            rows.append((y + extra + pitch / 2, a, lines))
            y += pitch + extra
    bottom = y + 4

    # (a) The consolidated card's replies over the SLO, on one axis of sessions.
    a_x = Scale(-0.5, len(sessions) - 0.5, 96, 392)
    a_y = Scale(0, 1, bottom, top + 18)
    s.text(24, 30, "(a) Replies over the SLO", size=LABEL)
    s.text(
        24,
        54,
        f"the consolidated card, {len(c.rounds)} rounds pooled",
        size=NOTE,
        cls="soft",
    )
    _pair_key(s, 24, top - 10, left - 48, filled=True)
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_axis(s, a_y, marks, [pct(m) for m in marks], a_x.start, a_x.end)
    last = x_axis(
        s,
        a_x,
        list(range(len(sessions))),
        [str(n) for n in sessions],
        None,
        a_y.start,
        title="Agent sessions beside both calls",
    )
    s.rule(a_x.start, a_x.end, a_y(c.requirement), "slo")
    s.text(
        a_x.end + 10,
        a_y(c.requirement) + 6,
        f"at most {c.requirement:.0%}",
        size=NOTE,
        cls="orange tint",
    )
    _cards_lines(s, c, a_x, a_y)

    # (b) Each arrangement's agents' tokens a second, against the hurdle: its bar the median of the rounds, over
    # each round's tick, which reaches past the bar so that it shows on the hatching too.
    s.text(left, 30, "(b) Agents' tokens/s", size=LABEL)
    s.text(
        left,
        54,
        f"summed over both cards, median of {len(c.rounds)} rounds",
        size=NOTE,
        cls="soft",
    )
    _cards_key(s, c, left, top - 10)
    ticks, labels = linear_ticks(b.hi, 5000)
    x_axis(s, b, ticks, labels, top, bottom, title=TOKENS_TITLE)
    for y, a, lines in rows:
        for k, line in enumerate(lines):
            s.text(right, y - 3 - 22 * (len(lines) - 1 - k), line, size=VALUE, anchor="end")
        s.text(right, y + 19, _card_sessions(a), size=NOTE, anchor="end", cls="soft")
        _cards_bar(s, c, a, b, y, thick)
    hurdle = b(c.threshold)
    s.vrule(hurdle, top, bottom, "slo")
    s.text(
        hurdle,
        top - 10,
        f"{c.hurdle:.0%} over the best dedicated, {c.threshold:,.0f}",
        size=NOTE,
        anchor="middle",
        cls="orange tint",
    )
    s.height = last + 14
    return s.save(path)


# The step cap's five rounds ----------------------------------------------------------------------------------------

STEP_CAP_ROWS = {
    ("tuned", 512): (data.label("tuned")[0], "512-token steps", "gray", "circle", False),
    ("change", 512): (data.label("deploy")[0], "512-token steps", "sky", "circle", False),
    ("change", 256): (data.label("deploy")[0], "256-token steps", "blue", "circle", True),
}
"""Each configuration of the step cap's comparison by (engine, step cap): its name (data.LABELS), its step cap as the
note under it, its hue, its marker and whether the marker is filled (the deployment's own step cap)."""
REQUIREMENT = 0.05
"""The share of a run's replies the SLO allows over its threshold (docs/SLO.md, the two-card gate's requirement): the
line the step cap's rounds and the soak's hours are read against."""
ALLOWED = f"at most {REQUIREMENT:.0%}"


def _step_cap_name(c: data.StepCap) -> str:
    name, note, _, _, _ = STEP_CAP_ROWS[(c.engine, c.cap)]
    return f"{name}, {note}"


def _step_cap_description(caps: Sequence[data.StepCap]) -> str:
    rows = "; ".join(
        f"{data.in_prose(_step_cap_name(c))}, "
        f"{', '.join(str(r.over) for r in c.rounds)} of {c.rounds[0].replies} by round, {c.over} of {c.replies} "
        f"in all, at {c.tokens:,.0f} agents' tokens a second"
        for c in caps
    )
    return (
        "In panel a, the share of each of five rounds' replies over that round's SLO beside 24 agent sessions, against "
        f"the requirement of {ALLOWED}, and in panel b each round's agents' tokens a second over the calls' first 240 s "
        f"with their median: {rows}."
    )


def step_cap(out: Path) -> list[Path]:
    """report-step-cap.svg and its phone layout: the step cap that ships. (a) Each round's replies over its SLO for
    tuned vLLM and the change at 512-token steps and at 256, against the requirement. (b) The agents' tokens a second
    in each round, and their median."""
    caps = data.step_caps()
    description = _step_cap_description(caps)
    return [
        _step_cap_wide(caps, out / "report-step-cap.svg", description),
        _step_cap_narrow(caps, out / "report-step-cap-narrow.svg", description),
    ]


def _step_cap_rounds(s: Chart, caps: Sequence[data.StepCap], x: Scale, y: Scale) -> None:
    """Panel a's marks: in each round the three configurations' shares over the SLO, joined by a hair."""
    for k in range(len(caps[0].rounds)):
        shares = [c.rounds[k].share for c in caps]
        at = x(caps[0].rounds[k].round)
        s.trace([(at, y(min(shares))), (at, y(max(shares)))], "hair", wire=False)
        for c in caps:
            _, _, hue, shape, filled = STEP_CAP_ROWS[(c.engine, c.cap)]
            s.mark(at, y(c.rounds[k].share), hue, shape, filled=filled)


def _step_cap_keys(caps: Sequence[data.StepCap]) -> list[tuple[Key, str]]:
    """Panel a's key: each configuration's marker and its name with its step cap."""
    entries = []
    for c in caps:
        name, note, hue, shape, filled = STEP_CAP_ROWS[(c.engine, c.cap)]
        entries.append((mark_key(hue, shape, filled=filled), f"{name}, {note}"))
    return entries


def _step_cap_requirement(s: Chart, x: Scale, y: Scale) -> None:
    s.rule(x.start, x.end, y(REQUIREMENT), "slo")
    s.text(x.start + 6, y(REQUIREMENT) - 8, ALLOWED, size=NOTE, cls="orange tint", halo=True)


def _step_cap_tokens(s: Chart, c: data.StepCap, b: Scale, y: float) -> None:
    """A configuration's agents' tokens a second, each round a tick, their median a marker with its value."""
    _, _, hue, shape, _ = STEP_CAP_ROWS[(c.engine, c.cap)]
    xs = [b(r.tokens) for r in c.rounds]
    rug(s, xs, y, hue, shape, median=b(c.tokens), half=RUN_HALF)
    s.text(max(xs) + 12, y + 6, f"{c.tokens:,.0f}", size=NOTE, halo=True)


def _step_cap_wide(caps: Sequence[data.StepCap], path: Path, description: str) -> Path:
    s = Chart(WIDTH, 620, "The step cap that ships", description)
    rounds = [r.round for r in caps[0].rounds]
    s.text(24, 30, "(a) Replies over the SLO, each round", size=LABEL)
    key = legend_columns(s, 24, 62, _step_cap_keys(caps), columns=1, column=0)
    s.text(24, key + 40, "Share of the round's replies", size=NOTE, cls="soft")
    x = Scale(0.5, len(rounds) + 0.5, 96, 420)
    y = Scale(0, 1, key + 416, key + 66)
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_axis(s, y, marks, [pct(m) for m in marks], x.start, x.end)
    x_axis(s, x, rounds, [str(r) for r in rounds], None, y.start, title="Round")
    _step_cap_requirement(s, x, y)
    _step_cap_rounds(s, caps, x, y)
    # (b) The agents' tokens a second, a row a configuration.
    b = Scale(0, 4000, 840, 1040)
    right = b.start - 24
    s.text(660, 30, "(b) Agents' tokens/s", size=LABEL)
    top, pitch = 90, 64
    ys = [top + 34 + pitch * k for k in range(len(caps))]
    ticks, labels = linear_ticks(4000, 2000)
    last = x_axis(s, b, ticks, labels, top, ys[-1] + 34, title=TOKENS_TITLE)
    for row_y, c in zip(ys, caps, strict=True):
        name, note, _, _, _ = STEP_CAP_ROWS[(c.engine, c.cap)]
        s.text(right, row_y - 4, name, size=VALUE, anchor="end")
        s.text(right, row_y + 19, note, size=NOTE, anchor="end", cls="soft")
        _step_cap_tokens(s, c, b, row_y)
    s.height = max(last, y.start + 50) + 14
    return s.save(path)


def _step_cap_narrow(caps: Sequence[data.StepCap], path: Path, description: str) -> Path:
    s = Chart(NARROW, 1000, "The step cap that ships", description)
    x0 = 18
    rounds = [r.round for r in caps[0].rounds]
    s.text(x0, 30, "(a) Replies over the SLO, each round", size=LABEL)
    baseline = legend_columns(s, x0, 62, _step_cap_keys(caps), columns=1, column=0)
    s.text(x0, baseline + 40, "Share of the round's replies", size=NOTE, cls="soft")
    x = Scale(0.5, len(rounds) + 0.5, 64, NARROW - 24)
    y = Scale(0, 1, baseline + 360, baseline + 70)
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_axis(s, y, marks, [pct(m) for m in marks], x.start, x.end)
    last = x_axis(s, x, rounds, [str(r) for r in rounds], None, y.start, title="Round")
    _step_cap_requirement(s, x, y)
    _step_cap_rounds(s, caps, x, y)
    # (b) The agents' tokens a second, a band a configuration under its name.
    b = Scale(0, 4000, x0, 300)
    yy = last + 56
    s.text(x0, yy, "(b) Agents' tokens/s", size=LABEL)
    yy += 12
    marks_b = [0, 2000, 4000]
    for c in caps:
        name, note, _, _, _ = STEP_CAP_ROWS[(c.engine, c.cap)]
        band = yy + 40 + _row_label(s, x0, yy + 18, name, note, NARROW - x0, gap=7)
        band_grid(s, b, marks_b, band - 12, band + 12)
        _step_cap_tokens(s, c, b, band)
        yy = band + 16
    last = x_axis(s, b, marks_b, linear_ticks(4000, 2000)[1], None, yy)
    for line in wrap(TOKENS_TITLE, NARROW - 36, NOTE):
        last += 25
        s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


# The 24-hour soak ---------------------------------------------------------------------------------------------------

SOAK_SHARE_TOP = 0.06
"""Panel a's scale, past the worst hour's 5.5%."""
SOAK_TOKENS_TOP = 10000
"""Panel b's scale, past the first hour's 9,300 tokens a second."""


def _soak_description(soak: data.Soak) -> str:
    worst = max(soak.hours, key=lambda h: h.share)
    first, rest = soak.hours[0], soak.hours[1:]
    return (
        f"Each of the soak's 24 hours as a column: in panel a, the share of that hour's replies over the SLO of "
        f"{soak.slo:,.0f} ms against the requirement of {ALLOWED}, {soak.over} of {soak.replies:,} replies "
        f"({soak.share:.1%}) over the day, at most {worst.over} of {worst.replies} ({worst.share:.1%}) in hour "
        f"{worst.hour}; in panel b, the agents' tokens a second, {soak.tokens:,.0f} over the day, "
        f"{first.tokens:,.0f} in hour 0 and {min(h.tokens for h in rest):,.0f} to "
        f"{max(h.tokens for h in rest):,.0f} in each hour after."
    )


def _columns(s: Chart, hours: Sequence[float], x: Scale, y: Scale, width: float) -> None:
    """Each hour's value as a column from the scale's start, in the blue the charts give the configuration to deploy
    (figure_style's "hi" bar), which the soak ran."""
    for hour, value in enumerate(hours):
        left = x(hour) - width / 2
        top = y(value)
        s.rect(left, top, width, y.start - top, "g-hi")
        s.marks.append(Box(left, top, left + width, y.start, f"hour {hour}'s column"))


def _day_rule(s: Chart, x: Scale, y: Scale, value: float, text: str, *, wide: bool) -> None:
    """The whole day's value as a dotted rule across the hours, named right of the plot, or on a phone in a key over
    the plot's top right."""
    s.ref((x.start, y(value)), (x.end, y(value)), "ink trace dotted")
    if wide:
        s.text(x.end + 10, y(value) + 6, text, size=NOTE, cls="soft")
        return
    entry = [(line_key("ink trace dotted", 24), text)]
    legend(s, x.end - legend_width(entry), y.end - 14, entry, width=x.end - x.start)


def soak(out: Path) -> list[Path]:
    """report-soak.svg and its phone layout: the 24-hour soak by hour. (a) Each hour's replies over the SLO, against
    the requirement, beside the day's share. (b) The agents' tokens a second, beside the day's."""
    s_data = data.soak()
    description = _soak_description(s_data)
    return [
        _soak_layout(s_data, out / "report-soak.svg", description, WIDTH),
        _soak_layout(s_data, out / "report-soak-narrow.svg", description, NARROW),
    ]


def _soak_layout(soak: data.Soak, path: Path, description: str, width: float) -> Path:
    wide = width == WIDTH
    s = Chart(width, 800, "The 24-hour soak", description)
    x0, x1 = (96.0, WIDTH - 190.0) if wide else (64.0, NARROW - 24.0)
    x = Scale(-0.5, 23.5, x0, x1)
    column = (x(1) - x(0)) * 0.62
    worst = max(soak.hours, key=lambda h: h.share)
    hour_ticks = [0, 6, 12, 18, 23]
    # (a) The replies over the SLO, by hour.
    s.text(24 if wide else 18, 30, "(a) Replies over the SLO, by hour", size=LABEL)
    s.text(24 if wide else 18, 64, "Share of the hour's replies", size=NOTE, cls="soft")
    a = Scale(0, SOAK_SHARE_TOP, 300, 100)
    ticks = [0.0, 0.02, 0.04, 0.06]
    y_axis(s, a, ticks, [pct(v) for v in ticks], x.start, x.end)
    _columns(s, [h.share for h in soak.hours], x, a, column)
    s.rule(x.start, x.end, a(REQUIREMENT), "slo")
    if wide:
        s.text(x.end + 10, a(REQUIREMENT) + 6, ALLOWED, size=NOTE, cls="orange tint")
    else:
        s.text(
            x.end,
            a(REQUIREMENT) - 8,
            ALLOWED,
            size=NOTE,
            anchor="end",
            cls="orange tint",
            halo=True,
        )
    s.text(
        x(worst.hour) + column / 2 + 6,
        a(worst.share) + 2,
        f"{worst.share:.1%}",
        size=NOTE,
        halo=True,
    )
    _day_rule(s, x, a, soak.share, f"the day, {soak.share:.1%}", wide=wide)
    x_axis(s, x, hour_ticks, [str(h) for h in hour_ticks], None, a.start)
    # (b) The agents' tokens a second, by hour.
    top_b = a.start + (96 if wide else 120)
    s.text(
        24 if wide else 18,
        top_b - (34 if wide else 58),
        "(b) Agents' tokens/s, by hour",
        size=LABEL,
    )
    b = Scale(0, SOAK_TOKENS_TOP, top_b + 190, top_b)
    marks_b, labels_b = linear_ticks(SOAK_TOKENS_TOP, 5000)
    y_axis(s, b, marks_b, labels_b, x.start, x.end)
    _columns(s, [h.tokens for h in soak.hours], x, b, column)
    first = soak.hours[0]
    s.text(
        x(0) + column / 2 + 6, b(first.tokens) + 12, f"{first.tokens:,.0f}", size=NOTE, halo=True
    )
    _day_rule(s, x, b, soak.tokens, f"the day, {soak.tokens:,.0f}", wide=wide)
    last = x_axis(
        s,
        x,
        hour_ticks,
        [str(h) for h in hour_ticks],
        None,
        b.start,
        title="Hours since the soak began",
    )
    s.height = last + 14
    return s.save(path)


# The engine's starts and the kill drill ------------------------------------------------------------------------------

PHASES = (
    ("load", "process, weights"),
    ("kv", "compile, KV cache"),
    ("graphs", "kernels, CUDA graphs"),
    ("serve", "until it serves"),
)
"""The phases of a start's bar, each its key and its name in the legend, from the unit's start
(flatshare_bench.drills.START_PHASES): the engine's process until its weights are loaded, torch.compile and the
profiling run until its KV cache is allocated, the kernels' warmup and JIT builds until its CUDA graphs are captured,
and its server until it serves: its unit ready (the health check), or after the kill a request answered. Before a
restart's, the killed engine down ("down", its unit's teardown and systemd's restart delay), which its row's note
names."""
PHASE_ENDS = {
    "load": "the weights are loaded",
    "kv": "the KV cache is allocated",
    "graphs": "the CUDA graphs are captured",
}
"""The start phase (drills.START_PHASES) each of the bar's phases ends at."""
READY = "the unit is ready (its health check)"
PHASE_CSS = (
    "svg{--phase-graphs:#858d99}@media(prefers-color-scheme:dark){svg{--phase-graphs:#969fab}}"
    ".p-down{fill:var(--page);stroke:var(--bar-line);stroke-width:1.1;stroke-dasharray:4 3}"
    ".p-load{fill:var(--bar-lo);stroke:var(--bar-line);stroke-width:1.1}"
    ".p-kv{fill:var(--bar-mid);stroke:var(--bar-line);stroke-width:1.1}"
    ".p-graphs{fill:var(--phase-graphs);stroke:var(--bar-line);stroke-width:1.1}"
    ".p-serve{fill:var(--page);stroke:var(--bar-line);stroke-width:1.1}"
)
"""The phases' fills: the engine down, open and dashed; a start's three long phases in grays that darken with the
start (figure_style's pale and mid gray, then a gray that clears a delta E (OKLab) of 15 from the mid gray in each
theme); its last seconds open."""
STARTS_TOP = 540
"""The time axis, past the cold start's 531.8 s."""


class Phases(Chart):
    """A chart whose style also holds the phases' fills (PHASE_CSS)."""

    def render(self) -> str:
        return super().render().replace("</style>", PHASE_CSS + "</style>", 1)

    def phases(
        self, x: Scale, y: float, thick: float, spans: Sequence[tuple[str, float, float]]
    ) -> None:
        """A bar of ``spans`` (key, from, to in seconds), its free end rounded, each a mark no label may cover."""
        for k, (key, start, end) in enumerate(spans):
            x0, x1 = x(start), x(end)
            self._hpath(x0, y, x1 - x0, thick, f"p-{key}", 3.0 if k == len(spans) - 1 else 0.0)
            self.marks.append(Box(x0, y, x1, y + thick, key))


def _phase_key(key: str) -> Key:
    def draw(s: Chart, x: float, baseline: float) -> None:
        s._hpath(x, baseline - 0.66 * NOTE, 22, 0.7 * NOTE, f"p-{key}", 0.0)

    return Key(22, draw)


def _start_spans(start: data.Start, offset: float, end: float) -> list[tuple[str, float, float]]:
    """A start's phases from ``offset`` seconds, the last until ``end``."""
    spans, at = [], offset
    for key, phase in PHASE_ENDS.items():
        to = offset + start.at(phase)
        spans.append((key, at, to))
        at = to
    return [*spans, ("serve", at, end)]


def _starts_rows(
    warm: data.Start, cold: data.Start, kill: data.Kill
) -> list[tuple[str, str, list[tuple[str, float, float]], str]]:
    """Each bar: its name and note, its spans, and the label at its end."""
    ready_warm, ready_cold = warm.at(READY), cold.at(READY)
    return [
        (
            "The engine killed",
            f"down {kill.restarted:.1f} s, then a warm start",
            [
                ("down", 0.0, kill.restarted),
                *_start_spans(kill.start, kill.restarted, kill.first_success),
            ],
            f"a request answered, {kill.first_success:.1f} s",
        ),
        (
            "A warm start",
            "compile and JIT caches full",
            _start_spans(warm, 0.0, ready_warm),
            f"ready, {ready_warm:.1f} s",
        ),
        (
            "A cold start",
            "empty caches, as on a new host",
            _start_spans(cold, 0.0, ready_cold),
            f"ready, {ready_cold:.1f} s",
        ),
    ]


def _starts_description(warm: data.Start, cold: data.Start, kill: data.Kill) -> str:
    weights, kv, graphs = (cold.at(PHASE_ENDS[key]) for key in ("load", "kv", "graphs"))
    return (
        "Three bars of the engine's phases on one axis of seconds: killed mid-decode, it was down "
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
    caches, each a bar of its phases on one time axis."""
    found = data.starts()
    warm, cold, kill = found["warm"], found["cold"], data.kill()
    rows = _starts_rows(warm, cold, kill)
    description = _starts_description(warm, cold, kill)
    return [
        _starts_layout(rows, out / "report-starts.svg", description, WIDTH),
        _starts_layout(rows, out / "report-starts-narrow.svg", description, NARROW),
    ]


def _starts_layout(
    rows: list[tuple[str, str, list[tuple[str, float, float]], str]],
    path: Path,
    description: str,
    width: float,
) -> Path:
    s = Phases(width, 800, "The engine's starts and its recovery", description)
    entries = [(_phase_key(key), name) for key, name in PHASES]
    title = "Seconds from the kill, or from the unit's start"
    if width == WIDTH:
        legend(s, 24, 30, entries, width=WIDTH - 48)
        x = Scale(0, STARTS_TOP, 300, 976)
        top, pitch, thick = 64.0, 70.0, 24.0
        ys = [top + 30 + pitch * k for k in range(len(rows))]
        bottom = ys[-1] + thick + 20
        ticks = list(range(0, STARTS_TOP + 1, 60))
        last = x_axis(s, x, ticks, [str(t) for t in ticks], top, bottom, title=title)
        for y, (name, note, spans, end) in zip(ys, rows, strict=True):
            s.text(276, y + 8, name, size=VALUE, anchor="end")
            s.text(276, y + 31, note, size=NOTE, anchor="end", cls="soft")
            s.phases(x, y, thick, spans)
            s.text(x(spans[-1][2]) + 10, y + thick / 2 + 6, end, size=NOTE, halo=True)
    else:
        x0 = 18.0
        y = legend(s, x0, 30, entries, width=NARROW - 2 * x0, pitch=25) + 16
        x = Scale(0, STARTS_TOP, x0, NARROW - x0)
        thick = 22.0
        ticks = [0, 120, 240, 360, 480]
        for name, note, spans, end in rows:
            s.text(x0, y + 20, name, size=VALUE)
            s.text(x0, y + 42, note, size=NOTE, cls="soft")
            band = y + 52
            band_grid(s, x, ticks, band - 4, band + thick + 4)
            s.phases(x, band, thick, spans)
            tip = x(spans[-1][2])
            if tip + 10 + measure(end, NOTE) <= NARROW - 4:
                s.text(tip + 10, band + thick / 2 + 6, end, size=NOTE, halo=True)
            else:  # over the bar's end, on its note's line
                s.text(tip, y + 42, end, size=NOTE, anchor="end")
            y = band + thick + 10
        last = x_axis(s, x, ticks, [str(t) for t in ticks], None, y)
        for line in wrap(title, NARROW - 2 * x0, NOTE):
            last += 25
            s.text(x0, last, line, size=NOTE, cls="soft")
    s.height = last + 14
    return s.save(path)


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
    margin,
    a100,
    cards,
    step_cap,
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
