"""The report's figures (:mod:`.report`) as SVG files that are the same bytes for the same report,
in the docs' figure style (:mod:`flatshare_bench.figure_style`): a frontier figure per workload,
the stage waterfall, and the README's voice-p95.svg with its phone layout.

Every number a figure prints is a record's, as the report's tables give it: a configuration's
median over its runs, or a stage's mean over the replies of its runs. Latency scales start at zero.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from flatshare_bench.e2.record import REPLY_TIMEOUT_MS
from flatshare_bench.e2.report import STACKED, arm_order, configurations
from flatshare_bench.figure_style import (
    NARROW,
    NOTE,
    TITLE,
    VALUE,
    WIDTH,
    Figure,
    measure,
    wrap,
)
from flatshare_stats import quantiles

FAMILIES = (
    ("stock vLLM", 1, 5),
    ("operator answers", 6, 6),
    ("engines", 7, 7),
    ("routers", 8, 8),
    ("make-room", 9, 9),
    ("MPS", 10, 10),
)
"""E2's arms by family (docs/EXPERIMENTS.md): its name, its first and last arm. A frontier figure
groups its rows under them."""
CHANGE_FAMILY = 9
"""The family of the engine change: make-room and the PRs' configurations, drawn in blue."""
TO_DEPLOY = "9-pr-ab48"
"""The configuration to deploy (docs/PRODUCTION.md): the two PRs with a 48-token window, the one
configuration a frontier figure draws in the saturated blue."""
WORKLOAD_CLAUSES = re.compile(
    r", (?:discriminating|standard) scenario(?: \(large contexts\))?"
    r"|, a warm-up turn opening each call|, \d+ callers"
)
"""The clauses of a configuration's label that name its workload, which every row of a figure
shares and the figure's title gives once."""
NUMBERS = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")


def label(text: str) -> str:
    """A configuration's label as a figure names it, under its workload's title or heading:
    without the clauses that name the workload, and a first word in lower case capitalized
    ("Frozen box", but "vLLM #63350")."""
    text = WORKLOAD_CLAUSES.sub("", text)
    first = text.split(" ", 1)[0]
    return text[:1].upper() + text[1:] if first.islower() else text


def ticks(top: float, length: float) -> list[float]:
    """Round ticks from zero that reach ``top``, at most seven, on an axis ``length`` long: the
    finest step whose labels keep a clear gap between them."""
    if top <= 0:
        return [0.0, 1.0]
    magnitude = 10 ** math.floor(math.log10(top))
    for factor in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20):
        step = factor * magnitude
        count = math.ceil(top / step - 1e-9)
        widest = max(measure(tick_text(step * k, step), NOTE) for k in range(count + 1))
        if count <= 6 and length / count >= widest + 16:
            return [step * k for k in range(count + 1)]
    return [0.0, 20 * magnitude * math.ceil(top / (20 * magnitude))]


def tick_text(value: float, step: float) -> str:
    """A tick's value, with as many decimals as its step needs, the same for every tick."""
    digits = next(d for d in range(4) if abs(step * 10**d - round(step * 10**d)) < 1e-9)
    return f"{value:,.{digits}f}"


def _family(arm: str) -> tuple[int, str]:
    """An arm's family, by its index in FAMILIES and as a heading names it."""
    number = arm_order(arm)[0]
    for index, (name, first, last) in enumerate(FAMILIES):
        if first <= number <= last:
            arms = f"arm {first}" if first == last else f"arms {first}\N{EN DASH}{last}"
            return index, f"{name[0].upper()}{name[1:]}, {arms}"
    return len(FAMILIES), "Other arms"


def _fence(values: Sequence[float]) -> float | None:
    """The V2V p95 past which a frontier figure cuts a configuration's bar at the scale's end, so
    that one collapse does not flatten the rest: Tukey's fence for far outliers, three
    interquartile ranges above the upper quartile, of four or more configurations. None when none
    is past it."""
    if len(values) < 4:
        return None
    q1, _, q3 = quantiles(values, 4)
    fence = q3 + 3 * (q3 - q1)
    return fence if max(values) > fence else None


def load_sweep(configurations: Sequence[Mapping[str, Any]]) -> bool:
    """Whether configurations span load levels (numbers of session trees)."""
    return len({c["runs"][0]["sessions"] for c in configurations}) > 1


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.2f}"


def _name(s: Figure, right: float, y: float, name: str, note: str) -> float:
    """A row's name, right-aligned at ``right`` from the label column's left edge (24) in lines that
    break evenly, and its ``note`` in the soft ink at the end of its last line: where it does not
    fit there, the name breaks narrower to make room, at the cost of one more line at most, else
    the note takes a line of its own. Returns the lines' height."""
    width, room = right - 24, measure(note, NOTE) + 10
    lines = wrap(name, width, VALUE)
    joined = bool(note) and measure(lines[-1], VALUE) + room <= width
    if note and not joined:
        narrower = wrap(name, width - room, VALUE)
        if len(narrower) <= len(lines) + 1 and measure(narrower[-1], VALUE) + room <= width:
            lines, joined = narrower, True
    for k, line in enumerate(lines):
        last = k == len(lines) - 1
        end = right - (room if last and joined else 0)
        s.text(end, y + 15 + 22 * k, line, size=VALUE, anchor="end")
    height = 22.0 * len(lines)
    if joined:
        s.text(right, y + 15 + height - 22, note, size=NOTE, anchor="end", cls="soft")
    elif note:
        s.text(right, y + 15 + height, note, size=NOTE, anchor="end", cls="soft")
        height += 20
    return height


def _panel_heads(s: Figure, cx: float, y: float, title: str, subtitle: str) -> None:
    s.text(cx, y, title, anchor="middle")
    s.text(cx, y + 25, subtitle, size=NOTE, anchor="middle", cls="soft")


def _axis(s: Figure, x0: float, length: float, marks: list[float], y0: float, y1: float) -> None:
    """A scale of ``length`` from zero to the last of ``marks``, its ticks' values in the units
    they are given in: gridlines from ``y0`` to ``y1``, the axis at zero, the baseline at ``y1`` and
    the values under it."""
    step = marks[1] - marks[0]
    for mark in marks:
        x = x0 + length * mark / marks[-1]
        s.vrule(x, y0, y1, "axis" if mark == 0 else "grid")
        s.text(x, y1 + 23, tick_text(mark, step), size=NOTE, anchor="middle", cls="mute")
    s.rule(x0, x0 + length, y1)


# The frontier figure -----------------------------------------------------------------------------


def plot(rows: Sequence[Mapping[str, Any]], path: Path, slo: float | None = None) -> Path:
    """A workload's frontier figure: a row per configuration, under its family's heading, with its
    voice-to-voice p95 against the voice SLO (``slo`` ms) beside the agents' goodput over the
    calls, each its median over the configuration's runs. A configuration on its arm's frontier has
    a filled bar and one off it an open bar; the change's configurations are blue, the one to
    deploy saturated, and the others gray. A configuration whose V2V p95 lies past the far
    outliers' fence of the rest, or that failed turns (its V2V censored at the reply timeout), has
    its bar cut at the scale's end, its value printed."""
    groups = configurations(rows)
    workload = rows[0]["workload"]
    measured = [g["v2v_p95_ms"] for g in groups if g["v2v_p95_ms"] is not None]
    fence = _fence([g["v2v_p95_ms"] for g in groups if not g["failed_turns"]])
    shown = [v for v in measured if (fence is None or v <= fence) and v < REPLY_TIMEOUT_MS]
    top_ms = max([*shown, *([slo] if slo else []), 1000.0])
    right, ax0, alen, bx0, blen = 388, 412, 344, 842, 176
    marks_a = ticks(top_ms / 1000, alen)
    top_a = marks_a[-1] * 1000
    goodputs = [g["background_tokens_per_s"] for g in groups if g["runs"][0]["sessions"]]
    marks_b = ticks(max([*goodputs, 1.0]), blen)
    top_b = marks_b[-1]
    pale = any(arm_order(g["arm"])[0] == CHANGE_FAMILY and g["arm"] != TO_DEPLOY for g in groups)

    s = Figure(WIDTH, 400, f"Voice tail and the agents' goodput: {workload}", "")
    s.text(
        WIDTH / 2,
        38,
        f"Voice tail and the agents' goodput: {workload}",
        size=TITLE,
        anchor="middle",
    )
    _panel_heads(
        s, ax0 + alen / 2, 92, "Voice-to-voice p95", "seconds, median of the runs • Lower is better"
    )
    _panel_heads(
        s, bx0 + blen / 2, 92, "Agents' goodput", "tokens/s over the calls • Higher is better"
    )
    top = 160  # the SLO's label; the plot starts under it
    y = top + 16
    family = None
    placed = []  # (y of the bar, configuration, lines' bottom)
    for g in groups:
        index, heading = _family(g["arm"])
        if index != family:
            family = index
            y += 2 if placed else 0
            s.text(24, y + 15, heading, size=NOTE, cls="mute")
            y += 26
        notes = []
        if len(g["runs"]) > 1:
            notes.append(f"{len(g['runs'])} runs")
        if g["failed_turns"]:
            notes.append(f"{g['failed_turns']} of {g['turns']} turns failed")
        height = _name(s, right, y, label(g["runs"][0]["label"]), ", ".join(notes))
        placed.append((y + max(height, 22) / 2 - 10, g))
        y += max(height, 22) + 10
    bottom = y + 2
    _axis(s, ax0, alen, marks_a, top + 8, bottom)
    _axis(s, bx0, blen, marks_b, top + 8, bottom)
    cut = False
    values = []
    fills = set()
    for bar_y, g in placed:
        number = arm_order(g["arm"])[0]
        bar = (
            "open"
            if not g["frontier"]
            else "hi"
            if g["arm"] == TO_DEPLOY
            else "soft"
            if number == CHANGE_FAMILY
            else "mid"
            if pale
            else "lo"
        )
        fills.add(bar)
        p95 = g["v2v_p95_ms"]
        if p95 is not None:
            over = p95 > top_a
            cut |= over
            s.hbar(ax0, bar_y, alen * min(p95, top_a) / top_a, 20, bar, cut=over)
            text = ("≥" if p95 >= REPLY_TIMEOUT_MS else "") + _seconds(p95)
            values.append((ax0 + alen * min(p95, top_a) / top_a + 7, bar_y + 15, text))
        if g["runs"][0]["sessions"]:
            tokens = g["background_tokens_per_s"]
            s.hbar(bx0, bar_y, blen * tokens / top_b, 20, bar)
            values.append((bx0 + blen * tokens / top_b + 7, bar_y + 15, f"{tokens:,.0f}"))
        else:
            values.append((bx0 + 7, bar_y + 15, "no agents"))
    if slo:
        x = ax0 + alen * slo / top_a
        s.vrule(x, top + 8, bottom, "slo")
        s.text(x, top, f"SLO {_seconds(slo)} s", size=NOTE, anchor="middle", cls="orange tint")
    for x, baseline, text in values:
        s.text(x, baseline, text, size=NOTE, halo=True, cls="soft" if text == "no agents" else "")
    notes = []
    if any(not g["frontier"] for g in groups):
        notes.append("An open bar is a configuration off its arm's frontier.")
    if {"hi", "soft"} <= fills:
        notes.append(DEPLOY_NOTE)
    if cut:
        notes.append("A bar cut square runs past the scale's end; its value says how far.")
    if slo is None:
        notes.append(f"No point of voice alone with {rows[0]['callers']} callers sets a voice SLO.")
    else:
        notes.append("The SLO is the p95 of voice alone with as many callers, plus 250 ms.")
    y = bottom + 58
    for note in notes:
        s.text(24, y, note, size=NOTE, cls="soft")
        y += 24
    s.height = y - 6
    return s.save(path)


# The stage waterfall -----------------------------------------------------------------------------

STAGES = (*STACKED, ("transport_ms", "Transport and playout"))
STAGE_FILLS = ("lo", "violet", "mid", "pale")
"""Each stage's fill: the reply's first sentence, the LLM's stage and where a shared GPU loses
its time, the one hue; the others grays of distinct lightness."""


def plot_waterfall(cells: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """waterfall.svg: a row per configuration whose bar stacks the means of its stages from the end
    of the caller's speech (STACKED, then transport and playout), so that its length is the mean
    V2V, printed at its end; each run of rows of one workload under the workload's name."""
    right, x0, length = 388, 412, 610
    top_ms = max(c["v2v_mean_ms"] for c in cells)
    marks = ticks(top_ms / 1000, length)
    top = marks[-1] * 1000
    s = Figure(WIDTH, 400, "The voice turn by stage", "")
    s.text(WIDTH / 2, 38, "The voice turn by stage", size=TITLE, anchor="middle")
    s.text(
        WIDTH / 2,
        64,
        "Mean over the replies of each configuration's runs, seconds • Shorter is better",
        size=NOTE,
        anchor="middle",
        cls="soft",
    )
    # The stages' swatches, centred under the subtitle.
    widths = [30 + measure(name, NOTE) for _, name in STAGES]
    x = (WIDTH - sum(widths) - 28 * (len(STAGES) - 1)) / 2
    for (_, name), fill, width in zip(STAGES, STAGE_FILLS, widths, strict=True):
        s.swatch(x, 98, fill, name)
        x += width + 28
    top_y = 132
    y = top_y + 8
    placed = []
    workload = None
    for c in cells:
        if c["workload"] != workload:  # each run of one workload's rows under its heading
            workload = c["workload"]
            y += 2 if placed else 0
            # In the label column, clear of the gridlines: a GPU's name makes one wider than it.
            lines = wrap(workload, right - 24, NOTE)
            for k, line in enumerate(lines):
                s.text(24, y + 15 + 20 * k, line, size=NOTE, cls="mute")
            y += 26 + 20 * (len(lines) - 1)
        height = _name(s, right, y, label(c["label"]), f"{c['runs']} runs" if c["runs"] > 1 else "")
        placed.append((y + max(height, 22) / 2 - 10, c))
        y += max(height, 22) + 10
    bottom = y + 2
    _axis(s, x0, length, marks, top_y, bottom)
    for bar_y, c in placed:
        parts = [
            (length * max(0.0, c[key]) / top, fill)
            for (key, _), fill in zip(STAGES, STAGE_FILLS, strict=True)
        ]
        s.segments(x0, bar_y, parts, 20)
        end = x0 + sum(width for width, _ in parts)
        s.text(end + 7, bar_y + 15, _seconds(c["v2v_mean_ms"]), size=NOTE, halo=True)
    s.height = bottom + 40
    return s.save(path)


# The README's voice-p95.svg ----------------------------------------------------------------------


def _title(groups: Sequence[Mapping[str, Any]]) -> str:
    """The figure's title, from its records: the calls, the agents and, for one group, the GPU."""
    callers, sessions = groups[0]["callers"], groups[0]["sessions"]
    calls = (
        f"{NUMBERS[callers]} voice calls" if callers < len(NUMBERS) else f"{callers} voice calls"
    )
    title = f"{calls} beside {sessions} coding-agent sessions"
    return title + (f" on one {groups[0]['gpu']}" if len(groups) == 1 else "")


def plot_voice(groups: Sequence[Mapping[str, Any]], path: Path, narrow: Path) -> list[Path]:
    """voice-p95.svg and its phone layout: each group's configurations (``rows``: the
    configuration and its row of figure.toml) as rows of two panels, the V2V p95 against the
    group's voice SLO (``slo`` ms) and the agents' tokens/s over the fixed window with each one's
    share of the group's first row, each the median over the configuration's runs. A group is the
    configurations of one GPU (``gpu``); with several, each is headed by its ``title``. The row to
    deploy (``featured``) is the saturated blue, the change's other rows (``change``) pale blue,
    the rest gray. ``callers`` and ``sessions`` are the groups' workload, which the title names."""
    p95_top = max(
        [g["slo"] or 0.0 for g in groups] + [c["v2v_p95_ms"] for g in groups for c, _ in g["rows"]]
    )
    tokens = max([c["fixed_window_tokens_per_s"] or 0.0 for g in groups for c, _ in g["rows"]])
    title = _title(groups)
    description = _describe(groups)
    return [
        _voice_wide(groups, path, title, description, p95_top, max(tokens, 1.0)),
        _voice_narrow(groups, narrow, title, description, p95_top, max(tokens, 1.0)),
    ]


def _bar(spec: Mapping[str, Any]) -> str:
    return "hi" if spec.get("featured") else "soft" if spec.get("change") else "mid"


DEPLOY_NOTE = "The darker blue bar is the configuration to deploy."
"""The note of a figure whose blue bars come in both tones: the configuration to deploy, and the change's others."""


def _voice_notes(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    """The voice figure's notes: what its two blues mean, where it has both, and what its SLOs are."""
    fills = {_bar(spec) for g in groups for _, spec in g["rows"]}
    notes = [DEPLOY_NOTE] if {"hi", "soft"} <= fills else []
    if any(g["slo"] for g in groups):
        notes.append(
            "Each group's SLO is the p95 of voice alone on its card, with as many callers, plus 250 ms."
            if len(groups) > 1
            else "The SLO is the p95 of voice alone with as many callers, plus 250 ms."
        )
    return notes


def _share(group: Mapping[str, Any], configuration: Mapping[str, Any]) -> str | None:
    base = group["rows"][0][0]["fixed_window_tokens_per_s"]
    tokens = configuration["fixed_window_tokens_per_s"]
    return None if not base or tokens is None else f"{tokens / base:.0%}"


def _note(configuration: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    runs = len(configuration["runs"])
    parts = [spec["note"]] if spec.get("note") else []
    parts.append(f"{runs} run{'' if runs == 1 else 's'}")
    if configuration["failed_turns"]:
        parts.append(f"{configuration['failed_turns']} of {configuration['turns']} turns failed")
    return ", ".join(parts)


def _describe(groups: Sequence[Mapping[str, Any]]) -> str:
    sentences = []
    for g in groups:
        head = f"{g['gpu']}: " if len(groups) > 1 else ""
        rows = "; ".join(
            f"{spec['label']}{', ' + spec['note'] if spec.get('note') else ''}, "
            f"{_seconds(c['v2v_p95_ms'])} s and {c['fixed_window_tokens_per_s']:,.0f} tokens/s"
            f" ({_share(g, c)} of the first row's) over {len(c['runs'])} runs"
            for c, spec in g["rows"]
        )
        slo = f" against an SLO of {_seconds(g['slo'])} s" if g["slo"] else ""
        sentences.append(f"{head}voice-to-voice p95{slo} and the agents' tokens/s: {rows}.")
    return " ".join(sentences)


def _voice_wide(groups, path, title, description, p95_top, tokens_top) -> Path:
    right, ax0, alen, bx0, blen = 238, 262, 420, 764, 200
    marks_a, marks_b = ticks(p95_top / 1000, alen), ticks(tokens_top, blen)
    top_a, top_b = marks_a[-1] * 1000, marks_b[-1]
    s = Figure(WIDTH, 400, title, description)
    s.text(WIDTH / 2, 38, title, size=TITLE, anchor="middle")
    _panel_heads(
        s, ax0 + alen / 2, 96, "Voice-to-voice p95", "seconds, median of the runs • Lower is better"
    )
    _panel_heads(
        s,
        bx0 + blen / 2 + 24,
        96,
        "Agents' throughput",
        "tokens/s, % of the baseline • Higher is better",
    )
    top = 164  # the SLO's label; the plot starts under it
    y = top + 16
    pitch, thick = 54, 26
    spans = []  # each group's rows' extent, for its SLO
    bars = []
    for g in groups:
        if len(groups) > 1:
            s.text(24, y + 22, g["title"], size=NOTE, cls="mute")
            y += 36
        first = y
        for configuration, spec in g["rows"]:
            s.text(right, y + 12, spec["label"], size=VALUE, anchor="end")
            s.text(right, y + 34, _note(configuration, spec), size=NOTE, anchor="end", cls="soft")
            bars.append((y, g, configuration, spec))
            y += pitch
        spans.append((first, y - pitch + thick + 6, g))
        y += 10
    bottom = y - 10 + 2
    _axis(s, ax0, alen, marks_a, top + 8, bottom)
    _axis(s, bx0, blen, marks_b, top + 8, bottom)
    values = []
    for bar_y, g, configuration, spec in bars:
        p95 = configuration["v2v_p95_ms"]
        s.hbar(ax0, bar_y, alen * p95 / top_a, thick, _bar(spec))
        values.append((ax0 + alen * p95 / top_a + 8, bar_y + 19, _seconds(p95), ""))
        tokens = configuration["fixed_window_tokens_per_s"]
        if tokens is None:
            values.append((bx0 + 8, bar_y + 19, "\N{EN DASH}", "soft"))
            continue
        s.hbar(bx0, bar_y, blen * tokens / top_b, thick, _bar(spec))
        values.append((bx0 + blen * tokens / top_b + 8, bar_y + 19, f"{tokens:,.0f}", ""))
        if share := _share(g, configuration):
            width = measure(f"{tokens:,.0f}", NOTE)
            values.append((bx0 + blen * tokens / top_b + 8 + width + 8, bar_y + 19, share, "soft"))
    for k, (first, last, g) in enumerate(spans):
        if g["slo"]:
            x = ax0 + alen * g["slo"] / top_a
            label_y = top if k == 0 else first - 8
            s.vrule(x, label_y + 8, last, "slo")
            s.text(
                x,
                label_y,
                f"SLO {_seconds(g['slo'])} s",
                size=NOTE,
                anchor="middle",
                cls="orange tint",
                halo=True,
            )
    for x, baseline, text, cls in values:
        s.text(x, baseline, text, size=NOTE, cls=cls, halo=True)
    y = bottom + 58
    for note in _voice_notes(groups):
        s.text(24, y, note, size=NOTE, cls="soft")
        y += 24
    s.height = y - 6
    return s.save(path)


def _voice_narrow(groups, path, title, description, p95_top, tokens_top) -> Path:
    s = Figure(NARROW, 400, title, description)
    cx = NARROW / 2
    y = 32
    for line in wrap(title, NARROW - 32, TITLE):
        s.text(cx, y, line, size=TITLE, anchor="middle")
        y += 29
    x0, length = 18, 296
    marks_a, marks_b = ticks(p95_top / 1000, length), ticks(tokens_top, length)

    def panel(y: float, head: str, sub: str, marks: list[float], scale: float, show) -> float:
        _panel_heads(s, cx, y + 20, head, sub)
        y += 62
        top = marks[-1] * scale
        thick = 20
        for g in groups:
            if len(groups) > 1:
                s.text(x0, y + 16, g["title"], size=NOTE, cls="mute")
                y += 28
            slo = g["slo"] if scale == 1000 else None
            if slo:
                s.text(
                    x0 + length * slo / top,
                    y + 14,
                    f"SLO {_seconds(slo)} s",
                    size=NOTE,
                    anchor="middle",
                    cls="orange tint",
                )
                y += 22
            for configuration, spec in g["rows"]:
                name = s.text(x0, y + 18, spec["label"], size=VALUE)
                if spec.get("note"):
                    s.text(name.x1 + 7, y + 18, spec["note"], size=NOTE, cls="soft")
                band = y + 27
                for mark in marks:
                    s.vrule(
                        x0 + length * mark * scale / top,
                        band,
                        band + thick + 8,
                        "axis" if mark == 0 else "grid",
                    )
                value = show(g, configuration)
                if value is None:
                    s.text(x0 + 8, band + 4 + 15, "\N{EN DASH}", size=NOTE, cls="soft", halo=True)
                else:
                    amount, texts = value
                    end = x0 + length * amount / top
                    s.hbar(x0, band + 4, end - x0, thick, _bar(spec))
                    x = end + 7
                    for text, cls in texts:
                        box = s.text(x, band + 4 + 15, text, size=NOTE, cls=cls, halo=True)
                        x = box.x1 + 7
                if slo:
                    s.vrule(x0 + length * slo / top, band, band + thick + 8, "slo")
                y += 62
        bottom = y - 62 + 27 + thick + 8
        s.rule(x0, x0 + length, bottom)
        step = marks[1] - marks[0]
        for mark in marks:
            s.text(
                x0 + length * mark * scale / top,
                bottom + 23,
                tick_text(mark, step),
                size=NOTE,
                anchor="middle",
                cls="mute",
            )
        return bottom + 42

    def p95(g, configuration):
        return configuration["v2v_p95_ms"], [(_seconds(configuration["v2v_p95_ms"]), "")]

    def tokens(g, configuration):
        amount = configuration["fixed_window_tokens_per_s"]
        if amount is None:
            return None
        share = _share(g, configuration)
        return amount, [(f"{amount:,.0f}", ""), *([(share, "soft")] if share else [])]

    y = panel(y + 6, "Voice-to-voice p95", "seconds • Lower is better", marks_a, 1000, p95)
    y = panel(
        y + 10,
        "Agents' throughput",
        "tokens/s, % of the baseline • Higher is better",
        marks_b,
        1,
        tokens,
    )
    y += 16  # the notes' first baseline as far under the ticks as the wide figure's
    for note in _voice_notes(groups):
        for line in wrap(note, NARROW - 36, NOTE):
            s.text(18, y, line, size=NOTE, cls="soft")
            y += 24
        y += 6
    s.height = y - 10
    return s.save(path)
