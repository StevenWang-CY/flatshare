"""The report's figures (:mod:`.report`) as SVG files that are the same bytes for the same report,
in the docs' figure style (:mod:`flatshare_bench.figure_style`): a frontier figure per workload,
the stage waterfall, and the README's voice-p95.svg with its phone layout.

Every number a figure prints is a record's, as the report's tables give it: a configuration's
median over its runs, or a stage's mean over the replies of its runs. Latency scales start at zero.

The frontier figure and the waterfall are set as the main report's figures are, on its chart parts
(:mod:`flatshare_bench.report.charts`): a row's name in a column at the left, then its agent
sessions, the table's own column, which parts the rows of one configuration at several loads; a
panel's title in semibold at its left and its unit under its axis; keys over the plot and
definitions under it. A hue has one meaning in every figure (:func:`_hue`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flatshare_bench import labels
from flatshare_bench.e2.groups import STACKED, arm_order, configurations
from flatshare_bench.e2.record import REPLY_TIMEOUT_MS
from flatshare_bench.figure_style import (
    LABEL,
    NARROW,
    NOTE,
    TITLE,
    VALUE,
    WIDTH,
    Box,
    Figure,
    measure,
    wrap,
)
from flatshare_bench.report.charts import (
    BAR_FILLS,
    Chart,
    Key,
    Scale,
    cut_breaks,
    cut_key,
    legend,
    stage_key,
    x_axis,
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
CHANGE_ENGINE = "make-room"
"""What the variant of an engine with the change begins with: an arm of the change's family on
another engine (the gateway's policy on tuned vLLM) is a comparator, and gray."""
DEPLOY_ARMS = re.compile(r"9-pr-ab48(?:-b256)?(?:-t0)?(?:-(?:calls|[cs]\d+))*")
"""The arms that run the configuration to deploy (docs/PRODUCTION.md): the two PRs with a 48-token
window and no gateway policy, at the box's own step cap or at 256 tokens a step, with any number
of callers and of agent sessions. An arm that adds a gateway's static cap, another window, another
step cap or another build is a configuration of its own."""
DEPLOY_ENGINE = {
    "--additional-config.make_room.reserve_tokens": "0",
    "--additional-config.make_room.budget_cap_tokens": "48",
}
"""The engine's flags the configuration to deploy sets: no reserve, the budget cap at 48 tokens."""
STEP_FLAGS = ("--max-num-batched-tokens", "--long-prefill-token-threshold")
"""The engine's flags that set its step cap and its long-prefill threshold."""
STEP_CAP_SHIPS, STEP_CAP_BOX = 256, 512
"""The step cap the serving profile ships (256 tokens a step, no long-prefill threshold), and the
one an engine started without STEP_FLAGS runs: its box's own, 512 with a threshold of 256."""
WORKLOAD_CLAUSES = re.compile(
    r", (?:discriminating|standard) scenario(?: \(large contexts\))?"
    r"|, a warm-up turn opening each call|, \d+ callers"
)
"""The clauses of a configuration's label that name its workload, which every row of a figure
shares and the figure's first line gives once."""
NUMBERS = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")

MARGIN = 24.0
"""The figure's left and right margins."""
BAR = 24.0
"""A bar's thickness, the main report's."""
VALUE_GAP = 9.0
"""Between a bar's end and its value."""
GROUP_HEAD, GROUP_GAP = 30.0, 8.0
"""A heading's line over its rows, and the gap after a heading's last row."""
ROW_HEIGHTS = (48.0, 58.0, 80.0)
"""A row's height by its lines of text, a name's and a note's: one, two, three."""
NAME_PITCH, NOTE_PITCH = 25.0, 22.0
"""From a name's line to its next, and from its last line to the note under it."""
NAMES_MIN, NAMES_MAX, NAMES_WIDE = 240.0, 406.0, 540.0
"""The names' column: its least width, and the most a name's line takes beside two panels and
beside one. A name stands in as few lines as that lets it, and the column is as wide as its
widest line."""
SESSIONS_GAP, PLOT_GAP = 20.0, 16.0
"""Between a name and its row's sessions, and between the last column and the plot."""
PANEL_GAP = 28.0
"""Between the widest value after panel (a) and panel (b)."""
PANEL_SHARE = 0.62
"""Panel (a)'s share of the two panels' length."""
HEAD = 46.0
"""From a panel's title to its plot's top, with the SLO's name between them."""
SLO_CLEAR = 12.0
"""The least from the plot's start to the SLO's name, which stands over its rule on the line of
the sessions' head: a rule near the scale's start has its name beside that head, not against
it."""
RECEDED = {"gray": "pale", "ink": "pale", "sky": "blue-pale", "blue": "blue-pale"}
"""The fill of a receded bar (a configuration off its arm's frontier): the palest step of its
hue, a stacked bar's part (figure_style's SEGMENTS)."""
SESSIONS = "Sessions"
"""The head of the column of each row's agent sessions: the tables' own word."""
DEPLOY_KEY = "The configuration to deploy"
BOLD_NOTE = "Bold: the configuration to deploy."
"""What the semibold says in a figure whose bars' fills are its stages, so that no bar is the
saturated one a key could name."""
RECEDED_KEY = "Off its arm's frontier"
CUT_KEY = "Runs past the scale's end"


def label(text: str) -> str:
    """A configuration's label as a figure names it, under its workload's line or heading:
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


def _marks(top: float, length: float) -> tuple[list[float], list[str]]:
    """:func:`ticks` to ``top`` and each one's label in its shortest form, as the main report's
    axes give them: "0", "2.5", "5"; "5,000"."""
    marks = ticks(top, length)
    texts = [tick_text(mark, marks[1] - marks[0]) for mark in marks]
    return marks, [t.rstrip("0").rstrip(".") if "." in t else t for t in texts]


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


# What a row is, and how its hue says it ----------------------------------------------------------


def _deploy_cap(arm: str, engine: str) -> int | None:
    """The step cap at which an arm (its id and its engine, the variant then its flags) runs the
    configuration to deploy: STEP_CAP_SHIPS, STEP_CAP_BOX, or None for any other configuration."""
    variant, *flags = engine.split()
    settings = dict(zip(flags[::2], flags[1::2], strict=False))
    if variant != CHANGE_ENGINE or len(flags) % 2 or not DEPLOY_ARMS.fullmatch(arm):
        return None
    if {flag: value for flag, value in settings.items() if flag not in STEP_FLAGS} != DEPLOY_ENGINE:
        return None
    step = tuple(settings.get(flag) for flag in STEP_FLAGS)
    return {(None, None): STEP_CAP_BOX, (f"{STEP_CAP_SHIPS}", "0"): STEP_CAP_SHIPS}.get(step)


def _deployed(arms: Sequence[tuple[str, str]]) -> list[bool]:
    """Which of one workload's rows (each an arm's id and its engine) are the configuration to
    deploy: the rows that run it at the step cap the profile ships, at every load they stand at;
    where the workload has none, the rows that run it at the box's own step cap. Where both stand,
    the one at the box's cap is one of the change's other configurations, as the main report's
    step-cap figure draws the pair."""
    caps = [_deploy_cap(arm, engine) for arm, engine in arms]
    ships = STEP_CAP_SHIPS if STEP_CAP_SHIPS in caps else STEP_CAP_BOX
    return [cap == ships for cap in caps]


def _hue(arm: str, engine: str, sessions: int, deploy: bool) -> str:
    """A row's hue (the charts' BAR_FILLS), by what the row is and nothing else in its figure:
    voice alone, the reference, is ink (the stronger gray); the configuration to deploy is blue,
    the saturated one; the change's other configurations (its family's arms on an engine with
    the change) are sky, the softer blue; every other comparator is gray."""
    if not sessions:
        return "ink"
    if deploy:
        return "blue"
    if arm_order(arm)[0] == CHANGE_FAMILY and engine.startswith(CHANGE_ENGINE):
        return "sky"
    return "gray"


# Rows: names, notes and sessions at the left of a plot -------------------------------------------


@dataclass(frozen=True)
class _Row:
    """A row of a record's figure: its configuration's name in its lines, its note (its runs, its
    failed turns), its agent sessions (none for voice alone), and how it is set: ``featured``,
    the configuration to deploy, its name and value in semibold; ``receded``, a configuration
    off its arm's frontier, its name and value in the soft ink."""

    lines: tuple[str, ...]
    note: str
    sessions: int
    hue: str = "gray"
    featured: bool = False
    receded: bool = False

    @property
    def height(self) -> float:
        count = len(self.lines) + bool(self.note)
        last = len(ROW_HEIGHTS)
        return ROW_HEIGHTS[min(count, last) - 1] + NAME_PITCH * max(0, count - last)

    @property
    def ink(self) -> str:
        return "soft" if self.receded else ""


@dataclass(frozen=True)
class _Columns:
    """Where a figure's columns stand: the right edge of the names, the right edge of the sessions
    (None without the column), and the plot's start."""

    names: float
    sessions: float | None
    plot: float


def _lines(name: str, bold: bool, width: float) -> tuple[str, ...]:
    """A name in as few lines as fit ``width``, broken evenly, in the weight it is set in:
    figure_style's wrap measures the regular weight, so a name in semibold is wrapped to a
    narrower measure until its lines fit."""
    lines = wrap(name, width, LABEL)
    while bold and max(measure(line, LABEL, bold=True) for line in lines) > width:
        narrower = wrap(name, max(measure(line, LABEL) for line in lines) - 0.5, LABEL)
        if narrower == lines:
            break
        lines = narrower
    return tuple(lines)


def _names_limit(names: Sequence[tuple[str, bool]]) -> float:
    """The most a name's line takes beside one panel, for names each with whether it is set in
    semibold: NAMES_MAX, or as much more, up to NAMES_WIDE, as the longest of them needs to stand
    in two lines."""
    limit = NAMES_MAX
    for name, bold in names:
        if len(_lines(name, bold, limit)) <= 2:
            continue
        words = name.split()
        halves = sorted(
            max(measure(" ".join(part), LABEL, bold=bold) for part in (words[:k], words[k:]))
            for k in range(1, len(words))
        )
        fits = (w for w in halves if w > limit and len(_lines(name, bold, w)) <= 2)
        limit = min(math.ceil(next(fits, NAMES_WIDE)), NAMES_WIDE)
    return limit


def _columns(groups: Sequence[tuple[Sequence[str], Sequence[_Row]]]) -> _Columns:
    """A figure's columns, from its groups (each its headings and its rows): the names' column as
    wide as its widest line, at least NAMES_MIN; the sessions' column where a row has agents; and
    a heading, which has no sessions, ends before the plot too."""
    rows = [row for _, members in groups for row in members]
    names = max(
        NAMES_MIN,
        *(measure(line, LABEL, bold=row.featured) for row in rows for line in row.lines),
        *(measure(row.note, NOTE) for row in rows),
    )
    counts = [measure(f"{row.sessions}", VALUE) for row in rows if row.sessions]
    column = SESSIONS_GAP + max(counts) if counts else 0.0
    heads = max((measure(head, NOTE) for heads, _ in groups for head in heads), default=0.0)
    names = math.ceil(max(names, heads - column))
    end = MARGIN + names + column
    return _Columns(MARGIN + names, end if counts else None, end + PLOT_GAP)


def _set_rows(
    s: Chart,
    groups: Sequence[tuple[Sequence[str], Sequence[_Row]]],
    columns: _Columns,
    top: float,
) -> tuple[list[float], float]:
    """Each group's headings, a line each in the soft ink, then its rows from ``top`` down: a
    row's name right-aligned in its column, its note under it as the main report sets a row's
    second line, and its sessions in the values' ink, level with its bar, under the column's
    head, which stands just over the plot's top. Returns each row's centre, where its bar and
    its values stand, and the plot's bottom."""
    if columns.sessions is not None:
        s.text(columns.sessions, top - 8, SESSIONS, size=NOTE, anchor="end", cls="soft")
    y = top
    centres = []
    for heads, members in groups:
        for head in heads:
            s.text(MARGIN, y + 21, head, size=NOTE, cls="soft")
            y += GROUP_HEAD
        for row in members:
            centre = y + row.height / 2
            block = NAME_PITCH * (len(row.lines) - 1) + (NOTE_PITCH - 2 if row.note else 0)
            base = centre + 7 - block / 2
            for k, line in enumerate(row.lines):
                s.text(
                    columns.names,
                    base + NAME_PITCH * k,
                    line,
                    anchor="end",
                    cls=row.ink,
                    bold=row.featured,
                )
            if row.note:
                last = base + NAME_PITCH * (len(row.lines) - 1)
                s.text(
                    columns.names, last + NOTE_PITCH, row.note, size=NOTE, anchor="end", cls="soft"
                )
            if row.sessions and columns.sessions is not None:
                s.text(
                    columns.sessions,
                    centre + 7,
                    f"{row.sessions}",
                    size=VALUE,
                    anchor="end",
                    cls=row.ink,
                )
            centres.append(centre)
            y += row.height
        y += GROUP_GAP
    return centres, y - GROUP_GAP + 2


def _runs(count: int) -> list[str]:
    """A row's note of its runs, where it has several."""
    return [f"{count} runs"] if count > 1 else []


def _fill_key(*fills: str) -> Key:
    """Bars' fills in a key (figure_style's SEGMENTS): a short stretch of each, side by side."""
    width, gap = 22.0, 4.0

    def draw(s: Chart, x: float, baseline: float) -> None:
        for k, fill in enumerate(fills):
            s.segments(x + (width + gap) * k, baseline - 0.66 * NOTE, [(width, fill)], 0.7 * NOTE)

    return Key((width + gap) * len(fills) - gap, draw)


def _foot(s: Chart, last: float, notes: Sequence[str]) -> None:
    """The definitions under a figure, a line each in the soft ink under its lowest baseline
    ``last``, and the figure's height to fit them."""
    for note in notes:
        last += 30 if note is notes[0] else 24
        s.text(MARGIN, last, note, size=NOTE, cls="soft")
    s.height = last + 14


# The frontier figure -----------------------------------------------------------------------------


def _lead(names: Sequence[str], bold: Sequence[bool]) -> int:
    """How many of their first words a heading states once for its rows' names (each with
    whether it is set in semibold): the words every name begins with, where they are at least
    three, end in a word (never a bare sign) and leave something of every name; of one name too
    long for two lines, its first clause. Zero where the rows keep their whole names."""
    words = [name.split() for name in names]
    if len(names) == 1:
        clause = next((k + 1 for k, word in enumerate(words[0]) if word.endswith(",")), 0)
        long = len(_lines(names[0], bold[0], NAMES_MAX)) > 2
        return clause if long and clause < len(words[0]) else 0
    count = 0
    while all(len(w) > count for w in words) and len({w[count].rstrip(",") for w in words}) == 1:
        count += 1
    while count and not any(c.isalnum() for c in words[0][count - 1]):
        count -= 1
    return count if count >= 3 and all(len(w) > count for w in words) else 0


def _hoisted(
    heading: str, names: Sequence[str], bold: Sequence[bool]
) -> tuple[list[str], list[str]]:
    """A family's heading and its rows' names (each with whether it is set in semibold), with
    the words the names share (:func:`_lead`) stated once, after the heading where the line fits
    before the plot and under it otherwise, in as many lines as keep it there, and each row
    named by what differs."""
    count = _lead(names, bold)
    if not count:
        return [heading], list(names)
    lead = " ".join(names[0].split()[:count]).rstrip(",:;")
    rest = [label(" ".join(name.split()[count:])) for name in names]
    one = f"{heading}: {lead}"
    limit = NAMES_MAX + SESSIONS_GAP
    if measure(one, NOTE) <= limit:
        return [one], rest
    return [f"{heading}:", *wrap(lead, limit, NOTE)], rest


def _row_bar(
    s: Chart, x: float, centre: float, length: float, row: _Row, *, cut: bool = False
) -> None:
    """A row's bar of ``length`` from ``x``: in its hue's fill, or receded in the hue's palest
    step; ``cut`` at the scale's end, with a square end where the fill is a bar's and the two
    breaks before it."""
    top = centre - BAR / 2
    if row.receded:
        s.segments(x, top, [(length, RECEDED[row.hue])], BAR)
    else:
        s.hbar(x, top, length, BAR, BAR_FILLS[row.hue], cut=cut)
    if cut:
        cut_breaks(s, x + length, top, top + BAR)
    s.marks.append(Box(x, top, x + length, top + BAR, "a bar"))


def plot(rows: Sequence[Mapping[str, Any]], path: Path, slo: float | None = None) -> Path:
    """A workload's frontier figure: a row per configuration, under its family's heading, with its
    agent sessions, its voice-to-voice p95 against the voice SLO (``slo`` ms) and, where a row of
    the workload has agents, the agents' token rate over the calls beside it, each its median
    over the configuration's runs. Voice alone is the stronger gray, the change's configurations
    are blue, the one to deploy saturated and in semibold, and every other is gray (:func:`_hue`);
    a configuration off its arm's frontier is receded, its bar pale and its name and values soft.
    A configuration whose V2V p95 lies past the far outliers' fence of the rest, or that failed
    turns (its V2V censored at the reply timeout), has its bar cut at the scale's end, broken
    before it, and its value printed after it. No value stands across the SLO's rule: one that
    would takes the rule's far side."""
    groups = configurations(rows)
    workload = rows[0]["workload"]
    deploy = _deployed([(g["arm"], g["runs"][0]["engine"]) for g in groups])

    families: list[tuple[str, list[int]]] = []
    for k, g in enumerate(groups):
        heading = _family(g["arm"])[1]
        if not families or families[-1][0] != heading:
            families.append((heading, []))
        families[-1][1].append(k)
    blocks = []
    for heading, members in families:
        heads, names = _hoisted(
            heading,
            [label(groups[k]["runs"][0]["label"]) for k in members],
            [deploy[k] for k in members],
        )
        block = []
        for k, name in zip(members, names, strict=True):
            g = groups[k]
            first = g["runs"][0]
            notes = _runs(len(g["runs"]))
            if g["failed_turns"]:
                notes.append(f"{g['failed_turns']} of {g['turns']} turns failed")
            block.append(
                _Row(
                    _lines(name, deploy[k], NAMES_MAX),
                    ", ".join(notes),
                    first["sessions"],
                    _hue(g["arm"], first["engine"], first["sessions"], deploy[k]),
                    featured=deploy[k],
                    receded=not g["frontier"] and not deploy[k],
                )
            )
        blocks.append((heads, block))
    placed = [row for _, block in blocks for row in block]
    columns = _columns(blocks)
    agents = columns.sessions is not None

    # What each panel prints after its bars, which sets the room after its scale.
    p95s = [g["v2v_p95_ms"] for g in groups]
    times = [
        None if p95 is None else ("≥" if p95 >= REPLY_TIMEOUT_MS else "") + f"{_seconds(p95)} s"
        for p95 in p95s
    ]
    rates = [
        f"{g['background_tokens_per_s']:,.0f}" if row.sessions else "no agents"
        for g, row in zip(groups, placed, strict=True)
    ]
    room_a = VALUE_GAP + max(
        (measure(t, VALUE, bold=r.featured) for t, r in zip(times, placed, strict=True) if t),
        default=0.0,
    )
    room_b = (
        VALUE_GAP
        + max(
            measure(t, VALUE, bold=r.featured)
            for t, r in zip(rates, placed, strict=True)
            if r.sessions
        )
        if agents
        else 0.0
    )
    right = WIDTH - MARGIN
    if agents:
        length = right - columns.plot - room_a - PANEL_GAP - room_b
        a_end = columns.plot + round(PANEL_SHARE * length)
        b_start, b_end = a_end + room_a + PANEL_GAP, right - room_b
    else:
        a_end = b_start = b_end = right - room_a

    measured = [p95 for p95 in p95s if p95 is not None]
    fence = _fence([g["v2v_p95_ms"] for g in groups if not g["failed_turns"]])
    shown = [v for v in measured if (fence is None or v <= fence) and v < REPLY_TIMEOUT_MS]
    top_ms = max([*shown, *([slo] if slo else []), 1000.0])
    marks_a, labels_a = _marks(top_ms / 1000, a_end - columns.plot)
    a = Scale(0, marks_a[-1] * 1000, columns.plot, a_end)
    over = [p95 is not None and p95 > a.hi for p95 in p95s]
    tokens = [
        g["background_tokens_per_s"] for g, row in zip(groups, placed, strict=True) if row.sessions
    ]
    marks_b, labels_b = _marks(max([*tokens, 1.0]), b_end - b_start) if agents else ([0.0], ["0"])
    b = Scale(0, marks_b[-1] or 1.0, b_start, b_end)

    title = "Voice-to-voice p95 and the agents' token rate" if agents else "Voice-to-voice p95"
    s = Chart(WIDTH, 400, f"{title}: {workload}", "")
    y = 30.0
    s.text(MARGIN, y, f"Workload: {workload}")
    entries = []
    if any(row.featured for row in placed):
        entries.append((_fill_key(BAR_FILLS["blue"]), DEPLOY_KEY))
    if any(row.receded for row in placed):
        fills = dict.fromkeys(RECEDED[row.hue] for row in placed if row.receded)
        entries.append((_fill_key(*fills), RECEDED_KEY))
    if any(over):
        entries.append((cut_key(), CUT_KEY))
    if entries:
        y = legend(s, MARGIN, y + 30, entries, width=WIDTH - 2 * MARGIN)
    y += 42
    s.text(a.start, y, "(a) Voice-to-voice p95" if agents else "Voice-to-voice p95", bold=True)
    if agents:
        s.text(b.start, y, "(b) Agents' token rate", bold=True)
    top = y + HEAD
    centres, bottom = _set_rows(s, blocks, columns, top)
    last = x_axis(s, a, [m * 1000 for m in marks_a], labels_a, top, bottom, title="Time (s)")
    if agents:
        x_axis(s, b, marks_b, labels_b, top, bottom, title="Tokens/s over the calls")
    slo_x = None if slo is None else a(slo)
    values = []
    for k, (centre, row) in enumerate(zip(centres, placed, strict=True)):
        if p95s[k] is not None:
            end = a(min(p95s[k], a.hi))
            _row_bar(s, a.start, centre, end - a.start, row, cut=over[k])
            x = end + VALUE_GAP
            if (
                slo_x is not None
                and x - 5 <= slo_x <= x + measure(times[k], VALUE, bold=row.featured) + 5
            ):
                x = slo_x + VALUE_GAP  # past the SLO's rule, never across it
            values.append((x, centre, times[k], row, row.ink))
        if row.sessions:
            end = b(groups[k]["background_tokens_per_s"])
            _row_bar(s, b.start, centre, end - b.start, row)
            values.append((end + VALUE_GAP, centre, rates[k], row, row.ink))
        elif agents:
            values.append((b.start + VALUE_GAP, centre, rates[k], row, "soft"))
    if slo_x is not None:
        s.vrule(slo_x, top, bottom, "slo")
        name = f"SLO {_seconds(slo)} s"
        x = max(slo_x, a.start + SLO_CLEAR + measure(name, NOTE) / 2)
        s.text(x, top - 8, name, size=NOTE, anchor="middle", cls="red tint", halo=True)
    for x, centre, text, row, ink in values:
        s.text(x, centre + 7, text, size=VALUE, cls=ink, halo=True, bold=row.featured)
    notes = []
    if any(len(g["runs"]) > 1 for g in groups):
        notes.append("A row with several runs gives the median of its runs.")
    if slo is None:
        notes.append(f"No point of voice alone with {rows[0]['callers']} callers sets a voice SLO.")
    else:
        notes.append("The SLO is the p95 of voice alone with as many callers, plus 250 ms.")
    _foot(s, last, notes)
    return s.save(path)


# The stage waterfall -----------------------------------------------------------------------------

STAGES = (*STACKED, ("transport_ms", "Transport and playout"))
"""The record's stages from the end of the caller's speech, each its column and its name."""
STAGE_FILLS = ("endpointing", "wait", "tts", "transport")
"""Each stage's fill, the main report's stage figure's (the charts' STAGES): the reply's first
sentence is the LLM's stage and where a shared GPU loses its time, in the red of that figure's
wait for the first token, which it holds together with the decode the record does not part from
it; endpointing, TTS and transport in their grays."""


def plot_waterfall(cells: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """waterfall.svg: a row per configuration whose bar stacks the means of its stages from the end
    of the caller's speech (STACKED, then transport and playout), so that its length is the mean
    V2V, printed at its end; each run of rows of one workload under the workload's name, each row
    with its agent sessions, and the configuration to deploy in semibold, which a note at the foot
    says where a row is it."""
    workloads: list[tuple[str, list[Mapping[str, Any]]]] = []
    for c in cells:
        if not workloads or workloads[-1][0] != c["workload"]:
            workloads.append((c["workload"], []))
        workloads[-1][1].append(c)
    deploy = [
        featured
        for _, members in workloads
        for featured in _deployed([(c["arm"], c["engine"]) for c in members])
    ]
    names = [label(c["label"]) for c in cells]
    limit = _names_limit(list(zip(names, deploy, strict=True)))
    placed = [
        _Row(
            _lines(name, featured, limit),
            ", ".join(_runs(c["runs"])),
            c["sessions"],
            featured=featured,
        )
        for c, name, featured in zip(cells, names, deploy, strict=True)
    ]
    blocks, count = [], 0
    for workload, members in workloads:
        blocks.append(([workload], placed[count : count + len(members)]))
        count += len(members)
    columns = _columns(blocks)
    totals = [f"{_seconds(c['v2v_mean_ms'])} s" for c in cells]
    room = VALUE_GAP + max(
        measure(t, VALUE, bold=row.featured) for t, row in zip(totals, placed, strict=True)
    )
    end = WIDTH - MARGIN - room
    marks, texts = _marks(max(c["v2v_mean_ms"] for c in cells) / 1000, end - columns.plot)
    scale = Scale(0, marks[-1] * 1000, columns.plot, end)

    s = Chart(WIDTH, 400, "The voice turn by stage", "")
    entries = [(stage_key(fill), name) for (_, name), fill in zip(STAGES, STAGE_FILLS, strict=True)]
    legend(s, MARGIN, 30, entries, width=WIDTH - 2 * MARGIN)
    top = 72.0
    centres, bottom = _set_rows(s, blocks, columns, top)
    last = x_axis(
        s,
        scale,
        [m * 1000 for m in marks],
        texts,
        top,
        bottom,
        title="Mean time from the end of the caller's speech (s)",
    )
    for centre, c, total, row in zip(centres, cells, totals, placed, strict=True):
        parts = [
            (scale(max(0.0, c[key])) - scale.start, fill)
            for (key, _), fill in zip(STAGES, STAGE_FILLS, strict=True)
        ]
        s.stack(scale.start, centre - BAR / 2, parts, BAR)
        length = sum(width for width, _ in parts)
        s.marks.append(
            Box(scale.start, centre - BAR / 2, scale.start + length, centre + BAR / 2, "a bar")
        )
        s.text(
            scale.start + length + VALUE_GAP,
            centre + 7,
            total,
            size=VALUE,
            halo=True,
            bold=row.featured,
        )
    notes = ["Each bar is the mean over the replies of its row's runs."]
    if any(row.featured for row in placed):
        notes.append(BOLD_NOTE)
    _foot(s, last, notes)
    return s.save(path)


# The README's voice-p95.svg ----------------------------------------------------------------------


def _workload(groups: Sequence[Mapping[str, Any]]) -> str:
    """The figure's workload, from its records: the calls, the agents and, for one group, the GPU."""
    callers, sessions = groups[0]["callers"], groups[0]["sessions"]
    calls = (
        f"{NUMBERS[callers]} voice calls" if callers < len(NUMBERS) else f"{callers} voice calls"
    )
    title = f"{calls} beside {sessions} coding-agent sessions"
    return title + (f" on one {groups[0]['gpu']}" if len(groups) == 1 else "")


def _title(groups: Sequence[Mapping[str, Any]]) -> str:
    """The figure's claim, from its records: how near its group's SLO the configuration to deploy brings the voice
    tail on each card, as far over it as the furthest, rounded up to the hundredth of a second."""
    featured = [
        (configuration["v2v_p95_ms"], g["slo"])
        for g in groups
        for configuration, spec in g["rows"]
        if spec.get("featured") and g["slo"]
    ]
    if not featured:
        return _workload(groups)
    where = "each card's SLO" if len(groups) > 1 else "the SLO"
    over = max(p95 - slo for p95, slo in featured)
    if over <= 0:
        return f"The configuration to deploy keeps the voice tail within {where}"
    return f"The configuration to deploy brings the voice tail within {math.ceil(over / 10) / 100:.2f} s of {where}"


def plot_voice(groups: Sequence[Mapping[str, Any]], path: Path, narrow: Path) -> list[Path]:
    """voice-p95.svg and its phone layout: each group's configurations (``rows``: the
    configuration and its row of figure.toml) as rows of three panels, the V2V p95 against the
    group's voice SLO (``slo`` ms), the agents' tokens/s over the fixed window with each one's
    share of the group's first row, and the agents' TTFT p99, each the median over the
    configuration's runs. The title states how near the SLO the configuration to deploy comes. A group is the
    configurations of one GPU (``gpu``); with several, each is headed by its ``title``. The row to
    deploy (``featured``) is the saturated blue, the change's other rows (``change``) pale blue,
    the rest gray. ``callers`` and ``sessions`` are the groups' workload, which the title names. A row that
    figure.toml gives no label takes its point's from the report's label table, and its note too unless the row
    gives one (:func:`_labelled`)."""
    groups = _labelled(groups)
    p95_top = max(
        [g["slo"] or 0.0 for g in groups] + [c["v2v_p95_ms"] for g in groups for c, _ in g["rows"]]
    )
    tokens = max([c["fixed_window_tokens_per_s"] or 0.0 for g in groups for c, _ in g["rows"]])
    ttft = max([c["agent_ttft_p99_ms"] or 0.0 for g in groups for c, _ in g["rows"]])
    title = _title(groups)
    description = _describe(groups)
    tops = (p95_top, max(tokens, 1.0), max(ttft, 1000.0))
    return [
        _voice_wide(groups, path, title, description, *tops),
        _voice_narrow(groups, narrow, title, description, *tops),
    ]


def _labelled(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The groups with each row's label and note: figure.toml's where it gives them, else the label table's for the
    row's point (:data:`flatshare_bench.labels.LABELS`), so that a configuration reads the same in this figure
    as in every other."""
    found = []
    for g in groups:
        rows = []
        for configuration, spec in g["rows"]:
            if "label" not in spec:
                name, note = labels.label(spec["point"])
                spec = {**spec, "label": name, "note": spec.get("note", note)}
            rows.append((configuration, spec))
        found.append({**g, "rows": rows})
    return found


def _bar(spec: Mapping[str, Any]) -> str:
    return "hi" if spec.get("featured") else "soft" if spec.get("change") else "mid"


DEPLOY_NOTE = "The saturated blue bar is the configuration to deploy."
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


def _ttft(configuration: Mapping[str, Any]) -> str:
    return f"{configuration['agent_ttft_p99_ms'] / 1000:.0f}"


def _describe(groups: Sequence[Mapping[str, Any]]) -> str:
    sentences = []
    for g in groups:
        head = f"{g['gpu']}: " if len(groups) > 1 else ""
        rows = "; ".join(
            f"{spec['label']}{', ' + spec['note'] if spec.get('note') else ''}, "
            f"{_seconds(c['v2v_p95_ms'])} s, {c['fixed_window_tokens_per_s']:,.0f} tokens/s"
            f" ({_share(g, c)} of the first row's) and a TTFT p99 of {_ttft(c)} s over {len(c['runs'])} runs"
            for c, spec in g["rows"]
        )
        slo = f" against an SLO of {_seconds(g['slo'])} s" if g["slo"] else ""
        sentences.append(
            f"{head}voice-to-voice p95{slo}, the agents' tokens/s and their TTFT p99: {rows}."
        )
    return " ".join(sentences)


def _voice_wide(groups, path, title, description, p95_top, tokens_top, ttft_top) -> Path:
    right = 238
    a, b, c = (262, 300), (632, 150), (930, 110)  # each panel's start and length
    marks_a, marks_b = ticks(p95_top / 1000, a[1]), ticks(tokens_top, b[1])
    marks_c = ticks(ttft_top / 1000, c[1])
    top_a, top_b, top_c = marks_a[-1] * 1000, marks_b[-1], marks_c[-1] * 1000
    s = Figure(WIDTH, 400, title, description)
    s.text(WIDTH / 2, 38, title, size=TITLE, anchor="middle")
    s.text(WIDTH / 2, 64, _workload(groups), size=NOTE, anchor="middle", cls="soft")
    for (x0, length), head, sub in (
        (a, "Voice-to-voice p95", "seconds • lower is better"),
        (b, "Agents' throughput", "tokens/s, % of the baseline"),
        (c, "Agents' TTFT p99", "seconds • lower is better"),
    ):
        _panel_heads(s, x0 + length / 2, 104, head, sub)
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
            lines = wrap(spec["label"], right - 24, VALUE)
            for k, line in enumerate(lines):
                s.text(right, y + 12 + 22 * k, line, size=VALUE, anchor="end")
            extra = 22 * (len(lines) - 1)
            s.text(
                right,
                y + 34 + extra,
                _note(configuration, spec),
                size=NOTE,
                anchor="end",
                cls="soft",
            )
            bars.append((y, g, configuration, spec))
            last = y
            y += pitch + extra
        spans.append((first, last + thick + 6, g))
        y += 10
    bottom = y - 10 + 2
    for (x0, length), marks in ((a, marks_a), (b, marks_b), (c, marks_c)):
        _axis(s, x0, length, marks, top + 8, bottom)
    values = []
    for bar_y, g, configuration, spec in bars:
        p95 = configuration["v2v_p95_ms"]
        end = a[0] + a[1] * p95 / top_a
        s.hbar(a[0], bar_y, end - a[0], thick, _bar(spec))
        values.append((end + 8, bar_y + 19, _seconds(p95), ""))
        tokens = configuration["fixed_window_tokens_per_s"]
        if tokens is None:
            values.append((b[0] + 8, bar_y + 19, "\N{EN DASH}", "soft"))
        else:
            end = b[0] + b[1] * tokens / top_b
            s.hbar(b[0], bar_y, end - b[0], thick, _bar(spec))
            values.append((end + 8, bar_y + 19, f"{tokens:,.0f}", ""))
            if share := _share(g, configuration):
                width = measure(f"{tokens:,.0f}", NOTE)
                values.append((end + 8 + width + 8, bar_y + 19, share, "soft"))
        if configuration["agent_ttft_p99_ms"] is None:
            values.append((c[0] + 8, bar_y + 19, "\N{EN DASH}", "soft"))
        else:
            end = c[0] + c[1] * configuration["agent_ttft_p99_ms"] / top_c
            s.hbar(c[0], bar_y, end - c[0], thick, _bar(spec))
            values.append((end + 8, bar_y + 19, _ttft(configuration), ""))
    for k, (first, last, g) in enumerate(spans):
        if g["slo"]:
            x = a[0] + a[1] * g["slo"] / top_a
            label_y = top if k == 0 else first - 8
            s.vrule(x, label_y + 8, last, "slo")
            s.text(
                x,
                label_y,
                f"SLO {_seconds(g['slo'])} s",
                size=NOTE,
                anchor="middle",
                cls="red tint",
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


def _voice_narrow(groups, path, title, description, p95_top, tokens_top, ttft_top) -> Path:
    s = Figure(NARROW, 400, title, description)
    cx = NARROW / 2
    y = 32
    for line in wrap(title, NARROW - 32, TITLE):
        s.text(cx, y, line, size=TITLE, anchor="middle")
        y += 29
    for line in wrap(_workload(groups), NARROW - 32, NOTE):
        s.text(cx, y - 4, line, size=NOTE, anchor="middle", cls="soft")
        y += 24
    x0, length = 18, 296
    marks_a, marks_b = ticks(p95_top / 1000, length), ticks(tokens_top, length)
    marks_c = ticks(ttft_top / 1000, length)

    def panel(
        y: float, head: str, sub: str, marks: list[float], scale: float, show, *, slos: bool = False
    ) -> float:
        _panel_heads(s, cx, y + 20, head, sub)
        y += 62
        top = marks[-1] * scale
        thick = 20
        for g in groups:
            if len(groups) > 1:
                s.text(x0, y + 16, g["title"], size=NOTE, cls="mute")
                y += 28
            slo = g["slo"] if slos else None
            if slo:
                s.text(
                    x0 + length * slo / top,
                    y + 14,
                    f"SLO {_seconds(slo)} s",
                    size=NOTE,
                    anchor="middle",
                    cls="red tint",
                )
                y += 22
            for configuration, spec in g["rows"]:
                name = s.text(x0, y + 18, spec["label"], size=VALUE)
                note = spec.get("note", "")
                if note and name.x1 + 7 + measure(note, NOTE) <= NARROW - 18:
                    s.text(name.x1 + 7, y + 18, note, size=NOTE, cls="soft")
                elif note:
                    y += 22
                    s.text(x0, y + 18, note, size=NOTE, cls="soft")
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

    def ttft(g, configuration):
        amount = configuration["agent_ttft_p99_ms"]
        return None if amount is None else (amount, [(_ttft(configuration), "")])

    y = panel(
        y + 6, "Voice-to-voice p95", "seconds • lower is better", marks_a, 1000, p95, slos=True
    )
    y = panel(
        y + 10,
        "Agents' throughput",
        "tokens/s, % of the baseline • higher is better",
        marks_b,
        1,
        tokens,
    )
    y = panel(y + 10, "Agents' TTFT p99", "seconds • lower is better", marks_c, 1000, ttft)
    y += 16  # the notes' first baseline as far under the ticks as the wide figure's
    for note in _voice_notes(groups):
        for line in wrap(note, NARROW - 36, NOTE):
            s.text(18, y, line, size=NOTE, cls="soft")
            y += 24
        y += 6
    s.height = y - 10
    return s.save(path)
