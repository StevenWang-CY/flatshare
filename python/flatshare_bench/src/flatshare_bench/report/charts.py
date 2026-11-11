"""The chart parts the report's figures share, on :mod:`flatshare_bench.figure_style`: value scales and their axes,
flat bars with their runs' spread, stacked stage bars, filled dots, traces, ECDF steps and legends.

The report's figures keep one encoding per meaning. A configuration has one hue in every figure,
flatshare_bench.labels' MARKS: blue is the change, saturated for the configuration to deploy and soft for the change's
other configurations; gray is vLLM without the change, tuned or stock, and every comparator; the stronger gray is voice
alone, the reference; red is what the voice loses, the SLO's dashed rule and a turn's wait. A quantity is a flat
mark with its value beside it: a bar from zero, a stacked bar's segment, a filled dot, or a line that carries its name
at its end. The spread of a bar's runs is one thin capped line over the bar's end. A bar is hatched on the page where
what it measures misses the requirement it is read against. The stage bars stack the voice turn's stages in their
order, the two the engine's scheduler sets in red (the wait for the LLM's first token strong, the decode of the
reply's first sentence pale) and the rest in grays of distinct lightness, so that neighbours stay apart in grayscale
and for a color-blind reader. Titles, axis titles and a key's entries are set in sentence case, and a panel's title
names its quantity; what a mark means and what a figure covers is its caption's.

:class:`Chart` adds these parts' classes to the figure's style and, to :meth:`Figure.check`'s refusals, a label over a
data mark.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from flatshare_bench.figure_style import (
    BAR_CORNER,
    NOTE,
    Box,
    Figure,
    _segment_hits_box,
    fmt,
    measure,
)

MARK = 5.0
"""A dot's radius."""
MARK_LINE = 1.4
"""An open marker's outline."""
SPREAD_CAP = 3.5
"""A spread's cap: half its length across the bar."""
SPREAD_MIN = 8.0
"""The least width at which a spread is drawn: under it the two caps close into one glyph, and the runs, which then
agree to within a few percent of the axis, stand as the bar's end."""
STAGES = ("endpointing", "wait", "decode", "tts", "transport")
"""The voice turn's stages from the end of the caller's speech, in their order, each a stacked bar's segment."""
PALETTE = {
    "stage-endpointing": ("#d1d5d9", "#40464d"),
    "stage-wait": ("#e06a64", "#af3c3a"),
    "stage-decode": ("#fdd0cb", "#652b28"),
    "stage-tts": ("#b5bfcb", "#535c66"),
    "stage-transport": ("#e4e6ea", "#30353b"),
}
"""The stage fills the charts add to figure_style's palette, light and dark, off the same ladder (tools: the wait is
red's mark tone and the decode its wash; endpointing, TTS and transport are grays of three lightnesses). Ink sets a
stage's value inside the wait at 4.5:1 or more in both themes."""
HUES = ("gray", "blue", "sky", "green", "ink", "red")
"""The hues a mark takes: figure_style's gray, blue, green and red, the charts' sky (the softer blue of the
change's other configurations) and ink."""


def _variables(theme: int) -> str:
    return ";".join(f"--{name}:{colors[theme]}" for name, colors in PALETTE.items())


CHART_CSS = (
    f"svg{{{_variables(0)}}}@media(prefers-color-scheme:dark){{svg{{{_variables(1)}}}}}"
    ".sky{--c:var(--blue-m);--m:var(--blue-m);--w:var(--blue-w);--a:var(--blue-a)}"
    ".ink{--c:var(--ink)}.soft-ink{--c:var(--soft)}"
    f".mark{{fill:var(--page);stroke:var(--c);stroke-width:{MARK_LINE}}}"
    ".mark-fill{fill:var(--c);stroke:var(--page);stroke-width:1.4}.gray.mark-fill{fill:var(--m)}"
    ".trace{fill:none;stroke:var(--c);stroke-width:2;stroke-linejoin:round}"
    ".trace.thick{stroke-width:2.8}.trace.dashed{stroke-dasharray:7 5}"
    ".trace.dotted{stroke-width:2.4;stroke-dasharray:0 5.5;stroke-linecap:round}"
    ".hair{fill:none;stroke:var(--axis);stroke-width:1}"
    ".ref{fill:none;stroke:var(--c);stroke-width:1.4}"
    ".range{fill:none;stroke:var(--c);stroke-width:1.4;stroke-linecap:round}"
    ".band{fill:var(--a)}"
    ".arrow{fill:none;stroke:var(--c);stroke-width:1.4}.arrow-head{fill:var(--c)}"
    + "".join(f".s-{stage}{{fill:var(--stage-{stage})}}" for stage in STAGES)
)


class Chart(Figure):
    """A figure_style Figure whose style also holds the charts' classes (CHART_CSS), and whose check also refuses a
    label set over a data mark, or across a reference rule."""

    def __init__(self, width: float, height: float, title: str, description: str):
        super().__init__(width, height, title, description)
        self.marks: list[Box] = []
        """The data marks' extents, which no label may cover."""
        self.covers: list[tuple[Box, int]] = []
        """Opaque fills (a stacked bar's segments) and where each is painted among the items: a label wholly inside
        one may sit where a rule painted before it runs, which the fill hides."""
        self.painted: list[int] = []
        """Where each of the rules is painted among the items."""
        self.kinds: list[str] = []
        """Each rule's class: a gridline ("grid") may pass behind a label set on a halo, and nothing else may."""

    def rule(self, x0: float, x1: float, y: float, cls: str = "axis") -> None:
        self.painted.append(len(self.items))
        self.kinds.append(cls)
        super().rule(x0, x1, y, cls)

    def vrule(self, x: float, y0: float, y1: float, cls: str = "axis") -> None:
        self.painted.append(len(self.items))
        self.kinds.append(cls)
        super().vrule(x, y0, y1, cls)

    def ref(self, a: tuple[float, float], b: tuple[float, float], cls: str) -> None:
        """A reference line drawn as a trace (a dotted or dashed level), a rule no label may cross."""
        self.trace([a, b], cls, wire=False)
        self.rules.append((a, b))
        self.painted.append(len(self.items) - 1)
        self.kinds.append(cls)

    def render(self) -> str:
        return super().render().replace("</style>", CHART_CSS + "</style>", 1)

    def check(self) -> None:
        """figure_style's checks, except that a halo lets a label sit on a gridline alone, never across an axis or
        a reference rule (an SLO, a requirement, a level), and a rule may pass under a label inside a segment painted
        after the rule, which hides it; and a label over a data mark is refused too."""
        rules, self.rules = self.rules, []
        try:
            super().check()
        finally:
            self.rules = rules
        for (a, b), at, kind in zip(rules, self.painted, self.kinds, strict=True):
            for box in self.boxes:
                if (box.halo and kind == "grid") or not _segment_hits_box(a, b, box, pad=1.0):
                    continue
                if not any(_inside(box, cover) and at < over for cover, over in self.covers):
                    raise ValueError(f"a rule crosses the text {box.label!r} near {a}")
        for box in self.boxes:
            for mark in self.marks:
                if box.overlaps(mark, pad=1.0):
                    raise ValueError(f"a label covers a mark: {box.label!r} near {mark.label}")

    # Marks ------------------------------------------------------------------------------------------------------

    def mark(
        self,
        x: float,
        y: float,
        hue: str,
        shape: str = "circle",
        *,
        filled: bool = True,
        r: float = MARK,
        name: str = "a mark",
    ) -> None:
        """A dot centred at (x, y) in ``hue``, ringed in the page's color so that two that touch stay two; ``shape``
        one of SHAPES. Its extent joins the marks no label may cover."""
        if hue not in HUES:
            raise ValueError(f"a mark's hue is one of {HUES}, not {hue!r}")
        draw, extent = SHAPES[shape]
        cls = "mark-fill" if filled else "mark"
        self.path(draw(x, y, r), f"{hue} {cls}")
        half = extent * r + (MARK_LINE / 2 if not filled else 0.6)
        self.marks.append(
            Box(x - half, y - half, x + half, y + half, f"{name} at ({x:.0f}, {y:.0f})")
        )

    def trace(self, points: Sequence[tuple[float, float]], cls: str, *, wire: bool = True) -> None:
        """A data line through ``points`` in the classes ``cls`` (a hue and "trace", or "hair"); with ``wire``, a
        line no label may cross. A level or upright segment is written as one, which halves an ECDF's steps."""
        d = f"M{fmt(points[0][0])} {fmt(points[0][1])}"
        for (px, py), (x, y) in itertools.pairwise(points):
            if fmt(y) == fmt(py):
                d += f"H{fmt(x)}"
            elif fmt(x) == fmt(px):
                d += f"V{fmt(y)}"
            else:
                d += f"L{fmt(x)} {fmt(y)}"
        self.path(d, cls)
        if wire:
            self.wires.append(([(float(x), float(y)) for x, y in points], ()))

    def arrow(
        self, a: tuple[float, float], b: tuple[float, float], hue: str, *, gap: float = 9
    ) -> None:
        """An arrow from ``a`` to ``b`` in ``hue``, its head ``gap`` short of ``b`` (a marker's centre)."""
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        tip = (b[0] - ux * gap, b[1] - uy * gap)
        start = (a[0] + ux * gap, a[1] + uy * gap)
        head, half = 9.0, 4.2
        base = (tip[0] - ux * head, tip[1] - uy * head)
        self.path(f"M{fmt(start[0])} {fmt(start[1])}L{fmt(base[0])} {fmt(base[1])}", f"{hue} arrow")
        corners = [
            tip,
            (base[0] - uy * half, base[1] + ux * half),
            (base[0] + uy * half, base[1] - ux * half),
        ]
        self.path(
            "M" + "L".join(f"{fmt(px)} {fmt(py)}" for px, py in corners) + "Z", f"{hue} arrow-head"
        )
        self.wires.append(([start, tip], ()))

    def stack(
        self, x: float, y: float, parts: Sequence[tuple[float, str]], thickness: float
    ) -> list[float]:
        """A stacked horizontal bar of the voice turn's stages: ``parts`` are (length, stage) from ``x``, each stage
        one of STAGES; the last segment's free end is rounded. Returns each segment's start."""
        starts = []
        for k, (length, stage) in enumerate(parts):
            if stage not in STAGES:
                raise ValueError(f"a stage is one of {STAGES}, not {stage!r}")
            starts.append(x)
            self._segment(x, y, length, thickness, f"s-{stage}", k, len(parts))
            self.covers.append((Box(x, y, x + length, y + thickness, stage), len(self.items) - 1))
            x += length
        return starts

    def span(self, x0: float, x1: float, y: float, *, name: str = "a spread") -> None:
        """The spread of a bar's runs, from the least to the greatest: one thin level line in ink over the bar's
        end with a cap at each end, as an error bar is drawn, in place of a tick a run. Its extent joins the marks
        no label may cover."""
        cap = SPREAD_CAP
        self.path(
            f"M{fmt(x0)} {fmt(y - cap)}V{fmt(y + cap)}M{fmt(x0)} {fmt(y)}H{fmt(x1)}"
            f"M{fmt(x1)} {fmt(y - cap)}V{fmt(y + cap)}",
            "soft-ink range",
        )
        self.marks.append(Box(x0 - 1, y - cap - 1, x1 + 1, y + cap + 1, name))

    def bar(
        self,
        x: float,
        y: float,
        length: float,
        thickness: float,
        hue: str,
        *,
        hatched: bool = False,
        name: str = "a bar",
    ) -> None:
        """A horizontal bar from the axis at ``x``, its top at ``y``, its free end rounded: pale in ``hue`` (gray or
        blue), or ``hatched`` in it on the page, for what misses its requirement. Its extent joins the marks no label
        may cover."""
        if hue not in BAR_FILLS:
            raise ValueError(f"a bar's hue is one of {tuple(BAR_FILLS)}, not {hue!r}")
        self.hbar(x, y, length, thickness, "open" if hatched else BAR_FILLS[hue])
        if hatched:
            self.path(_hatch(x, y, length, thickness, BAR_CORNER), f"{hue} hatch")
        self.marks.append(Box(x, y, x + length, y + thickness, name))


BAR_FILLS = {"gray": "lo", "blue": "hi", "sky": "soft", "ink": "mid"}
"""A bar's fill by its hue: the soft gray for a comparator, the saturated blue for the configuration to deploy, the
soft blue for the change's other configurations, and the stronger gray for voice alone, the reference."""
HATCH_STEP = 6.0
"""Between a hatched bar's lines, along it."""


def _hatch(x: float, y: float, length: float, thickness: float, corner: float) -> str:
    """A bar's hatching: rising diagonals HATCH_STEP apart along it, each clipped to the bar, its free end's corners
    cut where they round, so that no line leaves the outline."""
    x1, y1 = x + length, y + thickness
    d = ""
    # Each line is u + v = c; along it, inside the bar and short of its upper corner's cut, u runs over [lo, hi].
    c = x + y + HATCH_STEP
    while c < x1 + y1 - corner:
        lo, hi = max(x, c - y1), min(x1, c - y, (x1 - corner - y + c) / 2)
        if hi - lo > 0.5:
            d += f"M{fmt(lo)} {fmt(c - lo)}L{fmt(hi)} {fmt(c - hi)}"
        c += HATCH_STEP
    return d


def _inside(box: Box, cover: Box) -> bool:
    return cover.x0 <= box.x0 and box.x1 <= cover.x1 and cover.y0 <= box.y0 and box.y1 <= cover.y1


def _circle(x: float, y: float, r: float) -> str:
    return (
        f"M{fmt(x - r)} {fmt(y)}A{fmt(r)} {fmt(r)} 0 1 0 {fmt(x + r)} {fmt(y)}"
        f"A{fmt(r)} {fmt(r)} 0 1 0 {fmt(x - r)} {fmt(y)}Z"
    )


def _dot(x: float, y: float, r: float) -> str:
    return _circle(x, y, 0.62 * r)


SHAPES: dict[str, tuple[Callable[[float, float, float], str], float]] = {
    "circle": (_circle, 1.0),
    "dot": (_dot, 0.62),
}
"""Each marker's path at a radius, and its half extent in radii: a configuration's circle, and the smaller dot of
the build with its switches off (labels.MARKS)."""


# Scales and axes ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scale:
    """A value axis: ``lo`` to ``hi`` onto the canvas from ``start`` to ``end`` (a vertical axis runs from its bottom
    up), linear or logarithmic."""

    lo: float
    hi: float
    start: float
    end: float
    log: bool = False

    def __call__(self, value: float) -> float:
        if self.log:
            low, high = math.log10(self.lo), math.log10(self.hi)
            share = (math.log10(value) - low) / (high - low)
        else:
            share = (value - self.lo) / (self.hi - self.lo)
        return self.start + share * (self.end - self.start)


def x_axis(
    s: Figure,
    scale: Scale,
    marks: Sequence[float],
    labels: Sequence[str],
    top: float | None,
    bottom: float,
    *,
    title: str | None = None,
) -> float:
    """A horizontal value axis under a plot from ``top`` to ``bottom``: a gridline at each mark (the axis itself at
    the scale's start; none without ``top``), the baseline at ``bottom``, each mark's label under it, and ``title``
    centred under them. Returns the lowest baseline it set."""
    for mark, label in zip(marks, labels, strict=True):
        x = scale(mark)
        if top is not None:
            s.vrule(x, top, bottom, "axis" if mark == scale.lo else "grid")
        s.text(x, bottom + 23, label, size=NOTE, anchor="middle", cls="mute")
    s.rule(scale.start, scale.end, bottom)
    if title is None:
        return bottom + 23
    s.text(
        (scale.start + scale.end) / 2, bottom + 50, title, size=NOTE, anchor="middle", cls="soft"
    )
    return bottom + 50


def band_grid(s: Figure, scale: Scale, marks: Sequence[float], top: float, bottom: float) -> None:
    """A horizontal scale's gridlines over one row's band only, the axis at the scale's start: the phone layouts set
    a row's name above its band, clear of them."""
    for mark in marks:
        s.vrule(scale(mark), top, bottom, "axis" if mark == scale.lo else "grid")


def y_axis(
    s: Figure,
    scale: Scale,
    marks: Sequence[float],
    labels: Sequence[str],
    left: float,
    right: float,
    *,
    axis: bool = True,
) -> float:
    """A vertical value axis beside a plot from ``left`` to ``right``: a gridline at each mark (the baseline at the
    scale's start where ``axis``), each mark's label left of the plot, right-aligned. Returns the labels' left edge."""
    edge = left
    for mark, label in zip(marks, labels, strict=True):
        y = scale(mark)
        s.rule(left, right, y, "axis" if mark == scale.lo and axis else "grid")
        box = s.text(left - 8, y + 6, label, size=NOTE, anchor="end", cls="mute")
        edge = min(edge, box.x0)
    s.vrule(left, scale.end, scale.start)
    return edge


def linear_ticks(top: float, step: float) -> tuple[list[float], list[str]]:
    """Ticks from zero to ``top`` every ``step``, and their labels, with as many decimals as the step needs."""
    marks = [step * k for k in range(round(top / step) + 1)]
    digits = next(d for d in range(4) if abs(step * 10**d - round(step * 10**d)) < 1e-9)
    return marks, [f"{mark:,.{digits}f}" for mark in marks]


def seconds(ms: float) -> str:
    """A latency in ms as the report's figures print it: seconds to two decimals."""
    return f"{ms / 1000:.2f}"


def pct(share: float) -> str:
    return f"{share:.0%}"


# Steps ---------------------------------------------------------------------------------------------------------


def ecdf(
    samples: Sequence[float], x: Scale, y: Scale, *, cap: float | None = None
) -> list[tuple[float, float]]:
    """The ECDF of ``samples`` as the corners of its steps on the canvas: from zero at the smallest sample up to one
    at the largest, a sample past ``cap`` standing at it."""
    values = sorted(min(v, cap) if cap is not None else v for v in samples)
    n = len(values)
    points = [(x(values[0]), y(0.0))]
    k = 0
    while k < n:
        j = k
        while j + 1 < n and values[j + 1] == values[k]:
            j += 1
        at = x(values[k])
        points += [(at, points[-1][1]), (at, y((j + 1) / n))]
        k = j + 1
    return points


# Legends --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Key:
    """A legend entry's key: its width, and how it draws at a left edge and the entry's baseline."""

    width: float
    draw: Callable[[Chart, float, float], None]


def mark_key(hue: str, shape: str = "circle", *, filled: bool = True, r: float = MARK) -> Key:
    def draw(s: Chart, x: float, baseline: float) -> None:
        s.mark(
            x + r + 2, baseline - 0.32 * NOTE, hue, shape, filled=filled, r=r, name="a legend's key"
        )

    return Key(2 * r + 4, draw)


def stage_key(stage: str) -> Key:
    def draw(s: Chart, x: float, baseline: float) -> None:
        s._hpath(x, baseline - 0.66 * NOTE, 22, 0.7 * NOTE, f"s-{stage}", 2.0)

    return Key(22, draw)


def hatch_key(hue: str = "gray") -> Key:
    """A hatched bar's key: a short bar hatched in ``hue``, as a bar that misses its requirement is drawn."""

    def draw(s: Chart, x: float, baseline: float) -> None:
        s.bar(x, baseline - 0.66 * NOTE, 22, 0.7 * NOTE, hue, hatched=True, name="a legend's key")

    return Key(22, draw)


def line_key(cls: str, width: float = 32) -> Key:
    """A data line's key: a short level stroke in its classes (a hue and "trace", and its weight or dashes)."""

    def draw(s: Chart, x: float, baseline: float) -> None:
        y = baseline - 0.32 * NOTE
        s.path(f"M{fmt(x)} {fmt(y)}H{fmt(x + width)}", cls)

    return Key(width, draw)


KEY_GAP = 8
"""Between a key and its name."""


def legend_width(
    entries: Sequence[tuple[Key, str]], *, size: float = NOTE, gap: float = 24
) -> float:
    """The width of a legend's entries in one row."""
    return sum(key.width + KEY_GAP + measure(name, size) for key, name in entries) + gap * (
        len(entries) - 1
    )


def legend(
    s: Chart,
    x: float,
    baseline: float,
    entries: Sequence[tuple[Key, str]],
    *,
    width: float,
    size: float = NOTE,
    gap: float = 24,
    pitch: float = 26,
) -> float:
    """Entries from ``x`` at ``baseline``, each a key then its name in ink, in rows that break before an entry that
    would pass ``width``. Returns the last row's baseline."""
    left = x
    for key, name in entries:
        room = key.width + KEY_GAP + measure(name, size)
        if x > left and x + room > left + width:
            x, baseline = left, baseline + pitch
        key.draw(s, x, baseline)
        box = s.text(x + key.width + KEY_GAP, baseline, name, size=size)
        x = box.x1 + gap
    return baseline


# --- v1.27 review fixes (packet FR) ---

BREAK_WIDTH, BREAK_LEAN, BREAK_STEP, BREAK_TAIL = 2.5, 5.0, 6.5, 4.0
"""A cut bar's two breaks: each one's width, how far it leans from its foot to its head, the pitch from one to the
next, and the bar left after the last one, before the scale's end."""


def cut_breaks(s: Figure, end: float, top: float, bottom: float) -> None:
    """The two leaning breaks, in the page's color, that part a bar cut at ``end`` (the scale's end) near that end,
    across the bar from ``top`` to ``bottom``: the bar runs on past the scale, and its value, set after it, says how
    far."""
    for k in range(2):
        foot = end - BREAK_TAIL - BREAK_LEAN - BREAK_WIDTH - k * BREAK_STEP
        s.path(
            f"M{fmt(foot)} {fmt(bottom + 1)}L{fmt(foot + BREAK_LEAN)} {fmt(top - 1)}"
            f"H{fmt(foot + BREAK_LEAN + BREAK_WIDTH)}L{fmt(foot + BREAK_WIDTH)} {fmt(bottom + 1)}Z",
            "page",
        )


def cut_key(bar: str = "lo") -> Key:
    """A cut bar's key: a short bar in the fill ``bar`` (figure_style's BARS) with a square end and the two breaks."""
    width = 30.0

    def draw(s: Chart, x: float, baseline: float) -> None:
        top, thick = baseline - 0.66 * NOTE, 0.7 * NOTE
        s.hbar(x, top, width, thick, bar, cut=True)
        cut_breaks(s, x + width, top, top + thick)

    return Key(width, draw)
