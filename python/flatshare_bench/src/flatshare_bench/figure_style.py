"""Type, palette and drawing primitives for Flatshare's figures: the report's charts
(:mod:`.report.figures`), a record's own (:mod:`.e2.figures`), the method figure and the system
diagrams (:mod:`.diagrams`) and the brand marks (:mod:`.brand`). The README's animation
(``benchmarks/e2/demo.py``) draws with the same palette and type.

Every figure is a self-contained SVG whose labels stay selectable text. Text is set in STIX Two
(SIL Open Font License 1.1, ``fonts/OFL.txt``), and each export embeds a WOFF subset of exactly the
glyphs it draws, so GitHub shows the same type on every platform without fetching anything. Layout
measures the same font files, which lets :meth:`Figure.render` refuse what a reader would trip on:
overlapping labels, a connector across a label, text off the canvas, and an arrow too short to have
a shaft.

The conventions, which every figure here keeps:

- A figure is 1120 units wide and shows at about 880 CSS pixels (one unit is about 0.8 px); its
  phone layout is 420 units wide and shows at about 375.
- Four type sizes, one per role: a chart's title 23, a panel's title and a label 21, a category
  label and a bar's value 20, a note or a tick 18. Text is regular; the semibold sets only a panel's
  title and the name and value of the one row a figure features. Text is ink; the soft gray sets secondary
  text and the mute gray ticks and axis words. All three clear 4.5:1 on the page in both themes.
- Axis titles. An axis's title names its quantity and its unit, the unit in parentheses with
  ", log scale" where the scale is logarithmic: "Time to the first token (ms)", "Voice-to-voice
  p95 (s)". Where a panel's title names the measure, the axis names its kind. A share's percent
  signs and a count's numbers need no unit. No axis says which way is better.
- Four colors, each with one meaning. Blue is the change: the configuration to deploy in the
  saturated blue, the change's other configurations in a softer one, and the component that
  carries it. Green is the voice turn itself: its KV blocks, its steps, its calls. Gray is
  everything else: the agents, stock and tuned vLLM, every comparator. Red is what the
  voice loses: the SLO a result is read against, the wait, a miss.
- Marks are flat. A bar has no outline, a square end on its axis and a slightly rounded free
  end, and its value beside it; a stacked bar's segments are parted by a seam of the page's
  color. A component is a pale fill inside a thin outline of the same hue; a frame is a
  hairline, and a dash means "not active". An arrow is green where it carries the voice turn and
  gray everywhere else.
- Figures follow the reader's theme through ``prefers-color-scheme`` and paint their own page, so
  a figure stays legible where the system's theme and the page's differ.
"""

from __future__ import annotations

import base64
import io
import itertools
import math
import os
import re
import sys
from dataclasses import dataclass
from functools import cache
from html import escape
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

FONT_DIR = Path(__file__).resolve().parent / "fonts"

WIDTH = 1120
NARROW = 420

# Type, in canvas units.
TITLE = 23  # a chart's title
LABEL = 21  # a panel's title, a label, an operation
VALUE = 20  # a category label, a row's name, a bar's value
NOTE = 18  # a note, a subtitle, a tick

# Strokes, in canvas units.
HAIR = 1.0  # gridlines, frames, dividers, leaders
THIN = 1.4  # axes, a threshold, an open bar's outline
SEAM = 2.0  # the page-colored gap between a stacked bar's segments, and between two cells
BAR_CORNER = 2.0  # a bar's rounded free end, and a cell's corner
OUTLINE = 1.4  # a component's outline, in its fill's hue
CORNER = 5.0  # a component's corner
FLOW = 2.0  # dataflow, and a line a chart traces
FLOW_HEAD = 8.5  # dataflow's head: its length
FLOW_HALF = 3.7  # and its half width
FLOW_GAP = 2.5  # the clearance between a head and its target's edge
MIN_SHAFT = 2.2  # an arrow's last segment, at least this many head lengths

FONT_NOTICE = (
    " Text is set in subsets of STIX Two Text and STIX Two Math, SIL Open Font License 1.1 "
    "(python/flatshare_bench/src/flatshare_bench/fonts/OFL.txt). "
)
TEXT_FAMILY = "Flatshare Text"
MATH_FAMILY = "Flatshare Math"
FACES = {
    "regular": ("STIXTwoText-Regular.ttf", TEXT_FAMILY, "normal", 400),
    "italic": ("STIXTwoText-Italic.ttf", TEXT_FAMILY, "italic", 400),
    "bold": ("STIXTwoText-SemiBold.ttf", TEXT_FAMILY, "normal", 600),
    "math": ("STIXTwoMath-Regular.ttf", MATH_FAMILY, "normal", 400),
}

# The palette. Ink and page are GitHub's, so a figure sits naturally in either theme. Every other
# color comes off one ladder in OKLCH: each hue takes the same lightness at each of five steps, and
# its chroma falls by the same share from step to step, so two hues at one step weigh the same.
# A hue's own name is its line, which also sets text (4.5:1 on the page); "-m" is a mark beside it
# (3:1); "-s" is the soft fill of a quantity that is not the featured one, a bar or a cell; "-w" is
# a wash pale enough to hold ink; "-a" is the tint inside an object's outline.
LIGHT = {
    "ink": "#1f2328",
    "soft": "#3f444b",
    "mute": "#666c75",
    "page": "#ffffff",
    "grid": "#edeff1",
    "axis": "#8f969f",
    "rule": "#d6d9de",
    "flow": "#6b747f",
    "box": "#f6f8fa",
    "box-line": "#a9b0ba",
    "bar-hi": "#2669cd",
    "bar-soft": "#98bcf4",
    "bar-lo": "#b9bec4",
    "bar-mid": "#8d949d",
    "green": "#138159",
    "green-m": "#5fa583",
    "green-s": "#9ac7af",
    "green-w": "#cce3d6",
    "green-a": "#ecf5f0",
    "gray": "#6b747f",
    "gray-m": "#8d949d",
    "gray-s": "#b9bec4",
    "gray-w": "#e0e3e6",
    "gray-a": "#f2f3f5",
    "red": "#c3292e",
    "red-m": "#e06a64",
    "red-s": "#f4a29b",
    "red-w": "#fdd0cb",
    "red-a": "#feeeed",
    "blue": "#2669cd",
    "blue-m": "#6094e4",
    "blue-s": "#98bcf4",
    "blue-w": "#cbdefb",
    "blue-a": "#ecf3fe",
}
DARK = {
    "ink": "#e6edf3",
    "soft": "#cbd0d6",
    "mute": "#9399a1",
    "page": "#0d1117",
    "grid": "#1d2125",
    "axis": "#6e757e",
    "rule": "#31363c",
    "flow": "#8c939c",
    "box": "#181c21",
    "box-line": "#4f565e",
    "bar-hi": "#76abff",
    "bar-soft": "#3e639d",
    "bar-lo": "#565b62",
    "bar-mid": "#7b8189",
    "green": "#6fbe98",
    "green-m": "#508f71",
    "green-s": "#3d6f57",
    "green-w": "#274838",
    "green-a": "#17251e",
    "gray": "#9199a3",
    "gray-m": "#7b8189",
    "gray-s": "#565b62",
    "gray-w": "#3c4046",
    "gray-a": "#1d1e20",
    "red": "#ff7f77",
    "red-m": "#c55b55",
    "red-s": "#9a4541",
    "red-w": "#632d2a",
    "red-a": "#331917",
    "blue": "#76abff",
    "blue-m": "#5280c9",
    "blue-s": "#3e639d",
    "blue-w": "#284065",
    "blue-a": "#172234",
}
HUES = ("green", "gray", "red", "blue")
BARS = ("hi", "soft", "lo", "mid", "open")
"""A chart's bar fills: the configuration to deploy, a sibling of it, a comparison, the reference
a figure measures from (voice alone, the darker gray, so three bars stay apart in grayscale), and
an open bar, outlined on the page, which a chart hatches for an arrangement that misses its
requirement."""
SEGMENTS = {
    "hi": "bar-hi",
    "soft": "bar-soft",
    "lo": "bar-lo",
    "mid": "bar-mid",
    "pale": "gray-w",
    "red": "red-m",
    "red-pale": "red-w",
    "green-pale": "green-w",
    "blue": "blue-m",
    "blue-pale": "blue-w",
}
"""The fills of a stacked bar's segments, each the palette's color it takes."""
KINDS = ("box", "ours", *HUES)
"""What a component is: a plain one, the change's own (blue), or one in a hue."""
TEXT_CLASSES = {"", "soft", "mute", *(f"{hue} tint" for hue in HUES)}
"""The inks a label may take: the three neutrals, and a hue where the hue names an identity."""


def _variables(palette: dict[str, str]) -> str:
    return ";".join(f"--{name}:{color}" for name, color in palette.items())


THEME_CSS = (
    f"svg{{{_variables(LIGHT)}}}@media(prefers-color-scheme:dark){{svg{{{_variables(DARK)}}}}}"
)
BASE_CSS = (
    f"svg{{font-size:{LABEL}px}}"
    f"text{{font-family:'{TEXT_FAMILY}','{MATH_FAMILY}',serif;fill:var(--ink);"
    "font-synthesis:none;font-kerning:normal;white-space:pre}"
    ".i{font-style:italic}.b{font-weight:600}.soft{fill:var(--soft)}.mute{fill:var(--mute)}"
    + "".join(
        f".{h}{{--c:var(--{h});--m:var(--{h}-m);--s:var(--{h}-s);--w:var(--{h}-w);--a:var(--{h}-a)}}"
        for h in HUES
    )
    + ".tint{fill:var(--c)}.mid{fill:var(--m)}.wash{fill:var(--w)}.area{fill:var(--a)}"
    ".page{fill:var(--page)}"
    f".grid{{fill:none;stroke:var(--grid);stroke-width:{HAIR}}}"
    f".axis{{fill:none;stroke:var(--axis);stroke-width:{THIN}}}"
    ".axis-head{fill:var(--axis)}"
    f".rule{{fill:none;stroke:var(--rule);stroke-width:{HAIR}}}"
    + "".join(f".g-{bar}{{fill:var(--bar-{bar})}}" for bar in BARS if bar != "open")
    + f".g-open{{fill:var(--page);stroke:var(--bar-mid);stroke-width:{THIN}}}"
    + "".join(f".s-{name}{{fill:var(--{color})}}" for name, color in SEGMENTS.items())
    + ".halo{paint-order:stroke;stroke:var(--page);stroke-width:6px;stroke-linejoin:round}"
    f".slo{{fill:none;stroke:var(--red);stroke-width:{THIN};stroke-dasharray:5 4}}"
    f".panel{{fill:none;stroke:var(--rule);stroke-width:{HAIR}}}"
    f".group{{fill:none;stroke:var(--rule);stroke-width:{HAIR}}}"
    f".box{{fill:var(--box);stroke:var(--box-line);stroke-width:{OUTLINE}}}"
    f".ours{{fill:var(--blue-a);stroke:var(--blue);stroke-width:{OUTLINE}}}"
    f".obj{{fill:var(--a);stroke:var(--c);stroke-width:{OUTLINE}}}"
    f".memory{{fill:var(--box);stroke:var(--rule);stroke-width:{HAIR}}}"
    f".flow{{fill:none;stroke:var(--flow);stroke-width:{FLOW};stroke-linecap:round;"
    "stroke-linejoin:round}"
    ".flow-head{fill:var(--flow)}"
    f".line{{fill:none;stroke:var(--c);stroke-width:{FLOW};stroke-linecap:round;"
    "stroke-linejoin:round}"
    ".line-head{fill:var(--c)}"
    f".leader{{fill:none;stroke:var(--axis);stroke-width:{HAIR};stroke-dasharray:3 3}}"
    f".bracket{{fill:none;stroke:var(--axis);stroke-width:{HAIR};stroke-linejoin:round}}"
    f".hatch{{fill:none;stroke:var(--c);stroke-width:{HAIR}}}"
    ".cell{fill:var(--s)}.cell.strong{fill:var(--m)}.cell.taken{fill:var(--w)}"
    f".cell-empty{{fill:var(--page);stroke:var(--box-line);stroke-width:{HAIR}}}"
)

# Function names that stay upright inside math, as in LaTeX's \operatorname.
FUNCTIONS = ("max", "min", "exp")
# Commands accepted inside $...$; longer names are matched first.
COMMANDS = {
    r"\,": "\N{SIX-PER-EM SPACE}",
    r"\cdots": "⋯",
    r"\cdot": "·",
    r"\infty": "∞",
    r"\leftarrow": "←",
    r"\to": "→",
    r"\times": "\N{MULTIPLICATION SIGN}",
    r"\le": "≤",
    r"\ge": "≥",
    r"\in": "∈",
}
RELATIONS = set("=←→≤≥<>∈")
BINARY = {"+", "\N{MINUS SIGN}", "\N{MIDDLE DOT}", "\N{MULTIPLICATION SIGN}"}
OPENING = set("([{")
REL_SPACE = "\N{FOUR-PER-EM SPACE}"
BIN_SPACE = THIN_SPACE = "\N{SIX-PER-EM SPACE}"
_TOKEN = re.compile(r"\$[^$]*\$|[^$]+")


@cache
def _font(face: str) -> TTFont:
    return TTFont(FONT_DIR / FACES[face][0], recalcTimestamp=False, lazy=False)


@cache
def _advances(face: str) -> tuple[dict[int, float], frozenset[int]]:
    font = _font(face)
    scale = 1 / font["head"].unitsPerEm
    metrics = font["hmtx"].metrics
    cmap = font.getBestCmap()
    return {code: metrics[name][0] * scale for code, name in cmap.items()}, frozenset(cmap)


def _resolve(face: str, char: str) -> str:
    """The face a browser draws ``char`` in, as the CSS fallback does: the span's, else math."""
    if ord(char) in _advances(face)[1]:
        return face
    if ord(char) in _advances("math")[1]:
        return "math"
    raise ValueError(f"no embedded glyph for {char!r} (U+{ord(char):04X})")


@dataclass(frozen=True)
class Span:
    text: str
    face: str = "regular"
    scale: float = 1.0
    shift: float = 0.0  # baseline shift in em; positive moves up


def _group(source: str, i: int) -> tuple[str, int]:
    """A braced group, or a single character, starting at ``i``."""
    if source[i] != "{":
        return source[i], i + 1
    depth, k = 0, i
    while True:
        depth += {"{": 1, "}": -1}.get(source[k], 0)
        if depth == 0:
            return source[i + 1 : k], k + 1
        k += 1


def _math_spans(source: str, scale: float = 1.0, shift: float = 0.0) -> list[Span]:
    """A small LaTeX subset: italic letters, upright function names and digits, true minus signs,
    sub- and superscripts, and TeX's spacing around relations and binary operators."""
    for command in sorted(COMMANDS, key=len, reverse=True):
        source = source.replace(command, COMMANDS[command])
    source = source.replace("-", "\N{MINUS SIGN}")
    atoms: list[tuple[str, list[Span]]] = []
    i = 0
    while i < len(source):
        char = source[i]
        if char == " ":
            i += 1
        elif char in "_^" and i + 1 < len(source):
            inner, i = _group(source, i + 1)
            lift = -0.2 if char == "_" else 0.36
            nested = _math_spans(inner, scale * 0.72, shift + lift * scale)
            if atoms:
                atoms[-1] = (atoms[-1][0], atoms[-1][1] + nested)
            else:
                atoms.append(("ord", nested))
        elif source.startswith("\\mathrm{", i):
            inner, i = _group(source, i + 7)
            atoms.append(("ord", [Span(inner, "regular", scale, shift)]))
        elif word := next((w for w in FUNCTIONS if source.startswith(w, i)), None):
            atoms.append(("op", [Span(word, "regular", scale, shift)]))
            i += len(word)
        else:
            kind = (
                "rel"
                if char in RELATIONS
                else "bin"
                if char in BINARY
                else "punct"
                if char == ","
                else "open"
                if char in OPENING
                else "ord"
            )
            face = "italic" if char.isalpha() else "regular"
            atoms.append((kind, [Span(char, face, scale, shift)]))
            i += 1
    spans: list[Span] = []
    previous = None
    for k, (kind, parts) in enumerate(atoms):
        following = atoms[k + 1][0] if k + 1 < len(atoms) else None
        if kind == "bin" and previous in (None, "rel", "bin", "open", "punct", "op"):
            kind = "ord"  # a sign, not an operator
        space = {"rel": REL_SPACE, "bin": BIN_SPACE}.get(kind)
        if space and previous is not None:
            spans.append(Span(space, "regular", scale, shift))
        elif kind == "op" and previous == "ord":
            spans.append(Span(THIN_SPACE, "regular", scale, shift))
        spans += parts
        if space and following is not None:
            spans.append(Span(space, "regular", scale, shift))
        elif (kind == "op" and following == "ord") or (kind == "punct" and following):
            spans.append(Span(THIN_SPACE, "regular", scale, shift))
        previous = kind
    return spans


def parse(markup: str) -> list[Span]:
    """``$math$`` markup into measured spans; the rest is regular text."""
    spans: list[Span] = []
    for token in _TOKEN.findall(str(markup)):
        if token.startswith("$") and token.endswith("$") and len(token) > 1:
            spans += _math_spans(token[1:-1])
        else:
            spans.append(Span(token))
    merged: list[Span] = []
    for span in spans:
        last = merged[-1] if merged else None
        if last and (last.face, last.scale, last.shift) == (span.face, span.scale, span.shift):
            merged[-1] = Span(last.text + span.text, span.face, span.scale, span.shift)
        else:
            merged.append(span)
    return merged


def measure(markup: str, size: float, *, bold: bool = False) -> float:
    """The advance width of ``markup`` at ``size``, from the embedded fonts' metrics; ``bold`` sets
    its upright text in the semibold."""
    width = 0.0
    for span in parse(markup):
        for char in span.text:
            face = _resolve("bold" if bold and span.face == "regular" else span.face, char)
            width += _advances(face)[0][ord(char)] * size * span.scale
    return width


def wrap(text: str, width: float, size: float) -> list[str]:
    """``text`` in as few lines as fit ``width`` at ``size``, broken at spaces where the lines come
    out most even, so that no line is left with a word or two alone."""
    words = text.split()
    greedy: list[str] = []
    for word in words:
        if greedy and measure(f"{greedy[-1]} {word}", size) <= width:
            greedy[-1] += f" {word}"
        else:
            greedy.append(word)
    lines = len(greedy)
    if lines < 2:
        return greedy

    @cache
    def span(i: int, j: int) -> float:
        return measure(" ".join(words[i:j]), size)

    @cache
    def best(i: int, k: int) -> tuple[float, tuple[int, ...]]:
        """The narrowest widest line for words[i:] in k lines, and where the lines break."""
        if k == 1:
            return (span(i, len(words)), ()) if span(i, len(words)) <= width else (math.inf, ())
        found: tuple[float, tuple[int, ...]] = (math.inf, ())
        for j in range(i + 1, len(words) - k + 2):
            if span(i, j) > width:
                break
            rest, breaks = best(j, k - 1)
            found = min(found, (max(span(i, j), rest), (j, *breaks)))
        return found

    _, breaks = best(0, lines)
    edges = [0, *breaks, len(words)]
    return [" ".join(words[a:b]) for a, b in itertools.pairwise(edges)]


@dataclass(frozen=True)
class Box:
    """A label's extent on the canvas, and what it says (for the checks' messages)."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: str
    halo: bool = False
    """Set on the page by a halo of the page's color, which gridlines and rules pass beneath."""

    def overlaps(self, other: Box, pad: float = 0.0) -> bool:
        return not (
            self.x1 + pad <= other.x0
            or other.x1 + pad <= self.x0
            or self.y1 + pad <= other.y0
            or other.y1 + pad <= self.y0
        )


def fmt(value: float) -> str:
    """A coordinate as an export writes it: at most two decimals, no trailing zeros."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _segment_hits_box(a, b, box: Box, pad: float) -> bool:
    """Whether segment ab enters ``box`` grown by ``pad`` (a Liang-Barsky clip)."""
    x0, y0, x1, y1 = box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad
    dx, dy = b[0] - a[0], b[1] - a[1]
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
            if lo > hi:
                return False
    return True


class Figure:
    """An SVG canvas that records its labels and connectors, for the checks :meth:`render` runs."""

    def __init__(self, width: float, height: float, title: str, description: str):
        self.width, self.height = width, height
        self.title, self.description = title, description
        self.items: list[str] = []
        self.boxes: list[Box] = []
        self.wires: list[tuple[list[tuple[float, float]], tuple[str, ...]]] = []
        """Connectors, which no label may cross, and what each may (its ``ignore``)."""
        self.rules: list[tuple[tuple[float, float], tuple[float, float]]] = []
        """Gridlines, axes and reference rules, which only a label set on a halo may cross."""
        self.borders: list[tuple[float, float, float, float]] = []
        """Panels', groups' and the KV pool's outlines (x, y, w, h), which a label sits inside or outside, never
        across."""
        self.glyphs: dict[str, set[str]] = {face: set() for face in FACES}

    # Text -------------------------------------------------------------------------------------

    def text(
        self,
        x: float,
        y: float,
        markup: object,
        *,
        size: float = LABEL,
        anchor: str = "start",
        cls: str = "",
        collide: bool = True,
        name: str | None = None,
        halo: bool = False,
        bold: bool = False,
    ) -> Box:
        """A label at baseline ``y``. ``cls`` is its ink: "", "soft", "mute" or "<hue> tint".
        ``halo`` sets it on a halo of the page's color, for a value that gridlines run behind;
        ``bold`` sets it in the semibold, for a panel's title and the value a figure is drawn for."""
        if cls not in TEXT_CLASSES:
            raise ValueError(f"text takes ink, soft, mute or a hue's tint, not {cls!r}")
        markup = str(markup)
        spans = parse(markup)
        width = measure(markup, size, bold=bold)
        parts, level = [], 0.0
        for span in spans:
            for char in span.text:
                face = "bold" if bold and span.face == "regular" else span.face
                self.glyphs[_resolve(face, char)].add(char)
            attrs = []
            if span.face == "italic":
                attrs.append('class="i"')
            if span.scale != 1:
                attrs.append(f'font-size="{fmt(size * span.scale)}"')
            if span.shift != level:
                attrs.append(f'dy="{fmt((level - span.shift) * size)}"')
                level = span.shift
            text = escape(span.text, quote=False)
            parts.append(f"<tspan {' '.join(attrs)}>{text}</tspan>" if attrs else text)
        anchor_attr = "" if anchor == "start" else f' text-anchor="{anchor}"'
        classes = " ".join(c for c in (cls, "b" if bold else "", "halo" if halo else "") if c)
        cls_attr = f' class="{classes}"' if classes else ""
        size_attr = "" if size == LABEL else f' font-size="{fmt(size)}"'
        self.items.append(
            f'<text x="{fmt(x)}" y="{fmt(y)}"{anchor_attr}{cls_attr}{size_attr}>'
            + "".join(parts)
            + "</text>"
        )
        left = x - {"start": 0, "middle": width / 2, "end": width}[anchor]
        low = min((s.shift for s in spans), default=0)
        high = max((s.shift + 0.7 * s.scale for s in spans), default=0.7)
        descends = any(c in "gjpqyQ(),;[]{}/|" for s in spans for c in s.text)
        box = Box(
            left,
            y - max(high, 0.68) * size,
            left + width,
            y + (0.24 if descends else 0.04) * size - low * size,
            name or markup,
            halo,
        )
        if collide:
            self.boxes.append(box)
        return box

    # Geometry ---------------------------------------------------------------------------------

    def raw(self, svg: str) -> None:
        self.items.append(svg)

    def rect(self, x: float, y: float, w: float, h: float, cls: str, rx: float = 0) -> None:
        radius = f' rx="{fmt(rx)}"' if rx else ""
        self.items.append(
            f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}"{radius} '
            f'class="{cls}"/>'
        )

    def path(self, d: str, cls: str) -> None:
        self.items.append(f'<path d="{d}" class="{cls}"/>')

    def rule(self, x0: float, x1: float, y: float, cls: str = "axis") -> None:
        """A horizontal rule: an axis, a gridline (``cls`` "grid") or the SLO's ("slo")."""
        self.path(f"M{fmt(x0)} {fmt(y)}H{fmt(x1)}", cls)
        self.rules.append(((x0, y), (x1, y)))

    def vrule(self, x: float, y0: float, y1: float, cls: str = "axis") -> None:
        """A vertical rule, as :meth:`rule`."""
        self.path(f"M{fmt(x)} {fmt(y0)}V{fmt(y1)}", cls)
        self.rules.append(((x, y0), (x, y1)))

    def time_axis(self, x0: float, x1: float, y: float) -> None:
        """A time axis: a thin rule from ``x0`` that ends in a small solid head at ``x1``."""
        head, half = 7.0, 3.2
        self.rule(x0, x1 - head + 1, y)
        self.path(
            f"M{fmt(x1)} {fmt(y)}L{fmt(x1 - head)} {fmt(y - half)}L{fmt(x1 - head)} {fmt(y + half)}Z",
            "axis-head",
        )

    # Charts -----------------------------------------------------------------------------------

    def hbar(
        self, x: float, y: float, length: float, thickness: float, bar: str, *, cut: bool = False
    ) -> None:
        """A flat horizontal bar from the axis at ``x``, its free end's corners rounded; a bar
        ``cut`` at the scale's end has a square end, and its value says how far it goes."""
        if bar not in BARS:
            raise ValueError(f"a bar is one of {BARS}, not {bar!r}")
        self._hpath(x, y, length, thickness, f"g-{bar}", 0.0 if cut else BAR_CORNER)

    def segments(
        self, x: float, y: float, parts: list[tuple[float, str]], thickness: float
    ) -> list[float]:
        """A stacked horizontal bar: ``parts`` are (length, fill) from the axis at ``x``, each fill
        one of SEGMENTS, parted by a seam of the page's color; the last segment's free end is
        rounded. Returns each segment's start."""
        starts = []
        for k, (length, fill) in enumerate(parts):
            if fill not in SEGMENTS:
                raise ValueError(f"a segment is one of {tuple(SEGMENTS)}, not {fill!r}")
            starts.append(x)
            self._segment(x, y, length, thickness, f"s-{fill}", k, len(parts))
            x += length
        return starts

    def _segment(
        self, x: float, y: float, length: float, thickness: float, cls: str, k: int, count: int
    ) -> None:
        """A stacked bar's ``k``-th segment of ``count``: half a seam short on each side it shares
        with a neighbour, so that the bar's length stays its value's; the last one's end rounded."""
        left = SEAM / 2 if k else 0.0
        right = SEAM / 2 if k < count - 1 else 0.0
        if length - left - right < 0.6:  # too short to part: a sliver its full length
            left = right = 0.0
        corner = BAR_CORNER if k == count - 1 else 0.0
        self._hpath(x + left, y, length - left - right, thickness, cls, corner)

    def _hpath(self, x: float, y: float, length: float, thickness: float, cls: str, r: float):
        r = min(r, length / 2, thickness / 2)
        x1, y1 = x + length, y + thickness
        if r <= 0:
            self.path(f"M{fmt(x)} {fmt(y)}H{fmt(x1)}V{fmt(y1)}H{fmt(x)}Z", cls)
            return
        self.path(
            f"M{fmt(x)} {fmt(y)}H{fmt(x1 - r)}Q{fmt(x1)} {fmt(y)} {fmt(x1)} {fmt(y + r)}"
            f"V{fmt(y1 - r)}Q{fmt(x1)} {fmt(y1)} {fmt(x1 - r)} {fmt(y1)}H{fmt(x)}Z",
            cls,
        )

    def swatch(self, x: float, baseline: float, fill: str, name: str, *, size: float = NOTE):
        """A legend's entry: a small segment swatch and its name; returns the name's box."""
        self._hpath(x, baseline - 0.62 * size, 18, 0.66 * size, f"s-{fill}", 2.0)
        return self.text(x + 26, baseline, name, size=size)

    # Method figures ---------------------------------------------------------------------------

    def panel(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        letter: str | None = None,
        *,
        frame: bool = True,
        rx: float = 8,
    ) -> float:
        """A panel: a hairline frame, unless ``frame`` is off, with its letter and its title in
        the semibold inside the top left. Returns the title's baseline."""
        if frame:
            self.rect(x, y, w, h, "panel", rx=rx)
            self.borders.append((x, y, w, h))
        baseline = y + 16 + 0.7 * LABEL
        self.text(x + 20, baseline, f"({letter}) {title}" if letter else title, bold=True)
        return baseline

    def group(self, x: float, y: float, w: float, h: float, *, rx: float = 8) -> None:
        """A hairline container around components that act together."""
        self.rect(x, y, w, h, "group", rx=rx)
        self.borders.append((x, y, w, h))

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        lines: str | list[str],
        *,
        kind: str = "box",
        size: float = LABEL,
        pitch: float | None = None,
        rx: float = CORNER,
    ) -> None:
        """A component: a pale fill inside a thin outline of its hue, plain ("box"), the change's
        own ("ours") or in a hue, its name centred inside on one line or several. A label that
        does not fit inside is refused."""
        if kind not in KINDS:
            raise ValueError(f"a component is one of {KINDS}, not {kind!r}")
        self.rect(x, y, w, h, kind if kind in ("box", "ours") else f"{kind} obj", rx=rx)
        lines = [lines] if isinstance(lines, str) else lines
        pitch = pitch or size * 1.22
        first = y + h / 2 - pitch * (len(lines) - 1) / 2 + 0.33 * size
        for k, line in enumerate(lines):
            box = self.text(x + w / 2, first + k * pitch, line, size=size, anchor="middle")
            if box.x0 < x + 6 or box.x1 > x + w - 6 or box.y0 < y + 3 or box.y1 > y + h - 3:
                raise ValueError(f"{line!r} does not fit inside its box")

    def memory(self, x: float, y: float, w: float, h: float, *, rx: float = CORNER) -> None:
        """Physical KV memory: the tray its blocks lie in, a component's shape in the faintest
        neutral."""
        self.rect(x, y, w, h, "memory", rx=rx)
        self.borders.append((x, y, w, h))

    def cells(
        self,
        x: float,
        y: float,
        hues: list[str | None],
        *,
        cell: float,
        height: float,
        gap: float = 0,
        hatched: set[int] | frozenset[int] = frozenset(),
        strong: tuple[str, ...] = (),
        rx: float = 2,
    ) -> None:
        """A row of KV blocks, flat as a bar's segments are and parted by the same seam: each a
        cell in its owner's soft tone, an outlined empty cell where ``None``; a hatched cell is
        one being taken, paler under its hatch, and a hue in ``strong`` is filled in its mid
        tone, for the blocks a figure follows."""
        inset = SEAM / 2
        for k, hue in enumerate(hues):
            cx = x + k * (cell + gap)
            if hue is None:
                self.rect(
                    cx + inset + 0.5, y + 0.5, cell - SEAM - 1, height - 1, "cell-empty", rx=rx
                )
                continue
            cls = f"{hue} cell strong" if hue in strong else f"{hue} cell"
            if k in hatched:
                cls += " taken"
            self.rect(cx + inset, y, cell - SEAM, height, cls, rx=rx)
            if k in hatched:
                step = 5.0
                d = "".join(
                    f"M{fmt(cx + inset + max(0, o))} {fmt(y + height - max(0, -o))}"
                    f"L{fmt(cx + inset + min(cell - SEAM, o + height))} "
                    f"{fmt(y + height - min(height, cell - SEAM - o))}"
                    for o in (
                        j * step - height for j in range(int((cell - SEAM + height) / step) + 1)
                    )
                )
                self.path(d, f"{hue} hatch")

    def flow(
        self,
        points: list[tuple[float, float]],
        *,
        arrow: bool = True,
        radius: float = 8,
        collide: bool = True,
        ignore: tuple[str, ...] = (),
        hue: str | None = None,
    ) -> None:
        """Dataflow: a thin slate polyline of one weight, rounded at its corners, its swept head
        ending FLOW_GAP short of the last point, a target's edge. With ``hue``, the line takes that
        hue, and carries that hue's values (the voice turn's blocks). An arrow whose last segment
        is shorter than MIN_SHAFT head lengths is refused: it would read as a bare head."""
        points = [(float(x), float(y)) for x, y in points]
        head, half = FLOW_HEAD, FLOW_HALF
        if arrow:
            (ax, ay), (bx, by) = points[-2], points[-1]
            length = math.hypot(bx - ax, by - ay)
            if length < MIN_SHAFT * head:
                raise ValueError(
                    f"an arrow's last segment is {length:.1f} units, under "
                    f"{MIN_SHAFT * head:.1f}: make room for its shaft"
                )
            ux, uy = (bx - ax) / length, (by - ay) / length
            tip = (bx - ux * FLOW_GAP, by - uy * FLOW_GAP)
            # The shaft ends inside the head, so its round cap never shows past the head's notch.
            draw = [*points[:-1], (tip[0] - ux * head * 0.5, tip[1] - uy * head * 0.5)]
        else:
            draw = points
        d = f"M{fmt(draw[0][0])} {fmt(draw[0][1])}"
        for k in range(1, len(draw)):
            if radius and 0 < k < len(draw) - 1:
                (px, py), (cx, cy), (nx, ny) = draw[k - 1], draw[k], draw[k + 1]
                l1, l2 = math.hypot(cx - px, cy - py), math.hypot(nx - cx, ny - cy)
                r = min(radius, l1 / 2, l2 / 2)
                ax, ay = cx - (cx - px) / l1 * r, cy - (cy - py) / l1 * r
                bx, by = cx + (nx - cx) / l2 * r, cy + (ny - cy) / l2 * r
                d += f"L{fmt(ax)} {fmt(ay)}Q{fmt(cx)} {fmt(cy)} {fmt(bx)} {fmt(by)}"
            else:
                d += f"L{fmt(draw[k][0])} {fmt(draw[k][1])}"
        self.path(d, "flow" if hue is None else f"{hue} line")
        if arrow:
            nx, ny = -uy, ux
            x, y = tip
            base = (x - ux * head, y - uy * head)
            notch = (x - ux * head * 0.72, y - uy * head * 0.72)
            corners = [
                (x, y),
                (base[0] + nx * half, base[1] + ny * half),
                notch,
                (base[0] - nx * half, base[1] - ny * half),
            ]
            self.path(
                "M" + "L".join(f"{fmt(px)} {fmt(py)}" for px, py in corners) + "Z",
                "flow-head" if hue is None else f"{hue} line-head",
            )
        if collide:
            self.wires.append((points, ignore))

    def leader(self, points: list[tuple[float, float]]) -> None:
        """A zoom's dashed leader, from a dashed outline in a stage strip to the panel it expands."""
        self.path("M" + "L".join(f"{fmt(x)} {fmt(y)}" for x, y in points), "leader")
        self.wires.append(([(float(x), float(y)) for x, y in points], ()))

    def bracket(
        self, x0: float, x1: float, y: float, name: str, *, below: bool = True, size: float = NOTE
    ) -> Box:
        """A tick-ended bracket spanning exactly ``x0`` to ``x1`` at ``y``, its ticks toward what it
        names (up when it sits ``below`` it), its name once, centred on its far side."""
        tick = -6 if below else 6
        self.path(f"M{fmt(x0)} {fmt(y + tick)}V{fmt(y)}H{fmt(x1)}V{fmt(y + tick)}", "bracket")
        self.rules.append(((x0, y), (x1, y)))
        baseline = y + 0.72 * size + 6 if below else y - 8
        return self.text((x0 + x1) / 2, baseline, name, size=size, anchor="middle")

    # Checks and export ------------------------------------------------------------------------

    def check(self) -> None:
        """Refuse text off the canvas, overlapping text, a connector across text, a rule across text not on a
        halo, and text across a panel's, a group's or the KV pool's outline."""
        for k, box in enumerate(self.boxes):
            if box.x0 < 1 or box.y0 < 1 or box.x1 > self.width - 1 or box.y1 > self.height - 1:
                raise ValueError(f"text leaves the canvas: {box.label!r} {box}")
            for other in self.boxes[k + 1 :]:
                if box.overlaps(other, pad=1.0):
                    raise ValueError(f"overlapping text: {box.label!r} and {other.label!r}")
        for points, ignore in self.wires:
            for box in self.boxes:
                if box.label in ignore:
                    continue
                for a, b in itertools.pairwise(points):
                    if _segment_hits_box(a, b, box, pad=1.5):
                        raise ValueError(f"a connector crosses the text {box.label!r} near {a}")
        for a, b in self.rules:
            for box in self.boxes:
                if not box.halo and _segment_hits_box(a, b, box, pad=1.0):
                    raise ValueError(f"a rule crosses the text {box.label!r} near {a}")
        for x, y, w, h in self.borders:
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            edges = list(itertools.pairwise(corners))
            for box in self.boxes:
                if any(_segment_hits_box(a, b, box, pad=1.0) for a, b in edges):
                    raise ValueError(f"a label crosses a border: {box.label!r} near {(x, y)}")

    def _font_css(self) -> str:
        rules = []
        for face, chars in self.glyphs.items():
            if not chars:
                continue
            file, family, style, weight = FACES[face]
            options = subset.Options()
            options.flavor = "woff"
            options.layout_features = ["kern", "liga"]
            options.name_IDs = [0, 1, 2, 13, 14]
            options.notdef_outline = True
            options.hinting = False
            options.recalc_timestamp = False
            font = TTFont(FONT_DIR / file, recalcTimestamp=False)
            subsetter = subset.Subsetter(options)
            subsetter.populate(text="".join(sorted(chars)))
            subsetter.subset(font)
            buffer = io.BytesIO()
            font.flavor = "woff"
            font.save(buffer)
            data = base64.b64encode(buffer.getvalue()).decode()
            rules.append(
                f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight};"
                f"src:url(data:font/woff;base64,{data}) format('woff')}}"
            )
        return "".join(rules)

    def render(self) -> str:
        """The SVG, after the checks. With FLATSHARE_FIGURE_DRAFT set, a failed check is printed and the
        SVG written anyway, to look at while laying a figure out; nothing else sets it."""
        try:
            self.check()
        except ValueError as exc:
            if not os.environ.get("FLATSHARE_FIGURE_DRAFT"):
                raise
            print(f"draft: {self.title}: {exc}", file=sys.stderr)
        css = self._font_css() + THEME_CSS + BASE_CSS
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{fmt(self.width)}" '
                f'height="{fmt(self.height)}" viewBox="0 0 {fmt(self.width)} {fmt(self.height)}" '
                f'font-size="{LABEL}" role="img" aria-labelledby="title desc">',
                f'<title id="title">{escape(self.title, quote=False)}</title>',
                f'<desc id="desc">{escape(self.description, quote=False)}</desc>',
                f"<!--{FONT_NOTICE}-->",
                f"<style>{css}</style>",
                # An opaque page behind everything: where the reader's page theme and the system's
                # differ, the figure, which follows the system's, stays legible.
                f'<rect class="page" width="{fmt(self.width)}" height="{fmt(self.height)}"/>',
                *self.items,
                "</svg>",
                "",
            ]
        )

    def save(self, path: Path) -> Path:
        path.write_text(self.render())
        return path
