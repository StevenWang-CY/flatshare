"""Flatshare's logo, its mark alone and the repository's social preview (docs/brand), drawn from their
geometry and the figures' typeface, STIX Two Text, every glyph outlined to a path so that each viewer renders
them alike.

    python -m flatshare_bench.brand OUT_DIR

The mark is a KV pool of nine blocks whose middle column is the voice turn's: the agents' blocks gray and the
turn's lane green with a waveform, as the method figure draws a voice turn's blocks beside the agents'
(docs/figures/method.svg). The wordmark is set in STIX Two Text SemiBold, kerned by the font's own pair
adjustments, without ligatures.

Writes logo.svg and logo-dark.svg (the mark and the wordmark, for the README's light and dark themes), mark.svg
and mark-dark.svg (the mark alone), mark-small.svg and mark-small-dark.svg (the mark without its waveform, for 16
to 24 pixels), and social-preview.svg with social-preview.png (1280 x 640, GitHub's size for
a repository's social preview). resvg renders the PNG, which records the SHA-256 of the SVG it was rendered from.
"""

from __future__ import annotations

import hashlib
import sys
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

from flatshare_bench.figure_style import DARK, FONT_DIR, LIGHT

WORDMARK = "flatshare"
TAGLINE = "A voice agent and coding agents on one GPU"
SUBLINE = "Two scheduler changes, proposed to vLLM, make room for a voice agent's turn beside long-context coding agents"
URL = "github.com/StevenWang-CY/flatshare"
NOTICE = "An independent project · Apache-2.0"

SEMIBOLD, REGULAR = "STIXTwoText-SemiBold.ttf", "STIXTwoText-Regular.ttf"

UNIT = 64.0
"""The mark's grid: a 64-unit square."""
BLOCK, GAP, CORNER = 16.0, 4.0, 4.0
"""A KV block's side, the gap between blocks (and the mark's inset), and a block's corner radius."""
WAVE = ((-4.2, 12.0), (0.0, 24.0), (4.2, 16.0))
"""The waveform's bars in the lane: each one's offset from the lane's center and its height."""
WAVE_WIDTH = 2.8

THEMES = {
    "light": {
        "ink": LIGHT["ink"],
        "block": "#8c959f",
        "lane": LIGHT["green"],
        "wave": "#ffffff",
    },
    "dark": {
        "ink": DARK["ink"],
        "block": DARK["bar-mid"],
        "lane": LIGHT["green"],
        "wave": "#ffffff",
    },
}
"""Each theme's colors. The lane keeps the light theme's green in both themes, so that the mark is one object on
either page, and only its neutrals follow the theme: the wordmark takes the figures' ink, and the blocks a gray
that keeps 3:1 against the page, WCAG's floor for a graphic (#8c959f on white; the figures' dark bar gray on
#0d1117)."""

TEXT_SIZE, TEXT_GAP, BASELINE = 66.0, 20.0, 54.0
"""The wordmark beside a 64-unit mark: its size (its ascenders reach the top row), its distance from the mark,
and its baseline."""

PREVIEW = (1280, 640)
"""GitHub's recommended size for a repository's social preview."""


def ntos(value: float) -> str:
    """A coordinate as the SVGs write it: at most two decimals, no trailing zeros, no negative zero."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


@cache
def _font(name: str) -> TTFont:
    return TTFont(FONT_DIR / name, recalcTimestamp=False, lazy=False)


@cache
def _kerning(name: str) -> Callable[[str, str], int]:
    """The font's pair kerning from its GPOS `kern` feature: for two glyph names, the adjustment to the first
    one's advance, in font units. Each lookup adds the value of the first of its subtables that covers the pair,
    as a shaper applies them."""
    gpos = _font(name)["GPOS"].table
    indices = sorted(
        {
            index
            for record in gpos.FeatureList.FeatureRecord
            if record.FeatureTag == "kern"
            for index in record.Feature.LookupListIndex
        }
    )
    lookups = [
        [
            getattr(subtable, "ExtSubTable", subtable)
            for subtable in gpos.LookupList.Lookup[index].SubTable
            if type(getattr(subtable, "ExtSubTable", subtable)).__name__ == "PairPos"
        ]
        for index in indices
    ]

    def advance(record: object) -> int:
        return (getattr(record, "XAdvance", 0) or 0) if record is not None else 0

    def kern(left: str, right: str) -> int:
        total = 0
        for subtables in lookups:
            for subtable in subtables:
                if left not in subtable.Coverage.glyphs:
                    continue
                if subtable.Format == 1:
                    pairs = subtable.PairSet[subtable.Coverage.glyphs.index(left)].PairValueRecord
                    match = next((pair for pair in pairs if pair.SecondGlyph == right), None)
                    if match is None:
                        continue
                    total += advance(match.Value1)
                else:
                    first = subtable.ClassDef1.classDefs.get(left, 0)
                    second = subtable.ClassDef2.classDefs.get(right, 0)
                    total += advance(subtable.Class1Record[first].Class2Record[second].Value1)
                break
        return total

    return kern


@dataclass(frozen=True)
class Text:
    """Text set as outlines: its path data, its advance and its ink's bounds (x0, y0, x1, y1)."""

    path: str
    advance: float
    bounds: tuple[float, float, float, float]


def outline(text: str, name: str, size: float, x: float, baseline: float) -> Text:
    """``text`` in the font ``name`` at ``size`` canvas units from ``x``, its baseline at ``baseline``: kerned,
    without ligatures, every glyph a path."""
    font = _font(name)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    metrics = font["hmtx"].metrics
    kern = _kerning(name)
    scale = size / font["head"].unitsPerEm
    pen = SVGPathPen(glyphs, ntos=ntos)
    bounds = BoundsPen(glyphs)
    cursor, previous = 0, None
    for char in text:
        glyph = cmap[ord(char)]
        if previous is not None:
            cursor += kern(previous, glyph)
        transform = (scale, 0, 0, -scale, x + cursor * scale, baseline)
        glyphs[glyph].draw(TransformPen(pen, transform))
        glyphs[glyph].draw(TransformPen(bounds, transform))
        cursor += metrics[glyph][0]
        previous = glyph
    if bounds.bounds is None:
        raise ValueError(f"no ink in {text!r}")
    return Text(pen.getCommands(), cursor * scale, bounds.bounds)


def lines(text: str, name: str, size: float, width: float) -> list[str]:
    """``text`` broken at spaces into lines no wider than ``width`` at ``size``."""
    out: list[str] = []
    for word in text.split():
        trial = f"{out[-1]} {word}" if out else word
        if out and outline(trial, name, size, 0, 0).advance <= width:
            out[-1] = trial
        else:
            out.append(word)
    return out


def _rect(x: float, y: float, width: float, height: float, radius: float, fill: str) -> str:
    return (
        f'<rect x="{ntos(x)}" y="{ntos(y)}" width="{ntos(width)}" height="{ntos(height)}" '
        f'rx="{ntos(radius)}" fill="{fill}"/>'
    )


def pool(
    x: float,
    y: float,
    columns: int,
    rows: int,
    lane: int,
    scale: float,
    colors: dict[str, str],
    wave: bool = True,
) -> Iterator[str]:
    """A KV pool of ``columns`` x ``rows`` blocks from (x, y), ``scale`` canvas units to a mark unit; its
    column ``lane`` is the voice turn's, one green capsule the pool's height, with the waveform at its middle
    unless ``wave`` is false."""
    pitch, side = (BLOCK + GAP) * scale, BLOCK * scale
    for column in range(columns):
        left = x + column * pitch
        if column != lane:
            for row in range(rows):
                yield _rect(left, y + row * pitch, side, side, CORNER * scale, colors["block"])
            continue
        height = rows * pitch - GAP * scale
        yield _rect(left, y, side, height, side / 2, colors["lane"])
        if not wave:
            continue
        middle_x, middle_y = left + side / 2, y + height / 2
        width = WAVE_WIDTH * scale
        for offset, bar in WAVE:
            x0, length = middle_x + offset * scale - width / 2, bar * scale
            yield _rect(x0, middle_y - length / 2, width, length, width / 2, colors["wave"])


def mark(x: float, y: float, size: float, colors: dict[str, str], wave: bool = True) -> str:
    """The mark in a ``size``-unit square at (x, y): three columns of three blocks, the middle one the lane."""
    scale = size / UNIT
    return "".join(pool(x + GAP * scale, y + GAP * scale, 3, 3, 1, scale, colors, wave))


def _svg(
    box: tuple[float, float, float, float], body: str, label: str, ground: str | None = None
) -> str:
    x, y, width, height = box
    fill = (
        f'<rect x="{ntos(x)}" y="{ntos(y)}" width="{ntos(width)}" height="{ntos(height)}" fill="{ground}"/>'
        if ground
        else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{ntos(x)} {ntos(y)} {ntos(width)} {ntos(height)}" '
        f'width="{ntos(width)}" height="{ntos(height)}" role="img" aria-label="{label}">'
        f"<title>{label}</title>{fill}{body}</svg>\n"
    )


def logo(theme: str) -> str:
    """The mark beside the wordmark, cropped to their ink."""
    colors = THEMES[theme]
    word = outline(WORDMARK, SEMIBOLD, TEXT_SIZE, UNIT + TEXT_GAP, BASELINE)
    top, bottom = min(GAP, word.bounds[1]), max(UNIT - GAP, word.bounds[3])
    body = mark(0, 0, UNIT, colors) + f'<path d="{word.path}" fill="{colors["ink"]}"/>'
    return _svg((GAP, top, word.bounds[2] - GAP, bottom - top), body, "Flatshare")


def mark_svg(theme: str, wave: bool = True) -> str:
    """The mark alone, cropped to its ink; without its waveform for 16 to 24 pixels, where the bars would be
    narrower than a pixel."""
    body = mark(0, 0, UNIT, THEMES[theme], wave)
    return _svg((GAP, GAP, UNIT - 2 * GAP, UNIT - 2 * GAP), body, "Flatshare")


@dataclass(frozen=True)
class Placed:
    """A part of the social preview: its SVG and its ink's bounds (x0, y0, x1, y1)."""

    svg: str
    bounds: tuple[float, float, float, float]


def preview_parts() -> list[Placed]:
    """The social preview's parts, light: on the left the logo, the tagline and its subline as one group
    centered on the page's height, with the repository's address below; on the right a wider pool, its lane
    the fourth column."""
    width, height = PREVIEW
    colors = THEMES["light"]
    margin = 88.0
    scale, columns, rows, lane = 3.4, 5, 4, 3
    pool_width = (columns * (BLOCK + GAP) - GAP) * scale
    pool_height = (rows * (BLOCK + GAP) - GAP) * scale
    pool_x, pool_y = width - margin - pool_width, (height - pool_height) / 2
    parts = [
        Placed(
            "".join(pool(pool_x, pool_y, columns, rows, lane, scale, colors)),
            (pool_x, pool_y, pool_x + pool_width, pool_y + pool_height),
        )
    ]
    text_width = pool_x - margin - 56
    size = 1.42
    tagline_size = min(40.0, 40.0 * text_width / outline(TAGLINE, REGULAR, 40.0, 0, 0).advance)
    subline = lines(SUBLINE, REGULAR, 27.0, text_width)
    top = (height - (UNIT * size + 100 + 37 * len(subline))) / 2
    inset = GAP * size
    parts.append(
        Placed(
            mark(margin, top, UNIT * size, colors),
            (margin + inset, top + inset, margin + UNIT * size - inset, top + UNIT * size - inset),
        )
    )

    def text(line: str, name: str, size: float, x: float, y: float, fill: str) -> Placed:
        set_line = outline(line, name, size, x, y)
        return Placed(f'<path d="{set_line.path}" fill="{fill}"/>', set_line.bounds)

    word_x = margin + (UNIT + TEXT_GAP) * size
    parts.append(
        text(WORDMARK, SEMIBOLD, TEXT_SIZE * size, word_x, top + BASELINE * size, colors["ink"])
    )
    y = top + UNIT * size + 70
    parts.append(text(TAGLINE, REGULAR, tagline_size, margin, y, colors["ink"]))
    for i, line in enumerate(subline):
        parts.append(text(line, REGULAR, 27.0, margin, y + 67 + 37 * i, LIGHT["mute"]))
    parts.append(text(URL, REGULAR, 23.0, margin, height - 56, LIGHT["mute"]))
    notice_x = width - margin - outline(NOTICE, REGULAR, 23.0, 0, 0).advance
    parts.append(text(NOTICE, REGULAR, 23.0, notice_x, height - 56, LIGHT["mute"]))
    return parts


def preview() -> str:
    """The social preview's SVG."""
    body = "".join(part.svg for part in preview_parts())
    return _svg((0, 0, *PREVIEW), body, f"Flatshare: {TAGLINE}", LIGHT["page"])


def with_source(png: bytes, source: str) -> bytes:
    """``png`` with a tEXt chunk, before its image data, recording the SHA-256 of the SVG it was rendered from."""
    data = b"Source-SHA256\x00" + hashlib.sha256(source.encode()).hexdigest().encode()
    chunk = (
        len(data).to_bytes(4, "big")
        + b"tEXt"
        + data
        + zlib.crc32(b"tEXt" + data).to_bytes(4, "big")
    )
    at = png.index(b"IDAT") - 4
    return png[:at] + chunk + png[at:]


def source_of(png: bytes) -> str | None:
    """The SVG SHA-256 that :func:`with_source` recorded in ``png``, or None."""
    marker = b"tEXtSource-SHA256\x00"
    at = png.find(marker)
    return png[at + len(marker) : at + len(marker) + 64].decode() if at >= 0 else None


def svgs() -> dict[str, str]:
    """Every SVG this module writes, by file name."""
    out = {}
    for theme, suffix in (("light", ""), ("dark", "-dark")):
        out[f"logo{suffix}.svg"] = logo(theme)
        out[f"mark{suffix}.svg"] = mark_svg(theme)
        out[f"mark-small{suffix}.svg"] = mark_svg(theme, wave=False)
    out["social-preview.svg"] = preview()
    return out


def render(out_dir: Path) -> list[Path]:
    import resvg_py  # the brand extra: only rendering the PNG needs it

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in svgs().items():
        (out_dir / name).write_text(text)
        written.append(out_dir / name)
    source = (out_dir / "social-preview.svg").read_text()
    png = bytes(resvg_py.svg_to_bytes(svg_string=source, width=PREVIEW[0], height=PREVIEW[1]))
    (out_dir / "social-preview.png").write_bytes(with_source(png, source))
    return [*written, out_dir / "social-preview.png"]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m flatshare_bench.brand OUT_DIR", file=sys.stderr)
        return 2
    for path in render(Path(args[0])):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
