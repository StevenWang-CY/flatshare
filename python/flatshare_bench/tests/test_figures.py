"""The docs' figures: what their style refuses to draw, the contrast of its inks, and that every SVG under
docs/figures is what its generator draws today from the committed records (`make figures` rewrites them), so a
change to a generator, a record or figure.toml fails here until the figures are drawn again; and the README's
animation, which make demo-video draws from recorded calls outside the tree."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from flatshare_bench import diagrams
from flatshare_bench import figure_style as style
from flatshare_bench.report import figures as report_figures

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs" / "figures"
SIZES = {style.TITLE, style.LABEL, style.VALUE, style.NOTE}
"""The type's sizes, one per role; a sub- or superscript sets its role's size at 72%."""


def figure() -> style.Figure:
    return style.Figure(400, 200, "A figure", "What it shows.")


def test_the_style_refuses_what_a_reader_would_trip_on() -> None:
    s = figure()
    s.text(10, 40, "a label")
    s.text(30, 44, "another")
    with pytest.raises(ValueError, match="overlapping text: 'a label' and 'another'"):
        s.render()
    s = figure()
    s.text(360, 40, "off the canvas")
    with pytest.raises(ValueError, match="text leaves the canvas"):
        s.render()
    s = figure()
    s.text(100, 40, "crossed")
    s.flow([(120, 10), (120, 100)], arrow=False)
    with pytest.raises(ValueError, match="a connector crosses the text 'crossed'"):
        s.render()
    # A gridline may pass behind a value set on a halo of the page's color, and behind nothing else.
    s = figure()
    s.text(100, 40, "a value", halo=True)
    s.vrule(120, 10, 100, "grid")
    s.render()
    s.text(100, 80, "a name")
    with pytest.raises(ValueError, match="a rule crosses the text 'a name'"):
        s.render()
    # A label sits inside a panel, a group or the KV pool's outline, or outside it, never across its border.
    s = figure()
    s.group(20, 20, 100, 80)
    s.text(30, 50, "inside")
    s.text(140, 50, "outside")
    s.render()
    s.text(100, 90, "across")
    with pytest.raises(ValueError, match="a label crosses a border: 'across'"):
        s.render()
    with pytest.raises(ValueError, match="make room for its shaft"):
        figure().flow([(10, 50), (24, 50)])  # under 2.2 head lengths
    figure().flow([(10, 50), (10 + style.MIN_SHAFT * style.FLOW_HEAD, 50)])
    with pytest.raises(ValueError, match="does not fit inside its box"):
        figure().box(10, 10, 60, 30, "a long name")
    with pytest.raises(ValueError, match="text takes ink, soft, mute or a hue's tint"):
        figure().text(10, 40, "faint", cls="faint")
    with pytest.raises(ValueError, match="a bar is one of"):
        figure().hbar(10, 10, 50, 20, "pink")


def test_a_figure_is_the_same_bytes_every_time_and_embeds_only_its_glyphs() -> None:
    def draw() -> str:
        s = figure()
        s.text(10, 40, "Voice $p_{95}$ \N{MINUS SIGN}2")
        s.hbar(10, 60, 120, 20, "hi")
        return s.render()

    first = draw()
    assert first == draw()
    assert first.count("@font-face") == 2  # the text's faces, regular and italic (p); no math glyph
    assert not any("font-weight:600" in face for face in re.findall(r"@font-face\{[^}]*\}", first))
    # The semibold is embedded only where a figure sets a title or its featured value in it.
    s = figure()
    s.text(10, 40, "A panel", bold=True)
    s.text(10, 80, "a note")
    faces = re.findall(r"@font-face\{[^}]*\}", s.render())
    assert sorted("font-weight:600" in face for face in faces) == [False, True]


def luminance(color: str) -> float:
    channels = [int(color[k : k + 2], 16) / 255 for k in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("palette", [style.LIGHT, style.DARK], ids=["light", "dark"])
def test_every_ink_clears_four_and_a_half_to_one_on_every_ground(palette: dict[str, str]) -> None:
    # Text is ink, soft or mute, or a hue that names an identity: the voice's green, and the change's blue and the
    # SLO's red, which only the charts set, on the page. A label sits on the page or in a plain component; a name inside a hue's
    # pale fill is ink.
    grounds = ("page", "box")
    for ink, on in [
        *((ink, grounds) for ink in ("ink", "soft", "mute", "green")),
        ("blue", ("page",)),
        ("red", ("page",)),
    ]:
        for ground in on:
            ratio = contrast(palette[ink], palette[ground])
            assert ratio >= 4.5, (ink, ground, round(ratio, 2))
    for hue in style.HUES:
        for tone in ("w", "a"):
            assert contrast(palette["ink"], palette[f"{hue}-{tone}"]) >= 4.5, (hue, tone)


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Every figure `make figures` writes into docs/figures, drawn afresh, by the name it takes there."""
    out = tmp_path_factory.mktemp("figures")
    # make diagrams: the method figure and the system diagrams.
    drawn = diagrams.render(out / "diagrams")
    # make report-figures: the technical report's, each from the records it cites.
    report = report_figures.main(out / "report")
    return {path.name: path for path in (*drawn, *report)}


def test_every_svg_under_docs_figures_is_what_its_generator_draws(drawn: dict[str, Path]) -> None:
    assert sorted(p.name for p in DOCS.glob("*.svg")) == sorted(drawn)
    stale = [
        name for name, path in drawn.items() if path.read_bytes() != (DOCS / name).read_bytes()
    ]
    assert not stale, f"run make figures and commit what it draws: {stale}"


def webp_frames(path: Path) -> list[int]:
    """An animated WebP's frames, each its duration in ms, from its ANMF chunks: libwebp merges a
    frame that repeats the one before into it, so the count can differ where the time cannot."""
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP", path
    durations, at = [], 12
    while at + 8 <= len(data):
        kind, size = data[at : at + 4], int.from_bytes(data[at + 4 : at + 8], "little")
        if kind == b"ANMF":
            durations.append(int.from_bytes(data[at + 8 + 12 : at + 8 + 15], "little"))
        at += 8 + size + (size & 1)
    return durations


def test_the_readme_animation_is_a_looping_webp_in_each_theme_and_layout() -> None:
    """docs/figures/e2-demo.webp, its dark theme and its phone layout, which make demo-video draws
    from the demo campaign's recorded calls (a release's raw records, not this tree's): the same
    span of the calls in each, at twice the width the README shows each layout at, the same size in
    both themes, small enough to show inline."""
    from PIL import Image

    seen = {}
    for name, theme in (
        ("e2-demo", style.LIGHT),
        ("e2-demo-dark", style.DARK),
        ("e2-demo-narrow", style.LIGHT),
        ("e2-demo-narrow-dark", style.DARK),
    ):
        path = DOCS / f"{name}.webp"
        assert path.stat().st_size < 8_000_000, name
        with Image.open(path) as image:
            assert image.format == "WEBP" and image.is_animated, name
            assert image.info.get("loop") == 0, name  # it loops
            page = tuple(int(theme["page"][k : k + 2], 16) for k in (1, 3, 5))
            assert image.convert("RGB").getpixel((2, 2)) == page, name  # its own page
            seen[name] = (image.size, sum(webp_frames(path)))
    assert seen["e2-demo"][0] == seen["e2-demo-dark"][0] and seen["e2-demo"][0][0] == 1680
    assert seen["e2-demo-narrow"][0] == seen["e2-demo-narrow-dark"][0]
    assert seen["e2-demo-narrow"][0][0] == 680
    spans = {span for _, span in seen.values()}
    assert max(spans) - min(spans) <= 20 and min(spans) > 5_000, spans  # within a frame at 60 fps


def test_the_figures_set_their_type_in_its_sizes_and_its_two_weights(
    drawn: dict[str, Path],
) -> None:
    allowed = SIZES | {round(size * 0.72, 2) for size in SIZES}
    for name, path in drawn.items():
        svg = path.read_text()
        sizes = {float(size) for size in re.findall(r'font-size="([\d.]+)"', svg)}
        assert sizes <= allowed, (name, sizes - allowed)
        # Regular, and the semibold of a panel's title and a figure's featured value: no other weight, and none
        # synthesized.
        assert set(re.findall(r"font-weight:(\d+)", svg)) <= {"400", "600"}, name
        assert "font-synthesis:none" in svg and not re.search(r"font-weight=|bold", svg), name
        # Each paints its own page, under prefers-color-scheme's palette.
        assert '<rect class="page"' in svg and "prefers-color-scheme:dark" in svg, name
        # Red text clears 4.5:1 on the page alone, not on a panel's ground: the diagrams set none.
        if not name.startswith("report-"):
            assert 'class="red tint' not in svg, name


GLOSSARY = REPO / "docs" / "GLOSSARY.md"


def retired_names() -> list[str]:
    """The names GLOSSARY.md's "One name for each concept" table retires (its Never column), but those it retires for
    one use only, which it marks with a note in parentheses, as "make-room (for the rule)" stays the variant's name."""
    table = GLOSSARY.read_text().split("## One name for each concept", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("|")][2:]
    assert rows, "GLOSSARY.md has no One name for each concept table"
    names = [name.strip() for row in rows for name in row.strip("|").split("|")[2].split(",")]
    return [name for name in names if "(" not in name]


def test_every_figure_names_each_concept_as_the_glossary_does(drawn: dict[str, Path]) -> None:
    # No figure's drawn text, title or description gives a concept a name the glossary retired: the figures take their
    # configurations' names from flatshare_bench.labels, and their other words follow the same table.
    retired = retired_names()
    assert {"the prefill hold", "session tree", "abort-and-resubmit"} <= set(retired)
    for name, path in drawn.items():
        svg = re.sub(r"<style>.*?</style>", " ", path.read_text(), flags=re.S)
        text = " ".join(re.sub(r"<[^>]+>", " ", svg).split())
        for old in retired:
            assert not re.search(rf"(?<![\w-]){re.escape(old)}s?(?![\w-])", text, re.I), (name, old)


SHOWN_IN = ("README.md", "docs", "ops", "crates", "python", "deploy")
"""Where the hand-written documents that show the figures live; the records under benchmarks/results keep their own."""
SKIPPED = {".venv", "node_modules", "target", "build"}


def test_every_figure_the_docs_show_has_its_drawings_description_as_its_alt_text() -> None:
    # A reader who cannot see a figure reads what its generator states from the records, word for word, in every
    # document that shows it: the alt text of each <img> of a figure under docs/figures is that SVG's <desc>.
    image = re.compile(r'<img src="[^"]*figures/([a-z0-9-]+)\.svg"[^>]*? alt="([^"]*)"')
    docs = [REPO / "README.md"] + [
        doc
        for top in SHOWN_IN[1:]
        for doc in sorted((REPO / top).rglob("*.md"))
        if not SKIPPED & set(doc.relative_to(REPO).parts)
    ]
    shown = 0
    for doc in docs:
        for name, alt in image.findall(doc.read_text()):
            desc = (
                ET.parse(DOCS / f"{name}.svg")
                .getroot()
                .find("{http://www.w3.org/2000/svg}desc")
                .text
            )
            assert html.unescape(alt) == desc, (doc.relative_to(REPO).as_posix(), name)
            shown += 1
    assert shown >= 50, shown
