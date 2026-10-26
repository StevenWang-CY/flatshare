"""The docs' figures: what their style refuses to draw, the contrast of its inks, and that every SVG under
docs/figures is what its generator draws today from the committed records (`make figures` rewrites them), so a
change to a generator, a record or figure.toml fails here until the figures are drawn again; and the README's
animation, which make demo-video draws from recorded calls outside the tree."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from leaseholder_bench import diagrams
from leaseholder_bench import figure_style as style
from leaseholder_bench.e2.report import load_points, read_dots, write_report

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs" / "figures"
W4 = REPO / "benchmarks" / "results" / "2026-10-23-e2"
W5 = REPO / "benchmarks" / "results" / "2026-10-24-e2-w5"
A100 = REPO / "benchmarks" / "results" / "2026-10-26-a100"
SIZES = {style.TITLE, style.LABEL, style.VALUE, style.NOTE}
"""The type's sizes, one per role; a sub- or superscript sets its role's size at 72%."""


def makefile_list(name: str) -> list[str]:
    """A list the Makefile sets with ``name :=``, across its continued lines."""
    found = re.search(
        rf"^{name} := ((?:.*\\\n)*.*)$", (REPO / "Makefile").read_text(), re.MULTILINE
    )
    assert found, name
    return found[1].replace("\\\n", " ").split()


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
    with pytest.raises(ValueError, match="make room for its shaft"):
        figure().flow([(10, 50), (24, 50)])  # under 2.2 head lengths
    figure().flow([(10, 50), (10 + style.MIN_SHAFT * style.FLOW_HEAD, 50)])
    with pytest.raises(ValueError, match="does not fit inside its box"):
        figure().box(10, 10, 60, 30, "a long name")
    with pytest.raises(ValueError, match="text takes ink, soft, mute or a hue's tint"):
        figure().text(10, 40, "faint", cls="faint")
    with pytest.raises(ValueError, match="a bar is one of"):
        figure().hbar(10, 10, 50, 20, "red")


def test_a_figure_is_the_same_bytes_every_time_and_embeds_only_its_glyphs() -> None:
    def draw() -> str:
        s = figure()
        s.text(10, 40, "Voice $p_{95}$ \N{MINUS SIGN}2")
        s.hbar(10, 60, 120, 20, "hi")
        return s.render()

    first = draw()
    assert first == draw()
    assert first.count("@font-face") == 2  # the text's faces, regular and italic (p); no math glyph
    assert "font-weight:400" in first and "600" not in first and "bold" not in first


def luminance(color: str) -> float:
    channels = [int(color[k : k + 2], 16) / 255 for k in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("palette", [style.LIGHT, style.DARK], ids=["light", "dark"])
def test_every_ink_clears_four_and_a_half_to_one_on_every_ground(palette: dict[str, str]) -> None:
    # Text is ink, soft or mute, or a hue that names an identity: the change's blue, the voice turn's green, and
    # the SLO's orange, which only the charts set, on the page. A label sits on the page, a panel's ground or KV
    # memory; a component's name is ink.
    grounds = ("page", "ground-a", "ground-b", "memory")
    for ink, on in [
        *((ink, grounds) for ink in ("ink", "soft", "mute", "blue", "green")),
        ("orange", ("page",)),
    ]:
        for ground in on:
            ratio = contrast(palette[ink], palette[ground])
            assert ratio >= 4.5, (ink, ground, round(ratio, 2))
    for ground in ("box", "ours"):
        assert contrast(palette["ink"], palette[ground]) >= 4.5, ground


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Every figure `make figures` copies into docs/figures, drawn afresh, by the name it takes there."""
    out = tmp_path_factory.mktemp("figures")
    # make e2-figures: the configurations an operator could deploy, and the waterfall's points.
    deployable = [
        run.parent
        for name in makefile_list("E2_DEPLOYABLE")
        for run in W4.glob(f"*/{name}/run.json")
    ]
    rows, incomplete = load_points(deployable)
    assert rows and not incomplete
    write_report(rows, out / "e2", makefile_list("E2_STAGES"))
    # make e2-w5-figure: the rows figure.toml names, from all of W5's records.
    rows, incomplete = load_points(sorted(run.parent for run in W5.glob("*/*/run.json")))
    assert rows and not incomplete
    write_report(rows, out / "w5", dots=read_dots(W5 / "figure.toml"))
    # make e2-gpus-figure: the rows the A100's figure.toml names, from W5's records and the A100's core, tuned vLLM's
    # and the PRs' values' at its step cap.
    a100 = sorted(
        run.parent
        for campaign in ("a100-r", "a100-tuned-r", "a100-prtuned-r")
        for run in A100.glob(f"e2/{campaign}[0-9]/*/run.json")
    )
    rows, incomplete = load_points(sorted(run.parent for run in W5.glob("*/*/run.json")) + a100)
    assert rows and not incomplete
    write_report(rows, out / "gpus", dots=read_dots(A100 / "figure.toml"))
    # make method-figure.
    diagrams.render(out)
    return {
        "e2-frontier.svg": out / "e2" / "frontier-weka-x32-long.svg",
        "e2-waterfall.svg": out / "e2" / "waterfall.svg",
        "e2-voice-p95.svg": out / "w5" / "voice-p95.svg",
        "e2-voice-p95-narrow.svg": out / "w5" / "voice-p95-narrow.svg",
        "e2-voice-p95-gpus.svg": out / "gpus" / "voice-p95.svg",
        "e2-voice-p95-gpus-narrow.svg": out / "gpus" / "voice-p95-narrow.svg",
        "method.svg": out / "method.svg",
        "method-narrow.svg": out / "method-narrow.svg",
    }


def test_every_svg_under_docs_figures_is_what_its_generator_draws(drawn: dict[str, Path]) -> None:
    assert sorted(p.name for p in DOCS.glob("*.svg")) == sorted(drawn)
    stale = [
        name for name, path in drawn.items() if path.read_bytes() != (DOCS / name).read_bytes()
    ]
    assert not stale, f"run make figures and commit what it draws: {stale}"


def test_the_readme_animation_is_a_looping_webp_in_each_theme_and_layout() -> None:
    """docs/figures/e2-demo.webp, its dark theme and its phone layout, which make demo-video draws
    from the demo campaign's recorded calls (a release's raw records, not this tree's): the same
    frames in each theme, at twice the width the README shows each layout at, small enough to show
    inline."""
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
            seen[name] = (image.size, image.n_frames)
            page = tuple(int(theme["page"][k : k + 2], 16) for k in (1, 3, 5))
            assert image.convert("RGB").getpixel((2, 2)) == page, name  # its own page
    assert seen["e2-demo"] == seen["e2-demo-dark"] and seen["e2-demo"][0] == (1680, 588)
    assert seen["e2-demo-narrow"] == seen["e2-demo-narrow-dark"]
    assert seen["e2-demo-narrow"][0][0] == 680 and seen["e2-demo-narrow"][1] == seen["e2-demo"][1]


def test_the_figures_set_their_type_in_its_sizes_and_never_in_bold(drawn: dict[str, Path]) -> None:
    allowed = SIZES | {round(size * 0.72, 2) for size in SIZES}
    for name, path in drawn.items():
        svg = path.read_text()
        sizes = {float(size) for size in re.findall(r'font-size="([\d.]+)"', svg)}
        assert sizes <= allowed, (name, sizes - allowed)
        assert "font-weight:400" in svg and not re.search(r"font-weight:(?!400)|bold", svg), name
        # Each paints its own page, under prefers-color-scheme's palette.
        assert '<rect class="page"' in svg and "prefers-color-scheme:dark" in svg, name
        # Orange text clears 4.5:1 on the page alone, not on a panel's ground: the method figure sets none.
        if name.startswith("method"):
            assert 'class="orange tint' not in svg, name
