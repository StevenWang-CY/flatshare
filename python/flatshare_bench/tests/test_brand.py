"""The repository's logo, mark and social preview (docs/brand): each SVG is what flatshare_bench.brand draws
today (`make brand` rewrites them), the PNG was rendered from the committed social-preview.svg at GitHub's size,
the wordmark is kerned as a shaper would set it, and nothing in the preview leaves its page or touches the pool."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from flatshare_bench import brand

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs" / "brand"


@pytest.mark.parametrize("name", sorted(brand.svgs()))
def test_each_committed_svg_is_what_the_generator_draws(name: str) -> None:
    assert (DOCS / name).read_text() == brand.svgs()[name], f"make brand redraws {name}"


def test_the_social_preview_png_is_the_committed_svg_at_githubs_size() -> None:
    png = (DOCS / "social-preview.png").read_bytes()
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == brand.PREVIEW
    source = (DOCS / "social-preview.svg").read_text()
    assert brand.source_of(png) == hashlib.sha256(source.encode()).hexdigest()
    assert len(png) < 1_000_000  # GitHub takes a social preview under 1 MB


def test_the_wordmark_is_kerned_by_the_fonts_pairs() -> None:
    # The fonts' own pair adjustments, as HarfBuzz applies them to these pairs (font units of 1000 an em).
    for name in (brand.SEMIBOLD, brand.REGULAR):
        cmap = brand._font(name).getBestCmap()
        kern = brand._kerning(name)
        pairs = {
            pair: kern(cmap[ord(pair[0])], cmap[ord(pair[1])]) for pair in ("re", "Tu", "AV", "WA")
        }
        assert pairs == {"re": -10, "Tu": -30, "AV": -100, "WA": -105}, name
    # So the wordmark's advance is its glyphs' advances and its one kerned pair, r before e.
    font = brand._font(brand.SEMIBOLD)
    metrics, cmap = font["hmtx"].metrics, font.getBestCmap()
    units = sum(metrics[cmap[ord(char)]][0] for char in brand.WORDMARK) - 10
    assert brand.outline(brand.WORDMARK, brand.SEMIBOLD, 1000, 0, 0).advance == pytest.approx(units)


def test_nothing_in_the_social_preview_leaves_its_page_or_meets_another_part() -> None:
    width, height = brand.PREVIEW
    parts = brand.preview_parts()
    margin = 48
    for part in parts:
        x0, y0, x1, y1 = part.bounds
        assert margin <= x0 and x1 <= width - margin and margin <= y0 and y1 <= height - margin, (
            part.bounds
        )
    # No two parts' ink meets, the text keeps 40 units from the pool, and the lines keep 6 from each other.
    for i, a in enumerate(parts):
        for b in parts[i + 1 :]:
            gap = 40 if a is parts[0] else 6
            apart = (
                a.bounds[2] + gap <= b.bounds[0]
                or b.bounds[2] + gap <= a.bounds[0]
                or a.bounds[3] + 6 <= b.bounds[1]
                or b.bounds[3] + 6 <= a.bounds[1]
            )
            assert apart, (a.bounds, b.bounds)
