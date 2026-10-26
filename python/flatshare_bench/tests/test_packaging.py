"""What the repository's Python packages carry into their wheels: the repository's license, each its own copy, and the
license of the fonts flatshare-bench vendors."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PACKAGES = sorted(path.parent for path in (REPO / "python").glob("*/pyproject.toml"))
LICENSES = {
    # The figures' STIX Two fonts, under the SIL Open Font License 1.1, beside its text.
    "flatshare_bench": ("Apache-2.0 AND OFL-1.1", ["LICENSE", "src/flatshare_bench/fonts/OFL.txt"]),
}
"""Each package's license expression and license files where they are not the repository's alone."""
FONTS = {".ttf", ".otf", ".woff", ".woff2"}


def project(package: Path) -> dict:
    return tomllib.loads((package / "pyproject.toml").read_text())["project"]


def test_every_package_is_found() -> None:
    names = [package.name for package in PACKAGES]
    assert names == ["flatshare_bench", "flatshare_stats", "flatshare_vllm", "flatshare_voice"]


@pytest.mark.parametrize("package", PACKAGES, ids=lambda package: package.name)
def test_each_package_carries_its_licenses(package: Path) -> None:
    # A wheel built from a package's sdist cannot reach the repository's root, so each package keeps a copy of
    # LICENSE, which its pyproject.toml names (license-files): the copy must stay the root's, byte for byte.
    assert (package / "LICENSE").read_bytes() == (REPO / "LICENSE").read_bytes()
    meta = project(package)
    wanted = LICENSES.get(package.name, ("Apache-2.0", ["LICENSE"]))
    assert (meta["license"], meta["license-files"]) == wanted
    assert all((package / path).is_file() for path in meta["license-files"])


@pytest.mark.parametrize("package", PACKAGES, ids=lambda package: package.name)
def test_a_package_that_ships_fonts_ships_their_license(package: Path) -> None:
    # The fonts are under the SIL Open Font License, which travels with them: its text beside them, named in
    # license-files, and OFL-1.1 in the package's license expression.
    meta = project(package)
    for font in (path for path in (package / "src").rglob("*") if path.suffix in FONTS):
        ofl = font.parent / "OFL.txt"
        assert ofl.is_file(), font
        assert ofl.relative_to(package).as_posix() in meta["license-files"], font
        assert "OFL-1.1" in meta["license"].split(" AND "), font
