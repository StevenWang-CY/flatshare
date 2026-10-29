"""CITATION.cff against the CHANGELOG: the software's version and release date are the newest release's. The citation
once gave 2026-10-27 for 1.7.0, released on 2026-10-29; a release that moves one file and not the other fails here."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RELEASE = re.compile(
    r"^## \[v(\d+\.\d+(?:\.\d+)?)\]\([^)]*\) \((\d{4}-\d{2}-\d{2})\)", re.MULTILINE
)
"""A release's heading in CHANGELOG.md, newest first: `## [v1.8](its release page) (2026-10-29): its title`."""


def top_level(cff: str, key: str) -> str:
    """A top-level key's value in CITATION.cff, as written (the preferred citation's keys are indented)."""
    [value] = re.findall(rf"^{re.escape(key)}: (.+)$", cff, re.MULTILINE)
    return value.strip().strip('"')


def test_the_citation_is_the_newest_release() -> None:
    match = RELEASE.search((REPO / "CHANGELOG.md").read_text())
    assert match, "CHANGELOG.md names no release"
    version, date = match.groups()
    cff = (REPO / "CITATION.cff").read_text()
    # The release's tag names its minor version (v1.8); the packages, and so the citation, the patch too (1.8.0).
    assert top_level(cff, "version") == (version if version.count(".") == 2 else f"{version}.0")
    assert top_level(cff, "date-released") == date
