"""SECURITY.md's supported versions against the CHANGELOG: the version it names as the latest release is the newest
release's, and the unsupported row starts at the minor version before it. At v1.8 the table still named 1.7 as the
latest release; a release that moves the CHANGELOG and not this table fails here."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RELEASE = re.compile(r"^## \[v(\d+)\.(\d+)(?:\.\d+)?\]\(", re.MULTILINE)
"""A release's heading in CHANGELOG.md, newest first, as test_citation.py reads it: `## [v1.8](its page) ...`."""


def test_the_supported_versions_are_the_newest_release() -> None:
    match = RELEASE.search((REPO / "CHANGELOG.md").read_text())
    assert match, "CHANGELOG.md names no release"
    major, minor = map(int, match.groups())
    assert minor > 0, "a new major release names its predecessor by hand"
    security = (REPO / "SECURITY.md").read_text()
    assert f"At v{major}.{minor} that is:" in security
    assert f"| {major}.{minor} (the latest release) and `main` | Yes |" in security
    assert (
        f"| {major}.{minor - 1} and earlier, including the releases named Leaseholder | No |"
        in security
    )
