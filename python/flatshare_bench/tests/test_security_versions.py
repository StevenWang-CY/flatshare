"""SECURITY.md against the repository: its supported versions against the CHANGELOG, and the pins it lists with their
advisories against the requirements files. The version it names as the latest release is the newest release's, and
the unsupported row starts at the minor version before it; at v1.8 the table still named 1.7 as the latest release,
and a release that moves the CHANGELOG and not this table fails here. Each pin listed under "Advisories the pins
carry" is the pin of every file its entry names, so a release that moves the pin moves its entry too."""

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


PINNED = re.compile(r"^- \*\*`([\w.-]+)==([^`]+)`\*\*(.*?)(?=^- |^\S|\Z)", re.MULTILINE | re.DOTALL)
"""An entry under "Advisories the pins carry": `- **`name==version`**`, then its lines and sub-items."""
NAMED_FILE = re.compile(r"`((?:deploy|python)/[^`]+\.txt)`")


def pinned(security: str) -> list[tuple[str, str, list[str]]]:
    """The pins SECURITY.md lists with their advisories, each with the requirements files its entry names."""
    section = security.split("### Advisories the pins carry", 1)[1]
    return [
        (name, version, NAMED_FILE.findall(text)) for name, version, text in PINNED.findall(section)
    ]


def test_each_pin_with_an_advisory_is_the_pin_of_the_files_it_names() -> None:
    entries = pinned((REPO / "SECURITY.md").read_text())
    assert [name for name, _, _ in entries] == ["setuptools", "diskcache"]
    for name, version, files in entries:
        assert files, f"{name}'s entry names no requirements file"
        for file in files:
            lines = (REPO / file).read_text().splitlines()
            assert f"{name}=={version}" in lines, f"{file} no longer pins {name}=={version}"


def test_an_entry_names_each_of_its_files() -> None:
    entry = (
        "### Advisories the pins carry\n\n- **`setuptools==80.10.2`**, in (`deploy/a.txt`) and\n"
        "  (`python/b.txt`).\n  - Not `__init__.py` or `setuptools<81`.\n- **`diskcache==5.6.3`**, in\n"
        "  (`deploy/c.txt`).\n\nA paragraph naming `deploy/d.txt`.\n"
    )
    assert pinned(entry) == [
        ("setuptools", "80.10.2", ["deploy/a.txt", "python/b.txt"]),
        ("diskcache", "5.6.3", ["deploy/c.txt"]),
    ]
