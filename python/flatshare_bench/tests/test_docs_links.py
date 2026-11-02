"""Every relative link and image in the repository's Markdown resolves as GitHub resolves it: a path to a file or a
directory of the repository, and an anchor to a heading of its target (GitHub's slug of the heading's text, numbered
from its second use) or to an explicit <a id> or <a name>. v1.17's Markdown held none broken; a new one fails here,
so a renamed heading or a moved file takes its links with it.

External links (a scheme, or //) are not followed: their targets change outside the repository."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import pytest

REPO = Path(__file__).resolve().parents[3]
SKIP = {".git", ".venv", ".pytest_cache", ".ruff_cache", "build", "node_modules", "target"}
"""Directories whose Markdown is not the repository's: version control's, and what builds and test runs write."""
LINK = re.compile(r"(?<!!)\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
HTML_TARGET = re.compile(r"""(?:src|srcset|href)=["']([^"']+)["']""")
ANCHOR = re.compile(r"""<a\s+(?:id|name)=["']([^"']+)["']""")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
EXTERNAL = re.compile(r"^[a-z][a-z0-9+.-]*:|^//")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: its rendered text in lower case, without punctuation or symbols, its spaces
    turned to hyphens."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("`", "").replace("**", "").replace("__", "")
    text = re.sub(r"(?<!\w)[*_]|[*_](?!\w)", "", text)
    kept = (c for c in text.strip().lower() if c in " -_" or unicodedata.category(c)[0] in "LNM")
    return "".join(kept).replace(" ", "-")


def anchors(path: Path) -> set[str]:
    """The anchors a Markdown file offers: each heading's slug, numbered from its second use, and the explicit
    anchors."""
    found: set[str] = set()
    seen: dict[str, int] = {}
    fenced = False
    for line in path.read_text().splitlines():
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            found.update(ANCHOR.findall(line))
            if heading := HEADING.match(line):
                base = slug(heading.group(2))
                n = seen.get(base, 0)
                seen[base] = n + 1
                found.add(base if n == 0 else f"{base}-{n}")
    return found


def broken(path: Path) -> list[str]:
    """Each relative link, image or HTML target of ``path`` that does not resolve, with its line."""
    found = []
    fenced = False
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = re.sub(r"`[^`]*`", "", line)  # inline code is no link
        targets = LINK.findall(line) + IMAGE.findall(line)
        for value in HTML_TARGET.findall(line):
            targets += [part.strip().split(" ")[0] for part in value.split(",")]
        for target in targets:
            if EXTERNAL.match(target):
                continue
            file, _, fragment = unquote(target).partition("#")
            dest = (path.parent / file).resolve() if file else path
            if not dest.exists():
                found.append(f"{number}: {target}: no such path")
            elif fragment and dest.suffix == ".md" and fragment not in anchors(dest):
                found.append(f"{number}: {target}: no such anchor")
    return found


def documents(top: Path = REPO) -> list[Path]:
    """Every Markdown file under top, the repository by default, but those in SKIP's directories."""
    found = []
    for root, dirs, files in os.walk(top):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        found += [Path(root) / name for name in sorted(files) if name.endswith(".md")]
    return found


@pytest.mark.parametrize("path", documents(), ids=lambda path: path.relative_to(REPO).as_posix())
def test_every_relative_link_resolves(path: Path) -> None:
    assert not broken(path)


def test_a_missing_file_anchor_or_image_fails_and_code_is_no_link(tmp_path: Path) -> None:
    # The negative control: each kind of broken target is found, and a link inside code is not one.
    (tmp_path / "other.md").write_text("# Other\n\n## A section, in µs: `code` and *more*\n")
    good = tmp_path / "good.md"
    good.write_text(
        "# Good\n\n[ok](other.md#a-section-in-µs-code-and-more) [self](#good) [dir](.)\n"
    )
    assert broken(good) == []
    bad = tmp_path / "bad.md"
    bad.write_text(
        "# Bad\n\n[file](missing.md) [anchor](other.md#nowhere)\n\n"
        '<img src="figures/none.svg">\n\n```\n[in code](missing.md)\n```\n`[inline](missing.md)`\n'
    )
    assert broken(bad) == [
        "3: missing.md: no such path",
        "3: other.md#nowhere: no such anchor",
        "5: figures/none.svg: no such path",
    ]


def test_the_slug_drops_markup_and_punctuation_and_a_repeat_takes_a_number(tmp_path: Path) -> None:
    assert slug("6.10 The serving engine's additions") == "610-the-serving-engines-additions"
    assert (
        slug("[REPORT §6.10](REPORT.md#x): the `make-room` variant")
        == "report-610-the-make-room-variant"
    )
    doc = tmp_path / "doc.md"
    doc.write_text('# Doc\n\n## Files\n\n## Files\n\n```\n## Files\n```\n\n<a id="kept"></a>\n')
    assert anchors(doc) == {"doc", "files", "files-1", "kept"}
