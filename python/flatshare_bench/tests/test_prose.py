"""Every prose unit of the repository's Markdown holds at most 150 words, so that no document reads as a wall. A unit
is a paragraph or a list item joined across its wrapped lines; a link counts as its text and inline code as one word.
v1.9's Markdown outside the records held 67 units over 150 words in 13 files, the longest 305; v1.10 split them, and a
new one fails here.

Three kinds of Markdown are read as written: the records under benchmarks/results, CHANGELOG.md, and a forecast the
simulator wrote (it opens with "Written by `flatshare-sim forecast`"), which is its output from before the A100 run. A
new generated file fails until it is added to SKIPPED on purpose."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LIMIT = 150
"""The most words a prose unit may hold."""
SKIPPED = ("benchmarks/results/", "CHANGELOG.md")
"""Markdown kept as written: the records' directory and the release history."""
WRITTEN = "Written by `flatshare-sim forecast`"
"""How a forecast the simulator wrote opens."""
BUILD = {".venv", "build", "node_modules", "target"}
"""Directories a build or a test run fills, whose Markdown is not the repository's: build/ holds what make writes,
such as make reproduce-e2's report."""
HTML = re.compile(
    r"^</?(p|picture|source|img|h\d|div|a|br|details|summary|table|tr|td|th|sub|sup)\b",
    re.IGNORECASE,
)
LIST_ITEM = re.compile(r"^(?:[-*+]|\d+\.)\s+(.*)$")


def units(text: str) -> list[tuple[int, str]]:
    """The prose units of a Markdown text, each with its first line: paragraphs and list items, joined across wrapped
    lines. Fenced code, HTML blocks and comments, tables and headings are not prose."""
    found: list[tuple[int, str]] = []
    unit: list[str] = []
    start, fenced, comment, block = 0, False, False, ""

    def close() -> None:
        if unit:
            found.append((start, " ".join(unit)))
            unit.clear()

    for number, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        if s.startswith(("```", "~~~")):
            close()
            fenced = not fenced
        elif fenced:
            continue
        elif comment:
            comment = "-->" not in s
        elif s.startswith("<!--"):
            close()
            comment = "-->" not in s
        elif block:
            block = "" if re.search(rf"</{block}>", s, re.IGNORECASE) else block
        elif HTML.match(s):
            close()
            opened = re.match(r"^<(picture|p|div|details|table)\b", s, re.IGNORECASE)
            if opened and not re.search(rf"</{opened.group(1)}>", s, re.IGNORECASE):
                block = opened.group(1)
        elif not s or s.startswith(("|", "#")):
            close()
        else:
            s = s.lstrip("> ").strip() if s.startswith(">") else s
            if s.startswith("[!"):
                continue
            item = LIST_ITEM.match(s)
            if item:
                close()
                start, s = number, item.group(1)
            elif not unit:
                start = number
            unit.append(s)
    close()
    return found


def words(unit: str) -> int:
    """A unit's words: its tokens with a letter or a digit, a link counted as its text and inline code as one word."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", unit)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"``[^`]+``|`[^`]+`", "CODE", text)
    text = re.sub(r"https?://\S+", "URL", text)
    return sum(1 for token in text.split() if re.search(r"[A-Za-z0-9]", token))


def documents(top: Path = REPO) -> list[Path]:
    """Every Markdown file under top, the repository by default, but those SKIPPED and the forecasts the simulator
    wrote."""
    found = []
    for root, dirs, files in os.walk(top):
        dirs[:] = sorted(d for d in dirs if d not in BUILD)
        for name in sorted(files):
            path = Path(root) / name
            if name.endswith(".md") and not path.relative_to(top).as_posix().startswith(SKIPPED):
                first = units(path.read_text())[:1]
                if not (first and first[0][1].startswith(WRITTEN)):
                    found.append(path)
    return found


@pytest.mark.parametrize("path", documents(), ids=lambda path: path.relative_to(REPO).as_posix())
def test_no_prose_unit_holds_more_than_150_words(path: Path) -> None:
    walls = [(line, n) for line, unit in units(path.read_text()) if (n := words(unit)) > LIMIT]
    assert not walls, (
        f"units over {LIMIT} words (line, words): {walls}; split each where its ideas change"
    )


def test_the_reader_finds_a_wall_and_skips_code_tables_and_html() -> None:
    wall = " ".join(["word"] * 151)
    text = "\n".join(
        [
            "# A heading of words",
            "",
            " ".join(["word"] * 75),  # one paragraph, wrapped over two lines
            " ".join(["word"] * 76),
            "",
            "- a [linked item](https://example.com/a/b) with `inline code`",  # a, linked, item, with, the code
            "",
            "```",
            wall,
            "```",
            f"| {wall} |",
            f"<picture>\n{wall}\n</picture>",
            f"<!-- {wall} -->",
        ]
    )
    assert [(line, words(unit)) for line, unit in units(text)] == [(3, 151), (6, 5)]


def test_the_scan_reads_the_docs_and_skips_the_records() -> None:
    read = {path.relative_to(REPO).as_posix() for path in documents()}
    assert {
        "README.md",
        "docs/REPORT.md",
        "crates/flatshare-gateway/README.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
    } <= read
    assert not any(path.startswith(SKIPPED) for path in read)
    assert "benchmarks/a100/forecast.md" not in read
    assert "benchmarks/a100/rehearsal/README.md" in read


def test_the_scan_skips_what_a_build_or_a_run_writes(tmp_path: Path) -> None:
    for name in (
        "README.md",
        "build/reproduce-e2/frontier.md",
        "target/doc/index.md",
        "python/x/.venv/notes.md",
        "benchmarks/results/run/README.md",
        "CHANGELOG.md",
    ):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("# A record\n\nSome words.\n")
    assert [path.relative_to(tmp_path).as_posix() for path in documents(tmp_path)] == ["README.md"]
