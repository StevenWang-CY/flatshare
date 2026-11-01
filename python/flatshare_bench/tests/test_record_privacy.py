"""No record names an account. The records under benchmarks/results are copied from shared hosts, and a copy writes
the home directory as ~ (the compat record's commands as /home/USER). A check for the account's own name misses a path
that `ps` cut short inside it, so this one refuses a home directory by its shape: /home/NAME or /Users/NAME, whatever
NAME is."""

from __future__ import annotations

import re
from pathlib import Path

RECORDS = Path(__file__).resolve().parents[3] / "benchmarks" / "results"
HOME = re.compile(rb"/(?:home|Users)/(?!USER\b)[^/\s\"'`]+")
"""A home directory by its shape, a Linux host's /home/NAME or a Mac's /Users/NAME, other than the placeholder."""
COMPILED = "__pycache__"
"""What Python writes beside a record's scripts when a test imports them: its bytecode names the checkout's own path,
and it is not the record."""


def homes(root: Path) -> list[str]:
    """Each file under root, outside what Python compiled, that names a home directory, with the first one it names."""
    found = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if COMPILED not in path.relative_to(root).parts and (m := HOME.search(path.read_bytes())):
            found.append(f"{path.relative_to(root)}: {m.group().decode(errors='replace')}")
    return found


def test_no_record_names_a_home_directory() -> None:
    assert homes(RECORDS) == []


def test_the_scan_finds_a_home_however_it_ends_and_keeps_the_placeholder(tmp_path: Path) -> None:
    (tmp_path / "snapshot.txt").write_text("vllm serve --config /home/ali\n")
    (tmp_path / "run.json").write_text('{"cwd": "/Users/bob/x"}\n')
    (tmp_path / "compat.log").write_text("--clone /home/USER/.cache, `/home/USER`, ~/flatshare\n")
    (tmp_path / COMPILED).mkdir()
    (tmp_path / COMPILED / "table.cpython-313.pyc").write_bytes(
        b"\x00/home/carol/flatshare/table.py"
    )
    assert homes(tmp_path) == ["run.json: /Users/bob", "snapshot.txt: /home/ali"]
