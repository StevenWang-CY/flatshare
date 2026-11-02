"""The FP8 crash's record (benchmarks/results/2026-10-31-fp8-crash): its README's two tables of the runs, how each was
set up and what it did, are what benchmarks/fp8/summarize.py makes of the runs' files, each table whole."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-31-fp8-crash"


def _summarize() -> ModuleType:
    path = REPO / "benchmarks" / "fp8" / "summarize.py"
    spec = importlib.util.spec_from_file_location("fp8_summarize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block(text: str, header: str) -> str:
    """The table in text that opens with header, through its last row."""
    lines = text.split("\n")
    start = lines.index(header)
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return "\n".join(lines[start:end])


def test_the_readmes_tables_are_what_summarize_gives() -> None:
    readme = (RECORD / "README.md").read_text()
    for table in _summarize().tables(RECORD / "runs"):
        assert block(readme, table.split("\n")[0]) == table


def test_a_changed_cell_or_an_extra_row_fails() -> None:
    # The negative control: each table is compared whole, so a changed cell or a row after its last differs.
    _, outcome = _summarize().tables(RECORD / "runs")
    header = outcome.split("\n")[0]
    changed = outcome.replace("CRASHED at 124 s", "SURVIVED", 1)
    assert changed != outcome
    assert block(f"x\n\n{changed}\n\ny", header) != outcome
    extra = f"x\n\n{outcome}\n| r99 | SURVIVED | 900 | 1 / 0 | 1 | 0 of 1 |\n\ny"
    assert block(extra, header) != outcome
