"""benchmarks/results/README.md indexes the records: every record directory has exactly one row, and every row names a
record that exists and has a README. A new record fails here until the index names it."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "benchmarks" / "results"
ROW = re.compile(r"^\| \[`([^`]+)`\]\(([^)]+)\) \| \S", re.M)


def test_the_index_names_every_record_once() -> None:
    rows = ROW.findall((RESULTS / "README.md").read_text())
    names = [name for name, _ in rows]
    records = sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and any(p.iterdir()))
    assert sorted(names) == records
    for name, target in rows:
        assert target == f"{name}/README.md" and (RESULTS / target).is_file(), name
