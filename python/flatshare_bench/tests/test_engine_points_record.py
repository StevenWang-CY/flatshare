"""The engine points' record (benchmarks/results/2026-11-01-engine-points): its README's tables are what its table.py
makes of the record's files."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-11-01-engine-points"


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("engine_points_table", RECORD / "table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_readmes_tables_are_what_the_records_give() -> None:
    table = _table()
    readme = (RECORD / "README.md").read_text()
    for name in table.TABLES:
        found = re.search(
            rf"<!-- table.py {name} -->\n(.*?)\n<!-- /table.py {name} -->", readme, re.S
        )
        assert found is not None, name
        assert found.group(1) == table.compact(table.table(name))


def test_every_point_the_tables_name_is_in_the_record() -> None:
    """Each arm a table lists ran on at least one node, so that no row is silently missing from the README."""
    table = _table()
    ran = {arm for _, arm in table.points()}
    for name, arms in table.TABLES.items():
        assert set(arms) <= ran, (name, sorted(set(arms) - ran))


def test_benchmarks_gives_the_adopted_table_as_the_record_does() -> None:
    # BENCHMARKS.md's ledger of the serving engine's additions is the record's own table of them, row for row.
    benchmarks = (REPO / "docs" / "BENCHMARKS.md").read_text()
    # It prints each median over its range, two lines a cell: the record's table, set for the page.
    table = _table()
    assert table.compact(table.table("adopted")) in benchmarks
