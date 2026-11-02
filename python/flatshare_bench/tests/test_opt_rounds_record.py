"""The optimization rounds' record (benchmarks/results/2026-11-02-opt-rounds): its README's tables are what its table.py
makes of the rounds' points, and every round ran every point its campaign lists."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-11-02-opt-rounds"


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("opt_rounds_table", RECORD / "table.py")
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
        assert found.group(1) == table.table(name)


def test_every_round_ran_its_campaigns_points() -> None:
    table = _table()
    rows = table.points(table.ROUNDS)
    for name in table.ROUNDS:
        listed = re.findall(
            r"^arms/(\S+)\.toml$", (REPO / "benchmarks/e2" / f"{name}.campaign").read_text(), re.M
        )
        assert sorted(point for campaign, point in rows if campaign == name) == sorted(listed), name


def test_the_record_names_home_directories_as_a_tilde() -> None:
    # The export writes home directories as ~; a campaign's log, copied beside it, names its report under the home
    # directory of the host that ran it, which the record must not carry.
    leaked = [
        path.relative_to(RECORD)
        for path in RECORD.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and re.search(r"/(home|Users)/[^/\s]+", path.read_text(errors="replace"))
    ]
    assert not leaked, leaked
