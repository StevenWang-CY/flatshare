"""The optimization rounds' record (benchmarks/results/2026-11-02-opt-rounds): its README's tables are what its table.py
makes of the rounds' points, and every round ran every point its campaign lists."""

from __future__ import annotations

import importlib.util
import re
import statistics
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


def test_benchmarks_gives_the_rounds_tables_as_the_record_does() -> None:
    # BENCHMARKS.md's ledger of the rounds is the record's own tables of them, row for row.
    table = _table()
    benchmarks = (REPO / "docs" / "BENCHMARKS.md").read_text()
    for name in ("slo", "voice", "agents"):
        assert table.table(name) in benchmarks, name


def test_production_gives_the_two_callers_rounds_as_the_record_does() -> None:
    # PRODUCTION.md's rows of the rounds beside 24 sessions with two callers: each configuration's replies over the SLO,
    # pooled, and the medians over the rounds of the agents' token rate and their TTFT p50 and p99.
    table = _table()
    rows = table.points()
    production = (REPO / "docs" / "PRODUCTION.md").read_text()
    labels = (
        "| v1.14's engine beside 24 sessions, five more rounds on new samples | 24 |",
        "| **The same with v1.15's first four additions** | 24 |",
    )
    [(_, before, after)] = [s for s in table.SCENARIOS if s[0] == "24 agent sessions, 2 callers"]
    for label, point in zip(labels, (before, after), strict=True):
        rounds = table.by_round(point, rows)
        over, replies = sum(r[1] for r in rounds), sum(r[2] for r in rounds)
        got = [rows[name, point] for name in table.ran(rows) if (name, point) in rows]
        tokens = statistics.median(r["background_tokens_per_s"] for r in got)
        p50 = statistics.median(r["agent_ttft_p50_ms"] for r in got) / 1e3
        p99 = statistics.median(r["agent_ttft_p99_ms"] for r in got) / 1e3
        row = (
            f"{label} {over} / {replies} ({100 * over / replies:.1f}%) | {tokens:,.0f} | "
            f"{p50:.0f} / {p99:.0f} s |"
        )
        assert row in production, row
