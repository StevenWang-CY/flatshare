"""The envelope rounds' record (benchmarks/results/2026-11-08-envelope): its README's tables are what its table.py makes of
the points, every one of version 1.24's rounds ran every point its campaign lists, the record names no home directory,
and the serving profile's envelope is the gate's verdict."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-11-08-envelope"


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("envelope_table", RECORD / "table.py")
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
        assert found.group(1) == table.table(name), name


def test_every_round_ran_its_campaigns_points() -> None:
    table = _table()
    rows = table.points()
    for name in table.ROUNDS:
        listed = re.findall(
            r"^arms/(\S+)\.toml$", (REPO / "benchmarks/e2" / f"{name}.campaign").read_text(), re.M
        )
        assert sorted(point for campaign, point in rows if campaign == name) == sorted(listed), name


def test_the_rounds_ran_the_release_and_the_profiles_configuration() -> None:
    # Each of version 1.24's queue points ran the serve module's scheduler with the profile's two switches, and the
    # trial's round ran the trial's class, which the README names as such.
    table = _table()
    rows = table.points()
    serving = tomllib.loads((REPO / "deploy/profiles/rtx-5060-ti-serve.toml").read_text())[
        "serving"
    ]
    for campaign in table.ROUNDS:
        for _, _, point in table.GATE[1:]:
            engine = rows[campaign, point]["engine"]
            assert "--scheduler-cls flatshare_make_room_serve.AsyncScheduler" in engine, (
                campaign,
                point,
            )
            assert (
                f"--additional-config.flatshare_queue.max_wait_s {serving['queue_max_wait_s']}"
                in engine
            )
            assert (
                f"--additional-config.flatshare_copy.sm_load_programs {serving['sm_load_programs']}"
                in engine
            )
    for _, _, point in table.TRIAL_GATE[1:]:
        assert "--scheduler-cls opt2_srpf.AsyncScheduler" in rows[table.TRIAL, point]["engine"]


def test_the_profiles_envelope_is_the_gates_verdict() -> None:
    # max_agents is the most session trees at which the queue and the SM copy qualify, pooled over the three rounds.
    table = _table()
    rows = table.points()
    slo = table.slos(rows)
    assert sorted(slo) == list(table.ROUNDS)
    qualified = []
    for _, trees, point in table.GATE[1:]:
        found = [(c, rows[c, point]) for c in table.ROUNDS]
        over = sum(sum(v > slo[c] for v in row["v2v_samples"]) for c, row in found)
        total = sum(len(row["v2v_samples"]) for _, row in found)
        if over <= table.SHARE * total:
            qualified.append(trees)
    serving = tomllib.loads((REPO / "deploy/profiles/rtx-5060-ti-serve.toml").read_text())[
        "serving"
    ]
    assert serving["max_agents"] == max(qualified, default=4)


def test_the_record_names_home_directories_as_a_tilde() -> None:
    leaked = [
        path.relative_to(RECORD)
        for path in RECORD.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and re.search(r"/(home|Users)/[^/\s]+", path.read_text(errors="replace"))
    ]
    assert not leaked, leaked


def test_benchmarks_gives_the_records_tables() -> None:
    # BENCHMARKS.md's ledger of the windows and the rounds is the record's own tables, row for row.
    table = _table()
    benchmarks = (REPO / "docs" / "BENCHMARKS.md").read_text()
    for name in ("windows", "gate"):
        assert table.table(name) in benchmarks, name
