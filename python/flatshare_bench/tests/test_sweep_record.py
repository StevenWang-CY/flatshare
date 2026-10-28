"""The sweep's screens' record (benchmarks/results/2026-10-27-sweep) against its README's table."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-27-sweep"


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sweep_table", RECORD / "table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_readmes_table_is_what_the_records_give() -> None:
    readme = (RECORD / "README.md").read_text()
    found = re.search(r"<!-- table.py -->\n(.*?)\n<!-- /table.py -->", readme, re.S)
    assert found is not None
    assert found.group(1) == _table().table()


def test_the_screens_measure_against_w5s_slo() -> None:
    # W5's voice alone with two callers, the median of its five rounds' p95s (docs/SLO.md: 1,784 ms), plus 250 ms.
    assert _table().w5_slo() == 2034.3
