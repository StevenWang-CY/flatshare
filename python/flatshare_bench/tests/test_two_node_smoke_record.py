"""The two-node smokes' records (benchmarks/results/2026-10-28-two-node-smoke): their README's table against
table.py, and each summary against what the soak tool makes of its record today."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from flatshare_bench.soak.summary import load, render_markdown, summarize

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-28-two-node-smoke"


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_table", RECORD / "table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_readmes_table_is_what_the_records_give() -> None:
    readme = (RECORD / "README.md").read_text()
    found = re.search(r"<!-- table.py -->\n(.*?)\n<!-- /table.py -->", readme, re.S)
    assert found is not None
    assert found.group(1) == _table().table()


@pytest.mark.parametrize("smoke", ["idle-300", "idle-150"])
def test_each_summary_is_what_the_soak_tool_makes_of_its_record(smoke: str) -> None:
    data = load(RECORD / smoke)
    reference = (data.record.get("options") or {}).get("slo_reference_ms")
    made = render_markdown(summarize(data, reference, 250.0))
    assert made == (RECORD / smoke / "summary.md").read_text()
