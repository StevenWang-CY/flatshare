"""The back-to-back runs before and after the voice worker sent LiveKit its status as each call ends
(benchmarks/results/2026-10-29-status-send): the README's table against table.py, each summary against what the soak
tool makes of its record today, and table.py's split of the refusals against the summary's count."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from flatshare_bench.soak.summary import load, render_markdown, summarize

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-29-status-send"


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("status_send_table", RECORD / "table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_readmes_table_is_what_the_records_give() -> None:
    readme = (RECORD / "README.md").read_text()
    found = re.search(r"<!-- table.py table -->\n(.*?)\n<!-- /table.py table -->", readme, re.S)
    assert found is not None
    assert found.group(1) == _table().table()


@pytest.mark.parametrize("run", ["before", "after"])
def test_each_summary_is_what_the_soak_tool_makes_of_its_record(run: str) -> None:
    data = load(RECORD / run)
    reference = (data.record.get("options") or {}).get("slo_reference_ms")
    made = render_markdown(summarize(data, reference, 250.0))
    assert made == (RECORD / run / "summary.md").read_text()


@pytest.mark.parametrize("run", ["before", "after"])
def test_the_table_splits_every_refusal_the_summary_counts(run: str) -> None:
    """The attempts table.py reads hold the summary's refusals, so that its two columns of them add up to it."""
    table = _table()
    refused = [a for a in table.attempts(run) if a["outcome"] == "refused"]
    counts = table.counts(run)
    assert len(refused) == counts["busy"] + counts["free"]
