"""The gateway's cost on the deployed build (benchmarks/results/2026-10-29-gateway-window): the README's tables against
table.py, its findings against results.json, the run against what the README says of it, and the docs that cite the
record against its first-token figure."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-29-gateway-window"


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gateway_window_table", RECORD / "table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _results() -> dict:
    return json.loads((RECORD / "results.json").read_text())


def _first_token() -> str:
    return f"{_results()['a']['paired_gateway_minus_direct_ms']['first_ms']['p50']:+.2f} ms"


def test_the_readmes_tables_are_what_the_results_give() -> None:
    readme = (RECORD / "README.md").read_text()
    found = re.search(r"<!-- table.py table -->\n(.*?)\n<!-- /table.py table -->", readme, re.S)
    assert found is not None
    assert found.group(1) == _table().table()


def test_the_readmes_findings_are_the_results() -> None:
    record = _results()
    findings = (
        (RECORD / "README.md")
        .read_text()
        .split("<!-- /table.py table -->")[1]
        .split("## How it ran")[0]
    )
    first = record["a"]["paired_gateway_minus_direct_ms"]["first_ms"]
    assert (
        f"**The gateway adds {first['p50']:+.2f} ms to the first token at the median**" in findings
    )
    assert f"over {first['n']} pairs" in findings and f"(p95 {first['p95']:+.2f} ms)" in findings
    gaps = {
        round(record["a"]["summary"][path]["gaps_ms"]["p50"], 1) for path in ("direct", "gateway")
    }
    assert len(gaps) == 1 and f"{gaps.pop():.1f} ms between tokens" in findings
    buckets = _table().overhead(record)
    total = buckets[-1][1]
    over = total - next(count for bound, count in buckets if bound >= 0.002)
    assert f"({over:.0f} of {total:.0f})" in findings
    tts = record["c"]["summary"]
    clause, sentence = (tts[k]["first_audio_ms"]["p50"] for k in ("clause", "sentence"))
    assert (
        f"({sentence - clause:.1f} ms between the medians, {clause:.0f} and {sentence:.0f} ms)"
        in findings
    )


def test_the_run_is_the_one_the_readme_describes() -> None:
    record = _results()
    assert record["errors"] == []
    assert record["host"] == "cyan" and record["started"].startswith("2026-10-29T19:30")
    assert record["a"]["same_text"] is True
    assert {stage["n"] for stage in record["a"]["paired_gateway_minus_direct_ms"].values()} == {80}
    assert {kind["first_audio_ms"]["n"] for kind in record["c"]["summary"].values()} == {40}


@pytest.mark.parametrize(
    "doc", ["README.md", "docs/SLO.md", "docs/REPORT.md", "crates/flatshare-gateway/README.md"]
)
def test_each_doc_that_cites_the_record_gives_its_first_token_figure(doc: str) -> None:
    text = (REPO / doc).read_text()
    assert "benchmarks/results/2026-10-29-gateway-window" in text
    assert _first_token() in text
