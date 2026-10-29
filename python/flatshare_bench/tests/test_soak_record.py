"""The 24-hour soak's records (benchmarks/results/2026-10-28-soak): their README's tables and BENCHMARKS.md's against
table.py, each summary against what the soak tool makes of its record today, and the calls given no agent with a line
free against the worker's status as it sent it to LiveKit."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

from flatshare_bench.soak.summary import load, render_markdown, summarize

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-28-soak"


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("soak_table", RECORD / "table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("block", ["table", "hours", "spare", "rules"])
def test_the_readmes_tables_are_what_the_records_give(block: str) -> None:
    readme = (RECORD / "README.md").read_text()
    found = re.search(
        rf"<!-- table.py {block} -->\n(.*?)\n<!-- /table.py {block} -->", readme, re.S
    )
    assert found is not None
    assert found.group(1) == getattr(_table(), block)()


def test_benchmarks_gives_the_records_table() -> None:
    assert _table().table() in (REPO / "docs" / "BENCHMARKS.md").read_text()


@pytest.mark.parametrize("run", ["voice-alone", "serving-24h"])
def test_each_summary_is_what_the_soak_tool_makes_of_its_record(run: str) -> None:
    data = load(RECORD / run)
    reference = (data.record.get("options") or {}).get("slo_reference_ms")
    made = render_markdown(summarize(data, reference, 250.0))
    assert made == (RECORD / run / "summary.md").read_text()


def test_the_soak_is_held_to_voice_alones_p95() -> None:
    """The soak's SLO is voice alone's replies' p95 in the reference run beside it, plus 250 ms."""
    reference = summarize(load(RECORD / "voice-alone"))["overall"]["turns"]["replies"]["p95"]
    soak = load(RECORD / "serving-24h").record["options"]["slo_reference_ms"]
    assert soak == pytest.approx(reference, abs=0.05)


def _line_free(attempt: dict, events: list[list[str]]) -> bool:
    """Whether LiveKit turned ``attempt`` away while the worker had a line free: its gate open (the status file's
    reason "N of M calls"), the last status it had sent LiveKit full, and a call's process exited since. LiveKit
    dispatches a worker it holds full no call, so the worker's calls can only fall from that status on."""
    if not re.fullmatch(r"\d+ of \d+ calls", attempt["worker"]["reason"]):
        return False
    mine = [
        (float(at), event)
        for at, worker, event, _ in events
        if worker == str(attempt["worker"]["pid"])
    ]
    sent = [(at, event) for at, event in mine if event != "exit" and at <= attempt["started"]]
    return (
        bool(sent)
        and sent[-1][1] == "full"
        and any(sent[-1][0] < at < attempt["started"] for at, event in mine if event == "exit")
    )


def test_calls_given_no_agent_with_a_line_free_are_those_livekit_held_full() -> None:
    """The calls the README names as having found no agent with a line free are the refused attempts LiveKit turned
    away while the worker had one (worker-events.tsv), and they hold every one the soak counts as ``no_agent``."""
    events = [
        line.split("\t") for line in (RECORD / "worker-events.tsv").read_text().splitlines()[1:]
    ]
    attempts = (RECORD / "serving-24h" / "attempts.jsonl").read_text().splitlines()
    refused = [a for a in map(json.loads, attempts) if a["outcome"] == "refused"]
    line_free = {a["call"] for a in refused if _line_free(a, events)}
    assert {a["call"] for a in refused if a["reason"] == "no_agent"} <= line_free
    readme = " ".join((RECORD / "README.md").read_text().split())
    named = re.search(
        r"found no agent with a line free\.\*\* Calls? ((?:\d+(?:, | and )?)+) came", readme
    )
    assert named is not None
    assert {int(call) for call in re.findall(r"\d+", named.group(1))} == line_free
