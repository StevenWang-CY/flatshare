"""make-room on vLLM 0.33.0 (benchmarks/results/2026-11-02-port-v0.33.0): the README's table against table.py, its
findings against the probe's logs, the engine it ran, and its make-room command against the repository README's."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-11-02-port-v0.33.0"
WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("port_record_table", RECORD / "table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command(text: str, start: str) -> list[str]:
    """The words of the command in ``text`` whose first line starts with ``start``, its continued lines joined."""
    lines = text.splitlines()
    words: list[str] = []
    for line in lines[next(n for n, line in enumerate(lines) if line.startswith(start)) :]:
        words += line.rstrip().removesuffix("\\").split()
        if not line.rstrip().endswith("\\"):
            return words
    raise AssertionError(f"{start!r}: the command does not end")


def test_the_readmes_table_is_what_the_logs_give() -> None:
    readme = (RECORD / "README.md").read_text()
    found = re.search(r"<!-- table.py table -->\n(.*?)\n<!-- /table.py table -->", readme, re.S)
    assert found is not None
    assert found.group(1) == _table().table()


def test_the_readmes_findings_are_the_logs() -> None:
    table = _table()
    readme = (RECORD / "README.md").read_text()
    findings = " ".join(
        readme.split("<!-- /table.py table -->")[1].split("## How it ran")[0].split()
    )
    stock, make_room = ([run["ttft"] for run in table.runs(engine)] for engine in table.ENGINES)
    assert (
        f"**On vLLM 0.33.0 make-room answered the voice turn in {table.span(make_room, ' ms')}**"
        in findings
    )
    assert (
        f"against {table.span(stock, ' ms')} on stock vLLM 0.33.0, in {WORDS[len(stock) - 1]} runs each"
        in findings
    )


def test_the_engines_were_vllm_0_33_0_and_make_room_built_for_it() -> None:
    environment = (RECORD / "environment.txt").read_text()
    assert re.search(r"^vllm 0\.33\.0, torch 2\.14\.0\+cu130, CUDA 13\.0$", environment, re.M)
    assert "'scheduler_cls': 'flatshare_make_room_scheduler.AsyncScheduler'" in environment


def test_the_record_ran_the_readmes_make_room_switches() -> None:
    # The quick start's make-room command but for vLLM 0.33.0's module directory and --host 127.0.0.1.
    ran = _command((RECORD / "commands.sh").read_text(), "PYTHONPATH=~/make-room-033 vllm serve")
    readme = _command((REPO / "README.md").read_text(), "PYTHONPATH=~/make-room vllm serve")
    assert ran[-2:] == ["--host", "127.0.0.1"]
    assert ran[1:-2] == readme[1:]
