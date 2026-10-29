"""The quick start on the box (benchmarks/results/2026-10-29-quick-start): the README's table against table.py, its
findings against the probe's logs, and the commands that ran against the repository README's quick start."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-29-quick-start"
MAKE_ROOM = "PYTHONPATH=~/make-room vllm serve"
SWITCHES = ("--scheduler-cls", "--additional-config.")
"""make-room's arguments, each followed by its value."""
WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("quick_start_table", RECORD / "table.py")
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


def _without_switches(words: list[str]) -> list[str]:
    kept = []
    for previous, word in zip(["", *words[:-1]], words, strict=True):
        if not word.startswith(SWITCHES) and not previous.startswith(SWITCHES):
            kept.append(word)
    return kept


def test_the_readmes_table_is_what_the_logs_give() -> None:
    readme = (RECORD / "README.md").read_text()
    found = re.search(r"<!-- table.py table -->\n(.*?)\n<!-- /table.py table -->", readme, re.S)
    assert found is not None
    assert found.group(1) == _table().table()


def test_the_readmes_findings_are_the_logs() -> None:
    table = _table()
    findings = (
        (RECORD / "README.md")
        .read_text()
        .split("<!-- /table.py table -->")[1]
        .split("## How it ran")[0]
    )
    stock, make_room = ([run["ttft"] for run in table.runs(engine)] for engine in table.ENGINES)
    assert f"**make-room answered the voice turn in {table.span(make_room, ' ms')}**" in findings
    assert (
        f"against {table.span(stock, ' ms')} on stock vLLM, in {WORDS[len(stock) - 1]} runs each"
        in findings
    )


def test_both_engines_ran_the_same_settings_but_make_rooms_switches() -> None:
    commands = (RECORD / "commands.sh").read_text()
    stock, make_room = _command(commands, "vllm serve"), _command(commands, MAKE_ROOM)
    assert _without_switches(make_room)[1:] == stock
    assert stock[-2:] == ["--host", "127.0.0.1"]


def test_the_record_ran_the_readmes_make_room_command() -> None:
    ran = _command((RECORD / "commands.sh").read_text(), MAKE_ROOM)
    assert ran[-2:] == ["--host", "127.0.0.1"]
    assert ran[:-2] == _command((REPO / "README.md").read_text(), MAKE_ROOM)


def test_the_environment_is_the_one_the_readme_gives() -> None:
    environment = (RECORD / "environment.txt").read_text()
    readme = (RECORD / "README.md").read_text()
    versions = re.search(r"^vllm (\S+), torch (\S+), CUDA (\S+)$", environment, re.M)
    assert versions is not None and versions.group(1) == "0.32.0"
    assert all(version in readme for version in versions.groups())
    pool = {
        int(n.replace(",", ""))
        for n in re.findall(r"GPU KV cache size: ([\d,]+) tokens", environment)
    }
    assert len(pool) == 1 and f"{pool.pop():,} tokens" in readme
