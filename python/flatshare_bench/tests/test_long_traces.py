"""benchmarks/e2/long_traces.py: the long corpus E2's discriminating points replay, as links into the scaled corpus."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LONG_TRACES = REPO / "benchmarks" / "e2" / "long_traces.py"


def long_traces(*args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(LONG_TRACES), *map(str, args)], capture_output=True, text=True
    )


def corpus(root: Path, inputs: dict[str, list[int]]) -> Path:
    """A corpus of traces, each with requests of the given `in`s."""
    root.mkdir(parents=True)
    for name, ins in inputs.items():
        # An entry without `in` counts for nothing.
        requests = [{"in": n} for n in ins] + [{"out": 7}]
        (root / f"{name}.json").write_text(json.dumps({"requests": requests}))
    return root


def links(out: Path) -> dict[str, Path]:
    return {path.name: path.readlink() for path in sorted(out.iterdir()) if path.is_symlink()}


def test_the_largest_traces_are_linked_and_only_they(tmp_path: Path) -> None:
    traces = corpus(
        tmp_path / "weka-x32",
        {"a": [100, 300], "b": [4000], "c": [2000, 2000, 9000], "d": [500], "e": [3000]},
    )
    out = tmp_path / "weka-x32-long"
    done = long_traces(traces, out, 3)
    assert done.returncode == 0, done.stderr
    assert (
        done.stdout
        == "5 traces, median input 2,000; the 3 largest: median input 3,000, 5 requests\n"
    )
    assert links(out) == {
        name: (traces / name).resolve() for name in ("b.json", "c.json", "e.json")
    }
    # Fewer: the links of the earlier run's other traces go, where they stayed and were replayed too. A link to
    # another file is pointed at the trace, and a file that is not a link stays.
    (out / "b.json").unlink()
    (out / "b.json").symlink_to(tmp_path / "elsewhere.json")
    (out / "notes.txt").write_text("kept\n")
    done = long_traces(traces, out, 2)
    assert done.returncode == 0, done.stderr
    assert links(out) == {name: (traces / name).resolve() for name in ("b.json", "e.json")}
    assert (out / "notes.txt").read_text() == "kept\n"


def test_a_corpus_without_traces_is_an_error(tmp_path: Path) -> None:
    # It ended in a StatisticsError traceback, after making OUT.
    empty = corpus(tmp_path / "empty", {"a": []})
    done = long_traces(empty, tmp_path / "out", 48)
    assert done.returncode == 1
    assert done.stderr == f"long_traces: no trace with requests in {empty} (*.json)\n"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "args", [(), ("corpus", "out"), ("corpus", "out", "0"), ("corpus", "out", "x")]
)
def test_its_arguments_are_checked(args: tuple[str, ...]) -> None:
    done = long_traces(*args)
    assert done.returncode == 2
    assert done.stderr == "usage: long_traces.py CORPUS OUT N (N: 1 or more traces)\n"
