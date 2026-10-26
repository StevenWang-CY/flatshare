"""The perf check's negative control, which make perf-make-room runs after the check (FLATSHARE_PERF=1): make-room built
with two of #63350's optimizations undone fails the check against the checked-in baseline in the states where that work
runs, make-room built with a cost on its switches-off path fails it against stock vLLM, and make-room itself passes it
in those states, each timed in one process per state (a state that fails is measured again over two more).

Undone, each step that tries to make room sums the decodes' growth: in state B, where two victims cannot make room
(44937b87a gave up before the sum), and in state C, where the head fits (e87503f22 checked the fit against a bound
first). That is the change's cost with its switches on, which on/off measures. With both switches off, the peek at
the waiting head returns before it looks for one (_peek_make_room_request); without that return, each step looks the
head's prefix-cache hit up as it does with the switches on, a cost that stock vLLM does not pay and on/off cannot see,
which off/stock measures. The control times the scheduler, as the check does, so CI never runs it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import perf_schedule as P
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("FLATSHARE_PERF"),
    reason="it times the scheduler: make perf-make-room runs it",
)

# The baseline the check ran against: make passes its PERF_BASELINE; by default, the checked-in one.
BASELINE = Path(
    os.environ.get("FLATSHARE_PERF_BASELINE")
    or Path(__file__).resolve().parents[3] / "benchmarks/perf/make-room-schedule.json"
)
SLOWER = (
    (  # give up before counting growth: never
        "            num_blocks_short > 0\n            and not self.need_mamba_block_aligned_split\n",
        "            False\n            and not self.need_mamba_block_aligned_split\n",
    ),
    (  # check the fit against a bound first: never
        "            self._max_growth_blocks is not None\n            and num_blocks_short",
        "            False\n            and num_blocks_short",
    ),
)
STATES = ("B64", "C256")
# The peek's early return with both switches off, gone.
OFF_PATH = (
    (
        "            not (\n"
        "                self.max_priority_preemptions_per_step\n"
        "                or self.priority_prefill_budget_cap\n"
        "            )\n"
        "            or defer_prefills\n",
        "            defer_prefills\n",
    ),
)
# Where the head's lookup is a large share of the step: a few running requests, the head's 127-block hit.
OFF_STATES = ("B16", "C16")


@pytest.fixture(scope="module")
def real(tmp_path_factory) -> Path:
    pytest.importorskip("torch")
    return P.build_module(tmp_path_factory.mktemp("make-room"))


def mutant(real: Path, edits, out: Path) -> Path:
    """make-room's module with ``edits`` made, each found once, in a directory of its own."""
    source = real.read_text()
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    out.mkdir()
    (out / real.name).write_text(source)
    return out / real.name


def check(module: Path, out: Path, states) -> list[str]:
    """The check's failures for ``module`` in ``states``; its exit status must say whether there are any."""
    cmd = [sys.executable, P.__file__, "check", str(BASELINE), "--rounds", "1"]
    cmd += ["--states", ",".join(states), "--module", str(module), "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    failures = json.loads(out.read_text())["failures"]
    assert proc.returncode == (1 if failures else 0), proc.stdout + proc.stderr
    return failures


def test_a_module_made_slower_fails_the_check(real, tmp_path):
    failures = check(mutant(real, SLOWER, tmp_path / "slower"), tmp_path / "slower.json", STATES)
    assert sorted(failure.split(":")[0] for failure in failures) == sorted(STATES), failures
    assert all(" on/off " in failure for failure in failures), failures
    print("\nthe negative control, make-room with two of #63350's optimizations undone, fails:")
    print("\n".join(f"  {failure}" for failure in failures))


def test_a_module_costlier_with_its_switches_off_fails_the_check(real, tmp_path):
    off_path = mutant(real, OFF_PATH, tmp_path / "off-path")
    failures = check(off_path, tmp_path / "off-path.json", OFF_STATES)
    assert sorted(failure.split(":")[0] for failure in failures) == sorted(OFF_STATES), failures
    assert all(" off/stock " in failure for failure in failures), failures
    print("\nand make-room that looks the head up with its switches off fails:")
    print("\n".join(f"  {failure}" for failure in failures))


def test_make_room_passes_the_check_there(real, tmp_path):
    assert check(real, tmp_path / "make-room.json", STATES + OFF_STATES) == []
    print(f"\nand make-room passes it at {', '.join(STATES + OFF_STATES)}")
