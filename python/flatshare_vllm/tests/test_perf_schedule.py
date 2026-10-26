"""tests/perf_schedule.py (make perf-make-room): the cells' arithmetic, the tolerance and the check, the checked-in
baseline, and, on vLLM's real scheduler, that each of the 13 steady states is steady, for stock vLLM's scheduler as for
make-room's, and runs the path it is named for. Nothing here times anything: tests/test_perf_control.py, the check's
negative control, does, under make perf-make-room."""

import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import perf_schedule as P
import pytest

BASELINE = Path(__file__).resolve().parents[3] / "benchmarks/perf/make-room-schedule.json"


def process(
    state: str,
    off: float,
    on: float,
    off2: float | None = None,
    run: str = "a",
    stock: float | None = None,
) -> dict:
    """A process's timed steps: 40 of each configuration's two schedulers, the first's slowest 35 of them 50% above
    the given cost, as if something else had slowed them, and the second's all 20% above it, as if it ran slower.
    Stock vLLM costs what off does unless ``stock`` says otherwise."""
    times = {P.STOCK: off if stock is None else stock, "off": off, "on": on}
    times |= {} if off2 is None else {P.NOISE: off2}
    steps = {c: [[t] * 5 + [1.5 * t] * 35, [1.2 * t] * 40] for c, t in times.items()}
    return {"run": run, "rep": 1, "state": state, "order": list(steps), "times_us": steps}


def ratios(ratio: float, off_ratio: float = 0.0) -> dict:
    """A cell's two ratios: on/off - 1 and off/stock - 1."""
    return {"ratio": ratio, "off_ratio": off_ratio}


def test_a_configurations_cost_is_its_cheaper_schedulers_10th_percentile_step():
    """The 10th percentile is inclusive, as docs/GLOSSARY.md's percentiles are: of 40 steps, 0.9 of the way from the
    4th fastest to the 5th."""
    assert P.cost([15.0] * 35 + [10.0] * 5) == 10.0  # the 4th and 5th fastest of 40
    assert P.cost(list(range(1, 41))) == 4.9
    assert "method='inclusive'" in P.METHOD["cost"]
    assert P.cheapest([[12.0] * 40, [15.0] * 35 + [10.0] * 5]) == 10.0
    assert P.cheapest([[10.5] * 40, [15.0] * 35 + [10.0] * 5]) == 10.0
    assert P.cheapest([[9.0] * 40, [15.0] * 35 + [10.0] * 5]) == 9.0


def test_a_cells_ratio_is_the_median_of_its_processes_on_off():
    procs = [
        process("B64", 100, 160, 101),
        process("B64", 200, 300, 196),
        process("B64", 110, 187, 110),
    ]
    cell = P.cells(procs)["B64"]
    assert cell["ratios"] == [0.6, 0.5, 0.7] and cell["ratio"] == 0.6  # paired in each process
    assert cell["off_us"] == 110  # the median over processes
    assert cell["change_us"] == 66 and cell["on_us"] == 176  # at that cost: on = off x (1 + ratio)
    assert cell["noises"] == [0.01, -0.02, 0.0] and cell["noise"] == 0.0
    assert "noise" not in P.cells([process("C16", 100, 150)])["C16"]


def test_a_cells_off_ratio_is_the_median_of_its_processes_off_against_stock():
    """off/stock - 1 is what the change costs a step with its switches off, as a fraction of stock vLLM's step; stock's
    cost in µs is off's over that, so that off = stock x (1 + off_ratio)."""
    procs = [
        process("B16", 101, 250, stock=100),
        process("B16", 200, 500, stock=200),
        process("B16", 99, 250, stock=90),
    ]
    cell = P.cells(procs)["B16"]
    assert cell["off_ratios"] == [0.01, 0.0, 0.1] and cell["off_ratio"] == 0.01
    assert cell["off_us"] == 101 and cell["stock_us"] == 100 and cell["off_change_us"] == 1


def test_the_baseline_compares_its_half_b_with_its_half_a():
    procs = [process("C16", 100, 150, 101, run, stock=100) for run in "ab"]
    procs += [process("C16", 100, 160, 100, "a", 99), process("C16", 100, 158, 99, "b", 98)]
    procs += [process("C16", 100, 170, 100, "a", 98), process("C16", 100, 162, 101, "b", 100)]
    cell = P.baseline_cells(procs)["C16"]
    assert cell["ratios_a"] == [0.5, 0.6, 0.7] and cell["ratios_b"] == [0.5, 0.58, 0.62]
    assert cell["ratio_a"] == 0.6 and cell["ratio_b"] == 0.58
    assert cell["ratio"] == 0.59 and "ratios" not in cell  # over all six processes
    assert cell["aa"] == round(1.58 / 1.6 - 1, 4)
    # And off against stock, each half on its own.
    assert cell["off_ratios_a"] == [0.0, 0.0101, 0.0204] and cell["off_ratios_b"] == [
        0.0,
        0.0204,
        0.0,
    ]
    assert cell["off_ratio_a"] == 0.0101 and cell["off_ratio_b"] == 0.0
    assert cell["off_aa"] == round(1 / 1.0101 - 1, 4) and "off_ratios" not in cell


def test_the_tolerance_is_twice_the_worst_factor_of_any_noise():
    # The worst differed by a factor of 1/0.958: twice that is 1.0896, up to a whole percent.
    assert P.tolerance([0.01, -0.042], [0.02], [0.0]) == 0.09
    assert P.tolerance([0.01], [0.042, -0.001], [0.0]) == 0.09  # half against half in on/off
    assert P.tolerance([0.01], [0.0], [0.0, -0.042]) == 0.09  # and in off/stock
    assert P.tolerance([0.0, 0.001], [0.0], [0.0]) == 0.01
    assert P.tolerance([0.01], [0.0], [0.0]) == 0.03  # 1.01 ** 2 - 1 = 0.0201, rounded up


def test_the_check_fails_a_state_whose_on_off_grew_past_the_tolerance():
    baseline = {"tolerance": 0.10, "cells": {"B64": ratios(0.5), "C16": ratios(0.0)}}
    assert P.check({"B64": ratios(0.64)}, baseline) == []  # 1.64 / 1.5: +9.3%
    assert P.check({"B64": ratios(0.2), "C16": ratios(-0.05)}, baseline) == []  # cheaper
    [failure] = P.check({"B64": ratios(0.66), "C16": ratios(0.05)}, baseline)
    assert failure == "B64: on/off 1.66 against the baseline's 1.50, +10.7% (tolerance 10%)"
    assert P.check({"A16": ratios(0.0)}, baseline) == ["A16: not in the baseline"]


def test_the_check_fails_a_state_whose_off_against_stock_grew_past_the_tolerance():
    """With its switches off the change must cost a step what stock vLLM's does, as the PRs say it does: a state whose
    off/stock exceeds the baseline's by more than the tolerance fails."""
    baseline = {"tolerance": 0.04, "cells": {"B16": ratios(1.5, 0.002)}}
    assert P.check({"B16": ratios(1.5, 0.03)}, baseline) == []  # 1.03 / 1.002: +2.8%
    [failure] = P.check({"B16": ratios(1.5, 0.05)}, baseline)
    assert failure == "B16: off/stock 1.050 against the baseline's 1.002, +4.8% (tolerance 4%)"


def test_a_cost_common_to_off_and_on_fails_off_against_stock_and_reads_cheaper_in_on_off():
    """The same work added to every step, switches off or on: on/off falls, which alone would read as the change got
    cheaper, and off/stock fails."""
    baseline = {"tolerance": 0.04, "cells": P.cells([process("B64", 800, 1320, stock=800)])}
    measured = P.cells([process("B64", 1240, 1760, stock=800)])
    assert P.check(measured, baseline) == [
        "B64: off/stock 1.550 against the baseline's 1.000, +55.0% (tolerance 4%)"
    ]
    assert P.cheaper(measured, baseline) == ["B64: on/off 1.42 against the baseline's 1.65, -14.0%"]


def test_a_state_cheaper_past_the_tolerance_passes_and_is_reported():
    """A ratio that fell by more than the tolerance means that the configuration on top of it got cheaper or the one
    below it costlier: the baseline no longer describes the code, though nothing costs more."""
    baseline = {"tolerance": 0.04, "cells": {"B64": ratios(0.6), "C16": ratios(0.8, 0.05)}}
    # B64's on/off 1.5 against 1.6: -6.25%; C16's 1.75 against 1.8: -2.8%, and its off/stock 1.0 against 1.05: -4.8%.
    measured = {"B64": ratios(0.5), "C16": ratios(0.75)}
    assert P.check(measured, baseline) == []
    assert P.cheaper(measured, baseline) == [
        "B64: on/off 1.50 against the baseline's 1.60, -6.2%",
        "C16: off/stock 1.000 against the baseline's 1.050, -4.8%",
    ]
    assert (
        P.cheaper({"A16": ratios(0.0)}, baseline) == []
    )  # not in the baseline, which check() says
    lines = P.table(P.cells([process("B64", 1000, 1500)]), baseline).splitlines()
    assert lines[2].endswith("| +0.50 | +0.60 | -6.2% | +64 µs | cheaper |")


@pytest.mark.parametrize("again_ratio, fails", [(0.6, False), (0.7, True)])
def test_a_failing_state_is_measured_again_and_fails_only_if_it_still_exceeds(again_ratio, fails):
    baseline = {"tolerance": 0.04, "cells": {"B64": ratios(0.6), "C16": ratios(0.8)}}
    # B64: two of three processes strayed, +6.25% over the baseline's on/off; C16 within it.
    procs = [process("B64", 100, 170), process("B64", 100, 170), process("B64", 100, 160)]
    procs += [process("C16", 100, 180)] * 3
    calls = []

    def again(states):
        calls.append([s.key for s in states])
        return {"processes": [process("B64", 100, 100 * (1 + again_ratio))] * 6, "load": [{}]}

    result = {"processes": procs, "load": []}
    measured, failures, suspects = P.confirm(result, baseline, again)
    assert calls == [["B64"]] and suspects == ["B64"]  # C16 is not measured again
    assert len(measured["B64"]["ratios"]) == 9 and len(result["load"]) == 1
    assert bool(failures) == fails
    assert [f.split(":")[0] for f in failures] == (["B64"] if fails else [])


def test_a_state_failing_off_against_stock_is_measured_again_too():
    baseline = {"tolerance": 0.04, "cells": {"B16": ratios(1.5), "C16": ratios(0.8)}}
    procs = [process("B16", 105, 262.5, stock=100)] * 3  # off/stock +5%, on/off the baseline's
    procs += [process("C16", 100, 180)] * 3
    calls = []

    def again(states):
        calls.append([s.key for s in states])
        return {"processes": [process("B16", 100, 250, stock=100)] * 6, "load": []}

    measured, failures, suspects = P.confirm({"processes": procs, "load": []}, baseline, again)
    assert calls == [["B16"]] and suspects == ["B16"] and failures == []
    assert len(measured["B16"]["off_ratios"]) == 9 and measured["B16"]["off_ratio"] == 0.0


def test_a_passing_run_is_not_measured_again():
    baseline = {"tolerance": 0.04, "cells": {"B64": ratios(0.6)}}
    result = {"processes": [process("B64", 100, 161)] * 3, "load": []}
    measured, failures, suspects = P.confirm(result, baseline, lambda states: pytest.fail("again"))
    assert failures == [] and suspects == [] and measured["B64"]["ratio"] == 0.61


def test_the_table_says_how_much_more_a_step_may_cost_before_a_state_fails():
    """The change may cost the tolerance of the on step more before the state fails: off x tol x (1 + ratio)."""
    measured = P.cells([process("B64", 1000, 1620)])
    baseline = {"tolerance": 0.09, "cells": {"B64": ratios(0.62)}}
    head, _, line = P.table(measured, baseline).splitlines()
    assert head.endswith("| baseline | excess | fails past | |")
    assert line.endswith("| +0.62 | +0.0% | +146 µs | ok |")  # 1000 x 0.09 x 1.62


def test_a_cell_records_the_median_step_too_which_sees_work_on_half_the_steps():
    """The 10th-percentile step cannot see work done on fewer than about 9 steps in 10; the median-step ratio, kept
    as information beside it, can."""
    steps = {"stock": [[100.0] * 40] * 2, "off": [[100.0] * 40] * 2}
    steps["on"] = [[160.0] * 20 + [320.0] * 20] * 2
    cell = P.cells(
        [{"run": "a", "rep": 1, "state": "C16", "order": list(steps), "times_us": steps}]
    )["C16"]
    assert cell["ratio"] == 0.6 and cell["median_ratio"] == 1.4


def test_a_run_busier_than_its_baseline_by_half_says_so():
    base = [
        {"before": [2.2, 3.0, 3.1], "after": [3.6, 3.2, 3.1]},
        {"before": [3.6, 3.2, 3.1], "after": [3.1, 3, 3]},
    ]
    calm = [{"before": [3.0, 3.0, 3.0], "after": [5.3, 3.4, 3.1]}]
    assert P.load_warning(calm, base) is None  # 5.3 is under 1.5 x 3.6
    busy = [
        {"before": [3.0, 3.0, 3.0], "after": [5.5, 3.4, 3.1]},
        {"before": [5.5, 3, 3], "after": [9.1, 4, 3]},
    ]
    warning = P.load_warning(busy, base)
    assert "reached 9.1" in warning and "at most 3.6" in warning


def test_the_table_shows_off_on_the_change_and_the_ratio_against_the_baseline():
    measured = P.cells([process("B64", 1000, 1620), process("C16", 250, 450)])
    baseline = {"tolerance": 0.09, "cells": {"B64": ratios(0.62), "C16": ratios(0.6)}}
    lines = P.table(measured, baseline).splitlines()
    assert lines[0].startswith("| State | Running | off µs | on µs | change µs | ratio |")
    assert lines[2] == "| B | 64 | 1,000 | 1,620 | +620 | +0.62 | +0.62 | +0.0% | +146 µs | ok |"
    assert lines[3].endswith(
        "| 16 | 250 | 450 | +200 | +0.80 | +0.60 | +12.5% | +36 µs | costlier |"
    )
    assert "baseline" not in P.table(measured, None)


def test_the_off_table_shows_stock_off_and_their_ratio_against_the_baseline():
    """Its "fails past" is the tolerance of stock's step at the baseline's off/stock: 330 x 0.04 x 1.0, and
    200 x 0.04 x 0.96."""
    measured = P.cells([process("B16", 340, 860, stock=330), process("C16", 190, 340, stock=200)])
    baseline = {"tolerance": 0.04, "cells": {"B16": ratios(1.5), "C16": ratios(0.8, -0.04)}}
    lines = P.table(measured, baseline, "off/stock").splitlines()
    assert lines[0].startswith("| State | Running | stock µs | off µs | change µs | ratio |")
    assert lines[2].endswith("| 16 | 330 | 340 | +10 | +3.0% | +0.0% | +3.0% | +13 µs | ok |")
    assert lines[3].endswith("| 16 | 200 | 190 | -10 | -5.0% | -4.0% | -1.0% | +8 µs | ok |")
    both = P.tables(measured, None)
    assert "against stock vLLM (ratio: off/stock - 1):" in both and "baseline" not in both


def test_the_schedulers_take_turns_of_five_steps_in_a_rotating_order(monkeypatch):
    """measure() builds INSTANCES schedulers of each configuration, stock vLLM's of its own class and the others of
    the module's, and times each one's WARMUP + STEPS steps in turns, the order rotated each turn, keeping the steps
    after WARMUP; stand-in schedulers record the order."""
    seen, built = [], []
    monkeypatch.setattr(P, "load", lambda path, name: SimpleNamespace(Scheduler="make-room's"))
    monkeypatch.setattr(P, "suite", lambda: SimpleNamespace(StockScheduler="vLLM's"))

    def steady_state(cls, model, state, switches):
        built.append((cls, switches))
        return {"switches": switches}

    monkeypatch.setattr(P, "steady_state", steady_state)
    monkeypatch.setattr(P, "timed_step", lambda sched, state, k: seen.append((id(sched), k)) or k)
    times = P.measure(Path("m.py"), "model", P.BY_KEY["C16"], P.TIMED)
    timed = list(range(P.WARMUP, P.WARMUP + P.STEPS))
    assert times == {c: [timed, timed] for c in ["stock", "off", "on", "off2"]}
    on = P.CONFIGS["on"]
    kinds = [("vLLM's", None), ("make-room's", None), ("make-room's", on), ("make-room's", None)]
    assert built == kinds * P.INSTANCES
    order = [s for i, (s, _) in enumerate(seen) if i % P.TURN == 0]  # whose turn, turn by turn
    assert len(set(order[:8])) == 8  # a scheduler each
    assert order[8:16] == order[1:8] + order[:1]  # the next turn's order, rotated by one
    assert [k for s, k in seen if s == order[7]][:10] == list(range(10))


def test_the_method_records_what_a_process_times():
    """A check compares only with a baseline measured alike: the configurations, stock vLLM's class, how many
    schedulers a process times, a check's as a baseline's, and how a cost is taken."""
    assert P.TIMED == ["stock", "off", "on", "off2"] and P.CONFIGS["stock"] is None
    assert P.METHOD["schedulers"] == P.INSTANCES * len(P.TIMED) == 8
    assert P.METHOD["stock"] == "vllm.v1.core.sched.scheduler.Scheduler"
    assert P.METHOD["configs"] == P.CONFIGS


@pytest.fixture
def standins(monkeypatch) -> list:
    """Stand-ins for the timing, the machine and the build, so that command() runs without torch. Returns the runs
    asked for, as (their sets of rounds, their configurations)."""
    seen = []

    def run(module, model, states, configs, reps, runs="a"):
        seen.append((runs, configs))
        procs = [process(s.key, 100, 150, 100, name) for name in runs for s in states] * reps
        load = {"before": [1.0] * 3, "after": [1.0] * 3, "busy": []}
        loads = [{"run": name, "rep": r + 1} | load for r in range(reps) for name in runs]
        return {"processes": procs, "load": loads}

    here = {"host": "here", "cpu": "a CPU", "cores": 1, "os": "an OS"}
    here |= {"python": "3.12", "torch": "2.13", "vllm": "v0.32.0"}
    monkeypatch.setattr(P, "run", run)
    monkeypatch.setattr(P, "machine", lambda: here)
    monkeypatch.setattr(P, "importable", lambda modules: None)
    monkeypatch.setattr(P, "build_module", lambda out: out / "flatshare_make_room_scheduler.py")
    return seen


def command(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["perf_schedule.py", *argv])
    return P.command()


ONE = ("--states", "B16", "--rounds", "1")


def test_a_check_and_a_baseline_time_the_same_schedulers(standins, monkeypatch, tmp_path, capsys):
    """Both time stock, off, on and off2 in each process, so that a check's processes are its baseline's."""
    path = tmp_path / "baseline.json"
    assert command(monkeypatch, "baseline", str(path), *ONE, "--beside", "nothing") == 0
    assert command(monkeypatch, "check", str(path), *ONE) == 0
    assert standins == [("ab", P.TIMED), ("a", P.TIMED)]
    out = capsys.readouterr().out
    assert out.count("against stock vLLM (ratio: off/stock - 1):") == 2
    assert "ok: 1 state within it" in out


def test_the_record_names_the_module_it_timed_by_its_path(standins, monkeypatch, tmp_path):
    """Every generated module has the same file name, so the record keeps the whole path --module gave."""
    path, out = tmp_path / "baseline.json", tmp_path / "check.json"
    assert command(monkeypatch, "baseline", str(path), *ONE, "--beside", "nothing") == 0
    assert json.loads(path.read_text())["module"] == "make-room"
    module = tmp_path / "mutant" / "flatshare_make_room_scheduler.py"
    assert (
        command(monkeypatch, "check", str(path), *ONE, "--module", str(module), "--out", str(out))
        == 0
    )
    assert json.loads(out.read_text())["module"] == str(module.resolve())


@pytest.mark.parametrize(
    "scheduled, preempted, steady",
    [
        ({"run0": 1, "run1": 1}, set(), True),
        ({"run0": 1}, set(), False),  # a running request left
        ({"run0": 1, "head": 64}, set(), False),  # the head was admitted
        ({"run0": 1, "w0": 64}, {"run1"}, False),  # a preemption, the count unchanged
    ],
)
def test_a_step_that_admits_or_preempts_leaves_the_steady_state(
    monkeypatch, scheduled, preempted, steady
):
    out = SimpleNamespace(num_scheduled_tokens=scheduled, preempted_req_ids=preempted)
    sched = SimpleNamespace(schedule=lambda: out, update_from_output=lambda out, model_out: None)
    monkeypatch.setattr(P, "suite", lambda: SimpleNamespace(output=lambda sched, out: None))
    state = P.State("X", "X", "B", running=2)
    if steady:
        assert P.timed_step(sched, state, 0) >= 0
    else:
        with pytest.raises(P.StateChanged, match="the state changed at step 0"):
            P.timed_step(sched, state, 0)


def test_the_states_are_the_prs_tables_rows():
    assert [s.key for s in P.STATES] == [
        "A16", "A64", "A256", "A64-hit999", "A64-nohit",
        "B16", "B64", "B256", "B256-W10", "B64-hit999",
        "C16", "C64", "C256",
    ]  # fmt: skip
    assert P.CONFIGS["on"] == {
        "max_preemptions_per_step": 2,
        "budget_cap": True,
        "budget_cap_tokens": 48,
        "reserve_tokens": 0,
    }


def test_the_json_puts_a_key_a_line_and_a_short_list_on_one():
    record = {
        "cells": {"B64": {"ratio": 0.65, "ratios_a": [0.6, 0.7]}},
        "load": [{"busy": ["x 30%"]}],
    }
    text = P.dumps(record)
    assert json.loads(text) == record
    assert text.splitlines() == [
        "{",
        ' "cells": {',
        '  "B64": {',
        '   "ratio": 0.65,',
        '   "ratios_a": [0.6, 0.7]',
        "  }",
        " },",
        ' "load": [',
        "  {",
        '   "busy": ["x 30%"]',
        "  }",
        " ]",
        "}",
    ]


def script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, P.__file__, *args], capture_output=True, text=True)


def test_a_check_without_a_readable_baseline_of_this_method_exits_2(tmp_path):
    """Exit 2 is "could not measure", which a job tells from 1, "the change costs more"."""
    proc = script("check", str(tmp_path / "missing.json"))
    assert proc.returncode == 2, proc.stderr
    assert "cannot read the baseline" in proc.stderr and "Traceback" not in proc.stderr
    (tmp_path / "broken.json").write_text("{")
    assert script("check", str(tmp_path / "broken.json")).returncode == 2
    other = json.loads(BASELINE.read_text())
    other["method"] = other["method"] | {"steps": 30}
    (tmp_path / "other.json").write_text(json.dumps(other))
    proc = script("check", str(tmp_path / "other.json"))
    assert proc.returncode == 2 and "measured otherwise" in proc.stderr


def test_an_unknown_state_exits_2():
    proc = script("check", str(BASELINE), "--states", "A16,Z9")
    assert proc.returncode == 2 and "no state Z9" in proc.stderr


def test_a_python_without_torch_or_vllms_scheduler_is_told_why():
    assert P.importable(["json", "statistics"]) is None
    assert "no_such_module_flatshare" in P.importable(["json", "no_such_module_flatshare"])


def test_a_measuring_process_that_fails_means_it_could_not_measure(tmp_path):
    with pytest.raises(P.CannotMeasure, match="A16: its measuring process failed"):
        P.run(tmp_path / "no-such-module.py", str(tmp_path), [P.BY_KEY["A16"]], ["off"], 1)


def test_a_baseline_without_what_ran_beside_it_is_refused_before_anything_is_written(tmp_path):
    out = tmp_path / "baseline.json"
    for beside in ([], ["--beside", ""], ["--beside", "  "]):
        proc = script("baseline", str(out), *beside)
        assert proc.returncode == 2 and "needs --beside" in proc.stderr, proc.stderr
    assert not out.exists()


def test_the_control_checks_against_the_baseline_make_passes_it(tmp_path):
    code = "import test_perf_control as C; print(C.BASELINE)"
    env = os.environ | {"FLATSHARE_PERF_BASELINE": str(tmp_path / "mine.json")}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(P.__file__).parent,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == str(tmp_path / "mine.json"), proc.stderr
    env.pop("FLATSHARE_PERF_BASELINE")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(P.__file__).parent,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == str(BASELINE), proc.stderr


def test_another_machines_baseline_is_refused_unless_asked():
    there = {
        "host": "peacock",
        "cpu": "Xeon",
        "python": "3.12.15",
        "torch": "2.13.0+cpu",
        "os": "Linux 5.15",
    }
    baseline = {"machine": there}
    assert P.machine_check(there | {"os": "Linux 6.8", "cores": 8}, baseline, False) is None
    for key, value in (("cpu", "Apple M5 Max"), ("python", "3.12.12"), ("torch", "2.13.0")):
        with pytest.raises(
            P.CannotMeasure, match=f"{key}: .* there, {value} here.*measure one here"
        ):
            P.machine_check(there | {key: value}, baseline, False)
        warning = P.machine_check(there | {key: value}, baseline, True)
        assert warning.startswith("against another machine's baseline") and value in warning


def test_a_check_against_another_cpus_baseline_exits_2_before_timing(tmp_path):
    other = json.loads(BASELINE.read_text())
    other["machine"]["cpu"] = "a CPU no machine has"
    path = tmp_path / "other.json"
    path.write_text(json.dumps(other))
    proc = script("check", str(path), "--states", "A16", "--rounds", "1")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    if importlib.util.find_spec("torch") and importlib.util.find_spec("vllm"):
        assert "a CPU no machine has there" in proc.stderr and "measure one here" in proc.stderr


def test_the_checked_in_baseline_is_complete_and_its_numbers_agree():
    baseline = json.loads(BASELINE.read_text())
    assert {"host", "cpu", "cores", "os", "python", "torch", "vllm"} <= set(baseline["machine"])
    assert baseline["date"] and baseline["beside"] and baseline["tolerance_rule"]
    assert baseline["method"] == P.METHOD  # what a check compares against
    reps = baseline["reps"]
    assert [(load["run"], load["rep"]) for load in baseline["load"]] == [
        (run, rep) for rep in range(1, reps + 1) for run in "ab"
    ]
    assert all(len(load["before"]) == len(load["after"]) == 3 for load in baseline["load"])
    assert list(baseline["cells"]) == [s.key for s in P.STATES]
    for key, cell in baseline["cells"].items():
        for ratio in ("ratio", "off_ratio"):
            a, b = cell[f"{ratio}s_a"], cell[f"{ratio}s_b"]
            assert len(a) == len(b) == reps, key
            assert cell[f"{ratio}_a"] == round(statistics.median(a), 4), key
            assert cell[f"{ratio}_b"] == round(statistics.median(b), 4), key
            assert cell[ratio] == round(statistics.median(a + b), 4), key
        assert cell["aa"] == round(P.excess(cell["ratio_b"], cell["ratio_a"]), 4), key
        assert cell["off_aa"] == round(P.excess(cell["off_ratio_b"], cell["off_ratio_a"]), 4), key
        assert len(cell["noises"]) == 2 * reps, key
        assert cell["noise"] == round(statistics.median(cell["noises"]), 4), key
        costs = P.costs(cell["off_us"], cell["ratio"])
        assert {k: cell[k] for k in costs} == costs, key
        assert cell["stock_us"] == round(cell["off_us"] / (1 + cell["off_ratio"]), 1), key
        assert cell["off_change_us"] == round(cell["off_us"] - cell["stock_us"], 1), key
    # The tolerance is the one its rule gives from the noise it records.
    cells = baseline["cells"].values()
    noise, aa = [c["noise"] for c in cells], [c["aa"] for c in cells]
    off_aa = [c["off_aa"] for c in cells]
    assert baseline["tolerance"] == P.tolerance(noise, aa, off_aa)
    assert baseline["noise_max"] == max(noise, key=abs) and baseline["aa_max"] == max(aa, key=abs)
    assert baseline["off_aa_max"] == max(off_aa, key=abs)


# On vLLM's real scheduler: no timing, only what the states do.


@pytest.fixture(scope="module")
def module(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("vllm.v1.core.sched.scheduler")
    return P.load(P.build_module(tmp_path_factory.mktemp("make-room")), "flatshare_perf_make_room")


@pytest.fixture(scope="module")
def long_model(tmp_path_factory):
    from conftest import write_model

    path = tmp_path_factory.mktemp("model")
    write_model(path, max_position_embeddings=P.MAX_MODEL_LEN)
    return str(path)


def steps(module, model, key: str, config: str) -> list[int]:
    """WARMUP + STEPS steps of the state, as the measurement runs them (StateChanged if it is not steady), and in
    each the number of times make-room made room, or tried to: never, for stock vLLM's scheduler, which has none."""
    state = P.BY_KEY[key]
    sched = P.steady_state(P.scheduler_class(module, config), model, state, P.CONFIGS[config])
    calls = []
    make_room = getattr(sched, "_make_room", None)

    def counted(*args):
        calls[-1] += 1
        return make_room(*args)

    if make_room is not None:
        sched._make_room = counted
    for k in range(P.WARMUP + P.STEPS):
        calls.append(0)
        P.timed_step(sched, state, k)
    return calls


def test_stock_vllm_is_the_class_the_method_names(module):
    stock = P.scheduler_class(module, "stock")
    assert f"{stock.__module__}.{stock.__qualname__}" == P.METHOD["stock"]
    assert P.scheduler_class(module, "off") is P.scheduler_class(module, "on") is module.Scheduler


@pytest.mark.parametrize("key", list(P.BY_KEY))
def test_each_state_is_steady_and_runs_the_path_it_is_named_for(module, long_model, key):
    assert steps(module, long_model, key, "stock") == [0] * (P.WARMUP + P.STEPS)  # steady too
    on, off = steps(module, long_model, key, "on"), steps(module, long_model, key, "off")
    assert off == [0] * len(off)  # with the switches off, make-room never runs
    if key.startswith("A"):
        assert on == [0] * len(on)  # a priority-2 head: nothing to make room for
    else:
        # B: every step evaluates make-room for the priority-0 head and preempts nobody (the state would change);
        # C: every step finds that the head fits, with no slot for it.
        assert on == [1] * len(on)
