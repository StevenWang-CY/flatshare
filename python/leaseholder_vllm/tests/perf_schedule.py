"""make-room's cost per step: Scheduler.schedule()'s wall time on vLLM 0.32.0's real scheduler, on CPU, for stock vLLM
and for make-room with its switches off and in the configuration to deploy, and the regression check of those costs
against a measured baseline.

Stock vLLM is vLLM's own Scheduler, without the change. make-room's module is the one build.py generates at engine
start (``--module`` times another generated module), its switches off or in the configuration to deploy:
``max_preemptions_per_step 2``, ``budget_cap true``, ``budget_cap_tokens 48``, ``reserve_tokens 0``. They run in the
steady states the PRs' cost tables were measured in (#63350 and #63351), in which nothing is admitted or preempted for
the measured steps, so that every step does the same work:

  A  R running priority-2 decodes; the waiting head is a priority-2 request that does not fit the free KV (the
     saturated background tier), then W-1 small priority-2 requests behind it.
  B  as A, but the head is a priority-0 request too large for two victims: make-room evaluates every step and
     preempts nobody.
  C  every running slot taken (R = max_num_seqs); the head is a small priority-0 request that fits the KV but has no
     slot: make-room's fits path every step.

The head shares its first blocks with a running request's prompt (127 of them unless a state says otherwise), so its
prefix-cache lookup walks that many cached blocks.

A process times one state. It builds two schedulers of each configuration in that state, stock, off and on, and two more
of off (off2), then the eight take turns of 5 steps, 5 of warm-up and 40 timed each, so that whatever else runs on the
machine meanwhile slows them alike. A scheduler's cost is its 10th-percentile step (inclusive, as docs/GLOSSARY.md's
percentiles are): in a steady state every step does the same work, and the fastest are those that nothing else slowed
(nor a turn's first, cold after the others'). A configuration's cost in the process is its cheaper scheduler's: one
scheduler of a configuration may run a few percent slower than another through the whole process, as their places in
memory differ, and the pair makes that need both. A state has two ratios, each the median over its processes of each
one's own: off/stock - 1, what the change costs a step with its switches off as a fraction of stock vLLM's step, which
the PRs measured as main's within their tables' noise; and on/off - 1, what it costs in the configuration to deploy as a
fraction of the step with its switches off. Neither moves on a machine that is uniformly faster or slower. The state's
cost off in µs is the median over its processes; stock's is off's divided by off/stock, and on's off's times on/off. A
round times every state once, a process each, the states in a rotated order, and a check is 3 rounds.

What it can see: a state fails when the change's own cost a step grows by the tolerance of the step with the switches
on (the on/off table's "fails past", in µs), which is a small share of the change's cost where the change costs much
(the B states) and a large one where it costs little (the A states), or when its cost with the switches off grows by
the tolerance of stock's step (the off/stock table's): a cost common to off and on, which on/off cannot see, is one.
And what it cannot: with 40 timed steps the 10th-percentile step lies between the 4th and 5th fastest, so work added
to fewer than about 9 steps in 10 does not move it. Each cell records the on/off ratio by the median step as well, as
information the check does not read.

The check fails a state whose off/stock or on/off exceeds the baseline's by more than the baseline's tolerance, a
fraction of it (tolerance()), measured again over twice as many processes more and still exceeding it (confirm()). A
state whose ratio fell below the baseline's by more than the tolerance passes with a note (cheaper()): the
configuration on top of the ratio got cheaper or the one below it costlier, and either way the baseline no longer
describes the code. The baseline (benchmarks/perf/make-room-schedule.json) is two halves, A and B, of 3 rounds each,
their rounds interleaved: its noise is off against off within each process, and half B against half A in either
ratio, compared as the check compares itself with the baseline. A check's processes time the same eight schedulers,
its off2 giving its own noise, as information.

    python tests/perf_schedule.py check BASELINE [--out JSON] [--rounds 3] [--states A16,B64] [--module MODULE]
    python tests/perf_schedule.py baseline BASELINE [--rounds 3] [--beside TEXT]

Both need torch and vLLM 0.32.0's dependencies, with vLLM's source on PYTHONPATH (``make perf-make-room``). The exit
status is 0 when every state is within the tolerance, 1 when one costs more than it did, and 2 when it could not
measure: no torch or vLLM's scheduler, a baseline missing, unreadable or measured otherwise, or a measuring process
that failed or left its steady state.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "src/leaseholder_vllm/build.py"
WARMUP, STEPS = 5, 40
TURN = 5  # the steps a scheduler takes in a row
INSTANCES = 2  # schedulers of each configuration in a process; the cheaper one's cost counts
RUN_TOKENS = 2048  # each running request's prompt
MAX_MODEL_LEN = 32768
# The configurations, as make_room's keys: stock vLLM, vLLM's own Scheduler (STOCK_CLASS), without the change; make-room
# with its switches off; and make-room in the configuration to deploy (docs/PRODUCTION.md).
STOCK = "stock"
STOCK_CLASS = "vllm.v1.core.sched.scheduler.Scheduler"
CONFIGS: dict[str, dict | None] = {
    STOCK: None,
    "off": None,
    "on": {
        "max_preemptions_per_step": 2,
        "budget_cap": True,
        "budget_cap_tokens": 48,
        "reserve_tokens": 0,
    },
}
# Each process times the off configuration twice, the second time as "off2": the baseline's noise, and a check's, as
# information.
NOISE = "off2"
TIMED = [*CONFIGS, NOISE]
# A state's two ratios, by the key its cell holds each under: off/stock - 1 and on/off - 1.
RATIOS = {"off/stock": "off_ratio", "on/off": "ratio"}
# What a baseline must have been measured with for a check to compare against it.
METHOD = {
    "warmup_steps": WARMUP,
    "steps": STEPS,
    "turn_steps": TURN,
    "cost": "the 10th-percentile step, inclusive (statistics.quantiles(steps, n=10, method='inclusive')[0]), of the "
    "cheaper of a configuration's schedulers in a process",
    "instances": INSTANCES,
    "configs": CONFIGS,
    "stock": STOCK_CLASS,
    "schedulers": INSTANCES * len(TIMED),  # in each process, a check's and a baseline's alike
}


@dataclass(frozen=True)
class State:
    key: str
    label: str
    scenario: str
    running: int
    waiting: int = 1000
    hit_blocks: int = 127
    free: int = 0  # free blocks at the start; 0: the default sizing


# The rows of the PRs' cost tables, in their order.
STATES = (
    State("A16", "A: only priority 2 work, the head with a 127-block hit", "A", 16),
    State("A64", "A", "A", 64),
    State("A256", "A", "A", 256),
    State("A64-hit999", "A, the head with a 999-block hit (W = 100)", "A", 64, 100, 999, 500),
    State("A64-nohit", "A, the head with no hit (W = 100)", "A", 64, 100, 0),
    State("B16", "B: a priority 0 head that two victims cannot make room for", "B", 16),
    State("B64", "B", "B", 64),
    State("B256", "B", "B", 256),
    State("B256-W10", "B (W = 10)", "B", 256, 10),
    State("B64-hit999", "B, the head with a 999-block hit (W = 100)", "B", 64, 100, 999, 500),
    State("C16", "C: every slot taken, a priority 0 head that fits", "C", 16),
    State("C64", "C", "C", 64),
    State("C256", "C", "C", 256),
)
BY_KEY = {state.key: state for state in STATES}


class StateChanged(Exception):
    """The scheduler left the steady state during the measured steps."""


class CannotMeasure(Exception):
    """The check could not measure: the environment, the baseline or a measuring process, not the change. Exit 2."""


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# The scheduler, as the suites build it: tests/test_make_room.py's scheduler and requests, conftest.py's model.


def suite():
    """tests/test_make_room.py, which builds the suites' schedulers and requests; it needs torch and vLLM."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import test_make_room

    return test_make_room


def load(path: Path, name: str):
    """A generated scheduler module, from its file."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scheduler_class(module, config: str):
    """The class of a configuration's schedulers: stock vLLM's own Scheduler, as the suites import it, or
    ``module``'s."""
    return suite().StockScheduler if config == STOCK else module.Scheduler


def steady_state(cls, model: str, state: State, switches: dict | None):
    """A scheduler of the class ``cls`` with ``switches`` in ``state``: the running requests decoding, the head and the
    requests behind it waiting."""
    T = suite()
    block = T.BLOCK
    R = state.running
    head_tokens = 32000 if state.scenario in "AB" else 1024
    long_run = max(RUN_TOKENS, block * state.hit_blocks + block)
    run_blocks = (R - 1) * cdiv(RUN_TOKENS + 64, block) + cdiv(long_run + 64, block)
    growth = R * (WARMUP + STEPS + 2) // block + R + 64
    free = state.free or max(900 if state.scenario in "AB" else 400, growth)
    sched = T.scheduler(
        cls,
        model,
        run_blocks + free + 1,  # + vLLM's null block
        switches,
        max_num_batched_tokens=8192,
        max_model_len=MAX_MODEL_LEN,
        prefix_caching=True,
        max_num_seqs=R if state.scenario == "C" else 1024,
    )
    # R running priority-2 requests with unique prompts; the first one's is long enough to hold the head's shared
    # prefix.
    tok = 1000
    running = []
    for i in range(R):
        n = long_run if i == 0 else RUN_TOKENS
        prompt = list(range(tok, tok + n))
        running.append(T.request(f"run{i}", 0, 2, float(i), max_tokens=100_000, prompt=prompt))
        tok += n
    for req in running:
        sched.add_request(req)
    for _ in range(4000):
        if all(
            req.status == T.RequestStatus.RUNNING and not req.is_prefill_chunk for req in running
        ):
            break
        T.step(sched)
    else:
        raise StateChanged("the running requests never all decoded")
    shared = running[0].prompt_token_ids[: block * state.hit_blocks]
    suffix = list(range(tok, tok + head_tokens - len(shared)))
    tok += len(suffix)
    priority = 2 if state.scenario == "A" else 0
    sched.add_request(T.request("head", 0, priority, 1e6, max_tokens=64, prompt=shared + suffix))
    for j in range(state.waiting - 1):
        prompt = list(range(tok, tok + 64))
        sched.add_request(T.request(f"w{j}", 0, 2, 1e6 + 1 + j, max_tokens=64, prompt=prompt))
        tok += 64
    return sched


def timed_step(sched, state: State, k: int) -> float:
    """schedule()'s wall time in µs for one step, its output fed back as a synchronous engine does; StateChanged if
    the step schedules other than the running requests or preempts."""
    t0 = time.perf_counter_ns()
    out = sched.schedule()
    t1 = time.perf_counter_ns()
    sched.update_from_output(out, suite().output(sched, out))
    if (
        len(out.num_scheduled_tokens) != state.running
        or "head" in out.num_scheduled_tokens
        or out.preempted_req_ids
    ):
        raise StateChanged(
            f"{state.key}: the state changed at step {k}: "
            f"{len(out.num_scheduled_tokens)} scheduled, {len(out.preempted_req_ids or ())} preempted"
        )
    return (t1 - t0) / 1e3


def measure(
    module_path: Path, model: str, state: State, configs: list[str]
) -> dict[str, list[list[float]]]:
    """One process's measurement: INSTANCES schedulers of their own for each configuration in ``configs``, all in
    ``state``, then WARMUP and STEPS steps of each, the schedulers taking turns of TURN steps in an order rotated each
    turn. Returns each configuration's timed steps in µs, scheduler by scheduler."""
    module = load(module_path, module_path.stem)
    order = [(c, i) for i in range(INSTANCES) for c in configs]
    scheds = {}
    for c, i in order:
        config = "off" if c == NOISE else c
        scheds[c, i] = steady_state(scheduler_class(module, config), model, state, CONFIGS[config])
    gc.collect()
    gc.freeze()  # the states' objects: the collector need not walk them while steps are timed
    times: dict[tuple[str, int], list[float]] = {key: [] for key in order}
    for turn, first in enumerate(range(0, WARMUP + STEPS, TURN)):
        for key in rotated(order, turn):
            for k in range(first, min(first + TURN, WARMUP + STEPS)):
                t = timed_step(scheds[key], state, k)
                if k >= WARMUP:
                    times[key].append(t)
    gc.unfreeze()
    return {c: [times[c, i] for i in range(INSTANCES)] for c in configs}


# The rounds: every state once in each, a process each, the states and configurations rotated.


def rotated(items: list, k: int) -> list:
    k %= len(items)
    return items[k:] + items[:k]


def busy() -> list[str]:
    """What else uses the CPU: the commands whose processes use 25% of a CPU or more between them, as ps reports it
    (on Linux, each process's use averaged over its life), this process left out."""
    proc = subprocess.run(["ps", "-eo", "pid=,pcpu=,comm="], capture_output=True, text=True)
    use: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and int(fields[0]) != os.getpid():
            name = Path(fields[2].strip()).name
            use[name] = use.get(name, 0.0) + float(fields[1])
    top = sorted(use.items(), key=lambda item: -item[1])
    return [f"{name} {cpu:.0f}%" for name, cpu in top if cpu >= 25]


def run(
    module_path: Path,
    model: str,
    states: list[State],
    configs: list[str],
    reps: int,
    runs: str = "a",
) -> dict:
    """``reps`` rounds of each set of rounds in ``runs``, named by a letter (a check's ``a``, a baseline's halves
    ``ab``, the states measured again ``c``), their rounds interleaved, every state in ``states`` in each, a process
    each, timing ``configs``. Returns each process's timed steps by configuration, and for each round the load average
    before and after it and what else used the CPU as it started."""
    processes, load = [], []
    # The scheduler is Python; torch's thread pools would only take CPUs from what else runs.
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for rep in range(reps):
        for r, name in enumerate(runs):
            k = rep * len(runs) + r
            before, others = os.getloadavg(), busy()
            for j, state in rotated(list(enumerate(states)), k):
                order = rotated(configs, k + j)
                cmd = [sys.executable, str(Path(__file__).resolve()), "measure", str(module_path)]
                cmd += [model, state.key, *order]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if proc.returncode:
                    raise CannotMeasure(
                        f"{state.key}: its measuring process failed:\n{proc.stderr[-2000:]}"
                    )
                times = json.loads(proc.stdout)
                processes.append(
                    {
                        "run": name,
                        "rep": rep + 1,
                        "state": state.key,
                        "order": order,
                        "times_us": times,
                    }
                )
                where = f"{rep + 1}{name if len(runs) > 1 else ''}/{reps}"
                costs = "  ".join(f"{c} {cheapest(times[c]):,.0f}" for c in configs)
                print(f"  {where} {state.key:<11} {costs}", file=sys.stderr, flush=True)
            after = os.getloadavg()
            load.append(
                {
                    "run": name,
                    "rep": rep + 1,
                    "before": [round(x, 2) for x in before],
                    "after": [round(x, 2) for x in after],
                    "busy": others,
                }
            )
    return {"processes": processes, "load": load}


# Costs, ratios, the tolerance and the check: arithmetic on the timed steps, testable without torch.


def cost(times: list[float]) -> float:
    """A scheduler's cost: the 10th percentile of its timed steps, inclusive, as docs/GLOSSARY.md's percentiles are."""
    return statistics.quantiles(times, n=10, method="inclusive")[0]


def cheapest(schedulers: list[list[float]]) -> float:
    """A configuration's cost in a process: its cheaper scheduler's."""
    return min(map(cost, schedulers))


def cells(processes: list[dict]) -> dict[str, dict]:
    """Each state's cell from its processes, in STATES' order: its two ratios, each the median over the processes of
    each one's own, which its turns pair: on/off - 1 (``ratio``) and off/stock - 1 (``off_ratio``); the cost off, the
    median over the processes, in µs, and stock's and on's from it by the ratios, so that off = stock x (1 +
    off_ratio) and on = off x (1 + ratio), and the change's in µs with the switches off and on; where the off
    configuration was timed twice, the noise, the median of off2/off - 1; and, as information the check does not
    read, on/off by the median step, which work on fewer than about 9 steps in 10 moves and the 10th-percentile step
    does not."""
    by_state: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = {}
    for proc in processes:
        p10 = {c: cheapest(t) for c, t in proc["times_us"].items()}
        median = {c: min(map(statistics.median, t)) for c, t in proc["times_us"].items()}
        by_state.setdefault(proc["state"], []).append((p10, median))
    out = {}
    for key in (s.key for s in STATES if s.key in by_state):
        procs = [p10 for p10, _ in by_state[key]]
        ratios = [round(p["on"] / p["off"] - 1, 4) for p in procs]
        off_ratios = [round(p["off"] / p[STOCK] - 1, 4) for p in procs]
        off_us = round(statistics.median(p["off"] for p in procs), 1)
        off_ratio = round(statistics.median(off_ratios), 4)
        cell: dict = {"stock_us": round(off_us / (1 + off_ratio), 1), "off_us": off_us}
        cell |= costs(off_us, round(statistics.median(ratios), 4))
        cell["ratios"] = ratios
        cell["median_ratio"] = round(
            statistics.median(m["on"] / m["off"] - 1 for _, m in by_state[key]), 4
        )
        cell["off_change_us"] = round(off_us - cell["stock_us"], 1)
        cell |= {"off_ratio": off_ratio, "off_ratios": off_ratios}
        if NOISE in procs[0]:
            cell["noises"] = [round(p[NOISE] / p["off"] - 1, 4) for p in procs]
            cell["noise"] = round(statistics.median(cell["noises"]), 4)
        out[key] = cell
    return out


def costs(off_us: float, ratio: float) -> dict[str, float]:
    """A state's costs from its cost off and its ratio: on = off x (1 + ratio), the change on - off."""
    change = round(off_us * ratio, 1)
    return {"on_us": round(off_us + change, 1), "change_us": change, "ratio": ratio}


def excess(ratio: float, base: float) -> float:
    """How far a state's on/off exceeds the baseline's, as a fraction of the baseline's."""
    return (1 + ratio) / (1 + base) - 1


def baseline_cells(processes: list[dict]) -> dict[str, dict]:
    """The baseline's cells: over both halves' processes, with each half's ratios and half B's excess over half A, in
    on/off (``aa``) and in off/stock (``off_aa``)."""
    out = cells(processes)
    by_half = {name: cells([p for p in processes if p["run"] == name]) for name in "ab"}
    for key, cell in out.items():
        a, b = by_half["a"][key], by_half["b"][key]
        del cell["ratios"], cell["off_ratios"]
        cell |= {"ratios_a": a["ratios"], "ratios_b": b["ratios"]}
        cell |= {"ratio_a": a["ratio"], "ratio_b": b["ratio"]}
        cell["aa"] = round(excess(b["ratio"], a["ratio"]), 4)
        cell |= {"off_ratios_a": a["off_ratios"], "off_ratios_b": b["off_ratios"]}
        cell |= {"off_ratio_a": a["off_ratio"], "off_ratio_b": b["off_ratio"]}
        cell["off_aa"] = round(excess(b["off_ratio"], a["off_ratio"]), 4)
    return out


def tolerance(noise: list[float], aa: list[float], off_aa: list[float]) -> float:
    """The check's tolerance: twice the largest factor by which the baseline's two measurements of something
    unchanged differed in any state, as a fraction rounded up to a whole percent. Those are off against off in each
    process (``noise``: the median over the processes of off2/off - 1) and its half B against its half A in on/off
    (``aa``) and in off/stock (``off_aa``), each the excess as the check compares a ratio with the baseline's, and a
    check run and the baseline may each be off by that much."""
    worst = max(abs(math.log1p(x)) for x in [*noise, *aa, *off_aa])
    return math.ceil(round(100 * math.expm1(2 * worst), 6)) / 100


def against(key: str, what: str, ratio: float, base: float) -> str:
    """A state's ratio ``what`` (one of RATIOS) against the baseline's, and the excess: off/stock to three decimals,
    which moves little, and on/off to two."""
    digits = 3 if what == "off/stock" else 2
    return (
        f"{key}: {what} {1 + ratio:.{digits}f} against the baseline's {1 + base:.{digits}f}, "
        f"{excess(ratio, base):+.1%}"
    )


def check(measured: dict[str, dict], baseline: dict) -> list[str]:
    """The states in ``measured`` whose off/stock or on/off exceeds the baseline's by more than its tolerance, and by
    how much; a state the baseline lacks fails too."""
    failures = []
    for key, cell in measured.items():
        base = baseline["cells"].get(key)
        if base is None:
            failures.append(f"{key}: not in the baseline")
            continue
        for what, field in RATIOS.items():
            if excess(cell[field], base[field]) > baseline["tolerance"]:
                failures.append(
                    f"{against(key, what, cell[field], base[field])} "
                    f"(tolerance {baseline['tolerance']:.0%})"
                )
    return failures


def cheaper(measured: dict[str, dict], baseline: dict) -> list[str]:
    """The states in ``measured`` whose off/stock or on/off fell below the baseline's by more than its tolerance. They
    pass: the configuration on top of the ratio got cheaper or the one below it costlier, and either way the baseline
    no longer describes the code."""
    notes = []
    for key, cell in measured.items():
        base = baseline["cells"].get(key)
        if base is None:
            continue
        for what, field in RATIOS.items():
            if excess(cell[field], base[field]) < -baseline["tolerance"]:
                notes.append(against(key, what, cell[field], base[field]))
    return notes


def load_warning(load: list[dict], baseline_load: list[dict]) -> str | None:
    """A warning when the 1-minute load during a run went past the highest it reached as the baseline was measured by
    half of that or more, so that a state failing under such load reads as possibly the load's; None otherwise."""
    here = max(max(rep["before"][0], rep["after"][0]) for rep in load)
    there = max(max(rep["before"][0], rep["after"][0]) for rep in baseline_load)
    if here <= 1.5 * there:
        return None
    return (
        f"the 1-minute load reached {here:.1f} during this run, against at most {there:.1f} as the baseline was "
        "measured: a state that fails under such load may fail for it"
    )


def confirm(result: dict, baseline: dict, again) -> tuple[dict[str, dict], list[str], list[str]]:
    """The check of ``result``'s processes against ``baseline``, a state that fails measured again first: ``again``
    runs the failing states for twice as many processes more, and a state fails only if its ratio over all of its
    processes still exceeds the tolerance. Two of an unchanged module's three processes can stray the same way, and
    rarely do; a costlier module fails again. Returns the cells, the failures and the states measured again."""
    measured = cells(result["processes"])
    failing = {
        f.split(":")[0] for f in check(measured, baseline) if not f.endswith("not in the baseline")
    }
    suspects = [key for key in measured if key in failing]
    if suspects:
        more = again([BY_KEY[key] for key in suspects])
        result["processes"] += more["processes"]
        result["load"] += more["load"]
        measured = cells(result["processes"])
    return measured, check(measured, baseline), suspects


# How each of RATIOS reads in its table: the cell's keys of its two costs in µs, the one below the ratio and the one on
# top, of the change between them, and how the ratio is shown.
TABLES = {
    "off/stock": ("stock_us", "off_us", "off_change_us", "{:+.1%}"),
    "on/off": ("off_us", "on_us", "change_us", "{:+.2f}"),
}


def table(measured: dict[str, dict], baseline: dict | None, what: str = "on/off") -> str:
    """The cells' ratio ``what`` (one of RATIOS) as a Markdown table, in the PRs' rows' order: the costs of the
    configurations below and on top of it in µs, the change between them and the ratio. Against a baseline, "fails
    past" is how many µs a step more the configuration on top may cost before the state fails: the tolerance of its
    step at the baseline's ratio."""
    below, top, change, form = TABLES[what]
    field = RATIOS[what]
    head = f"| State | Running | {below[:-3]} µs | {top[:-3]} µs | change µs | ratio |"
    rule = "|---|---:|---:|---:|---:|---:|"
    if baseline:
        head += " baseline | excess | fails past | |"
        rule += "---:|---:|---:|---|"
    lines = [head, rule]
    for state in STATES:
        cell = measured.get(state.key)
        if cell is None:
            continue
        line = (
            f"| {state.label} | {state.running} | {cell[below]:,.0f} | {cell[top]:,.0f} "
            f"| {cell[change]:+,.0f} | {form.format(cell[field])} |"
        )
        if baseline:
            base = baseline["cells"].get(state.key)
            if base is None:
                line += " — | — | not in the baseline |"
            else:
                over = excess(cell[field], base[field])
                verdict = "ok"
                if abs(over) > baseline["tolerance"]:
                    verdict = "costlier" if over > 0 else "cheaper"
                # The configuration on top may cost this much more a step before the state fails.
                room = cell[below] * baseline["tolerance"] * (1 + base[field])
                line += f" {form.format(base[field])} | {over:+.1%} | +{room:,.0f} µs | {verdict} |"
        lines.append(line)
    return "\n".join(lines)


def tables(measured: dict[str, dict], baseline: dict | None) -> str:
    """Both ratios' tables, each under what it compares."""
    return (
        "make-room with its switches off against stock vLLM (ratio: off/stock - 1):\n"
        f"{table(measured, baseline, 'off/stock')}\n\n"
        "make-room in the configuration to deploy against its switches off (ratio: on/off - 1):\n"
        f"{table(measured, baseline, 'on/off')}"
    )


def dumps(value, indent: int = 0) -> str:
    """JSON with one key of an object a line and a list of numbers or strings on one line."""
    pad = " " * (indent + 1)
    if isinstance(value, dict) and value:
        items = [f"{pad}{json.dumps(k)}: {dumps(v, indent + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if isinstance(value, list) and any(isinstance(v, dict | list) for v in value):
        items = [pad + dumps(v, indent + 1) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(value, ensure_ascii=False)


# The machine and the environment, for the record.


def machine() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        if sys.platform == "darwin":
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            cpu = proc.stdout.strip() or cpu
    import torch
    import vllm

    src = Path(vllm.__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "-C", str(src), "describe", "--tags", "--always"], capture_output=True, text=True
    )
    return {
        "host": platform.node().split(".")[0],
        "cpu": cpu,
        "cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "vllm": proc.stdout.strip() or str(src),
    }


def importable(modules: list[str]) -> str | None:
    """None if this Python imports ``modules``, or why not."""
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as error:
            return f"{name}: {error}"
    return None


def read_baseline(path: Path) -> dict:
    """The baseline at ``path``, measured as this check measures; CannotMeasure if it is not one."""
    try:
        baseline = json.loads(path.read_text())
        missing = [
            k for k in ("date", "machine", "method", "tolerance", "cells") if k not in baseline
        ]
    except (OSError, ValueError, TypeError) as error:
        raise CannotMeasure(f"cannot read the baseline {path}: {error}") from error
    if missing:
        raise CannotMeasure(f"cannot read the baseline {path}: it has no {', '.join(missing)}")
    if baseline["method"] != METHOD:
        raise CannotMeasure(
            f"{path} was measured otherwise than this check measures: measure a new baseline "
            "(make perf-make-room UPDATE=1)"
        )
    return baseline


def machine_check(here: dict, baseline: dict, any_machine: bool) -> str | None:
    """Whether a check here may compare with ``baseline``: the ratios depend on the CPU, Python and torch, so another's
    raises CannotMeasure, or with ``any_machine`` returns a warning to print. None when they agree."""
    there = baseline["machine"]
    keys = [k for k in ("cpu", "python", "torch") if here.get(k) != there.get(k)]
    if not keys:
        return None
    diffs = "; ".join(f"{k}: {there.get(k)} there, {here.get(k)} here" for k in keys)
    what = f"the baseline is {there.get('host')}'s and this machine differs ({diffs})"
    if any_machine:
        return f"against another machine's baseline: {what}; ratios are CPU-specific, so read it with care"
    raise CannotMeasure(
        f"{what}: ratios are CPU-specific, so measure one here (make perf-make-room UPDATE=1 "
        "PERF_BASELINE=build/perf-baseline.json PERF_BESIDE=...) and check against it, or pass --any-machine"
    )


def build_module(out: Path) -> Path:
    """make-room's module, as build.py writes it at engine start, from the vLLM source on PYTHONPATH."""
    build = load(BUILD, "lh_perf_build")
    return build.build("make-room", out, source_tree=True)


def count_states(lines) -> str:
    """How many states ``lines`` name, each line led by its state's key, or the keys themselves: "1 state", "2
    states"."""
    n = len({line.split(":")[0] for line in lines})
    return f"{n} state" if n == 1 else f"{n} states"


def main() -> None:
    try:
        sys.exit(command())
    except CannotMeasure as error:
        print(f"perf_schedule.py: {error}", file=sys.stderr)
        sys.exit(2)


def command() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "baseline"):
        p = sub.add_parser(name)
        p.add_argument("baseline", type=Path)
        p.add_argument("--rounds", type=int, default=3, help="rounds: every state once in each")
        p.add_argument("--states", default=",".join(BY_KEY), help="comma-separated states")
        p.add_argument("--module", type=Path, help="a generated module in place of make-room's")
        p.add_argument("--raw", type=Path, help="every process's timed steps, as JSON")
    sub.choices["check"].add_argument("--out", type=Path, help="the run's JSON")
    sub.choices["check"].add_argument(
        "--any-machine", action="store_true", help="check against another machine's baseline"
    )
    sub.choices["baseline"].add_argument("--beside", default="", help="what else ran meanwhile")
    m = sub.add_parser("measure")  # one process's measurement, run by run()
    m.add_argument("module", type=Path)
    m.add_argument("model")
    m.add_argument("state", choices=list(BY_KEY))
    m.add_argument("configs", nargs="+", choices=TIMED)
    args = parser.parse_args()

    if args.command == "measure":
        times = measure(args.module, args.model, BY_KEY[args.state], args.configs)
        print(json.dumps({c: [[round(t, 2) for t in s] for s in v] for c, v in times.items()}))
        return 0

    unknown = [key for key in args.states.split(",") if key not in BY_KEY]
    if unknown:
        raise CannotMeasure(f"no state {', '.join(unknown)}; the states are {', '.join(BY_KEY)}")
    states = [BY_KEY[key] for key in args.states.split(",")]
    if args.command == "baseline" and not args.beside.strip():
        raise CannotMeasure(
            "a baseline needs --beside: what else ran on the machine as it was measured"
        )
    baseline = read_baseline(args.baseline) if args.command == "check" else None
    why = importable(["pytest", "torch", "vllm.v1.core.sched.scheduler"])
    if why:
        raise CannotMeasure(f"this Python cannot import what the scheduler needs ({why})")
    here = machine()
    warning = machine_check(here, baseline, args.any_machine) if baseline else None
    if warning:
        print(f"perf_schedule.py: {warning}", file=sys.stderr)
    from conftest import write_model

    runs = "ab" if args.command == "baseline" else "a"
    with tempfile.TemporaryDirectory() as tmp:
        module = args.module or build_module(Path(tmp))
        model = Path(tmp) / "model"
        model.mkdir()
        write_model(model, max_position_embeddings=MAX_MODEL_LEN)
        started = datetime.now(UTC)
        result = run(module, str(model), states, TIMED, args.rounds, runs)
        if baseline is not None:

            def again(failing: list[State]) -> dict:
                keys = ", ".join(s.key for s in failing)
                print(
                    f"  measuring {keys} again, {2 * args.rounds} processes each", file=sys.stderr
                )
                return run(module, str(model), failing, TIMED, 2 * args.rounds, "c")

            measured, failures, suspects = confirm(result, baseline, again)
    if args.raw:
        args.raw.write_text(json.dumps(result) + "\n")
    record = {
        "machine": here,
        "date": started.strftime("%Y-%m-%dT%H:%MZ"),
        "module": "make-room" if args.module is None else str(args.module.resolve()),
        "method": METHOD,
        "reps": args.rounds,
        "load": result["load"],
    }

    if args.command == "baseline":
        measured = baseline_cells(result["processes"])
        noise = [cell["noise"] for cell in measured.values()]
        aa = [cell["aa"] for cell in measured.values()]
        off_aa = [cell["off_aa"] for cell in measured.values()]
        record = {
            "what": "the cost per step of stock vLLM's scheduler and of make-room's with its switches off and in the "
            "configuration to deploy, Scheduler.schedule()'s wall time on vLLM 0.32.0's real scheduler on CPU, "
            "written by python/leaseholder_vllm/tests/perf_schedule.py baseline",
            **record,
            "beside": args.beside,
            "tolerance": tolerance(noise, aa, off_aa),
            "tolerance_rule": " ".join(tolerance.__doc__.split()),
            "noise_max": max(noise, key=abs),
            "aa_max": max(aa, key=abs),
            "off_aa_max": max(off_aa, key=abs),
            "cells": measured,
        }
        print(tables(measured, None))
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(dumps(record) + "\n")
        print(
            f"\nbaseline written to {args.baseline}: the worst off-against-off noise "
            f"{record['noise_max']:+.1%}, the worst half B against half A {record['aa_max']:+.1%} in on/off "
            f"and {record['off_aa_max']:+.1%} in off/stock, so a tolerance of {record['tolerance']:.0%}"
        )
        return 0

    record |= {
        "cells": measured,
        "measured_again": suspects,
        "baseline": {
            "date": baseline["date"],
            "host": baseline["machine"]["host"],
            "tolerance": baseline["tolerance"],
        },
        "failures": failures,
        "cheaper": cheaper(measured, baseline),
    } | ({"machine_differs": warning} if warning else {})
    busy = load_warning(result["load"], baseline["load"])
    if busy:
        record["load_warning"] = busy
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(dumps(record) + "\n")
    print(tables(measured, baseline))
    print(
        f"\nagainst the baseline of {baseline['machine']['host']}, {baseline['date']} "
        f"(tolerance {baseline['tolerance']:.0%}):"
    )
    if busy:
        print(f"warning: {busy}")
    if failures:
        print(f"FAIL: the change costs more than it did in {count_states(failures)}:")
        print("\n".join(f"  {failure}" for failure in failures))
    else:
        print(f"ok: {count_states(measured)} within it")
    if record["cheaper"]:
        print(
            f"note: {count_states(record['cheaper'])} cheaper than the baseline by more than the tolerance, which no "
            "longer describes the code: once the change is meant, measure a new baseline"
        )
        print("\n".join(f"  {note}" for note in record["cheaper"]))
    return 1 if failures else 0


if __name__ == "__main__":
    main()
