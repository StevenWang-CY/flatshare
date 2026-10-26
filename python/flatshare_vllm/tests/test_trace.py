"""make-room against the PRs, step for step, on vLLM 0.32.0's real scheduler, on CPU: what E2 measured is what #63350
and #63351 propose.

The PRs' module is v0.32.0's scheduler with the PRs' two patches alone (src/flatshare_vllm/make-room), their three
SchedulerConfig fields read from the variant's ``make_room`` keys; the variant's has its own patch on top. Both run the
same request traces, and every step's scheduled tokens by request, preempted, new, cached and resumed requests and
free blocks must be equal, in each switch set E2 ran (the reserve, which the PRs do not have, off), synchronous and
asynchronous, at watermarks 0 and 0.07. Four workloads: the behavior suite's, KV pressure (make-room preempting under
the budget cap, past two victims, with shared prefixes), random ones with shared prefixes, and random ones with a KV
connector loading asynchronously, answering "not yet" and holding frees (the untried rule). One-line mutants of the
PRs' module are the negative control: each must make a trace diverge.

``TRACE_SEEDS`` sets the random workloads' seeds (20; RV2's identity check ran 100). ``TRACE_REFERENCE``, a generated
module's path, compares the variant with that module instead, the reserve and the variant's counters included: for
example main's module against a branch's. A module from before #63351 bounded a step's held tokens by its budget (the
old build.py's, which W5's records ran) counted each held chunk against the step's whole budget, so with two chunks
held in a step it counts more held tokens, never fewer: against such a module, the variant's held tokens need only be
at most its. The scheduler needs torch and vLLM 0.32.0's dependencies, so these run as tests/test_make_room.py does:
``make test-trace TORCH_PYTHON=...``.
"""

import importlib.util
import os
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.core.sched.scheduler")

import test_make_room as T
from vllm.v1.core.sched.async_scheduler import AsyncScheduler as StockAsyncScheduler
from vllm.v1.request import RequestStatus

PACKAGE = Path(__file__).resolve().parents[1] / "src/flatshare_vllm"
STEPS = 240
SEEDS = int(os.environ.get("TRACE_SEEDS", "20"))
REFERENCE = os.environ.get("TRACE_REFERENCE")
B = T.BLOCK

# The PRs' SchedulerConfig fields, the variant's make_room keys and the PRs' defaults.
FIELDS = (
    ("max_priority_preemptions_per_step", "max_preemptions_per_step", "0"),
    ("priority_prefill_budget_cap", "budget_cap", "False"),
    ("priority_prefill_budget_cap_tokens", "budget_cap_tokens", "16"),
)

# Deliberate changes of behavior in the PRs' module, the negative control.
MUTANTS = {
    # Victims in arrival order instead of the latest first.
    "victim-order": (
        "key=lambda req: (req.priority, req.arrival_time),\n            reverse=True,\n",
        "key=lambda req: (req.priority, -req.arrival_time),\n            reverse=True,\n",
    ),
    # The room ignores the waiting request's prefix-cache hit.
    "ignore-hit": (
        "hit_blocks = set() if hit is None else {id(b) for g in hit[0] for b in g}\n",
        "hit_blocks = set()\n",
    ),
    # Room is made again for a request the waiting loop has not tried since.
    "no-untried": ("and make_room_req not in self._make_room_untried\n", "and True\n"),
    # Room is made while the budget cap holds the request back.
    "no-cap-gate": (
        "and (hold_priority is None or make_room_req.priority <= hold_priority)\n",
        "and True\n",
    ),
    # With deferred frees, the decodes' growth is counted for one step only.
    "one-step-growth": ("num_steps = 2 if self.defer_block_free else 1\n", "num_steps = 1\n"),
}

SWITCHES = {
    # E2's PR configurations (disc-pr-a, disc-pr-ab, disc-pr-ab48), with the reserve off, as the PRs have none.
    "A2": {"max_preemptions_per_step": 2, "budget_cap": False},
    "AB16": {"max_preemptions_per_step": 2, "budget_cap": True, "budget_cap_tokens": 16},
    "AB48": {"max_preemptions_per_step": 2, "budget_cap": True, "budget_cap_tokens": 48},
    # More preemptions a step, and the budget cap alone.
    "A4": {"max_preemptions_per_step": 4, "budget_cap": False},
    "B16": {"max_preemptions_per_step": 0, "budget_cap": True, "budget_cap_tokens": 16},
    "B48": {"max_preemptions_per_step": 0, "budget_cap": True, "budget_cap_tokens": 48},
}
if REFERENCE:
    SWITCHES |= {
        # Arm 9's shape at the suite's pools, and the reserve with each part.
        "AB16R": {"max_preemptions_per_step": 2, "budget_cap": True, "reserve_tokens": 4 * B},
        "AB48R": {
            "max_preemptions_per_step": 2,
            "budget_cap": True,
            "budget_cap_tokens": 48,
            "reserve_tokens": 8 * B,
        },
        "A2R": {"max_preemptions_per_step": 2, "reserve_tokens": 4 * B},
        "B16R": {"budget_cap": True, "reserve_tokens": 4 * B},
        "R": {"reserve_tokens": 8 * B},
    }
# The counters the variant logs, each as (its attribute, the attribute in a module build.py wrote before the PRs
# had counters), the first three the PRs'.
COUNTERS = (
    ("_num_priority_preemptions", "make_room_preemptions"),
    ("_num_prefill_held_steps", "make_room_held_steps"),
    ("_num_prefill_held_tokens", "make_room_held_tokens"),
    ("make_room_held_admissions", "make_room_held_admissions"),
)
HELD_TOKENS = 2
# Whether the reference bounds a step's held tokens by the step's budget, as #63351 does.
REFERENCE_BOUNDS_HELD = (
    bool(REFERENCE) and "_prefill_held_step_tokens" in Path(REFERENCE).read_text()
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prs_source(build) -> str:
    """The PRs' module: their two patches, their SchedulerConfig fields read from additional_config["make_room"]."""
    sched = build.sched_dir(source_tree=True)
    patches = [PACKAGE / "make-room" / name for name in ("63350.patch", "63351.patch")]
    source = build.module_source(sched, patches)
    for field, key, default in FIELDS:
        old = f"self.scheduler_config.{field}\n"
        assert source.count(old) == 1, field
        new = f'vllm_config.additional_config.get("make_room", {{}}).get("{key}", {default})\n'
        source = source.replace(old, new)
    return source


@pytest.fixture(scope="session")
def build():
    return load(PACKAGE / "build.py", "flatshare_trace_build")


@pytest.fixture(scope="session")
def variant(tmp_path_factory, build):
    """The variant, as its engine unit builds it."""
    out = tmp_path_factory.mktemp("variant")
    return load(build.build("make-room", out, source_tree=True), "flatshare_trace_variant")


@pytest.fixture(scope="session")
def other(tmp_path_factory, build):
    """What the variant is compared with: the PRs' module, or TRACE_REFERENCE."""
    if REFERENCE:
        return load(Path(REFERENCE), "flatshare_trace_reference")
    path = tmp_path_factory.mktemp("prs") / "flatshare_trace_prs.py"
    path.write_text(prs_source(build))
    return load(path, "flatshare_trace_prs")


def record(sched, out) -> tuple:
    cached = out.scheduled_cached_reqs
    return (
        dict(out.num_scheduled_tokens),
        sorted(out.preempted_req_ids or ()),
        sorted(req.req_id for req in out.scheduled_new_reqs),
        list(cached.req_ids),
        sorted(cached.resumed_req_ids),
        T.free_blocks(sched),
    )


def pressure_workload() -> list[tuple[int, T.Request]]:
    """(step, request): background decodes (priority 2) fill the pool and keep growing; interactive requests
    (priority 1) and voice turns (priority 0) arrive beside them, some needing more than two victims' blocks; two
    turns share four blocks of prompt with a running background request."""
    out = []
    for i in range(6):
        out.append(
            (i, T.request(f"bg{i}", 0, 2, float(i), max_tokens=280, prompt=[100 + i] * (6 * B)))
        )
    for i, step in enumerate((60, 75, 90, 105)):
        prompt = [110 + i] * (5 * B)
        out.append((step, T.request(f"bg{6 + i}", 0, 2, 6.0 + i, max_tokens=200, prompt=prompt)))
    for i, step in enumerate((30, 70, 110, 150)):
        prompt = [120 + i] * (4 * B)
        out.append((step, T.request(f"it{i}", 0, 1, 20.0 + i, max_tokens=40, prompt=prompt)))
    sizes = (2, 4, 12, 20, 3, 14, 8, 18)
    for i, step in enumerate((20, 40, 55, 80, 100, 120, 140, 160)):
        shared = [101] * (4 * B) if i in (2, 6) else []  # bg1's first four blocks
        prompt = shared + [130 + i] * (sizes[i] * B - len(shared) - 5)
        out.append((step, T.request(f"v{i}", 0, 0, 30.0 + i, max_tokens=30 + 4 * i, prompt=prompt)))
    return sorted(out, key=lambda x: (x[0], x[1].request_id))


def random_workload(seed: int) -> list[tuple[int, T.Request]]:
    """(step, request): 24 requests over 150 steps, half background (priority 2), some interactive (1) and voice
    turns (0), whose prompts start with one of three shared prefixes or none."""
    rng = random.Random(seed)
    prefixes = [[], [201] * (2 * B), [202] * (4 * B), [203] * (6 * B)]
    out = []
    for i in range(24):
        priority = rng.choices((0, 1, 2), weights=(3, 2, 5))[0]
        prompt = rng.choice(prefixes) + [300 + i] * (rng.randint(1, 12) * B - rng.randint(0, B - 1))
        max_tokens = rng.randint(50, 250) if priority == 2 else rng.randint(10, 60)
        max_tokens = max(1, min(max_tokens, 24 * B - len(prompt) - 4))  # the suite's max_model_len
        step = rng.randint(0, 150)
        request = T.request(
            f"q{i}", 0, priority, step + i / 100, max_tokens=max_tokens, prompt=prompt
        )
        out.append((step, request))
    return sorted(out, key=lambda x: (x[0], x[1].request_id))


def trace(sched, workload, sync: bool) -> list[tuple]:
    pipe = None if sync else T.Pipelined(sched)
    pending = list(workload)
    seen = []
    for n in range(STEPS):
        while pending and pending[0][0] == n:
            sched.add_request(pending.pop(0)[1])
        seen.append(record(sched, T.step(sched) if sync else pipe.step()))
    return seen


class PlannedConnector(T.Connector):
    """T.Connector with matched tokens bounded by what the request still lacks, in whole blocks."""

    def __init__(self, matched: dict[str, int]):
        super().__init__()
        self.plan = matched

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if request.request_id in self.delayed:
            return None, False
        num_tokens = min(self.plan.get(request.request_id, 0), request.num_tokens - 1)
        num_tokens = max(0, (num_tokens - num_computed_tokens) // B * B)
        return num_tokens, num_tokens > 0


def connector_trace(sched, seed: int, sync: bool) -> list[tuple]:
    """A random workload with a KV connector: 40% of the requests find part of their prompt outside the GPU and load
    it in 1-8 steps (again after a preemption), a quarter answer "not yet" for a few steps after they arrive, and in
    three windows the connector has frees pending."""
    workload = random_workload(seed)
    rng = random.Random(10_000 + seed)
    matched, delay, not_ready = {}, {}, {}
    for step, req in workload:
        if rng.random() < 0.4:
            matched[req.request_id] = rng.randint(1, max(1, req.num_prompt_tokens // B)) * B
        delay[req.request_id] = rng.randint(1, 8)
        if rng.random() < 0.25:
            not_ready[req.request_id] = range(step, step + rng.randint(1, 6))
    pending_frees = set()
    for _ in range(3):
        start = rng.randint(0, STEPS - 5)
        pending_frees.update(range(start, start + rng.randint(2, 4)))
    connector = PlannedConnector(matched)
    sched.connector = connector
    if not sync:
        sched.defer_block_free = True
    pending = list(workload)
    loads: dict[str, int] = {}  # request id -> the step its current load was first seen
    reported: set[str] = set()  # loads reported done whose request still waits for its KV
    in_flight, seen = [], []
    for n in range(STEPS):
        while pending and pending[0][0] == n:
            sched.add_request(pending.pop(0)[1])
        connector.delayed = {rid for rid, steps in not_ready.items() if n in steps}
        connector.pending_frees = n in pending_frees
        out = sched.schedule()
        for rid, req in sched.requests.items():
            if req.status != RequestStatus.WAITING_FOR_REMOTE_KVS:
                reported.discard(rid)
            elif rid not in loads and rid not in reported:
                loads[rid] = n
        done = {rid for rid, start in loads.items() if n >= start + delay[rid]}
        for rid in done:
            del loads[rid]
        reported |= done
        model_output = T.output(sched, out, done or None)
        if sync:
            sched.update_from_output(out, model_output)
        else:
            in_flight.append((out, model_output))
            if len(in_flight) > 1:
                sched.update_from_output(*in_flight.pop(0))
        seen.append((*record(sched, out), sorted(done)))
    return seen


def scheduler(
    module, sync: bool, model, switches, watermark=0.0, prefix_caching=True, num_blocks=48
):
    cls = module.Scheduler if sync else module.AsyncScheduler
    return T.scheduler(
        cls,
        model,
        num_blocks,
        switches,
        max_num_batched_tokens=128,
        long_prefill_token_threshold=64,
        watermark=watermark,
        prefix_caching=prefix_caching,
    )


def run(module, model, workload: str, name: str, mode: str, watermark: float, **kwargs):
    """The trace of ``workload`` ("suite", "pressure", "random-SEED" or "connector-SEED") on ``module``, and the
    counters it has (None for one it has not)."""
    sync = mode == "sync"
    sched = scheduler(module, sync, model, SWITCHES[name], watermark, **kwargs)
    kind, _, seed = workload.partition("-")
    if kind == "connector":
        seen = connector_trace(sched, int(seed), sync)
    else:
        requests = {"suite": T.workload, "pressure": pressure_workload}.get(kind)
        seen = trace(sched, requests() if requests else random_workload(int(seed)), sync)
    return seen, tuple(getattr(sched, name, getattr(sched, old, None)) for name, old in COUNTERS)


def assert_same(variant, other, model, *args, **kwargs):
    v, v_counters = run(variant, model, *args, **kwargs)
    o, o_counters = run(other, model, *args, **kwargs)
    first = next((i for i, (a, b) in enumerate(zip(v, o, strict=True)) if a != b), None)
    assert first is None, (
        f"first divergence at step {first}: the variant {v[first]}, the other {o[first]}"
    )
    assert None not in v_counters
    if REFERENCE_BOUNDS_HELD:
        assert v_counters == o_counters
    elif REFERENCE:
        held = HELD_TOKENS
        assert (
            v_counters[:held] + v_counters[held + 1 :] == o_counters[:held] + o_counters[held + 1 :]
        )
        assert v_counters[held] <= o_counters[held]
    else:
        assert v_counters[:3] == o_counters[:3]  # the PRs' counters


@pytest.mark.parametrize("workload", ["suite", "pressure"])
@pytest.mark.parametrize("num_blocks", [48, 64])
@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize("watermark", [0.0, 0.07])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("name", list(SWITCHES))
def test_the_variant_schedules_as_the_prs(
    variant, other, model, name, mode, watermark, prefix_caching, num_blocks, workload
):
    assert_same(
        variant,
        other,
        model,
        workload,
        name,
        mode,
        watermark,
        prefix_caching=prefix_caching,
        num_blocks=num_blocks,
    )


RANDOM = [
    (name, mode, watermark)
    for name in SWITCHES
    if name != "B16"
    for mode in ("sync", "async")
    for watermark in (0.0, 0.07)
]


@pytest.mark.parametrize("seed", range(SEEDS))
@pytest.mark.parametrize("config", RANDOM, ids=lambda c: "-".join(map(str, c)))
def test_the_variant_schedules_as_the_prs_on_random_workloads(variant, other, model, seed, config):
    name, mode, watermark = config
    assert_same(variant, other, model, f"random-{seed}", name, mode, watermark)


CONNECTOR = [
    (name, mode, watermark)
    for name in SWITCHES
    for mode in ("sync", "async")
    for watermark in (0.0, 0.07)
]


@pytest.mark.parametrize("seed", range(SEEDS))
@pytest.mark.parametrize("config", CONNECTOR, ids=lambda c: "-".join(map(str, c)))
def test_the_variant_schedules_as_the_prs_with_a_kv_connector(variant, other, model, seed, config):
    name, mode, watermark = config
    assert_same(variant, other, model, f"connector-{seed}", name, mode, watermark)


@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_with_the_switches_off_both_schedule_as_stock(variant, other, model, mode, prefix_caching):
    sync = mode == "sync"
    stock_cls = T.StockScheduler if sync else StockAsyncScheduler
    stock_sched = T.scheduler(
        stock_cls,
        model,
        48,
        None,
        max_num_batched_tokens=128,
        long_prefill_token_threshold=64,
        prefix_caching=prefix_caching,
    )
    stock = trace(stock_sched, T.workload(), sync)
    assert any(step[1] for step in stock)  # the workload preempts
    for module in (variant, other):
        sched = scheduler(module, sync, model, None, prefix_caching=prefix_caching)
        assert trace(sched, T.workload(), sync) == stock


# The configurations a mutant is looked for in, in order: each mutant changes a rule some of them reach.
SAMPLE = [
    *(
        ("pressure", name, mode, 0.0)
        for name in ("A2", "AB16", "AB48", "A4")
        for mode in ("sync", "async")
    ),
    *((f"random-{seed}", name, mode, w) for seed in range(5) for name, mode, w in RANDOM),
    *((f"connector-{seed}", name, mode, w) for seed in range(5) for name, mode, w in CONNECTOR),
]


@pytest.mark.skipif(bool(REFERENCE), reason="the mutants are of the PRs' module")
@pytest.mark.parametrize("mutant", list(MUTANTS))
def test_a_mutant_of_the_prs_diverges(tmp_path, build, variant, model, mutant):
    old, new = MUTANTS[mutant]
    source = prs_source(build)
    assert source.count(old) == 1, mutant
    path = tmp_path / f"flatshare_trace_{mutant.replace('-', '_')}.py"
    path.write_text(source.replace(old, new))
    mutated = load(path, path.stem)
    for workload, name, mode, watermark in SAMPLE:
        v, _ = run(variant, model, workload, name, mode, watermark)
        m, _ = run(mutated, model, workload, name, mode, watermark)
        if v != m:
            return
    pytest.fail(f"no sampled configuration tells {mutant} from the variant")
