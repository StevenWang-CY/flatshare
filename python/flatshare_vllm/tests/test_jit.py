"""Just-in-time scheduling of the next batch (src/flatshare_vllm/serve/flatshare_jit.py), on a stand-in engine core
and a clock of its own: the step-time model as crates/flatshare-core/src/step.rs counts a step, and when the core
waits, how long, and what it handles while it waits. Needs neither vLLM nor torch."""

import importlib.util
import queue
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

JIT = Path(__file__).resolve().parents[1] / "src/flatshare_vllm/serve/flatshare_jit.py"
_SPEC = importlib.util.spec_from_file_location("flatshare_jit", JIT)
jit = importlib.util.module_from_spec(_SPEC)
sys.modules["flatshare_jit"] = jit  # a dataclass resolves its annotations through its module
_SPEC.loader.exec_module(jit)

# StepCost::frozen_box() (crates/flatshare-core/src/step.rs).
BOX = jit.StepCost(
    step_ms=11.56, us_per_token=47.99, ns_per_pair=14.17, ns_per_context=429.3, mixed_ms=7.6
)
MARGIN_S = 0.030


def output(prefills: dict[str, tuple[int, int]], decodes: dict[str, int]) -> SimpleNamespace:
    """A scheduler output: each prefill's (tokens, tokens computed before) and each decode's context before it, as
    vLLM 0.32's SchedulerOutput carries them (new requests' computed tokens, cached requests' parallel lists)."""
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, num_computed_tokens=before)
            for r, (_, before) in prefills.items()
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(decodes), num_computed_tokens=list(decodes.values())
        ),
        num_scheduled_tokens={
            **{r: n for r, (n, _) in prefills.items()},
            **dict.fromkeys(decodes, 1),
        },
    )


def test_a_steps_duration_is_step_rs_model():
    """step.rs's own case: a 256-token chunk after 1,000 cached tokens beside a decode at context 3,000."""
    load = output({"agent": (256, 1_000)}, {"call": 3_000})
    pairs = 256 * 1_000 + 256 * 257 // 2 + 3_001
    expected = 11.56 + 7.6 + 47.99 * 0.257 + 14.17 * pairs / 1e6 + 429.3 * (1_256 + 3_001) / 1e6
    assert BOX.seconds(load) * 1e3 == pytest.approx(expected, abs=1e-9)
    # A decode alone adds no mixed step's cost; an agent's chunk deep in its context is most of a long step.
    assert BOX.seconds(output({}, {"call": 3_000})) * 1e3 == pytest.approx(
        11.56
        + 0.04799
        + 14.17 * 3_001 / 1e6
        + 429.3 * 3_001 / 1e6  # a decode's pairs: its context and itself
    )
    assert BOX.seconds(output({"agent": (256, 20_000)}, {})) > 0.08


class Clock:
    """The stand-in's time: the core's waits advance it, as its input queue's timeouts do."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class Inputs:
    """The core's input queue: each request arrives at its time, and a get that times out first moves the clock on."""

    def __init__(self, clock: Clock, arrivals: list[tuple[float, str]]) -> None:
        self.clock = clock
        self.arrivals = deque(sorted(arrivals))

    def get(self, timeout: float):
        if self.arrivals and self.arrivals[0][0] <= self.clock.now + timeout:
            at, request = self.arrivals.popleft()
            self.clock.now = max(self.clock.now, at)
            return ("ADD", request)
        self.clock.now += timeout
        raise queue.Empty


class Future:
    def __init__(self, clock: Clock, end: float) -> None:
        self.clock, self.end = clock, end

    def done(self) -> bool:
        return self.clock.now >= self.end


class Core:
    """vLLM 0.32's EngineCore as far as the wrapper sees it: one batch in flight, which ends at ``end``, a scheduler
    with or without requests, and an original step that records when it was called and what had been added."""

    def __init__(
        self, clock: Clock, inflight: SimpleNamespace, end: float, arrivals=(), has_requests=True
    ) -> None:
        self.clock = clock
        self.batch_queue = deque([(Future(clock, end), inflight, None)])
        self.input_queue = Inputs(clock, list(arrivals))
        self.added: list[str] = []
        self.calls: list[tuple[float, list[str]]] = []
        self.scheduler = SimpleNamespace(
            has_requests=lambda: has_requests, update_from_output=lambda *args: None
        )

    def _handle_client_request(self, kind: str, request: str) -> None:
        self.added.append(request)

    def step_with_batch_queue(self):
        self.calls.append((self.clock.now, list(self.added)))


@pytest.fixture
def clock(monkeypatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(jit, "time", clock)
    return clock


def installed() -> type:
    cls = type("JitCore", (Core,), {})
    jit.install(cls, MARGIN_S, BOX)
    return cls


LONG = output({"agent": (256, 20_000)}, {"agent-2": 9_000})  # ~110 ms on the model


def test_a_request_arriving_during_a_long_step_joins_the_next_batch(clock):
    """The step in flight started at 0 and the model gives it ~110 ms: the core handles the call that arrives at 20 ms
    and schedules the next batch at the step's end less the margin, with the call in it."""
    cls = installed()
    core = cls(clock, LONG, end=BOX.seconds(LONG), arrivals=[(0.020, "call")])
    core.flatshare_jit_started = 0.0
    core.step_with_batch_queue()
    [(at, added)] = core.calls
    assert added == ["call"]
    assert at == pytest.approx(BOX.seconds(LONG) - MARGIN_S, abs=jit.POLL_S)


def test_a_step_that_ends_early_is_followed_at_once(clock):
    """The model gives the step ~110 ms and it ends at 40: the next batch is scheduled then, not at the deadline."""
    cls = installed()
    core = cls(clock, LONG, end=0.040)
    core.flatshare_jit_started = 0.0
    core.step_with_batch_queue()
    assert core.calls[0][0] == pytest.approx(0.040, abs=jit.POLL_S)


@pytest.mark.parametrize(
    ("inflight", "has_requests", "started"),
    [
        (
            output({}, {"call": 3_000, "agent": 9_000}),
            True,
            0.0,
        ),  # a decode-only step, shorter than the margin
        (LONG, False, 0.0),  # nothing to schedule
        (LONG, True, None),  # a step whose start is not known
    ],
)
def test_the_core_does_not_wait(clock, inflight, has_requests, started):
    cls = installed()
    core = cls(clock, inflight, end=1.0, has_requests=has_requests)
    core.flatshare_jit_started = started
    core.step_with_batch_queue()
    assert core.calls == [(0.0, [])]


def test_a_batch_launched_beside_another_starts_as_that_ones_output_arrives(clock):
    """The wrapper has the scheduler record when each output reaches it: a batch the original step launched beside
    the one in flight, and then waited on, started on the GPU at that moment, not when the step returned."""

    class Beside(Core):
        def step_with_batch_queue(
            self,
        ):  # launches beside the batch in flight, then waits on it and processes it
            self.batch_queue.appendleft((Future(clock, 1.0), LONG, None))
            clock.now = 0.050
            self.batch_queue.pop()
            self.scheduler.update_from_output(None, None)
            clock.now = 0.053

    jit.install(Beside, MARGIN_S, BOX)
    core = Beside(clock, LONG, end=0.050, has_requests=False)
    core.flatshare_jit_started = None
    core.step_with_batch_queue()
    assert core.flatshare_jit_started == pytest.approx(0.050)


def test_installing_again_replaces_the_settings_not_the_wrapper(clock):
    cls = installed()
    wrapper = cls.step_with_batch_queue
    jit.install(cls, 0.010, BOX)
    assert cls.step_with_batch_queue.flatshare_jit_original is wrapper.flatshare_jit_original
    assert cls.step_with_batch_queue.flatshare_jit == (0.010, BOX)
    core = cls(clock, LONG, end=BOX.seconds(LONG))
    core.flatshare_jit_started = 0.0
    core.step_with_batch_queue()
    assert core.calls[0][0] == pytest.approx(BOX.seconds(LONG) - 0.010, abs=jit.POLL_S)
