"""flatshare_queue (src/flatshare_vllm/serve/flatshare_queue.py) as the serve module installs it, on vLLM's real scheduler on
CPU: the order of the waiting queue, net of what the GPU pool and a stand-in CPU tier hold; priorities stay strict; the
bound puts the longest waiting first; a step refreshes few counts and none beside a call; the order stands for a step;
and the control, no flatshare_queue table or a bound of 0, is vLLM's queue."""

import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.core.sched.scheduler")

from test_make_room import module, request, scheduler, step  # noqa: F401
from vllm.v1.core.sched.request_queue import PriorityRequestQueue
from vllm.v1.kv_offload.base import LookupResult

SERVE = Path(__file__).resolve().parents[1] / "src/flatshare_vllm/serve"
SWITCHES = {"max_preemptions_per_step": 2, "budget_cap": True, "budget_cap_tokens": 48}
SERVED = ("flatshare_copy", "flatshare_jit", "flatshare_queue", "flatshare_make_room_serve")
NEVER = 10**10
"""A bound no request here reaches: most arrive at the epoch's first seconds, which a bound of 120 s puts past it, all
of them, so that the order would be the longest waiting first (the bound's own tests use times near now)."""


@pytest.fixture
def serve(module, monkeypatch):  # noqa: F811
    """The serve module over make-room's, imported fresh with the modules it installs from."""
    monkeypatch.syspath_prepend(str(SERVE))
    monkeypatch.setitem(sys.modules, "flatshare_make_room_scheduler", module)
    for name in SERVED:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("flatshare_make_room_serve")


def make(serve, model, max_wait_s=NEVER, **kwargs):  # noqa: F811
    return scheduler(
        serve.AsyncScheduler, model, 512, dict(SWITCHES), max_num_batched_tokens=64, max_model_len=4096,
        prefix_caching=True, additional={"flatshare_queue": {"max_wait_s": max_wait_s}}, **kwargs,
    )


@pytest.mark.parametrize("additional", [None, {"flatshare_queue": {"max_wait_s": 0}}])
def test_without_a_bound_it_is_vllms_queue(serve, model, additional):  # noqa: F811
    """The negative control: no table, or the profile's off (0), leaves make-room's queue in place."""
    sched = scheduler(serve.AsyncScheduler, model, 512, dict(SWITCHES), max_model_len=4096, additional=additional)
    assert type(sched.waiting) is PriorityRequestQueue
    assert serve.effective(sched)["queue"]["max_wait_s"] == 0


def test_a_negative_bound_is_refused(serve, model):  # noqa: F811
    with pytest.raises(ValueError, match="max_wait_s"):
        make(serve, model, max_wait_s=-1)


def test_the_shortest_goes_first_and_priorities_stay_strict(serve, model):  # noqa: F811
    sched = make(serve, model)
    assert type(sched.waiting) is serve.flatshare_queue.CacheAwareQueue
    assert serve.effective(sched)["queue"] == {
        "max_wait_s": NEVER, "waiting_queue": "flatshare_queue.CacheAwareQueue",
    }
    sched.add_request(request("long", 800, 2, 1.0, token=1))
    sched.add_request(request("short", 64, 2, 2.0, token=2))
    sched.add_request(request("call", 400, 0, 3.0, token=3))
    sched.waiting.begin_step()
    assert [r.request_id for r in sched.waiting] == ["call", "short", "long"]
    assert sched.waiting.peek_request().request_id == "call"
    assert sched.waiting.pop_request().request_id == "call"
    assert sched.waiting.pop_request().request_id == "short"
    assert sched.waiting.pop_request().request_id == "long"


def test_the_calls_keep_their_arrival_order(serve, model):  # noqa: F811
    sched = make(serve, model)
    sched.add_request(request("call-long", 800, 0, 1.0, token=1))
    sched.add_request(request("call-short", 64, 0, 2.0, token=2))
    sched.waiting.begin_step()
    assert [r.request_id for r in sched.waiting] == ["call-long", "call-short"]


def test_what_the_gpu_holds_is_not_left_to_compute(serve, model):  # noqa: F811
    sched = make(serve, model)
    first = request("first", 320, 2, 1.0, token=1, max_tokens=1)
    sched.add_request(first)
    for _ in range(12):
        step(sched)
    assert first.is_finished()
    # 320 tokens of the same prefix against 200 new ones: the longer prompt has less left.
    sched.add_request(request("fresh", 200, 2, 2.0, token=2))
    sched.add_request(request("again", 336, 2, 3.0, token=1))
    sched.waiting.begin_step()
    assert [r.request_id for r in sched.waiting] == ["again", "fresh"]


def tier(sched, held, chunks=50):
    """A stand-in CPU tier on ``sched`` that holds, of a request's ``chunks`` 16-token chunks, those numbered in ``held``
    (an int: the first that many)."""
    present = set(range(held)) if isinstance(held, int) else set(held)
    keys = [f"k{i}" for i in range(chunks)]
    status = SimpleNamespace(
        update_offload_keys=lambda: None,
        group_states=[SimpleNamespace(offload_keys=keys)],
        req_context=None,
    )
    manager = SimpleNamespace(
        lookup=lambda key, context: LookupResult.HIT if int(key[1:]) in present else LookupResult.MISS
    )
    sched.connector = SimpleNamespace(
        connector_scheduler=SimpleNamespace(
            _req_status={"loaded": status},
            manager=manager,
            config=SimpleNamespace(kv_group_configs=[SimpleNamespace(tokens_per_chunk=16)]),
        ),
        on_new_request=lambda request: None,
    )


@pytest.mark.parametrize(
    "held, local, want",
    [
        (0, 0, 0),
        (1, 0, 16),
        (40, 0, 640),
        (50, 0, 800),
        # A hole in the prefix's middle with its tail held, as the LRU leaves it when a turn found the head on the GPU:
        # vLLM loads the chunks before the hole only, and a bisection would have counted to the tail's end (800).
        ([*range(10), *range(20, 50)], 0, 160),
        # The head evicted from the tier but on the GPU: the count goes on from the GPU hit's chunk.
        (range(20, 50), 320, 800),
        (range(20, 50), 160, 0),  # the tier misses at the GPU hit's end: nothing beyond it
        (range(20, 30), 320, 480),
    ],
)
def test_what_the_tier_holds_is_counted_as_vllm_loads_it(serve, model, held, local, want):  # noqa: F811
    sched = make(serve, model)
    tier(sched, held)
    loaded = request("loaded", 800, 2, 1.0, token=1)
    assert serve.flatshare_queue.tier_tokens(sched, loaded, local) == want
    # A request the tier has no status for: nothing.
    assert serve.flatshare_queue.tier_tokens(sched, request("other", 400, 2, 2.0, token=2), local) == 0


def test_cached_tokens_are_the_gpu_hit_extended_by_the_tier(serve, model, monkeypatch):  # noqa: F811
    sched = make(serve, model)
    tier(sched, range(20, 30))
    loaded = request("loaded", 800, 2, 1.0, token=1)
    monkeypatch.setattr(sched, "_local_prefix_hit", lambda request: ((), 320))
    assert serve.flatshare_queue.cached_tokens(sched, loaded) == 480
    monkeypatch.setattr(sched, "_local_prefix_hit", lambda request: ((), 640))
    assert serve.flatshare_queue.cached_tokens(sched, loaded) == 640  # the tier misses at 640: the GPU's hit
    monkeypatch.setattr(sched, "_local_prefix_hit", lambda request: None)
    assert serve.flatshare_queue.cached_tokens(sched, loaded) == 0


def test_a_request_that_skips_the_prefix_cache_has_nothing_cached(serve, model):  # noqa: F811
    sched = make(serve, model)
    tier(sched, 50)
    skipping = SimpleNamespace(request_id="loaded", skip_reading_prefix_cache=True)
    assert serve.flatshare_queue.tier_tokens(sched, skipping) == 0
    reading = SimpleNamespace(request_id="loaded", skip_reading_prefix_cache=False)
    assert serve.flatshare_queue.tier_tokens(sched, reading) == 800  # the negative control


def test_what_the_tier_holds_is_not_left_to_compute(serve, model):  # noqa: F811
    sched = make(serve, model)
    tier(sched, 40)
    loaded = request("loaded", 800, 2, 1.0, token=1)
    other = request("other", 400, 2, 2.0, token=2)
    sched.waiting.add_request(loaded)
    sched.waiting.add_request(other)
    sched.waiting.refresh_per_step = 8
    sched.waiting.begin_step()
    # 800 - 640 = 160 left against 400; without the tier's count, the negative control, 800 against 400.
    assert sched.waiting.peek_request().request_id == "loaded"
    tier(sched, 0)
    sched.waiting._cached.clear()
    sched.waiting.begin_step()
    assert sched.waiting.peek_request().request_id == "other"


def test_the_bound_puts_the_longest_waiting_first(serve, model):  # noqa: F811
    sched = make(serve, model, max_wait_s=60)
    now = time.time()
    sched.waiting.add_request(request("short", 64, 2, now, token=2))
    sched.waiting.add_request(request("long", 800, 2, now - 30.0, token=1))
    sched.waiting.begin_step()
    assert sched.waiting.peek_request().request_id == "short"
    sched.waiting.add_request(request("older", 900, 2, now - 90.0, token=3))
    sched.waiting.add_request(request("oldest", 2000, 2, now - 120.0, token=4))
    sched.waiting.begin_step()
    assert [r.request_id for r in sched.waiting][:2] == ["oldest", "older"]
    # A realtime request still goes first.
    sched.waiting.add_request(request("call", 400, 0, now, token=5))
    sched.waiting.begin_step()
    assert sched.waiting.peek_request().request_id == "call"


def test_a_step_refreshes_few_counts_and_none_beside_a_call(serve, model, monkeypatch):  # noqa: F811
    """A step counts at most REFRESH_PER_STEP requests, the never counted first, and none while a call waits or runs."""
    sched = make(serve, model)
    calls = []
    monkeypatch.setattr(serve.flatshare_queue, "cached_tokens", lambda s, request: calls.append(request.request_id) or 0)
    now = time.time()
    for i in range(6):
        sched.waiting.add_request(request(f"a{i}", 64, 2, now - i, token=i + 1))
    sched.waiting.begin_step()
    assert len(calls) == 2
    sched.waiting.begin_step()
    sched.waiting.begin_step()
    assert len(calls) == 6 and len(set(calls)) == 6  # each counted once before any is counted again
    calls.clear()
    sched.waiting.add_request(request("call", 64, 0, now, token=9))
    sched.waiting.begin_step()
    assert calls == []
    # The order still stands: the call first, then the agents by their (stale) counts.
    assert sched.waiting.peek_request().request_id == "call"


def test_a_running_call_or_an_announced_turn_stops_the_refresh(serve, model, monkeypatch):  # noqa: F811
    sched = make(serve, model)
    calls = []
    monkeypatch.setattr(serve.flatshare_queue, "cached_tokens", lambda s, request: calls.append(request.request_id) or 0)
    sched.waiting.add_request(request("agent", 64, 2, time.time(), token=1))
    sched.running.append(SimpleNamespace(priority=0))
    sched.waiting.begin_step()
    assert calls == []
    sched.running.clear()
    sched._hint_until = time.monotonic() + 60
    sched.waiting.begin_step()
    assert calls == []
    sched._hint_until = 0.0
    sched.waiting.begin_step()
    assert calls == ["agent"]


def test_the_order_stands_for_a_step(serve, model):  # noqa: F811
    """The waiting loop peeks a request and pops it later in the step: what changes meanwhile moves nothing until the
    next step."""
    sched = make(serve, model)
    now = time.time()
    sched.waiting.add_request(request("a", 64, 2, now, token=1))
    sched.waiting.add_request(request("b", 800, 2, now, token=2))
    sched.waiting.begin_step()
    peeked = sched.waiting.peek_request()
    assert peeked.request_id == "a"
    sched.waiting._cached["b"] = (time.monotonic(), 800)  # b has nothing left: the next step's order differs
    assert sched.waiting.peek_request() is peeked and sched.waiting.pop_request() is peeked
    sched.waiting.add_request(peeked)
    sched.waiting.begin_step()
    assert sched.waiting.peek_request().request_id == "b"


def test_schedule_fixes_the_order_each_step(serve, model, monkeypatch):  # noqa: F811
    sched = make(serve, model)
    begun = []
    monkeypatch.setattr(sched.waiting, "begin_step", lambda: begun.append(True))
    step(sched)
    step(sched)
    assert len(begun) == 2


def test_make_room_sees_a_lower_priority_anywhere_in_the_queue(serve, model):  # noqa: F811
    """make-room reads a PriorityRequestQueue's heap for a lower priority, its leaves alone: this queue is not one."""
    sched = make(serve, model)
    sched.add_request(request("agent", 64, 2, 1.0, token=1))
    sched.add_request(request("call-1", 64, 0, 2.0, token=2))
    sched.add_request(request("call-2", 64, 0, 3.0, token=3))
    sched.add_request(request("call-3", 64, 0, 4.0, token=4))
    sched.waiting.begin_step()
    assert sched._has_lower_priority_than(sched.waiting, 0)
    assert not sched._has_lower_priority_than(sched.waiting, 2)


@pytest.mark.parametrize("max_wait_s", [1, 10**10])
def test_it_runs_as_a_queue_in_the_scheduler(serve, model, max_wait_s):  # noqa: F811
    """Every request ends, whatever the order (the requests arrived at the epoch's first seconds: past a bound of 1 s,
    within one of 10**10), and a request removed while waiting leaves no state behind."""
    sched = make(serve, model, max_wait_s=max_wait_s)
    reqs = [request(f"r{i}", 64 * (1 + i % 5), 2 + i % 2, float(i), token=i + 1, max_tokens=2) for i in range(12)]
    for req in reqs:
        sched.add_request(req)
    gone = reqs[-1]
    sched.finish_requests(gone.request_id, type(gone.status).FINISHED_ABORTED)
    for _ in range(200):
        if all(r.is_finished() for r in reqs):
            break
        step(sched)
    assert all(r.is_finished() for r in reqs)
    assert not sched.waiting
    step(sched)
    assert not sched.waiting._keys and not sched.waiting._cached


def test_it_runs_beside_make_rooms_preemption(serve, model):  # noqa: F811
    """test_trace.py's pressure workload: background decodes fill a pool of 48 blocks, and interactive requests and
    voice turns preempt them, which puts them back in the waiting queue: every request ends, preemptions happened, and
    the queue keeps no state after."""
    import test_trace as TR

    sched = scheduler(
        serve.AsyncScheduler, model, 48, dict(SWITCHES), max_num_batched_tokens=128, long_prefill_token_threshold=64,
        prefix_caching=True, additional={"flatshare_queue": {"max_wait_s": NEVER}},
    )
    pending = TR.pressure_workload()
    reqs = [req for _, req in pending]
    preempted = 0
    for n in range(8 * TR.STEPS):  # 1,123 steps here; vLLM's own queue, 1,062
        while pending and pending[0][0] == n:
            sched.add_request(pending.pop(0)[1])
        preempted += len(step(sched).preempted_req_ids or ())
        if not pending and all(req.is_finished() for req in reqs):
            break
    assert all(req.is_finished() for req in reqs)
    assert preempted > 0
    assert not sched.waiting
    step(sched)
    assert not sched.waiting._keys and not sched.waiting._cached


@pytest.mark.parametrize("workload", ["suite", "pressure"])
def test_past_its_bound_it_schedules_as_vllms_queue(serve, model, workload):  # noqa: F811
    """Past the bound every request goes in arrival order within its priority, which is vLLM's priority queue: with a
    bound of 1 s, which these requests' arrival times (the epoch's first seconds) are all past, the queue schedules as
    make-room's own queue, step for step, preemptions and skips included. Its contract with the scheduler (peek, pop,
    prepend, removal) is vLLM's queue's."""
    import test_trace as TR

    requests = {"suite": TR.T.workload, "pressure": TR.pressure_workload}[workload]
    seen = [
        TR.trace(
            scheduler(
                serve.AsyncScheduler, model, 48, dict(SWITCHES), max_num_batched_tokens=128,
                long_prefill_token_threshold=64, prefix_caching=True, additional=additional,
            ),
            requests(),
            sync=True,
        )
        for additional in (None, {"flatshare_queue": {"max_wait_s": 1}})
    ]
    first = next((i for i, (a, b) in enumerate(zip(*seen, strict=True)) if a != b), None)
    assert first is None, f"first divergence at step {first}: {seen[0][first]} against {seen[1][first]}"
    if workload == "pressure":
        # Its waiting agents of one priority do not compete in the 240 traced steps, so both orders schedule alike
        # there (they part later: all its requests end at step 1,123 against 1,062 for vLLM's queue).
        return
    # The negative control: within the bound's reach (10**10 s), the order is by what is left, and the schedule differs.
    ordered = TR.trace(
        scheduler(
            serve.AsyncScheduler, model, 48, dict(SWITCHES), max_num_batched_tokens=128,
            long_prefill_token_threshold=64, prefix_caching=True, additional={"flatshare_queue": {"max_wait_s": NEVER}},
        ),
        requests(),
        sync=True,
    )
    assert ordered != seen[0]
