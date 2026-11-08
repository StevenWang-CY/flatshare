"""How long flatshare_queue's count of a waiting request's cached tokens takes on this CPU: a prefix of BLOCKS blocks in
the GPU pool, 24 requests waiting on it, each counted as begin_step() counts it when its count is stale; the prefix held by the GPU pool,
or by a stand-in tier that holds the whole prefix with nothing on the GPU (the forward scan's longest case, a dict
read a chunk, as vLLM's CPUOffloadingManager.lookup is). Not a test: run with pytest -s."""

import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import conftest
from test_make_room import module, request, scheduler, step  # noqa: F401
from test_queue import SWITCHES, serve  # noqa: F401
from vllm.v1.kv_offload.base import LookupResult


def tier(sched, ids, chunks):
    held = {f"k{i}" for i in range(chunks)}
    statuses = {
        rid: SimpleNamespace(
            update_offload_keys=lambda: None,
            group_states=[SimpleNamespace(offload_keys=[f"k{i}" for i in range(chunks)])],
            req_context=None,
        )
        for rid in ids
    }
    sched.connector = SimpleNamespace(
        connector_scheduler=SimpleNamespace(
            _req_status=statuses,
            manager=SimpleNamespace(
                lookup=lambda key, context: LookupResult.HIT if key in held else LookupResult.MISS
            ),
            config=SimpleNamespace(kv_group_configs=[SimpleNamespace(tokens_per_chunk=16)]),
        ),
        on_new_request=lambda request: None,
    )


@pytest.mark.parametrize("where", ["gpu", "tier"])
@pytest.mark.parametrize("blocks", [250, 1250, 1875])
def test_count_cost(serve, tmp_path, blocks, where):  # noqa: F811
    model = str(tmp_path / "model")
    (tmp_path / "model").mkdir()
    conftest.write_model(tmp_path / "model", 65536)
    tokens = blocks * 16
    sched = scheduler(
        serve.AsyncScheduler,
        model,
        2 * blocks + 64,
        dict(SWITCHES),
        max_num_batched_tokens=8192,
        max_model_len=tokens + 256,
        prefix_caching=True,
        additional={"flatshare_queue": {"max_wait_s": 120}},
    )
    if where == "gpu":
        first = request("first", tokens, 2, 0.0, max_tokens=1, token=3)
        sched.add_request(first)
        for _ in range(4 + tokens // 8192):
            step(sched)
        assert first.is_finished()
    waiting = [request(f"w{i}", tokens + 64, 2, 1.0 + i, token=3) for i in range(24)]
    if where == "tier":
        tier(sched, [r.request_id for r in waiting], blocks)
    for req in waiting:
        sched.waiting.add_request(req)
    start = time.perf_counter()
    counts = [serve.flatshare_queue.cached_tokens(sched, req) for req in waiting]
    per = (time.perf_counter() - start) / len(waiting)
    assert counts[0] >= tokens - 16
    # Every count at once, as a step would refresh them without the per-step bound.
    sched.waiting.refresh_per_step = len(waiting)
    sched.waiting._cached.clear()
    start = time.perf_counter()
    sched.waiting.begin_step()
    every = time.perf_counter() - start
    print(
        f"\nblocks {blocks}, held on the {where}: one count {per * 1e3:.2f} ms; all 24 in one step {every * 1e3:.1f} ms"
    )
