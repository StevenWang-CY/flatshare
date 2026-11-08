"""flatshare_queue: within a priority, the waiting request with the least prefill left, net of what is cached, goes first.

First come first served over an LRU CPU tier is the tier's cyclic worst case: under load a session's next request waits
longer than the tier keeps its prefix, so each request computes its context again, which makes the queue longer. On the
card 89.5% of the prompt tokens beside 24 agent sessions, nearly all of them the agents', were computed again although
they had been computed a turn before (benchmarks/results/2026-11-08-envelope). Taking first the request with the least
left to compute, counting what the GPU pool and the CPU tier hold now, serves the sessions whose prefix is still cached,
which keeps it cached. A request that has waited ``max_wait_s`` goes ahead of the rest, the longest waiting first, so
that a long prompt is not starved; the rounds measured a bound of 120 s. The wait counts from the request's arrival, so a
request preempted after running counts its running time too.

The serve module (flatshare_make_room_serve) installs it as the scheduler's waiting queue where the engine's additional
config gives ``flatshare_queue.max_wait_s`` above 0 (the serving profile's queue_max_wait_s), and fixes its order at the
start of each step. Priorities stay strict; only requests of priority 1 or more (the agents, not the calls) are ordered
this way, and the calls keep their arrival order, so traffic of priority 0 alone is served as by vLLM's queue. Under a KV
connector other than the OffloadingConnector, or none, the count is the GPU pool's alone.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import Any

from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.kv_offload.base import LookupResult

FROM_PRIORITY = 1
"""The priorities ordered by what is left to compute: the agents'. Priority 0, the calls, keeps arrival order."""
REFRESH_S = 0.25
"""How long a request's count of cached tokens stands before it is looked up again."""
REFRESH_PER_STEP = 2
"""How many counts a step may refresh, the never counted and then the stalest first: a count costs the host 1.0-2.5 ms at
20-30k tokens held on the GPU and every one of 24 at once 24-60 ms (benchmarks/results/2026-11-08-envelope/count-cost.out),
which in one step would hold a call's next decode back."""


def max_wait_s(config: Any) -> int:
    """The bound the engine's additional config gives the queue, in seconds; 0 where it asks for vLLM's own queue."""
    queue = (config.additional_config or {}).get("flatshare_queue") or {}
    bound = int(queue.get("max_wait_s", 0))
    if bound < 0:
        raise ValueError(f"flatshare_queue.max_wait_s must be 0 (off) or more, not {bound}")
    return bound


def realtime_present(scheduler: Any) -> bool:
    """Whether a realtime request (priority 0) waits or runs, or a call's announced turn holds."""
    if scheduler._hint_until and time.monotonic() < scheduler._hint_until:
        return True
    for queue in (scheduler.running, scheduler.kv_holding_waiting):
        if any(request.priority == 0 for request in queue):
            return True
    # The queue's own list, unordered: its order is what the caller is fixing.
    waiting = scheduler.waiting
    return isinstance(waiting, CacheAwareQueue) and any(
        request.priority == 0 for request in waiting._items
    )


def tier_tokens(scheduler: Any, request: Any, local: int = 0) -> int:
    """The tokens of ``request``'s prefix the GPU pool and the CPU tier hold together, as vLLM 0.32.0's own lookup
    counts them (OffloadingConnectorScheduler._maximal_prefix_lookup): the tier's chunks from the one the block-aligned
    GPU hit of ``local`` tokens ends in, up to the first it misses; 0 where the tier holds none of them. A turn that finds
    part of its prefix on the GPU refreshes in the tier's LRU only the chunks it loads or stores, so a prefix can lose a
    run of chunks in its middle while its tail stays: vLLM loads nothing past the hole, and nothing past it is counted.
    A chunk whose lookup is pending or to be retried counts as held, where vLLM defers the request until it knows; the
    box's CPUOffloadingManager answers hit or miss. update_offload_keys() fills the request's keys, which vLLM does only
    when the scheduler first tries the request."""
    if request.skip_reading_prefix_cache:
        return 0
    connector = getattr(getattr(scheduler, "connector", None), "connector_scheduler", None)
    status = getattr(connector, "_req_status", {}).get(request.request_id) if connector else None
    if status is None:
        return 0
    status.update_offload_keys()
    keys = status.group_states[0].offload_keys
    chunk = connector.config.kv_group_configs[0].tokens_per_chunk
    start = (local - local % scheduler.block_size) // chunk
    hits = 0
    for key in keys[start:]:
        if connector.manager.lookup(key, status.req_context) is LookupResult.MISS:
            break
        hits += 1
    return (start + hits) * chunk if hits else 0


def cached_tokens(scheduler: Any, request: Any) -> int:
    """The tokens of a waiting ``request`` that need not be computed: what the GPU pool holds, extended by what the CPU
    tier holds beyond it. A request that has computed tokens already waits in the scheduler's kv_holding_waiting, not in
    this queue, and a preempted one starts again from none; the first branch keeps the count right if one does come."""
    if request.num_computed_tokens:
        return request.num_computed_tokens
    hit = scheduler._local_prefix_hit(request)
    local = hit[1] if hit is not None else 0
    return max(local, tier_tokens(scheduler, request, local))


class CacheAwareQueue(RequestQueue):
    """A priority queue, ordered within a priority by what is left to compute. The scheduler's waiting loop peeks a
    request and pops it later in the step, so the order stands for a step: begin_step() fixes each request's key, and a
    request added meanwhile takes its key once. Not a PriorityRequestQueue, whose heap make-room reads."""

    def __init__(
        self, scheduler: Any, max_wait_s: float, refresh_per_step: int = REFRESH_PER_STEP
    ) -> None:
        self._items: list[Any] = []
        self.scheduler = scheduler
        self.max_wait_s = max_wait_s
        self.refresh_per_step = refresh_per_step
        self._cached: dict[str, tuple[float, int]] = {}
        self._keys: dict[str, tuple] = {}

    def _compute_key(self, request: Any) -> tuple:
        if request.priority < FROM_PRIORITY:
            return (request.priority, 0.0, request.arrival_time, request.request_id)
        waited = max(0.0, time.time() - request.arrival_time)
        if waited >= self.max_wait_s:
            # Past the bound, the longest waiting first, ahead of the rest.
            left = -1e12 - waited
        else:
            left = request.num_tokens - self._cached.get(request.request_id, (None, 0))[1]
        return (request.priority, left, request.arrival_time, request.request_id)

    def _key(self, request: Any) -> tuple:
        key = self._keys.get(request.request_id)
        if key is None:
            key = self._keys[request.request_id] = self._compute_key(request)
        return key

    def begin_step(self) -> None:
        """Fix the step's order. A few stale counts are refreshed first, the never counted and then the stalest, and none
        while a realtime request waits or runs or a call's announced turn holds: a count's host time would delay the
        call's next step. Meanwhile a request never counted is ordered by its whole prompt, as if nothing were cached."""
        waiting = {request.request_id for request in self._items}
        self._cached = {key: value for key, value in self._cached.items() if key in waiting}
        if self.refresh_per_step and not realtime_present(self.scheduler):
            now = time.monotonic()
            stale = [
                (self._cached.get(request.request_id, (float("-inf"), 0))[0], request)
                for request in self._items
                if request.priority >= FROM_PRIORITY
            ]
            stale = [entry for entry in stale if now - entry[0] >= REFRESH_S]
            stale.sort(key=lambda entry: entry[0])
            for _, request in stale[: self.refresh_per_step]:
                self._cached[request.request_id] = (now, cached_tokens(self.scheduler, request))
        self._keys = {request.request_id: self._compute_key(request) for request in self._items}

    def add_request(self, request: Any) -> None:
        self._items.append(request)

    def pop_request(self) -> Any:
        if not self._items:
            raise IndexError("pop from empty queue")
        request = min(self._items, key=self._key)
        self._items.remove(request)
        return request

    def peek_request(self) -> Any:
        if not self._items:
            raise IndexError("peek from empty queue")
        return min(self._items, key=self._key)

    def prepend_request(self, request: Any) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[Any]) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Any) -> None:
        self._items.remove(request)

    def remove_requests(self, requests: Iterable[Any]) -> None:
        gone = requests if isinstance(requests, set) else set(requests)
        self._items = [request for request in self._items if request not in gone]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(sorted(self._items, key=self._key))
