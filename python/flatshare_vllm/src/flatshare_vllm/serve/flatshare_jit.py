"""Just-in-time scheduling of the next batch, which the serve module installs on the engine core when the serving
profile asks for it.

Under async scheduling vLLM 0.32's ``EngineCore.step_with_batch_queue`` schedules batch k+1 as soon as batch k is
launched, so a request that arrives while k runs is added only after k's output, scheduled into k+2, and waits for the
whole of k+1, which was committed before it arrived. On the box k+1 is often an agent's long-context prefill chunk
(65 ms at the median, 108 at p95, beside the calls), most of a call's first token past the step it arrived in.
Installed, the core waits to schedule k+1 until k is due to finish less a margin, handling its input queue meanwhile,
so that a request arriving during k joins k+1. It schedules at once when k finishes early, and never waits on a step
the model gives less than the margin (a decode-only step on the box). A step's duration is the profile's step-time
model over what the step carries (``StepCost``, crates/flatshare-core/src/step.rs fits it on the box); the margin
covers a schedule and a launch, the box's piecewise launch of a mixed step (25-26 ms) included.

A batch launched while another is in flight starts on the GPU when that one finishes: the time its output reaches the
scheduler, which ``install`` has the scheduler record.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any

POLL_S = 0.002
"""How long the wait blocks on the input queue at a time, between looks at the step in flight."""
STEP_COST = ("step_ms", "us_per_token", "ns_per_pair", "ns_per_context", "mixed_ms")
"""StepCost's coefficients, by name, as the serving profile gives them (jit_ and each) and the engine's additional
config carries them (flatshare_jit.step_cost)."""


@dataclass(frozen=True)
class StepCost:
    """The step-time model (crates/flatshare-core/src/step.rs, ``StepCost``): a step's fixed ms, the µs of each token
    it computes, the ns of each attention pair and of each token of context it reads, and the ms a step with both a
    prefill chunk and a decode adds."""

    step_ms: float
    us_per_token: float
    ns_per_pair: float
    ns_per_context: float
    mixed_ms: float

    def seconds(self, scheduler_output: Any) -> float:
        """The model's duration of the step ``scheduler_output`` schedules, in seconds: each request's tokens after
        those it had computed, as step.rs's ``StepLoad`` counts them."""
        computed = {r.req_id: r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        # Parallel lists in vLLM 0.32; a mismatch must not raise inside the engine core's loop.
        computed.update(zip(cached.req_ids, cached.num_computed_tokens, strict=False))
        tokens = pairs = context = prefills = decodes = 0
        for req_id, n in scheduler_output.num_scheduled_tokens.items():
            before = computed.get(req_id, 0)
            tokens += n
            pairs += n * before + n * (n + 1) // 2
            context += before + n
            if n > 1:
                prefills += 1
            else:
                decodes += 1
        mixed = self.mixed_ms if prefills and decodes else 0.0
        ms = (
            self.step_ms
            + mixed
            + self.us_per_token * tokens / 1e3
            + self.ns_per_pair * pairs / 1e6
            + self.ns_per_context * context / 1e6
        )
        return ms / 1e3


def finished(future: Any) -> bool:
    """Whether the step behind ``future`` is done, without waiting for it: the in-process executor's future completes
    only when read, so its output's copy-ready CUDA event is queried."""
    if future.done():
        return True
    output = getattr(future, "async_output", None)
    event = getattr(output, "async_copy_ready_event", None)
    return event is not None and event.query()


def install(core_cls: type, margin_s: float, cost: StepCost) -> None:
    """Make ``core_cls.step_with_batch_queue`` schedule just in time, with ``margin_s`` and ``cost``; installing again
    replaces the settings, not the wrapper."""
    original = getattr(core_cls.step_with_batch_queue, "flatshare_jit_original", None)
    original = original or core_cls.step_with_batch_queue

    def step_with_batch_queue(self: Any) -> Any:
        scheduler = self.scheduler
        if not hasattr(scheduler, "flatshare_output_t"):
            update = scheduler.update_from_output

            def stamped(*args: Any, **kwargs: Any) -> Any:
                scheduler.flatshare_output_t = time.monotonic()
                return update(*args, **kwargs)

            scheduler.flatshare_output_t = None
            scheduler.update_from_output = stamped
        batch_queue = self.batch_queue
        started = getattr(self, "flatshare_jit_started", None)
        if len(batch_queue) == 1 and started is not None and scheduler.has_requests():
            future, scheduler_output, _ = batch_queue[0]
            deadline = started + cost.seconds(scheduler_output) - margin_s
            while not finished(future):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    request = self.input_queue.get(timeout=min(remaining, POLL_S))
                except queue.Empty:
                    continue
                self._handle_client_request(*request)
        before = len(batch_queue)
        outputs = original(self)
        if len(batch_queue) == 1:
            if before == 0:
                # Launched with nothing in flight: it started at once.
                self.flatshare_jit_started = time.monotonic()
            elif before == 1:
                # Launched beside the one before, it started as that one's output reached the scheduler.
                self.flatshare_jit_started = scheduler.flatshare_output_t or time.monotonic()
        elif not batch_queue:
            self.flatshare_jit_started = None
        return outputs

    step_with_batch_queue.flatshare_jit_original = original
    step_with_batch_queue.flatshare_jit = (margin_s, cost)
    core_cls.step_with_batch_queue = step_with_batch_queue
