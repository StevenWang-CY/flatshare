# ADR-0003: Realtime is never shed

- **Status:** accepted, 2026-10-22; the room is made in the engine ([ADR-0006](0006-the-engine-change-carries-the-realtime-tier.md)), and the serving gateway bounds every class's requests, realtime among them ([ADR-0007](0007-the-gateway-is-the-serving-control-plane.md))
- **Deciders:** Chuyue Wang

## Context

A voice caller expects an answer about a second after they stop talking. Rejecting a voice turn with a 429 leaves the caller without an answer mid-conversation. Load-shedding routers (llm-d's flow control, for example) treat every class the same way at the limit.

On this hardware, the threat to voice is not request volume but background work holding the GPU:
- one server-side voice turn takes 415 ms idle and 876 ms median while background agents run 8k-token prefills, and LLM time to first token rises from 39 to 371 ms (the 2026-10-22 probe; [BENCHMARKS.md](../BENCHMARKS.md#voice-on-the-shared-gpu) has the frozen box's W1 values);
- vLLM's priority cannot preempt running requests to admit a waiting one (ADR-0001).

## Decision

Admission for the `realtime` class never rejects. Under pressure it does two things, in order:
1. **Make room.**
   - Cap background prefill admission at a small token budget.
   - Keep free KV at least equal to the realtime working set, by deferring new background turns and letting finished ones park in the CPU tier.
   - Prefetch the speaking session's KV while the user is still talking.
2. **Degrade the reply.** Use a shorter `max_tokens` or a filler phrase, so the reply still arrives.

Background traffic absorbs the overload. It keeps a minimum share and ages, so it is never starved.

## Consequences

- Capacity for realtime must be planned: a bounded number of concurrent calls per GPU, set from the voice SLO calibration.
- The agents' token rate becomes the measure of how well the GPU is used. The E2 frontier plots it against voice V2V p95.
- Rejections exist only for `interactive` (a deadline can be missed) and for `background` (deferral beyond its age bound). Both are reported as their own metrics.
- The decision binds Flatshare's own admission policy. E2's baseline policies emulate what others ship (W2): llm-d's flow control, for one, can make a realtime request wait or refuse it when realtime and interactive requests fill its ceiling, and those outcomes are measured, not prevented.

## Later

Under the lifecycle policy a realtime request is never queued or refused. Its steps to make room in the control plane (the prefill budget and the KV headroom) could not act in E2's discriminating scenario, where running agent requests held the pool; the room is made in the engine ([ADR-0006](0006-the-engine-change-carries-the-realtime-tier.md)). Degrade is built as a 16-token `max_tokens`, and a filler phrase and prefetch are not; a 48-token first-tokens window fits full replies instead, all but one of W5's 5,571, at a slightly higher p95 ([BENCHMARKS.md](../BENCHMARKS.md#voice-under-agent-load-e2)), and is what [PRODUCTION.md](../PRODUCTION.md) recommends.

Since 2026-10-27 the gateway bounds a realtime request as it does any other, with or without a policy. It refuses one beyond 128 realtime requests in flight, ends one that gets no first content within 60 s, and with `[[nodes]]` refuses a call that finds no free call slot (503 `no_call_slot`). These bounds are a limit on the gateway's own safety. The calls themselves are bounded where they arrive, by the voice worker's admission ([ADR-0007](0007-the-gateway-is-the-serving-control-plane.md)). The decision binds the admission policies, and none of them queues or refuses a realtime request.

As built, background traffic keeps no minimum share: the lifecycle policy ages a deferred background request (it goes upstream after `max_defer_s`, 10 s) and refuses a request only when its class's queue is full (429). It never holds interactive requests, and it has no deadline for them. In the engine, the budget cap holds lower-priority prefill while a higher-priority request's first tokens decode, so overlapping realtime replies can stop the background tier entirely. W5 bounds only the budget cap's hold for a KV load that never completes; its hold for overlapping replies is the capacity limit [PRODUCTION.md](../PRODUCTION.md#how-it-fails) states.
