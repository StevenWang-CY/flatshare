# ADR-0002: One lifecycle-event schema for every workload

- **Status:** accepted, 2026-10-22; the lease and prefetch consequences are not pursued ([ADR-0006](0006-the-engine-change-carries-the-realtime-tier.md))
- **Deciders:** Chuyue Wang

## Context

Retention, prefetch and headroom all depend on one prediction: when will this session next need its KV, and how fast must it resume? Agents and voice give that signal on different clocks:
- an agent's response that ends in tool calls, or a server-side tool that starts and returns, on tool time (seconds to minutes);
- a caller who starts speaking, finishes a turn or barges in, on speech time (sub-second).

Separate mechanisms per workload would duplicate the prefetch and headroom logic and drift apart.

## Decision

- Define one `LifecycleEvent` type in `crates/flatshare-proto`. It is a flat JSON object whose `type` selects the variant and its fields (`turn_finished`, `tool_wait_started`, `user_speech_started`, `interrupted`, and the rest; see [PROTOCOL.md](../PROTOCOL.md)).
- Generate JSON Schemas from the Rust types for the Python components (`crates/flatshare-proto/schema/`). A test fails when the checked-in schemas fall behind the code.
- Events arrive at `POST /v1/events`, idempotent by `(source, event_id)`, or the control plane infers them from the traffic it proxies.
- Readers ignore fields they do not know, so newer emitters do not break older control planes. They reject unknown event types, so a typo cannot silently change behavior.

## Consequences

- A tool return and a speech onset drive the same lease transitions and the same prefetch.
- Adding a workload means emitting events, not adding a mechanism.
- One example file, `crates/flatshare-proto/testdata/events.jsonl`, pins the wire format. The Rust tests read it; the Python emitters are to test against it once they send events (W3).
- The schema is a public interface. Changes to it are additive, or they get a new event type.

## Later

Since 2026-10-23 the voice worker posts its events to `/v1/events`, and the gateway's lifecycle policy follows each call's turn by them. The worker's tests check its events against the generated `lifecycle_event.schema.json` rather than against the example file. Nothing sends tool-wait events, which the gateway accepts without effect ([PROTOCOL.md](../PROTOCOL.md#lifecycle-events)), and the shared lease and prefetch transitions are not pursued ([ADR-0006](0006-the-engine-change-carries-the-realtime-tier.md)).

Since 2026-10-24 (W5) events are idempotent by `(tenant, source, event_id)`, so that a trusted tenant cannot make another's events repeats, and the control plane infers no events: the `inferred` source is gone.

Since 2026-10-28, with `[[nodes]]`, a `session_closed` event also ends the session's placement and frees its call or agent slot ([ADR-0008](0008-two-cards-consolidation-with-a-warm-spare.md)), without an admission policy. In the arrangement to deploy it is the one event that acts ([ADR-0007](0007-the-gateway-is-the-serving-control-plane.md)).

Since v1.8 a tenant can end a session's placement with `DELETE /v1/sessions/{id}` too, and the voice worker does so at each call's end. In serve mode the worker posts `session_closed` alone, and the turns' events only in bench mode, for E2's lifecycle policy; the gateway keeps `session_closed` as a client's way to end a session.
