# ADR-0007: The gateway is the serving control plane

- **Status:** accepted, 2026-10-29. It records what v1.6 built on 2026-10-27 and v1.7 on 2026-10-28.
- **Deciders:** Chuyue Wang
- **Supersedes in part:**
  - [ADR-0006](0006-the-engine-change-carries-the-realtime-tier.md): "the control plane is a harness";
  - [ADR-0001](0001-control-plane-in-front-of-stock-vllm.md)'s status: "this control plane is the experiment harness";
  - [ADR-0003](0003-realtime-is-never-shed.md), at the request level.

## Context

- **ADR-0006 left the gateway E2's harness.** It put the realtime tier in the engine, and in the configuration E2
  measured, the gateway only set each request's priority from its class.
- **An external audit (2026-10-27) found that gateway not safe to deploy**, and each finding held when checked in the
  code:
  - a chat body could bypass the realtime reply cap;
  - the whole body was read before anything bounded the requests in flight;
  - nothing ended a stream whose engine stopped emitting;
  - the service level improved when requests failed, since it counted only first bytes;
  - a deploy wrote into the live tree, with no manifest and no rollback.
- **The production program built the missing parts into the gateway.**
  - v1.6: bounds on what a request asks for and on the work held, deadlines, failure-inclusive accounting and an
    inference canary.
  - v1.7: placement on two GPU nodes with failover ([ADR-0008](0008-two-cards-consolidation-with-a-warm-spare.md)).
  - A deployment now depends on all of it: the calls and agents it accepts, its refusals and its recovery are the
    gateway's.

## Decision

- **`flatshare-gateway` is the serving control plane.** In front of the engines it:
  - gives each request a class and its priority from the caller's identity, capped per tenant (unchanged from
    ADR-0006);
  - bounds what a request asks for (a reply cap per class, at most `max_choices` choices), and the work it holds (the
    requests in flight in all, by class and by tenant, and the bodies they buffer). Beyond a bound it refuses at once,
    with `Retry-After`;
  - ends every wait a request goes through by its class's deadline (upload, queue, response headers, first content,
    each frame after, and the whole request), and cancels the engine's work;
  - accounts every inference request, failures included: offered, admitted, one outcome each, and a service level of
    good over valid requests;
  - judges the engine by inference (an idle-only canary, answered by `/ready`);
  - with `[[nodes]]`, places calls and agents' sessions on the GPU nodes and fails over between them (ADR-0008).

  The gateway's README says how ([What it does](../../crates/flatshare-gateway/README.md#what-it-does)).
- **No admission policy serves.** The gateway refuses an `[admission]` policy beside `[[nodes]]`, and the arrangement
  to deploy runs none with one upstream either. The change makes the room ([ADR-0006](0006-the-engine-change-carries-the-realtime-tier.md)).
- **E2's admission policies stay in the same binary, as its harness** (`src/harness/`). The proxy reaches them only
  through a request's admission ticket. A gateway without `[admission]` runs none of their code and exports none of
  their metric families, and the reproduction of E2's records needs them.

## Consequences

- **Realtime requests are bounded per request**, which ADR-0003 did not foresee. By default the bounds are:
  - 128 realtime requests in flight;
  - a 60 s deadline for the first content and 300 s for the whole reply;
  - with `[[nodes]]`, a call without a free slot gets 503 `no_call_slot`.

  These bounds limit the gateway's own safety, and they are set well above what the voice worker admits: it admits
  calls by the serving profile's count ([ADR-0009](0009-immutable-releases-and-the-exact-serving-profile.md)). ADR-0003
  still binds the admission policies: none of them queues or refuses a realtime request.
- **The gateway's cost is the service's.** Its overhead has an objective, p99 at most 2 ms a request
  ([SLO.md](../SLO.md#indicators)), which its README measures.
- **The harness is split by configuration, not by build.** Moving E2's policies into a second binary was considered
  and declined (RV3, 2026-10-29). It would add a build and ship path, and E2 re-runs on the GPUs, for readability
  alone, while the configuration already keeps the policies out of the arrangement to deploy.
