# ADR-0006: The engine change carries the realtime tier; the control plane is a harness

- **Status:** accepted, 2026-10-24; superseded in part by [ADR-0007](0007-the-gateway-is-the-serving-control-plane.md): the gateway is the serving control plane, and only its admission policies are E2's harness
- **Deciders:** Chuyue Wang
- **Supersedes in part:** [ADR-0001](0001-control-plane-in-front-of-stock-vllm.md) (the engine stays stock, and the control plane compensates with admission and headroom), [ADR-0003](0003-realtime-is-never-shed.md) (room made by the control plane), and the lease and prefetch consequences of [ADR-0002](0002-one-lifecycle-event-schema.md) and [ADR-0004](0004-native-cpu-offload-tier.md)

## Context

E2 measured two calls beside 24 coding-agent sessions whose contexts fill the GPU's KV pool, on one RTX 5060 Ti ([BENCHMARKS.md](../BENCHMARKS.md#voice-under-agent-load-e2)):
- **Nothing deployable today fixes it.** Tuned vLLM (the frozen box) leaves the voice tail at 10.1 s at p95. Every gateway baseline and both engines' priority preemption leave it at 8.8–12.1 s. Aborting background work before its first byte and resubmitting it reaches 2.6 s, at the cost of the agents' tail.
- **The engine change does.** With the two PRs and a 48-token window the tail is 2.19 s, against an SLO of 2.02 s, at 94% of stock's background throughput. Flatshare's arm 9 reaches 2.06 s with every reply capped at 16 tokens.
- **The control plane could not make room.** Flatshare's admission policy holds KV headroom from speech onset and throttles background prefill while a turn is answered. In front of stock vLLM it left the tail where stock does (10,993 ms). Beside the engine change its mechanisms could not act. Running agent requests held the pool, and a control plane can only defer new ones, so the headroom was never there: degrade capped 60 of 60 replies in every run. The throttle, checked after the headroom, deferred nothing.
- **E1 measured the agent side.** In the simulator, even an oracle that knows when each agent returns closes under 9% of resume TTFT at p50; at p90 it closes 28% at low pressure, but session history, the information a deployment has, closes nothing ([EXPERIMENTS.md](../EXPERIMENTS.md#e1-parking)).

## Decision

- **The engine carries the realtime tier.** It runs priority scheduling with two changes:
  - [#63350](https://github.com/vllm-project/vllm/pull/63350): a waiting request of a higher priority preempts running lower-priority work for KV, at most two requests a step;
  - [#63351](https://github.com/vllm-project/vllm/pull/63351): lower-priority prefill sits out the request's first tokens, with a window long enough for the reply's first sentence (48 tokens for this voice agent).

  There is no reserve. The PRs are proposed to vLLM and under review; the `make-room` variant runs the same change on vLLM 0.32.0 meanwhile.
- **The priority comes from the caller's identity.** A gateway maps each tenant to a class and caps it, so a client cannot raise its own priority.
- **No admission policy in front.** `flatshare-gateway` stays the experiment harness: its baselines and the lifecycle policy remain E2's arms, and the measured configuration ran without either.
- **Not pursued:**
  - leases, prefetch, the per-node sidecar, cache contracts and the recompute ledger;
  - they would be reopened by a source of tool-return times ahead of the return, which the oracle's tail gain needs, or by an engine whose running KV a control plane can reclaim.

## Consequences

- **What deploys is a set of vLLM flags** ([PRODUCTION.md](../PRODUCTION.md)). The change is proposed to vLLM rather than kept as a fork; while it is under review, the variant pins vLLM 0.32.0, and its build refuses any other version.
- **The other ADRs stand with these changes:**
  - ADR-0001's design describes the experiment harness, not the deployment;
  - ADR-0003's decision stands, with the room made in the engine;
  - a window that covers the reply's first sentence makes degrade unnecessary.
- **The v0 lease and ledger code is removed after v0.3:** `flatshare-proto`'s lease hint and ledger rows, the header that marked synthetic requests, and `flatshare_vllm`'s `LeasePolicy` and `LeaseConnector`. Done on 2026-10-24; the v0.3 tree keeps them.
- **The change's known gaps** ([ARCHITECTURE.md §2](../ARCHITECTURE.md#2-the-change-63350-and-63351)) were open in the v0.3 variant and in both PRs as first opened, and E2 measured the change with them. W5 fixed them, with the further defects its review found, in the variant (main, up to b656c25); the PRs carry the same fixes.

## Later

- **Aborting work makes room** (2026-10-25). The Context's "a control plane can only defer new ones" is too strong: a control plane can make room only by aborting background work, as E2's preempt-resubmit baseline does at the agents' tail. Only the engine can take KV from a request already decoding, so the decision stands.
- **W5's re-measure** (2026-10-25). W5 fixed the change's gaps and re-measured it in rounds, each on its own sample of the agents' traces: over fifteen runs in three campaigns, the PRs with a 48-token window reach a p95 of 2.24 s at the median (2.13–2.40 s) against W5's SLO of 2.03 s, and the frozen box 10.5 s over ten runs (8.7–12.1 s) ([BENCHMARKS.md](../BENCHMARKS.md#the-discriminating-scenario-after-w5s-fixes-w5)). The Context keeps the numbers the decision was taken on.
- **Another GPU** (2026-10-26). On an A100 SXM4 80 GB at the box's pool, the change took the p95 from 4.99 s to 1.63 s, under that card's SLO in each of five rounds, and the agents' throughput held ([HARDWARE.md](../HARDWARE.md#what-it-measured-2026-10-26)).
- **The gateway serves** (2026-10-27). The production program built bounds, deadlines, failure-inclusive accounting and, on 2026-10-28, two-node placement into the gateway, which the deployment depends on. "Stays the experiment harness" holds only for its admission policies ([ADR-0007](0007-the-gateway-is-the-serving-control-plane.md)). The decision on the engine and the priorities stands.
