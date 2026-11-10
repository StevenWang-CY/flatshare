# flatshare-core

The models and policies that the simulator and the gateway share, for a developer who changes either. It is pure,
deterministic code with no I/O and no clock. Callers pass token counts and events, and get durations and decisions
back. `flatshare-sim` and the gateway's throttle use its cost, step and KV-geometry models, and only the gateway runs
its admission policies, as E2's harness ([ARCHITECTURE.md §6.1](../../docs/ARCHITECTURE.md#61-e2s-admission-policies)).

| Module | What it holds | Used by |
|---|---|---|
| `cost` | What serving a prompt costs from each tier, the E0 model fitted on the frozen box ([BENCHMARKS.md](../../docs/BENCHMARKS.md#what-does-resuming-an-agent-turn-cost)), and decode time | `flatshare-sim` |
| `step` | What one engine step costs from what it carries (note 1) | `flatshare-sim`'s contention model; `admission`'s prefill price |
| `admission` | Whether a request goes upstream now, waits or is refused (note 2) | `flatshare-gateway`'s admission point |
| `geometry` | KV bytes per token, the engine block and tier capacities in blocks | `flatshare-sim`; the gateway's default pool (`FROZEN_BOX_POOL_TOKENS`) |

1. A step carries its step tokens, attention pairs and context, and a prefill beside decodes. `flatshare-sim`'s
   `scripts/fit_step_cost.py` fits the model on three operating points of the frozen box.
2. It holds E2's baselines (static cap, token reservation, flow control, preempt-resubmit) and the lifecycle policy,
   driven by arrivals, first bytes, departures and the realtime sessions' turns, and the token estimate they share.

## The lifecycle policy

`Policy::Lifecycle`, the gateway's `policy = "lifecycle"`, is a variant of `admission::Policy` beside the baselines,
pure and without a clock like them, and E2's arm 9 runs it with make-room. Realtime requests go upstream at once, never
queued and never refused ([ADR-0003](../../docs/adr/0003-realtime-is-never-shed.md)). Background requests yield to the
realtime sessions' turns, which the caller tracks from lifecycle events and reports as two counts (`Turns`). These are
the sessions whose user is speaking, and those in their critical window, from the end of the user's turn to the
reply's first audio. Each mechanism can be switched off for the ablations:

- **Headroom.** Its `Trigger` arms a session from the user's speech onset, or only in its critical window. While
  sessions are armed, a background request goes upstream only if the estimated tokens of the background requests in
  flight and its own leave `headroom_tokens` of the pool free for each armed session.
- **Throttle.** The background requests admitted in a critical window share a budget of predicted prefill time. Each is
  priced by the `PrefillCost` the admission point is made with, a step model and the engine's step budget, and
  `Admission::new` refuses the throttle without one. On the frozen box's fit, 512 tokens after a 30,000-token context
  take 269 ms, and after none 38 ms. The policy carries no fit of its own, so a different engine or GPU needs one from
  `scripts/fit_step_cost.py`.
- **Degrade.** A realtime request that arrives while the background requests in flight leave less than one turn's
  headroom free has its reply capped.
- **Aging.** A request the caller has let wait long enough is admitted regardless (`Admission::age`), and is `oversize`
  if its estimate alone is over what the headroom leaves.

`Admission::held_back` gives the estimated background tokens the headroom holds back, which the gateway integrates over
time to price false alarms. [The gateway's README](../flatshare-gateway/README.md#the-lifecycle-policy) gives the
policy's configuration, and [ARCHITECTURE.md §7](../../docs/ARCHITECTURE.md#7-designs-measured-and-dropped) what E2
measured of it.

## What every policy keeps

`Elsewhere` is what another admission point holds on the same engine. In the gateway it is a reload's predecessor while
it drains, which the successor's point counts wherever its policy counts requests in flight
(`Admission::set_elsewhere`). `Admission::held` gives a point's own, with what it counts, for its own successor.
`Admission::close` refuses every waiting request and admits nothing more, as a predecessor hands its sockets over, and
`Admission::with_max_waiting` bounds the queues together.

## Tests

`cargo test -p flatshare-core` runs the unit tests beside each module: every admission policy's order, holds and
refusals, the lifecycle policy's mechanisms alone and together, and a point closed or holding requests elsewhere. They
also check the cost and step models against E0's measured cells and fits, and the frozen box's geometry, 2,275 GPU and
9,102 CPU blocks.

The `estimate_error` example measures the admission token estimate's error against the prompt tokens an engine reported
in an AIPerf replay ([the gateway's figures](../flatshare-gateway/README.md#admission)):

```sh
cargo run --release -p flatshare-core --example estimate_error -- RUN/aiperf/profile_export.jsonl...
```
