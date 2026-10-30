# ADR-0008: Two cards: consolidation with a warm spare

- **Status:** accepted, 2026-10-28
- **Deciders:** Chuyue Wang

## Context

The box has two RTX 5060 Ti cards. A resource gate ([RESOURCE-GATE.md](../../benchmarks/quality/RESOURCE-GATE.md)) asked which arrangement of them does the most background work at the same foreground service, and whether sharing a card is worth its complexity. Three rounds measured each card's part of every arrangement, on fresh samples of the agents' traces, with the nodes swapped from round to round ([the rounds](../../benchmarks/results/2026-10-28-rounds/README.md)):
- **Consolidation qualifies.** One card ran both calls and four agent sessions on the change, with 5 of 177 replies over the SLO. Beside it, the other card ran agents alone. Together they did 34.2% more background work than the best dedicated arrangement: the calls on a card of their own, and the agents-only card at 14 agent sessions, 7,893 tokens a second.
- **Most of the gain is the arrangement's.** Consolidated on tuned vLLM, at its own envelope of two sessions, the same cards did 27.3% more than the best dedicated arrangement. The change's own gain over that is 5.4%.
- **Pooling qualifies on neither engine.** With one call and half the agents on each card, 24 of 87 replies on the change, and 53 of 87 on stock vLLM, were over the one-caller SLO.
- **A warm spare keeps most of it.** With the second card the change's engine, its speech servers resident so that it can take the calls, and its agents at 12 agent sessions, consolidation did 22.3% more than the best dedicated arrangement, which has no spare.

## Decision

- **The arrangement to deploy** is a `[[nodes]]` entry per card in `gateway.toml` ([the example](../../deploy/cyan/gateway.example.toml)):
  - both cards take both roles and two calls (`max_calls = 2`), so that either can carry every call the worker admits;
  - the first card's agents are bounded by its envelope (`max_agents = 4`), and the spare runs up to 12 agents' sessions;
  - `[placement] calls = "fill"` puts each call on the first card with a free call slot, so the second takes calls only while the first is full or not available.
- **The spare covers while it carries any call.** Its agents' requests are held to `failover_background_max_in_flight` (0) until those calls end, even after the first card is healthy again. A recovery therefore never leaves the calls beside the spare's whole agent load. The first card never covers for its own calls, and only `max_agents` bounds its agents.
- **Sessions stick to their card**, since KV never moves between engines.
  - A session whose card is not available is placed again by its next request.
  - A realtime request whose card fails before its first content is sent once more on the other card, within the deadlines of its receipt.
  - A request whose first content has reached the client is never sent again, so a voice reply is never a second generation after half-delivered audio.
- **A card's health is its whole pipeline**: its engine serving (its own canary) and ready, and its STT and TTS answering `/health`.

## Consequences

- **Failover is drilled.** With the engine of the first card killed, both calls' next turns went to the spare 0.7 s after the kill; with it stopped, 1.9 s after the stop. No turn failed after the move ([the drills](../../benchmarks/results/2026-10-28-node-failover/README.md)).
- **The arrangement needs a bound that holds.** Without `max_agents` the first card runs the measured failure: beside 24 agent sessions, 37 of 177 replies were over the SLO. The gateway now refuses a node with both roles and no bound, and each release holds the running gateway's nodes to the serving profile's envelope.
- **Gray is shared**, and serves in booked windows: its own servers are started for a window and stopped after ([the runbook](../../ops/RUNBOOK.md#two-gpu-nodes)).
- **The gateway places sessions but does not move KV.** A moved session pays its prefill again on the other card.
