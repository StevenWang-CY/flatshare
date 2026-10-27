# The resource gate: does sharing make better use of the two cards than dedicated roles

Fixed before any two-card arrangement is measured. The box has two RTX 5060 Ti 16 GB cards on two hosts (Phastform
and Gray). The workload is the one E2 measures: two calls of the voice agent, beside coding agents replayed as
background work (until the useful-workload pilot, when the quality gate passes, replaces the replay with accepted
enrichment records). The question an operator asks: at the same foreground service, which arrangement of the two cards
does the most background work, and is sharing worth its complexity.

## The foreground requirement

Every arrangement serves both calls, and qualifies only if, over its runs pooled, at most 5% of the calls' replies take
longer than the SLO for their card's number of callers (voice alone on that card with as many callers, plus 250 ms,
measured in the same campaign). An arrangement that misses it does no qualifying background work, however much it
does.

## The background measure

The agents' tokens a second over the calls' first 240 s, summed over the arrangement's cards, the median over its
runs; beside it, the agents' completed requests and their time to first token at p50 and p99. The measure counts the
replay's work as E2 always has. Neither card is given work the arrangement would not run in production.

## The arrangements

- **Dedicated roles.** Card A runs the calls alone; card B runs the agents alone. Controlled: card B with the voice
  servers' memory left unused, the same KV pool and settings as a shared card. Best practical: card B without the
  voice servers, its KV pool grown into their memory, with the step cap and scheduling that give the agents the most.
- **Stock pooling.** Both cards run tuned stock vLLM with one call and half the agents each.
- **Flatshare pooling.** Both cards run the serving profile with one call and half the agents each.
- **Consolidation.** One card runs both calls and the most agents it can beside them while meeting the foreground
  requirement (the profile's envelope); the other card runs agents alone, as dedicated roles' card B does.

## The criterion

Sharing has resource value when a qualifying shared arrangement (Flatshare pooling or consolidation) does at least 20%
more background work than the best qualifying dedicated arrangement on the same two cards, measured on both
machines with their roles swapped between rounds. Flatshare's own value is the same comparison against stock pooling
and against consolidation on stock vLLM. A margin under 20% is reported as such, and a shared arrangement that does not
qualify has no resource value whatever its background measure. A warm spare improves recovery and is not counted as a
resource benefit here.

## How it is measured

Each card's part of an arrangement is measured as a point of its own on one host at a time: the cards share nothing
but the control plane on cyan, which E2 shows is not the bottleneck at these loads (the gateway's overhead p99 is under
a millisecond). Each arrangement's points run in three rounds on fresh samples of the agents, with the two hosts'
roles swapped from round to round and the order of the arrangements rotated, every attempt kept.
