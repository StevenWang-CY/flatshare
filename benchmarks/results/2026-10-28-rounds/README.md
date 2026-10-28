# The two-card comparison and the sweep's confirmation (2026-10-28)

[RESOURCE-GATE.md](../../quality/RESOURCE-GATE.md) asks, before any two-card arrangement was measured, which
arrangement of the box's two RTX 5060 Ti cards does the most background work at the same foreground service, and
whether sharing a card is worth its complexity. The sweep's screens ([2026-10-27-sweep](../2026-10-27-sweep)) picked
the change at a step cap of 256 and found no setting that kept 24 session trees within the foreground requirement. These
records answer both: three rounds of each card's part of every arrangement (`rg-r1` to `rg-r3`), and two more rounds
of the sweep's configurations at 24 session trees (`rg-r4`, `rg-r5`), which with the first three confirm the step cap
on five fresh samples of the agents.

## How they ran

- **The rounds** ([`benchmarks/e2`](../../e2): `rg-r1.campaign` to `rg-r5.campaign`), each on a fresh sample of the
  agents' traces (seeds 21 to 25), each on one GPU node, the nodes swapped from round to round: Gray in rounds 2 and 4;
  Phastform in rounds 1, 3 and 5. Each round runs its engine groups in an order rotated from the round before. Every
  point is E2's: two callers (one, for a pooled card), 25 turns each after a warm-up turn, beside the session trees of
  the discriminating scenario (`weka-x32-long`), on vLLM 0.32.0 with Qwen3-4B-Instruct-2507-FP8. Every host ran the
  release of v1.6-27-gea7efd0 (`versions` in each `run.json`).
- **The cards.** Each card's part of an arrangement is a point of its own:
  - voice alone with two callers (`w5-a1-voice`) and with one (`rg-voice-c1`), the dedicated arrangement's voice card
    and each round's SLO (its p95 plus 250 ms);
  - the agents alone, on the frozen box with its speech servers idle (`rg-disc-agents-frozen-s24` and `-s12`, the
    controlled dedicated card) and on the agents-only card (`rg-disc-agents-only-s24`, `-s20`, `-s16`, `-s14` and
    `-s12`: no speech servers, a pool of 61,152 tokens in their memory, FCFS at a step cap of 2,048, the best practical
    dedicated card and consolidation's second card). Each arrangement runs a dedicated card at the number of session
    trees at which it does the most, the highest median over the rounds: how many agents a card runs at once is part of
    the scheduling that gives them the most. The counts below 24 were added with the envelope's, once round 1 had shown
    the agents-only card at 24 doing 4,232 tokens/s over the calls' first 240 s, with a median time to first token of
    48 s, where its pool thrashes, and a pooled card on the frozen box, one call beside 12 session trees, doing 6,617;
  - one caller beside 12 session trees, a pooled card: on the frozen box (`rg-disc-frozen-c1-s12`, stock pooling) and
    on the change at a step cap of 256 (`rg-disc-pr-ab48-b256-c1-s12`, Flatshare pooling);
  - two callers beside 24, 12, 8, 4 and 2 session trees, a consolidated card, whose envelope is the most of these
    at which it qualifies: on the change at 256 (`sweep-disc-pr-ab48-b256-t0-s24`, `rg-disc-pr-ab48-b256-s12`,
    `-s8`, `-s4`, `-s2`), on the frozen box (`w5-disc-frozen-s24`, `rg-disc-frozen-s12`, `-s8`, `-s4`, `-s2`), and
    at 24 on W5's 48-token configuration (`w5-disc-pr-ab48-s24`). The points at 8, 4 and 2 were added once round
    1's first eleven had put 7 of 59 replies over the SLO on the change's card at 12 and 20 of 59 on stock vLLM's.
    Round 1 ran them after its first eleven points, in two more sittings: from 97c48b4 (the envelope and the
    dedicated card at 20, 16 and 12) and from 91de180 (the agents-only card at 14, once 12 had been its busiest short
    of saturation, and the spare below), each adding only arm files to the harness its first sitting ran
    (ea7efd0); rounds 2 and 3 ran every point inside their engine groups.
  - beside the gate, a warm spare's card: the agents alone on the change at 256 with its speech servers idle
    (`rg-disc-agents-pr-ab48-b256-s16` and `-s12`), the second card when it must be able to take the calls.
- **The records** are kept as W5's are (`run.json`, `arm.toml`, `engine.json`, `summary.json`, the callers'
  `bots/summary.json` and `bots/turns.jsonl`), with each campaign's `campaign.log` and report, home directories
  written as `~`. A point that did not complete is kept beside the one that did (`run.json` names its status): the
  one here is round 1's first voice alone (`rg-r1/w5-a1-voice.incomplete-20261028T020909Z`), stopped at its first
  turns on v1.6-21-ga4e6676, whose callers waited out every reply to 60 s, misreading the agent's state; the release
  the rounds ran fixed it.

[`gate.py`](gate.py) gives the gate's verdict and the sweep's table from these records
(`cd python/flatshare_bench && uv run python ../../benchmarks/results/2026-10-28-rounds/gate.py`), and
`python/flatshare_bench/tests/test_rounds_record.py` fails while this README's tables are not what it gives.

## The two-card comparison

<!-- gate.py -->
Each round's SLO, ms (voice alone's p95 plus 250):

| Round | Node | Two callers | One caller |
|---|---|---:|---:|
| rg-r1 | phastform | 2,000.4 | 1,988.3 |
| rg-r2 | gray | 1,972.0 | 1,943.4 |
| rg-r3 | phastform | 2,010.0 | 1,981.7 |

The agents alone on a dedicated card, by session trees; each arrangement runs the card at its busiest:

| Card | Session trees | Agents, tokens/s (240 s): median | Rounds | Agents' TTFT p50/p99, s |
|---|---:|---:|---|---:|
| controlled: rg-disc-agents-frozen-s24 | 24 | 3,451 | 3,451, 3,499, 2,960 | 66.6/112.1 |
| controlled: rg-disc-agents-frozen-s12 | 12 (busiest) | 6,302 | 7,008, 6,302, 6,041 | 4.4/31.0 |
| best practical: rg-disc-agents-only-s24 | 24 | 4,232 | 4,232, 4,294, 3,750 | 47.9/86.6 |
| best practical: rg-disc-agents-only-s20 | 20 | 4,203 | 4,203, 4,780, 3,733 | 42.1/71.8 |
| best practical: rg-disc-agents-only-s16 | 16 | 5,624 | 5,624, 6,489, 3,829 | 5.9/59.1 |
| best practical: rg-disc-agents-only-s14 | 14 (busiest) | 7,893 | 8,282, 7,893, 4,710 | 5.1/41.0 |
| best practical: rg-disc-agents-only-s12 | 12 | 7,213 | 7,213, 7,372, 6,909 | 1.1/18.4 |

| Arrangement | Cards | Replies over the SLO | Qualifies | Agents, tokens/s (240 s): median | Rounds | Agents' TTFT p50/p99, s |
|---|---|---:|---|---:|---|---:|
| Dedicated roles, controlled, 12 session trees | w5-a1-voice + rg-disc-agents-frozen-s12 | 0/177 (0.0%) | yes | 6,302 | 7,008, 6,302, 6,041 | 4.4/31.0 |
| Dedicated roles, best practical, 14 session trees | w5-a1-voice + rg-disc-agents-only-s14 | 0/177 (0.0%) | yes | 7,893 | 8,282, 7,893, 4,710 | 5.1/41.0 |
| Stock pooling | 2 x rg-disc-frozen-c1-s12 | 53/87 (60.9%) | no | 13,188 | 13,233, 13,188, 11,951 | 4.7/29.2 |
| Flatshare pooling | 2 x rg-disc-pr-ab48-b256-c1-s12 | 24/87 (27.6%) | no | 12,111 | 13,138, 12,111, 11,374 | 4.8/37.2 |
| Consolidation, the change at a step cap of 256, 24 session trees | sweep-disc-pr-ab48-b256-t0-s24 + rg-disc-agents-only-s14 | 37/177 (20.9%) | no | 11,075 | 11,466, 11,075, 7,435 | 48.0/92.5 |
| Consolidation, the change at a step cap of 256, 12 session trees | rg-disc-pr-ab48-b256-s12 + rg-disc-agents-only-s14 | 33/177 (18.6%) | no | 13,880 | 15,107, 13,880, 10,082 | 5.1/41.6 |
| Consolidation, the change at a step cap of 256, 8 session trees | rg-disc-pr-ab48-b256-s8 + rg-disc-agents-only-s14 | 31/177 (17.5%) | no | 13,537 | 13,537, 14,352, 10,392 | 0.7/11.7 |
| Consolidation, the change at a step cap of 256, 4 session trees | rg-disc-pr-ab48-b256-s4 + rg-disc-agents-only-s14 | 5/177 (2.8%) | yes | 10,596 | 11,655, 10,596, 8,308 | 0.6/10.8 |
| Consolidation, the change at a step cap of 256, 2 session trees | rg-disc-pr-ab48-b256-s2 + rg-disc-agents-only-s14 | 2/177 (1.1%) | yes | 10,050 | 10,982, 10,050, 6,213 | 0.4/9.8 |
| Consolidation, stock vLLM, 24 session trees | w5-disc-frozen-s24 + rg-disc-agents-only-s14 | 163/177 (92.1%) | no | 11,378 | 11,681, 11,378, 7,776 | 39.7/85.8 |
| Consolidation, stock vLLM, 12 session trees | rg-disc-frozen-s12 + rg-disc-agents-only-s14 | 95/177 (53.7%) | no | 14,191 | 14,987, 14,191, 11,055 | 4.6/33.5 |
| Consolidation, stock vLLM, 8 session trees | rg-disc-frozen-s8 + rg-disc-agents-only-s14 | 69/177 (39.0%) | no | 13,573 | 13,573, 14,267, 10,547 | 0.7/11.6 |
| Consolidation, stock vLLM, 4 session trees | rg-disc-frozen-s4 + rg-disc-agents-only-s14 | 18/177 (10.2%) | no | 10,596 | 11,655, 10,596, 8,308 | 0.6/10.4 |
| Consolidation, stock vLLM, 2 session trees | rg-disc-frozen-s2 + rg-disc-agents-only-s14 | 4/177 (2.3%) | yes | 10,051 | 10,984, 10,051, 6,234 | 0.4/7.8 |
| Consolidation, the change at W5's configuration, 24 session trees | w5-disc-pr-ab48-s24 + rg-disc-agents-only-s14 | 70/177 (39.5%) | no | 11,196 | 11,596, 11,196, 7,553 | 44.4/88.0 |

The best qualifying dedicated arrangement: Dedicated roles, best practical, 14 session trees, 7,893 tokens/s.

Sharing's resource value:
- Flatshare pooling: does not qualify
- Consolidation on the change, at its envelope (Consolidation, the change at a step cap of 256, 4 session trees): +34.2% against Dedicated roles, best practical, 14 session trees (7,893): meets the 20% hurdle
- Beside the criterion, whose consolidated card runs the serving profile: consolidation on stock vLLM, at its envelope (Consolidation, stock vLLM, 2 session trees): +27.3% against Dedicated roles, best practical, 14 session trees (7,893): meets the 20% hurdle

Flatshare's own value:
- Flatshare pooling against stock pooling: does not qualify
- Consolidation on the change against consolidation on stock vLLM (Consolidation, stock vLLM, 2 session trees): +5.4% against Consolidation, stock vLLM, 2 session trees (10,051): does not meet the 20% hurdle

Beside the gate, which counts no spare as a resource benefit: the second card as a warm spare, on the serving card's engine with its speech servers idle, so that it can take the calls (busiest at 12 session trees, rg-disc-agents-pr-ab48-b256-s12):
- Dedicated roles with a warm spare (w5-a1-voice + rg-disc-agents-pr-ab48-b256-s12): 0/177 replies over the SLO, 6,715 tokens/s (7,189, 6,715, 6,054), -14.9% against Dedicated roles, best practical, 14 session trees, agents' TTFT p50/p99 3.7/30.1 s
- Consolidation on the change at 4 session trees, with a warm spare (rg-disc-pr-ab48-b256-s4 + rg-disc-agents-pr-ab48-b256-s12): 5/177 replies over the SLO, 9,652 tokens/s (10,563, 9,418, 9,652), +22.3% against Dedicated roles, best practical, 14 session trees, agents' TTFT p50/p99 0.5/10.7 s
<!-- /gate.py -->

The gate's verdict:
- **Sharing has resource value, by consolidation.** One card running both calls and four session trees on the change,
  beside the agents-only card, did 34.2% more background work than the best dedicated arrangement, the calls on a card
  of their own and the agents-only card at 14 session trees: past the 20% hurdle. Four trees is the change's envelope:
  at eight, 31 of its 177 replies were over the SLO, at four 5 (2.8%). Most of that gain is the arrangement's, not the
  change's: consolidated on stock vLLM, at its own envelope of two trees, the same cards did 27.3% more than the best
  dedicated arrangement, which the criterion does not count, since its consolidated card runs the serving profile.
- **Pooling qualifies on neither engine.** With one call beside 12 session trees, 24 of the pooled cards' 87 replies
  on the change and 53 on stock vLLM were over the one-caller SLO.
- **Flatshare's own value does not meet the hurdle.** Stock vLLM's consolidated card qualifies at two session trees,
  the change's at four, and the agents-only card's work dominates both sums: +5.4%.
- **The dedicated card's best number of agents is not all of them.** The agents-only card did the most at 14 session
  trees, not 24, where its pool thrashes and its agents' first tokens wait 48 s at the median: measured only at 24, the
  dedicated baseline would have been 46% below the card's best (4,232 tokens a second against 7,893). The busiest count
  also moves with the sample: 14 in two rounds, 12 in the third, whose trees made the card thrash already at 16.
- **A warm spare keeps most of it.** The gate counts no spare as a resource benefit; beside it, consolidation at four
  trees with the second card a warm spare (the change's engine with its speech servers resident, so that it can take
  the calls, its agents at 12 session trees) did 22.3% more than the best dedicated arrangement, which has no spare.

## Beside the gate: the margin

<!-- gate.py margins -->
| Session trees | the change at a step cap of 256, +250 ms | the change at a step cap of 256, +500 ms | stock vLLM, +250 ms | stock vLLM, +500 ms |
|---:|---:|---:|---:|---:|
| 24 | 37/177 (20.9%) | 8/177 (4.5%) | 163/177 (92.1%) | 155/177 (87.6%) |
| 12 | 33/177 (18.6%) | 6/177 (3.4%) | 95/177 (53.7%) | 72/177 (40.7%) |
| 8 | 31/177 (17.5%) | 5/177 (2.8%) | 69/177 (39.0%) | 41/177 (23.2%) |
| 4 | 5/177 (2.8%) | 0/177 (0.0%) | 18/177 (10.2%) | 12/177 (6.8%) |
| 2 | 2/177 (1.1%) | 1/177 (0.6%) | 4/177 (2.3%) | 1/177 (0.6%) |

The envelope, the most session trees whose rounds pooled qualify: the change at a step cap of 256, +250 ms: 4 session trees; the change at a step cap of 256, +500 ms: 24 session trees; stock vLLM, +250 ms: 2 session trees; stock vLLM, +500 ms: 2 session trees.
<!-- /gate.py margins -->

The verdict rests on the margin the SLO fixed, 250 ms over voice alone's p95. At twice that, the change's
consolidated card would qualify at every count measured up to 24 session trees (8 of 177 replies over at 24) and stock
vLLM's still only at two (12 of 177 at four). The change keeps the tail within about half a second of voice alone at
any of these loads; what it cannot keep is the 250 ms.

## Beside the gate: what the shared card's agents measure

<!-- gate.py pace -->
| Session trees | the change at a step cap of 256: requests by round | stock vLLM: requests by round |
|---:|---:|---:|
| 24 | 67, 57, 46 | 71, 65, 52 |
| 12 | 163, 104, 95 | 166, 107, 107 |
| 8 | 114, 98, 93 | 116, 97, 95 |
| 4 | 44, 41, 37 | 44, 41, 37 |
| 2 | 108, 43, 38 | 108, 43, 39 |
<!-- /gate.py pace -->

The agents' requests that completed in the calls' first 240 s, by round. At two and four session trees both engines
completed the same requests in every round, so there the shared card's part of the background measure is the trees' own
pace, which the traces' think times set, not what the card could do: the change's 5.4% over stock vLLM is the two trees
more at which it qualifies. From 12 trees on the counts part, and at 24 stock vLLM completed 71, 65 and 52 against the
change's 67, 57 and 46.

## Beside the gate: where the replies spend their time

<!-- gate.py stages -->
| Card | Transcript | LLM first token | LLM first token, p95 | First sentence | First sentence, p95 | TTS first audio |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 745 | 79 | 97 | 290 | 437 | 197 |
| the change at a step cap of 256, 2 trees | 756 | 81 | 129 | 306 | 467 | 190 |
| the change at a step cap of 256, 4 trees | 749 | 91 | 242 | 312 | 534 | 195 |
| the change at a step cap of 256, 8 trees | 800 | 145 | 442 | 411 | 737 | 203 |
| the change at a step cap of 256, 12 trees | 808 | 223 | 452 | 476 | 765 | 203 |
| the change at a step cap of 256, 24 trees | 811 | 215 | 333 | 464 | 768 | 215 |
| stock vLLM, 2 trees | 757 | 84 | 125 | 308 | 472 | 192 |
| stock vLLM, 4 trees | 746 | 92 | 414 | 294 | 909 | 194 |
| stock vLLM, 8 trees | 796 | 189 | 808 | 521 | 2,424 | 201 |
| stock vLLM, 12 trees | 811 | 284 | 2,678 | 675 | 3,150 | 202 |
| stock vLLM, 24 trees | 816 | 2,929 | 8,493 | 3,792 | 9,719 | 210 |
<!-- /gate.py stages -->

The voice worker's own timings say where the margin goes. From eight session trees on, the transcript comes 51 to 71 ms
later on either engine than with the calls alone (745 ms at the median): the agents' steps share the GPU with the speech
server, which no scheduling in vLLM removes. Stock vLLM then adds its waits for KV: the first sentence's p95 grows from
437 ms alone to 909 at four trees, 2,424 at eight and 9,719 at 24, where the change holds it at 534, 737 and 768. The
change removes the waits for KV and the prefill beside the reply's first tokens, as the earlier records show; the
contention for the card stays: on the change, from eight trees on, the transcript and the first sentence together take
176 to 249 ms longer at the median than with the calls alone, against a margin of 250.

## The sweep's confirmation

<!-- gate.py sweep -->
| Configuration | Runs | Median run p95, ms | Pooled p95, ms | Replies over the SLO | Agents, tokens/s (240 s): median | Agents' TTFT p50/p99, s |
|---|---:|---:|---:|---:|---:|---:|
| w5-disc-frozen-s24 | 5 | 12,010 | 11,766 | 261/295 (88.5%) | 3,399 | 68.7/109.4 |
| w5-disc-pr-ab48-s24 | 5 | 2,246 | 2,259 | 94/295 (31.9%) | 3,303 | 77.7/116.7 |
| sweep-disc-pr-ab48-b256-t0-s24 | 5 | 2,098 | 2,151 | 44/295 (14.9%) | 3,067 | 84.8/120.2 |

Replies over the round's SLO, by round:

| Configuration | rg-r1 | rg-r2 | rg-r3 | rg-r4 | rg-r5 |
|---|---:|---:|---:|---:|---:|
| The round's SLO, ms | 2,000.4 | 1,972.0 | 2,010.0 | 2,079.6 | 2,091.5 |
| w5-disc-frozen-s24 | 53/59 | 55/59 | 55/59 | 54/59 | 44/59 |
| w5-disc-pr-ab48-s24 | 16/59 | 29/59 | 25/59 | 9/59 | 15/59 |
| sweep-disc-pr-ab48-b256-t0-s24 | 10/59 | 17/59 | 10/59 | 4/59 | 3/59 |
<!-- /gate.py sweep -->

Two callers beside 24 session trees in all five rounds, each held to its own round's SLO. The change at the step cap
of 256 without a per-request threshold had fewer replies over the SLO than W5's configuration (the box's step cap of
512 with a threshold of 256) in each of the five rounds, 44 of 295 against 94 (14.9% against 31.9%), with a pooled p95
of 2,151 ms against 2,259; stock vLLM had 261 of 295 over (88.5%). The agents pay for it: 3,067 tokens a second over the
calls' first 240 s against 3,303 at the median, 7% less, and a median time to first token of 85 s against 78. The
screens' choice stands, and the serving profile runs 256 without a threshold. No step cap keeps 24 session trees within
the requirement of at most 5% over; the change's consolidated card meets it at four ([above](#the-two-card-comparison)).
