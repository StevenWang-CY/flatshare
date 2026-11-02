# Engine points (2026-11-01)

What each engine-level change does to a call's reply and to the agents beside it on an RTX 5060 Ti: two calls of the
voice agent's turns beside 12 coding agents' long-context requests, 240 s a point, on Phastform and on Gray, with the
engine's scheduler traced step by step. These points chose what the serving profile adds to v1.14's; the optimization
rounds measure the result in the whole pipeline.

## What was adopted

<!-- table.py adopted -->
| Node | Arm | Runs | First token p50, ms | p95 | 16th token p50 | p95 | Later tokens, ms each | Agents' tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| phastform | v1.14's serving profile (asynchronous scheduling) | 2 | 175 (170-180) | 354 (339-368) | 408 (407-409) | 597 (594-600) | 13.7 | 2,718 |
| phastform | + the calls' prefixes kept on the GPU (retention) | 2 | 122 (118-127) | 230 (223-236) | 354 (349-359) | 502 (495-508) | 13.9 | 2,761 |
| phastform | + the next batch just in time (30 ms margin) | 2 | 98 (94-101) | 185 (170-200) | 323 (320-326) | 449 (448-450) | 13.8 | 2,675 |
| phastform | + each turn announced (250 ms ahead, held 400 ms) | 2 | 64 (63-65) | 90 (83-96) | 258 (256-259) | 394 (388-401) | 13.7 | 2,637 |
| phastform | + the turn announced and just in time | 1 | 63 | 98 | 257 | 403 | 13.6 | 2,628 |
| phastform | + the calls taking the agents' reservations | 2 | 64 (64-64) | 92 (91-94) | 261 (260-262) | 392 (391-392) | 13.6 | 2,625 |
| phastform | + the agents' decodes held from a call's first 16 tokens | 2 | 64 (63-65) | 94 (92-95) | 258 (256-261) | 285 (284-286) | 13.8 | 2,615 |
| phastform | the same, the decodes held from a call's first 48 tokens | 2 | 66 (65-66) | 88 (85-92) | 257 (257-257) | 284 (278-291) | 13.4 | 2,604 |
| gray | v1.14's serving profile (asynchronous scheduling) | 2 | 182 (179-184) | 308 (299-316) | 400 (396-403) | 608 (591-624) | 13.8 | 2,682 |
| gray | + the calls' prefixes kept on the GPU (retention) | 2 | 139 (136-142) | 256 (254-259) | 374 (373-376) | 502 (497-507) | 13.8 | 2,686 |
| gray | + each turn announced (250 ms ahead, held 400 ms) | 2 | 72 (71-74) | 110 (109-110) | 270 (269-270) | 405 (404-406) | 13.7 | 2,606 |
| gray | + the turn announced and just in time | 2 | 72 (71-72) | 111 (108-114) | 270 (270-270) | 438 (413-462) | 13.5 | 2,584 |
| gray | + the calls taking the agents' reservations | 2 | 71 (71-71) | 106 (96-115) | 268 (266-269) | 402 (395-408) | 13.5 | 2,554 |
| gray | + the agents' decodes held from a call's first 16 tokens | 1 | 72 | 108 | 268 | 300 | 13.8 | 2,569 |
| gray | the same, the decodes held from a call's first 48 tokens | 2 | 74 (73-74) | 106 (103-109) | 267 (267-267) | 300 (299-301) | 13.3 | 2,600 |
<!-- /table.py adopted -->

- **The calls' prefixes kept on the GPU** (`make_room.retain_tokens 3584`, `retain_s 30`): a call's next turn finds
  its prefix on the GPU instead of loading it from the CPU tier, which costs two or three steps however small the load
  is. The engine's load stage falls from 69 to 12 ms a turn, and the first token from 175 to 122 ms at the median and
  from 354 to 230 ms at p95 on Phastform, with the agents unchanged.
- **The next batch just in time** (`flatshare_jit`, a 30 ms margin): a call that arrives while an agent's long prefill
  step runs joins the next batch instead of the one after it. The first token falls to 98 ms (p95 185), and the wait
  adds no idle time to the GPU.
- **Each turn announced** (`make_room.hint_hold_ms 400` and the voice worker's `turn_hints`): when the user stops
  speaking, one speech recognition round trip before the turn's request, the worker sends a realtime request for one
  token; the budget cap holds the agents' prefill back from it, so the turn arrives into decode-only steps. The first
  token falls to 64 ms (p95 90), and the 16th token, about the first sentence, from 408 to 258 ms. The agents give up
  4% of their tokens a second.
- **The calls taking the agents' reservations** (`make_room.take_reservations`): see [The stalls](#the-stalls). It
  costs the calls nothing and the agents 1%, within the replicates' spread.
- **The agents' decodes held from a call's first tokens** (`make_room.decode_hold_tokens`): a step that also decodes
  an agent at 15,000 tokens of context takes 18-20 ms against 13 alone, and in one turn in twenty such steps added
  108-135 ms to the first sentence. With the agents' decodes sitting out a call's first 16 tokens, the 16th token's p95
  falls from 392 to 285 ms, its median and the agents unchanged. The serving profile holds them for 48 tokens, the
  budget cap's window ([how long the hold lasts](#how-long-the-hold-lasts)).

With everything adopted, the agents' decodes held for 48 tokens, a call's first token is 66 ms at the median and 88 ms
at p95 on Phastform, against v1.14's 175 and 354 ms; its 16th token, about a first sentence, comes at 257 ms and 284 ms
at p95, against 408 and 597; and its 48th, the whole reply, at 688 and 722 ms, against 867 and 1,227.

## What was not

<!-- table.py alternatives -->
| Node | Arm | Runs | First token p50, ms | p95 | 16th token p50 | p95 | Later tokens, ms each | Agents' tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| phastform | synchronous scheduling | 2 | 114 (112-117) | 210 (197-224) | 380 (380-381) | 562 (553-572) | 16.5 | 2,374 |
| phastform | synchronous scheduling, retention | 2 | 87 (85-89) | 164 (151-177) | 372 (369-376) | 480 (478-483) | 16.6 | 2,465 |
| phastform | Triton attention, full CUDA graphs | 1 | 189 | 333 | 433 | 768 | 13.2 | 2,301 |
| phastform | Triton, full graphs, synchronous | 1 | 139 | 265 | 458 | 775 | 16.2 | 1,950 |
| phastform | Triton, full graphs, retention, just in time (6 ms) | 1 | 93 | 177 | 351 | 616 | 13.0 | 2,227 |
| phastform | Triton, full graphs, FP8 KV | 1 | 283 | 539 | 709 | 1088 | 17.3 | 1,609 |
| phastform | FlashInfer, FP8 KV (#64165's wait) | 1 | 244 | 397 | 511 | 690 | 14.9 | 2,084 |
| phastform | retention, two API servers | 1 | 121 | 234 | 364 | 527 | 13.9 | 2,792 |
| gray | synchronous scheduling | 2 | 130 (122-138) | 211 (211-211) | 427 (426-428) | 586 (584-588) | 18.0 | 2,193 |
| gray | synchronous scheduling, retention | 2 | 93 (91-95) | 181 (171-191) | 376 (375-378) | 534 (526-541) | 18.1 | 2,208 |
| gray | Triton attention, full CUDA graphs | 1 | 224 | 372 | 453 | 818 | 13.1 | 2,086 |
| gray | Triton, full graphs, synchronous | 1 | 133 | 250 | 459 | 817 | 17.8 | 1,863 |
| gray | Triton, full graphs, FP8 KV | 1 | 245 | 581 | 725 | 1151 | 24.9 | 1,531 |
| gray | FlashInfer, FP8 KV (#64165's wait) | 1 | 250 | 431 | 484 | 683 | 14.0 | 2,061 |
| gray | retention, EAGLE-3 with 3 drafts | 1 | 219 | 390 | 329 | 572 | 10.8 | 2,576 |
| gray | retention, EAGLE-3 with 2 drafts | 1 | 226 | 374 | 342 | 607 | 8.4 | 2,587 |
| gray | retention, EAGLE-3 with 3 drafts, short loads recomputed | 1 | 160 | 261 | 298 | 453 | 9.8 | 2,618 |
| gray | retention, EAGLE-3 with 2 drafts, short loads recomputed | 1 | 160 | 293 | 303 | 587 | 8.3 | 2,578 |
| gray | the turn announced and just in time, EAGLE-3 with 2 drafts, short loads recomputed | 2 | 76 (75-76) | 137 (136-138) | 171 (166-176) | 387 (381-393) | 7.6 | 2,572 |
| gray | the turn announced, just in time, the reservations taken, EAGLE-3 with 2 drafts, short loads recomputed | 2 | 78 (76-79) | 130 (115-144) | 194 (191-197) | 400 (368-432) | 9.1 | 2,527 |
| gray | the same, the agents' decodes held from a call's first 16 tokens | 2 | 78 (78-78) | 140 (136-143) | 182 (177-186) | 260 (259-262) | 9.6 | 2,519 |
<!-- /table.py alternatives -->

- **Synchronous scheduling** gives the first token earlier, but each later token takes 16.5 to 18 ms instead of 13.7,
  so with retention the first sentence is no earlier (372 against 354 ms on Phastform), and the agents lose 11 to
  18%.
- **Triton attention with full CUDA graphs** cuts the call's launch from 29 to 8 ms, but its long-context prefill is
  1.2 times FlashAttention 2's, so the agents lose 15 to 22% and the first sentence comes later.
- **FP8 KV** doubles the pool, but FlashInfer plans every step on the host (+23 ms a step) and Triton's FP8 attention
  is slower still: the agents lose 23 to 43%.
- **Two API servers** change nothing: the input stage is not the API server's.
- **Speculative decoding** (EAGLE-3, 2 drafts) nearly halves each later token, 13.7 to 7.6-8.4 ms. Beside the agents,
  its drafter drops the last block of every prefix hit, so 68 of 71 turns loaded that block from the CPU tier and the
  first token rose to 219-226 ms. Recomputing a call's short loads instead (`make_room.recompute_tokens 256`) removes
  the loads. On top of the announced turn and just in time, it brings the first sentence from 270 to 166-176 ms on
  Gray, at 4 ms more on the first token's median and 26 ms more at its p95. The decode hold keeps its effect beside
  it: the 16th token's p95 259-262 ms against 368-432 without the hold, the first token and the agents within the
  runs' spread. It is measured in the whole pipeline before it is adopted.

## How long the hold lasts

<!-- table.py hold -->
| Node | Arm | Runs | First token p50, ms | p95 | 16th token p50 | p95 | 48th token p50 | p95 | Agents' tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phastform | + the calls taking the agents' reservations | 2 | 64 (64-64) | 92 (91-94) | 261 (260-262) | 392 (391-392) | 700 (696-705) | 998 (953-1043) | 2,625 |
| phastform | + the agents' decodes held from a call's first 16 tokens | 2 | 64 (63-65) | 94 (92-95) | 258 (256-261) | 285 (284-286) | 704 (701-707) | 966 (948-983) | 2,615 |
| phastform | the same, the decodes held from a call's first 48 tokens | 2 | 66 (65-66) | 88 (85-92) | 257 (257-257) | 284 (278-291) | 688 (687-690) | 722 (722-722) | 2,604 |
| gray | + the calls taking the agents' reservations | 2 | 71 (71-71) | 106 (96-115) | 268 (266-269) | 402 (395-408) | 698 (698-699) | 1016 (989-1042) | 2,554 |
| gray | + the agents' decodes held from a call's first 16 tokens | 1 | 72 | 108 | 268 | 300 | 714 | 980 | 2,569 |
| gray | the same, the decodes held from a call's first 32 tokens | 2 | 72 (72-73) | 108 (108-108) | 264 (264-265) | 299 (298-300) | 701 (700-702) | 841 (838-844) | 2,582 |
| gray | the same, the decodes held from a call's first 48 tokens | 2 | 74 (73-74) | 106 (103-109) | 267 (267-267) | 300 (299-301) | 693 (692-694) | 735 (734-736) | 2,600 |
<!-- /table.py hold -->

- **At 16 tokens the hold covers a first sentence of 16 tokens**: the 16th token's p95 falls to 285-300 ms, but the
  48th token's barely moves, 966-980 ms against 998-1,016, since the agents' decodes return to the reply's steps after
  its 16th token.
- **At 48 tokens, the budget cap's window, the whole reply comes out of the agents' way**: the 48th token's p95 is 722
  ms on Phastform and 735 on Gray, within 50 ms of its median, against 966 and 980 at 16 and 998 and 1,016 without the
  hold, with the first token, the 16th token and the agents unchanged. The voice agent's first sentences run up to 39 tokens and its replies 23 at the median
  ([the report's §3.3](../../../docs/REPORT.md#33-sizing-the-window)), and a hold ends with its reply, so the serving
  profile holds the agents' decodes for 48 tokens, as the budget cap holds their prefill.

## Controls

<!-- table.py controls -->
| Node | Arm | Runs | First token p50, ms | p95 | 16th token p50 | p95 | Later tokens, ms each | Agents' tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| phastform | retention, the turns announced but nothing held | 1 | 119 | 218 | 336 | 485 | 13.8 | 2,692 |
| phastform | retention, the hint scheduler with no turn announced | 1 | 126 | 253 | 371 | 540 | 13.9 | 2,763 |
| phastform | retention, the agents' reservations taken, no turn announced | 1 | 121 | 224 | 360 | 484 | 13.9 | 2,746 |
| gray | retention, the turns announced but nothing held | 1 | 140 | 239 | 356 | 516 | 13.8 | 2,646 |
| gray | retention, the hint scheduler with no turn announced | 1 | 148 | 279 | 406 | 553 | 13.9 | 2,660 |
| gray | retention, short loads recomputed, no speculation | 1 | 142 | 251 | 384 | 501 | 13.8 | 2,669 |
<!-- /table.py controls -->

The turns announced with nothing held keep only the prefix they warm (the load stage), and the hint's scheduler with no
turn announced, short loads recomputed without speculation, and the reservations taken without a turn announced each run
as retention alone.

## With fresh agent prompts

<!-- table.py fresh -->
| Node | Arm | Runs | First token p50, ms | p95 | 16th token p50 | p95 | Later tokens, ms each | Agents' tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| phastform | v1.14's serving profile | 1 | 173 | 269 | 390 | 547 | 13.7 | 3,082 |
| phastform | retention | 1 | 131 | 259 | 378 | 481 | 13.8 | 3,101 |
| phastform | retention, just in time | 1 | 95 | 197 | 324 | 431 | 13.7 | 3,063 |
| phastform | retention, each turn announced | 1 | 63 | 86 | 256 | 347 | 13.6 | 2,968 |
| phastform | synchronous scheduling | 1 | 100 | 164 | 371 | 477 | 16.6 | 2,679 |
| phastform | synchronous scheduling, retention | 1 | 90 | 150 | 367 | 487 | 16.6 | 2,687 |
| gray | v1.14's serving profile | 1 | 174 | 268 | 392 | 541 | 13.8 | 3,010 |
| gray | retention | 1 | 149 | 269 | 381 | 506 | 13.8 | 3,032 |
| gray | retention, each turn announced | 1 | 74 | 102 | 267 | 392 | 13.6 | 2,908 |
| gray | synchronous scheduling | 1 | 104 | 167 | 395 | 545 | 18.0 | 2,520 |
| gray | synchronous scheduling, retention | 1 | 110 | 171 | 410 | 552 | 17.9 | 2,521 |
| gray | retention, EAGLE-3 with 3 drafts | 1 | 208 | 337 | 308 | 531 | 10.6 | 2,974 |
<!-- /table.py fresh -->

The same with agents whose every turn is a new prompt, as E2's discriminating replay computes about 95% of its prompt
tokens: the order of the arms holds.

## Where the first token goes

<!-- table.py stages -->
| Node | Arm | First token in the engine, mean ms | input | load | queue | prep | gpu | own | Turns loading from the CPU tier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phastform | v1.14's serving profile (asynchronous scheduling) | 183 | 51 | 69 | 6 | 29 | 13 | 15 | 72 of 136 |
| phastform | + the calls' prefixes kept on the GPU (retention) | 130 | 46 | 12 | 1 | 29 | 23 | 19 | 12 of 138 |
| phastform | + the next batch just in time (30 ms margin) | 99 | 25 | 7 | 22 | 30 | 3 | 13 | 9 of 138 |
| phastform | + each turn announced (250 ms ahead, held 400 ms) | 58 | 17 | 0 | 0 | 28 | 3 | 10 | 0 of 136 |
| phastform | synchronous scheduling | 119 | 48 | 29 | 7 | 16 | 0 | 20 | 71 of 136 |
| phastform | Triton attention, full CUDA graphs | 190 | 48 | 68 | 7 | 8 | 36 | 22 | 34 of 68 |
| phastform | FlashInfer, FP8 KV (#64165's wait) | 235 | 68 | 76 | 9 | 52 | 15 | 16 | 42 of 68 |
| gray | v1.14's serving profile (asynchronous scheduling) | 180 | 49 | 59 | 6 | 30 | 17 | 18 | 62 of 137 |
| gray | + the calls' prefixes kept on the GPU (retention) | 141 | 51 | 9 | 0 | 28 | 31 | 21 | 9 of 138 |
| gray | + each turn announced (250 ms ahead, held 400 ms) | 65 | 24 | 0 | 0 | 27 | 2 | 12 | 0 of 136 |
| gray | synchronous scheduling | 125 | 49 | 31 | 8 | 15 | 0 | 22 | 73 of 134 |
| gray | Triton attention, full CUDA graphs | 225 | 72 | 75 | 7 | 9 | 41 | 21 | 36 of 68 |
| gray | FlashInfer, FP8 KV (#64165's wait) | 246 | 82 | 99 | 9 | 34 | 12 | 9 | 49 of 68 |
| gray | retention, EAGLE-3 with 2 drafts | 242 | 59 | 120 | 12 | 25 | 6 | 21 | 68 of 71 |
| gray | retention, EAGLE-3 with 2 drafts, short loads recomputed | 222 | 53 | 9 | 65 | 32 | 37 | 26 | 4 of 70 |
<!-- /table.py stages -->

The engine's first token by stage: the request's way in (input), its prefix's load from the CPU tier (load), the steps
it waits in the queue (queue), its batch's preparation and launch on the host (prep), the step it waits behind on the
GPU (gpu), and its own step (own).

## The stalls

On Gray, three points without a turn held had call turns that waited 1.4 to 4.5 s for their first step (the arms
without a hint and with speculation, above). vLLM admits a load from the CPU tier only if its prompt fits beside every
in-flight prefill's reservation, the blocks that prompt still needs, and the waiting loop stops at the first request
holding KV that it cannot admit. An agent's 20,000-token prompt reserves about a thousand blocks, so a call whose turn
loaded a few blocks of its last reply, or that sat behind an agent whose load was done but whose prompt no longer fit,
waited until the agents moved on; make-room could not preempt for it, as the reservations belonged to requests that
were loading, not running. With `take_reservations` a call leaves free only the reservations of its own priority or a
higher one, and the budget cap holds the agents' prefill back as it is admitted. make-room's tests reproduce both
cases (`test_a_lower_priority_loads_reservation_keeps_no_call_waiting`,
`test_a_lower_priority_load_that_cannot_be_promoted_keeps_no_call_waiting`).

## Speculative decoding alone

<!-- table.py speculation -->
| Drafts | Tokens a step | ms a token, p50 | First token p50, ms | 16th token p50, ms |
|---|---:|---:|---:|---:|
| none | 1.00 | 12.4 | 47 | 233 |
| 2 | 1.97 | 7.7 | 63 | 181 |
| 3 | 2.15 | 8.0 | 62 | 190 |
| 4 | 2.23 | 8.0 | 62 | 187 |
<!-- /table.py speculation -->

The two calls with nothing beside them, on Gray, with the voice agent's own prompt and the server's sampling defaults:
two drafts accept almost as many tokens a step as four, at the lowest cost a token.

## The CPU tier's restore

<!-- table.py restore -->
| Node | Voice conversations answered | Long documents | Restored from the CPU tier, identical to computed |
|---|---:|---:|---:|
| phastform | 24 | 1 | 1 of 1 |
| gray | 24 | 8 | 8 of 8 |
<!-- /table.py restore -->

Greedy answers to long documents whose prefixes were restored from the CPU tier, after filler requests had taken the
GPU pool, against the same answers computed, on FlashAttention 2 with BF16 KV.

## How it ran

Each point started one engine on its node from the box's `engine.yaml` (FlashAttention 2, BF16 KV in a 36,400-token
pool, the CPU tier of 20 GiB, asynchronous scheduling, priority scheduling) with make-room at the serving profile's
switches (two preemptions a step, the budget cap's 48-token window, no reserve, a 256-token step cap with no
long-prefill threshold) and the point's own flags (`flags.txt`), then ran
[tail.py](../../../python/flatshare_vllm/src/flatshare_vllm/make-room/tail.py) for 240 s: two calls of 48-token
replies, a turn every 4 to 8 s, beside 12 agents whose prompts grow from 6,000 to 30,000 tokens a turn (`grow`) or are
new each turn (`fresh`); with a hint, each call turn was announced 250 ms before it left. Gray's speculative points
used the EAGLE-3 head taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3 at a pool of the same 2,275 blocks.

One run is left out of the tables, [`left-out/phastform/p-r0hjtd48-grow-1`](left-out/phastform/p-r0hjtd48-grow-1/tail.txt):
its engine started while Phastform's host ran about 2.5 times slow, its CUDA graph capture taking 24 s against 9-11 s
in every other start, and every measure of it moved with the host, the agents' tokens a second too (1,991 against
2,604-2,625). The next two starts captured in 11 and 9 s, and those two runs are the table's.

## Files

- `<node>/<point>/tail.txt`: tail.py's output; `flags.txt`: the point's engine flags; `stages.json`: the calls' first
  token by stage, the steps by kind against the step-time model, the host's time a step and the agents' computed
  tokens a second, from the point's scheduler trace, which the record does not keep.
- `gray/g-spec-*/spec.json`: each turn's first, 16th and last token, and the engine's speculative counters.
- `phastform/p-agree-fa2/agree.json`, `gray/g-agree-fa2/agree.json`: every answer's token ids, computed and restored.
- `left-out/`: the run the tables leave out, and why above.
- `table.py`: the tables above (`python3 table.py NAME`).
