# The optimization rounds (2026-11-02)

What the serving profile's additions to v1.14 do in the whole pipeline: five rounds, odd ones on Phastform and even
ones on Gray, each on a fresh sample of the agents. In each round, v1.14's configuration to deploy (the change at the
serving profile's step cap of 256) runs beside the same with the first four additions: the calls' prefixes kept on the
GPU, the next batch scheduled just in time, each turn announced when the user stops speaking, and the calls taking the
agents' block reservations, beside 24 agent sessions with 2, 4 and 6 callers and beside 16 agent sessions of the whole
corpus. [The engine points](../2026-11-01-engine-points/README.md) chose each addition; the fifth, the agents' decodes
held through the budget cap's window, came after the rounds began, and the follow-up rounds measure it.

## The SLO

<!-- table.py slo -->
| Scenario | Configuration | Rounds | Replies over the SLO | V2V p50, ms | V2V p95 | The SLO |
|---|---|---:|---:|---:|---:|---:|
| 24 agent sessions, 2 callers | v1.14 | 5 | 55 of 295 (18.6%) | 1,894 (1,853–1,934) | 2,176 (2,066–2,215) | 2,010 (1,976–2,051) |
| 24 agent sessions, 2 callers | The first four additions | 5 | 16 of 295 (5.4%) | 1,712 (1,692–1,765) | 2,023 (1,927–2,139) | 2,010 (1,976–2,051) |
| 24 agent sessions, 4 callers | v1.14 | 5 | 110 of 595 (18.5%) | 1,897 (1,849–1,934) | 2,274 (2,126–2,299) | 2,065 (2,046–2,142) |
| 24 agent sessions, 4 callers | The first four additions | 5 | 38 of 595 (6.4%) | 1,744 (1,709–1,800) | 2,106 (2,015–2,168) | 2,065 (2,046–2,142) |
| 24 agent sessions, 6 callers | v1.14 | 5 | 164 of 910 (18.0%) | 1,913 (1,890–1,944) | 2,320 (2,260–2,361) | 2,119 (2,047–2,174) |
| 24 agent sessions, 6 callers | The first four additions | 5 | 53 of 910 (5.8%) | 1,754 (1,729–1,778) | 2,152 (2,062–2,172) | 2,119 (2,047–2,174) |
| 16 agent sessions of the whole corpus, 2 callers | v1.14 | 5 | 22 of 295 (7.5%) | 1,661 (1,635–1,801) | 2,012 (1,931–2,322) | 2,010 (1,976–2,051) |
| 16 agent sessions of the whole corpus, 2 callers | The first four additions | 5 | 10 of 295 (3.4%) | 1,675 (1,641–1,708) | 1,916 (1,820–2,049) | 2,010 (1,976–2,051) |
<!-- /table.py slo -->

- **Fewer replies over the SLO in every round of every scenario.** Pooled over the five rounds, the first four
  additions put 5.4% of the replies over their round's SLO beside 24 agent sessions with two callers, against v1.14's
  18.6%; 6.4% against 18.5% with four and 5.8% against 18.0% with six; and 3.4% against 7.5% beside 16 sessions of the
  whole corpus. They had fewer in each of the 20 rounds and scenarios.
- **The 5% allowance.** Beside 16 sessions the first four additions met it pooled and in four of the five rounds.
  Beside 24 they met it in 5 of the 15 rounds and caller counts, and v1.14 in none.
- **The tail.** The rounds' median V2V p95 fell by 153 ms with two callers, 168 with four, 169 with six and 96 beside
  16 sessions. With four and six callers it stays 41 and 33 ms over the rounds' median SLO, against 209 and 201 ms for
  v1.14.

## The voice turn's LLM part

<!-- table.py voice -->
| Scenario | Configuration | LLM first token p50, ms | p95 | First sentence p50 | p95 | E2E p95 |
|---|---|---:|---:|---:|---:|---:|
| 24 agent sessions, 2 callers | v1.14 | 220 (204–237) | 314 (290–345) | 464 (443–484) | 706 (653–718) | 1,783 (1,680–1,815) |
| 24 agent sessions, 2 callers | The first four additions | 94 (86–102) | 136 (124–139) | 327 (307–332) | 560 (470–623) | 1,643 (1,526–1,720) |
| 24 agent sessions, 4 callers | v1.14 | 210 (198–225) | 334 (329–340) | 456 (430–487) | 787 (608–816) | 1,898 (1,707–1,965) |
| 24 agent sessions, 4 callers | The first four additions | 95 (92–103) | 139 (128–145) | 324 (308–368) | 580 (505–672) | 1,740 (1,585–1,785) |
| 24 agent sessions, 6 callers | v1.14 | 206 (199–232) | 325 (312–346) | 437 (425–496) | 737 (686–861) | 1,939 (1,889–1,953) |
| 24 agent sessions, 6 callers | The first four additions | 95 (88–99) | 139 (129–166) | 316 (298–339) | 557 (535–593) | 1,692 (1,684–1,753) |
| 16 agent sessions of the whole corpus, 2 callers | v1.14 | 102 (88–150) | 255 (188–303) | 329 (302–402) | 570 (494–822) | 1,635 (1,534–1,959) |
| 16 agent sessions of the whole corpus, 2 callers | The first four additions | 90 (84–94) | 125 (112–145) | 305 (272–337) | 458 (438–627) | 1,517 (1,496–1,684) |
<!-- /table.py voice -->

- **The engine points' gains reach the pipeline.** Beside 24 sessions the LLM's first token at p95 falls from 314–334
  ms to 136–139 ms and its first sentence at p95 from 706–787 ms to 557–580 ms, with two, four or six callers.

## The agents

<!-- table.py agents -->
| Scenario | Configuration | Agents' tokens/s | Against v1.14 | Agents' TTFT p50, s | p99 |
|---|---|---:|---:|---:|---:|
| 24 agent sessions, 2 callers | v1.14 | 3,010 (2,720–3,275) | – | 100.5 (74.9–116.4) | 125.2 (112.2–138.7) |
| 24 agent sessions, 2 callers | The first four additions | 2,933 (2,692–3,221) | -1.6% (-4.3%–-0.5%) | 105.0 (74.5–120.2) | 125.9 (111.7–141.9) |
| 24 agent sessions, 4 callers | v1.14 | 2,753 (2,548–3,062) | – | 112.2 (79.8–130.0) | 130.7 (116.0–147.8) |
| 24 agent sessions, 4 callers | The first four additions | 2,754 (2,508–2,947) | -3.7% (-4.3%–+0.0%) | 121.6 (87.0–135.4) | 139.8 (117.1–153.4) |
| 24 agent sessions, 6 callers | v1.14 | 2,580 (2,415–2,888) | – | 124.7 (88.8–137.6) | 142.1 (118.6–154.0) |
| 24 agent sessions, 6 callers | The first four additions | 2,476 (2,215–2,780) | -4.9% (-8.3%–-2.1%) | 131.8 (95.5–145.0) | 148.3 (129.6–160.5) |
| 16 agent sessions of the whole corpus, 2 callers | v1.14 | 4,948 (4,633–8,364) | – | 0.1 (0.1–0.2) | 6.6 (1.2–7.7) |
| 16 agent sessions of the whole corpus, 2 callers | The first four additions | 4,900 (4,666–9,034) | -1.0% (-2.3%–+8.0%) | 0.1 (0.1–0.2) | 7.3 (2.9–8.2) |
<!-- /table.py agents -->

- **The agents give up 2–5% of their token rate** beside 24 sessions, at the median of each round's pair: 1.6% with
  two callers, 3.7% with four and 4.9% with six, as more callers announce more turns and hold the agents' prefill for
  more first tokens. Their median time to first token rises by 4.5–9.4 s, within the rounds' spread.

## Where a turn's time goes

<!-- table.py stages -->
| Scenario | Configuration | Transcript | LLM first token | LLM first sentence | TTS first audio |
|---|---|---:|---:|---:|---:|
| 24 agent sessions, 2 callers | v1.14 | 821 | 223 | 479 | 212 |
| 24 agent sessions, 2 callers | The first four additions | 815 | 96 | 352 | 207 |
| 24 agent sessions, 4 callers | v1.14 | 833 | 215 | 483 | 218 |
| 24 agent sessions, 4 callers | The first four additions | 822 | 97 | 348 | 214 |
| 24 agent sessions, 6 callers | v1.14 | 842 | 213 | 474 | 225 |
| 24 agent sessions, 6 callers | The first four additions | 826 | 97 | 344 | 211 |
| 16 agent sessions of the whole corpus, 2 callers | v1.14 | 790 | 132 | 370 | 192 |
| 16 agent sessions of the whole corpus, 2 callers | The first four additions | 807 | 91 | 317 | 187 |
<!-- /table.py stages -->

The voice turn's mean stage times over every reply of each configuration's rounds: the caller's transcript from the
end of their speech, the LLM's first token and first sentence from the request, and TTS's first audio from the first
sentence.

- **Speech to text is now the turn's longest stage.** Beside 24 sessions the caller's transcript takes 815–826 ms from
  the end of their speech, more than twice the LLM's first sentence, 344–352 ms, which the first four additions took
  from 474–483 ms.

## The decode hold and speculative decoding

<!-- table.py follow -->
| Scenario | Configuration | Rounds | Replies over the SLO | V2V p95, ms | LLM first token p95 | First sentence p50 | p95 | Agents' tokens/s | Against the first four additions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 agent sessions, 2 callers | The first four additions | 2 | 11 of 118 (9.3%) | 2,048 (2,034–2,061) | 127 (118–135) | 340 (302–378) | 596 (559–634) | 3,034 (2,838–3,231) | – |
| 24 agent sessions, 2 callers | + the decode hold | 2 | 6 of 118 (5.1%) | 1,939 (1,907–1,972) | 130 (125–135) | 311 (309–312) | 455 (449–461) | 2,897 (2,770–3,024) | -4.4% (-6.4%–-2.4%) |
| 24 agent sessions, 2 callers | + the hold and speculation | 2 | 1 of 118 (0.8%) | 1,828 (1,825–1,830) | 168 (155–181) | 225 (222–228) | 345 (337–352) | 2,758 (2,566–2,951) | -9.1% (-9.6%–-8.7%) |
| 24 agent sessions, 4 callers | The first four additions | 2 | 15 of 238 (6.3%) | 2,157 (2,143–2,172) | 142 (129–155) | 331 (323–340) | 606 (580–631) | 2,710 (2,500–2,919) | – |
| 24 agent sessions, 4 callers | + the decode hold | 2 | 2 of 238 (0.8%) | 1,955 (1,879–2,030) | 138 (138–139) | 304 (292–316) | 433 (419–447) | 2,609 (2,385–2,834) | -3.8% (-4.6%–-2.9%) |
| 24 agent sessions, 4 callers | + the hold and speculation | 2 | 4 of 238 (1.7%) | 1,915 (1,892–1,938) | 177 (167–186) | 237 (236–238) | 377 (372–382) | 2,568 (2,343–2,793) | -5.3% (-6.3%–-4.3%) |
| 24 agent sessions, 6 callers | The first four additions | 2 | 29 of 364 (8.0%) | 2,156 (2,118–2,195) | 141 (138–145) | 323 (319–327) | 641 (623–658) | 2,464 (2,294–2,633) | – |
| 24 agent sessions, 6 callers | + the decode hold | 2 | 8 of 364 (2.2%) | 2,000 (1,985–2,016) | 136 (134–138) | 306 (302–310) | 459 (454–464) | 2,390 (2,272–2,507) | -2.9% (-4.8%–-1.0%) |
| 24 agent sessions, 6 callers | + the hold and speculation | 2 | 7 of 364 (1.9%) | 1,991 (1,980–2,002) | 172 (161–184) | 235 (230–240) | 371 (357–385) | 2,328 (2,189–2,467) | -5.4% (-6.3%–-4.6%) |
<!-- /table.py follow -->

Two follow-up rounds, Phastform then Gray, on new samples of the agents: the first four additions, the same with the
agents' decodes held from a reply's first 48 tokens, the budget cap's window, and that with speculative decoding too
(EAGLE-3, two drafts a step, the calls' short loads recomputed, the KV pool kept at 2,275 blocks beside the drafter),
beside 24 agent sessions with 2, 4 and 6 callers, and voice alone with as many callers for each round's SLO.

- **The decode hold cut the replies over the SLO by more than two thirds**: 16 of 720 against 55 for the first four
  additions over the two rounds and three caller counts, fewer in five of the six and as many in the sixth, its first
  sentence's p95 433–459 ms against 596–641. It cost the agents 3.8% of their token rate at the median of the six,
  2.9–4.4% by caller count.
- **Speculative decoding on top** put 12 of 720 over against the hold's 16, fewer in three of the six, its first
  sentence's p95 345–377 ms but its first token's 168–177 ms against 130–138, at 9.1% of the agents' token rate with two
  callers and 5.3–5.4% with four and six against the first four additions.
- **Neither is in the serving profile**, by rules stated before these rounds ran: the decode hold was to stay if it had
  no more replies over the SLO and cost the agents no more than 3% at the median of the six, and speculative decoding
  was to come in only with fewer replies over the SLO than the hold, pooled and in at least five of the six. Both stay
  switches an operator can set (`make_room.decode_hold_tokens 48`; EAGLE-3 as `opt-disc-spec-*` runs it).

## How it ran

| | |
|---|---|
| The rounds | [`benchmarks/e2/opt-r1.campaign`](../../e2/opt-r1.campaign) to `opt-r5.campaign`, seeds 31 to 35, each round's engine groups in an order rotated from the round before |
| The release | v1.14-17-g157f6b7 on cyan, Phastform and Gray for every round |
| v1.14's configuration | `sweep-disc-pr-ab48-b256-t0-s24`, `opt-disc-b256-c4-s24`, `opt-disc-b256-c6-s24`, `opt-std-b256-s16` |
| The first four additions | `opt-disc-ret-s24`, `opt-disc-ret-c4-s24`, `opt-disc-ret-c6-s24`, `opt-std-ret-s16`: the same with `make_room.retain_tokens 3584`, `retain_s 30`, `hint_hold_ms 400`, `take_reservations true`, the serve module's scheduler with a 30 ms margin, and the voice worker's `turn_hints` |
| The follow-up rounds | [`opt-s1.campaign`](../../e2/opt-s1.campaign) and [`opt-s2.campaign`](../../e2/opt-s2.campaign), seeds 36 and 37, on v1.14-42-g51a6d45 on cyan, Phastform and Gray: `opt-disc-ret-*`, `opt-disc-hold-*` (+ `make_room.decode_hold_tokens 48`) and `opt-disc-spec-*` (+ EAGLE-3 with 2 drafts, `make_room.recompute_tokens 256`, `--kv-cache-memory 5517000000`), at 2, 4 and 6 callers, with voice alone at each count |
| The SLO | A reply is over it when its voice-to-voice time exceeds the same round's voice alone with as many callers, plus 250 ms ([RESOURCE-GATE.md](../../quality/RESOURCE-GATE.md)) |

## Files

- `opt-rN/<point>/`, `opt-sN/<point>/`: each point as `flatshare-bench e2 export` writes it (`run.json`, `arm.toml`, `engine.json`,
  `summary.json`, the callers' `bots/summary.json` and `bots/turns.jsonl`), home directories as `~`; each round's
  `campaign.log`.
- `table.py`: the tables above (`cd python/flatshare_bench && uv run python
  ../../benchmarks/results/2026-11-02-opt-rounds/table.py NAME`).
