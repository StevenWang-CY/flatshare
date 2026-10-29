# The step-cap and window sweep's screens (2026-10-27)

W5 measured the change at the box's step cap of 512 tokens, where stock vLLM had done better at 1,024 and the change
never ran ([BENCHMARKS](../../../docs/BENCHMARKS.md)). The sweep runs the change, and stock vLLM beside it, over the step
cap, the long-prefill threshold, the first-tokens window and the gateway's static cap on background requests, first as
two screens on the sample of the agents that W5 tuned on (seed 11), then on fresh samples in five rounds
([`rg-r1`](../../e2/rg-r1.campaign) to [`rg-r5`](../../e2/rg-r5.campaign)), which confirm or overturn what the screens
pick. These are the screens' records, kept as [W5's](../2026-10-24-e2-w5) keep each point: `run.json`, `arm.toml`,
`engine.json`, `summary.json` and the callers' `bots/summary.json` and `bots/turns.jsonl`, with the first screen's
report (`sweep-screen/report`).

Both screens ran on Phastform's RTX 5060 Ti with vLLM 0.32.0 and Qwen3-4B-Instruct-2507-FP8, Phastform and cyan at
v1.5 (`versions` in each `run.json`), one run a point: two callers, each call opening with a warm-up turn, beside 24
session trees of the discriminating scenario (`weka-x32-long`). [`sweep-screen`](sweep-screen) ran from 20:54 to 22:45
UTC, [`sweep-screen2`](sweep-screen2) from 22:46 to 00:10 UTC. Each campaign's `campaign.log` names its arm files'
revision at every start. Each began with a dry run; the first screen then ran at v1.5-1-g41a9ff9 and, after the
workstation's SSH connections failed at 21:52 UTC, resumed at v1.5-9-g85061ea with its complete points kept; the second
ran at v1.5-14-g7d4e760, after a dry run from a tree with changes not yet committed. Each point's `arm.toml` is the arm
it ran.

The settings: the step cap (`--max-num-batched-tokens`) at 256, at the box's 512 (W5's configuration, `w5-disc-pr-ab48-s24`)
and at 1,024, with a long-prefill threshold of 0 or 256; the change's window at 48 tokens, or 96 (W5 measured 16, which a
whole reply outruns: [BENCHMARKS](../../../docs/BENCHMARKS.md)); stock vLLM, the frozen box, at step caps of 1,024 and 256; and the gateway's static cap of 1, 2,
4 or 8 background requests in flight (`policy = "static-cap"`), on the frozen box and on the change, alone and with
the step cap of 256.

[`table.py`](table.py) gives the table from these records (`cd python/flatshare_bench && uv run python
../../benchmarks/results/2026-10-27-sweep/table.py`), and `python/flatshare_bench/tests/test_sweep_record.py` fails
while this README's table is not what it gives:

<!-- table.py -->
W5's SLO 2,034.3 ms; the screen's own 2,001.2 ms (its voice alone's p95 1,751.2 ms plus 250)

| Point | Runs | Median run p95, ms | Pooled p95, ms | Over W5's SLO | Over the screen's | Agents, tokens/s (240 s) | Generated tokens/s | Agents' TTFT p50/p99, s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sweep-disc-b1024-t0-s24 | 1 | 9,115 | 9,115 | 57/59 (97%) | 57/59 (97%) | 3,956 | 17.4 | 57/86 |
| sweep-disc-frozen-b256-t0-s24 | 1 | 9,522 | 9,522 | 51/59 (86%) | 52/59 (88%) | 3,304 | 15.4 | 74/104 |
| sweep-disc-frozen-cap1-s24 | 1 | 4,421 | 4,421 | 41/59 (69%) | 41/59 (69%) | 3,357 | 15.9 | 76/105 |
| sweep-disc-frozen-cap2-s24 | 1 | 9,361 | 9,361 | 52/59 (88%) | 54/59 (92%) | 3,446 | 16.2 | 71/103 |
| sweep-disc-frozen-cap4-s24 | 1 | 10,884 | 10,885 | 54/59 (92%) | 57/59 (97%) | 3,587 | 16.1 | 63/96 |
| sweep-disc-frozen-cap8-s24 | 1 | 9,228 | 9,228 | 58/59 (98%) | 58/59 (98%) | 3,644 | 16.0 | 64/101 |
| sweep-disc-pr-ab48-b1024-t0-s24 | 1 | 2,420 | 2,420 | 33/59 (56%) | 37/59 (63%) | 3,832 | 16.8 | 63/90 |
| sweep-disc-pr-ab48-b1024-t256-s24 | 1 | 3,505 | 3,505 | 28/59 (47%) | 34/59 (58%) | 3,442 | 15.9 | 70/98 |
| sweep-disc-pr-ab48-b256-cap2-s24 | 1 | 2,072 | 2,071 | 8/59 (14%) | 13/59 (22%) | 3,108 | 15.4 | 79/111 |
| sweep-disc-pr-ab48-b256-cap4-s24 | 1 | 2,158 | 2,158 | 9/59 (15%) | 10/59 (17%) | 3,212 | 15.2 | 78/107 |
| sweep-disc-pr-ab48-b256-t0-s24 | 1 | 2,128 | 2,128 | 9/59 (15%) | 12/59 (20%) | 3,158 | 15.5 | 79/108 |
| sweep-disc-pr-ab48-cap1-s24 | 1 | 2,136 | 2,136 | 10/59 (17%) | 18/59 (31%) | 3,212 | 15.5 | 78/108 |
| sweep-disc-pr-ab48-cap2-s24 | 1 | 2,167 | 2,167 | 15/59 (25%) | 18/59 (31%) | 3,402 | 16.0 | 74/102 |
| sweep-disc-pr-ab48-cap4-s24 | 1 | 2,194 | 2,194 | 15/59 (25%) | 20/59 (34%) | 3,421 | 15.9 | 72/102 |
| sweep-disc-pr-ab48-cap8-s24 | 1 | 2,230 | 2,230 | 19/59 (32%) | 23/59 (39%) | 3,454 | 16.2 | 71/99 |
| sweep-disc-pr-ab96-b1024-t0-s24 | 1 | 2,415 | 2,415 | 33/59 (56%) | 36/59 (61%) | 3,767 | 16.7 | 63/89 |
| w5-a1-voice | 1 | 1,751 | 1,751 | 0/59 (0%) | 0/59 (0%) | - | - | - |
| w5-disc-pr-ab48-s24 | 1 | 2,236 | 2,236 | 24/59 (41%) | 26/59 (44%) | 3,355 | 16.3 | 71/100 |
<!-- /table.py -->

The points' names say their settings: `pr-ab48` is the change with the PRs' switches and a 48-token window (`ab96`,
96), `b256`/`b1024` the step cap, `t0`/`t256` the long-prefill threshold, `capN` the static cap, `frozen` or none of
`pr` stock vLLM. Each point measured 59 replies: its 50 turns' and the 9 that answered an interruption.

What the screens say, each from one run on one sample, for the rounds to confirm:

- **The step cap moves the change most.** At a cap of 256, 9 of its 59 replies were over W5's SLO (15%), against 24
  at the box's 512 (41%) and 33 at 1,024 (56%); the agents did 3,158 tokens a second over the first 240 s, against
  3,355 and 3,832.
- **Stock vLLM fails at every setting**: 41 to 58 of 59 replies over the SLO (69% to 98%), at either step cap and every
  static cap. Its best, one background request in flight, had a pooled p95 of 4.4 s.
- **The gateway's static cap does less for the change than the step cap**: 10 to 19 replies over (17% to 32%) at caps
  of 1 to 8 without it, and with the step cap of 256, 8 (cap 2) and 9 (cap 4), within a reply of the step cap alone.
- **The 96-token window does as the 48 does** at a step cap of 1,024: 33 replies over the SLO each.
- **No setting keeps 24 session trees within the requirement of at most 5% over the SLO**: the best, 8 of 59, is 14%.
  The rounds therefore also run 12 session trees, beside one call (a pooled card) and beside both (a consolidated
  card), and carry the step cap of 256 against W5's configuration and the frozen box at 24 to confirm the choice.

The FP8 point of the second screen (`sweep-screen2/sweep-disc-pr-ab48-fp8-s24`, the change with an FP8 KV cache in the
same 5 GiB) has a `run.json` and no summary: its engine died a second into the calls, a minute after the agents' replay
began profiling, and the point was stopped ([below](#the-fp8-kv-cache)). The campaign stopped there, so
`sweep-disc-pr-ab48-b256-fp8-s24` never ran.

## The FP8 KV cache

An FP8 KV cache (`--kv-cache-dtype fp8`) holds 72,816 tokens in the box's 5 GiB, against 36,400 in BF16, and moves
attention to FlashInfer, since FlashAttention 2 has no FP8 KV cache on this card; the quality gate's screen saw no
quality it costs ([2026-10-27-quality](../2026-10-27-quality)). Under E2's load, beside vLLM's CPU tier (the
`OffloadingConnector`, 20 GiB), the change's engine died both times it ran:

| Run | Phastform | The calls began | The engine died |
|---|---|---|---|
| [`sweep-screen2/sweep-disc-pr-ab48-fp8-s24`](sweep-screen2/sweep-disc-pr-ab48-fp8-s24) | v1.5 | 2026-10-27 23:58:47 UTC | 23:58:48 UTC |
| [`fp8-repro/sweep-disc-pr-ab48-fp8-s24`](fp8-repro/sweep-disc-pr-ab48-fp8-s24) | v1.6-21-ga4e6676 | 2026-10-28 01:37:07 UTC | 01:39:27 UTC |

Each time the engine core stopped on a CUDA fault raised where it waits for a step's output copy
(`vllm/v1/worker/gpu/async_utils.py`, `copy_event.synchronize()`), with a transfer of the CPU tier's in flight
(`gpu_worker.py`: "Failed to synchronize transfer end event; skipping 1 remaining transfers"), and the API server then
reported `EngineDeadError`: "an illegal memory access was encountered" the first time,
"operation not supported on global/shared address space" the second. CUDA reports such a fault at the next
synchronization rather than in the kernel that made it, so the journals do not name the kernel:
[`fp8-journal-1.txt`](fp8-journal-1.txt) and [`fp8-journal-2.txt`](fp8-journal-2.txt) hold the engine's lines from
its start to its death, home directories written as `~`. The second run is
[`fp8-repro.campaign`](../../e2/fp8-repro.campaign); its campaign file was a copy outside the repository, so its
`campaign.log` names no revision for its arm file, and `arm.toml` is the arm it ran. Each run was stopped once its
engine had died, since its turns then failed one by one at the reply timeout, so neither has a summary.

An FP8 KV cache beside the CPU tier is therefore not deployable on vLLM 0.32.0 on this card, and the serving profile
keeps its BF16 cache.
