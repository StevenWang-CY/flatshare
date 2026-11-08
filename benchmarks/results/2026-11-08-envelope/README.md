# The envelope rounds (2026-11-08)

What the serving profile gains from ordering the agents' waiting queue by what each request has left to compute, and from
loading the CPU tier through the SM copy kernel: three rounds of the card's envelope beside the two calls, by the
resource gate's rule ([RESOURCE-GATE.md](../../quality/RESOURCE-GATE.md)), and the Gray windows before them that found
the queue's gain and the cause of its voice tail. Version 1.24 serves both (`queue_max_wait_s = 120`,
`sm_load_programs = 4` in [the serving profile](../../../deploy/profiles/rtx-5060-ti-serve.toml)).

## The gate

<!-- table.py gate -->
| Configuration | Session trees | Rounds | Replies over the SLO | Qualifies | V2V p50, ms | V2V p95 | Agents' tokens/s | Agents' TTFT p50, s | p99 | Computed |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| The serving profile through v1.23 | 4 | 3 | 4 of 177 (2.3%) | yes | 1,635<br><sub>1,618⁠–⁠1,642</sub> | 1,934<br><sub>1,852⁠–⁠2,008</sub> | 3,278<br><sub>2,703⁠–⁠3,598</sub> | 0.3<br><sub>0.3⁠–⁠0.7</sub> | 8<br><sub>6⁠–⁠8</sub> | 2.4%<br><sub>2.0%⁠–⁠3.1%</sub> |
| The queue and the SM copy | 12 | 3 | 4 of 177 (2.3%) | yes | 1,753<br><sub>1,659⁠–⁠1,765</sub> | 1,905<br><sub>1,876⁠–⁠1,959</sub> | 6,906<br><sub>5,949⁠–⁠7,090</sub> | 2.3<br><sub>0.5⁠–⁠2.9</sub> | 24<br><sub>19⁠–⁠73</sub> | 14.2%<br><sub>13.1%⁠–⁠14.7%</sub> |
| The queue and the SM copy | 24 | 3 | 3 of 177 (1.7%) | yes | 1,690<br><sub>1,674⁠–⁠1,712</sub> | 1,872<br><sub>1,865⁠–⁠1,887</sub> | 7,253<br><sub>4,490⁠–⁠8,094</sub> | 4.3<br><sub>3.8⁠–⁠4.5</sub> | 133<br><sub>132⁠–⁠182</sub> | 26.1%<br><sub>22.8%⁠–⁠42.7%</sub> |
<!-- /table.py gate -->

<!-- table.py rounds -->
| Round | Node | Voice alone V2V p95, ms | The SLO | The serving profile through v1.23, 4 trees | The queue and the SM copy, 12 trees | The queue and the SM copy, 24 trees |
|---|---|---:|---:|---:|---:|---:|
| env-r1 | phastform | 1,814 | 2,064 | 0 of 59, 3,278 tokens/s | 2 of 59, 5,949 tokens/s | 0 of 59, 8,094 tokens/s |
| env-r2 | gray | 1,714 | 1,964 | 2 of 59, 2,703 tokens/s | 2 of 59, 7,090 tokens/s | 3 of 59, 7,253 tokens/s |
| env-r3 | phastform | 1,817 | 2,067 | 2 of 59, 3,598 tokens/s | 0 of 59, 6,906 tokens/s | 0 of 59, 4,490 tokens/s |
<!-- /table.py rounds -->

## First come first served over the LRU tier

The agents' sessions grow one turn at a time, and each turn's prompt is the session so far. Through version 1.23 the
waiting queue takes them in arrival order. Beside 24 session trees a request waits long enough that the CPU tier, which
evicts the least recently used chunks, has evicted its session's prefix by the time it is scheduled, so it computes its
whole context again, which makes the next request wait longer: the tier's cyclic worst case. On Gray 89.5% of the
prompt tokens, nearly all of them the agents', were computed at their first schedule, though computed a turn before, and the agents'
time to first token was 77 s at the median. A tier of 30 GiB in place of 20 does not leave it (68.0% computed, the token
rate 7% lower).

The queue takes first, within a priority, the request with the least left to compute, counting what the GPU pool and the
CPU tier hold now, so the sessions whose prefix is still cached are served while it is, which keeps it cached: 20% of the
prompt tokens computed, the agents' token rate 2.3 times the control's and their time to first token 4.5 s. A request that
has waited 120 s goes ahead of the rest, the longest waiting first, so no long prompt starves. The calls are priority 0
and keep their arrival order.

<!-- table.py windows -->
| Window | Point | Agents' tokens/s | Against the control | Computed | Loaded from the CPU tier, GB | Agents' TTFT p50, s | Calls' first byte p95, ms | LLM first sentence p95 | V2V p50 | V2V p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gray, window 1 (11-07) | v1.23, the control | 3,173 | — | 89.5% | 2.2 | 76.8 | 202 | 403 | 1,711 | 1,843 |
| Gray, window 1 (11-07) | a 30 GiB CPU tier | 2,958 | −7% | 68.0% | 37.7 | 76.7 | 255 | 456 | 1,700 | 1,840 |
| Gray, window 1 (11-07) | the queue, every count refreshed each step | 7,144 | +125% | 23.5% | 274.2 | 4.7 | 301 | 630 | 1,787 | 2,051 |
| Gray, window 1 (11-07) | the queue, two counts a step, none beside a call | 7,270 | +129% | 20.2% | 306.5 | 4.5 | 324 | 471 | 1,664 | 2,025 |
| Gray, window 1 (11-07) | the queue (every count) and a 30 GiB tier | 8,622 | +172% | 11.7% | 368.0 | 4.6 | 405 | 825 | 1,762 | 2,226 |
| Gray, window 2 (11-08) | v1.23, the control | 3,089 | — | 89.8% | 2.0 | 76.2 | 239 | 437 | 1,694 | 1,835 |
| Gray, window 2 (11-08) | the queue and the SM copy | 7,849 | +154% | 20.3% | 324.1 | 4.6 | 183 | 454 | 1,742 | 1,924 |
| Gray, window 2 (11-08) | the queue, the SM copy and a 30 GiB tier | 9,537 | +209% | 12.4% | 399.7 | 3.6 | 200 | 506 | 1,737 | 1,994 |
| Gray, the whole points (11-08) | v1.23, the control | 3,113 | — | 89.9% | 2.4 | 75.8 | 232 | 408 | 1,669 | 1,915 |
| Gray, the whole points (11-08) | the queue and the SM copy | 7,777 | +150% | 21.3% | 286.4 | 4.1 | 211 | 462 | 1,700 | 1,927 |
| Gray, the whole points (11-08) | endpointing 0.2/0.65 s alone (not in the profile) | 2,932 | −6% | 88.8% | 2.4 | 78.0 | 225 | 477 | 1,342 | 1,615 |
| Gray, the whole points (11-08) | the queue, the SM copy and endpointing 0.2/0.65 s | 7,912 | +154% | 22.6% | 303.2 | 3.7 | 179 | 435 | 1,343 | 1,582 |
| Gray, the whole points (11-08) | v1.23, 4 callers | 2,643 | — | 84.0% | 3.4 | 84.9 | 234 | 516 | 1,723 | 2,016 |
| Gray, the whole points (11-08) | the queue, the SM copy and endpointing 0.2/0.65 s, 4 callers | 5,876 | +122% | 24.6% | 252.2 | 5.7 | 247 | 479 | 1,385 | 1,649 |
<!-- /table.py windows -->

Each window's points ran on one sample of the agents (seed 36), each against its own window's control, and the controls'
V2V p95 moved by 80 ms between windows (1,843, 1,835 and 1,915 ms).

## The voice tail: the copy engine on PCIe 3.0 x8

With the queue alone the calls' tail grew: V2V p95 2,025 ms against the control's 1,843, the calls' first byte p95 324 ms
against 202. The queue loads from the CPU tier far more (307 GB in its point against the control's 2.2), and both boxes' cards sit on
PCIe 3.0 x8 links (6.8 GB/s). Probes on Gray's bare card with the engine's own packages ([`probes/`](probes), each
script with its output), a step (a 64 KiB input copy, a 5.7 ms kernel, a 64 B output copy) beside one 1.46 GB load as
vLLM 0.32.0 issues it (`swap_blocks_batch`, one `cuMemcpyBatchAsync`):

- the step waits for the load's copy to end before its own input reaches the card: up to 192 ms with 32 KiB pages and
  209 ms with 2 MiB ones, with the load submitted from another thread in 2 ms, so the wait is the copy engine's, not the
  host's; the load cut into 64 MB or 16 MB jobs on chained streams still holds the step for up to 152 and 136 ms;
- the same load through vLLM's own Triton kernel for small pages (its path for pages under 28 KiB, which the box's 64 KiB
  pages do not take) with 4 programs moves 6.1 GB/s, and the steps beside it take 6.1 ms against 5.7 alone; with 12
  programs a step took up to 19.2 ms.

`flatshare_copy` routes the tier's loads through that kernel with 4 programs. With it the calls' first byte p95 fell to
183 ms beside the queue (window 2), and the agents' token rate rose further, as steps no longer stall behind loads.

## The trial's round

Before version 1.24's rounds the same configurations ran from the trial's modules on Gray (seed 22). A review of that code
then found that its count of a request's CPU-tier hit could overcount by a whole prompt: it found the held chunks by
bisection, which assumes the tier holds a prefix's head whenever it holds its tail, and vLLM's LRU does not keep that (a
turn that finds part of its prefix on the GPU refreshes only the chunks it loads or stores, so a hole opens in the
prefix's middle while its tail stays). vLLM loads nothing past such a hole, and the bisection counted to the tail's end.
Version 1.24 counts as vLLM's lookup does, and its rounds run the release's own code; the trial's round stands here
outside the gate.

<!-- table.py trial -->
| Configuration | Session trees | Rounds | Replies over the SLO | Qualifies | V2V p50, ms | V2V p95 | Agents' tokens/s | Agents' TTFT p50, s | p99 | Computed |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| The serving profile through v1.23 | 4 | 1 | 5 of 59 (8.5%) | no | 1,630 | 1,992 | 2,715 | 0.3 | 6 | 3.1% |
| The trial's queue and SM copy | 12 | 1 | 0 of 59 (0.0%) | yes | 1,680 | 1,838 | 7,156 | 2.6 | 30 | 12.0% |
| The trial's queue and SM copy | 24 | 1 | 5 of 59 (8.5%) | no | 1,764 | 1,962 | 6,919 | 7.2 | 138 | 26.2% |
<!-- /table.py trial -->

## flatshare_jit's early finish

flatshare_jit schedules the next batch when the one in flight is due to end, and ends the wait early when the step is
done. It reads Model Runner V1's event; the box runs Model Runner V2, whose output names its event otherwise, so the early
finish never fired in any measurement of the serving profile. Reading V2's as well, in the trial's round, adjacent to the
same configuration without it:

<!-- table.py jit -->
| flatshare_jit's early finish | Agents' tokens/s | Agents' TTFT p50, s | LLM first token p95, ms | LLM first sentence p95 | V2V p50 | V2V p95 |
|---|---:|---:|---:|---:|---:|---:|
| off (v1.23, as measured) | 3,085 | 82.1 | 137 | 389 | 1,653 | 1,850 |
| on (Model Runner V2's event read) | 3,193 | 83.0 | 146 | 427 | 1,654 | 1,846 |
<!-- /table.py jit -->

Every difference is within one run's noise: beside 24 session trees the steps are prefill-bound and the step model's
deadline sits near the real step's end. The profile keeps the behaviour it was measured with.

## Files

- `env-r1`, `env-r2`, `env-r3`: version 1.24's rounds ([env-r1.campaign](../../e2/env-r1.campaign) and the next two).
  `env-r1` and `env-r2` ran the release candidate at c445329. `env-r3` ran a working commit of the release's branch,
  254bc89 in its `run.json` (`versions.source`), since a release of c445329 built from another worktree would not deploy
  over the one already on the hosts. Its serve module and the three it installs from are c445329's code with only their
  docstrings changed, as in the release, and the patches, the arms and the campaigns are c445329's.
- `opt2-env-r2`: the trial's round, with the early finish's pair.
- `opt2-w1g`, `opt2-w1g-fix`, `opt2-w2g`, `opt2-i5g`: Gray's windows. `opt2-w1g-fix` is the queue after its count's
  refresh was bounded to two requests a step and none beside a call.
- `probes/`: the copy probes on Gray's bare card, each script with its output (`opt2_dma_probe4.py` checked the SM
  kernel's bytes against the copy engine's on 1.46 GB before window 2).
- `count-cost.out`: what one count of a waiting request's cached tokens costs the host, and every one of 24 at once
  (`python/flatshare_vllm/tests/bench_queue_count.py`, four runs on peacock): 1.0–2.5 ms and 24–60 ms at 20–30k tokens
  held on the GPU, which is why a step refreshes at most two.
- Each point's compact record (`flatshare-bench e2 export`): `run.json`, `arm.toml`, the callers' `bots/`, `summary.json`
  and the engine's `engine.json`. The trial's points name the trial's scheduler class (`opt2_srpf.AsyncScheduler`) and
  its keys (`opt2.srpf_max_wait_s`, `opt2.sm_load_programs`) in place of the serve module's.
- [table.py](table.py): the tables above, from the records.

```
cd python/flatshare_bench && uv run python ../../benchmarks/results/2026-11-08-envelope/table.py
```
