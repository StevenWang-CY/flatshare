# Forecast: the A100 SXM4 80 GB against the frozen box

Written by `flatshare-sim forecast` (`make forecast-a100 GPU=a100-80gb`) from the box's measured costs, the card's published peaks and the path vLLM 0.32.0 takes for the box's checkpoint on it (Marlin W8A16; `crates/flatshare-sim/src/hardware.rs`). Every number for the A100 SXM4 80 GB is modeled; the box's are measured where they say so. Three settings bound it: **low**, the peaks' ratios alone with Marlin as fast as a BF16 GEMM; **central**, with the box's 1.7 ms of step overhead kept, Marlin 1.42× a BF16 GEMM (vLLM PR #61224's L4) and FlashAttention 2 at its best published efficiency on the card; **high**, with FlashAttention 2 at its worst published. The E1 map's model, on which `flatshare-sim hardware` places every card, scales the fixed step with bandwidth and keeps the box's attention efficiency.

## Step and tier costs, Qwen3-4B-Instruct-2507-FP8

| Card | Costs | Step: fixed | per token | per pair | per KV token read | Restore | Cold prefill of 31,744 tokens |
|---|---|---:|---:|---:|---:|---|---:|
| RTX 5060 Ti 16 GB | measured | 11.56 ms | 47.99 µs | 14.17 ns | 429.3 ns | 10.3 ms + 23.64 µs/token | 10,239 ms |
| A100 SXM4 80 GB | the E1 map's | 2.54 ms | 41.33 µs | 2.15 ns | 94.3 ns | 10.3 ms + 5.91 µs/token | 2,813 ms |
| A100 SXM4 80 GB | low | 2.54 ms | 29.19 µs | 2.15 ns | 94.3 ns | 10.3 ms + 5.91 µs/token | 2,427 ms |
| A100 SXM4 80 GB | central | 3.88 ms | 41.33 µs | 2.59 ns | 94.3 ns | 10.3 ms + 5.91 µs/token | 3,123 ms |
| A100 SXM4 80 GB | high | 3.88 ms | 41.33 µs | 3.78 ns | 94.3 ns | 10.3 ms + 5.91 µs/token | 3,748 ms |

## Step and tier costs, Qwen3-14B-FP8

| Card | Costs | Step: fixed | per token | per pair | per KV token read | Restore | Cold prefill of 31,744 tokens |
|---|---|---:|---:|---:|---:|---|---:|
| A100 SXM4 80 GB | the E1 map's | 7.62 ms | 150.21 µs | 2.99 ns | 104.8 ns | 10.3 ms + 6.57 µs/token | 7,040 ms |
| A100 SXM4 80 GB | low | 7.62 ms | 106.07 µs | 2.99 ns | 104.8 ns | 10.3 ms + 6.57 µs/token | 5,635 ms |
| A100 SXM4 80 GB | central | 8.96 ms | 150.21 µs | 3.60 ns | 104.8 ns | 10.3 ms + 6.57 µs/token | 7,438 ms |
| A100 SXM4 80 GB | high | 8.96 ms | 150.21 µs | 5.25 ns | 104.8 ns | 10.3 ms + 6.57 µs/token | 8,301 ms |

## E0's cells

Time to first token of a request alone, in ms: the box's measured p50 (2026-10-22), and the forecast as that times the modeled ratio of the two cards, central (low–high).

| Prompt tokens | GPU hit: box | forecast | CPU restore: box | forecast | Recompute: box | forecast |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 45.0 | 35.4 (33.6–35.4) | 64.0 | 41.0 (39.7–41.1) | 135 | 98.4 (78.9–99.2) |
| 2,048 | 40.0 | 31.7 (30.3–31.7) | 95.0 | 52.8 (51.4–52.8) | 198 | 130 (100–133) |
| 4,096 | 48.0 | 38.6 (37.1–38.6) | 157 | 74.8 (73.3–74.9) | 437 | 251 (189–261) |
| 8,192 | 64.0 | 52.3 (50.7–52.4) | 281 | 117 (116–117) | 1,114 | 534 (400–573) |
| 16,384 | 97.0 | 80.6 (79.0–81.0) | 509 | 193 (191–193) | 3,264 | 1,254 (953–1,415) |
| 24,576 | 128 | 107 (106–108) | 722 | 264 (262–264) | 6,506 | 2,185 (1,681–2,553) |
| 31,744 | 156 | 131 (130–132) | 903 | 324 (322–324) | 10,239 | 3,169 (2,459–3,788) |

## The engine probes

- **The KV case** (`probe.py kv`). Stock vLLM's voice turn waits while two priority-2 requests decode into the pool's last free blocks: on Gray 4,923–5,077 ms, which the box's model brackets at 176–192 steps of 27.6 ms (4,861–5,303 ms). On the A100 SXM4 80 GB: 1,328–1,369 ms (low–high 1,082–1,377). make-room's first token, 225 and 297 ms on Gray against 107–137 ms on an idle pool: 126–145 ms (109–146).
- **The prefill case** (`probe.py prefill`). The reply's tokens come 69 ms apart behind the agents' chunks on stock vLLM and 17 ms apart with the budget cap holding them, on Gray (the medians' 1st to 16th tokens). On the A100 SXM4 80 GB: 32.8 ms (25.7–35.4) and 5.6 ms (3.7–5.6), with the chunks at 5,000 tokens of context.
- **The time-equivalent step cap** at the discriminating scenario's median request context (13,900 tokens), where the box's 512-token chunk takes 145 ms: 2,048 tokens (low, 130 ms); 1,536 tokens (central, 127 ms); 1,280 tokens (high, 129 ms). The long-prefill threshold is half the cap, as the box's 256 is of its 512.
- **The time-equivalent step cap** at the whole corpus's context (7,900 tokens), where the box's 512-token chunk takes 99 ms: 1,792 tokens (low, 90 ms); 1,280 tokens (central, 86 ms); 1,024 tokens (high, 80 ms). The long-prefill threshold is half the cap, as the box's 256 is of its 512.

## E2's scenarios on the regime map

Each cell replays E2's agents in `flatshare-sim`'s model of vLLM 0.32.0's two-tier KV cache and step loop with the card's central costs; the discriminating scenario replays `weka-x32-long`, the calibration point the whole 32x corpus.

Means over seeds 11; `lru-cpu`, the stock engine's policy; a realtime tier of two calls. Running / working set: the time running agent requests / the agents' working set leave those calls short of KV. Busy: the card's step time over the replay's. Stalled: the requests that waited for free blocks.

| Scenario | Card | Sessions | Pool | CPU tier | Pressure | π | ρ | Running / working set | Busy | Stalled | Computed | Resume TTFT p50 / p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: the core, the box's pool | RTX 5060 Ti 16 GB (measured) | 24 | 36,400 tokens | 20 GiB | 1.4 / 1.5 | 36.2 | 0.29 | 0% / 100% | 63% | 0% | 13% | 88 / 366 ms |
| A: the core, the box's pool | A100 SXM4 80 GB (the E1 map's) | 24 | 36,400 tokens | 20 GiB | 1.3 / 1.5 | 70.5 | 0.19 | 0% / 100% | 23% | 0% | 14% | 58 / 184 ms |
| A: the core, the box's pool | A100 SXM4 80 GB (central) | 24 | 36,400 tokens | 20 GiB | 1.3 / 1.5 | 70.5 | 0.18 | 0% / 100% | 31% | 0% | 14% | 62 / 176 ms |
| the calibration point, the whole corpus | RTX 5060 Ti 16 GB (measured) | 16 | 36,400 tokens | 20 GiB | 0.9 / 1.0 | 34.7 | 0.29 | 0% / 20% | 58% | 0% | 17% | 91 / 299 ms |
| the calibration point, the whole corpus | A100 SXM4 80 GB (central) | 16 | 36,400 tokens | 20 GiB | 0.9 / 1.0 | 80.5 | 0.18 | 0% / 14% | 23% | 0% | 16% | 64 / 247 ms |
| A': the box's pool, more agents | A100 SXM4 80 GB (central) | 32 | 36,400 tokens | 20 GiB | 1.8 / 1.9 | 94.3 | 0.18 | 0% / 100% | 39% | 0% | 16% | 74 / 555 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 SXM4 80 GB (central) | 24 | 379,488 tokens | 232 GiB | 0.1 / 0.1 | 68.8 | 0.07 | 0% / 0% | 54% | 0% | 14% | 85 / 402 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 SXM4 80 GB (central) | 32 | 379,488 tokens | 232 GiB | 0.2 / 0.2 | 90.6 | 0.07 | 0% / 0% | 60% | 0% | 16% | 108 / 524 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 SXM4 80 GB (central) | 40 | 379,488 tokens | 232 GiB | 0.2 / 0.2 | 76.6 | 0.07 | 0% / 0% | 70% | 0% | 15% | 109 / 695 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 SXM4 80 GB (central) | 48 | 379,488 tokens | 232 GiB | 0.3 / 0.3 | 85.8 | 0.07 | 0% / 0% | 71% | 0% | 16% | 141 / 1,098 ms |
| B: twice the box's pool and sessions | A100 SXM4 80 GB (central) | 48 | 72,800 tokens | 40 GiB | 1.4 / 1.4 | 84.1 | 0.18 | 0% / 100% | 46% | 0% | 16% | 81 / 1,006 ms |
| D: Qwen3-4B at the card's own pool | A100 SXM4 80 GB (central) | 24 | 502,432 tokens | 276 GiB | 0.1 / 0.1 | 70.5 | 0.18 | 0% / 0% | 31% | 0% | 14% | 62 / 186 ms |

## What the forecast says the run will show

- **The core, at the box's pool and 24 sessions.** On the A100 SXM4 80 GB: pressure 1.3, running share 0%, 31% busy, 0% of requests stalled; on the box: pressure 1.4, running share 0%, 63% busy, 0% of requests stalled. Running agent requests leave two calls short of KV 0% of the time against the box's 0%, and such a wait, a decode-bound stretch, runs at 0.27 of the box's step time: the KV-blocking regime the change addresses, with shorter waits.
- **More agents at the box's pool.** 32 sessions: pressure 1.8, running share 0%, 39% busy, 0% of requests stalled.
- **Qwen3-14B-FP8 at the card's own pool.** 24 sessions: pressure 0.1, running share 0%, 54% busy, 0% of requests stalled; 32 sessions: pressure 0.2, running share 0%, 60% busy, 0% of requests stalled; 40 sessions: pressure 0.2, running share 0%, 70% busy, 0% of requests stalled; 48 sessions: pressure 0.3, running share 0%, 71% busy, 0% of requests stalled.
- **Twice the box's pool and sessions.** 48 sessions: pressure 1.4, running share 0%, 46% busy, 0% of requests stalled.
- **Qwen3-4B at the card's own pool.** 24 sessions: pressure 0.1, running share 0%, 31% busy, 0% of requests stalled.
- **The most time short of KV** among the A100 SXM4 80 GB's scenarios: D: Qwen3-4B at the card's own pool, 24 sessions (0%, against the box's 0% in the core).
