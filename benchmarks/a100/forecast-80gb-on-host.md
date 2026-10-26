# Forecast: the A100 SXM4 80 GB against the frozen box

Written by `flatshare-sim forecast` (`make forecast-a100 GPU=a100-80gb HOST_JSON=host.json`) from the box's measured costs, the card's published peaks, the path vLLM 0.32.0 takes for the box's checkpoint on it (Marlin W8A16; `crates/flatshare-sim/src/hardware.rs`) and the host it is placed on, as `benchmarks/a100/bench_host.py` measured it: its CPU (Intel(R) Xeon(R) CPU @ 2.20GHz) runs the Python workload in 227.2 ms against the box's 381.1, and its link copies pinned host memory to the card at 12.32 GB/s in one copy against the box's 6.76. Every number for the A100 SXM4 80 GB is modeled; the box's are measured where they say so. On this host the API server's time is the box's over 1.68, and a restore's transfer runs at 92% of the host's one-copy rate, as the box's E0 restore does of the box's; the restore's fixed part, the mixed-step extra and the restore stall stay the box's. Three settings bound it: **low**, the peaks' ratios alone with Marlin as fast as a BF16 GEMM; **central**, with the box's 1.7 ms of step overhead kept, Marlin 1.42× a BF16 GEMM (vLLM PR #61224's L4) and FlashAttention 2 at its best published efficiency on the card; **high**, with FlashAttention 2 at its worst published. The E1 map's model, on which `flatshare-sim hardware` places every card, scales the fixed step with bandwidth and keeps the box's attention efficiency.

## Step and tier costs, Qwen3-4B-Instruct-2507-FP8

| Card | Costs | Step: fixed | per token | per pair | per KV token read | Restore | Cold prefill of 31,744 tokens |
|---|---|---:|---:|---:|---:|---|---:|
| RTX 5060 Ti 16 GB | measured | 11.56 ms | 47.99 µs | 14.17 ns | 429.3 ns | 10.3 ms + 23.64 µs/token | 10,239 ms |
| A100 SXM4 80 GB | the E1 map's | 2.54 ms | 41.33 µs | 2.15 ns | 94.3 ns | 10.3 ms + 5.91 µs/token | 2,813 ms |
| A100 SXM4 80 GB | low | 2.54 ms | 29.19 µs | 2.15 ns | 94.3 ns | 10.3 ms + 5.91 µs/token | 2,427 ms |
| A100 SXM4 80 GB | central | 3.88 ms | 41.33 µs | 2.59 ns | 94.3 ns | 10.3 ms + 5.91 µs/token | 3,123 ms |
| A100 SXM4 80 GB | high | 3.88 ms | 41.33 µs | 3.78 ns | 94.3 ns | 10.3 ms + 5.91 µs/token | 3,748 ms |
| A100 SXM4 80 GB | the E1 map's, on this host | 2.54 ms | 41.33 µs | 2.15 ns | 94.3 ns | 10.3 ms + 12.98 µs/token | 2,766 ms |
| A100 SXM4 80 GB | low, on this host | 2.54 ms | 29.19 µs | 2.15 ns | 94.3 ns | 10.3 ms + 12.98 µs/token | 2,380 ms |
| A100 SXM4 80 GB | central, on this host | 3.88 ms | 41.33 µs | 2.59 ns | 94.3 ns | 10.3 ms + 12.98 µs/token | 3,077 ms |
| A100 SXM4 80 GB | high, on this host | 3.88 ms | 41.33 µs | 3.78 ns | 94.3 ns | 10.3 ms + 12.98 µs/token | 3,702 ms |

## Step and tier costs, Qwen3-14B-FP8

| Card | Costs | Step: fixed | per token | per pair | per KV token read | Restore | Cold prefill of 31,744 tokens |
|---|---|---:|---:|---:|---:|---|---:|
| A100 SXM4 80 GB | the E1 map's | 7.62 ms | 150.21 µs | 2.99 ns | 104.8 ns | 10.3 ms + 6.57 µs/token | 7,040 ms |
| A100 SXM4 80 GB | low | 7.62 ms | 106.07 µs | 2.99 ns | 104.8 ns | 10.3 ms + 6.57 µs/token | 5,635 ms |
| A100 SXM4 80 GB | central | 8.96 ms | 150.21 µs | 3.60 ns | 104.8 ns | 10.3 ms + 6.57 µs/token | 7,438 ms |
| A100 SXM4 80 GB | high | 8.96 ms | 150.21 µs | 5.25 ns | 104.8 ns | 10.3 ms + 6.57 µs/token | 8,301 ms |
| A100 SXM4 80 GB | the E1 map's, on this host | 7.62 ms | 150.21 µs | 2.99 ns | 104.8 ns | 10.3 ms + 14.42 µs/token | 6,993 ms |
| A100 SXM4 80 GB | low, on this host | 7.62 ms | 106.07 µs | 2.99 ns | 104.8 ns | 10.3 ms + 14.42 µs/token | 5,587 ms |
| A100 SXM4 80 GB | central, on this host | 8.96 ms | 150.21 µs | 3.60 ns | 104.8 ns | 10.3 ms + 14.42 µs/token | 7,390 ms |
| A100 SXM4 80 GB | high, on this host | 8.96 ms | 150.21 µs | 5.25 ns | 104.8 ns | 10.3 ms + 14.42 µs/token | 8,254 ms |

## E0's cells

Time to first token of a request alone, in ms: the box's measured p50 (2026-10-22), and the forecast as that times the modeled ratio of the two cards, the card on this host and the box on its own, central (low–high).

| Prompt tokens | GPU hit: box | forecast | CPU restore: box | forecast | Recompute: box | forecast |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 45.0 | 23.3 (21.5–23.3) | 64.0 | 38.1 (36.8–38.1) | 135 | 85.2 (65.7–86.0) |
| 2,048 | 40.0 | 20.7 (19.3–20.8) | 95.0 | 55.3 (53.8–55.3) | 198 | 119 (88.9–121) |
| 4,096 | 48.0 | 24.9 (23.5–25.0) | 157 | 89.4 (87.8–89.5) | 437 | 237 (175–247) |
| 8,192 | 64.0 | 33.4 (31.9–33.5) | 281 | 157 (156–157) | 1,114 | 515 (381–554) |
| 16,384 | 97.0 | 50.8 (49.2–51.1) | 509 | 282 (280–282) | 3,264 | 1,224 (923–1,385) |
| 24,576 | 128 | 67.2 (65.5–67.7) | 722 | 398 (396–399) | 6,506 | 2,144 (1,640–2,512) |
| 31,744 | 156 | 82.0 (80.2–82.6) | 903 | 497 (495–497) | 10,239 | 3,118 (2,409–3,737) |

## The engine probes

- **The KV case** (`probe.py kv`). Stock vLLM's voice turn waits while two priority-2 requests decode into the pool's last free blocks: on Gray 4,923–5,077 ms, which the box's model brackets at 176–192 steps of 27.6 ms (4,861–5,303 ms). On the A100 SXM4 80 GB: 1,328–1,369 ms (low–high 1,082–1,377). make-room's first token, 225 and 297 ms on Gray against 107–137 ms on an idle pool: 126–145 ms (109–146).
- **The prefill case** (`probe.py prefill`). The reply's tokens come 69 ms apart behind the agents' chunks on stock vLLM and 17 ms apart with the budget cap holding them, on Gray (the medians' 1st to 16th tokens). On the A100 SXM4 80 GB: 32.8 ms (25.7–35.4) and 5.6 ms (3.7–5.6), with the chunks at 5,000 tokens of context.
- **The time-equivalent step cap** at the discriminating scenario's median request context (13,900 tokens), where the box's 512-token chunk takes 145 ms: 2,048 tokens (low, 130 ms); 1,536 tokens (central, 127 ms); 1,280 tokens (high, 129 ms). The long-prefill threshold is half the cap, as the box's 256 is of its 512.
- **The time-equivalent step cap** at the whole corpus's context (7,900 tokens), where the box's 512-token chunk takes 99 ms: 1,792 tokens (low, 90 ms); 1,280 tokens (central, 86 ms); 1,024 tokens (high, 80 ms). The long-prefill threshold is half the cap, as the box's 256 is of its 512.

## E2's scenarios on the regime map

Each cell replays E2's agents in `flatshare-sim`'s model of vLLM 0.32.0's two-tier KV cache and step loop with the card's central costs on this host; the discriminating scenario replays `weka-x32-long`, the calibration point the whole 32x corpus.

Means over seeds 11, 12, 13, 14, 15; `lru-cpu`, the stock engine's policy; a realtime tier of two calls. Running / working set: the time running agent requests / the agents' working set leave those calls short of KV. Busy: the card's step time over the replay's. Stalled: the requests that waited for free blocks.

| Scenario | Card | Sessions | Pool | CPU tier | Pressure | π | ρ | Running / working set | Busy | Stalled | Computed | Resume TTFT p50 / p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: the core, the box's pool | RTX 5060 Ti 16 GB (measured) | 24 | 36,400 tokens | 20 GiB | 9.1 / 10.0 | 27.8 | 0.17 | 17% / 100% | 100% | 99% | 97% | 59,331 / 79,873 ms |
| A: the core, the box's pool | A100 SXM4 80 GB (the E1 map's, on this host) | 24 | 36,400 tokens | 20 GiB | 9.0 / 10.0 | 49.6 | 0.25 | 9% / 100% | 66% | 49% | 39% | 968 / 8,045 ms |
| A: the core, the box's pool | A100 SXM4 80 GB (central, on this host) | 24 | 36,400 tokens | 20 GiB | 9.0 / 10.0 | 50.6 | 0.23 | 11% / 100% | 74% | 57% | 42% | 1,601 / 10,736 ms |
| the calibration point, the whole corpus | RTX 5060 Ti 16 GB (measured) | 16 | 36,400 tokens | 20 GiB | 3.5 / 4.0 | 22.8 | 0.25 | 3% / 100% | 53% | 12% | 6% | 158 / 1,041 ms |
| the calibration point, the whole corpus | A100 SXM4 80 GB (central, on this host) | 16 | 36,400 tokens | 20 GiB | 3.4 / 4.0 | 38.5 | 0.27 | 0% / 100% | 21% | 4% | 6% | 60 / 238 ms |
| A': the box's pool, more agents | A100 SXM4 80 GB (central, on this host) | 32 | 36,400 tokens | 20 GiB | 12.6 / 13.5 | 52.1 | 0.23 | 18% / 100% | 93% | 99% | 92% | 18,922 / 27,264 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 SXM4 80 GB (central, on this host) | 24 | 379,488 tokens | 232 GiB | 0.9 / 1.0 | 47.1 | 0.09 | 0% / 3% | 58% | 0% | 2% | 113 / 200 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 SXM4 80 GB (central, on this host) | 32 | 379,488 tokens | 232 GiB | 1.2 / 1.3 | 40.3 | 0.09 | 0% / 100% | 73% | 0% | 2% | 118 / 283 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 SXM4 80 GB (central, on this host) | 40 | 379,488 tokens | 232 GiB | 1.5 / 1.6 | 42.6 | 0.09 | 0% / 100% | 78% | 1% | 2% | 132 / 393 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 SXM4 80 GB (central, on this host) | 48 | 379,488 tokens | 232 GiB | 1.8 / 1.9 | 40.0 | 0.09 | 0% / 100% | 85% | 2% | 2% | 149 / 490 ms |
| B: twice the box's pool and sessions | A100 SXM4 80 GB (central, on this host) | 48 | 72,800 tokens | 40 GiB | 9.3 / 9.9 | 52.2 | 0.22 | 8% / 100% | 96% | 97% | 87% | 30,334 / 42,239 ms |
| D: Qwen3-4B at the card's own pool | A100 SXM4 80 GB (central, on this host) | 24 | 502,432 tokens | 276 GiB | 0.6 / 0.7 | 51.5 | 0.23 | 0% / 0% | 35% | 0% | 2% | 87 / 155 ms |

## What the forecast says the run will show

- **The core, at the box's pool and 24 sessions.** On the A100 SXM4 80 GB: pressure 9.0, running share 11%, 74% busy, 57% of requests stalled; on the box: pressure 9.1, running share 17%, 100% busy, 99% of requests stalled. Running agent requests leave two calls short of KV 11% of the time against the box's 17%, and such a wait, a decode-bound stretch, runs at 0.27 of the box's step time: the KV-blocking regime the change addresses, with shorter waits.
- **More agents at the box's pool.** 32 sessions: pressure 12.6, running share 18%, 93% busy, 99% of requests stalled.
- **Qwen3-14B-FP8 at the card's own pool.** 24 sessions: pressure 0.9, running share 0%, 58% busy, 0% of requests stalled; 32 sessions: pressure 1.2, running share 0%, 73% busy, 0% of requests stalled; 40 sessions: pressure 1.5, running share 0%, 78% busy, 1% of requests stalled; 48 sessions: pressure 1.8, running share 0%, 85% busy, 2% of requests stalled.
- **Twice the box's pool and sessions.** 48 sessions: pressure 9.3, running share 8%, 96% busy, 97% of requests stalled.
- **Qwen3-4B at the card's own pool.** 24 sessions: pressure 0.6, running share 0%, 35% busy, 0% of requests stalled.
- **The most time short of KV** among the A100 SXM4 80 GB's scenarios: A': the box's pool, more agents, 32 sessions (18%, against the box's 17% in the core).
