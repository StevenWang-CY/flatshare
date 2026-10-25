# Forecast: the A100 40 GB against the frozen box

Written by `lh-sim forecast` (`make forecast-a100`) from the box's measured costs, the card's published peaks and the path vLLM 0.32.0 takes for the box's checkpoint on it (Marlin W8A16; `crates/lh-sim/src/hardware.rs`). Every number for the A100 40 GB is modeled; the box's are measured where they say so. Three settings bound it: **low**, the peaks' ratios alone with Marlin as fast as a BF16 GEMM; **central**, with the box's 1.7 ms of step overhead kept, Marlin 1.42× a BF16 GEMM (vLLM PR #61224's L4) and FlashAttention 2 at its best published efficiency on the card; **high**, with FlashAttention 2 at its worst published. The E1 map's model, on which `lh-sim hardware` places every card, scales the fixed step with bandwidth and keeps the box's attention efficiency.

## Step and tier costs, Qwen3-4B-Instruct-2507-FP8

| Card | Costs | Step: fixed | per token | per pair | per KV token read | Restore | Cold prefill of 31,744 tokens |
|---|---|---:|---:|---:|---:|---|---:|
| RTX 5060 Ti 16 GB | measured | 11.56 ms | 47.99 µs | 14.17 ns | 429.3 ns | 10.3 ms + 23.64 µs/token | 10,239 ms |
| A100 40 GB | the E1 map's | 3.33 ms | 41.33 µs | 2.15 ns | 123.7 ns | 10.3 ms + 5.91 µs/token | 2,892 ms |
| A100 40 GB | low | 3.33 ms | 29.19 µs | 2.15 ns | 123.7 ns | 10.3 ms + 5.91 µs/token | 2,506 ms |
| A100 40 GB | central | 4.55 ms | 41.33 µs | 2.59 ns | 123.7 ns | 10.3 ms + 5.91 µs/token | 3,196 ms |
| A100 40 GB | high | 4.55 ms | 41.33 µs | 3.78 ns | 123.7 ns | 10.3 ms + 5.91 µs/token | 3,820 ms |

## Step and tier costs, Qwen3-14B-FP8

| Card | Costs | Step: fixed | per token | per pair | per KV token read | Restore | Cold prefill of 31,744 tokens |
|---|---|---:|---:|---:|---:|---|---:|
| A100 40 GB | the E1 map's | 9.99 ms | 150.21 µs | 2.99 ns | 137.4 ns | 10.3 ms + 6.57 µs/token | 7,221 ms |
| A100 40 GB | low | 9.99 ms | 106.07 µs | 2.99 ns | 137.4 ns | 10.3 ms + 6.57 µs/token | 5,816 ms |
| A100 40 GB | central | 11.21 ms | 150.21 µs | 3.60 ns | 137.4 ns | 10.3 ms + 6.57 µs/token | 7,612 ms |
| A100 40 GB | high | 11.21 ms | 150.21 µs | 5.25 ns | 137.4 ns | 10.3 ms + 6.57 µs/token | 8,475 ms |

## E0's cells

Time to first token of a request alone, in ms: the box's measured p50 (2026-10-22), and the forecast as that times the modeled ratio of the two cards, central (low–high).

| Prompt tokens | GPU hit: box | forecast | CPU restore: box | forecast | Recompute: box | forecast |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 45.0 | 36.2 (34.5–36.2) | 64.0 | 41.7 (40.4–41.7) | 135 | 100 (81.0–101) |
| 2,048 | 40.0 | 32.4 (31.1–32.4) | 95.0 | 53.5 (52.2–53.5) | 198 | 133 (103–135) |
| 4,096 | 48.0 | 39.3 (37.9–39.4) | 157 | 75.6 (74.2–75.7) | 437 | 257 (196–266) |
| 8,192 | 64.0 | 53.2 (51.8–53.3) | 281 | 118 (117–118) | 1,114 | 546 (415–586) |
| 16,384 | 97.0 | 81.8 (80.3–82.1) | 509 | 194 (193–195) | 3,264 | 1,284 (986–1,445) |
| 24,576 | 128 | 109 (107–109) | 722 | 265 (264–266) | 6,506 | 2,236 (1,737–2,604) |
| 31,744 | 156 | 133 (131–134) | 903 | 325 (324–326) | 10,239 | 3,242 (2,540–3,861) |

## The engine probes

- **The KV case** (`probe.py kv`). Stock vLLM's voice turn waits while two priority-2 requests decode into the pool's last free blocks: on Gray 4,923–5,077 ms, which the box's model brackets at 176–192 steps of 27.6 ms (4,861–5,303 ms). On the A100 40 GB: 1,636–1,688 ms (low–high 1,411–1,695). make-room's first token, 225 and 297 ms on Gray against 107–137 ms on an idle pool: 134–158 ms (117–158).
- **The prefill case** (`probe.py prefill`). The reply's tokens come 69 ms apart behind the agents' chunks on stock vLLM and 17 ms apart with the budget cap holding them, on Gray (the medians' 1st to 16th tokens). On the A100 40 GB: 33.5 ms (26.5–36.1) and 6.6 ms (4.9–6.6), with the chunks at 5,000 tokens of context.
- **The time-equivalent step cap** at the discriminating scenario's median request context (13,900 tokens), where the box's 512-token chunk takes 145 ms: 2,048 tokens (low, 131 ms); 1,536 tokens (central, 128 ms); 1,280 tokens (high, 130 ms). The long-prefill threshold is half the cap, as the box's 256 is of its 512.
- **The time-equivalent step cap** at the whole corpus's context (7,900 tokens), where the box's 512-token chunk takes 99 ms: 1,792 tokens (low, 91 ms); 1,280 tokens (central, 87 ms); 1,024 tokens (high, 81 ms). The long-prefill threshold is half the cap, as the box's 256 is of its 512.

## E2's scenarios on the regime map

Each cell replays E2's agents in `lh-sim`'s model of vLLM 0.32.0's two-tier KV cache and step loop with the card's central costs; the discriminating scenario replays `weka-x32-long`, the calibration point the whole 32x corpus.

Means over seeds 11, 12, 13, 14, 15; `lru-cpu`, the stock engine's policy; a realtime tier of two calls. Running / working set: the time running agent requests / the agents' working set leave those calls short of KV. Busy: the card's step time over the replay's. Stalled: the requests that waited for free blocks.

| Scenario | Card | Sessions | Pool | CPU tier | Pressure | π | ρ | Running / working set | Busy | Stalled | Computed | Resume TTFT p50 / p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: the core, the box's pool | RTX 5060 Ti 16 GB (measured) | 24 | 36,400 tokens | 20 GiB | 9.1 / 10.0 | 27.8 | 0.17 | 17% / 100% | 100% | 99% | 97% | 59,331 / 79,873 ms |
| A: the core, the box's pool | A100 40 GB (the E1 map's) | 24 | 36,400 tokens | 20 GiB | 9.0 / 10.0 | 97.3 | 0.12 | 11% / 100% | 70% | 50% | 41% | 1,349 / 9,597 ms |
| A: the core, the box's pool | A100 40 GB (central) | 24 | 36,400 tokens | 20 GiB | 9.0 / 10.0 | 100.0 | 0.11 | 13% / 100% | 76% | 57% | 43% | 2,003 / 11,334 ms |
| the calibration point, the whole corpus | RTX 5060 Ti 16 GB (measured) | 16 | 36,400 tokens | 20 GiB | 3.5 / 4.0 | 22.8 | 0.25 | 3% / 100% | 53% | 12% | 6% | 158 / 1,041 ms |
| the calibration point, the whole corpus | A100 40 GB (central) | 16 | 36,400 tokens | 20 GiB | 3.4 / 4.0 | 69.2 | 0.14 | 0% / 100% | 24% | 3% | 6% | 89 / 244 ms |
| A': the box's pool, more agents | A100 40 GB (central) | 32 | 36,400 tokens | 20 GiB | 12.6 / 13.5 | 98.9 | 0.11 | 18% / 100% | 95% | 99% | 94% | 21,698 / 29,949 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 40 GB (central) | 24 | 117,344 tokens | 72 GiB | 2.8 / 3.1 | 92.3 | 0.04 | 0% / 100% | 66% | 4% | 2% | 207 / 490 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 40 GB (central) | 32 | 117,344 tokens | 72 GiB | 3.9 / 4.2 | 77.8 | 0.04 | 2% / 100% | 84% | 11% | 5% | 312 / 2,429 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 40 GB (central) | 40 | 117,344 tokens | 72 GiB | 4.9 / 5.1 | 83.8 | 0.04 | 8% / 100% | 92% | 45% | 13% | 1,545 / 12,458 ms |
| C': Qwen3-14B-FP8 at its own pool | A100 40 GB (central) | 48 | 117,344 tokens | 72 GiB | 5.8 / 6.1 | 91.3 | 0.04 | 6% / 100% | 96% | 90% | 43% | 35,767 / 82,774 ms |
| B: twice the box's pool and sessions | A100 40 GB (central) | 48 | 72,800 tokens | 40 GiB | 9.3 / 9.9 | 99.2 | 0.11 | 8% / 100% | 96% | 95% | 87% | 32,136 / 45,939 ms |
| D: Qwen3-4B at the card's own pool | A100 40 GB (central) | 24 | 211,168 tokens | 116 GiB | 1.5 / 1.7 | 99.9 | 0.11 | 0% / 100% | 40% | 1% | 2% | 151 / 321 ms |

## What the forecast says the run will show

- **The core, at the box's pool and 24 sessions.** On the A100 40 GB: pressure 9.0, running share 13%, 76% busy, 57% of requests stalled; on the box: pressure 9.1, running share 17%, 100% busy, 99% of requests stalled. Running agent requests leave two calls short of KV 13% of the time against the box's 17%, and such a wait, a decode-bound stretch, runs at 0.33 of the box's step time: the KV-blocking regime the change addresses, with shorter waits.
- **More agents at the box's pool.** 32 sessions: pressure 12.6, running share 18%, 95% busy, 99% of requests stalled.
- **Qwen3-14B-FP8 at the card's own pool.** 24 sessions: pressure 2.8, running share 0%, 66% busy, 4% of requests stalled; 32 sessions: pressure 3.9, running share 2%, 84% busy, 11% of requests stalled; 40 sessions: pressure 4.9, running share 8%, 92% busy, 45% of requests stalled; 48 sessions: pressure 5.8, running share 6%, 96% busy, 90% of requests stalled.
- **Twice the box's pool and sessions.** 48 sessions: pressure 9.3, running share 8%, 96% busy, 95% of requests stalled.
- **Qwen3-4B at the card's own pool.** 24 sessions: pressure 1.5, running share 0%, 40% busy, 1% of requests stalled.
- **The most time short of KV** among the A100 40 GB's scenarios: A': the box's pool, more agents, 32 sessions (18%, against the box's 17% in the core).
