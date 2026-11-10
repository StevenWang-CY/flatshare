# flatshare-sim

`flatshare-sim` replays the SemiAnalysis Claude Code traces as AIPerf 0.13.0's `agentic_replay` sends them, through a
model of vLLM 0.32's GPU prefix cache, its OffloadingConnector CPU tier and its step loop on the frozen box. It is for a
reader who runs the simulator, checks its calibration, or reads its placements of other GPUs. For each cache policy a
run reports where prompt tokens came from, what that costs in time to first token and PCIe traffic, and where the run
sits in E1's regime coordinates. A clairvoyant reference gives the gap each policy leaves. E1's results are in
[BENCHMARKS.md](../../docs/BENCHMARKS.md#does-knowing-when-an-agent-returns-help), and
[ARCHITECTURE.md §6.4](../../docs/ARCHITECTURE.md#64-the-simulator-e1) places the simulator in the system.

## Usage

`DATA/weka-x32` is the 32× corpus, SemiAnalysis's traces as `flatshare-bench scale-traces` writes them
([scaling](../../python/flatshare_bench/README.md#trace-scaling), and the download in its
[quick start](../../python/flatshare_bench/README.md#quick-start-cyan)). It stays outside the repository.

```sh
cargo build --release -p flatshare-sim
# the W0 configuration, the stock arms; then the CPU tier at 2x, 4x and 8x the pool
flatshare-sim run DATA/weka-x32 --json w0.json --csv w0.csv
flatshare-sim run DATA/weka-x32 --cpu-ratio 2,4,8 --arms lru-cpu,clairvoyant
# the regime map, seeds 11-13; then BENCHMARKS' realistic regime over 20 seeds
flatshare-sim regime DATA/weka-x32 --csv regime.csv --markdown regime.md
flatshare-sim regime DATA/weka-x32 --sessions 8,12,16,24 --cpu-ratio 4 --pause-sources trace \
  --seeds $(seq 11 30 | paste -sd, -)
# other GPUs, modeled: on the 32x replay, and each at its own scaling
flatshare-sim hardware DATA/weka-x32 --markdown hw.md
flatshare-sim hardware DATA/weka-x32 --fit --markdown hw-fit.md
# a GPU run's forecast; miss-ratio curves
flatshare-sim forecast DATA/weka-x32 DATA/weka-x32-long --markdown forecast.md
flatshare-sim mrc DATA/weka-x32 --service idle --markdown mrc.md
# a measured run through the cache model, then served by the step loop
flatshare-sim replay-log DATA/weka-x32 runs/w0/aiperf/profile_export.jsonl --timed --csv w0-timed.csv
```

`make reproduce-e1 CORPUS=DATA/weka-x32` writes every E1 table, BENCHMARKS' and this README's other hardware, to
`build/reproduce-e1`. `flatshare-sim COMMAND --help` lists every option with its default.

| Command | What it runs, by default |
|---|---|
| `run` | `gpu-lru`, `lru-cpu` and `clairvoyant`, or the arms of `--arms`, at each CPU tier size |
| `regime` | `lru-cpu`, the E1 arms and the reference at 5–40 sessions, CPU tiers of 2, 4 and 8× the pool, both pause sources and seeds 11–13: the regime map |
| `hardware` | The same on four GPUs at 5–256 sessions, with the CPU tier 4× each pool and a realtime tier of two calls ([below](#other-hardware-modeled)) |
| `forecast` | A run on another GPU, before it runs ([below](#a-gpu-runs-forecast)) |
| `mrc` | Miss-ratio curves of the GPU tier: LRU from Mattson stack distances, MIN at 0.25 to 8 times the pool |
| `replay-log` | A measured run's requests through the cache model at their measured times; with `--timed`, served by the step loop |

In `regime`'s CSV each arm closes `(lru − arm) / (lru − clairvoyant)` of the clairvoyant gap, left empty where the
reference is less than 5% faster than `lru-cpu`. The replay's options default to W0's settings, not to those of
`flatshare-bench`'s `configs/agentic_replay.yaml`:

| Option | Default | Meaning |
|---|---|---|
| `--concurrency` | 16 | Live agent sessions, AIPerf's lanes |
| `--duration`, `--grace` | 600, 60 | The profiling phase and the grace after it, in seconds |
| `--think-cap`, `--idle-cap` | 120, 10 | Caps in seconds on each think time and on the engine's idle gaps; 0 for none |
| `--seed`, `--start-min`, `--start-max` | 11, 0, 1 | AIPerf's seed, and where in its trace a lane may start |
| `--token-scale` | 1 | 32 replays a 32× corpus at its original token counts |
| `--pauses` | `trace` | A turn's pause: the trace's think time, or `tracelab`'s tool wait |
| `--pool-tokens`, `--cpu-ratio` or `--cpu-gib` | 36,400 tokens, 20 GiB | The GPU pool, and the CPU tier as multiples of it or in GiB |
| `--service` | `contention` | vLLM's step loop, or `idle`: v0's idle-box cost per request |
| `--costs`, `--step-costs` | the frozen box's | E0's fits (`flatshare-bench tiers`' `summary.json`) and the step-time model (`scripts/fit_step_cost.py`'s JSON) |

## The replay

- **AIPerf's structure, ported.** A prompt is `len(hash_ids) × block_size` tokens. The Weka loader's chain detection
  splits each trace into the main agent and its agent and subagent chains, with their think times, spawns and joins.
  `scripts/aiperf_reference.py` dumps AIPerf's own plan and `flatshare-sim aiperf-plan` the port's. On the full 32×
  corpus, 9,843 conversations and 98,827 turns, they agree field for field for 393 lanes at seed 11, and for 450 at
  seed 7 with start ratios 0.25–0.75 and a 30 s cap.
- **Dispatch.** Every stream live at a lane's start first sends its last earlier request with `max_tokens = 1`, and
  profiling starts once these return. A turn's successor goes its delay after the turn returns, and a drained lane
  starts the next trace under a fresh KV namespace, AIPerf's cache-bust marker. Requests still running after the grace
  are dropped.
- **Pauses.** A turn's delay is the trace's think time, or with `--pauses tracelab` a TraceLab tool wait capped the same
  way, drawn from that tool's percentiles (`data/tracelab-pauses.tsv`). Every arm draws the same pause for a turn.
- **Service.** The contention model, the default, sends a request to the step loop after its host time, where it
  shares every step with the requests beside it. The idle service is v0, its zero-load limit, which charges E0's cost
  for the request's hit pattern and then a constant inter-token latency, with no queueing.

## The cache

- **Blocks.** Prompts are cut into 16-token blocks whose identity is their exact prefix, so two prompts share a block
  iff they share every token up to its end, as vLLM's chained block hashing does without collisions.
- **The GPU pool** is 2,275 blocks (36,400 tokens) on the frozen box. A request takes the longest cached prefix of its
  full blocks, at most `⌊(L − 1) / 16⌋`, since vLLM always computes the last prompt token. A finished request frees its
  blocks in reverse order to the back of the free queue, so a prefix's end is evicted before its start.
- **The CPU tier** is 9,102 blocks (20 GiB at 144 KiB a token) on the frozen box, or a ratio of the pool. It extends a
  GPU hit with the following blocks it holds, and a finished request offers it every full block under vLLM's
  `LRUCachePolicy`. Output KV is never reused, since AIPerf synthesizes each turn's history from the trace.
- **Admission.** A request needs blocks for `L + out − 1` tokens and waits while the free blocks cannot hold them, and
  the first waiting request that does not fit holds back those behind it, as in vLLM's scheduler. Each wait is a stall.
- **Retention.** For the E1 arms the engine can pin a prompt's cached start, prefetch the rest from the CPU tier and
  hold it, and demote free cached blocks. A request that does not fit releases holds, oldest first, before it waits.

## The step loop

`scheduler.rs` is vLLM 0.32's `Scheduler.schedule()` on the frozen box, with async scheduling:

- **Host.** A request reaches the scheduler `29.6 ms + 7.994 µs` a prompt token after it is sent, plus 6.5 ms for every
  request then decoding. The API server prepares one request at a time, so a burst queues there.
- **Budget.** A step carries at most 512 tokens (`max_num_batched_tokens`). Running requests come first, a decode taking
  one token and a prefill at most 256 (`long_prefill_token_threshold`) while another request runs or waits. Waiting
  requests follow, first come first served, since the replay sends every request at the background priority.
- **Restores.** A request with blocks in the CPU tier waits for their load without taking budget. Loads run one after
  another, each taking E0's restore term (`10.3 ms + 23.64 µs` a token) and stalling the step loop for 0.80 of it, and
  the request computes from the second step after its load. A prefetch is a load no request waits for.
- **Step time.** A step that computes `T` tokens takes `11.56 ms + 47.99 µs·T + 14.17 ns·P + 429.3 ns·K`, plus 7.6 ms
  when it carries both a prefill chunk and a decode. `P` is the attention pairs and `K` the KV tokens the scheduled
  requests read, its coefficient 147,456 bytes a token at 343 GB/s against the card's 448 GB/s.

`scripts/fit_step_cost.py` fits each term from its own data in AIPerf exports and their `replay-log` CSVs:

| Term | Fitted on |
|---|---|
| The step terms and the host time | 46,315 decode-only steps at [the three operating points](#calibration-against-the-gpu-operating-points), 517 lone requests and E0's 14 cells, by least squares on relative error |
| The mixed-step extra | The median excess of 1,443 decode steps beside one long prefill |
| The host time per decoding request | The slope over 92 GPU hits that arrived while only decodes ran |
| The restore stall | The median lost decode time over the transfer time of 36 restores of 2,000 tokens or more |

## Simplifications

- **Whole footprint at admission.** A request reserves its prompt and output blocks when admitted, where vLLM allocates
  output blocks as it decodes, so nothing is preempted. The operating points saw 0 to 3 preemptions a run.
- **Transfers.** Stores happen when a request finishes, take no time and never delay a block's reuse. But vLLM 0.32
  stores as a request computes, and waits for a pending store before it reallocates the block. That is likely what the
  model misses at 24 sessions ([the calibration](#calibration-against-the-gpu-operating-points)).
- **Per-step costs.** The terms are fitted per step, not per kernel, so CUDA-graph and chunk-shape effects are folded
  into the coefficients.
- **Corpus tokens.** Prompts hold no chat-template or cache-bust tokens, so at W0 the engine counted 4.7% more prompt
  tokens than the corpus holds.

## Arms

| Arm | GPU pool | CPU tier | While a session pauses |
|---|---|---|---|
| `gpu-lru` | vLLM's LRU free queue | none | nothing |
| `lru-cpu` | vLLM's LRU free queue | `LRUCachePolicy`, write-through: the frozen box | nothing |
| `clairvoyant` | Belady's MIN | Belady's MIN, with bypass | nothing |
| `ttl-5`, `ttl-30`, `ttl-120` | LRU | LRU | hold its cached blocks for the TTL |
| `history-prefetch` | LRU | LRU | prefetch it to end as the history predicts it returns |
| `oracle-prefetch` | LRU | LRU | prefetch it to end as it returns |
| `tokencake` | LRU | LRU | offload it, then prefetch it by history |
| `mori` | the free block expected back last | LRU | nothing |
| `infercept` | LRU | LRU | hold it or offload it, by minimum waste |

**The clairvoyant reference** runs twice. The first pass runs `lru-cpu` and records the order in which requests were
admitted and finished. The second replays that order with farthest-next-use eviction in each tier, keeps the first
pass's waits, and re-prices each request's time to first token from its new hit pattern. It is a reference, not a bound,
since a block's hit counts only when its whole prefix hits, a CPU copy is worth nothing while the GPU holds the block,
and faster service would change which requests overlap. `flatshare-sim mrc` reports the case where MIN is a bound.

**The E1 arms** (`retention.rs`) act on a session's latest prompt while it pauses, from a turn's return to its
successor's admission. Their information is none (`lru-cpu` and the TTLs), history or the oracle, the exact time of the
next turn. History is an EWMA, α = 0.5, of the session's own pauses, per tool where they name one. Each arm simplifies
its paper:

- **Continuum** ([arXiv 2511.02230](https://arxiv.org/abs/2511.02230)) picks a TTL per tool call, and here a fixed TTL
  is swept.
- **The prefetches** load the session's blocks from the CPU tier to end 1 ms before the predicted or exact return. The
  oracle is Ask the Tool's progress-reporting level ([arXiv 2609.18849](https://arxiv.org/abs/2609.18849)).
- **Tokencake** ([arXiv 2510.18586](https://arxiv.org/abs/2510.18586)) offloads at a function call, here by demoting the
  session's free blocks, and uploads by the history prefetch. Its space scheduling is not modeled.
- **MORI** ([arXiv 2606.00866](https://arxiv.org/abs/2606.00866)) keeps only its ranking, as the pool's eviction order,
  so that the free block whose session is expected back last goes first.
- **InferCept** (Abhyankar et al., ICML 2024) chooses by minimum waste in memory × time. Preserving holds the blocks,
  swapping demotes them, and discarding never wins with a CPU tier, since ρ < 1.

A hold ends once its session is later than twice its predicted pause. A prefetch moves only the blocks the pool can take
when it begins, and runs on the load stream like any restore.

## What a run reports

- **Tokens.** The shares of prompt tokens served from the GPU pool, restored from the CPU tier and computed, the
  computed ones split into cold first turns, new context and recomputed blocks, the avoidable recompute.
- **Latency and traffic.** Time to first token p50, p90 and p99, overall and for resumed turns, and the PCIe bytes
  restored, prefetched and stored, with evictions and stalls.
- **The regime.** KV pressure is the live working set over the pool, as a time-weighted mean and a peak. π is the median
  pause over the median prompt's restore time, E0's restore extra over a GPU hit, and ρ that restore time over the
  median prompt's recompute time.

Shares and latencies cover the profiling requests that completed, as AIPerf's results do.

## Calibration against the GPU operating points

The contention model meets its targets at 8 and 16 sessions and is 18–35% fast at 24, where restores take longer than it
models. Three operating points ran on the frozen box, each on a freshly started engine, as AIPerf's `agentic_replay` at
8, 16 and 24 sessions with W0's settings, straight to the engine. The simulator runs
`flatshare-sim run DATA/weka-x32 --arms lru-cpu --concurrency N`. Measured shares are the engine's
`prompt_tokens_cached_by_source` counters over the profiling phase, and resume TTFT covers the requests after a
conversation's first turn.

| Point | Source | Requests | Resume TTFT p50 / p90 / p99, ms | GPU / CPU / computed, % |
|---|---|---:|---:|---:|
| 8 sessions | measured | 408 | 83 / 400 / 1,463 | 66.6 / 29.0 / 4.4 |
| | contention | 405 (−0.7%) | 81 / 382 / 1,749 (−3 / −5 / +20%) | 66.1 / 29.5 / 4.4 |
| | leave-one-out | 405 | 83 / 400 / 1,772 (−0 / −0 / +21%) | 66.1 / 29.5 / 4.4 |
| 16 sessions | measured | 686 | 132 / 743 / 8,102 | 57.4 / 36.0 / 6.6 |
| | contention | 691 (+0.7%) | 127 / 685 / 7,776 (−4 / −8 / −4%) | 58.8 / 34.3 / 6.9 |
| | leave-one-out | 690 | 125 / 751 / 7,830 (−5 / +1 / −3%) | 59.1 / 33.9 / 7.0 |
| 24 sessions | measured | 1,568 | 173 / 1,345 / 15,172 | 52.9 / 34.3 / 12.9 |
| | contention | 1,649 (+5.2%) | 142 / 942 / 9,896 (−18 / −30 / −35%) | 55.9 / 32.4 / 11.7 |
| | leave-one-out | 1,646 | 137 / 826 / 9,931 (−21 / −39 / −35%) | 55.9 / 32.9 / 11.2 |

Leave-one-out refits every coefficient on the other two points and predicts the third. The idle service (v0) is 19–80%
fast at p50 and p90 at every point.

- **8 and 16 sessions meet the targets**, p50 and p90 within 10%, shares within 2.1 points and requests within 1%, in
  sample and out of sample. p99 at 8 sessions is four requests, and moves 20% with one of them.
- **The measurements spread too.** Three 16-session runs measured 132–140 ms at p50 and 743–966 ms at p90. A 24-session
  run on a warm engine measured 213 / 1,798 / 18,403 ms, so the starting state matters more there.
- **The cache model** on each point's own timeline (`replay-log`) is within 0.7, 0.3 and 1.8 points of the counters' GPU
  share, and its per-request cached share within 0.6, 1.3 and 2.0 points on average.
- **24 sessions misses on transfers.** Open loop (`replay-log --timed`) the model is −17 / −47 / −56% there, against +1
  / −2 / −33% at 8 sessions and −6 / −19 / +18% at 16. Hits match at every point, but restores take 163 ms longer than
  modeled at 24 sessions, against 15 ms at 8 and 31 ms at 16. The excess grows with the other restores that overlap a
  restore's wait: 41 ms with none, 163 ms with three and 1.2 s with six or more.

`make reproduce-calibration EXPORTS=<dir> CORPUS=DATA/weka-x32` writes every number of this section and the next, and
BENCHMARKS' table of the three points, to `build/reproduce-calibration/calibration.md`. EXPORTS is the unpacked
`leaseholder-calibration-exports-2026-10-24.tar.gz` of the
[v0.3 release](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3), named before the project's rename, which
holds six runs' AIPerf exports and `/metrics` snapshots.

## Validation against W0

On W0's own timeline the cache model is within half a point of the measured GPU and CPU shares. W0 replayed the full 32×
corpus at 16 sessions against the frozen box, and the simulator runs the same configuration at seed 11.

- **The cache model** gives 56.96 / 37.31 / 5.73% (GPU / CPU / computed) on W0's timeline, against the measured 56.54 /
  36.90 / 6.56%, and its per-request cached share is within 1.35 points on average.
- **The full simulation** gives 58.8 / 34.3 / 6.9% under contention. The idle service gives 62.8 / 31.1 / 6.1%, because
  at the idle box's speed sessions return sooner, a mean TTFT of 237 ms against W0's 541, so more prompts still hit.
- **The seed** sets where each lane starts, so it changes the workload. Under the idle service seeds 11–15 give 639–955
  requests and 60.9–72.2% / 21.8–32.5% / 4.4–7.4%, so one 10-minute run is a small sample.

## Other hardware, modeled

On the L4 and the A100 40 GB, modeled from their published peaks, running agents leave two calls short of KV at most 1%
of the time up to 24 sessions of the 32× replay. On the H100 SXM they never do. At its own scaling each card reaches the
box's KV pressure, and lifecycle information buys at most 15%, 13% and 9% of resume TTFT p90 there. Every cost is
measured on one RTX 5060 Ti and scaled. The A100 run of 2026-10-26 measured an A100 SXM4 80 GB against its forecast
([BENCHMARKS.md](../../docs/BENCHMARKS.md#do-the-rules-carry-to-a-datacenter-card)).

### How a GPU is modeled

A GPU's profile is its published peaks and the kernel path its linear layers take. Each of the box's measured cost terms
is divided by the ratio of the peaks of the kind that bounds it, so the box's measured efficiency carries over.
Bandwidth scales the KV reads and the fixed step term, which is mostly the weight read of 4.41 GB a step, 9.8 ms of the
11.56 ms. The linear layers' rate scales the per-token term, the BF16 rate the per-pair term, and the link the restore's
transfer. E0's cold-prefill fit, which gives ρ, follows the scaled step loop. The host's and the software's terms stay:
the API server's time, the mixed-step extra, the restore stall and the 512/256 step budget.

Each GPU keeps beside its pool what the box keeps, from the weights to STT and TTS on the same card, and the box's
headroom, so its pool is `16 · ⌊(memory − 11 GiB) / (16 · 144 KiB)⌋` tokens. The 11 GiB is the box's 9,288 MiB of non-KV
footprint and its 1,976 MiB of headroom, and the formula gives the box's 36,400 tokens back. The CPU tier is 4× each
pool.

| GPU | Memory, GiB | Bandwidth, GB/s | Dense FP8 / BF16, TFLOPS | Linear layers | Host link, each way | Pool, tokens |
|---|---:|---:|---:|---|---|---:|
| RTX 5060 Ti 16 GB | 16 | 448 | 189.8 / 47.4 | FP8 W8A8 | PCIe 3.0 x8, 7.9 GB/s | 36,400 |
| L4 24 GB | 22.5 | 300 | 242.5 / 121.0 | Marlin W8A16, 1.42× a BF16 GEMM | PCIe 4.0 x16, 31.5 GB/s | 83,728 |
| A100 40 GB | 40 | 1,555 | none / 312.0 | Marlin W8A16, 1.42× a BF16 GEMM | PCIe 4.0 x16, 31.5 GB/s | 211,168 |
| H100 SXM 80 GB | 80 | 3,350 | 1,979.0 / 989.5 | FP8 W8A8 | PCIe 5.0 x16, 63.0 GB/s | 502,432 |

| GPU | Costs | Fixed step, ms | Per token, µs | Per pair, ns | Per KV token read, ns | Restore | Cold prefill |
|---|---|---:|---:|---:|---:|---|---|
| RTX 5060 Ti 16 GB | measured | 11.56 | 47.99 | 14.17 | 429.3 | 10.3 ms + 23.64 µs/token | 47.8 ms + 63.11 µs/token + 8.126 ns/token² |
| L4 24 GB | modeled | 17.26 | 106.58 | 5.56 | 641.1 | 10.3 ms + 5.91 µs/token | 53.5 ms + 134.38 µs/token + 3.798 ns/token² |
| A100 40 GB | modeled | 3.33 | 41.33 | 2.15 | 123.7 | 10.3 ms + 5.91 µs/token | 35.2 ms + 47.47 µs/token + 1.340 ns/token² |
| H100 SXM 80 GB | modeled | 1.55 | 4.60 | 0.68 | 57.4 | 10.3 ms + 2.96 µs/token | 27.4 ms + 9.57 µs/token + 0.440 ns/token² |

The peaks are NVIDIA's: the
[RTX 5060 family page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/), the
[50-series comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/), the
[RTX Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf),
and the [L4](https://www.nvidia.com/en-us/data-center/l4/),
[A100](https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
and [H100](https://www.nvidia.com/en-us/data-center/h100/) (SXM) datasheets, whose sparse throughputs are halved.

- **The box's tensor rate.** NVIDIA quotes only 759 AI TOPS, FP4 with sparsity. Every GeForce Blackwell card in the
  whitepaper runs FP8 with FP16 accumulate at half its FP4 rate and BF16 at a quarter of that. E0's recompute cell rules
  out FP8 with FP32 accumulate, at half that rate. At 4,096 tokens the linear layers and attention would take 418 ms at
  it, more than the 389 ms the cell leaves beyond its fixed part.
- **The A100 SXM4 80 GB**, the card the A100 run drew, is the 40 GB card with 80 GB of HBM2e at 2,039 GB/s. That gives
  it a fixed step of 2.54 ms, 94.3 ns a KV token read and the H100's pool. The map leaves it out, and
  `forecast --gpu a100-80gb` forecast the run on it ([`forecast-80gb.md`](../../benchmarks/a100/forecast-80gb.md)).
- **The links** are line rates with 128b/130b encoding. The box's is what its hosts negotiate, not the card's PCIe 5.0.
- **ECC.** The L4's GDDR6 runs with ECC on
  ([product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf)), its check
  bits taking 6.25% of the memory
  ([CUDA guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#theoretical-bandwidth-calculation),
  9.2.1) and leaving 22.5 GiB. HBM keeps its own resources for ECC, so the A100s and the H100 keep all their memory, as
  `nvidia-smi` shows on an [H100 SXM](https://www.icc-usa.com/blog/hgx-8x-h100-benchmarks), an
  [A100-SXM4-40GB](https://forums.developer.nvidia.com/t/a100-sxm4-40gb-shows-in-nvidia-smi-but-not-accessible-in-cuda/322730)
  and an [A100-SXM4-80GB](https://github.com/NVIDIA/DCGM/issues/93).

### The linear layers' path

Qwen3-4B-Instruct-2507-FP8 quantizes its weights in 128×128 blocks, and vLLM 0.32.0's `Fp8LinearMethod` serves every
linear layer with the first block-FP8 kernel that supports the card
(`model_executor/kernels/linear/__init__.py:478-488, 743-755` at v0.32.0). On sm75 to sm89 every kernel ahead of
`MarlinFP8ScaledMMLinearKernel` refuses the card, so the L4 (sm89) and the A100 (sm80) run Marlin, which is weight-only
W8A16 at the BF16 tensor rate. The box (sm120) runs `DeepGemmFp8BlockScaledMMKernel`, as its engine log shows, and the
H100 (sm90) DeepGEMM, both W8A8.

[vLLM PR #61224](https://github.com/vllm-project/vllm/pull/61224) measured Qwen3-8B's four linear layers on an L4 at 512
tokens, 1.42× slower on Marlin than in BF16 (5,046.9 against 3,563.7 µs). The model uses that factor on both cards, so
the L4's linear layers take 106.6 µs a token, against the box's 48.0, and the A100's 41.3.

### Checks against measurements

- **Restores on PCIe Gen4.** The Gen4 x16 term gives 34.5, 58.7 and 107.1 ms for 4k, 8k and 16k tokens, where vLLM 0.29
  measured 34, 65 and 104 ms for Qwen3-4B's CPU-tier restores on an L20
  ([#57103](https://github.com/vllm-project/vllm/issues/57103#issuecomment-5733975478)). So the box's 79% of its link
  carries over to Gen4 within 10%, and nothing checks Gen5.
- **The fixed step is a floor.** vLLM 0.30 decoding Qwen3-4B in BF16 at batch 1 on an H100 SXM takes 4.29 ms a step
  ([decode-decoded](https://github.com/zhebrak/decode-decoded)), 2.45 ms of it the weight read and 1.8 ms the rest. The
  box's fixed term has about as much beyond its weight read, 1.7 ms, which bandwidth scaling shrinks to 0.23 ms. Kept,
  it would put the H100's fixed step at 3.0 ms, not 1.55, and the A100's at 4.55, not 3.33.
- **Attention.** FlashAttention 2's forward pass reaches 50–73% of the A100's peak
  ([its paper](https://arxiv.org/abs/2307.08691)), where the box's attention term carries 88% of the box's. The map
  keeps the box's efficiency, and the forecast takes FlashAttention 2's range.
- **The A100 run** fitted the step model on its calibration point
  ([`a100.md`](../../benchmarks/results/2026-10-26-a100/a100.md)). Marlin's per-token term came in at 41.36 µs against
  41.33 modeled. Every other term came in above the model: the fixed step at 4.5 ms against 2.54, 4.07 ns a pair against
  2.15, and 171.9 ns a KV token read against 94.3. Its decode attention read KV at 42% of the card's bandwidth.

### Where each GPU sits on the map

The 32× scaling was chosen for the box's pool, so each GPU is placed twice: on the same 32× replay, and at its own
scaling (`--fit`). That is the least power of two whose longest request fits both its pool and Qwen3-4B's 262,144-token
context: 16× on the L4, 8× on the A100 and 4× on the H100. A GPU whose pool holds r times the box's runs the session
counts up to 40·r. Each cell is the mean over seeds 11–13 with the traces' pauses.

The realtime tier is two calls of 1,024 tokens of KV each, as in E2's discriminating scenario. The running share is the
time the KV of running agent requests leaves less than 2,048 tokens of the pool free. A voice turn then waits for
blocks, since vLLM 0.32 preempts no running request for a waiting one, the KV-blocking regime of vLLM issue
[#40004](https://github.com/vllm-project/vllm/issues/40004). Otherwise a voice turn meets compute interference alone,
behind steps that carry a 512-token prefill chunk at the median prompt. π, ρ and that chunk hardly move with the
sessions on a card:

| GPU | Scaling | π | ρ | Chunk, ms |
|---|---|---:|---:|---:|
| RTX 5060 Ti 16 GB, measured | 32× | 21–30 | 0.25–0.27 | 56–66 |
| L4 24 GB, modeled | 32× | 62–83 | 0.05–0.06 | 81–85 |
| | 16× | 38–56 | 0.04–0.05 | 90–99 |
| A100 40 GB, modeled | 32× | 66–109 | 0.14–0.16 | 28–29 |
| | 8× | 21–29 | 0.10–0.11 | 37–43 |
| H100 SXM 80 GB, modeled | 32× | 100–131 | 0.31–0.33 | 5 |
| | 4× | 21–43 | 0.14–0.18 | 12–16 |

The box's cells are [E1's map](../../docs/BENCHMARKS.md#where-does-knowing-the-returns-help-most), where its running
share is 0, 0, 1, 4, 12, 25 and 33% at 5 to 40 sessions. On the 32× replay:

| GPU, modeled | Sessions | Pressure | Running share | `lru-cpu` resume p50, ms | `oracle-prefetch` gain, p50 / p90 |
|---|---:|---:|---:|---:|---:|
| L4 24 GB | 5 | 0.5 | 0% | 101 | +0 / +0% |
| | 8 | 0.8 | 0% | 97 | +0 / +0% |
| | 12 | 1.1 | 0% | 127 | +2 / +7% |
| | 16 | 1.5 | 1% | 156 | +3 / +8% |
| | 24 | 2.0 | 1% | 158 | +2 / +14% |
| | 32 | 2.7 | 3% | 224 | +10 / +12% |
| | 40 | 3.3 | 7% | 377 | +16 / +8% |
| | 64 | 5.5 | 23% | 10,851 | −1 / −0% |
| A100 40 GB | 5 | 0.2 | 0% | 69 | +0 / +0% |
| | 8 | 0.3 | 0% | 63 | +0 / +0% |
| | 12 | 0.4 | 0% | 75 | +0 / +0% |
| | 16 | 0.6 | 0% | 81 | −0 / +0% |
| | 24 | 0.8 | 0% | 79 | +0 / +1% |
| | 32 | 1.1 | 0% | 96 | +1 / +5% |
| | 40 | 1.3 | 0% | 103 | +3 / +4% |
| | 64 | 2.2 | 0% | 135 | +5 / +3% |
| | 96 | 3.1 | 0% | 270 | −3 / −3% |
| | 128 | 4.1 | 1% | 5,876 | −25 / −22% |
| | 192 | 6.2 | 0% | 51,789 | −62 / −5% |
| H100 SXM 80 GB | 5 | 0.1 | 0% | 62 | +0 / +0% |
| | 8 | 0.1 | 0% | 54 | +0 / +0% |
| | 12 | 0.2 | 0% | 65 | +0 / +0% |
| | 16 | 0.3 | 0% | 70 | +0 / +0% |
| | 24 | 0.3 | 0% | 67 | +0 / +0% |
| | 32 | 0.5 | 0% | 81 | +0 / +0% |
| | 40 | 0.5 | 0% | 84 | +0 / +0% |
| | 64 | 0.9 | 0% | 102 | +2 / +1% |
| | 96 | 1.3 | 0% | 158 | +2 / +1% |
| | 128 | 1.7 | 0% | 481 | +1 / +1% |
| | 192 | 2.6 | 0% | 4,756 | −2 / −4% |
| | 256 | 3.5 | 0% | 12,192 | −6 / −7% |

At each GPU's own scaling:

| GPU, modeled | Sessions | Pressure | Running share | `lru-cpu` resume p50, ms | `oracle-prefetch` gain, p50 / p90 |
|---|---:|---:|---:|---:|---:|
| L4 24 GB | 5 | 1.1 | 0% | 173 | +4 / +5% |
| | 8 | 1.6 | 1% | 175 | +5 / +15% |
| | 12 | 2.3 | 2% | 237 | +6 / +10% |
| | 16 | 3.1 | 9% | 2,019 | +4 / −5% |
| | 24 | 4.2 | 22% | 7,432 | +2 / −0% |
| | 32 | 5.3 | 19% | 27,550 | −4 / −1% |
| | 40 | 6.4 | 11% | 81,713 | −1 / −0% |
| A100 40 GB | 5 | 0.9 | 0% | 193 | −0 / −1% |
| | 8 | 1.2 | 0% | 188 | +3 / +13% |
| | 12 | 1.8 | 0% | 251 | +7 / +11% |
| | 16 | 2.5 | 2% | 456 | −2 / −3% |
| | 24 | 3.3 | 8% | 2,402 | −14 / −34% |
| | 32 | 4.2 | 10% | 9,380 | −5 / −6% |
| | 40 | 5.2 | 10% | 28,145 | −6 / −2% |
| H100 SXM 80 GB | 5 | 0.7 | 0% | 327 | +0 / +0% |
| | 8 | 1.0 | 0% | 288 | −0 / +5% |
| | 12 | 1.5 | 1% | 426 | +1 / +9% |
| | 16 | 2.1 | 2% | 1,223 | +4 / +7% |
| | 24 | 2.8 | 6% | 4,531 | −5 / −4% |
| | 32 | 3.6 | 7% | 11,191 | −3 / −5% |
| | 40 | 4.3 | 6% | 30,759 | −11 / −7% |
| | 64 | 7.2 | 1% | 266,358 | −1 / −1% |

- **The working set** leaves two calls short, displacing cached agent KV, 0–11% of the time up to a pressure of 0.9,
  69–85% at 1.0–1.1 and 94–100% from 1.2, on every GPU and the box.
- **The other arms.** The clairvoyant reference gains at most 2% at p50, and at p90 at most 5% on the 32× replay and 10%
  at the GPUs' own scalings. Session history and Tokencake gain at most 5% anywhere, and lose up to 25% and 44%.
- **Saturation.** Rows whose `lru-cpu` p50 runs to seconds are saturated, and there the running share falls as requests
  queue before admission. The H100 saturates on attention at 4× from 16 sessions, and at 32× from 128 sessions at the
  modeled API server, its GPU 61–73% busy.

What each regime means for a realtime tier beside agents:

- **On the box** a voice turn meets compute interference up to 16 sessions. From 24 it is in #40004's regime an eighth
  to a third of the time, where E2 needed the budget cap as well as the KV preemption
  ([the report's §6.2](../../docs/REPORT.md#62-the-kv-preemption-removes-the-wait-for-kv-and-the-budget-cap-removes-agent-prefill-from-the-turns-steps)).
- **The L4** has 2.3× the box's pool but two thirds of its bandwidth, and Marlin layers at 2.2× the box's FP8 time, so
  at its own 16× its compute runs out at 12 to 16 sessions, before KV does.
- **The H100 SXM** at its own 4× has two calls blocked 0–7% of the time, but a realtime tier sized like the box's, 28
  calls, would be blocked 25% of the time at 16 sessions and 49–65% at 24–32.

So the KV-blocking regime follows the agents' context lengths against the pool, and the realtime tier's share of it, not
the consumer card.

**What the assumptions could move.** Each assumption was varied with the H100 at 4× and the L4 at 16×, over seeds 11–13.
These runs predate the L4's Marlin path and the refitted cold prefill, so the L4's sizes come from its earlier FP8
model.

1. **The workload's scaling.** On the H100, 4× against 32× multiplies pressure by 8 at equal sessions, roughly halves ρ
   and nearly triples the chunk.
2. **The realtime tier's size.** Sized at the box's share of the pool, 5 calls on the L4 and 28 on the H100, it raises
   the running share from 9–21% to 23–44% on the L4 at 16–32 sessions. On the H100 at 16–40 it rises from 2–7% to
   25–65%.
3. **The restore's efficiency**, the box's 79% of the line rate. Restores 1.6× slower would cut π by a third and raise ρ
   by half.
4. **Attention at the box's BF16 peak.** If the box's attention were bound by twice that rate, the modeled GPUs'
   attention would run half as fast. ρ would fall by a third on the H100 and a fifth on the L4, and the chunk lengthen
   by 60–70% and 25–35%.
5. **The headroom.** A headroom of 12% of memory, the box's share, would shrink the L4's pool by 7% and the H100's by
   11%, raising the running share by up to 3 points.
6. **The fixed step and the host.** Keeping the box's 1.7 ms of step overhead, or a host three times faster, moves no
   coordinate beyond the seeds' noise. The faster host cuts low-load resume TTFT by 40–60%.
7. **Not varied:** the mixed-step extra, the restore stall, the step budget, the 4× CPU tier, Qwen3-4B-FP8 on every card
   and the L4's bandwidth with ECC on. Stores are not modeled, so restores at high pressure are optimistic on every GPU.

## A GPU run's forecast

`flatshare-sim forecast WHOLE LONG --gpu a100-40gb` forecasts a run on another GPU before it runs, from the 32× corpus
and its 48 traces with the largest contexts, E2's discriminating corpus (`benchmarks/e2/long_traces.py` links them).
`make forecast-a100 CORPUS=DATA/weka-x32` writes [`forecast.md`](../../benchmarks/a100/forecast.md) for the A100 40 GB,
and with `GPU=a100-80gb` [`forecast-80gb.md`](../../benchmarks/a100/forecast-80gb.md) for the A100 SXM4 80 GB. Where the
box measured the same thing, the forecast is that measurement times the modeled ratio of the two cards, so that the box
model's own error does not carry over. It gives:

- **The costs in three settings**, for Qwen3-4B-FP8 and Qwen3-14B-FP8. *Low* is the peaks' ratios alone, with Marlin as
  fast as BF16. *Central* keeps the box's 1.7 ms of step overhead, runs Marlin as on the L4, and puts attention at
  FlashAttention 2's best published efficiency on the card. *High* takes FlashAttention 2's worst.
- **E0's cells and the engine probes' cases**, and the time-equivalent step cap, the largest multiple of 256 tokens
  whose chunk takes the card no longer than the box's 512-token chunk takes the box.
- **E2's scenarios on the regime map** over seeds 11–15, among them the core, the discriminating scenario at the box's
  pool and 24 sessions.

**Fitted costs.** Once a run has measured the card, `--costs` and `--step-costs` take its E0 `summary.json` and the step
model `scripts/fit_step_cost.py` fits on its calibration point. They place the core at each of `--fitted-sessions` and
name the count whose running share comes closest to the box's, the count to run A' at.
`make forecast-fitted CORPUS=DATA/weka-x32 COSTS=SUMMARY STEPS=STEPS` writes it to `build/forecast-fitted.md`, for the
A100 SXM4 80 GB unless `GPU` names another card.

**A measured host.** Without `--host` the card's API server runs at the box's CPU speed and its restores at the
datasheet's link. The A100 run showed both to be the host's. Its API server took 14.2 ms + 4.971 µs a prompt token
against the box's 29.6 ms + 7.994 µs, and its restores moved 11.46 GB/s, not 25, through PCIe 3.0 root ports.
`--host HOST_JSON` takes the `host.json` that `benchmarks/a100/bench_host.py` writes on the host, and
`make forecast-a100 HOST_JSON=<host.json>` writes the forecast on it beside the one made before the run.

## Runtime

On an Apple M5 Max, in CPU time, a run of the W0 configuration takes 0.22 s with the stock arms and 0.33 s with the E1
arms, and every trace through 16 recycling lanes 13.2 s. `regime`'s 42 cells × 3 seeds × 10 arms take 89 s, or 9.7 s on
18 threads in 2.2 GB. On peacock, a 16-thread Xeon Silver 4110, `make reproduce-e1` ran all five tables in 2 min 18 s.

## Tests

`cargo test -p flatshare-sim` holds the KV model, both clairvoyant tiers and MIN against exhaustive search, the step
loop against hand-computed schedules, the GPU profiles (the box's gives its measured costs back exactly) and the
forecast (the box forecasts its own measurements). `tests/subset.rs` runs the 16-trace subset end to end, the plan
against AIPerf's and every arm. The subset is a lighter workload than W0's, 559 requests at a KV pressure of 0.9 against
3.3.

**Test data.** `testdata/weka-x32-sub16` holds 16 traces of
[semianalysisai/cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
(Apache-2.0), block-scaled 32×, and `testdata/aiperf-0.13.0-sub16.json` AIPerf's plan for them, which
`scripts/aiperf_reference.py` regenerates. Beside them are a record of W0's AIPerf export, the box's `host.json` from
`benchmarks/a100/bench_host.py` and `forecast`'s report on the subset, whose test gives the command that rewrites it.
`data/tracelab-pauses.tsv` holds per-tool percentiles of the wait between a coding agent's tool calls and their
results, derived by `scripts/tracelab_pauses.py` from the [TraceLab](https://github.com/uw-syfi/TraceLab) public
coding-agent trace dataset v0.0.2 (CC BY 4.0), which is not in the repository.
