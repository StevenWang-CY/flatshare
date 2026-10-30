# flatshare-sim

`flatshare-sim` replays the SemiAnalysis Claude Code traces as AIPerf 0.13.0's `agentic_replay` sends them, through a model of vLLM 0.32's GPU prefix cache and OffloadingConnector CPU tier, served by a model of vLLM's step loop on the frozen box (`flatshare-core`'s step-time model; E0's idle-box cost model is its zero-load limit). For each cache policy it reports where prompt tokens came from (the GPU pool, the CPU tier, or computation), what that costs in time to first token and PCIe traffic, and where the run sits in the regime coordinates E1 is plotted in. A clairvoyant reference gives the gap each policy leaves, and the E1 arms measure how much of it each level of information about a session's return closes. Other GPUs are placed on the same map with costs modeled from their published peaks. This README is for a reader who runs the simulator, checks its calibration, or reads E1's tables and the modeled hardware.

## Usage

`DATA/weka-x32` is the 32× corpus: SemiAnalysis's traces as `flatshare-bench scale-traces` writes them ([how](../../python/flatshare_bench/README.md#trace-scaling); the download is in its quickstart). It stays outside the repository.

```sh
cargo build --release -p flatshare-sim
flatshare-sim run DATA/weka-x32 --json w0.json --csv w0.csv                       # the W0 configuration, the stock arms
flatshare-sim run DATA/weka-x32 --cpu-ratio 2,4,8 --arms lru-cpu,clairvoyant      # CPU tier at 2x, 4x and 8x the pool
flatshare-sim run DATA/weka-x32 --service idle                                    # v0's idle-box service
flatshare-sim run DATA/weka-x32 --pauses tracelab --arms lru-cpu,oracle-prefetch
flatshare-sim regime DATA/weka-x32 --csv regime.csv --markdown regime.md          # the regime map, seeds 11-13
flatshare-sim regime DATA/weka-x32 --sessions 8,12,16,24 --cpu-ratio 4 --pause-sources trace --seeds $(seq 11 30 | paste -sd, -)
flatshare-sim hardware DATA/weka-x32 --csv hw.csv --markdown hw.md                # other GPUs, modeled
flatshare-sim hardware DATA/weka-x32 --fit --csv hw-fit.csv --markdown hw-fit.md  # each at its own scaling
flatshare-sim forecast DATA/weka-x32 DATA/weka-x32-long --markdown forecast.md    # a GPU run's forecast (the A100's)
flatshare-sim mrc DATA/weka-x32 --service idle --csv mrc.csv --markdown mrc.md    # miss-ratio curves (BENCHMARKS' table)
flatshare-sim replay-log DATA/weka-x32 runs/w0/aiperf/profile_export.jsonl        # a measured run through the engine model
flatshare-sim replay-log DATA/weka-x32 runs/w0/aiperf/profile_export.jsonl --timed --csv w0-timed.csv
```

The replay options are the fields of `flatshare-bench`'s `configs/agentic_replay.yaml`, but they default to the W0 settings, not to that file's (1,800 s plus a 120 s grace and a 600 s think-time cap): 16 lanes, 600 s plus a 60 s grace, a 120 s think-time cap, seed 11, start ratios 0 to 1, burst starts and a 10 s idle cap. Capacities default to the frozen box (`--pool-tokens 36400`, a 20 GiB CPU tier; `--cpu-ratio` or `--cpu-gib` change it), costs to the E0 fits (`--costs` takes the `summary.json` of `flatshare-bench tiers`; `--itl-ms` sets the idle service's decode rate) and the step-time model to the frozen box's fit (`--step-costs`). `run` writes a JSON summary (`--json`), one CSV row per arm and capacity (`--csv`) and one CSV row per request (`--requests`). `--token-scale 32` replays a 32× corpus at its original token counts.

`regime` runs `lru-cpu`, the E1 arms and the reference in every cell of `--sessions` (5, 8, 12, 16, 24, 32, 40), `--cpu-ratio` (2, 4, 8) and `--pause-sources` (trace, tracelab), at each of `--seeds` (11, 12, 13), on `--jobs` threads (default: every core). It writes one CSV row per cell, seed and arm (`--csv`), with the share of the clairvoyant gap each arm closes, `(lru − arm) / (lru − clairvoyant)`, left empty where the reference is less than 5% faster than `lru-cpu`. The Markdown (`--markdown`) is the regime map ([BENCHMARKS.md](../../docs/BENCHMARKS.md#the-regime-map)), one table per CPU tier: per cell, the coordinates and each arm's gain over `lru-cpu`, the mean over the seeds. With several seeds a table per pause source and tier gives their spread, as BENCHMARKS' realistic regime does. `make reproduce-e1 CORPUS=DATA/weka-x32` writes every E1 table, BENCHMARKS' and this README's other hardware, to `build/reproduce-e1`.

`hardware` runs the same arms on each of `--gpus` (`rtx-5060-ti`, the frozen box with its measured costs; `l4`, `a100-40gb` and `h100-sxm`, modeled: see [Other hardware, modeled](#other-hardware-modeled); `a100-80gb`, modeled, is not in the default) over `--sessions` and `--seeds` (11, 12, 13), with the CPU tier at `--cpu-ratio` (4) times each GPU's pool and a realtime tier of `--calls` (2) voice calls. The sessions are the regime map's and 64, 96, 128, 192 and 256, of which a GPU whose pool holds r times the box's, in the replay's tokens, runs those up to 40·r: the box runs 5 to 40. Every GPU replays the corpus at `--token-scale`, which unlike `regime`'s scales no pool, or with `--fit` at its own scaling. It writes one CSV row per cell and arm (`--csv`: `regime`'s columns, after the GPU, its costs' label, the token scale, the pool and the seed, and before the peak pressure, the median prompt, the realtime tier's two shares and the chunk) and the GPUs' costs and placements (`--markdown`), each cell the mean over the seeds.

`replay-log` sends a measured run's requests at their measured start and end times through the engine model, which separates the cache model from the replay timing; with `--timed` it sends each at its measured start and serves it with the step loop, open loop, which separates the service model from the closed loop's feedback, and adds the measured and simulated TTFT to the CSV.

## The replay

- **Traces.** A directory of `<trace id>.json` as `flatshare-bench scale-traces` writes it, or the original JSONL. A prompt is `len(hash_ids) × block_size` tokens, which is the trace's `in` on every request of the corpus (98,827 of 98,827).
- **AIPerf's structure, ported.** The Weka loader's chain detection splits each trace into the main agent, flattened agent chains and subagent chains, with their think-time delays (`useThinkTimeOnly`, capped at `interTurnDelayCapSeconds`), SPAWN branches and joins, and the cross-stream replay barriers. `TrajectorySource` gives each lane its start instant `t*` (SHA-256 of seed, trace and lane, then numpy's PCG64) and the streams live at `t*`. `scripts/aiperf_reference.py` dumps AIPerf's own plan with prompt synthesis stubbed out, and `flatshare-sim aiperf-plan` dumps the port's: they agree field for field on the full 32× corpus (9,843 conversations, 98,827 turns) for 393 lanes with seed 11 and for 450 lanes with seed 7, start ratios 0.25–0.75 and a 30 s cap. The test subset's plan is pinned in `testdata/aiperf-0.13.0-sub16.json`.
- **Dispatch.** `concurrency` lanes take traces from AIPerf's sequential sampler. Every stream live at `t*` first sends its last earlier request with `max_tokens = 1`, all at once; profiling starts when the last of these returns. Each lane then sends its earliest live turn at once and the others at their recorded offsets. A turn's successor goes its delay after the turn returns; a root turn spawns its children when it returns (when it is sent, if the recorded spawn overlapped the request), each at its recorded offset from the spawn; a joining turn waits for its children; a turn whose replay predecessors have not completed is held. A lane whose agent session has drained starts the next trace from its first turn, under a fresh KV namespace (AIPerf's cache-bust marker). No request is sent after the duration, requests still running at the end of the grace period are dropped, and when nothing is in flight the pending think-time timers are pulled forward so the engine idles at most `systemIdleGapCapSeconds`.
- **Pauses.** A turn's delay is the trace's think time (`--pauses trace`, the default), or, with `--pauses tracelab`, a TraceLab tool wait drawn in its place and capped the same way: a tool by its share of TraceLab's tool rounds, then a wait from that tool's percentiles (`data/tracelab-pauses.tsv`, below). Every arm draws the same pause for the same turn.
- **Service.** Under contention (`--service contention`, the default) a request reaches vLLM's step loop after its host time and shares every step with the requests beside it (the contention model below). `--service idle` is v0: a request's time to first token is E0's cost for its hit pattern after any wait for free blocks, then it decodes at a constant inter-token latency, with no queueing behind other requests; it reproduces v0's outputs exactly (the tests pin v0's subset run).

## The engine model

- **Blocks.** Prompts are cut into 16-token blocks. A block's identity is its exact prefix: the trace's hash-id trie node its last token falls in, its offset inside that hash block, and the session play. Two prompts share a block iff they share every token up to its end, which is vLLM's chained block hashing without collisions. Every request goes to one served model, as in an AIPerf replay, so requests the traces sent to different models share KV.
- **GPU pool.** 2,275 blocks (36,400 tokens) on the frozen box, one of them vLLM's null block. A request takes the longest cached prefix of its full blocks, at most `⌊(L − 1) / 16⌋` blocks, because vLLM always computes the last prompt token. Running requests pin their blocks (reference counted, shared by common prefixes). A finished request frees its blocks in reverse order: cached blocks join the back of the free queue, so the end of a prefix is evicted before its start, and the partial last block joins the front. A block recomputed while a copy is cached is cached a second time, as vLLM keeps every block per hash. A hit takes a copy that running requests already pin when there is one, where vLLM takes the first copy cached; the two differ only for such recomputed blocks, and the clairvoyant pass relies on the rule (see below).
- **CPU tier.** 9,102 blocks (20 GiB at 144 KiB per token) on the frozen box, or a ratio of the pool. After the GPU hit, the tier extends it with the following blocks it holds; they are restored into new GPU blocks. When a request finishes, every full block it holds, prompt and output (`offload_prompt_only: false`), is offered to the tier: from the start of the prompt, or from past the restored blocks when it restored any (vLLM's chunk-level store index). LRU follows vLLM's `LRUCachePolicy`: blocks become most recent when stored and when a request that used them finishes, in reverse prefix order. A store never evicts a block it offers, and stores nothing when it cannot make room.
- **Output KV.** Output tokens take blocks and are written through, but they are never reused: AIPerf synthesizes each turn's history from the trace, not from what the model generated.
- **Admission.** A request needs blocks for `L + out − 1` tokens. It waits when the free blocks, less the free cached blocks its hit takes, cannot hold the blocks it must allocate; waiting requests are admitted in order as blocks free up, and the first that does not fit holds back those behind it, as in vLLM's scheduler. Each wait is a stall. A request larger than the whole pool is an error, as vLLM refuses to start with `max_model_len` above its pool.
- **Retention.** For the E1 arms the engine can hold the pool's cached start of a prompt (pin it), prefetch the rest from the CPU tier into free blocks and hold it, and demote free cached blocks to be reused first. Holds give way: a request that does not fit releases holds, oldest first, before it waits, so no arm makes a request wait for memory vLLM would give it.

## The contention model

`scheduler.rs` is vLLM 0.32's step loop on the frozen box (`Scheduler.schedule()` in `v1/core/sched`, async scheduling on):

- **Host.** A request reaches the scheduler after its host time: the API server, the tunnel, rendering, tokenization and block hashing, `29.6 ms + 7.994 µs` per prompt token, plus 6.5 ms for every request decoding when it is sent. The API server prepares one request at a time in the order they are sent (its renderer pool has `--renderer-num-workers` threads, 1 by default), so a burst of sends queues there.
- **Budget.** A step carries at most 512 tokens (`max_num_batched_tokens`). Running requests come first, in admission order: a decode takes one token, a prefill the rest of its prompt, at most 256 per step (`long_prefill_token_threshold`) when more than one request is running or waiting; alone, a request takes the whole budget. A request reached with the budget spent waits, decodes included.
- **Waiting requests** follow, first come first served (the replay sends every request with the background class's priority), after those whose restore has finished. A request is admitted into the cache model when it is first reached with budget left; one that does not fit stops the loop. A prompt found whole recomputes its last token.
- **Restores.** A request with blocks in the CPU tier waits for their load without taking budget. The load starts once the step that admitted it has run and the loads before it have finished (the connector issues loads one after another on one stream), and takes E0's restore term (`10.3 ms + 23.64 µs` per token). The step loop stalls for 0.80 of it, charged to the first step that starts during the load. The engine learns that a load has finished from the output of the step in flight and has already scheduled the next step, so the request computes from the second step that starts after its load, or at once when nothing else runs. A prefetch is a load that no request waits for.
- **Step time.** A step that computes `T` tokens takes `11.56 ms + 47.99 µs·T + 14.17 ns·P + 429.3 ns·K`, plus 7.6 ms when it carries both a prefill chunk and a decode, where `P` is the attention pairs (each computed token times the tokens it attends to) and `K` the KV tokens every scheduled request reads. `K`'s coefficient is 147,456 bytes per token read at 343 GB/s (the card's memory bandwidth is 448 GB/s); `P`'s is within 13% of twice E0's quadratic recompute term; the mixed-step extra is measured, not derived. Each output token comes out at the end of a step; a request with all its tokens finishes and frees its blocks.

`scripts/fit_step_cost.py` fits every coefficient from AIPerf exports and their `replay-log` CSVs, each term from its own data: the step terms and host time by least squares on relative error over decode-only steps (the gaps between a request's tokens while every request in flight was decoding: 46,315 at the three fresh points), requests that ran alone (517) and E0's 14 GPU-hit and recompute cells, with E0's token-id requests given a host time of their own; the mixed-step extra from 1,443 decode steps beside one long prefill (median excess); the host time per decoding request from 92 GPU hits that arrived while only decodes ran (slope); and the restore stall from 36 restores of at least 2,000 tokens with only decodes beside them (median lost decode time over the transfer time; quartiles 0.74 and 1.01). `--step-costs` takes the JSON it writes. The scheduler is synchronous otherwise: a GPU hit that arrives beside one decode waits about half a step in the measured runs, as here, not the step and a half that scheduling a step ahead would add.

## Simplifications

- **Whole footprint at admission.** A request reserves its prompt and output blocks when it is admitted, where vLLM reserves the prompt (`scheduler_reserve_full_isl`) and allocates output blocks as it decodes; so nothing is preempted (the operating points saw 0 to 3 preemptions per run).
- **Transfers.** Loads take E0's restore term each, one after another, and stall the step loop by a fixed share; stores happen when a request finishes, take no time and never delay a block's reuse. vLLM 0.32 stores as a request computes, and when the scheduler reallocates a block whose store is still pending the worker waits for that store before the step (the connector's `jobs_to_flush`); with the pool full this is likely what the model misses at 24 sessions (see the calibration).
- **Fixed step costs.** The step-time terms are fitted per step, not per kernel: CUDA-graph and chunk-shape effects are folded into the coefficients and the measured mixed-step extra.
- **Corpus tokens.** Prompts have no chat-template or cache-bust-marker tokens: the engine counted 4.7% more prompt tokens than the corpus holds for the same requests at W0.

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

`run` runs the first three by default; `--arms` picks any, and `regime` runs `lru-cpu`, the E1 arms and the reference.

The **clairvoyant reference** runs in two passes. The first runs the reference arm (`lru-cpu` unless `--reference` says otherwise) and records the order in which the engine admitted and finished requests. The second replays that fixed order with farthest-next-use eviction in each tier: the pool reuses the free block whose next use is farthest, and the CPU tier keeps the blocks with the nearest next uses, declining to store a block whose next use is farther than every block it holds. A block's next use is the next request whose prompt holds it, whichever tier serves that request. Waits for free blocks, and so the whole timeline, are the reference arm's; time to first token is re-priced from the new hit pattern (under contention, the reference run's queueing stays and each request's own service, as it would take alone, is re-priced). Under any eviction policy a request takes the blocks its footprint needs minus those it shares with running requests (because a hit takes a pinned copy when one exists), so every admission of the first pass fits in the second.

It is a reference, not a bound. MIN per tier is optimal only for the hit count of one tier over a fixed sequence of unit-size blocks; here a block's hit counts only when its whole prefix hits, a CPU copy is worth nothing while the GPU holds the block, faster service would change which requests overlap, and PCIe has limits. `flatshare-sim mrc` reports the case where MIN is a bound.

**The E1 arms** (`retention.rs`, and `replay/retain.rs` in the replay) act on a session's KV, its latest prompt, while the session pauses between turns: from when a turn returns to when its successor, due after a delay, is admitted. Turns sent at once, or gated on children, are not pauses. The information levels are none (`lru-cpu`, and the fixed TTLs), history (an EWMA, α = 0.5, of the session's own pauses, per tool when the pauses name one, falling back to its mean over every tool, then to the mean over every session) and the oracle (the exact time the next turn is sent). Each arm simplifies its paper to this engine:

- **Continuum** ([arXiv 2511.02230](https://arxiv.org/abs/2511.02230)) pins a paused agent's KV with a TTL it picks per tool call; here a fixed TTL is swept.
- **history-prefetch** and **oracle-prefetch** start a load of the session's blocks that the CPU tier holds so it ends 1 ms before the predicted (exact) return, and hold them until the turn is admitted; the oracle is Ask the Tool's progress-reporting level ([arXiv 2609.18849](https://arxiv.org/abs/2609.18849)).
- **Tokencake** ([arXiv 2510.18586](https://arxiv.org/abs/2510.18586)) offloads an agent's KV at the start of a function call and uploads it before the predicted completion. The CPU tier already has a write-through copy, so offloading demotes the session's free blocks; the upload is the history prefetch. Its space scheduling across agents is not modeled.
- **MORI** ([arXiv 2606.00866](https://arxiv.org/abs/2606.00866)) splits KV between HBM and DRAM by idleness; here only its ranking, as the pool's eviction order: the free block whose session is expected back last (by history) goes first. Its proactive moves are not modeled.
- **InferCept** (Abhyankar et al., ICML 2024) chooses per interception among preserving, discarding and swapping the KV by minimum waste in memory × time: preserving wastes the context for the predicted pause, swapping wastes the running batch for the step-loop stall of the restore (0.80 of E0's restore term), discarding wastes it for the recompute. Discarding never wins with a CPU tier here (ρ < 1). Preserving holds the blocks; swapping demotes them.

A hold also ends once its session is later than twice its predicted pause. A prefetch takes only free blocks and runs on the load stream like any restore: it delays the restores queued behind it and stalls the step loop as they do.

## Results per arm

- **Tokens.** Requests and prompt tokens, and the shares served from the GPU pool, restored from the CPU tier and computed. Computed tokens split into cold first turns of conversations, new context (blocks the run had not seen), and recomputed blocks (seen before, since evicted from every tier, plus the last block of a fully cached block-aligned prompt): the avoidable recompute.
- **Latency.** Time to first token p50/p90/p99, overall and for resumed turns.
- **Traffic.** PCIe bytes restored, prefetched and stored, GPU and CPU evictions, and stalls (requests that waited for free blocks, and the total wait). Under contention `engine` also counts the steps run, their tokens and their total time.
- **Regime.** KV pressure, the live working set (distinct blocks in the latest prompt of every live stream) over the pool, as a time-weighted mean and a peak; π, the median pause (over the turns that pause: a turn sent at once has no pause) over the restore time of the median prompt (E0's restore extra over a GPU hit); ρ, that restore time over the recompute time of the median prompt.

Shares, latencies, restored bytes and stalls cover the profiling requests that completed, as AIPerf's results do; store, prefetch and eviction counts cover the whole run.

## Calibration against the GPU operating points

Three operating points on the frozen box, each on a freshly started engine: AIPerf `agentic_replay` at 8, 16 and 24 sessions (the 32× corpus, 120 s think cap, 600 s plus 60 s grace, seed 11, burst starts, straight to the engine as W0). The simulator runs `flatshare-sim run DATA/weka-x32 --arms lru-cpu --concurrency N`. Measured shares are the engine's `prompt_tokens_cached_by_source` counters over the profiling phase (the counters' deltas less the warm-up requests' own tokens); simulated shares cover the profiling requests. Resume TTFT covers profiling requests after a conversation's first turn. Percentiles interpolate between ranks. `make reproduce-calibration EXPORTS=<dir> CORPUS=DATA/weka-x32` writes every number of this section and the next, and BENCHMARKS' table of the three points, to `build/reproduce-calibration/calibration.md` (`scripts/calibration.py`). It needs six runs' AIPerf `profile_export.jsonl` and the engine's `/metrics` before and after: the three points, W0, and the 16- and 24-session repeats on a warm engine. They are an asset of the [v0.3 release](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3), `leaseholder-calibration-exports-2026-10-24.tar.gz` (named before the project's rename), and EXPORTS is its unpacked directory.

| Point | | Requests | Resume TTFT p50 / p90 / p99 (ms) | GPU / CPU / computed (%) |
|---|---|---|---|---|
| 8 sessions | measured | 408 | 83 / 400 / 1,463 | 66.6 / 29.0 / 4.4 |
| | idle (v0) | 410 (+0.5%) | 67 / 200 / 563 (−19 / −50 / −61%) | 67.6 / 28.1 / 4.3 |
| | contention | 405 (−0.7%) | 81 / 382 / 1,749 (−3 / −5 / +20%) | 66.1 / 29.5 / 4.4 |
| | leave-one-out | 405 | 83 / 400 / 1,772 (−0 / −0 / +21%) | 66.1 / 29.5 / 4.4 |
| 16 sessions | measured | 686 | 132 / 743 / 8,102 | 57.4 / 36.0 / 6.6 |
| | idle (v0) | 685 (−0.1%) | 79 / 317 / 4,287 (−40 / −57 / −47%) | 62.8 / 31.1 / 6.1 |
| | contention | 691 (+0.7%) | 127 / 685 / 7,776 (−4 / −8 / −4%) | 58.8 / 34.3 / 6.9 |
| | leave-one-out | 690 | 125 / 751 / 7,830 (−5 / +1 / −3%) | 59.1 / 33.9 / 7.0 |
| 24 sessions | measured | 1,568 | 173 / 1,345 / 15,172 | 52.9 / 34.3 / 12.9 |
| | idle (v0) | 1,756 (+12.0%) | 71 / 272 / 4,766 (−59 / −80 / −69%) | 59.5 / 28.6 / 11.9 |
| | contention | 1,649 (+5.2%) | 142 / 942 / 9,896 (−18 / −30 / −35%) | 55.9 / 32.4 / 11.7 |
| | leave-one-out | 1,646 | 137 / 826 / 9,931 (−21 / −39 / −35%) | 55.9 / 32.9 / 11.2 |

Leave-one-out refits every coefficient on the other two points and predicts the third. The repeats give the measurements' own spread: three 16-session runs (fresh, after the 8-session run, and W0) measured 132–140 ms at p50, 743–966 ms at p90 and 8,102–8,456 ms at p99, against which the model is −4 to −9%, −8 to −29% and −4 to −8%. The 24-session run after the 16-session one measured 213 / 1,798 / 18,403 ms (−33 / −48 / −46%): the starting state matters more there.

- **The cache model.** On each point's own timeline (`replay-log`) the engine model's GPU share is within 0.7, 0.3 and 1.8 points of the counters' at 8, 16 and 24 sessions, and its per-request cached share within 0.6, 1.3 and 2.0 points on average: about half of the closed loop's +3.0-point GPU share at 24 sessions is the cache model's.
- **8 and 16 sessions meet the targets** (p50 and p90 within 10%, shares within 2.1 points, requests within 1%) in sample and out of sample. p99 at 8 sessions is four requests; it moves 20% with one of them.
- **24 sessions does not**, and the evidence points at transfers. Open loop (`replay-log --timed`), the model is −17 / −47 / −56% at 24 sessions against +1 / −2 / −33% at 8 and −6 / −19 / +18% at 16. Hits match at every point (median residual 0, 0 and −5 ms) and so do computed prompts at 8 and 16 sessions (+12, +11 ms), but restores take 163 ms longer than modeled at 24 sessions (15 ms at 8, 31 ms at 16), the more so the more other restores overlap their measured wait: 41 ms with none, 102 ms with one, 163 ms with three, 1.2 s with six or more. The model gives each load E0's transfer time and never delays a block's reuse for a store; vLLM 0.32 stores as requests compute and, when it reallocates a block whose store is pending, waits for the store before the step, which happens most when the pool is full. Modeling store traffic is left for when E1 or E2 needs the 24-session regime.

## Validation against W0 (v0, the idle service)

W0 replayed the full 32× corpus against the frozen box: AIPerf's sequential sampler gave its 16 lanes the first 16 traces in name order, and its 10 recycled plays the next 10. The simulator runs the same configuration with seed 11, here with `--service idle`. `replay-log` feeds the engine model W0's own requests at their measured start and end times, which separates the cache model from the replay timing.

| | W0, measured | Engine model on W0's timeline | Full simulation |
|---|---|---|---|
| Profiling requests | 688 | 688 | 685 |
| Session plays, warmup requests | 26, 16 | — | 26, 16 |
| Prompt tokens | 3,268,924 | 3,122,538 | 3,113,792 |
| From the GPU pool | 56.54% | 56.96% (+0.42) | 62.84% (+6.30) |
| Restored from the CPU tier | 36.90% | 37.31% (+0.41) | 31.05% (−5.85) |
| Computed | 6.56% | 5.73% (−0.83) | 6.11% (−0.45) |
| TTFT p50 / p90 / p99 | 135 / 815 / 8,501 ms | — | 79 / 322 / 4,420 ms |

The measured split is the engine's `prompt_tokens_cached_by_source` counters over the run; its cached share agrees with the per-request `cached_tokens` (93.44%).

- **The cache model.** On W0's timeline the engine model is within half a point of the measured GPU and CPU shares, and its per-request cached share is within 1.35 points on average.
- **The timeline.** At the idle box's speed the full simulation's sessions return sooner than W0's (a mean modeled time to first token of 237 ms and service time of 783 ms, where W0 measured 541 ms and 1,269 ms), so fewer blocks pass through the pool between a session's turns and more prompts still hit the GPU. The contention model closes the gap: 58.8 / 34.3 / 6.9% against W0's 56.5 / 36.9 / 6.6% (profiling phase), and resume TTFT within 10% (above).
- **Prompt length.** The engine counted chat-template and cache-bust tokens the corpus does not hold (4.7% more).
- **Spread.** The seed sets where each lane starts in its trace, so it changes the workload itself. Seeds 11–15 give 639–955 requests and 60.9–72.2% / 21.8–32.5% / 4.4–7.4% (GPU / CPU / computed): one 10-minute, 16-lane run is a small sample.
- **The 16-trace subset** in `testdata/` is not W0's trace pool. Replayed the same way it is a lighter workload (559 requests, 0.74 M prompt tokens, KV pressure 0.9 against 3.3; 72.4 / 13.3 / 14.3% with seed 11), and it serves as the end-to-end test.

## The E1 regime map

E1's results, the regime map over its arms and the miss-ratio curves, are in
[BENCHMARKS.md](../../docs/BENCHMARKS.md#e1-parking-in-the-simulator), from `make reproduce-e1` (its `regime.md` and
`mrc.md`).

## Other hardware, modeled

Every cost above is measured on one RTX 5060 Ti. `flatshare-sim hardware` places other GPUs on the same map with costs modeled from their published specifications and the path vLLM 0.32.0 takes for the box's checkpoint on them. The box's rows use its measured costs (its placement is simulated like the others'); every other row is labeled modeled. `flatshare-sim forecast` writes the forecast for a run on one of them ([below](#a-gpu-runs-forecast)), and the A100 run of 2026-10-26 measured one, an A100 SXM4 80 GB, against its forecast ([HARDWARE.md](../../docs/HARDWARE.md#what-it-measured-2026-10-26)).

**Costs.** A GPU's profile is its published peaks (memory, memory bandwidth, dense FP8 and BF16 tensor throughput, the host link) and the kernel path its linear layers take. The peaks are used only as ratios to the box's: each measured cost term is divided by the ratio of the published peaks of the kind that bounds it (bandwidth to bandwidth, the linear layers' rate to the box's FP8 rate, link to link), so the box's measured efficiency carries over.
- **Memory bandwidth** scales the fixed step term and the KV reads. The fixed term is mostly the weight read: Qwen3-4B-FP8 reads 4.41 GB a step (3.63 G FP8 linear weights and the 0.78 GB BF16 embedding, tied to the LM head), 9.8 ms of the 11.56 ms at 448 GB/s.
- **The linear layers' rate** scales the per-token term, **BF16** the per-pair term (attention over the BF16 cache), and **the link** the restore's transfer; the restore's fixed 10.3 ms stays. The linear layers run at the FP8 rate where vLLM serves the box's block-FP8 checkpoint W8A8 (DeepGEMM or CUTLASS: the box, the H100), and at the BF16 rate over Marlin's slowdown where it serves it W8A16 through Marlin (the L4, the A100; [below](#the-linear-layers-path)).
- **E0's cold-prefill fit**, which gives ρ, is the box's fit times how much longer the scaled step loop takes a lone prefill than the box's step loop does, at each prompt length from 1k to 249k tokens (with E0's host time), refitted as a quadratic. It keeps, on every card, the relation it has on the box to the box's own loop (2% below it at 4k and 8k tokens, 14% above at 1k), within the refit's 5%.
- **Unchanged**, as the host's and the software's: the API server's host time and its extra per decoding request, the mixed-step extra, the restore stall, and the 512/256 step budget. `flatshare-sim forecast --host` takes the host's from a measured host instead ([below](#a-gpu-runs-forecast)).

The box's profile gives `StepCost::frozen_box()` and `TierCost::frozen_box()` back exactly.

**The pool.** Each GPU keeps beside its pool what the box keeps beside its own (the edge pattern: the engine's weights, activations and CUDA graphs, and STT and TTS on the same card), and the box's headroom:

`pool tokens = 16 · ⌊(memory − 11 GiB) / (16 · 144 KiB)⌋`

The 11 GiB is the box's 9,288 MiB of non-KV footprint (14,408 MiB in use with the three servers idle, less the 5 GiB pool) and 1,976 MiB of headroom of its 16 GiB (`nvidia-smi` shows 16,311 MiB, 1,903 MiB of them free). The formula gives the box's 36,400 tokens back. The memory is what each card leaves with the ECC it ships with: 22.5 GiB of the L4's 24, and all of the box's, the A100s' and the H100's (below). The CPU tier is 4× each pool, as in E1's realistic regime.

| GPU | Costs | Memory | Bandwidth | Dense FP8 / BF16 | Linear layers | Host link, each way | Pool | Step: fixed | per token | per pair | per KV token read | Restore | Cold prefill |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|
| RTX 5060 Ti 16 GB | measured | 16 GiB | 448 GB/s | 189.8 / 47.4 TFLOPS | FP8 W8A8 | PCIe 3.0 x8, 7.9 GB/s | 36,400 tokens | 11.56 ms | 47.99 µs | 14.17 ns | 429.3 ns | 10.3 ms + 23.64 µs/token | 47.8 ms + 63.11 µs/token + 8.126 ns/token² |
| L4 24 GB | modeled | 22.5 GiB | 300 GB/s | 242.5 / 121.0 TFLOPS | Marlin W8A16, 1.42× a BF16 GEMM | PCIe 4.0 x16, 31.5 GB/s | 83,728 tokens | 17.26 ms | 106.58 µs | 5.56 ns | 641.1 ns | 10.3 ms + 5.91 µs/token | 53.5 ms + 134.38 µs/token + 3.798 ns/token² |
| A100 40 GB | modeled | 40 GiB | 1,555 GB/s | none / 312.0 TFLOPS | Marlin W8A16, 1.42× a BF16 GEMM | PCIe 4.0 x16, 31.5 GB/s | 211,168 tokens | 3.33 ms | 41.33 µs | 2.15 ns | 123.7 ns | 10.3 ms + 5.91 µs/token | 35.2 ms + 47.47 µs/token + 1.340 ns/token² |
| A100 SXM4 80 GB | modeled | 80 GiB | 2,039 GB/s | none / 312.0 TFLOPS | Marlin W8A16, 1.42× a BF16 GEMM | PCIe 4.0 x16, 31.5 GB/s | 502,432 tokens | 2.54 ms | 41.33 µs | 2.15 ns | 94.3 ns | 10.3 ms + 5.91 µs/token | 34.9 ms + 46.00 µs/token + 1.307 ns/token² |
| H100 SXM 80 GB | modeled | 80 GiB | 3,350 GB/s | 1,979.0 / 989.5 TFLOPS | FP8 W8A8 | PCIe 5.0 x16, 63.0 GB/s | 502,432 tokens | 1.55 ms | 4.60 µs | 0.68 ns | 57.4 ns | 10.3 ms + 2.96 µs/token | 27.4 ms + 9.57 µs/token + 0.440 ns/token² |

The peaks come from NVIDIA: the [RTX 5060 family page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/) and [50-series comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/), the [RTX Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), and the [L4](https://www.nvidia.com/en-us/data-center/l4/), [A100](https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) and [H100](https://www.nvidia.com/en-us/data-center/h100/) (SXM) datasheets.
- **The box's tensor throughput.** NVIDIA quotes it only as 759 AI TOPS, FP4 with sparsity, which is 379.5 dense FP4 TFLOPS. The 5060 Ti is not in the whitepaper's tables; its rates follow by the ratios every GeForce Blackwell card there has: half its FP4 rate in FP8 with FP16 accumulate, and a quarter of that in BF16, whose only rate is with FP32 accumulate. FP8 with FP32 accumulate runs at half rate on these cards, and E0's recompute cell says the box's linear layers beat that rate: at 4,096 tokens, 437 ms less the 48 ms fixed part leaves 389 ms, where the linear layers alone at the FP32-accumulate peak (94.9 TFLOPS) would take 314 ms and attention at 47.4 TFLOPS about 104 ms more, 418 ms. At the FP16-accumulate peak they take 157 ms, which fits, so FP16 accumulate is the peak of the same kind.
- **The others.** The L4's, A100's and H100's throughputs are quoted with sparsity and halved. The A100 has no FP8.
- **The A100 80 GB.** The datasheet's A100 80GB SXM has 80 GB of HBM2e at 2,039 GB/s (its PCIe card 1,935), and the 40 GB card's BF16 rate, sm80 and host link, so its profile is the 40 GB card's with the memory and the bandwidth changed: its costs differ in the bandwidth terms alone, and its pool is the H100's. The A100 run drew one; the map below does not place it, and `flatshare-sim forecast --gpu a100-80gb` forecast the run on it before it ran ([`benchmarks/a100/forecast-80gb.md`](../../benchmarks/a100/forecast-80gb.md)).
- **The links.** The box's link is what both of its hosts negotiate (`nvidia-smi`'s `pcie.link.gen.hostmax` 3, width 8), not the card's PCIe 5.0. Each rate is the line rate with 128b/130b encoding.
- **ECC.** The L4's 24 GB are GDDR6 with ECC enabled by default ([product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf), tables 2 and 3), and with ECC on, GDDR memory keeps its check bits in 6.25% of itself ([CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#theoretical-bandwidth-calculation), 9.2.1). That leaves 22.5 GiB, of which `nvidia-smi` shows 23,034 MiB ([TensorRT-LLM #1823](https://github.com/NVIDIA/TensorRT-LLM/issues/1823)), as it shows 16,311 MiB of the box's 16 GiB, from which the box's headroom is counted. HBM has resources of its own for ECC (the guide, of HBM2): an H100 SXM with ECC on shows 81,559 MiB ([an HGX H100's `nvidia-smi`](https://www.icc-usa.com/blog/hgx-8x-h100-benchmarks)), more than a sixteenth less would leave, as do the box's 16,311 MiB, an A100-SXM4-40GB shows 40,960 MiB ([NVIDIA's forum](https://forums.developer.nvidia.com/t/a100-sxm4-40gb-shows-in-nvidia-smi-but-not-accessible-in-cuda/322730)) and an A100-SXM4-80GB 81,920 MiB ([NVIDIA's DCGM #93](https://github.com/NVIDIA/DCGM/issues/93)).

### The linear layers' path

Qwen3-4B-Instruct-2507-FP8 quantizes its weights in 128×128 blocks with dynamic activations. For such a checkpoint vLLM 0.32.0's `Fp8LinearMethod` takes the block-FP8 kernel list, and the first kernel in it that supports the card serves every linear layer (`model_executor/kernels/linear/__init__.py:478-488, 743-755` at v0.32.0). On sm75 to sm89 every kernel ahead of `MarlinFP8ScaledMMLinearKernel` refuses the card (DeepGEMM needs sm90 or the sm100 or sm120 families, CUTLASS's block kernel sm90 or newer, B12x the sm120 family), so the L4 (sm89) and the A100 (sm80) run Marlin, which is weight-only: FP8 weights, BF16 activations, at the BF16 tensor rate. [vLLM PR #61224](https://github.com/vllm-project/vllm/pull/61224) observed it on an L4 ("vLLM logs 'Your GPU does not have native support for FP8 computation'") and measured Qwen3-8B's four linear layers at the step cap's 512 tokens 1.42× slower on Marlin than in BF16 (5,046.9 µs against 3,563.7, the sums of its medians). The model uses that factor on both cards; nothing published measured Marlin's FP8 path on an A100 before the A100 run (below). The box (sm120) runs `DeepGemmFp8BlockScaledMMKernel` (its engine log), the H100 (sm90) DeepGEMM or FlashInfer's DeepGEMM kernel, both W8A8. So the L4's linear layers take 106.6 µs a token, against the box's 48.0 and the 37.5 its FP8 rate would give; the A100's 41.3.

**Checks against published vLLM measurements.**
- **Restores.** The Gen4 x16 term gives 34.5, 58.7 and 107.1 ms for 4k, 8k and 16k tokens. vLLM 0.29 measured 34, 65 and 104 ms for Qwen3-4B's CPU-tier restores on an L20, also PCIe Gen4 ([#57103](https://github.com/vllm-project/vllm/issues/57103#issuecomment-5733975478)), so the box's 79% of its link carries over to Gen4 within 10%. Nothing checks it on Gen5.
- **The fixed step.** vLLM 0.30 decoding Qwen3-4B in BF16 at batch 1 on an H100 SXM takes 4.29 ms a step ([decode-decoded](https://github.com/zhebrak/decode-decoded), `results/cells.csv`): 2.45 ms of it is the weight read at the datasheet's bandwidth, and 1.8 ms the rest (launches, kernel excess, bandwidth short of the datasheet's). The box's fixed term has about the same 1.7 ms beyond its weight read, and bandwidth scaling shrinks that to 0.23 ms. So the H100's 1.55 ms is a floor: with the overhead kept it would be 3.0 ms, and the A100's 4.55 ms against 3.33.
- **The L4 and the A100.** No published vLLM decode measurement of a model this size on either was found. FlashAttention 2's forward pass reaches 50-73% of the A100's peak ([its paper](https://arxiv.org/abs/2307.08691)), where the box's attention term carries 88% of the box's: the A100's attention is modeled at the box's efficiency here and over FlashAttention 2's range in the forecast.

**Checked on an A100.** The A100 run (2026-10-26) fitted the step model on its calibration point ([`a100.md`](../../benchmarks/results/2026-10-26-a100/a100.md)): the per-token term, Marlin's linear layers, came in at 41.36 µs against the 41.33 modeled, and every other term above the model's: the fixed step at 4.5 ms against 2.54 (3.88 with the box's overhead kept), 4.07 ns a pair against 2.15, and 171.9 ns a KV token read against 94.3, its decode attention reading KV at 42% of the card's bandwidth where the box's reaches 77% ([HARDWARE.md](../../docs/HARDWARE.md#what-it-measured-2026-10-26)).

**Workloads.** The 32× scaling was chosen for the box's pool, so the GPUs are placed twice:
- **The 32× replay**, the same on every GPU.
- **Each GPU's own scaling** (`--fit`): the least power-of-two scaling whose longest request fits both the pool and Qwen3-4B's 262,144-token context, as the 32× corpus fits the box's. The box replays 32×, the L4 16×, the A100 8× and the H100 4×; on the H100 the model's context binds before its pool does.

The original-scale corpus (`cc-traces-weka-062126`) fits no GPU (`--fit` says so): its median prompt is 142k tokens, and its longest request has 996,579 tokens of prompt and output. Of its requests, 88% exceed the box's pool, 68% the L4's and 12% the H100's, and 30% exceed Qwen3-4B's context.

**The realtime tier** is two voice calls, as in E2's discriminating scenario, at 1,024 tokens of KV each: a voice turn's prompt and reply came to about 680 tokens in the W2 runs on the box. The replay measures two shares of its window:
- **Working set:** the time the agents' live working set leaves less than 2,048 tokens of the pool free. A voice turn's KV then displaces cached agent KV, which vLLM evicts for it; the agents pay for that when they resume.
- **Running:** the time the KV of running agent requests leaves less than that free. A voice turn then waits for blocks, as vLLM 0.32 preempts no running request for a waiting one: #40004's KV-blocking regime, where tuned vLLM fails. Otherwise a voice turn meets compute interference alone, which the step cap and priority bound: it waits behind steps carrying the agents' prefill chunks. The chunk column gives one 512-token chunk at the median prompt; on the box a decode step alone takes about 14 ms.

Means over seeds 11, 12 and 13, trace pauses. Pressure is mean / peak; the voice KV shares are working set / running; resume TTFT p50 / p90 is in ms; gains are against `lru-cpu` at p50 / p90. The box's rows are the E1 map's cells. On the 32× replay:

| GPU | Sessions | Pressure | π | ρ | Voice KV short | Chunk | `lru-cpu` | `clairvoyant` | `oracle-prefetch` | `history-prefetch` | `tokencake` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX 5060 Ti 16 GB (measured), 32× | 5 | 1.3 / 1.5 | 27.0 | 0.26 | 94 / 0% | 64 | 90 / 243 | +1 / +7% | +2 / +13% | −1 / +0% | −5 / −9% |
|  | 8 | 1.8 / 2.0 | 29.6 | 0.27 | 100 / 0% | 56 | 86 / 355 | +1 / +27% | +3 / +40% | +1 / +11% | −6 / −8% |
|  | 12 | 2.6 / 3.1 | 21.4 | 0.25 | 100 / 1% | 64 | 113 / 340 | +0 / +7% | +5 / +32% | −1 / −6% | −10 / −21% |
|  | 16 | 3.6 / 4.1 | 23.4 | 0.25 | 100 / 4% | 66 | 180 / 1,399 | +2 / +8% | +15 / +10% | −11 / −27% | −27 / −22% |
|  | 24 | 4.7 / 5.4 | 23.9 | 0.26 | 100 / 12% | 61 | 364 / 2,334 | +3 / +8% | −25 / −14% | −31 / −22% | −60 / −23% |
|  | 32 | 6.2 / 6.8 | 23.5 | 0.26 | 100 / 25% | 64 | 1,849 / 9,109 | +3 / +2% | −48 / −24% | −40 / −17% | −37 / −11% |
|  | 40 | 7.5 / 8.1 | 22.9 | 0.26 | 100 / 33% | 63 | 8,216 / 19,336 | +1 / +1% | −55 / −27% | −46 / −29% | −47 / −22% |
| L4 24 GB (modeled), 32× | 5 | 0.5 / 0.7 | 78.0 | 0.06 | 0 / 0% | 85 | 101 / 212 | +0 / +0% | +0 / +0% | +0 / +0% | −5 / −6% |
|  | 8 | 0.8 / 0.9 | 83.2 | 0.06 | 0 / 0% | 81 | 97 / 205 | +0 / +0% | +0 / +0% | +0 / +0% | −4 / −12% |
|  | 12 | 1.1 / 1.3 | 62.3 | 0.06 | 83 / 0% | 85 | 127 / 249 | +0 / +3% | +2 / +7% | +1 / +3% | −7 / −14% |
|  | 16 | 1.5 / 1.8 | 70.4 | 0.05 | 100 / 1% | 85 | 156 / 325 | +0 / +5% | +4 / +7% | +1 / −5% | −11 / −14% |
|  | 24 | 2.0 / 2.3 | 68.4 | 0.06 | 100 / 1% | 83 | 158 / 354 | +0 / +3% | +2 / +13% | −1 / −1% | −18 / −13% |
|  | 32 | 2.7 / 3.0 | 71.0 | 0.05 | 100 / 3% | 85 | 224 / 640 | +0 / +1% | +9 / +10% | −8 / −8% | −25 / −10% |
|  | 40 | 3.3 / 3.6 | 70.3 | 0.05 | 100 / 7% | 85 | 377 / 2,054 | +1 / +3% | +18 / +3% | −7 / −4% | −15 / −12% |
|  | 64 | 5.5 / 5.8 | 77.7 | 0.06 | 100 / 23% | 84 | 10,851 / 26,546 | +0 / +2% | −6 / −3% | −15 / −4% | −12 / −6% |
| A100 40 GB (modeled), 32× | 5 | 0.2 / 0.3 | 81.9 | 0.14 | 0 / 0% | 29 | 69 / 155 | +0 / +0% | +0 / +0% | +0 / +0% | −6 / −3% |
|  | 8 | 0.3 / 0.4 | 76.5 | 0.16 | 0 / 0% | 28 | 63 / 133 | +0 / +0% | +0 / +0% | +0 / +0% | −4 / −11% |
|  | 12 | 0.4 / 0.5 | 66.2 | 0.14 | 0 / 0% | 29 | 75 / 141 | +0 / +0% | +0 / +0% | +0 / +0% | −9 / −12% |
|  | 16 | 0.6 / 0.7 | 71.7 | 0.14 | 0 / 0% | 29 | 81 / 180 | +0 / +0% | −0 / +0% | −0 / +0% | −10 / −14% |
|  | 24 | 0.8 / 0.9 | 72.4 | 0.15 | 1 / 0% | 28 | 79 / 171 | +0 / +1% | +0 / +1% | −0 / +1% | −12 / −14% |
|  | 32 | 1.1 / 1.2 | 72.4 | 0.14 | 85 / 0% | 29 | 96 / 243 | +1 / +2% | +1 / +5% | +1 / +0% | −16 / −8% |
|  | 40 | 1.3 / 1.4 | 72.1 | 0.14 | 100 / 0% | 29 | 103 / 298 | +1 / +3% | +2 / +4% | +1 / −2% | −21 / −11% |
|  | 64 | 2.2 / 2.3 | 85.7 | 0.15 | 100 / 0% | 29 | 135 / 735 | +1 / +1% | +5 / +5% | −2 / −6% | −26 / −11% |
|  | 96 | 3.1 / 3.3 | 79.5 | 0.15 | 100 / 0% | 28 | 270 / 2,231 | +2 / +1% | +3 / −2% | −19 / −13% | −44 / −11% |
|  | 128 | 4.1 / 4.3 | 86.1 | 0.15 | 100 / 1% | 28 | 5,876 / 9,281 | +1 / +1% | −26 / −21% | −28 / −25% | −23 / −14% |
|  | 192 | 6.2 / 6.4 | 108.7 | 0.15 | 100 / 0% | 28 | 51,789 / 81,315 | +0 / +0% | −12 / −4% | −7 / −3% | −3 / −5% |
| H100 SXM 80 GB (modeled), 32× | 5 | 0.1 / 0.1 | 124.4 | 0.32 | 0 / 0% | 5 | 62 / 133 | +0 / +0% | +0 / +0% | +0 / +0% | −2 / −12% |
|  | 8 | 0.1 / 0.1 | 102.8 | 0.33 | 0 / 0% | 5 | 54 / 126 | +0 / +0% | +0 / +0% | +0 / +0% | −8 / −7% |
|  | 12 | 0.2 / 0.2 | 100.0 | 0.31 | 0 / 0% | 5 | 65 / 127 | +0 / +0% | +0 / +0% | +0 / +0% | −7 / −9% |
|  | 16 | 0.3 / 0.3 | 109.4 | 0.31 | 0 / 0% | 5 | 70 / 164 | +0 / +0% | +0 / +0% | +0 / +0% | −10 / −11% |
|  | 24 | 0.3 / 0.4 | 106.2 | 0.32 | 0 / 0% | 5 | 67 / 143 | +0 / +0% | +0 / +0% | +0 / +0% | −11 / −14% |
|  | 32 | 0.5 / 0.5 | 108.6 | 0.31 | 0 / 0% | 5 | 81 / 205 | +0 / +0% | +0 / +0% | +0 / +0% | −13 / −6% |
|  | 40 | 0.5 / 0.6 | 108.1 | 0.31 | 0 / 0% | 5 | 84 / 251 | +0 / +0% | +0 / +0% | +0 / +0% | −15 / −7% |
|  | 64 | 0.9 / 1.0 | 126.8 | 0.32 | 0 / 0% | 5 | 102 / 566 | +1 / +1% | +2 / +1% | +0 / −0% | −14 / −6% |
|  | 96 | 1.3 / 1.4 | 116.4 | 0.32 | 100 / 0% | 5 | 158 / 1,419 | +1 / +0% | +1 / +1% | +0 / +1% | −12 / +2% |
|  | 128 | 1.7 / 1.8 | 120.1 | 0.32 | 100 / 0% | 5 | 481 / 2,996 | +1 / +0% | −1 / +0% | −2 / −0% | −29 / −10% |
|  | 192 | 2.6 / 2.8 | 131.3 | 0.32 | 100 / 0% | 5 | 4,756 / 7,599 | +0 / +0% | −2 / −3% | −8 / −5% | −12 / −10% |
|  | 256 | 3.5 / 3.6 | 129.4 | 0.32 | 100 / 0% | 5 | 12,192 / 13,843 | +0 / +0% | −5 / −6% | −6 / −7% | −9 / −12% |

At each GPU's own scaling (`--fit`; the box's rows are those above):

| GPU | Sessions | Pressure | π | ρ | Voice KV short | Chunk | `lru-cpu` | `clairvoyant` | `oracle-prefetch` | `history-prefetch` | `tokencake` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L4 24 GB (modeled), 16× | 5 | 1.1 / 1.2 | 52.7 | 0.04 | 70 / 0% | 99 | 173 / 456 | +2 / +0% | +4 / +5% | −3 / +0% | −7 / −3% |
|  | 8 | 1.6 / 1.8 | 51.4 | 0.05 | 100 / 1% | 90 | 175 / 510 | +1 / +3% | +5 / +14% | +2 / −4% | −5 / −11% |
|  | 12 | 2.3 / 2.6 | 38.3 | 0.05 | 100 / 2% | 95 | 237 / 788 | +0 / +1% | +7 / +9% | +2 / −3% | −10 / −7% |
|  | 16 | 3.1 / 3.5 | 38.6 | 0.04 | 100 / 9% | 97 | 2,019 / 10,927 | +1 / +0% | +15 / +4% | −2 / −0% | −6 / +4% |
|  | 24 | 4.2 / 4.6 | 38.4 | 0.05 | 100 / 22% | 94 | 7,432 / 22,720 | +1 / +1% | −16 / −8% | −10 / −3% | −13 / −6% |
|  | 32 | 5.3 / 5.6 | 40.9 | 0.05 | 100 / 19% | 93 | 27,550 / 59,576 | +1 / +1% | −20 / −4% | −13 / −4% | −13 / −4% |
|  | 40 | 6.4 / 6.6 | 56.0 | 0.05 | 100 / 11% | 92 | 81,713 / 136,552 | +0 / +0% | +1 / −0% | +2 / −0% | +1 / −1% |
| A100 40 GB (modeled), 8× | 5 | 0.9 / 1.0 | 27.1 | 0.10 | 11 / 0% | 42 | 193 / 536 | +1 / +1% | −0 / −1% | −2 / −3% | −6 / −5% |
|  | 8 | 1.2 / 1.4 | 29.3 | 0.11 | 100 / 0% | 37 | 188 / 646 | +1 / +10% | +3 / +13% | +2 / +4% | −1 / +0% |
|  | 12 | 1.8 / 2.1 | 20.9 | 0.10 | 100 / 0% | 41 | 251 / 715 | +0 / +4% | +7 / +11% | +2 / −2% | −4 / −10% |
|  | 16 | 2.5 / 2.8 | 21.8 | 0.10 | 100 / 2% | 43 | 456 / 3,403 | +1 / +1% | −2 / +3% | −11 / −15% | −16 / −20% |
|  | 24 | 3.3 / 3.6 | 23.3 | 0.10 | 100 / 8% | 40 | 2,402 / 6,573 | +2 / +1% | −18 / −27% | −6 / −28% | −18 / −21% |
|  | 32 | 4.2 / 4.5 | 21.3 | 0.10 | 100 / 10% | 39 | 9,380 / 26,816 | +1 / +1% | −21 / −5% | −16 / −7% | −15 / −8% |
|  | 40 | 5.2 / 5.5 | 24.0 | 0.10 | 100 / 10% | 39 | 28,145 / 49,509 | +1 / +1% | −10 / −5% | −9 / −10% | −11 / −10% |
| H100 SXM 80 GB (modeled), 4× | 5 | 0.7 / 0.8 | 29.1 | 0.14 | 0 / 0% | 16 | 327 / 1,017 | +0 / +1% | +0 / +0% | +0 / +0% | −3 / −1% |
|  | 8 | 1.0 / 1.2 | 30.5 | 0.18 | 69 / 0% | 12 | 288 / 1,095 | +0 / +3% | −1 / +6% | +1 / +1% | −1 / −1% |
|  | 12 | 1.5 / 1.8 | 21.5 | 0.15 | 100 / 1% | 14 | 426 / 1,321 | +0 / +2% | +2 / +7% | −2 / +8% | −0 / +3% |
|  | 16 | 2.1 / 2.3 | 21.1 | 0.15 | 100 / 2% | 15 | 1,223 / 8,026 | +1 / +0% | +1 / +4% | −1 / +0% | −3 / −1% |
|  | 24 | 2.8 / 3.0 | 21.6 | 0.16 | 100 / 6% | 14 | 4,531 / 14,102 | +1 / +0% | −13 / −4% | −15 / −3% | −25 / −2% |
|  | 32 | 3.6 / 3.8 | 21.5 | 0.16 | 100 / 7% | 13 | 11,191 / 29,027 | +0 / +0% | −10 / −13% | −14 / −11% | −11 / −12% |
|  | 40 | 4.3 / 4.6 | 25.3 | 0.16 | 100 / 6% | 13 | 30,759 / 50,569 | +0 / +0% | −14 / −5% | −15 / −7% | −15 / −6% |
|  | 64 | 7.2 / 7.4 | 43.3 | 0.14 | 100 / 1% | 16 | 266,358 / 329,898 | +0 / +0% | −2 / −2% | −2 / −2% | −2 / −2% |

Rows whose resume TTFT runs to seconds are saturated, and there the running share falls, as requests queue before admission. The modeled GPUs saturate at these points:
- **The L4** saturates its compute from 64 sessions at 32× (resume TTFT p50 10.9 s) and from 16 sessions at 16× (2.0 s).
- **The A100** from 128 sessions at 32× (5.9 s) and from 24 sessions at 8× (2.4 s).
- **The H100 at 4×** saturates its compute from 16 sessions (90% busy), on attention over contexts of 20–30k tokens.
- **The H100 at 32×, from 128 sessions,** queues at the modeled API server instead (one request at a time, at the box's host cost per token and per decoding request), with its GPU 61–73% busy.

**What each GPU's regime means for a realtime tier beside agents.**
- **RTX 5060 Ti (measured costs).** From 8 sessions the agents' working set leaves less than two calls' KV free all the time, so a voice turn's KV always displaces cached agent KV. Running agents hold the pool against a voice turn 0–4% of the time up to the box's 16-session operating point, 12% at 24 sessions and 25–33% at 32–40.
  - Up to 16 sessions, a realtime tier on the box meets compute interference. It waits behind a 56–66 ms prefill chunk, which the step cap bounds and priority orders.
  - From 24 sessions, it is in #40004's regime an eighth to a third of the time, where it needs KV freed for it. In E2's discriminating scenario (24 agent sessions of the longest-context traces), freeing KV was not enough: the engine also had to keep lower-priority prefill out of the turn's first tokens (make-room's budget cap; [EXPERIMENTS.md](../../docs/EXPERIMENTS.md#e2-voice-under-agent-load)). What headroom held ahead of the turn could not do there is in [METHODS.md](../../docs/METHODS.md#what-lifecycle-timing-could-and-could-not-do).
  - π is 21–30 and ρ 0.25–0.27. Lifecycle information buys 13% at p90 at pressure 1.3, 32–40% at 1.8–2.6 and 10% at 3.6, and it loses from 4.7 (the E1 map).
- **L4 (modeled).** Against the box it has 2.3× the pool, a 4× faster link and 2.6× the BF16 throughput, but it runs the box's checkpoint through Marlin, whose linear layers take 2.2× the box's FP8 time, and it has two thirds of the memory bandwidth: its decode steps and its prefill are both slower.
  - On the 32× replay, running agents block two calls at most 1% of the time up to 24 sessions. π rises to 62–83 (restores four times cheaper per token) and ρ falls to 0.05–0.06, so restoring beats recomputing about twenty to one.
  - At its own 16×, it sits where the box sits at fewer sessions: pressure 1.1–3.1 at 5–16 sessions, π 38–53. Its compute runs out first (resume TTFT p50 0.2 s at 12 sessions, 2.0 s at 16), where running agents block two calls 9% of the time.
  - A realtime tier on an L4 meets compute interference behind chunks longer than the box's (81–99 ms) until the agents saturate the card, from 12 to 16 sessions.
- **A100 40 GB (modeled).** Against the box it has 5.8× the pool, 3.5× the bandwidth, a 4× faster link and 6.6× the BF16 throughput, and its Marlin linear layers are slightly faster than the box's FP8 ones (41.3 against 48.0 µs a token).
  - **The 32× replay barely fills it.** Pressure stays under 1.5 up to 40 sessions, and running agents block two calls at most 1% of the time at any count to 192. The chunk is 28–29 ms, half the box's, π 66–109 and ρ 0.14–0.16.
  - **At its own 8×**, its π (21–29) is the box's and its ρ lower (0.10–0.11). Running agents block two calls 2–10% of the time from 16 to 40 sessions, and its compute saturates from 24 sessions, at pressure 3.3.
  - Lifecycle information buys up to 13% at p90 at 8× (`oracle-prefetch` at 8 sessions).
  - The A100 run (2026-10-26) measured the box's discriminating scenario on an A100 SXM4 80 GB at the box's pool, which these cells do not place; its forecast is [`benchmarks/a100/forecast-80gb.md`](../../benchmarks/a100/forecast-80gb.md), and what it measured is in [HARDWARE.md](../../docs/HARDWARE.md#what-it-measured-2026-10-26).
- **H100 SXM (modeled).**
  - **The 32× replay cannot fill it.** Pressure stays under 1 up to 64 sessions, and running agents never block two calls (nor 28) at any count. The chunk is 5 ms, π 100–131 and ρ 0.31–0.33, and the clairvoyant reference gains at most 1%: lifecycle information has nothing to buy.
  - **At its own 4×**, its π (21–31 up to 40 sessions) is near the box's and its ρ lower (0.14–0.18). Long recomputes are dear in attention while the Gen5 link keeps restores cheap. Its compute saturates from 16 sessions, at pressure 2.
  - **A small realtime tier** meets compute interference with 12–16 ms chunks, a quarter of the box's; running agents block two calls 0–7% of the time.
  - **A realtime tier sized like the box's** (28 calls, the 5.6% of the pool that two calls are of the box's) is blocked 25% of the time at 16 sessions and 49–65% at 24–32, more often than on the box.
  - So the KV-blocking regime follows the agents' context lengths against the pool, and the realtime tier's share of it, not the consumer card.

**Assumptions, by how much they could move a placement.** Each was rerun with the H100 at 4× and the L4 at 16× (seeds 11–13), except where the size is given otherwise. The reruns predate the L4's Marlin path and the refitted cold prefill: their L4 sizes are those of the L4's earlier FP8 model.
1. **The scaling of the workload.** On the H100, 4× against 32× multiplies pressure by 8 at equal sessions (the working set is in tokens), roughly halves ρ (from 0.32–0.34 to 0.15–0.19) and nearly triples the chunk (from 5 to 12–16 ms). At the box's share of the pool, it takes the running share from 0% to 25–65%; hence the two workloads. `--fit` is bounded by Qwen3-4B's context: a model with a 1M-token context would put the H100 at 2×, with longer contexts and more pressure.
2. **The realtime tier's size** (two calls of 1,024 tokens). Given the box's share of the pool (5 calls on the L4, 28 on the H100), the running share rises from 9–21% to 23–44% on the L4 at 16–32 sessions, and from 2–7% to 25–65% on the H100 at 16–40.
3. **The restore's efficiency on the link** (the box's 79% of the line rate; checked within 10% on Gen4, not on Gen5). Restores 1.6× slower would cut π by a third and raise ρ by half on both GPUs, and move nothing else on the map.
4. **Attention scaling with the box's BF16 peak** (FP32 accumulate, 47.4 TFLOPS). If the box's attention were bound by a rate twice that, the modeled GPUs' attention would run half as fast. ρ would fall by a third on the H100 and a fifth on the L4, and the chunk would lengthen by 60–70% and 25–35%.
5. **The footprint and headroom** (the box's absolute 9,288 and 1,976 MiB on every card). A headroom of the box's 12% of memory would shrink the L4's pool by 7% and the H100's by 11%, raising pressure as much and the running share by up to 3 points.
6. **The fixed step** (scaled by bandwidth; the H100 measurement above says this is a floor). Keeping the box's 1.7 ms of overhead (3.0 ms on the H100) moves no coordinate beyond the seeds' noise. It adds 1–2 ms to the chunk and up to 7 points of GPU busy at 4×, and on the 32× replay 2 ms and up to 17 points; on the L4 it is 5% shorter than modeled.
7. **The host** (the box's API server: one request at a time, 29.6 ms + 8 µs a token and 6.5 ms per decoding request, from a 1.7 GHz Xeon through a tunnel). A host three times faster cuts low-load resume TTFT by 40–60%. It moves the H100's saturation on the 32× replay from about 128 to past 192 sessions (TTFT p50 4.8 s to 59 ms at 192), and no coordinate beyond the seeds' noise.
8. **Not varied:**
   - the mixed-step extra (7.6 ms), the restore stall (0.8) and the 512/256 step budget;
   - a CPU tier of 4× (276 GiB of host RAM per H100);
   - Qwen3-4B-FP8 on every card (an H100 would serve a larger model, with more KV per token);
   - the L4's bandwidth with ECC on: its datasheet's 300 GB/s, where the CUDA guide above puts ECC's cost on a GDDR card at about 20% of effective bandwidth, more for some access patterns;
   - the calibration's caveat: stores are not modeled, so restores at high pressure are optimistic on every GPU.

## A GPU run's forecast

`flatshare-sim forecast WHOLE LONG --gpu a100-40gb` writes the forecast for a run on another GPU, before the run: the 32× corpus (`WHOLE`) and its 48 traces with the largest contexts (`LONG`, E2's discriminating corpus, which `benchmarks/e2/long_traces.py` links). `make forecast-a100 CORPUS=DATA/weka-x32` writes the A100 40 GB's to [`benchmarks/a100/forecast.md`](../../benchmarks/a100/forecast.md), and with `GPU=a100-80gb` the A100 SXM4 80 GB's to [`benchmarks/a100/forecast-80gb.md`](../../benchmarks/a100/forecast-80gb.md). Everything in it for the card is modeled from the box's measured costs; where the box has a measurement of the same thing, the forecast is that measurement times the modeled ratio of the two cards, so that what the box's model misses of its own measurement does not carry over. It gives:
- **The costs in three settings**, besides the E1 map's: **low**, the peaks' ratios alone with Marlin as fast as a BF16 GEMM; **central**, with the box's 1.7 ms of step overhead kept (the H100 measurement above), Marlin as on the L4, and attention at FlashAttention 2's best published efficiency on the card where one is published (73% on the A100); **high**, with FlashAttention 2 at its worst published (50%). For Qwen3-14B-FP8 too, scaled by its bytes and FLOPs (`Model`): 14.77 GB read a step, 26.4 GFLOP a token, 819,200 FLOPs an attention pair, 160 KiB of KV a token.
- **E0's cells**, the box's measured p50s times the ratio of the two cards' modeled lone requests (GPU hit, restore, recompute).
- **The engine probes' cases**: the KV case's stock wait (two 18k-token decodes into the last free blocks, on Gray 4.9–5.1 s, by the ratio of the two cards' decode steps), make-room's first token there (225 and 297 ms on Gray), the reply's token gaps behind agent chunks and with the budget cap (69 and 17 ms on Gray); and the **time-equivalent step cap**, the largest multiple of 256 tokens whose chunk takes the card no longer than the box's 512-token chunk takes the box, at the discriminating scenario's median context (13.9k tokens) and the whole corpus's (7.9k).
- **E2's scenarios on the regime map**, `lru-cpu` (the stock engine's policy) with a realtime tier of two calls, over seeds 11 to 15 (E2's rounds' agent samples): the core (the discriminating scenario at the box's pool and 24 sessions, with the box's own cells beside it), the calibration point (the whole corpus at 16 sessions), Qwen3-14B-FP8 at the card's own pool, twice the box's pool and sessions, and Qwen3-4B at the card's own pool. The table adds the share of time the card's steps run (busy) and the requests that waited for free blocks (stalled) to the map's coordinates.

Once the run has measured the card, `--costs SUMMARY --step-costs STEPS` (both or neither, read as `run` reads them: the card's E0 `summary.json` and the step-time model `scripts/fit_step_cost.py` fits on its calibration point) add a section after the scenario table with the card's fitted costs: the calibration point, to read the fitted model against the measured point, and the core at the box's pool at each of `--fitted-sessions` (24 to 40 by default), with the count whose running share comes closest to the box's in the core, the count to run A' at. The other tables do not change. `make forecast-fitted CORPUS=DATA/weka-x32 COSTS=SUMMARY STEPS=STEPS` writes it for the A100 SXM4 80 GB to `build/forecast-fitted.md` (`GPU` and `OUT` change them).

Without `--host`, the card's API server runs at the box's CPU's speed and its restores at its datasheet's link. The A100 run showed both to be the host's. Its host's API server took 14.2 ms + 4.971 µs a prompt token (`scripts/fit_step_cost.py` on its calibration point) where the box's takes 29.6 ms + 7.994, and E0's GPU hit grew with the prompt by 1.80 µs a token there against the box's 3.79 (least squares over E0's seven lengths). Its restores moved 11.46 GB/s of KV, 12.86 µs a token beyond the GPU hit, where the forecast took 25 GB/s from the card's PCIe Gen4 x16 (5.91 µs): the host's Cascade Lake CPU has PCIe 3.0 root ports, so the card's Gen4 link to its switch is not the host's. `--host HOST_JSON` places the card on a host as `benchmarks/a100/bench_host.py` measured it (its `host.json`, refused without `python_ms.p50` or the link's `h2d` row of one copy). The API server's time, on a prompt and for each request decoding, and E0's are then the box's over how many times faster the host's CPU runs the script's Python workload (the box's Xeon E5-2609 v4 at 1.70 GHz takes 381.1 ms), and a restore's transfer is a token's KV at the host's one-copy rate times 0.92, the share of its own the box's E0 restore reaches (6.24 GB/s of 6.76). The restore's fixed part stays the connector's, and the mixed-step extra and the restore stall the software's: one is a step's, the other a share of the transfer. The report names the host in its introduction, adds the card's costs on it to each costs table (`central, on this host`), and places E0's cells and the card's scenario cells there. `make forecast-a100` and `make forecast-fitted` take it as `HOST_JSON=<host.json>`; `make forecast-a100` then writes the forecast beside the one made before the run, with `-on-host` in its name.

## Runtime

Release builds on the full corpus (393 traces, 98,827 requests, 1.7 GB of JSON), on an Apple M5 Max. Times are the whole process's CPU time (user and system), with the wall time on 18 threads and the peak memory for the runs that use every core; each time is the lower of two runs, each peak the higher. Other work loaded the machine while they were measured (load average 15 to 49), and that moved even CPU time by up to a quarter between two runs. The full-corpus runs use `--duration 100000000 --grace 0 --max-sessions 393 --start-max 0`, so every trace plays once from its start.

| Run | Contention | Idle |
|---|---|---|
| The W0 configuration, the three stock arms | 0.22 s | 0.22 s |
| The W0 configuration, `lru-cpu`, the E1 arms and the reference | 0.33 s | — |
| Every trace through 16 recycling lanes, the three stock arms | 13.2 s | 13.5 s |
| `regime` at one seed on one thread, 42 cells × 10 arms | 13.6 s | — |
| `regime`, 42 cells × 3 seeds × 10 arms | 89 s (9.7 s on 18 threads, 2.2 GB) | — |
| The realistic regime, 4 cells × 20 seeds × 10 arms | 41 s (4.0 s on 18 threads, 2.2 GB) | — |
| `hardware`, the 32× replay on four GPUs, 114 cells × 10 arms, on peacock | 207 s (53 s on 16 threads, 3.2 GB) | — |
| `hardware --fit`, 87 cells × 10 arms, on peacock | 236 s (51 s on 16 threads, 1.7 GB) | — |

The two `hardware` rows were measured on peacock (Xeon Silver 4110 at 2.1 GHz, 16 threads, Linux) once the A100 joined the profiles, one run each; there `make reproduce-e1` ran all five tables in 2 min 18 s. W1 measured the idle runs on cyan (Xeon Silver 4310 at 2.1 GHz, Linux) too, with identical results: 0.53 s for the W0 configuration and 29.7 s through 16 recycling lanes, then with four stock arms.

## Tests

`cargo test -p flatshare-sim` checks, grouped by what each part holds to:

- **The KV model:** block identity and namespacing; LRU tail-first freeing, pinning, recomputed copies, CPU write-through and restore; holds, their release under pressure, prefetch, demotion and the idleness ranking.
- **The references:** both clairvoyant tiers and MIN against exhaustive search on small sequences; Mattson stack distances against a direct LRU simulation.
- **The step loop**, against hand-computed schedules: budget and cap, decodes behind prefills, head-of-line blocking, restore loads, stalls and the two-step lag, mixed steps, the serialized host.
- **The replay scheduler:** the order of its pending events and the idle cap's shift; on small synthetic traces, the think-time and idle caps, concurrent subagents and joins, recycling, the TTL hold and the prefetches on a paused session, and the time a realtime tier is short of KV.
- **The policies' inputs:** the history predictor and InferCept's choice; the TraceLab table and its draws; the regime map and its spread over seeds.
- **The GPU profiles** and their tables: the box's gives its measured costs and pool back exactly under every scaling; each peak scales only its own step terms and cheapens the cold prefill; the L4's and the A100's linear layers run through Marlin at the BF16 rate; the A100's three settings; a second model's bytes, FLOPs and pool; the L4's ECC keeps a sixteenth of its memory; the modeled cold prefill keeps the box's relation to the scaled step loop; the quadratic refit, the fitted scales and the session counts; a host's file and its refusals, the box's own host giving every card the box's CPU and measured link and the box its measured costs, and a host's CPU scaling only the API server's time and E0's, its link only the restore's transfer.
- **The forecast:** the box forecasts its own measurements, on its own host too; the A100's probe cases and time-equivalent caps; the scenarios' cells and the tables; the fitted cells, their section and the count closest to the box's running share, the lower on a tie; on a host, the card's cells and rows, which say so, and the report naming the host.
- **The command line**, which refuses a value that describes no run by its option's name, and AIPerf's seed derivation against numpy.
- **On the 16-trace subset** (`tests/subset.rs`): the plan against AIPerf's; every stock arm end to end; v0's run under the idle service; the E1 arms under TraceLab pauses, twice, with identical results; the box placed by `hardware` exactly as `regime` runs it; the forecast's cells placed as `run` places them; `forecast` byte for byte as before it took fitted costs, and with the box's own costs as the card's fitted ones, which give the box's rows; `forecast` on the box's own host, the same but for the introduction, the costs tables' rows on it, E0's restores and the card's scenario cells; `replay-log` on a record of AIPerf's own and on a simulated run written as AIPerf's export, where every request finds the sources the simulation gave it.

`testdata/aiperf-0.13.0-record.jsonl` is one record of W0's AIPerf export, on a trace of the subset. `testdata/forecast-a100-80gb-sub16.md` is `forecast`'s report on the subset, for the A100 80 GB at seed 11 in 120 s replays, as it was before the fitted costs (the test that reads it gives the command that rewrites it). `testdata/box-host.json` is `benchmarks/a100/bench_host.py`'s `host.json` on the box (Phastform, 2026-10-26), whose figures the box's host constants hold. `testdata/weka-x32-sub16` holds 16 traces of [semianalysisai/cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) (Apache-2.0), block-scaled 32× by `flatshare-bench scale-traces`. To regenerate the pinned AIPerf plan, see `scripts/aiperf_reference.py`. `data/tracelab-pauses.tsv` holds the percentiles of the wait between a coding agent's tool calls and their results, per tool, derived by `scripts/tracelab_pauses.py` from the [TraceLab](https://github.com/uw-syfi/TraceLab) public coding-agent trace dataset v0.0.2 (CC BY 4.0); the dataset itself is not in the repository.
