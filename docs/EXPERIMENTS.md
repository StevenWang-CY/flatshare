# Experiments

How Flatshare's measurements were designed and run, for a reader who wants to audit an experiment or repeat it: each experiment's question, arms, scenarios, metrics, repeats and builds. Every experiment runs on one frozen box (the same model, precision, memory split and SLO), E2's core on an A100 SXM4 80 GB as well, and compares against tuned vLLM and against what engines and routers already ship. The numbers are in [BENCHMARKS.md](BENCHMARKS.md), what they rule in and out in [the report](REPORT.md#6-results), and the terms (arms, campaigns, W0 to W8) in [GLOSSARY.md](GLOSSARY.md).

| # | Experiment | Question | Ran, and where its numbers are |
|---|---|---|---|
| E0 | **The tier cost model** | What does resuming an agent turn cost from the GPU cache, from host RAM and recomputed? | On the frozen box, 2026-10-22 ([BENCHMARKS.md](BENCHMARKS.md#what-does-resuming-an-agent-turn-cost)); `flatshare-core` carries the fits |
| E2 | **Voice under agent load** | How close does each configuration an operator can deploy bring the voice tail to its SLO, at what cost to background agents, and what does each mechanism of arm 9 (make-room with the lifecycle policy) contribute? | On one GPU node from W2 to RG ([BENCHMARKS.md](BENCHMARKS.md#how-close-does-each-configuration-bring-the-voice-tail-to-its-slo)), its core again on an A100 SXM4 80 GB ([BENCHMARKS.md](BENCHMARKS.md#what-it-measured-2026-10-26)), and on two cards in v1.7's rounds, each card's part of an arrangement a run of its own ([the rounds](../benchmarks/results/2026-10-28-rounds/README.md)); no point fronted two nodes at once ([below](#e2-voice-under-agent-load)) |
| E3 | **The background tier's cost of make-room** | What does make-room take from the agents, and through which rule: the budget cap's hold on their prefill, or the KV preemption's victims and their restores? | On W5's points, from the engine's own counts ([BENCHMARKS.md](BENCHMARKS.md#what-does-the-change-take-from-the-agents)) |
| E1 | **Parking** | How much of the gap between LRU plus CPU offload and a clairvoyant reference does each level of lifecycle information close, on resume latency? | In the simulator ([BENCHMARKS.md](BENCHMARKS.md#does-knowing-when-an-agent-returns-help)) |
| Soak | **The production path for hours** | Does the box, as it serves, keep its voice SLO, its admission and its background service over a day of arriving calls and agents, and what fails, restarts or is refused meanwhile? | Once, for 24 hours of the arrangement to deploy on two GPU nodes, 2026-10-28 to 29 ([BENCHMARKS.md](BENCHMARKS.md#does-the-arrangement-to-deploy-keep-its-slo-over-a-day); [the record](../benchmarks/results/2026-10-28-soak/README.md)) |

The original E3, a recompute ledger validated by injected causes with cache contracts as an SLI, was dropped with leases ([ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md)).

## Which campaign measured what, on which code

Each result in [BENCHMARKS.md](BENCHMARKS.md) names the campaign that measured it. The map gives each campaign its code, what it measured, when it ran, its rounds and its samples of the agents' traces, and the release its code belonged to, from its record's `run.json` files, which name the exact revision ([REVISIONS.md](REVISIONS.md) gives each one's commit in this history).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-campaigns-narrow.svg">
  <img src="figures/report-campaigns.svg" width="100%" alt="A timeline of 22 to 29 October 2026, UTC, a bar for each campaign from its first run's start to its last run's end, with what it measured, its rounds and samples, and the code it ran: W2, every arm an operator can deploy, once each, 23 October 00:13 to 08:59, one run a point, one sample, on code before v0.1; W3–W4, the headline configurations, the lifecycle policy and the change, 23 October 10:57 to 24 October 01:35, 5 rounds, one sample, on code before v0.1; W5, the fixed make-room and the configuration to deploy, 24 October 18:56 to 25 October 09:57, 18 rounds, 5 samples, on code after v0.3; FB, the configuration to deploy on the code of 25 October, 25 October 12:47 to 15:22, 5 rounds, 5 samples, on code after v0.3; the A100 run, the main comparison like for like on an A100, 26 October 04:42 to 09:05, 5 rounds and its arms, 5 samples, on v1.0; RS, how often two calls find KV short, 26 October 20:11 to 21:57, 5 rounds, 5 samples, on v1.2; SW, the step cap, the threshold and the window, screened, 27 October 20:56 to 28 October 00:10, one run a setting, one sample, on v1.5; RG, two cards' arrangements, and the step cap confirmed, 28 October 01:41 to 13:19, 5 rounds, 5 samples, on code after v1.6; the soak, the arrangement to deploy, serving for a day, 28 October 15:20 to 29 October 17:24, 2 hours of voice alone, then 24 hours serving, on code after v1.6.">
</picture>

**Nine campaigns from 23 to 29 October 2026 measured the results, each on code its records name.** Each bar runs from a campaign's first run to its last, UTC; a round runs each of a campaign's configurations once, on a sample of the agents' traces of its own.

| Campaign | Ran, UTC | Points | Rounds and samples | Code | Records |
|---|---|---:|---|---|---|
| W2 | 23 October 00:13 to 08:59 | 45 | one run a point, one sample | before v0.1 | [`2026-10-23-e2`](../benchmarks/results/2026-10-23-e2/README.md) |
| W3–W4 | 23 October 10:57 to 24 October 01:35 | 84 | 5 rounds, one sample | before v0.1 | [`2026-10-23-e2`](../benchmarks/results/2026-10-23-e2/README.md) |
| W5 | 24 October 18:56 to 25 October 09:57 | 72 | 18 rounds, 5 samples | after v0.3 | [`2026-10-24-e2-w5`](../benchmarks/results/2026-10-24-e2-w5/README.md) |
| FB | 25 October 12:47 to 15:22 | 15 | 5 rounds, 5 samples | after v0.3 | [`2026-10-25-e2-fb`](../benchmarks/results/2026-10-25-e2-fb/README.md) |
| The A100 run | 26 October 04:42 to 09:05 | 21 | 5 rounds and its arms, 5 samples | v1.0 | [`2026-10-26-a100`](../benchmarks/results/2026-10-26-a100/README.md) |
| RS | 26 October 20:11 to 21:57 | 10 | 5 rounds, 5 samples | v1.2 | [`2026-10-26-e2-rs`](../benchmarks/results/2026-10-26-e2-rs/README.md) |
| SW | 27 October 20:56 to 28 October 00:10 | 18 | one run a setting, one sample | v1.5 | [`2026-10-27-sweep`](../benchmarks/results/2026-10-27-sweep/README.md) |
| RG | 28 October 01:41 to 13:19 | 80 | 5 rounds, 5 samples | after v1.6 | [`2026-10-28-rounds`](../benchmarks/results/2026-10-28-rounds/README.md) |
| The soak | 28 October 15:20 to 29 October 17:24 | – | 2 hours of voice alone, then 24 hours serving | after v1.6 | [`2026-10-28-soak`](../benchmarks/results/2026-10-28-soak/README.md) |

## E2: voice under agent load

**A point** is one run of an arm file ([`benchmarks/e2`](../benchmarks/e2/README.md)):
- **Voice.** Two callers (`flatshare-voice bots`) call the voice agent through LiveKit, 25 turns each, with scripted lines of known end times and a barge-in on a quarter of the turns: 50 turns a point. V2V runs from the end of the caller's speech to the agent's first audio, read from waveforms on one clock, with fixed endpointing.
- **Agents.** AIPerf replays the SemiAnalysis Claude Code traces closed-loop through the gateway as background requests, block-scaled 32× so that every request fits the 32k context, with their subagents and think times capped at 120 s.
- **Scenarios.**
  - The standard scenario replays the whole corpus at 8, 16 and 24 agent sessions.
  - The discriminating scenario replays the 48 traces with the largest contexts (13.9k tokens at the median request, against 7.9k for the whole corpus) at 24 agent sessions. Their KV fills the pool, the regime where a waiting voice turn cannot take KV from running work.
- **Before each point** the engine's prefix cache is reset (`POST /reset_prefix_cache?reset_external=true`), and the harness checks the engine and each flag it reports against the arm file.

**The arms**, as run:

| # | Arm | Configurations |
|---|---|---|
| 1 | Voice alone | Two callers, no agents: the SLO's reference |
| 2 | Stock defaults | FCFS, 2,048-token chunks |
| 3 | Priority scheduling | Voice at priority 0, agents at 2 |
| 4 | Arm 3 tuned | Step cap 512 or 1,024 tokens with a long-prefill threshold of 0, 256 or 512 (five cells); tuned vLLM is 512/256 |
| 5 | vLLM's own knobs | `--watermark 0.1`; `--max-num-queued-tokens 8192` |
| 6 | Static answers | A background concurrency cap of 2 or 8; a static 2,048-token KV reservation |
| 7 | What engines ship | `--max-num-seqs 12`, with and without #54530 ported to 0.32.0; SGLang 0.5.22's priority preemption, with and without its host tier (HiCache) |
| 8 | What routers ship | llm-d's flow control, emulated in the gateway (ceilings 6/6/6); SMG 1.13.0's priority scheduler, as shipped; preempt-resubmit, which aborts background work before its first byte and resubmits it |
| 9 | make-room with the lifecycle policy | The change and the lifecycle policy, below, with single-mechanism ablations |

Arm 10 of E2's original plan (MPS with an active-thread split, audio off the GPU) was not run: W1's MPS comparison had already shown that the interference that matters is inside the engine ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)).

**Arm 9** ([`disc-a9-s24.toml`](../benchmarks/e2/arms/disc-a9-s24.toml)):
- **The engine** runs the `make-room` variant ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351)): at most two preemptions a step for a waiting request that lacks KV, the budget cap with a 16-token window, and a 2,048-token reserve. The budget cap took two changes before the repeat rounds, each found from a point's stage waterfall and reproduced on a probe; the earlier versions' points are kept apart and not placed on the frontier.
- **The gateway** runs the lifecycle policy (`policy = "lifecycle"`; [ARCHITECTURE.md §7](ARCHITECTURE.md#6-tested-and-not-adopted)): 1,024 tokens of KV headroom per armed call, armed from speech onset; 150 ms of predicted background prefill per critical window; replies capped at 16 tokens when the headroom was not there (degrade); aging after 10 s.
- **The voice worker** posts its lifecycle events.
- **Ablations**, each arm 9 without one part: the throttle, the headroom, degrade, the events, the gateway policy (the engine only), the change (the gateway only), and each engine switch. The headroom is also armed by the request instead of speech onset, and the throttle's budget run at 0 and 500 ms.
- **Beyond the original plan:**
  - arm 9 without degrade but with a 48-token window;
  - the engine as the pull requests propose it (no reserve, no gateway policy), with the KV preemption alone, with the budget cap, and with a 48-token window;
  - arm 9 and tuned vLLM in the standard scenario.

**Factors**, at 16 agent sessions in the standard scenario: LiveKit's preemptive generation off (on by default), and four and six callers.

**Metrics** the report computes, per point:
- **Voice.** V2V p50, p95 and p99, and failed turns, each counted as a sample censored at the callers' 30 s reply timeout. Barge-in to silence.
- **The voice worker's stage times** (endpointing, the reply's first sentence, TTS's first audio, transport and playout) and the realtime first byte, exact from the gateway's request log.
- **Agents.** Their token rate (the prompt and completion tokens of the agent requests completed during the calls, over the calls' duration) and the agents' TTFT p50 and p99.
- **The engine.** Prompt tokens by source (computed, cached, restored from the CPU tier) and preemptions.
- **The gateway.** Its decisions: deferrals by reason, degraded replies and resubmissions. It also counts the headroom's false alarms, and a record keeps the counters that moved: none of E2's records has one, since every utterance of the callers is a turn.
- **The GPU.** Its clocks and temperature.

Abort latency, wasted tokens and cost per token were in the original plan and are not computed. Its audio gaps of 100 ms or more are read, from v1.6 and v1.7, as the pauses inside a reply, the concealed audio and the agent's playout ([SLO.md](SLO.md#indicators)).

**How the campaigns ran.** [The map](#which-campaign-measured-what-on-which-code) gives each campaign's rounds, samples and code; [BENCHMARKS.md](BENCHMARKS.md) what each one measured, and with which configurations.
- **Rounds.** W2 measured each baseline once (50 turns). W3–W4 repeated the discriminating scenario in rounds, rotating the order of the engine groups so that no configuration always runs at the same hour. Each headline configuration has five runs (250 turns); each ablation and PR configuration has three (150). Every run of W2–W4 replays the same workload sample (agent seed 11, caller seed 0), so their ranges measure timing noise only.
- **W5's rounds.** W5 ran three campaigns of five rounds and a fourth, of three, of calls per GPU. Round N of each replays a sample of the agents' traces of its own (agent seed 10 + N, the same in every campaign) and runs its configurations in an order rotated from round to round. Every call opens with an unmeasured warm-up turn, and each point checked the engine's scheduler class and switches before it started. W5's records are kept apart from W2–W4's ([`benchmarks/results/2026-10-24-e2-w5`](../benchmarks/results/2026-10-24-e2-w5)).
- **Builds.** W2's first two campaigns (`w2`, `w2b`) ran the W2 gateway and voice worker; everything else ran the W3 build, with lifecycle events, and later versions of the variant where noted. The change's configurations ran as the make-room variant on vLLM 0.32.0 with the pull requests' switch values; ported to 0.32.0, the pull requests' diff schedules identically to it ([GLOSSARY.md](GLOSSARY.md#the-change-and-its-parts)).

**Measurement rules.**
- **The SLO's reference** is each campaign's own voice alone with as many callers: in W2 its one run, 1,744 ms, so 1,994 ms; in W3–W4 the median of five runs, 1,767 ms, so 2,017 ms (one of those runs is from the W2 build; the four on the W3 build give 1,768.5 ms); in W5 the median of its five rounds, 1,784 ms, so 2,034 ms; in RG each round's own voice alone, with two callers and with one ([SLO.md](SLO.md)).
- **Every turn counts** in W2–W4, the first of each call included (W5's calls open with a warm-up turn the report leaves out). Under load the first turn is slower than the rest, which raises the standard scenario's p95 and leaves the discriminating scenario's where it is; [BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2) gives both with and without it.
- **The token rate's windows differ.** The agents' token rate is counted over the calls, whose duration differs by configuration (about 280 s for arm 9, about 430 s for tuned vLLM), against agent requests of about 75 s. The report also gives it over the calls' first 240 s, a window of the same length for every point.
- **Load generator health.** The callers record how late they play their lines ([BENCHMARKS.md](BENCHMARKS.md#is-the-replayed-load-faithful-to-the-traces)), and the report does not use it to reject points.

**Results** are in [BENCHMARKS.md](BENCHMARKS.md#how-close-does-each-configuration-bring-the-voice-tail-to-its-slo), every number with its runs and ranges, and [the report's §6](REPORT.md#6-results), what they rule in and out.

**Other hardware.** E2 runs on the frozen box, and its core ran like for like on an A100 SXM4 80 GB on 2026-10-26. [The report's §6.7](REPORT.md#67-the-rules-carry-to-an-a100) says what transfers to another card and what does not, [HARDWARE.md](HARDWARE.md) how a node of another card is deployed, and [BENCHMARKS.md](BENCHMARKS.md#the-a100-run-2026-10-26) how the A100 run was laid out and what it measured.

**Two GPU nodes.** Three configurations of two GPU nodes were planned: a voice node and an agent node (dedicate); two mixed nodes with least-loaded placement; and two mixed nodes with Flatshare's placement, which sends a new call to the node with the fewest calls and a new agent session to the node holding the least background KV. None ran in E2, while another user's work held the second card from 2026-10-23; their arm files and the harness's points that front two nodes at once stay on the `two-node` branch ([`benchmarks/e2`](../benchmarks/e2/README.md)).

v1.7's rounds compared dedicated roles, pooling and consolidation on two cards instead, each card's part of an arrangement a run of its own ([RESOURCE-GATE.md](../benchmarks/quality/RESOURCE-GATE.md), [BENCHMARKS.md](BENCHMARKS.md#two-cards-rg)). The gateway's placement of sessions on several nodes, which keeps each on its node, ships ([Nodes](../crates/flatshare-gateway/README.md#nodes)).

## E3: the background tier's cost of make-room

E2 prices make-room by the agents' token rate and first-token latency; E3 says where the cost comes from. It uses no new infrastructure: every W5 point's record keeps the engine's counters over the calls, and the engine's own lines add make-room's counts.
- **The budget cap's hold.** make-room logs its running totals at most every 10 s: the steps in which the budget cap held lower-priority prefill back, the prefill tokens and admissions it held back in them (summed over the steps) and the requests it preempted to make room. Over a point's calls they are the difference of the totals at its two ends, read from the engine's journal (`e3_counters.py` in W5's records; up to 10 s of moves at either end can fall outside). The engine's step count over the calls, from its metrics at their start and end, gives the budget cap's share of the steps.
- **The victims.** The engine's count of every preemption (`vllm:num_preemptions_total`, which counts a victim when its next output reaches the frontend), make-room's own count, and the CPU tier's loads beyond the requests' first scheduling: its load bytes less its hits at first scheduling, which are the victims' restores, give or take the requests in flight at the window's edges.
- **What it cannot see.** What a victim recomputes when it resumes: vLLM 0.32 counts a request's computed prompt tokens, and its steps' tokens, at its first scheduling only. The steps' tokens less the decoded ones equal the prompt tokens computed at first scheduling at every point, so they are no second instrument.

## E1: parking

Parking KV during a tool call and prefetching it before the return is published ([PRIOR-ART.md](PRIOR-ART.md)), so E1 measures how much of the gap between LRU plus CPU offload and a clairvoyant reference each level of lifecycle information closes (none, session history, an oracle), per regime, with resume latency as the objective. It runs in the simulator first, and the GPU runs only the arms the simulator says matter.

- **Workload.** The block-scaled Claude Code traces, replayed as AIPerf's `agentic_replay` sends them, with recorded think times capped at 120 s, as in W0 and W2, or TraceLab's tool latencies. Session counts are swept to KV pressures of 1–8×, with the CPU tier at 2×, 4× and 8× the GPU pool.
- **Arms.** GPU-only LRU; LRU + CPU tier; a fixed-TTL hold (Continuum); prefetch timed by session history and by an oracle; simplified Tokencake, MORI and InferCept; the clairvoyant reference (Belady's MIN in each tier, replayed on the LRU + CPU timeline: a reference, not a bound).
- **Metrics.** Resume TTFT p50, p90 and p99 by source tier, and stalls; avoidable recompute; PCIe bytes.
- **The gate (G-sim).** If even the oracle closes under about 10% of resume TTFT over LRU + CPU in the realistic regime, E1 is a regime map with a negative result, and no lease-aware policy runs on the GPU. The gate names no percentile; it was read at p50, and the p90 result is stated beside it. It was first written (affb8f0, 2026-10-22) on the clairvoyant reference's headroom over LRU + CPU, and changed to the oracle's gain in the commit that reported the result (ceb0d67). [BENCHMARKS.md](BENCHMARKS.md#does-knowing-when-an-agent-returns-help) gives how both forms of it read.

**Reproduce.** `make reproduce-e1 CORPUS=DIR` regenerates its tables from the 32× corpus in about a minute; `make reproduce-calibration EXPORTS=DIR CORPUS=DIR` recomputes the calibration's from the six measured runs' exports, an asset of the [v0.3 release](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3).

## The soak: the production path for hours

E2 measures a fixed point for a few minutes on a box switched into its bench mode: the engine's development routes on, its prefix cache emptied before each point, the gateway's canary off, and a voice worker that takes every call. A soak measures the box as it serves and changes nothing on it, for `--hours` (24 to 72 for the audit's soak), so that what only time shows is counted: refusals at the call limit, restarts, a slow leak, an engine that degrades, the tail of a day rather than of 50 turns ([`flatshare-bench soak`](../python/flatshare_bench/README.md#soaks), [E2's README](../benchmarks/e2/README.md#soaks)).

- **Calls.** Synthetic callers (`flatshare-voice arrivals`, the measurement callers' turns unchanged) arrive as a Poisson process (`--calls-per-hour`, 20 by default), each for a warm-up turn and `--turns` measured turns (10) with the callers' think times, against the voice worker in serve mode, whose admission takes at most its serving profile's `max_calls`. Calls of 11 turns last about two and a half to three and a half minutes at E2's pace (W5's two-caller calls of 26 turns lasted 336 to 476 s), so 20 an hour offer the worker's two call slots 0.79 to 1.12 Erlang, and by the Erlang B formula about one call in seven to one in four finds them taken (`python/flatshare_bench/tests/test_soak.py`). Every attempt is recorded: accepted, refused (and why: the worker full, no GPU node to place it on, the engine not serving, a drain, a dispatch LiveKit lost) or failed to connect.
- **Agents.** AIPerf's agentic replay as E2 runs it (the same corpus, seed and think-time cap), through the gateway as the agents tenant, at `--agents-sessions` (16), for the whole window; a run that ends early is recorded and started again.
- **The service's own counts**, every minute, as a production scraper reads them: the gateway's offered, admitted and ended requests by outcome, its failure-inclusive service level by class, its tokens, in-flight requests and engine states (per node with `[[nodes]]`, with each node's health and placements), each engine's preemptions, KV use and tokens through the gateway's relay, the voice worker's status, and every start, stop, failure and restart of the flatshare units, cyan's and, through the guard, the GPU nodes'. No development route and no `/server_info`.
- **A guard.** A node's GPU is read every minute, with its units' journal lines since the last check (`node-report`, over SSH with a key that may run nothing else); while another's process holds a GPU the soak loads, its load stops, and that is an event of the record.

**Its summary**, per hour and overall: calls attempted, accepted, refused (by reason) and failed; measured turns and failed ones; the replies' pooled p50, p95 and p99, and the share over the SLO, against a voice-alone reference given on the command line (its p95 plus 250 ms, [SLO.md](SLO.md)); replies with a pause over 500 ms; the gateway's realtime service level from its counters; background requests offered, admitted, completed and failed, and their tokens a second; each engine's preemptions and KV use; errors, restarts and the guard's pauses. An attempt belongs to the hour it was due in, a turn to the hour it ended in, a counter's increase to the hour of its later scrape.

**What it cannot say.** The callers are synthetic: the same 40 lines, in fixed voices, with no echo and no room noise; a serving worker's microphone configuration (the end-of-turn model, LiveKit's AEC warm-up) adds its own endpointing to every V2V, so the SLO's reference is a voice-alone soak of the same configuration (`--agents-sessions 0`), not E2's. The agents replay traces; they do no task. A week of real use, the audit's other step, needs real callers.

## How runs are made

- **One frozen box** for every experiment ([`deploy/phastform`](../deploy/phastform)). Engine variants are drop-ins ([RUNBOOK.md](../ops/RUNBOOK.md#engine-variants)), and runs are grouped by engine, because a restart costs minutes.
- **Campaigns** list arm files. [`campaign.sh`](../benchmarks/e2/campaign.sh) measures them in order from a workstation and fetches each run directory, and a stopped campaign resumes at its first incomplete point. While a campaign measures, only the box's own units may hold the GPU; a point measured beside another process is measured again ([RUNBOOK.md](../ops/RUNBOOK.md#checks-before-heavy-work-on-a-shared-host)).
- **Records.** Each run directory holds the arm file as run, the engine's `/server_info`, the gateway's request log, AIPerf's exports, the worker's journal, the GPU's samples and the metrics around the calls. A compact record of each point (`flatshare-bench e2 export`) is committed under [`benchmarks/results`](../benchmarks/results/2026-10-23-e2/README.md), from which `make reproduce-e2` rebuilds the report (`make reproduce-e2-w5` W5's). W2–W4's raw run directories are an asset of the v0.3 release.
- **Reproduce.** `make reproduce-e2` rebuilds the report, its tables, frontiers and waterfall from the compact records of W2–W4's 129 points, `make reproduce-e2-w5` W5's from its 72 and `make reproduce-e2-a100` the A100 run's, and `make figures` the figures; `make reproduce-e1` rebuilds E1's tables from the 32× corpus, and `make reproduce-calibration` the simulator's calibration from its runs' exports. The raw records are release assets: the calibration exports and W2–W4's run directories in v0.3, W5's, FB's and the demo's in v1.0, and the A100 run's in v1.1.
- **The demo video**, in v1.2, plays two calls side by side, on tuned vLLM and on the configuration to deploy, measured again for it with the calls' audio saved: a V2V p95 of 11,232 and 2,213 ms over both calls, not in the tables ([the demo video](../benchmarks/e2/README.md#the-demo-video)).
- **Reporting.** Each configuration is reported per run, with the median and the range of its runs, never as a mean alone.
