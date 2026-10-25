# Experiments

How Leaseholder's measurements were made. Every experiment runs on one frozen box (the same model, precision, memory split and SLO) and compares against tuned vLLM and against what engines and routers already ship. The numbers are in [BENCHMARKS.md](BENCHMARKS.md), the story in [METHODS.md](METHODS.md), and the terms (arms, campaigns, W0 to W8) in [GLOSSARY.md](GLOSSARY.md).

| # | Experiment | Question | State |
|---|---|---|---|
| E0 | **The tier cost model** | What does resuming an agent turn cost from the GPU cache, from host RAM and recomputed? | Done ([BENCHMARKS.md](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0)); `lh-core` carries the fits |
| E2 | **Voice under agent load** | How close does each configuration an operator can deploy bring the voice tail to its SLO, at what cost to background agents, and what does each mechanism of Leaseholder's arm 9 contribute? | Done on one GPU node; the two-node points never ran, and their code is parked on the `two-node` branch |
| E1 | **Parking** | How much of the gap between LRU plus CPU offload and a clairvoyant reference does each level of lifecycle information close, on resume latency? | Done in the simulator: negative at the median, and for the information a deployment has |
| E3 | **The background tier's cost of make-room** | What does make-room take from the agents, and through which rule: the budget cap's hold on their prefill, or the KV preemption's victims and their restores? | Done on W5's points ([BENCHMARKS.md](BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)), from the engine's own counts; what a victim recomputes when it resumes is not measured |

The original E3, a recompute ledger validated by injected causes with cache contracts as an SLI, was dropped with leases ([ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md)).

## E2: voice under agent load

**A point** is one run of an arm file ([`benchmarks/e2`](../benchmarks/e2/README.md)):
- **Voice.** Two callers (`lh-voice bots`) call the voice agent through LiveKit, 25 turns each, with scripted lines of known end times and a barge-in on a quarter of the turns: 50 turns a point. V2V runs from the end of the caller's speech to the agent's first audio, read from waveforms on one clock, with fixed endpointing.
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
| 4 | Arm 3 tuned | Step cap 512 or 1,024 tokens with a long-prefill threshold of 0, 256 or 512 (five cells); the frozen box is 512/256 |
| 5 | vLLM's own knobs | `--watermark 0.1`; `--max-num-queued-tokens 8192` |
| 6 | Static answers | A background concurrency cap of 2 or 8; a static 2,048-token KV reservation |
| 7 | What engines ship | `--max-num-seqs 12`, with and without #54530 ported to 0.32.0; SGLang 0.5.22's priority preemption, with and without its host tier (HiCache) |
| 8 | What routers ship | llm-d's flow control, emulated in the gateway (ceilings 6/6/6); SMG 1.13.0's priority scheduler, as shipped; abort-and-resubmit of background work before its first byte (preempt-resubmit) |
| 9 | make-room with the lifecycle policy | The change and the lifecycle policy, below, with single-mechanism ablations |

Arm 10 of the plan (MPS with an active-thread split, audio off the GPU) was not run in E2: W1's MPS comparison had already shown that the interference that matters is inside the engine ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)).

**Arm 9** ([`disc-a9-s24.toml`](../benchmarks/e2/arms/disc-a9-s24.toml)):
- **The engine** runs the `make-room` variant ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351)): at most two preemptions a step for a waiting request that lacks KV, the budget cap with a 16-token window, and a 2,048-token reserve. The budget cap took two changes before the repeat rounds, each found from a point's stage waterfall and reproduced on a probe; the earlier versions' points are kept apart and not placed on the frontier.
- **The gateway** runs the lifecycle policy (`policy = "lifecycle"`; E2's arm files spell it `leaseholder`; [ARCHITECTURE.md §6](ARCHITECTURE.md#6-tested-and-not-adopted)): 1,024 tokens of KV headroom per armed call, armed from speech onset; 150 ms of predicted background prefill per critical window; replies capped at 16 tokens when the headroom was not there (degrade); aging after 10 s.
- **The voice worker** posts its lifecycle events.
- **Ablations**, each arm 9 without one part: the throttle, the headroom, degrade, the events, the gateway policy (the engine only), the change (the gateway only), and each engine switch. The headroom is also armed by the request instead of speech onset, and the throttle's budget run at 0 and 500 ms.
- **Beyond the plan:**
  - arm 9 without degrade but with a 48-token window;
  - the engine as the vLLM PRs propose it (no reserve, no gateway policy), with the KV preemption alone, with the budget cap, and with a 48-token window;
  - arm 9 and the frozen box in the standard scenario.

**Factors**, at 16 agent sessions in the standard scenario: LiveKit's preemptive generation off (on by default), and four and six callers.

**Metrics** the report computes, per point:
- **Voice.** V2V p50, p95 and p99, and failed turns, each counted as a sample censored at the callers' 30 s reply timeout. Barge-in to silence.
- **The voice worker's stage times** (endpointing, the reply's first sentence, TTS's first audio, transport and playout) and the realtime first byte, exact from the gateway's request log.
- **Agents.** Background goodput (the prompt and completion tokens of the agent requests completed during the calls, over the calls' duration) and the agents' TTFT p50 and p99.
- **The engine.** Prompt tokens by source (computed, cached, restored from the CPU tier) and preemptions.
- **The gateway.** Its decisions: deferrals by reason, degraded replies and resubmissions. It also counts the headroom's false alarms, and a record keeps the counters that moved: none of E2's records has one, since every utterance of the callers is a turn.
- **The GPU.** Its clocks and temperature.

Audio gaps, abort latency, wasted tokens and cost per token were in the plan and are not computed.

**Volume and builds.**
- **Runs.** W2 measured each baseline once (50 turns). W3–W4 repeated the discriminating scenario in rounds, rotating the order of the engine groups so that no configuration always runs at the same hour. Each headline configuration has five runs (250 turns); each ablation and PR configuration has three (150). Two exceptions: the vLLM PRs' configurations and arm 9's 48-token window ran after the rounds, in a session of their own (10-23 23:16Z – 10-24 01:35Z) with no frozen-box, arm-9 or voice-alone point, and arm 9's ablations always ran in one order. Every run of W2–W4 replays the same workload sample (agent seed 11, caller seed 0), so their ranges measure timing noise only.
- **W5's re-measure.** After W5's fixes to make-room ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351)) and to this harness, two campaigns of five rounds re-measured the configurations the PRs and the README cite. W5's campaign ran voice alone, the frozen box and the PRs' 48-token configuration on the fixed make-room and, as its control, on make-room as v0.3 built it; the PR campaign ran the frozen box, the KV preemption alone and both PRs at 16 and at 48 tokens. Two more campaigns use the same samples: the extension (five rounds) runs preempt-resubmit, the best arm without an engine change, and the PRs' 48-token configuration once more in the discriminating scenario, and the frozen box and the PRs' 48-token configuration in the standard scenario at 16 agent sessions; calls per GPU (three rounds, the first three samples) runs the PRs' 48-token configuration with four and with six callers, beside voice alone with as many, its SLO's reference. Round N of each replays a sample of the agents' traces of its own (agent seed 10 + N, the same in every campaign) and runs its configurations in an order rotated from round to round. Every call opens with an unmeasured warm-up turn, and each point checked the engine's scheduler class and switches before it started. W5's records are kept apart from W2–W4's ([`benchmarks/results/2026-10-24-e2-w5`](../benchmarks/results/2026-10-24-e2-w5)).
- **Builds.** W2's first two campaigns (`w2`, `w2b`) ran the W2 gateway and voice worker; everything else ran the W3 build, with lifecycle events, and later versions of the variant where noted. The vLLM PRs' configurations ran as the make-room variant on vLLM 0.32.0 with the PRs' switch values; ported to 0.32.0, the PRs' diff schedules identically to it ([GLOSSARY.md](GLOSSARY.md#the-change-and-its-parts)).
- **The SLO's reference** is each campaign's own voice alone with as many callers: in W3–W4 the median of five runs, 1,767 ms, so 2,017 ms (one of those runs is from the W2 build; the four on the W3 build give 1,768.5 ms); in W5 the median of its five rounds, 1,784 ms, so 2,034 ms ([SLO.md](SLO.md)).
- **Every turn counts** in W2–W4, the first of each call included (W5's calls open with a warm-up turn the report leaves out). Under load the first turn is slower than the rest, which raises the standard scenario's p95 and leaves the discriminating scenario's where it is; [BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2) gives both with and without it.
- **Goodput windows differ.** Background goodput is counted over the calls, whose duration differs by configuration (about 280 s for arm 9, about 430 s for the frozen box), against agent requests of about 75 s. The report also gives it over the calls' first 240 s, a window of the same length for every point.
- **Load generator health.** The callers record how late they play their lines: at most 116 ms, at six callers, and under 60 ms elsewhere. The report does not use it to reject points.

**Results** are in [BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2), every number with its runs and ranges,
and [METHODS.md](METHODS.md#voice-under-agent-load-e2), the story they tell.

**Other hardware.** E2 runs on the frozen box. [METHODS.md](METHODS.md#what-transfers-to-other-hardware) says what
transfers to another card and what does not, and [HARDWARE.md](HARDWARE.md) how a node of another card is deployed
and measured, the A100 40 GB run included.

## E3: the background tier's cost of make-room

E2 prices make-room by the agents' goodput and first-token latency; E3 says where the cost comes from. It uses no new infrastructure: every W5 point's record keeps the engine's counters over the calls, and the engine's own lines add make-room's counts.
- **The budget cap's hold.** make-room logs its running totals at most every 10 s: the steps in which the budget cap held lower-priority prefill back, the prefill tokens and admissions it held back in them (summed over the steps) and the requests it preempted to make room. Over a point's calls they are the difference of the totals at its two ends, read from the engine's journal (`e3_counters.py` in W5's records; up to 10 s of moves at either end can fall outside). The engine's step count over the calls, from its metrics at their start and end, gives the hold's share of the steps.
- **The victims.** The engine's count of every preemption (`vllm:num_preemptions_total`, which counts a victim when its next output reaches the frontend), make-room's own count, and the CPU tier's loads beyond the requests' first scheduling: its load bytes less its hits at first scheduling, which are the victims' restores, give or take the requests in flight at the window's edges.
- **What it cannot see.** What a victim recomputes when it resumes: vLLM 0.32 counts a request's computed prompt tokens, and its steps' tokens, at its first scheduling only. The steps' tokens less the decoded ones equal the prompt tokens computed at first scheduling at every point, so they are no second instrument.

## E1: parking

Parking KV during a tool call and prefetching it before the return is published ([PRIOR-ART.md](PRIOR-ART.md)), so E1 measures how much of the gap between LRU plus CPU offload and a clairvoyant reference each level of lifecycle information closes (none, session history, an oracle), per regime, with resume latency as the objective. It runs in the simulator first, and the GPU runs only the arms the simulator says matter.

- **Workload.** The block-scaled Claude Code traces, replayed as AIPerf's `agentic_replay` sends them, with recorded think times capped at 120 s, as in W0 and W2, or TraceLab's tool latencies. Session counts are swept to KV pressures of 1–8×, with the CPU tier at 2×, 4× and 8× the GPU pool.
- **Arms.** GPU-only LRU; LRU + CPU tier; a fixed-TTL hold (Continuum); prefetch timed by session history and by an oracle; simplified Tokencake, MORI and InferCept; the clairvoyant reference (Belady's MIN in each tier, replayed on the LRU + CPU timeline: a reference, not a bound).
- **Metrics.** Resume TTFT p50, p90 and p99 by source tier, and stalls; avoidable recompute; PCIe bytes.
- **The gate.** If even the oracle closes under about 10% of resume TTFT over LRU + CPU in the realistic regime, E1 is a regime map with a negative result, and no lease-aware policy runs on the GPU. The gate names no percentile; it was read at p50, and the p90 result is stated beside it. It was first written (2c26e61, 2026-10-22) on the clairvoyant reference's headroom over LRU + CPU, and changed to the oracle's gain in the commit that reported the result (102ca22). Both read the same way: at p50 the clairvoyant reference gains 0.5–2.3% and the oracle −2 to +8%, under the gate; at p90 the clairvoyant reference gains 8–16% (over 10% at 8 and 24 sessions) and the oracle up to 29%. The decision not to deploy a lease-aware policy rests on what a deployment can know: no client sends tool return times, and session history gains nothing.

**Result: the negative branch at p50; at p90 only an oracle gains** ([`crates/lh-sim`](../crates/lh-sim/README.md)). `make reproduce-e1 CORPUS=DIR` regenerates its tables from the 32× corpus in about a minute; `make reproduce-calibration EXPORTS=DIR CORPUS=DIR` recomputes the calibration's from the six measured runs' exports, an asset of the [v0.3 release](https://github.com/StevenWang-CY/leaseholder/releases/tag/v0.3).
- **The model.** `lh-sim` serves requests through a model of vLLM 0.32's step loop, fitted to the E0 cells and to three operating points of the frozen box. It reproduces resume TTFT p50 and p90 within 8% at 8 and 16 sessions. At 24 it is 18–30% fast, because restores there take longer than modeled.
- **The realistic regime** (the traces' think times, the CPU tier at 4×, 20 seeds). The oracle gains under 9% of resume TTFT p50 at every pressure (3% at 8 and 12 sessions, 8% at 16, where the gain of the seeds' mean latencies is 10%), because the median resume is already a GPU hit or a short restore. At p90 it gains 28% at pressures of 1.6–2.4 (about 80–90 ms), more than the clairvoyant reference there, since a prefetch hides a restore inside the pause; 13% on average at the box's 16 sessions (−3% by the mean latencies), with a seed range wider than the effect; and it loses 17% at 24, where its prefetches queue behind demand restores. The history level, which a deployment has without the tools' own return times, gains at most 4% at p90 and does worse than LRU + CPU from 12 sessions on, and TTL holds are neutral. With TraceLab's pauses the oracle gains at most 7%.
- **Modeled on an L4, an A100 and an H100,** at their own scalings, lifecycle information buys at most 14%, 13% and 8% of resume TTFT p90.
- **What follows.** The lease-aware CPU-tier policy built for E1, `LeasePolicy`, was never switched on and was removed after v0.3: the tail gain needs each tool's return time ahead of the return, which no client sends, and session history gains nothing. E1's regime map went to vLLM #57103 and router #428, corrected there on 2026-10-24 after a timing bug in the simulated oracle was fixed: its prefetch fired late for the turns AIPerf's idle cap sends early.

## How runs are made

- **One frozen box** for every experiment ([`deploy/phastform`](../deploy/phastform)). Engine variants are drop-ins ([RUNBOOK.md](../ops/RUNBOOK.md#engine-variants)), and runs are grouped by engine, because a restart costs minutes.
- **Campaigns** list arm files. [`campaign.sh`](../benchmarks/e2/campaign.sh) measures them in order from a workstation and fetches each run directory, and a stopped campaign resumes at its first incomplete point. While a campaign measures, only the box's own units may hold the GPU; a point measured beside another process is measured again ([RUNBOOK.md](../ops/RUNBOOK.md#checks-before-heavy-work-on-a-shared-host)).
- **Records.** Each run directory holds the arm file as run, the engine's `/server_info`, the gateway's request log, AIPerf's exports, the worker's journal, the GPU's samples and the metrics around the calls. A compact record of each point (`lh-bench e2 export`) is committed under [`benchmarks/results`](../benchmarks/results/2026-10-23-e2/README.md), from which `make reproduce-e2` rebuilds the report. The raw run directories are an asset of the v0.3 release.
- **Reporting.** Each configuration is reported per run, with the median and the range of its runs, never as a mean alone.
