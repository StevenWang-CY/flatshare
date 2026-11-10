# Experiments

How Flatshare's measurements were designed, for a reader who wants to audit an experiment or repeat it: each
experiment's question, arms, scenarios, metrics and repeats, and which campaign ran on which code. The numbers are in
[BENCHMARKS.md](BENCHMARKS.md), what they rule in and out in [the report](REPORT.md#6-results), and the terms in
[GLOSSARY.md](GLOSSARY.md). Every experiment runs the frozen box's configuration, on Phastform or on Gray, its second
RTX 5060 Ti node, against tuned vLLM and what engines and routers already ship.

| # | Experiment | Question | Its numbers |
|---|---|---|---|
| E0 | **The tier cost model** | What does resuming an agent turn cost from the GPU cache, from host RAM and recomputed? | [BENCHMARKS.md](BENCHMARKS.md#what-does-resuming-an-agent-turn-cost) |
| E2 | **Voice under agent load** | How close does each configuration an operator can deploy bring the voice tail to its SLO, and at what cost to the agents? | [BENCHMARKS.md](BENCHMARKS.md#how-close-does-each-configuration-bring-the-voice-tail-to-its-slo), and [on an A100](BENCHMARKS.md#the-a100-run-2026-10-26) |
| E3 | **The background tier's cost of make-room** | Through which rule does make-room cost the agents, the budget cap or the KV preemption? | [BENCHMARKS.md](BENCHMARKS.md#what-does-the-change-take-from-the-agents) |
| E1 | **Parking** | How much of the gap to a clairvoyant reference does each level of lifecycle information close? | [BENCHMARKS.md](BENCHMARKS.md#does-knowing-when-an-agent-returns-help) |
| Soak | **The production path for hours** | Does the box, as it serves, keep its SLO, admission and background service over a day? | [BENCHMARKS.md](BENCHMARKS.md#does-the-arrangement-to-deploy-keep-its-slo-over-a-day) |
| Engine points | **A call's reply inside the engine** | What does each change to the engine do to a call's reply beside closed-loop agents? | [BENCHMARKS.md](BENCHMARKS.md#what-do-the-serving-engines-additions-do-to-a-voice-turn) |

## Which campaign measured what, on which code

Each result in [BENCHMARKS.md](BENCHMARKS.md) names the campaign that measured it. The map gives each campaign's span,
rounds, samples of the agents' traces and the release its code belonged to. Each record's `run.json` files name the
exact revision, and [REVISIONS.md](REVISIONS.md) resolves the older names to commits.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-campaigns-narrow.svg">
  <img src="figures/report-campaigns.svg" width="100%" alt="A timeline of 22 October to 2 November 2026, UTC, a bar for each campaign from its first run's start to its last run's end, with what it measured, its rounds and samples, and the code it ran: W2, every arm an operator can deploy, once each, 23 October 00:13 to 08:59, one run a point, one sample, on code before v0.1; W3–W4, the headline configurations, the lifecycle policy and the change, 23 October 10:57 to 24 October 01:35, 5 rounds, one sample, on code before v0.1; W5, the fixed make-room and the configuration to deploy, 24 October 18:56 to 25 October 09:57, 18 rounds, 5 samples, on code after v0.3; FB, the configuration to deploy on the code of 25 October, 25 October 12:47 to 15:22, 5 rounds, 5 samples, on code after v0.3; the A100 run, the main comparison like for like on an A100, 26 October 04:42 to 09:05, 5 rounds and its arms, 5 samples, on v1.0; RS, how often two calls find KV short, 26 October 20:11 to 21:57, 5 rounds, 5 samples, on v1.2; SW, the step cap, the threshold and the window, screened, 27 October 20:56 to 28 October 00:10, one run a setting, one sample, on v1.5; RG, two cards' arrangements, and the step cap confirmed, 28 October 01:41 to 13:19, 5 rounds, 5 samples, on code after v1.6; the soak, the arrangement to deploy, serving for a day, 28 October 15:20 to 29 October 17:24, 2 hours of voice alone, then 24 hours serving, on code after v1.6; OR, the serving engine's additions against v1.14's configuration, 2 November 00:32 to 08:18, 5 rounds, 5 samples, on code after v1.14; OS, the decode hold and speculative decoding on the four additions, 2 November 08:21 to 11:46, 2 rounds, 2 samples, on code after v1.14.">
</picture>

**Eleven campaigns from 23 October to 2 November 2026 measured the results, each on code its records name.** Each bar runs from a campaign's first run to its last, UTC; a round runs each of a campaign's configurations once, on a sample of the agents' traces of its own.

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
| The soak | 28 October 15:20 to 29 October 17:24 | — | 2 hours of voice alone, then 24 hours serving | after v1.6 | [`2026-10-28-soak`](../benchmarks/results/2026-10-28-soak/README.md) |
| OR | 2 November 00:32 to 08:18 | 55 | 5 rounds, 5 samples | after v1.14 | [`2026-11-02-opt-rounds`](../benchmarks/results/2026-11-02-opt-rounds/README.md) |
| OS | 2 November 08:21 to 11:46 | 24 | 2 rounds, 2 samples | after v1.14 | [`2026-11-02-opt-rounds`](../benchmarks/results/2026-11-02-opt-rounds/README.md) |

The map leaves out [the engine points](../benchmarks/results/2026-11-01-engine-points/README.md) of 1 and 2 November,
which tried each addition on v1.14's serving profile, and
[version 1.24's windows and rounds](../benchmarks/results/2026-11-08-envelope/README.md) of 7 and 8 November, on code of
v1.23 and after.

## E2: voice under agent load

**A point** is one run of an arm file ([`benchmarks/e2`](../benchmarks/e2/README.md)):
- **Voice.** Two callers (`flatshare-voice bots`) call the voice agent through LiveKit, 25 turns each, with scripted
  lines of known end times and a barge-in on 9 of the 50 turns. V2V runs from the end of the caller's speech to the
  agent's first audio, read from waveforms on one clock, with fixed endpointing.
- **Agents.** AIPerf replays the SemiAnalysis Claude Code traces closed-loop through the gateway as background requests,
  block-scaled 32× so that every request fits the 32k context, with think times capped at 120 s.
- **Scenarios.** The standard scenario replays the whole corpus at 8, 16 and 24 agent sessions. The discriminating
  scenario replays the 48 traces with the largest contexts, 13.9k tokens at the median request against 7.9k, at 24
  sessions, so that their KV fills the pool.
- **Before each point** the engine's prefix cache is reset, and the harness checks the engine and each flag it reports
  against the arm file.

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
| 8 | What routers ship | llm-d's flow control, emulated in the gateway (ceilings 6/6/6); SMG 1.13.0's priority scheduler, as shipped; preempt-resubmit (note 1) |
| 9 | make-room with the lifecycle policy | The change and the lifecycle policy, below, with single-mechanism ablations |

1. Preempt-resubmit aborts background work before its first byte and resubmits it.

Arm 10, MPS with an active-thread split, did not run, since W1 had shown that the interference that matters is inside
the engine ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)).

**Arm 9** ([`disc-a9-s24.toml`](../benchmarks/e2/arms/disc-a9-s24.toml)) combines three parts:
- **The engine** runs the `make-room` variant ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change)), with at most two
  preemptions a step for a waiting request that lacks KV, the budget cap at a 16-token window and a 2,048-token reserve.
- **The gateway** runs the lifecycle policy ([ARCHITECTURE.md §7](ARCHITECTURE.md#7-designs-measured-and-dropped)). It
  holds 1,024 tokens of KV headroom for each call from speech onset, allows 150 ms of predicted agent prefill per
  critical window, and caps a reply at 16 tokens when the headroom is not there (degrade).
- **The voice worker** posts its lifecycle events.

Each ablation removes one part or engine switch, and two vary the headroom's arming and the throttle's budget. Also run
were the engine as the pull requests propose it, without the reserve or a gateway policy, at each of its steps: the KV
preemption alone, then the budget cap at 16 tokens and at 48. Arm 9 and tuned vLLM also ran in the standard scenario.
The factors, at 16 sessions there, were LiveKit's preemptive generation off and four and six callers.

**Metrics**, per point:
- **voice**: V2V p50, p95 and p99, with failed turns at the callers' 30 s timeout, and barge-in to silence;
- **the voice worker's stages** and the realtime first byte, from the gateway's request log;
- **agents**: their token rate and TTFT p50 and p99 ([SLO.md](SLO.md#indicators) defines both);
- **the engine**: prompt tokens by source and preemptions;
- **the gateway**: its deferrals by reason, degraded replies and resubmissions.

**Rounds and samples**, each campaign's in [the map](#which-campaign-measured-what-on-which-code):
- **W2 to W4** replay one sample of the agents' traces (agent seed 11), so their ranges measure timing noise only. W2
  measured each baseline once, and W3–W4 repeated the discriminating scenario in rounds, rotating the engine groups so
  that no configuration always runs at the same hour.
- **W5's rounds** each replay a sample of their own, agent seed 10 + N in round N of every campaign, which draws where
  each session enters its trace. The order rotates from round to round, and every call opens with an unmeasured warm-up
  turn.
- **From RG on** each round has a new sample, the rounds alternate Phastform and Gray, and each is held to its own voice
  alone with as many callers.
- **Builds.** W2's first two batches ran the W2 gateway and voice worker, everything else the W3 build. The change's
  configurations ran the make-room variant on vLLM 0.32.0 with the pull requests' switch values. The pull requests'
  revisions of 3 and 5 November change how make-room schedules, so every campaign up to 2 November measured it as it
  was before them ([the report's §4](REPORT.md#4-implementation)).

**Measurement rules.** The SLO's reference is each campaign's own voice alone with as many callers, plus 250 ms
([SLO.md](SLO.md#the-slos-reference-campaign-by-campaign)). Every turn counts in W2–W4, the first of each call included,
and from W5 on a warm-up turn opens each call. The callers record how late they play their lines, and no point is
rejected for it ([the replay's fidelity](BENCHMARKS.md#how-closely-does-the-replay-follow-the-traces)).

**Other hardware and two nodes.** E2's core ran like for like on an A100 SXM4 80 GB
([BENCHMARKS.md](BENCHMARKS.md#the-a100-run-2026-10-26)). No point fronts two GPU nodes at once, and the two-node arm
files stay on the `two-node` branch. RG compared arrangements of two cards instead, each card's part a run of its own
([BENCHMARKS.md](BENCHMARKS.md#which-arrangement-of-two-cards-does-the-most-background-work)), and the gateway's
placement of sessions on several nodes ships ([Nodes](../crates/flatshare-gateway/README.md#nodes)).

## E3: the background tier's cost of make-room

E2 prices make-room by the agents' token rate and first-token latency, and E3 says where the cost comes from, from the
counters every W5 point's record keeps.
- **The budget cap's hold.** make-room logs its running totals at most every 10 s: the steps in which the budget cap
  held lower-priority prefill back, the prefill tokens and admissions it held back, and the requests it preempted. Over
  a point's calls they are the difference of the totals at its two ends, so up to 10 s of moves at either end can fall
  outside. The engine's step count over the calls gives the cap's share of the steps.
- **The victims.** The engine counts every preemption when the victim's next output reaches the frontend, and make-room
  counts its own. The CPU tier's loads beyond the requests' first scheduling are the victims' restores, give or take the
  requests in flight at the window's edges.
- **What it cannot see** is what a victim recomputes when it resumes, since vLLM 0.32 counts a request's computed prompt
  tokens at its first scheduling only.

## E1: parking

Prior work parks KV during a tool call and prefetches it before the return ([PRIOR-ART.md](PRIOR-ART.md)). E1 measures
how much of the gap between LRU with CPU offload and a clairvoyant reference each level of lifecycle information closes:
none, session history, an oracle. It runs in the simulator, so that the GPU runs only the arms the simulator says
matter.
- **Workload.** The block-scaled Claude Code traces as AIPerf's `agentic_replay` sends them, with recorded think times
  capped at 120 s, or TraceLab's tool latencies, at KV pressures of 1–8× with the CPU tier at 2×, 4× and 8× the pool.
- **Arms.** GPU-only LRU, LRU with the CPU tier, a fixed-TTL hold (Continuum), prefetch timed by session history and by
  an oracle, and simplified Tokencake, MORI and InferCept. Belady's MIN in each tier, replayed on the LRU timeline, is
  the clairvoyant reference, not a bound.
- **Metrics.** Resume TTFT p50, p90 and p99 by source tier, stalls, avoidable recompute and PCIe bytes.
- **The gate (G-sim).** If even the oracle closes under about 10% of resume TTFT in the realistic regime, the traces'
  think times with the CPU tier at 4×, no lease-aware policy runs on the GPU. The gate names no percentile, and was read
  at p50. It was first written on the clairvoyant reference's headroom, and changed to the oracle's gain in the commit
  that reported the result ([how both forms read](BENCHMARKS.md#does-knowing-when-an-agent-returns-help)).

`make reproduce-e1 CORPUS=DIR` regenerates its tables from the 32× corpus in about two minutes on 16 threads, and `make
reproduce-calibration EXPORTS=DIR CORPUS=DIR` the calibration's from the measured runs' exports, an asset of the [v0.3
release](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3).

## The soak: the production path for hours

E2 measures a fixed point for minutes, on a box in its bench mode. A soak measures the box as it serves, changing
nothing on it, for `--hours`, so that what only time shows is counted: refusals at the call limit, restarts, a slow
leak, a day's tail ([`flatshare-bench soak`](../python/flatshare_bench/README.md#soaks),
[how to run one](../benchmarks/e2/README.md#soaks)).
- **Calls.** Synthetic callers arrive as a Poisson process (`--calls-per-hour`, 20 by default), each for a warm-up turn
  and `--turns` measured turns (10), against the voice worker in serve mode. Such calls last 2.5 to 3.5 minutes, so 20
  an hour offer the worker's two call slots 0.79 to 1.12 Erlang, and by the Erlang B formula one call in seven to one in
  four finds them taken. Every attempt is recorded, with a refusal's reason.
- **Agents.** AIPerf's agentic replay as E2 runs it, through the gateway, at `--agents-sessions` (16) for the whole
  window.
- **The service's own counts**, every minute, as a production scraper reads them, and every start, stop, failure and
  restart of the units. A guard stops the load while another's process holds a GPU the soak loads.

**Its summary**, per hour and overall, gives the calls by outcome, the turns, the replies' percentiles and share over
the SLO, the realtime service level, the agents' tokens a second, and errors, restarts and pauses. The SLO's reference
is a voice-alone soak of the same configuration (`--agents-sessions 0`), since a serving worker adds its own endpointing
to every V2V. The callers are synthetic, the same 40 lines in fixed voices, and the agents replay traces without doing a
task, so a week of real use needs real callers.

## The engine points: a call's reply inside the engine

E2 measures a reply at the caller's ear, and the engine points inside the engine, quickly enough to try one change at a
time ([the record](../benchmarks/results/2026-11-01-engine-points/README.md)).
- **A point** starts one engine from the box's `engine.yaml`, with the serving profile's make-room switches and the
  point's own flags, and runs [`tail.py`](../python/flatshare_vllm/src/flatshare_vllm/make-room/tail.py) for 240 s. Two
  calls of 48-token replies turn every 4 to 8 s beside 12 closed-loop agents whose prompts grow from 6,000 to
  30,000 tokens. With a hint, each call turn is announced 250 ms ahead, as the voice worker does when the caller stops
  speaking.
- **Measures.** The calls' first, 16th and 48th tokens at p50 and p95, and the agents' computed tokens a second from the
  scheduler's trace, which also splits the first token by stage.
- **Repeats and controls.** Each change runs against the point before it, two runs a point on each node, and each
  mechanism has a control, such as a hint sent with nothing held.
- **What it cannot say** is what a change does to V2V, since the calls go straight to the engine. OR's and OS's rounds
  of E2 measure that.

## How runs are made

- **The frozen box's configuration** for every experiment ([`deploy/phastform`](../deploy/phastform)). Engine variants
  are drop-ins ([RUNBOOK.md](../ops/RUNBOOK.md#engine-variants)), and runs are grouped by engine, because a restart
  costs minutes.
- **Campaigns** list arm files. [`campaign.sh`](../benchmarks/e2/campaign.sh) measures them in order from a workstation,
  fetches each run directory, and resumes a stopped campaign at its first incomplete point. While a campaign measures,
  only the box's own units may hold the GPU ([RUNBOOK.md](../ops/RUNBOOK.md#checks-before-heavy-work-on-a-shared-host)).
- **Records.** A compact record of each point (`flatshare-bench e2 export`) is committed under
  [`benchmarks/results`](../benchmarks/results/2026-10-23-e2/README.md). The raw run directories are release assets: the
  calibration exports and W2–W4's in v0.3, W5's and FB's in v1.0, the A100 run's in v1.1 and RS's in v1.3.
- **Reproduce.** `make reproduce-e2` rebuilds the report of W2–W4's 129 points from their compact records,
  `make reproduce-e2-w5` W5's 72 and `make reproduce-e2-a100` the A100 run's, and `make figures` the figures.
- **Reporting.** Each configuration is reported per run, with the median and the range of its runs, never as a mean
  alone.
