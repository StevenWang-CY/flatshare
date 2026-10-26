# One GPU, two clocks: a voice agent beside coding agents

A voice agent and a fleet of coding agents can share one GPU only if the GPU answers a voice turn on speech time, about a second after the caller stops talking, while the agents run on tool time, with seconds to minutes between their turns. This is how Flatshare measured what sharing costs on one RTX 5060 Ti 16 GB, what stock vLLM and the alternatives an operator can deploy today do about it, where they stop, the change that closes most of the gap, and which of Flatshare's own mechanisms paid for themselves. Every number comes from a scripted run in this repository; [the last section](#how-the-runs-were-made) says how to rerun them, and [BENCHMARKS.md](BENCHMARKS.md) has every run.

## The box

One RTX 5060 Ti 16 GB (sm_120) on PCIe Gen3 x8, in a host with two Xeon E5-2609 v4 at 1.7 GHz and 188 GiB of RAM. On it, the frozen box: one pinned configuration every experiment runs on.
- **LLM:** vLLM 0.32.0 serving Qwen3-4B-Instruct-2507-FP8 with a 32k context. The KV pool is fixed at 5 GiB, 36,400 tokens: two or three coding-agent contexts. Prefix caching, priority scheduling, and a step cap of 512 tokens with a 256-token long-prefill threshold.
- **CPU tier:** vLLM's native `OffloadingConnector`, 20 GiB in `/dev/shm`, written through (generated tokens too).
- **Audio on the same card:** Whisper large-v3-turbo (STT) on vLLM and Kokoro-82M (TTS), time-sliced with the LLM. This is the edge pattern; NVIDIA's single-GPU voice-agent profile co-locates the same three.

A second, identical card (Gray) is deployed the same way, for the probes and the two-node points, which never ran ([below](#two-gpus-dedicate-or-pool)). Other hardware is placed on the results by a model, labeled as such, except an A100 SXM4 80 GB, which ran E2's core like for like on 2026-10-26 ([below](#what-transfers-to-other-hardware)).

## What an agent turn costs to resume (E0)

`flatshare-bench tiers` sends a unique prompt three ways and reads vLLM's prefix-cache counters around each request, to verify where its tokens came from:

| Prompt tokens | Recomputed | GPU prefix-cache hit | Restored from host RAM |
|---|---:|---:|---:|
| 4,096 | 437 ms | 48 ms | 157 ms |
| 16,384 | 3,264 ms | 97 ms | 509 ms |
| 31,744 | 10,239 ms | 156 ms | 903 ms |

- A restore adds about 10 ms plus 23.6 µs a token over a GPU hit: 6.2 GB/s of KV, 79% of what PCIe Gen3 x8 carries. It is bound by the link.
- Recompute is 48 ms + 63 µs a token + 8.1 ns a token squared; the quadratic term is attention.
- Restoring beats recomputing at every length measured, so for agents the CPU tier turns an eviction into a restore rather than a recompute.

The same costs say why voice suffers beside agents: a 512-token prefill chunk after a 30k-token context takes about 270 ms of GPU time, seven times one after an empty context. A voice turn waiting behind agent chunks waits behind time, not tokens.

## Voice under agent load (E2)

### How a point is measured

- **Voice.** Two synthesized callers (`flatshare-voice bots`) call a stock LiveKit Agents 1.8.8 voice agent through a LiveKit server on the control host, 25 turns each, barging in on a quarter of the turns. Voice-to-voice latency (V2V) runs from the end of the caller's speech to the agent's first audio, read from waveforms at the caller on one clock. A turn the agent does not answer within 30 s counts at 30 s. Turns end a fixed delay after speech, the same for every arm: LiveKit's end-of-turn model judged about one synthesized line in six unfinished and waited its 3 s maximum.
- **Agents.** AIPerf replays the SemiAnalysis Claude Code traces closed-loop as background requests, block-scaled 32× so that every request fits the 32k context, with their subagents and think times capped at 120 s.
- **Records.** Per point: the callers' turns, AIPerf's records, the gateway's request log (realtime first bytes exact to the millisecond), the voice worker's per-turn stage times, the GPU's clocks and temperature once a second, and the engine's and gateway's metrics before and after.
- **Goodput.** Background goodput counts the prompt and completion tokens of the agent requests that completed during the calls. Shed or failed work counts as lost.
- **The SLO.** V2V p95 at most 250 ms above voice alone with as many callers, which each campaign measures itself: once in W2 (1,744 ms), so W2's SLO is 1,994 ms; five times in W4 (1,744–1,805 ms, median 1,767, W2's run among them), so W4's is 2,017 ms; and once a round in W5 (1,764–1,819 ms, median 1,784), so W5's is 2,034 ms.

### The arms

Each baseline is what an operator could deploy today, configured as its authors recommend, on the same engine and model ([EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load) has the table): voice alone, the SLO's reference (arm 1); stock vLLM and its own knobs (arms 2 to 5: FCFS, priority scheduling, priority with the step cap swept, whose best cell is the frozen box, and the headroom flags); a static background cap or KV reservation in the gateway (6); other engines' preemption (7a, vLLM #54530 ported to 0.32.0; 7b, SGLang 0.5.22 with and without its host tier); routers (8a, llm-d's flow control, emulated; 8b, SMG 1.13.0; 8c, preempt-resubmit, which aborts background work before its first byte and resubmits it); and arm 9, make-room with the lifecycle policy ([below](#arm-9-make-room-with-the-lifecycle-policy)).

### Where the pool has room: the standard scenario

The whole corpus at 8, 16 and 24 agent sessions. At 8, W2's vLLM arms meet W2's SLO, and arm 9, run there once in W4, lands 43 ms above W4's. At 16, the realistic operating point, none clearly does: the best land 110–221 ms above W2's SLO and within noise of each other, and no gateway policy or engine knob separates itself from tuned vLLM. At 24 none meets it. Repeated three times at 16 on one sample (W4), arm 9 (2,196 ms) and the frozen box (2,323 ms) were within noise of each other: where the pool has room the KV preemption has little to do, a full reply outruns arm 9's 16-token window, and what remains is agent prefill sharing the steps of the reply. The PRs' 48-token window covers the reply's first sentence: over W5's five rounds at 16, each on its own sample, it had the lower p95 in every round, by 0.1 to 1.4 s (2.02–2.42 s against the frozen box's 2.15–3.83 s), and met the SLO in one, with 0–7 preemptions a run and the cap holding agent prefill back in 3–13% of the steps. Five rounds resolve the direction; the size follows the sample.

### Where agent KV fills the pool: the discriminating scenario

24 agent sessions of the 48 traces with the largest contexts (a median of 13.9k tokens a request). The pool holds two or three of these contexts, and under the frozen box about 95% of the agents' prompt tokens are computed rather than read from a cache. Here every stock, engine and router baseline but one collapses, with a V2V p95 of 5.7–12.1 s:
- **The loss is the wait for the first sentence.** Endpointing and TTS barely move; on the frozen box the reply's first sentence takes 3.8 s of a 5.2 s mean turn (W4; 3.9 of 5.3 s in W5), against 0.3 s alone, most of it before the LLM's first token.
- **Engine priority does not reach it.** vLLM's priority scheduling cannot let a waiting request preempt running ones for KV ([#40004](https://github.com/vllm-project/vllm/issues/40004)); #54530 preempts at the request-count cap, which does not bind here; SGLang's priority preemption does not fire on KV either, with or without its host tier.
- **Static answers help or cheat.** A 2,048-token KV reservation halves the tail (5.7 s); concurrency caps do not help, since two contexts fill the pool; vLLM's queue limit refuses the voice turns with the rest (48 of 50 failed).
- **Preempt-resubmit comes closest** at 2.6 s (W4; 2.52 s in W5): aborting the background requests that have no first byte yet frees their KV at once, and their resubmissions restore the prefixes already written through to the CPU tier instead of recomputing them. The agents pay in their tail (TTFT p99 160 s against 99 in W4, 193 s against 106 in W5), and it cannot reach a request that is already decoding.

### Arm 9: make-room with the lifecycle policy

Arm 9 has two halves and a signal between them:
- **The engine** runs `make-room`, the change's variant on vLLM 0.32: a waiting request that lacks KV preempts running work of a strictly lower priority (as few as it needs and at most two a step, the lowest priority and the most KV first, their KV kept in the CPU tier); lower-priority prefill sits out the steps of a higher-priority request's first 16 tokens, from the step it fits, through its KV load; and lower classes leave a 2,048-token reserve free. The first two are proposed upstream as vLLM [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351), without the reserve. That is the change as W4's rounds ran it, in v0.3's victim order; the variant and the PRs now preempt in vLLM's own order, the lowest priority and then the latest arrival, and [ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351) states the rules as they are.
- **The gateway** runs the lifecycle policy (`policy = "lifecycle"`). From a caller's speech onset it holds back background requests that would leave less than one voice turn of KV (1,024 tokens) free; in the critical window, from the end of the caller's turn to the reply's first audio, it admits background prefill only up to 150 ms of predicted GPU time, priced by a step model fitted on the box; when the headroom was not there it caps the reply at 16 tokens rather than refusing it (degrade); a background request waits at most 10 s.
- **The voice worker** tells the gateway where each call is: speech onset, the end of the turn, the reply's first audio and its end, a barge-in, the call's end.

The engine half took three rounds, each found in a point's stage breakdown and isolated by a probe before it changed:
1. KV preemption alone passed its probe (a voice turn behind two 18k-token requests in a full pool: 4.9 s to the first token on stock vLLM, 0.25–0.32 s with it) and lost the first E2 point to preempt-resubmit (V2V p95 4.3 s against 2.6 s). The stages put the loss between the reply's first token and its first sentence: the agents' 512-token prefill chunks shared every step of the reply.
2. Holding lower-priority prefill for a turn's first tokens fixed the decode (232 ms from first token to first sentence at the median, against 300 ms for preempt-resubmit) and left a first-token tail (p95 2.1 s).
3. The tail was a turn whose prefix the agents had evicted from the GPU between turns, loading back from the CPU tier: once its KV had arrived, it waited for the scheduler's waiting loop, which runs only on budget the running prefills leave. Holding prefill through the load took a synthetic load's call TTFT p95 from 2.8 s to 0.3 s.

The points measured on the first two versions are kept and reported, not placed on the frontier.

### The results

W4's runs (2026-10-23/24, one sample of the agents' traces): five of each headline configuration and three of each ablation and PR configuration, in rounds that rotate the order of the engine groups, so that no configuration always runs at the same hour; the PRs' configurations (the make-room variant with the PRs' switch values, [GLOSSARY.md](GLOSSARY.md#the-change-and-its-parts)) ran later, in a session of their own with no control (10-23 23:16Z – 10-24 01:35Z). V2V p95, the median of the runs and their range:

| Configuration | V2V p95 | Background tokens/s | Agent TTFT p99 |
|---|---:|---:|---:|
| The frozen box (tuned vLLM) | 10,138 ms (9,257–10,811) | 3,710 | 99 s |
| Static reservation, 2,048 tokens | 5,671 ms (4,906–5,798) | 3,564 | 102 s |
| Preempt-resubmit | 2,591 ms (2,110–2,663) | 3,630 | 160 s |
| Arm 9, every reply capped at 16 tokens | 2,057 ms (1,980–2,134) | 3,411 | 103 s |
| #63350 alone | 5,773 ms (5,612–6,063) | 3,633 | 97 s |
| #63350 + #63351, 16-token window | 3,229 ms (3,220–4,772) | 3,568 | 100 s |
| #63350 + #63351, 48-token window | 2,193 ms (2,168–2,286) | 3,493 | 97 s |

![E2's frontier: voice-to-voice p95 against the agents' goodput for every configuration an operator can deploy](figures/e2-frontier.svg)

Each configuration at the median of its runs, whiskers over their range. The open diamond is the frozen box; 4 is its best step cap, 6 the static reservation, 7b SGLang with HiCache and 8c preempt-resubmit. Arm 9 (left) and the PRs' configuration with a 48-token window (9-pr-ab48, right) are separate configurations, each measured at one load, so no line joins them.

- **The change takes the voice tail to the SLO's edge.** With full replies and a 48-token window, the PRs' configuration is 176 ms over the SLO (2,193 ms). Arm 9, whose gateway capped every reply at 16 tokens, is 40 ms over it at the median, and two of its five runs are under it. No configuration without the change has a run under the SLO.
- **The budget cap is the necessary half.** Without it the p95 stays at 5.3 s even with the KV preemption and the reserve. Either the KV preemption or the reserve makes the room. With full replies they gave 2,193 ms (the KV preemption, as the PRs) and 2,205 ms (the reserve, with arm 9's policy and a 48-token window). With capped replies the reserve alone gave 2,037 ms and the KV preemption alone 2,116 ms (2,115–2,155, every run above arm 9's median); with both, the reserve kept the pool from filling and the KV preemption never fired (0 preemptions in five runs). #63350 alone gets the turn into the batch, and its reply then decodes beside agent prefill (5.8 s).
- **The window has to cover the reply's first sentence**, which the voice agent hands TTS whole. This voice agent's replies run 23 tokens at the median and 52 at most (W5's 5,571 replies). With a 16-token window, a full reply outruns it and decodes beside agent prefill again: 3.2–4.0 s, with the gateway or without. Degrade capped every reply here, which is what fits arm 9's replies into the window; a 48-token window holds all but one full replies instead, about 140 ms above the capped ones (526 against 429 ms to the first sentence).
- **The agents keep most of their throughput.** Arm 9 keeps 92% of the frozen box's background goodput and the PRs' configuration 94%, with the agents' TTFT p99 unchanged. Preempt-resubmit keeps 98% at a 160 s tail, so neither it nor arm 9 dominates the other on the frontier.

**W5's re-measure** (2026-10-24/25, [BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)). After W5's fixes, three campaigns of five rounds, each round on its own sample of the agents' traces, re-measured the configurations the PRs and the README cite against W5's own SLO, 2,034 ms: the frozen box 10,477 ms over ten runs, preempt-resubmit 2,522 ms with its agents' TTFT p99 at 193 s, #63350 alone 5,085 ms, both PRs at 16 tokens 3,094 ms, and at 48 tokens 2,236 ms over fifteen runs, 202 ms over the SLO, with the agents keeping 97% of their goodput over the calls' first 240 s. In every round of the PR campaign the configurations rank the same (the frozen box, #63350 alone, both PRs at 16 tokens, both at 48), and the 48-token window is lower than preempt-resubmit in every round of the extension, by 223 to 723 ms. W5 did not re-measure the ablations, the static reservation or arm 9.

![The voice turn by stage](figures/e2-waterfall.svg)

The stages, as a mean over each configuration's replies, show where each fix acts: endpointing and TTS are the same in every row, and the bar that moves is the reply's first sentence.

## What lifecycle timing could and could not do

Flatshare's premise was that a realtime tier needs to know what is coming: when a caller starts to speak, when an agent's tool returns. Neither experiment found a use for the knowledge a deployment actually has, for different reasons.
- **Agents (E1, in simulation).** Knowing exactly when an agent returns from its tool buys under 9% of resume TTFT at p50 over LRU with the CPU tier: the median resume is already a GPU hit or a short restore, and restores are bound by the link at 6.2 GB/s. At p90 that oracle buys 28% at low pressure, about 80–90 ms, by hiding a restore inside the pause; session history, which is what a deployment knows without the tools' own return times, buys nothing.
- **Voice (E2).** The gateway's lifecycle-timed mechanisms could not act in the regime where voice suffers. Running agent requests held the pool, and the headroom protects KV only by deferring new requests, so the headroom armed at speech onset was never there: degrade capped 60 of 60 realtime requests in every run of arm 9. The critical-window throttle, checked after the headroom, deferred nothing while the headroom ran, even at a 0 ms budget. Leaving out or changing any of these mechanisms therefore landed within the spread of arm 9's runs (medians of 2,033–2,124 ms against 2,057, three runs each), and the same policy in front of the stock engine left voice where stock vLLM does (10,993 ms; W4, which W5 did not re-measure).

So a control plane in front can make room for a voice turn only by aborting background work it has admitted: E2's preempt-resubmit arm aborts the agent requests that have no first byte yet and resubmits them, which reaches 2.52 s at the agents' cost (their TTFT p99 193 s against 106 s for the frozen box over W5's rounds; 2.6 s, and 160 s against 99 s, in W4; [BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)). Only the engine can take KV from a request already decoding, and the change does, by preempting it. This does not show that lifecycle timing never helps: where its mechanisms engage, as in the standard scenario at 24 agent sessions, where the throttle deferred 27 requests, they were not isolated from the change.

What did pay is scheduling: the engine running a realtime request's admission, KV load and first tokens without lower-priority prefill beside them, and a reply that fits that window. [PRODUCTION.md](PRODUCTION.md) turns that into a configuration to deploy.

## What transfers to other hardware

The regime is stated in hardware-free coordinates: KV pressure (the live working set over the pool), π (the median pause over the restore time) and ρ (restore over recompute at the median prompt). They locate a card and a workload on the map. They do not carry an arm's gain to another card, which also depends on how much of a resumed turn is the restore: at the same π (21) and pressure 2.1–2.6, the oracle gains 32% at p90 on the box and 4% on the modeled H100 at 4×. `flatshare-sim hardware` places an L4, an A100 40 GB and an H100 SXM on the same map, with the box's measured costs scaled by the ratios of their published peaks (bandwidth for the weight and KV reads, the rate of the kernel path vLLM 0.32.0 takes for the linear layers, FP8 on the H100 and BF16 through Marlin on the L4 and the A100, BF16 for attention, the link for restores). The model's restores on PCIe Gen4 match vLLM's published L20 measurement within 10%.

The regime where a waiting voice turn is blocked by KV (#40004) follows how long the agents' contexts are against the pool, which the card's memory sets, and how much of the pool the realtime tier needs. On the 32× replay, running agents leave two calls short of KV 12% of the time at 24 sessions on the box, at most 1% of the time up to 24 sessions on the L4 and at any count on the A100, and never on the H100. Replayed at each GPU's own scaling (the L4 at 16×, the A100 at 8×, the H100 at 4×), the L4 is short 9–22% of the time at 16–32 sessions, the A100 2–10% at 16–40 and the H100 0–7%, or 25–65% for a realtime tier sized to the box's share of the pool. Everything off the box but the A100 run below is modeled.

**Measured on one card: an A100 SXM4 80 GB** ([the A100 run](HARDWARE.md#what-it-measured-2026-10-26)). At the box's pool and 24 sessions, the card ran E2's core like for like on 2026-10-26. The scaling by peaks placed each mechanism and missed the KV case's waits by about 30%: the step model fitted on the card's own calibration point put decode attention at 1.82 times the modeled cost (KV read at 42% of the card's bandwidth, where the box's reaches 77%), the fixed step cost at 1.16 times and the per-token cost at 1.00 (Marlin's W8A16 GEMMs as modeled). The host, which the peaks leave out, set E0's GPU hit and restore: placed on the measured host (`bench_host.py`), the forecast's E0 GPU hits and CPU restores came within 10% at the mean, from 69% and 22% off; recompute, which the host does not set, stayed 17% off. What carried is the mechanism and the regime's sign: the change took the voice p95 from 4.99 s to 1.63 s there, where running agents left two calls short of KV 6% of the time, measured, and the simulator puts the box at 17%. What did not is a wait's length, which a card's own calibration point gives.

## Two GPUs: dedicate or pool

Three configurations of two GPU nodes were planned: a voice node and an agent node (dedicate); two mixed nodes with least-loaded placement; two mixed nodes with Flatshare's placement, which sends a new call to the node with the fewest calls and a new agent session to the node holding the least background KV. They never ran: another user's work has held the second card since 2026-10-23, and none was free before v1.0. The gateway's placement of sessions on several nodes, which keeps each on its node, and the harness's points on two nodes were built for them, and are parked on the `two-node` branch ([EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load)).

## How the runs were made

- **One frozen box, campaigns of arm files.** The box is pinned by lock files and model revisions, and [`benchmarks/e2/campaign.sh`](../benchmarks/e2/campaign.sh) measures a campaign's arm files in order from a workstation, grouped by engine because a restart costs minutes; [EXPERIMENTS.md](EXPERIMENTS.md#how-runs-are-made) has its rules, a point measured beside another process among them.
- **Repeats are rounds**: each round runs every headline configuration once, from a campaign file of its own, with the engine groups' order rotated. Results are reported per run with their spread, never as a mean alone.
- **Control-plane overhead.** The gateway's own time is a metric on every request (`flatshare_gateway_overhead_seconds`: a p99 of 0.44–0.58 ms in place on cyan). Paired against the engine's own socket, the gateway added 0.38 ms to a voice request's first token on the W1 build and 0.29 ms on the final transport (W6).
- **Reproduce.** `make reproduce-e2` rebuilds the report, its tables, frontiers and waterfall from the compact records of W2–W4's 129 points ([`benchmarks/results`](../benchmarks/results/2026-10-23-e2/README.md)), `make reproduce-e2-w5` W5's from its 72 and `make reproduce-e2-a100` the A100 run's, and `make e2-figures` the figures above; `make reproduce-e1` rebuilds E1's tables from the 32× corpus, and `make reproduce-calibration` the simulator's calibration from its runs' exports. The exports and E2's raw run directories are assets of the v0.3 release, with the demo video: two calls side by side, the frozen box's and the PRs' configuration, measured again for it with the calls' audio saved (V2V p95 10,851 and 2,202 ms, not in the tables).

## Limitations

- One model (Qwen3-4B-FP8); one GPU model and one host for every experiment but the A100 run, which repeated E2's core on one A100 SXM4 80 GB; synthesized callers and fixed endpointing.
- 50 turns a point: a p95 lies between the third- and fourth-largest turn, so single runs differ by a few hundred milliseconds; a configuration near the SLO moves about 150 ms from run to run, one that misses it up to 1.9 s, and SGLang 7 s.
- The agent traces are block-scaled 32×, which keeps their reuse structure and changes their timing.
- W2–W4 count every turn, the first of each call included, which is slower under load: without it, the standard scenario's p95 at 16 agent sessions drops from 2,323 to 2,186 ms for the frozen box. W5's calls open with an unmeasured warm-up turn. Background goodput is counted over the calls, whose duration differs by configuration, and over their first 240 s, the same span for every point.
- vLLM counts a preemption when the victim next produces output, so per-point preemption counts are lower bounds.
- Every run of W2–W4 replays the same workload sample (agent seed 11, caller seed 0): their ranges are timing noise, not workload variation. W5's five rounds replay five samples (agent seeds 11–15), the same five for every configuration.
- In W4 the vLLM PRs' configurations ran in a session of their own, after the rounds and with no control (10-23 23:16Z – 10-24 01:35Z); W5 re-measured them in rounds beside the frozen box.
- vLLM 0.32's prompt-token counters see only a request's first scheduling: what a preempted request reloads or recomputes when it resumes is not in them. The records also keep the CPU tier's own load counter, which sees a resume's loads; what a resume recomputes is not measured.
- The two-node points never ran, and their code is parked on the `two-node` branch.
