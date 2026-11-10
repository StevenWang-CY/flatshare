# Benchmarks

The measurements behind Flatshare's claims, by the question each answers, for a reader checking a number. Each section
gives the answer, then its table or figure, then the record of the runs. [The report](REPORT.md) argues from these
numbers, [EXPERIMENTS.md](EXPERIMENTS.md) gives each experiment's design and
[the campaign map](EXPERIMENTS.md#which-campaign-measured-what-on-which-code), and
[GLOSSARY.md](GLOSSARY.md#experiments-and-how-they-are-counted) the terms.

Unless a section says otherwise, the numbers come from vLLM 0.32.0 serving Qwen3-4B-Instruct-2507-FP8 on
[the frozen box](#how-is-the-frozen-box-set-up), the RTX 5060 Ti in Phastform or its twin node Gray, with clients on cyan reaching it
through an SSH tunnel. The change is make-room, built from pull requests #65282 and #65283 proposed to vLLM (first
#63350 and #63351), which are open and rebased, with no maintainer review as of 2026-11-09
([GLOSSARY.md](GLOSSARY.md#the-change-and-its-parts)). It now builds on vLLM 0.33.0, which
[its port's record](../benchmarks/results/2026-11-02-port-v0.33.0/README.md) checks and no round has measured.

**Contents.**
- [How is the frozen box set up?](#how-is-the-frozen-box-set-up)
- [What does resuming an agent turn cost?](#what-does-resuming-an-agent-turn-cost)
- [What does agent load do to a voice turn on the same GPU?](#what-does-agent-load-do-to-a-voice-turn-on-the-same-gpu)
- [Why does a voice turn wait beside agents?](#why-does-a-voice-turn-wait-beside-agents)
- [How is a point of voice under agent load measured?](#how-is-a-point-of-voice-under-agent-load-measured)
- [How close does each configuration bring the voice tail to its SLO?](#how-close-does-each-configuration-bring-the-voice-tail-to-its-slo)
- [Does any configuration without the change reach the SLO?](#does-any-configuration-without-the-change-reach-the-slo)
- [What changes where the pool has room, and with more calls?](#what-changes-where-the-pool-has-room-and-with-more-calls)
- [What does the change take from the agents?](#what-does-the-change-take-from-the-agents)
- [How often do running agents leave two calls short of KV?](#how-often-do-running-agents-leave-two-calls-short-of-kv)
- [Which step cap should the change run at?](#which-step-cap-should-the-change-run-at)
- [What do the serving engine's additions do to a voice turn?](#what-do-the-serving-engines-additions-do-to-a-voice-turn)
- [How many agents' sessions does one card take beside both calls?](#how-many-agents-sessions-does-one-card-take-beside-both-calls)
- [Which arrangement of two cards does the most background work?](#which-arrangement-of-two-cards-does-the-most-background-work)
- [Does the arrangement to deploy keep its SLO over a day?](#does-the-arrangement-to-deploy-keep-its-slo-over-a-day)
- [Do the rules carry to a datacenter card?](#do-the-rules-carry-to-a-datacenter-card)
- [What does a failure do to the calls and the agents?](#what-does-a-failure-do-to-the-calls-and-the-agents)
- [What does the gateway add to a request?](#what-does-the-gateway-add-to-a-request)
- [What does an API layer add to an agent's turn?](#what-does-an-api-layer-add-to-an-agents-turn)
- [How do the agents use the KV cache?](#how-do-the-agents-use-the-kv-cache)
- [Does knowing when an agent returns help?](#does-knowing-when-an-agent-returns-help)
- [How closely does the replay follow the traces?](#how-closely-does-the-replay-follow-the-traces)

## How is the frozen box set up?

The engine's KV pool is fixed at 5 GiB, 36,400 tokens, beside a 20 GiB CPU tier in host RAM, and the three servers leave
1.9 GiB of the card free when idle. W0 measured these on 2026-10-22, and the stress peak on 2026-10-21. Start times are
under [failures](#what-does-a-failure-do-to-the-calls-and-the-agents).

| | Value |
|---|---|
| Host | Phastform: two Xeon E5-2609 v4 at 1.7 GHz, 188 GiB of RAM, the RTX 5060 Ti 16 GB on PCIe Gen3 x8 |
| GPU KV pool | 36,400 tokens, fixed at 5 GiB (32k context) |
| CPU tier | 20 GiB in `/dev/shm`: 17.35 GiB on NUMA node 1 (GPU-local), 2.65 GiB spilled to node 0 (note 1) |
| GPU memory, all three servers idle | 14,408 / 16,311 MiB |
| GPU memory, stress peak (LLM c=16, 30k prefill, STT c=4, TTS c=4) | 14,086 / 16,311 MiB, with no OOM or preemption |

1. A remote NUMA node cost at most 1% of pinned bandwidth.

## What does resuming an agent turn cost?

Restoring a prompt's KV from host RAM beats recomputing it at every length measured, from 1,024 tokens. At 16,384 tokens
the first token takes 97 ms from the GPU prefix cache, 509 ms from the CPU tier and 3,264 ms recomputed on the box, and
46, 268 and 1,471 ms on the A100.

`flatshare-bench tiers` sends a unique prompt three times for one token each: cold, again at once, and after other
prompts have pushed it out of the GPU pool. vLLM's prefix-cache counters confirm where its tokens came from. The table
gives the median of 5 samples on the box and 3 on the A100, with the audio servers resident, the box's pool and CPU
tier, and 512-token steps.

| Prompt tokens | Recomputed, box | GPU hit, box | Restored, box | Recomputed, A100 | GPU hit, A100 | Restored, A100 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 135 ms | 45 ms | 64 ms | 64 ms | 18 ms | 32 ms |
| 2,048 | 198 ms | 40 ms | 95 ms | 128 ms | 21 ms | 49 ms |
| 4,096 | 437 ms | 48 ms | 157 ms | 269 ms | 25 ms | 81 ms |
| 8,192 | 1,114 ms | 64 ms | 281 ms | 602 ms | 32 ms | 139 ms |
| 16,384 | **3,264 ms** | **97 ms** | **509 ms** | **1,471 ms** | **46 ms** | **268 ms** |
| 24,576 | 6,506 ms | 128 ms | 722 ms | 2,607 ms | 61 ms | 379 ms |
| 31,744 | 10,239 ms | 156 ms | 903 ms | 3,825 ms | 75 ms | 483 ms |

The box's fits, which [`flatshare-core`'s cost model](../crates/flatshare-core/src/cost.rs) keeps, with the A100's in
parentheses:
- **a GPU hit** takes 35 ms + 3.8 µs a token (17 ms + 1.80 µs);
- **a restore** adds 10 ms + 23.6 µs a token over a hit, 6.2 GB/s of KV at 144 KiB a token (2.8 ms + 12.9 µs);
- **a recompute** takes 48 ms + 63 µs a token + 8.1 ns a token², the quadratic term being attention (4.2 ms, 56.5 µs and
  2.01 ns).

Background load slows every path. On the box, eight decode streams add about 85 ms to a hit and a restore alike, so PCIe
is not the bottleneck. Behind a stream of cold 8k prefills even a cached 1k prompt waits about 250 ms, and recompute
slows 1.6–2.5×.

## What does agent load do to a voice turn on the same GPU?

It slows the LLM's part of the turn most. Beside one stream of cold 8k-token prefills the server-side turn's median
doubles, from 589 to 1,261 ms. Beside 16 replayed agent sessions the calls' V2V p95 rises by 552 ms and their p99 by
1.1 s, where the SLO allows 250 ms at p95.

`flatshare-bench voice-turn` sends a LibriSpeech clip of 2.0 or 3.2 s to STT, streams the transcript through the LLM at
the realtime priority and the first sentence to TTS. It times the turn to TTS's first byte, the server-side share of
V2V. W1 ran 30 turns a row on 2026-10-22, p50 / p90 in ms.

| Background agent load | STT | LLM first token | LLM first sentence | TTS first audio | **Turn** | Background prefill |
|---|---:|---:|---:|---:|---:|---:|
| None | 160 / 168 | 49 / 51 | 251 / 319 | 176 / 187 | **589 / 665** | — |
| One stream of cold 8k-token prefills | 233 / 240 | 172 / 197 | 807 / 937 | 226 / 251 | **1,261 / 1,413** | 6,220 tokens/s |
| Two streams | 233 / 248 | 565 / 1,207 | 1,802 / 2,113 | 231 / 254 | **2,279 / 2,593** | 6,966 tokens/s |

- **The LLM's first sentence carries most of the damage**, its decode steps shared with 512-token prefill chunks. STT
  and TTS, time-sliced with the engine, slow by up to 46% and 31%.
- **MPS does not help.** With the three servers as clients of one MPS daemon, the turn beside one prefill stream worsens
  to 1,403 / 2,061 ms and background prefill drops 13%. The interference that matters is inside the engine, so the box
  stays on time-slicing.

End to end, two callers of 25 turns call the stock voice agent through LiveKit, and V2V runs from the end of the
caller's speech to the agent's first audio, read at the caller on one clock.

| Setup | V2V p50 | p95 | p99 | Barge-in to silence p50 / p95 |
|---|---:|---:|---:|---:|
| Voice alone, LiveKit's end-of-turn model | 1,646 ms | 3,226 ms | 3,444 ms | 637 / 686 ms |
| Voice alone, fixed endpointing (experiments) | **1,614 ms** | **1,760 ms** | **1,889 ms** | 634 / 696 ms |
| Beside the agent replay (16 agent sessions, 1.3 requests/s), fixed endpointing | 1,664 ms | **2,312 ms** | 3,013 ms | 563 / 687 ms |

- **Endpointing is most of voice alone.** About 0.75 s of each turn is endpointing, the VAD's 0.55 s of silence and
  about 0.2 s more. LiveKit's end-of-turn model judged one synthesized line in six unfinished and waited its 3 s
  maximum, so experiments end a turn a fixed delay after speech, the same for every arm.
- **Shorter silence would take a quarter second off.** In the endpointing sweep 0.3 s of silence brought the V2V p50
  from 1,612 to 1,346 ms, and answered one line of 40 before it ended. Every E2 point keeps 0.55 s
  ([the sweep's record](../benchmarks/results/2026-10-28-endpointing/README.md)).

<a id="the-kv-case-make-room"></a>

## Why does a voice turn wait beside agents?

It waits for KV when running agents fill the pool, and for agent prefill, which rides every step of its reply. In one
session on the box, stock vLLM 0.32 gave a voice-sized request sent into a full pool its first token after 4.9–5.0 s,
and behind agent prefill its tokens came 60–66 ms apart. The KV preemption cut the first to 0.22–0.29 s, and the budget
cap the second to 16–20 ms.

Three probes isolate the waits ([`probe.py`](../python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py),
[`tail.py`](../python/flatshare_vllm/src/flatshare_vllm/make-room/tail.py)). The KV case sends a 750-token priority-0
request into a pool that two decoding 18k-token agent requests fill, and the prefill case sends it beside three
10k-token agent prefills. The calls' tail runs two calls beside 12 closed-loop agent sessions for 240 s.

### Why does a full pool keep a voice turn waiting?

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-kv-timeline-narrow.svg">
  <img src="figures/report-kv-timeline.svg" width="100%" alt="In panel a, the share of the KV pool's 2,274 blocks in use on stock vLLM while the KV case's two agent requests of 18k prompt tokens prefill and decode, read about every 0.55 s: 98.7% at 7.7 s, 20 blocks free at 8.3 s, where the KV case sends its voice turn, none freed until 12.6 s, and 50.1% at 13.1 s, once vLLM has preempted agent B so that agent A can grow; in panel b, each request's output tokens, both decoding into the last free blocks until agent B stops at 192 tokens while agent A reaches 812 by 24.5 s.">
</picture>

**On vLLM 0.32 a full KV pool frees nothing until a running request cannot grow.** The KV case's two agent requests on
stock vLLM, sampled every half second.

| Scheduler | Runs | Priority-0 TTFT | Then |
|---|---:|---:|---|
| Stock vLLM 0.32.0, priority scheduling | 3 | 4,923, 5,069, 5,077 ms | 2 running, 1 waiting |
| `make-room`, the KV preemption alone (#63350) | 2 | 247, 321 ms | 2 running, 1 waiting |
| `make-room`, the KV preemption and the budget cap (#63350 and #63351) | 2 | 225, 297 ms | 2 running, 1 waiting |
| Either, on an idle pool | 4 | 107–137 ms | — |

The two agent requests fill the pool to 98.7% at 7.7 s and decode into its last blocks until 12.6 s, when one cannot
grow and vLLM preempts the other. A priority-0 request that arrives in between waits, since running requests are
preempted only for each other's growth. These runs are Gray's, on 2026-10-23.
[Their record](../benchmarks/results/2026-10-23-make-room-probes/README.md) also holds that day's prefill case and
calls' tail, with arm 9's reserve.

<a id="w5-the-fixed-make-room-on-phastform"></a>

### What does each rule remove?

The KV preemption removes the wait for KV, and the budget cap removes agent prefill from the reply's steps, so beside
closed-loop agents only the two together bring the calls' p95 under 0.4 s. All three engines ran in one session on
Phastform on 2026-10-25, the budget cap at its default 16-token window.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-isolation-narrow.svg">
  <img src="figures/report-isolation.svg" width="100%" alt="Four probe runs on the RTX 5060 Ti for each of stock vLLM, the KV preemption and the change at its 16-token window: in panel a, the KV case, the voice turn's first token while two agent requests of 18k tokens hold the pool (stock vLLM 4,900–4,971 ms; the KV preemption 219–290 ms; the change 216–294 ms); in panel b, the prefill case, its first token to its 16th while three agent prefills of 10k tokens run, the 16th at stock vLLM 1,220–1,529 ms; the KV preemption 1,211–1,528 ms; the change 536–615 ms.">
</picture>

**In isolation, a voice-sized request waits 5.0 s for KV on stock vLLM and 0.29 s with the KV preemption, and agent
prefill slows its tokens three- to fourfold until the budget cap holds it back.** Four runs of each probe, the budget
cap at a 16-token window. (a) The KV case's first token, each circle a run and the label the median. (b) The prefill
case, each line from a run's first token to its 16th, the bar and label at the median 16th token.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-tail-narrow.svg">
  <img src="figures/report-tail.svg" width="100%" alt="The cumulative share of two calls' turns by their time to first token beside 12 closed-loop agent sessions, pooled over two 240 s runs of each scheduler, on a log scale with the 95th percentile marked: in panel a, with the agents' prompts fresh each turn, the p50 and p95 at 2,244 and 6,830 ms on stock vLLM (96 turns), 716 and 4,111 ms with the KV preemption (117 turns), 236 and 388 ms with the change (146 turns); in panel b, with each agent's prompt grown by a tool result, the p50 and p95 at 1,324 and 7,158 ms on stock vLLM (102 turns), 615 and 1,073 ms with the KV preemption (125 turns), 258 and 358 ms with the change (146 turns).">
</picture>

**Beside closed-loop agents, the KV preemption shortens the calls' first-token tail and the budget cap removes it.** The
agents' prompts are fresh each turn in panel a and grow by a tool result in panel b. The change is both rules, the
budget cap at a 16-token window.

| Engine | KV case TTFT | Prefill case TTFT / 16th token | Calls' TTFT, fresh prompts, p50 / p95 | Growing prompts, p50 / p95 |
|---|---:|---:|---:|---:|
| Stock vLLM 0.32.0, priority scheduling | 4,968 ms (4,900–4,971) | 529 (313–558) / 1,500 (1,220–1,529) ms | 2,244 / 6,830 ms | 1,324 / 7,158 ms |
| #63350 alone (the KV preemption) | 290 ms (219–290) | 416 (304–556) / 1,350 (1,211–1,528) ms | 716 / 4,111 ms | 615 / 1,073 ms |
| #63350 and #63351 (the budget cap, 16 tokens) | 290 ms (216–294) | 304 (292–318) / 575 (536–615) ms | 236 / 388 ms | 258 / 358 ms |

Behind the agents' prefill the reply's tokens came 60–66 ms apart on stock vLLM, 60–65 ms with the KV preemption alone
and 16–20 ms with both rules.

A KV load needs the budget cap's hold too, since a call whose prefix was evicted reloads it from the CPU tier over
several steps. A module that held nothing during the load gave the calls' first token 125–131 ms later at the median,
and holding agent prefill for the load's first 16 steps gave it back.

[W5's probe record](../benchmarks/results/2026-10-24-w5-probes/README.md) holds these runs, four of each case and two of
each tail, and the modules'.

<a id="the-probes-on-the-pull-requests-revision-of-3-november-2026-11-04"></a>

### Does the revision of 3 November schedule them the same?

Yes. On that code, v1.21's engine, the KV preemption gives the voice turn its first token after 0.20–0.30 s, and the
budget cap its 16th after 0.52–0.58 s instead of 1.24–1.51 s. Stock vLLM, the control in the same session, waited
4,906–4,981 ms in the KV case, as on 2026-10-25.

| Engine | KV case TTFT | Prefill case TTFT / 16th token |
|---|---:|---:|
| Stock vLLM 0.32.0, priority scheduling | 4,976 ms (4,906–4,981) | 420 (325–541) / 1,374 (1,236–1,514) ms |
| #65282 alone (the KV preemption) | 284 ms (198–298) | 328 (257–470) / 1,242 (1,164–1,429) ms |
| #65282 and #65283 (the budget cap, 16 tokens) | 274 ms (198–299) | 292 (232–325) / 560 (518–582) ms |

Each case ran four times on Phastform on 2026-11-04, without the tails
([the record](../benchmarks/results/2026-11-04-r3-probes/README.md)).

<a id="the-search-for-stalls-2026-11-05"></a>

### Can the two rules stall the engine?

Not as revised. A random search over 400,000 workloads on small KV pools found no stall the options cause on #65283,
against 241 before its fix. Each workload ran the real scheduler on CPU with the options off and on, and a stall is the
options' when it ends once both are turned off.

| Code | Workloads | Stalls the options cause | Workloads stuck, options off / on |
|---|---:|---:|---:|
| #65283 before the fix | 400,000 | 241 | 5,578 / 5,225 |
| #65283 | 400,000 | 0 | 5,578 / 5,287 |
| #65282 alone | 400,000 | 0 | 5,578 / 5,634 |
| The make-room module of v1.22, the serving profile's switches | 200,000 | 153 | 2,830 / 2,156 |
| The make-room module of v1.23, the serving profile's switches | 200,000 | 0 | 2,830 / 2,071 |

Every stall left has nothing running and stays stuck with the options off, as on vLLM's main. A third search of 60,000
workloads, over LoRA, late connector lookups, grammars and sliding windows, found 100 before the fix and none after. The
workloads are small and short, so they show which schedules can stall, not how often a real engine meets one
([the record](../benchmarks/results/2026-11-05-liveness/README.md)).

### How long should the window be?

As long as the replies. The p95 of a 40-token reply's last token falls from 3.8 s at a 16-token window to 1.0 s at 48,
and no further at 64. The agents' prompt throughput falls 8% from 16 to 48 and no further, since a hold ends with its
reply.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-window-narrow.svg">
  <img src="figures/report-window.svg" width="100%" alt="In panel a, the cumulative share of 5,571 voice replies by their output tokens, 23 at the median and 52 at most, 21% ending within a 16-token window and all but 1 within a 48-token one; in panel b, the p95 of a 40-token reply's last token by the window, the median of two runs (16 tokens 3.76 s; 24 tokens 2.64 s; 32 tokens 1.85 s; 48 tokens 1.04 s; 64 tokens 1.02 s), its first token's staying between 326 and 448 ms; in panel c, the agents' prompt tokens a second, 3,483 at 16 tokens and 3,205 at 48, 8% less.">
</picture>

**A window that covers the replies keeps their tokens fast, and a longer one costs the agents nothing more.** Panel a
draws the replies of W5's points, and panels b and c the table.

| Window (tokens) | First token p95 | 16th token p95 | 40th token p95 | Agents' prompt tokens/s |
|---:|---:|---:|---:|---:|
| 16 | 448 | 726 | 3,759 | 3,483 |
| 24 | 387 | 648 | 2,642 | 3,443 |
| 32 | 326 | 572 | 1,854 | 3,300 |
| 48 | 347 | 616 | 1,037 | 3,205 |
| 64 | 366 | 612 | 1,018 | 3,246 |

`knee.sh` ran each window twice for 240 s beside the closed-loop agents, with 40-token call replies, on 2026-10-25, and
the table gives the medians in ms ([the record](../benchmarks/results/2026-10-25-window-knee/README.md)).
[PRODUCTION.md](PRODUCTION.md#the-replys-first-tokens) sets the window from it.

## How is a point of voice under agent load measured?

Two callers, 25 turns each with a barge-in on 9 of the 50, call the stock voice agent through LiveKit, beside AIPerf's
replay of the 32× Claude Code corpus through `flatshare-gateway`
([the design](EXPERIMENTS.md#e2-voice-under-agent-load), [the indicators](SLO.md#indicators)).
- **V2V** is read from waveforms at the callers. A turn unanswered within 30 s counts as 30,000 ms, and ≥ marks a
  percentile it sets.
- **The agents' token rate** counts the agent requests completed during the calls, and also over the calls' first 240 s,
  the same window for every point.
- **The engine's counts** are partial. vLLM counts a preemption only at the victim's next output, and a request's prompt
  tokens by source only at its first scheduling.

The SLO is V2V p95 at most 250 ms above voice alone with as many callers, measured in each campaign and, from
2026-10-28 on, in each round ([SLO.md](SLO.md#the-slos-reference-campaign-by-campaign)). With 59 replies a point, a p95
lies between the third- and fourth-slowest, so single runs differ by a few hundred milliseconds.

## How close does each configuration bring the voice tail to its SLO?

To 0.20 s over it. Beside 24 agent sessions whose KV fills the pool, the configuration to deploy brings the V2V p95 from
tuned vLLM's 10.4–10.5 s to 2.16–2.25 s, while the agents keep 97% of their token rate. That is 202 ms over the SLO of
2,034 ms at the median of its fifteen runs. The KV preemption alone reaches 5.09 s, and both rules at a 16-token window
3.09 s, since a whole reply outruns that window.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-headline-narrow.svg">
  <img src="figures/report-headline.svg" width="100%" alt="Each run's voice-to-voice p95 on a log scale against the SLO of 2.03 s, and the agents' tokens a second over the calls' first 240 s, with each configuration's median: voice alone, no agents: 1.78 s at the median of 5 runs (1.76 to 1.82 s); tuned vLLM, the baseline: 10.48 s at the median of 10 runs (8.69 to 12.12 s), its agents at 3,305 tokens/s, 100% of tuned vLLM's; preempt-resubmit, before the first byte: 2.52 s at the median of 5 runs (2.52 to 2.87 s), its agents at 3,332 tokens/s, 101% of tuned vLLM's; the KV preemption, #63350: 5.09 s at the median of 5 runs (4.69 to 6.26 s), its agents at 3,414 tokens/s, 103% of tuned vLLM's; the change, 16-token window: 3.09 s at the median of 5 runs (2.87 to 3.19 s), its agents at 3,298 tokens/s, 100% of tuned vLLM's; and the configuration to deploy, 48-token window: 2.24 s at the median of 15 runs (2.13 to 2.40 s), its agents at 3,203 tokens/s, 97% of tuned vLLM's.">
</picture>

**The two rules bring the voice tail to 0.20 s over the SLO, and the agents keep most of their throughput.** (a) Each
run's p95 V2V as a tick and each configuration's median as a marker and a label, on a log scale, against the SLO
(dashed). (b) The agents' tokens a second, each run a tick, with each median's share of tuned vLLM's.

W5 ran three campaigns of five rounds on 2026-10-24 and 25, round N of each on the agents' sample of seed 10 + N, and
voice alone gave 1,784 ms. Each ran the configuration to deploy beside its own comparisons. W5's own ran tuned vLLM and
the same configuration on v0.3's module, the PR campaign tuned vLLM, the KV preemption alone and the change at
16 tokens, and the extension preempt-resubmit, which aborts agent requests before their first byte. The table gives each
campaign's median of five runs in ms, and for V2V p95 their range.

| Configuration | Campaign | V2V p50 / p95 / p99 | p95 range | First byte p95 | Agents' tokens/s, first 240 s | Agent TTFT p50 / p99 |
|---|---|---:|---:|---:|---:|---:|
| Voice alone | W5 | 1,619 / 1,784 / 1,892 | 1,764–1,819 | 60 | — | — |
| Tuned vLLM | W5 | 4,772 / 10,432 / 11,361 | 8,686–10,770 | 6,942 | 3,306 | 63 / 106 s |
| Tuned vLLM | PRs | 5,134 / 10,523 / 12,466 | 9,946–12,115 | 7,786 | 3,304 | 63 / 107 s |
| Preempt-resubmit (arm 8c) | extension | 1,990 / 2,522 / 2,919 | 2,516–2,868 | 349 | 3,332 | 32 / 193 s |
| The KV preemption (#63350 alone) | PRs | 3,191 / 5,085 / 5,970 | 4,693–6,261 | 1,737 | 3,414 | 67 / 112 s |
| The change, 16-token window | PRs | 1,983 / 3,094 / 3,885 | 2,868–3,192 | 364 | 3,298 | 71 / 105 s |
| **The configuration to deploy** | W5 | 1,967 / **2,250** / 2,384 | 2,175–2,307 | 376 | 3,156 | 70 / 110 s |
| **The configuration to deploy** | PRs | 1,982 / **2,232** / 2,379 | 2,174–2,397 | 376 | 3,236 | 74 / 115 s |
| **The configuration to deploy** | extension | 1,940 / **2,162** / 2,300 | 2,132–2,293 | 363 | 3,203 | 72 / 111 s |
| The same on v0.3's module | W5 | 1,941 / 2,241 / 2,356 | 2,134–2,333 | 382 | 3,132 | 74 / 113 s |

The differences between configurations on the same sample, round by round, and their median:

| Pair | V2V p95 (ms) | Median | First byte p95, median | Agents' tokens/s over 240 s, median |
|---|---|---:|---:|---:|
| The fixed make-room less v0.3's (W5's campaign) | −37, −158, +164, +8, +93 | +8 | −6 | −24 |
| Tuned vLLM, the PR campaign less W5's | −630, −637, +2,124, +2,333, +92 | +92 | +1,000 | −51 |
| The configuration to deploy, the PR campaign less W5's | −62, +126, −109, −17, +91 | −17 | +26 | +58 |
| The configuration to deploy, the extension less W5's | +57, +82, −166, −88, −161 | −88 | −10 | +22 |
| The configuration to deploy less preempt-resubmit (the extension) | −223, −603, −390, −360, −723 | −390 | +9 | −130 |

- **The result holds over five samples.** The configuration to deploy stays at 2.13–2.40 s while tuned vLLM moves from
  8.7 to 12.1 s, and the configurations rank the same in every round of the PR campaign. Pooled, its fifteen runs put
  286 of 885 replies over the SLO, 32% where it allows 5%, against 532 of tuned vLLM's 590.
- **Preempt-resubmit makes room at the agents' tail.** Its agents keep their token rate, but their TTFT p99 is 193 s
  against tuned vLLM's 106 s.
- **Nearly every reply fits a 48-token window.** W5's gateway logged 5,571 voice replies, 23 tokens at the median and 52
  at most, 79% of them past 16 tokens and one past 48.

[W5's record](../benchmarks/results/2026-10-24-e2-w5) holds every run, and its `compare.py` prints these tables.

<a id="after-w5-the-first-tokens-in-flight"></a>

### Do W5's numbers hold on the code of 25 October?

Yes. FB re-ran voice alone, tuned vLLM and the configuration to deploy on that code, each round on W5's sample of the
same number. The configuration to deploy came out at 2,268 ms against W5's 2,250, within W5's fifteen runs of it
(2,132–2,397 ms).

That code moved the GPU servers to Unix sockets, and its budget cap counts a reply's first tokens in flight, as #63351
did from its revision of 2026-10-25 and #65283 does. W5's module counted them a step late, so it held agent prefill one
step past the window, which met 81% (248 of 305) of the replies in W5's 16-token points but 1 of the 2,469 in its
48-token points.

| Configuration | Rounds | W5 V2V p95 | FB V2V p95 | FB − W5, round by round | Agents' tokens/s, W5 / FB |
|---|---:|---:|---:|---|---:|
| Voice alone | 5 | 1,784 (1,764–1,819) | 1,796 (1,768–1,819) | −4, +18, −51, +55, +11 | — |
| Tuned vLLM | 5 | 10,432 (8,686–10,770) | 10,399 (8,374–10,866) | −177, −2,396, +1,706, +761, +435 | 3,306 / 3,368 |
| The configuration to deploy | 5 | 2,250 (2,175–2,307) | 2,268 (2,246–2,343) | +77, +71, −45, +19, +36 | 3,156 / 3,252 |

No turn failed ([FB's record](../benchmarks/results/2026-10-25-e2-fb), whose `compare.py` prints the table).

## Does any configuration without the change reach the SLO?

No. Beside 24 agent sessions every configuration without the change stayed 0.5 s or more over the SLO of 2,017 ms, at
the median of its runs. Preempt-resubmit came nearest, at 2,591 ms, by aborting agent work, and a gateway policy timed
by the conversation left tuned vLLM's tail at 10,993 ms.

These are W2's to W4's runs, from 2026-10-22 to 24, on one sample of the agents' traces and one to five runs a
configuration, so their ranges measure timing noise only. Every turn counted, the first of each call included. The
change's rows ran an earlier make-room, after the rounds.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-deployable-narrow.svg">
  <img src="figures/report-deployable.svg" width="100%" alt="Each of 7 configurations beside 24 agent sessions of the traces with the largest contexts, a row under its family, in panel a its voice-to-voice p95 on a log scale against the SLO of 2.02 s, the median of its runs and their range, and in panel b the agents' tokens a second over the calls, only the lifecycle policy over make-room and the configuration to deploy within 0.2 s of the SLO, every other 0.5 s or more over it: tuned vLLM 10.14 s at 3,710 tokens/s; 1,024-token steps 8.79 s at 4,097 tokens/s; a token reservation 5.67 s at 3,564 tokens/s; SGLang with HiCache 11.95 s at 4,723 tokens/s; preempt-resubmit 2.59 s at 3,630 tokens/s; the lifecycle policy over make-room 2.06 s at 3,411 tokens/s; and the configuration to deploy 2.19 s at 3,493 tokens/s.">
</picture>

**Only the configuration to deploy and the lifecycle policy over make-room, both on the change, come within 0.2 s of the
SLO at the median of their runs; every other configuration an operator can deploy stays 0.5 s or more over it.** Panel b
gives the agents' tokens a second over the whole calls, as the table does, and the table adds the KV preemption and the
change at its 16-token window.

| Configuration | Runs | V2V p50 / p95 / p99 | p95 range | First byte p95 | Background tokens/s | Agent TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 5 | 1,611 / 1,767 / 1,799 | 1,744–1,805 | 66 | — | — |
| Tuned vLLM | 5 | 4,520 / 10,138 / 12,110 | 9,257–10,811 | 7,834 | 3,710 | 65 / 99 s |
| 1,024-token steps, no threshold | 5 | 4,348 / 8,789 / 10,024 | 8,161–9,490 | 6,523 | 4,097 | 57 / 85 s |
| A token reservation, 2,048 tokens | 5 | 3,127 / 5,671 / 6,681 | 4,906–5,798 | 3,190 | 3,564 | 69 / 102 s |
| SGLang with HiCache | 5 | 4,705 / 11,948 / 18,063 | 9,394–16,702 | 8,012 | 4,723 | 46 / 82 s |
| Preempt-resubmit before the first byte | 5 | 1,960 / 2,591 / 2,684 | 2,110–2,663 | 374 | 3,630 | 31 / 160 s |
| **The lifecycle policy over make-room** (arm 9), every reply capped at 16 tokens | 5 | 1,825 / **2,057** / 2,139 | 1,980–2,134 | 290 | 3,411 | 75 / 103 s |
| The KV preemption (#63350 alone) | 3 | 3,246 / 5,773 / 8,110 | 5,612–6,063 | 2,785 | 3,633 | 66 / 97 s |
| The change, 16-token window | 3 | 1,986 / 3,229 / 4,135 | 3,220–4,772 | 338 | 3,568 | 71 / 100 s |
| **The configuration to deploy, 48-token window** | 3 | 1,943 / **2,193** / 2,281 | 2,168–2,286 | 372 | 3,493 | 72 / 97 s |

W2's single runs of the other settings fared no better. vLLM's KV watermark, static caps on the agents, #54530, SGLang's
priority preemption, llm-d's flow control and SMG gave 8,573–12,051 ms. A limit on queued tokens refused 48 of the 50
voice turns ([every point](../benchmarks/results/2026-10-23-e2/report/frontier.md)).

Arm 9 is the lifecycle policy over make-room at a 16-token window, with a 2,048-token reserve and every reply capped at
16 tokens. With one part removed or changed at a time, three runs each:

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-ablations-narrow.svg">
  <img src="figures/report-ablations.svg" width="100%" alt="The voice-to-voice p95 on a log scale against the SLO of 2.02 s, beside 24 agent sessions of the traces with the largest contexts, of the lifecycle policy over make-room with all its parts, 2.06 s at the median of 5 runs (1.98 to 2.13 s), and with one part removed or changed, the median of three runs and their range: in its gateway policy, without the critical-window throttle 2.05 s (1.99 to 2.08 s), without the headroom 2.03 s (1.95 to 2.07 s), the headroom armed by the request 2.12 s (2.01 to 2.17 s), the throttle's budget at 0 ms 2.11 s (2.09 to 2.17 s), the throttle's budget at 500 ms 2.06 s (1.99 to 2.08 s), without the lifecycle events 2.04 s (2.01 to 2.05 s), without degrade 3.84 s (2.89 to 4.08 s), without degrade, with a 48-token window 2.20 s (2.10 to 2.29 s), without the gateway policy 4.03 s (2.98 to 4.06 s); in the engine under it, without the KV preemption 2.04 s (1.99 to 2.04 s), without the reserve 2.12 s (2.12 to 2.15 s), without the budget cap 5.29 s (5.06 to 6.11 s), without the change 10.99 s (10.03 to 11.96 s).">
</picture>

**Removing degrade, the gateway policy, the budget cap or the change puts the p95 1.8 s or more over the SLO; removing
or changing any other part, or replacing degrade with a 48-token window, keeps it within 0.2 s.** Each row is the median
p95 of its runs, five with all the parts and three with one removed or changed, a line over their range; the dashed line
is the SLO, 2.02 s.

| Arm 9 | Its part | V2V p95 | Range |
|---|---|---:|---:|
| without the critical-window throttle | gateway | 2,049 | 1,995–2,081 |
| without the headroom | gateway | 2,033 | 1,947–2,071 |
| the headroom armed by the request | gateway | 2,124 | 2,008–2,165 |
| the throttle's budget at 0 ms / 500 ms | gateway | 2,109 / 2,060 | 2,094–2,166 / 1,990–2,080 |
| without the lifecycle events | gateway | 2,036 | 2,005–2,048 |
| without degrade | gateway | 3,837 | 2,890–4,079 |
| without degrade, with a 48-token window | gateway | 2,205 | 2,098–2,289 |
| without the gateway policy (the engine only) | gateway | 4,034 | 2,984–4,062 |
| without the KV preemption | engine | 2,037 | 1,990–2,039 |
| without the reserve | engine | 2,116 | 2,115–2,155 |
| without the budget cap | engine | 5,295 | 5,063–6,115 |
| without the change (the gateway only) | engine | 10,993 | 10,028–11,958 |

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-deployable-stages-narrow.svg">
  <img src="figures/report-deployable-stages.svg" width="100%" alt="The voice turn's mean voice-to-voice time stacked by stage from the end of the caller's speech, beside 24 agent sessions but for voice alone: voice alone, 1.60 s, 0.09 s of it waiting for the LLM's first token and 0.21 s decoding the first sentence; tuned vLLM, 5.21 s, 2.78 s of it waiting for the LLM's first token and 1.02 s decoding the first sentence; a token reservation, 3.38 s, 0.95 s of it waiting for the LLM's first token and 1.01 s decoding the first sentence; preempt-resubmit, 2.02 s, 0.28 s of it waiting for the LLM's first token and 0.34 s decoding the first sentence; the lifecycle policy, 1.83 s, 0.23 s of it waiting for the LLM's first token and 0.20 s decoding the first sentence; the configuration to deploy, 1.95 s, 0.28 s of it waiting for the LLM's first token and 0.25 s decoding the first sentence.">
</picture>

**Only the reply to its first sentence changes from row to row, and endpointing, TTS and transport stay within 0.1 s in
every configuration.** Mean stage times in seconds, the reply split at the LLM's first token into the wait (violet) and
the first sentence's decode (green); the table gives them in ms and adds the change at its 16-token window. Voice alone
has four runs, since its W2 run predates the worker's stage times.

| Configuration | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | LLM first token |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 757 | 297 | 199 | 352 | 1,605 | 80 |
| Tuned vLLM | 822 | 3,801 | 244 | 343 | 5,211 | 2,760 |
| A token reservation | 827 | 1,967 | 238 | 346 | 3,378 | 935 |
| Preempt-resubmit | 826 | 622 | 222 | 353 | 2,023 | 274 |
| The lifecycle policy over make-room | 824 | 429 | 222 | 354 | 1,830 | 221 |
| The change, 16-token window | 826 | 914 | 245 | 353 | 2,336 | 265 |
| The configuration to deploy | 824 | 526 | 249 | 353 | 1,953 | 270 |

- **Engine priority and static answers fall short.** Priority scheduling at any step cap and SGLang's, whose pool came
  out 5–7% smaller than vLLM's, leave the p95 at 8.8–12.1 s. The 2,048-token reservation halves the tail, still 3.7 s
  over, and caps on the agents do not help, because two contexts fill the pool.
- **Preempt-resubmit makes room by aborting** agent requests before their first byte, so it cannot reach one already
  decoding.
- **The budget cap is necessary, and either the KV preemption or the reserve makes room.** With both, the reserve kept
  the pool from filling, so the KV preemption never fired.
- **The reply has to fit the window.** Full replies, 20–22 tokens at the median and up to 40, outrun a 16-token window
  and decode beside agent prefill again.
- **The gateway's lifecycle mechanisms could not act.** Deferring new agent requests frees none of the KV running ones
  hold, so degrade capped all 60 realtime requests in every run of arm 9, and the throttle deferred nothing. Beside 24
  sessions of the whole corpus it deferred 27 requests, which no run isolated.
- **Neither the first turn nor the run sets this.** Without each call's first turn the medians moved by at most 86 ms,
  and a configuration near the SLO moves about 150 ms in p95 from run to run.

Over the calls' first 240 s, arm 9 kept 94.5% of tuned vLLM's 3,552 agents' tokens a second, the configuration to deploy
94.7% and preempt-resubmit 101.2%. [W2–W4's record](../benchmarks/results/2026-10-23-e2/README.md) holds all 129 points
(`make reproduce-e2`).

## What changes where the pool has room, and with more calls?

The change helps less where the pool has room, and overlapping calls hold the agents' prefill for longer. Beside 16
sessions of the whole corpus the configuration to deploy had the lower p95 in all five rounds, 2,050 ms at the median
against tuned vLLM's 2,516. With four or six callers its p95 stayed 257–269 ms over their SLO, about as far as with two.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-margin-narrow.svg">
  <img src="figures/report-margin.svg" width="100%" alt="In panel a, with 16 agent sessions of the whole corpus, where the pool has room, tuned vLLM's V2V p95 and the configuration to deploy's in each of five rounds against the SLO of 2.03 s (2.15 and 2.04 s, 3.83 and 2.42 s, 3.07 and 2.12 s, 2.30 and 2.05 s, 2.52 and 2.02 s); in panels b to d, the configuration to deploy with two, four and six callers beside 24 agent sessions, on the same three samples of the agents, each run's p95 over that many callers' SLO, the share of the engine's steps in which the budget cap held agent prefill, and the agents' tokens a second over the calls' first 240 s (2 callers, +202 ms over their SLO of 2.03 s at the median of 9 runs (+98 to +267), agent prefill held in 33% of the steps, the agents at 3,156 tokens/s; 4 callers, +269 ms over their SLO of 2.04 s at the median of 3 runs (+206 to +370), agent prefill held in 53% of the steps, the agents at 3,055 tokens/s; 6 callers, +257 ms over their SLO of 2.13 s at the median of 3 runs (+226 to +333), agent prefill held in 64% of the steps, the agents at 2,716 tokens/s).">
</picture>

**The configuration to deploy helps less where the pool has room, and overlapping calls hold the agents' prefill for
longer.** Panel a draws the five rounds beside 16 sessions, and panels b to d two, four and six callers beside 24, each
count's p95 over its own SLO (2.03, 2.04 and 2.13 s).

Where the pool has room the configuration to deploy was lower by 109–1,413 ms, at 6,815 against 7,131 agents' tokens a
second. It met W5's SLO in one round and tuned vLLM in none. Five rounds resolve the direction, not the size, since
without a difference they would agree in sign one time in sixteen.

With 4 and 6 callers beside the 24 sessions, three rounds each, every round was over its SLO. Overlapping replies held
agent prefill in 53% and 64% of the steps, against 33% with two callers on the same samples, and the agents' rate fell
to 3,055 and 2,716 tokens a second, from 3,156.

| Configuration | Callers | V2V p50 / p95 / p99 | p95 range | First byte p95 | Agents' tokens/s, first 240 s | Agent TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 4 | 1,647 / 1,794 / 1,936 | 1,780–1,894 | 66 | — | — |
| **The configuration to deploy** | 4 | 1,993 / 2,313 / 2,512 | 2,250–2,415 | 385 | 3,055 | 78 / 109 s |
| Voice alone | 6 | 1,637 / 1,885 / 2,028 | 1,878–1,885 | 82 | — | — |
| **The configuration to deploy** | 6 | 2,016 / 2,392 / 2,533 | 2,360–2,468 | 379 | 2,716 | 83 / 125 s |

These are W5's runs of 2026-10-25, each count held to voice alone with as many callers plus 250 ms
([W5's record](../benchmarks/results/2026-10-24-e2-w5)).

## What does the change take from the agents?

About 3% of their token rate, through the budget cap's holds rather than the preemptions. At a 48-token window the cap
held agent prefill back in a third of the engine's steps over the calls and make-room preempted about 31 requests a run,
while the agents' TTFT p99 stayed at tuned vLLM's.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-agents-narrow.svg">
  <img src="figures/report-agents.svg" width="100%" alt="For each configuration beside 24 agent sessions, each run and the median of the agents' TTFT p99, of the share of the engine's steps over the calls in which the budget cap kept agent prefill out, and of make-room's own preemptions over the calls: tuned vLLM (the baseline, 10 runs): TTFT p99 106 s (96 to 129), no budget cap, no KV preemption; preempt-resubmit (before the first byte, 5 runs): TTFT p99 193 s (160 to 232), no budget cap, no KV preemption; the KV preemption (#63350, 5 runs): TTFT p99 112 s (98 to 131), agent prefill held in 0% of the steps, 34 preemptions by make-room; the change (16-token window, 5 runs): TTFT p99 105 s (99 to 130), agent prefill held in 25% of the steps, 32 preemptions by make-room; and the configuration to deploy (48-token window, 15 runs): TTFT p99 111 s (94 to 132), agent prefill held in 33% of the steps, 31 preemptions by make-room.">
</picture>

**The agents' first-token tail stays at tuned vLLM's with the configuration to deploy and nearly doubles with
preempt-resubmit.** Each run a tick and each median a marker and a label; tuned vLLM and preempt-resubmit run neither
rule.

| Configuration | Runs | Engine steps over the calls | Steps the cap held agent prefill back in (share) | Prefill tokens held back (an upper bound) | Agents' tokens/s, first 240 s |
|---|---:|---:|---:|---:|---:|
| Tuned vLLM | 10 | 5,778 | — | — | 3,305 |
| Preempt-resubmit (arm 8c) | 5 | 5,390 | — | — | 3,332 |
| The KV preemption | 5 | 5,430 | 0 (0%) | 0 | 3,414 |
| The change, 16-token window | 5 | 5,161 | 1,310 (25%) | 325,689 | 3,298 |
| **The configuration to deploy** | 15 | 5,404 | 1,778 (33%) | 382,720 | 3,203 |
| The same on v0.3's module | 5 | 4,939 | — | — | 3,132 |
| Standard scenario: tuned vLLM | 5 | 10,755 | — | — | 7,131 |
| Standard scenario: the configuration to deploy | 5 | 10,988 | 505 (4%) | 31,520 | 6,815 |
| 4 callers: the configuration to deploy | 3 | 6,349 | 3,321 (53%) | 747,879 | 3,055 |
| 6 callers: the configuration to deploy | 3 | 6,727 | 4,332 (64%) | 1,067,013 | 2,716 |

| Configuration | make-room's preemptions | vLLM's preemptions | Tokens loaded beyond first scheduling |
|---|---:|---:|---:|
| Tuned vLLM | — | 31 | 212,630 |
| Preempt-resubmit (arm 8c) | — | 7 | 273,461 |
| The KV preemption | 34 | 51 | 171,736 |
| The change, 16-token window | 32 | 47 | 117,049 |
| **The configuration to deploy** | 31 | 46 | 117,044 |
| The same on v0.3's module | — | 41 | 123,193 |
| Standard scenario: tuned vLLM | — | 0 | −74 |
| Standard scenario: the configuration to deploy | 1 | 1 | 115 |
| 4 callers: the configuration to deploy | 54 | 65 | 111,679 |
| 6 callers: the configuration to deploy | 64 | 70 | 82,548 |

- **The victims cost little.** The CPU tier's loads beyond the requests' first scheduling, the victims' restores among
  them, are 117,044 tokens with the configuration to deploy, against tuned vLLM's 212,630.
- **Where the pool has room** make-room preempted 1 request a run at the median (0–7) and the cap held agent prefill in
  4% of the steps (3–13%).
- **The held tokens are an upper bound**, since W5's module counted a held chunk against each step's remaining budget.
  The 48-token runs count 215 tokens a held step at the median, about one chunk.

What a resume recomputes is not measured ([E3's counters](EXPERIMENTS.md#e3-the-background-tiers-cost-of-make-room)). E3
read these from W5's points, the medians of each configuration's runs
([W5's record](../benchmarks/results/2026-10-24-e2-w5)).

## How often do running agents leave two calls short of KV?

About 6–8% of the time on the box and 6% on the A100, about half of what the simulator models. Beside tuned vLLM two
calls found free KV under their 2,048 tokens 8.1% of the time on the box and 5.7% on the A100, on the same five samples,
and 6.1% and 6.3% beside the configuration to deploy.

Five rounds re-ran both on the box on 2026-10-26, sampling the engine's free KV every 0.5 s, each round on W5's sample
of the same number, as the A100's core did. Medians of the five rounds, with their range:

| Configuration | Rounds | Running share, the box | Running share, the A100 | V2V p95 on the box, ms | Agents' tokens/s on the box |
|---|---:|---:|---:|---:|---:|
| Tuned vLLM | 5 | 8.1% (7.1–8.9%) | 5.7% (4.2–9.1%) | 9,818 (9,200–11,636) | 3,356 (3,040–3,690) |
| The configuration to deploy | 5 | 6.1% (5.4–7.4%) | 6.3% (2.4–6.9%) | 2,169 (2,143–2,277) | 3,160 (2,903–3,496) |

The simulator places the box at 17% and the A100 at 10% [modeled]. What sets the cards' tails apart is how long a turn
waits once KV is short, 4.9–5.0 s on the box and 1.72–1.75 s on the A100 in the KV case. No turn of the 500 failed
([the record](../benchmarks/results/2026-10-26-e2-rs), whose `running_share.py` prints the table).

## Which step cap should the change run at?

256 tokens a step, without a per-request threshold. Over five rounds beside 24 sessions the configuration to deploy put
44 of 295 replies over the SLO at that cap, against 94 at 512 tokens and fewer in every round, for 7% of the agents'
token rate.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-step-cap-narrow.svg">
  <img src="figures/report-step-cap.svg" width="100%" alt="In panel a, the share of each of five rounds' replies over that round's SLO beside 24 agent sessions, against the requirement of at most 5%, and in panel b each round's agents' tokens a second over the calls' first 240 s with their median: tuned vLLM, 512-token steps, 53, 55, 55, 54, 44 of 59 by round, 261 of 295 in all, at 3,399 agents' tokens a second; the configuration to deploy, 512-token steps, 16, 29, 25, 9, 15 of 59 by round, 94 of 295 in all, at 3,303 agents' tokens a second; the configuration to deploy, 256-token steps, 10, 17, 10, 4, 3 of 59 by round, 44 of 295 in all, at 3,067 agents' tokens a second.">
</picture>

**At 256 tokens a step the configuration to deploy had fewer replies over the SLO than at 512 in every round, and no
configuration met the 5% requirement in any.** (a) Each round's share of its 59 replies over that round's SLO, beside 24
coding-agent sessions; the dashed line is the requirement. (b) The agents' tokens a second over the calls' first 240 s,
each round a tick and the median a marker.

| Configuration | Rounds | V2V p95: median of the rounds / pooled | Replies over the SLO | By round | Agents' tokens/s, first 240 s | Agent TTFT p50 / p99 |
|---|---:|---:|---:|---|---:|---:|
| Tuned vLLM, 512-token steps | 5 | 12,010 / 11,766 | 261 / 295 (88.5%) | 53, 55, 55, 54, 44 | 3,399 | 69 / 109 s |
| The configuration to deploy, 512-token steps (W5's 512/256) | 5 | 2,246 / 2,259 | 94 / 295 (31.9%) | 16, 29, 25, 9, 15 | 3,303 | 78 / 117 s |
| **The configuration to deploy, 256-token steps, no threshold** | 5 | **2,098 / 2,151** | **44 / 295 (14.9%)** | 10, 17, 10, 4, 3 | 3,067 | 85 / 120 s |

Screens on 2026-10-27, one run a setting on one sample, chose the cap. At 256 tokens 9 of 59 replies were over,
against 24 at 512 and 33 at 1,024, and 41 to 58 for stock vLLM at every setting
([the sweep's record](../benchmarks/results/2026-10-27-sweep/README.md)). The five rounds confirmed it on 2026-10-28
([their record](../benchmarks/results/2026-10-28-rounds/README.md), whose `gate.py` prints the table). At 24 sessions
14.9% of the replies were still over, and the shared card meets the requirement at four
([two cards](#which-arrangement-of-two-cards-does-the-most-background-work)).

## What do the serving engine's additions do to a voice turn?

They cut the replies over the SLO by more than half. Inside the engine they bring a call's first token from 175 to 64 ms
at the median, and in the whole pipeline beside 24 sessions they put 16 of 295 replies over the SLO, against 55.

### What does each addition do inside the engine?

The serving profile's engine ran alone on Phastform and on Gray, two calls beside 12 closed-loop coding agents for
240 s. Each addition ran against the point before it, two runs a row unless the table says one, the medians with their
range beneath.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-engine-points-narrow.svg">
  <img src="figures/report-engine-points.svg" width="100%" alt="On the box, two calls beside 12 closed-loop agents, the calls' first token and their 16th token at p50 and p95, each the median of the runs, as the serving profile gained each addition it adopted, each row adding to the one before it but the announced turn's, which adds it to retention, and the next, which adds just in time to both: v1.14's serving profile, 2 runs: the first token 175 and 354 ms, the 16th 408 and 597 ms; + the calls' prefixes kept on the GPU, 2 runs: the first token 122 and 230 ms, the 16th 354 and 502 ms; + the next batch just in time, 2 runs: the first token 98 and 185 ms, the 16th 323 and 449 ms; retention + each turn announced, 2 runs: the first token 64 and 90 ms, the 16th 258 and 394 ms; + just in time too, 1 run: the first token 63 and 98 ms, the 16th 257 and 403 ms; + the agents' reservations taken, 2 runs: the first token 64 and 92 ms, the 16th 261 and 392 ms; + their decodes held for 16 tokens, 2 runs: the first token 64 and 94 ms, the 16th 258 and 285 ms; + held for 48 tokens, the window: the profile, 2 runs: the first token 66 and 88 ms, the 16th 257 and 284 ms.">
</picture>

**Retention, an announced turn and just in time took a call's first token from 175 to 63 ms at the median, and the
decode hold took its 16th token's p95 from 392 to 284 ms.** The serving profile's engine on the box beside 12
closed-loop agents, each addition against the point before it, in the order the profile gained them; the announced
turn's row adds it to retention, and the next adds just in time to both. Each row's filled marker is the p50 and its
open marker the p95 of (a) the calls' first token and (b) their 16th token, each the median of the row's runs. The table
gives both nodes, with each median's range.

| Node | Arm | Runs | First token p50, ms | p95 | 16th token p50 | p95 | Later tokens, ms each | Agents' tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Phastform | v1.14's serving profile (asynchronous scheduling) | 2 | 175<br><sub>170⁠–⁠180</sub> | 354<br><sub>339⁠–⁠368</sub> | 408<br><sub>407⁠–⁠409</sub> | 597<br><sub>594⁠–⁠600</sub> | 13.7 | 2,718 |
| Phastform | + the calls' prefixes kept on the GPU (retention) | 2 | 122<br><sub>118⁠–⁠127</sub> | 230<br><sub>223⁠–⁠236</sub> | 354<br><sub>349⁠–⁠359</sub> | 502<br><sub>495⁠–⁠508</sub> | 13.9 | 2,761 |
| Phastform | + the next batch just in time (30 ms margin) | 2 | 98<br><sub>94⁠–⁠101</sub> | 185<br><sub>170⁠–⁠200</sub> | 323<br><sub>320⁠–⁠326</sub> | 449<br><sub>448⁠–⁠450</sub> | 13.8 | 2,675 |
| Phastform | + each turn announced (250 ms ahead, held 400 ms) | 2 | 64<br><sub>63⁠–⁠65</sub> | 90<br><sub>83⁠–⁠96</sub> | 258<br><sub>256⁠–⁠259</sub> | 394<br><sub>388⁠–⁠401</sub> | 13.7 | 2,637 |
| Phastform | + the turn announced and just in time | 1 | 63 | 98 | 257 | 403 | 13.6 | 2,628 |
| Phastform | + the calls taking the agents' reservations | 2 | 64<br><sub>64⁠–⁠64</sub> | 92<br><sub>91⁠–⁠94</sub> | 261<br><sub>260⁠–⁠262</sub> | 392<br><sub>391⁠–⁠392</sub> | 13.6 | 2,625 |
| Phastform | + the agents' decodes held from a call's first 16 tokens | 2 | 64<br><sub>63⁠–⁠65</sub> | 94<br><sub>92⁠–⁠95</sub> | 258<br><sub>256⁠–⁠261</sub> | 285<br><sub>284⁠–⁠286</sub> | 13.8 | 2,615 |
| Phastform | the same, the decodes held from a call's first 48 tokens | 2 | 66<br><sub>65⁠–⁠66</sub> | 88<br><sub>85⁠–⁠92</sub> | 257<br><sub>257⁠–⁠257</sub> | 284<br><sub>278⁠–⁠291</sub> | 13.4 | 2,604 |
| Gray | v1.14's serving profile (asynchronous scheduling) | 2 | 182<br><sub>179⁠–⁠184</sub> | 308<br><sub>299⁠–⁠316</sub> | 400<br><sub>396⁠–⁠403</sub> | 608<br><sub>591⁠–⁠624</sub> | 13.8 | 2,682 |
| Gray | + the calls' prefixes kept on the GPU (retention) | 2 | 139<br><sub>136⁠–⁠142</sub> | 256<br><sub>254⁠–⁠259</sub> | 374<br><sub>373⁠–⁠376</sub> | 502<br><sub>497⁠–⁠507</sub> | 13.8 | 2,686 |
| Gray | + each turn announced (250 ms ahead, held 400 ms) | 2 | 72<br><sub>71⁠–⁠74</sub> | 110<br><sub>109⁠–⁠110</sub> | 270<br><sub>269⁠–⁠270</sub> | 405<br><sub>404⁠–⁠406</sub> | 13.7 | 2,606 |
| Gray | + the turn announced and just in time | 2 | 72<br><sub>71⁠–⁠72</sub> | 111<br><sub>108⁠–⁠114</sub> | 270<br><sub>270⁠–⁠270</sub> | 438<br><sub>413⁠–⁠462</sub> | 13.5 | 2,584 |
| Gray | + the calls taking the agents' reservations | 2 | 71<br><sub>71⁠–⁠71</sub> | 106<br><sub>96⁠–⁠115</sub> | 268<br><sub>266⁠–⁠269</sub> | 402<br><sub>395⁠–⁠408</sub> | 13.5 | 2,554 |
| Gray | + the agents' decodes held from a call's first 16 tokens | 1 | 72 | 108 | 268 | 300 | 13.8 | 2,569 |
| Gray | the same, the decodes held from a call's first 48 tokens | 2 | 74<br><sub>73⁠–⁠74</sub> | 106<br><sub>103⁠–⁠109</sub> | 267<br><sub>267⁠–⁠267</sub> | 300<br><sub>299⁠–⁠301</sub> | 13.3 | 2,600 |

- **Retention, just in time and an announced turn take the first token** from 175 to 122, 98 and 64 ms at the median on
  Phastform. The announced turn takes the 16th from 408 to 258 ms, at 4% of the agents' token rate.
- **Taking the agents' reservations removes the 1.4–4.5 s stalls** of three Gray points, at no cost
  ([the stalls](../benchmarks/results/2026-11-01-engine-points/README.md#the-stalls)).
- **The decode hold takes the reply's 48th token from 998 to 722 ms at p95**, and its 16th from 392 to 284 ms. In the
  whole pipeline it cost the agents more ([below](#should-the-decode-hold-and-speculative-decoding-join-them)), and the
  serving profile takes it from version 1.17.

[The record](../benchmarks/results/2026-11-01-engine-points/README.md) of 2026-11-01 and 02 also gives the alternatives
not adopted and the controls.

### What do the first four do to the whole pipeline?

They put fewer than half as many replies over the SLO in every scenario, at 1.0–4.9% of the agents' token rate. Five
rounds ran v1.14's configuration to deploy against the same with the first four additions, beside 24 sessions with two,
four and six callers and beside 16 of the whole corpus.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-opt-rounds-narrow.svg">
  <img src="figures/report-opt-rounds.svg" width="100%" alt="In panel a, each scenario's replies over their round's SLO in 5 rounds, v1.14's configuration to deploy and the same with the first four of version 1.15's additions to the engine, each round a tick and the rounds pooled a marker, against the requirement of at most 5%, and in panel b the agents' tokens a second over the calls' first 240 s, each round a tick and their median a marker: 24 agent sessions, 2 callers, v1.14's configuration to deploy 10, 13, 10, 6, 16 by round, 55 of 295 in all, at 3,010 agents' tokens a second, the same with the first four additions 7, 1, 4, 4, 0 by round, 16 of 295 in all, at 2,933 agents' tokens a second; 24 agent sessions, 4 callers, v1.14's configuration to deploy 19, 25, 29, 12, 25 by round, 110 of 595 in all, at 2,753 agents' tokens a second, the same with the first four additions 5, 10, 9, 6, 8 by round, 38 of 595 in all, at 2,754 agents' tokens a second; 24 agent sessions, 6 callers, v1.14's configuration to deploy 23, 56, 27, 29, 29 by round, 164 of 910 in all, at 2,580 agents' tokens a second, the same with the first four additions 3, 20, 10, 9, 11 by round, 53 of 910 in all, at 2,476 agents' tokens a second; 16 agent sessions of the whole corpus, 2 callers, v1.14's configuration to deploy 5, 3, 3, 9, 2 by round, 22 of 295 in all, at 4,948 agents' tokens a second, the same with the first four additions 2, 2, 1, 5, 0 by round, 10 of 295 in all, at 4,900 agents' tokens a second.">
</picture>

**In every scenario the first four additions put fewer than half as many of the pooled replies over the SLO, and beside
16 sessions they met the 5% requirement.** (a) Each round's share of its replies over that round's SLO and the five
rounds pooled; the dashed line is the requirement. (b) The agents' tokens a second over the calls' first 240 s, each
round a tick and the median a marker.

The medians over the rounds, with their range beneath:

| Scenario | Configuration | Rounds | Replies over the SLO | V2V p50, ms | V2V p95 | The SLO |
|---|---|---:|---:|---:|---:|---:|
| 24 agent sessions, 2 callers | v1.14 | 5 | 55 of 295 (18.6%) | 1,894<br><sub>1,853⁠–⁠1,934</sub> | 2,176<br><sub>2,066⁠–⁠2,215</sub> | 2,010<br><sub>1,976⁠–⁠2,051</sub> |
| 24 agent sessions, 2 callers | The first four additions | 5 | 16 of 295 (5.4%) | 1,712<br><sub>1,692⁠–⁠1,765</sub> | 2,023<br><sub>1,927⁠–⁠2,139</sub> | 2,010<br><sub>1,976⁠–⁠2,051</sub> |
| 24 agent sessions, 4 callers | v1.14 | 5 | 110 of 595 (18.5%) | 1,897<br><sub>1,849⁠–⁠1,934</sub> | 2,274<br><sub>2,126⁠–⁠2,299</sub> | 2,065<br><sub>2,046⁠–⁠2,142</sub> |
| 24 agent sessions, 4 callers | The first four additions | 5 | 38 of 595 (6.4%) | 1,744<br><sub>1,709⁠–⁠1,800</sub> | 2,106<br><sub>2,015⁠–⁠2,168</sub> | 2,065<br><sub>2,046⁠–⁠2,142</sub> |
| 24 agent sessions, 6 callers | v1.14 | 5 | 164 of 910 (18.0%) | 1,913<br><sub>1,890⁠–⁠1,944</sub> | 2,320<br><sub>2,260⁠–⁠2,361</sub> | 2,119<br><sub>2,047⁠–⁠2,174</sub> |
| 24 agent sessions, 6 callers | The first four additions | 5 | 53 of 910 (5.8%) | 1,754<br><sub>1,729⁠–⁠1,778</sub> | 2,152<br><sub>2,062⁠–⁠2,172</sub> | 2,119<br><sub>2,047⁠–⁠2,174</sub> |
| 16 agent sessions of the whole corpus, 2 callers | v1.14 | 5 | 22 of 295 (7.5%) | 1,661<br><sub>1,635⁠–⁠1,801</sub> | 2,012<br><sub>1,931⁠–⁠2,322</sub> | 2,010<br><sub>1,976⁠–⁠2,051</sub> |
| 16 agent sessions of the whole corpus, 2 callers | The first four additions | 5 | 10 of 295 (3.4%) | 1,675<br><sub>1,641⁠–⁠1,708</sub> | 1,916<br><sub>1,820⁠–⁠2,049</sub> | 2,010<br><sub>1,976⁠–⁠2,051</sub> |

| Scenario | Configuration | LLM first token p50, ms | p95 | First sentence p50 | p95 | E2E p95 |
|---|---|---:|---:|---:|---:|---:|
| 24 agent sessions, 2 callers | v1.14 | 220<br><sub>204⁠–⁠237</sub> | 314<br><sub>290⁠–⁠345</sub> | 464<br><sub>443⁠–⁠484</sub> | 706<br><sub>653⁠–⁠718</sub> | 1,783<br><sub>1,680⁠–⁠1,815</sub> |
| 24 agent sessions, 2 callers | The first four additions | 94<br><sub>86⁠–⁠102</sub> | 136<br><sub>124⁠–⁠139</sub> | 327<br><sub>307⁠–⁠332</sub> | 560<br><sub>470⁠–⁠623</sub> | 1,643<br><sub>1,526⁠–⁠1,720</sub> |
| 24 agent sessions, 4 callers | v1.14 | 210<br><sub>198⁠–⁠225</sub> | 334<br><sub>329⁠–⁠340</sub> | 456<br><sub>430⁠–⁠487</sub> | 787<br><sub>608⁠–⁠816</sub> | 1,898<br><sub>1,707⁠–⁠1,965</sub> |
| 24 agent sessions, 4 callers | The first four additions | 95<br><sub>92⁠–⁠103</sub> | 139<br><sub>128⁠–⁠145</sub> | 324<br><sub>308⁠–⁠368</sub> | 580<br><sub>505⁠–⁠672</sub> | 1,740<br><sub>1,585⁠–⁠1,785</sub> |
| 24 agent sessions, 6 callers | v1.14 | 206<br><sub>199⁠–⁠232</sub> | 325<br><sub>312⁠–⁠346</sub> | 437<br><sub>425⁠–⁠496</sub> | 737<br><sub>686⁠–⁠861</sub> | 1,939<br><sub>1,889⁠–⁠1,953</sub> |
| 24 agent sessions, 6 callers | The first four additions | 95<br><sub>88⁠–⁠99</sub> | 139<br><sub>129⁠–⁠166</sub> | 316<br><sub>298⁠–⁠339</sub> | 557<br><sub>535⁠–⁠593</sub> | 1,692<br><sub>1,684⁠–⁠1,753</sub> |
| 16 agent sessions of the whole corpus, 2 callers | v1.14 | 102<br><sub>88⁠–⁠150</sub> | 255<br><sub>188⁠–⁠303</sub> | 329<br><sub>302⁠–⁠402</sub> | 570<br><sub>494⁠–⁠822</sub> | 1,635<br><sub>1,534⁠–⁠1,959</sub> |
| 16 agent sessions of the whole corpus, 2 callers | The first four additions | 90<br><sub>84⁠–⁠94</sub> | 125<br><sub>112⁠–⁠145</sub> | 305<br><sub>272⁠–⁠337</sub> | 458<br><sub>438⁠–⁠627</sub> | 1,517<br><sub>1,496⁠–⁠1,684</sub> |

| Scenario | Configuration | Agents' tokens/s | Against v1.14 | Agents' TTFT p50, s | p99 |
|---|---|---:|---:|---:|---:|
| 24 agent sessions, 2 callers | v1.14 | 3,010<br><sub>2,720⁠–⁠3,275</sub> | — | 100.5<br><sub>74.9⁠–⁠116.4</sub> | 125.2<br><sub>112.2⁠–⁠138.7</sub> |
| 24 agent sessions, 2 callers | The first four additions | 2,933<br><sub>2,692⁠–⁠3,221</sub> | −1.6%<br><sub>−4.3% to −0.5%</sub> | 105.0<br><sub>74.5⁠–⁠120.2</sub> | 125.9<br><sub>111.7⁠–⁠141.9</sub> |
| 24 agent sessions, 4 callers | v1.14 | 2,753<br><sub>2,548⁠–⁠3,062</sub> | — | 112.2<br><sub>79.8⁠–⁠130.0</sub> | 130.7<br><sub>116.0⁠–⁠147.8</sub> |
| 24 agent sessions, 4 callers | The first four additions | 2,754<br><sub>2,508⁠–⁠2,947</sub> | −3.7%<br><sub>−4.3% to +0.0%</sub> | 121.6<br><sub>87.0⁠–⁠135.4</sub> | 139.8<br><sub>117.1⁠–⁠153.4</sub> |
| 24 agent sessions, 6 callers | v1.14 | 2,580<br><sub>2,415⁠–⁠2,888</sub> | — | 124.7<br><sub>88.8⁠–⁠137.6</sub> | 142.1<br><sub>118.6⁠–⁠154.0</sub> |
| 24 agent sessions, 6 callers | The first four additions | 2,476<br><sub>2,215⁠–⁠2,780</sub> | −4.9%<br><sub>−8.3% to −2.1%</sub> | 131.8<br><sub>95.5⁠–⁠145.0</sub> | 148.3<br><sub>129.6⁠–⁠160.5</sub> |
| 16 agent sessions of the whole corpus, 2 callers | v1.14 | 4,948<br><sub>4,633⁠–⁠8,364</sub> | — | 0.1<br><sub>0.1⁠–⁠0.2</sub> | 6.6<br><sub>1.2⁠–⁠7.7</sub> |
| 16 agent sessions of the whole corpus, 2 callers | The first four additions | 4,900<br><sub>4,666⁠–⁠9,034</sub> | −1.0%<br><sub>−2.3% to +8.0%</sub> | 0.1<br><sub>0.1⁠–⁠0.2</sub> | 7.3<br><sub>2.9⁠–⁠8.2</sub> |

- **Fewer replies over the SLO in every round of every scenario**, 20 of 20. Beside 24 sessions the additions met the 5%
  allowance in 5 of the 15 rounds and caller counts, v1.14 in none.
- **The LLM's part of the turn comes sooner.** Beside 24 sessions its first token's p95 falls from 314–334 ms to
  136–139 ms, and its first sentence's from 706–787 ms to 557–580 ms.

The rounds ran on 2026-11-02 ([their record](../benchmarks/results/2026-11-02-opt-rounds/README.md), whose `table.py`
prints the tables).

### Should the decode hold and speculative decoding join them?

Not by the rules stated before the rounds, though the decode hold cut the replies over the SLO by more than two thirds.
Two rounds set the decode hold at 48 tokens, and speculative decoding on top of it, against the first four additions.

| Scenario | Configuration | Rounds | Replies over the SLO | V2V p95, ms | LLM first token p95 | First sentence p50 | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 24 agent sessions, 2 callers | The first four additions | 2 | 11 of 118 (9.3%) | 2,048<br><sub>2,034⁠–⁠2,061</sub> | 127<br><sub>118⁠–⁠135</sub> | 340<br><sub>302⁠–⁠378</sub> | 596<br><sub>559⁠–⁠634</sub> |
| 24 agent sessions, 2 callers | + the decode hold | 2 | 6 of 118 (5.1%) | 1,939<br><sub>1,907⁠–⁠1,972</sub> | 130<br><sub>125⁠–⁠135</sub> | 311<br><sub>309⁠–⁠312</sub> | 455<br><sub>449⁠–⁠461</sub> |
| 24 agent sessions, 2 callers | + the decode hold and speculation | 2 | 1 of 118 (0.8%) | 1,828<br><sub>1,825⁠–⁠1,830</sub> | 168<br><sub>155⁠–⁠181</sub> | 225<br><sub>222⁠–⁠228</sub> | 345<br><sub>337⁠–⁠352</sub> |
| 24 agent sessions, 4 callers | The first four additions | 2 | 15 of 238 (6.3%) | 2,157<br><sub>2,143⁠–⁠2,172</sub> | 142<br><sub>129⁠–⁠155</sub> | 331<br><sub>323⁠–⁠340</sub> | 606<br><sub>580⁠–⁠631</sub> |
| 24 agent sessions, 4 callers | + the decode hold | 2 | 2 of 238 (0.8%) | 1,955<br><sub>1,879⁠–⁠2,030</sub> | 138<br><sub>138⁠–⁠139</sub> | 304<br><sub>292⁠–⁠316</sub> | 433<br><sub>419⁠–⁠447</sub> |
| 24 agent sessions, 4 callers | + the decode hold and speculation | 2 | 4 of 238 (1.7%) | 1,915<br><sub>1,892⁠–⁠1,938</sub> | 177<br><sub>167⁠–⁠186</sub> | 237<br><sub>236⁠–⁠238</sub> | 377<br><sub>372⁠–⁠382</sub> |
| 24 agent sessions, 6 callers | The first four additions | 2 | 29 of 364 (8.0%) | 2,156<br><sub>2,118⁠–⁠2,195</sub> | 141<br><sub>138⁠–⁠145</sub> | 323<br><sub>319⁠–⁠327</sub> | 641<br><sub>623⁠–⁠658</sub> |
| 24 agent sessions, 6 callers | + the decode hold | 2 | 8 of 364 (2.2%) | 2,000<br><sub>1,985⁠–⁠2,016</sub> | 136<br><sub>134⁠–⁠138</sub> | 306<br><sub>302⁠–⁠310</sub> | 459<br><sub>454⁠–⁠464</sub> |
| 24 agent sessions, 6 callers | + the decode hold and speculation | 2 | 7 of 364 (1.9%) | 1,991<br><sub>1,980⁠–⁠2,002</sub> | 172<br><sub>161⁠–⁠184</sub> | 235<br><sub>230⁠–⁠240</sub> | 371<br><sub>357⁠–⁠385</sub> |

| Scenario | Configuration | Agents' tokens/s | Against the first four additions |
|---|---|---:|---:|
| 24 agent sessions, 2 callers | The first four additions | 3,034<br><sub>2,838⁠–⁠3,231</sub> | — |
| 24 agent sessions, 2 callers | + the decode hold | 2,897<br><sub>2,770⁠–⁠3,024</sub> | −4.4%<br><sub>−6.4% to −2.4%</sub> |
| 24 agent sessions, 2 callers | + the decode hold and speculation | 2,758<br><sub>2,566⁠–⁠2,951</sub> | −9.1%<br><sub>−9.6% to −8.7%</sub> |
| 24 agent sessions, 4 callers | The first four additions | 2,710<br><sub>2,500⁠–⁠2,919</sub> | — |
| 24 agent sessions, 4 callers | + the decode hold | 2,609<br><sub>2,385⁠–⁠2,834</sub> | −3.8%<br><sub>−4.6% to −2.9%</sub> |
| 24 agent sessions, 4 callers | + the decode hold and speculation | 2,568<br><sub>2,343⁠–⁠2,793</sub> | −5.3%<br><sub>−6.3% to −4.3%</sub> |
| 24 agent sessions, 6 callers | The first four additions | 2,464<br><sub>2,294⁠–⁠2,633</sub> | — |
| 24 agent sessions, 6 callers | + the decode hold | 2,390<br><sub>2,272⁠–⁠2,507</sub> | −2.9%<br><sub>−4.8% to −1.0%</sub> |
| 24 agent sessions, 6 callers | + the decode hold and speculation | 2,328<br><sub>2,189⁠–⁠2,467</sub> | −5.4%<br><sub>−6.3% to −4.6%</sub> |

- **The decode hold put 16 of 720 replies over the SLO against 55**, fewer in five of the six rounds and caller counts
  and as many in the sixth. It came at 3.8% of the agents' token rate at the median of the six.
- **Speculative decoding on top put 12 over against the decode hold's 16**, fewer in three of the six, its first
  token's p95 168–177 ms against 130–138.
- **The rules turned both down.** The decode hold was to cost the agents at most 3%, and speculative decoding was to
  beat it in five of the six. The serving profile takes the decode hold all the same from version 1.17, for the voice
  tail, and speculative decoding stays a switch.

The two rounds ran on 2026-11-02 ([their record](../benchmarks/results/2026-11-02-opt-rounds/README.md)).

## How many agents' sessions does one card take beside both calls?

Twenty-four, the most measured, once the agents' queue is ordered by the prefill each has left. With that queue and the
SM copy, 3 of 177 pooled replies were over the SLO beside 24 sessions, within the 5% requirement, at 2.2 times the
agents' token rate of the serving profile's envelope of four. The agents' TTFT p99 of 133 s carries the cost.

Beside 24 sessions, first come first served over the LRU CPU tier computes 89.5% of the prompt tokens, the tier's cyclic
worst case. Version 1.24's serve module orders the agents' queue by the prefill each has left, net of what the pool and
tier hold, any of them first after 120 s. It loads the tier through vLLM's SM copy kernel, because on these PCIe 3.0 x8
links a copy-engine load holds a concurrent step's inputs. Three windows on Gray measured each point against its
window's control with as many callers, on one sample of the agents.

**The first window, 2026-11-07.**

| Point | Agents' tokens/s | Against the control | Computed | Loaded from the CPU tier, GB | Agents' TTFT p50, s | Calls' first byte p95, ms | LLM first sentence p95 | V2V p50 | V2V p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.23, the control | 3,173 | — | 89.5% | 2.2 | 76.8 | 202 | 403 | 1,711 | 1,843 |
| a 30 GiB CPU tier | 2,958 | −7% | 68.0% | 37.7 | 76.7 | 255 | 456 | 1,700 | 1,840 |
| the queue, every count refreshed each step | 7,144 | +125% | 23.5% | 274.2 | 4.7 | 301 | 630 | 1,787 | 2,051 |
| the queue, two counts a step, none beside a call | 7,270 | +129% | 20.2% | 306.5 | 4.5 | 324 | 471 | 1,664 | 2,025 |
| the queue (every count) and a 30 GiB tier | 8,622 | +172% | 11.7% | 368.0 | 4.6 | 405 | 825 | 1,762 | 2,226 |

**The second window, 2026-11-08.**

| Point | Agents' tokens/s | Against the control | Computed | Loaded from the CPU tier, GB | Agents' TTFT p50, s | Calls' first byte p95, ms | LLM first sentence p95 | V2V p50 | V2V p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.23, the control | 3,089 | — | 89.8% | 2.0 | 76.2 | 239 | 437 | 1,694 | 1,835 |
| the queue and the SM copy | 7,849 | +154% | 20.3% | 324.1 | 4.6 | 183 | 454 | 1,742 | 1,924 |
| the queue, the SM copy and a 30 GiB tier | 9,537 | +209% | 12.4% | 399.7 | 3.6 | 200 | 506 | 1,737 | 1,994 |

**The third window, 2026-11-08.**

| Point | Agents' tokens/s | Against the control | Computed | Loaded from the CPU tier, GB | Agents' TTFT p50, s | Calls' first byte p95, ms | LLM first sentence p95 | V2V p50 | V2V p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.23, the control | 3,113 | — | 89.9% | 2.4 | 75.8 | 232 | 408 | 1,669 | 1,915 |
| the queue and the SM copy | 7,777 | +150% | 21.3% | 286.4 | 4.1 | 211 | 462 | 1,700 | 1,927 |
| endpointing 0.2/0.65 s alone (not in the profile) | 2,932 | −6% | 88.8% | 2.4 | 78.0 | 225 | 477 | 1,342 | 1,615 |
| the queue, the SM copy and endpointing 0.2/0.65 s | 7,912 | +154% | 22.6% | 303.2 | 3.7 | 179 | 435 | 1,343 | 1,582 |
| v1.23, 4 callers | 2,643 | — | 84.0% | 3.4 | 84.9 | 234 | 516 | 1,723 | 2,016 |
| the queue, the SM copy and endpointing 0.2/0.65 s, 4 callers | 5,876 | +122% | 24.6% | 252.2 | 5.7 | 247 | 479 | 1,385 | 1,649 |

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-tier-narrow.svg">
  <img src="figures/report-tier.svg" width="100%" alt="In panel a, where vLLM found the prompt tokens at their first schedule beside 24 agent sessions on the second GPU node, as shares of them, with the agents' tokens a second over the calls' first 240 s and their median time to first token: first come first served over a 20 GiB CPU tier, 9.1% found in the GPU pool, 1.5% loaded from the CPU tier and 89.5% computed, at 3,173 agents' tokens a second and a median time to first token of 76.8 s; first come first served over a 30 GiB CPU tier, 8.3%, 23.7% and 68.0%, at 2,958 and 76.7 s; and the queue over a 20 GiB CPU tier, 5.9%, 73.9% and 20.2%, at 7,270 and 4.5 s; in panel b, a step's time beside one 1.46 GB load from the CPU tier on that node's bare card, 5.7 ms alone at the median of the steps before each load, and the longest during it 192 ms on the copy engine in one batch of 32 KiB pages, 209 ms in one of 2 MiB pages, 152 ms in 23 jobs of 64 MB, 136 ms in 92 jobs of 16 MB on chained streams, and 6.1 ms through vLLM's Triton kernel with 4 programs.">
</picture>

**First come first served computed most of the prompt tokens and the queue a fifth, and beside a load from the CPU tier
a step took up to 209 ms on the copy engine and 6.1 ms through the SM copy.** (a) Where vLLM found the prompt tokens in
the second GPU node's first window. (b) A step's time beside one 1.46 GB load from the tier on that node's bare card, in
32 KiB pages unless marked. Alone is the median of the ten steps before each load, the copy engine the longest of the
two (one batch) or four (chained jobs) steps during it, and vLLM's Triton kernel the longest of 22.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-envelope-narrow.svg">
  <img src="figures/report-envelope.svg" width="100%" alt="In panel a, each configuration's replies over their round's SLO in version 1.24's 3 envelope rounds, each round a tick and the rounds pooled a marker, against the requirement of at most 5%, in panel b the agents' tokens a second over the calls' first 240 s and in panel c their time to first token at p99, each round a tick and their median a marker: the serving profile through v1.23 beside 4 agent sessions, 0, 2, 2 by round, 4 of 177 in all, its agents at 3,278 tokens a second (2,703 to 3,598) and a p99 of 8 s (6 to 8); the queue and the copy beside 12 agent sessions, 2, 2, 0 by round, 4 of 177 in all, its agents at 6,906 tokens a second (5,949 to 7,090) and a p99 of 24 s (19 to 73); and beside 24, 0, 3, 0 by round, 3 of 177 in all, its agents at 7,253 tokens a second (4,490 to 8,094) and a p99 of 133 s (132 to 182).">
</picture>

**Beside 24 sessions the queue and the copy kept the pooled replies over the SLO within the 5% requirement, at 2.2 times
the agents' token rate of the serving profile at 4, and the agents' first-token tail carries the cost.** (a) Each
round's share of its replies over that round's SLO, voice alone's p95 plus 250 ms, and the three rounds pooled; the
dashed line is the requirement. (b) The agents' tokens a second over the calls' first 240 s and (c) their time to first
token at p99, each round a tick and the median a marker.

Version 1.24's three envelope rounds, on samples 21–23, applied the resource gate's rule, under which a configuration
qualifies when at most 5% of its replies, pooled over the rounds, are over each round's SLO. The medians over the
rounds, with their range beneath:

| Configuration | Agent sessions | Rounds | Replies over the SLO | Qualifies | V2V p50, ms | V2V p95 | Agents' tokens/s | Agents' TTFT p50, s | p99 | Computed |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| The serving profile through v1.23 | 4 | 3 | 4 of 177 (2.3%) | yes | 1,635<br><sub>1,618⁠–⁠1,642</sub> | 1,934<br><sub>1,852⁠–⁠2,008</sub> | 3,278<br><sub>2,703⁠–⁠3,598</sub> | 0.3<br><sub>0.3⁠–⁠0.7</sub> | 8<br><sub>6⁠–⁠8</sub> | 2.4%<br><sub>2.0%⁠–⁠3.1%</sub> |
| The queue and the SM copy | 12 | 3 | 4 of 177 (2.3%) | yes | 1,753<br><sub>1,659⁠–⁠1,765</sub> | 1,905<br><sub>1,876⁠–⁠1,959</sub> | 6,906<br><sub>5,949⁠–⁠7,090</sub> | 2.3<br><sub>0.5⁠–⁠2.9</sub> | 24<br><sub>19⁠–⁠73</sub> | 14.2%<br><sub>13.1%⁠–⁠14.7%</sub> |
| The queue and the SM copy | 24 | 3 | 3 of 177 (1.7%) | yes | 1,690<br><sub>1,674⁠–⁠1,712</sub> | 1,872<br><sub>1,865⁠–⁠1,887</sub> | 7,253<br><sub>4,490⁠–⁠8,094</sub> | 4.3<br><sub>3.8⁠–⁠4.5</sub> | 133<br><sub>132⁠–⁠182</sub> | 26.1%<br><sub>22.8%⁠–⁠42.7%</sub> |

The profile's `max_agents` is 24 from version 1.24. At 24 sessions a request can wait out the queue's 120 s bound, hence
the agents' p99 of 133 s. [The record](../benchmarks/results/2026-11-08-envelope/README.md)'s `table.py` prints both
tables, and it holds the copy probes and what the windows measured and the profile did not adopt: a 30 GiB tier alone,
shorter endpointing, Parakeet in place of Whisper and flatshare_jit's early finish.

## Which arrangement of two cards does the most background work?

Consolidating both calls on one card, on either engine. With the change the shared card qualified at four sessions, 2.8%
of its replies over the SLO, and the pair did 10,596 agents' tokens a second, 34.2% more than the best dedicated
arrangement and past the gate's 20% hurdle. Most of that is the arrangement's, since tuned vLLM's shared card qualified
at two sessions and did 27.3% more.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-cards-narrow.svg">
  <img src="figures/report-cards.svg" width="100%" alt="In panel a, the share of the consolidated card's replies over their round's SLO, the 3 rounds pooled, by its agent sessions beside both calls, against the requirement of at most 5%: at 2, 4, 8, 12 and 24 sessions, 2.3%, 10.2%, 39.0%, 53.7% and 92.1% on tuned vLLM and 1.1%, 2.8%, 17.5%, 18.6% and 20.9% on the configuration to deploy; in panel b, each arrangement's agents' tokens a second over the calls' first 240 s summed over both cards, at the median of the 3 rounds, against 9,471, 20% over the best dedicated arrangement: dedicated roles, best practical, 0 + 14 sessions: 7,893; dedicated roles, controlled, 0 + 12 sessions: 6,302; consolidation on tuned vLLM, 2 + 14 sessions: 10,051; consolidation on the configuration to deploy, 4 + 14 sessions: 10,596; consolidation on the configuration to deploy with a warm spare, 4 + 12 sessions: 9,652; consolidation on the configuration to deploy, 8 + 14 sessions: 13,537, missing the requirement; pooling on tuned vLLM, 12 + 12 sessions: 13,188, missing the requirement; pooling on the configuration to deploy, 12 + 12 sessions: 12,111, missing the requirement.">
</picture>

**On two cards, consolidating both calls on one card does the most background work within the SLO, on either engine.**
(a) The consolidated card's replies over the SLO by its agent sessions, three rounds pooled, against the 5% requirement
(dashed). (b) The agents' tokens a second summed over both cards, bars the median of the rounds and ticks the rounds;
hatched arrangements miss the requirement, and the dashed line is 20% over the best dedicated one.

The resource gate's comparison ([RESOURCE-GATE.md](../benchmarks/quality/RESOURCE-GATE.md)) ran in three rounds on
2026-10-28, each card's part of an arrangement a point of its own, the change at the serving profile's settings. Replies
over each round's SLO are pooled over the rounds, and the agents' tokens a second are summed over the cards, the median
of the rounds with each round's sum.

| Arrangement | Sessions (shared + second card) | Replies over the SLO | Agents' tokens/s, first 240 s | Rounds | Agent TTFT p50 / p99 |
|---|---:|---:|---:|---|---:|
| Dedicated roles, controlled (the second card tuned vLLM, its speech servers idle) | 0 + 12 | 0 / 177 | 6,302 | 7,008, 6,302, 6,041 | 4.4 / 31.0 s |
| **Dedicated roles, best practical** (the second card agents alone: a 61,152-token pool, FCFS, 2,048-token steps) | 0 + 14 | 0 / 177 | **7,893** | 8,282, 7,893, 4,710 | 5.1 / 41.0 s |
| Pooling, tuned vLLM | 12 + 12 | 53 / 87 | 13,188 | 13,233, 13,188, 11,951 | 4.7 / 29.2 s |
| Pooling, the change | 12 + 12 | 24 / 87 | 12,111 | 13,138, 12,111, 11,374 | 4.8 / 37.2 s |
| Consolidation, tuned vLLM | 24 + 14 | 163 / 177 | 11,378 | 11,681, 11,378, 7,776 | 39.7 / 85.8 s |
| Consolidation, tuned vLLM | 12 + 14 | 95 / 177 | 14,191 | 14,987, 14,191, 11,055 | 4.6 / 33.5 s |
| Consolidation, tuned vLLM | 8 + 14 | 69 / 177 | 13,573 | 13,573, 14,267, 10,547 | 0.7 / 11.6 s |
| Consolidation, tuned vLLM | 4 + 14 | 18 / 177 | 10,596 | 11,655, 10,596, 8,308 | 0.6 / 10.4 s |
| Consolidation, tuned vLLM | 2 + 14 | 4 / 177 | 10,051 | 10,984, 10,051, 6,234 | 0.4 / 7.8 s |
| Consolidation, the change | 24 + 14 | 37 / 177 | 11,075 | 11,466, 11,075, 7,435 | 48.0 / 92.5 s |
| Consolidation, the change | 12 + 14 | 33 / 177 | 13,880 | 15,107, 13,880, 10,082 | 5.1 / 41.6 s |
| Consolidation, the change | 8 + 14 | 31 / 177 | 13,537 | 13,537, 14,352, 10,392 | 0.7 / 11.7 s |
| **Consolidation, the change** | 4 + 14 | 5 / 177 | **10,596** | 11,655, 10,596, 8,308 | 0.6 / 10.8 s |
| Consolidation, the change | 2 + 14 | 2 / 177 | 10,050 | 10,982, 10,050, 6,213 | 0.4 / 9.8 s |
| Consolidation, the change at W5's configuration | 24 + 14 | 70 / 177 | 11,196 | 11,596, 11,196, 7,553 | 44.4 / 88.0 s |

- **The change itself adds 5.4%**, its shared card at four sessions against tuned vLLM's at two, short of the hurdle.
- **A wider margin would change the verdict.** With 500 ms over voice alone in place of 250, the change's shared card
  would qualify at every count measured, and tuned vLLM's still only at two.
- **A warm spare keeps 22.3% over the best dedicated arrangement**, the pair doing 9,652 tokens a second.

[The record](../benchmarks/results/2026-10-28-rounds/README.md)'s `gate.py` prints the table, the margins and the
stages.

## Does the arrangement to deploy keep its SLO over a day?

Yes. Over 24 hours 74 of 5,058 replies (1.5%) were over the SLO of 1,976 ms, inside the 5% the contract allows, and no
call or turn failed. Only hour 4 went over 5%, with 14 of 254.

The soak served [the arrangement to deploy](PRODUCTION.md#two-cards) on 2026-10-28 and 29, both calls on Phastform and
Gray a warm spare, with calls arriving at 20 an hour beside 14 agent sessions and every attempt counted
([its design](EXPERIMENTS.md#the-soak-the-production-path-for-hours)). Its SLO is voice alone's p95 over two hours of
the same calls, plus 250 ms.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-soak-narrow.svg">
  <img src="figures/report-soak.svg" width="100%" alt="Each of the soak's 24 hours as a column: in panel a, the share of that hour's replies over the SLO of 1,976 ms against the requirement of at most 5%, 74 of 5,058 replies (1.5%) over the day, at most 14 of 254 (5.5%) in hour 4; in panel b, the agents' tokens a second, 3,529 over the day, 9,300 in hour 0 and 582 to 5,402 in each hour after.">
</picture>

**Over the 24 hours 1.5% of the replies were over the SLO, and only hour 4, at 5.5%, went over the 5% requirement.** (a)
Each hour's share of its replies over the SLO. (b) The agents' tokens a second by hour, with 9,300 in hour 0, as AIPerf
resumes its 14 sessions at once.

| Measure | Voice alone | The arrangement to deploy |
|---|---:|---:|
| Hours | 2 | 24 |
| Calls (accepted, refused, failed) | 35 (31, 4, 0) | 476 (402, 74, 0) |
| Turns (failed) | 310 (0) | 4,020 (0) |
| Replies | 395 | 5,058 |
| Over the SLO | the reference | 74 (1.5%) |
| Replies p95, ms | 1,726 | 1,828 |
| Realtime SLI | 1.000 | 0.996 |
| Background tokens/s | 0 | 3,529 |
| Restarts | 0 | 0 |
| Guard pauses | 0 | 0 |

- **No call failed.** Of 476 calls the worker answered 402, and 74 found no agent, 72 of them with both lines busy. The
  other two came 1.0 and 1.2 s after a call ended, while LiveKit still held the worker full, since LiveKit Agents 1.8.8
  reports a worker's status only every 2.5 s.
- **Nothing restarted, paused or went unread**, and Gray carried no call.
- **The agents ran at their traces' pace**, 3,529 tokens a second over the day. The rounds' 240 s windows sit in the
  replay's opening burst, so their agents' rates are its first minutes, not a day's.

[The soak's record](../benchmarks/results/2026-10-28-soak/README.md)'s `table.py` prints the table.

## Do the rules carry to a datacenter card?

Yes. On an A100 SXM4 80 GB at the box's pool and step cap, beside the same 24 sessions, the configuration to deploy met
the card's SLO of 1,669 ms in all five rounds. It gave 1,634 ms at the median against tuned vLLM's 4,986. What does not
carry is the length of a wait, 1.75 s in the KV case against the box's 4.9–5.0 s.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-a100-narrow.svg">
  <img src="figures/report-a100.svg" width="100%" alt="On the A100 beside 24 agent sessions: in panel a, at the RTX 5060 Ti's 512-token steps, tuned vLLM's V2V p95 and the configuration to deploy's in each of five rounds against the card's SLO of 1.67 s (4.16 and 1.66 s, 4.78 and 1.66 s, 4.99 and 1.61 s, 5.17 and 1.63 s, 5.71 and 1.61 s); in panel b, the agents' tokens a second over the calls' first 240 s against the V2V p95, each run and the medians, an arrow from tuned vLLM's median to the configuration to deploy's at the same step cap (at 512-token steps tuned vLLM 4.99 s at 8,633 tokens/s, the configuration to deploy 1.63 s at 8,651 tokens/s, medians of 5 runs each; at 2,048-token steps tuned vLLM 3.66 s at 14,536 tokens/s, the configuration to deploy 1.83 s at 14,635 tokens/s, medians of 2 runs each).">
</picture>

**On an A100 the rules meet the card's SLO in every round at the RTX 5060 Ti's 512-token steps, and at 2,048-token steps
they trade 0.19 s of the voice's tail for 69% more agent throughput.** (a) Each round's p95 V2V at 512-token steps,
joined by round, against the card's SLO of 1.67 s. (b) The p95 V2V against the agents' tokens a second at 512- and
2,048-token steps, small marks runs and large marks medians, an arrow joining tuned vLLM's median to the configuration
to deploy's at each step cap; tuned vLLM's marks are open.

The A100 ran E2's core like for like on 2026-10-26, with the box's model, pool, steps and CPU tier and W5's arm files
and samples. Only the card, its precision path and the host changed. The card has no FP8 units, so it runs the
checkpoint through Marlin's W8A16 kernels. V2V p95 in ms and the agents' tokens a second, each round on W5's sample of
its number:

| Agents' sample (seed) | Voice alone | Tuned vLLM, the box's 512/256 | The configuration to deploy | Its tokens/s, of tuned vLLM's |
|---|---:|---:|---:|---:|
| 11 | 1,445 | 4,163 / 10,749 | 1,663 / 10,125 | 94% |
| 12 | 1,414 | 4,783 / 9,820 | 1,658 / 10,851 | 111% |
| 13 | 1,419 | 4,986 / 8,398 | 1,611 / 8,025 | 96% |
| 14 | 1,391 | 5,169 / 8,633 | 1,634 / 8,651 | 100% |
| 15 | 1,424 | 5,710 / 8,312 | 1,609 / 7,922 | 95% |
| Median | 1,419 | 4,986 / 8,633 | 1,634 / 8,651 | 96% |

- **Every round was under the SLO**, by 7–60 ms, where tuned vLLM was 2.5–4.0 s over, and no turn of the 750 failed.
  With 32 sessions in one round, A', the configuration to deploy gave 1,617 ms against 5,222, at 98% of the agents'
  tokens.

At vLLM's default 2,048 tokens a step, the card's best stock step cap in the probes:

| Agents' sample (seed) | Tuned vLLM, 2,048/0 | The configuration to deploy, 2,048/0 | Of tuned vLLM's tokens/s | Its LLM first token p95: 2,048/0, 512/256 |
|---|---:|---:|---:|---:|
| 11 | 4,036 / 15,082 | 1,810 / 15,124 | 100% | 412 ms, 183 ms |
| 12 | 3,289 / 13,990 | 1,843 / 14,145 | 101% | 414 ms, 216 ms |

At 2,048 tokens tuned vLLM gives the agents 1.40 and 1.42 times their tokens at 512/256, and the configuration to deploy
keeps those tokens 140 and 173 ms over the SLO. A turn waits for the steps in flight, two under asynchronous scheduling,
and the budget cap acts only on the steps scheduled after it arrives. So 512/256 meets the SLO at 70% of the card's best
throughput.

The probes, against [the forecast](../benchmarks/a100/forecast-80gb.md) written for this card before the run and the
box's runs:

| Probe | A100 SXM4 80 GB | Forecast, central (low–high) [modeled] | The box |
|---|---:|---:|---:|
| KV case, stock 512/256: the voice turn's wait | 1,748 ms (1,721–1,750, 3 runs) | 1,328–1,369 ms (1,082–1,377) | 4,968 ms (4,900–4,971) |
| KV case, stock at vLLM's default 2,048/0 | 1,642 ms (1,620–1,645, 3 runs) | — | — |
| KV case, the configuration to deploy: the first token | 175 ms (150–176, 3 runs) | 126–145 ms (109–146) | 290 ms (216–294) |
| Prefill case, stock: the reply's token gap (1st to 16th) | 32 ms (3 runs) | 32.8 ms (25.7–35.4) | 60–66 ms |
| Prefill case, the configuration to deploy: the token gap | 7 ms (3 runs) | 5.6 ms (3.7–5.6) | 16–20 ms |
| The calls' first-token p95 beside 12 agents, 512/256 | 3,114 ms (3,102–3,126, 2 runs) | — | 7,631 and 7,643 ms (the kit's rehearsal) |
| The same at 2,048/0, and at 1,536/768 | 2,526 ms (2,518–2,535) and 3,566 ms (3,446–3,687) | — | — |
| The same, the configuration to deploy | 173 ms (167–179, 2 runs) | — | — |

- **The forecast placed each mechanism and missed the KV case's wait by about 30%.** Decode attention, fitted on the
  card's calibration point, read KV at 858 GB/s, 42% of the card's bandwidth, 1.82 times the forecast's cost. The box's
  reaches 77% of its own. FlashInfer's attention shortened the wait by 8%.
- **The host sets E0's GPU hit and restore.** Its Cascade Lake root ports are PCIe 3.0, so copies ran at 10–12 GB/s, 39%
  of the card's Gen4 link. Placed on this host, the forecast came within 10% of both
  ([the forecast on this host](../benchmarks/a100/forecast-80gb-on-host.md)).

At a datacenter batch, 256 requests of 48 prompt and 64 output tokens in flight, 26 at priority 0:

| Engine | Output tokens/s | Priority 0: TTFT p50 / p99 | Priority 0: TPOT p50 | Priority 2: TTFT p50 / p99 | Priority 2: TPOT p50 | Running p50 / max |
|---|---:|---:|---:|---:|---:|---:|
| Tuned vLLM | 7,524 | 166 / 574 ms | 28.6 ms | 341 / 1,086 ms | 29.1 ms | 201 / 254 |
| make-room, its switches off | 7,661 | 133 / 510 ms | 30.9 ms | 214 / 1,212 ms | 30.8 ms | 215 / 256 |
| make-room, the configuration to deploy | 8,672 | 90 / 177 ms | 6.9 ms | 283 / 7,971 ms | 28.6 ms | 154 / 230 |

With its switches off make-room made 1.018 times vLLM's output tokens a second, so the code path costs nothing. With the
deployed values priority 0's p99 TTFT fell from 574 to 177 ms and priority 2's rose 7.3 times, the cost of the budget
cap's holds.

[The A100 run's record](../benchmarks/results/2026-10-26-a100/README.md) gives each block of the run, its time and the
Colab host, and [`a100.md`](../benchmarks/results/2026-10-26-a100/a100.md) reads it against the forecast
(`make reproduce-a100`). On the box beforehand, [a rehearsal](../benchmarks/results/2026-10-26-a100-rehearsal/README.md)
ran the kit end to end, and [a dry run](../benchmarks/results/2026-10-25-marlin-dry/README.md) took the checkpoint
through Marlin.

<a id="failure-drills"></a>

## What does a failure do to the calls and the agents?

A killed engine ends every stream with an error event within 11 ms and serves again 106 s later, mostly its warm start.
A gateway reload or restart fails no new request, and with two GPU nodes the calls move to the spare within a turn.

The drills of 2026-10-25 ran from `ops/drills` against the deployed box, beside W8's build
([RUNBOOK.md](../ops/RUNBOOK.md#drills) runs them). Each opens two realtime and four background streams through the
gateway, breaks something once every stream has its first byte, and then probes both planes every second until a request
succeeds. Times count from the drill's stamp just before its `systemctl kill`.

| Drill | What the six streams saw | What new requests got | W8, before W6 |
|---|---|---|---|
| The engine killed mid-decode (SIGKILL to every process of `flatshare-engine`) | vLLM's error event, `upstream_unavailable`, 6–11 ms after the kill command | 210 of 212 probes failed at once, until one succeeded 106.2 s after the kill | Every body cut 53–57 ms after the kill command, with no error event, and a request succeeded 112.3 s after it |
| The gateway reloaded, as a deploy does | The realtime streams whole 20.9 s after, the background ones `draining` 5.0 s after | None of 42 failed or waited, the slowest answered in 0.13 s | — (a deploy restarted the gateway) |
| The gateway restarted, 30 s after the reload | As the reload's, 20.7 s and 5.0 s after | None of 42 failed, those sent during the drain answered once the new gateway started | Every stream finished while the gateway refused new connections |
| The gateway killed (SIGKILL, its socket unit holding the listeners) | All broke 10–14 ms after the kill command | None of 4 failed, the slowest waiting 1.3 s for the restart | — |

- **Where an engine's recovery goes.** The unit's last process exited 7.6 s after the kill, systemd started it 5.1 s
  later, and its warm start listened 92.1 s after that. Until then the gateway answers 503 with `Retry-After: 1`.
- **No KV or host memory outlives a killed engine.** `/dev/shm` held the CPU tier's 20.0 GiB before and after, made
  again by each start.
- **A deploy reloads the gateway without a gap**, handing its listening sockets to a successor that serves at once
  ([PRODUCTION.md](PRODUCTION.md#how-it-fails)).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-starts-narrow.svg">
  <img src="figures/report-starts.svg" width="100%" alt="Three bars of the engine's phases on one axis of seconds: killed mid-decode, it was down 12.7 s, its unit's last process gone 7.6 s after the kill and systemd's restart 5.1 s later, and its warm start listened 92.1 s after the restart, a request answered 106.2 s after the kill; a warm start was ready at 88.8 s, its weights loaded at 54.1 s and its CUDA graphs captured at 84.1 s; and a cold start, with empty caches, was ready at 531.8 s, its weights at 86.1 s, its KV cache at 189.3 s after compiling and profiling and its CUDA graphs at 526.6 s after the kernels' warmup and JIT builds.">
</picture>

**The engine's recovery from a kill is mostly its warm start, and on empty caches a start takes six times as long.**
Each bar stacks a start's phases from its journal, in seconds from the kill or the unit's start, and the dashed box is
the killed engine down, until systemd starts it again.

| Start | Unit start to ready | Where the time goes |
|---|---:|---|
| The engine, warm caches | 88.8 s | The weights loaded at 54.1 s and the CUDA graphs captured at 84.1 s |
| The engine, empty caches, as on a new host | 531.8 s | The weights at 86.1 s, the KV cache at 189.3 s after compiling and profiling, and the CUDA graphs at 526.6 s after the kernels' warmup and JIT builds |
| All three servers restarted (`systemctl --user restart flatshare.target`) | 102.7 s | The units stopped within 8.1 s and started at once, the engine ready last, 94.6 s after its start |

With both calls on Phastform and Gray a warm spare, `ops/drills/node-failover.sh` broke Phastform's engine on
2026-10-28, a 32-token turn of each call running every 3 s
([the record](../benchmarks/results/2026-10-28-node-failover/README.md)). The calls moved within a turn, and none of the
300 turns after the events failed.

| Drill | Turns before / after the event, whole | Calls moved to Gray | Turns failed after the first move | Phastform passed over / unhealthy / available again | Gray covered |
|---|---|---:|---:|---|---:|
| SIGKILL to the engine; systemd starts it again | 10 of 10 / 148 of 148 | +0.7 s | 0 | +0.10 s / +2.1 s / +101.0 s | 238 s |
| The engine stopped to the drill's end | 10 of 10 / 152 of 152 | +1.9 s | 0 | +1.3 s / +3.3 s / down to the end | 244 s |

[The drills' records](../benchmarks/results/2026-10-25-drills/README.md) hold every drill.

<a id="the-control-plane-flatshare-gateway-in-front-of-the-engine"></a>

## What does the gateway add to a request?

Under half a millisecond. Paired against the engine's own socket, the gateway added 0.29–0.45 ms at the median to a
voice request's first token, first sentence and end, and on v1.8 0.06–0.20 ms. End to end its cost is inside the
run-to-run spread.

The voice agent's own LLM request ran 40 rounds alternating between the engine's socket and the gateway's, with greedy
decoding so that both give the same text, on 2026-10-25.

| Path | First token p50 | First sentence p50 | End p50 | Gap between chunks, p50 / p99 |
|---|---:|---:|---:|---:|
| Engine socket | 43.3 ms | 162.1 ms | 344.1 ms | 12.08 / 12.53 ms |
| Through flatshare-gateway | 43.8 ms | 162.7 ms | 344.7 ms | 12.09 / 12.52 ms |
| **Paired median difference** | **+0.29 ms** | **+0.40 ms** | **+0.45 ms** | — |

On v1.8, over 80 rounds, the same pairing measured +0.12, +0.06 and +0.20 ms
([its record](../benchmarks/results/2026-10-29-gateway-window/README.md)).

Moving the servers from TCP forwards to Unix sockets, each connection an SSH session, left the turn the same and dropped
a 40 ms tail from each stream's end. That is Linux's shortest delayed acknowledgment, though the records do not say
which connection held it. p50 / p90 in ms ([the records](../benchmarks/results/2026-10-25-w6-transport), whose `ab.py`
prints the table):

| Stage | TCP forwards (W5's tunnel) | Sockets (flatshare-tunnel) | Sockets − TCP, p50 |
|---|---:|---:|---:|
| STT | 152.6 / 161.3 | 152.5 / 160.9 | −0.1 |
| LLM first token | 49.1 / 50.5 | 40.7 / 41.5 | −8.4 |
| LLM first sentence | 252.5 / 317.9 | 249.9 / 316.6 | −2.6 |
| TTS first audio | 176.3 / 187.6 | 175.6 / 186.7 | −0.8 |
| **Turn** | 584.6 / 655.2 | 581.0 / 655.1 | −3.6 |
| LLM whole reply | 303.0 / 369.2 | 263.3 / 330.0 | −39.7 |

- **In place** the gateway's own time, from the end of a request's body to the upstream write, averaged 0.13 ms over
  voice turns seconds apart and 0.16 ms beside the agent replay, at most 0.58 ms at p99. In process it is 32 µs
  ([the gateway's README](../crates/flatshare-gateway/README.md#overhead)).
- **Connections must outlive the turn gap.** vLLM drops idle connections after 5 s, less than many gaps between voice
  turns, and a new one costs about 1 ms and at times 40 ms more. Keeping them 60 s took the realtime first byte's p99
  from 205 to 99 ms.
- **End to end** voice alone gave a V2V p50 of 1,600–1,684 ms through the gateway and 1,579–1,614 ms straight to the
  engine, and beside the agent replay every request completed.

## What does an API layer add to an agent's turn?

Nothing measurable. The vLLM project's [agentic-api](https://github.com/vllm-project/agentic-api), which serves coding
agents' APIs, added 0–3 ms to the per-turn median of a 4-round tool loop, below the noise at up to 50 concurrent loops.
Its own `response_provider` benchmark ran all five arms at once against the box on 2026-10-22, p50 in ms:

| Sessions per arm | agentic-api HTTP/SSE | vLLM HTTP/SSE | Paired median difference |
|---:|---:|---:|---:|
| 1 | 3,173 | 3,176 | 0 ms |
| 5 | 4,489 | 4,482 | +3 ms |
| 10 | 6,102 | 6,072 | +1 ms |

It used about 5.5 ms of CPU a round and 44–54 MiB of memory, and since it holds the conversation state, clients sent
3.6–3.8 KB a turn against 8.4 KB direct ([the record](../benchmarks/results/2026-10-22-agentic-api-http-baseline/)).

## How do the agents use the KV cache?

Mostly from cache. Beside 16 sessions about 90% of the agents' prompt tokens come from the GPU prefix cache or the CPU
tier, and the share computed grows with the load, from 7.7% at 8 sessions to 15.2% at 24.

W2 replayed the 32× corpus at three loads on 2026-10-22, straight to the engine, each run on a fresh engine for 10
minutes with think times capped at 120 s. The prompt-token split is vLLM's own counters over the whole run.

| Agent sessions | Requests | TTFT p50 / p90 / p99 | ITL p50 | Prompt tokens: GPU / CPU tier / computed | Preemptions |
|---:|---:|---:|---:|---:|---:|
| 8 | 408 | 84 / 376 / 1,554 ms | 13.2 ms | 64.3 / 28.0 / 7.7% | 0 |
| 16 | 686 | 132 / 787 / 9,198 ms | 15.6 ms | 55.2 / 34.6 / 10.2% | 3 |
| 24 | 1,568 | 175 / 1,415 / 15,605 ms | 22.0 ms | 51.4 / 33.4 / 15.2% | 2 |

- **LRU evicts sessions that are coming back.** In W0's replay at 16 sessions, half of the evicted blocks had been idle
  no longer than the 90th-percentile reuse gap of 6.33 s. With the 4× write-through CPU tier those misses cost a
  restore, not a recompute.
- **The simulator reproduces them.** `flatshare-sim`, fitted to E0's cells and these points, gives resume TTFT p50 and
  p90 within 3–8% at 8 and 16 sessions, 0–5% leaving each point out. At 24 sessions it is 18% and 30% fast, because
  restores there take 163 ms longer than modeled.

`make reproduce-calibration` recomputes the table from the runs' exports, an asset of the
[v0.3 release](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3)
([the simulator's calibration](../crates/flatshare-sim/README.md#calibration-against-the-gpu-operating-points)).

## Does knowing when an agent returns help?

Little at the median, and only with knowledge no deployment has. In the simulator even an oracle that knows each return
closes under 8% of the median resume time over LRU and the CPU tier. At low pressure it closes 28–29% of the p90, about
80–90 ms, by hiding a restore inside the pause. Session history, the knowledge a deployment has, gains nothing.

`flatshare-sim` ([its README](../crates/flatshare-sim/README.md)) replays the 32× corpus through a model of vLLM 0.32's
two-tier KV cache and step loop, each parking arm in place of the engine's LRU
([the design](EXPERIMENTS.md#e1-parking)). In the realistic regime, the traces' think times with the CPU tier at 4× (π
about 23), over 20 seeds, each cell gives the mean of the per-seed gains over `lru-cpu`, then their median and range:

| Sessions | Pressure | π | `lru-cpu` resume p50 / p90 | `oracle-prefetch` p50 | p90 | `oracle-prefetch` on mean latencies, p50 / p90 | `clairvoyant` p50 / p90 | Best history arm, p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 1.6 | 28.6 | 86 / 267 ms | +3.6% (+3.4) [−3, +11] | +28.1% (+28.5) [+10, +48] | +3.7 / +29.4% | +1.4 / +15.5% | `history-prefetch` +3.9% |
| 12 | 2.4 | 20.6 | 103 / 299 ms | +3.3% (+3.3) [−1, +8] | +28.6% (+30.8) [+15, +41] | +3.4 / +29.5% | +0.5 / +8.0% | `history-prefetch` −6.6% |
| 16 | 3.2 | 21.3 | 139 / 895 ms | +7.6% (+5.3) [+1, +32] | +15.3% (+22.6) [−59, +41] | +9.7 / −0.7% | +1.7 / +9.4% | `infercept` −9.8% |
| 24 | 4.4 | 23.3 | 208 / 1,645 ms | +4.9% (+6.2) [−41, +25] | −3.9% (−1.4) [−42, +26] | −2.3 / −8.2% | +2.3 / +10.9% | `infercept` −7.0% |

- **The gate reads negative at p50**, where the oracle closes under 8% by the mean of the per-seed gains and 10% at 16
  sessions by the gain of the mean latencies, at the gate rather than under it. So no lease-aware CPU-tier policy runs,
  and `LeasePolicy`, built for E1 and never switched on, was removed after v0.3
  ([ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md)).
- **The tail gain needs each tool's return time**, which no client sends, and at 24 sessions the oracle loses 4% at p90.
- **The history level closes nothing**, its best arm gaining 3.9% at p90 at 8 sessions and losing 7–10% from 12 on.
- **The gate's first form, on the clairvoyant reference's headroom, reads the same**, at 0.5–2.3% at p50.
- **On an L4, an A100 and an H100**, modeled at their own scalings, lifecycle information buys at most 15%, 13% and 9%
  of resume TTFT p90 ([other hardware, modeled](../crates/flatshare-sim/README.md#other-hardware-modeled)).

### Where does knowing the returns help most?

The oracle's tail gain peaks at 8–12 sessions with the traces' pauses and turns to a loss at high pressure. Each cell is
the resume TTFT gain over `lru-cpu` at p50 / p90, the mean over seeds 11–13, with the CPU tier at 4× and ρ 0.25–0.27.
`make reproduce-e1` writes the map as `regime.md`, with the 2× and 8× tiers.

| Pauses | Sessions | Pressure | π | `oracle-prefetch` | `history-prefetch` | `tokencake` | `ttl-30` | `clairvoyant` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trace | 5 | 1.3 | 27.0 | +2 / +13% | −1 / +0% | −5 / −9% | −1 / +2% | +1 / +7% |
| trace | 8 | 1.8 | 29.6 | +3 / +40% | −0 / +12% | −6 / −10% | +1 / +4% | +1 / +27% |
| trace | 12 | 2.6 | 21.4 | +5 / +32% | +1 / −8% | −11 / −21% | +0 / −2% | +0 / +7% |
| trace | 16 | 3.6 | 23.4 | +9 / +2% | −9 / −9% | −24 / −18% | +2 / −5% | +2 / +8% |
| trace | 24 | 4.7 | 23.9 | +2 / −5% | −19 / −17% | −36 / −17% | +5 / −3% | +3 / +8% |
| trace | 32 | 6.2 | 23.5 | −5 / −23% | −24 / −11% | −27 / −5% | −7 / +1% | +3 / +2% |
| trace | 40 | 7.5 | 22.9 | −25 / −21% | −10 / −6% | −10 / −3% | +4 / +6% | +1 / +1% |
| tracelab | 5 | 1.2 | 3.2 | −0 / +1% | −0 / +0% | −3 / −4% | −1 / +1% | +0 / +1% |
| tracelab | 8 | 1.8 | 2.9 | +2 / +4% | −2 / −1% | −4 / −10% | +0 / +3% | +1 / +4% |
| tracelab | 12 | 2.4 | 2.7 | +1 / +9% | −0 / −0% | −10 / −8% | +0 / +0% | +0 / +7% |
| tracelab | 16 | 3.4 | 2.6 | +17 / −11% | −9 / −13% | −18 / −18% | +7 / −2% | +4 / +2% |
| tracelab | 24 | 4.7 | 2.9 | −14 / −10% | −19 / +0% | −36 / −1% | +13 / +1% | +2 / +0% |
| tracelab | 32 | 6.1 | 3.3 | −13 / −3% | −11 / −12% | −12 / −13% | −0 / −0% | +1 / +1% |
| tracelab | 40 | 7.3 | 2.8 | −1 / −1% | −6 / −1% | −4 / −1% | +0 / +0% | +1 / +1% |

With the traces' pauses up to 24 sessions, `mori` loses 1–15% at p50 and `infercept` 18–36%. With TraceLab's tool
pauses, π about 3, the oracle gains at most 9% at p90.

### How much could any eviction order save?

About a third of LRU's misses at the pool's size, where Belady's MIN misses 22.9% of the prompt tokens and LRU 35.6%.
MIN is down to the cold misses, 4.1%, at four times the pool, LRU at eight. `flatshare-sim mrc` computes the per-block
miss ratios on the W0 configuration's admission sequence, for the 32× replay and for the same order at the original
token counts:

| Pool multiple | LRU, 32× | LRU, original tokens, same order | MIN, 32× | MIN, original tokens, same order |
|---:|---:|---:|---:|---:|
| 0.25 | 0.718 | 0.718 | 0.503 | 0.503 |
| 0.5 | 0.541 | 0.540 | 0.341 | 0.340 |
| 1 | 0.356 | 0.355 | 0.229 | 0.228 |
| 2 | 0.217 | 0.216 | 0.108 | 0.107 |
| 4 | 0.058 | 0.057 | 0.041 | 0.039 |
| 8 | 0.041 | 0.039 | 0.041 | 0.039 |

The curves agree within 0.2 points at every capacity, so scaling keeps the reuse structure. It changes the timing, since
a cold 100k-token prompt is priced at 88 s at the original counts, so E1 states its results in the regime coordinates.

## How closely does the replay follow the traces?

Closely enough that no point was rejected for its load.
- **The turns arrive on time.** AIPerf 0.13.0's `agentic_replay` of the SemiAnalysis Claude Code traces delivers 99.4%
  of turns within 50 ms of their recorded delay.
- **Block scaling makes all 98,827 requests fit.** At 32× they fit a 32k context, the main agent's context at 7.9k
  tokens at the median and 20.3k at p90.
- **The load generators run on pinned cores** on cyan, and the callers play their lines at most 116 ms late, at six
  callers.
