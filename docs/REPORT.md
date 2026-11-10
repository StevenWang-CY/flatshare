# Making Room in the Engine: A Realtime Voice Tier beside Coding Agents on One GPU

**Chuyue Wang**

Flatshare technical report, version 1.25, 9 November 2026. Code, records and figures at tag
[v1.25](https://github.com/StevenWang-CY/flatshare/tree/v1.25) of
[github.com/StevenWang-CY/flatshare](https://github.com/StevenWang-CY/flatshare); earlier versions in [Appendix
E](#e-revision-history). The report is for engineers who serve a voice agent and background LLM work from one GPU, and
argues for two rules in vLLM's scheduler from the measurements behind them.

> [!NOTE]
> The change this report measures is proposed to vLLM as pull requests
> [#65282](https://github.com/vllm-project/vllm/pull/65282) and
> [#65283](https://github.com/vllm-project/vllm/pull/65283), open, with no maintainer review as of 2026-11-09. They
> succeed #63350 and #63351, the numbers the measurements use. The cache-aware queue and the SM copy of §6.11 are
> proposed as the drafts [#66558](https://github.com/vllm-project/vllm/pull/66558) and
> [#66527](https://github.com/vllm-project/vllm/pull/66527). Flatshare is an independent project and is not affiliated
> with vLLM.

## Abstract

A voice agent needs its GPU only while it computes a reply, and coding agents could use the rest. On vLLM 0.32, beside
24 coding-agent sessions on one RTX 5060 Ti, two calls' p95 voice-to-voice latency rises from 1.78 s alone to 10.48 s,
against an SLO of voice alone plus 250 ms, 2.03 s. A waiting voice turn cannot take KV from running agent requests, and
an admitted one shares its steps with agent prefill. The nearest deployable configuration we measured, a gateway policy
that aborts agent requests, reaches 2.52 s at a 193 s agent p99 time to first token. We add two rules to vLLM's priority
scheduler. The KV preemption lets the waiting turn preempt lower-priority requests for KV, and the budget cap keeps
their prefill out of the turn's steps through its reply's first sentence.

Together the rules bring the p95 to 2.24 s and the replies over the SLO from 90% to 32%, 286 of 885, while the agents
keep 97% of their token rate. On an A100 at 512-token steps they take the p95 from 4.99 s to 1.63 s, under that card's
SLO in each of five rounds. Consolidating both calls and four agent sessions on one of two cards does 34% more
background work than dedicated cards; tuned vLLM, consolidated at two sessions, does 27% more. A 256-token step cap cuts
the replies over the SLO from 94 of 295 to 44 of 295, and four additions to the engine more than halve them in rounds of
their own. Ordering the agents' queue by the prefill each has left lets one card carry 24 agent sessions beside both
calls instead of 4, at a 133 s agent p99.

## Contents

1. [Introduction](#1-introduction)
2. [Background and motivation](#2-background-and-motivation)
3. [Design](#3-design)
4. [Implementation](#4-implementation)
5. [Evaluation setup](#5-evaluation-setup)
6. [Results](#6-results)
7. [Operating the change](#7-operating-the-change)
8. [Related work](#8-related-work)
9. [Limitations](#9-limitations)
10. [Conclusion](#10-conclusion)
- [References](#references)
- Appendices: [A. Reproducing the results](#a-reproducing-the-results) · [B. Results by round](#b-results-by-round) ·
  [C. Configurations](#c-configurations) · [D. Failure drills](#d-failure-drills) · [E. Revision
  history](#e-revision-history)

## 1. Introduction

A self-hosted voice agent is a cascade of speech recognition, an LLM and speech synthesis, provisioned for how soon it
answers. Its GPU idles for most of each call, because the LLM works only while it computes a reply, and serving systems
fill such idle capacity with best-effort work [[1](#ref-1), [2](#ref-2), [3](#ref-3)]. Coding agents suit the role. They
send long prompts that share long prefixes, and they spend much of each session waiting for tools and for their user.

The two workloads do not share a GPU well on today's engines. On one RTX 5060 Ti running vLLM 0.32 [[4](#ref-4)], tuned
as an operator would tune it (§5.3), a voice agent's median reply beside 24 Claude Code sessions begins 4.61 s after the
caller stops speaking, against 1.62 s alone. The p95 voice-to-voice latency (V2V) of two calls rises from 1.78 s to
10.48 s. The turn waits 2.81 s at the mean for the LLM's first token, against 0.09 s alone, and then takes 1.07 s to
decode the first sentence it can speak, against 0.22 s (Figure 1).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-stages-narrow.svg">
  <img src="figures/report-stages.svg" width="100%" alt="The voice turn's mean voice-to-voice time stacked by stage from the end of the caller's speech, beside 24 agent sessions but for voice alone: voice alone, 1.62 s, 0.09 s of it waiting for the LLM's first token and 0.22 s decoding the first sentence; tuned vLLM, 5.30 s, 2.81 s of it waiting for the LLM's first token and 1.07 s decoding the first sentence; preempt-resubmit, 2.08 s, 0.28 s of it waiting for the LLM's first token and 0.39 s decoding the first sentence; the KV preemption, 3.41 s, 0.78 s of it waiting for the LLM's first token and 1.21 s decoding the first sentence; the change, 2.15 s, 0.30 s of it waiting for the LLM's first token and 0.44 s decoding the first sentence; the configuration to deploy, 1.97 s, 0.29 s of it waiting for the LLM's first token and 0.26 s decoding the first sentence.">
</picture>

**Figure 1. Sharing the GPU stretches two stages of the voice turn, and the configuration to deploy brings both back
near voice alone's.** Mean time of each stage of a voice turn from the end of the caller's speech, over every reply of
each configuration's runs, with two callers beside 24 coding-agent sessions on the RTX 5060 Ti. Violet is the wait for
the LLM's first token, green the decode of the reply's first sentence, and the number after each bar the mean V2V.

No configuration an operator can deploy today meets the SLO, voice alone's p95 plus 250 ms. vLLM's own settings,
SGLang's priority preemption with its host tier [[5](#ref-5), [6](#ref-6)], a static KV reservation and two routers miss
it by seconds. Preempt-resubmit, a gateway policy that aborts agent requests before their first byte and resubmits them,
misses it by 0.49 s, and the agents' p99 time to first token (TTFT) grows from 106 s to 193 s.

Two mechanisms in vLLM's scheduler explain the loss, and neither is specific to the card. The V1 scheduler preempts a
running request only so that another running request can grow, never for a waiting one [[7](#ref-7)]. When the agents'
running requests hold the KV pool, a voice turn waits until one of them finishes or is preempted for another's growth,
4.9–5.0 s in isolation behind two 18k-token agent requests. Once the turn is admitted, every step of its prefill and its
reply also carries the agents' prefill chunks, so its reply's tokens arrive 60–66 ms apart instead of 16–20 ms. A
control plane in front of the engine sees requests, not KV blocks or steps, so it can defer or abort agent requests but
cannot preempt one and keep its work, or reach into a step.

We change the scheduler instead, with two rules (Figure 5). The KV preemption lets the highest-priority waiting request
preempt lower-priority running requests when it lacks KV, at most N in a step, through vLLM's own preemption path.
vLLM's V0 scheduler did so [[8](#ref-8)], but only with chunked prefill off and by recomputing its victims. Here the
victims' KV is already in the CPU tier, so a resume reloads what the tier still holds. The budget cap keeps
lower-priority prefill out of that request's steps until it has T output tokens, enough for the reply's first sentence,
while every decode proceeds. Both rules count blocks, tokens and steps, need no model of the hardware and are off by
default, and they are proposed to vLLM as two stacked pull requests [[9](#ref-9), [10](#ref-10)].

The report makes six contributions:

- **It measures the failure end to end** (§2.4, §5, §6.1), with V2V read from waveforms at the caller beside a
  closed-loop replay of real coding-agent traces.
- **It changes vLLM's priority scheduling** with two rules that run on vLLM 0.32 without modifying the installed engine
  (§3, §4). Neither idea is new on its own (§8).
- **It shows what each rule recovers and what the agents pay** (§6.1–§6.4). The p95 comes to 2.24 s, 0.20 s over the
  SLO, and the agents keep 97% of their token rate.
- **It bounds what a control plane in front of the engine can do** (§6.5). Short of aborting agent work, the room a
  waiting turn needs comes only from inside the engine.
- **It maps where the gain holds** (§6.6–§6.8), from a pool with room and overlapping calls to an A100 and two cards.
- **It tunes the serving engine around the rules** (§6.9–§6.11) until one card carries 24 agent sessions beside both
  calls.

## 2. Background and motivation

### 2.1 A voice turn

A cascaded voice agent answers in four stages. It transcribes the caller's speech, decides that the caller has finished
(endpointing), streams the transcript through the LLM, and hands speech synthesis the reply's first complete sentence as
soon as the LLM has produced it. The caller hears nothing until that sentence is spoken, and voice-to-voice latency
(V2V) runs from the end of the caller's speech to the agent's first audio.

With the voice agent alone on the GPU, a turn's mean V2V is 1.62 s, of which the LLM's first sentence takes 0.30 s. That
share is small but sits on the critical path, and the first sentence rather than the whole reply sets the wait. The SLO
allows voice alone's p95 V2V plus 250 ms, 2.03 s with two callers.

### 2.2 Coding agents as background work

Coding agents are the opposite workload. In the Claude Code traces we replay [[11](#ref-11)], each session alternates
model calls with tool calls and the user's think time, and each call re-sends a context that extends the prefix of the
last one. Scaled to fit the GPU (§5.2), the 48 traces with the longest contexts send a median request of 13.9k tokens.

The KV pool beside the audio models holds 36,400 tokens, two or three such contexts, so a few agents fill it. Losing a
context's KV is expensive. On this GPU a 16k-token prompt takes 3.26 s to recompute, 509 ms to restore from the CPU tier
and 97 ms when its blocks are still in the GPU's prefix cache.

### 2.3 Scheduling in vLLM

vLLM batches requests continuously and stores KV in fixed-size blocks [[4](#ref-4)]. In each engine step, its V1
scheduler first schedules the running requests, a decode token each or the next chunk of a prefill, and then admits
waiting requests while the step's token budget and the free blocks last. Chunked prefill [[12](#ref-12)] splits a long
prompt over many steps so that decodes do not stall behind it, and a step cap (512 tokens here) with a per-request
threshold (256 tokens) bounds each step's prefill.

Under priority scheduling a lower value is served first. The waiting loop takes the requests that already hold KV first,
then the others in priority order, and stops at the first request it cannot allocate. When a running request cannot
grow, the scheduler preempts the lowest-priority running request, the latest arrival among equals, and nothing preempts
on a waiting request's behalf [[7](#ref-7)].

vLLM's offloading connector keeps a CPU tier of KV, written through as tokens are computed, so a request whose blocks
were evicted from the GPU reloads them instead of recomputing them. Asynchronous scheduling plans each step before the
previous step's output is processed, which leaves up to two steps in flight, so with a CPU tier a preempted request's
blocks are freed one step late.

### 2.4 The two waits in isolation

Three probes on the RTX 5060 Ti isolate the waits, with the budget cap at its default window of 16 tokens. In the KV
case, two priority-2 requests of 18k prompt tokens fill the pool and keep decoding, and a 750-token priority-0 request
arrives once at most 24 blocks are free. Nothing frees a block until one agent request cannot grow, about 5 s later, and
vLLM preempts the later arrival so that the other can (Figure 2). The voice turn's first token comes 4.9–5.0 s after it
is sent, at that preemption, and 0.22–0.29 s after it with the KV preemption (Figure 3a).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-kv-timeline-narrow.svg">
  <img src="figures/report-kv-timeline.svg" width="100%" alt="In panel a, the share of the KV pool's 2,274 blocks in use on stock vLLM while the KV case's two agent requests of 18k prompt tokens prefill and decode, read about every 0.55 s: 98.7% at 7.7 s, 20 blocks free at 8.3 s, where the KV case sends its voice turn, none freed until 12.6 s, and 50.1% at 13.1 s, once vLLM has preempted agent B so that agent A can grow; in panel b, each request's output tokens, both decoding into the last free blocks until agent B stops at 192 tokens while agent A reaches 812 by 24.5 s.">
</picture>

**Figure 2. On vLLM 0.32 a full KV pool frees nothing until a running request cannot grow.** Two priority-2 requests of
18k prompt tokens, the engine's state sampled about every half second. (a) The share of the KV pool's 2,274 blocks in
use, shaded from where the KV case sends its voice turn to the preemption that frees half the pool. (b) Each agent
request's output tokens.

In the prefill case, three priority-2 requests of 10,000 prompt tokens prefill side by side with room left in the pool.
The priority-0 request is admitted at once, but every step of its reply carries their prefill chunks, so its tokens come
60–66 ms apart, against 16–20 ms with the budget cap (Figure 3b). A step cap bounds the tokens in a step, not its time,
and a reply's first sentence takes a step for each of its tokens, 15 at the median and 39 at most (§6.3).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-isolation-narrow.svg">
  <img src="figures/report-isolation.svg" width="100%" alt="Four probe runs on the RTX 5060 Ti for each of stock vLLM, the KV preemption and the change at its 16-token window: in panel a, the KV case, the voice turn's first token while two agent requests of 18k tokens hold the pool (stock vLLM 4,900–4,971 ms; the KV preemption 219–290 ms; the change 216–294 ms); in panel b, the prefill case, its first token to its 16th while three agent prefills of 10k tokens run, the 16th at stock vLLM 1,220–1,529 ms; the KV preemption 1,211–1,528 ms; the change 536–615 ms.">
</picture>

**Figure 3. In isolation, a voice-sized request waits 5.0 s for KV on stock vLLM and 0.29 s with the KV preemption, and
agent prefill slows its tokens three- to fourfold until the budget cap holds it back.** Four runs of each probe, the
budget cap at a 16-token window. (a) The KV case's time to the first token, each circle a run and the label the median.
(b) The prefill case, each line from a run's first token to its 16th, the bar and label at the median 16th token.

Beside closed-loop agents the two waits occur together (Figure 4). With two calls beside 12 agent sessions whose prompts
are 6–30k tokens, the calls' p95 time to first token is 6.83 s on vLLM 0.32, 4.11 s with the KV preemption alone and
0.39 s with both rules. When each agent's prompt instead grows by a tool result, so that the agents reuse the CPU tier,
it is 7.16, 1.07 and 0.36 s.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-tail-narrow.svg">
  <img src="figures/report-tail.svg" width="100%" alt="The cumulative share of two calls' turns by their time to first token beside 12 closed-loop agent sessions, pooled over two 240 s runs of each scheduler, on a log scale with the 95th percentile marked: in panel a, with the agents' prompts fresh each turn, the p50 and p95 at 2,244 and 6,830 ms on stock vLLM (96 turns), 716 and 4,111 ms with the KV preemption (117 turns), 236 and 388 ms with the change (146 turns); in panel b, with each agent's prompt grown by a tool result, the p50 and p95 at 1,324 and 7,158 ms on stock vLLM (102 turns), 615 and 1,073 ms with the KV preemption (125 turns), 258 and 358 ms with the change (146 turns).">
</picture>

**Figure 4. Beside closed-loop agents, the KV preemption shortens the calls' first-token tail and the budget cap removes
it.** The share of two calls' turns whose first token came within a given time, on a log scale, over two 240 s runs per
engine. (a) The agents' prompts are fresh each turn. (b) Each grows by a tool result. The change is both rules, the
budget cap at a 16-token window, and the thin line marks the 95th percentile.

## 3. Design

The pull requests add two rules to vLLM's priority scheduler (Figure 5). We write p(x) for a request's priority, a lower
value served first, with the voice tier at 0 and the agents at 2. Each rule is a switch, off by default, and with both
off the scheduler produces vLLM's schedules. Scheduling among equal priorities, preemption for a running request's
growth and the order of decodes stay vLLM's.

<picture>
  <source media="(max-width: 767px)" srcset="figures/method-narrow.svg">
  <img src="figures/method.svg" width="100%" alt="Panel a: a caller's speech goes to speech to text, then to vLLM 0.32's scheduler, which carries #63350 and #63351, then to text to speech and back to the caller; 24 coding-agent sessions send their requests to the same scheduler at priority 2, the voice turn's at priority 0; the scheduler allocates from the KV pool beneath it. Panel b, the KV preemption, #63350: in a pool of 12 KV blocks, two agent requests hold 5 each and 2 are free, fewer than the 4 a waiting voice turn needs; on vLLM 0.32 the turn keeps waiting, while the KV preemption preempts the latest agent request, agent B, and the turn takes 4 of its blocks. Panel c, the budget cap, #63351: the engine's steps in time, one box a step, from the reply's first token: on vLLM 0.32 each step of the reply also carries an agent's prefill chunk, so in the time the budget cap gives the reply its first 48 tokens the reply gets 14 without it; after the window, agent prefill resumes. Blocks and steps are illustrative.">
</picture>

**Figure 5. The KV preemption takes KV for the waiting voice turn, and the budget cap keeps agent prefill out of its
first tokens.** (a) The voice agent and 24 coding-agent sessions share one GPU and one vLLM engine, at priorities 0
and 2. (b) With the KV pool full, vLLM 0.32 leaves the voice turn waiting until an agent request frees its blocks. The
KV preemption takes agent B instead, whose KV is in the CPU tier. (c) Engine steps from the reply's first token.
Without the budget cap every step also carries an agent's prefill chunk, and the reply decodes 14 tokens in the span.
With it the reply's first 48 tokens run beside decodes only. Blocks and steps are illustrative.

### 3.1 The KV preemption

At the start of each step, before the running loop, the KV preemption finds r, the highest-priority waiting request that
is ready to run in this step (Algorithm 1). If r lacks KV for its whole remaining sequence, net of its prefix-cache hit,
the rule preempts running requests of a strictly lower priority through vLLM's own preemption path. It takes them in the
order vLLM's running loop preempts in, the lowest priority first and then the latest arrival, and preempts as few as
make r fit, at most N in a step and none if N would not suffice.

The count of room matches what vLLM allocates. A block shared through the prefix cache counts as freed only when every
holder is a victim, and the running decodes' growth until the frees land is added to r's shortfall. The room is counted
beside vLLM's watermark and what in-flight prefills still need. Preemptions do not cascade, since the rule makes no room
for r again until the waiting loop has tried it, and no lower-priority request can take the room back in that step. A
victim's pending stores to the CPU tier are flushed before its blocks are freed, so a resume reloads what the tier still
holds and recomputes the rest.

```text
Algorithm 1. The KV preemption, in each scheduler step, before its running loop (at most N victims).
p(x) is a request's priority, and a lower value is served first.

 1  if prefills are deferred, new requests are paused, or the KV connector has block frees
        pending: return
 2  r ← the highest-priority waiting request that is ready to run in this step
 3  if r is none, or no running request has p > p(r), or room was made for r and the waiting loop
        has not tried r since, or (with the budget cap) h < p(r) in Algorithm 2: return
 4  C ← up to N running requests with p > p(r), the lowest priority first, then the latest arrival
 5  s ← the free blocks r lacks for its whole sequence, net of its prefix-cache hit as the waiting
        loop bounds it, beside the watermark and what in-flight prefills still need
 6  if s > freed(C): return                         ▹ not even every candidate makes room
 7  s ← s + growth(running)                          ▹ the decodes' growth until the frees land
 8  if s ≤ freed(∅): return                          ▹ r fits, or deferred frees make the room
 9  V ← the shortest prefix of C with s − growth(V) ≤ freed(V); if none: return
10  preempt each request in V through vLLM's preemption path ▹ its KV is in the CPU tier
11  mark r untried, and in this step's waiting loop stop at a request in V and skip each q with
        p(q) > p(r) that holds KV blocks but no reservation for the rest
12  in every step's waiting loop, over requests holding no KV: if a prefill x gets a chunk in this
        step, stop at the first q with p(q) > p(x) that would take blocks in-flight prefills still need

freed(X): the blocks whose every holder is in X or has a free deferred, less those r takes as its
          prefix-cache hit, plus the blocks that in-flight prefills in X still reserve.
growth(X): the blocks the requests in X would take until the frees land, over one step, or two with
           deferred frees.
```

### 3.2 The budget cap

The budget cap keeps the batch's prefill out of the admitted turn's way (Algorithm 2). Lower-priority prefill sits out
the steps of a higher-priority request from the step in which it fits, through its prefill, until it has T output
tokens, a step in flight counted as one. A held step skips running lower-priority prefill chunks and admits no waiting
lower-priority request, while every decode proceeds. A voice agent speaks once its first sentence is complete, so the
cap covers a reply's first tokens and not its whole decode.

Exemptions keep the queue moving. The first waiting request is held only after its prefix-cache and KV connector
lookups, so its KV load starts meanwhile, and a request with one token left samples like a decode and is never held. A
lower-priority request that already holds KV blocks is held only while a request of its priority or a lower one runs,
since preemption cannot reach its blocks while it waits. A prefill midway through a cached encoder input is not held for
a request that needs encoder cache space, since only that prefill's progress frees the cache.

Bounds keep the cap from stalling the engine. Nothing is held for a waiting request or a KV load while every running
slot is taken. A request whose first tokens are due holds for at most T steps in a row in which it computes nothing,
since what it waits for may be what it holds back. A KV load holds for its first T steps and again for at most T steps
from the step it completes in. Holding through a load matters, because its completion shows only at the end of a step
and agent prefill makes those steps about three times longer. At T = 16, a build that held nothing during loads made the
calls' first token 125–131 ms later at the median.

```text
Algorithm 2. The budget cap, in each scheduler step (window T, in output tokens).

 1  H ← the requests whose first tokens are due: running requests with fewer than T output
        tokens, a step in flight counted as one, and, while a running slot is free, requests
        whose KV load is done or began fewer than T steps ago; but for a load in flight,
        none that has computed nothing for T steps in a row
 2  if r (Algorithm 1) fits once the victims' frees land, a running slot is free, no request
        in H has a priority above r's, and r has computed nothing for fewer than T steps:
        add r to H
 3  if H is empty: schedule the step as vLLM does; else h ← the highest priority in H
 4  running loop: for each running request q
 5      if q is prefilling with more than one token left, p(q) > h, and not (q is midway
            through a cached encoder input and a request in H of a priority above p(q) needs
            encoder input):
            skip q                                   ▹ its chunk sits out this step
 6      else schedule q as vLLM does                 ▹ every decode runs
 7  waiting loop: for each waiting request q, in vLLM's order
 8      if p(q) > h, q holds KV blocks, has more than one token left, and a running request
            has a priority of p(q) or lower: skip q
 9      after q's prefix-cache and KV connector lookups (a KV load starts):
            if p(q) > h, q holds no KV blocks and has more than one token left: stop
10      admit q as vLLM does
```

### 3.3 Sizing the window

T is counted in output tokens, but the voice agent speaks its first complete sentence, at least 20 characters by
LiveKit's default [[13](#ref-13)], so T must cover that sentence. The records keep each reply's tokens but not its text,
so T = 48 is sized to cover whole replies, which bound the first sentence. The 5,571 replies of the main evaluation ran
23 tokens at the median and 52 at most (Figure 6a). A 16-token window ends before 79% of them, a 48-token window covers
all but one, and a longer window costs the agents nothing more, since a hold ends with its reply (§6.3). Asked again for
320 replies, the engine completed every first sentence within 39 tokens (§6.3).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-window-narrow.svg">
  <img src="figures/report-window.svg" width="100%" alt="In panel a, the cumulative share of 5,571 voice replies by their output tokens, 23 at the median and 52 at most, 21% ending within a 16-token window and all but 1 within a 48-token one; in panel b, the p95 of a 40-token reply's last token by the window, the median of two runs (16 tokens 3.76 s; 24 tokens 2.64 s; 32 tokens 1.85 s; 48 tokens 1.04 s; 64 tokens 1.02 s), its first token's staying between 326 and 448 ms; in panel c, the agents' prompt tokens a second, 3,483 at 16 tokens and 3,205 at 48, 8% less.">
</picture>

**Figure 6. A window that covers the replies keeps their tokens fast, and a longer one costs the agents nothing more.**
(a) The share of the 5,571 replies that end within a given number of output tokens. (b) The p95 time to the 40th token,
and to the first, of 40-token replies beside closed-loop agents, by window, circles runs and lines their medians. (c)
The agents' prompt tokens per second in the same runs.

## 4. Implementation

The rules are proposed to vLLM as two stacked pull requests on vLLM's main, #65282 and #65283, which succeed #63350
and #63351 with the same change [[9](#ref-9), [10](#ref-10)]. They add `--max-priority-preemptions-per-step N` and
`--priority-prefill-budget-cap` with `--priority-prefill-budget-cap-tokens T`, 16 by default. Both carry tests that
run vLLM's real scheduler on CPU, and the budget cap counts the steps it held and the prefill tokens it held back.

The evaluation runs vLLM 0.32.0, which predates both. At each engine start a build step generates a scheduler from the
installed vLLM's without modifying the installed package, the engine variant make-room, which the engine loads with
`--scheduler-cls`. Make-room applies the pull requests' scheduler diff, ported to 0.32.0, reads the switches from vLLM's
additional configuration, logs the rules' counts, and adds a class-aware KV reserve that the pull requests leave out and
the evaluated configuration turns off. A trace test runs make-room and the pull requests' patches on the same request
traces and requires every step's schedule to match.

Every GPU number in this report comes from make-room, since the pull requests' branches ran only in their CPU tests. The
main evaluation's make-room matched the pull requests before their revision of 25 October 2026, which ends the budget
cap's hold one step sooner by counting a reply's first tokens still in flight. Only a reply at least T tokens long
reaches that step, 248 of the 305 at T = 16 and 1 of the 2,469 at T = 48. Five rounds of the configuration to deploy on
the code of 25 October gave a p95 of 2.27 s against 2.25 s on the same samples before, −45 to +77 ms round by round.

Two later revisions followed reviews. That of 3 November adds most of the exemptions and bounds of §3.2, and that of 5
November bounds every holder by the steps in which it computes nothing. A search of 400,000 random workloads on small KV
pools finds no stall the options cause on the pull requests, and one of 200,000 finds none on make-room with the serving
profile's switches ([the record](../benchmarks/results/2026-11-05-liveness/README.md)). On the code before the revision
of 5 November, they found 241 and 153. Against the module the main evaluation ran, make-room now schedules differently
in 1,080 of the 1,264 traced configurations with the budget cap on, and in 98 of the 772 with it off, all with a KV
connector. The rounds of §6.1–§6.10 ran before both revisions, and those of §6.11 after them.

**What a step costs.** The rules run on the host, inside `Scheduler.schedule()`, which Figure 7 times on vLLM 0.32.0's
real scheduler on a 2.1 GHz Xeon Silver 4110, 1.61 times as fast as the GPU node's CPU at this work. With its switches
off the change costs what vLLM's call does, within 1.8% in the pull requests' 13 steady states and 4.6% in the serving
profile's 4. With the deployed values it changes a call by −63 µs to 1.5 ms. The step that preempts for a voice turn
takes 1.1 ms on the box's CPU, against a voice reply's decode step of 12 ms alone.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-step-cost-narrow.svg">
  <img src="figures/report-step-cost.svg" width="100%" alt="The wall time of one call of vLLM 0.32.0's Scheduler.schedule() on the host's CPU in 17 states, on a log scale, beside a band of the box's steps from a voice reply's alone (12 ms; beside held agent prefill, 16 to 20 ms) to the agents' at 24 agent sessions (38 ms), each divided by 1.61, how many times as long the box's CPU takes: the change with its switches off within 4.6% of stock vLLM in every state, and the serving profile's change to the call in each scenario, in the pull requests' cost tables: only lower-priority work (+48 to +182 µs), a waiting turn needs more room than two victims free (+326 to +1,491 µs) and every running slot taken (+158 to +299 µs); in the serving profile: only agents (+20 µs), a reply's first tokens, an agent waiting that fits (−63 µs), a waiting turn needs more room than two victims free (+78 µs) and the step that preempts for a waiting turn (+471 µs), its slowest call 3,158 µs against the band's lower edge at 7,484 µs.">
</picture>

**Figure 7. With its switches off the scheduler costs what vLLM's does, and the deployed values change a scheduler call
by −63 µs to 1.5 ms, far below the box's steps.** Each mark is a state's 10th-percentile call on a log scale, a ring for
stock vLLM, a small dot for the change with its switches off and a blue dot, labeled with what it adds, for the serving
profile's values. The gray band is the box's steps, from a voice reply's decode step alone (12 ms) to the agents' at 24
sessions (38 ms). It is divided by 1.61, the host's speed over the box's CPU at this work, so that each call stands
against the steps as it would on the box.

## 5. Evaluation setup

### 5.1 Hardware and software

The GPU node holds one RTX 5060 Ti 16 GB on a PCIe 3.0 x8 link, in a host with two 1.7 GHz Xeon E5-2609 v4 and 188 GiB
of RAM, and serves the whole cascade. Whisper large-v3-turbo [[15](#ref-15)] on vLLM transcribes the caller, Kokoro-82M
[[16](#ref-16)] speaks the reply, and vLLM 0.32.0 serves Qwen3-4B-Instruct-2507 in FP8 [[17](#ref-17)]. The engine has a
fixed 5 GiB KV pool (36,400 tokens), prefix caching, a 20 GiB CPU tier written through, priority scheduling and the
512/256 step cap. §6.7 repeats the main comparison on an A100 SXM4 80 GB, and Appendix C gives both nodes.

A second host runs the media server, the gateway and the load generators. The gateway, the control plane the deployment
runs, gives each client a class and a priority from its identity. It adds +0.12 ms to a voice request's first token at
the median, paired against a direct call ([its record](../benchmarks/results/2026-10-29-gateway-window/README.md)). The
voice agent is a LiveKit Agents worker [[13](#ref-13)] with its stock turn handling, the callers are LiveKit clients
that play scripted lines and record what they hear, and AIPerf [[14](#ref-14)] replays the agents' traces.

### 5.2 Workloads

**Voice.** Two scripted callers place calls through LiveKit, each with one unmeasured warm-up turn and 25 measured
turns, the same script in every run, in six Kokoro voices and with a barge-in on 9 of the 50 measured turns. The agent
ends a turn a fixed delay after voice activity stops, so that a turn detector's variance does not enter the comparison.
V2V is read from the waveforms at the caller on one clock, and a turn not answered within 30 s counts at 30 s.

**Agents.** AIPerf replays the SemiAnalysis Claude Code traces closed-loop through the gateway, with their sessions,
subagents and think times, the think times capped at 120 s. Each trace records a request as a list of 64-token prefix
blocks, and we shrink every block 32×, which keeps which requests share which prefixes and fits every request in the 32k
context. The gateway sends voice requests at priority 0 and agent requests at 2.

The main scenario runs 24 sessions on the 48 traces with the longest contexts (a median request of 13.9k tokens), where
running agents' KV fills the pool. A second runs 16 on the whole corpus (a median of 7.9k tokens), where the pool has
room. On tuned vLLM the agents are oversubscribed, with a median TTFT of 63 s, so their throughput measures what the GPU
leaves them.

### 5.3 Configurations

Table 1 lists what the evaluation compares, every configuration but voice alone with the agents beside the calls. Tuned
vLLM's 512-token step cap with a 256-token threshold per request bounds how long a decode waits behind a prefill chunk.
On the whole corpus at 16 sessions it lay within the noise of every other cap swept. In the campaign before the main
evaluation, beside the 24 long-context sessions, a 1,024-token cap without a threshold gave 8.79 s against 10.14 s with
9% more agent throughput, still 6.8 s over the SLO.

**Table 1. The configurations compared.** Each differs from tuned vLLM only in what the second column says. N is the KV
preemption's cap on victims a step, and T the budget cap's window in output tokens.

| Configuration | What it changes | Where |
|---|---|---|
| Voice alone | No agents; the SLO's reference | §6.1 |
| Tuned vLLM | vLLM 0.32 with priority scheduling (voice 0, agents 2), a 512-token step cap, prefix caching and a CPU KV tier (note 1) | §6 |
| The KV preemption | #63350 with N = 2 | §6.1–§6.4 |
| The change (T = 16) | #63350 and #63351 with N = 2 and T = 16, the budget cap's default window | §6.1–§6.4 |
| **The configuration to deploy** | #63350 and #63351 with N = 2 and T = 48 | §6 |
| Preempt-resubmit | A gateway that aborts agent requests without a first byte when a voice request arrives (note 2) | §6.1, §6.4, §6.5 |
| vLLM's settings | A 1,024-token step cap without a per-request threshold; a KV watermark of 10%; a limit of 8,192 queued tokens; `--max-num-seqs 12` with #54530 [[18](#ref-18)] | §6.5 |
| Static answers | A gateway that caps agent requests in flight at 2 or 8, or reserves 2,048 tokens of KV for voice | §6.5 |
| Other engines and routers | SGLang 0.5.22's priority preemption, with and without its host tier [[5](#ref-5), [6](#ref-6)]; llm-d's flow control, emulated in the gateway [[19](#ref-19)]; SMG 1.13.0 [[20](#ref-20)] | §6.5 |
| A gateway policy timed by the conversation (the lifecycle policy) | KV headroom held from the caller's speech onset, a throttle on agent prefill and a reply cap (note 3) | §6.5 |

1. The step cap has a 256-token threshold per request, and the CPU KV tier is written through.
2. It resubmits them once the voice request has its first byte.
3. The headroom is held by deferring agent requests, the throttle acts while a turn is answered, and replies are capped
   at 16 tokens when headroom is short.

### 5.4 Metrics, rounds and estimation

Each run reports the p50, p95 and p99 of V2V over its 59 replies, one for each measured turn and one for each barge-in.
The agents' token rate is the prompt and completion tokens a second of their requests that completed in the calls' first
240 s, a window of the same length for every configuration. Their latency is the p50 and p99 of their TTFT. The SLO is
voice alone's p95 V2V plus 250 ms with as many callers, 2.03 s with two callers, 2.04 and 2.13 s with four and six, and
1.67 s on the A100.

In the main scenario each configuration runs in five rounds. Round n replays its own sample of the agents, the same for
every configuration, in an order rotated from round to round. AIPerf's sequential sampler gives every round the same
traces in the same order, so a sample is a draw of where each session enters its trace. The rounds vary which part of
each trace runs beside the calls, not which traces run. Pooled over three sets of rounds on the same samples, the
configuration to deploy has 15 runs and tuned vLLM 10.

We report each configuration's median p95 over its runs with their range, the share of its replies over the SLO and the
p95 of its replies pooled, and compare configurations round by round. A p95 over 59 replies lies between the third- and
fourth-slowest reply, so single runs differ by a few hundred milliseconds. Five rounds that agree in sign would arise by
chance one time in sixteen if two configurations did not differ, so the rounds resolve the direction of a difference and
its size only roughly.

## 6. Results

Sections 6.1–6.7 measure the two rules at tuned vLLM's 512-token step cap, and Sections 6.8–6.11 what an operator
deploys around them. Appendix A maps every result to its records.

### 6.1 The two rules bring the voice tail to 2.24 s while the agents keep 97% of their throughput

With both rules and T = 48, the median p95 V2V of 15 runs is 2.24 s, against 10.48 s on tuned vLLM, and the agents
complete 3,203 tokens a second against 3,305 (Figure 8, Table 2). The configuration to deploy misses the SLO by 0.20 s
at the median, and none of its runs met it. Of its 885 replies, 286 (32%) took longer than the SLO's 2,034 ms, against
532 of 590 (90%) on tuned vLLM. Preempt-resubmit, the nearest configuration an operator can deploy today, is 0.49 s over
the SLO, and the configuration to deploy was faster than it in all five rounds on the same samples, by 223–723 ms.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-headline-narrow.svg">
  <img src="figures/report-headline.svg" width="100%" alt="Each run's voice-to-voice p95 on a log scale against the SLO of 2.03 s, and the agents' tokens a second over the calls' first 240 s, with each configuration's median: voice alone, no agents: 1.78 s at the median of 5 runs (1.76 to 1.82 s); tuned vLLM, the baseline: 10.48 s at the median of 10 runs (8.69 to 12.12 s), its agents at 3,305 tokens/s, 100% of tuned vLLM's; preempt-resubmit, before the first byte: 2.52 s at the median of 5 runs (2.52 to 2.87 s), its agents at 3,332 tokens/s, 101% of tuned vLLM's; the KV preemption, #63350: 5.09 s at the median of 5 runs (4.69 to 6.26 s), its agents at 3,414 tokens/s, 103% of tuned vLLM's; the change, 16-token window: 3.09 s at the median of 5 runs (2.87 to 3.19 s), its agents at 3,298 tokens/s, 100% of tuned vLLM's; and the configuration to deploy, 48-token window: 2.24 s at the median of 15 runs (2.13 to 2.40 s), its agents at 3,203 tokens/s, 97% of tuned vLLM's.">
</picture>

**Figure 8. The two rules bring the voice tail to 0.20 s over the SLO, and the agents keep most of their throughput.**
Two callers beside 24 coding-agent sessions. (a) Each run's p95 V2V as a tick and each configuration's median as a
marker and a label, on a log scale, against the SLO (dashed). (b) The agents' tokens per second, each run a tick, with
each median's share of tuned vLLM's.

**Table 2. The headline, beside 24 coding-agent sessions whose KV fills the pool.** The p95 V2V is the median of the
runs' p95 with their range, and the pooled p95 that of all their replies together. A reply over the SLO took longer than
2,034 ms.

| Configuration | Runs | p95 V2V | Pooled p95 | Replies over the SLO | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 5 | 1,784 ms (1,764–1,819) | 1,796 ms | 1 / 295 (0.3%) | — | — |
| Tuned vLLM | 10 | 10,477 ms (8,686–12,115) | 10,609 ms | 532 / 590 (90%) | 3,305 | 63 / 106 s |
| Preempt-resubmit | 5 | 2,522 ms (2,516–2,868) | 2,717 ms | 130 / 295 (44%) | 3,332 | 32 / 193 s |
| The KV preemption | 5 | 5,085 ms (4,693–6,261) | 5,290 ms | 263 / 295 (89%) | 3,414 | 67 / 112 s |
| The change (T = 16) | 5 | 3,094 ms (2,868–3,192) | 3,090 ms | 128 / 295 (43%) | 3,298 | 71 / 105 s |
| **The configuration to deploy** (T = 48) | 15 | **2,236 ms** (2,132–2,397) | 2,268 ms | 286 / 885 (32%) | 3,203 | 72 / 111 s |

The configuration to deploy moves the whole distribution of replies, not only its tail (Figure 9). Its median reply
takes 1.96 s and its slowest 3.71 s, where tuned vLLM's median reply takes 4.61 s and 46% of its replies take longer
than 5 s. Preempt-resubmit's replies were slower at every percentile from the median up, by 58 ms at the median and 703
ms at the 99th.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-cdf-narrow.svg">
  <img src="figures/report-cdf.svg" width="100%" alt="The cumulative share of every measured turn's voice-to-voice time, pooled over each configuration's runs, on a log scale from 1.2 to 20 s with the SLO of 2.03 s and the 95th percentile marked: voice alone, 295 replies over 5 runs: 100% within the SLO, the slowest at 2.04 s; tuned vLLM, 590 replies over 10 runs: 10% within the SLO, the slowest at 18.62 s; preempt-resubmit, 295 replies over 5 runs: 56% within the SLO, the slowest at 3.37 s; the configuration to deploy, 885 replies over 15 runs: 68% within the SLO, the slowest at 3.71 s.">
</picture>

**Figure 9. The configuration to deploy shifts every voice reply, not only the slowest.** The share of replies that
began within a given V2V, on a log scale, each line pooling a configuration's replies (295 for voice alone and for
preempt-resubmit, 590 for tuned vLLM, 885 for the configuration to deploy). The dashed line is the SLO, and the thin
line the 95th percentile.

### 6.2 The KV preemption removes the wait for KV, and the budget cap removes agent prefill from the turn's steps

Of the 3.68 s that sharing the GPU adds to the mean turn on tuned vLLM, 3.58 s fall between the end-of-turn decision and
the LLM's first sentence (Table 3). Endpointing, speech synthesis and transport move by at most 69 ms from voice alone's
in any configuration.

**Table 3. Where the voice turn's time goes, by configuration.** Mean stage times in ms over every reply, in the order a
turn passes through them, which sum to the mean V2V. The last column is the LLM's own time to its first token.

| Configuration | Endpointing | Wait for the LLM's first token | Decode of the first sentence | TTS first audio | Transport and playout | Mean V2V | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Voice alone | 756 | 85 | 218 | 211 | 352 | 1,623 | 80 |
| Tuned vLLM | 825 | 2,815 | 1,066 | 250 | 343 | 5,299 | 2,795 |
| The KV preemption | 820 | 783 | 1,207 | 250 | 356 | 3,415 | 771 |
| The change (T = 16) | 822 | 299 | 439 | 236 | 355 | 2,151 | 289 |
| **The configuration to deploy** (T = 48) | 822 | 289 | 262 | 242 | 355 | 1,971 | 282 |
| Preempt-resubmit | 822 | 283 | 390 | 232 | 357 | 2,084 | 275 |

The KV preemption takes the wait for the first token from 2,815 ms to 783 ms at the mean, as the KV case predicts
(Figure 3a). It leaves the turn's prefill and reply sharing steps with agent prefill, so the first sentence's decode
rises to 1,207 ms from 1,066 ms, and the p95 stays at 5.09 s. At T = 48 the budget cap, which also covers the turn's
prefill and KV load, takes the wait down to 289 ms and the decode to 262 ms, near voice alone's 218 ms, as the prefill
case predicts (Figure 3b). The configurations rank the same in every round.

### 6.3 A window shorter than the replies leaves their later tokens beside agent prefill

At T = 16 the p95 stays at 3.09 s (2.87–3.19 s), 1.06 s over the SLO. The first token comes as fast as at T = 48 (299
against 289 ms at the mean), but the first sentence's decode takes 439 against 262 ms. Of the replies, 79% run past 16
tokens (Figure 6a), and their later tokens decode beside agent prefill again.

A sweep of the window against 40-token replies beside closed-loop agents puts the knee where the window covers the reply
(Figure 6b). The 40th token's p95 fell from 3.76 s at a 16-token window to 1.04 s at 48 and 1.02 s at 64, while the
first token stayed at 326–448 ms. The agents' prompt throughput fell 8% from 16 to 48 tokens and no further at 64
(Figure 6c).

This voice agent is prompted for one or two short sentences, and one prompted for longer answers needs a longer window.
Asked again for 320 of its replies to the callers' lines, the engine took 15 tokens at the median to complete a reply's
first sentence, 29 at the 95th percentile and 39 at most ([the
record](../benchmarks/results/2026-10-27-first-sentence/README.md)). A 48-token window covers every first sentence, and
a 16-token window 56% of them.

### 6.4 The agents pay through the budget cap, not through the preemptions

The agents keep 97% of their token rate with the configuration to deploy (Table 4, Figure 10), 94.2–98.7% round by
round. Their median TTFT rises from 63 s to 72 s, and their p99 of 111 s lies within the spread of tuned vLLM's runs
(96–129 s).

The budget cap is where the 3% goes. It kept lower-priority prefill out of 33% of the engine's steps at T = 48 and 25%
at T = 16. With the KV preemption alone, which holds nothing, the agents completed about as many tokens as on tuned vLLM
in the same rounds, 0.1–6.2% more round by round and 0.4% more at the median.

**Table 4. What each configuration takes from the agents.** Medians over each configuration's runs. The KV preemptions
are make-room's own count, while vLLM counts a preemption, one for growth included, only once the victim produces output
again, so its count is a lower bound. The loads are the tokens the CPU tier loaded beyond what vLLM counts at a
request's first scheduling, the victims' restores among them.

| Configuration | Steps held | KV preemptions | vLLM's preemptions | CPU-tier loads | Agents' tokens/s | TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Tuned vLLM | — | — | 31 | 212,630 | 3,305 | 63 / 106 s |
| Preempt-resubmit | — | — | 7 | 273,461 | 3,332 | 32 / 193 s |
| The KV preemption | 0% | 34 | 51 | 171,736 | 3,414 | 67 / 112 s |
| The change (T = 16) | 25% | 32 | 47 | 117,049 | 3,298 | 71 / 105 s |
| **The configuration to deploy** (T = 48) | 33% | 31 | 46 | 117,044 | 3,203 | 72 / 111 s |

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-agents-narrow.svg">
  <img src="figures/report-agents.svg" width="100%" alt="For each configuration beside 24 agent sessions, each run and the median of the agents' TTFT p99, of the share of the engine's steps over the calls in which the budget cap kept agent prefill out, and of make-room's own preemptions over the calls: tuned vLLM (the baseline, 10 runs): TTFT p99 106 s (96 to 129), no budget cap, no KV preemption; preempt-resubmit (before the first byte, 5 runs): TTFT p99 193 s (160 to 232), no budget cap, no KV preemption; the KV preemption (#63350, 5 runs): TTFT p99 112 s (98 to 131), agent prefill held in 0% of the steps, 34 preemptions by make-room; the change (16-token window, 5 runs): TTFT p99 105 s (99 to 130), agent prefill held in 25% of the steps, 32 preemptions by make-room; and the configuration to deploy (48-token window, 15 runs): TTFT p99 111 s (94 to 132), agent prefill held in 33% of the steps, 31 preemptions by make-room.">
</picture>

**Figure 10. The agents' first-token tail stays at tuned vLLM's with the configuration to deploy and nearly doubles with
preempt-resubmit.** Each run a tick and each median a marker and a label. (a) The agents' p99 time to first token. (b)
The share of steps in which the budget cap kept agent prefill out. (c) The KV preemption's preemptions, as make-room
counts them. Tuned vLLM and preempt-resubmit run neither rule.

The preemptions cost the agents little. With the configuration to deploy, the KV preemption made a median of 31
preemptions a run, about one for every two voice replies and equal to about 30% of the agent requests completed during
the calls. Yet the CPU tier loaded less than on tuned vLLM, 117k tokens over a median of 46 preemptions that vLLM
counted against 213k over 31, which suggests that the victims, the latest arrivals, bring little back from the tier.

Preempt-resubmit pays differently. Aborting agent requests that have no first byte frees their KV at once and keeps 101%
of tuned vLLM's throughput. But an aborted request waits again at the front of the queue, once for each abort, so the
agents' p99 TTFT reaches 193 s (160–232 s over the runs).

### 6.5 A control plane in front of the engine makes room only by aborting agent work

No configuration without an engine change reaches the SLO beside the 24 sessions (Figure 11). These runs come from a
campaign before the main evaluation on one sample of the agents' traces, against an SLO of 2.02 s ([its
tables](../benchmarks/results/2026-10-23-e2/report/frontier.md)). Other settings of vLLM, caps on the agents'
concurrency, SGLang with or without its host tier, llm-d's flow control and SMG leave the p95 between 8.6 and 12.1 s. A
static 2,048-token reservation halves the tail to 5.67 s and stays 3.7 s over the SLO, and a limit on queued tokens left
48 of 50 voice turns unanswered.

That campaign counted each call's first turn too, and its rows with the change ran an earlier make-room (Appendix C),
whose p95 at T = 48 differed from the main evaluation's version by 8 ms at the median of five rounds.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-frontier-narrow.svg">
  <img src="figures/report-frontier.svg" width="100%" alt="Each of 18 configurations beside 24 agent sessions of the traces with the largest contexts, a row under its family, in panel a its voice-to-voice p95 on a log scale against the SLO of 2.02 s, the median of its runs and their range, and in panel b the agents' tokens a second over the calls' first 240 s, only preempt-resubmit, the lifecycle policy over make-room and the configuration to deploy within 1 s of the SLO: tuned vLLM 10.14 s at 3,552 tokens/s; 1,024-token steps 8.79 s at 3,888 tokens/s; a KV watermark 10.65 s at 3,649 tokens/s; a queued-token limit at least 30 s, 48 turns failed, at 5,584 tokens/s; a token reservation 5.67 s at 3,551 tokens/s; a static cap (2 agent requests in flight) 8.57 s at 3,446 tokens/s; a static cap (8 agent requests in flight) 9.88 s at 3,586 tokens/s; vLLM with #54530 10.93 s at 3,624 tokens/s; SGLang 0.5.22 12.05 s at 3,577 tokens/s; SGLang with HiCache 11.95 s at 4,330 tokens/s; llm-d's flow control 10.99 s at 3,684 tokens/s; SMG 1.13.0 11.05 s at 3,446 tokens/s; preempt-resubmit 2.59 s at 3,593 tokens/s; the lifecycle policy over make-room 2.06 s at 3,355 tokens/s; the lifecycle policy alone on tuned vLLM 10.99 s at 3,633 tokens/s; the KV preemption 5.77 s at 3,572 tokens/s; the change 3.23 s at 3,540 tokens/s; and the configuration to deploy 2.19 s at 3,362 tokens/s.">
</picture>

**Figure 11. No configuration without the change comes within a second of the SLO except by aborting agent work.** Each
configuration is a row under its family, on one sample of the agents' traces, with (a) its median p95 V2V and a line
over the range of its runs, and (b) its agents' tokens per second. Gray is vLLM without the change and every comparator,
open for tuned vLLM and preempt-resubmit. Blue is the configuration to deploy and a fainter blue the change's other
configurations on the earlier make-room, the open diamond among them the lifecycle policy over it with the KV reserve on
and replies capped at 16 tokens. The dashed line is the SLO, 2.02 s, and unanswered turns count at 30 s.

A gateway policy timed by the conversation, the lifecycle policy, could not act here. It holds KV headroom for a call
from the caller's speech onset by deferring new agent requests, throttles agent prefill while a turn is answered, and
caps a reply at 16 tokens when headroom is short. On tuned vLLM it left the p95 at 10.99 s, against 10.14 s for tuned
vLLM alone on the same sample. Deferring new requests frees nothing that running ones hold, so the headroom was never
there and the policy capped all 60 of its realtime requests in every run.

Over the earlier make-room, with the budget cap at T = 16 and a 2,048-token KV reserve, the same policy reached 2.06 s.
The reserve kept the pool from filling, so the KV preemption never fired, and every reply was capped to fit the window.
Without the reply cap the p95 rose to 3.84 s and without the budget cap to 5.29 s, while leaving out the headroom, the
throttle or the lifecycle events changed nothing beyond the policy's own spread. The room came from the engine's
reserve, and the policy added a shorter reply.

Preempt-resubmit is the one control-plane answer that makes room, and it does so by aborting admitted work (§6.4). A
control plane can hold back work that has not started and end work that has, losing what it computed. Only the engine
can preempt a decoding request and keep its work, or keep agent prefill out of a reply's steps.

Knowing what the conversation will do next found no use on either side. For the voice turn the headroom could not act.
In the repository's simulator, knowing exactly when each agent returns shortens its median resume by under 8%, and its
p90 by up to 29% only at low pressure ([the simulator's result](BENCHMARKS.md#does-knowing-when-an-agent-returns-help)).

### 6.6 The gain shrinks where the pool has room, and the tail grows as calls overlap

Where the pool has room, the KV preemption has little to do. With 16 sessions of the whole corpus it preempted a median
of 1 agent request a run, and the budget cap kept agent prefill out of 4% of the steps. The configuration to deploy had
the lower p95 in every round, by 109–1,413 ms, but met the SLO in only one (Figure 12a). Its p95 was 2.05 s, 15 ms over
the SLO at the median, against tuned vLLM's 2.52 s, at 6,815 agents' tokens a second against 7,131. The gain of 0.47 s
compares with 8.24 s beside the long-context sessions.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-margin-narrow.svg">
  <img src="figures/report-margin.svg" width="100%" alt="In panel a, with 16 agent sessions of the whole corpus, where the pool has room, tuned vLLM's V2V p95 and the configuration to deploy's in each of five rounds against the SLO of 2.03 s (2.15 and 2.04 s, 3.83 and 2.42 s, 3.07 and 2.12 s, 2.30 and 2.05 s, 2.52 and 2.02 s); in panels b to d, the configuration to deploy with two, four and six callers beside 24 agent sessions, on the same three samples of the agents, each run's p95 over that many callers' SLO, the share of the engine's steps in which the budget cap held agent prefill, and the agents' tokens a second over the calls' first 240 s (2 callers, +202 ms over their SLO of 2.03 s at the median of 9 runs (+98 to +267), agent prefill held in 33% of the steps, the agents at 3,156 tokens/s; 4 callers, +269 ms over their SLO of 2.04 s at the median of 3 runs (+206 to +370), agent prefill held in 53% of the steps, the agents at 3,055 tokens/s; 6 callers, +257 ms over their SLO of 2.13 s at the median of 3 runs (+226 to +333), agent prefill held in 64% of the steps, the agents at 2,716 tokens/s).">
</picture>

**Figure 12. The configuration to deploy helps less where the pool has room, and overlapping calls hold the agents'
prefill for longer.** (a) Sixteen sessions of the whole corpus, each round's p95 V2V on tuned vLLM (gray) and with the
configuration to deploy (blue), joined by round. (b–d) The configuration to deploy beside the 24 sessions with 2, 4 and
6 callers, with the p95 over each caller count's SLO (2.03, 2.04 and 2.13 s), the share of steps held, and the agents'
tokens per second. Ticks are runs and markers their medians.

More calls make the budget cap hold longer, because it keeps lower-priority prefill out while any call's first tokens
are due (Figure 12b and 12c). With 4 and 6 callers the configuration to deploy missed each caller count's own SLO in
every round, by 269 and 257 ms at the median, against 202 ms with two on the same samples. The held steps grew to 53%
and 64% from 33%, and the agents' throughput fell 3% and 14%.

The limit is where the windows of successive replies overlap. A test on vLLM's scheduler on CPU sends replies of at
least 48 tokens at a fixed rate beside six agents' prefills. The agents kept 82% of their prompt tokens at one reply
every 200 steps and 9% at one every 50, and none at one every 40, where each reply's window of about 49 steps overlaps
the next. At 16–20 ms a step, that is about one realtime reply a second across all calls.

### 6.7 The rules carry to an A100

An A100 SXM4 80 GB ran the main comparison like for like on the same five samples of the agents, with the same model, KV
pool, step cap, CPU tier, 24 sessions and two callers. Only the card, its precision path and the host changed, and the
model ran on Marlin's weight-only FP8 kernels [[21](#ref-21)], since the A100 has no FP8 units.

Voice alone gave a p95 of 1.42 s, so this card's SLO is 1.67 s. The configuration to deploy took the p95 from 4.99 s to
1.63 s, under the SLO in all five rounds by 7–60 ms (Figure 13a, Table 5), with the agents' throughput at 94–111% of
tuned vLLM's round by round.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-a100-narrow.svg">
  <img src="figures/report-a100.svg" width="100%" alt="On the A100 beside 24 agent sessions: in panel a, at the RTX 5060 Ti's 512-token steps, tuned vLLM's V2V p95 and the configuration to deploy's in each of five rounds against the card's SLO of 1.67 s (4.16 and 1.66 s, 4.78 and 1.66 s, 4.99 and 1.61 s, 5.17 and 1.63 s, 5.71 and 1.61 s); in panel b, the agents' tokens a second over the calls' first 240 s against the V2V p95, each run and the medians, an arrow from tuned vLLM's median to the configuration to deploy's at the same step cap (at 512-token steps tuned vLLM 4.99 s at 8,633 tokens/s, the configuration to deploy 1.63 s at 8,651 tokens/s, medians of 5 runs each; at 2,048-token steps tuned vLLM 3.66 s at 14,536 tokens/s, the configuration to deploy 1.83 s at 14,635 tokens/s, medians of 2 runs each).">
</picture>

**Figure 13. On an A100 the rules meet the card's SLO in every round at the RTX 5060 Ti's 512-token steps, and at
2,048-token steps they trade 0.19 s of the voice's tail for 69% more agent throughput.** An A100 SXM4 80 GB at the RTX
5060 Ti's KV pool. (a) Each round's p95 V2V at 512-token steps on tuned vLLM (gray) and with the configuration to deploy
(blue), joined by round, against the card's SLO of 1.67 s. (b) The p95 V2V against the agents' tokens per second at 512-
and 2,048-token steps, small marks runs and large marks medians, an arrow joining tuned vLLM's median to the
configuration to deploy's at each step cap. In both panels tuned vLLM's marks are open and the configuration to deploy's
filled.

**Table 5. The A100 SXM4 80 GB at the RTX 5060 Ti's KV pool, beside the same 24 sessions.** Medians over the runs, with
the range of the p95 V2V. The 2,048-token rows run vLLM's default steps, 2,048 tokens with no per-request threshold, on
the first two samples of the agents only.

| Configuration | Runs | p95 V2V | Over the SLO | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|
| Voice alone | 5 | 1,419 ms (1,391–1,445) | — | — | — |
| Tuned vLLM, 512-token steps | 5 | 4,986 ms (4,163–5,710) | +3,317 ms | 8,633 | 12 / 26 s |
| **The configuration to deploy**, 512-token steps | 5 | **1,634 ms** (1,609–1,663) | −35 ms | 8,651 | 14 / 26 s |
| Tuned vLLM, 2,048-token steps | 2 | 3,663 ms (3,289–4,036) | +1,994 ms | 14,536 | 1.5 / 14 s |
| The configuration to deploy, 2,048-token steps | 2 | 1,826 ms (1,810–1,843) | +157 ms | 14,635 | 1.5 / 15 s |

The cards differ in how long a turn waits once KV is short, more than in how often. Beside tuned vLLM the engine's free
KV was under two calls' 2,048 tokens 5.7% of the time on the A100 and 8.1% on the RTX 5060 Ti, but in the KV case the
A100's turn waited 1.75 s against 4.9–5.0 s. A forecast that scaled the RTX 5060 Ti's measured costs by the A100's
published peaks underestimated that wait by about 30%, because the A100's decode attention read KV at 42% of its memory
bandwidth ([the run against its forecast](../benchmarks/results/2026-10-26-a100/a100.md)).

The step cap trades the voice's tail against the agents' throughput (Figure 13b). At vLLM's default of 2,048 tokens a
step with no threshold, tuned vLLM gave the agents 1.40 and 1.42 times their throughput at 512-token steps on two
samples, with the p95 at 4.04 and 3.29 s. With the rules at that cap the agents kept 100% and 101% of it, and the p95
was 1.81 and 1.84 s, 0.14 and 0.17 s over the SLO. A turn that arrives while longer steps are in flight waits for them.

At a datacenter batch of 256 short requests, 26 at priority 0, the switches off made 1.018 times vLLM's output tokens a
second. The deployed values cut the priority-0 requests' p99 TTFT from 574 to 177 ms, while the other 230 requests'
output fell 8.8% and their p99 TTFT rose from 1.1 s to 8.0 s. Both costs come from the budget cap holding prefill, at a
rate of first tokens that no voice tier reaches (§6.6).

What carries to another card is the mechanism and the regime's sign, not the length of a wait. The repository's
simulator places an L4, an A100 40 GB and an H100 SXM on a map of KV pressure, pause and restore cost only as a model,
from the RTX 5060 Ti's costs scaled by published peaks ([other hardware,
modeled](../crates/flatshare-sim/README.md#other-hardware-modeled)). It has running agents leave two calls short of KV
at most 1% of the time up to 24 sessions on the L4 and the A100 and never on the H100. At the RTX 5060 Ti's pool it gave
17% and 10% where the two cards measured 8.1% and 5.7%.

### 6.8 On two cards, consolidating both calls on one does 34% more background work, most of it without the change

A second RTX 5060 Ti 16 GB runs on another host. An operator with two cards can give the calls one and the agents the
other (dedicated roles), or run both mixed with one call each (pooling). Consolidation puts both calls on one card
beside as many agent sessions as it serves within the SLO, and agents alone on the other.

A resource gate, its rules fixed before its runs, decides between them
([RESOURCE-GATE.md](../benchmarks/quality/RESOURCE-GATE.md)). An arrangement qualifies if at most 5% of its replies, its
rounds pooled, are over its round's SLO. Sharing has resource value if a qualifying shared arrangement does at least 20%
more background work than the best qualifying dedicated one, counted as the agents' tokens a second over the calls'
first 240 s summed over both cards. Three rounds each replayed a new sample of the agents, with the configuration to
deploy at the serving profile's 256-token step cap (§6.9).

A card of agents alone does the most well short of 24 sessions. With its KV pool grown into the speech servers' memory
to 61,152 tokens and on vLLM's default scheduling, it did 7,893 tokens a second at 14 and 4,232 at 24, where its pool
thrashes. The gate takes each dedicated card at its busiest count.

Consolidation on the configuration to deploy qualifies with four sessions on the shared card, 5 of 177 replies over the
SLO, against 31 at eight (Figure 14a). Beside the agents-only card at 14 sessions it did 10,596 tokens a second, 34.2%
more than the best dedicated arrangement, past the hurdle (Table 6, Figure 14b).

Most of that gain is the arrangement's. Tuned vLLM's consolidated card qualifies at two sessions, with 4 of 177 replies
over and 18 at four, and that arrangement did 27.3% more than the best dedicated one. The change's own value, the two
sessions more at which it qualifies, is therefore 5.4%, short of the hurdle. Pooling qualifies on neither engine, with
24 of 87 replies over the one-caller SLO on the configuration to deploy and 53 on tuned vLLM.

**Table 6. The two cards' arrangements.** Three rounds, each on a new sample of the agents' traces. Replies over the SLO
are pooled over the rounds, and the agents' tokens a second are summed over both cards at the median of the rounds. The
shared card's sessions come first.

| Arrangement | Agent sessions | Replies over the SLO | Agents' tokens/s | Over the best dedicated | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|
| Dedicated roles, controlled | 0 + 12 | 0 / 177 | 6,302 | — | 4.4 / 31.0 s |
| **Dedicated roles, best practical** | 0 + 14 | 0 / 177 | **7,893** | — | 5.1 / 41.0 s |
| Pooling, tuned vLLM | 12 + 12 | 53 / 87 | 13,188 | does not qualify | 4.7 / 29.2 s |
| Pooling, the configuration to deploy | 12 + 12 | 24 / 87 | 12,111 | does not qualify | 4.8 / 37.2 s |
| Consolidation, tuned vLLM | 4 + 14 | 18 / 177 | 10,596 | does not qualify | 0.6 / 10.4 s |
| Consolidation, tuned vLLM | 2 + 14 | 4 / 177 | 10,051 | +27.3% | 0.4 / 7.8 s |
| Consolidation, the configuration to deploy | 8 + 14 | 31 / 177 | 13,537 | does not qualify | 0.7 / 11.7 s |
| **Consolidation, the configuration to deploy** | 4 + 14 | 5 / 177 | **10,596** | **+34.2%** | 0.6 / 10.8 s |
| Consolidation, the configuration to deploy, with a warm spare | 4 + 12 | 5 / 177 | 9,652 | +22.3% | 0.5 / 10.7 s |

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-cards-narrow.svg">
  <img src="figures/report-cards.svg" width="100%" alt="In panel a, the share of the consolidated card's replies over their round's SLO, the 3 rounds pooled, by its agent sessions beside both calls, against the requirement of at most 5%: at 2, 4, 8, 12 and 24 sessions, 2.3%, 10.2%, 39.0%, 53.7% and 92.1% on tuned vLLM and 1.1%, 2.8%, 17.5%, 18.6% and 20.9% on the configuration to deploy; in panel b, each arrangement's agents' tokens a second over the calls' first 240 s summed over both cards, at the median of the 3 rounds, against 9,471, 20% over the best dedicated arrangement: dedicated roles, best practical, 0 + 14 sessions: 7,893; dedicated roles, controlled, 0 + 12 sessions: 6,302; consolidation on tuned vLLM, 2 + 14 sessions: 10,051; consolidation on the configuration to deploy, 4 + 14 sessions: 10,596; consolidation on the configuration to deploy with a warm spare, 4 + 12 sessions: 9,652; consolidation on the configuration to deploy, 8 + 14 sessions: 13,537, missing the requirement; pooling on tuned vLLM, 12 + 12 sessions: 13,188, missing the requirement; pooling on the configuration to deploy, 12 + 12 sessions: 12,111, missing the requirement.">
</picture>

**Figure 14. On two cards, consolidating both calls on one card does the most background work within the SLO, on either
engine.** (a) The consolidated card's replies over the SLO by its agent sessions, three rounds pooled, on tuned vLLM
(gray) and with the configuration to deploy (blue), against the 5% requirement (dashed). (b) Each arrangement's agents'
tokens per second summed over both cards, bars the median of the rounds and ticks the rounds. Arrangements that miss the
requirement are hatched, and the dashed line is 20% over the best dedicated arrangement.

The SLO's 250 ms margin sets the envelope. With 500 ms the consolidated card would qualify with the configuration to
deploy at every count measured, up to 24 sessions, and with tuned vLLM still only at two. From eight sessions on, the
transcript comes 51–71 ms later on either engine, as the agents' steps take the GPU from the speech server. With the
configuration to deploy, the transcript and the first sentence then take 176–249 ms longer than with the calls alone,
nearly all of the margin ([the
record](../benchmarks/results/2026-10-28-rounds/README.md#beside-the-gate-where-the-replies-spend-their-time)).

The gate counts no standby. A warm spare keeps the speech servers resident on the second card so that it can take the
calls, with its agents on the configuration to deploy at 12 sessions. Consolidation at four sessions with a warm spare
did 9,652 tokens a second, 22.3% more than the best dedicated arrangement, which has no standby (§7, Appendix D).

### 6.9 A 256-token step cap halves the change's replies over the SLO beside 24 sessions

The evaluation of §6.1–§6.7 ran the configuration to deploy at tuned vLLM's step cap of 512 tokens with a 256-token
threshold per request. A sweep of caps, windows and static caps on agent requests, one run a setting, found that the cap
moved it most ([the sweep's record](../benchmarks/results/2026-10-27-sweep/README.md)). At 256 tokens without the
threshold, 9 of 59 replies were over the SLO, against 24 at 512 and 33 at 1,024.

Five rounds on new samples ran three configurations side by side at 24 sessions, each against its round's own SLO
(Figure 15, Table 7). At 256 the configuration to deploy had fewer replies over the SLO than at 512 in each of the five
rounds, 44 of 295 against 94 (14.9% against 31.9%), with a pooled p95 of 2.15 s against 2.26 s. Against both, tuned vLLM
had 261 of 295 over. The smaller steps cost the agents 7% of their token rate, 3,067 tokens a second against 3,303, and
their median TTFT rose from 78 s to 85 s. No cap brought 24 sessions within the 5% allowance, which the shared card
meets at four (§6.8).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-step-cap-narrow.svg">
  <img src="figures/report-step-cap.svg" width="100%" alt="In panel a, the share of each of five rounds' replies over that round's SLO beside 24 agent sessions, against the requirement of at most 5%, and in panel b each round's agents' tokens a second over the calls' first 240 s with their median: tuned vLLM, 512-token steps, 53, 55, 55, 54, 44 of 59 by round, 261 of 295 in all, at 3,399 agents' tokens a second; the configuration to deploy, 512-token steps, 16, 29, 25, 9, 15 of 59 by round, 94 of 295 in all, at 3,303 agents' tokens a second; the configuration to deploy, 256-token steps, 10, 17, 10, 4, 3 of 59 by round, 44 of 295 in all, at 3,067 agents' tokens a second.">
</picture>

**Figure 15. At 256 tokens a step the configuration to deploy had fewer replies over the SLO than at 512 in every round,
and no configuration met the 5% requirement in any.** (a) Each round's share of its 59 replies over that round's SLO,
beside 24 coding-agent sessions; the dashed line is the requirement. (b) The agents' tokens a second over the calls'
first 240 s, each round a tick and the median a marker. The 512-token configurations keep the 256-token threshold per
request.

The smaller steps also left the pool room. At 256 tokens its fewest free blocks in a run were 10–89, against 0–2 at 512,
and vLLM counted 0–4 preemptions a run against 39–50. In the sweep's run at 256 it counted none, so there the 9 replies
over the SLO, against stock vLLM's 51 of 59 at the same cap, are the budget cap's. Beside the envelope's four sessions
(§6.8) the KV preemption took no victim in any round. The serving profile runs the change at 256 tokens a step without a
threshold and keeps the KV preemption for a pool that fills.

**Table 7. The step cap beside 24 sessions, in five rounds on new samples of the agents.** Each reply is held to its
round's SLO, and the replies over it are pooled over the rounds and given by round. The 512-token rows keep the
256-token threshold per request, and the 256-token row has none. Medians over the rounds otherwise.

| Configuration | Median p95 V2V | Pooled p95 | Replies over the SLO | By round | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---|---:|---:|
| Tuned vLLM, 512-token steps | 12,010 ms | 11,766 ms | 261 / 295 | 53, 55, 55, 54, 44 | 3,399 | 69 / 109 s |
| The configuration to deploy, 512-token steps | 2,246 ms | 2,259 ms | 94 / 295 | 16, 29, 25, 9, 15 | 3,303 | 78 / 117 s |
| **The configuration to deploy, 256-token steps** | **2,098 ms** | 2,151 ms | **44 / 295** | 10, 17, 10, 4, 3 | 3,067 | 85 / 120 s |

### 6.10 The serving engine's additions cut the replies over the SLO by more than half

The serving profile sets five additions to the engine around the two rules, each a switch of make-room or of the serve
module. Each was found in the scheduler's own traces and measured against the configuration before it with a control
([the engine points](../benchmarks/results/2026-11-01-engine-points/README.md)).

1. **The calls' prefixes kept on the GPU** (`retain_tokens` 3,584, `retain_s` 30), so that about half the turns no
   longer reload their prefix from the CPU tier, a load of two or three steps at any size.
2. **The next batch just in time** (`flatshare_jit`, a 30 ms margin). Asynchronous scheduling commits the next step
   before the one in flight ends, so the serve module schedules it when that step is due to end, and a call that arrives
   meanwhile joins it.
3. **Each turn announced** (`hint_hold_ms` 400, the voice worker's `turn_hints`). At the caller's end of speech a
   one-token realtime request holds agent prefill back for 400 ms, so the turn arrives into short steps.
4. **A call never waits behind an agent's reservations** (`take_reservations`). Calls waited 1.4–4.5 s while agents'
   prefix loads held the free blocks, so a request now counts only the reservations of its own priority or higher.
5. **The agents' decodes held through the budget cap's window** (`decode_hold_tokens` 48). A reply's step that also
   decodes an agent at 15–16k tokens of context takes 18–20 ms against 13 ms alone, so lower-priority decodes sit out a
   reply's first 48 tokens, as their prefill does.

Table 8 gives the engine points in the order the additions came, with two calls of 48-token replies beside 12
closed-loop agents whose prompts grow from 6,000 to 30,000 tokens. Together the additions take a reply's first token
from 354 to 88 ms at p95 and its 48th token, the whole reply, from 1,226 to 722 ms. The agents give up 4% of their token
rate (2,604 against 2,718 tokens a second). Without the decode hold the 48th token is 998 ms.

**Table 8. The serving engine's additions at the engine level, beside 12 agents.** Medians of the runs, from the calls'
own clocks and the scheduler's trace. The announced turn's row adds it to retention, the next adds just in time to both,
and the last two add the decode hold.

| Configuration | First token p50 / p95 | 16th token p50 / p95 | 48th token p95 | Agents' tokens/s |
|---|---:|---:|---:|---:|
| v1.14's serving profile | 175 / 354 ms | 408 / 597 ms | 1,226 ms | 2,718 |
| + the calls' prefixes kept on the GPU | 122 / 230 | 354 / 502 | 1,136 | 2,761 |
| + the next batch just in time | 98 / 185 | 323 / 449 | 1,050 | 2,675 |
| retention + each turn announced | 64 / 90 | 258 / 394 | 1,004 | 2,637 |
| + just in time too | 63 / 98 | 257 / 403 | 982 | 2,628 |
| + the agents' reservations taken | 64 / 92 | 261 / 392 | 998 | 2,625 |
| + their decodes held for 16 tokens | 64 / 94 | 258 / 285 | 966 | 2,615 |
| + held for 48 tokens, the window: the profile | 66 / 88 | 257 / 284 | 722 | 2,604 |

Five rounds on new samples ran the configuration to deploy with and without the first four additions, beside 24 agent
sessions with two, four and six callers and beside 16 sessions of the whole corpus (Figure 16, Table 9). The additions
had fewer replies over the SLO in every round of every scenario. Pooled, 5.4%, 6.4% and 5.8% of the replies were over
beside 24 sessions, against 18.6%, 18.5% and 18.0% without them on these samples, and 3.4% beside 16 sessions, within
the 5% allowance, against 7.5%. The agents gave up 1.0–4.9% of their token rate.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-opt-rounds-narrow.svg">
  <img src="figures/report-opt-rounds.svg" width="100%" alt="In panel a, each scenario's replies over their round's SLO in 5 rounds, v1.14's configuration to deploy and the same with the first four of version 1.15's additions to the engine, each round a tick and the rounds pooled a marker, against the requirement of at most 5%, and in panel b the agents' tokens a second over the calls' first 240 s, each round a tick and their median a marker: 24 agent sessions, 2 callers, v1.14's configuration to deploy 10, 13, 10, 6, 16 by round, 55 of 295 in all, at 3,010 agents' tokens a second, the same with the first four additions 7, 1, 4, 4, 0 by round, 16 of 295 in all, at 2,933 agents' tokens a second; 24 agent sessions, 4 callers, v1.14's configuration to deploy 19, 25, 29, 12, 25 by round, 110 of 595 in all, at 2,753 agents' tokens a second, the same with the first four additions 5, 10, 9, 6, 8 by round, 38 of 595 in all, at 2,754 agents' tokens a second; 24 agent sessions, 6 callers, v1.14's configuration to deploy 23, 56, 27, 29, 29 by round, 164 of 910 in all, at 2,580 agents' tokens a second, the same with the first four additions 3, 20, 10, 9, 11 by round, 53 of 910 in all, at 2,476 agents' tokens a second; 16 agent sessions of the whole corpus, 2 callers, v1.14's configuration to deploy 5, 3, 3, 9, 2 by round, 22 of 295 in all, at 4,948 agents' tokens a second, the same with the first four additions 2, 2, 1, 5, 0 by round, 10 of 295 in all, at 4,900 agents' tokens a second.">
</picture>

**Figure 16. In every scenario the first four additions put fewer than half as many of the pooled replies over the SLO,
and beside 16 sessions they met the 5% requirement.** (a) Each round's share of its replies over that round's SLO and
the five rounds pooled, beside 24 coding-agent sessions with two, four and six callers and beside 16 sessions of the
whole corpus; the dashed line is the requirement. (b) The agents' tokens a second over the calls' first 240 s, each
round a tick and the median a marker.

**Table 9. v1.14's configuration to deploy and the same with the first four additions, in five rounds on new samples of
the agents.** Each reply is held to its round's SLO, voice alone with as many callers plus 250 ms, and the replies over
it are pooled over the rounds and given by round. Medians over the rounds otherwise.

| Scenario | Configuration | Replies over the SLO | By round | Median p95 V2V | LLM first token p95 | Agents' tokens/s |
|---|---|---:|---|---:|---:|---:|
| 24 agent sessions, 2 callers | v1.14's configuration to deploy | 55 / 295 (18.6%) | 10, 13, 10, 6, 16 | 2,176 ms | 314 ms | 3,010 |
| 24 agent sessions, 2 callers | + the first four additions | **16 / 295 (5.4%)** | 7, 1, 4, 4, 0 | 2,023 ms | 136 ms | 2,933 |
| 24 agent sessions, 4 callers | v1.14's configuration to deploy | 110 / 595 (18.5%) | 19, 25, 29, 12, 25 | 2,274 ms | 334 ms | 2,753 |
| 24 agent sessions, 4 callers | + the first four additions | **38 / 595 (6.4%)** | 5, 10, 9, 6, 8 | 2,106 ms | 139 ms | 2,754 |
| 24 agent sessions, 6 callers | v1.14's configuration to deploy | 164 / 910 (18.0%) | 23, 56, 27, 29, 29 | 2,320 ms | 325 ms | 2,580 |
| 24 agent sessions, 6 callers | + the first four additions | **53 / 910 (5.8%)** | 3, 20, 10, 9, 11 | 2,152 ms | 139 ms | 2,476 |
| 16 agent sessions of the whole corpus, 2 callers | v1.14's configuration to deploy | 22 / 295 (7.5%) | 5, 3, 3, 9, 2 | 2,012 ms | 255 ms | 4,948 |
| 16 agent sessions of the whole corpus, 2 callers | + the first four additions | **10 / 295 (3.4%)** | 2, 2, 1, 5, 0 | 1,916 ms | 125 ms | 4,900 |

Two more rounds set the decode hold at 48, and speculative decoding on top of it, against the first four additions with
two, four and six callers. The decode hold put 16 of 720 replies over the SLO against 55, fewer in five of the six
rounds and caller counts. But it cost the agents 3.8% of their token rate at the median of the six, past the 3% that a
rule fixed before the rounds allowed. The serving profile takes it for the voice tail. Speculative decoding put 12 over
against the decode hold's 16, fewer in three of the six where the rule asked for five, and stays a switch ([the
rounds](../benchmarks/results/2026-11-02-opt-rounds/README.md)). Synchronous scheduling, full CUDA graphs through Triton
attention, an FP8 KV cache and a second API server were measured and not adopted ([what was
not](../benchmarks/results/2026-11-01-engine-points/README.md#what-was-not)).

### 6.11 Ordering the agents by what they have left raises the card's envelope from 4 sessions to 24

Beside 24 agent sessions the CPU tier stops working as a cache. Each session's next turn is its context so far plus a
little, and the waiting queue takes requests in arrival order. Under load a request waits longer than the 20 GiB LRU
tier keeps its session's prefix, so it computes the whole context again, which makes the next request wait longer. In
one window 89.5% of the prompt tokens were computed at their first schedule, and the agents' median time to first token
was 77 s (Figure 17a). A 30 GiB tier still computed 68.0%.

The serve module orders the waiting queue within a priority by the prefill each request has left, net of what the GPU
pool and the CPU tier hold now. The sessions whose prefix is still cached go first, which keeps it cached. A request
that has waited 120 s goes ahead of the rest (`queue_max_wait_s` 120), and the calls, at priority 0, keep their arrival
order. Each step refreshes at most two requests' counts, and none while a call waits or runs, since a count costs the
host 1.0–2.5 ms.

The queue alone took the agents' token rate from 3,173 to 7,270 tokens a second, with 20% of the prompt tokens computed.
But it put the calls' V2V p95 at 2,025 ms against 1,843, because it loads far more from the tier over the card's PCIe
3.0 x8 link. Beside a 1.46 GB tier load on the copy engine, a step's small input copy waits for the load to end, and the
slowest sampled step took 192–209 ms. vLLM's own Triton kernel for small pages moves the same load at 6.1 GB/s with 4
programs and leaves the step beside it at 6.1 ms against 5.7 ms alone (Figure 17b). The serve module sends the tier's
loads through that kernel, its size cutoff lifted for the box's 64 KiB pages (`sm_load_programs` 4, [the
record](../benchmarks/results/2026-11-08-envelope/README.md)).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-tier-narrow.svg">
  <img src="figures/report-tier.svg" width="100%" alt="In panel a, where vLLM found the prompt tokens at their first schedule beside 24 agent sessions on the second GPU node, as shares of them, with the agents' tokens a second over the calls' first 240 s and their median time to first token: first come first served over a 20 GiB CPU tier, 9.1% found in the GPU pool, 1.5% loaded from the CPU tier and 89.5% computed, at 3,173 agents' tokens a second and a median time to first token of 76.8 s; first come first served over a 30 GiB CPU tier, 8.3%, 23.7% and 68.0%, at 2,958 and 76.7 s; and the queue over a 20 GiB CPU tier, 5.9%, 73.9% and 20.2%, at 7,270 and 4.5 s; in panel b, a step's time beside one 1.46 GB load from the CPU tier on that node's bare card, 5.7 ms alone at the median of the steps before each load, and the longest during it 192 ms on the copy engine in one batch of 32 KiB pages, 209 ms in one of 2 MiB pages, 152 ms in 23 jobs of 64 MB, 136 ms in 92 jobs of 16 MB on chained streams, and 6.1 ms through vLLM's Triton kernel with 4 programs.">
</picture>

**Figure 17. First come first served computed most of the prompt tokens and the queue a fifth, and beside a load from
the CPU tier a step took up to 209 ms on the copy engine and 6.1 ms through the SM copy.** (a) Where vLLM found the
prompt tokens at their first schedule beside 24 agent sessions, in one window on one sample of the agents, with the
agents' tokens a second over the calls' first 240 s and their median time to first token beside each bar. (b) A step's
time beside one 1.46 GB load from the tier on the bare card, in 32 KiB pages unless marked. Alone is the median of the
ten steps before each load, the copy engine the longest of the two (one batch) or four (chained jobs) steps during it,
and vLLM's Triton kernel the longest of 22.

Three rounds then measured the card's envelope beside both calls by the resource gate's rule, without the queue and the
copy at 4 sessions and with them at 12 and 24 (Table 10).

**Table 10. The card's envelope beside both calls, three rounds.** Each reply is held to its round's SLO, voice alone's
p95 plus 250 ms, and the replies over it are pooled over the rounds; a configuration qualifies at 5% or fewer. Medians
over the rounds otherwise, with the agents' tokens a second over the calls' first 240 s.

| Configuration | Agents' sessions | Replies over the SLO | V2V p95 | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|
| The serving profile through v1.23 | 4 | 4 / 177 (2.3%) | 1,934 ms | 3,278 | 0.3 / 8 s |
| + the queue and the copy | 12 | 4 / 177 (2.3%) | 1,905 ms | 6,906 | 2.3 / 24 s |
| + the queue and the copy | **24** | **3 / 177 (1.7%)** | 1,872 ms | **7,253** | 4.3 / 133 s |

Beside 24 sessions the queue and the copy qualify, with 2.2 times the agents' token rate of the profile at 4 (Figure
18), so the serving profile sets the envelope, `max_agents`, to 24. The agents' tail pays, since a request can wait out
the 120 s bound, and their p99 time to first token is 133 s against 8 s at 4 sessions. Both parts are proposed to vLLM,
as cache-aware admission that counts the CPU tier's hits [[62](#ref-62)] and as the tier's loads through the Triton
kernel [[63](#ref-63)].

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-envelope-narrow.svg">
  <img src="figures/report-envelope.svg" width="100%" alt="In panel a, each configuration's replies over their round's SLO in version 1.24's 3 envelope rounds, each round a tick and the rounds pooled a marker, against the requirement of at most 5%, in panel b the agents' tokens a second over the calls' first 240 s and in panel c their time to first token at p99, each round a tick and their median a marker: the serving profile through v1.23 beside 4 agent sessions, 0, 2, 2 by round, 4 of 177 in all, its agents at 3,278 tokens a second (2,703 to 3,598) and a p99 of 8 s (6 to 8); the queue and the copy beside 12 agent sessions, 2, 2, 0 by round, 4 of 177 in all, its agents at 6,906 tokens a second (5,949 to 7,090) and a p99 of 24 s (19 to 73); and beside 24, 0, 3, 0 by round, 3 of 177 in all, its agents at 7,253 tokens a second (4,490 to 8,094) and a p99 of 133 s (132 to 182).">
</picture>

**Figure 18. Beside 24 sessions the queue and the copy kept the pooled replies over the SLO within the 5% requirement,
at 2.2 times the agents' token rate of the serving profile at 4, and the agents' first-token tail carries the cost.**
(a) Each round's share of its replies over that round's SLO, voice alone's p95 plus 250 ms, and the three rounds pooled;
the dashed line is the requirement. (b) The agents' tokens a second over the calls' first 240 s and (c) their time to
first token at p99, each round a tick and the median a marker.

## 7. Operating the change

The configuration to deploy runs the KV preemption at N = 2 and the budget cap at T = 48, at 256 tokens a step without a
per-request threshold. It needs priority scheduling, prefix caching and a CPU KV tier about four times the GPU pool,
written through. The gateway sets voice requests to priority 0 and background work to 2 from each client's identity, so
that a background client cannot raise its own priority. The serving profile adds the switches of §6.10 and §6.11, each
off by default, and [PRODUCTION.md](PRODUCTION.md) gives the whole deployment.

On two cards the arrangement to deploy is consolidation with a warm spare (§6.8). The first card carries both calls and
its envelope of agent sessions, and the second runs agents with its speech servers resident. When the first card fails,
its calls move to the second by their next turn, and the second holds its agents' new requests while it carries any
call. Over a 24-hour soak 1.5% of 5,058 replies were over the SLO, no call failed and nothing restarted (Appendix D,
Figure D2).

The change matters where background KV fills the pool, which tuned vLLM shows as KV usage near 100% while requests wait.
T should cover the voice agent's first sentence at its longest (§3.3, §6.3), and a faster GPU takes a larger step cap
only where its voice tail stays under the SLO (§6.7). An engine failure breaks every stream it serves. In the failure
drills each stream of a killed engine received an error event 6–11 ms after the kill command, and a request succeeded
again 106 s after the kill (Appendix D).

## 8. Related work

**Co-serving a latency class with best-effort work.** ConServe [[1](#ref-1)], HyGen [[2](#ref-2)], BROS [[22](#ref-22)],
Niyama [[23](#ref-23)], Chiron [[24](#ref-24)], Echo [[25](#ref-25)] and ProServe [[26](#ref-26)] schedule online
requests beside offline ones, and Valve colocates them in production on 8,054 GPUs with sub-millisecond compute
preemption and dynamic memory reservation [[3](#ref-3)]. We change an existing engine's scheduler for a workload pair
these systems do not report together.

**Preemption for a waiting request.** vLLM's V0 scheduler preempted lower-priority running requests for the
highest-priority waiting request from v0.6.2 until V0 was removed, with chunked prefill off and by recomputing the
victims [[8](#ref-8)]. Priority scheduling in the V1 scheduler [[27](#ref-27)] preempts among running requests only
[[7](#ref-7)]. Its open fixes preempt at the request-count cap [[18](#ref-18)], which does not bind when KV fills, and
two also preempt lower-priority requests one at a time when a waiting request's allocation fails, until it fits or none
is left [[28](#ref-28), [59](#ref-59)]. SGLang's priority preemption [[6](#ref-6)] stops when KV memory runs out, and an
open pull request adds preemption on KV exhaustion with a best-effort keep of the victim's KV [[29](#ref-29)].

Research systems that preempt for a waiting or arriving request keep the victim's KV in host memory, as TokenFlow
[[30](#ref-30)], ConServe [[1](#ref-1)], Chiron [[24](#ref-24)], ProServe and FastServe [[26](#ref-26), [31](#ref-31)]
do. Semantic Scheduling, implemented in vLLM, preempts lower-urgency requests for an urgent one, discarding or
offloading each victim's KV [[32](#ref-32)]. Llumnix keeps memory headroom for high-priority requests [[33](#ref-33)],
and LayerKV traces long-context TTFT to queueing for KV blocks [[34](#ref-34)]. The KV preemption is V0's rule in V1's
scheduler on vLLM's own CPU tier, and unlike the open fixes it preempts only when at most N victims make room, and then
as few as do.

**Keeping prefill out of a priority request's way.** Chunked prefill bounds how long a decode waits behind a prefill
[[12](#ref-12)], and disaggregated serving moves prefill to other GPUs [[35](#ref-35), [36](#ref-36)], which one GPU
cannot do. Semantic Scheduling keeps new prefills out while its most urgent request decodes [[32](#ref-32)], VoxServe
prioritizes newly admitted speech requests until their first audio chunk [[37](#ref-37)], and SLOWeave sizes every
prefill chunk to the earliest decode deadline [[38](#ref-38)]. In vLLM a proposed cadence that admitted prefill only
every N steps while any decode ran was closed unmerged [[39](#ref-39)]. vLLM's `prefill_schedule_interval` admits new
prefill only every N steps, in v0.32.0 in the data-parallel MoE engine alone and since in a single engine too
[[40](#ref-40)]. These hold prefill back for every decode alike, where the budget cap holds only lower-priority prefill
through a reply's first tokens.

**Routers and gateways.** llm-d's flow control [[19](#ref-19)], SMG [[20](#ref-20)], Dynamo [[41](#ref-41)], AIBrix
[[42](#ref-42)] and vLLM's router [[43](#ref-43)] admit, order, shed or evict requests in front of the engine. Some end
running background requests, as llm-d's in-flight eviction does, but none can preempt a running request and keep its
work.

**KV across agents' pauses.** InferCept [[44](#ref-44)], KVFlow [[45](#ref-45)], TokenCake [[46](#ref-46)], Continuum
[[47](#ref-47)] and MORI [[48](#ref-48)] park, retain or prefetch agents' KV around tool calls. CachedAttention
[[49](#ref-49)] and Pensieve [[50](#ref-50)] keep a conversation's KV between turns, and on production prefix-cache
traces LRU is hard to beat [[51](#ref-51)]. These systems decide which agent's KV stays. The change decides when a
realtime request may take KV from them, and leaves the rest to vLLM's eviction and CPU tier (§6.5).

**Ordering by what is cached, and the copy path.** SGLang orders its waiting requests by the longest cached prefix
[[5](#ref-5)], and since 4 November vLLM's main reorders a bounded window of its queue by the GPU prefix cache's hits
alone [[60](#ref-60)], which cannot see the CPU tier where the cycle of §6.11 forms. vLLM moves pages under 28 KiB with
a Triton kernel and the rest on the copy engine, and an open issue and its pull requests calibrate that cutoff per
device [[61](#ref-61)]. §6.11 measures what the cutoff leaves out on PCIe 3.0 x8.

**Voice under contention.** LiveKit's preemptive generation, Endpoint Anticipation and PredGen start the LLM earlier in
the turn [[13](#ref-13), [52](#ref-52), [53](#ref-53)]. Unmute and NVIDIA's single-GPU voice agent document the cost of
colocating the cascade [[54](#ref-54), [55](#ref-55)], and Metronome shows full-duplex models collapsing when session KV
fills the pool [[56](#ref-56)].

**Adaptive prefill control.** DLFP resizes each prefill chunk from the interval between the host's scheduler calls,
which in vLLM cut a 0.6B model's p99 inter-token latency by 27.7% at a 34.8% higher p99 TTFT and did not carry over to
larger models [[57](#ref-57)]. AgentServe, on one consumer GPU, isolates agents' prefills from their decodes with CUDA
Green Contexts [[58](#ref-58)]. Both treat every request alike, with no class above the rest.

**What is new.** Co-serving a latency class with best-effort work, preempting for a waiting request while keeping the
victim's KV in host memory, and holding prefill back for a priority request are all published. Three things are this
report's:

- **The KV preemption** is V0's rule in vLLM's V1 scheduler, on vLLM's own CPU tier, proposed to vLLM.
- **The budget cap** is bounded to a reply's first tokens and sized to the first sentence a voice agent speaks, which we
  did not find elsewhere.
- **The measurements** cover a cascaded voice pipeline beside long-context coding agents on one GPU and two, against the
  configurations an operator can deploy.

## 9. Limitations

**One model, two kinds of GPU.** Every result uses Qwen3-4B, on RTX 5060 Ti cards whose 36,400-token pool the agents'
contexts fill and on an A100 held to the same pool. A larger pool, another model or a faster GPU changes how often the
pool is full and how long a step with agent prefill takes, and other GPUs are placed only by a model.

**A few calls per GPU.** The headline runs two callers. With four and six, the configuration to deploy missed each SLO
in every round, and with the engine's first four additions 6.4% and 5.8% of the replies were still over it. With the
decode hold too, two more rounds put 0.8% and 2.2% over it, 2 of 238 and 8 of 364 replies (§6.10). Replies whose
windows overlap without a gap would stop the agents' prefill altogether (§6.6).

**Two cards in three rounds.** The two-card comparison rests on three rounds and measured the consolidated card at 2, 4,
8, 12 and 24 sessions only. Its envelope therefore lies between four and eight sessions with the configuration to deploy
and between two and four with tuned vLLM. Each card's part ran as a run of its own, and the sums assume that one card's
work does not slow the other's. The two-card rounds predate the queue and the copy, whose envelope of 24 was measured on
one card at 12 and 24 sessions only.

**What the agents' rate counts.** The replay counts tokens, 99.5% of them prompt tokens, not tasks done, so the results
say what sharing costs the call, not what the agents accomplish, and whether the queue's 133 s p99 matters to an agent's
task is unmeasured. At the envelope the agents' work is the traces' own pace, so the measure there counts the load
offered. The calls' first 240 s also hold the replay's start, when AIPerf resumes every session at once, and over the
24-hour soak the agents did 9,300 tokens a second in the first hour and 582–5,402 in each hour after.

**What the shared model can do.** The quality gate ([GATE.md](../benchmarks/quality/GATE.md)), its rules fixed before
its runs, tried answering questions about private documents in a call (QASPER) beside enriching new documents into
searchable records (Re-DocRED). The served model failed 9 of its 11 criteria, with 50.6% of its answers and 23.9% of its
records' facts right against bars of 80%, and Qwen3-8B-FP8 on a card of its own failed as well ([the gate's
record](../benchmarks/results/2026-10-27-quality/README.md)). A deployment that shares a card for background work has to
check first that the model does that work.

**Synthesized callers.** The callers speak scripted lines in six synthesized voices, and the agent's endpointing is
fixed, taking 0.76 s of voice alone's V2V and 0.82 s under load. Shorter silence, 0.3 s in place of 0.55, brought voice
alone's median V2V from 1.61 s to 1.35 s but answered one of the corpus's 40 lines before it had ended ([the sweep's
record](../benchmarks/results/2026-10-28-endpointing/README.md)). Real callers vary in pace, pauses and barge-ins.

**Scaled traces and their start points.** The traces are scaled 32× in blocks of tokens, which keeps which requests
share which prefixes but changes their timing and sets the KV pressure the first wait depends on (§6.6). Every round
enters the same traces, so the results generalize over where each session enters its trace, not over the corpus's other
traces.

**One step cap sweep, one cap on victims.** The sweep screened caps of 256, 512 and 1,024 tokens on one sample before
five rounds confirmed 256 against 512 (§6.9), and caps between them were not measured. Every run with the KV preemption
used N = 2.

**What a resume recomputes.** vLLM 0.32's counters bound what resumes reload, at most the 117k tokens loaded beyond
first scheduling, but not what they recompute, which bears on the agents' cost (§6.4), not the voice tier's latency.

**Timing by the conversation.** Where the lifecycle policy's mechanisms engage, as beside 24 sessions of the whole
corpus where its throttle deferred 27 requests, they were not isolated from the change.

**Make-room on one vLLM release.** Every measurement ran make-room on vLLM 0.32.0 (§4). Make-room is rebased onto vLLM
0.33.0, where it passes the same CPU suites and schedules as the pull requests ([the port's
record](../benchmarks/results/2026-11-02-port-v0.33.0/README.md)), but no round measured it.

**The pool stays as it is.** An FP8 KV cache doubles the pool's tokens at no quality cost the gate's screen saw, but
beside the CPU tier the engine died both times it ran under the main scenario's load ([the sweep's
record](../benchmarks/results/2026-10-27-sweep/README.md)). The fault is stock vLLM's, and stock 0.32.0 and 0.33.0rc3
died the same way. FlashInfer's `plan()` refills a buffer whose previous copy may still be queued behind the CPU tier's
transfers, and with the wait of vLLM #64165, 0.32.0 ran 30 minutes clean ([the FP8
record](../benchmarks/results/2026-10-31-fp8-crash/README.md)). Moved to the CPU, the speech servers would free their
memory, but Kokoro's first audio took 991 ms against 211–242 ms on the GPU ([the CPU
record](../benchmarks/results/2026-10-27-cpu-audio/README.md)).

## 10. Conclusion

A voice agent that shares its GPU with coding agents loses its seconds inside the engine's scheduler, where a waiting
turn cannot take KV from running background requests and a running reply shares its steps with background prefill. A
control plane in front of the engine sees neither. Two rules inside vLLM's priority scheduler remove both waits by
counting blocks and tokens alone. Beside 24 agent sessions they bring the voice tail to within 0.20 s of a strict SLO
while the agents keep 97% of their token rate, and under an A100's own SLO in every round. Around them the serving
engine lets one card carry 24 agent sessions beside both calls. For a realtime tier beside agents whose KV fills the
pool, the place to act is the engine's scheduler, which can preempt for the waiting turn and keep background prefill
out of its first sentence.

## References

<a name="ref-1"></a>[1] Y. Qiao et al. [ConServe: Fine-Grained GPU Harvesting for LLM Online and Offline Co-Serving](https://arxiv.org/abs/2410.01228). In *ICML 2026*.

<a name="ref-2"></a>[2] T. Sun, P. Wang, and F. Lai. [HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](https://arxiv.org/abs/2501.14808). In *NeurIPS 2025*.

<a name="ref-3"></a>[3] F. Liu et al. [Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate](https://arxiv.org/abs/2604.07874). arXiv:2604.07874, 2026.

<a name="ref-4"></a>[4] W. Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). In *SOSP 2023*.

<a name="ref-5"></a>[5] L. Zheng et al. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). In *NeurIPS 2024*.

<a name="ref-6"></a>[6] SGLang v0.5.22: [priority scheduling with preemption](https://github.com/sgl-project/sglang/pull/8746) (pull requests #8746 and #17026, merged 2025-09-17 and 2026-03-04) and [HiCache](https://github.com/sgl-project/sglang/blob/v0.5.22/docs/docs/advanced_features/hicache.mdx), its hierarchical KV cache in host memory.

<a name="ref-7"></a>[7] vLLM issue #40004. [Priority scheduling supports preemption of requests in the running queue by requests in the waiting queue](https://github.com/vllm-project/vllm/issues/40004). Opened 2026-04-16, open on 2026-10-27.

<a name="ref-8"></a>[8] vLLM pull request #5958. [Adding priority scheduling](https://github.com/vllm-project/vllm/pull/5958). Merged 2024-09-25, in vLLM v0.6.2 to v0.10.2.

<a name="ref-9"></a>[9] C. Wang. vLLM pull request #65282. [Priority scheduling: preempt lower-priority requests for KV cache space](https://github.com/vllm-project/vllm/pull/65282). Opened 2026-11-03, open, with no maintainer review as of 2026-11-09; it succeeds #63350, opened 2026-10-23.

<a name="ref-10"></a>[10] C. Wang. vLLM pull request #65283. [Priority scheduling: hold lower-priority prefill for first tokens](https://github.com/vllm-project/vllm/pull/65283). Opened 2026-11-03, open, with no maintainer review as of 2026-11-09; it succeeds #63351, opened 2026-10-23.

<a name="ref-11"></a>[11] SemiAnalysis. [Claude Code traces (cc-traces-weka-062126)](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126). Hugging Face dataset.

<a name="ref-12"></a>[12] A. Agrawal et al. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310). In *OSDI 2024*.

<a name="ref-13"></a>[13] LiveKit. [LiveKit Agents](https://github.com/livekit/agents), version 1.8.8, and its [documentation of preemptive generation](https://docs.livekit.io/agents/multimodality/audio.md).

<a name="ref-14"></a>[14] NVIDIA. [AIPerf](https://github.com/ai-dynamo/aiperf), version 0.13.0.

<a name="ref-15"></a>[15] A. Radford et al. [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356). In *ICML 2023*.

<a name="ref-16"></a>[16] hexgrad. [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Hugging Face model.

<a name="ref-17"></a>[17] A. Yang et al. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). arXiv:2505.09388, 2025.

<a name="ref-18"></a>[18] vLLM pull request #54530. [Fix priority scheduling: allow waiting high-priority request to preempt running lower-priority one at max_num_seqs](https://github.com/vllm-project/vllm/pull/54530). Opened 2026-08-31, open on 2026-10-27.

<a name="ref-19"></a>[19] llm-d. [llm-d](https://github.com/llm-d/llm-d) v1.0.0 and its router's [flow control](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/flowcontrol/README.md).

<a name="ref-20"></a>[20] [SMG](https://github.com/smg-project/smg), version 1.13.0.

<a name="ref-21"></a>[21] E. Frantar et al. [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743). In *PPoPP 2025*.

<a name="ref-22"></a>[22] B. Wan et al. [Efficient LLM Serving on Hybrid Real-time and Best-effort Requests](https://arxiv.org/abs/2504.09590). arXiv:2504.09590, 2025.

<a name="ref-23"></a>[23] K. Goel et al. [Niyama: Breaking the Silos of LLM Inference Serving](https://arxiv.org/abs/2503.22562). arXiv:2503.22562, 2025.

<a name="ref-24"></a>[24] A. Patke et al. [Hierarchical Autoscaling for Large Language Model Serving with Chiron](https://arxiv.org/abs/2501.08090). arXiv:2501.08090, 2025.

<a name="ref-25"></a>[25] Z. Wang et al. [Echo: Efficient Co-Scheduling of Hybrid Online-Offline Tasks for Large Language Model Serving](https://arxiv.org/abs/2504.03651). arXiv:2504.03651, 2025.

<a name="ref-26"></a>[26] W. Huang et al. [ProServe: Unified Multi-Priority Request Scheduling for LLM Serving](https://arxiv.org/abs/2512.12928). arXiv:2512.12928, 2025.

<a name="ref-27"></a>[27] vLLM pull request #19057. [Implement priority scheduling in V1 engine](https://github.com/vllm-project/vllm/pull/19057). Merged 2025-06-23.

<a name="ref-28"></a>[28] vLLM pull request #40087. [Priority scheduling supports preemption of requests in the running queue by requests in the waiting queue](https://github.com/vllm-project/vllm/pull/40087). Opened 2026-04-17, last pushed 2026-07-14, open on 2026-10-29.

<a name="ref-29"></a>[29] SGLang pull request #45686. [Fire priority preemption when KV runs out, and keep retracted KV behind a flag](https://github.com/sgl-project/sglang/pull/45686). Opened 2026-10-21, open on 2026-10-29.

<a name="ref-30"></a>[30] J. Chen et al. [TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling](https://arxiv.org/abs/2510.02758). In *EuroSys 2026*.

<a name="ref-31"></a>[31] B. Wu et al. [FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang). In *NSDI 2026*.

<a name="ref-32"></a>[32] W. Hua et al. [Semantic Scheduling for LLM Inference](https://arxiv.org/abs/2506.12204). arXiv:2506.12204, 2025.

<a name="ref-33"></a>[33] B. Sun et al. [Llumnix: Dynamic Scheduling for Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/sun-biao). In *OSDI 2024*.

<a name="ref-34"></a>[34] Y. Xiong et al. [LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management](https://arxiv.org/abs/2410.00428). arXiv:2410.00428, 2024.

<a name="ref-35"></a>[35] Y. Zhong et al. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670). In *OSDI 2024*.

<a name="ref-36"></a>[36] P. Patel et al. [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677). In *ISCA 2024*.

<a name="ref-37"></a>[37] K. Kamahori et al. [VoxServe: Streaming-Centric Serving System for Speech Language Models](https://arxiv.org/abs/2602.00269). arXiv:2602.00269, 2026.

<a name="ref-38"></a>[38] S. Song et al. [Deadline-Aware Adaptive Prefill Chunking for Efficient Large Language Model Serving](https://arxiv.org/abs/2609.07883). arXiv:2609.07883, 2026.

<a name="ref-39"></a>[39] vLLM pull request #56848. [Add decode-priority prefill cadence for single-node decode ITL](https://github.com/vllm-project/vllm/pull/56848). Opened 2026-09-14, closed unmerged 2026-09-17.

<a name="ref-40"></a>[40] vLLM pull request #54627. [Apply prefill_schedule_interval outside data parallelism](https://github.com/vllm-project/vllm/pull/54627). Merged 2026-10-15; not in v0.32.0.

<a name="ref-41"></a>[41] NVIDIA. [Dynamo](https://github.com/ai-dynamo/dynamo), version 1.5.1.

<a name="ref-42"></a>[42] [AIBrix](https://github.com/vllm-project/aibrix), version 0.8.0.

<a name="ref-43"></a>[43] [vLLM router](https://github.com/vllm-project/router), version 0.1.16.

<a name="ref-44"></a>[44] R. Abhyankar et al. [InferCept: Efficient Intercept Support for Augmented Large Language Model Inference](https://proceedings.mlr.press/v235/abhyankar24a.html). In *ICML 2024*.

<a name="ref-45"></a>[45] Z. Pan et al. [KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows](https://arxiv.org/abs/2507.07400). In *NeurIPS 2025*.

<a name="ref-46"></a>[46] Z. Bian et al. [TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications](https://arxiv.org/abs/2510.18586). In *EuroSys 2027*.

<a name="ref-47"></a>[47] H. Li et al. [Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](https://arxiv.org/abs/2511.02230). In *ICLR 2026 Workshop on Lifelong Agents*.

<a name="ref-48"></a>[48] T. Xia et al. [Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI](https://arxiv.org/abs/2606.00866). arXiv:2606.00866, 2026.

<a name="ref-49"></a>[49] B. Gao et al. [Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention](https://arxiv.org/abs/2403.19708). In *USENIX ATC 2024*.

<a name="ref-50"></a>[50] L. Yu, J. Lin, and J. Li. [Stateful Large Language Model Serving with Pensieve](https://arxiv.org/abs/2312.05516). In *EuroSys 2025*.

<a name="ref-51"></a>[51] Y. Liu, M. Yu, and J. Yang. [When Fancy Eviction Fails: Rethinking Cache Replacement For LLM Prefix Reuse](https://arxiv.org/abs/2609.28870). arXiv:2609.28870, 2026.

<a name="ref-52"></a>[52] S. Udupa et al. [Endpoint Anticipation for Low-Latency Spoken Dialogue](https://arxiv.org/abs/2606.13450). In *Interspeech 2026*.

<a name="ref-53"></a>[53] S. Li and A. Grover. [PredGen: Accelerated Inference of Large Language Models through Input-Time Speculation for Real-Time Speech Interaction](https://arxiv.org/abs/2506.15556). arXiv:2506.15556, 2025.

<a name="ref-54"></a>[54] Kyutai. [Unmute](https://github.com/kyutai-labs/unmute).

<a name="ref-55"></a>[55] NVIDIA. [Nemotron voice agent blueprint](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent).

<a name="ref-56"></a>[56] J. Meng and B. Li. [Metronome: Bound the Cache, Keep the Beat for Real-Time Interaction Model Serving](https://arxiv.org/abs/2607.02640). arXiv:2607.02640, 2026.

<a name="ref-57"></a>[57] G. Agarwal, A. Garg, and I. Singhal. [Decode-Latency Feedback Prefill: A Model-Free Controller and Its Generalization Limits](https://arxiv.org/abs/2609.38386). arXiv:2609.38386, 2026.

<a name="ref-58"></a>[58] Y. Zhang, Y. Yan, N. Yang, and D. Yuan. [AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU](https://arxiv.org/abs/2603.10342). arXiv:2603.10342, 2026.

<a name="ref-59"></a>[59] vLLM pull request #45558. [Priority-based preemption at max_num_seqs](https://github.com/vllm-project/vllm/pull/45558). Opened 2026-06-14, last pushed 2026-08-20, open on 2026-10-31.

<a name="ref-60"></a>[60] vLLM pull request #54625. [Add cache-aware admission ordering](https://github.com/vllm-project/vllm/pull/54625). Merged 2026-11-04.

<a name="ref-61"></a>[61] vLLM issue #57555. [KV offload: DMA/Triton load threshold needs per-device profiling](https://github.com/vllm-project/vllm/issues/57555). Opened 2026-09-18, open as of 2026-11-09, with its pull requests #66510 and #66511 (reopened that day from #57343 and #57569).

<a name="ref-62"></a>[62] vLLM pull request #66558. [[Core] Count KV connector hits in cache-aware admission](https://github.com/vllm-project/vllm/pull/66558). Opened 2026-11-09 as a draft.

<a name="ref-63"></a>[63] vLLM pull request #66527. [[Perf][KV Offload] Load pages of any size through the Triton kernel (opt-in)](https://github.com/vllm-project/vllm/pull/66527). Opened 2026-11-09 as a draft.

## Appendices

### A. Reproducing the results

Every number and table in this report, and every figure but the illustrative Figure 5, is computed from records
committed to the repository, or from its tests where Table A1 says so, and the repository's tests fail when a committed
figure differs from what its records give. [BENCHMARKS.md](BENCHMARKS.md) lists the campaigns' measurements, the
records' READMEs the probes and checks around them, and [EXPERIMENTS.md](EXPERIMENTS.md) each experiment's design.

**Table A1. Where each result comes from.**

| What | Records | Regenerated by |
|---|---|---|
| §6.1–§6.4 and §6.6, Figures 1, 8, 9, 10 and 12, Tables 2–4 and B1 | [`2026-10-24-e2-w5`](../benchmarks/results/2026-10-24-e2-w5) | `make reproduce-e2-w5`, `make figures` |
| The code of 25 October (§4) | [`2026-10-25-e2-fb`](../benchmarks/results/2026-10-25-e2-fb) | its `compare.py` |
| §6.5 and Figure 11 | [`2026-10-23-e2`](../benchmarks/results/2026-10-23-e2) | `make reproduce-e2`, `make figures` |
| §6.7, Figure 13 and Table 5 | [`2026-10-26-a100`](../benchmarks/results/2026-10-26-a100) with its [`batch`](../benchmarks/results/2026-10-26-a100/batch), and [`2026-10-26-e2-rs`](../benchmarks/results/2026-10-26-e2-rs) | `make reproduce-e2-a100`, `make reproduce-a100`, `running_share.py`, `make figures` |
| The probes (Figures 2–4) | [`2026-10-23-make-room-probes`](../benchmarks/results/2026-10-23-make-room-probes) and [`2026-10-24-w5-probes`](../benchmarks/results/2026-10-24-w5-probes) | the probes' `check.py`, `make figures` |
| The window (Figure 6) | [`2026-10-25-window-knee`](../benchmarks/results/2026-10-25-window-knee) and [`replies.tsv`](../benchmarks/results/2026-10-24-e2-w5/replies.tsv) | their `table.py` and `replies.py`, `make figures` |
| A step's cost (Figure 7) | [`make-room-schedule.json`](../benchmarks/perf/make-room-schedule.json) | `make perf-make-room`, `make figures` |
| A prompt's cost recomputed, restored or hit (§2.2) | [The tier cost table](BENCHMARKS.md#what-does-resuming-an-agent-turn-cost), kept in [`cost.rs`](../crates/flatshare-core/src/cost.rs) | `flatshare-bench tiers` on the GPU node |
| The gateway's overhead (§5.1) | [`2026-10-29-gateway-window`](../benchmarks/results/2026-10-29-gateway-window) | its `table.py` |
| The search for stalls (§4) | [`2026-11-05-liveness`](../benchmarks/results/2026-11-05-liveness) | its README |
| The simulator's results (§6.5, §6.7), modeled | [Its tables in BENCHMARKS.md](BENCHMARKS.md#does-knowing-when-an-agent-returns-help) and [other hardware, modeled](../crates/flatshare-sim/README.md#other-hardware-modeled) | `make reproduce-e1`, `flatshare-sim hardware` |
| Replies whose windows overlap (§6.6) | The test of overlapping replies in [`test_make_room.py`](../python/flatshare_vllm/tests/test_make_room.py) | `make test-make-room` |
| The first sentences' tokens (§6.3) | [`2026-10-27-first-sentence`](../benchmarks/results/2026-10-27-first-sentence) | [`first_sentence.py`](../benchmarks/replies/first_sentence.py) |
| §6.8, Figure 14 and Table 6 | [`2026-10-28-rounds`](../benchmarks/results/2026-10-28-rounds) | its `gate.py`, `make figures` |
| §6.9, Figure 15 and Table 7 | [`2026-10-28-rounds`](../benchmarks/results/2026-10-28-rounds), and the screens in [`2026-10-27-sweep`](../benchmarks/results/2026-10-27-sweep) | `gate.py`'s sweep, the screens' `table.py`, `make figures` |
| §6.10 and Table 8 | [`2026-11-01-engine-points`](../benchmarks/results/2026-11-01-engine-points) | its `table.py` |
| §6.10, Figure 16 and Table 9 | [`2026-11-02-opt-rounds`](../benchmarks/results/2026-11-02-opt-rounds) | its `table.py`, `make figures` |
| §6.11, Figures 17 and 18 and Table 10 | [`2026-11-08-envelope`](../benchmarks/results/2026-11-08-envelope) | its `table.py`, `make figures` |
| The failure drills (Appendix D, Figure D1) | [`2026-10-25-drills`](../benchmarks/results/2026-10-25-drills) and [`2026-10-28-node-failover`](../benchmarks/results/2026-10-28-node-failover) | their READMEs, `make figures` |
| The 24-hour soak (§7, Appendix D, Figure D2) | [`2026-10-28-soak`](../benchmarks/results/2026-10-28-soak) | its `table.py`, `flatshare-bench soak report`, `make figures` |
| The limits of §9 | [`2026-10-27-quality`](../benchmarks/results/2026-10-27-quality), [`2026-10-28-endpointing`](../benchmarks/results/2026-10-28-endpointing), [`2026-10-27-sweep`](../benchmarks/results/2026-10-27-sweep), [`2026-10-31-fp8-crash`](../benchmarks/results/2026-10-31-fp8-crash), [`2026-10-27-cpu-audio`](../benchmarks/results/2026-10-27-cpu-audio) and [`2026-11-02-port-v0.33.0`](../benchmarks/results/2026-11-02-port-v0.33.0) | their READMEs, [`gate.py`](../benchmarks/quality/gate.py) and [`summary.py`](../benchmarks/endpointing/summary.py) |

**Hosting.** The tag in the header holds the code, the records and this report. The raw records are release assets of
[v1.0](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.0) (the main evaluation),
[v1.1](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.1) (the A100),
[v1.3](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.3) (the KV sampling) and
[v0.3](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3) (the campaign before the main evaluation).

**Requirements and variation.** Regenerating the tables and figures needs Python and the repository alone, and measuring
again a GPU node configured as in Appendix C ([HARDWARE.md](HARDWARE.md) deploys one on another GPU). A repeat should
land within the ranges of Tables 2 and B1 rather than on their medians.

### B. Results by round

A column compares configurations on the same sample of the agents (§5.4). The records' own tables give every other
comparison run by run: [the main evaluation's](../benchmarks/results/2026-10-24-e2-w5/report/frontier.md), [the campaign
before it](../benchmarks/results/2026-10-23-e2/report/frontier.md), [the
A100's](../benchmarks/results/2026-10-26-a100/report/frontier.md) and [the two
cards'](../benchmarks/results/2026-10-28-rounds/README.md).

**Table B1. The p95 V2V of every run behind Table 2, by round (ms).** Three sets of rounds replayed the same five
samples, and the last row is the configuration to deploy on the code of 25 October (§4).

| Configuration | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Median |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 1,810 | 1,774 | 1,819 | 1,764 | 1,784 | 1,784 |
| Tuned vLLM, first set | 10,576 | 10,770 | 8,686 | 9,782 | 10,432 | 10,432 |
| Tuned vLLM, second set | 9,946 | 10,133 | 10,810 | 12,115 | 10,523 | 10,523 |
| Preempt-resubmit, third set | 2,516 | 2,860 | 2,522 | 2,522 | 2,868 | 2,522 |
| The KV preemption, second set | 6,261 | 5,376 | 4,693 | 4,970 | 5,085 | 5,085 |
| The change (T = 16), second set | 3,022 | 3,100 | 2,868 | 3,192 | 3,094 | 3,094 |
| The configuration to deploy, first set | 2,236 | 2,175 | 2,298 | 2,250 | 2,307 | 2,250 |
| The configuration to deploy, second set | 2,174 | 2,301 | 2,189 | 2,232 | 2,397 | 2,232 |
| The configuration to deploy, third set | 2,293 | 2,257 | 2,132 | 2,162 | 2,145 | 2,162 |
| The configuration to deploy, code of 25 October | 2,313 | 2,246 | 2,253 | 2,268 | 2,343 | 2,268 |

### C. Configurations

**Table C1. The two GPU nodes.** The A100 node runs the RTX 5060 Ti node's configuration but for the card, its precision
path and the host.

| | RTX 5060 Ti node | A100 node |
|---|---|---|
| GPU | GeForce RTX 5060 Ti 16 GB, PCIe 3.0 x8 | A100 SXM4 80 GB, PCIe 3.0 to the host |
| Host | 2× Xeon E5-2609 v4 (1.7 GHz), 188 GiB of RAM | 12 vCPUs of a 2.2 GHz Xeon (Cascade Lake), in a Google Colab container |
| LLM | Qwen3-4B-Instruct-2507-FP8 on vLLM 0.32.0, 32k context | The same, on Marlin's weight-only FP8 kernels |
| KV | A fixed 5 GiB pool (36,400 tokens), prefix caching, a 20 GiB CPU tier written through, generated tokens included | The same |
| Scheduling | Priority, asynchronous, `--max-num-batched-tokens 512 --long-prefill-token-threshold 256` | The same, and vLLM's default 2,048-token steps in Table 5's last rows |
| Speech | Whisper large-v3-turbo on vLLM 0.32.0; Kokoro-82M | The same |
| Control plane | A second host: LiveKit, the voice agent (LiveKit Agents 1.8.8), the callers, the gateway and AIPerf 0.13.0 | The same programs on the GPU node itself |

**The configuration to deploy** runs on make-room, with `make_room.max_preemptions_per_step 2`, `make_room.budget_cap
true`, `make_room.budget_cap_tokens 48` and `make_room.reserve_tokens 0` in vLLM's `--additional-config`, and the engine
loads it with `--scheduler-cls`. On vLLM's main the same rules are `--max-priority-preemptions-per-step 2
--priority-prefill-budget-cap --priority-prefill-budget-cap-tokens 48`. The KV preemption alone turns the budget cap off
(`make_room.budget_cap false`), and the change at T = 16 leaves `budget_cap_tokens` at its default of 16.
[PRODUCTION.md](PRODUCTION.md#the-configuration) gives the whole deployment.

**The earlier make-room** of §6.5's campaign took, among victims of the lowest priority, those holding the most KV
first, and lacked three fixes the main evaluation's version and the pull requests have. Its budget cap did not hold a
lower-priority request whose KV load had finished, its KV shortfall counted a request's cached prefix as needed, and it
sized victims before the next step's decode growth.

**The hosts.** The records name the RTX 5060 Ti node Phastform, an identical second node Gray, the control host cyan,
and peacock, where Figure 7 was measured.

### D. Failure drills

The deployed configuration was broken on purpose while two voice streams and four agent streams were decoding, and each
probe after the event used a new connection. Times after a kill count from the drill's stamp just before its `systemctl
kill` (Table D1).

**Table D1. What a failure does to the streams in flight and to new requests.**

| Drill | The six streams in flight | New requests |
|---|---|---|
| The engine killed mid-decode | Each received an error event 6–11 ms after the kill command | Refused at once until the engine's warm start; a request succeeded 106.2 s after the kill |
| The gateway reloaded, as a deploy does | The voice streams finished whole, 20.9 s later; the agents' streams ended after 5.0 s with a draining event | None of 42 failed or waited; the slowest was answered in 0.13 s |
| The gateway restarted | As in the reload | None of 42 failed; those sent during the 20.7 s drain were answered once the new gateway started |
| The gateway killed | All broke 10–14 ms after the kill command | None of 4 failed; the slowest waited 1.3 s for the gateway's restart |

The engine's recovery is mostly its warm start (Figure D1). Its last process exited 7.6 s after the kill, systemd
started it again 5.1 s later, and it listened 92.1 s after that, while on empty caches its first start takes 531.8 s.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-starts-narrow.svg">
  <img src="figures/report-starts.svg" width="100%" alt="Three bars of the engine's phases on one axis of seconds: killed mid-decode, it was down 12.7 s, its unit's last process gone 7.6 s after the kill and systemd's restart 5.1 s later, and its warm start listened 92.1 s after the restart, a request answered 106.2 s after the kill; a warm start was ready at 88.8 s, its weights loaded at 54.1 s and its CUDA graphs captured at 84.1 s; and a cold start, with empty caches, was ready at 531.8 s, its weights at 86.1 s, its KV cache at 189.3 s after compiling and profiling and its CUDA graphs at 526.6 s after the kernels' warmup and JIT builds.">
</picture>

**Figure D1. The engine's recovery from a kill is mostly its warm start, and on empty caches a start takes six times as
long.** Each bar stacks a start's phases from its journal, in seconds from the kill or from the unit's start; compiling
counts in the KV cache's phase, and the kernels' warmup and JIT builds in the CUDA graphs'. The dashed box is the killed
engine down, until systemd starts it again.

On two GPU nodes, with both calls on the first and the second a warm spare (§6.8, §7), a SIGKILL of the first node's
engine moved both calls' next turns to the spare 0.7 s later. After a clean stop they moved 1.9 s later. None of the 300
turns after the events failed, and the spare held its agents' new requests while it carried the calls.

A 24-hour soak of that arrangement, calls arriving at 20 an hour beside 14 agent sessions, put 74 of its 5,058 replies
over the SLO (1.5%, and 5.5% in the worst hour, Figure D2). None of its 476 calls failed, no unit restarted, and the
spare never carried a call. On the calls' card the budget cap held prefill back in 4,853 steps, and the KV preemption
never fired ([the record](../benchmarks/results/2026-10-28-soak/README.md)).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-soak-narrow.svg">
  <img src="figures/report-soak.svg" width="100%" alt="Each of the soak's 24 hours as a column: in panel a, the share of that hour's replies over the SLO of 1,976 ms against the requirement of at most 5%, 74 of 5,058 replies (1.5%) over the day, at most 14 of 254 (5.5%) in hour 4; in panel b, the agents' tokens a second, 3,529 over the day, 9,300 in hour 0 and 582 to 5,402 in each hour after.">
</picture>

**Figure D2. Over the 24 hours 1.5% of the replies were over the SLO, and only hour 4, at 5.5%, went over the 5%
requirement.** (a) Each hour's share of its replies over the SLO, the requirement dashed and the day's share dotted. (b)
The agents' tokens a second by hour, the day's dotted: 9,300 in hour 0, as AIPerf resumes its 14 sessions at once.

### E. Revision history

The report is versioned with the repository's releases. A correction that changes a number, a figure or a claim takes a
new version, and [CHANGELOG.md](../CHANGELOG.md) lists what each release changed.

**Table E1. Versions of this report.**

| Version | Date | What changed |
|---|---|---|
| 1.25 | 2026-11-09 | Figures 17 and 18, §8 on ordering by what is cached and on the copy path |
| 1.24 | 2026-11-08 | §6.11, the queue and the copy |
| 1.23 | 2026-11-05 | §3 and §4: the pull requests' revision of 5 November |
| 1.21 | 2026-11-04 | §3 and §4: the pull requests' revision of 3 November |
| 1.19–1.20 | 2026-11-03 | The pull requests renumbered #65282 and #65283; §3.2's encoder exemption |
| 1.15–1.18 | 2026-11-02 | §6.10, the serving engine's additions and the decode hold; make-room on vLLM 0.33.0; Appendix A |
| 1.14 | 2026-11-01 | §9 on the FP8 crash and what the rounds sample |
| 1.9–1.11 | 2026-10-30 | Results at a glance, Figures 15, D1 and D2, phone layouts |
| 1.7–1.8 | 2026-10-29 | Two cards, the 256-token step cap, the soak and the step cost |
| 1.6 | 2026-10-28 | Replies over the SLO and the pooled p95 |
| 1.5 | 2026-10-27 | Rewritten in full, with the design, algorithms, figures and appendices |
| 1.0–1.3 | 2026-10-26 | First version; the A100 run; the name Flatshare; the RTX 5060 Ti's time short of KV |
