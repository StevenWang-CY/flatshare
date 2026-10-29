# Making Room in the Engine: A Realtime Voice Tier beside Coding Agents on One GPU

**Chuyue Wang**

Flatshare technical report, version 1.7, 29 October 2026. Code, records and figures at tag
[v1.7](https://github.com/StevenWang-CY/flatshare/tree/v1.7) of
[github.com/StevenWang-CY/flatshare](https://github.com/StevenWang-CY/flatshare); earlier versions in [Appendix
F](#f-revision-history).

> [!NOTE]
> The scheduler change this report measures is proposed to vLLM as pull requests
> [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351),
> open and under review on 27 October 2026. Flatshare is an independent project and is not affiliated with vLLM.

## Abstract

A self-hosted voice agent needs its GPU only while it computes a reply, and coding agents could use the rest. On vLLM
0.32, beside 24 coding-agent sessions on one RTX 5060 Ti, the p95 voice-to-voice latency of two calls rises from 1.78 s
alone to 10.48 s. No configuration an operator can deploy today meets the SLO of voice alone plus 250 ms, 2.03 s. The
time is lost in the scheduler. A waiting voice turn cannot take KV cache from running agent requests, and once admitted,
the turn shares every step of its prefill and reply with agent prefill. We add two rules to vLLM's priority scheduler.
Make-room lets the waiting turn preempt lower-priority requests for KV, keeping their cache in the CPU tier, and the
prefill hold keeps lower-priority prefill out of the turn's steps until its reply's first sentence. Together they bring
the p95 to 2.24 s at the median of 15 runs, 0.20 s over the SLO, where tuned vLLM leaves 90% of the replies over it. A
third of the replies, 286 of 885, still take longer than the SLO allows, so the change narrows the gap without closing
it. The agents keep 97% of their token rate on tuned vLLM at the same 512-token step cap. Make-room alone leaves 5.09 s.
The nearest any control plane in front of the engine came is 2.52 s, by aborting and resubmitting agent requests, at a
193 s p99 time to first token for the agents. On an A100 at the same KV pool and step cap, the rules take the p95 from
4.99 s to 1.63 s, 7–60 ms under that card's SLO in each of five rounds. At the card's default 2,048-token steps they
keep all of the agents' throughput and leave the voice 0.14–0.17 s over the SLO. On two RTX 5060 Ti cards, under a
resource gate fixed before the runs, both calls and four agent sessions on one card with the rules, beside a card of
agents alone, do 34% more background work than a card per workload, with 2.8% of the replies over the SLO. Most of that
gain is the arrangement's: on stock vLLM the shared card qualifies at two sessions, and the same cards do 27% more.

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
  [C. Configurations](#c-configurations) · [D. Failure drills](#d-failure-drills) · [E. The forecast for the
  A100](#e-the-forecast-for-the-a100) · [F. Revision history](#f-revision-history)

## 1. Introduction

A self-hosted voice agent is a cascade of speech recognition, an LLM and speech synthesis, provisioned for how soon it
answers. Its GPU idles for most of each call, because the LLM works only while it computes a reply, and serving systems
fill such idle capacity with best-effort work [[1](#ref-1), [2](#ref-2), [3](#ref-3)]. Coding agents suit the role. They
send long prompts that share long prefixes, and they spend much of each session waiting for tools and for their user.

The two workloads do not share a GPU well on today's engines. We run a voice agent beside 24 Claude Code sessions on one
RTX 5060 Ti with vLLM 0.32 [[4](#ref-4)], tuned on a typical load as an operator would tune it (§5.4). The voice tier
runs at the highest priority, a 512-token step cap bounds how long a decode waits behind a prefill chunk, and a CPU KV
tier is written through as tokens are computed. With the voice agent alone, the median reply begins 1.62 s after the
caller stops speaking. Beside the agents, the median reply begins after 4.61 s and 7% after more than 10 s, and the p95
voice-to-voice latency (V2V) of two calls rises from 1.78 s to 10.48 s. Figure 1 shows where the seconds go.
Endpointing, speech synthesis and the network barely move. The turn waits 2.81 s at the mean for the LLM's first token,
against 0.09 s alone, and then takes 1.07 s to decode the first sentence it can speak, against 0.22 s.

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-stages-narrow.svg">
  <img src="figures/report-stages.svg" width="100%" alt="Stacked horizontal bars of a voice turn's mean stage times in four rows. Voice alone: 1.62 s, of which the wait for the first token is 0.09 s and the first sentence's decode 0.22 s. Tuned vLLM: 5.30 s, the wait 2.81 s and the decode 1.07 s. With make-room: 3.41 s, the wait 0.78 s and the decode 1.21 s. With make-room and the prefill hold at 48 tokens: 1.97 s, the wait 0.29 s and the decode 0.26 s. Endpointing, speech synthesis and transport change little between rows.">
</picture>

**Figure 1. Sharing the GPU stretches two stages of the voice turn, and the two rules bring both back near voice
alone's.** Mean time of each stage of a voice turn from the end of the caller's speech, over every reply of each
configuration's runs, with two callers beside 24 coding-agent sessions on the RTX 5060 Ti. Violet is the wait for the
LLM's first token and green the decode of the reply's first sentence. The number after each bar is the mean V2V.

No configuration an operator can deploy today meets the SLO of voice alone plus 250 ms. vLLM's own knobs, SGLang's
priority preemption with its host tier [[5](#ref-5), [6](#ref-6)], a static KV reservation and two routers miss it by
seconds. A router that aborts agent requests before their first byte and resubmits them misses it by 0.49 s, and the
agents' p99 time to first token (TTFT) grows from 106 s to 193 s.

Two mechanisms in vLLM's scheduler explain the loss, and neither is specific to the card. First, the V1 scheduler
preempts a running request only so that another running request can grow, never for a waiting one [[7](#ref-7)]. When
the agents' running requests hold the KV pool, a voice turn waits until one of them finishes or is preempted for
another's growth. In isolation, a 750-token turn waits 4.9–5.0 s behind two 18k-token agent requests. Second, once the
turn is admitted, every step of its prefill and its reply also carries the agents' prefill chunks, so its reply's tokens
arrive 60–66 ms apart instead of 16–20 ms.

Neither wait can be removed from outside the engine without discarding the agents' work. A control plane in front of the
engine sees requests, not KV blocks or steps. It can defer new background requests and abort those in flight, but it
cannot preempt a request and keep its work, and it cannot reach into a step. Inside vLLM, the V0 scheduler did preempt
for a waiting request of the highest priority [[8](#ref-8)], but only with chunked prefill off and by recomputing its
victims, and the V1 scheduler that replaced it has no such rule.

We change the scheduler instead, with two rules (Figure 5). Make-room lets the highest-priority waiting request preempt
lower-priority running requests when it lacks KV, through vLLM's own preemption path and in vLLM's own victim order, at
most N in a step. The victims' KV is already in the CPU tier, so a resume can reload what the tier still holds. The
prefill hold keeps lower-priority prefill out of the steps of that request's prefill and first T output tokens, while
every decode proceeds. We size T to cover whole replies, and with them the first sentence, the text the voice agent
hands to speech synthesis. Both rules count blocks, tokens and steps, so neither needs a model of the hardware, and both
are off by default. They are proposed to vLLM as two stacked pull requests [[9](#ref-9), [10](#ref-10)].

With both rules and T = 48, the p95 V2V falls to 2.24 s at the median of 15 runs, 0.20 s over the SLO, and the agents
keep 97% of their token rate (§6.1). The SLO allows 5% of the replies over its threshold, and 32% of the change's are
over it, against 90% on tuned vLLM. Make-room cuts the wait for the first token from 2.81 s to 0.78 s. The hold removes
most of the rest, which is the turn's own prefill sharing steps with agent prefill, and cuts the decode of the first
sentence from 1.21 s to 0.26 s (§6.2). A window shorter than the replies leaves their later tokens beside agent prefill,
so at T = 16 the p95 stays at 3.09 s (§6.3). The agents pay through the hold, which keeps their prefill out of a third
of the engine's steps, and not through the preemptions (§6.4). A router that aborts agent requests makes room at the
agents' tail. A gateway policy timed by the conversation, with KV headroom held from the moment the caller starts to
speak, leaves the tail where tuned vLLM leaves it (§6.5). The gain shrinks where the pool has room. As calls overlap,
the tail grows and the hold keeps the agents' prefill out for longer (§6.6). On an A100 at the same KV pool and step
cap, the rules take the p95 from 4.99 s to 1.63 s, 7–60 ms under that card's SLO in each of five rounds (§6.7). On two
RTX 5060 Ti cards, a resource gate fixed before the runs compared a card per workload, pooling and consolidation. Both
calls and four agent sessions on one card, beside a card of agents alone, did 34% more background work than a card per
workload within the SLO's 5% allowance, and stock vLLM's shared card, at two sessions, did 27% more (§6.8).

This report makes five contributions.

- **It measures the failure end to end.** V2V is read from waveforms at the caller beside a replay of real coding-agent
  traces. Tuned vLLM, the router and the change run in rotated rounds over five samples of the traces, and every other
  configuration an operator can deploy in a single-sample campaign of one to five runs each (§5, §6.1, §6.5).
- **It changes vLLM's priority scheduling** with two rules that run on vLLM 0.32 without modifying the installed engine
  (§3, §4). The first brings V0's preemption for a waiting request to the V1 scheduler under chunked prefill, with the
  CPU tier keeping the victims' KV. The second bounds the hold to a reply's first tokens, sized to cover whole replies.
  Neither idea is new on its own (§8).
- **It shows what each rule contributes and what it costs the agents**, from the engine's own counts of what each rule
  did (§6.2–§6.4). The hold is necessary. The room for the waiting turn can come from make-room or, in a single-sample
  campaign, from a class-aware KV reserve (§6.5).
- **It bounds what a control plane in front of the engine can do** when agents' KV fills the pool (§6.5), and carries
  the result to a datacenter GPU (§6.7).
- **It measures what sharing a card buys** over a card per workload on two cards, under a resource gate fixed before
  the runs, and finds most of the gain in the arrangement rather than in the scheduler: the rules' own share is the two
  agent sessions more that the shared card carries within the SLO (§6.8).

## 2. Background and motivation

### 2.1 A voice turn

A cascaded voice agent answers in four stages. It transcribes the caller's speech, decides that the caller has finished
(endpointing), streams the transcript through the LLM, and hands speech synthesis the reply's first complete sentence as
soon as the LLM has produced it. The caller hears nothing until that sentence is spoken. We measure voice-to-voice
latency (V2V) from the end of the caller's speech to the agent's first audio. With the voice agent alone on the GPU, a
turn's mean V2V is 1.62 s. Endpointing, which here ends a fixed delay after voice activity stops and once the final
transcript is in, takes 0.76 s, the LLM's first sentence 0.30 s, the first audio 0.21 s, and transport and playout 0.35
s. The LLM's share is small, but it sits on the critical path, and its first sentence rather than its whole reply sets
the wait. Our SLO allows voice alone's p95 V2V plus 250 ms, which is 2.03 s with two callers.

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
threshold (256 tokens) bounds each step's prefill. Under priority scheduling the waiting loop first takes the requests
that already hold KV, then the others in priority order, and a lower value is served first. The running loop takes
running requests in the order they were admitted, and when one cannot grow, the scheduler preempts the lowest-priority
running request, the latest arrival among equals. The waiting loop stops at the first request it cannot allocate, and
nothing preempts on a waiting request's behalf [[7](#ref-7)].

Two further mechanisms matter here. vLLM's offloading connector keeps a CPU tier of KV that is written through as tokens
are computed, by default a prompt's alone and here generated tokens too (`offload_prompt_only: false`), so a request
whose blocks were evicted from the GPU reloads them instead of recomputing them. Asynchronous scheduling plans each step
before the previous step's output is processed. It keeps the GPU busy, and it leaves up to two steps in flight. With it
and a CPU tier, as here, a preempted request's blocks are freed one step late.

### 2.4 The two waits in isolation

Three probes on the RTX 5060 Ti isolate the waits, with the hold at the pull requests' default window of 16 tokens
wherever it runs. In the KV case, two priority-2 requests of 18k prompt tokens fill the pool and keep decoding, and a
750-token priority-0 request arrives once at most 24 blocks are free. Figure 2 shows why it waits. The two agent
requests fill the pool within 8 s and then decode into its last free blocks. Nothing frees a block until one of them
cannot grow, about 5 s later, and vLLM preempts the later arrival so that the other can, which frees half the pool at
once. On vLLM 0.32 the voice turn's first token comes 4.9–5.0 s after it is sent, at that preemption, and with make-room
0.22–0.29 s after (Figure 3a).

<img src="figures/report-kv-timeline.svg" width="100%" alt="Two panels over 25 seconds. (a) KV pool use climbs to 99% by about 8 s, stays at the top until about 12.6 s, then drops to 50% when agent B is preempted so that agent A can grow. The span from where the KV case sends its voice turn to the preemption is shaded and labeled no room for a waiting request. (b) Both agent requests decode from about 8 s; agent B stops at 192 output tokens when it is preempted, and agent A continues to about 810.">

**Figure 2. On vLLM 0.32 a full KV pool frees nothing until a running request cannot grow.** Two priority-2 requests of
18k prompt tokens sent together to vLLM 0.32 on the RTX 5060 Ti, with the engine's state sampled about every half
second. (a) The share of the KV pool's 2,274 blocks in use. The shaded span runs from where the KV case sends its voice
turn to the preemption that frees half the pool. (b) Each agent request's output tokens.

In the prefill case, three priority-2 requests of 10,000 prompt tokens prefill side by side with room left in the pool.
The priority-0 request is admitted at once, but every step of its reply carries their prefill chunks, so its tokens come
60–66 ms apart. With the hold its steps carry no agent prefill, and its tokens come 16–20 ms apart (Figure 3b). A step
cap bounds the tokens in a step, not its time, and a voice reply needs a few dozen steps before it can be spoken.

<img src="figures/report-isolation.svg" width="100%" alt="Two panels of probe runs in three rows: vLLM 0.32, with make-room, and with make-room and the prefill hold. (a) The KV case's first token: 4,968 ms at the median on vLLM 0.32, 290 ms with make-room and 290 ms with both rules. (b) The prefill case, first token to 16th token per run: the 16th token at 1,500 ms on vLLM 0.32, 1,350 ms with make-room alone and 575 ms with both rules.">

**Figure 3. In isolation, a voice-sized request waits seconds for KV, and agent prefill slows each of its tokens three-
to fourfold.** Four runs of each probe on the RTX 5060 Ti, with the prefill hold at a 16-token window. (a) Time to the
first token of a 750-token priority-0 request sent while two 18k-token priority-2 requests fill the KV pool. Each circle
is a run, and the label is the median. (b) The same request behind three 10,000-token priority-2 prefills. Each line
runs from a run's first token to its 16th, and the bar and label mark the median 16th token.

Beside closed-loop agents the two waits occur together (Figure 4). Two calls, each turning every 4–8 s, run for 240 s
beside 12 agent sessions whose prompts are 6–30k tokens. When the agents' prompts are fresh each turn, the calls' p95
time to first token is 6.83 s on vLLM 0.32, 4.11 s with make-room alone and 0.39 s with both rules. When each agent's
prompt grows by a tool result instead, so that the agents reuse the CPU tier, it is 7.16, 1.07 and 0.36 s.

<img src="figures/report-tail.svg" width="100%" alt="Two panels of cumulative distributions of the calls' time to first token on a log axis. With fresh agent prompts the 95th percentile is 6.83 s on vLLM 0.32, 4.11 s with make-room and 0.39 s with both rules. With growing prompts it is 7.16 s, 1.07 s and 0.36 s. The curve with both rules lies left of the others.">

**Figure 4. Beside closed-loop agents, make-room shortens the calls' first-token tail and the hold removes it.** The
share of two calls' turns whose first token came within a given time, on a log scale, pooled over two 240 s runs per
engine. The calls run beside 12 closed-loop agent sessions with prompts of 6–30k tokens, on the RTX 5060 Ti. (a) The
agents' prompts are fresh each turn. (b) Each grows by a tool result, so the agents reuse the CPU tier. The thin
horizontal line marks the 95th percentile.

## 3. Design

The pull requests add two rules to vLLM's priority scheduler (Figure 5, Algorithms 1 and 2). We write p(x) for a
request's priority, where a lower value is served first. The voice tier runs at priority 0 and the agents at 2. Each
rule is a switch, off by default, and each counts blocks, tokens and steps rather than time.

<picture>
  <source media="(max-width: 767px)" srcset="figures/method-narrow.svg">
  <img src="figures/method.svg" width="100%" alt="Three panels. (a) A caller's speech goes through speech to text to the vLLM 0.32 scheduler with #63350 and #63351 at priority 0, beside 24 coding-agent sessions at priority 2, and the reply goes through text to speech to the caller. (b) With the KV pool full of agents A and B, vLLM 0.32 leaves the voice turn waiting; with #63350, agent B is preempted, its KV in the CPU tier, and the voice turn takes its blocks. (c) Engine steps from the reply's first token: without #63351 every step also carries an agent's prefill chunk and the reply gets 14 tokens in the span; with #63351 the reply's first 48 tokens run without agent prefill, which then resumes.">
</picture>

**Figure 5. Make-room takes KV for the waiting voice turn, and the prefill hold keeps agent prefill out of its first
tokens.** (a) The voice agent and 24 coding-agent sessions share one GPU and one vLLM engine, at priorities 0 and 2. (b)
With the KV pool full, vLLM 0.32 leaves the voice turn waiting until an agent request frees its blocks, by finishing or
by being preempted for another's growth. Make-room preempts agent B instead, and B's KV is in the CPU tier when it is
preempted. (c) Engine steps from the reply's first token. Without the hold, every step also carries an agent's prefill
chunk, and in the same span the reply decodes 14 tokens. With it, the reply's first 48 tokens run beside decodes only,
and agent prefill resumes after them.

### 3.1 Make-room

At the start of each step, before the running loop, make-room finds r, the highest-priority waiting request that the
waiting loop can schedule in this step (Algorithm 1, line 2). A request still loading its KV from the CPU tier,
compiling a grammar, awaiting streaming input or awaiting output still in flight from before a preemption is passed
over, as is one whose LoRA adapter cannot join the running ones. If r lacks KV for its whole remaining sequence, net of
its prefix-cache hit, make-room preempts running requests of a strictly lower priority through vLLM's own preemption
path. It takes victims in the order vLLM's running loop preempts in, the lowest priority first and then the latest
arrival, which spares older work. It preempts as few as make r fit, at most N in a step, and none if N would not suffice
(lines 4–9). It does nothing in a step whose prefills are deferred, while new requests are paused, or while the KV
connector has block frees pending (line 1).

Three details make the count of room match what vLLM allocates. Prefix caching shares blocks between requests, so a
block counts as freed only when every request that holds it is a victim, and r's own prefix-hit blocks never count,
since admitting r keeps them. The running decodes keep growing until the victims' blocks are freed, one step later under
asynchronous scheduling with the CPU tier, so their growth over those steps is added to r's shortfall (line 7). Blocks
whose free is already deferred count as freed. The room is also counted beside vLLM's watermark and what in-flight
prefills still need, so admitting r cannot leave a running prefill short.

Preemptions do not cascade. Make-room makes no room for r again until the waiting loop has tried to allocate it, and a
victim is not resumed in the step that preempted it. While a higher-priority prefill is in flight, the waiting loop
stops at the first lower-priority request that holds no KV and would take blocks in-flight prefills still need, so a
victim cannot take its own blocks back.

Victims keep their KV. With vLLM's offloading connector, a victim's pending stores to the CPU tier are flushed before
its blocks are freed through vLLM's own path, so its KV is in the CPU tier when it is preempted. A resume reloads what
the tier still holds and recomputes the rest.

vLLM's V0 scheduler had a form of this rule [[8](#ref-8)]. It preempted for the highest-priority waiting request until
the request fit, with no cap, but only with chunked prefill off and by recomputing its victims. Make-room brings the
rule to V1, under chunked prefill and asynchronous scheduling, with a cap of N a step and the victims' KV kept.

```text
Algorithm 1. Make-room, before the running loop of each scheduler step (at most N victims a step).
p(x) is a request's priority, and a lower value is served first.

 1  if prefills are deferred, new requests are paused, or the KV connector has block frees pending: return
 2  r ← the highest-priority waiting request that the waiting loop can schedule in this step
 3  if r is none, or no running request has p > p(r), or room was made for r and the waiting loop
        has not tried r since, or (with the hold) h < p(r) in Algorithm 2: return
 4  C ← up to N running requests with p > p(r), the lowest priority first, then the latest arrival
 5  s ← the free blocks r lacks for its whole sequence, net of its prefix-cache hit, beside the
        watermark and what in-flight prefills still need
 6  if s > freed(C): return                         ▹ not even every candidate makes room
 7  s ← s + growth(running)                          ▹ the decodes' growth until the frees land
 8  if s ≤ freed(∅): return                          ▹ r fits, or deferred frees make the room
 9  V ← the shortest prefix of C with s − growth(V) ≤ freed(V); if none: return
10  preempt each request in V through vLLM's preemption path ▹ its KV is in the CPU tier
11  mark r untried, and in this step's waiting loop stop at a request in V
12  in every step's waiting loop, over requests holding no KV: while a prefill x is in flight, stop
        at the first q with p(q) > p(x) that would take blocks in-flight prefills still need

freed(X): the blocks whose every holder is in X or has a free deferred, less those r takes as its
          prefix-cache hit, plus the blocks that in-flight prefills in X still reserve.
growth(X): the blocks the requests in X would take until the frees land, over one step, or two with
           deferred frees.
```

### 3.2 The prefill hold

Make-room gets the voice turn into the batch, and the prefill hold keeps the batch's prefill out of the turn's way.
Lower-priority prefill sits out the steps of a higher-priority request's first T output tokens (Algorithm 2). The hold
starts in the step in which the highest-priority waiting request fits, its victims' frees counted, continues while that
request prefills, and ends once it has T output tokens, counting those still in flight under asynchronous scheduling. In
a held step, running lower-priority prefill chunks are skipped and no waiting lower-priority request is admitted, while
every decode proceeds. A held request that already holds KV blocks is skipped rather than allowed to block the queue, so
the requests behind it are still considered.

Two bounds keep the hold from stalling the engine. A waiting request or a KV load holds nothing while the engine already
runs as many requests as it may (its running slots), since neither could be promoted and holding prefill would only
stall the running requests. A voice turn whose prefix was evicted between turns reloads it from the CPU tier over
several steps. The load holds lower-priority prefill for its first T steps and again from the step in which it
completes, so a load that never completes holds nothing after T steps. In a closed-loop probe at T = 16, a build that
held nothing during loads made the calls' first token 125–131 ms later at the median than the version measured before
the main evaluation, which held through them (§6.5). The load's completion is seen only at a step boundary, and agent
prefill makes those steps about three times longer.

The hold covers a reply's first tokens and not its whole decode. A voice agent speaks once its first sentence is
complete, so protecting the rest of the reply buys the caller nothing and costs the agents prefill steps.


```text
Algorithm 2. The prefill hold, in each scheduler step (window T, in output tokens).

 1  h ← the highest priority among the requests whose first tokens are due: running requests
        with fewer than T output tokens, those still in flight counted, and, while a running
        slot is free, requests whose KV load is done or began fewer than T steps ago
 2  if r (Algorithm 1) fits once the victims' frees land and a running slot is free:
        h ← min(h, p(r))
 3  if h is none: schedule the step as vLLM does
 4  running loop: for each running request q
 5      if q is prefilling and p(q) > h: skip q      ▹ its chunk sits out this step
 6      else schedule q as vLLM does                 ▹ every decode runs
 7  waiting loop: for each waiting request q, in vLLM's order
 8      if p(q) > h: skip q if it already holds KV blocks, else stop
 9      else admit q as vLLM does
```

### 3.3 Sizing the window

T is counted in output tokens, but the voice agent does not speak its first token. It hands speech synthesis its first
complete sentence, at least 20 characters by LiveKit's default [[13](#ref-13)], so T must cover at least the first
sentence. The evaluation's records keep each reply's tokens but not its text, so we size T to cover whole replies, which
bounds them; §6.3 counts the first sentences afterwards, 39 tokens at most. In a single-sample campaign before the main
evaluation, this agent's replies ran 20–22 tokens at the median and at most 40, and we set T = 48. Figure 6a gives the
lengths of the 5,571 replies the voice agent completed in the main evaluation. They ran 23 tokens at the median and 52
at most, so a 16-token window ends before 79% of them do and a 48-token window covers all but one. Within a reply a
longer window costs the agents nothing more, because a hold ends when its reply ends (§6.3), though a KV load that never
completes holds for up to T steps.

<img src="figures/report-window.svg" width="100%" alt="Three panels. (a) A step curve of the replies' output tokens: 21% end within 16 tokens and all but 1 of 5,571 within 48. (b) The 40th token's p95 falls from about 3.8 s at a 16-token window to about 1.0 s at 48 and 64, while the first token stays near 0.4 s. (c) The agents' prompt tokens per second fall 8% from 16 to 48 and no further.">

**Figure 6. A window that covers the replies keeps their tokens fast, and a longer one costs the agents nothing more.**
(a) The share of the voice agent's 5,571 completed replies in the main evaluation that end within a given number of
output tokens, with the 16- and 48-token windows marked. (b) The p95 time to the 40th token, and to the first, of
40-token replies beside closed-loop agents against the window. Circles are runs and lines join their medians. (c) The
agents' prompt tokens per second in the same runs.

### 3.4 Properties

The rules share four properties that bear on deploying them.

- **Off by default.** With both switches off, the scheduler produces the same schedules as vLLM's.
- **Counted, not timed.** No rule reads a clock or models the hardware. The step cap, which the rules leave to the
  operator, decides what a held step costs on a given GPU (§6.7).
- **Bounded.** Make-room preempts at most N requests a step and none that would not make room. The hold never cuts a
  decode, holds nothing for a waiting request or a KV load while every running slot is taken, and ends after T output
  tokens, or after T steps of a load that never completes.
- **Narrow.** Scheduling among requests of equal priority, preemption for a running request's growth, and the order of
  decodes are vLLM's.

## 4. Implementation

The rules are proposed to vLLM as two stacked pull requests. #63350 adds make-room as
`--max-priority-preemptions-per-step N` [[9](#ref-9)], and #63351 adds the prefill hold as
`--priority-prefill-budget-cap` with `--priority-prefill-budget-cap-tokens T`, 16 tokens by default [[10](#ref-10)]. The
repository's other documents call the hold the budget cap, after this flag. Both pull requests carry tests that run
vLLM's real scheduler on CPU, and the hold adds counters for the steps it held and the prefill tokens it held back in
them.

The evaluation runs vLLM 0.32.0, which predates both. At each engine start, a build step generates a scheduler from the
installed vLLM's, without modifying the installed package, and we call the result the generated scheduler. It applies
the pull requests' scheduler diff, ported to 0.32.0, and a patch of its own that reads the switches from vLLM's
additional configuration, logs the rules' counts, and adds a third switch, a class-aware KV reserve. The pull requests
leave the reserve out, and the evaluated configuration turns it off. In the campaign before the main evaluation, the
reserve made room in place of make-room (§6.5). The engine loads the generated scheduler with `--scheduler-cls`, and the
build refuses any vLLM other than 0.32.0. A trace test runs the generated scheduler and the pull requests' patches on
the same request traces and requires every step's schedule to match.

Two disclosures bound what the evaluation measured. The pull requests' branches on vLLM's main ran only in their CPU
tests, so every GPU number in this report comes from the generated scheduler. And it matched the pull requests as they
stood before their revision of 25 October 2026. The revision counts a reply's first tokens still in flight under
asynchronous scheduling, so the hold ends one step sooner. Only a reply at least T tokens long reaches that step, which
248 of the 305 replies with T = 16 (81%) did and 1 of the 2,469 with T = 48. §6.1 reports five rounds of the change on
the final code, at T = 48 only, so Table 2's T = 16 row holds one step longer than the final code would for most of its
replies.

**What a step costs.** The rules run on the host, inside `Scheduler.schedule()`. Figure 7 times that call on vLLM
0.32.0's real scheduler on a 2.1 GHz Xeon Silver 4110, a CPU host faster than the GPU node's (Appendix C), in the 13
steady states of the pull requests' cost tables. With the switches off, the call costs what vLLM's does, within 1.1% in
every state. With the deployed values it costs 36–149 µs more while only lower-priority work runs, at most 18% of the
call's own time. It costs 0.54–0.64 ms more when a waiting high-priority request needs more room than two victims free,
1.7 ms more when that request's prefix-cache hit is 999 blocks long, and 160–247 µs more when every running slot is
taken. A voice reply's decode step on the RTX 5060 Ti takes 12 ms with nothing beside it, and 16–20 ms beside
agent prefill that the hold keeps back.

<img src="figures/report-step-cost.svg" width="100%" alt="A dot plot of 13 scheduler states in three groups on a log axis from 0.1 to 20 ms. In each state the change with its switches off sits on vLLM 0.32's ring. As deployed it adds 36 to 149 microseconds when only lower-priority work runs, 539 to 1,719 microseconds when a waiting turn needs more room than two victims free, and 160 to 247 microseconds with every running slot taken, all far left of a 16 to 20 ms decode step.">

**Figure 7. With its switches off the scheduler costs what vLLM's does, and the deployed values add 36 µs to 1.7 ms to a
scheduler call.** Wall time of one `Scheduler.schedule()` call on vLLM 0.32.0's scheduler on a 2.1 GHz Xeon, in the 13
steady states of the pull requests' cost tables, each mark the 10th-percentile step of its state, on a log scale. Rings
are vLLM 0.32, small dots the change with its switches off, and blue dots the change as deployed, labeled with what it
adds. The shaded band is a voice reply's decode step on the RTX 5060 Ti beside agent prefill that the hold keeps
back; with nothing beside it, a step takes 12 ms.

**The measurement harness.** Around the engine, the repository holds the system that measured it. A gateway written in
Rust gives each client a class and a priority from its identity, and adds less than 0.5 ms to a voice request's first
token at the median, paired against a direct call. The voice agent is a LiveKit Agents worker [[13](#ref-13)] with its
stock turn handling and fixed endpointing, the callers are LiveKit clients that play scripted lines and record what they
hear, and AIPerf [[14](#ref-14)] replays the agents' traces. Appendix A gives the records behind every table and figure,
and what regenerates each.

## 5. Evaluation setup

### 5.1 Hardware and software

The GPU node holds one RTX 5060 Ti 16 GB on a PCIe 3.0 x8 link, in a host with two 1.7 GHz Xeon E5-2609 v4 and 188 GiB
of RAM. It serves the whole cascade. Whisper large-v3-turbo [[15](#ref-15)] on vLLM transcribes the caller, Kokoro-82M
[[16](#ref-16)] speaks the reply, and vLLM 0.32.0 serves Qwen3-4B-Instruct-2507 in FP8 [[17](#ref-17)]. The engine has a
fixed 5 GiB KV pool (36,400 tokens), prefix caching, a 20 GiB CPU tier written through, priority scheduling and the
512/256 step cap. A second host runs the media server, the gateway and the load generators. Every configuration in
§6.1–§6.6 runs on this node with the same models and KV pool, and differs from tuned vLLM in its engine, its scheduler
or what sits in front of it. §6.7 repeats the main comparison on an A100 SXM4 80 GB (Appendix C).

### 5.2 Workloads

**Voice.** Two scripted callers place calls through LiveKit to the voice agent. Each call has one unmeasured warm-up
turn and 25 measured turns, and the callers play the same script in every run, with a barge-in on 9 of the 50 measured
turns. The callers speak lines synthesized in six Kokoro voices, and the agent ends a turn a fixed delay after voice
activity stops, so that a turn detector's variance does not enter the comparison. V2V is read from the waveforms at the
caller, on one clock. A turn the agent does not answer within 30 s counts at 30 s, and no turn of the runs in §6.1 went
unanswered.

**Agents.** AIPerf replays the SemiAnalysis Claude Code traces closed-loop through the gateway, with their sessions,
subagents and think times, the think times capped at 120 s. Each trace records a request as a list of 64-token prefix
blocks. We shrink every block 32×, which shrinks each request and its output 32× while keeping which requests share
which prefixes, so every request fits the 32k context. The main scenario runs 24 sessions drawn from the 48 traces with
the longest contexts (a median request of 13.9k tokens), where running agents' KV fills the pool. A second scenario runs
16 sessions of the whole corpus (a median of 7.9k tokens), where the pool has room (§6.6). The gateway sends voice
requests at priority 0 and agent requests at priority 2. On tuned vLLM the agents are oversubscribed, with a median TTFT
of 63 s, so their throughput measures what the GPU leaves them.

### 5.3 Metrics and the SLO

Each run reports the p50, p95 and p99 of V2V over its 59 replies, one for each measured turn and, after each barge-in,
the reply to the interruption. The agents' throughput is the prompt and completion tokens per second of the agent
requests that completed in the calls' first 240 s, a window of the same length for every configuration. Their latency is
the p50 and p99 of their TTFT. The SLO is voice alone's p95 V2V plus 250 ms with as many callers. Voice alone ran in the
main evaluation's first set of rounds with two callers, which gives 2.03 s, in the calls-per-GPU rounds with four and
six, 2.04 and 2.13 s, and in the A100's rounds, 1.67 s.

### 5.4 Configurations

Table 1 lists what the evaluation compares. Every configuration but voice alone runs the agents beside the calls. Tuned
vLLM's step cap of 512 tokens with a 256-token threshold per request bounds how long a decode waits behind a prefill
chunk. On the whole corpus at 16 sessions it lay within the noise of every other step cap swept. Beside the 24
long-context sessions, a 1,024-token cap without a threshold did better in each of its five runs on one sample of the
traces. It gave 8.79 s against 10.14 s with 9% more agent throughput, though still 6.8 s over the SLO (Table B3). The
sweep of §6.9 measured the change at that cap and others.

**Table 1. The configurations compared.** Each differs from tuned vLLM only in what the second column says. N is
make-room's cap on victims a step, and T the prefill hold's window in output tokens.

| Configuration | What it changes | Where |
|---|---|---|
| Voice alone | No agents; the SLO's reference | §6.1 |
| Tuned vLLM | vLLM 0.32 with priority scheduling (voice 0, agents 2), a 512-token step cap with a 256-token threshold per request, prefix caching and a CPU KV tier written through | §6 |
| Make-room | #63350 with N = 2 | §6.1–§6.4 |
| Make-room and the hold, T = 16 | #63350 and #63351 with N = 2 and T = 16 | §6.1–§6.4 |
| **The change** | #63350 and #63351 with N = 2 and T = 48 | §6 |
| The abort-and-resubmit router | A gateway that aborts agent requests without a first byte when a voice request arrives, and resubmits them once the voice request has its first byte | §6.1, §6.4, §6.5 |
| vLLM's settings | A 1,024-token step cap without a per-request threshold; a KV watermark of 10%; a limit of 8,192 queued tokens; `--max-num-seqs 12` with #54530 [[18](#ref-18)] | §6.5 |
| Static answers | A gateway that caps agent requests in flight at 2 or 8, or reserves 2,048 tokens of KV for voice | §6.5 |
| Other engines and routers | SGLang 0.5.22's priority preemption, with and without its host tier [[5](#ref-5), [6](#ref-6)]; llm-d's flow control, emulated in the gateway [[19](#ref-19)]; SMG 1.13.0 [[20](#ref-20)] | §6.5 |
| A gateway policy timed by the conversation (the lifecycle policy) | KV headroom held from the caller's speech onset by deferring agent requests, a throttle on agent prefill while a turn is answered, and replies capped at 16 tokens when headroom is short | §6.5 |

### 5.5 Rounds and estimation

In the main scenario each configuration runs in five rounds. Round n replays its own sample of the agents' traces, the
same sample for every configuration, and runs the configurations in an order rotated from round to round, so that none
always runs at the same hour. Other comparisons run fewer: §6.5's single-sample campaign one to five times, the calls
per GPU three rounds, and the A100's 2,048-token rows two. A configuration measured in more than one set of rounds is
pooled over them, which gives 10 runs of tuned vLLM and 15 of the change. We report each configuration's median p95 over
its runs, with their range, beside the share of its replies over the SLO and the p95 of its replies pooled. We compare
configurations round by round on the same sample of the agents. The three sets of rounds replay the same five samples of
the agents with the same callers, so the change's 15 runs are five samples measured three times, not fifteen independent
workloads. A p95 over 59 replies lies between the third- and fourth-slowest reply, so single runs differ by a few
hundred milliseconds. Five rounds that agree in sign would arise by chance one time in sixteen if two configurations did
not differ, so the rounds resolve the direction of a difference, and its size only roughly.

## 6. Results

### 6.1 The two rules bring the voice tail to 2.24 s while the agents keep 97% of their throughput

With both rules and T = 48, the p95 V2V is 2.24 s over 15 runs, against 10.48 s on tuned vLLM. The agents complete 3,203
tokens a second against 3,305 at the same 512-token step cap (Figure 8, Table 2). A 1,024-token step cap gave tuned vLLM
9% more agent throughput in the earlier campaign, and still missed the SLO by seconds (§5.4). The change misses the SLO
by 0.20 s at the median, and none of its 15 runs met it (2.13–2.40 s). Reply by reply, 286 of its 885 replies (32%) took
longer than the SLO's 2,034 ms, where the SLO allows 5%, and the p95 of all of them taken together is 2,268 ms. On tuned
vLLM 532 of 590 replies (90%) took longer. The fastest run of tuned vLLM missed it by 6.7 s. Of the configurations an
operator can deploy today, the abort-and-resubmit router comes nearest, 0.49 s over the SLO. Round by round, on the same
sample of the agents, the change was faster than the router in all five rounds, by 223–723 ms (390 ms at the median).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-headline-narrow.svg">
  <img src="figures/report-headline.svg" width="100%" alt="Two panels with six rows. (a) Median p95 voice-to-voice latency on a log axis: voice alone 1.78 s, tuned vLLM 10.48 s, the abort-and-resubmit router 2.52 s, make-room 5.09 s, make-room with a 16-token hold 3.09 s, and with a 48-token hold 2.24 s, against an SLO of 2.03 s. (b) The agents' tokens per second: 3,305, 3,332, 3,414, 3,298 and 3,203, that is 100%, 101%, 103%, 100% and 97% of tuned vLLM's.">
</picture>

**Figure 8. The two rules bring the voice tail to 0.20 s over the SLO, and the agents keep most of their throughput.**
Two callers beside 24 coding-agent sessions on the RTX 5060 Ti. (a) Each run's p95 V2V as a tick, and the median of each
configuration's runs as a marker and a label, on a log scale. The dashed line is the SLO, voice alone's median plus 250
ms. (b) The agents' tokens per second over the calls' first 240 s, each run a tick, with each median's share of tuned
vLLM's ten-run median. Round by round in its own set, make-room alone gave 0.4% more than tuned vLLM at the median
(§6.4).

**Table 2. The headline, beside 24 coding-agent sessions whose KV fills the pool.** The p95 V2V is the median of the
runs' p95, with their range, and the pooled p95 that of all the runs' replies taken together. A reply over the SLO took
longer than 2,034 ms, and the SLO allows 5% of them. The agents' tokens a second are the median of the runs, counted
over the calls' first 240 s, and are nearly all prompt tokens (§6.4).

| Configuration | Runs | p95 V2V | Pooled p95 | Replies over the SLO | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 5 | 1,784 ms (1,764–1,819) | 1,796 ms | 1 / 295 (0.3%) | – | – |
| Tuned vLLM | 10 | 10,477 ms (8,686–12,115) | 10,609 ms | 532 / 590 (90%) | 3,305 | 63 / 106 s |
| The abort-and-resubmit router | 5 | 2,522 ms (2,516–2,868) | 2,717 ms | 130 / 295 (44%) | 3,332 | 32 / 193 s |
| Make-room | 5 | 5,085 ms (4,693–6,261) | 5,290 ms | 263 / 295 (89%) | 3,414 | 67 / 112 s |
| Make-room and the hold, T = 16 | 5 | 3,094 ms (2,868–3,192) | 3,090 ms | 128 / 295 (43%) | 3,298 | 71 / 105 s |
| **The change** (T = 48) | 15 | **2,236 ms** (2,132–2,397) | 2,268 ms | 286 / 885 (32%) | 3,203 | 72 / 111 s |

The change moves the whole distribution of replies, not only its tail (Figure 9). Over its 885 replies the median V2V is
1.96 s and the slowest takes 3.71 s. On tuned vLLM the median reply takes 4.61 s, and 46% of its 590 replies take longer
than 5 s. In the five rounds the router ran, its replies were slower than the change's at every percentile from the
median up, by 58 ms at the median, 364 ms at the 90th percentile and 703 ms at the 99th.

<img src="figures/report-cdf.svg" width="100%" alt="Cumulative distributions of every reply's voice-to-voice time on a log axis. All but one of voice alone's replies fall below the 2.03 s SLO. The change's lie between about 1.6 and 3.7 s, with a median of 1.96 s. The router's lie just to the right of the change's, except for the change's few slowest. Tuned vLLM's spread from about 1.6 to 19 s, with a median of 4.61 s.">

**Figure 9. The change shifts every voice reply, not only the slowest.** The share of replies that began within each
voice-to-voice time, beside 24 coding-agent sessions, on a log scale. Each line pools a configuration's replies over its
runs: 295 for voice alone, 590 for tuned vLLM, 295 for the router and 885 for the change. The dashed vertical line is
the SLO, and the thin horizontal line marks the 95th percentile.

The change's rows ran on the generated scheduler as it stood before the pull requests' revision (§4). Five more rounds
on the final code replayed the same five samples of the agents. Besides counting the first tokens in flight, they
reached the GPU servers' Unix sockets through an SSH tunnel and staggered the callers' joins by a quarter second. They
gave a p95 of 2.27 s against 2.25 s for the same rounds before, −45 to +77 ms round by round. The differences lie within
the spread of the change's own runs, so the numbers above stand for the code proposed to vLLM (Appendix B).

### 6.2 Make-room removes the wait for KV, and the hold removes agent prefill from the turn's steps

Of the 3.68 s that sharing the GPU adds to the mean turn on tuned vLLM, 3.58 s fall between the end-of-turn decision and
the LLM's first sentence (Table 3). Endpointing, speech synthesis and transport move by at most 69 ms from voice alone's
in any configuration. The rest divides into the two waits of §2.4.

**Table 3. Where the voice turn's time goes, by configuration.** Mean stage times in ms over every reply of each
configuration's runs, in the order a turn passes through them. Their sum is the mean V2V. The LLM's own time to its
first token is in the last column, measured from its request.

| Configuration | Endpointing | Wait for the first token | Decode to the first sentence | Speech synthesis | Transport and playout | Mean V2V | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Voice alone | 756 | 85 | 218 | 211 | 352 | 1,623 | 80 |
| Tuned vLLM | 825 | 2,815 | 1,066 | 250 | 343 | 5,299 | 2,795 |
| Make-room | 820 | 783 | 1,207 | 250 | 356 | 3,415 | 771 |
| Make-room and the hold, T = 16 | 822 | 299 | 439 | 236 | 355 | 2,151 | 289 |
| **The change** (T = 48) | 822 | 289 | 262 | 242 | 355 | 1,971 | 282 |
| The abort-and-resubmit router | 822 | 283 | 390 | 232 | 357 | 2,084 | 275 |

Make-room takes the wait for the first token from 2,815 ms to 783 ms at the mean, as the KV case predicts (Figure 3a).
It leaves the turn's prefill and reply sharing steps with agent prefill, so the decode of the first sentence rises to
1,207 ms, from 1,066 ms on tuned vLLM, and the p95 is 5.09 s, 3.05 s over the SLO. The hold covers the turn's prefill
and KV load as well as its first tokens. At T = 48 it takes the wait for the first token on to 289 ms and the decode to
262 ms, against 218 ms with voice alone, as the prefill case predicts (Figure 3b). In every round the configurations
rank the same. Tuned vLLM is slowest, then make-room alone, then both rules at T = 16, and both rules at T = 48 are
fastest.

### 6.3 A window shorter than the replies leaves their later tokens beside agent prefill

At T = 16 the p95 stays at 3.09 s (2.87–3.19 s), 1.06 s over the SLO. The turn's first token comes as fast as at T = 48
(299 against 289 ms at the mean), and what differs is the decode of the first sentence, 439 against 262 ms. Of the
replies, 79% run past 16 tokens (Figure 6a), and their tokens after the 16th decode beside agent prefill again.

A sweep of the window against whole replies of 40 tokens, beside closed-loop agents, places the knee where the window
covers the reply (Figure 6b). Windows of 16, 24 and 32 tokens left the 40th token's p95 at 3.76, 2.64 and 1.85 s, and
windows of 48 and 64 tokens at 1.04 and 1.02 s. The first token, which the window does not touch, stayed at 326–448 ms
at every window. The agents' prompt throughput fell 8% from 16 to 48 tokens and no further at 64 (Figure 6c), since a
hold ends with its reply, so a window longer than the replies costs no more than one just long enough. The voice agent
here is prompted for one or two short sentences, and an agent prompted for longer answers needs a longer window. The
records keep each reply's tokens but not its text. So we asked the engine for 320 of the agent's replies to the callers'
lines again, and counted the tokens up to the one that completes each reply's first sentence. They took 15 tokens at the
median, 29 at the 95th percentile and 39 at most: a 48-token window covers every first sentence, and a 16-token window
56% of them.

### 6.4 The agents pay through the hold, not through the preemptions

The agents keep 97% of their token rate with the change (Table 4, Figure 10): 96.7% at the median of the ten rounds that
ran both on the same sample, and 94.2–98.7% round by round. The rate counts the tokens of a replay of coding agents'
requests over the calls' first 240 s. The replay rebuilds each recorded request's prompt from its prefix blocks and asks
for its recorded output, both scaled down 32 times; it runs no task and judges no result. Over the whole call 99.5% of
the agents' tokens are prompt tokens, and the engine generates 16 tokens a second for them with the change and 17 on
tuned vLLM. So the 97% measures the prefill work the agents keep, not the tasks they would complete. Their median TTFT
rises from 63 s to 72 s, and their p99 of 111 s lies within the spread of tuned vLLM's runs (96–129 s). The engine's own
counts show where the 3% goes. The hold kept lower-priority prefill out of 33% of the engine's steps at T = 48 and 25%
at T = 16. With make-room alone, which holds nothing, the agents completed about as many tokens as on tuned vLLM,
0.1–6.2% more round by round and 0.4% more at the median.

**Table 4. What each configuration takes from the agents.** Medians over each configuration's runs, counted over the
calls, except the agents' tokens a second, which count the calls' first 240 s as in Table 2. Make-room's preemptions are
its own count. vLLM counts every preemption once the victim produces output again, including those for a running
request's growth, so its count is a lower bound. vLLM counts where a request's prompt tokens came from only at its first
scheduling, and the loads are the tokens the CPU tier loaded beyond those, the victims' restores among them.

| Configuration | Steps held | Make-room's preemptions | vLLM's preemptions | CPU-tier loads | Agents' tokens/s | TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Tuned vLLM | – | – | 31 | 212,630 | 3,305 | 63 / 106 s |
| The abort-and-resubmit router | – | – | 7 | 273,461 | 3,332 | 32 / 193 s |
| Make-room | 0% | 34 | 51 | 171,736 | 3,414 | 67 / 112 s |
| Make-room and the hold, T = 16 | 25% | 32 | 47 | 117,049 | 3,298 | 71 / 105 s |
| **The change** (T = 48) | 33% | 31 | 46 | 117,044 | 3,203 | 72 / 111 s |

<img src="figures/report-agents.svg" width="100%" alt="Three panels in five rows. (a) The agents' p99 time to first token: 106 s on tuned vLLM, 193 s with the router, and 112, 105 and 111 s with make-room alone, with the hold at 16 tokens and at 48. (b) The hold's share of steps: 0%, 25% and 33% for the three configurations with make-room. (c) Make-room's preemptions a run: 34, 32 and 31.">

**Figure 10. The agents' first-token tail stays at tuned vLLM's with the change and nearly doubles with the router.**
Beside the 24 sessions on the RTX 5060 Ti, each run a tick and each configuration's median a marker and a label. (a) The
agents' p99 time to first token. (b) The share of the engine's steps in which the hold kept agent prefill out. (c)
Make-room's own count of its preemptions. Tuned vLLM and the router run neither rule.

Make-room made a median of 31 preemptions a run with the change, about one for every two of the voice agent's replies.
The preemptions equal about 30% of the agent requests completed during the calls (18–41% over the 15 runs). Yet the CPU
tier loaded less beyond the requests' first scheduling with the change, 117k tokens over a median of 46 preemptions that
vLLM counted, than on tuned vLLM, 213k over 31, about 2.7k tokens a counted preemption against 7.0k. This suggests that
make-room's victims, the latest arrivals, bring little back from the tier. What a resume recomputes rather than restores
is not visible in vLLM 0.32's counters, which count a request's computed prompt tokens at its first scheduling only.

The router pays differently. It aborts agent requests that have no first byte yet, which frees their KV at once and
keeps 101% of tuned vLLM's throughput, and their resubmissions restore their prefixes from the CPU tier. But an aborted
request waits again at the front of the queue, and one aborted more than once waits each time, so the agents' p99 TTFT
reaches 193 s (160–232 s over the runs).

### 6.5 A control plane in front of the engine makes room only by aborting agent work

No configuration without an engine change reaches the SLO beside the 24 sessions (Figure 11, Table B3). These runs come
from a campaign before the main evaluation, which replayed one sample of the agents' traces, one to five runs a
configuration, against an SLO of 2.02 s. Other settings of vLLM, caps on the agents' concurrency, SGLang with or without
its host tier, llm-d's flow control and SMG leave the p95 between 8.6 and 12.1 s. A static 2,048-token reservation
halves the tail to 5.67 s and stays 3.7 s over the SLO. A limit on queued tokens, another of vLLM's settings, refuses
voice turns with the agents' requests, and 48 of 50 turns went unanswered.

That campaign differs from the main evaluation in three ways. Every turn counted, the first of each call included,
though leaving the first turns out moved the three medians checked by at most 86 ms. Its rows with the engine change ran
an earlier version of the generated scheduler, which took the victims holding the most KV first and had three gaps fixed
since (Appendix C). In the main evaluation's first five rounds, which ran both versions, the fixed version's p95 at T =
48 differed from the earlier one's by +8 ms at the median (−158 to +164 ms). And the rows of the pull requests'
configurations ran after the campaign's rounds, in a session of their own.

<img src="figures/report-frontier.svg" width="100%" alt="A scatter of configurations: the agents' tokens per second against the median p95 voice-to-voice latency on a log axis. Tuned vLLM, its settings, the static answers, other engines and the routers sit between about 5.7 and 12 s, and a queued-token limit at the 30 s cap. The abort-and-resubmit router is at 2.59 s, and the configurations with the engine change run from 2.06 s to make-room alone's 5.77 s, beside the static reservation's 5.67 s. The SLO is 2.02 s.">

**Figure 11. No configuration without the engine change comes within a second of the SLO except by aborting agent
work.** Each configuration's median p95 V2V against its agents' tokens per second over the calls' first 240 s, with a
vertical line over the range of its runs, beside 24 coding-agent sessions on one sample of their traces. These runs come
from the campaign before the main evaluation (Table B3). Gray marks vLLM without the engine change and every comparator,
and blue the engine change, which ran the earlier version of the generated scheduler (§6.5). The pale blue diamond is
the lifecycle policy over it, with the KV reserve on, the hold at T = 16 and every reply capped at 16 tokens. Shapes
mark the families in the legend. The dashed line is the SLO, 2.02 s, and unanswered turns count at 30 s.

A gateway policy timed by the conversation, the lifecycle policy, could not act here. It holds KV headroom for a call
from the caller's speech onset by deferring new agent requests, throttles agent prefill while a turn is answered, and
caps a reply at 16 tokens when the headroom is short. On tuned vLLM it left the p95 at 10.99 s, against 10.14 s for
tuned vLLM's own five runs on the same sample of the agents' traces. Running agent requests held the pool, and deferring
new requests frees nothing that running ones hold, so the headroom was never there and the policy capped all 60 of its
realtime requests in every run. Over the earlier generated scheduler, with the hold at T = 16 and its 2,048-token KV
reserve on, the same policy reached 2.06 s. There the reserve kept the pool from filling, so make-room never fired, and
every reply was capped at 16 tokens and so fitted the window. Leaving out the policy's headroom, its throttle or its
lifecycle events moved the p95 by less than the spread of its runs (2.03–2.05 s). Its reply cap and the hold did matter.
Without the cap the p95 rose to 3.84 s, and the engine alone, with no policy, gave 4.03 s, because full replies outran
the 16-token window. Without the hold it rose to 5.29 s. With a 48-token window and full replies, the reserve again made
room without a preemption, and the policy gave 2.21 s against 2.19 s for the pull requests' configuration. The room came
from the engine's reserve, and what the policy added was a shorter reply.

The abort-and-resubmit router is the one control-plane answer that makes room, and it does so by aborting admitted work
(§6.4). A control plane sees requests, not KV blocks or steps. It can hold back work that has not started and end work
that has, losing what it computed, but only the engine can preempt a decoding request and keep its work, or keep agent
prefill out of a reply's steps.

### 6.6 The gain shrinks where the pool has room, and the tail grows as calls overlap

Where the pool has room, make-room has little to do. With 16 sessions of the whole corpus, it preempted a median of 1
agent request a run (0–7), and the hold kept agent prefill out of 4% of the steps (3–13%). The change still had the
lower p95 in every round, by 109–1,413 ms, but met the SLO in only one (Figure 12a). Its p95 was 2.05 s (2.02–2.42 s),
15 ms over the SLO at the median, against tuned vLLM's 2.52 s (2.15–3.83 s), at 6,815 agents' tokens a second against
7,131. The gain over tuned vLLM is 0.47 s here, against 8.24 s beside the long-context sessions. What remains of the
turn's delay is again in the LLM's first sentence, 0.42 s from the end of the turn at the mean, against 0.58 s on tuned
vLLM and 0.30 s with voice alone.

<img src="figures/report-margin.svg" width="100%" alt="Four panels. (a) Over five rounds with 16 sessions of the whole corpus, the change's p95 is at or below tuned vLLM's in every round, from 2.02 to 2.42 s against 2.15 to 3.83 s, near the 2.03 s SLO. (b) Beside the 24 sessions the change's p95 is over each SLO by a median of 202, 269 and 257 ms with 2, 4 and 6 callers. (c) The hold covers 33%, 53% and 64% of the steps. (d) The agents make 3,156, 3,055 and 2,716 tokens a second, on the same three samples.">

**Figure 12. The change helps less where the pool has room, and overlapping calls hold the agents' prefill for longer.**
(a) Sixteen sessions of the whole corpus beside two callers. Each round's p95 V2V on tuned vLLM (gray) and with the
change (blue), joined by round. (b–d) The change beside the 24 sessions with 2, 4 and 6 callers, on the first three
samples of the agents. They give the p95 over each caller count's own SLO, the share of engine steps in which the hold
kept agent prefill out, and the agents' tokens per second over the calls' first 240 s. Ticks are runs and markers their
medians.

More calls make the hold longer (Figure 12b and 12c). The hold keeps lower-priority prefill out while any call's first
tokens are due, so overlapping replies extend it. The rounds with 4 and 6 callers replayed the first three samples of
the agents. From the second of them on, the callers joined their rooms a quarter second apart rather than at once, a
stagger that ends within each call's warm-up turn. Each against its own SLO, the change's p95 missed in every round, by
269 ms (206–370) and 257 ms (226–333) at the median, against 202 ms with two callers on the same samples. The hold grew
to 53% and 64% of the engine's steps, from 33%, and the agents' throughput fell 3% and 14%, to 3,055 and 2,716 tokens a
second from 3,156. The limit is where the windows of successive replies overlap. A test on vLLM's scheduler on CPU sends
replies at least 48 tokens long at a fixed rate beside six agents' prefills. At one reply every 200 steps the agents
kept 82% of the prompt tokens they get on vLLM, and at one every 50 they kept 9%. At one every 40, where each reply's
window of about 49 steps overlaps the next, they kept none. At 16–20 ms a step, that is about one realtime reply a
second across all calls.

### 6.7 The rules carry to an A100

An A100 SXM4 80 GB ran the main comparison like for like, on the same five samples of the agents. It ran the same model,
a 5 GiB KV pool, the 512-token step cap, a 20 GiB CPU tier, 24 sessions of the long traces and two callers, and only the
card, its precision path and the host changed. The A100 has no FP8 units, so the model ran on Marlin's weight-only FP8
kernels [[21](#ref-21)]. Voice alone gave a p95 of 1.42 s, so this card's SLO is 1.67 s. The change took the p95 from
4.99 s to 1.63 s, under the SLO in all five rounds by 7–60 ms (Figure 13a, Table 5). Round by round the agents'
throughput was 94–111% of tuned vLLM's, and their median TTFT moved by −1.2 to +1.8 s. With 32 sessions at the same
pool, in one round, the p95 was 1.62 s against 5.22 s, at 98% of tuned vLLM's throughput.

<img src="figures/report-a100.svg" width="100%" alt="Two panels. (a) In each of five rounds tuned vLLM's p95 is 4.2 to 5.7 s and the change's 1.61 to 1.66 s, under the 1.67 s SLO. (b) At 512-token steps the change moves the median from 4.99 s to 1.63 s at about 8,600 agent tokens a second; at 2,048-token steps from 3.66 s to 1.83 s at about 14,600 tokens a second, just over the SLO.">

**Figure 13. On an A100 the rules meet the card's SLO in every round, and the step cap trades the voice's tail against
the agents' throughput.** An A100 SXM4 80 GB at the RTX 5060 Ti's KV pool, beside the same 24 sessions. (a) Each round's
p95 V2V at 512-token steps on tuned vLLM (gray) and with the change (blue), joined by round, against the card's SLO of
1.67 s. (b) The p95 V2V against the agents' tokens per second at 512- and 2,048-token steps. Open circles are runs and
filled circles medians, and arrows lead from tuned vLLM to the change at each step cap.

**Table 5. The A100 SXM4 80 GB at the RTX 5060 Ti's KV pool, beside the same 24 sessions.** Medians over the runs, with
the range of the p95 V2V. The 2,048-token rows run vLLM's default steps, 2,048 tokens with no per-request threshold, on
the first two samples of the agents only.

| Configuration | Runs | p95 V2V | Over the SLO | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|
| Voice alone | 5 | 1,419 ms (1,391–1,445) | – | – | – |
| Tuned vLLM, 512-token steps | 5 | 4,986 ms (4,163–5,710) | +3,317 ms | 8,633 | 12 / 26 s |
| **The change**, 512-token steps | 5 | **1,634 ms** (1,609–1,663) | −35 ms | 8,651 | 14 / 26 s |
| Tuned vLLM, 2,048-token steps | 2 | 3,663 ms (3,289–4,036) | +1,994 ms | 14,536 | 1.5 / 14 s |
| The change, 2,048-token steps | 2 | 1,826 ms (1,810–1,843) | +157 ms | 14,635 | 1.5 / 15 s |

The cards differ in how long a turn waits once KV is short, more than in how often. Sampled every half second beside
tuned vLLM, on the same five samples, the engine's free KV was under the 2,048 tokens that two calls' turns take 5.7% of
the time on the A100 and 8.1% on the RTX 5060 Ti. But in the KV case the A100's turn waited 1.75 s, against 4.9–5.0 s. A
forecast written before the run scaled the RTX 5060 Ti's measured costs by the A100's published peaks. It placed each
mechanism but underestimated that wait by about 30%, because the A100's decode attention read KV at 42% of its memory
bandwidth (Appendix E).

The step cap trades the voice's tail against the agents' throughput (Figure 13b). The 512-token cap was chosen on the
RTX 5060 Ti (§5.4), and on the A100 it costs the agents about 30% of their throughput on the two samples both caps ran.
At vLLM's default of 2,048 tokens a step with no per-request threshold, tuned vLLM gave the agents 1.40 and 1.42 times
their throughput at 512-token steps on the two samples it ran, with the p95 at 4.04 and 3.29 s. With the rules at that
cap the agents kept 100% and 101% of it, and the p95 was 1.81 and 1.84 s, 0.14 and 0.17 s over the SLO. The larger steps
lengthen the LLM's first token to 412 and 414 ms at p95, against 183 and 216 ms with 512-token steps on the same two
samples. A turn that arrives while steps are in flight waits for them, and no scheduling rule takes back a step already
running.

At a datacenter batch, the scheduler change costs nothing measurable and the hold is the cost. With 256 requests of 48
prompt and 64 output tokens in flight, 26 at priority 0 and 230 at priority 2 in closed loop, the switches off made
1.018 times vLLM's output tokens a second. With the deployed values, the priority-0 requests' p99 TTFT fell from 574 to
177 ms and their tokens came 4.1 times as fast. The priority-2 requests' output fell 8.8%, from 6,589 to 6,008 tokens a
second, and their p99 TTFT rose from 1.1 s to 8.0 s while their median fell. Both are the hold of §6.6, at a rate of
first tokens that no voice tier reaches.

### 6.8 On two cards, consolidating both calls on one does 34% more background work, most of it without the change

A second RTX 5060 Ti 16 GB runs on another host. An operator with the two cards can give the calls one and the agents
the other (dedicated roles), run both cards mixed with one call each (pooling), or put both calls on one card beside as
many agent sessions as it serves within the SLO and run agents alone on the other (consolidation). A resource gate fixed
before the runs decides between them ([RESOURCE-GATE.md](../benchmarks/quality/RESOURCE-GATE.md)). An arrangement
qualifies if at most 5% of its replies, over its rounds pooled, are over its round's SLO for the card's callers. Sharing
has resource value if a qualifying shared arrangement does at least 20% more background work than the best qualifying
dedicated one, counted as the agents' tokens a second over the calls' first 240 s summed over both cards, at the median
of the rounds. Three rounds each replayed a new sample of the agents' traces on one host, the hosts alternating from
round to round, and measured each card's part of every arrangement as a run of its own. The change ran at the serving
profile's 256-token step cap without a per-request threshold (§6.9).

A card of agents alone does the most well short of 24 sessions (Table 6). Without speech servers, with its KV pool grown
into their memory to 61,152 tokens and vLLM's default first-come-first-served scheduling at 2,048-token steps, it did
7,893 tokens a second at 14 sessions, at the median of the rounds, and 4,232 at 24, where its pool thrashes and the
agents' median TTFT is 48 s. The gate runs each dedicated card at its busiest count, so a baseline taken at 24 sessions
would have been 46% lower. The controlled card, tuned vLLM with a shared card's pool and its speech servers idle, did
6,302 tokens a second at its busiest, 12 sessions.

Consolidation on the change qualifies with four sessions on the shared card (Figure 14a). Five of its 177 replies were
over the SLO (2.8%), and at eight sessions 31 (17.5%). Beside the agents-only card at 14 sessions, the arrangement did
10,596 tokens a second, 34.2% more than the best dedicated arrangement, past the hurdle (Figure 14b). Most of that gain
is the arrangement's, not the change's. Stock vLLM's consolidated card qualifies at two sessions, with 4 of 177 replies
over and 18 at four, and that arrangement did 27.3% more than the best dedicated one. The gate counts consolidation on
the serving profile, and against stock vLLM's consolidation the change's own value is 5.4%, short of the hurdle. At two
and four sessions both engines completed the same agents' requests in the window in every round, so there the shared
card's part is the sessions' own pace, which the traces' think times set: the change's 5.4% is the two sessions more at
which it qualifies. Pooling, one call a card beside 12 sessions each, qualifies on neither engine: of each one's 87
replies, 24 of the change's and 53 of stock vLLM's were over the one-caller SLO.

**Table 6. The two cards' arrangements.** Three rounds, each on a new sample of the agents' traces. Replies over the SLO
are pooled over the rounds, and the agents' tokens a second over the calls' first 240 s are summed over both cards, at
the median of the rounds, beside the share they add over the best dedicated arrangement. The agents' TTFT is the median
of their cards' runs. The shared card's sessions are given first, then the second card's.

| Arrangement | Agent sessions | Replies over the SLO | Agents' tokens/s | Over the best dedicated | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|
| Dedicated roles, controlled | 0 + 12 | 0 / 177 | 6,302 | – | 4.4 / 31.0 s |
| **Dedicated roles, best practical** | 0 + 14 | 0 / 177 | **7,893** | – | 5.1 / 41.0 s |
| Pooling, stock vLLM | 12 + 12 | 53 / 87 | 13,188 | does not qualify | 4.7 / 29.2 s |
| Pooling, the change | 12 + 12 | 24 / 87 | 12,111 | does not qualify | 4.8 / 37.2 s |
| Consolidation, stock vLLM | 4 + 14 | 18 / 177 | 10,596 | does not qualify | 0.6 / 10.4 s |
| Consolidation, stock vLLM | 2 + 14 | 4 / 177 | 10,051 | +27.3% | 0.4 / 7.8 s |
| Consolidation, the change | 8 + 14 | 31 / 177 | 13,537 | does not qualify | 0.7 / 11.7 s |
| **Consolidation, the change** | 4 + 14 | 5 / 177 | **10,596** | **+34.2%** | 0.6 / 10.8 s |
| Consolidation, the change, with a warm spare | 4 + 12 | 5 / 177 | 9,652 | +22.3% | 0.5 / 10.7 s |

<img src="figures/report-cards.svg" width="100%" alt="Two panels. (a) The consolidated card's share of replies over the SLO by agent sessions beside both calls, three rounds pooled: stock vLLM 2.3% at 2 sessions, 10.2% at 4, 39.0% at 8, 53.7% at 12 and 92.1% at 24; the change 1.1%, 2.8%, 17.5%, 18.6% and 20.9%, against the 5% requirement. (b) Each arrangement's agents' tokens per second summed over both cards, against 9,471, 20% over the best dedicated arrangement: dedicated roles 7,893 and 6,302; consolidation on stock vLLM at 2 sessions 10,051, on the change at 4 sessions 10,596 and 9,652 with a warm spare; the change at 8 sessions 13,537 and pooling 13,188 and 12,111, all three missing the requirement.">

**Figure 14. On two cards, consolidating both calls on one card does the most background work within the SLO, on
either engine.** (a) The consolidated card's replies over its round's SLO by its agent sessions, three rounds pooled, on
stock vLLM (gray) and with the change (blue), against the requirement of at most 5% (dashed). (b) Each arrangement's
agents' tokens per second over the calls' first 240 s, summed over both cards: bars the median of the rounds, ticks the
rounds. Arrangements that miss the requirement are hatched, and the dashed line is 20% over the best dedicated
arrangement.

The SLO's 250 ms margin sets the envelope. Had it allowed 500 ms over voice alone, the change's consolidated card would
qualify at every count measured, up to 24 sessions with 8 of 177 replies over, and stock vLLM's still only at two, with
12 of 177 over at four. The voice agent's stage times show where the margin goes (Table 7). From eight sessions on, the
transcript comes 51–71 ms later than with the calls alone, on either engine: the agents' steps take the GPU from the
speech server, and the processes time-slice it. On stock vLLM the waits of §6.2 come on top. Its first sentence's p95
grows from 437 ms with the calls alone to 909 ms at four sessions, 2,424 at eight and 9,719 at 24, where the change
holds it to 534, 737 and 768. With the change, from eight sessions on, the transcript and the first sentence take
176–249 ms longer at the median than with the calls alone, nearly all of the margin before any reply's own spread.

**Table 7. The voice turn's stages on the consolidated card, in ms: the transcript from the end of the caller's speech,
the LLM's first token and first sentence from its request, and TTS's first audio from that sentence.** Medians over the
three rounds of each run's median, or of its p95 where the column says so.

| Card | Transcript | LLM first token | LLM first token, p95 | First sentence | First sentence, p95 | TTS first audio |
|---|---:|---:|---:|---:|---:|---:|
| The calls alone | 745 | 79 | 97 | 290 | 437 | 197 |
| The change, 2 sessions | 756 | 81 | 129 | 306 | 467 | 190 |
| The change, 4 sessions | 749 | 91 | 242 | 312 | 534 | 195 |
| The change, 8 sessions | 800 | 145 | 442 | 411 | 737 | 203 |
| The change, 12 sessions | 808 | 223 | 452 | 476 | 765 | 203 |
| The change, 24 sessions | 811 | 215 | 333 | 464 | 768 | 215 |
| Stock vLLM, 2 sessions | 757 | 84 | 125 | 308 | 472 | 192 |
| Stock vLLM, 4 sessions | 746 | 92 | 414 | 294 | 909 | 194 |
| Stock vLLM, 8 sessions | 796 | 189 | 808 | 521 | 2,424 | 201 |
| Stock vLLM, 12 sessions | 811 | 284 | 2,678 | 675 | 3,150 | 202 |
| Stock vLLM, 24 sessions | 816 | 2,929 | 8,493 | 3,792 | 9,719 | 210 |

The gate counts no standby. In the best practical arrangements the second card runs no speech servers, so it cannot
take the calls if the first card fails. A warm spare keeps them resident and runs its agents on the change's engine
beside them. Alone on such a card the agents did the most at 12 sessions, 6,715 tokens a second, and consolidation at
four sessions with a warm spare did 9,652 tokens a second, 22.3% more than the best dedicated arrangement, which has no
standby. Section 7 gives this arrangement's deployment and Appendix D its failover drill.

### 6.9 A 256-token step cap halves the change's replies over the SLO beside 24 sessions

The evaluation of §6.1–§6.7 ran the change at tuned vLLM's step cap of 512 tokens with a 256-token threshold per
request. A sweep screened the change at caps of 256, 512 and 1,024 tokens, with and without the threshold, at windows of
48 and 96 tokens and beside the gateway's static caps on agent requests in flight, and stock vLLM at caps of 256 and
1,024, one run a setting on the sample of the agents the evaluation was tuned on
([the sweep's record](../benchmarks/results/2026-10-27-sweep/README.md)). The cap moved the change most. At 256 without
the threshold, 9 of its 59 replies were over the SLO, against 24 at 512 and 33 at 1,024, and the static caps did less.
Five rounds on new samples then ran the three configurations side by side at 24 sessions, each against its round's own
SLO (Table 8). At 256 the change had fewer replies over the SLO than at 512 in each of the five rounds, 44 of 295
against 94 (14.9% against 31.9%), with a pooled p95 of 2.15 s against 2.26 s, while tuned vLLM had 261 of 295 over. The
smaller steps cost the agents 7% of their token rate, 3,067 tokens a second against 3,303, and their median TTFT rose
from 78 s to 85 s. The smaller steps also left the pool room. At 256 tokens its fewest free blocks in a run were 10–89,
against 0–2 at 512, and vLLM counted 0–4 preemptions a run, make-room's among them, against 39–50. In the sweep's run at
256 it counted none, so there the change's 9 replies over the SLO, against stock vLLM's 51 of 59 at the same cap, are
the hold's. Beside the envelope's four sessions (§6.8) make-room preempted none in any of three rounds. At 256 tokens
the hold and the step cap carry the change, and make-room stays in the profile for a pool that fills, as it does at 512.
The serving profile runs the change at 256 tokens a step without a threshold. No cap brought 24 sessions within the
SLO's 5% allowance, which the shared card meets at four (§6.8).

**Table 8. The step cap beside 24 sessions, in five rounds on new samples of the agents.** Each reply is held to its
round's SLO, and the replies over it are pooled over the rounds and given by round. The 512-token rows keep the
256-token threshold per request, and the 256-token row has none. Medians over the rounds otherwise.

| Configuration | Median p95 V2V | Pooled p95 | Replies over the SLO | By round | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---|---:|---:|
| Tuned vLLM, 512-token steps | 12,010 ms | 11,766 ms | 261 / 295 | 53, 55, 55, 54, 44 | 3,399 | 69 / 109 s |
| The change, 512-token steps | 2,246 ms | 2,259 ms | 94 / 295 | 16, 29, 25, 9, 15 | 3,303 | 78 / 117 s |
| **The change, 256-token steps** | **2,098 ms** | 2,151 ms | **44 / 295** | 10, 17, 10, 4, 3 | 3,067 | 85 / 120 s |

## 7. Operating the change

**The configuration.** The change runs on vLLM with priority scheduling, prefix caching, and a CPU KV tier about four
times the GPU pool, written through. When the audio models share the GPU, the KV pool's size is fixed. Make-room runs
with N = 2 and the prefill hold with T = 48. The evaluation of §6.1–§6.7 ran a step cap of 512 tokens and 256 per
request, and the serving profile runs the cap the sweep chose for the change, 256 tokens a step without a per-request
threshold (§6.9). The gateway sets voice requests to priority 0 and background work to 2 from each client's identity,
so that a background client cannot raise its own priority.

**Two cards.** The arrangement to deploy on two cards is consolidation with a warm spare (§6.8). The first card carries
both calls and at most four agent sessions, the envelope measured beside them, and the second runs agents on the same
engine with its speech servers resident, so that it can take the calls. The gateway places each call on the first card
with a free call slot, in its configured order, and each agent session on the card with the fewest, up to each card's
bound (`max_agents`); a session beyond every bound gets 503 `no_agent_slot` and tries again with its next request. When
the first card fails, its calls move to the second by their next turn, and while the second card carries any call it
holds its agents' requests, letting those already running finish. Over a 24-hour soak of this arrangement, 1.5% of its
5,058 replies were over the SLO and 5.5% in its worst hour, no call failed, nothing restarted, and the second card
carried no call (Appendix D).

**Where it pays.** The change matters when background KV fills the pool, as when agent contexts of 10k tokens or more
meet a pool that holds a few of them. On tuned vLLM the signs are KV usage near 100% while requests wait, and a realtime
TTFT of seconds at p95. Where the pool has room, the change does less (§6.6).

**Choosing T and the step cap.** T should cover the agent's whole replies, as T = 48 covered all but one of these
(§3.3), and a window longer than the replies costs nothing more (§6.3). The step cap bounds the wait for a step in
flight, so a faster GPU can take a larger cap where its voice tail stays under the SLO. On the A100 the default
2,048-token cap did not (§6.7).

**What to watch.** The voice tier's p95 V2V and its time to first token at p95. The engine's KV usage, running and
waiting requests, and preemptions, whose count lags. The hold's counters of held steps and prefill tokens. The agents'
completed tokens a second and their p99 TTFT, where a preemption policy moves its cost.

**How it fails.** A voice turn that N preemptions cannot make room for waits as on vLLM, and load made only of realtime
requests leaves nothing to preempt. Overlapping calls hold the background tier for longer, up to the limit of §6.6. An
engine failure breaks every stream it serves. In the failure drills, each stream of a killed engine received an error
event within 8–11 ms, and a request succeeded again 106 s after the kill, 92 s of it the engine's warm start. A gateway
reload, as a deploy performs, failed none of 42 requests (Appendix D).

## 8. Related work

**Co-serving a latency class with best-effort work.** ConServe [[1](#ref-1)], HyGen [[2](#ref-2)], BROS [[22](#ref-22)],
Niyama [[23](#ref-23)], Chiron [[24](#ref-24)], Echo [[25](#ref-25)] and ProServe [[26](#ref-26)] schedule online
requests beside offline ones, and Valve colocates them in production on 8,054 GPUs with sub-millisecond compute
preemption and dynamic memory reservation [[3](#ref-3)]. We do not claim the goal. We measure it for a workload pair
these systems do not report together, a cascaded voice agent beside long-context coding agents on one GPU, and change an
existing engine's scheduler rather than build a new one.

**Preemption for a waiting request, with the victim's KV kept.** vLLM's V0 scheduler preempted lower-priority running
requests for the highest-priority waiting request from v0.6.2 until V0 was removed, but only with chunked prefill off
and by recomputing the victims [[8](#ref-8)]. Priority scheduling in the V1 scheduler [[27](#ref-27)] preempts among
running requests only [[7](#ref-7)]. Its open fixes preempt at the request-count cap [[18](#ref-18)], which does not
bind when KV fills, and one of them also preempts lower-priority running requests, one at a time, when a waiting
request's allocation fails, until it fits or none is left [[28](#ref-28)]. SGLang's priority preemption [[6](#ref-6)]
stops when KV memory runs out, and an open pull request adds preemption on KV exhaustion with a best-effort keep of the
victim's KV in its prefix cache [[29](#ref-29)]. Research systems preempt for a waiting or arriving request and keep the
victim's KV in host memory. TokenFlow writes KV through as it is generated [[30](#ref-30)], ConServe checkpoints each
new token's KV [[1](#ref-1)], Chiron migrates evicted batch requests' KV to CPU memory [[24](#ref-24)], and ProServe and
FastServe copy it there ahead of need [[26](#ref-26), [31](#ref-31)]. Semantic Scheduling, implemented in vLLM, preempts
running requests of lower urgency for an urgent one until it fits, discarding or offloading each victim's KV, whichever
is cheaper [[32](#ref-32)]. Llumnix keeps memory headroom for high-priority requests and migrates others away
[[33](#ref-33)], and LayerKV traces long-context TTFT to queueing for KV blocks [[34](#ref-34)]. Make-room is V0's rule
in V1's scheduler, under chunked prefill and asynchronous scheduling, on vLLM's own CPU tier. Unlike the open fix that
preempts for KV [[28](#ref-28)], it preempts only when at most N victims make room, and then as few as do.

**Keeping prefill out of a priority request's way.** Chunked prefill bounds how long a decode waits behind a prefill
[[12](#ref-12)], and disaggregated serving moves prefill to other GPUs altogether [[35](#ref-35), [36](#ref-36)], which
a single GPU cannot do. Semantic Scheduling also keeps new prefills out of the batch while its most urgent request
decodes [[32](#ref-32)], and VoxServe, serving speech language models, prioritizes newly admitted requests until their
first audio chunk [[37](#ref-37)]. SLOWeave sizes every prefill chunk to the earliest decode deadline, for all decodes
alike [[38](#ref-38)]. Within vLLM, a proposed decode-priority cadence admitted prefill only every N steps while any
decode ran, and was closed unmerged [[39](#ref-39)]. In v0.32.0, vLLM's `prefill_schedule_interval` admits new prefill
requests only every N steps, in the data-parallel MoE engine alone; a change that v0.32.0 does not include applies it to
a single engine too, and a deferred step then carries no prefill at all, chunks of a prompt already admitted included
[[40](#ref-40)]. Both hold prefill back for every decode alike. The prefill hold differs in its bound. It holds only
lower-priority prefill, and only through a reply's first tokens, a window sized to cover whole replies and with them the
first sentence the voice agent hands to speech synthesis.

**Routers and gateways.** llm-d's flow control [[19](#ref-19)], SMG [[20](#ref-20)], Dynamo [[41](#ref-41)], AIBrix
[[42](#ref-42)] and vLLM's router [[43](#ref-43)] admit, order, shed or evict requests in front of the engine. llm-d's
flow control (emulated), SMG and an abort-and-resubmit policy are baselines here. Some end running background requests,
as llm-d's in-flight eviction does, but none can preempt a running request and keep its work, and the voice turn waits
for the KV that running requests hold.

**KV across agents' pauses.** InferCept [[44](#ref-44)], KVFlow [[45](#ref-45)], TokenCake [[46](#ref-46)], Continuum
[[47](#ref-47)] and MORI [[48](#ref-48)] park, retain or prefetch agents' KV around tool calls. CachedAttention
[[49](#ref-49)] and Pensieve [[50](#ref-50)] keep a conversation's KV between its turns, and on production prefix-cache
traces LRU is hard to beat [[51](#ref-51)]. These systems decide which agent's KV stays. The change decides when a
realtime request may take KV from them, and it uses vLLM's own eviction and CPU tier. In the repository's simulator,
even knowing exactly when each agent returns from a tool call shortens the median resume by under 9% against LRU with
the CPU tier, and its p90 by up to 28% at low pressure. The session history a deployment has gains nothing. The change
therefore leaves the agents' KV to vLLM.

**Voice under contention.** LiveKit's preemptive generation, Endpoint Anticipation and PredGen start the LLM earlier in
the turn [[13](#ref-13), [52](#ref-52), [53](#ref-53)]. Unmute and NVIDIA's single-GPU voice agent document the cost of
colocating the cascade [[54](#ref-54), [55](#ref-55)], and Metronome shows full-duplex models collapsing when session KV
fills the pool [[56](#ref-56)]. We add the voice tier's tail beside a background agent class in one engine, against the
configurations an operator can deploy.

**Adaptive prefill control.** Beside SLOWeave's sizing of chunks to deadlines [[38](#ref-38)], two recent controllers
adapt the prefill that runs beside decodes from what they measure. DLFP resizes each prefill chunk from the interval
between the host's scheduler calls. In vLLM it cut a 0.6B model's p99 inter-token latency by 27.7% at a 34.8% higher p99
TTFT, and did not carry over to 8B and 32B models or to two-GPU tensor parallelism, which its authors trace to the
interval being only a proxy for when the GPU finished a step [[57](#ref-57)]. AgentServe, on one consumer GPU as here,
isolates agents' prefills from their decodes, budgets the prefills that resume a cached context, and reserves streaming
multiprocessors for decodes with CUDA Green Contexts [[58](#ref-58)]. Both treat every request of their workload alike,
with no class above the rest. We built no adaptive controller. At the two-card comparison's envelope (§6.8), what the
voice tier still loses on a shared card is the speech servers' share of the GPU, which time-slices between processes:
the transcript comes 51–71 ms later from eight agent sessions on, on stock vLLM and with the change alike, and no
scheduling inside the engine removes that. DLFP's result says what such a controller would need, the device's step
times rather than the host's rhythm.

**What is new.** The mechanisms are not: co-serving a latency class with best-effort work, preempting for a waiting
request while keeping the victim's KV in host memory, and holding prefill back for a priority request are all
published, as above. Three things are this report's. Make-room brings V0's waiting-request preemption to vLLM's V1
scheduler under chunked prefill and asynchronous scheduling, on vLLM's own CPU tier, as a change proposed to vLLM
rather than a new engine. The prefill hold is bounded to a reply's first tokens and sized to the first sentence that a
cascaded voice agent hands to speech synthesis, which we did not find elsewhere. And the measurements: a cascaded voice
pipeline beside long-context coding agents on one GPU, with V2V read from waveforms, against the configurations an
operator can deploy, and, across two cards under a resource gate fixed before the runs, what sharing a card buys over a
card per workload.

## 9. Limitations

Each result rests on assumptions about the setting, which we state here with how much the result depends on each.

**One model, two kinds of GPU.** Every result uses Qwen3-4B, on RTX 5060 Ti cards whose 36,400-token pool beside the
speech servers the agents' contexts fill, one or, in §6.8, two on two hosts, and on an A100 SXM4 80 GB held to the same
pool. The first wait depends on how often the pool is full and how long a turn then waits, and the second on how long a
step with agent prefill takes. A larger pool, another model or a faster GPU changes both, and §6.6 and §6.7 show the
change helping less where the pool has room and the step cap setting its cost on a faster card. On the A100 the rules
meet the SLO at the RTX 5060 Ti's 512-token cap, which costs the agents about 30% of their throughput there. At vLLM's
default of 2,048 tokens they keep all of it and leave the voice 0.14–0.17 s over the SLO. The repository's simulator
places other GPUs only by costs scaled from published peaks, which on the A100 missed decode attention by 1.8 times.

**A few calls per GPU.** The headline runs two callers. The hold's share of the engine's steps grows with the calls,
from 33% with two to 64% with six, and with 4 and 6 callers the change misses each SLO in every round, by 0.26–0.27 s at
the median. Replies whose windows overlap without a gap would stop the agents' prefill altogether (§6.6).

**Two cards in three rounds.** The two-card comparison rests on three rounds, each on a new sample of the agents, and
measured the consolidated card at 2, 4, 8, 12 and 24 sessions only, so the change's envelope lies between four and eight
sessions and stock vLLM's between two and four. Each card's part of an arrangement ran as a run of its own, and the sums
assume that one card's work does not slow the other's, as with the cards on two hosts that share only the control plane.
At the envelope the agents' work is the traces' own pace, so the background measure there counts the load offered, not
what the shared card could do. Their window, the calls' first 240 s a minute into the replay, holds its start, when
AIPerf resumes every session at once: over the 24-hour soak the same arrangement's agents did 9,300 tokens a second in
the first hour and 582–5,402 in each hour after.

**Synthesized callers.** The callers speak scripted lines synthesized in six voices, and the agent's endpointing is
fixed. Endpointing, which waits for the final transcript, takes 0.76 s of voice alone's V2V and 0.82 s of every
configuration's under load, within 5 ms of one another. A sweep of the fixed endpointing found it to be the VAD's
silence and about 0.2 s more: 0.3 s of silence in place of 0.55 brought voice alone's median V2V from 1.61 s to
1.35 s and answered one of the corpus's 40 lines before it had ended. Real callers vary in pace, pauses and barge-ins,
which these runs do not sample.

**A step cap chosen on one sample.** The sweep screened the change's caps of 256, 512 and 1,024 tokens on the sample
the evaluation was tuned on, one run a setting, and five rounds on new samples confirmed 256 against 512 beside the
24 sessions (§6.9). Caps between those were not measured, and the results of §6.1–§6.7 stay at 512 tokens a step.

**What the shared model can do.** The agents' replay needs no answer to be right, so the results say what sharing costs
the call, not whether the shared model's background work is worth keeping. The quality gate
([GATE.md](../benchmarks/quality/GATE.md)), fixed before it ran, tried a useful pair of jobs: answering questions about
private documents in a call (QASPER), beside enriching new documents into searchable records (Re-DocRED). The served
model failed 9 of its 11 criteria. Of the questions it answered, 50.6% were answered correctly, against a bar of 80%,
and of the facts in its records 23.9% were right, against 80%. Qwen3-8B-FP8 on a card of its own failed as well
([the gate's record](../benchmarks/results/2026-10-27-quality/README.md)). A deployment that shares a card for
background work has to check first that the model does that work.

**The pool stays as it is.** Two ways to a larger KV pool on this card did not hold. An FP8 KV cache doubles the pool's
tokens and cost no quality the gate's screen saw. Beside the CPU tier, though, the engine died both times it ran under
the main scenario's load, on a CUDA fault at a step's output copy
([the sweep's record](../benchmarks/results/2026-10-27-sweep/README.md)). On the CPU the speech servers would give their
memory to the pool, but Kokoro's first audio took 991 ms with one call against 211–242 ms for the whole TTS stage on the
GPU ([the CPU record](../benchmarks/results/2026-10-27-cpu-audio/README.md)).

**Scaled traces.** The agents replay Claude Code traces scaled 32× in blocks of tokens, so that every request fits the
32k context. The scaling keeps which requests share which prefixes and changes the traces' timing. It also sets the KV
pressure that the first wait depends on, and the scenario where the pool has room (§6.6) shows how much the result
depends on that pressure.

**What a resume recomputes.** A preempted request's KV is in the CPU tier when it is preempted, and on resume vLLM
reloads what the tier still holds. vLLM 0.32's counters bound what resumes reload, at most the 117k tokens loaded beyond
first scheduling, about 2.7k a counted preemption, but not what they recompute. This bears on the agents' cost (§6.4),
not on the voice tier's latency.

**The generated scheduler, not the branches.** The evaluation ran the change on the generated scheduler on vLLM 0.32.0
with the pull requests' switch values. The trace test holds it and the pull requests' diff, ported to 0.32.0, to the
same schedules, but the branches on vLLM's main ran only in their CPU tests.

**One vLLM version.** Every measurement ran on vLLM 0.32.0, the latest release when the results were taken. On
2026-10-27 both pull requests still merged cleanly into vLLM's main in a local merge and passed their tests there on the
CPU, and make-room's patches did not apply to the next release candidate, v0.33.0rc0, whose scheduler no longer matches
their first hunk, so an upgrade starts by rebasing them
([the check's record](../benchmarks/results/2026-10-27-compat/README.md)).

**One cap on victims.** Every run with make-room used N = 2, so how the results depend on N is not measured.

**What five runs resolve.** In the main scenario each comparison rests on five runs a configuration, 10 for tuned vLLM
and 15 for the change. The caller counts rest on three, the A100's 2,048-token rows on two and its 32-session round on
one, and §6.5's single-sample campaign on one to five. Differences of a few hundred milliseconds between configurations
near the SLO lie within the spread of their runs, which is why we compare such configurations round by round on the same
sample.

## 10. Conclusion

A voice agent that shares its GPU with coding agents loses its seconds inside the engine's scheduler, in two places. A
waiting turn cannot take KV from running background requests, and a running reply shares its steps with background
prefill. A control plane in front of the engine sees neither, so it can make room only by aborting the agents' work. Two
rules inside vLLM's priority scheduler remove both waits by counting blocks and tokens alone, a preemption for the
waiting turn and a hold on lower-priority prefill through the reply's first sentence. With them the voice tail on a
consumer GPU comes to 2.24 s, 0.20 s over a strict SLO, with a third of the replies still over it, while the agents keep
97% of their token rate. On an A100 at the same pool and step cap it meets that card's SLO in each of five rounds, by
7–60 ms. The gain shrinks where the pool has room, the tail grows as calls overlap, and the step cap sets how much of
the agents' throughput a tail under the SLO costs. For a realtime tier that shares a GPU with agents whose KV fills the
pool, the place to act is the engine's scheduler: preempt for the waiting turn, and keep background prefill out of its
first sentence. With two cards, the larger gain is the arrangement: both calls and a few agents on one card, and the
other card full of agents. What then limits the shared card is the speech servers' share of the GPU, which no
scheduling inside the engine removes.

## References

<a name="ref-1"></a>[1] Y. Qiao et al. [ConServe: Fine-Grained GPU Harvesting for LLM Online and Offline Co-Serving](https://arxiv.org/abs/2410.01228). In *ICML 2026*.

<a name="ref-2"></a>[2] T. Sun, P. Wang, and F. Lai. [HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](https://arxiv.org/abs/2501.14808). In *NeurIPS 2025*.

<a name="ref-3"></a>[3] F. Liu et al. [Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate](https://arxiv.org/abs/2604.07874). arXiv:2604.07874, 2026.

<a name="ref-4"></a>[4] W. Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). In *SOSP 2023*.

<a name="ref-5"></a>[5] L. Zheng et al. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). In *NeurIPS 2024*.

<a name="ref-6"></a>[6] SGLang v0.5.22: [priority scheduling with preemption](https://github.com/sgl-project/sglang/pull/8746) (pull requests #8746 and #17026, merged 2025-09-17 and 2026-03-04) and [HiCache](https://github.com/sgl-project/sglang/blob/v0.5.22/docs/docs/advanced_features/hicache.mdx), its hierarchical KV cache in host memory.

<a name="ref-7"></a>[7] vLLM issue #40004. [Priority scheduling supports preemption of requests in the running queue by requests in the waiting queue](https://github.com/vllm-project/vllm/issues/40004). Opened 2026-04-16, open on 2026-10-27.

<a name="ref-8"></a>[8] vLLM pull request #5958. [Adding priority scheduling](https://github.com/vllm-project/vllm/pull/5958). Merged 2024-09-25, in vLLM v0.6.2 to v0.10.2.

<a name="ref-9"></a>[9] C. Wang. vLLM pull request #63350. [Priority scheduling: preempt for a waiting request under KV pressure](https://github.com/vllm-project/vllm/pull/63350). Opened 2026-10-23, open and under review on 2026-10-27.

<a name="ref-10"></a>[10] C. Wang. vLLM pull request #63351. [Priority scheduling: hold lower-priority prefill for first tokens](https://github.com/vllm-project/vllm/pull/63351). Opened 2026-10-23, open and under review on 2026-10-27.

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

<a name="ref-59"></a>[59] Z. Ye et al. [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005). In *MLSys 2025*.

<a name="ref-60"></a>[60] T. Dao. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). In *ICLR 2024*.

## Appendices

### A. Reproducing the results

**Scope.** Every number, table and figure in this report is computed from records committed to the repository, or from
its tests where Table A1 says so. Table A1 maps each part of the report to its records and to what regenerates it.

**Table A1. Where each result comes from.**

| What | Records | Regenerated by |
|---|---|---|
| §6.1–§6.4 and §6.6, Figures 1, 8, 9, 10 and 12, Tables 2–4 | [`2026-10-24-e2-w5`](../benchmarks/results/2026-10-24-e2-w5) | `make reproduce-e2-w5`, `make figures` |
| The final code's five rounds (§6.1) | [`2026-10-25-e2-fb`](../benchmarks/results/2026-10-25-e2-fb) | its `compare.py` |
| §6.5, Figure 11 and Table B3 | [`2026-10-23-e2`](../benchmarks/results/2026-10-23-e2) | `make reproduce-e2`, `make figures` |
| §6.7, Figure 13 and Table 5 | [`2026-10-26-a100`](../benchmarks/results/2026-10-26-a100) | `make reproduce-e2-a100`, `make figures` |
| The A100 at a batch of 256 (§6.7) | [`2026-10-26-a100/batch`](../benchmarks/results/2026-10-26-a100/batch) | `make reproduce-a100` |
| The share of time short of KV (§6.7) | [`2026-10-26-e2-rs`](../benchmarks/results/2026-10-26-e2-rs) | its `running_share.py` |
| The KV pool over time (Figure 2) | [`2026-10-23-make-room-probes`](../benchmarks/results/2026-10-23-make-room-probes) | `make figures` |
| The KV and prefill cases and the calls beside closed-loop agents (Figures 3 and 4) | [`2026-10-24-w5-probes`](../benchmarks/results/2026-10-24-w5-probes) | its `check.py`, `make figures` |
| The window sweep (Figure 6b, 6c) | [`2026-10-25-window-knee`](../benchmarks/results/2026-10-25-window-knee) | its `table.py`, `make figures` |
| The replies' lengths (Figure 6a) | [`replies.tsv`](../benchmarks/results/2026-10-24-e2-w5/replies.tsv) | `replies.py`, from the raw records |
| A step's cost (Figure 7) | [`make-room-schedule.json`](../benchmarks/perf/make-room-schedule.json) | `make perf-make-room`, `make figures` |
| A prompt's cost recomputed, restored or hit (§2.2) | [The tier cost table](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0), kept as the cost model's constants in [`cost.rs`](../crates/flatshare-core/src/cost.rs) | `flatshare-bench tiers` on the GPU node |
| The gateway's overhead (§4) | [`2026-10-25-w6-transport`](../benchmarks/results/2026-10-25-w6-transport) | its `llm_paths.py` |
| The earlier campaign's reply lengths (§3.3) | [The earlier campaign's section of BENCHMARKS.md](BENCHMARKS.md#the-discriminating-scenario-in-rounds-with-arm-9-and-the-change-w4), from the gateway's log | The v0.3 release's raw records |
| The simulator's result on agents' KV (§8) | [The simulator's tables in BENCHMARKS.md](BENCHMARKS.md#e1-parking-in-the-simulator) | `make reproduce-e1` |
| Replies whose windows overlap (§6.6) | The test of overlapping replies in [`test_make_room.py`](../python/flatshare_vllm/tests/test_make_room.py) | `make test-make-room` |
| The failure drills (Appendix D) | [`2026-10-25-drills`](../benchmarks/results/2026-10-25-drills) | its README |
| The first sentences' tokens (§6.3) | [`2026-10-27-first-sentence`](../benchmarks/results/2026-10-27-first-sentence) | [`first_sentence.py`](../benchmarks/replies/first_sentence.py) |
| §6.8, Figure 14 and Tables 6 and 7 | [`2026-10-28-rounds`](../benchmarks/results/2026-10-28-rounds) | its `gate.py`, `make figures` |
| §6.9 and Table 8 | [`2026-10-28-rounds`](../benchmarks/results/2026-10-28-rounds), and the screens in [`2026-10-27-sweep`](../benchmarks/results/2026-10-27-sweep) | `gate.py`'s sweep, the screens' `table.py` |
| The endpointing sweep (§9) | [`2026-10-28-endpointing`](../benchmarks/results/2026-10-28-endpointing) | [`summary.py`](../benchmarks/endpointing/summary.py) |
| The drills on two GPU nodes (Appendix D) | [`2026-10-28-node-failover`](../benchmarks/results/2026-10-28-node-failover) | its README |
| The 24-hour soak (§7, Appendix D) | [`2026-10-28-soak`](../benchmarks/results/2026-10-28-soak) | its `table.py`, `flatshare-bench soak report` |
| The quality gate (§9) | [`2026-10-27-quality`](../benchmarks/results/2026-10-27-quality) | [`gate.py`](../benchmarks/quality/gate.py) |
| The pool's two ways out (§9) | [`2026-10-27-sweep`](../benchmarks/results/2026-10-27-sweep), [`2026-10-27-quality/fp8-kv`](../benchmarks/results/2026-10-27-quality/fp8-kv) and [`2026-10-27-cpu-audio`](../benchmarks/results/2026-10-27-cpu-audio) | their READMEs |
| The next vLLM release (§9) | [`2026-10-27-compat`](../benchmarks/results/2026-10-27-compat) | `make compat-upstream` |

The repository's tests fail when a committed figure differs from what its records give, so the figures here are the
records' figures. [BENCHMARKS.md](BENCHMARKS.md) lists every measurement run by run, and
[EXPERIMENTS.md](EXPERIMENTS.md) gives each experiment's design.

**Hosting.** Tag v1.7 holds the code, the records and this report. The raw records, the gateway's request logs and each
run's directory, are release assets: the main evaluation's in
[v1.0](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.0), the A100 run's in
[v1.1](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.1), the KV-sampling runs' in
[v1.3](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.3), and those of the campaign before the main
evaluation in [v0.3](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3).

**Requirements.** Regenerating the tables and figures from the records needs Python and the repository alone. Measuring
again needs a GPU node configured as in Appendix C, the models and the traces. The media server, the gateway and the
load generators ran on a second host beside the RTX 5060 Ti, and on the GPU node itself on the A100.
[HARDWARE.md](HARDWARE.md) deploys the node on another GPU from a profile, and on the A100 one round of the main
comparison took 26 minutes.

**Expected variation.** A p95 over 59 replies lies between the third- and fourth-slowest reply. In the main evaluation
the change's p95 ranged over 2.13–2.40 s in 15 runs, and tuned vLLM's over 8.7–12.1 s in ten. Five more rounds on the
same samples moved tuned vLLM's p95 by −2,396 to +1,706 ms round by round, and the change's, on its final code, by −45
to +77 ms. A repeat should therefore land near the ranges of Tables 2 and B1 rather than on their medians, and one round
of the repeat put tuned vLLM at 8.37 s, below its range in Table 2.

**Differences from the code proposed to vLLM.** The measurements ran the generated scheduler on vLLM 0.32.0 (§4). It
applies the pull requests' scheduler diff, ported to 0.32.0, and adds the reserve, which the measured configurations
turn off, and its own log of the rules' counts. Its schedules match the pull requests' as they stood before their
revision of 25 October 2026, which ends the hold one step sooner for a reply at least T tokens long. The final code's
five rounds in §6.1 measure that difference.

### B. Results by round

Each round replays its own sample of the agents' traces, the same sample in every configuration and set of rounds, so a
column compares configurations on the same agents. Table B1 gives every run behind Table 2.

**Table B1. The p95 V2V of every run behind Table 2, and of the change on the final code, by round (ms).** Three sets of
rounds replayed the same five samples. The first ran voice alone, tuned vLLM and the change, the second tuned vLLM,
make-room alone, both rules at T = 16 and the change, and the third the router and the change, beside the scenario where
the pool has room. The last row is the change on the final code (§6.1).

| Configuration | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Median |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 1,810 | 1,774 | 1,819 | 1,764 | 1,784 | 1,784 |
| Tuned vLLM, first set | 10,576 | 10,770 | 8,686 | 9,782 | 10,432 | 10,432 |
| Tuned vLLM, second set | 9,946 | 10,133 | 10,810 | 12,115 | 10,523 | 10,523 |
| The abort-and-resubmit router, third set | 2,516 | 2,860 | 2,522 | 2,522 | 2,868 | 2,522 |
| Make-room, second set | 6,261 | 5,376 | 4,693 | 4,970 | 5,085 | 5,085 |
| Make-room and the hold, T = 16, second set | 3,022 | 3,100 | 2,868 | 3,192 | 3,094 | 3,094 |
| The change, first set | 2,236 | 2,175 | 2,298 | 2,250 | 2,307 | 2,250 |
| The change, second set | 2,174 | 2,301 | 2,189 | 2,232 | 2,397 | 2,232 |
| The change, third set | 2,293 | 2,257 | 2,132 | 2,162 | 2,145 | 2,162 |
| The change, final code | 2,313 | 2,246 | 2,253 | 2,268 | 2,343 | 2,268 |

**Table B2. Where the pool has room, and with more callers (p95 V2V, ms).** Sixteen sessions of the whole corpus beside
two callers, and the 24 sessions beside four and six callers, against each caller count's own SLO.

| Configuration | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Median | SLO |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 sessions, tuned vLLM | 2,147 | 3,832 | 3,067 | 2,300 | 2,516 | 2,516 | 2,034 |
| 16 sessions, the change | 2,038 | 2,419 | 2,122 | 2,050 | 2,021 | 2,050 | 2,034 |
| 4 callers, voice alone | 1,894 | 1,794 | 1,780 | | | 1,794 | |
| 4 callers, the change | 2,313 | 2,415 | 2,250 | | | 2,313 | 2,044 |
| 6 callers, voice alone | 1,885 | 1,885 | 1,878 | | | 1,885 | |
| 6 callers, the change | 2,392 | 2,360 | 2,468 | | | 2,392 | 2,135 |

**Table B3. Every configuration measured beside the 24 sessions on a single sample of the agents' traces.** Medians over
each configuration's runs, with the range of the p95 V2V. Voice alone's p95 was 1,767 ms over five runs, so the SLO is
2,017 ms. The campaign before the main evaluation ran these, and counted every turn, the first of each call included.
The table lists all 30 of its configurations beside the 24 sessions. Its rows with the engine change ran the earlier
version of the generated scheduler (Appendix C), and the last three ran after its rounds, in a session of their own.
Figure 11 plots every row but the policy's twelve ablations.

| Configuration | Runs | p95 V2V (ms) | Agents' tokens/s |
|---|---:|---:|---:|
| Tuned vLLM | 5 | 10,138 (9,257–10,811) | 3,552 |
| A 1,024-token step cap without a per-request threshold | 5 | 8,789 (8,161–9,490) | 3,888 |
| A KV watermark of 10% | 1 | 10,647 | 3,649 |
| A limit of 8,192 queued tokens | 1 | ≥ 30,000 (48 of 50 turns unanswered) | 5,584 |
| Agent requests in flight capped at 2 | 1 | 8,573 | 3,446 |
| Agent requests in flight capped at 8 | 1 | 9,876 | 3,586 |
| A static 2,048-token KV reservation for voice | 5 | 5,671 (4,906–5,798) | 3,551 |
| `--max-num-seqs 12` with #54530 | 1 | 10,930 | 3,624 |
| SGLang 0.5.22, priority preemption | 1 | 12,051 | 3,577 |
| SGLang 0.5.22 with its host tier | 5 | 11,948 (9,394–16,702) | 4,330 |
| llm-d's flow control, emulated | 1 | 10,992 | 3,684 |
| SMG 1.13.0 | 1 | 11,050 | 3,446 |
| The abort-and-resubmit router | 5 | 2,591 (2,110–2,663) | 3,593 |
| The gateway policy timed by the conversation, on tuned vLLM | 3 | 10,993 (10,028–11,958) | 3,633 |
| The same policy over the earlier scheduler, with the hold at T = 16, the reserve on and replies capped at 16 tokens | 5 | 2,057 (1,980–2,134) | 3,355 |
| The same, without its headroom | 3 | 2,033 (1,947–2,071) | 3,339 |
| The same, without its prefill throttle | 3 | 2,049 (1,995–2,081) | 3,357 |
| The same, without its lifecycle events | 3 | 2,036 (2,005–2,048) | 3,326 |
| The same, without its reply cap | 3 | 3,837 (2,890–4,079) | 3,336 |
| The same, without its reply cap, with a 48-token window | 3 | 2,205 (2,098–2,289) | 3,343 |
| The same engine without the policy | 3 | 4,034 (2,984–4,062) | 3,406 |
| The same, without the hold | 3 | 5,295 (5,063–6,115) | 3,465 |
| The same, without make-room | 3 | 2,037 (1,990–2,039) | 3,326 |
| The same, without the reserve | 3 | 2,116 (2,115–2,155) | 3,438 |
| The same, with a prefill throttle budget of 0 ms | 3 | 2,109 (2,094–2,166) | 3,406 |
| The same, with a prefill throttle budget of 500 ms | 3 | 2,060 (1,990–2,080) | 3,355 |
| The same, with its headroom armed by the voice request | 3 | 2,124 (2,008–2,165) | 3,339 |
| Make-room | 3 | 5,773 (5,612–6,063) | 3,572 |
| Make-room and the hold, T = 16 | 3 | 3,229 (3,220–4,772) | 3,540 |
| The change | 3 | 2,193 (2,168–2,286) | 3,362 |

**Table B4. The A100 by round (p95 V2V in ms, and the agents' tokens a second over the calls' first 240 s).** The SLO is
1,669 ms.

| Configuration | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Median |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 1,445 | 1,414 | 1,419 | 1,391 | 1,424 | 1,419 |
| Tuned vLLM, 512-token steps | 4,163 | 4,783 | 4,986 | 5,169 | 5,710 | 4,986 |
| The change, 512-token steps | 1,663 | 1,658 | 1,611 | 1,634 | 1,609 | 1,634 |
| Tuned vLLM, 2,048-token steps | 4,036 | 3,289 | | | | 3,663 |
| The change, 2,048-token steps | 1,810 | 1,843 | | | | 1,826 |
| Agents' tokens/s, tuned vLLM, 512 | 10,749 | 9,820 | 8,398 | 8,633 | 8,312 | 8,633 |
| Agents' tokens/s, the change, 512 | 10,125 | 10,851 | 8,025 | 8,651 | 7,922 | 8,651 |
| Agents' tokens/s, tuned vLLM, 2,048 | 15,082 | 13,990 | | | | 14,536 |
| Agents' tokens/s, the change, 2,048 | 15,124 | 14,145 | | | | 14,635 |

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

**The change** runs on the generated scheduler, which the records call make-room, with
`make_room.max_preemptions_per_step 2`, `make_room.budget_cap true`, `make_room.budget_cap_tokens 48` and
`make_room.reserve_tokens 0` in vLLM's `--additional-config`, and the engine loads it with `--scheduler-cls`. On vLLM's
main the same rules are `--max-priority-preemptions-per-step
2 --priority-prefill-budget-cap --priority-prefill-budget-cap-tokens 48`. Make-room alone turns the hold off (`make_room.budget_cap false`), and T = 16
leaves `budget_cap_tokens` at its default of 16. [PRODUCTION.md](PRODUCTION.md#the-configuration) gives the whole
deployment.

**The earlier version.** Table B3's rows with the engine change ran an earlier version of the generated scheduler. Among
victims of the lowest priority it took those holding the most KV first, and it had three gaps that the main evaluation's
version fixes, as the pull requests do. The hold did not hold a lower-priority request that already held its KV, such as
one whose load from the CPU tier had finished. The KV shortfall counted the blocks of a request's cached prefix as
needed, so the reserve and the guard on in-flight prefills could refuse a request that vLLM admits. And victims were
sized before the next step's decode growth, so a turn could need a second preemption a step later.

**The records' directories.** Their names carry the campaigns' codes. `2026-10-24-e2-w5` holds the main evaluation,
`2026-10-25-e2-fb` its repeat on the final code, `2026-10-23-e2` the campaign before it, `2026-10-26-e2-rs` the
KV-sampling runs, and `2026-10-24-w5-probes` and `2026-10-25-w6-transport` the probes and the transport's check.

**The hosts in the records.** The records name the RTX 5060 Ti node Phastform and an identical second node Gray, on
which the earliest KV and prefill cases ran. The control host is cyan, and the CPU cost of a step (Figure 7) was
measured on peacock.

### D. Failure drills

The deployed configuration was broken on purpose while two voice streams and four agent streams were decoding, and each
probe after the event used a new connection. Table D1 gives what the streams and the new requests saw.

**Table D1. What a failure does to the streams in flight and to new requests.**

| Drill | The six streams in flight | New requests |
|---|---|---|
| The engine killed mid-decode | Each received an error event 8–11 ms after the kill | Refused at once until the engine's warm start; a request succeeded 106.2 s after the kill |
| The gateway reloaded, as a deploy does | The voice streams finished whole, 20.9 s later; the agents' streams ended after 5.0 s with a draining event | None of 42 failed or waited; the slowest was answered in 0.13 s |
| The gateway restarted | As in the reload | None of 42 failed; those sent during the 20.7 s drain were answered once the new gateway started |
| The gateway killed | All broke within 11–14 ms | None of 4 failed; the slowest waited 1.3 s for the gateway's restart |

The engine's recovery is mostly its warm start. Its last process exited 7.6 s after the kill, systemd started it again
5.1 s later, and it listened 92.1 s after that. On empty caches, as on a new host, the engine's first start takes 531.8
s, and all three GPU servers restart together in 102.7 s on warm caches. A CPU tier filled to twice its size kept the
engine healthy, and it restored the latest 8,192-token prompt in 294 ms against 1,112 ms to recompute it.

On two GPU nodes, with both calls on the first and the second a warm spare (§6.8, §7), the drill places two calls as the
voice worker does, runs a turn of each every 3 s beside two agents' streams, and then breaks the first node's engine. A
SIGKILL moved both calls' next turns to the spare 0.7 s later, and stopping the engine 1.9 s later. None of the 300
turns after the events failed, and none was in flight at either event. The spare held its agents' new requests while it
carried the calls, and each got 503 `queue_timeout` 120 s later; the killed engine was back 101 s after the kill.

A 24-hour soak of that arrangement, calls arriving at 20 an hour beside 14 agents' session trees, put 74 of its 5,058
replies over the SLO (1.5%; 5.5% in the worst hour). Of its 476 calls, 72 found both lines busy and 2 were turned away
with a line free, LiveKit still holding the worker full from its last status
([PRODUCTION.md](PRODUCTION.md#calls-what-the-box-admits)); none failed, no unit restarted, and the spare never carried
a call; on the calls' card the budget cap held prefill back in 4,853 steps and make-room never had to preempt
([the record](../benchmarks/results/2026-10-28-soak/README.md)).

### E. The forecast for the A100

A forecast for the A100 SXM4 80 GB was written before the run, from the RTX 5060 Ti's measured costs scaled by the
card's published peaks ([forecast-80gb.md](../benchmarks/a100/forecast-80gb.md)). It scaled weight and KV reads by the
memory bandwidth, each layer by the throughput of the kernels vLLM 0.32.0 runs for it, and restores by the host link. It
placed each mechanism, missed the length of the KV case's wait by about 30%, and also fell short on the change's first
token, 175 ms against 126–145, and its token gap, 7 ms against 5.6 (Table E1).

**Table E1. The probes on the A100, against the forecast and the RTX 5060 Ti.** The A100's medians and ranges over three
runs, and the RTX 5060 Ti's over the four runs of §2.4, whose hold at T = 16 acts as T = 48 does on a reply's first 16
tokens. The forecast gives its central range and, in parentheses, its low–high range.

| Probe | A100, measured | Forecast | RTX 5060 Ti |
|---|---|---|---|
| KV case on tuned vLLM, the voice turn's wait | 1,748 ms (1,721–1,750) | 1,328–1,369 ms (1,082–1,377) | 4,968 ms (4,900–4,971) |
| KV case with the change, the first token | 175 ms (150–176) | 126–145 ms (109–146) | 290 ms (216–294) |
| Prefill case on tuned vLLM, the reply's token gap | 32 ms | 32.8 ms (25.7–35.4) | 60–66 ms |
| Prefill case with the change, the token gap | 7 ms | 5.6 ms (3.7–5.6) | 16–20 ms |

A step model fitted on the A100's own calibration run explains the miss. Its decode attention cost 171.9 ns per token of
KV read, 1.82 times the forecast, which is 858 GB/s or 42% of the card's bandwidth, where the RTX 5060 Ti's reaches 77%
of its own. The per-token cost of the linear layers came in at the forecast's 41.4 µs. A step on the A100 is cheaper
than on the RTX 5060 Ti by less than the cards' peaks say, and the KV case's wait is a count of decode steps.
FlashInfer's attention [[59](#ref-59)] in place of FlashAttention 2 [[60](#ref-60)] shortened the wait by 8%, to
1,580–1,607 ms.

The forecast also overstated how often two calls run short of KV. It placed the A100 at 10% of the time and the RTX 5060
Ti at 17%, against the 5.7% and 8.1% measured beside tuned vLLM.

### F. Revision history

The report is versioned with the repository's releases. A correction that changes a number, a figure or a claim takes a
new version, and [CHANGELOG.md](../CHANGELOG.md) lists what each release changed.

**Table F1. Versions of this report.**

| Version | Date | What changed |
|---|---|---|
| 1.7 | 2026-10-29 | Two cards measured against a resource gate fixed before the runs, the serving profile's 256-token step cap, operating two cards and their 24-hour soak, the endpointing sweep, adaptive prefill control and what is new beside the related work, Table A1's records for them and for the first sentences, §3.3's sentence on those, and the version and tag the report names, which 1.6 left at 1.5. Corrected: §8 on vLLM's open fixes (#40087 preempts for KV too), on `prefill_schedule_interval` and on SGLang's #45686, and §2.3 on what the CPU tier keeps by default; added to §9: the quality gate, the two ways to a larger pool and the next vLLM release, which 1.6 measured and left out |
| 1.6 | 2026-10-28 | The share of replies over the SLO and the pooled p95 beside each configuration's median p95, what the agents' token rate measures, the reuse of the five samples of the agents, and how long the first sentences are |
| 1.5 | 2026-10-27 | Rewritten in full: the design and two algorithms, thirteen figures and five tables drawn from the records, references, and appendices |
| 1.4 | 2026-10-27 | Unchanged: the release rewrote the repository's history |
| 1.3 | 2026-10-26 | The RTX 5060 Ti's measured share of time short of KV, beside the A100's |
| 1.2 | 2026-10-26 | The project renamed Flatshare |
| 1.1 | 2026-10-26 | The A100 run |
| 1.0 | 2026-10-26 | First version |
