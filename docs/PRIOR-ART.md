# Prior art

This page is for an engineer asking how Flatshare differs from an engine, router or paper that co-serves latency
classes. It states what Flatshare claims, where each mechanism already exists, and what Flatshare does not claim. Each
source was checked against its primary source, a paper, thread, release note, documentation or code, between 2026-10-22
and 2026-11-09, and vLLM's threads and Flatshare's pull requests on 2026-11-09.

## What Flatshare claims

Flatshare serves a realtime tier, voice turns, beside a best-effort tier, background coding agents, on one vLLM replica.
It applies published co-serving and KV-tiering mechanisms to a workload pair that no published system we found measures
together. Three things are its own, as [the report's §8](REPORT.md#8-related-work) also argues.

1. **A change to vLLM's V1 scheduler, measured end to end.** It is proposed to vLLM as pull requests
   [#65282](https://github.com/vllm-project/vllm/pull/65282) and [#65283](https://github.com/vllm-project/vllm/pull/65283),
   first opened as #63350 and #63351. Both are open and rebased, with no maintainer review as of 2026-11-09.
   - **The KV preemption** (#65282) brings V0's preemption for a waiting request to V1, extended to chunked prefill and
     to the CPU tier. The tier keeps the victim's KV when the offloading connector offloads generated tokens too
     (`offload_prompt_only: false`, which Flatshare sets). It preempts only when at most N victims make room, and then
     as few as do. The open fixes #40087 and #45558 preempt one victim at a time until the request fits, so they can
     preempt without making room.
   - **The budget cap** (#65283) holds lower-priority prefill out of a priority request's first tokens, a window sized
     to cover the first sentence the voice agent hands to text to speech. Semantic Scheduling holds prefill through an
     urgent request's whole decode, and VoxServe holds steady streams until a new request's first audio chunk. We found
     no hold bounded to a reply's first tokens.
   - **The measurement.** In E2's discriminating scenario the change takes the voice tail from 10.5 s to 2.24 s, 0.20 s
     over the SLO. On an A100 SXM4 80 GB at the same pool it takes it from 4.99 s to 1.63 s, under that card's SLO
     ([§6.1](REPORT.md#61-the-two-rules-bring-the-voice-tail-to-224-s-while-the-agents-keep-97-of-their-throughput),
     [§6.7](REPORT.md#67-the-rules-carry-to-an-a100)).
2. **Headroom timed by voice events, a measured negative result.** KV room for the next voice turn is held from speech
   onset or end of turn, before the request exists, by deferring new background requests. We found no system that times
   headroom from a voice event. In E2's discriminating scenario it could not act, since running agents held the KV.
   Arm 9 with its headroom left out, or armed only by the request, stayed within the spread of its own runs. A control
   plane in front of the engine makes room only by aborting admitted work
   ([§6.5](REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work)).
3. **The workload pair, measured.** A cascaded voice pipeline beside long-context agents on one GPU, with voice-to-voice
   latency read from waveforms, against the configurations an operator can deploy. The co-serving systems below serve
   online and offline LLM requests, and the voice systems measure voice alone.

E1, the simulator's study, measures what lifecycle information adds to the parking mechanisms of Tokencake, KVFlow,
MORI and Ask the Tool. Knowing when agents return bought under 8% of the median resume latency, and session history
nothing ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)).

Flatshare also proposed version 1.24's cache-aware queue and SM copy to vLLM, as drafts
[#66558](https://github.com/vllm-project/vllm/pull/66558) and [#66527](https://github.com/vllm-project/vllm/pull/66527).
E1's negative result went to vLLM #57103 and [vllm-project/router#428](https://github.com/vllm-project/router/issues/428)
as comments, and a lifecycle-event sink to [agentic-api#364](https://github.com/vllm-project/agentic-api/issues/364).

## At a glance

| Mechanism | Where it exists | Flatshare |
|---|---|---|
| A latency class co-served with best-effort work | ConServe, HyGen, BROS, Niyama, Chiron, Echo, ProServe; Valve in production on 8,054 GPUs | Not claimed |
| Priority classes, budgets, aging, queue TTLs, in-flight eviction | llm-d v1.0's flow control, SMG, Dynamo, AIBrix | Not claimed; llm-d's flow control (emulated) and SMG are E2 baselines |
| Headroom held back for priority work | Static: SMG's reserved slots, llm-d's holdback, vLLM's class-blind `watermark`. For a running request: Llumnix. From load or the agent graph: Valve, Echo, Tokencake | Not claimed; an equal-size static reservation is a baseline |
| Headroom timed from a voice event, before the request exists | Not found | A measured negative result; in E2 it could not act |
| Engine priority preemption | vLLM V1 among running requests; SGLang at the request cap | Baseline; the KV-full case in V1's scheduler is the claimed fix |
| Preempting for a waiting request, the victim's KV kept in host memory | vLLM V0 (its victims recomputed), TokenFlow, ConServe, Chiron, ProServe, FastServe | Not claimed; #65282 is its form in vLLM V1's scheduler, on vLLM's own CPU tier |
| Holding other work back for a priority request's output | Semantic Scheduling, VoxServe; class-blind: SLOWeave, vLLM #56848 (closed) | Claimed in one form, the budget cap (#65283), bounded to a reply's first tokens |
| Parking KV during a tool call, prefetching before the return | Tokencake, KVFlow, CacheScout, MORI; proposed for vLLM (#57103, #58198) | Not claimed; E1 measures what lifecycle information adds |
| TTL holds and leases | Continuum, TensorRT-LLM, vLLM #37003, llm-d-router #2273, AIBrix #2814 | Not claimed; Continuum's TTL hold is an E1 baseline |
| Program- or session-level admission | Autellix, Parrot, ThunderAgent, vllm-project/router's Progress-TTL | Not claimed |
| Starting the LLM before the turn ends | LiveKit's preemptive generation (on by default), Endpoint Anticipation, PredGen | Not claimed; triggers to compare |
| Ordering waiting requests by what is cached | SGLang, by the longest cached prefix; vLLM's main since 2026-11-04 ([#54625](https://github.com/vllm-project/vllm/pull/54625)), in a bounded window, by the GPU prefix cache's hits alone | Not claimed; version 1.24's queue also counts the CPU tier's hits, within each priority (#66558) |
| The copy path of KV loads from host memory | vLLM: a Triton kernel for pages under 28 KiB, the copy engine above, with a per-device cutoff asked in [#57555](https://github.com/vllm-project/vllm/issues/57555) and proposed in [#66511](https://github.com/vllm-project/vllm/pull/66511) | Not claimed; on PCIe 3.0 x8 a load on the copy engine holds a concurrent step's inputs until it ends (measured), and #66527 loads the tier through the kernel |
| A priority tier that degrades instead of failing | Gemini's Priority overflow, OpenAI's Fast mode, Niyama's relegation | Not claimed |
| A gap between LRU and Belady on prefix caches | "When Fancy Eviction Fails", LPC | Not claimed; E1 reports the gap closed per level of information |
| Cascaded voice beside long-context agents on one GPU, measured | Not found; NVIDIA's single-GPU voice-agent profile documents the regression | The measured workload |

The sections below give each source, and their Relation column says how it bears on Flatshare. A **baseline** is
compared against in E1 or E2. **Prior** marks a mechanism that the source published or shipped before Flatshare, which
does not claim it. A **complement** is an input, a venue for upstream work, or evidence for the problem.

## Co-serving latency-critical and best-effort work

| Work | Venue, date | What it does | Relation |
|---|---|---|---|
| [ConServe](https://arxiv.org/abs/2410.01228) | ICML 2026 | Co-serves online and offline tasks on vLLM, preempting offline requests for online ones, their KV checkpointed to host memory token by token | Prior: harvest-and-protect with resume from host memory; E2's preempt-resubmit approximates it from outside the engine |
| [HyGen](https://arxiv.org/abs/2501.14808) | NeurIPS 2025 | A latency predictor co-locates online and offline requests within latency SLOs | Prior: the co-location goal |
| [BROS](https://arxiv.org/abs/2504.09590) | arXiv 2025-04-13 | Real-time and best-effort requests in one engine, with dynamic priorities and KV shared between the classes | Prior: two classes sharing KV in one engine |
| [Niyama](https://arxiv.org/abs/2503.22562) | arXiv 2025-03-28 | QoS classes on shared infrastructure that relegate work under overload | Prior: class-aware admission with graceful degradation |
| [Valve](https://arxiv.org/abs/2604.07874) | arXiv 2026-04-09 | Online–offline colocation on 8,054 GPUs, with sub-millisecond compute preemption and dynamic memory reservation | Prior: co-location at fleet scale; the closest production analogue of E2 |
| [Chiron](https://arxiv.org/abs/2501.08090) | arXiv 2025-01-14 | Autoscales instances by SLO; interactive requests evict batch ones, whose KV moves to CPU memory for a fast restart | Prior: eviction with the victim's KV kept in host memory |
| [Echo](https://arxiv.org/abs/2504.03651) | arXiv 2025-03-01 | On vLLM V0's priority scheduling, online tasks preempt offline ones, which recompute, beside a KV reserve sized from load | Prior: preemption on V0's scheduler, and a reserve sized from load |
| [ProServe](https://arxiv.org/abs/2512.12928) | arXiv 2025-12-15 | Multi-priority scheduling that preempts even in decode, the victims' KV copied to host memory ahead of need | Prior: priority preemption with the victim's KV kept |
| [TokenFlow](https://arxiv.org/abs/2510.02758) | EuroSys 2026 | Preempts running requests for waiting ones on SGLang, ranked by clients' token buffers, with KV written through to host memory | Prior: preemption for a waiting request with KV kept; its priorities come from buffers, not classes |
| [FastServe](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang) | NSDI 2026 | Iteration-level preemption with a skip-join multi-level feedback queue, swapping KV to host memory ahead of need | Prior: preemption with proactive offload; its priority is a job's size |
| [Semantic Scheduling](https://arxiv.org/abs/2506.12204) | arXiv 2025-06-13 | In vLLM, an urgent request preempts less urgent ones until it fits, and new prefills sit out while the most urgent request decodes | Prior: both rules' ideas; its hold lasts the whole decode, the budget cap's a reply's first tokens |
| [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao) | OSDI 2024 | Migrates requests between instances, keeping memory headroom for high-priority ones | Prior: class headroom, held for a running request where Flatshare's is held before the request arrives |
| [SLOs-Serve](https://arxiv.org/abs/2504.08784) | arXiv 2025-04-05 | Allocates tokens per SLO over chunked prefill and speculative decoding | Prior: per-class token budgets |
| [QLM](https://arxiv.org/abs/2407.00047) | SoCC 2024 | Holds batch and interactive requests in one queue, ordered by estimated waiting time, with eviction and model swapping | Prior: SLO-ordered queueing |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | OSDI 2024 | Chunked prefill and stall-free batching | Baseline: vLLM's chunked prefill, swept in the tuned-engine arms |
| [FlowPrefill](https://arxiv.org/abs/2602.16603) | arXiv 2026-02-18 | Preempts running prefill at operator boundaries, so a long prefill does not delay a higher-priority first token | Complement: it protects a first token, where the budget cap protects a reply's decode steps |
| [SLOWeave](https://arxiv.org/abs/2609.07883) | arXiv 2026-09-07 | Sizes each prefill chunk to finish before the earliest decode deadline, for every decode alike | Complement: prefill held back for every decode alike, by a time budget |
| [Andes](https://arxiv.org/abs/2404.16283) | arXiv 2024-04-25 | Preemptive token-level scheduling for the quality of experience of streamed text | Complement: the text analogue of pacing decode to audio playback |
| ["Beyond Prediction"](https://arxiv.org/abs/2606.18431) (UNIBOOST) | ICML 2026 | Boosts priorities softly from lightweight statistics instead of predicting output lengths | Complement: preemption on arrival that prices KV movement, by job size |
| [QLLM](https://arxiv.org/abs/2503.09304) | EuroMLSys 2025 | In MoE inference, preempts best-effort jobs for latency-sensitive ones at any layer, down to an expert | Complement: class preemption inside a step, for MoE models |
| [AgentServe](https://arxiv.org/abs/2603.10342) | arXiv 2026-03-11 | Agents on one consumer GPU: prefills isolated from decodes, resume prefills budgeted, the GPU partitioned with CUDA Green Contexts | Complement: the same contention among agents alone |
| [Agent.xpu](https://arxiv.org/abs/2506.24045) | arXiv 2025-06-30 | Reactive and proactive agent flows on a heterogeneous SoC, with fine-grained preemption and slack-aware piggybacking | Complement: two classes of agent work on one device, an SoC rather than a GPU |
| [Aladdin](https://arxiv.org/abs/2405.06856) | arXiv 2024-05-11 | SLO-aware query placement and worker scaling | Complement: the cluster-scale question |
| [DLFP](https://arxiv.org/abs/2609.38386) | arXiv 2026-09-29 | Sizes prefill chunks from the interval between the host's scheduler calls; it did not carry over to 8B or 32B models | Lesson: an adaptive cap must time the device's steps, not the host's rhythm |
| [LayerKV](https://arxiv.org/abs/2410.00428) | arXiv 2024-10-01 | Traces the rise of TTFT with context to queueing for KV blocks; allocates and offloads KV layer by layer | Complement: the wait for KV that E2's KV case measures |

## Routers and gateways

Routers and gateways ship classes, admission and eviction today, and act in front of the engine.

| Stack (release) | Classes and admission | Making room | Capacity held for a class | Relation |
|---|---|---|---|---|
| [llm-d](https://github.com/llm-d/llm-d/releases/tag/v1.0.0) v1.0.0 and [llm-d-router](https://github.com/llm-d/llm-d-router/releases/tag/v1.0.0) v1.0.0 | [Flow control](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/flowcontrol/README.md), off by default: priority bands, fairness, queue TTLs, SLO-deadline ordering, a latency-SLO admitter | [In-flight eviction](https://github.com/llm-d/llm-d-router/pull/2093), off by default: sheddable streams end with a 429 | [`priority-holdback-policy`](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/flowcontrol/usagelimits/priorityholdback/README.md) (beta) lowers low priorities' admission ceiling as saturation rises | Baseline: flow control with holdback and eviction, emulated in `flatshare-core` and labeled as such |
| [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/tag/v1.6.2) v1.6.2 | Moved its Endpoint Picker and the `InferenceObjective` priority API to llm-d in v1.6.0 | — | — | Prior: the priority API; its policy lives in llm-d |
| [SMG](https://github.com/smg-project/smg) v1.13.0 | Four classes from an `x-smg-priority` header, per-class queues, strict priority with starvation promotion | Preempts a lower class only before its first byte, with a 503 | Reserved slots per class, in concurrency rather than tokens | Baseline: run as shipped in front of vLLM |
| [Dynamo](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.1) v1.6.1 | A router queue ordered by strict priority and deadline; a soft `priority` forwarded to vLLM and SGLang | Does "not preempt admitted work" | Does "not reserve capacity for high-priority requests" | Prior: priority queueing at the router |
| [AIBrix](https://github.com/vllm-project/aibrix/releases/tag/v0.8.0) v0.8.0 | SLO tiers at the gateway, passed to vLLM as the request priority | Through engine priority only | None found | Prior: SLO tiers mapped onto engine priority |
| [production-stack](https://github.com/vllm-project/production-stack/blob/main/src/vllm_router/routers/routing_logic.py) | `PriorityRouter` sends high priority to the least-loaded engine, so "no engine is ever reserved and left idle" | No | No | Prior: priority-aware routing |
| [vllm-project/router](https://github.com/vllm-project/router) v0.1.16 | Program scheduling with [Progress-TTL](https://github.com/vllm-project/router/pull/286): admission against logical KV capacity, with no physical eviction, offload or prefetch | No | Program capacity | Prior: active-session admission |

- **A gate cannot act on admitted requests.** Red Hat's
  [write-up](https://www.redhat.com/en/blog/shared-gpus-without-shared-latency-priority-aware-serving-in-red-hat-ai)
  of llm-d's flow control, four tiers on one H100, names the limit. "When responses run to thousands of tokens, requests
  already past the gate dominate latency while they generate, so flow control can no longer guarantee premium
  protection." That is where E2's voice turn waits. llm-d's partial eviction, which would keep the victim's KV for a
  retry, is not built ([#1119](https://github.com/llm-d/llm-d-router/issues/1119)).
- **llm-d's turn priority is not a voice baseline.** Its
  [guide](https://github.com/llm-d/llm-d/blob/main/guides/turn-priority-fairness/README.md) ranks agent turns by
  `turnNumber + 0.05 × headWaitSeconds` for fairness. The voice-relevant llm-d policy is flow control.

## Engines

### vLLM 0.32.0 (2026-10-18)

- **Priority preempts among running requests, never for a waiting one.** Under `--scheduling-policy priority`, a running
  request that cannot grow preempts the lowest-priority running request. The waiting loop stops when `allocate_slots`
  fails or the running set is full
  ([`scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py)).
- **The open fixes.** Issue [#40004](https://github.com/vllm-project/vllm/issues/40004), open since 2026-04-16, asks for
  waiting-request preemption. [#54530](https://github.com/vllm-project/vllm/pull/54530) and
  [#45561](https://github.com/vllm-project/vllm/pull/45561) add it at `max_num_seqs`, which does not bind when KV fills.
  [#45558](https://github.com/vllm-project/vllm/pull/45558) and [#40087](https://github.com/vllm-project/vllm/pull/40087)
  also preempt lower-priority running requests when a waiting request's allocation fails, one at a time, until it fits
  or none is left.
- **A self-preemption fix came after 0.32.0.** [#61976](https://github.com/vllm-project/vllm/pull/61976) is in v0.33.0,
  so stock vLLM's priority baseline may do slightly better there, which no round measured.
- **Scheduling policies under review do not preempt.** The SLO-aware [#53571](https://github.com/vllm-project/vllm/pull/53571)
  "only reorders waiting requests", and #58420 adds shortest remaining prefill first with aging.
- **Class-blind knobs**, from [`config/scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/scheduler.py):
  - `watermark`, a fraction of KV blocks kept free when admitting, off by default;
  - `max_num_queued_tokens`, which rejects new requests with a 503 once the prefill backlog reaches a limit;
  - `prefill_schedule_interval`, which admits new prefill only every N steps. In v0.32.0 only the data-parallel MoE
    engine applies it. [#54627](https://github.com/vllm-project/vllm/pull/54627), in v0.33.0, applies it to a single
    engine, and a deferred step then carries no prefill at all.
- **Holding prefill back for decodes was proposed class-blind.**
  [#56848](https://github.com/vllm-project/vllm/pull/56848)'s `--decode-priority-cadence N` was closed unmerged, and
  [#53277](https://github.com/vllm-project/vllm/issues/53277) asks for cooperative prefill preemption.

**V0 preempted for a waiting request.** [#5958](https://github.com/vllm-project/vllm/pull/5958), in v0.6.2 to v0.10.2,
preempted lower-priority running requests for the highest-priority waiting request, the lowest priority and the latest
arrival first, until it fit ([`_schedule_priority_preemption`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/core/scheduler.py#L959)).
It ran only with chunked prefill off, and its victims were recomputed, since under `--preemption-mode swap` it dropped
the swap-out list it built. V1's priority scheduling, [#19057](https://github.com/vllm-project/vllm/pull/19057),
preempts only in the running loop, and v0.11.0 removed V0.

**Relation.** Baselines: priority scheduling with the step cap, `watermark`, `max_num_queued_tokens`,
`prefill_schedule_interval` on a build with #54627, and vLLM with #54530 applied. The KV-full case is the claimed fix.

### SGLang 0.5.22 (2026-10-19)

- **Priority preemption ships** ([#8746](https://github.com/sgl-project/sglang/pull/8746),
  [#17026](https://github.com/sgl-project/sglang/pull/17026),
  [flags](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/arg_groups/fields/schedule.py)). With
  `--enable-priority-scheduling` an incoming request preempts running ones whose priority is lower by at least a
  threshold, 10 by default, and `--retraction-policy priority` retracts lower-priority decodes first when KV is full.
- **But not when KV memory runs out.** The admission loop tries preemption only at the request cap.
  [#45686](https://github.com/sgl-project/sglang/pull/45686), open, adds preemption on KV exhaustion, with
  `--preempt-keep-kv` keeping the victim's KV best-effort in the radix tree. It is SGLang's counterpart of #65282.
- **Relation.** Baseline: SGLang's priority preemption, on the same model.

### TensorRT-LLM and DeepInfer

TensorRT-LLM's [`KvCacheRetentionConfig`](https://nvidia.github.io/TensorRT-LLM/1.2.1/features/kvcache.html) gives
token ranges a retention priority with an optional duration, and its LLM API serves waiting requests by priority
([#12362](https://github.com/NVIDIA/TensorRT-LLM/pull/12362)). It is prior art for retention priority with expiry inside
an engine, and is not a baseline, since Flatshare targets vLLM.

[DeepInfer](https://github.com/deepseek-ai/DeepInfer), DeepSeek's engine for DeepSeek-V4.1-Flash, has no latency
tiers. It admits prefill first come, first served and only while KV use is at most 90%, recomputes a preempted request,
and budgets steps by their measured time. It serves one model on many GPUs and is not a baseline. Flatshare makes two of
its mechanisms class-aware: the critical-window throttle budgets background prefill in predicted time, and a KV reserve
for the realtime class is the class-aware form of its 90% watermark.

## KV across tool calls and turns

Parking KV during a tool call and prefetching it before the return is published, and E1 does not claim it. E1 measures
how much of the gap to a clairvoyant reference each level of lifecycle information closes (none, session history, a
progress oracle), per regime.

| Work | Venue, date | What it does | Relation |
|---|---|---|---|
| [InferCept](https://proceedings.mlr.press/v235/abhyankar24a.html) | ICML 2024 | At each tool interception, preserves, discards or swaps the KV to waste the least GPU time | Prior: per-pause retention choices; its min-waste policy is an E1 baseline |
| [CachedAttention](https://arxiv.org/abs/2403.19708) | USENIX ATC 2024 | Multi-turn KV in host memory and disk, preloaded layer by layer from the job scheduler's hints | Prior: prefetch from a known future |
| [Pensieve](https://arxiv.org/abs/2312.05516) | EuroSys 2025 | Conversation state cached across GPU and CPU memory | Prior: multi-turn parking |
| [HCache](https://arxiv.org/abs/2410.05004) | EuroSys 2025 | Restores state from cached intermediate activations instead of KV or tokens | Complement: a third restore path |
| [KVFlow](https://arxiv.org/abs/2507.07400) | NeurIPS 2025 | Evicts by steps to execution and prefetches for the agents scheduled next | Prior: prefetch timed by predicted activation |
| [Tokencake](https://arxiv.org/abs/2510.18586) | EuroSys '27 | Offloads idle KV during function calls and uploads it predictively | Prior: "park in host RAM, return before the tool does"; E1 baseline |
| [Continuum](https://arxiv.org/abs/2511.02230) | ICLR 2026 workshop | Pins KV in GPU memory after a tool-calling turn, with a TTL set from reload cost and queueing delay | Prior: TTL holds; E1 baseline |
| [Autellix](https://arxiv.org/abs/2502.13965) | arXiv 2025-02-19 | Program-level preemptive scheduling of LLM calls | Prior: program-level scheduling |
| [Parrot](https://arxiv.org/abs/2405.19888) | OSDI 2024 | Semantic Variables expose application structure to the server | Prior: application hints to the server |
| [ThunderAgent](https://arxiv.org/abs/2602.13692) | ICML 2026 | Pauses LLM programs under memory pressure and resumes them where capacity exists | Prior: active-session admission |
| [MORI](https://arxiv.org/abs/2606.00866) | arXiv 2026-05-30 | Places programs between HBM and DRAM by how idle they are | Prior: idleness-ranked parking; E1 baseline and the closest work to E1 |
| [CacheWise](https://arxiv.org/abs/2606.16824) | arXiv 2026-06-15 | Evicts by expected reuse, read from tool-call metadata, in vLLM | Prior: eviction from history |
| [CacheScout](https://arxiv.org/abs/2608.14624) | arXiv 2026-07-16 | Learns agents' execution transitions online to drive eviction and prefetch, in vLLM | Prior: prefetch from history |
| [Adaptive KV retention](https://arxiv.org/abs/2608.30830) | arXiv 2026-08-31 | Prices retention in GPU opportunity cost for waits of minutes to hours, such as a human's approval | Prior: retention for the think-time regime |
| [Ask the Tool](https://arxiv.org/abs/2609.18849) | arXiv 2026-09-16 | Tools report their progress, and p90 TTFT after a tool call falls about 21% against LRU, "close to an oracle" | Complement: defines E1's oracle level, and the number E1 is compared with |
| [Leyline](https://arxiv.org/abs/2606.01065) | arXiv 2026-05-31 | KV directives for an agent's context edits, such as retries and dropped tool output, without prefilling the rest again | Complement: it removes a cause of recompute, context edits, that Flatshare does not address |
| [Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang) | OSDI 2026 | Hierarchical context caching across HBM, host memory and SSD, with GPU-assisted I/O and a cache-aware scheduler, in SGLang | Complement: restore engineering |

| Thread | Status | Relation |
|---|---|---|
| vLLM [#57103](https://github.com/vllm-project/vllm/issues/57103), the Programmable KV Cache RFC, and draft [#62382](https://github.com/vllm-project/vllm/pull/62382) | Open: Retain, Pin, Prefetch and Demote directives | Complement: the upstream path of Tokencake's mechanism, and where E1's data went |
| vLLM [#58198](https://github.com/vllm-project/vllm/pull/58198), request-free CPU→GPU prefetch | Draft, no reviews | Complement: the primitive an agent-side prefetch policy would use; E1 measured what one gains, and Flatshare proposes none |
| vLLM [#37003](https://github.com/vllm-project/vllm/issues/37003), the retention API RFC, and [#38514](https://github.com/vllm-project/vllm/pull/38514) | Open, with no approving review | Complement: engine retention that Flatshare would adopt if it lands |
| vLLM's `session_id` ([#48048](https://github.com/vllm-project/vllm/pull/48048)), session ids on KV events, `KvHintsEnvelope` ([#53423](https://github.com/vllm-project/vllm/pull/53423)) | Merged | Complement: the gateway sets each request's `session_id` ([PROTOCOL.md](PROTOCOL.md#sessions)) and uses none of the others |
| SGLang's [session radix cache](https://github.com/sgl-project/sglang/pull/29173) | Merged, "soft protection, not memory pins"; the HiCache TTL pin was reverted ([#21884](https://github.com/sgl-project/sglang/pull/21884)) | Prior: session-aware retention in an engine |
| Dynamo's `agent_hints` | `speculative_prefill` warms the predicted next-turn prefix; TTL pinning was removed in [v1.1.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0) | Prior: next-turn prefill |
| llm-d-router [#2273](https://github.com/llm-d/llm-d-router/issues/2273), KV leases | Open umbrella: an advisory "evict last until time T", renewed by every turn | Prior: the agent-side lease design, which Flatshare never built |
| AIBrix [#2814](https://github.com/vllm-project/aibrix/issues/2814) | RFC, no code: session pinning with a TTL, driven by `open`, `heartbeat` and `close` signals | Prior: pinning driven by lifecycle signals |

[TraceLab](https://github.com/uw-syfi/TraceLab) ([paper](https://arxiv.org/abs/2606.30560)), a CC BY 4.0 corpus of 665,453
steps of Claude Code and Codex sessions with each tool call timed, supplies the pause distributions for E1's history
level.

## Voice under contention

We found no system that schedules cascaded voice turns beside a background agent class in one engine while managing KV
for both. VoxServe schedules speech streams alone, by their phase. Starting the LLM before the turn ends is published
and shipped at the application layer, and Flatshare compares those signals as triggers.

| Work | Venue, date | What it does | Relation |
|---|---|---|---|
| [LiveKit preemptive generation](https://docs.livekit.io/agents/multimodality/audio.md) | On by default in LiveKit Agents 1.8.8 | Starts the LLM request on a final transcript, before the end of turn is confirmed, and regenerates if the context changes | Baseline: an E2 factor, on and off; its transcript is one of the headroom triggers compared |
| [Endpoint Anticipation](https://arxiv.org/abs/2606.13450) | Interspeech 2026 | Forecasts a turn's end up to 2.56 s ahead | Complement: an earlier trigger |
| [PredGen](https://arxiv.org/abs/2506.15556) | arXiv 2025-06-18 | Generates candidate responses while the user is still speaking | Prior: input-time speculation |
| [Unmute](https://github.com/kyutai-labs/unmute) (Kyutai) | Open source, 2025 | STT, an LLM on vLLM and TTS; TTS latency about 750 ms on one L40S, about 450 ms on separate GPUs | Complement: the co-location cost in practice, fixed by separate GPUs |
| [Voice-Light](https://arxiv.org/abs/2609.20995) | arXiv 2026-09-17 | A full-duplex cascaded agent, 758 ms at the median from the final VAD endpoint to the first audio, over 36 turns | Complement: practice data, from a small sample |
| [Realtime voice agents from scratch](https://arxiv.org/abs/2603.05413) | arXiv 2026-03-05 | A tutorial cascade that sends each complete sentence to TTS while the LLM generates | Complement: the first-sentence handoff the budget cap's window must cover |
| [Metronome](https://arxiv.org/abs/2607.02640) | arXiv 2026-07-02 | Full-duplex models on one GPU collapse when session KV fills the pool, 14 of 20 runs without its KV window and none with it | Complement: why E2 reports outcomes per run, not only means |
| [VoxServe](https://arxiv.org/abs/2602.00269) | arXiv 2026-01-30 | Serves speech language models, prioritizing new requests until their first audio chunk, within a bounded concurrency | Complement: the nearest idea to the budget cap's window |
| [llmovoice](https://arxiv.org/abs/2609.04288) | SOSP 2026 | Context-management middleware that makes a call's speaking rate, background noise and packet loss explicit context | Complement: voice serving above the engine |
| [NVIDIA Nemotron Voice Agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent) | Blueprint | Single-GPU profiles run ASR, the LLM and TTS on one GPU, and list "a conversational-experience regression" as a known limitation | Complement: the edge pattern E2 reproduces |
| [LiveKit Inference](https://livekit.com/blog/latency-optimized-inference-gemma-4-on-livekit) | 2026-07-02 | Serves Gemma behind SGLang with more headroom than a throughput-maximized deployment would, since "queueing delay is very noticeable" | Complement: the headroom E2's frontier prices |
| vLLM realtime | Shipping | A WebSocket `/v1/realtime` endpoint for streaming ASR, on session-based streaming input ([#28973](https://github.com/vllm-project/vllm/pull/28973)) | Complement: a KV-holding realtime session inside vLLM |

## Simulators and eviction bounds

| Work | Venue, date | What it shows | Relation |
|---|---|---|---|
| ["When Fancy Eviction Fails"](https://arxiv.org/abs/2609.28870) | arXiv 2026-09-24 | On two companies' production traces, 14 eviction policies give "little benefit over LRU" despite "a large gap to Belady", because recency is unusually predictive | The main risk to E1: gains must come from information, not a cleverer policy |
| [LPC](https://neurips.cc/virtual/2025/poster/117662) | NeurIPS 2025 | A learned policy needs 18–47% less cache than LRU for the same hit ratio | The opposite reading of the same gap |
| [KVSET](https://arxiv.org/abs/2609.27746) | arXiv 2026-09-23 | The KV working set from LRU stack distances | Complement: miss-ratio curves for block-scaled traces |
| [Vidur](https://arxiv.org/abs/2405.05465) | MLSys 2024 | A serving simulator built from profiled operators comes within 9% of measured latency | Complement |
| [LLMServingSim 2.0](https://arxiv.org/abs/2602.23036) | arXiv 2026-02-26 | Batching, routing, offloading and memory simulated in one runtime loop come within 0.95% of real deployments on average | Complement |
| ["Where Should the KV Cache Live?"](https://arxiv.org/abs/2609.16215) | arXiv 2026-09-14 | Gains come from tier capacity, not placement, and an oracle never beats no prefetch on migration traffic | Not the E1 risk: its verdict is on migration traffic, which prefetch cannot reduce, and E1's objective is resume latency |

E1 also rests on the theory of prefetching and caching: Cao et al. (SIGMETRICS 1995), Albers, Garg and Leonardi (JACM
2000), Berger et al. on variable-size objects (SIGMETRICS 2018), SHARDS (FAST 2015) and Jung et al. on TTL caches
(INFOCOM 2003). Belady's MIN is exact only for single-tier hit counts, so with a CPU tier, prefetch and a PCIe limit E1
calls it a clairvoyant reference, not an optimum.

## Commercial cache contracts and latency tiers

Providers sell the two tiers Flatshare runs on one replica, and they harvest the cheap one by shedding it and having
clients retry. Their cache TTLs are billing terms, not residency promises, and the gateway relays only the engine's
OpenAI-compatible API.

| Provider | Latency tier | Best-effort tier | Cache contract |
|---|---|---|---|
| Anthropic | [Priority Tier](https://platform.claude.com/docs/en/api/service-tiers), overflowing to Standard; [Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode), up to 2.5× output speed | Batch | [`cache_control`](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): 5 minutes, refreshed on each use, or 1 hour, with no residency guarantee stated |
| OpenAI | [Fast mode](https://developers.openai.com/api/docs/guides/fast-mode) (`service_tier: "fast"`) | [Flex](https://developers.openai.com/api/docs/guides/flex-processing), with "occasional resource unavailability"; Batch | [`prompt_cache_options.ttl`](https://developers.openai.com/api/docs/guides/prompt-caching) "governs eligibility for cache-read pricing" and "does not guarantee" reuse |
| Google Gemini | [Priority](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-flex-and-priority-inference/), whose overflow is "served at the Standard tier instead of failing" | [Flex](https://ai.google.dev/gemini-api/docs/flex-inference), on "sheddable" capacity that "may be preempted or evicted" | [Implicit caching](https://ai.google.dev/gemini-api/docs/caching), on by default, with no hit guarantee |
| AWS Bedrock | [Priority](https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-bedrock-priority-flex-inference-service-tiers), up to 25% more output tokens a second | Flex, at a discount | — |

## What we do not claim

- **That any mechanism marked "Not claimed" is new.** Each is published or shipped ([at a glance](#at-a-glance)).
- **A cache residency guarantee.** Providers promise billing eligibility, not residency, and Flatshare makes no cache
  promise.
- **A better router.** `flatshare-gateway` is the serving control plane of one or two GPU nodes without Kubernetes, not
  a fleet router ([ADR-0007](adr/0007-the-gateway-is-the-serving-control-plane.md)).
- **Results beyond the two cards measured.** E2 ran on one 16 GB consumer GPU with a 4B model and traces block-scaled
  32×, and its core ran like for like on one A100 SXM4 80 GB at the box's pool
  ([the A100 run](BENCHMARKS.md#do-the-rules-carry-to-a-datacenter-card)). Other cards, and the A100 at its own pool,
  are placed on E1's map by `flatshare-sim` from modeled cost fits, labeled as modeled.
- **Precedence.** Nothing here claims to be first or the only one. Each comparison rests on the sources cited.
