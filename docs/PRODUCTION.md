# Running a voice agent beside background work on one GPU

This guide is for an operator who serves a realtime voice agent (speech to text, an LLM, text to speech) with vLLM on
their own GPU, and wants the same GPU to run background LLM work such as coding agents. It says what the deployment
promises, whether a deployment is in the regime where the change pays, how to configure it, what to watch and how it
fails. Its numbers come from an RTX 5060 Ti 16 GB serving Qwen3-4B-Instruct-2507-FP8 beside Whisper and Kokoro, and
[the technical report](REPORT.md) and [BENCHMARKS.md](BENCHMARKS.md) hold their evidence.

## What you deploy and what it promises

Where a second card is available, deploy consolidation with a warm spare ([Two cards](#two-cards)). Both calls and four
agent sessions run on the first card, and agents on a spare that keeps its speech servers resident so that it can take
the calls. With one card, the card carries both calls and up to 24 agent sessions
([the envelope](#the-envelope-from-version-124)). The two-card counts predate the engine's additions and version 1.24's
queue and copy, on which the one card's 24 rests. On this lab's hosts Phastform is the deployment, and the second card
runs in Gray's windows ([the runbook](../ops/RUNBOOK.md#two-gpu-nodes)).

### The service contract

What the two cards promise:

| | The promise |
|---|---|
| Calls accepted | Two at once, which the voice worker's serve mode admits ([Calls](#calls-what-the-box-admits)); each card's `max_calls` holds both |
| A call's turns | One at a time, replies of one or two short sentences, the first within the 48-token window, and interruptions answered |
| Agents accepted | 16 sessions at once, four on the first card and 12 on the spare; one more gets 503 `no_agent_slot` with `Retry-After: 1` |
| The calls' outcome | At most 5% of the replies over the SLO, voice alone's p95 plus 250 ms on that card |
| The agents' outcome | Their token rate and time to first token as measured ([below](#what-the-configuration-to-deploy-measured)) |
| Overload | Refused, never queued without bound: a call beyond two, an agent session beyond 16, a request beyond the gateway's bounds |
| Recovery | The first card's calls move to the spare by their next turn, and the spare's agents wait while it carries them ([How it fails](#the-first-of-two-cards-fails)) |
| Not covered | More than two calls at once, replies longer than the window, real callers' microphones, and an outage of both cards or of cyan's control plane ([below](#what-the-measurements-do-not-cover)) |

- **Agents' slots.** A slot frees when its session ends or idles out (`idle_s`, 150 s here). End a session with
  `DELETE /v1/sessions/{id}` or a `session_closed` event, or the client holds its slot until then.
- **Requests without a session.** A background request that names no session (no trusted `X-Session-ID`,
  `X-Claude-Code-Session-ID` or `prompt_cache_key`) takes no agent slot. It runs only on a card that carries no call
  and holds fewer sessions than its `max_agents`, and gets 503 `no_agent_slot` when no card does
  ([Nodes](../crates/flatshare-gateway/README.md#nodes)).
- **What waits.** Nothing is deferred but the spare's agents, while the spare carries any call.

### What the configuration to deploy measured

| Measured | Agent sessions | Replies over the SLO | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|
| **One card from version 1.24**, three rounds | 24 | 3 / 177 (1.7%) | 7,253 | 4.3 / 133 s |
| The same card with fewer agents, three rounds | 12 | 4 / 177 (2.3%) | 6,906 | 2.3 / 24 s |
| **Two cards, consolidation with a warm spare**, three rounds | 4 + 12 | 5 / 177 (2.8%) | 9,652 | 0.5 / 10.7 s |
| The same two cards for 24 hours, calls arriving at 20 an hour | 14, at most 4 beside the calls | 74 / 5,058 (1.5%) | 3,529 | — |

Each round ran two callers beside a new sample of the agents, against the round's own SLO. The agents' tokens a second
are the median over the calls' first 240 s, summed over both cards where there are two, and the soak's are over its
whole day ([the envelope rounds](../benchmarks/results/2026-11-08-envelope/README.md),
[the two-card rounds](../benchmarks/results/2026-10-28-rounds/README.md),
[the soak](../benchmarks/results/2026-10-28-soak/README.md)).

The two rules are what move the voice tail. Beside 24 sessions at the box's 512-token step cap, they take two calls'
p95 voice-to-voice from 10,477 ms on tuned vLLM to 2,236 ms, and the agents keep 97% of their token rate. There the
SLO is voice alone plus 250 ms at p95, 2,034 ms, and 32% of the replies are still over it; the serving profile's step
cap and additions bring that within the SLO's 5% allowance ([the report's results](REPORT.md#6-results)).

### The envelope from version 1.24

From version 1.24 the serve module orders the agents' waiting queue by the prefill each request has left, net of what
the GPU pool and the CPU tier hold. It also loads the CPU tier through the SM copy kernel
([the report's §6.11](REPORT.md#611-ordering-the-agents-by-what-they-have-left-raises-the-cards-envelope-from-4-sessions-to-24)).
With both, one card qualifies beside both calls with 24 agent sessions, the most the rounds measured, and the profile's
`max_agents` is 24. The agents pay in their tail. A long prompt can wait out the queue's 120 s bound, so their time to
first token at p99 grows with their sessions. A node that needs a shorter tail takes fewer sessions.

### What the measurements do not cover

- **A lab deployment.** The hosts are shared, the second card serves in measurement windows, and the callers are
  synthetic. Real callers need a domain and its certificate, TLS and TURN in front of LiveKit, a service that issues
  their tokens, and [ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md)'s loopback design revisited.
- **Real callers.** No run measured serve mode's audio for real microphones ([below](#real-callers-audio)), or a voice
  agent other than this repository's LiveKit worker.
- **The agents' work.** Their token rate counts the replay's tokens, not tasks done, and the served model failed the
  quality gate for useful background work ([below](#can-the-served-model-do-the-background-work)).
- **Paging and licences.** The alerts are rules for an operator's Prometheus, and nothing pages anyone. The contract
  does not cover what the third-party licences ask of whoever ships it ([THIRD-PARTY.md](../THIRD-PARTY.md)).
- **Other GPUs.** Only the RTX 5060 Ti and an A100 were measured; [HARDWARE.md](HARDWARE.md) renders a node for another.

## Is your deployment in this regime?

The change matters when background KV fills the pool, with agent contexts of 10k tokens or more against a pool that
holds a few of them. On stock vLLM the signs are KV cache usage near 1 (`vllm:kv_cache_usage_perc`) while requests wait
(`vllm:num_requests_waiting`), and a realtime time to first token of seconds at p95. Where the pool has room the change
does less. At 16 agent sessions of the whole corpus the configuration to deploy had the lower p95 in each of five
rounds, but met the SLO in one ([BENCHMARKS.md](BENCHMARKS.md#the-standard-scenario-in-five-rounds-w5)).

### Can the served model do the background work?

Sharing the card shares the model. The quality gate's useful pair of jobs is answering questions about private
documents in a call (QASPER) beside enriching new documents into searchable records (Re-DocRED). The served model
failed 9 of its 11 criteria. Of its answers 50.6% were right, and of the facts in its records 23.9%, against 80% for
each. Qwen3-8B-FP8 on a card of its own failed too ([the gate](../benchmarks/quality/GATE.md);
[its record](../benchmarks/results/2026-10-27-quality/README.md)). Check your model on your background work before
sharing a card for it; [`benchmarks/quality`](../benchmarks/quality) runs the gate on another model or pair of jobs.

## The configuration

### The engine

The serving profile ([`rtx-5060-ti-serve.toml`](../deploy/profiles/rtx-5060-ti-serve.toml)) runs the frozen box's
[`engine.yaml`](../deploy/phastform/engine.yaml), whose settings are the first four rows below. The
[serving variant](../deploy/phastform/variants/make-room-serve/flatshare-engine.conf) passes the rest over it, loads
`flatshare_make_room_serve.AsyncScheduler` with `--scheduler-cls`, and gives each switch as `--additional-config.KEY`:

| Setting | Value | Why |
|---|---|---|
| `--scheduling-policy` | `priority` | Voice at priority 0 runs before agents at 2 |
| `--enable-prefix-caching` | on | The agents' requests share long prefixes |
| `--kv-cache-memory` | 5 GiB, 36,400 tokens | The pool depends neither on the profiler nor on the audio servers' memory at start |
| `--kv-transfer-config` | `OffloadingConnector`, 20 GiB, written through | A preempted agent reloads its KV: 509 ms against 3.26 s recomputed for a 16k-token context ([its cost](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0)) |
| `--max-num-batched-tokens`, `--long-prefill-token-threshold` | `256`, `0` | The step cap, which bounds how long a decode waits behind a prefill chunk ([the sweep](../benchmarks/results/2026-10-27-sweep/README.md)) |
| `make_room.max_preemptions_per_step` | `2` | The KV preemption: a waiting higher-priority request that lacks KV preempts lower-priority ones |
| `make_room.budget_cap`, `make_room.budget_cap_tokens` | `true`, `48` | The budget cap: lower-priority prefill sits out a reply's first 48 tokens ([the window](#the-replys-first-tokens)) |
| `make_room.reserve_tokens` | `0` | No reserve: beside the KV preemption it showed no effect |
| `make_room.take_reservations` | `true` | A call leaves free only its own class's block reservations, so it does not wait behind the agents' loads ([the stalls](../benchmarks/results/2026-11-01-engine-points/README.md#the-stalls)) |
| `make_room.retain_tokens`, `make_room.retain_s` | `3584`, `30` | The calls' prefixes stay on the GPU between turns, within 3,584 tokens and 30 s ([the engine points](../benchmarks/results/2026-11-01-engine-points/README.md#what-was-adopted)) |
| `make_room.hint_hold_ms` | `400` | The voice worker announces each turn when the user stops speaking, and the budget cap holds agent prefill for 400 ms from it |
| `make_room.decode_hold_tokens` | `48` | The decode hold: the agents' decodes sit out a call's first 48 tokens. In the follow-up rounds it put 16 of 720 replies over the SLO against 55 but cost the agents 3.8% of their token rate at the median ([the rounds](../benchmarks/results/2026-11-02-opt-rounds/README.md#the-decode-hold-and-speculative-decoding)) |
| `flatshare_jit.margin_ms`, `flatshare_jit.step_cost.*` | `30`, the card's step-time model | The next batch is scheduled when the step in flight is due to end, so a call that arrives during a long agent step joins it |
| `flatshare_queue.max_wait_s` | `120` | The cache-aware queue: the agents with the least prefill left go first, and one that has waited 120 s goes ahead |
| `flatshare_copy.sm_load_programs` | `4` | The CPU tier's loads take vLLM's Triton copy kernel, which on PCIe 3.0 x8 does not hold the steps beside each load |

- **Upstream.** The KV preemption and the budget cap are the pull requests
  [#65282](https://github.com/vllm-project/vllm/pull/65282) and
  [#65283](https://github.com/vllm-project/vllm/pull/65283), proposed to vLLM and open, with no maintainer review as of
  2026-11-09. On a vLLM with them the same rules are `--max-priority-preemptions-per-step 2
  --priority-prefill-budget-cap --priority-prefill-budget-cap-tokens 48`. The other switches are make-room's or the
  serve module's, and each is off at 0 ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change)).
- **The step cap's trade.** A voice turn that arrives with steps in flight waits for them. On an A100, vLLM's default
  2,048/0 gave the agents 1.4 times the tokens a second of 512/256 but left the voice p95 0.14–0.17 s over its SLO,
  where 512/256 kept it under ([the A100 run](BENCHMARKS.md#what-it-measured-2026-10-26)). Pick the cap by the voice
  SLO's margin.
- **The build.** On vLLM 0.32.0 or 0.33.0, `build.py` writes make-room's module from the installed release without
  modifying it (`python python/flatshare_vllm/src/flatshare_vllm/build.py make-room OUT_DIR`), and the engine loads it
  from `PYTHONPATH=OUT_DIR`. A trace test holds it to the pull requests' diff on 0.32.0, step for step. Version 1.24's
  rounds ran it after the pull requests' revisions of 3 and 5 November, and every other measurement on this page before
  them ([the report's §4](REPORT.md#4-implementation)).
- **Held to the profile.** [`serving.json`](../deploy/phastform/serving.json) holds every setting above, and each start
  of the engine fails on any difference from it
  ([the serving profile](../deploy/phastform/variants/README.md#the-serving-profile)).
  [`deploy/container`](../deploy/container/README.md) runs the same engine and check from vLLM 0.32.0's own image.

### The priorities

Voice requests go at priority 0 and background work at 2; in vLLM a lower value is served first. A client can send
`priority` in the request body or the `X-Vllm-Priority` header, but a gateway should set it from the caller's identity,
so that a background client cannot raise its own. `flatshare-gateway` maps each tenant to a class, capped by its
`max_class`, and the configuration to deploy needs no admission policy in front.

The engine must be reachable only through the gateway, the serving control plane
([ADR-0007](adr/0007-the-gateway-is-the-serving-control-plane.md)), since vLLM's `--api-key` guards only some of its
routes. Serve the engine on a Unix socket in a directory only its user may enter (`--uds`, as
[`deploy/phastform/systemd`](../deploy/phastform/systemd) does), and reach it on that socket or through a tunnel that
may reach nothing else ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md#unix-sockets-end-to-end-w6-2026-10-25)).
[The gateway's README](../crates/flatshare-gateway/README.md#what-it-does-not-do) lists what it leaves to others, such
as TLS.

### An existing LiveKit or Pipecat agent

The voice worker here is a LiveKit Agents worker ([`flatshare_voice`](../python/flatshare_voice/README.md)). Another
voice agent keeps its own pipeline and points its LLM client at the gateway:

| | What to set |
|---|---|
| The endpoint | The gateway's OpenAI-compatible routes: on cyan, `http://127.0.0.1:18080/v1` with a tenant's key, or the voice plane's Unix socket ([`gateway.example.toml`](../deploy/cyan/gateway.example.toml)) |
| The class | The call's requests in `realtime`: a tenant of that `default_class`, as `[tenants.voice]`, or `X-Flatshare-Class: realtime` from a trusted tenant whose `max_class` allows it |
| The session | One `X-Session-ID` a call, from a trusted tenant ([PROTOCOL.md](PROTOCOL.md#sessions)) |
| The priorities | The gateway sends each class's, realtime 0 and background 2, and removes a client's own ([PROTOCOL.md](PROTOCOL.md#headers)) |
| The agents | A tenant of class `background`, as `[tenants.agents]` |
| LiveKit Agents | `openai.LLM(model=..., base_url=..., api_key=..., extra_headers={...})`, the headers as [`plugins.py`](../python/flatshare_voice/src/flatshare_voice/plugins.py)'s `make_llm` sets them |
| Pipecat | `OpenAILLMService(base_url=..., api_key=..., default_headers={...})` |

With two cards a call's speech servers are its node's. The voice worker places each call with `POST /v1/sessions` and
binds its STT and TTS to the node that answers ([Nodes](../crates/flatshare-gateway/README.md#nodes)).

### The reply's first tokens

The budget cap protects a reply for its first T tokens, its window. The caller hears nothing until the reply's first
complete sentence reaches TTS, so T must cover the first sentence, not the first token. The rest of a longer sentence
decodes beside agent prefill again, its tokens 60–65 ms apart instead of 16–20 ms
([the prefill case](BENCHMARKS.md#the-prs-configuration-in-one-session-w5)).

Choose T to cover your agent's first sentence at its longest. This voice agent's first sentences run 15 tokens at the
median and 39 at most ([the first sentences](../benchmarks/results/2026-10-27-first-sentence/README.md)), and at T = 16
its p95 was 3,094 ms, against 2,236 ms at 48. A longer window costs the agents little, since a hold ends with its reply.
Their prompt throughput fell 8% from 16 to 48 and no further at 64
([the window sweep](../benchmarks/results/2026-10-25-window-knee/README.md)). When in doubt, choose T on the long side.

### Calls: what the box admits

The voice worker's serve mode, the serving configuration
([`deploy/cyan/variants/serve`](../deploy/cyan/variants/serve/flatshare-voice-worker.conf)), admits at most the serving
profile's two calls at once, beside which the envelope was measured. Bench mode, which every measurement but the soak
ran, takes every call LiveKit dispatches.

- **A full worker** is marked full, so LiveKit sends it no job, and a job request beyond the limit is refused.
- **A line that frees** reaches LiveKit at once, since the worker sends its status as soon as a call's process closes.
  At the SDK's own pace, a status every 2.5 s, 10 of 233 calls arriving faster than two lines take them were turned
  away with a line free, against none with the status sent at once
  ([the runs](../benchmarks/results/2026-10-29-status-send/README.md)).
- **No call** is taken while the gateway's `/ready` says the engine does not serve, or while a release drains the
  worker.
- **A refused call** gets no agent, so a client should take an agent that has not joined within a few seconds for a
  busy line. Admission never cuts a call in progress.

### Real callers' audio

Serve mode runs the profile's `microphone` audio configuration for real callers. LiveKit's end-of-turn model ends a
turn, the VAD needs a clearer onset to count speech and 0.6 s of it for an interruption, LiveKit's AEC warm-up stays
on, and noise suppression is the caller's client's
([its settings](../python/flatshare_voice/README.md#audio-configurations)).

### What the voice path keeps

| Data | Where | Kept until |
|---|---|---|
| Callers' audio | Memory only: LiveKit records nothing (it has no egress), and the STT and TTS servers keep no audio | The call's end |
| Transcripts and the agent's replies | The voice worker's memory, which a failover's re-synthesis needs for the sentence it cut | The call's end |
| The engine's KV | GPU memory, and the CPU tier's mapping in `/dev/shm` (mode 0600, the deploying user's) | Eviction, or the engine's stop |
| Metadata: session ids, room and participant names, timings, token counts, outcomes | The hosts' user journals: the gateway's request lines, the worker's turn lines, LiveKit's room events | As long as the hosts' journald keeps them |

The logs keep no transcript or reply at the levels the units run: the engines log no request, LiveKit Agents logs a
transcript only at DEBUG, and the TTS server's unit sets `API_LOG_LEVEL=WARNING`. Nothing in the repository sets the
journals' retention. An operator with real callers sets it, for example with `SystemMaxUse=` in journald.conf, and
decides consent.

### The audio models

Whisper and Kokoro share the GPU by time-slicing. MPS does not help, since the interference that matters is inside the
LLM engine ([Voice on the shared GPU](BENCHMARKS.md#voice-on-the-shared-gpu)). Pin each server to its own CPU cores,
the engine on the GPU's NUMA node and the audio servers off it. Keep the audio models on the GPU. On the CPU, Kokoro's
first audio took 991 ms with one call and 2,736 ms with two, against 211–242 ms for the GPU's whole TTS stage
([the CPU record](../benchmarks/results/2026-10-27-cpu-audio/README.md)).

## Two cards

With two cards, put both calls on one and fill the other with agents. In a resource gate's three rounds, both calls and
four agent sessions on one card did 34% more background work than dedicated roles within the SLO, and tuned vLLM's
consolidation, at two sessions, 27% more ([the report's
§6.8](REPORT.md#68-on-two-cards-consolidating-both-calls-on-one-does-34-more-background-work-most-of-it-without-the-change)).
The arrangement to deploy also keeps the speech servers resident on the agents' card, which can then take the calls, and
still does 22% more background work than dedicated roles (Figure 1).

<picture>
  <source media="(max-width: 767px)" srcset="figures/two-cards-narrow.svg">
  <img src="figures/two-cards.svg" width="100%" alt="Two panels of the two cards' slots as the gateway bounds them, two call slots each and up to 4 agents' sessions on Phastform and 12 on Gray: in panel a, in service, both calls fill Phastform and Gray, the warm spare, runs agents only; in panel b, Phastform's engine failed, both calls take Gray's call slots at their next turn, 0.7 s after a kill and 1.9 s after a stop in the drills, and Gray holds its agents' requests until the moved calls end.">
</picture>

**Figure 1. Both calls fill the first card, and the spare takes them by their next turn when the first fails.** (a) Each
card has two call slots (`max_calls = 2`) and up to 4 or 12 agent sessions (`max_agents`), and `calls = "fill"` places
each call on the first card with a free call slot. (b) The timings are the failover drills'
([the record](../benchmarks/results/2026-10-28-node-failover/README.md)), and the spare holds its agents' requests,
hatched, until the moved calls end.

### Setting it up

In `gateway.toml` ([`deploy/cyan/gateway.example.toml`](../deploy/cyan/gateway.example.toml) has both `[[nodes]]`
blocks):

| Key | Value | Why |
|---|---|---|
| `[[nodes]]` | An entry per card, both with the realtime and background roles and `max_calls = 2` | Each card can carry both calls, so that the spare can take them |
| `max_agents` | 4 on the first, 12 on the spare | The counts the two-card rounds measured; a release allows the first up to the profile's 24 |
| `[placement] calls` | `"fill"` | Each call goes on the first node with a free call slot, in the file's order |

A node with both roles must give the agent sessions it takes beside its calls, a count or `"unbounded"`, or the gateway
refuses to start. Each release on cyan holds the running gateway's nodes to the serving profile's envelope
(`serving-check --nodes`). The runbook's [Two GPU nodes](../ops/RUNBOOK.md#two-gpu-nodes) says what each key does, and
how health, placement and failover work.

## What not to deploy

Each p95 is the median of its configuration's runs beside the same 24 sessions. The row of #63350 alone comes from
[the main evaluation's rounds](BENCHMARKS.md#how-close-does-each-configuration-bring-the-voice-tail-to-its-slo), and
the others from [the campaigns before them](BENCHMARKS.md#the-discriminating-scenario-in-rounds-with-arm-9-and-the-change-w4), on
one sample of the agents' traces.

| Don't deploy | Measured | Why not |
|---|---|---|
| Lifecycle-timed protection in a gateway: KV headroom from speech onset, a prefill throttle while a turn is answered | 10,993 ms without the change, as stock | With agents' KV filling the pool it could not act ([why](REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work)) |
| #63350 alone | 5,085 ms | The turn is admitted, and its reply decodes beside agent prefill |
| A static KV reservation | 5,671 ms | It halves the tail but stays seconds over the SLO |
| SGLang 0.5.22's priority preemption, with or without its host tier | 11,948 ms with HiCache | It preempts only at the request cap, not when KV runs out, which SGLang's [#45686](https://github.com/sgl-project/sglang/pull/45686) proposes |
| `--max-num-queued-tokens` | 48 of 50 voice turns failed | It refuses voice turns too |
| `--watermark`, concurrency caps, [#54530](https://github.com/vllm-project/vllm/pull/54530) | — | The watermark is class-blind and changed nothing, two agent contexts fill the pool under any cap, and #54530 acts at a request-count cap that does not bind here |
| An FP8 KV cache beside the CPU tier (`--kv-cache-dtype fp8`) | Twice the tokens, and no quality lost in the gate's screen | The engine died both times beside two calls and 24 agent sessions, on a CUDA fault of stock vLLM 0.32.0 that vLLM #64165 would fix ([the record](../benchmarks/results/2026-10-31-fp8-crash/README.md)) |
| vLLM's development mode (`VLLM_SERVER_DEV_MODE=1`) | — | `--api-key` does not guard its routes, such as `/collective_rpc` ([the routes](../deploy/phastform/variants/README.md#the-measurement-drop-in)) |
| KV cache events on TCP | — | They send every cached block's token ids to whoever binds the endpoint |

Without an engine change, the gateway's `preempt-resubmit` policy, which aborts background requests that have no first
byte yet and resubmits them, comes nearest the SLO at 2,522 ms. It costs the agents' tail (a TTFT p99 of 193 s against
106 s), and cannot reach a request that is already decoding.

## Releasing a change

`deploy/deploy.sh HOST` releases a commit to a host as an immutable release, `~/flatshare/releases/ID`, with a manifest
of everything it holds. It verifies the release, drains what the switch would cut, switches one link and restarts only
the units whose inputs changed. It then checks them, the engine's configuration against the serving profile among them,
and switches back by itself on any failure. `deploy.sh HOST --rollback` returns to the previous release by the same
steps ([the runbook](../ops/RUNBOOK.md#deploy); [ARCHITECTURE.md §3.6](ARCHITECTURE.md#36-releases)).

## What to watch

- **Voice.** The V2V p95 per call, and the realtime time to first token at p95 from the gateway's request log. The
  voice agent's stage times split a slow turn into endpointing, the LLM's first sentence and TTS (Figure 2).
- **The engine.** KV usage, running and waiting requests, and preemptions. `vllm:num_preemptions_total` counts a
  victim when its next output reaches the frontend, so it lags, and no vLLM 0.32 counter shows what a preempted request
  recomputes ([what the agents pay](BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)).
- **What the engine held.** At most every 10 s the engine logs the steps it held, the prefill and admissions it held
  back, its preemptions, the turns announced and the decodes held. Turns announced that stay at 0 while calls run mean
  the voice worker's hints do not reach the engine at priority 0.
- **The agents.** Their completed tokens a second and their TTFT p99, where a preemption policy moves the cost.
- **As code.** [`ops/`](../ops) holds the Prometheus scrape configuration, the burn-rate alerts of the two production
  objectives ([SLO.md](SLO.md#the-production-objectives)), health alerts and a Grafana dashboard
  ([the runbook](../ops/RUNBOOK.md#monitoring)).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-stages-narrow.svg">
  <img src="figures/report-stages.svg" width="100%" alt="The voice turn's mean voice-to-voice time stacked by stage from the end of the caller's speech, beside 24 agent sessions but for voice alone: voice alone, 1.62 s, 0.09 s of it waiting for the LLM's first token and 0.22 s decoding the first sentence; tuned vLLM, 5.30 s, 2.81 s of it waiting for the LLM's first token and 1.07 s decoding the first sentence; preempt-resubmit, 2.08 s, 0.28 s of it waiting for the LLM's first token and 0.39 s decoding the first sentence; the KV preemption, 3.41 s, 0.78 s of it waiting for the LLM's first token and 1.21 s decoding the first sentence; the change, 2.15 s, 0.30 s of it waiting for the LLM's first token and 0.44 s decoding the first sentence; the configuration to deploy, 1.97 s, 0.29 s of it waiting for the LLM's first token and 0.26 s decoding the first sentence.">
</picture>

**Figure 2. A slow turn shows in two stages, the wait for the LLM's first token and the decode of its first
sentence.** The mean time of each stage from the end of the caller's speech, over every reply of each configuration's
runs beside 24 coding-agent sessions ([the report's Figure 1](REPORT.md#1-introduction)).

## How it fails

| Failure | What it does | What to do or expect |
|---|---|---|
| A voice turn that two preemptions cannot make room for | It waits as on stock vLLM: the change preempts at most `N` requests a step, and none if `N` would not do | — |
| Many calls at once | Replies that overlap hold lower-priority prefill back continuously | Watch what the engine held ([below](#many-calls-at-once)) |
| A KV load that never completes, or a request that computes nothing | It holds lower-priority prefill for at most `budget_cap_tokens` steps in a row | — |
| An agent's long prompt beside agents whose prefixes are cached | It waits for the agents with less prefill left, at most 120 s | The agents' TTFT p99 grows with their sessions ([the envelope](#the-envelope-from-version-124)) |
| The SM copy kernel cannot take the CPU tier's loads (host memory not pinned, no Triton) | The engine refuses to start | Set `sm_load_programs = 0` for vLLM's copy engine |
| The engine dies (an out-of-memory kill, a crash) | Every stream it serves breaks, and with one GPU node voice is down until it restarts | systemd restarts it ([below](#the-engine-dies)) |
| The first of two cards fails | Its calls move to the spare by their next turn, and the spare's agents wait while it carries any call | The moved calls stay on the spare until they end ([below](#the-first-of-two-cards-fails)) |
| The gateway restarts | A reload refuses nothing and makes no request wait; background streams are cut after 5 s | Clients may send the cut requests again ([below](#the-gateway-restarts)) |
| Host memory pressure | The CPU tier's shared memory is charged to the engine's cgroup, so `systemd-oomd` picks the engine first | Keep memory pressure low, and set `ManagedOOMPreference=avoid` in a drop-in for `flatshare-engine.service` |
| Another process on the GPU | It slows every stage | Keep the GPU exclusive, as `gpu-guard` does for the box's units |

### Many calls at once

The budget cap holds lower-priority prefill while any realtime request's first tokens are due, so the agents' prefill
stops once each reply's window overlaps the next. On vLLM's real scheduler on CPU, beside six agents, that came at one
48-token reply every 40 steps ([the test](../python/flatshare_vllm/tests/test_make_room.py)). At the 16–20 ms a step
the box measures under the cap, it is about one realtime reply a second, some six calls turning every 6 s. The
evaluation measured up to six callers ([calls per GPU](BENCHMARKS.md#calls-per-gpu-w5)).

### The engine dies

The gateway sends each event stream vLLM's error event, of type `upstream_unavailable`, before it aborts it. The next
request gets one 502, and later ones 503 `upstream_not_ready` with `Retry-After: 1` until the engine's `/health`
answers. systemd starts the unit 5 s after its last process exits, and `StartLimitBurst=3` in 30 minutes ends a crash
loop. In the drill each stream got the error event 6–11 ms after the kill command. A request succeeded 106.2 s after
the kill, most of it the engine's warm start, and on cold caches after an upgrade a start takes about nine minutes
([the drills](BENCHMARKS.md#failure-drills)).

### The first of two cards fails

The first card's calls move to the spare by their next turn, and none of the 300 turns after the failure failed in
the drills. While the spare carries any call its agents' new requests wait out their queue timeout (120 s) and get 503
`queue_timeout`, and the first card's agent sessions get 503 `no_agent_slot` until a slot frees. Once healthy, the
first card takes the new calls, and the moved calls stay on the spare until they end. For maintenance, drain a card
instead ([the runbook's steps](../ops/RUNBOOK.md#drain-one-node)), a drill the two cards have not run.

### The gateway restarts

A deploy reloads the gateway (`systemctl --user reload flatshare-gateway`), which hands its listening sockets to a
successor that serves at once, and drains. Realtime and interactive requests in flight finish in the old process, and
background ones are cut after `background_drain_s` (5 s) with a `draining` error event or a 503 with `Retry-After`.
`flatshare-gateway.socket` holds the sockets across a restart or a crash, so new connections wait rather than fail. In
the drills a reload and a restart failed none of 42 requests, and a kill none
([the drills](BENCHMARKS.md#failure-drills)).

## Checking it on your hardware

On another host, [HARDWARE.md](HARDWARE.md) renders the node from a hardware profile and runs the A100 run's kit. The
kit is not a deployment. Nothing of it starts at boot, and releases, rollback and the serving check reach only this
lab's hosts.

- **The probes.** `python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py ENGINE kv` sends a voice-sized request
  at priority 0 while long priority-2 requests hold the pool, and `probe.py ENGINE prefill` sends it behind three long
  prefills. ENGINE is `unix:` and the engine's socket, or `http://host:port`.
- **Agents beside calls.** `tail.py ENGINE SECONDS [AGENTS] [fresh|grow] [--reply TOKENS]` runs closed-loop agents
  beside two calls and prints the calls' TTFT distribution and the agents' throughput.
- **The whole measurement.** An arm file of the end-to-end harness with your traces, and `flatshare-bench e2 run`
  ([benchmarks/e2](../benchmarks/e2/README.md)).
- **The change's upkeep.** Before changing the engine's code or moving off vLLM 0.32.0, `make perf-make-room` and
  `make compat-upstream` measure the change's cost a scheduler step and its fit on vLLM's main
  ([how to run and read them](../python/flatshare_vllm/README.md#upkeep-the-changes-cost-and-its-fit-upstream)).
