# Running a voice agent beside background work on one GPU

This guide is for an operator who serves a realtime voice agent (speech to text, an LLM, text to speech) from their own GPU with vLLM, and wants the same GPU to run background LLM work: coding agents, post-call summaries, batch jobs. It opens with what the deployment promises and what it measured, then gives the configuration, what each part buys, what to watch and how it fails. The numbers come from one RTX 5060 Ti 16 GB serving Qwen3-4B-Instruct-2507-FP8 beside Whisper and Kokoro ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)), and an A100 SXM4 80 GB ran the same comparison on 2026-10-26 ([the A100 run](BENCHMARKS.md#what-it-measured-2026-10-26)). The mechanisms are not specific to the card, though what each costs is; the simulator's regime map places other hardware by a model ([the report's §6.7](REPORT.md#67-the-rules-carry-to-an-a100)).

## What you deploy and what it promises

The arrangement to deploy, where a second card is available, is consolidation with a warm spare (both calls and four agents' sessions on one card, agents on a spare that keeps its speech servers resident: 22% more background work than dedicated roles, a card each for the calls and the agents, within the SLO, and a standby that arrangement lacks; [Two cards](#two-cards)). On this lab's hosts, one node, Phastform, is the deployment. The two cards run in Gray's windows ([the runbook's Gray's windows](../ops/RUNBOOK.md#two-gpu-nodes)), and the box is paused when idle.

### The service contract

What the two cards promise at that configuration, and what they do not:

| | The promise |
|---|---|
| Calls accepted | Two at once in all, which the voice worker's serve mode admits ([Calls](#calls-what-the-box-admits)); each card's `max_calls` holds both, so that the spare can take them |
| A call's turns | The voice agent's: one at a time, its replies one or two short sentences, the first sentence within the 48-token window, and interruptions answered |
| Agents accepted | 16 sessions at once, four on the first card and 12 on the spare; one more gets 503 `no_agent_slot` with `Retry-After: 1` |
| The calls' outcome | At most 5% of the replies over the SLO: the rounds measured 2.8% at four sessions, the 24-hour soak 1.5% |
| The agents' outcome | Their tokens a second and time to first token, as measured: 9,652 tokens a second; 0.5 s at the median, 10.7 s at p99 |
| Overload | Refused, never queued without bound: a call beyond two, an agent's session beyond 16, a request beyond the intake's bounds |
| Recovery | The first card's calls move to the spare by their next turn; the spare's agents wait while it carries them ([How it fails](#how-it-fails)) |
| Not covered | More than two calls at once, replies longer than the window, real callers' microphones, and an outage of both cards or of cyan's control plane |

- **Agents' slots.** A slot frees when its session ends or idles out (`idle_s`, 150 s here, without a request), and every request is bounded per class and tenant by the gateway's intake, refused at once beyond it with `Retry-After`. A client that starts new sessions without ending the old ones holds the slots of those it has left until they idle out: end a session with `DELETE /v1/sessions/{id}` or a `session_closed` event, or offer a few sessions under the bounds ([the two-node smokes](../benchmarks/results/2026-10-28-two-node-smoke/README.md)).
- **Requests without a session.** The bounds count agents' sessions: a background request that names none (no trusted `X-Session-ID`, `X-Claude-Code-Session-ID` or `prompt_cache_key`) takes no agent slot, so it runs only on a card that carries no call and holds fewer sessions than its `max_agents`, and gets 503 `no_agent_slot` when no card does ([Nodes](../crates/flatshare-gateway/README.md#nodes)).
- **What waits.** Nothing is deferred but the agents of a card that covers: the spare's while it carries any call, and a card's that carries calls moved to it from a failed one.
- **Real callers' microphones** are serve mode's audio, which no run measured ([below](#real-callers-audio)), and the contract does not cover what the third-party licences ask of whoever ships the deployment ([THIRD-PARTY.md](../THIRD-PARTY.md)).
- **The agents' tokens** count the replay's tokens, not tasks done, and the served model did not pass the quality gate for useful background work ([below](#can-the-served-model-do-the-background-work)).

### What the configuration to deploy measured

The serving profile's engine, 256 tokens a step without a per-request threshold, against each round's SLO, voice alone's p95 plus 250 ms on that card ([the rounds](../benchmarks/results/2026-10-28-rounds/README.md); [the soak](../benchmarks/results/2026-10-28-soak/README.md)). The agents' tokens a second are the rounds' median over the calls' first 240 s, over both cards where there are two, and the soak's the whole day's:

| Measured | Agents' sessions | Replies over the SLO | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|
| **Two cards, consolidation with a warm spare**, three rounds | 4 + 12 | 5 / 177 (2.8%) | 9,652 | 0.5 / 10.7 s |
| The same for 24 hours, calls arriving at 20 an hour | 14, at most 4 beside the calls | 74 / 5,058 (1.5%) | 3,529 | — |
| One card beside 24 sessions, five rounds on new samples | 24 | 44 / 295 (14.9%) | 3,067 | 85 / 120 s |

- **The soak's day.** Its worst hour put 5.5% of its replies over the SLO, no call failed and nothing restarted. The agents ran at 9,300 tokens a second in its first hour, where the replay resumes every session at once and the rounds' windows sit, and at 582–5,402 in each hour after, at their traces' own pace.
- **Beside 24 sessions**, past the card's envelope, 256-token steps put 44 of 295 replies over each round's SLO against 94 at 512 tokens a step with a 256-token threshold, at 7% less of the agents' token rate, and tuned vLLM put 261 over ([the step cap](BENCHMARKS.md#the-step-cap-screened-and-confirmed-sw-rg)).

**Where it ran.** The numbers on this page come from a lab deployment: shared hosts, whose GPUs other users' work can hold, the second card served in measurement windows, and synthetic callers only, since real callers cannot reach it. Opening it to them would take a domain and a CA's certificate, TLS at a proxy in front of LiveKit, TURN over TLS on port 443 and a UDP port range for the media, a service that issues the callers' tokens, and the loopback design of [ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md) revisited. The alerts are rules for an operator's Prometheus ([`ops/`](../ops)), and nothing pages anyone.

## What it buys

Two callers, 25 turns each, beside 24 coding-agent sessions whose contexts fill the GPU's KV pool. Voice-to-voice latency (V2V) runs from the end of the caller's speech to the agent's first audio, read from waveforms; the SLO is voice alone plus 250 ms at p95, 2,034 ms. Medians of the main evaluation's runs (W5), each round on its own sample of the agents' traces (a configuration more than one campaign ran is pooled over them), with the p95's range:

| Configuration | Runs | V2V p95 | Replies over the SLO | Agents' tokens/s, first 240 s | Agent TTFT p99 |
|---|---:|---:|---:|---:|---:|
| Voice alone | 5 | 1,784 ms (1,764–1,819) | 1 / 295 | — | — |
| Shared, tuned vLLM 0.32 (priority, step cap 512/256, a CPU KV tier) | 10 | 10,477 ms (8,686–12,115) | 532 / 590 | 3,305 | 106 s |
| Shared, preempt-resubmit in a router, the same engine | 5 | 2,522 ms (2,516–2,868) | 130 / 295 | 3,332 | 193 s |
| **Shared, this configuration at the box's 512/256** | 15 | **2,236 ms (2,132–2,397)** | 286 / 885 | 3,203 | 111 s |

These rows ran the box's step cap, 512 tokens a step with a 256-token threshold; the serving profile's 256-token steps are [above](#what-the-configuration-to-deploy-measured).

On stock vLLM the shared GPU leaves the voice p95 at over five times its SLO: a waiting voice turn cannot take KV from running agents ([vLLM #40004](https://github.com/vllm-project/vllm/issues/40004)), and once it runs, agent prefill chunks share every step of its reply.

With the configuration to deploy the tail comes within about 0.2 s of the SLO (202 ms at the median), the agents keep 97% of their token rate over the calls' first 240 s, and their TTFT p99 stays within the runs' spread. The SLO is still missed: it allows 5% of the replies over its threshold, and 32% of its replies are over it, against 90% on tuned vLLM. A deployment that must meet it admits less background work beside the calls: on the serving profile, four of this scenario's sessions beside two calls, the envelope that [Two cards](#two-cards) gives. Without the change, the alternative is a GPU per workload.

On an A100 SXM4 80 GB held to the box's pool, model and agents, the same configuration took the voice p95 from 4,986 to 1,634 ms, under that card's SLO of 1,669 ms in each of five rounds, and the agents' throughput held ([the A100 run](BENCHMARKS.md#what-it-measured-2026-10-26)).

The campaign before these rounds ran an earlier build of make-room, with three gaps since fixed, and the fixed build differed from it by +8 ms in V2V p95 at the median of five rounds ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351); [BENCHMARKS.md](BENCHMARKS.md#the-discriminating-scenario-after-w5s-fixes-w5)). In it, a gateway that also capped every reply at 16 tokens came nearer the SLO, at the cost of the replies ([BENCHMARKS.md](BENCHMARKS.md#the-discriminating-scenario-in-rounds-with-arm-9-and-the-change-w4)).

## Is your deployment in this regime?

The change matters when background KV fills the pool: agent contexts of 10k tokens or more against a pool that holds a few of them. On stock vLLM the signs are KV cache usage near 1 (`vllm:kv_cache_usage_perc`) while requests wait (`vllm:num_requests_waiting`), and a realtime time to first token of seconds at p95.

Where the pool has room, the change does less: at the standard scenario's realistic load (16 agent sessions of the whole corpus) the KV preemption has little to do (0–7 preemptions a run), and the budget cap still keeps agent prefill out of each reply's first tokens. Over five rounds, each on its own sample of the agents, the configuration to deploy had the lower p95 in every round, 2.02–2.42 s against tuned vLLM's 2.15–3.83 s, and met the SLO in one ([BENCHMARKS.md](BENCHMARKS.md#the-standard-scenario-in-five-rounds-w5)).

### Can the served model do the background work?

Sharing the card shares the model: the background work runs on the model the call needs. Check that it does that work well enough before sharing a card for it. On a useful pair of jobs, answering questions about private documents in a call (QASPER) beside enriching new documents into searchable records (Re-DocRED), the served Qwen3-4B-Instruct-2507-FP8 failed nine of the quality gate's eleven criteria, both tasks' central ones among them: 50.6% of the questions it answered were answered correctly, against 80%, and 23.9% of the facts in its records were right, against 80%.

The screen failed as well with an FP8 KV cache and with Qwen3-8B-FP8 on a card of its own ([the quality record](../benchmarks/results/2026-10-27-quality/README.md); [`benchmarks/quality`](../benchmarks/quality) runs it on another model or pair). The measurements on this page replay coding agents' traffic, which needs no answers to be right; they say what sharing costs the call, not whether the shared model's background answers are worth keeping.

## The configuration

### The engine

The serving profile's engine: the frozen box's [`engine.yaml`](../deploy/phastform/engine.yaml), with the step cap and the change that [`variants/make-room-serve`](../deploy/phastform/variants/make-room-serve/flatshare-engine.conf) passes over it:

| Flag | Value | Why | File |
|---|---|---|---|
| `--scheduling-policy` | `priority` | The class's priority orders the queue: voice at 0 runs before agents at 2 | `engine.yaml` |
| `--max-num-batched-tokens` | `256` | The step cap: it bounds how long a running decode waits behind a prefill chunk, and the sweep chose 256 for the change ([its screens](../benchmarks/results/2026-10-27-sweep/README.md)) | [`rtx-5060-ti-serve.toml`](../deploy/profiles/rtx-5060-ti-serve.toml) |
| `--long-prefill-token-threshold` | `0` | No per-request prefill cap: steps of at most 256 tokens | `rtx-5060-ti-serve.toml` |
| `--enable-prefix-caching` | on | Agents' requests share long prefixes | `engine.yaml` |
| `--kv-transfer-config` | A CPU KV tier written through: vLLM's `OffloadingConnector`, `offload_prompt_only: false`, 20 GiB, about four times the GPU pool | A preempted agent reloads its KV from host memory: 509 ms against 3.26 s recomputed for a 16k-token context ([the tier cost model](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0)) | `engine.yaml` |
| `--kv-cache-memory` | 5 GiB, a fixed pool of 36,400 tokens | When the audio models share the GPU, the pool depends neither on the profiler nor on the audio servers' memory at start | `engine.yaml` |
| `--max-priority-preemptions-per-step` | `2` | Make-room ([#63350](https://github.com/vllm-project/vllm/pull/63350)): a waiting request of a higher priority preempts running requests of a lower priority when it lacks KV, at most 2 a step | `rtx-5060-ti-serve.toml` |
| `--priority-prefill-budget-cap`, `--priority-prefill-budget-cap-tokens` | on, `48` | The budget cap ([#63351](https://github.com/vllm-project/vllm/pull/63351)): prefills of a lower priority sit out the steps of its first tokens ([below](#the-replys-first-tokens) for the 48) | `rtx-5060-ti-serve.toml` |

- **The step cap's trade.** The frozen box keeps its own, 512/256 in [`engine.yaml`](../deploy/phastform/engine.yaml), as the baseline every measurement changes, and the serving variant passes the profile's over it. A faster card's own cap may be larger, and with the change the cap sets one wait it cannot remove: a voice turn that arrives with steps in flight waits for them. On an A100 SXM4 80 GB, vLLM's default 2,048/0 gave the agents 1.4 times the tokens a second of 512/256, and with the configuration to deploy the voice p95 stayed 0.14–0.17 s over its SLO at all of that throughput, against under it at 512/256 ([the A100 run](BENCHMARKS.md#what-it-measured-2026-10-26)): pick the cap by the voice SLO's margin.
- **The change** is proposed to vLLM as pull requests [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351), open, with no maintainer review as of 2026-10-29.
- **No reserve.** The variant's third switch keeps KV free for priority 0; beside the KV preemption it showed no effect: with a 48-token window, 2,205 ms (2,098–2,289) with it and the lifecycle policy against 2,193 ms (2,168–2,286) for the configuration to deploy alone, three runs each on one sample of the agents' traces in the campaign before the rounds ([BENCHMARKS.md](BENCHMARKS.md#the-discriminating-scenario-in-rounds-with-arm-9-and-the-change-w4)), which leave a difference of up to about 0.2 s either way unresolved. The pull requests leave it out.

On vLLM 0.32.0, the [`make-room` variant](../deploy/phastform/variants/README.md) generates the same rules from the installed vLLM, without modifying it (`python python/flatshare_vllm/src/flatshare_vllm/build.py make-room OUT_DIR`), and the engine loads it with `PYTHONPATH=OUT_DIR` and `--scheduler-cls flatshare_make_room_scheduler.AsyncScheduler`. Its switches are keys of `--additional-config`: `make_room.max_preemptions_per_step 2`, `make_room.budget_cap true`, `make_room.budget_cap_tokens 48` and `make_room.reserve_tokens 0`.

The repository's serving variant, [`variants/make-room-serve`](../deploy/phastform/variants/make-room-serve/flatshare-engine.conf), runs exactly that, rendered from the serving profile [`deploy/profiles/rtx-5060-ti-serve.toml`](../deploy/profiles/rtx-5060-ti-serve.toml), whose [`serving.json`](../deploy/phastform/serving.json) holds every setting of this section with the revisions it needs: the vLLM version, the model's and the tokenizer's revisions, the SHA-256 of the scheduler module and of the patches it is built from. Each start of the engine checks the configuration the engine reports from inside against it and fails on any difference ([the serving profile](../deploy/phastform/variants/README.md#the-serving-profile)); `variants/make-room` keeps the engine configuration the lifecycle policy ran over (arm 9), the reserve and a 16-token window, for the experiments. The variants refuse any other vLLM version, since their patches match 0.32.0's source, and a bad `make_room` key or value fails the engine start, naming it.

The evaluation measured the configuration to deploy as this variant with these switch values; ported to 0.32.0, the pull requests' diff schedules identically to it ([GLOSSARY.md](GLOSSARY.md#the-change-and-its-parts)). Since #63351's revision of 2026-10-25, the PR and the variant both count a request's first tokens in flight under asynchronous scheduling, which the frozen box runs; the main evaluation's measurements (W5) ran the earlier count, which held a step past the window ([after the main evaluation](BENCHMARKS.md#after-w5-the-first-tokens-in-flight)).

### The priorities

Voice requests go at priority 0 and background work at 2; in vLLM a lower value is served first. A client can send `priority` in the request body or the `X-Vllm-Priority` header. A gateway in front should set it from the caller's identity instead, so that a background client cannot raise its own; `flatshare-gateway` maps each tenant to a class and caps it with `max_class`. Nothing else in front is needed: the configuration to deploy ran without an admission policy.

`flatshare-gateway` is the deployment's serving control plane ([ADR-0007](adr/0007-the-gateway-is-the-serving-control-plane.md)): what it does not do, such as TLS or a bound on idle connections, is listed in [its README](../crates/flatshare-gateway/README.md#what-it-does-not-do), and the engine must be reachable only through it. vLLM's `--api-key` guards only its `/v1`, `/v2`, `/inference` and `/cohere` routes, so serve the engine on a Unix socket in a directory only its user may enter (`--uds`, as [`deploy/phastform/systemd`](../deploy/phastform/systemd) does) and reach it from the gateway on that socket, or through a tunnel that may reach nothing else ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md#unix-sockets-end-to-end-w6-2026-10-25)).

### An existing LiveKit or Pipecat agent

The voice worker here is a LiveKit Agents worker ([`flatshare_voice`](../python/flatshare_voice/README.md)). Another voice agent keeps its own pipeline and points its LLM client at the gateway, under the same contract:

| | What to set |
|---|---|
| The endpoint | The gateway's OpenAI-compatible routes (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`): on cyan, `http://127.0.0.1:18080/v1` with a tenant's key, or the voice plane's Unix socket, whose tenant needs none ([`gateway.example.toml`](../deploy/cyan/gateway.example.toml)) |
| The class | The call's requests in `realtime`: a tenant of that `default_class`, as `[tenants.voice]`, or `X-Flatshare-Class: realtime` from a trusted tenant whose `max_class` allows it |
| The session | One `X-Session-ID` a call, from a trusted tenant ([PROTOCOL.md](PROTOCOL.md#sessions)) |
| The priorities | The gateway sends each class's, realtime 0 and background 2, and removes a client's own `priority` or `X-Vllm-Priority` ([PROTOCOL.md](PROTOCOL.md#headers)) |
| The agents | A tenant of class `background`, as `[tenants.agents]` |
| The window | `--priority-prefill-budget-cap-tokens` covers the agent's first sentence at its longest ([The reply's first tokens](#the-replys-first-tokens)) |
| LiveKit Agents | `openai.LLM(model=..., base_url=..., api_key=..., extra_headers={...})`, the headers as [`plugins.py`](../python/flatshare_voice/src/flatshare_voice/plugins.py)'s `make_llm` sets them |
| Pipecat | `OpenAILLMService(base_url=..., api_key=..., default_headers={...})`; no Pipecat agent ran here |

With two cards a call's speech servers are its node's: the voice worker places each call with `POST /v1/sessions` and binds its STT and TTS to the node that answers ([Nodes](../crates/flatshare-gateway/README.md#nodes)). Only this repository's worker has been measured.

### The reply's first tokens

The budget cap protects a reply for its first T tokens, its window. The caller hears nothing until the reply's first complete sentence reaches TTS (the voice agent hands TTS whole sentences, a sentence once it reaches 20 characters, LiveKit's default), so T must cover the first sentence, not the first token. A reply whose first sentence is longer decodes the rest of it beside agent prefill again: its tokens come 60–65 ms apart instead of 16–20 ms ([the prefill case](BENCHMARKS.md#the-prs-configuration-in-one-session-w5)).

Choose T to cover your agent's first sentence at its longest; this voice agent is prompted for "one or two short, natural sentences" ([`worker.py`](../python/flatshare_voice/src/flatshare_voice/worker.py)), and an agent prompted for longer answers needs a longer window. This voice agent's replies run 23 tokens at the median and 52 at most (the main evaluation's 5,571 replies):
- at T = 16 with full replies the p95 is 3,094 ms (2,868–3,192);
- at T = 48 it is 2,236 ms (2,132–2,397, fifteen runs);
- at T = 16 with every reply capped at 16 tokens (`max_tokens`) it came nearer still in the campaign before, at the price of truncated replies ([BENCHMARKS.md](BENCHMARKS.md#the-discriminating-scenario-in-rounds-with-arm-9-and-the-change-w4)). The realtime class's reply cap in the gateway (`[classes.realtime] max_reply_tokens`) sets such a cap on every realtime reply by itself, with or without an admission policy.

A longer window holds background prefill for longer each turn: the agents' throughput is 2% lower at 48 than at 16 (3,236 against 3,298 tokens/s over the calls' first 240 s, in the main evaluation's second set of rounds, which ran both windows).

A sweep of the window against whole replies of 40 tokens, beside the box's closed-loop agents, places the knee ([`knee.sh`'s records](../benchmarks/results/2026-10-25-window-knee/README.md)): the reply's 40th token came 3,759 ms after the request at p95 with a 16-token window, 2,642 at 24, 1,854 at 32 and 1,037 at 48, and no sooner at 64 (1,018 ms), while the first token stayed at 326–448 ms. Those agents' prompt throughput, all of it prefill, fell 8% from 16 to 48 and no further at 64: a hold ends when its reply ends, so a window longer than the replies costs no more than one just long enough. When in doubt, choose T on the long side of your agent's first sentence.

### Calls: what the box admits

The voice worker runs in one of two modes ([`worker.py`](../python/flatshare_voice/src/flatshare_voice/worker.py)):

| Mode | What it admits |
|---|---|
| Bench mode, what every measurement ran | Every call LiveKit dispatches |
| Serve mode, the serving configuration ([`deploy/cyan/variants/serve`](../deploy/cyan/variants/serve/flatshare-voice-worker.conf)) | At most the serving profile's calls at once, two |

- **Why two.** Two are the calls at which the sweep and the two-card rounds measured the profile, its envelope of agents beside them ([Two cards](#two-cards)); at the main evaluation's step cap, 512 tokens, 4 and 6 callers were further over each one's SLO than two ([calls per GPU](BENCHMARKS.md#calls-per-gpu-w5)).
- **A full worker.** Its load, which LiveKit's agents SDK reads every half second, is its calls over that number, with LiveKit's load threshold at 1, so a full worker is marked full and the server sends it no job; a job request beyond it is refused, counted exactly when requests arrive together.
- **A line that frees.** The worker sends LiveKit its status as soon as a call's process closes, besides the SDK's rounds every 2.5 s (LiveKit Agents 1.8.8's `UPDATE_STATUS_INTERVAL`): LiveKit dispatches a room once and to no worker it holds full, and at the SDK's pace alone it held a full worker full for up to 3 s after a call ended, which turned 2 of the 24-hour soak's 476 calls away with a line free ([the record](../benchmarks/results/2026-10-28-soak/README.md)). With calls arriving faster than two lines take them, a free line reached LiveKit 44 ms after the call's process exited, against 1.5 s before (medians), and no call was turned away with a line free, against 10 of 233 ([the runs](../benchmarks/results/2026-10-29-status-send/README.md)).
- **An engine that does not serve.** It takes no new call while the gateway's `/ready` says the engine does not serve (warming, degraded or drained, or 503 while the engine's `/health` fails; a gateway without the route leaves capacity alone to admit), nor while a release drains it.
- **A refused call** gets no agent: a client should take an agent that has not joined within a few seconds for a busy line. Calls in progress are never cut by admission.

### Real callers' audio

The benchmark's callers are synthetic: files played into the room, with no echo and no room noise. Serve mode's audio is the profile's other configuration, `microphone`, for real callers, which no benchmark measured:
- LiveKit's end-of-turn model ends a turn, since people pause inside a sentence (the experiments' fixed endpointing is refused with it);
- the VAD needs a clearer and longer onset to count speech (an activation of 0.6 against 0.5, 0.1 s against 0.05) and an interruption 0.6 s of speech, so that a click or a door does not cut the agent off, and an interruption that no transcript follows within 2 s lets the agent resume;
- LiveKit's 3 s AEC warm-up stays on, so that a loudspeaker's echo of the agent does not cut its first reply;
- noise suppression is the caller's client's (a browser's WebRTC applies its own), since LiveKit's noise cancellation needs LiveKit Cloud.

Measure it with your own callers before relying on it. With fixed endpointing, as the measurements run it, a turn waits out the VAD's silence (0.55 s) and about 0.2 s more, and LiveKit's 0.5 s endpointing delay does not bind: 0.3 s of silence answered 0.27 s sooner at the median, and took the pause after a first sentence for the end of one line in 40 ([the endpointing sweep](../benchmarks/results/2026-10-28-endpointing/README.md)).

### What the voice path keeps

| Data | Where | Kept until |
|---|---|---|
| Callers' audio | Memory only: LiveKit records nothing (it has no egress), and the STT and TTS servers keep no audio | The call's end |
| Transcripts and the agent's replies | The voice worker's memory, which a failover's re-synthesis needs for the sentence it cut | The call's end |
| The engine's KV | GPU memory, and the CPU tier's mapping in `/dev/shm` (mode 0600, the deploying user's) | Eviction, or the engine's stop |
| Metadata: session ids, room and participant names, timings, token counts, outcomes | The hosts' user journals: the gateway's request lines, the worker's turn lines, LiveKit's room events | As long as the hosts' journald keeps them |

The logs keep no transcript or reply at the levels the units run: the gateway's request line holds ids, timings, token counts and the outcome; the engines log no request (vLLM's `--enable-log-requests` is off); LiveKit Agents logs a transcript only at DEBUG, below the worker's INFO; and the TTS server's unit sets `API_LOG_LEVEL=WARNING`, since at INFO Kokoro-FastAPI logs the first 50 characters of every chunk it synthesizes. Nothing in the repository sets the journals' retention: an operator with real callers sets it, for example with `SystemMaxUse=` in journald.conf or `journalctl --vacuum-time=`, and decides consent and retention.

### The audio models

- Whisper and Kokoro share the GPU by time-slicing; MPS does not help ([Voice on the shared GPU](BENCHMARKS.md#voice-on-the-shared-gpu)): the interference that matters is inside the LLM engine. Sharing still costs them a little: in the shared configurations of [What it buys](#what-it-buys), endpointing takes 822–825 ms instead of 756, and TTS's first audio 232–250 ms instead of 211 (means over the main evaluation's replies, from [its report](../benchmarks/results/2026-10-24-e2-w5/report/frontier.md#stage-waterfall)).
- Pin each server to its own CPU cores: the engine on the GPU's NUMA node, the audio servers off it, as the box's units do ([`deploy/phastform/systemd`](../deploy/phastform/systemd)).
- Keep them on the GPU. On the CPU they would give their 3.4 GiB to the KV pool, but Kokoro's first audio took 991 ms at the median with one call and 2,736 ms with two, against 211–242 ms for the GPU's whole TTS stage, and a Whisper transcript 9 s or more, on a 1.8 GHz Xeon faster than the box's ([the CPU record](../benchmarks/results/2026-10-27-cpu-audio/README.md)).

## Two cards

With two cards, put both calls on one and fill the other with agents. A resource gate compared the arrangements of two RTX 5060 Ti cards in three rounds, each on a new sample of the agents and on the two hosts in turn ([RESOURCE-GATE.md](../benchmarks/quality/RESOURCE-GATE.md); [the rounds' record](../benchmarks/results/2026-10-28-rounds/README.md); [the report, §6.8](REPORT.md#68-on-two-cards-consolidating-both-calls-on-one-does-34-more-background-work-most-of-it-without-the-change)). An arrangement qualifies when at most 5% of its replies are over the SLO, voice alone's p95 plus 250 ms on that card:

| Arrangement | Agents' sessions | Replies over the SLO | Agents' tokens/s, first 240 s | Against dedicated roles |
|---|---:|---:|---:|---:|
| Dedicated roles: the calls on one card, agents alone on the other | 0 + 14 | 0 / 177 | 7,893 | — |
| **Consolidation**: both calls and at most four sessions on one card, with the serving profile, beside the agents' card | 4 + 14 | 5 / 177 (2.8%) | 10,596 | 34% more |
| Consolidation on tuned vLLM | 2 + 14 | 4 / 177 | 10,051 | 27% more |
| **Consolidation with a warm spare** | 4 + 12 | 5 / 177 | 9,652 | 22% more |
| Pooling: one call and 12 sessions a card | 12 + 12 | 24 / 87 with the configuration to deploy, 53 / 87 on tuned vLLM | 12,111 with the configuration to deploy, 13,188 on tuned vLLM | does not qualify |

- **Dedicated roles** qualify, and their agents' card does the most at 14 sessions: at 24 its pool thrashes and it does 4,232.
- **Consolidation** at eight sessions put 31 of its 177 replies over. On stock vLLM the shared card qualifies at two sessions only: the second card's agents dominate both sums, and the change's own share is the two sessions more it carries.
- **A warm spare.** In the other arrangements the agents' card runs no speech servers and cannot take the calls. With them resident and the agents on the serving profile's engine, the second card did the most at 12 sessions, and the arrangement keeps the standby that dedicated roles lack.

<picture>
  <source media="(max-width: 767px)" srcset="figures/two-cards-narrow.svg">
  <img src="figures/two-cards.svg" width="100%" alt="Two panels of the two cards' slots as the gateway bounds them, two call slots each and up to 4 agents' sessions on Phastform and 12 on Gray: in panel a, in service, both calls fill Phastform and Gray, the warm spare, runs agents only; in panel b, Phastform's engine failed, both calls take Gray's call slots at their next turn, 0.7 s after a kill and 1.9 s after a stop in the drills, and Gray holds its agents' requests until the moved calls end.">
</picture>

**Figure 1. Both calls fill the first card, and the spare takes them by their next turn when the first fails.** (a) Each card has two call slots (`max_calls = 2`) and up to 4 or 12 agents' sessions (`max_agents`), and `calls = "fill"` places each call on the first card with a free call slot. (b) In the failover drills, both calls' next turns reached the spare 0.7 s after a SIGKILL to the first card's engine and 1.9 s after its stop ([the record](../benchmarks/results/2026-10-28-node-failover/README.md)); the spare holds its agents' requests, hatched, until the moved calls end.

### The envelope at a looser margin

The envelope is the most agents' sessions beside both calls at which a card qualifies, three rounds pooled, and the SLO's margin over voice alone sets it ([the rounds' margins](../benchmarks/results/2026-10-28-rounds/README.md#beside-the-gate-the-margin)):

| Margin over voice alone's p95 | The configuration to deploy, 256-token steps | Tuned vLLM |
|---|---|---|
| 250 ms, the SLO | 4 sessions: 5 of 177 replies over | 2 sessions: 4 of 177 over |
| 500 ms | 24 sessions, every count measured: 8 of 177 over at 24 | 2 sessions: 1 of 177 over, and 12 of 177 at four |

At 500 ms the card qualifies with the configuration to deploy at every count measured, up to 24 sessions, and with tuned vLLM still only at two.

### Setting it up

In `gateway.toml` ([`deploy/cyan/gateway.example.toml`](../deploy/cyan/gateway.example.toml) has it in its `[[nodes]]` blocks):

| Key | Value | Why |
|---|---|---|
| `[[nodes]]` | an entry per card, both with the realtime and background roles and `max_calls = 2` | Each card can carry both calls, so that the spare can take them |
| `max_agents` | 4 on the first, 12 on the spare | The envelope beside both calls, and the spare's busiest count |
| `[placement] calls` | `"fill"` | Each call goes on the first node with a free call slot in the file's order |

A node with both roles must say how many agents' sessions it takes beside its calls, a count or `"unbounded"`: the gateway refuses to start without it.

Each release on cyan holds the running gateway's nodes to the serving profile's envelope (`serving-check --nodes`): every node has `max_calls = 2`, the worker's calls, the first at most four agents' sessions beside them, and a covering node at most four agents' requests in flight; a release that finds otherwise fails and switches back ([Releasing a change](#releasing-a-change)). The runbook's [Two GPU nodes](../ops/RUNBOOK.md#two-gpu-nodes) says what each key does, and how the nodes' health, placement and failover work.

## What not to deploy

Measured in the same scenario, beside the 24 sessions; the p95s are medians of each configuration's runs:

| Don't deploy | Measured | Why not |
|---|---|---|
| Lifecycle-timed protection in a gateway: KV headroom held from speech onset, a prefill throttle while a voice turn is answered | 10,993 ms without the change, as stock | With agents' KV filling the pool these could not act; a gateway can make room only by aborting admitted work, at the agents' tail ([why](REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work)) |
| vLLM's development mode (`VLLM_SERVER_DEV_MODE=1`) | — | It serves routes such as `/collective_rpc`, `/sleep` and `/update_weights`, which `--api-key` does not guard ([the routes](../deploy/phastform/variants/README.md#the-measurement-drop-in)) |
| KV cache events on TCP | — | They carry every cached block's token ids and session id, and vLLM sends them to whoever binds the endpoint. The frozen box publishes none |
| `--watermark` | no change | — |
| `--max-num-queued-tokens` | 48 of 50 voice turns failed | It refuses voice turns too |
| Concurrency caps | — | Two agent contexts fill the pool |
| [#54530](https://github.com/vllm-project/vllm/pull/54530) | — | It preempts at the request-count cap, which does not bind here |
| SGLang 0.5.22's priority preemption, with or without its host tier | 11,948 ms with HiCache, five runs | — |
| A static KV reservation | 5,671 ms | It halves the tail but stays seconds over the SLO |
| #63350 alone | 5,085 ms | The KV preemption without the budget cap gets the turn into the batch, and its reply then decodes beside agent prefill |
| An FP8 KV cache beside the CPU tier (`--kv-cache-dtype fp8`), on vLLM 0.32.0 on this card | Twice the tokens in the same memory (72,816 against 36,400), and no quality lost in the gate's screen | Beside two calls and 24 agent sessions the engine died both times it ran, on a CUDA fault (note 1) |

1. The fault came at a step's output copy, with a CPU-tier transfer in flight ([the sweep's record](../benchmarks/results/2026-10-27-sweep/README.md#the-fp8-kv-cache)).

- **The lifecycle policy's timing** came from the voice agent's turn events, which serve mode does not send.
- **Development mode** runs only on a measured box, from the measurement drop-in (`variant.sh measure on`), for the harness's `/server_info` and cache resets.

#63350 alone ran in the rounds of [What it buys](#what-it-buys); the other p95s come from the campaign before them, on one sample of the agents' traces ([BENCHMARKS.md](BENCHMARKS.md#the-discriminating-scenario-in-rounds-with-arm-9-and-the-change-w4)).

Without any engine change, aborting background requests that have no first byte yet and resubmitting them (`flatshare-gateway`'s `preempt-resubmit` policy) holds 2,522 ms (2,516–2,868) at tuned vLLM's throughput (3,332 against 3,305 agents' tokens/s over the calls' first 240 s), at the cost of the agents' tail (TTFT p99 193 s against 106 s): of the configurations measured on stock vLLM, the nearest to the SLO. It cannot reach a request that is already decoding, which the change can.

## Releasing a change

`deploy/deploy.sh HOST` releases a commit to a host as an immutable release, `~/flatshare/releases/ID`, with a manifest of what it holds (every file's SHA-256, the lock files, the Python environments by key, the gateway's binary and where it was built, the models' and tokenizer's revisions, the scheduler's patches and module, the serving profile); the environments are built once per lock and linked by every release that has the same.

It verifies the release before it switches (the files, each environment's imports, the binaries' versions, the scheduler module that the patches build on the installed vLLM, the drop-ins and models the box uses), drains what the switch would cut (the voice worker's calls in serve mode, the engine's requests), switches one link, restarts or reloads only the units whose inputs changed, checks them (the engine's configuration against the serving profile among them, and on cyan the gateway's nodes against its envelope), and switches back by itself on any failure. `deploy.sh HOST --rollback` returns to the previous release by the same steps ([the runbook](../ops/RUNBOOK.md#deploy)).

## What to watch

- **Voice**: V2V p95 per call from the voice stack, and the realtime time to first token at p95 (the gateway's request log, or vLLM's per-request metrics). The voice agent's per-turn stage times split a slow turn into endpointing, the LLM's first sentence and TTS ([the stages](BENCHMARKS.md#voice-under-agent-load-e2)).
- **The engine**: KV usage, running and waiting requests, and preemptions; `vllm:num_preemptions_total` counts a victim when its next output reaches the frontend, so it lags. The CPU tier's loads beyond the requests' first scheduling show what preempted requests restored; what they recompute, no vLLM 0.32 counter shows, since it counts a request's computed prompt tokens at its first scheduling only ([what the agents pay](BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)).
- **Background**: completed tokens a second and the agents' TTFT p99, where a preemption policy moves the cost.
- [`ops/`](../ops) has these as code: the Prometheus scrape configuration (the gateway, and the engine through it), the burn-rate alerts of the two production objectives, the voice requests' first byte (95% within 250 ms) and the gateway's overhead ([SLO.md](SLO.md#the-production-objectives)), health alerts, and a Grafana dashboard ([the runbook](../ops/RUNBOOK.md#monitoring)).

<picture>
  <source media="(max-width: 767px)" srcset="figures/report-stages-narrow.svg">
  <img src="figures/report-stages.svg" width="100%" alt="The voice turn's mean voice-to-voice time stacked by stage from the end of the caller's speech, beside 24 agent sessions but for voice alone: voice alone, 1.62 s, 0.09 s of it waiting for the LLM's first token and 0.22 s decoding the first sentence; tuned vLLM, 5.30 s, 2.81 s of it waiting for the LLM's first token and 1.07 s decoding the first sentence; the KV preemption, 3.41 s, 0.78 s of it waiting for the LLM's first token and 1.21 s decoding the first sentence; the configuration to deploy, 1.97 s, 0.29 s of it waiting for the LLM's first token and 0.26 s decoding the first sentence.">
</picture>

**Figure 2. A slow turn shows in two stages, the wait for the LLM's first token and the decode of its first sentence.** Mean time of each stage of a voice turn from the end of the caller's speech, over every reply of each configuration's runs, with two callers beside 24 coding-agent sessions on the RTX 5060 Ti ([the report's Figure 1](REPORT.md#1-introduction)). Violet is the wait for the LLM's first token and green the decode of the reply's first sentence, and the number after each bar is the mean V2V; endpointing, speech synthesis and transport change little between rows.

## How it fails

| Failure | What it does | What to do or expect |
|---|---|---|
| A voice turn that two preemptions cannot make room for | It waits as on stock: the change preempts at most `N` requests a step, and none if `N` would not do | — |
| Only realtime work | Nothing to preempt: the engine schedules it by priority and arrival, as stock | — |
| Many calls at once | The budget cap holds lower-priority prefill while any realtime request's first tokens are due, so replies that overlap hold it continuously | Watch the engine's log of what it held ([below](#many-calls-at-once)) |
| A KV load that never completes | It holds lower-priority prefill for its first `budget_cap_tokens` steps, then nothing | — |
| The engine dies (an out-of-memory kill, a crash) | Every stream it serves breaks at once, and voice is down for all of it: one GPU node has no standby | systemd restarts it: a request succeeded 106.2 s after a kill ([below](#the-engine-dies)) |
| The first of two cards fails | Its calls move to the spare by their next turn; the spare's agents wait while it carries any call | The moved calls stay on the spare until they end ([below](#the-first-of-two-cards-fails)) |
| The gateway restarts | A reload refuses nothing and makes no request wait; background streams are cut after 5 s | Clients may send the cut background requests again ([below](#the-gateway-restarts)) |
| Host memory pressure | A memory-pressure killer such as `systemd-oomd` picks the engine first under a host-wide stall | Keep swap and memory pressure low; `ManagedOOMPreference=avoid` on the engine's unit |
| Another process on the GPU | It slows every stage | Keep the GPU exclusive, as `gpu-guard` does for the box's units |
| A reserve too large to start | The engine refuses to start if the reserve and one `max_model_len` request do not fit in the pool together | Only with `make_room.reserve_tokens` above 0 |

### Many calls at once

The engine logs, at most every 10 s, the steps it held, the prefill tokens and admissions it held back, and its preemptions to make room: watch them.
- **The limit.** On vLLM's real scheduler, on CPU, beside six agents' prefill, replies at least the 48-token window long left the agents 82% of stock's prompt tokens at one every 200 steps, 9% at one every 50 and none at one every 40, where each reply's window (about 49 steps) overlaps the next ([the test](../python/flatshare_vllm/tests/test_make_room.py)). At the 16–20 ms a step the box measures under the cap, that is about one realtime reply a second in aggregate, some six calls turning every 6 s. A 16-token window kept 91% at one every 40 steps.
- **Up to six calls, measured.** The evaluation measured up to six calls with this voice agent, whose replies are nearly all shorter than the window. With 4 and 6 callers the p95 of the configuration to deploy stayed 269 and 257 ms over each one's SLO at the median, against 202 ms with two; the cap held agent prefill back in 53% and 64% of the engine's steps, against a third, and the agents' throughput over the first 240 s fell to 3,055 and 2,716 tokens/s, from 3,203 ([calls per GPU](BENCHMARKS.md#calls-per-gpu-w5)).
- **A datacenter batch.** On an A100, with 256 requests in flight and 26 of them at priority 0 in closed loop, the budget cap took the priority 2 requests' TTFT p99 from 1.1 s to 8.0 s while their median fell ([the A100 run](BENCHMARKS.md#the-a100-run-2026-10-26)).

### The engine dies

The gateway sends each event stream vLLM's error event, of type `upstream_unavailable`, before it aborts it, so a client that reads events learns why; the next request gets one 502, and the ones after it 503 `upstream_not_ready` with `Retry-After: 1` without reaching for the engine, until its `/health` answers. In serve mode the voice worker takes no new call meanwhile, as the gateway's `/ready` says.
- **The restart.** systemd starts the unit RestartSec after its last process exits, so the delay is pure wait, and the units' `StartLimitBurst=3` in 30 minutes ends a crash loop whatever it is: the box's GPU servers wait 5 s.
- **The drill** (SIGKILL to the engine's unit): each stream got the error event 8–11 ms after the kill; the main process was gone 0.13 s after it and the unit's last process 7.6 s after it (the GPU's teardown). The warm start listened 92.1 s after the restart, and the first request succeeded 106.2 s after the kill; on cold caches (a start after an upgrade) expect about nine minutes ([the drills](BENCHMARKS.md#failure-drills)).

### The first of two cards fails

When the first card's engine or speech servers fail, its calls move to the spare by their next turn, and while the spare carries any call its agents' new requests wait for their class's queue timeout (120 s) and are then answered 503 `queue_timeout` (those already running finish), and the first card's agents' sessions find the spare at its bound and get 503 `no_agent_slot` until a slot frees.

In the drills ([the record](../benchmarks/results/2026-10-28-node-failover/README.md)) both calls' next turns reached the spare 0.7 s after a SIGKILL to the first card's engine and 1.9 s after its stop, and none of the 300 turns after failed; the killed engine was back 101 s after the kill. Once the first card is healthy again it takes the new calls, the calls moved to the spare stay there until they end, and the spare's agents run again after them; there is no graceful drain of one card.

### The gateway restarts

A deploy takes a new gateway with `systemctl --user reload flatshare-gateway`: the gateway hands its listening sockets to a successor started from the new binary and configuration, which serves at once, and drains. No request is refused or waits; realtime and interactive requests in flight finish in the old process, and background ones are cut after `background_drain_s` (5 s) with a `draining` error event or a 503 with `Retry-After`, which their clients may send again. While a reload's predecessor drains, two admission points serve one engine, neither seeing the other's requests; the configuration to deploy has none.
- **A restart or a crash.** `flatshare-gateway.socket` holds the sockets across both: a plain restart refuses nothing but answers the connections made during its drain only after it, and after a crash systemd starts the gateway again in about a second, the connections waiting meanwhile.
- **The drills.** A reload failed none of 42 requests, the slowest answered in 0.13 s, with its voice turns whole; a restart failed none either, the requests sent during its drain answered once it ended, 20.7 s on; and a kill none, the slowest answered in 1.3 s ([the drills](BENCHMARKS.md#failure-drills)).

### Host memory pressure

The CPU tier lives in shared memory charged to the engine's cgroup, so a memory-pressure killer such as `systemd-oomd` picks the engine first under a host-wide stall ([the runbook](../ops/RUNBOOK.md#failures)). Keep the host's swap and memory pressure low; where `systemd-oomd` acts for the engine's own manager, `ManagedOOMPreference=avoid` on the engine's unit makes it a last resort.

## Checking it on your hardware

On another host, [HARDWARE.md](HARDWARE.md) renders the node from a hardware profile and runs the A100 run's kit, which is not a deployment: nothing of it starts at boot, and releases, rollback and the release's serving check reach only this lab's hosts, phastform, gray and cyan.

- `python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py ENGINE kv` sends a voice-sized request at priority 0 while long priority-2 requests hold the pool, and `probe.py ENGINE prefill` sends it behind three long prefills; each sizes its requests from the engine's pool and context, takes about a minute and prints the turn's first token and its 16th, with the engine's state before and after (`--reply 40`: a whole 40-token reply, timed to its last token too). ENGINE is the engine's address, `unix:` and its socket (the GPU node's, by default) or `http://host:port`.
- `tail.py ENGINE SECONDS [AGENTS] [fresh|grow] [--reply TOKENS]` runs closed-loop agents beside two calls, by default as many as keep the frozen box's KV pressure on your pool (12 on the box's), and prints the calls' TTFT distribution and the agents' throughput; `--reply` times whole replies to their 16th and last token.
- The whole measurement: an arm file of the end-to-end harness with your traces and `flatshare-bench e2 run` ([benchmarks/e2](../benchmarks/e2/README.md)).

The change itself has two checks a checkout runs on CPU ([python/flatshare_vllm](../python/flatshare_vllm/README.md#upkeep-the-changes-cost-and-its-fit-upstream) says how to read them). `make perf-make-room TORCH_PYTHON=<CPU torch 2.13 with vLLM 0.32.0's dependencies>` times what the change costs a scheduler step with its switches off, against stock vLLM's step, and in the configuration to deploy, against the step with them off, in the 13 states of the pull requests' cost tables and four of the serving profile, the step that preempts among them, against the baseline measured on peacock, a 2.1 GHz Xeon. Another CPU weights the change's code differently (on an M5 Max, on/off came 2.5% and 3.3% above the Xeon's in B16 and C16 and 5.0% and 5.6% below it in hold-R2 and fire-R2, against its 7% tolerance), so measure your machine's own baseline first (`make perf-make-room UPDATE=1 PERF_BASELINE=build/perf-baseline.json PERF_BESIDE=<what else ran>`), then check against it (`PERF_BASELINE=build/perf-baseline.json`) after changing the engine's code.

Before moving off vLLM 0.32.0, `make compat-upstream` says whether #63350 and #63351 still merge into vLLM's main (with `TORCH_PYTHON`, an interpreter for main, whether their tests pass there), whether make-room's backport is still the pull requests' heads but for the adaptations its patches' headers name, and how make-room's patches fit the newest release: they apply to 0.32.0 alone, by design, and where they do not, it names the first hunk that build.py refuses and gives each hunk's fit as `git apply` finds it with three lines of context and with one, so that a rebase starts from the hunks rejected at `-C3`.
