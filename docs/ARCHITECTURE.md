# Architecture

Leaseholder runs a realtime voice agent beside background LLM work on one GPU. The part that makes this work is a change to vLLM's scheduler: a waiting realtime request can take KV from running background requests, and background prefill sits out the realtime reply's first tokens. It is proposed upstream as [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351) and runs on vLLM 0.32 as the engine variant `make-room` (§2). Around it are the deployment it was measured on (§3), the harness that measured it against everything an operator can deploy today (§4) and a KV-cache simulator (§5). Two mechanisms were built, measured and not adopted (§6), and the rest of the original design was dropped (§7).

This document describes what is built, as of v0.3 (2026-10-24). [PRODUCTION.md](PRODUCTION.md) gives the configuration to deploy, [BENCHMARKS.md](BENCHMARKS.md) the numbers, and [PRIOR-ART.md](PRIOR-ART.md) what is ours and what is not.

## 1. The problem in the engine

A voice turn on a GPU shared with coding agents loses its time in the LLM's first sentence. On stock vLLM 0.32, tuned (priority scheduling, a 512-token step cap and a write-through CPU KV tier), with 24 agent sessions of long contexts beside two calls, the reply's first sentence takes 3.9 s of a 5.3 s mean turn in W5, against 0.3 s with voice alone ([W5's stage means](../benchmarks/results/2026-10-24-e2-w5/report/frontier.md#stage-waterfall)). Two mechanisms in the scheduler cause it:
- **A waiting request cannot take KV.** In vLLM 0.32's V1 scheduler, the running loop preempts only so that a running request can grow, and the waiting loop stops at the first request it cannot allocate, so nothing preempts for a waiting request ([#40004](https://github.com/vllm-project/vllm/issues/40004)). vLLM's V0 scheduler did, from v0.6.2 until V0 was removed in v0.11.0, but only with chunked prefill off, and it recomputed its victims ([#5958](https://github.com/vllm-project/vllm/pull/5958); [PRIOR-ART.md](PRIOR-ART.md#vllm-0320-2026-10-18)). When running agents hold the KV pool, a voice turn waits for one of them to finish: 4.9 s to the first token of a 746-token turn behind two 18k-token agent requests ([the KV case](BENCHMARKS.md#the-kv-case-make-room)).
- **Prefill shares the reply's steps.** Once the turn runs, every step of its reply also carries the agents' prefill chunks, up to about 270 ms each after a 30k-token context, so the first sentence decodes slowly.

Neither mechanism is specific to the card, though what each costs is: the first appears wherever agents' contexts are long against the pool, which the card's memory sets, the second wherever background prefill runs beside a realtime decode ([METHODS.md](METHODS.md#what-transfers-to-other-hardware) places other GPUs by a model).

## 2. The change: #63350 and #63351

The variant `make-room` ([`python/leaseholder_vllm`](../python/leaseholder_vllm/README.md)) is built at each engine start from the installed vLLM 0.32.0's scheduler, without modifying vLLM: `build.py` applies the PRs' scheduler diff, backported to 0.32.0, and the variant's own patch, every hunk where it must be, and the engine loads the module with `--scheduler-cls`. It adds three switches to priority scheduling, each a key of `--additional-config` and off by default:

| Switch | Variant key | Upstream |
|---|---|---|
| KV make-room | `make_room.max_preemptions_per_step` | `--max-priority-preemptions-per-step` ([#63350](https://github.com/vllm-project/vllm/pull/63350)) |
| The budget cap | `make_room.budget_cap`, `make_room.budget_cap_tokens` | `--priority-prefill-budget-cap`, `--priority-prefill-budget-cap-tokens` ([#63351](https://github.com/vllm-project/vllm/pull/63351)) |
| The reserve | `make_room.reserve_tokens` | none: carried in the variant only |

- **KV make-room.** When the highest-priority waiting request the waiting loop can schedule in the step (not one loading its KV, compiling a grammar, paused for streaming input or awaiting stale output, nor one whose LoRA adapter could not join the running requests') lacks KV for its whole sequence, net of its prefix-cache hit, it preempts running requests of a strictly lower priority through vLLM's own preemption path, before the running loop, in vLLM's own preemption order: the lowest priority first, then the latest arrival. It preempts as few as make it fit, at most N a step, and none if N would not suffice. The room counts a block as freed only when every holder of it is a victim, never the blocks of the request's own prefix hit, and the running decodes' growth until the victims' frees land (two steps with deferred frees). It makes no room again for a request the waiting loop has not since tried. A victim, its blocks still cached, waits until the prefills in flight have the blocks they still need. With the `OffloadingConnector`, a victim's pending stores are flushed, and its KV stays in the CPU tier for its resume. It is V0's waiting-request preemption (§1) on V1's scheduler, with chunked prefill, a cap of N a step and the CPU tier keeping the victims' KV.
- **The budget cap.** Lower-priority prefill sits out the steps of a request's first tokens: from the step in which the highest-priority waiting request fits (deferred frees counted), while it prefills, until it has its first `budget_cap_tokens` output tokens (16 by default), a step's still in flight among them. Running lower-priority prefill chunks are held, waiting lower-priority requests are not admitted, and one that holds its KV is skipped so that the requests behind it are still considered; decodes are never cut, and nothing is held while every running slot is taken. A turn's prefix is usually evicted from the GPU between turns under agent load and loads back from the CPU tier: the load holds lower-priority prefill for its first `budget_cap_tokens` steps, and again from the step it completes, so a stuck load holds nothing after that. Holding nothing during the load cost the calls 125–131 ms at p50 and 175–202 ms at p95 on the box ([W5's GPU check](BENCHMARKS.md#w5-the-fixed-make-room-on-phastform)), because the load's completion is seen at a step boundary and agent prefill makes the steps three times longer. The engine logs, at most every 10 s, the steps held, the prefill tokens and admissions held back, and make-room's preemptions.
- **The reserve.** Requests of a positive priority value are admitted only while their whole sequences leave the reserve free: a class-aware watermark, where `--watermark` is class-blind.

What E2 found ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)): the budget cap is necessary, and either KV make-room or the reserve does the rest. The window has to cover the reply's first sentence: this voice agent's replies run 23 tokens at the median and 52 at most, and at 16 tokens full replies decode beside agent prefill again (3,094 ms at p95 in W5), where 48 tokens covers all but one of them (2,236 ms). The configuration to deploy is the two PRs' switches with a 48-token window and no reserve.

**The gaps E2 measured with**, in the v0.3 variant and the PRs as first opened, fixed in W5 in the variant (main, up to dff9744) with the further defects W5's review found; the PRs follow (W4's E2 numbers include their effect; W5's ran the fixed variant):
- the budget cap does not hold a lower-priority request that already holds its KV, such as one whose CPU-tier load has finished (vLLM's `kv_holding_waiting` queue, which it drains before `waiting`);
- the KV shortfall counts the blocks of a request's cached prefix as needed, so the reserve and the in-flight gate can refuse a request that stock vLLM admits;
- victims are sized before the next step's decode growth, so a turn can need a second preemption a step later.

**Tests.** CPU tests run the variant on vLLM's real scheduler with CPU torch, and trace it against the PRs' own patches step for step; CI builds the generated module from v0.32.0's source. GPU probes measure the KV case, the prefill case and closed-loop agents beside calls ([`probe.py`, `tail.py`](../python/leaseholder_vllm/src/leaseholder_vllm/make-room)).

## 3. The deployment

### 3.1 The frozen box

Every measurement runs on one pinned configuration with the audio models on the same GPU ([`deploy/phastform`](../deploy/phastform); measured values in [BENCHMARKS.md](BENCHMARKS.md#the-frozen-box)):

| Component | Configuration | GPU memory |
|---|---|---|
| LLM | vLLM 0.32.0, `Qwen/Qwen3-4B-Instruct-2507-FP8`, 32k context, `--kv-cache-memory` 5 GiB (36,400 tokens), prefix caching, priority scheduling, step cap 512/256, KV cache metrics, the `OffloadingConnector` (20 GiB in `/dev/shm`, write-through, generated tokens included) | ≈ 10.6 GiB |
| STT | vLLM 0.32.0 serving `openai/whisper-large-v3-turbo` at `/v1/audio/transcriptions`, `--kv-cache-memory` 437.5 MiB (4,020 tokens) | ≈ 2.4 GiB |
| TTS | Kokoro-82M on Kokoro-FastAPI 0.9.1-rc1 at `/v1/audio/speech` | ≈ 1.0 GiB |

- **Units.** Each server is a `systemd --user` unit, restarted on failure. `leaseholder.target` starts the three at once: both vLLM servers fix their KV pools, so neither profiles memory at its start, and each one's free-memory check passes in any order while the budgets fit the card together ([HARDWARE.md](HARDWARE.md#the-rules), R2); a unit counts as started once its health endpoint answers, and the box once the engine does. `gpu-guard` refuses to start a GPU unit while another user's process holds the GPU.
- **NUMA.** vLLM runs on the GPU's node (node 1, cores 8–15), the audio servers on node 0. The CPU tier's shared memory prefers node 1.
- **Pinned environment.** vLLM 0.32.0 with torch 2.13.0+cu130 serves the LLM and Whisper, Kokoro runs on torch 2.8.0+cu128; lock files are in `deploy/phastform/requirements/`.
- **Time-slicing, not MPS.** As MPS clients, STT and TTS gain a little under agent prefill, but the LLM stage and its tail get worse: the interference that matters is inside the engine ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)).
- **Engine variants** are drop-ins switched by `deploy/phastform/variant.sh`: `make-room`, vLLM #54530 ported to 0.32.0, SGLang 0.5.22, MPS, and engine flags from an arm file.

### 3.2 Hosts

```text
┌──────────────────────────────── cyan: control host, 48 threads, CPU only ───────────────────────────────┐
│  host network: one TCP listener, the gateway's     │  voice plane: network namespace, no route out      │
│  lh-gateway (Rust) · 127.0.0.1:18080, with keys    │  LiveKit server · LiveKit Agents voice worker      │
│  AIPerf agent replay (pinned cores)                │  measurement callers (livekit-rtc) · SMG (arm 8b)  │
│  lh-tunnel@NODE → $XDG_RUNTIME_DIR/lh/NODE-*.sock  │  reach out only through Unix sockets               │
└──────────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                       │ SSH sessions, one key per socket, each the bridge to it
                                                       ▼
┌─────────────── Phastform: RTX 5060 Ti 16 GB, 188 GiB RAM, the frozen box (Gray: the same) ──────────────┐
│  vLLM 0.32 + make-room · Qwen3-4B-FP8 · KV pool 36,400 tokens · CPU tier 20 GiB    lh/engine.sock       │
│  vLLM Whisper large-v3-turbo (STT) lh/stt.sock · Kokoro-82M (TTS) lh/tts.sock · all in a 0700 directory │
│  each server in a network namespace of its own, with a loopback alone (deploy/bin/in-private-net)       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Host | Role |
|---|---|
| **Phastform** | The GPU node every measurement runs on, paused when idle |
| **Gray** | A second, identical GPU node, used in booked windows for the probes; cyan reaches it through `lh-tunnel@gray` |
| **cyan** | The control host: `lh-gateway`, LiveKit and the voice worker, the measurement callers and AIPerf, each on its own cores |
| **peacock** | Linux release builds of `lh-gateway` and `lh-sim`, and simulator runs |

Another GPU node's files come from a hardware profile (`deploy/profiles/`, rendered by `deploy/profile.py`, which renders the box's own files byte for byte), and a host without cyan beside it runs the GPU servers and the control plane together, on loopback and its own cores, under systemd or a small supervisor where it has none ([HARDWARE.md](HARDWARE.md)).

### 3.3 Network and security

See [ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md).
- **Unix sockets end to end.** On the GPU nodes the engine, STT and TTS serve on Unix sockets in `$XDG_RUNTIME_DIR/lh`, a directory of mode 0700, and listen on no TCP port: vLLM's `--api-key` guards only its `/v1`, `/v2`, `/inference` and `/cohere` routes, so the directory is what keeps the hosts' other users from the engine. cyan reaches each socket through `lh-tunnel@NODE`, a `systemd --user` unit: one SSH key per socket, which the node lets run one command for every session whatever its client asks, `exec nc -q0 -U SOCKET`, under `restrict`, which allows no forwarding of any kind; each connection on cyan's socket of the same name is one session, multiplexed over a few SSH connections. On cyan the gateway's 127.0.0.1:18080, with keys, is the only TCP listener in the host's network; its `/metrics`, and the engine's that it relays, need the metrics key there. SMG's arm runs in the voice plane's namespace, between two proxies. `deploy/bin/audit-listeners` checks each host after every deploy and in the smoke test: no TCP listener of the box's units outside that set, and their sockets only in `$XDG_RUNTIME_DIR`. The engine publishes no KV events: they carry every cached block's token ids, and vLLM sends them to whoever binds its TCP endpoint.
- **The servers' own network namespaces.** Each GPU server runs in a network namespace of its own with a loopback alone ([`deploy/bin/in-private-net`](../deploy/bin/in-private-net)): vLLM's engine cores listen on the loopback for torch's gloo groups, and CUDA keeps abstract sockets, which in the host's namespace every user of the node reaches. Their sockets are files, which a namespace does not hide. Where the kernel refuses unprivileged user namespaces, which the single host's preflight finds, its drop-ins mark the servers `LH_PRIVATE_NET=host` and they run in the host's, where the listener audit accepts only their engine cores' loopback listeners on ephemeral ports and CUDA's abstract sockets; anywhere else a namespace that cannot be made fails the unit.
- **Development mode for measurement only.** A production box runs vLLM without its development and RL routes (`/collective_rpc`, `/sleep`, `/update_weights` and the others); a measured box adds them with a drop-in (`variant.sh measure on`) for the harness's `/server_info` and cache resets, on the engine's socket only.
- **The voice plane** runs in a network namespace with no route out. libwebrtc binds ICE sockets on every non-loopback interface and gathers no candidate on loopback alone, so the namespace holds loopback plus one dummy interface (10.255.0.1/32).
- **Sanitized inputs.** The gateway strips the request fields and headers that vLLM, SGLang and SMG trust for scheduling, caching and routing, and sets its own: the class's priority, `X-Session-ID` and the tenant's `cache_salt`, an HMAC of the tenant's name under a secret only the gateway reads, so tenants never share prefix-cache entries and a client that reaches the engine directly cannot name another tenant's salt ([PROTOCOL.md](PROTOCOL.md#headers)). Sessions are scoped to the tenant: a session id from an untrusted tenant is hashed with the tenant's name.
- **KV never crosses the network.** A session's KV stays in the engine that serves it, and a session that loses it, to an engine restart, prefills again.
- **What stays open** ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md#unix-sockets-end-to-end-w6-2026-10-25)). The sockets keep the hosts' other users out, not the deploying user's other processes: on the GPU nodes, other work under the same user (which `gpu-guard` tells from the box's units) reaches the servers, and the voice plane's processes keep the deploying user's uid and reach its systemd bus. While variant `sglang` runs, SGLang listens on 127.0.0.1:8000, which every user of the node reaches: it serves no Unix socket.

## 4. The measurement harness

### 4.1 The gateway

`lh-gateway` ([`crates/lh-gateway`](../crates/lh-gateway/README.md)) is a small Rust proxy in front of the engines, built to run E2's arms on one GPU without Kubernetes. A deployment of the change needs only what it does first: the class's priority set from the caller's identity.
1. **Authenticate** the API key and resolve the tenant and its class, capped by the tenant's `max_class`.
2. **Sanitize** and **resolve the session** ([PROTOCOL.md](PROTOCOL.md#sessions)).
3. **Admit** at the admission point, by the `[admission]` policy: E2's baselines (a background cap, a token reservation, llm-d-style flow control with eviction, preempt-resubmit before the first byte) or the lifecycle policy, arm 9's gateway half (§6).
4. **Forward** with the priority where the upstream reads it (`X-Vllm-Priority`, SGLang's body field or SMG's header), `X-Session-ID` and `cache_salt`.
5. **Stream back** through a bounded buffer: backpressure on a slow client, an upstream abort when the client leaves.

It serves its metrics at `/metrics` (requests, first bytes, the admission point's decisions by policy and reason, the realtime sessions' turn states) and logs one JSON line per request with its first byte to the millisecond, which E2 reads.

The Rust workspace has four crates: `lh-proto` (wire types and their JSON Schemas), `lh-core` (pure models of the frozen box, the E0 tier cost model, KV geometry and a step-time model of vLLM 0.32, which `lh-sim` runs, and the admission policies, which only the gateway runs and whose throttle prices by the step model, compiled in: no I/O and no clock), `lh-gateway` and `lh-sim` (§5).

### 4.2 The voice pipeline

- **Components.** Stock LiveKit server 1.13 and LiveKit Agents 1.8.8 in the voice-plane namespace on cyan (`lh-voice-worker`, [`python/leaseholder_voice`](../python/leaseholder_voice)), with Silero VAD on CPU; STT and TTS reach the GPU node through Unix-socket forwards, and the LLM goes through the gateway's socket with `X-LH-Class: realtime` and the room's session id.
- **Turn handling.** LiveKit's defaults, with preemptive generation on. Experiments end each turn a fixed delay after the end of speech, because LiveKit's end-of-turn model waits its 3 s maximum on some synthesized lines ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)). Barge-in pauses the reply after 0.5 s of speech and interrupts it on the final transcript.
- **Worker settings.** Bound to loopback; a small pool of idle processes; a 30 s LLM timeout, since a cold prefill can exceed LiveKit's 10 s default, and one retry instead of LiveKit's three, for a request that fails fast; no AEC warm-up, since the callers play files.
- **Lifecycle events and stage times.** The worker posts speech onset, end of turn, playback and barge-in to the gateway's `/v1/events`, never blocking the session (one sender task per session, a 1 s timeout, no retry), and logs each reply's stage times (end of speech to the transcript and the end-of-turn decision, the LLM's first token and first sentence, TTS's first audio), from which E2 draws its waterfall.
- **Measurement callers** (`lh-voice bots`) are `livekit-rtc` callers in the same namespace: they play utterances with known end times, record the agent's track and read V2V and barge-in to silence from the waveforms on one clock.

### 4.3 E2's harness

`lh-bench e2` ([`benchmarks/e2`](../benchmarks/e2/README.md)) runs an arm file as a point: it switches the GPU node to the arm's engine, checks the engine against its `/server_info` (each setting the frozen box's, the variant's and the arm's flags set, the scheduler class the variant loads, and exactly the `additional_config` its flags build, so that no variant passes for another; a setting the engine does not report is listed as unchecked), starts AIPerf's agent replay through the gateway, runs the calls, and saves the point's records: the callers' turns, the gateway's request log, AIPerf's exports, the worker's journal, the GPU's clocks once a second, the metrics around the calls, and the revisions deployed on cyan and on the GPU node. A campaign measures a list of arm files in order and resumes where it stopped. `lh-bench e2 export` writes each point's compact record, committed in [`benchmarks/results`](../benchmarks/results/2026-10-23-e2/README.md); `make reproduce-e2` rebuilds the report (tables, frontiers and the stage waterfall) from them, and `make e2-figures` the docs' figures.

## 5. The simulator (E1)

`lh-sim` ([`crates/lh-sim`](../crates/lh-sim/README.md)) replays the SemiAnalysis Claude Code traces, block-scaled 32× (32k context), as AIPerf's `agentic_replay` sends them, through a model of vLLM 0.32's two-tier prefix cache and its step loop (the step cap, chunked prefill, first come first served, as at one priority), with a step time fitted to the box. It reproduces resume TTFT p50 and p90 within 8% at 8 and 16 agent sessions and is 18–30% optimistic at 24. Its arms are LRU with and without the CPU tier, TTL holds, prefetch timed by session history or by an oracle, simplified Tokencake, MORI and InferCept, and a clairvoyant reference (Belady's MIN in each tier). `lh-sim hardware` places an L4 and an H100 SXM on the same regime map, with the box's costs scaled by their published peaks, labeled as modeled.

## 6. Tested and not adopted

- **Protection timed by the conversation, in the control plane** (E2). Under the lifecycle policy the gateway follows each call's turn from the worker's lifecycle events: it holds KV headroom from speech onset by deferring new background requests, throttles background prefill in the critical window from the end of the turn to the reply's first audio (priced by the step model), caps the reply at 16 tokens when the headroom was not there (degrade), and admits deferred work after 10 s (aging). In E2's discriminating scenario (W4) these mechanisms could not act: running agent requests held the pool, which the headroom can protect only by deferring new ones, so it never had its room (degrade capped 60 of 60 replies in every run), and the throttle, checked after the headroom, deferred nothing while the headroom ran. Room for the turn came from the engine alone: the same policy in front of stock vLLM left the tail where stock does (10,993 ms). So a control plane in front can make room only by aborting work it has admitted, as preempt-resubmit does ([METHODS.md](METHODS.md#what-lifecycle-timing-could-and-could-not-do)); whether lifecycle timing helps where its mechanisms engage was not isolated. The policy stays as an experiment arm, with its ablations ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)).
- **Retention and prefetch of agents' KV by their lifecycle** (E1). In the simulator, even an oracle that knows when each agent returns from its tool closes under 9% of resume TTFT at p50 over LRU with the CPU tier, since the median resume is already a GPU hit or a short restore; at p90 it closes 28% at low pressure, but session history, what a deployment knows without the tools' return times, closes nothing ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)). The CPU-tier policy built for it, `LeasePolicy`, and the connector that carried its hint, `LeaseConnector`, were checked live on the GPU, never used, and removed after v0.3 ([the v0.3 tree](https://github.com/StevenWang-CY/leaseholder/tree/v0.3/python/leaseholder_vllm) keeps them).

## 7. Not built

The original design put leases in the control plane: each session's claim on KV residency, with prefetch timed from lifecycle events, a sidecar per GPU node (`lh-node`) with engine epochs, cache contracts from Anthropic's `cache_control`, and a ledger attributing every recomputed token to a cause. After E1 and E2 these were dropped: §6's two results remove their premise, and sticky placement, parked with the two-node points (below), does the routing leases were also for. They would be reopened by a source of tool-return times ahead of the return, which the oracle's tail gain needs, or by an engine whose running KV a control plane can reclaim. The types and plugins built for them (`lh-proto`'s lease hint and ledger rows with their schemas, `X-LH-Synthetic`, the `leaseholder_vllm` plugins) were removed after v0.3.

The two-node points (dedicate against pool, in a window on Gray) never ran: no second GPU is free before v1.0. Their code, the gateway's placement of sessions on several GPU nodes with the voice worker's and the harness's halves, is parked on the `two-node` branch.

Still planned: the background tier's cost of make-room (restored against recomputed prompt tokens, from the counters every E2 point keeps).

## 8. Failure handling

| Failure | Behavior | Evidence |
|---|---|---|
| Engine crash | The unit restarts 5 s after its last process exits; streams in flight end with the gateway's `upstream_unavailable` error event, then one 502 and 503s with `Retry-After` until the engine's `/health` answers; sessions re-prefill on the next turn | Once in E2, an out-of-memory kill; the W8 engine-kill drill: streams broken within 57 ms, the first success 112.3 s after the kill with the 15 s delay of then ([PRODUCTION.md](PRODUCTION.md#how-it-fails)) |
| Gateway restart or crash | `lh-gateway.socket` holds the listeners. A reload hands them to a successor and drains: nothing is refused or waits. A restart or a crash leaves new connections waiting in the backlog, for the drain or about a second. Background requests in flight are cut 5 s into a drain with a `draining` error | The binary's reload tests (a request every 20 ms on each socket through a reload, all answered); the W8 gateway drills |
| GPU node loss | As an engine crash: a request that finds the engine broken starts a probe of its `/health`, and the gateway answers 503 with `Retry-After` without going upstream until it answers | The gateway's upstream tests |
| Slow client | Backpressure through the bounded buffer, then an upstream abort | A stalled reader |
| Client disconnect | Upstream abort; vLLM aborts the request | Mid-stream disconnects |
| Host memory pressure | `systemd-oomd` may pick the engine; a campaign run again resumes at its failed point | [RUNBOOK.md](../ops/RUNBOOK.md#failures) |
| Another user's GPU work | `gpu-guard` refuses to start the box; a point measured beside foreign GPU work is measured again | [RUNBOOK.md](../ops/RUNBOOK.md#checks-before-heavy-work-on-a-shared-host) |

## 9. Testing

| Layer | Tests |
|---|---|
| `lh-proto` | Wire formats pinned by examples; schemas kept current |
| `lh-core` | The cost model against E0's cells, the KV geometry, the step-time model; each admission policy's decisions on hand-made sequences of requests and turns |
| `lh-sim` | The AIPerf port against AIPerf's own plan; the step loop on hand-made sequences; the clairvoyant tiers against exhaustive search; every arm end to end on a 16-trace subset; `make reproduce-e1` regenerates E1's tables (not in CI: the corpus stays outside the repository) |
| `lh-gateway` | Against an in-process mock engine: disconnects, backpressure, sanitization and tenant-scoped sessions, event idempotency, every admission policy, the engine's readiness, the reload and its drain; the proxy's overhead in process |
| Engine | The generated `make-room` module built from v0.32.0's source; its behavior on vLLM's real scheduler with CPU torch; the GPU probes |
| Voice | A hermetic LiveKit test in the voice-plane namespace (barge-in, headers, lifecycle events in order); unit tests of the event delivery |
| E2 | The report, its figures and the demo renderer on synthetic points; `make reproduce-e2` and `make e2-figures` rebuild them from the committed records (not in CI) |
| CI | GitHub Actions on CPU: rustfmt, clippy and ruff, every Rust test, the engine variants' build tests against vLLM's source, the Python unit tests |
