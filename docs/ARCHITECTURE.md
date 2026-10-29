# Architecture

Flatshare runs a realtime voice agent beside background LLM work on one GPU, or on two. Its central part is a change to vLLM's scheduler: a waiting realtime request can take KV from running background requests, and background prefill sits out the realtime reply's first tokens. It is proposed to vLLM as pull requests [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351), open and under review, and runs on vLLM 0.32 as the engine variant `make-room` (§2). Around it are:
- the serving path the deployment runs (§3): one GPU node, or two cards with a warm spare where a second card is available;
- its network and security (§4);
- the configuration and hosts it was measured on (§5);
- the harness that measured it against the configurations of vLLM 0.32, SGLang 0.5.22, SMG and llm-d's flow control
  (emulated) that an operator can deploy without it, and a KV-cache simulator (§6).

The designs that were measured and dropped are in §7, failure handling in §8 and testing in §9.

This document describes what is built, as of v1.7, for a reader who wants to know how the parts work and fit together before reading or changing the code. [PRODUCTION.md](PRODUCTION.md) gives the configuration to deploy, [BENCHMARKS.md](BENCHMARKS.md) the numbers, [PRIOR-ART.md](PRIOR-ART.md) what is new here and what is prior work, the [ADRs](README.md#decisions) the decisions, and [GLOSSARY.md](GLOSSARY.md) the terms.

## 1. The problem in the engine

A voice turn on a GPU shared with coding agents loses its time in the LLM's first sentence. On stock vLLM 0.32, tuned (priority scheduling, a 512-token step cap and a write-through CPU KV tier), with 24 agent sessions of long contexts beside two calls, the reply's first sentence takes 3.9 s of a 5.3 s mean turn in W5, against 0.3 s with voice alone ([W5's stage means](../benchmarks/results/2026-10-24-e2-w5/report/frontier.md#stage-waterfall)). Two mechanisms in the scheduler cause it:
- **A waiting request cannot take KV.** In vLLM 0.32's V1 scheduler, the running loop preempts only so that a running request can grow, and the waiting loop stops at the first request it cannot allocate, so nothing preempts for a waiting request ([#40004](https://github.com/vllm-project/vllm/issues/40004)). vLLM's V0 scheduler did, from v0.6.2 until V0 was removed in v0.11.0, but only with chunked prefill off, and it recomputed its victims ([#5958](https://github.com/vllm-project/vllm/pull/5958); [PRIOR-ART.md](PRIOR-ART.md#vllm-0320-2026-10-18)). When running agents hold the KV pool, a voice turn waits for one of them to finish: 4.9–5.0 s to the first token of a 750-token turn behind two 18k-token agent requests ([the KV case](BENCHMARKS.md#the-kv-case-make-room)).
- **Prefill shares the reply's steps.** Once the turn runs, every step of its reply also carries the agents' prefill chunks, up to about 270 ms each after a 30k-token context, so the first sentence decodes slowly.

Neither mechanism is specific to the card, though what each costs is: the first appears wherever agents' contexts are long against the pool, which the card's memory sets, the second wherever background prefill runs beside a realtime decode ([METHODS.md](METHODS.md#what-transfers-to-other-hardware) places other GPUs by a model, and the A100 run measured one).

## 2. The change: #63350 and #63351

The variant `make-room` ([`python/flatshare_vllm`](../python/flatshare_vllm/README.md)) is built at each engine start from the installed vLLM 0.32.0's scheduler, without modifying vLLM: `build.py` applies the PRs' scheduler diff, backported to 0.32.0, and the variant's own patch, every hunk where it must be, and the engine loads the module with `--scheduler-cls`. It adds three switches to priority scheduling, each a key of `--additional-config` and off by default:

| Switch | Variant key | Upstream |
|---|---|---|
| KV make-room | `make_room.max_preemptions_per_step` | `--max-priority-preemptions-per-step` ([#63350](https://github.com/vllm-project/vllm/pull/63350)) |
| The budget cap | `make_room.budget_cap`, `make_room.budget_cap_tokens` | `--priority-prefill-budget-cap`, `--priority-prefill-budget-cap-tokens` ([#63351](https://github.com/vllm-project/vllm/pull/63351)) |
| The reserve | `make_room.reserve_tokens` | none: carried in the variant only |

Each switch in a sentence; [REPORT §3](REPORT.md#3-design) gives the two rules in full, with its Figure 5 and Algorithms 1 and 2:
- **KV make-room.** When the highest-priority waiting request lacks KV for its whole sequence, the scheduler preempts running requests of a strictly lower priority through vLLM's own path, the lowest priority and then the latest arrival first, as few as make it fit, at most N a step and none if N would not suffice, their KV kept in the CPU tier ([§3.1](REPORT.md#31-make-room)).
- **The budget cap.** Lower-priority prefill sits out the steps of a request's first `budget_cap_tokens` output tokens (16 by default), from the step in which the request fits, and decodes are never cut ([§3.2](REPORT.md#32-the-prefill-hold)).
- **The reserve.** Requests of a positive priority value are admitted only while their whole sequences leave the reserve free: a class-aware watermark, where `--watermark` is class-blind.

What E2 found ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)): the budget cap is necessary, and either KV make-room or the reserve does the rest. The window has to cover the reply's first sentence: this voice agent's replies run 23 tokens at the median and 52 at most, their first sentences 15 tokens at the median and 39 at most ([the first-sentence record](../benchmarks/results/2026-10-27-first-sentence)), and at 16 tokens full replies decode beside agent prefill again (3,094 ms at p95 in W5), where 48 tokens covers all but one of them (2,236 ms). The configuration to deploy is the two PRs' switches with a 48-token window and no reserve.

**The gaps E2 measured with before W5.** The v0.3 variant and the PRs as first opened had three gaps, whose effect W4's E2 numbers include. W5 fixed them in the variant (main, up to b656c25), together with further defects found in review, and the PRs carry the same fixes; W5's numbers ran the fixed variant. The gaps were:
- the budget cap did not hold a lower-priority request that already held its KV, such as one whose CPU-tier load had finished (vLLM's `kv_holding_waiting` queue, which it drains before `waiting`);
- the KV shortfall counted the blocks of a request's cached prefix as needed, so the reserve and the in-flight gate could refuse a request that stock vLLM admits;
- victims were sized before the next step's decode growth, so a turn could need a second preemption a step later.

**Tests.** CPU tests run the variant on vLLM's real scheduler with CPU torch, and trace it against the PRs' own patches step for step. CI builds the generated module from v0.32.0's source, and its make-room job runs both suites on Linux x86-64 (`make test-make-room-ci`). GPU probes measure the KV case, the prefill case and closed-loop agents beside calls ([`probe.py`, `tail.py`](../python/flatshare_vllm/src/flatshare_vllm/make-room)).

## 3. The serving path

This section is the serving path, from the configuration to the alerts: the serving profile, a GPU node, the gateway, the voice worker, two cards, releases and what to watch. [PRODUCTION.md](PRODUCTION.md) gives the operator's view of it, and [RUNBOOK.md](../ops/RUNBOOK.md) the commands. Figure 1 shows its hosts and the sockets between them.

<picture>
  <source media="(max-width: 767px)" srcset="figures/architecture-narrow.svg">
  <img src="figures/architecture.svg" width="100%" alt="Three hosts as dashed groups. On cyan, the control host, the callers, the LiveKit server and the voice worker sit in the voice plane's network namespace, which has no route out. The worker sends its LLM requests to the gateway through gateway.sock and its speech to the tunnel to Phastform through stt.sock and tts.sock; the coding agents reach the gateway through 127.0.0.1:18080, TCP with a tenant's key; the gateway sends its engine requests through engine.sock to the tunnel to Phastform, and, dashed, to the tunnel to Gray. Each tunnel carries SSH sessions to its node's three servers: vLLM 0.32 with make-room on engine.sock, shaded as the change, Whisper STT on stt.sock and Kokoro TTS on tts.sock. On Phastform, an RTX 5060 Ti, each server has a network namespace of its own; Gray, a second card when present, is drawn dashed, its servers in the host's namespace.">
</picture>

**Figure 1. Outside the voice plane the units meet only at Unix sockets, which SSH bridges between hosts, and the gateway's port is the only TCP listener in cyan's host network.** Dashed groups are hosts and network namespaces, and the change's engine is blue. A tunnel per node serves the node's sockets on cyan as `NODE-engine.sock`, `NODE-stt.sock` and `NODE-tts.sock` (§4). Gray, dashed, is the second card of the two-card arrangement (§3.5), which the gateway reaches only through `[[nodes]]`. The callers are the measurement's (§6.2).

### 3.1 The serving profile

`deploy/profiles/rtx-5060-ti-serve.toml` names the configuration to serve ([ADR-0009](adr/0009-immutable-releases-and-the-exact-serving-profile.md)):
- the frozen box (§5.1) running the change with the PRs' switches: two preemptions a step, the budget cap at 48 tokens, and no reserve;
- the step cap the sweep chose, 256 tokens a step without a long-prefill threshold ([the sweep](../benchmarks/results/2026-10-27-sweep/README.md), confirmed in [the rounds](../benchmarks/results/2026-10-28-rounds/README.md));
- the hash of the scheduler module, the calls the voice worker admits (two), and real callers' audio.

`deploy/profile.py` renders it into `deploy/phastform/serving.json`, every setting the engine must run with, named as the engine's own configuration names it. It also renders the `make-room-serve` variant, which passes the profile's step cap over the frozen box's.
- **Held at every start.** The variant's engine loads the serve module, whose scheduler writes the configuration it runs to a file as the engine starts. `deploy/bin/serving-check`, the unit's `ExecStartPost`, compares that file with serving.json and fails the start on any difference, naming the setting. A release runs the same check, and holds the voice worker's admission to the profile's ([the serving profile](../deploy/phastform/variants/README.md#the-serving-profile)).
- **No development routes.** A serving box runs vLLM without its development and RL routes. The measurement drop-in that adds them is off (§4).

### 3.2 A GPU node

A GPU node runs the three servers of §5.1, the engine on the serving variant. Each is a `systemd --user` unit, and `flatshare.target` starts the three together. Each serves on a Unix socket in `$XDG_RUNTIME_DIR/flatshare` and listens on no TCP port (§4). `gpu-guard` refuses to start a unit while another user's process holds the GPU. Phastform carries the calls; with a second card, Gray is the warm spare (§3.5).

### 3.3 The gateway

`flatshare-gateway` ([`crates/flatshare-gateway`](../crates/flatshare-gateway/README.md)) is the serving control plane, on the control host ([ADR-0007](adr/0007-the-gateway-is-the-serving-control-plane.md)). A request takes the steps of Figure 2.

<picture>
  <source media="(max-width: 767px)" srcset="figures/request-path-narrow.svg">
  <img src="figures/request-path.svg" width="100%" alt="A column of seven steps, each step's main refusals leaving to its right with their status and code. Authenticate, on no known key or an unknown class: 401 invalid_api_key, 400 invalid_class. Bound, on a bound full or a body too large: 503 in_flight_limit, 429 tenant_in_flight_limit, 413 body_too_large. Sanitize, on a body that is no JSON object or asks too many choices: 400 invalid_json, 400 too_many_choices. Place, with no engine ready or no free slot: 503 upstream_not_ready, no_healthy_node, no_call_slot, no_agent_slot. Forward, on no answer or no headers in time: 502 upstream_unavailable, 504 upstream_timeout. Relay, on a stream cut short: an error event of type upstream_unavailable, upstream_timeout or draining. Account records one outcome for every request. The gateway's error table lists every code.">
</picture>

**Figure 2. A request is refused at the first step it fails, and the first four refuse it before the engine sees it.** Each refusal carries its status and code; once a stream has begun, it ends with an error event of that type instead. The figure names each step's main codes; [the gateway's errors](../crates/flatshare-gateway/README.md#errors) lists every one. A passed deadline, or a client that leaves, cancels the engine's work.

| Step | What it does |
|---|---|
| Authenticate | The key gives the tenant and its class, capped by its `max_class`; the headers the engines trust go ([PROTOCOL.md](PROTOCOL.md#headers)) |
| Bound | In flight in all, by class and by tenant, before the body is read, then its bytes; past a bound, refused at once with `Retry-After` |
| Sanitize | The body's fields the engines trust go, the class's reply cap and `max_choices` bound it, and its session is resolved ([PROTOCOL.md](PROTOCOL.md#sessions)) |
| Place | With `[[nodes]]`, its session's node, or a node with its role and a free slot (§3.5); with one engine, while that engine is ready |
| Forward | With the class's priority where the engine reads it (`X-Vllm-Priority`), `X-Session-ID` and the tenant's `cache_salt` |
| Relay | Through a bounded buffer, under the class's deadlines for the first content, each frame after it and the whole reply |
| Account | Once: offered, admitted and one outcome, with a service level of good over valid requests by class |

The engine's state comes from inference: its requests' successes, and an idle-only canary. `/ready` answers by that state. A reload (SIGHUP) hands the listeners and the placements to a successor without a gap. Only E2 configures an `[admission]` policy, under which a request also waits at the admission point before it is placed (§6.1).

### 3.4 The voice worker

LiveKit server 1.13 and LiveKit Agents 1.8.8, with Silero VAD on CPU, run the voice agent on the control host, in a network namespace with no route out (§4) ([`python/flatshare_voice`](../python/flatshare_voice/README.md)). STT and TTS reach the GPU node through Unix-socket tunnels, and the LLM goes through the gateway's socket as the realtime class. In serve mode, which the serving arrangement runs, the worker:
- **admits calls by capacity:** at most the profile's two at once, and none while the engine does not serve (the gateway's `/ready`), so that a call it could not answer gets no agent rather than a slow one;
- **places each call** on a GPU node with `POST /v1/sessions` before it speaks, binds the call's STT and TTS to that node's sockets, refuses a call that finds no free slot, and ends the placement with `DELETE /v1/sessions/{id}` when the call ends or its job does not run, which frees the slot at once;
- **hears real callers,** with LiveKit's end-of-turn model and its AEC warm-up, which no benchmark measured (§6.2);
- **drains for a release:** while a drain file exists it takes no new call, and the release restarts it once its calls end;
- **writes its state** every second, which `serving-check` holds to the profile.

Of the lifecycle events it posts only each call's `session_closed`; the turns' events go out in bench mode, for E2's lifecycle policy (§6.3).

### 3.5 Two cards

Where a second card is available, the arrangement puts each card in a `[[nodes]]` entry of `gateway.toml` ([ADR-0008](adr/0008-two-cards-consolidation-with-a-warm-spare.md); [`deploy/cyan/gateway.example.toml`](../deploy/cyan/gateway.example.toml) has both), and the runbook's [Two GPU nodes](../ops/RUNBOOK.md#two-gpu-nodes) says how to run it. Figure 3 shows its placement and its failover.

<picture>
  <source media="(max-width: 767px)" srcset="figures/two-cards-narrow.svg">
  <img src="figures/two-cards.svg" width="100%" alt="Two panels of two cards, each with a row of two call slots and a row of agents' slots, four on Phastform and twelve on Gray. (a) In service: both calls fill Phastform's call slots beside its four agents' slots, and Gray, the warm spare, runs agents only, its call slots empty, under calls = fill. (b) The first card failed: Phastform's engine is down and its slots empty; both calls take Gray's call slots at their next turn, 0.7 s after a kill and 1.9 s after a stop, and Gray, covering, holds its twelve agents' slots, hatched, under a bracket that reads held until the calls end.">
</picture>

**Figure 3. Both calls fill the first card, and the spare takes them by their next turn when the first fails.** (a) Each card has two call slots (`max_calls = 2`) and up to 4 or 12 agents' sessions (`max_agents`), and `calls = "fill"` places each call on the first card with a free call slot. (b) In the failover drills, both calls' next turns reached the spare 0.7 s after a SIGKILL to the first card's engine and 1.9 s after its stop ([the record](../benchmarks/results/2026-10-28-node-failover)); the spare holds its agents' requests, hatched, until the moved calls end.

- **Health.** A node is healthy while its whole pipeline is: its engine serving and ready, and its STT and TTS answering `/health`.
- **Sessions stay on their node,** since KV never moves: a session whose node fails is placed again by its next request, and a realtime request whose node failed before its first content is sent once more on the other node.
- **Covering.** The spare covers while it carries any call: its agents' requests wait (`failover_background_max_in_flight`, 0) until those calls end.

### 3.6 Releases

`deploy/deploy.sh HOST` makes an immutable release under `~/flatshare/releases/`, with a manifest of what it holds ([PRODUCTION.md](PRODUCTION.md#releasing-a-change); [ADR-0009](adr/0009-immutable-releases-and-the-exact-serving-profile.md)), and takes it through the states of Figure 4. The drain is of what the switch would cut, the worker's calls and the engine's requests; the switch moves the `current` link and restarts only the units whose inputs changed; and the check runs those of §3.1. A release rolls back to the previous one offline.

<picture>
  <source media="(max-width: 767px)" srcset="figures/release-narrow.svg">
  <img src="figures/release.svg" width="100%" alt="A column of a release's states from deploy.sh HOST: prepare, verify, drain, switch, restart, check and serving. Prepare exits 2 with nothing switched when the host is locked or the upload changed; verify exits 1 with nothing switched when a check fails; drain exits 2 with nothing switched when calls do not end or an E2 point runs. A failure in the switch, the restart or the check leads to switch back, which exits 1 with the release before running again, or 3 when switching back failed too. Serving exits 0, and --rollback enters at verify.">
</picture>

**Figure 4. A release switches nothing until it has verified and drained, and switches back by itself when anything after the switch fails.** `deploy.sh`'s exit status says where it stopped, and `--rollback` takes the previous release through the same states from the verification.

### 3.7 What to watch

The gateway's `/metrics`, and each engine's, relayed by node, feed Prometheus ([`ops/prometheus`](../ops/prometheus)):
- **Two objectives**, each with a fast and a slow burn-rate alert against its target ([SLO.md](SLO.md)): the realtime first byte's service level, failures counted, and the gateway's overhead.
- **Health alerts** ([`rules.yml`](../ops/prometheus/rules.yml), tested with promtool) on:
  - the gateway, and the engine's readiness, state and metrics;
  - broken streams, refusals for capacity and missed deadlines;
  - each node's health and covering, refused calls, and a full KV pool with requests waiting.

A Grafana dashboard shows the same ([`ops/grafana`](../ops/grafana/flatshare.json)). The alerts are rules for an operator's Prometheus: nothing in the repository pages anyone. The 24-hour soak ([RUNBOOK.md](../ops/RUNBOOK.md#soaks)) is the arrangement's long-run check.

<a id="33-network-and-security"></a>

## 4. Network and security

See [ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md).
- **Unix sockets end to end** (Figure 1). On the GPU nodes the engine, STT and TTS serve on Unix sockets in `$XDG_RUNTIME_DIR/flatshare`, a directory of mode 0700, and listen on no TCP port: vLLM's `--api-key` guards only its `/v1`, `/v2`, `/inference` and `/cohere` routes, so the directory is what keeps the hosts' other users from the engine. cyan reaches each socket through `flatshare-tunnel@NODE`, a `systemd --user` unit, with one SSH key per socket. The node lets that key run one command for every session, whatever its client asks, `exec nc -q0 -U SOCKET`, under `restrict`, which allows no forwarding of any kind. Each connection on cyan's socket of the same name is one session, multiplexed over a few SSH connections.
- **One TCP listener on cyan** (Figure 1). The gateway's 127.0.0.1:18080, with keys, is the only TCP listener in cyan's host network; its `/metrics`, and the engine's that it relays, need the metrics key there. SMG's arm runs in the voice plane's namespace, between two proxies. `deploy/bin/audit-listeners` checks each host after every deploy and in the smoke test: no TCP listener of the box's units outside that set, and their sockets only in `$XDG_RUNTIME_DIR`.
- **No KV events.** The engine publishes none: they carry every cached block's token ids, and vLLM sends them to whoever binds its TCP endpoint.
- **The servers' own network namespaces.** Each GPU server runs in a network namespace of its own with a loopback alone ([`deploy/bin/in-private-net`](../deploy/bin/in-private-net)): vLLM's engine cores listen on the loopback for torch's gloo groups, and CUDA keeps abstract sockets, which in the host's namespace every user of the node reaches. Their sockets are files, which a namespace does not hide. Where the kernel refuses unprivileged user namespaces, which the single host's preflight finds and Gray's install finds at every release (`deploy/bin/private-net-check`), their drop-ins mark the servers `FLATSHARE_PRIVATE_NET=host` and they run in the host's, where the listener audit accepts only their engine cores' loopback listeners on ephemeral ports and CUDA's abstract sockets; anywhere else a namespace that cannot be made fails the unit.
- **Development mode for measurement only.** A production box runs vLLM without its development and RL routes (`/collective_rpc`, `/sleep`, `/update_weights` and the others); a measured box adds them with a drop-in (`variant.sh measure on`) for the harness's `/server_info` and cache resets, on the engine's socket only.
- **The voice plane** runs in a network namespace with no route out. libwebrtc binds ICE sockets on every non-loopback interface and gathers no candidate on loopback alone, so the namespace holds loopback plus one dummy interface (10.255.0.1/32).
- **Sanitized inputs.** The gateway strips the request fields and headers that vLLM, SGLang and SMG trust for scheduling, caching and routing, and sets its own: the class's priority, `X-Session-ID` and the tenant's `cache_salt`, an HMAC of the tenant's name under a secret only the gateway reads, so tenants never share prefix-cache entries and a client that reaches the engine directly cannot name another tenant's salt ([PROTOCOL.md](PROTOCOL.md#headers)). Sessions are scoped to the tenant: a session id from an untrusted tenant is hashed with the tenant's name.
- **KV never crosses the network.** A session's KV stays in the engine that serves it, and a session that loses it, to an engine restart, prefills again.
- **What stays open** ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md#unix-sockets-end-to-end-w6-2026-10-25)). The sockets keep the hosts' other users out, not the deploying user's other processes: on the GPU nodes, other work under the same user (which `gpu-guard` tells from the box's units) reaches the servers, and the voice plane's processes keep the deploying user's uid and reach its systemd bus. While variant `sglang` runs, SGLang listens on 127.0.0.1:8000, which every user of the node reaches: it serves no Unix socket.

## 5. The measurement configuration

<a id="31-the-frozen-box"></a>

### 5.1 The frozen box

Every measurement on the box runs one pinned configuration, with the audio models on the same GPU ([`deploy/phastform`](../deploy/phastform); measured values in [BENCHMARKS.md](BENCHMARKS.md#the-frozen-box)). The serving profile is the same box, running the change at its own step cap (§3.1):

| Component | Configuration | GPU memory |
|---|---|---|
| LLM | vLLM 0.32.0, `Qwen/Qwen3-4B-Instruct-2507-FP8`, 32k context, `--kv-cache-memory` 5 GiB (36,400 tokens), prefix caching, priority scheduling, step cap 512/256, KV cache metrics, the `OffloadingConnector` (20 GiB in `/dev/shm`, write-through, generated tokens included) | ≈ 10.6 GiB |
| STT | vLLM 0.32.0 serving `openai/whisper-large-v3-turbo` at `/v1/audio/transcriptions`, `--kv-cache-memory` 437.5 MiB (4,020 tokens) | ≈ 2.4 GiB |
| TTS | Kokoro-82M on Kokoro-FastAPI 0.9.1-rc1 at `/v1/audio/speech` | ≈ 1.0 GiB |

- **Units.** Each server is a `systemd --user` unit, restarted on failure. `flatshare.target` starts the three at once: both vLLM servers fix their KV pools, so neither profiles memory at its start, and each one's free-memory check passes in any order while the budgets fit the card together ([HARDWARE.md](HARDWARE.md#the-rules), R2); a unit counts as started once its health endpoint answers, and the box once the engine does. `gpu-guard` refuses to start a GPU unit while another user's process holds the GPU, or when `nvidia-smi` cannot list the GPU's processes.
- **NUMA.** vLLM runs on the GPU's node (node 1, cores 8–15), the audio servers on node 0. The CPU tier's shared memory prefers node 1.
- **Pinned environment.** vLLM 0.32.0 with torch 2.13.0+cu130 serves the LLM and Whisper, Kokoro runs on torch 2.8.0+cu128; lock files are in `deploy/phastform/requirements/`.
- **Time-slicing, not MPS.** As MPS clients, STT and TTS gain a little under agent prefill, but the LLM stage and its tail get worse: the interference that matters is inside the engine ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)).
- **Engine variants** are drop-ins switched by `deploy/phastform/variant.sh`: `make-room`, `make-room-v03` (make-room as v0.3 built it, W5's control), vLLM #54530 ported to 0.32.0, SGLang 0.5.22, MPS, `agents-only` (a card for the agents alone: the STT and the TTS stopped and their 3.4 GiB in the engine's KV pool, 8.4 GiB, under vLLM's own scheduling for the card), and engine flags from an arm file ([the variants](../deploy/phastform/variants/README.md)).

### 5.2 Hosts

Figure 1 shows the hosts. cyan has 48 threads and no GPU; Phastform and Gray each have an RTX 5060 Ti 16 GB and 188 GiB of RAM, and run the frozen box (§5.1). A measurement adds, in cyan's host network, AIPerf's agent replay on pinned cores, and in the voice plane's namespace the measurement callers (`livekit-rtc`) and SMG (arm 8b).

| Host | Role |
|---|---|
| **Phastform** | The first GPU node: every E2 point but the A100 run's before the two-card rounds, which ran on it and Gray in turn, and the serving arrangement's first card; paused when idle |
| **Gray** | A second, identical GPU node: the probes' booked windows, the second card of the two-card rounds, and the warm spare of the serving arrangement on two nodes; cyan reaches it through `flatshare-tunnel@gray` |
| **cyan** | The control host: `flatshare-gateway`, LiveKit and the voice worker, the measurement callers and AIPerf, each on its own cores |
| **peacock** | Linux release builds of `flatshare-gateway` and `flatshare-sim`, and simulator runs |

Another GPU node's files come from a hardware profile (`deploy/profiles/`, rendered by `deploy/profile.py`, which renders the box's own files byte for byte), and a host without cyan beside it runs the GPU servers and the control plane together, on loopback and its own cores, under systemd or a small supervisor where it has none ([HARDWARE.md](HARDWARE.md)). The A100 run of 2026-10-26 ran that way, in a Google Colab container under the small supervisor ([the A100 run](HARDWARE.md#the-a100-run)).

## 6. The measurement harness and the simulator

### 6.1 E2's admission policies

Under an `[admission]` policy, which only E2's arms configure, the gateway's admission point holds a request between its bounds and its placement until the policy admits it ([Admission](../crates/flatshare-gateway/README.md#admission)). The policies are E2's baselines (a background cap, a token reservation, llm-d-style flow control with eviction, and preempt-resubmit before the first byte) and the lifecycle policy, arm 9's gateway half (§7). They live in the gateway's `src/harness/`, which the proxy reaches only through a request's admission ticket, and the gateway refuses them beside `[[nodes]]`. The gateway's `/metrics` adds the admission point's decisions by policy and reason, and the realtime sessions' turn states, and it logs one JSON line per request with its first byte to the millisecond, which E2 reads.

The Rust workspace has four crates: `flatshare-proto` (wire types and their JSON Schemas), `flatshare-core`, `flatshare-gateway` and `flatshare-sim` (§6.4). `flatshare-core` is pure code, compiled in, with no I/O and no clock: models of the frozen box (the E0 tier cost model, KV geometry and a step-time model of vLLM 0.32), which `flatshare-sim` runs, and the admission policies, which only the gateway runs and whose throttle prices by the step model.

<a id="42-the-voice-pipeline"></a>

### 6.2 The voice pipeline in measurement

E2 ran the worker of §3.4 in bench mode, which takes every call LiveKit dispatches, beside callers that play files:
- **Turn handling.** LiveKit's defaults, with preemptive generation on. Experiments end each turn a fixed delay after the end of speech, because LiveKit's end-of-turn model waits its 3 s maximum on some synthesized lines ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)). Barge-in pauses the reply after 0.5 s of speech and interrupts it on the final transcript.
- **Worker settings.** Bound to loopback; a small pool of idle processes; a 30 s LLM timeout, since a cold prefill can exceed LiveKit's 10 s default, and one retry instead of LiveKit's three, for a request that fails fast; no AEC warm-up, since the callers play files.
- **Lifecycle events and stage times.** The worker posts speech onset, end of turn, playback and barge-in to the gateway's `/v1/events` in bench mode, for E2's lifecycle policy (in serve mode only the session's close, `session_closed`), never blocking the session (one sender task per session, a 1 s timeout, no retry), and logs each reply's stage times (end of speech to the transcript and the end-of-turn decision, the LLM's first token and first sentence, TTS's first audio), from which E2 draws its waterfall.
- **Measurement callers** (`flatshare-voice bots`) are `livekit-rtc` callers in the same namespace: they play utterances with known end times, record the agent's track and read V2V and barge-in to silence from the waveforms on one clock.

### 6.3 E2's harness

`flatshare-bench e2` ([`benchmarks/e2`](../benchmarks/e2/README.md)) runs an arm file as a point: it switches the GPU node to the arm's engine, checks the engine against its `/server_info` (each setting the frozen box's, the variant's and the arm's flags set, the scheduler class the variant loads, and exactly the `additional_config` its flags build, so that no variant passes for another; a setting the engine does not report is listed as unchecked), starts AIPerf's agent replay through the gateway, runs the calls, and saves the point's records: the callers' turns, the gateway's request log, AIPerf's exports, the worker's journal, the GPU's clocks once a second, the metrics around the calls, and the revisions deployed on cyan and on the GPU node. A campaign measures a list of arm files in order and resumes where it stopped. `flatshare-bench e2 export` writes each point's compact record, committed in [`benchmarks/results`](../benchmarks/results/2026-10-23-e2/README.md); `make reproduce-e2` rebuilds the report (tables, frontiers and the stage waterfall) from them, and `make figures` the docs' figures.

No counter shows what a preempted request recomputes when it resumes: vLLM 0.32 counts a request's computed prompt tokens at its first scheduling only. E3 measured the rest of what make-room takes from the agents from the counters every W5 point keeps ([E3](BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)).

Beside E2, two gates in [`benchmarks/quality`](../benchmarks/quality) judged whether sharing a card is worth it. Consolidation met the resource gate ([ADR-0008](adr/0008-two-cards-consolidation-with-a-warm-spare.md)). The served model failed the quality gate on a useful pair of jobs, questions about private documents in a call beside enriching new documents into records: it missed 9 of the gate's 11 criteria ([the gate's record](../benchmarks/results/2026-10-27-quality/README.md)).

### 6.4 The simulator (E1)

`flatshare-sim` ([`crates/flatshare-sim`](../crates/flatshare-sim/README.md)) replays the SemiAnalysis Claude Code traces, block-scaled 32× (32k context), as AIPerf's `agentic_replay` sends them, through a model of vLLM 0.32's two-tier prefix cache and its step loop (the step cap, chunked prefill, first come first served, as at one priority), with a step time fitted to the box. It reproduces resume TTFT p50 and p90 within 8% at 8 and 16 agent sessions and is 18–30% optimistic at 24. Its arms are LRU with and without the CPU tier, TTL holds, prefetch timed by session history or by an oracle, simplified Tokencake, MORI and InferCept, and a clairvoyant reference (Belady's MIN in each tier). `flatshare-sim hardware` places an L4, an A100 40 GB and an H100 SXM on the same regime map, with the box's costs scaled by their published peaks, labeled as modeled. `flatshare-sim forecast` wrote the A100 run's forecast before it ran ([`forecast-80gb.md`](../benchmarks/a100/forecast-80gb.md)), which placed each mechanism and missed the KV case's waits by about 30% ([the A100 run](HARDWARE.md#what-it-measured-2026-10-26)).

<a id="6-tested-and-not-adopted"></a>

## 7. Designs measured and dropped

Two mechanisms were built, measured and not adopted, and the rest of the original design went with them ([ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md)):
- **Protection timed by the conversation, in the control plane** (E2). Under the lifecycle policy the gateway follows each call's turn from the worker's lifecycle events: it holds KV headroom from speech onset by deferring new background requests, throttles background prefill in the critical window from the end of the turn to the reply's first audio (priced by the step model), caps the reply at 16 tokens when the headroom was not there (degrade), and admits deferred work after 10 s (aging). The policy stays as an experiment arm, with its ablations ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)).
  - **What E2 found.** In the discriminating scenario (W4) these mechanisms could not act. Running agent requests held the pool, which the headroom can protect only by deferring new ones, so it never had its room (degrade capped 60 of 60 replies in every run), and the throttle, checked after the headroom, deferred nothing while the headroom ran. Room for the turn came from the engine alone: the same policy in front of stock vLLM left the tail where stock does (10,993 ms).
  - **What follows.** A control plane in front can make room only by aborting work it has admitted, as preempt-resubmit does ([METHODS.md](METHODS.md#what-lifecycle-timing-could-and-could-not-do)). Whether lifecycle timing helps where its mechanisms engage was not isolated.
- **Retention and prefetch of agents' KV by their lifecycle** (E1). In the simulator, even an oracle that knows when each agent returns from its tool closes under 9% of resume TTFT at p50 over LRU with the CPU tier, since the median resume is already a GPU hit or a short restore; at p90 it closes 28% at low pressure, but session history, what a deployment knows without the tools' return times, closes nothing ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)). The CPU-tier policy built for it, `LeasePolicy`, and the connector that carried its hint, `LeaseConnector`, were checked live on the GPU, never used, and removed after v0.3 ([the v0.3 tree](https://github.com/StevenWang-CY/flatshare/tree/v0.3/python/leaseholder_vllm) keeps them).
- **Leases.** The original design put leases in the control plane: each session's claim on KV residency, with prefetch timed from lifecycle events, a sidecar per GPU node with engine epochs, cache contracts from Anthropic's `cache_control`, and a ledger attributing every recomputed token to a cause. After E1 and E2 these were dropped: the two results above remove their premise, and the gateway's sticky placement on several nodes (below) does the routing leases were also for. They would be reopened by a source of tool-return times ahead of the return, which the oracle's tail gain needs, or by an engine whose running KV a control plane can reclaim. The types and plugins built for them (`flatshare-proto`'s lease hint and ledger rows with their schemas, the header that marked synthetic requests, the `flatshare_vllm` plugins) were removed after v0.3.
- **An admission point per engine.** The gateway places sessions on several GPU nodes (§3.5), but with `[[nodes]]` it runs no admission policy: an admission point per engine stays parked on the `two-node` branch.

## 8. Failure handling

| Failure | Behavior | Evidence |
|---|---|---|
| Engine crash | The unit restarts 5 s after its last process exits; streams in flight end with the gateway's `upstream_unavailable` error event, then one 502 and 503s with `Retry-After` until the engine's `/health` answers; sessions re-prefill on the next turn | Once in E2, an out-of-memory kill; the engine-kill drill of 2026-10-25: each stream got the error event 8–11 ms after the kill, and a request succeeded 106.2 s after it, 92.1 s of that the engine's warm start ([the drills](BENCHMARKS.md#failure-drills); [PRODUCTION.md](PRODUCTION.md#how-it-fails)) |
| Gateway restart or crash | `flatshare-gateway.socket` holds the listeners. A reload hands them to a successor and drains: nothing is refused or waits. A restart or a crash leaves new connections waiting in the backlog, for the drain or about a second. Background requests in flight are cut 5 s into a drain with a `draining` error | The binary's reload tests (a request every 20 ms on each socket through a reload, all answered); the gateway drills of 2026-10-25: a reload and a restart failed none of 42 requests, and a kill none of 4 ([the drills](BENCHMARKS.md#failure-drills)) |
| GPU node loss | With one node, as an engine crash: a request that finds the engine broken starts a probe of its `/health`, and the gateway answers 503 with `Retry-After` without going upstream until it answers. With `[[nodes]]` the node takes no new session, and each of its sessions is placed again by its next request on an available node of its role, its KV left behind; a realtime request whose node fails before its first content is sent once more on another node, and a node that covers defers its agents' requests beyond `failover_background_max_in_flight` (0 by default) until the moved calls end | The gateway's upstream and node tests; the failover drills, where both calls' next turns went to the spare 0.7 s after a kill and 1.9 s after a stop and none of the 300 turns after failed ([`2026-10-28-node-failover`](../benchmarks/results/2026-10-28-node-failover)) |
| Slow client | Backpressure through the bounded buffer, then an upstream abort | A stalled reader |
| Client disconnect | Upstream abort; vLLM aborts the request | Mid-stream disconnects |
| Host memory pressure | `systemd-oomd` may pick the engine; a campaign run again resumes at its failed point | [RUNBOOK.md](../ops/RUNBOOK.md#failures) |
| Another user's GPU work | `gpu-guard` refuses to start the box; a point measured beside foreign GPU work is measured again | [RUNBOOK.md](../ops/RUNBOOK.md#checks-before-heavy-work-on-a-shared-host) |

## 9. Testing

| Layer | Tests |
|---|---|
| `flatshare-proto` | Wire formats pinned by examples; schemas kept current |
| `flatshare-core` | The cost model against E0's cells, the KV geometry, the step-time model; each admission policy's decisions on hand-made sequences of requests and turns |
| `flatshare-sim` | The AIPerf port against AIPerf's own plan; the step loop on hand-made sequences; the clairvoyant tiers against exhaustive search; every arm end to end on a 16-trace subset; `make reproduce-e1` regenerates E1's tables (not in CI: the corpus stays outside the repository) |
| `flatshare-gateway` | Against in-process mock engines and speech servers: disconnects, backpressure, sanitization and tenant-scoped sessions, the intake's bounds, every deadline, the accounting and the engine's state, the nodes' placement, failover and covering, event idempotency, every admission policy, the reload and its drain; the proxy's overhead in process |
| Engine | The generated `make-room` module built from v0.32.0's source; its behavior on vLLM's real scheduler with CPU torch, and its trace against the PRs' own patches; the GPU probes |
| Voice | A hermetic LiveKit test in the voice-plane namespace (barge-in, headers, lifecycle events in order); unit tests of the event delivery, serve mode's admission and the node binding |
| Deploy and ops | The serving profile's rendering and `serving-check` against an engine that differs; the release tool's prepare, switch, switch-back and rollback on stand-in hosts; the units, install scripts and listener audit; the alert rules under promtool |
| E2 | The report, its figures and the demo renderer on synthetic points; `make reproduce-e2` and `make e2-figures` rebuild them from the committed records (not in CI) |
| CI | GitHub Actions on CPU: rustfmt, clippy and ruff, every Rust test, the engine variants' build tests against vLLM's source, the Python unit tests, and make-room's behavior and trace suites on CPU torch |
