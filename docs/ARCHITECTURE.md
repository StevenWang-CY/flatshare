# Architecture

Flatshare runs a realtime voice agent beside background LLM work on one GPU, or on two. This document describes what
is built, as of v1.25, for a reader who wants to know how the parts work and fit together before reading or changing
the code. [PRODUCTION.md](PRODUCTION.md) gives the configuration to deploy, [BENCHMARKS.md](BENCHMARKS.md) the numbers,
[PRIOR-ART.md](PRIOR-ART.md) what is prior work, the [ADRs](README.md#decisions) the decisions, and
[GLOSSARY.md](GLOSSARY.md) the terms.

Its central part is a change to vLLM's scheduler. With it a waiting realtime request can take KV from running
background requests, and background prefill sits out the realtime reply's first tokens. The change is proposed to vLLM as pull
requests [#65282](https://github.com/vllm-project/vllm/pull/65282) and
[#65283](https://github.com/vllm-project/vllm/pull/65283), open, with no maintainer review as of 2026-11-09. This page
and its figures name them by their first numbers, #63350 and #63351.

## At a glance

Flatshare is one change to vLLM's scheduler and three planes around it (Figure 1). In the serving plane the gateway
gives each client its class and priority and places it on a GPU node, where vLLM with #63350 and #63351 schedules
voice and agents over one KV pool. The measurement plane made every number the report gives, and the operations plane
holds production to the measured configuration.

<picture>
  <source media="(max-width: 767px)" srcset="figures/overview-narrow.svg">
  <img src="figures/overview.svg" width="100%" alt="Three panels around one engine change: in the serving plane, callers reach the voice worker and coding agents reach the gateway, which sends both to vLLM 0.32 with #63350 and #63351, the KV preemption and the budget cap, over the GPU's KV pool and the CPU tier beside the speech servers; beside it the measurement plane, with the scripted callers, AIPerf's replay of Claude Code traces, the E2 harness, probes and counters, gates, and the cost model and simulator; and the operations plane, with the serving profile and its serving-check, immutable releases, two cards with a warm spare, drain and failover, failure drills and a 24-hour soak, and the container.">
</picture>

**Figure 1. The change sits in the serving plane, and the measurement and operations planes stand beside it.** Blue is
the change, white a component, and a dashed outline the engine's KV memory. Each part's home in the repository and the
section that describes it:

| Plane | Part | In the repository | Here |
|---|---|---|---|
| Serving | The change on vLLM 0.32 and 0.33: the pull requests as patches, the build, probes and tests | [`python/flatshare_vllm`](../python/flatshare_vllm/README.md) | §2 |
| Serving | The gateway, and the wire types it speaks | [`crates/flatshare-gateway`](../crates/flatshare-gateway/README.md), [`crates/flatshare-proto`](../crates/flatshare-proto) | §3.3 |
| Serving | The voice worker, and the callers that measure it | [`python/flatshare_voice`](../python/flatshare_voice/README.md) | §3.4 |
| Serving | A GPU node's units and the engine's variants | [`deploy/phastform`](../deploy/phastform) | §3.2 |
| Measurement | The E2 harness, trace conversion, probes, drills and figures | [`python/flatshare_bench`](../python/flatshare_bench), [`benchmarks/`](../benchmarks) | §6.3 |
| Measurement | The statistics the records publish, each defined once | [`python/flatshare_stats`](../python/flatshare_stats) | §6 |
| Measurement | The cost and step-time models, and the KV cache simulator | [`crates/flatshare-core`](../crates/flatshare-core), [`crates/flatshare-sim`](../crates/flatshare-sim) | §6.4 |
| Operations | The serving profile, releases, two cards and the container | [`deploy/`](../deploy) | §3.1, §3.5, §3.6 |
| Operations | The runbook, the failure drills, alert rules and a dashboard | [`ops/`](../ops) | §8 |

## 1. The problem in the engine

A voice turn on a GPU shared with coding agents loses its time in the LLM's first sentence. Beside 24 agent sessions
on tuned vLLM 0.32, the reply's first sentence takes 3.9 s of a 5.3 s mean turn, against 0.3 s with voice alone
([W5's stage means](../benchmarks/results/2026-10-24-e2-w5/report/frontier.md#stage-waterfall)). Two mechanisms in the
scheduler cause it:

- **A waiting request cannot take KV.** vLLM 0.32 preempts only so that a running request can grow, and its waiting
  loop stops at the first request it cannot allocate ([#40004](https://github.com/vllm-project/vllm/issues/40004)).
  When running agents hold the pool, a voice turn waits for one of them to finish
  ([the KV case](BENCHMARKS.md#the-kv-case-make-room); [vLLM's former V0 scheduler](PRIOR-ART.md#vllm-0320-2026-10-18)).
- **Prefill shares the reply's steps.** Once the turn runs, every step of its reply also carries agent prefill chunks,
  up to about 270 ms each after a 30k-token context, so the first sentence decodes slowly.

Neither mechanism is specific to the card, though what each costs is
([the report's §6.7](REPORT.md#67-the-rules-carry-to-an-a100)).

## 2. The change

The variant `make-room` ([`python/flatshare_vllm`](../python/flatshare_vllm/README.md)) is built at each engine start
from the installed vLLM's scheduler, 0.32.0's or 0.33.0's, without modifying vLLM. `build.py` applies the pull
requests' scheduler diff, backported to that release, and the variant's own patch, and the engine loads the module
with `--scheduler-cls`. The serve module adds three parts outside the scheduler. Each switch is a key of
`--additional-config`, off by default, and those with no upstream flag are carried here only:

| Switch | Variant key | Upstream |
|---|---|---|
| The KV preemption | `make_room.max_preemptions_per_step` | `--max-priority-preemptions-per-step` ([#65282](https://github.com/vllm-project/vllm/pull/65282)) |
| The budget cap | `make_room.budget_cap`, `make_room.budget_cap_tokens` | `--priority-prefill-budget-cap`, `--priority-prefill-budget-cap-tokens` ([#65283](https://github.com/vllm-project/vllm/pull/65283)) |
| The reserve | `make_room.reserve_tokens` | — |
| A lower priority's reservations taken | `make_room.take_reservations` | — |
| Lower-priority decodes held from a call's first tokens | `make_room.decode_hold_tokens` | — |
| The calls' prefixes kept on the GPU | `make_room.retain_tokens`, `make_room.retain_s` | — |
| A realtime request's short load recomputed | `make_room.recompute_tokens` | — |
| A call's announced turn held for | `make_room.hint_hold_ms` | — |
| The next batch scheduled just in time | `flatshare_jit.margin_ms` and the card's step-time model (the serve module) | — |
| The agents' waiting queue ordered by the prefill left net of the cache | `flatshare_queue.max_wait_s` (the serve module) | in vLLM's own terms, the draft [#66558](https://github.com/vllm-project/vllm/pull/66558) |
| The CPU tier's loads through the SM copy kernel | `flatshare_copy.sm_load_programs` (the serve module) | the draft [#66527](https://github.com/vllm-project/vllm/pull/66527) |

What each does ([REPORT §3](REPORT.md#3-design) gives the two rules in full):

- **The KV preemption.** When the highest-priority waiting request lacks KV for its whole sequence, the scheduler
  preempts running requests of a strictly lower priority, the lowest priority and the latest arrival first, as few as
  make it fit. It takes at most N a step, none if N would not suffice, and their KV stays in the CPU tier
  ([§3.1](REPORT.md#31-make-room)).
- **The budget cap.** Lower-priority prefill sits out the steps of a request's first `budget_cap_tokens` output tokens
  (16 by default), and decodes are never cut ([§3.2](REPORT.md#32-the-prefill-hold)).
- **The reserve.** Requests of a positive priority value are admitted only while they leave the reserve free, a
  class-aware watermark.
- **The reservations taken.** vLLM admits a load from the CPU tier only beside every in-flight prefill's reservation.
  With this switch, under the budget cap, a request leaves free only the reservations of its own priority or a higher
  one, so a call's turn does not wait behind the agents' loads.
- **The decode hold.** Lower-priority decodes sit out a running request's first `decode_hold_tokens` output tokens, as
  its prefill does under the budget cap, since a step that also decodes an agent's long context is slower.
- **Retention.** A finished priority-0 request's blocks stay on the GPU, within `retain_tokens`, until its
  conversation's next turn takes them or `retain_s` passes, since a load from the CPU tier costs two or three steps.
- **The short load recomputed.** A priority-0 request whose CPU-tier hit extends its GPU hit by at most
  `recompute_tokens` recomputes the extension in its own step instead of loading it.
- **The announced turn.** The voice worker sends a one-token realtime request when the user stops speaking, and the
  budget cap holds lower-priority prefill for `hint_hold_ms` from it, so that the turn arrives into decode-only steps.
  Only the realtime class gets priority 0, so no agent can announce one.
- **Just in time.** The engine core schedules the next batch when the one in flight is due to end, less the margin,
  so that a call arriving during a long agent step joins the next batch.
- **The cache-aware queue.** Within a priority, the waiting request with the least prefill left, net of what the GPU
  pool and the CPU tier hold, goes first, and one that has waited `max_wait_s` goes ahead of the rest. In arrival
  order a session's next request can wait longer than the LRU tier keeps its prefix, and computes it all again.
- **The SM copy.** The CPU tier's loads copy through vLLM's own Triton kernel instead of the copy engine, which on the
  box's PCIe 3.0 x8 link holds a concurrent step's input copy until the whole load ends. Stores keep the copy engine.

E2 found the budget cap necessary, and either the KV preemption or the reserve enough for the rest
([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)). The window has to cover the reply's first sentence, 39
tokens at most for this voice agent ([the first sentences](../benchmarks/results/2026-10-27-first-sentence/README.md)),
so the configuration to deploy is the pull requests' switches with a 48-token window and no reserve.

## 3. The serving path

[PRODUCTION.md](PRODUCTION.md) gives the operator's view of the serving path, and [RUNBOOK.md](../ops/RUNBOOK.md) the
commands. Figure 2 shows its hosts and the sockets between them.

<picture>
  <source media="(max-width: 767px)" srcset="figures/architecture-narrow.svg">
  <img src="figures/architecture.svg" width="100%" alt="Three hosts as dashed groups. On cyan, the control host, the callers, the LiveKit server and the voice worker sit in the voice plane's network namespace, which has no route out. The worker sends its LLM requests to the gateway through gateway.sock and its speech to the tunnel to Phastform through stt.sock and tts.sock; the coding agents reach the gateway through 127.0.0.1:18080, TCP with a tenant's key; the gateway sends its engine requests through engine.sock to the tunnel to Phastform, and, dashed, to the tunnel to Gray. Each tunnel carries SSH sessions to its node's three servers: vLLM 0.32 with make-room on engine.sock, shaded as the change, Whisper STT on stt.sock and Kokoro TTS on tts.sock. On Phastform, an RTX 5060 Ti, each server has a network namespace of its own; Gray, a second card when present, is drawn dashed, its servers in the host's namespace.">
</picture>

**Figure 2. Outside the voice plane the units meet only at Unix sockets, which SSH bridges between hosts, and the
gateway's port is the only TCP listener in cyan's host network.** Dashed groups are hosts and network namespaces, and
the change's engine is blue. A tunnel per node serves the node's sockets on cyan as `NODE-engine.sock`,
`NODE-stt.sock` and `NODE-tts.sock` (§4). Gray, dashed, is the second card (§3.5).

### 3.1 The serving profile

`deploy/profiles/rtx-5060-ti-serve.toml` names the configuration to serve
([ADR-0009](adr/0009-immutable-releases-and-the-exact-serving-profile.md)). It holds the frozen box (§5.1) with the
configuration to deploy, the sweep's 256-token step cap and the §2 switches it turns on. It also holds the scheduler
module's hash, two calls and 24 agent sessions a card, and real callers' audio. `deploy/profile.py` renders it into
`deploy/phastform/serving.json`, every setting the engine must run with, and into the `make-room-serve` variant.

- **Held at every start.** The serve module's scheduler writes the configuration it runs to a file as the engine
  starts. `deploy/bin/serving-check`, the unit's `ExecStartPost`, compares that file with serving.json and fails the
  start on any difference, naming it. A release runs the same check, and holds the voice worker's admission and the
  gateway's nodes to the profile ([the serving profile](../deploy/phastform/variants/README.md#the-serving-profile)).
- **No development routes.** A serving box runs vLLM without its development and RL routes (§4).

### 3.2 A GPU node

A GPU node runs the three servers of §5.1, the engine on the serving variant. Each is a `systemd --user` unit, and
`flatshare.target` starts the three together. Each serves on a Unix socket in `$XDG_RUNTIME_DIR/flatshare` and listens
on no TCP port (§4). `gpu-guard` refuses to start a unit while another user's process holds the GPU.

### 3.3 The gateway

`flatshare-gateway` ([`crates/flatshare-gateway`](../crates/flatshare-gateway/README.md)) is the serving control plane,
on the control host ([ADR-0007](adr/0007-the-gateway-is-the-serving-control-plane.md)). It serves the OpenAI routes
and the Anthropic Messages API, through which Claude Code connects. A request takes the steps of Figure 3.

<picture>
  <source media="(max-width: 767px)" srcset="figures/request-path-narrow.svg">
  <img src="figures/request-path.svg" width="100%" alt="A column of seven steps, each step's main refusals leaving to its right with their status and code. Authenticate, on no known key or an unknown class: 401 invalid_api_key, 400 invalid_class. Bound, on a bound full or a body too large: 503 in_flight_limit, 429 tenant_in_flight_limit, 413 body_too_large. Sanitize, on a body that is no JSON object or asks too many choices: 400 invalid_json, 400 too_many_choices. Place, with no engine ready or no free slot: 503 upstream_not_ready, no_healthy_node, no_call_slot, no_agent_slot. Forward, on no answer or no headers in time: 502 upstream_unavailable, 504 upstream_timeout. Relay, on a stream cut short: an error event of type upstream_unavailable, upstream_timeout or draining. Account records one outcome for every request. The gateway's error table lists every code.">
</picture>

**Figure 3. A request is refused at the first step it fails, and the first four refuse it before the engine sees it.**
Once a stream has begun, a refusal ends it with an error event instead
([every code](../crates/flatshare-gateway/README.md#errors)). A passed deadline, or a client that leaves, cancels the
engine's work.

| Step | What it does |
|---|---|
| Authenticate | The key gives the tenant and its class, capped by its `max_class`; the headers the engines trust go ([PROTOCOL.md](PROTOCOL.md#headers)) |
| Bound | In flight in all, by class and by tenant, before the body is read, then its bytes; past a bound, refused at once with `Retry-After` |
| Sanitize | The body's fields the engines trust go, the class's reply cap and `max_choices` bound it, and its session is resolved ([PROTOCOL.md](PROTOCOL.md#sessions)) |
| Place | With `[[nodes]]`, its session's node, or a node with its role and a free slot (§3.5); with one engine, while that engine is ready |
| Forward | With the class's priority where the engine reads it (`X-Vllm-Priority`), `X-Session-ID` and the tenant's `cache_salt` |
| Relay | Through a bounded buffer, under the class's deadlines for the first content, each frame after it and the whole reply |
| Account | Once: offered, admitted and one outcome, with a service level of good over valid requests by class |

The engine's state comes from its requests' successes and an idle-only canary, and `/ready` answers by it. A reload
(SIGHUP) hands the listeners and the placements to a successor without a gap. Only E2 configures an `[admission]`
policy (§6.1).

### 3.4 The voice worker

LiveKit server 1.13 and LiveKit Agents 1.8.8, with Silero VAD on CPU, run the voice agent on the control host, in a
network namespace with no route out (§4) ([`python/flatshare_voice`](../python/flatshare_voice/README.md)). STT and TTS
reach the GPU node through the tunnels, and the LLM goes through the gateway's socket as the realtime class.

In serve mode, which the deployment runs, the worker admits at most the profile's two calls at once, and none while the
gateway's `/ready` says the engine does not serve or a release drains it. It places each call on a GPU node with
`POST /v1/sessions`, binds the call's STT and TTS to that node, and frees the slot with `DELETE /v1/sessions/{id}` when
the call ends. It hears real callers with LiveKit's end-of-turn model, and writes its state every second, which
`serving-check` holds to the profile.

### 3.5 Two cards

Where a second card is available, each card is a `[[nodes]]` entry of `gateway.toml`
([ADR-0008](adr/0008-two-cards-consolidation-with-a-warm-spare.md);
[`deploy/cyan/gateway.example.toml`](../deploy/cyan/gateway.example.toml) has both). Figure 4 shows the placement and
the failover.

<picture>
  <source media="(max-width: 767px)" srcset="figures/two-cards-narrow.svg">
  <img src="figures/two-cards.svg" width="100%" alt="Two panels of two cards, each with a row of two call slots and a row of agents' slots, four on Phastform and twelve on Gray. (a) In service: both calls fill Phastform's call slots beside its four agents' slots, and Gray, the warm spare, runs agents only, its call slots empty, under calls = fill. (b) The first card failed: Phastform's engine is down and its slots empty; both calls take Gray's call slots at their next turn, 0.7 s after a kill and 1.9 s after a stop, and Gray, covering, holds its twelve agents' slots, hatched, under a bracket that reads held until the calls end.">
</picture>

**Figure 4. Calls fill the first card in the file's order, and a failed card's calls are placed again on the other by
their next turn.** The spare covers while it carries any call, holding its agents' requests, hatched, until those calls
end ([the failover drills](../benchmarks/results/2026-10-28-node-failover)).

- **Health.** A node is healthy while its whole pipeline is: its engine serving and ready, and its STT and TTS
  answering `/health`.
- **Sessions stay on their node,** since KV never moves. A session whose node fails is placed again by its next
  request, and a realtime request whose node failed before its first content is sent once more on the other node.
- **Covering.** While the spare carries any call, its agents' requests wait (`failover_background_max_in_flight`, 0).
- **Draining.** A node set `draining = true` takes no new session; its calls end on it and its agent sessions move by
  their next request, until `/nodes` says `"drained": true` ([Nodes](../crates/flatshare-gateway/README.md#nodes)).

### 3.6 Releases

`deploy/deploy.sh HOST` makes an immutable release under `~/flatshare/releases/`, with a manifest of what it holds
([ADR-0009](adr/0009-immutable-releases-and-the-exact-serving-profile.md)), and takes it through the states of
Figure 5. The drain is of what the switch would cut, the worker's calls and the engine's requests. The switch moves the
`current` link and restarts only the units whose inputs changed, and the check runs those of §3.1.

<picture>
  <source media="(max-width: 767px)" srcset="figures/release-narrow.svg">
  <img src="figures/release.svg" width="100%" alt="A column of a release's states from deploy.sh HOST: prepare, verify, drain, switch, restart, check and serving. Prepare exits 2 with nothing switched when the host is locked or the upload changed; verify exits 1 with nothing switched when a check fails; drain exits 2 with nothing switched when calls do not end or an E2 point runs. A failure in the switch, the restart or the check leads to switch back, which exits 1 with the release before running again, or 3 when switching back failed too. Serving exits 0, and --rollback enters at verify.">
</picture>

**Figure 5. A release switches nothing until it has verified and drained, and switches back by itself when anything
after the switch fails.** `deploy.sh`'s exit status says where it stopped, and `--rollback` takes the previous release
through the same states from the verification.

### 3.7 What to watch

The gateway's `/metrics`, and each engine's that it relays by node, feed Prometheus
([`ops/prometheus`](../ops/prometheus)). Two objectives, the realtime first byte's service level and the gateway's
overhead, each have a fast and a slow burn-rate alert ([SLO.md](SLO.md)), and health alerts in
[`rules.yml`](../ops/prometheus/rules.yml) cover the gateway, the engines, the nodes and the KV pool. A Grafana
dashboard shows the same ([`ops/grafana`](../ops/grafana/flatshare.json)), and
[PRODUCTION.md](PRODUCTION.md#what-to-watch) says what to watch.

<a id="33-network-and-security"></a>

## 4. Network and security

The services keep off the hosts' networks ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md)):

- **Unix sockets end to end.** On the GPU nodes the engine, STT and TTS serve on Unix sockets in a directory of mode
  0700, and listen on no TCP port. vLLM's `--api-key` guards only its `/v1`, `/v2`, `/inference` and `/cohere` routes,
  so the directory is what keeps the hosts' other users from the engine. cyan reaches each socket through
  `flatshare-tunnel@NODE`, by an SSH key that may run only the bridge to that socket.
- **One TCP listener on cyan.** The gateway's 127.0.0.1:18080, with keys, is the only TCP listener in cyan's host
  network. `deploy/bin/audit-listeners` checks each host after every deploy for any other listener of the box's units.
- **The servers' own network namespaces.** Each GPU server runs in a network namespace with a loopback alone
  ([`deploy/bin/in-private-net`](../deploy/bin/in-private-net)), since vLLM's engine cores listen on the loopback and
  CUDA keeps abstract sockets that every user of a host's namespace reaches. Where the kernel refuses unprivileged user
  namespaces, as on Gray, the servers run in the host's namespace (`FLATSHARE_PRIVATE_NET=host`).
- **The voice plane** runs in a network namespace with no route out, holding loopback and one dummy interface
  (10.255.0.1/32), since libwebrtc gathers no ICE candidate on loopback alone.
- **No development routes, no KV events.** A production box runs vLLM without its development and RL routes, such as
  `/collective_rpc`; a measured box adds them on the engine's socket only (`variant.sh measure on`). The engine
  publishes no KV cache events, which carry every cached block's token ids to whoever binds their endpoint.
- **Sanitized inputs.** The gateway strips the request fields and headers that the engines trust, and sets its own:
  the class's priority, `X-Session-ID` and the tenant's `cache_salt`, so that tenants never share prefix-cache entries
  ([PROTOCOL.md](PROTOCOL.md#headers)).
- **KV never crosses the network.** A session that loses its engine's KV prefills again.
- **What stays open.** The sockets keep the hosts' other users out, not the deploying user's other processes
  ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md#unix-sockets-end-to-end-w6-2026-10-25)), and the
  `sglang` variant listens on 127.0.0.1:8000, which every user of the node reaches.

## 5. The measurement configuration

<a id="31-the-frozen-box"></a>

### 5.1 The frozen box

Every measurement on the box runs one pinned configuration, with the audio models on the same GPU
([`deploy/phastform`](../deploy/phastform); measured values in [BENCHMARKS.md](BENCHMARKS.md#the-frozen-box)). The
serving profile is the same box, running the change at its own step cap (§3.1):

| Component | Configuration | GPU memory |
|---|---|---|
| LLM | vLLM 0.32.0, `Qwen/Qwen3-4B-Instruct-2507-FP8`, 32k context, `--kv-cache-memory` 5 GiB (36,400 tokens), step cap 512/256 (note 1) | ≈ 10.6 GiB |
| STT | vLLM 0.32.0 serving `openai/whisper-large-v3-turbo` at `/v1/audio/transcriptions`, `--kv-cache-memory` 437.5 MiB (4,020 tokens) | ≈ 2.4 GiB |
| TTS | Kokoro-82M on Kokoro-FastAPI 0.9.1-rc1 at `/v1/audio/speech` | ≈ 1.0 GiB |

1. The LLM also runs with prefix caching, priority scheduling, KV cache metrics and the `OffloadingConnector` (20 GiB
   in `/dev/shm`, write-through, generated tokens included).

- **Units.** `flatshare.target` starts the three servers at once. Both vLLM servers fix their KV pools, so neither
  profiles memory at its start, and the budgets fit the card together ([HARDWARE.md](HARDWARE.md#the-rules), R2).
- **NUMA and environment.** vLLM runs on the GPU's node (node 1, cores 8–15) and the audio servers on node 0. vLLM
  0.32.0 runs on torch 2.13.0+cu130 and Kokoro on torch 2.8.0+cu128, pinned in `deploy/phastform/requirements/`.
- **Time-slicing, not MPS.** As MPS clients STT and TTS gain a little, but the LLM stage and its tail get worse
  ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)).
- **Engine variants** are drop-ins that `deploy/phastform/variant.sh` switches
  ([the variants](../deploy/phastform/variants/README.md)). Besides `make-room` they include vLLM #54530 ported to
  0.32.0, SGLang 0.5.22, MPS, `agents-only` (the speech servers' 3.4 GiB in the KV pool), and an arm file's flags.

### 5.2 Hosts

cyan, the control host, has 48 threads and no GPU. It runs `flatshare-gateway`, LiveKit and the voice worker, and in a
measurement AIPerf's agent replay and the callers, each on its own cores. Phastform and Gray each have an RTX 5060 Ti
16 GB and run the frozen box. Phastform, the deployment's first card, ran every E2 point before the two-card rounds but
the A100 run's, and later rounds ran on it and Gray, its warm spare, in turn. peacock builds the Linux release binaries
and runs the simulator.

Another GPU node's files come from a hardware profile (`deploy/profiles/`, rendered by `deploy/profile.py`). A host
without cyan beside it runs the GPU servers and the control plane together, on loopback ([HARDWARE.md](HARDWARE.md)),
as the A100 run did in a Google Colab container ([the A100 run](BENCHMARKS.md#on-google-colab)).

## 6. The measurement harness and the simulator

### 6.1 E2's admission policies

Under an `[admission]` policy, which only E2's arms configure, the gateway holds a request between its bounds and its
placement until the policy admits it ([Admission](../crates/flatshare-gateway/README.md#admission)). The policies are
E2's baselines (a background cap, a token reservation, llm-d-style flow control with eviction, and preempt-resubmit)
and the lifecycle policy (§7), and the gateway refuses them beside `[[nodes]]`. They live in `flatshare-core`, pure
code with no I/O and no clock, beside the models of the frozen box that `flatshare-sim` runs: the E0 tier cost model,
KV geometry and a step-time model.

<a id="42-the-voice-pipeline"></a>

### 6.2 The voice pipeline in measurement

E2 ran the worker of §3.4 in bench mode, which takes every call LiveKit dispatches, and ended each turn a fixed delay
after the end of speech, since LiveKit's end-of-turn model waits its 3 s maximum on some synthesized lines. The worker
posts each turn's lifecycle events for E2's lifecycle policy and logs each reply's stage times. The measurement callers
(`flatshare-voice bots`) play utterances with known end times and read V2V from the waveforms on one clock.

### 6.3 E2's harness

`flatshare-bench e2` ([`benchmarks/e2`](../benchmarks/e2/README.md)) runs an arm file as a point. It switches the GPU
node to the arm's engine and checks the engine's `/server_info` against the arm, so that no variant passes for
another, then runs AIPerf's agent replay and the calls and saves the point's records. A campaign measures its arm files
in order and resumes where it stopped. `flatshare-bench e2 export` writes each point's compact record into
[`benchmarks/results`](../benchmarks/results/2026-10-23-e2/README.md), from which `make reproduce-e2` rebuilds the
report and `make figures` the docs' figures.

### 6.4 The simulator (E1)

`flatshare-sim` ([`crates/flatshare-sim`](../crates/flatshare-sim/README.md)) replays the SemiAnalysis Claude Code
traces, block-scaled 32× (32k context), as AIPerf's `agentic_replay` sends them, through a model of vLLM 0.32's
two-tier prefix cache and its step loop, with a step time fitted to the box. Its arms run from LRU with and without the
CPU tier to a clairvoyant reference (Belady's MIN in each tier). `flatshare-sim hardware` places an L4, an A100 40 GB
and an H100 SXM on the same regime map, labeled as modeled, and `flatshare-sim forecast` wrote the A100 run's forecast
before it ran ([`forecast-80gb.md`](../benchmarks/a100/forecast-80gb.md)).

<a id="6-tested-and-not-adopted"></a>

## 7. Designs measured and dropped

Two mechanisms were built, measured and not adopted, and the rest of the original design went with them
([ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md)):

- **Protection timed by the conversation, in the control plane** (E2). The lifecycle policy follows each call's turn
  from the worker's events, holding KV headroom from speech onset and throttling background prefill until the reply's
  first audio. With running agents holding the pool these could not act, and the policy in front of stock vLLM left the
  tail where stock does
  ([the report's §6.5](REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work)).
  It stays as an experiment arm.
- **Retention and prefetch of agents' KV by their lifecycle** (E1). In the simulator, session history, which a
  deployment knows without the tools' return times, gains nothing over LRU with the CPU tier
  ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)). The policy and connector built for it were removed after v0.3
  ([the v0.3 tree](https://github.com/StevenWang-CY/flatshare/tree/v0.3/python/leaseholder_vllm) keeps them).
- **Leases.** The original design gave each session a claim on KV residency in the control plane. The two results above
  remove its premise, and the gateway's sticky placement does the routing leases were also for. A source of tool-return
  times, or an engine whose running KV a control plane can reclaim, would reopen it.
- **An admission point per engine.** With `[[nodes]]` the gateway runs no admission policy; an admission point per
  engine stays parked on the `two-node` branch.

## 8. Failure handling

| Failure | Behavior | Evidence |
|---|---|---|
| Engine crash | Streams in flight end with an `upstream_unavailable` error event, then one 502 and 503s until the engine's `/health` answers; the unit restarts 5 s after its last process exits | The engine-kill drill (note 1) |
| Gateway restart or crash | `flatshare-gateway.socket` holds the listeners. A reload hands them to a successor and drains, refusing nothing; background requests are cut 5 s into a drain | The reload tests; the gateway drills (note 2) |
| GPU node loss | With one node, as an engine crash. With `[[nodes]]` the node takes no new session, and each of its sessions is placed again by its next request | The node tests; the failover drills (§3.5) |
| Slow client | Backpressure through the bounded buffer, then an upstream abort | A stalled reader |
| Client disconnect | Upstream abort; vLLM aborts the request | Mid-stream disconnects |
| Host memory pressure | `systemd-oomd` may pick the engine; a campaign run again resumes at its failed point | [RUNBOOK.md](../ops/RUNBOOK.md#failures) |
| Another user's GPU work | `gpu-guard` refuses to start the box; a point measured beside foreign GPU work is measured again | [RUNBOOK.md](../ops/RUNBOOK.md#checks-before-heavy-work-on-a-shared-host) |

1. Each stream got the error event 6–11 ms after the kill command, and a request succeeded 106.2 s after it, most of
   that the engine's warm start ([the drills](BENCHMARKS.md#failure-drills);
   [PRODUCTION.md](PRODUCTION.md#how-it-fails)).
2. The reload tests send a request every 20 ms through a reload, all answered. In the gateway drills a reload and a
   restart failed none of 42 requests, and a kill none of 4.

## 9. Testing

| Part | What its tests hold |
|---|---|
| `flatshare-proto` | The wire formats, pinned by examples, and the schemas |
| `flatshare-core` | The cost model against E0's cells, the KV geometry and the step-time model, and each admission policy's decisions on hand-made sequences |
| `flatshare-sim` | The AIPerf port against AIPerf's own plan, the step loop, the clairvoyant tiers against exhaustive search, and every arm on a 16-trace subset |
| `flatshare-gateway` | Against in-process mock engines and speech servers: backpressure, sanitization, the bounds and deadlines, the accounting, the nodes' placement and failover, every admission policy, and the reload |
| make-room | The module built from v0.32.0's and v0.33.0's source, its behavior on vLLM's real scheduler with CPU torch, and its trace against the pull requests' own patches step for step (`make test-make-room-ci`) |
| The voice worker | A hermetic LiveKit test in the voice-plane namespace, and unit tests of the event delivery, serve mode's admission and the node binding |
| Deploy and ops | The profile's rendering and `serving-check`, the release tool on stand-in hosts, the units, install scripts and listener audit, and the alert rules under promtool |
| E2 | The report, its figures and the demo renderer on synthetic points |

CI runs the Rust and Python unit tests, the engine variants' build tests and make-room's suites on CPU, with rustfmt,
clippy and ruff. The hermetic LiveKit test (`pytest -m integration`), the promtool tests (`make check-rules`),
`make reproduce-e1`, `make reproduce-e2` and the GPU probes
([`probe.py`, `tail.py`](../python/flatshare_vllm/src/flatshare_vllm/make-room)) run outside it.
