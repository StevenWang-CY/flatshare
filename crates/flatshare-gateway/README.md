# flatshare-gateway

`flatshare-gateway` is the serving control plane in front of one vLLM engine, or of several GPU nodes
([ADR-0007](../../docs/adr/0007-the-gateway-is-the-serving-control-plane.md)). This README is its reference for
integrators and operators: what it does to a request, its configuration, errors and metrics, and what it costs.
[ARCHITECTURE.md §3.3](../../docs/ARCHITECTURE.md#33-the-gateway) places it in the deployment, and
[the runbook](../../ops/RUNBOOK.md#the-gateway) installs and operates it.

The gateway carries Flatshare's request contract: who a request belongs to, its session and SLO class, and which
engine inputs a client may not set. It bounds, times and accounts every request, relays responses byte for byte, and
cancels the engine's request when the client leaves. It serves one upstream, or several GPU nodes with sticky sessions,
on loopback listeners. The configuration to deploy sets each request's priority from its class
([PRODUCTION.md](../../docs/PRODUCTION.md#the-priorities)) and runs no admission policy, which only E2's harness
configures ([Admission](#admission)).

```sh
flatshare-gateway --config ~/.config/flatshare/gateway.toml
```

[`deploy/cyan/gateway.example.toml`](../../deploy/cyan/gateway.example.toml) is cyan's configuration,
[`flatshare-gateway.service`](../../deploy/cyan/systemd/flatshare-gateway.service) its unit, and
[`flatshare-gateway.socket`](../../deploy/cyan/systemd/flatshare-gateway.socket) the socket unit that holds its
listeners.

## What it does

<picture>
  <source media="(max-width: 767px)" srcset="../../docs/figures/request-path-narrow.svg">
  <img src="../../docs/figures/request-path.svg" width="100%" alt="A column of seven steps a request takes through the gateway, each step's main refusals leaving to its right, which the gateway's error table gives in full: authenticate (401 invalid_api_key, 400 invalid_class), bound (503 in_flight_limit, 429 tenant_in_flight_limit, 413 body_too_large), sanitize (400 invalid_json, 400 too_many_choices), place (503 upstream_not_ready, no_healthy_node, no_call_slot or no_agent_slot), forward (502 upstream_unavailable, 504 upstream_timeout), relay (an error event of type upstream_unavailable, upstream_timeout or draining once a stream has begun), and account, which records one outcome for every request.">
</picture>

**A request is refused at the first step it fails, and the first four refuse it before the engine sees it.** Once a
stream has begun, a refusal ends it with an error event instead ([Errors](#errors)).

- **Routes.** `POST /v1/chat/completions`, `/v1/completions`, `/v1/responses`, and the Anthropic Messages API's
  `/v1/messages` and `/v1/messages/count_tokens` are proxied with session, class and sanitization, and `GET /v1/models`
  with sanitized headers. `POST /v1/events` takes lifecycle events, and `/v1/sessions` and `/nodes` serve
  [Nodes](#nodes). `GET /health` and `/ready` need no key, and `GET /metrics` and `/metrics/engine` (the engine's own,
  relayed) need the metrics key on a TCP listener. Other paths are 404 and wrong methods 405.
- **Claude Code** reaches it with `ANTHROPIC_BASE_URL` set to the gateway and the tenant's key in
  `ANTHROPIC_AUTH_TOKEN`, sent as a bearer token. A key in `x-api-key` is neither read nor forwarded.
- **Listeners** are loopback TCP ports and Unix sockets, and a TCP address beyond loopback is refused at start. Only a
  Unix listener may name a tenant for requests without a known key, since every user of the host reaches loopback.
  Sockets systemd passes (`LISTEN_FDS`) are matched to their `[[listen]]` entries, and one that no entry names stops
  the start. The gateway's own Unix sockets get mode 0600, in a directory that must be closed to other users (mode 700).
- **Tenants and keys.** `Authorization: Bearer <key>` is hashed with SHA-256 and compared with every digest in constant
  time. A known key selects its tenant on any listener, and otherwise the request is the listener's tenant's or gets
  401.
- **Sanitization.** Every request loses the engine-control headers (`X-Vllm-Priority`, `X-SMG-Priority`,
  `X-data-parallel-rank`, and `X-Session-ID` unless the tenant is trusted), `X-Request-Id`, every `X-Flatshare-`
  header, `X-Claude-Code-Session-ID`, `Authorization` and the hop-by-hop headers, those `Connection` names among
  them. Every JSON body loses vLLM 0.32's `session_id`, `priority`, `kv_transfer_params`, `ec_transfer_params`,
  `cache_salt`, `request_id` and `vllm_xargs`, and SGLang 0.5.22's `rid`, `extra_key`, `session_params`,
  `custom_labels`, `routed_dp_rank`, `data_parallel_rank`, `disagg_prefill_dp_rank`, `bootstrap_host`,
  `bootstrap_port` and `bootstrap_room`. Only the body's top level is parsed, its keys compared after unescaping, and
  `flatshare_gateway_sanitized_total` counts each removal.
- **What it sets.** `X-Session-ID` carries the resolved session, and `X-Request-Id` a fresh id that the client gets
  too. The body's `cache_salt` is the first 32 hex digits of HMAC-SHA-256 of the tenant's name, keyed by
  `cache_salt_secret_file`'s secret. The class's priority goes in the signals `upstream_priority` lists:
  `X-Vllm-Priority` by default (realtime 0, interactive 1, background 2), the body's `priority`, or `X-SMG-Priority`
  (`system`, `interactive`, `bulk`). Each inference response carries `X-Flatshare-Session-ID`.
- **Sessions.** The session is the first present source in [PROTOCOL.md](../../docs/PROTOCOL.md#sessions) order: a
  trusted tenant's `X-Session-ID`, `X-Claude-Code-Session-ID`, the body's `prompt_cache_key`. A trusted tenant's valid
  id is used as it is. Any other value becomes `h-<first 32 hex of SHA-256(tenant, NUL, value)>`, so no tenant can name
  another's session, and with no source the gateway mints `m-<32 random hex>`.
- **Classes.** A trusted tenant may choose one with `X-Flatshare-Class`, an unknown value being 400. Otherwise the
  tenant's `default_class` applies, and a class stricter than its `max_class` is clamped.
- **An engine that restarts.** A request that finds the upstream broken, with no connection or one closed before the
  response headers, gets 502 `upstream_unavailable` and starts a probe of the engine's `/health`. Until it answers,
  requests get 503 `upstream_not_ready` with `Retry-After: 1` without going upstream. A timeout starts no probe, since
  the engine is slow, not gone.
- **Relay.** A pump reads the upstream body into a bounded channel (`stream_buffer_frames`), so a slow client slows
  the engine's socket rather than filling memory. A channel full for `stall_timeout_s` aborts the request. A client that
  leaves drops the upstream request, which vLLM aborts. A stream the upstream breaks off gets vLLM's own error event,
  `data: {"error": {..., "type": "upstream_unavailable", "param": null, "code": 502}}`, then an abort, never a clean
  end.
- **Usage** is read from JSON bodies and SSE `data:` lines for every route: `prompt_tokens` or `input_tokens`,
  `completion_tokens` or `output_tokens`, and `cached_tokens`. The Messages API's prompt includes its cache reads and
  writes, and its `stop_reason` is logged as the matching `finish_reason`. The stream is never rewritten.
- **Lifecycle events.** `POST /v1/events` takes one `flatshare_proto::LifecycleEvent` or an array, from trusted tenants
  only (403 otherwise), validated in full before any is recorded. Events are deduplicated by
  `(tenant, source, event_id)` within `event_dedup_capacity` ids and `event_dedup_ttl_s`. The answer is 202
  `{"accepted": n, "duplicates": m}`.

### Bounds

Each bound holds with or without an admission policy. A request beyond one is answered at once and never queued, so a
flood costs the gateway neither memory nor a wait.

| Bound | Key | Default | Beyond it |
|---|---|---|---|
| A reply's tokens, by class | `[classes.<class>] max_reply_tokens` | none | The cap, in the route's own limit fields |
| A request's choices, `n` and `best_of` | `max_choices` | 1 | 400 `too_many_choices` |
| A request's body | `max_body_bytes` | 32 MiB | 413 |
| Inference requests in flight | `max_in_flight` | 1,024 | 503 `in_flight_limit`, `Retry-After: 1` |
| A class's requests in flight | `[classes.<class>] max_in_flight` | realtime 128, interactive 256, background 512 | 503 `class_in_flight_limit`, `Retry-After: 1` |
| A tenant's requests in flight | `[tenants.<name>] max_in_flight` | 512 | 429 `tenant_in_flight_limit`, `Retry-After: 1` |
| The bytes of every body held | `max_buffered_body_bytes` | 256 MiB | 503 `body_budget_exhausted` |
| Requests waiting at the admission point | `max_queued` | 256 | 429 `queue_full` |

| Route | The limit vLLM 0.32 reads | A capped body gets |
|---|---|---|
| Chat Completions | `max_completion_tokens`, or `max_tokens` when that is null or absent | Both, at the same value |
| Completions | `max_tokens` alone: 16 when it is absent, no limit when it is null | `max_tokens` |
| Responses | `max_output_tokens` alone | `max_output_tokens` |
| Messages | `max_tokens`, which it requires; count_tokens none, since it generates nothing | `max_tokens` |

- **The reply cap** writes its route's fields at the lower of the client's limit and the cap, and removes those its
  route does not read. A limit or choice field set twice, or to anything but null or a whole number, is a 400 naming
  the field.
- **A place in flight** is taken before the body is read and held to the request's end. The class defaults keep room for
  each other, since interactive and background together reach 768 of the gateway's 1,024 and leave realtime its 128.
- **The defaults never refuse a recorded workload.** E2's heaviest point held about 30 requests in flight, with bodies
  of about 130 KB.
- **[`tests/fixtures/reply-caps.json`](tests/fixtures/reply-caps.json)** holds 380 bodies as the gateway caps them,
  which `python/flatshare_vllm/tests/test_gateway_caps.py` checks against vLLM 0.32.0's own request models
  (`make test-make-room`). `make gateway-fixtures` writes it anew.

### Deadlines

Every wait a request goes through has a deadline from its class's `[classes.<class>]` table, and each wait ends at the
earliest that applies (`src/deadline.rs`).

| Wait | Key | Default | Then |
|---|---|---|---|
| The body's upload, from the receipt of its headers | `upload_timeout_s` | 60 s | 408 `upload_timeout` |
| Each wait at the admission point | `queue_timeout_s` | 120 s, after the policy's own `queue_timeout_s` (60 s) or `max_defer_s` (10 s) | 503 `queue_timeout`, `Retry-After: 1` |
| Each attempt's response headers | `headers_timeout_s` | The top-level `headers_timeout_s`, 600 s | 504 `upstream_timeout` |
| The first content, from the request's receipt | `first_content_timeout_s` | realtime 60 s, the others 600 s | 504 `upstream_timeout`, or the error event once the response has begun |
| Each frame after the first | `idle_timeout_s` | realtime 30 s, the others 600 s | The error event and an abort |
| The whole request, from its receipt to its response's end | `total_timeout_s` | realtime 300 s, interactive 1,800 s, background 3,600 s | 504 `upstream_timeout`, or the error event once the response has begun |

- **A passed deadline** aborts the engine's work, gives back the request's place, and is recorded once. A stream that
  has begun gets the error event of type `upstream_timeout` and an abort, never `[DONE]`.
- **The first content and the total count from the receipt**, so a retry never buys more time.
- **The defaults are a few times the slowest recorded p99s.** Background first bytes ran to 232 s at p99 behind
  preemptions, in W5's resubmitting arm, realtime ones to 13.4 s in the worst arm, and no run met the 600 s headers
  timeout.

### Stop, reload and exit

| Event | What the gateway does | Exit |
|---|---|---|
| SIGTERM or SIGINT | Stops the listeners, drains in-flight requests for up to `drain_timeout_s` (30 s), background ones for `background_drain_s` (5 s), and removes its socket files | 0 |
| SIGHUP | Reloads without a gap: a successor takes the listening sockets, and the old process drains as for a stop | 0, the old process |
| A successor that exits (a broken configuration, say) or does not take over within 10 s | Changes nothing: the old process serves on, and `systemctl --user reload flatshare-gateway` fails | — |
| A configuration error | Stops the start, naming the file and the cause | 2 |
| A listener that cannot bind, or an inherited socket that no listener names | Stops the start | 1 |

- **The drain** cuts a background stream with the `draining` error event, and answers a request without a response
  503 `draining` with `Retry-After: 1`. Requests still running when `drain_timeout_s` passes end with the process.
- **The reload** starts the successor from the binary at the gateway's path, with its arguments and listening sockets.
  The successor tells systemd it is the main process and ready (`MAINPID=`, `READY=1`), then sends its predecessor
  SIGTERM, and both accept until then.
  [`deploy/bin/flatshare-gateway-reload`](../../deploy/bin/flatshare-gateway-reload) waits for the main process to
  change.
- **One admission point over both processes** (`src/ledger.rs`). The predecessor admits nothing more once it hands its
  sockets over, and answers what still waits there 503 `draining`. The successor counts what the predecessor still
  holds in its own bounds until it exits (`flatshare_gateway_predecessor_in_flight`), and serves only once the
  predecessor admits nothing more, 2 s at most. No request reaches both.
- **The listeners outlive the process.** A change to `[[listen]]` takes a restart. Under socket activation, connections
  wait in the backlog until the next gateway serves.

## What it does not do

- **Leases, prefetch or a recompute ledger.** The lease design was not pursued
  ([ADR-0006](../../docs/adr/0006-the-engine-change-carries-the-realtime-tier.md)).
- **Retries or per-request balancing.** Only a realtime request whose node fails before its first content is sent
  again ([Nodes](#nodes)). A session stays on the node that holds its KV, and no admission policy runs with
  `[[nodes]]`.
- **TLS.** Every listener binds loopback or a private socket, and the upstream is a tunnel's socket or a loopback port.
- **Turns from traffic, or the Responses API's stop reasons.** `turn_started` and `turn_finished` come only from
  lifecycle events.
- **A Responses API `background` request.** vLLM 0.32 refuses it unless `VLLM_ENABLE_RESPONSES_API_STORE=1`, which no
  Flatshare configuration sets, since such a response would run outside the gateway's bounds.
- **A bound on idle connections**, or an answer to a request sent on a keep-alive connection that a draining gateway
  closes. That request never reached the gateway, so a client may resend it, as the voice worker does once.

## Configuration

Unknown keys are errors, so a misspelled setting fails at start instead of taking its default, and so is a duration
(a key that ends in `_s`) over 1e9 seconds.

| Key | Default | Meaning |
|---|---|---|
| `upstream` | required without `[[nodes]]` | The engine, `http://host:port` or `unix:` and its socket's absolute path, over HTTP/1.1 with pooled connections |
| `upstream_idle_timeout_s` | 4 | How long an idle upstream connection is kept, below the engine's keep-alive (`VLLM_HTTP_TIMEOUT_KEEP_ALIVE`, 5 s by default). On cyan it is 55 s, with the engine's at 60 s, so that voice turns reuse their connections ([BENCHMARKS.md](../../docs/BENCHMARKS.md#what-does-the-gateway-add-to-a-request)) |
| `upstream_priority` | `["vllm-header"]` | Where the class's priority goes: any of `"vllm-header"`, `"body"` and `"smg-header"`, or none (`[]`) |
| `cache_salt_secret_file` | required | 32 random bytes in hex that key every tenant's `cache_salt` (note 1) |
| `metrics_key_file` | required with a TCP listener | The key `/metrics` and `/metrics/engine` ask for on a TCP listener (note 1) |
| `max_body_bytes` | 33554432 (32 MiB) | [Bounds](#bounds) |
| `max_buffered_body_bytes` | 268435456 (256 MiB) | [Bounds](#bounds), at least `max_body_bytes` |
| `max_in_flight` | 1024 | [Bounds](#bounds), Envoy's circuit breakers' default |
| `max_queued` | 256 | [Bounds](#bounds), in all the admission point's queues |
| `max_choices` | 1 | [Bounds](#bounds): `n`, the beam's width with `use_beam_search`, and `best_of`. vLLM runs every choice, each a reply's worth of decode and KV |
| `stream_buffer_frames` | 64 | Frames buffered between the upstream and a client |
| `stall_timeout_s` | 30 | How long that buffer may stay full before the stream is aborted |
| `headers_timeout_s` | 600 | Every class's default `headers_timeout_s`. Its old name, `first_byte_timeout_s`, is refused |
| `drain_timeout_s` | 30 | How long a stop waits for requests in flight |
| `background_drain_s` | 5 | How far into a drain background requests go on, from 0 to `drain_timeout_s` |
| `event_dedup_capacity` | 100000 | Lifecycle-event ids remembered for deduplication |
| `event_dedup_ttl_s` | 600 | How long an event id is remembered |
| `log_format` | `"text"` | `"text"` or `"json"` (one object per line) |
| `log_level` | `"info"` | `error`, `warn`, `info`, `debug` or `trace` |

1. An absolute path to a file that `deploy/cyan/install.sh` writes once, mode 600. A missing file, one other users may
   open, or one that holds anything else stops the start. The secret keeps a client that reaches the engine directly
   from computing another tenant's salt, and a new one orphans every tenant's prefix cache.

Each `[classes.<class>]` table (`realtime`, `interactive`, `background`) sets:

| Key | Default | Meaning |
|---|---|---|
| `max_reply_tokens` | none | The cap on the class's replies, in tokens ([Bounds](#bounds)) |
| `max_in_flight` | realtime 128, interactive 256, background 512 | [Bounds](#bounds) |
| `upload_timeout_s`, `queue_timeout_s`, `headers_timeout_s`, `first_content_timeout_s`, `idle_timeout_s`, `total_timeout_s` | As in [Deadlines](#deadlines) | [Deadlines](#deadlines) |
| `first_content_objective_s` | realtime 0.25, the others none | The class's objective in the service level ([Accounting](#accounting)). None: completing is enough |

The `[canary]` table sets the inference canary ([The engine's state](#the-engines-state)):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` sends none and leaves the engine `serving`, as E2's points run |
| `interval_s` | 30 | How long no inference may succeed before the canary asks; every second while the engine warms |
| `timeout_s` | 60 | How long one canary may take, at the lowest class's priority |
| `degraded_after` | 3 | The canaries failed in a row, with no success between, that make the engine `degraded` |

Each `[[nodes]]` entry is a GPU node ([Nodes](#nodes)). With any, `upstream` and `[admission]` are not set, and some
node must take each role:

| Key | Default | Meaning |
|---|---|---|
| `name` | required | From `[A-Za-z0-9._-]`, the metrics' `node` label |
| `upstream` | required | Its engine, as the top-level `upstream`: `unix:/run/user/1007/flatshare/gray-engine.sock` |
| `roles` | `["realtime", "background"]` | `realtime` takes calls, `background` the agents' interactive and background requests |
| `max_calls` | 2 | The calls placed on it at most, positive with the realtime role |
| `max_agents` | required with both roles | The agents' sessions placed on it at most, a positive count or `"unbounded"`. A background-only node takes any number unless it sets one |
| `stt`, `tts` | none | Its speech servers, as `upstream`. Its health needs their `/health` |
| `draining` | false | Whether it [drains](#nodes) |

The `[placement]` table, which needs `[[nodes]]`, sets:

| Key | Default | Meaning |
|---|---|---|
| `idle_s` | 300 | How long a session keeps its node without a request |
| `health_interval_s` | 1 | How often each node is checked |
| `health_timeout_s` | 1 | How long a speech server's `/health` may take |
| `unhealthy_after` | 3 | The failed checks in a row that make a node unhealthy |
| `healthy_after` | 2 | The passed checks in a row that make it healthy again |
| `failover_background_max_in_flight` | 0 | The interactive and background requests in flight on a node that covers; 0 defers them all |
| `calls` | `"spread"` | Where a call goes: `"spread"` to the node with the fewest calls, `"fill"` to the first in the file's order with a free slot |

Each `[[listen]]` entry sets exactly one of `tcp = "127.0.0.1:18080"`, a loopback address, or
`unix = "/absolute/path.sock"`, and a Unix entry may set `tenant`. Each `[tenants.<name>]` table (names from
`[A-Za-z0-9._-]`) sets:

| Key | Default | Meaning |
|---|---|---|
| `keys_sha256` | `[]` | Hex SHA-256 digests of the tenant's API keys. A tenant with no keys must be named by a listener |
| `trusted` | `false` | May set `X-Session-ID` and `X-Flatshare-Class`, post lifecycle events, and place and end sessions |
| `default_class` | `"interactive"` | The class of requests that choose none |
| `max_class` | `default_class` | The strictest class the tenant may use |
| `max_in_flight` | 512 | [Bounds](#bounds), half the gateway's, so that one tenant's flood leaves the others room |

The `[admission]` table, which only E2's harness sets, is in [Admission](#admission).

## Nodes

With `[[nodes]]` the gateway fronts several GPU nodes, each an engine and its speech servers behind cyan's tunnels
(`src/nodes/`). KV never moves between them, so each session is placed on one node and stays there.

- **Whole-pipeline health.** A node passes a check while its engine serves by its own idle-only canary and is ready,
  and its STT and TTS answer `/health`. `unhealthy_after` failed checks make it unhealthy and `healthy_after` passed
  ones healthy again. A node whose engine a request found broken is passed over at once.
- **Sticky placement.** A session is placed by its first request, or by `POST /v1/sessions`, on an available node with
  its class's role. A call takes one of the node's `max_calls` slots, as `[placement] calls` chooses. An agent's
  session goes to a node that does not cover, then to the one with the fewest agents' sessions. A session stays until
  it has gone `idle_s` without a request, or its tenant ends it by `DELETE /v1/sessions/{id}` or `session_closed`. At
  most 4,096 sessions are remembered, the untrusted tenants' at most half.
- **Agents beside calls.** No admission policy holds agents back with `[[nodes]]`, and one card meets the voice SLO
  beside its calls only with a bounded number of agents, [the profile's
  envelope](../../docs/PRODUCTION.md#the-envelope-from-version-124). So a node with `max_agents` holds at most that
  many agents' sessions. An agent's session that finds every available background node at its bound gets 503
  `no_agent_slot`, and its next request tries again. A request that names no session takes no slot, and a bounded
  node takes it only while it carries no call and is below its bound.
- **Failover.** A node that is not available takes no new session, and a session placed on it moves by its next
  request, its KV left behind (`reason="moved"`). A session that finds no available node of its role with a free slot
  gets 503 `no_call_slot`, `no_agent_slot` or `no_healthy_node`.
- **Covering.** A node that carries realtime sessions moved to it covers. Its interactive and background requests in
  flight are held to `failover_background_max_in_flight`, 0 by default, and the others wait within their class's
  `queue_timeout_s` until the moved calls end. They wait even once the failed node is back, so that the calls never run
  beside the survivor's whole agent load. With `calls = "fill"` a later realtime node also covers while it carries any
  call, a warm spare.
- **Interrupted turns.** A realtime request is held back until its first body frame. If its node fails before then, it
  is sent once more on another realtime node within its original deadlines (`reason="retried"`), or answered 502
  `upstream_unavailable`. After its first content it is never sent again, so a voice reply is never a second generation
  appended to half-delivered audio.
- **Draining.** A node set `draining = true` takes no new session once a reload takes the key. Its calls stay until
  they end and its agents' sessions move. `/nodes` says `"drained": true` once it holds no session and no request, and
  `flatshare_gateway_node_draining` keeps `FlatshareNodeUnhealthy` quiet
  ([the runbook's steps](../../ops/RUNBOOK.md#drain-one-node)).
- **Readiness.** `GET /ready` answers 200 while each role has an available node that does not drain, else 503
  `no healthy node for realtime` (or `background`), or `gateway drained`. `GET /metrics/engine?node=NAME` relays one
  node's engine.
- **A reload** moves no session, since the successor takes the placements from the ledger
  ([`src/ledger.rs`](src/ledger.rs)).

| Route, from a trusted tenant | Answer |
|---|---|
| `POST /v1/sessions` with `{"session_id": "voice-RM_a1B2c3", "class": "realtime"}` | 200 `{"session_id": "voice-RM_a1B2c3", "class": "realtime", "node": "gray", "stt": "unix:/run/user/1007/flatshare/gray-stt.sock", "tts": "unix:/run/user/1007/flatshare/gray-tts.sock"}`, the node's speech servers or `null`. `class` defaults to the tenant's `default_class`. Asking again answers the same node while the placement holds, so a call that asks each turn learns of a move |
| `DELETE /v1/sessions/{id}` | 200 `{"session_id": "voice-RM_a1B2c3", "ended": true}`, or `"ended": false` for a session without a placement, so ending one twice succeeds. The slot frees at once, and a node that covered for it covers no more |
| `GET /nodes`, under the metrics' rule | `{"nodes": [...], "sessions": n, "placement": {...}}`. Each node gives its `name`, `roles`, `verdict` (`unknown`, `healthy` or `unhealthy`), `available`, `engine` (`address`, `state`, `ready`, last `check`), `stt`, `tts`, `max_calls`, `max_agents`, `covering`, `draining` and `drained`, and its `sessions` and `in_flight` by class. The placement gives `calls` and `failover_background_max_in_flight`, which a release checks (`deploy/bin/serving-check --nodes`) |

The voice worker places each call before it speaks, binds its STT and TTS to the node's sockets, and ends the
placement when the call ends or its job does not run, so that no slot waits out `idle_s`.

## Errors

Responses from the engine pass through unchanged. The gateway's own errors come in two shapes:

- **An answer**, sent before any response has begun, has OpenAI's format, `{"error": {"message", "type", "code"}}`,
  and carries `X-Request-Id`. Its `code` names the failure, and its `type` is the status's OpenAI class.
- **An error event** ends an event stream that has begun, in vLLM's own shape,
  `data: {"error": {"message", "type", "param", "code"}}`. Its `type` names the failure (`upstream_unavailable`,
  `upstream_timeout` or `draining`), and its `code` is the status.

[`tests/fixtures/errors.json`](tests/fixtures/errors.json) holds the errors the voice worker reads in both shapes.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_class`, `invalid_json`, `invalid_event`, `invalid_session`, `unreadable_body` | An unknown `X-Flatshare-Class`, a body that is no JSON object or cannot be read, or an invalid lifecycle event or session |
| 400 | `duplicate_field`, `invalid_reply_limit`, `invalid_choices`, `too_many_choices` | A limit or choice field set twice, or to anything but null or a whole number; more choices than `max_choices` |
| 400, 404 | `node_required`, `unknown_node` | With `[[nodes]]`, `GET /metrics/engine` without `?node=NAME`, or naming no node |
| 401 | `invalid_api_key` | No known key on a listener without a tenant, or no metrics key for `/metrics` on a TCP listener |
| 403 | `untrusted_tenant` | Lifecycle events or the sessions routes from an untrusted tenant |
| 404, 405 | `not_found`, `method_not_allowed` | Any other route or method |
| 408 | `upload_timeout` | The body did not arrive within its class's `upload_timeout_s` or `total_timeout_s` |
| 413 | `body_too_large` | A body over `max_body_bytes` |
| 429 | `queue_full`, `shed` | A full admission queue, with `Retry-After: 1`; a background request shed by flow control |
| 429 | `tenant_in_flight_limit` | The tenant has its `max_in_flight` requests in flight, with `Retry-After: 1` |
| 502 | `upstream_unavailable` | The engine could not be reached or closed the connection before answering, or `/metrics/engine` got no answer in 10 s |
| 503 | `queue_timeout`, `upstream_not_ready` | Not admitted in time; the engine found broken and its `/health` silent since. Each with `Retry-After: 1` |
| 503 | `no_healthy_node`, `no_call_slot`, `no_agent_slot`, `placement_full` | With `[[nodes]]`, no node of the session's role available, no free slot, or a full session table. Each with `Retry-After: 1` |
| 503 | `in_flight_limit`, `class_in_flight_limit`, `body_budget_exhausted`, `draining` | The gateway or class at its `max_in_flight`, the body budget full, or a draining gateway. Each with `Retry-After: 1` |
| 504 | `upstream_timeout` | No response headers, first content or end in time, before a response began |

## Accounting

Every inference request is counted three times, as offered when its tenant and class are known, as admitted when it
goes upstream, and as ended with its one outcome. Admitted over offered is the admission coverage, which the
dashboard shows beside the admitted requests' latencies, since refusing requests would keep those latencies low.

| Outcome | How the request ended |
|---|---|
| `completed` | A 2xx response relayed to its end: the only success |
| `upstream_client_error`, `upstream_server_error` | A response relayed to its end with a 4xx status, such as a prompt over the context, or with another status |
| `transport_error` | The upstream failed: no connection, one closed before the response headers, or a body that broke off |
| `cancelled`, `stalled` | The client left first; it stopped reading for `stall_timeout_s` |
| `shed`, `drained` | A policy gave its place to a stricter class; a drain cut it or kept it out |
| `timeout_upload`, `timeout_queue`, `timeout_headers`, `timeout_first_content`, `timeout_idle`, `timeout_total` | One of its [deadlines](#deadlines) passed |
| `refused_in_flight`, `refused_class`, `refused_tenant`, `refused_body_bytes`, `refused_queue` | A bound or the admission queues refused it before its body was read |
| `not_ready` | The engine was not ready or, with `[[nodes]]`, no node could take it |
| `invalid` | Refused as malformed or unauthorized: 400, 401, 404, 405 or 413 |

The service level counts, by class, the valid requests and the good ones, and `ops/prometheus/rules.yml`'s objective
reads 1 − good/valid.

- **Valid** is every request that ended, but for the client's faults: `invalid`, `upstream_client_error`,
  `timeout_upload`, `refused_tenant` and `stalled`.
- **Good** is `completed` with its first content within the class's `first_content_objective_s`, so a request that
  fails, is refused for capacity, times out or hangs is a miss.
- **A cancellation** in a class with an objective is good or bad by its first content if it had one, and otherwise bad
  once the objective has passed and not valid before. A class without an objective leaves its cancellations out.

## The engine's state

vLLM 0.32's `/health` answers while its processes live and runs no inference, so an engine whose generation stalls
looks healthy to it. The gateway keeps the engine's state as inference finds it (`src/canary.rs`):

- `warming` from the gateway's start until an inference succeeds;
- `serving` once one has, and again after any success;
- `degraded` once the canary has failed `degraded_after` times in a row with nothing succeeding between;
- `drained` from the start of the gateway's drain.

`GET /ready` answers 200 only while the engine is `serving` and its `/health` answered within the last second, and
otherwise 503 with `engine warming`, `engine degraded` or `engine drained`, or with `upstream not ready` while its
`/health` is silent or a request has found it broken. Requests go upstream in every state. An inference succeeds when a
2xx response's first content arrives or the canary answers. The canary is a one-token completion at the background
class's priority, with `X-Flatshare-Canary: 1`, sent only when no inference succeeded within `interval_s`, and outside
every bound, count and outcome above. A gateway that has just started or reloaded sends its first canary at once, so it
is ready as soon as that succeeds.

## Metrics and logs

`GET /metrics` serves the OpenMetrics text format, and `GET /metrics/engine` relays the engine's own within 10 s.
Label values come from fixed sets or the configuration, and every series exists from the start at zero, but for
requests by status, lifecycle events and the requests offered, admitted and ended. [`metrics.txt`](metrics.txt) is
what a gateway under the lifecycle policy exports before any traffic, which a test keeps current (`make metrics`
writes it) and the checks of `ops/` read.

| Metric | Labels | Meaning |
|---|---|---|
| `flatshare_gateway_requests_total` | `tenant`, `route`, `class`, `status` | API requests; 499 means the client left before any response |
| `flatshare_gateway_request_outcomes_total` | `class`, `outcome` | Inference requests with a class, one outcome each, which the outcome alerts read |
| `flatshare_gateway_requests_offered_total` | `tenant`, `route`, `class` | Inference requests received, each once |
| `flatshare_gateway_requests_admitted_total` | `tenant`, `route`, `class`, `admission` | Inference requests sent upstream, each once |
| `flatshare_gateway_requests_terminal_total` | `tenant`, `route`, `class`, `outcome` | Inference requests ended, each once, by outcome |
| `flatshare_gateway_sli_valid_total` | `class` | The requests the service level counts |
| `flatshare_gateway_sli_good_total` | `class` | The valid requests that met their class's objective |
| `flatshare_gateway_overhead_seconds` | `route` | The end of a request's body to its write upstream: parsing, sanitization, admission and connection checkout, without an admission wait. Its buckets start at 50 µs |
| `flatshare_gateway_request_body_seconds` | `route` | Receipt of a request's headers to the end of its body |
| `flatshare_gateway_first_byte_seconds` | `class` | Receipt to the first response body byte from the upstream |
| `flatshare_gateway_duration_seconds` | `class` | Receipt to the end of the upstream response |
| `flatshare_gateway_streams_active` | `class` | Responses being relayed |
| `flatshare_gateway_in_flight` | `class` | Inference requests past the intake and not yet ended |
| `flatshare_gateway_buffered_body_bytes` | — | Bytes of request bodies held |
| `flatshare_gateway_predecessor_in_flight` | — | A reload's predecessor's requests in flight |
| `flatshare_gateway_sanitized_total` | `field` | Inputs removed because clients may not set them |
| `flatshare_gateway_class_clamped_total` | — | Requests whose class was clamped to their tenant's `max_class` |
| `flatshare_gateway_upstream_errors_total` | `kind` | `connect`, `request`, `timeout`, `body` |
| `flatshare_gateway_error_events_total` | `class`, `type` | Event streams that got an error event, by its type |
| `flatshare_gateway_events_total` | `type`, `source`, `duplicate` | Lifecycle events received |
| `flatshare_gateway_usage_tokens_total` | `class`, `kind` | `prompt`, `completion` and `cached` tokens from response usage |
| `flatshare_gateway_admission_admitted_total` | `class` | Requests the admission point let go upstream (note 1) |
| `flatshare_gateway_admission_waiting` | `class` | Requests waiting at the admission point |
| `flatshare_gateway_admission_wait_seconds` | `class` | Arrival at the admission point to admission |
| `flatshare_gateway_admission_refused_total` | `class`, `reason` | `queue_full` and `max_queued` (429), `timeout` and `draining` (503) |
| `flatshare_gateway_admission_resubmits_total` | `class` | Requests sent again after a preemption dropped their attempt |
| `flatshare_gateway_admission_discarded_tokens_total` | `class` | Estimated prompt tokens of the dropped attempts |
| `flatshare_gateway_admission_deferred_total` | `reason` | Background requests the lifecycle policy deferred: `headroom` or `critical` (note 1) |
| `flatshare_gateway_admission_deferral_seconds` | `reason` | Their arrival to admission |
| `flatshare_gateway_admission_aged_total` | `oversize` | Deferred requests admitted after `max_defer_s` |
| `flatshare_gateway_admission_degraded_total` | — | Realtime replies capped at `degrade_max_tokens` |
| `flatshare_gateway_admission_false_alarms_total` | — | Armings of the headroom that no realtime request followed |
| `flatshare_gateway_admission_false_alarm_token_seconds_total` | — | Background tokens held back during false alarms, times how long |
| `flatshare_gateway_realtime_sessions` | `state` | Realtime sessions by state |
| `flatshare_gateway_realtime_state_timeouts_total` | `state` | Sessions that stayed in a state for `state_timeout_s` |
| `flatshare_gateway_node_ready` | — | 1 while the engine is ready, 0 while it answers requests 503 |
| `flatshare_gateway_engine_state` | `state` | 1 for the engine's state, 0 for the others |
| `flatshare_gateway_canary_probes_total` | `result` | Inference canaries, by `success` or `failure` |
| `flatshare_gateway_canary_seconds` | — | The time a canary took |

1. The admission series exist only with an `[admission]` policy, and the lifecycle policy's, `admission_deferred_total`
   through `realtime_state_timeouts_total`, only under it, so that a count a gateway does not keep reads as absent
   rather than zero.

With `[[nodes]]` the engine's four families, `node_ready`, `engine_state`, `canary_probes_total` and `canary_seconds`,
carry a `node` label, and the nodes add theirs. [`metrics-nodes.txt`](metrics-nodes.txt) is what a gateway of two nodes
exports.

| Metric | Labels | Meaning |
|---|---|---|
| `flatshare_gateway_node_healthy` | `node` | 1 while the node is healthy |
| `flatshare_gateway_node_component_healthy` | `node`, `component` | Each component's last check: `engine`, `stt`, `tts` |
| `flatshare_gateway_node_sessions` | `node`, `class` | The sessions placed on the node |
| `flatshare_gateway_node_in_flight` | `node`, `class` | Its inference requests in flight |
| `flatshare_gateway_node_covering` | `node` | 1 while it covers |
| `flatshare_gateway_node_failover_waiting` | `node` | Its agents' requests waiting while it covers |
| `flatshare_gateway_node_draining` | `node` | 1 while it drains |
| `flatshare_gateway_placements_total` | `node`, `class`, `reason` | Sessions placed: `new`, `moved` or `retried` |
| `flatshare_gateway_placements_refused_total` | `class`, `reason` | Sessions no node took: `no_node`, `no_call_slot`, `no_agent_slot`, `table_full` |
| `flatshare_gateway_node_retries_total` | `node` | Realtime requests sent elsewhere after this node failed before their first content |

**Each request logs one line at its end**, under `flatshare_gateway::request` and at level `warn` for 5xx,
`transport_error`, `stalled` and the upstream's deadlines. It gives `request_id`, `tenant`, `session`, `node`,
`class`, `route`, `status`, `outcome`, `first_byte_ms`, `total_ms`, the token counts and `finish_reason`, and where
they apply `admission_wait_ms`, `deferred`, `aged`, `degraded`, `resubmits`, `retries` and `error`.

Where a name says first byte, it times the first content, the response body's first frame from the request's receipt:
`flatshare_gateway_first_byte_seconds`, `first_byte_ms` and `ops/`'s `FlatshareRealtimeFirstByteBudget*` alerts. It is
the moment `first_content_timeout_s` bounds and `first_content_objective_s` judges. The names stay, since records, the
dashboard and the runbook's anchors read them.

## Admission

The admission point and its policies are E2's harness (`src/harness/`), and a gateway without an `[admission]` table
runs none of it. It sits between resolving a request and the upstream request. `flatshare_core::admission` decides
from the order of events alone, and the gateway does the waiting, the timeouts and the cancellation. A policy holds
some classes and lets the others go at once. Within a class requests go in arrival order, behind any waiting request
of a stricter class.

The `[admission]` table sets the admission policy. A key of another policy than the chosen one is an error, and so,
under `lifecycle`, is a key that no switched-on mechanism reads.

| Key | Default | Policies | Meaning |
|---|---|---|---|
| `policy` | `"none"` | — | `"none"`, `"static-cap"`, `"token-reservation"`, `"flow-control"`, `"preempt-resubmit"` or `"lifecycle"` |
| `background_max_in_flight` | required | `static-cap` | Background requests in flight at most |
| `pool_tokens` | 36400 | `token-reservation`; `lifecycle` with `headroom` or `degrade` | The KV pool in tokens, the frozen box's by default |
| `reserved_tokens` | required | `token-reservation` | Pool tokens background requests leave free |
| `ceilings` | required | `flow-control` | `{ realtime = 16, interactive = 12, background = 8 }`, not increasing from realtime to background |
| `headroom` | `true` | `lifecycle` | Keep KV headroom for the armed realtime sessions |
| `headroom_tokens` | required with `headroom` or `degrade` | `lifecycle` | The KV of one realtime turn, prompt and reply |
| `trigger` | `"onset"` | `lifecycle` with `headroom` | What arms the headroom: `"onset"` or `"request"` |
| `throttle` | `true` | `lifecycle` | Budget the background prefill admitted in a critical window |
| `critical_prefill_budget_ms` | required with `throttle` | `lifecycle` | That budget, in predicted prefill time; 0 admits none |
| `max_num_batched_tokens` | 512 | `lifecycle` with `throttle` | The engine's `--max-num-batched-tokens`, in whose steps a prefill is priced |
| `degrade` | `true` | `lifecycle` | Cap the reply of a realtime request that finds less than `headroom_tokens` free |
| `degrade_max_tokens` | required with `degrade` | `lifecycle` | That cap, in tokens |
| `max_defer_s` | 10 | `lifecycle` with `headroom` or `throttle` | How long a background request may be deferred before it is admitted regardless |
| `state_timeout_s` | 10 | `lifecycle` | How long a realtime session stays in a state of its turn that no event moves on |
| `queue_limit` | 1024 | all but `none`; `lifecycle` with `headroom` or `throttle` | Requests each class's queue holds; one more gets 429 |
| `queue_timeout_s` | 60 | all but `none` and `lifecycle` | How long a request may wait before it gets 503 |

The first four policies are E2's baselines, what an operator would try and what routers ship, run so that every arm
has the same engine, clients and measurement:

| Policy | E2 arm | What it holds | What it reproduces, and what it does not |
|---|---|---|---|
| `static-cap` | 6 | At most `background_max_in_flight` background requests in flight | An operator's background concurrency limit |
| `token-reservation` | 6 | A background request goes while the estimated background tokens in flight, its own and `reserved_tokens` fit in `pool_tokens` | The equal-size static reservation |
| `flow-control` | 8 | Strict priority bands, each dispatched while fewer requests than its ceiling are in flight. A blocked realtime or interactive request sheds the newest background ones in flight, which get 429 `shed` | llm-d v1.0's flow control with eviction (`flowControl.enableEviction`), its utilization a count of requests and its eviction immediate, without pacing, fairness or byte limits |
| `preempt-resubmit` | 8 | A realtime arrival drops the background requests without a first byte, which go again once every realtime request has its first byte | A router's abort and resubmit with SMG's rule. SMG itself (arm 8b) answers the preempted request 503 instead |

- **When shedding is not enough**, a realtime request waits like any other and can be refused, which the lifecycle
  policy never does ([ADR-0003](../../docs/adr/0003-realtime-is-never-shed.md)).
- **A preempted request's client** sees one response, whose first byte comes later.
- **The token estimate** is a session's last reported `prompt_tokens` plus the body's growth since, at 4 bytes per
  token, with no tokenizer, and a body smaller than the session's last is estimated whole. It counts prompts, not
  decode growth, so it is a proxy for the pool's use.
- **Its error.** On W2's agent replay at 8, 16 and 24 sessions (415 to 1,521 requests) it was off by 0.4 to 0.5% of all
  prompt tokens, against 1.9 to 2.2% for bytes / 4. Of the requests of 1,000 tokens or more, 90% were within 1.3 to 1.6%
  (bytes / 4: 3.4 to 4.5%) and all within 9.4% (bytes / 4: 8.2%). AIPerf's synthetic prompts run at 3.9 to 4.0 bytes a
  token, near the estimate's rate. `cargo run --release -p flatshare-core --example estimate_error --
  RUN/aiperf/profile_export.jsonl...` replays an export.
- **The session table** keeps at most 4,096 sessions, the untrusted tenants' at most half, and replaces the one seen
  longest ago that is not in a turn.

### The lifecycle policy

`policy = "lifecycle"` is what E2's arm 9 runs with make-room. Realtime requests go upstream at once and are never
queued or refused, background requests yield to the realtime sessions' turns, and interactive requests are not held.

Accepted lifecycle events move a realtime session through `idle`, `user_speaking` (`user_speech_started`), `critical`
(`user_turn_ended`), `playback` (`playback_started`) and back to `idle` (`playback_ended`). `interrupted` is the user
speaking again, and `session_closed` forgets the session. A realtime request also puts its session in `critical`. Every
state but `idle` ends after `state_timeout_s` without an event or a request.

| Switch | Keys | What it does |
|---|---|---|
| Headroom | `headroom`, `headroom_tokens`, `trigger` | While sessions are armed, a background request goes only if the estimated background tokens in flight and its own leave `headroom_tokens` free for each. A session is armed from speech onset to the reply's first audio, or in its critical window alone with `trigger = "request"` |
| Critical-window throttle | `throttle`, `critical_prefill_budget_ms`, `max_num_batched_tokens` | While any session is `critical`, the background requests admitted share `critical_prefill_budget_ms` of predicted prefill time |
| Degrade | `degrade`, `degrade_max_tokens` | A realtime request that finds less than `headroom_tokens` free is still admitted at once, its reply capped at the lower of `degrade_max_tokens` and its class's cap |
| Aging | `max_defer_s` | A background request deferred that long is admitted regardless, `oversize` when its estimate alone exceeds what the headroom leaves |

- **The step model** prices a prefill by the steps it takes alone. On the frozen box 512 tokens take 269 ms after a
  30,000-token context and 38 ms after none, a difference a token budget cannot see. The fit is compiled in
  (`flatshare_core::StepCost::frozen_box`) and no key changes it, so another engine or GPU needs its own, from
  `flatshare-sim`'s `scripts/fit_step_cost.py`, built into `flatshare-core`.
- **It counts** deferrals, aged and degraded admissions, sessions by state, state timeouts, and false alarms, armings
  of the headroom that no realtime request followed.
- **What it leaves out.** Its headroom works from the gateway's estimate, where the engine's own reserve counts blocks
  exactly. No event says that speech ended without a turn, so a false alarm keeps its headroom until
  `state_timeout_s`.
- **What E2 measured.** In a campaign of 2026-10-23 on one sample of the agents' traces, arm 9 with the change landed
  within the spread of its own runs without the headroom, the throttle or the events, or with `trigger = "request"`,
  while without degrade's reply cap its p95 rose from 2.06 to 3.84 s. On the stock engine the policy left voice where
  stock vLLM does
  ([why](../../docs/REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work);
  [EXPERIMENTS.md](../../docs/EXPERIMENTS.md#e2-voice-under-agent-load)).

## Overhead

**On a deployed build, version 1.8, the gateway added about a tenth of a millisecond.** Paired against the engine's
own socket on cyan, it added +0.12 ms to a voice request's first token at the median over 80 pairs (p95 +1.22 ms).
None of 82 requests spent over 2 ms in its own leg
([the record](../../benchmarks/results/2026-10-29-gateway-window/README.md);
[BENCHMARKS.md](../../docs/BENCHMARKS.md#what-does-the-gateway-add-to-a-request)).

`cargo run --release -p flatshare-gateway --example overhead` measures its latency and CPU in one process: a mock engine
that answers at once, the client and two proxies, each on 4 tokio worker threads, over loopback TCP. Each scenario runs
direct, through a bare reverse proxy on the same stack (the HTTP hop alone), and through the gateway with a key, a
session and its log line. "Added" is the gateway's quantile minus the direct one, and CPU is the proxy's busy time per
request or 8-chunk stream. On 16 logical CPUs of cyan's two Xeon Silver 4310 at 2.1 GHz, on 2026-10-22, p50 / p99:

| Scenario | Clients | Samples | Direct, µs | Bare proxy, µs | Gateway, µs | Added, µs | CPU per request, bare / gateway, µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-streaming, 2 KiB body | 1 | 5,000 | 44 / 54 | 156 / 171 | 270 / 306 | 226 / 251 | 80 / 191 |
| Non-streaming, 2 KiB body | 8 | 40,000 | 43 / 123 | 107 / 212 | 125 / 395 | 81 / 271 | 27 / 57 |
| Non-streaming, 2 KiB body | 64 | 40,000 | 243 / 481 | 690 / 1,405 | 927 / 2,400 | 684 / 1,919 | 40 / 55 |
| Non-streaming, 64 KiB body | 1 | 1,000 | 36 / 44 | 71 / 80 | 381 / 433 | 345 / 389 | 32 / 254 |
| Non-streaming, 64 KiB body | 8 | 8,000 | 106 / 178 | 153 / 321 | 206 / 527 | 100 / 349 | 41 / 109 |
| Streaming, first byte | 1 | 500 | 95 / 129 | 199 / 258 | 270 / 357 | 175 / 228 | 300 / 508 |
| Streaming, per chunk | 1 | 3,500 | 31 / 40 | 61 / 71 | 65 / 85 | 34 / 45 | 300 / 508 |
| Streaming, first byte | 8 | 4,000 | 97 / 156 | 230 / 342 | 303 / 421 | 206 / 265 | 302 / 452 |
| Streaming, per chunk | 8 | 28,000 | 44 / 81 | 92 / 158 | 94 / 162 | 51 / 82 | 302 / 452 |
| Streaming, first byte | 64 | 4,000 | 143 / 404 | 422 / 1,865 | 440 / 1,874 | 296 / 1,469 | 247 / 292 |
| Streaming, per chunk | 64 | 28,000 | 72 / 223 | 210 / 775 | 234 / 893 | 163 / 670 | 247 / 292 |

- **Its own time meets the 2 ms objective with room, and the latency it adds stays under 2 ms at p99.** Its own time
  averaged 32 µs, with 94.3% of requests within 100 µs, 99.6% within 500 µs and 99.99% within 2 ms. The largest added
  p99 is 1.92 ms, at 64 concurrent non-streaming requests in a process that also runs the client and the engine.
- **At one client the hop and the gateway's work cost about the same.** The bare proxy adds 112 µs at p50 and the
  gateway 114 µs more. Under load its own work costs about 15 µs of CPU per request beyond the hop.
- **A 64 KiB body adds 345 µs at one client**, a validating parse and a copy, against agents' times to first token of
  100 ms and more. A streamed chunk costs 65 µs at p50, near the bare hop's 61 µs.
- **On an Apple M5 Max** (18 cores) it adds 39 µs at one client, and 0.82 ms at most at p99.
- **An admission policy adds 6 to 24 µs to the gateway's own time.** On the M5 Max, in the medians of six runs on
  2026-10-23, the own time averaged 28 µs without a policy, and `lifecycle` added 9 µs. Every policy kept all its
  requests within 2 ms. A one-event `POST /v1/events` took 62 to 63 µs at p50 with or without the policy, and one
  connection posted 15,000 events a second.

## Tests

`cargo test -p flatshare-gateway` runs in a few seconds on loopback alone. The unit tests sit beside their modules,
and the integration tests (`tests/gateway/`) put the gateway between a raw HTTP/1.1 client and an in-process mock
engine. Each test file's module doc lists what it checks.
