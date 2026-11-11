# Runbook

How to deploy, check, operate and recover the Flatshare hosts, and run their failure drills, for whoever operates them:
the GPU nodes Phastform and Gray and the control host cyan. Every service is a `systemd --user` unit, and lingering is
on, so units survive logout and start at boot. Other hardware, or one host, runs the measurement kit of
[HARDWARE.md](../docs/HARDWARE.md) instead, which starts nothing at boot and has none of the releases below
([PRODUCTION.md](../docs/PRODUCTION.md#checking-it-on-your-hardware)).

| Host | Units | Listeners |
|---|---|---|
| Phastform (GPU) | `flatshare.target` → `flatshare-stt`, `flatshare-tts`, `flatshare-engine` | No TCP. The engine, STT and TTS on Unix sockets in `/run/user/$UID/flatshare/` |
| Gray (second GPU node, in windows) | Phastform's, with Gray's drop-ins; `flatshare.target` not enabled at boot | Unix sockets as on Phastform, and the engine cores' loopback TCP ports, which the listener audit accepts there ([below](#a-gpu-node-without-user-namespaces-gray)) |
| cyan (control) | `flatshare-tunnel@phastform`, `flatshare-gateway`, `flatshare-voicenet`, `flatshare-livekit`, `flatshare-voice-worker`; at times `flatshare-smg` and `flatshare-tunnel@gray` | The gateway's 127.0.0.1:18080, the host network's only TCP listener; Unix sockets in `/run/user/$UID/flatshare/` |

- **The sockets** live in a directory only the deployment's user may enter (mode 0700): the servers' `engine.sock`,
  `stt.sock` and `tts.sock`, and on cyan `gateway.sock` and each node's through its tunnel, such as
  `phastform-engine.sock`. LiveKit and SMG listen only inside the voice-plane namespace
  ([ARCHITECTURE.md's Figure 2](../docs/ARCHITECTURE.md#3-the-serving-path)).
- **Gray** is shared, used in windows when no other user's work holds its GPU. Its drop-ins (`deploy/gray/systemd/`)
  put the engine on cores 0–7 of NUMA node 0, the GPU's, and the audio servers on cores 8–15 of node 1.
- **cyan's occasional units.** `flatshare-smg` serves E2's arm 8b only, and `flatshare-tunnel@gray` runs while
  `gateway.toml` reaches Gray ([Gray's windows](#two-gpu-nodes)).

| Cores on cyan (48 threads) | Used by |
|---|---|
| 0–7 | The control plane (`flatshare-gateway`) |
| 8–15 | agentic-api and tools |
| 16–31 | Load generators: AIPerf on 16–23, `flatshare-bench` on 16–31, callers on 24–31 |
| 32–47 | The voice plane: LiveKit and the worker |

Load generators run under `taskset`, and the other units set their own affinity.

## Deploy

Deploy from a workstation with SSH to the hosts, by any name whose first label is the host's directory under
`deploy/`.

```sh
export FLATSHARE_SSH="ssh -F $HOME/.ssh/flatshare_config"   # your SSH command for the hosts
deploy/deploy.sh phastform                       # release HEAD's commit to
                                                 # Phastform: prepare, verify, switch
deploy/build-gateway.sh peacock                  # this commit's gateway, built
                                                 # on a Linux host with rustup
deploy/deploy.sh cyan --gateway build/flatshare-gateway-<id>
deploy/deploy.sh gray                            # the second GPU node
deploy/deploy.sh phastform --status              # its releases, current and previous
deploy/deploy.sh phastform --rollback            # back to the previous release
                                                 # (--to ID: another one)
```

**Drain the voice plane before a GPU node's release** that restarts its engine, or the calls on it break:

```sh
$FLATSHARE_SSH cyan python3 flatshare/src/deploy/bin/flatshare-release drain     # serve mode; waits for the calls
deploy/deploy.sh phastform
$FLATSHARE_SSH cyan python3 flatshare/src/deploy/bin/flatshare-release undrain
```

**First time on a host:**

1. Enable lingering: `loginctl enable-linger $USER`.
2. Release to it with `deploy/deploy.sh HOST`, which needs uv 0.11.27 or newer in `~/.local/bin` or set by
   `FLATSHARE_RELEASE_UV`. Nothing runs yet.
3. Run `bash ~/flatshare/src/deploy/HOST/bootstrap.sh`. A GPU node fetches its models and the Kokoro checkout, and cyan
   installs LiveKit and makes its keys and the three tunnel keys (`~/.ssh/flatshare_tunnel_{engine,stt,tts}_ed25519`).
   On cyan, write `FLATSHARE_TUNNEL_TARGET=<user>@<gpu node>` to `~/.config/flatshare/tunnel-phastform.env`.
4. Authorize the tunnel keys with `deploy/authorize-tunnel.sh cyan phastform` (and `gray`). Each key may only bridge
   to its node's socket (`exec nc -q0 -U /run/user/$UID/flatshare/engine.sock`) under `restrict`, so the node needs
   `netcat-openbsd`. A soak's guard takes `deploy/authorize-guard.sh cyan phastform` too
   ([Soaks](../benchmarks/e2/README.md#soaks)).
5. On cyan, write `~/.config/flatshare/gateway.toml` (mode 600) from `deploy/cyan/gateway.example.toml`, with only
   each tenant key's digest: `printf '%s' "$key" | sha256sum`. `install.sh` writes the files `cache_salt_secret_file`
   and `metrics_key_file` name, once.
6. On cyan, `systemctl --user enable --now flatshare-gateway.socket flatshare-gateway`.

**A release** is `HEAD`'s commit as `git archive` makes it, with `DEPLOYED`, its `git describe`. `deploy.sh` refuses a
tree with changes it would not carry. On the host, [`deploy/bin/flatshare-release`](../deploy/bin/flatshare-release)
(Python 3.10 or newer) keeps the deployment under `~/flatshare`:

| Path | What it is |
|---|---|
| `releases/ID/` | A release, ID the commit's 12-character abbreviation: its read-only tree, `MANIFEST.json` and `venvs/NAME` |
| `envs/NAME-KEY/` | A Python environment, KEY the SHA-256 of its lock file and other inputs |
| `current`, `previous` | The release the host runs, switched in one rename, and the one it replaced |
| `src`, `venvs/NAME`, `bin/flatshare-gateway` | Links into `current`, the paths the units name (`%h/flatshare/src/...`). `venvs/sglang` stays a directory, and `bin/` keeps the host's own tools |
| `releases/release.log` | What each command that changes the host printed (`release.log.1` past 1 MiB) |
| `releases/history.jsonl`, `releases/running.json` | A line per switch, rollback, switch back and adoption; the release each unit last started under |
| `releases/switching.json` | Present while a switch runs. Left behind, it makes the next switch restart every active unit |

| `MANIFEST.json` key | What it records |
|---|---|
| `commit`, `describe`, `files`, `tree_sha256` | The commit, its `git describe`, and the SHA-256 of every file and of the tree |
| `locks`, `envs` | Every lock file; each environment's key, inputs, interpreter, uv version and package list's hash |
| `artifacts`, `tools` | On cyan, the gateway binary's SHA-256 and where it was built or carried from; the host binaries the units run, such as `livekit-server` |
| `models`, `scheduler`, `serving` | The models' revisions, make-room's patches and scheduler module, and the serving profile's effective configuration |
| `units`, `isolation` | Each unit's inputs, hashed; on a GPU node, each server's network namespace and `private-net-check`'s verdict |

<picture>
  <source media="(max-width: 767px)" srcset="../docs/figures/release-narrow.svg">
  <img src="../docs/figures/release.svg" width="100%" alt="The release's states in order, prepare, verify, drain, switch, restart, check and serving: a host that is locked or an upload that changed exits 2, a failed verification exits 1, and a drain whose calls do not end or that meets a running E2 point exits 2, each with nothing switched; any failure from the switch through the check switches back, which exits 1 with the previous release running again or 3 when switching back fails too; a release that passes its checks serves and exits 0, and a rollback enters at the verification with the previous release.">
</picture>

**A release switches nothing until it has verified and drained, and switches back by itself when anything after the
switch fails.** `flatshare-release deploy`, which `deploy.sh` runs, takes these steps:

1. **Prepare.** The upload must hash to the staged tree. Environments no release has built are built from their locks.
2. **Verify** every file against the manifest, each environment's imports offline, the binaries' versions, the serving
   profile's patches and module, and the models a running unit needs. A failed check switches nothing.
3. **Drain** what the switch would cut. The voice worker in serve mode takes no new call, and the release waits up to
   `--drain-timeout` (300 s) for calls to end, else switches nothing unless `--force`. On a GPU node it waits at most
   60 s for the engine's requests in flight.
4. **Switch** `previous` to what `current` names and `current` to the release, then run `install.sh` and
   `daemon-reload`.
5. **Restart or reload** each active unit whose inputs changed, so a release of documentation restarts nothing. A new
   gateway binary reloads the gateway without a gap.
6. **Check** each restarted unit for `--settle` seconds (10), the engine's, worker's and nodes' configuration against
   `serving.json`, and the listener audit (`deploy/bin/audit-listeners`).
7. **On a failure after the switch** it switches back to the release before. On success it keeps the newest 3 releases
   (`--keep`), current, previous, and any that a running process imports from or a unit was started under, and removes
   the rest and the environments none of them links.

**Exit status** of `flatshare-release`, which `deploy.sh` passes on:

| Status | When |
|---|---|
| 0 | Success |
| 1 | A check or a command failed, or a switch failed and was switched back |
| 2 | Used wrongly, or the host or the upload was not ready: nothing switched |
| 3 | Switching back failed too |

- **A dropped session** (Ctrl-C, a lost link) does not stop the release, which runs on and logs to
  `releases/release.log`. Read it, or `deploy.sh HOST --status`, before deploying again.
- **`--no-switch`** prepares and verifies only, to warm a release. `--rollback` switches to `previous` by steps 2 to 7.
- **Never release to cyan while an E2 campaign runs**, or the next point reads changed arm files. **Never run scripts
  from a release's own directory**, since links made from there outlive it.
- **A unit the repository no longer has** is unlinked by cyan's `install.sh` but runs until the next reboot, so stop
  it by hand (`systemctl --user stop UNIT`).

**The gateway's binary.** `deploy/build-gateway.sh BUILD_HOST [OUT]` builds this commit's gateway on a Linux host with
rustup and a glibc no newer than cyan's, such as peacock. `deploy.sh cyan --gateway FILE` releases it, refusing one
built from another commit. Without `--gateway`, a release carries cyan's running binary if `crates/`, `Cargo.toml`,
`Cargo.lock` and `rust-toolchain.toml` are unchanged, and refuses otherwise.

## Upgrading a component

| Component | Upgrade | Rollback |
|---|---|---|
| vLLM and the Python environments | A new lock, from which the release builds a new environment | `deploy.sh HOST --rollback`, offline and atomic |
| The gateway's binary, Flatshare's code and the serving profile | Each release | Atomic, with the release |
| The models | Change the revision in `engine.yaml` or `stt.yaml`, run `deploy/phastform/bootstrap.sh` on each node to fetch it, then release | Atomic while the old revision is in the Hugging Face cache |
| LiveKit | Change its pins (`livekit=` in `deploy/cyan/bootstrap.sh`, `tools` in `deploy/bin/flatshare-release`), run `bootstrap.sh`, then release | Not atomic. Run the old release's `bootstrap.sh` first |
| Kokoro-FastAPI | Change its commit in `flatshare-release`'s `vendor`, `deploy/phastform/bootstrap.sh` and `deploy/a100/bootstrap.sh`, run `deploy/phastform/bootstrap.sh`, then release | Not atomic. First `git -C ~/flatshare/vendor/Kokoro-FastAPI checkout COMMIT` |

A release refuses to switch rather than run a component its manifest does not name.

## State and secrets

A gateway reload keeps its listeners, placements and nodes' verdicts, and a restart or crash loses the placements,
which the sessions' next requests make again. A restart of LiveKit or the worker ends the calls, and one of the engine
loses its KV.

| Item | Where | If lost | Rotation |
|---|---|---|---|
| `gateway.toml` | cyan, `~/.config/flatshare/` | Keep a copy | Edit, then `systemctl --user reload flatshare-gateway` |
| Tenants' keys | Clients; `gateway.toml` holds their SHA-256 (`keys_sha256`) | Issue new ones | Add the new digest, reload, move the clients, remove the old digest, reload |
| The metrics key | cyan, `~/.config/flatshare/metrics-key` | `install.sh` makes one | Replace the file and reload. Prometheus rereads it (`credentials_file`) at each scrape |
| The cache-salt secret | cyan, `~/.config/flatshare/cache-salt-secret` | `install.sh` makes one | Replace the file and reload. Every prefix cache starts cold |
| LiveKit's key and secret | cyan, `~/.config/flatshare/livekit.env` | `bootstrap.sh` makes them | Remove the file, run `bootstrap.sh`, restart LiveKit and the worker in a drain window |
| The tunnels' keys | cyan, `~/.ssh/flatshare_tunnel_*_ed25519` | `bootstrap.sh` makes them | Remove them, run `bootstrap.sh` and `deploy/authorize-tunnel.sh cyan NODE`, restart `flatshare-tunnel@NODE` |
| Each tunnel's target | cyan, `~/.config/flatshare/tunnel-NODE.env` | Keep a copy | Edit, then restart `flatshare-tunnel@NODE` |
| A node's engine variant | `~/.config/systemd/user/flatshare-engine.service.d/` | `variant.sh use make-room-serve` | — |

## Start, stop, status

```sh
# Phastform
systemctl --user start flatshare.target          # the three servers start at once;
                                                 # the box is ready when the engine is
systemctl --user stop flatshare.target
systemctl --user restart flatshare-engine        # one server; returns once /health answers
systemctl --user list-units 'flatshare-*'
journalctl --user -u flatshare-engine -f         # logs
bash ~/flatshare/src/deploy/phastform/smoke.sh   # end-to-end voice turn, GPU memory, listener audit
nsenter --target "$(systemctl --user show -p MainPID --value flatshare-engine)" --user --net \
  --preserve-credentials \
  ss -lntup                                      # what listens inside the
                                                 # engine's own network namespace

# cyan
systemctl --user start flatshare-tunnel@phastform flatshare-voicenet flatshare-livekit \
  flatshare-gateway
socket_dir=unix:$XDG_RUNTIME_DIR/flatshare           # the tunnel's sockets
bash ~/flatshare/src/deploy/phastform/smoke.sh $socket_dir/phastform-engine.sock \
  $socket_dir/phastform-stt.sock \
  $socket_dir/phastform-tts.sock
python3 ~/flatshare/src/deploy/bin/audit-listeners   # no TCP listener of ours but the gateway's
~/flatshare/src/deploy/bin/in-voicenet ss -lntup     # what listens inside the voice-plane namespace
```

A unit counts as started once its server answers its health endpoint, and units that depend on it start only then.

**Pause a GPU node when idle** with `systemctl --user stop flatshare.target flatshare-engine flatshare-stt
flatshare-tts`, which returns once the units are down, the engine after about 20 s. `systemctl --user start
flatshare.target` starts it again. On Phastform, whose target is enabled, a reboot does too, on the variant it ran. A
node that `gateway.toml` names pages when paused, so [drain it](#drain-one-node) first.

## The gateway

`flatshare-gateway` serves the OpenAI routes, the Anthropic Messages API and `/v1/events` on 127.0.0.1:18080 with an API
key, and without one on `gateway.sock`, the voice plane's way in. [Its README](../crates/flatshare-gateway/README.md) is
the reference for every key and metric. `flatshare-gateway.socket` holds the listening sockets, so a connection waits in
the backlog while no gateway accepts.

- **Reload** takes a new binary or a changed `gateway.toml` without a gap. A successor takes the sockets while the old
  process drains, and one not up within 10 s fails the reload and changes nothing.
- **Restart** drains, then starts. Realtime and interactive requests drain for up to `drain_timeout_s` (30 s), and
  background ones are cut after `background_drain_s` (5 s).
- **A stop or restart after a reload** waits for the drain in the unit's `ExecStop`
  (`deploy/bin/flatshare-gateway-stop`), and kills the gateway after its `--wait` of 45 s, the drain's 30 s and a
  margin. When you raise `drain_timeout_s`, raise `--wait` on the `ExecStop` line with it.
- **A change to `[[listen]]`** takes a restart of both units, with the socket unit's `ListenStream=` changed to match.
- **A new binary** comes with a release ([Deploy](#deploy)).

```sh
systemctl --user reload flatshare-gateway           # a new binary or gateway.toml, without a gap;
                                                    # fails if the successor does not take over
systemctl --user restart flatshare-gateway          # drains (background requests cut after 5 s),
                                                    # then starts; connections wait meanwhile
systemctl --user show -p MainPID flatshare-gateway  # a reload changes it: the successor
                                                    # is the unit's main process
```

```sh
curl -s 127.0.0.1:18080/health                    # the process is up
curl -s 127.0.0.1:18080/ready                     # 200 while the engine serves and its
                                                  # /health answers; else 503 and why
curl -s --unix-socket $XDG_RUNTIME_DIR/flatshare/gateway.sock localhost/metrics | grep \
  -E 'flatshare_gateway_(requests_terminal|upstream_errors)_total'
journalctl --user -u flatshare-gateway -f         # one JSON line per request
                                                  # and per lifecycle event
```

E2 reads each point's request lines from this journal, so `gateway.toml` keeps `log_format = "json"`. `/ready` is 200
only while the engine is `serving` by its inference, not by its `/health` alone
([the engine's state](../crates/flatshare-gateway/README.md#the-engines-state)).

## Monitoring

To watch the hosts, on cyan:

1. **Prometheus.** Run it as the deploy user with `ops/prometheus/prometheus.example.yml`, so that it may read the
   metrics key's file. It scrapes the gateway's `/metrics` and the engine's `/metrics/engine` through the gateway. With
   `[[nodes]]`, replace the `flatshare-engine` job with one job per node that scrapes `/metrics/engine?node=NAME`, as
   the example's commented jobs do. The gateway answers the single job 400 `node_required`, which
   `FlatshareEngineMetricsDown` reports.
2. **The rules.** Load `ops/prometheus/rules.yml`, which holds the SLIs, the burn-rate alerts of
   [the two production objectives](../docs/SLO.md#the-production-objectives) and the health alerts.
   `make check-rules PROMTOOL=<Prometheus 3.5.0's promtool>` checks them and runs their unit tests,
   `ops/prometheus/rules.test.yml`.
3. **The dashboard.** Import `ops/grafana/flatshare.json` with a Prometheus data source.

The first-byte objective counts a request that fails, is refused, times out or hangs as a miss
([the service level](../crates/flatshare-gateway/README.md#accounting)). With `[[nodes]]` the engine's alerts hold for
each node, and each alert's `runbook_url` opens its row in [Failures](#failures). The voice worker's LiveKit metrics
are not scraped, since its namespace has no route in.

## The voice plane

| Unit | What it runs | Notes |
|---|---|---|
| `flatshare-voicenet` | The network namespace holder | Loopback plus the dummy `flatshare0` (10.255.0.1/32), no route out |
| `flatshare-livekit` | LiveKit server in the namespace | 127.0.0.1:17880 (signalling), UDP 17882 (media) |
| `flatshare-voice-worker` | `flatshare-voice-worker start` (LiveKit Agents) in the namespace | Agent `flatshare-voice`. STT and TTS through `%t/flatshare/phastform-{stt,tts}.sock`, the LLM through `%t/flatshare/gateway.sock` |

```sh
systemctl --user start flatshare-tunnel@phastform flatshare-gateway flatshare-voicenet \
  flatshare-livekit \
  flatshare-voice-worker
journalctl --user -u flatshare-voice-worker -f          # JSON logs; "registered
                                                        # worker" means it is ready
```

**Bench and serve modes.** The worker's unit runs bench mode, which takes every call and which every measurement ran.
The `serve` variant admits the profile's two calls, none while `/ready` fails or a release drains
(`$XDG_RUNTIME_DIR/flatshare/voice-worker.drain`), and writes its state
to `$XDG_RUNTIME_DIR/flatshare/voice-worker.json` ([the
modes](../python/flatshare_voice/README.md#bench-and-serve-modes)). To serve, unlink `vad-turns`, which its microphone
configuration refuses, and link `serve`. `flatshare-bench e2 run` refuses a worker in serve mode. Check the worker and
the gateway's nodes against the profile:

```sh
units=~/.config/systemd/user/flatshare-voice-worker.service.d
rm $units/variant-vad-turns.conf
ln -s ~/flatshare/src/deploy/cyan/variants/serve/flatshare-voice-worker.conf \
  $units/variant-serve.conf
systemctl --user daemon-reload && systemctl --user restart flatshare-voice-worker
python3 ~/flatshare/src/deploy/bin/serving-check ~/flatshare/src/deploy/phastform/serving.json \
  --voice $XDG_RUNTIME_DIR/flatshare/voice-worker.json \
  --voice-pid "$(systemctl --user show -p MainPID --value flatshare-voice-worker)"
curl -s --unix-socket $XDG_RUNTIME_DIR/flatshare/gateway.sock localhost/nodes > \
  $XDG_RUNTIME_DIR/flatshare/gateway-nodes.json
python3 ~/flatshare/src/deploy/bin/serving-check ~/flatshare/src/deploy/phastform/serving.json \
  --nodes $XDG_RUNTIME_DIR/flatshare/gateway-nodes.json
```

**Worker variants** are drop-ins, `deploy/cyan/variants/NAME/flatshare-voice-worker.conf`, linked into
`~/.config/systemd/user/flatshare-voice-worker.service.d/` and followed by `systemctl --user daemon-reload` and a
restart:

| Variant | What it sets | For |
|---|---|---|
| `vad-turns` | Fixed endpointing (`FLATSHARE_TURN_DETECTION=vad`) | Every experiment, since LiveKit's end-of-turn model waits 3 s on some synthesized turns ([BENCHMARKS.md](../docs/BENCHMARKS.md#what-does-agent-load-do-to-a-voice-turn-on-the-same-gpu)) |
| `serve` | Serve mode, each call placed on a GPU node ([nodes](../python/flatshare_voice/README.md#nodes)) | Serving |
| `no-preemptive-generation` | `FLATSHARE_PREEMPTIVE_GENERATION=false` | E2's factor |
| `no-events` | `FLATSHARE_EVENTS=off` | Arm 9's ablation |
| `node-binding` | `FLATSHARE_NODE_BINDING=on`: bench mode's calls are placed too | Points and drills on two GPU nodes |
| `qwen3-14b` | `FLATSHARE_LLM_MODEL` names Qwen3-14B-FP8 | The A100 kit's arm C' (`benchmarks/a100/cprime.sh`) |

A drop-in with `Environment=FLATSHARE_LLM_UDS=%t/flatshare/phastform-engine.sock` sends the LLM straight to the engine.
The worker's journal counts each session's lifecycle events (`journalctl --user -u flatshare-voice-worker | grep
'"session ended"'`: `events_sent`, `events_dropped`, `events_failed`), and the gateway counts them in
`flatshare_gateway_events_total{source="voice-worker"}`.

**Measurement callers.** Render the corpus once with the TTS server. The callers run inside the namespace:

```sh
~/flatshare/venvs/voice/bin/flatshare-voice render-callers ~/flatshare/data/callers \
  --tts-url http://localhost/v1 \
  --uds $XDG_RUNTIME_DIR/flatshare/phastform-tts.sock
set -a; . ~/.config/flatshare/livekit.env; set +a
taskset -c 24-31 ~/flatshare/src/deploy/bin/in-voicenet \
  ~/flatshare/venvs/voice/bin/flatshare-voice bots \
  --corpus ~/flatshare/data/callers --callers 2 --turns 25 --barge-in 0.25 \
  --out ~/flatshare/runs/<run>/bots
```

## Engine variants

The frozen box is `deploy/phastform/engine.yaml` and the units in `deploy/phastform/systemd/`.
`deploy/phastform/variant.sh use NAME|frozen [FLAG...]` makes the box run exactly one variant plus engine flags,
restarting only what changes. Flags reach the engine through `FLATSHARE_ENGINE_FLAGS` in a generated drop-in,
`variant-flags.conf`, and `systemctl --user cat flatshare-engine` shows what the box runs.
[The variants' README](../deploy/phastform/variants/README.md) describes each variant and the E2 arms' flags.

| Variant | What the box runs |
|---|---|
| `make-room-serve` | The serving configuration, from the serving profile `deploy/profiles/rtx-5060-ti-serve.toml`. Each start holds the engine to `serving.json` (`deploy/bin/serving-check`) and fails on any difference, and the engine writes what it runs to `$XDG_RUNTIME_DIR/flatshare/engine-effective.json` ([the profile](../deploy/phastform/variants/README.md#the-serving-profile)) |
| `make-room` | The change on vLLM 0.32.0 with arm 9's values, as `--additional-config.make_room.*` keys that an arm's flags override |
| `pr54530` | vLLM #54530 ported to 0.32.0, loaded with `--scheduler-cls` |
| `sglang` | SGLang 0.5.22 on 127.0.0.1:8000 behind the engine's socket (`flatshare-engine-proxy.socket`). Build its environment once with `bash ~/flatshare/src/deploy/phastform/variants/sglang/install.sh`. Its unit counts as started after one sampled request, since the first compiles FlashInfer's kernels for minutes |
| `mps` | The three servers as clients of one CUDA MPS daemon (`flatshare-mps.service`) |
| `agents-only` | A card for the agents alone, its speech servers stopped and their memory in the KV pool ([the variant](../deploy/phastform/variants/README.md#a-card-for-the-agents-alone)) |

- **Serve** with `variant.sh use make-room-serve` and `variant.sh measure off`.
- **`variant.sh measure on|off`** adds or removes the measurement drop-in, vLLM's development routes, which E2 and the
  drills read and a production box does without ([the
  routes](../deploy/phastform/variants/README.md#the-measurement-drop-in)).
- **`variant.sh on|off NAME`** adds or removes a single variant, for work outside E2.
- **The generated variants** (`pr54530`, `make-room`, `make-room-serve`) build their scheduler module at each start, and
  a release restarts a running engine when their `build.py` or patches change.

## Two GPU nodes

The gateway can front two GPU nodes, Phastform and Gray, with deliberate failover between them and no claim of high
availability ([RESOURCE-GATE.md](../benchmarks/quality/RESOURCE-GATE.md) compares the arrangements). cyan reaches each
through its own tunnel, `flatshare-tunnel@NODE`, whose sockets are `NODE-engine.sock`, `NODE-stt.sock` and
`NODE-tts.sock`.

`gateway.toml` then has a `[[nodes]]` entry per node in place of `upstream`, no `[admission]` table, and optionally
`[placement]` ([every key](../crates/flatshare-gateway/README.md#configuration); `deploy/cyan/gateway.example.toml` has both
nodes). A node with both roles must set `max_agents`, a count or `"unbounded"`. A reload keeps every session's node,
so lower bounds apply to new sessions only.

```sh
# each node's verdict, components, calls and agents, and the placement
curl -s --unix-socket $XDG_RUNTIME_DIR/flatshare/gateway.sock localhost/nodes | jq
# 200 while each role has an available node
curl -s 127.0.0.1:18080/ready
```

- **Health.** A node is healthy while its engine serves and its STT and TTS answer `/health`. Checked every second, it
  is unhealthy after 3 failed checks and healthy after 2 passed ones.
- **Placement.** A session stays on its node until it ends or has gone `idle_s` without a request (150 s in the
  arrangement to deploy). The voice worker places each call with `POST /v1/sessions` and ends it with
  `DELETE /v1/sessions/{id}`.
- **Failover** (`FlatshareNodeUnhealthy`). The failed node's sessions move to the other by their next request. The
  survivor needs a free slot for each, or answers 503 `no_call_slot` or `no_agent_slot`.
- **Covering** (`FlatshareNodeCovering`). While the survivor carries moved calls, its agents' requests wait until those
  calls end, even once the failed node is back.
- **Recovery.** Restart the failed server on the node (`systemctl --user restart flatshare-engine` there), or let
  `Restart=on-failure` do it. After 2 passed checks the node is healthy again and takes new sessions, and the moved
  sessions stay where they are.

**Sizing.** Each node's `max_calls` must hold every call the worker admits, two in the arrangement to deploy. A node
that carries calls bounds its agents with `max_agents`, what its card serves beside both calls within the voice SLO.
With `calls = "fill"` the second node is a warm spare running up to its own `max_agents` (12) while it carries no call.
Set `idle_s` just above the agents' longest pause between requests, 150 s for the replay, whose think times are capped
at 120 s, and offer a few sessions under the bounds. A session holds its slot until its client ends it or it idles out,
so a client that starts new sessions without ending the old ones, as AIPerf's replay does, gets 503 `no_agent_slot`
until those idle out ([the two-node smokes](../benchmarks/results/2026-10-28-two-node-smoke/README.md)).

<a id="drain-one-node"></a>**Drain one node** for maintenance, such as a rolling engine upgrade, so that no call fails:

1. In the node's `[[nodes]]` entry in `~/.config/flatshare/gateway.toml` on cyan, set `draining = true`, and reload the
   gateway (`systemctl --user reload flatshare-gateway`). New calls go to the other node, and its agents' sessions
   move there by their next request.
2. Wait until its last call has ended and `/nodes` says it is drained:

   ```sh
   curl -s --unix-socket $XDG_RUNTIME_DIR/flatshare/gateway.sock localhost/nodes | jq \
     '.nodes[] | {name, draining, drained, sessions}'
   ```

3. Stop the node (`systemctl --user stop flatshare.target flatshare-engine flatshare-stt flatshare-tts` there). A
   draining node does not page. Upgrade it, or release to it, and start it again.
4. Once `/nodes` shows it healthy, remove `draining` and reload.

Drain in a quiet window, since the other node needs a call slot for each call. With `calls = "fill"`, the spare that
takes them covers, holding its agents' requests until those calls end, as in a failover. The gateway's tests drain one
of two nodes, but no drill has drained a GPU node.

**Gray's windows.** Start Gray's servers by hand for a window (`systemctl --user start flatshare.target` on Gray, and
`stop` after). cyan's `install.sh` enables `flatshare-tunnel@gray` while `gateway.toml` names Gray's sockets and
`tunnel-gray.env` exists, but never disables it. When a window ends, drain Gray, reload the gateway without its node,
then `systemctl --user disable --now flatshare-tunnel@gray`.

## Drills

The failure drills run from a workstation on the variant the GPU node runs, with no E2 point running, since they take
its lock on cyan. Each writes its directory under `~/flatshare/runs/drills` and prints its summary.

| Drill | What breaks | What it should show |
|---|---|---|
| `ops/drills/drill.sh engine-kill` | SIGKILL to every process of `flatshare-engine`, beside two realtime and four background streams | Each stream ends with the `upstream_unavailable` error event, and requests succeed once the engine is back |
| `ops/drills/drill.sh gateway-reload` | `systemctl --user reload flatshare-gateway`, as a deploy does | No failed probe, and every voice turn whole |
| `ops/drills/drill.sh gateway-restart` | `systemctl --user restart flatshare-gateway` | No failed probe, answers coming once the drain is done |
| `ops/drills/drill.sh gateway-kill` | SIGKILL to the gateway | Streams broken, the engine's requests gone within a second or so, probes answered once it is back |
| `ops/drills/node-failover.sh kill\|stop` | With `[[nodes]]`, the engine of two calls' node, killed or stopped | The calls move within a turn or two, and the agents wait while the survivor covers |
| `ops/drills/cold-start.sh engine\|box` | A restart of the engine or the box, timed from the journals; `--cold` empties the compile and JIT caches first | Each server's start time |
| `~/flatshare/venvs/bench/bin/flatshare-bench drill cpu-tier --out FILE` | Twice the CPU tier's capacity of unique prompts, on cyan with the engine idle | The tier evicted the oldest prompt and still restores the latest |

`cold-start.sh` does not take cyan's lock, so start nothing else meanwhile. In the recorded drills every stream of the
engine kill got the error event 6 to 11 ms after the kill command, and the failover's calls moved within 1.9 s
([BENCHMARKS.md](../docs/BENCHMARKS.md#what-does-a-failure-do-to-the-calls-and-the-agents)).

<a id="soaks"></a>

E2's points and soaks take the same lock. [E2's README](../benchmarks/e2/README.md#a-point-on-cyan) says what a point
changes on cyan and how to undo it, and how [a soak](../benchmarks/e2/README.md#soaks) runs.

## Checks before heavy work on a shared host

- **`nvidia-smi` and `who`.** `gpu-guard` refuses to start a GPU unit while another user's process holds the GPU, but
  cannot stop work that starts later. Another user's tmux does not show in `tmux ls`.
- **While a campaign measures**, `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` on the GPU node lists
  only the box's units. A point measured while any other process held the GPU is measured again.
- **`free -g` and `df -h /dev/shm`**, where the CPU tier occupies 20 GiB.
- **The listener audit**, which `deploy.sh` runs after each install, finds on Phastform the servers on their sockets and
  none of the box's units on TCP, but for the SGLang variant's engine on loopback. On cyan, `ss -Hlntup` shows no
  LiveKit socket on the host network.

  ```sh
  python3 ~/flatshare/src/deploy/bin/audit-listeners \
    --expect $XDG_RUNTIME_DIR/flatshare/engine.sock \
    --expect $XDG_RUNTIME_DIR/flatshare/stt.sock --expect $XDG_RUNTIME_DIR/flatshare/tts.sock
  ```

  Each server listens in a network namespace of its own, so the host's `ss -Hlx` does not list its socket. The
  `nsenter` line under [Start, stop, status](#start-stop-status) shows what listens in the engine's.

## A GPU node without user namespaces (Gray)

Gray runs Ubuntu 24.04 with `kernel.apparmor_restrict_unprivileged_userns=1`, and the deploying user has no root. So
`in-private-net` cannot give a GPU server a network namespace of its own, and fails the unit of one that asks
([HARDWARE.md](../docs/HARDWARE.md#a-gpu-node-without-user-namespaces-gray)).

- **Each release** runs `deploy/gray/install.sh`, whose `deploy/bin/private-net-check` prints `allowed`,
  `refused: REASON` or `unknown: REASON`. Refused, it links `deploy/gray/private-net-host.conf` into the three servers'
  drop-ins, marking them `FLATSHARE_PRIVATE_NET=host`. Allowed, it removes them, and unknown, it changes nothing.
- **The servers** then run in the host's network namespace, on their Unix sockets alone. The listener audit, with the
  same three `--expect` sockets as above, accepts only the engine cores' loopback ports and CUDA's abstract sockets,
  which the host's other users can reach.
- **The manifest** records it (`isolation`), and `deploy.sh gray --status` prints it.
- **To undo it**, an administrator runs `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`. Then run any
  release or `bash ~/flatshare/src/deploy/gray/install.sh`, and restart the servers.

## Failures

Every alert's `runbook_url` opens its row here, and without an alert [the triage](#triage) leads to one. A point's
errors are in [E2's README](../benchmarks/e2/README.md#when-a-point-fails).

### The alerts

| Alert | Severity | First check | Row |
|---|---|---|---|
| `FlatshareRealtimeFirstByteBudgetBurn` | Page | `flatshare:requests_terminal:rate5m` by outcome, then `/nodes` | [Objectives](#flatsharerealtimefirstbytebudgetburn) |
| `FlatshareRealtimeFirstByteBudgetSlowBurn` | Ticket | The same | [Objectives](#flatsharerealtimefirstbytebudgetslowburn) |
| `FlatshareOverheadBudgetBurn` | Page | The dashboard's "The gateway's own time, p99" panel | [Objectives](#flatshareoverheadbudgetburn) |
| `FlatshareOverheadBudgetSlowBurn` | Ticket | The same | [Objectives](#flatshareoverheadbudgetslowburn) |
| `FlatshareGatewayDown` | Page after 2 min | `systemctl --user status flatshare-gateway flatshare-gateway.socket` on cyan | [Gateway](#flatsharegatewaydown) |
| `FlatshareRequestsRefusedForCapacity` | Ticket | `flatshare_gateway_in_flight` by class | [Gateway](#flatsharerequestsrefusedforcapacity) |
| `FlatshareDeadlinesMissed` | Ticket | The request's log line, which names the deadline | [Gateway](#flatsharedeadlinesmissed) |
| `FlatshareEngineNotReady` | Page after 1 min | The engine's `/health` through its tunnel's socket on cyan | [Engine](#flatshareenginenotready) |
| `FlatshareEngineDegraded` | Page after 2 min | The engine's journal on the GPU node | [Engine](#flatshareenginedegraded) |
| `FlatshareEngineWarming` | Ticket after 15 min | Whether the engine is still starting | [Engine](#flatshareenginewarming) |
| `FlatshareEngineMetricsDown` | Ticket after 5 min | The gateway's `/metrics/engine` relay, on cyan | [Engine](#flatshareenginemetricsdown) |
| `FlatshareStreamsBroken` | Ticket | The engine's and the tunnel's journals at those times | [Engine](#flatsharestreamsbroken) |
| `FlatshareKvPoolFullWithWaiting` | Ticket after 10 min | Whether the engine runs `make-room-serve` | [Engine](#flatsharekvpoolfullwithwaiting) |
| `FlatshareNodeUnhealthy` | Page after 1 min | `/nodes`, which names the failed component | [Two nodes](#flatsharenodeunhealthy) |
| `FlatshareNodeCovering` | Ticket after 5 min | Expected while moved calls end | [Two nodes](#flatsharenodecovering) |
| `FlatshareCallsRefused` | Ticket | `/nodes`: each node's calls | [Two nodes](#flatsharecallsrefused) |

A severity "after" a time is the rule's `for`, how long its condition held.

### Triage

<picture>
  <source media="(max-width: 767px)" srcset="../docs/figures/triage-narrow.svg">
  <img src="../docs/figures/triage.svg" width="100%" alt="A column of five checks from a call that fails or waits, each leading to the runbook rows it names: the gateway down (FlatshareGatewayDown); /ready not 200, the engine not ready, degraded or warming (FlatshareEngineNotReady, FlatshareEngineDegraded, FlatshareEngineWarming); a node unhealthy or covering (FlatshareNodeUnhealthy, FlatshareNodeCovering); refusals for a full bound or no slot (FlatshareRequestsRefusedForCapacity, FlatshareCallsRefused, 503 no_agent_slot); a full KV pool with requests waiting (FlatshareKvPoolFullWithWaiting); and, when all five pass, the voice plane: the worker's journal and LiveKit's.">
</picture>

**Five checks lead from a call that fails or waits to its rows, the gateway first.** Each answer names the alerts whose
rows say what to do, or 503 `no_agent_slot`, which has no alert. When all five pass, the voice plane's journals are
next, the worker's and LiveKit's ([Voice plane](#voice-plane)).

### Objectives

| Symptom | Likely cause | Action |
|---|---|---|
| <a id="flatsharerealtimefirstbytebudgetburn"></a><a id="flatsharerealtimefirstbytebudgetslowburn"></a>`FlatshareRealtimeFirstByteBudgetBurn` or `FlatshareRealtimeFirstByteBudgetSlowBurn`: voice requests fail, are refused, time out or wait over 250 ms for their first content | Agents' KV filling the calls' node, an engine that does not serve, or calls moved to a survivor | `flatshare:requests_terminal:rate5m` by outcome, then `/nodes` and the calls' node's KV and engine journal. With agents beyond the envelope, lower that node's `max_agents` and reload |
| <a id="flatshareoverheadbudgetburn"></a><a id="flatshareoverheadbudgetslowburn"></a>`FlatshareOverheadBudgetBurn` or `FlatshareOverheadBudgetSlowBurn`: over 1% of requests spend over 2 ms in the gateway | The gateway's cores (0–7 on cyan) taken by other work, or an admission policy's slow decisions | `flatshare_gateway_overhead_seconds` by route, and what else runs on cores 0–7 |

### Gateway

| Symptom | Likely cause | Action |
|---|---|---|
| <a id="flatsharegatewaydown"></a>`FlatshareGatewayDown`: Prometheus has not read `/metrics` for 2 minutes | The gateway is stopped, crashed or refusing its configuration, or Prometheus's metrics key no longer matches (401) | `systemctl --user status flatshare-gateway flatshare-gateway.socket`. If it serves, Prometheus's target page gives the error |
| <a id="flatsharerequestsrefusedforcapacity"></a>503 `in_flight_limit`, `class_in_flight_limit` or `body_budget_exhausted`, or 429 `tenant_in_flight_limit` (`FlatshareRequestsRefusedForCapacity`) | The gateway, a class or a tenant is at its `max_in_flight`, or the bodies held would pass `max_buffered_body_bytes`, a reload's old process counting too | `flatshare_gateway_in_flight` by class, `flatshare_gateway_predecessor_in_flight` and `flatshare_gateway_buffered_body_bytes`. If the load is real, raise the bound in `gateway.toml` and reload |
| <a id="flatsharedeadlinesmissed"></a>504 `upstream_timeout`, the `upstream_timeout` error event, or 408 `upload_timeout` (`FlatshareDeadlinesMissed`) | The engine missed a class deadline, or a client uploaded too slowly (408) | The log line's `error` names the deadline, "classes.CLASS.KEY (N s)". Then the engine's waiting requests and KV |
| 401 `invalid_api_key` | The client's key has no digest in `gateway.toml` | `printf '%s' "$key" \| sha256sum` must match an entry; reload after editing |
| Exit at start: "another process is accepting on this socket" | A second gateway serves `gateway.sock`, or the gateway runs outside its unit | Stop the other process, or start the gateway with `systemctl --user start flatshare-gateway` |
| Exit at start: "inherited sockets that no [[listen]] entry names" | `flatshare-gateway.socket`'s `ListenStream=` and `gateway.toml`'s `[[listen]]` disagree | Make them name the same addresses (`%t` is `/run/user/$UID`), `systemctl --user daemon-reload`, restart both units |
| `systemctl --user reload flatshare-gateway` fails: "its successor did not take over" | The successor exited at start or was not ready within 10 s, and the old process serves on | `journalctl --user -u flatshare-gateway -n 50` says why; fix and reload again |
| `flatshare-gateway.socket` failed with `service-start-limit-hit` | The gateway crashed five times within 10 s | `journalctl --user -u flatshare-gateway -n 200`, fix, `systemctl --user reset-failed flatshare-gateway flatshare-gateway.socket`, start both |
| 429 `queue_full` or 503 `queue_timeout` | An `[admission]` policy's class queue (`queue_limit`) or all its queues (`max_queued`) are full, or a wait ran out (`queue_timeout_s`) | Expected under E2's policies. `flatshare_gateway_admission_refused_total` counts each reason |

### Engine and GPU servers

| Symptom | Likely cause | Action |
|---|---|---|
| <a id="flatshareenginenotready"></a>502 `upstream_unavailable`, or `/ready` 503 `upstream not ready` (`FlatshareEngineNotReady`) | The tunnel or the engine is down | [The tunnel's row](#tunnels-and-keys) checks both. An engine restarts by itself in about two minutes |
| <a id="flatshareenginemetricsdown"></a>`FlatshareEngineMetricsDown`: no engine metrics through the gateway | The engine or its tunnel is down, or the engine did not answer the relay within 10 s (502) | As above, then on cyan `curl -s --unix-socket $XDG_RUNTIME_DIR/flatshare/gateway.sock localhost/metrics/engine`, or `'localhost/metrics/engine?node=NODE'` under `[[nodes]]` |
| <a id="flatsharestreamsbroken"></a>`FlatshareStreamsBroken`: streams ended with the `upstream_unavailable` error event | An engine that died or restarted, or a tunnel that dropped | The engine's and the tunnel's journals at those times |
| <a id="flatshareenginedegraded"></a>`/ready` 503 `engine degraded` (`FlatshareEngineDegraded`) | The engine answers `/health`, but three canaries in a row failed with no success between | `journalctl --user -u flatshare-engine -n 200` on the node, `flatshare_gateway_canary_probes_total`, and the gateway's journal at `log_level = "debug"`. Any success serves it again, else restart the engine |
| <a id="flatshareenginewarming"></a>`/ready` 503 `engine warming` for minutes (`FlatshareEngineWarming`) | No inference has succeeded since the gateway started. An engine starts in about 9 minutes on cold compile caches, as after an upgrade, and in about 1.5 minutes on warm ones ([the starts](../docs/BENCHMARKS.md#what-does-a-failure-do-to-the-calls-and-the-agents)) | Wait for the start, whose caches then persist in `~/flatshare/cache/flashinfer` and `~/.cache/vllm`. If it has started, as for `engine degraded` |
| <a id="flatsharekvpoolfullwithwaiting"></a>`FlatshareKvPoolFullWithWaiting`: a KV pool over 95% full with requests waiting for 10 minutes | Agents' KV fills the pool, [the regime](../docs/PRODUCTION.md#is-your-deployment-in-this-regime) where a voice turn waits without the change | Expected beside agents under `make-room-serve`, so watch the first-byte objective. Without it, run `deploy/phastform/variant.sh use make-room-serve`, or keep calls off that node |
| `flatshare-engine` fails at start: "Free memory on device … less than desired GPU memory utilization" | Another process holds GPU memory, or the audio servers grew | `nvidia-smi`. The engine needs 10.83 GiB free at start |
| `flatshare-stt` fails at start with the same message | The STT started while the engine was above its steady state | Start it once the engine is ready (2.32 GiB free); `Restart=on-failure` retries it |
| A unit will not start (`start-limit-hit`) | A fourth start within 30 min, deliberate restarts counted too (`StartLimitBurst=3`) | `journalctl --user -u <unit> -n 200`, fix, `systemctl --user reset-failed <unit>`, start |
| The engine's journal: "systemd-oomd killed N process(es) in this unit" | Host memory pressure over the `ManagedOOMMemoryPressure=kill` limit of 50%, the engine holding the 20 GiB CPU tier | The unit restarts by itself. Once `/proc/pressure/memory` is near 0, rerun the campaign, which resumes at the failed point |
| A GPU server fails: "unshare failed: Operation not permitted" | The kernel refuses unprivileged user namespaces | Allow them (`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`) and restart the unit. Otherwise run Gray's install ([Gray](#a-gpu-node-without-user-namespaces-gray)), or a single host's `bootstrap.sh stage`, again |
| `make-room-serve` fails at start: "serving-check: N setting(s) differ from rtx-5060-ti-serve's" | Engine flags over the variant, an unnamed module or patch, or another vLLM; the lines above name each setting | `variant.sh use make-room-serve` without flags. A changed profile is a new release: `make profiles UPDATE=1`, commit, `deploy.sh` |
| The engine fails at start: "make_room.reserve_tokens leaves no room for a request of max_model_len tokens" | The reserve and one `max_model_len` request do not fit in the KV pool | Lower `make_room.reserve_tokens`, to about 3,600 tokens on the frozen box |
| The engine's `/server_info` answers 500 | vLLM cannot find uv | The unit's `PATH` must include `~/.local/bin`, and `VIRTUAL_ENV` must name the engine venv |

### Tunnels and keys

| Symptom | Likely cause | Action |
|---|---|---|
| The tunnel is down: `curl --unix-socket $XDG_RUNTIME_DIR/flatshare/phastform-engine.sock localhost/health` fails on cyan | A network blip, a restarted GPU node, or the node's server down | `flatshare-tunnel@phastform` reconnects within 30 s and restarts 5 s after it exits. Its journal says why each connection ended, and on the node `ls -l /run/user/$UID/flatshare/` lists the servers' sockets |
| `flatshare-tunnel@NODE`: "Permission denied (publickey)" | The node lacks the key's entry | `deploy/authorize-tunnel.sh cyan NODE_HOST` |
| `flatshare-tunnel@NODE` failed with status 2 | `~/.config/flatshare/tunnel-NODE.env` names no target, or a session count under 1 | Fix the env file, then `systemctl --user restart flatshare-tunnel@NODE` |
| Slow first bytes, and sshd logs "no more sessions" | The node's sshd allows fewer sessions a connection than `FLATSHARE_TUNNEL_SESSIONS` (10) | Set `FLATSHARE_TUNNEL_SESSIONS` in `tunnel-NODE.env` to the node's `MaxSessions` |
| `flatshare-bench soak run`: "the guard cannot read NODE's GPU" | cyan's guard key is not authorized on the node, or the node has no tunnel target | `deploy/authorize-guard.sh cyan NODE_HOST`, or `--guard NODE=ssh:TARGET`, or `--guard NODE=off` |

### Two nodes

| Symptom | Likely cause | Action |
|---|---|---|
| <a id="flatsharenodeunhealthy"></a>A node is unhealthy (`FlatshareNodeUnhealthy`) | Its engine does not serve or its tunnel is down, or a speech server does not answer `/health` | `/nodes` names the component; read its journal on the node, and `flatshare-tunnel@NODE`'s on cyan. Its calls moved to the other node. Restart the failed server, or let `Restart=on-failure` do it ([recovery](#two-gpu-nodes)) |
| <a id="flatsharenodecovering"></a>A node has covered for 5 minutes (`FlatshareNodeCovering`) | It carries calls moved to it, or calls placed on the spare while the first node was down | Expected until those calls end. `flatshare_gateway_node_failover_waiting` counts its agents' waiting requests |
| <a id="flatsharecallsrefused"></a>Calls get 503 `no_call_slot`, or 502 when retried after their node failed (`FlatshareCallsRefused`) | Every available realtime node has `max_calls` calls | Bring the failed node back, or admit fewer calls at the worker. A slot frees when its call ends or `idle_s` after its last turn |
| With `[[nodes]]`, agents' requests get 503 `no_agent_slot` | Every available background node has `max_agents` agents' sessions, or a request without a session met a node carrying a call | `/nodes` shows each node's `max_agents` and `sessions`. A slot frees when its session ends or idles out |

### Voice plane

| Symptom | Likely cause | Action |
|---|---|---|
| LiveKit is unreachable from the worker | The voice-plane namespace was restarted | `systemctl --user restart flatshare-voicenet flatshare-livekit`. Clients run inside it (`in-voicenet`) |
| Some voice turns wait about 3 s before the agent answers | LiveKit's end-of-turn model waits its maximum delay | Link the `vad-turns` worker variant |
| A caller run fails: "the agent never listened" | The worker is not registered, or LiveKit restarted under it | `journalctl --user -u flatshare-voice-worker -n 50`, then restart the worker |
| The worker restarts every 5 s: "FLATSHARE_… must be …" or "… is not set" | A drop-in sets a value the worker does not take, or leaves out a socket ([the settings](../python/flatshare_voice/README.md#the-voice-agent)) | Fix the drop-in (`systemctl --user cat flatshare-voice-worker`), `systemctl --user daemon-reload`, restart |
| In serve mode callers get no agent: "refusing a call: …" | The box is full ("2 of 2 calls in progress"), `/ready` fails ("the engine does not serve"), or a drain holds calls off ("draining: … exists") | `cat $XDG_RUNTIME_DIR/flatshare/voice-worker.json`. A drain file left by a manual `drain` takes `flatshare-release undrain` |
| The worker exits at start: "FLATSHARE_TURN_DETECTION=vad is the synthetic callers' fixed endpointing …" | `serve` is linked beside `vad-turns` | Unlink `variant-vad-turns.conf` ([the voice plane](#the-voice-plane)) |
| "lifecycle events do not reach the control plane" | The gateway is down, or `FLATSHARE_LLM_UDS` is not its socket | `systemctl --user status flatshare-gateway`. Lost events are not resent, and session states time out after `state_timeout_s` |

### Releases

| Symptom | Likely cause | Action |
|---|---|---|
| `deploy.sh` exits 1: "release ID fails its checks (…): nothing switched" | A check of step 2 failed, named in the lines before | The host runs what it ran. Fix it (a model: the bootstrap; a binary: `--gateway`) and deploy again |
| `deploy.sh` exits 1: "release ID failed after the switch: …", then "the host runs OLD again" | A restarted unit or a check failed, and the host switched back | `journalctl --user -u UNIT -n 100` for the unit named; `deploy.sh HOST --status` |
| `deploy.sh cyan` fails at "the gateway's nodes against the serving profile's envelope", naming `nodes.NODE.max_calls`, `nodes.NODE.max_agents` or `placement.failover_background_max_in_flight` | `gateway.toml`'s `[[nodes]]` run outside the profile's envelope, such as `max_agents` over 24 on the node that carries calls | Set the named key to what the check prints, reload the gateway, release again |
| `deploy.sh` exits 3: "switching back to OLD failed too" | Something outside the releases changed: the GPU, a model, a file in `~/.config` | `systemctl --user list-units 'flatshare-*'`; `flatshare-release rollback --to ID` once fixed |
| "the voice worker still has N call(s) after S s: nothing switched" | Calls outlasted `--drain-timeout` | Deploy later, with a longer `--drain-timeout`, or `--force`, which cuts them |
| "an E2 point (pid N) measures on this host" | A point or drill holds `~/flatshare/runs/.flatshare-bench-e2.lock` | Wait for it; `--force` releases anyway |
| `deploy.sh cyan`: "the units carry an E2 point's drop-ins (…)" | A point left its `variant-*.conf` linked, E2's gateway overlay among them | `flatshare-bench e2 reset` on cyan, then release; `--force` releases anyway |
| "the switch from A to B begun at … did not finish" | A killed run or a host down mid-switch | Deploy again. The switch restarts and checks every active unit |
| "is a deployment from before releases: deploy with --adopt" | A real `~/flatshare/src` | `deploy.sh HOST --adopt` once, in a quiet window, on cyan with `--gateway FILE`. Its first switch restarts every active unit |
