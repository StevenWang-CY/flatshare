# Runbook

How to deploy, check, operate and recover the Flatshare hosts, for whoever operates the box and its control host; a deployment on other hardware, or on one host, follows [HARDWARE.md](../docs/HARDWARE.md). Every service runs as a `systemd --user` unit. Lingering is enabled on each host, so units survive logout and start at boot.

| Host | Units | Listeners |
|---|---|---|
| Phastform (GPU) | `flatshare.target` → `flatshare-stt`, `flatshare-tts`, `flatshare-engine` | No TCP: the engine, STT and TTS on Unix sockets in `/run/user/$UID/flatshare/` (`engine.sock`, `stt.sock`, `tts.sock`), a directory only the deployment's user may enter. Variant `sglang` serves 127.0.0.1:8000 too, behind `engine.sock` (`flatshare-engine-proxy.socket`) |
| Gray (second GPU node, booked windows; held by another user's work since 2026-10-23) | The same units as Phastform, with Gray's drop-ins (`deploy/gray/systemd/`: the CUDA compiler from the engine venv, the engine on cores 0–7 of NUMA node 0, the GPU's, and the audio servers on cores 8–15 of node 1). `flatshare.target` is not enabled at boot | As on Phastform |
| cyan (control) | `flatshare-tunnel@phastform`, `flatshare-gateway`, `flatshare-voicenet`, `flatshare-livekit`, `flatshare-voice-worker`; `flatshare-smg` for E2 arm 8b only | The gateway's 127.0.0.1:18080, the only TCP listener in the host's network. Unix sockets in `/run/user/$UID/flatshare/` (0700): the gateway's `gateway.sock`, and the GPU node's through its tunnel, `phastform-engine.sock`, `phastform-stt.sock` and `phastform-tts.sock`. LiveKit 17880 (TCP) and 17882 (UDP), SMG 18090 and its metrics 18091 (arm 8b), inside the voice-plane namespace only; the gateway reaches SMG at `smg.sock` |

**Cores on cyan** (48 threads):

| Cores | Used by |
|---|---|
| 0–7 | The control plane (`flatshare-gateway`) |
| 8–15 | agentic-api and tools |
| 16–31 | Load generators: AIPerf on 16–23, `flatshare-bench` on 16–31, callers on 24–31 |
| 32–47 | The voice plane: LiveKit and the worker |

The `flatshare-gateway`, `flatshare-livekit` and `flatshare-voice-worker` units set their own affinity, and load generators run under `taskset`.

## Deploy

Deploy from a workstation with SSH access to the hosts. The commands name the hosts by SSH aliases (`phastform`, `cyan`, `gray`); any host name or alias that SSH resolves works, as long as its first label is the host's directory under `deploy/`, from which `deploy.sh` releases.

```sh
export FLATSHARE_SSH="ssh -F $HOME/.ssh/flatshare_config"   # your SSH command for the hosts
deploy/deploy.sh phastform                       # release HEAD's commit to Phastform: prepare, verify, switch
deploy/build-gateway.sh peacock                  # this commit's gateway, built on a Linux host with rustup
deploy/deploy.sh cyan --gateway build/flatshare-gateway-<id>
deploy/deploy.sh gray                            # the second GPU node
deploy/deploy.sh phastform --status              # its releases, current and previous
deploy/deploy.sh phastform --rollback            # back to the previous release (--to ID: another one)
```

**A release** is the commit `HEAD` names (`git archive`) and nothing else, no untracked, ignored or uncommitted file, with `DEPLOYED`, the commit's `git describe`, which every E2 point records: `deploy.sh` refuses a tree that has changes it would not carry. It uploads the release to the host and runs [`deploy/bin/flatshare-release`](../deploy/bin/flatshare-release) there, which keeps the deployment under `~/flatshare`:

| Path | What it is |
|---|---|
| `releases/ID/` | One release, ID the commit's 12-character abbreviation: its tree, read-only, with `MANIFEST.json` and `venvs/NAME`, links to its environments. The packages are compiled in it, so that nothing writes to it |
| `envs/NAME-KEY/` | A Python environment, KEY the SHA-256 of its lock file and what else decides it (a project's `pyproject.toml` and its path dependencies', uv's arguments, a vendored checkout's commit). A release whose inputs give the same KEY links it; only a changed lock builds one |
| `current`, `previous` | Links to the release the host runs, replaced in one rename, and to the one it replaced |
| `src` → `current` | The path the units and scripts name (`%h/flatshare/src/...`), and E2's `src/DEPLOYED` |
| `venvs/NAME` → `../current/venvs/NAME` | The units' `%h/flatshare/venvs/NAME/...`; `venvs/sglang`, the sglang variant's, stays a directory of its own |
| `bin/flatshare-gateway` → `../current/bin/flatshare-gateway` | On cyan, the gateway's binary, the release's; `bin/` keeps the host's own tools (`livekit-server`, `uv`) |
| `releases/history.jsonl`, `releases/running.json` | A line per switch, rollback, switch back and adoption; the release each unit was last started under |

**The manifest** (`releases/ID/MANIFEST.json`) says what the release is: the commit and its `git describe`; every file's SHA-256 and the tree's; the lock files (every `uv.lock`, `Cargo.lock` and requirements file); each environment's key, inputs, interpreter, build time, uv version and the hash of its package list; the host's own binaries that its units run (cyan's `livekit-server`), by SHA-256 and version; the gateway's binary on cyan, with its SHA-256 and its provenance (the commit, build host, toolchain and `Cargo.lock` it was built from, or the release it was carried from); the models' and tokenizer's revisions; make-room's patches and the scheduler module they build (`serving.json`'s hashes); the serving profile's effective configuration, `serving.json`; each unit's inputs, hashed; and on a GPU node whether each server runs in a network namespace of its own or in the host's (`FLATSHARE_PRIVATE_NET=host`, as on Gray, whose kernel refuses unprivileged user namespaces).

**The steps** (`flatshare-release deploy`, which `deploy.sh` runs):
1. **Prepare.** The upload must hash to the tree `deploy.sh` staged. Each environment no release has built yet is built from its lock (the engine's with `--torch-backend cu130`, the TTS's with cu128 and the Kokoro-FastAPI checkout installed editable at its pinned commit; cyan's three from their locks, `uv sync --locked`), a build cut short is removed and built again, the manifest is written, the tree made read-only and renamed into `releases/ID`. The same commit again is the release it was.
2. **Verify**, before anything switches: every file against the manifest; each environment's imports, offline as the units run (`import vllm` at 0.32.0 and torch, the TTS's torch and Kokoro's version, the voice worker and the harness from the release); `flatshare-gateway --version` against the source's; on a GPU node the serving profile's patches and the scheduler module they build on the installed vLLM, against `serving.json`'s hash, and the Kokoro checkout at its pin; the drop-ins the host links into its units, which must exist in the release; and the models and `livekit-server`, required where a unit that runs them runs or will restart (a paused box only reports them). A check that fails switches nothing.
3. **Drain** what the switch would cut: on cyan, the voice worker in serve mode takes no new call (its drain file) and the release waits for the calls in progress to end, at most `--drain-timeout` (300 s), and switches nothing if they do not, unless `--force`; in bench mode it has no drain, and a release waits for no E2 point: one that holds the host's lock stops the release unless `--force`. On a GPU node the release waits for the engine's requests in flight to end, 60 s at most (agents keep it busy), and its restart cuts those left (the gateway answers 503 meanwhile).
4. **Switch**: `previous` to what `current` names, `current` to the release, the host's `install.sh` (its unit links, through `~/flatshare/src`), `daemon-reload`.
5. **Restart or reload** each active unit whose inputs changed: the release files it reads, its environments and binaries, compared by hash with the release before. A release of documentation or of cyan's files restarts nothing on a GPU node; a new gateway binary reloads the gateway, which hands its sockets over without a gap, and a change to its unit restarts it; `flatshare-voicenet` takes LiveKit, SMG and the voice worker with it. An inactive unit stays stopped: a paused box starts the new release when it is started.
6. **Check**: each restarted unit active and not restarting for `--settle` seconds (10); on the box, the engine's configuration against `serving.json` where it runs `make-room-serve`, printed; on cyan, the gateway's `/health` and, in serve mode, the voice worker's admission against `serving.json`; the listener audit (`deploy/bin/audit-listeners`).
7. **On any failure after the switch** it switches back: `current` to the release before, `previous` as it was, the same units restarted on it and checked; `deploy.sh` exits 1, or 3 if switching back failed too. On success it keeps the newest 3 releases (`--keep`), current, previous and any release a running process imports from, and removes the rest and the environments none of them links.

A unit's Python processes import the release current named when the first of them started, whatever a switch does meanwhile: the environments' editable installs read the release that `FLATSHARE_RELEASE` names, which that first interpreter sets from `current` and its children inherit (the voice worker's job processes, an E2 point's tools), and pruning keeps any release a process names so. Do not release to cyan while an E2 campaign runs: a release refuses only while a point holds the lock, and one between two points would change the arm files and the harness that the next point reads. `--no-switch` prepares and verifies only, to warm a release before a window; `deploy.sh HOST --rollback` switches to `previous` by steps 2 to 7. Do not run scripts from a release's own directory: links made from there (by `variant.sh`, say) would outlive it, which is why the install scripts link through `~/flatshare/src` whichever path ran them.

**The gateway's binary.** `deploy/build-gateway.sh BUILD_HOST [OUT]` builds this commit's on a Linux host with rustup whose glibc is not newer than cyan's (peacock and cyan run Ubuntu 22.04), from `git archive HEAD` with `cargo build --release --locked -p flatshare-gateway`, into `build/flatshare-gateway-ID` and its provenance beside it (`.json`), and `deploy.sh cyan --gateway FILE` puts it in the release, refusing one built from another commit. Without `--gateway`, `deploy.sh` carries the binary of the release cyan runs when its crates (`crates/`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`) are this commit's, and refuses otherwise. `flatshare-gateway --version` names the version it was built at, which every E2 point records (`versions.flatshare_gateway`).

**Drain the voice plane before a GPU node's release.** A release of Phastform restarts its engine when the engine's inputs changed, and calls on it break. To release it without cutting calls, stop cyan's worker taking them first and let it again after: `$FLATSHARE_SSH cyan python3 flatshare/src/deploy/bin/flatshare-release drain` (serve mode; it waits for the calls in progress), then the node's `deploy.sh`, then `... flatshare-release undrain`.

**The first release on a host deployed before releases** (both hosts, once, paused or idle, with no E2 point or drill running): `--adopt` moves the old layout into a release in place before it prepares the new one: `~/flatshare/src` becomes `releases/legacy-REV` (REV its `DEPLOYED`) and `current`, the environments under `venvs/` become `envs/NAME-legacy`, cyan's `bin/flatshare-gateway` moves into the adopted release, and each old path becomes a link into `current`, each by one rename, so that a unit that runs keeps its files. The new release then builds its environments afresh and switches as above. The adopted tree need not be what its units run (the old deploy restarted none of them), so the adopted release has no unit hashes, and this first switch restarts every active unit, once: on cyan the voice plane, the tunnel and the gateway with its socket, so release it in a quiet window. `--rollback` returns to the adopted release, which has no `flatshare-release` of its own: `deploy.sh HOST --status` and `--rollback` then run the newest release's, so a second `--rollback` goes forward again. `deploy.sh HOST` does too (the release is kept, and switched to), on cyan with `--gateway FILE` again, since the adopted binary has no build to carry from.

```sh
deploy/build-gateway.sh peacock
deploy/deploy.sh phastform --adopt
deploy/deploy.sh cyan --adopt --gateway build/flatshare-gateway-<id>
```

On cyan, `install.sh` also disables and unlinks the units of this tree that the repository no longer has (the tunnels' from before `flatshare-tunnel@.service`, say); one that still runs keeps running until the next reboot, so stop it by hand (`systemctl --user stop UNIT`).

**First time on a host:**
1. Enable lingering: `loginctl enable-linger $USER`.
2. Release to it: `deploy/deploy.sh HOST` builds its Python environments from the lock files (uv 0.11.27 or newer, in `~/.local/bin` or set by `FLATSHARE_RELEASE_UV`) and switches to the release; nothing runs yet.
3. Run its bootstrap from the release, `bash ~/flatshare/src/deploy/HOST/bootstrap.sh`:
   - Phastform: the Kokoro checkout at its pin and its weights, the FlashInfer JIT workspace and the curand header links, and the models at the revisions `engine.yaml` and `stt.yaml` pin.
   - Gray: Phastform's.
   - cyan: the LiveKit server, a check that the worker's models load offline, the LiveKit keys in `~/.config/flatshare/livekit.env` (mode 600) and the three tunnel keys (`~/.ssh/flatshare_tunnel_{engine,stt,tts}_ed25519`). Write the tunnel's SSH target to `~/.config/flatshare/tunnel-phastform.env` as `FLATSHARE_TUNNEL_TARGET=<user>@<gpu node>`.
4. Authorize the tunnel keys on each GPU node: `deploy/authorize-tunnel.sh cyan phastform` (and `gray`). Each key's entry runs one command for every session, whatever its client asks, the bridge to the node's socket of its name (`exec nc -q0 -U /run/user/$UID/flatshare/engine.sock`), under `restrict`: no forwarding of any kind, no pty, no agent, no `~/.ssh/rc`. The node needs OpenBSD's netcat (Ubuntu's `netcat-openbsd`). Re-running the script replaces the entries, and removes those of cyan's former keys: the single tunnel key that forwarded the node's ports, `~/.ssh/flatshare_tunnel_ed25519` (`lh_tunnel_ed25519` before the rename), and the sockets' keys under the former name, `lh_tunnel_engine_ed25519` and so on. They may then be deleted on cyan.
5. On cyan, configure the gateway:
   - Write `~/.config/flatshare/gateway.toml` (mode 600) from `deploy/cyan/gateway.example.toml`. For each keyed tenant, make a key, give it to the client, and put only its digest in the file: `printf '%s' "$key" | sha256sum`. `cache_salt_secret_file` names the secret the tenants' cache salts are keyed by, which `deploy/cyan/install.sh` writes once (32 random bytes in hex, mode 600); the gateway refuses to start without it, or if any other user may read it. `metrics_key_file` names the key that `/metrics` asks for on the TCP listener, which `install.sh` writes once too; with a TCP listener the gateway refuses to start without it, or if any other user may read it.
   - `systemctl --user enable --now flatshare-gateway.socket flatshare-gateway`. The socket unit holds the gateway's listeners, 127.0.0.1:18080 and `gateway.sock`, and hands them to each gateway process ([below](#the-gateway)).

**A host deployed before the rename** (v1.1 and earlier, as Leaseholder): `deploy.sh` installs the `flatshare-*` units beside the old ones, and nothing reads the old names, so the host moves first, paused and with no E2 point or drill running. Both hosts moved this way on 2026-10-26:
1. Disable the `lh-*` units (and `leaseholder.target` on a GPU node), remove their links and drop-in directories from `~/.config/systemd/user`, and run `systemctl --user daemon-reload`.
2. Move `~/leaseholder` to `~/flatshare` (on cyan `~/.config/leaseholder` to `~/.config/flatshare` too), point the absolute links into the old tree at the new one (cyan's long traces), and remove `$XDG_RUNTIME_DIR/lh`.
3. On cyan, rename the variables in the env files from `LH_*` to `FLATSHARE_*`, their values unchanged; in `gateway.toml`, the paths under `/run/user/UID/lh/` and `~/.config/leaseholder/` and the units it names; and the tunnel keys from `~/.ssh/lh_tunnel_*` to `flatshare_tunnel_*`.
4. Deploy as on a new host: `deploy.sh`, the bootstrap (which rebuilds the venvs, whose scripts name the old tree), the gateway, `authorize-tunnel.sh` (which takes the entries of the keys' former names off the node too) and the units; on Phastform, `variant.sh use` the variant the box ran.

## Start, stop, status

```sh
# Phastform
systemctl --user start flatshare.target          # the three servers start at once; the box is ready when the engine is
systemctl --user stop flatshare.target
systemctl --user restart flatshare-engine        # one server; returns once /health answers
systemctl --user list-units 'flatshare-*'
journalctl --user -u flatshare-engine -f         # logs
bash ~/flatshare/src/deploy/phastform/smoke.sh   # end-to-end voice turn, GPU memory, listener audit
nsenter --target "$(systemctl --user show -p MainPID --value flatshare-engine)" --user --net --preserve-credentials \
  ss -lntup                                      # what listens inside the engine's own network namespace

# cyan
systemctl --user start flatshare-tunnel@phastform flatshare-voicenet flatshare-livekit flatshare-gateway
socket_dir=unix:$XDG_RUNTIME_DIR/flatshare           # the tunnel's sockets
bash ~/flatshare/src/deploy/phastform/smoke.sh $socket_dir/phastform-engine.sock $socket_dir/phastform-stt.sock \
  $socket_dir/phastform-tts.sock
python3 ~/flatshare/src/deploy/bin/audit-listeners   # no TCP listener of ours but the gateway's
~/flatshare/src/deploy/bin/in-voicenet ss -lntup     # what listens inside the voice-plane namespace
```

A unit counts as started only once its server answers its health endpoint (`wait-healthy` in `ExecStartPost`), the gateway once it says it is ready (`Type=notify`). Units that depend on it start only then.

## The gateway

`flatshare-gateway` serves the OpenAI routes and `/v1/events` on 127.0.0.1:18080, with an API key, and on `/run/user/$UID/flatshare/gateway.sock`, the voice plane's way in, without one. Its own `/health` and `/ready` need no key; `/metrics` and `/metrics/engine` (the engine's own, relayed) need the metrics key on the TCP listener and nothing on the socket. Its configuration reference and metrics list are in the [`flatshare-gateway` README](../crates/flatshare-gateway/README.md), including the E2 settings: `upstream_priority` (where each class's priority goes upstream: `X-Vllm-Priority`, the body's `priority`, or `X-SMG-Priority`), and the `[admission]` policies.

`flatshare-gateway.socket` holds the listening sockets and passes them to the gateway (systemd's socket activation), so they outlive any one gateway process: a connection made while no gateway accepts waits in the socket's backlog instead of being refused.

- **Reload** to take a new binary or a changed `gateway.toml`: the gateway starts its successor from the files on disk and hands it the sockets, the successor serves as soon as the old process says that it admits nothing more (at once, 2 s at most), and the old process drains, so no connection is refused. Until the old process exits, the successor counts its requests in flight and its admission point's reservations with its own, so that `max_in_flight`, the classes' and tenants' bounds and an `[admission]` policy hold over both (`flatshare_gateway_predecessor_in_flight`); a request still waiting at the old process's admission point gets 503 `draining` with `Retry-After: 1`, never having gone upstream. Copy a new binary with `install -m 755 target/release/flatshare-gateway ~/flatshare/bin/`, which replaces the file instead of writing into it: a running binary cannot be overwritten. A reload whose successor does not come up within 10 s (a broken `gateway.toml`, say) fails and changes nothing: the old process serves on, and the journal says why.
- **Restart** stops before it starts, so the connections made during its drain wait for it (realtime and interactive requests up to `drain_timeout_s`, 30 s; background ones are cut after `background_drain_s`, 5 s), and are then served.
- **A stop or restart after a reload** drains too: the unit's `ExecStop` (`deploy/bin/flatshare-gateway-stop`) sends the SIGTERM and waits for the gateway to exit, since systemd 249 would not (the successor was its predecessor's child when it said it is the main process, which the journal's "Supervising process … which is not our child" notes) and would send SIGKILL at once. The wait ends after 45 s (`--wait` on the `ExecStop` line: the drain's 30 s and a margin; raise it with `drain_timeout_s`): `flatshare-gateway-stop` then sends SIGKILL and fails, so a gateway that hangs in its drain holds up a stop or restart for 45 s, not for twice systemd's 90 s stop timeout. On a single host `deploy/bin/flatshare-svc` runs the same `ExecStop`, with `%n` expanded as systemd expands it, and sends the SIGTERM itself if the command fails.
- **A change to `[[listen]]`** takes a restart of both units, with `flatshare-gateway.socket`'s `ListenStream=` changed to match: the gateway refuses a socket that no `[[listen]]` entry names.

```sh
systemctl --user reload flatshare-gateway           # a new binary or gateway.toml, without a gap; fails if the successor does not take over
systemctl --user restart flatshare-gateway          # drains (background requests cut after 5 s), then starts; connections wait meanwhile
systemctl --user show -p MainPID flatshare-gateway  # a reload changes it: the successor is the unit's main process
```

**Moving a gateway that binds its own sockets to the socket unit** (the first deploy of `flatshare-gateway.socket`) takes one stop and start, during which connections are refused: `systemctl --user daemon-reload`, `systemctl --user stop flatshare-gateway`, `systemctl --user enable --now flatshare-gateway.socket`, `systemctl --user start flatshare-gateway`. Do not start the socket unit while the old gateway runs: its port is taken. The gateway's `[[listen]]` entries in `gateway.toml` stay as they are (`deploy/cyan/gateway.example.toml`'s).

```sh
curl -s 127.0.0.1:18080/health                    # the process is up
curl -s 127.0.0.1:18080/ready                     # 200 while the engine serves and its /health answers; else 503 and why
curl -s --unix-socket $XDG_RUNTIME_DIR/flatshare/gateway.sock localhost/metrics | grep -E 'flatshare_gateway_(request_outcomes|upstream_errors)_total'
journalctl --user -u flatshare-gateway -f         # one JSON line per request and per lifecycle event
```

E2 reads each point's request lines from this journal, so `gateway.toml` keeps `log_format = "json"`, as the example does.

`/ready` answers by the engine's state as inference finds it, not by its `/health` alone, which vLLM answers while its process lives: `warming` from the gateway's start until an inference succeeds, `serving`, `degraded` once the gateway's inference canary has failed three times in a row (`[canary] degraded_after`) with no request succeeding between, and `drained` in a drain; only `serving` is ready (`flatshare_gateway_engine_state`). The canary is a one-token completion at background priority, sent only when no request succeeded for `[canary] interval_s` (30 s) and every second while the engine warms, so a busy gateway sends none; a client's success counts as a passing canary. Requests go upstream in every state. A gateway that has just started or reloaded is warming for the moment its first canary takes, and E2's points run with the canary off ([E2 points](#e2-points)).

## Monitoring

`ops/prometheus/prometheus.example.yml` scrapes the gateway's `/metrics` and, through the gateway, the engine's (`/metrics/engine`), with the metrics key; run Prometheus as the deploy user, so that it may read the key's file. `ops/prometheus/rules.yml` holds the SLIs, the burn-rate alerts of [the SLO's two objectives](../docs/SLO.md#indicators) (a page at 14.4 times the budget's rate over an hour or 6 times over six hours, a ticket at 1 times over three days) and the health alerts: the gateway unreadable, the engine not ready (the gateway answering 503), the engine degraded (its inference failing while its `/health` answers: a page) or still warming 15 minutes after the gateway's start (a ticket), the engine's metrics unrelayed, streams broken off, requests refused for capacity or ended by their deadlines (tickets), and a full KV pool with requests waiting. The realtime objective reads the gateway's failure-inclusive service level, 1 − good/valid by class ([the gateway's README](../crates/flatshare-gateway/README.md#accounting)): a request that fails, is refused for capacity, times out or hangs is a miss, where the first byte's histogram it read before never saw one. `ops/prometheus/rules.test.yml` is promtool's unit test of the rules (a healthy class, an engine that fails every request, half the requests timing out, another class's misses, three days at 7%, the admission coverage, a degraded and a warming engine): `make check-rules PROMTOOL=<Prometheus 3.5.0's promtool>` checks the rules and runs it. `ops/grafana/flatshare.json` is the dashboard, with the miss fraction by class, the requests offered, admitted and ended by outcome, the admission coverage, the engine's state, the canaries and the requests in flight: import it with a Prometheus data source. The voice worker's LiveKit metrics are not scraped: its namespace has no route in. `ops/metrics/` lists the metric families of vLLM 0.32.0 and of the worker, from their sources; with the gateway's `crates/flatshare-gateway/metrics.txt` they are what `python/flatshare_bench/tests/test_ops.py` checks every expression against. `promtool check rules` and `check config` (Prometheus 3.5.0) pass on the files, the dashboard's expressions included.

## The voice plane

| Unit | What it runs | Notes |
|---|---|---|
| `flatshare-voicenet` | The network namespace holder | Loopback plus the dummy `flatshare0` (10.255.0.1/32), no route out |
| `flatshare-livekit` | LiveKit server in the namespace | 127.0.0.1:17880 (signalling), UDP 17882 (media) |
| `flatshare-voice-worker` | `flatshare-voice-worker start` (LiveKit Agents) in the namespace | Agent `flatshare-voice`. STT and TTS through `%t/flatshare/phastform-{stt,tts}.sock`, the LLM through `%t/flatshare/gateway.sock` |

```sh
systemctl --user start flatshare-tunnel@phastform flatshare-gateway flatshare-voicenet flatshare-livekit \
  flatshare-voice-worker
journalctl --user -u flatshare-voice-worker -f          # JSON logs; "registered worker" means it is ready
```

**Bench and serve modes.** The worker's unit runs bench mode, what every measurement ran: every call taken. The `serve` variant is the serving configuration: serve mode admits the serving profile's calls at once (two), with LiveKit's load threshold at 1 so that a full worker gets no job, takes no new call while the gateway's `/ready` says the engine does not serve or while a release drains it (the drain file, `$XDG_RUNTIME_DIR/flatshare/voice-worker.drain`), and writes its state every second to `$XDG_RUNTIME_DIR/flatshare/voice-worker.json`; its audio is the real-microphone configuration ([the modes](../python/flatshare_voice/README.md#bench-and-serve-modes)). To serve, unlink `vad-turns` (the microphone configuration refuses fixed endpointing) and link `serve`; `flatshare-bench e2 run` refuses a worker in serve mode.

```sh
units=~/.config/systemd/user/flatshare-voice-worker.service.d
rm $units/variant-vad-turns.conf
ln -s ~/flatshare/src/deploy/cyan/variants/serve/flatshare-voice-worker.conf $units/variant-serve.conf
systemctl --user daemon-reload && systemctl --user restart flatshare-voice-worker
python3 ~/flatshare/src/deploy/bin/serving-check ~/flatshare/src/deploy/phastform/serving.json \
  --voice $XDG_RUNTIME_DIR/flatshare/voice-worker.json --voice-pid "$(systemctl --user show -p MainPID --value flatshare-voice-worker)"
```

**Worker variants** are drop-ins linked into `~/.config/systemd/user/flatshare-voice-worker.service.d/`, followed by `systemctl --user daemon-reload` and a restart:
- `deploy/cyan/variants/vad-turns/flatshare-voice-worker.conf` sets fixed endpointing (`FLATSHARE_TURN_DETECTION=vad`). Every experiment runs with it, because LiveKit's end-of-turn model waits its 3 s maximum on some synthesized turns ([BENCHMARKS.md](../docs/BENCHMARKS.md#voice-on-the-shared-gpu)).
- An `Environment=FLATSHARE_LLM_UDS=%t/flatshare/phastform-engine.sock` drop-in sends the LLM straight to the engine, bypassing the gateway.
- `deploy/cyan/variants/no-preemptive-generation/flatshare-voice-worker.conf` turns LiveKit's preemptive generation off (`FLATSHARE_PREEMPTIVE_GENERATION=false`), for E2's factor.
- `deploy/cyan/variants/no-events/flatshare-voice-worker.conf` stops the worker's lifecycle events (`FLATSHARE_EVENTS=off`), for arm 9's ablation; the turn lines stay.
- `deploy/cyan/variants/serve/flatshare-voice-worker.conf` runs serve mode with the box's serving profile (above).
- `deploy/cyan/variants/qwen3-14b/flatshare-voice-worker.conf` makes the agent's requests name Qwen3-14B-FP8 (`FLATSHARE_LLM_MODEL`), for an engine that serves it in place of the box's model: the A100 kit's optional arm C' (`benchmarks/a100/cprime.sh` links it and removes it).

**Lifecycle events.** Each session's end logs its counts (`journalctl --user -u flatshare-voice-worker | grep '"session ended"'`: `events_sent`, `events_dropped`, `events_failed`), and the gateway counts what it received in `flatshare_gateway_events_total{source="voice-worker"}`. With the LLM on an engine's socket (the bypass) the worker sends none.

**Measurement callers.** Render the corpus once with the TTS server. The callers run inside the namespace:

```sh
~/flatshare/venvs/voice/bin/flatshare-voice render-callers ~/flatshare/data/callers --tts-url http://localhost/v1 \
  --uds $XDG_RUNTIME_DIR/flatshare/phastform-tts.sock
set -a; . ~/.config/flatshare/livekit.env; set +a
taskset -c 24-31 ~/flatshare/src/deploy/bin/in-voicenet ~/flatshare/venvs/voice/bin/flatshare-voice bots \
  --corpus ~/flatshare/data/callers --callers 2 --turns 25 --barge-in 0.25 --out ~/flatshare/runs/<run>/bots
```

## Engine variants

The frozen box is `deploy/phastform/engine.yaml` and the units in `deploy/phastform/systemd/`. `deploy/phastform/variant.sh use NAME|frozen [FLAG...]` makes the box run exactly one variant plus engine flags, restarting only what changes; `variant.sh use frozen` restores the frozen box. The variants, the flags the E2 arms use and how each was checked are in [`deploy/phastform/variants/README.md`](../deploy/phastform/variants/README.md):
- **Flags** go on the engine's command line through `FLATSHARE_ENGINE_FLAGS`, set by a generated drop-in (`variant-flags.conf`); vLLM takes the command line over `engine.yaml`.
- **`pr54530`**: vLLM #54530 ported to 0.32.0, generated at each start by `build.py` and loaded with `--scheduler-cls`.
- **`make-room-serve`**: the box's serving configuration, rendered from the serving profile `deploy/profiles/rtx-5060-ti-serve.toml`: `make-room` with the PRs' switches (2 preemptions a step, the budget cap with a 48-token window, no reserve) in its drop-in, under a scheduler class that writes the configuration the engine runs to `$XDG_RUNTIME_DIR/flatshare/engine-effective.json`. Each start holds it to the profile's `serving.json` (`deploy/bin/serving-check`, after the wait for `/health`) and fails on any difference. Serve with it as `variant.sh use make-room-serve` and `variant.sh measure off` ([the serving profile](../deploy/phastform/variants/README.md#the-serving-profile)).
- **`make-room`**: the change for #40004 on vLLM 0.32.0, arm 9's engine side, generated and loaded the same way: KV preemption for a waiting request of a higher priority, the budget cap that keeps lower-priority prefill out of its first tokens, and a reserve ([ARCHITECTURE.md §2](../docs/ARCHITECTURE.md#2-the-change-63350-and-63351) states the rules). Its switches are `--additional-config.make_room.*` keys: its unit sets arm 9's values, and an arm's engine flags override one. `/server_info` reports them under `additional_config`. `probe.py` measures a priority-0 request's first token behind two long priority-2 requests in a full pool (`kv`, with the reserve off), and its first and 16th tokens behind three agent prefills (`prefill`); `tail.py` measures the calls' first-token tail beside closed-loop agents.
- **`sglang`**: SGLang 0.5.22 in the engine's place, on 127.0.0.1:8000 behind the engine's socket (`flatshare-engine-proxy.socket`). Build its environment once with `bash ~/flatshare/src/deploy/phastform/variants/sglang/install.sh`. Its unit counts as started only after one sampled request, because the first one compiles FlashInfer's kernels for minutes.
- **`mps`**: the three servers as clients of one CUDA MPS daemon (`flatshare-mps.service`); the frozen box uses time-slicing ([BENCHMARKS.md](../docs/BENCHMARKS.md#voice-on-the-shared-gpu)).

`systemctl --user cat flatshare-engine` shows what the box runs. The generated variants (`pr54530`, `make-room`, `make-room-serve`) build their scheduler module when the engine starts, and `variant.sh` restarts only for a change of drop-ins or flags; a release restarts a running engine when its `build.py` or patches change, as they are among the engine's inputs, and a stopped one builds the new module when it starts. `variant.sh on|off NAME` adds or removes a single variant, for work outside E2. `variant.sh measure on|off` adds or removes the measurement drop-in, vLLM's development routes, which `flatshare-bench e2 run` and the drills read and a production box does without; a campaign turns it on itself ([the routes](../deploy/phastform/variants/README.md#the-measurement-drop-in)).

## E2 points

An E2 point is one arm file (`benchmarks/e2/arms/`); `flatshare-bench e2 run ARM --out DIR` measures it on cyan, and `benchmarks/e2/campaign.sh` runs a campaign file from a workstation, switching the engine before each point ([`benchmarks/e2/README.md`](../benchmarks/e2/README.md)). What a point changes on cyan, and how to undo it:
- **The gateway.** Every point runs the gateway from `~/.config/flatshare/gateway-e2.toml` through the `e2-gateway` drop-in: `gateway.toml` with the inference canary off (`[canary] enabled = false`, so that the engine serves only the point's requests and `/ready` answers as for the recorded campaigns) and the arm's `[gateway]` table merged in.
- **The worker.** `preemptive_generation = false` links the `no-preemptive-generation` worker drop-in, and `[voice] events = false` the `no-events` one.
- **SMG.** An arm with `router = "smg"` starts `flatshare-smg`, and every other arm stops it. SMG takes only http:// workers and listens on TCP only, so it runs in the voice-plane namespace between two proxies (`deploy/cyan/smg.sh`): the gateway reaches it at `$XDG_RUNTIME_DIR/flatshare/smg.sock`, and it reaches the engine at the namespace's 127.0.0.1:18000, the tunnel's socket. No other user of cyan reaches either port.
- **Undo.** `~/flatshare/venvs/bench/bin/flatshare-bench e2 reset` puts the gateway back on `gateway.toml`, preemptive generation and the events on, and SMG stopped. A campaign runs it at its end.

Beside AIPerf's and the callers' records, each point keeps the gateway's request lines (`gateway.jsonl`, the exact first bytes), the worker's journal (`voice-worker.log`: turn lines and event counts) and the GPU's clocks and temperature (`gpu.csv`, which `campaign.sh` samples over SSH once a second while the point runs). The [`flatshare-bench` README](../python/flatshare_bench/README.md#e2-points) lists every file.

One point runs at a time: the pid of the running one is in `~/flatshare/runs/.flatshare-bench-e2.lock` on cyan, and SIGINT to it stops AIPerf and the callers and marks the point failed.

**Demo points.** The demo video needs points whose callers saved their audio. A campaign file's `save-audio` line has `campaign.sh` pass `--save-audio` to its points, as `benchmarks/e2/demo.campaign` does; by hand on cyan, with the GPU node already on the arm's engine and the measurement drop-in on (`variant.sh measure on`), it is `~/flatshare/venvs/bench/bin/flatshare-bench e2 run ARM --out ~/flatshare/runs/e2/demo/POINT --save-audio`. Each caller writes `call-<i>.wav` and `call-<i>.json` beside its turns, and `make demo-video` renders the video from `demo.campaign`'s two points ([`benchmarks/e2/README.md`](../benchmarks/e2/README.md)).

## Drills

The failure drills run from a workstation with SSH to the hosts, on the variant the GPU node runs, with no E2 point running (they take the same lock on cyan). Each writes its directory under `~/flatshare/runs/drills` and prints its summary:
- `ops/drills/drill.sh engine-kill` opens two realtime streams (the voice plane's socket) and four background ones (the TCP port, the agents tenant's key) through the gateway, then sends SIGKILL to every process of `flatshare-engine`, as an OOM kill or a crash would end it. It records how each stream ended and when, what new requests get from the kill on, how long until one succeeds again, the engine's restart from its journal, and `/dev/shm` before and after, where the CPU tier lives. The final build's drill, on 2026-10-25: every stream got the `upstream_unavailable` error event 8 to 11 ms after the kill; the engine's main process was gone 0.14 s after it and its last process 7.6 s after it (the GPU's teardown, which systemd waits for); systemd started it RestartSec (5 s) after that; its warm start listened 92.1 s later; and the first request succeeded 106.2 s after the kill ([the drills](../docs/BENCHMARKS.md#failure-drills) set it beside W8's, before the W6 hardening).
- `ops/drills/drill.sh gateway-reload`, `gateway-restart` and `gateway-kill` do the same on cyan with `systemctl --user reload flatshare-gateway` (a deploy's), `restart flatshare-gateway`, and SIGKILL to the gateway (systemd starts it again a second later). The drill script writes the event's time to the run's `event` file in the same command as the event, and from the moment the client sees it, it probes both planes every second, each probe on a new connection, beside the streams, until they have ended and requests succeed again; every attempt is kept with when it was sent and answered. The client also samples the engine's running and waiting requests from the streams' first bytes on, each sample stamped, and the report says when none was left: a gateway that dies must not leave its requests running in the engine. A reload should show no failed probe and every voice turn whole; a restart no failed probe either, but answers that come only once its drain is done; a kill the in-flight streams broken, the engine's requests gone within a second or so, and the probes answered once the gateway is back.
- `~/flatshare/venvs/bench/bin/flatshare-bench drill cpu-tier --out FILE`, on cyan with the engine otherwise idle, stores twice the CPU tier's capacity of unique prompts and checks that the tier evicted the oldest prompt and still restores the latest.
- `ops/drills/cold-start.sh engine|box` restarts the engine alone or the whole box and times each server's start from its journal; `--cold` empties the compile and JIT caches first, as on a new host, and puts them back after, even when the start fails. A failed start still leaves its journals. The script refuses to start while an E2 point or a drill holds cyan's lock, but does not hold the lock itself: start nothing else meanwhile.

## Checks before heavy work on a shared host

- `nvidia-smi` and `who`. `gpu-guard` refuses to start a GPU unit while another user's process holds the GPU, or when `nvidia-smi` cannot list the GPU's processes. It cannot stop work that starts after the units; another user's processes run in their own tmux, which `tmux ls` does not show.
- While a campaign measures, `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` on the GPU node lists only the box's units (their cgroups end in `flatshare-*.service`). A point measured while any other process held the GPU is measured again.
- `free -g`, and `df -h /dev/shm`: the CPU tier occupies 20 GiB there.
- Listener audit, as `python3 ~/flatshare/src/deploy/bin/audit-listeners` checks it on either host (`deploy.sh` runs it after each install):
  - on Phastform, the audit with the three sockets expected finds the servers listening on them and none of the box's units on TCP, but for the SGLang variant's engine on loopback:

    ```sh
    python3 ~/flatshare/src/deploy/bin/audit-listeners --expect $XDG_RUNTIME_DIR/flatshare/engine.sock \
      --expect $XDG_RUNTIME_DIR/flatshare/stt.sock --expect $XDG_RUNTIME_DIR/flatshare/tts.sock
    ```

    The host's `ss -Hlx` does not list those sockets: each server listens in a network namespace of its own, whose sockets the audit reads from the namespace's own table (`ls -l /run/user/$UID/flatshare/` shows the files; the `nsenter` line under [Start, stop, status](#start-stop-status) shows what listens in the engine's namespace);
  - on cyan: `ss -Hlntup` shows no LiveKit socket on the host network.

## Failures

| Symptom | Likely cause | Action |
|---|---|---|
| `flatshare-engine` fails at start with "Free memory on device … less than desired GPU memory utilization" | Another process holds GPU memory, or the audio servers grew | `nvidia-smi`; stop the foreign job or wait. The engine needs 10.83 GiB free at start |
| `flatshare-stt` fails at start with the same message, the engine running | The three servers start at once, and the STT's check caught the engine above its steady state (its profiling pass or its graph capture) or above its budget | `systemctl --user start flatshare-stt` once the engine is ready: the STT needs 2.32 GiB free at start. `Restart=on-failure` retries it by itself |
| A unit restarts in a loop, then stops, or will not start (`start-limit-hit`) | A fourth start within 30 min: the units' crash-loop guard (`StartLimitBurst=3`) counts deliberate starts too, such as `systemctl --user restart` by hand | A crash loop: `journalctl --user -u <unit> -n 200`, fix, `systemctl --user reset-failed <unit>`, start. Deliberate restarts: `systemctl --user reset-failed <unit>` first (`variant.sh` does so before each switch) |
| `flatshare-engine` dies during a run and starts again 5 s after its last process exits; its journal says "systemd-oomd killed N process(es) in this unit" | A host-wide memory stall (swap full, `/proc/pressure/memory` high) held the user manager's memory pressure above its `ManagedOOMMemoryPressure=kill` limit (50% on these hosts), and `systemd-oomd` killed a cgroup under it: the engine, which holds the 20 GiB CPU tier in `/dev/shm`. Our units set no memory limit, and the host can have plenty of memory free | The unit restarts by itself (`Restart=on-failure`). A point measured across the kill fails: run its campaign again once `/proc/pressure/memory` is back near 0, and it resumes from that point |
| The first start after an upgrade takes many minutes | FlashInfer JIT and torch.compile caches are cold | Expected once; caches persist in `~/flatshare/cache/flashinfer` and `~/.cache/vllm` |
| The tunnel is down (`curl --unix-socket $XDG_RUNTIME_DIR/flatshare/phastform-engine.sock localhost/health` fails on cyan) | Network blip, a restarted GPU node, or the node's server is down (the connection closes at once) | `flatshare-tunnel@phastform` opens its SSH connections again after 1 to 30 s, or within 1 s while a connection waits for one, which it does up to 10 s before it connects on its own, and restarts itself 5 s after it exits; `journalctl --user -u flatshare-tunnel@phastform` says why each one ended. On the node, `ls -l /run/user/$UID/flatshare/` lists the servers' sockets |
| `flatshare-tunnel@NODE`'s journal: "Permission denied (publickey)" | The node lacks the key's entry | `deploy/authorize-tunnel.sh cyan NODE_HOST` |
| `flatshare-tunnel@NODE` failed with status 2 | `~/.config/flatshare/tunnel-NODE.env` names no target, or a session count under 1; the journal says which, and the unit does not restart a configuration it refuses | Fix the env file, then `systemctl --user restart flatshare-tunnel@NODE` |
| A tunnel's sessions all go through new connections (slow first bytes, sshd logs "no more sessions") | The node's sshd allows fewer sessions a connection than `FLATSHARE_TUNNEL_SESSIONS` (10, sshd's default `MaxSessions`) | Set `FLATSHARE_TUNNEL_SESSIONS` in `tunnel-NODE.env` to the node's `MaxSessions` |
| A GPU server does not start, its journal saying "unshare: unshare failed: Operation not permitted" | The kernel refuses unprivileged user namespaces (Ubuntu 24.04 restricts them), and `in-private-net` runs no server in the host's network namespace unless the host's units say so | Allow them (`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`) and restart the unit; or, on a single host, run its `bootstrap.sh stage` again: its preflight then renders the degraded path, `FLATSHARE_PRIVATE_NET=host` in the servers' drop-ins, where their engine cores' loopback listeners and CUDA's abstract sockets are any local user's to reach |
| LiveKit is unreachable from the worker | The voice-plane namespace was restarted | `systemctl --user restart flatshare-voicenet flatshare-livekit`. Clients must run inside the namespace (`in-voicenet`) |
| The gateway answers 502 `upstream_unavailable` or `/ready` is 503 `upstream not ready` | The tunnel or the engine is down | `curl --unix-socket $XDG_RUNTIME_DIR/flatshare/phastform-engine.sock localhost/health`; see the tunnel row above |
| `/ready` is 503 `engine degraded` (alert `FlatshareEngineDegraded`) | The engine answers `/health`, but its inference fails or hangs: the canary failed three times in a row and no request succeeded between | The engine's journal on the GPU node (`journalctl --user -u flatshare-engine -n 200`); `flatshare_gateway_canary_probes_total`, and the gateway's journal at `log_level = "debug"` for each canary's cause. Any success serves it again; restart the engine if none comes |
| `/ready` is 503 `engine warming` for minutes (alert `FlatshareEngineWarming` after 15) | No inference has succeeded since the gateway started: the engine is still starting (531.8 s with cold compile caches, 88.8 s warm; [the hardware notes](../docs/HARDWARE.md)), or it fails every request | Wait for the engine's start; if it has started, as for `engine degraded` |
| The gateway answers 503 `in_flight_limit`, `class_in_flight_limit` or `body_budget_exhausted`, or 429 `tenant_in_flight_limit` (alert `FlatshareRequestsRefusedForCapacity`) | The gateway, a class or a tenant has its `max_in_flight` requests in flight, or the bodies it holds would pass `max_buffered_body_bytes`; during a reload the old process's requests count too | `flatshare_gateway_in_flight` by class, `flatshare_gateway_predecessor_in_flight` and `flatshare_gateway_buffered_body_bytes`. One tenant's flood gets 429 and leaves the others room; if the load is real, raise the bound in `gateway.toml` and reload |
| Requests end with 504 `upstream_timeout`, streams with the `upstream_timeout` error event, or uploads with 408 `upload_timeout` (alert `FlatshareDeadlinesMissed`) | The engine sent no headers, first content or next frame within the class's deadlines, or a request ran past its total; a 408 is a client too slow to upload | The request's log line names the deadline (`error`: "… within classes.CLASS.KEY (N s)"); the engine's waiting requests and KV usage. Raise the class's deadline in `gateway.toml` only for work that is that long |
| The gateway answers 401 `invalid_api_key` | The client's key has no digest in `gateway.toml` | `printf '%s' "$key" \| sha256sum` must match an entry; reload the gateway after editing |
| `flatshare-gateway` exits at start: "another process is accepting on this socket" | A second gateway serves `gateway.sock`, or the socket unit holds it but the gateway runs outside its unit | Stop the other process, or start the gateway with `systemctl --user start flatshare-gateway`, which passes it the socket unit's sockets; a socket file left by a crash is replaced automatically |
| `flatshare-gateway` exits at start: "inherited sockets that no [[listen]] entry names" | `flatshare-gateway.socket`'s `ListenStream=` and `gateway.toml`'s `[[listen]]` disagree | Make them name the same addresses (`%t` is `/run/user/$UID`), then `systemctl --user daemon-reload` and restart both units |
| `systemctl --user reload flatshare-gateway` fails: "its successor did not take over" | The successor exited at start (a broken `gateway.toml`, a binary that does not run) or was not ready within 10 s; the old process serves on | `journalctl --user -u flatshare-gateway -n 50` says why ("reload failed: ..."); fix and reload again |
| `flatshare-gateway.socket` failed with `service-start-limit-hit`, and connections are refused | The gateway crashed five times within 10 s, so systemd stopped it, and the socket unit with it | `journalctl --user -u flatshare-gateway -n 200`, fix, `systemctl --user reset-failed flatshare-gateway flatshare-gateway.socket`, start both |
| Some voice turns wait about 3 s before the agent answers | LiveKit's end-of-turn model is waiting its maximum endpointing delay | Link the `vad-turns` worker variant |
| A caller run fails with "the agent never listened" | The worker is not registered ("registered worker" missing from its log), or LiveKit was restarted under it | `journalctl --user -u flatshare-voice-worker -n 50`; `systemctl --user restart flatshare-voice-worker` |
| `flatshare-voice-worker` restarts every 5 s without registering, its journal saying "flatshare-voice-worker: FLATSHARE_… must be …" or "… is not set" | A drop-in sets one of the worker's settings to a value it does not take, or leaves out a socket ([the settings](../python/flatshare_voice/README.md#the-voice-agent)) | Fix the drop-in (`systemctl --user cat flatshare-voice-worker` lists them), then `systemctl --user daemon-reload` and restart the worker |
| `flatshare-bench e2 run` fails at its engine check | The GPU node runs another variant or other flags than the arm names | `variant.sh use` with what `flatshare-bench e2 engine ARM` prints; a campaign does this itself |
| `flatshare-bench e2 run` or `flatshare-bench drill cpu-tier`: "no /server_info" | The engine runs without the measurement drop-in, as a production box does | `variant.sh measure on` (restarts the engine); a campaign does this itself |
| `flatshare-bench e2 run`: "another flatshare-bench e2 run or reset holds" the lock | A point is still running, or one was interrupted | The pid is in `~/flatshare/runs/.flatshare-bench-e2.lock`; stop it with SIGINT, or wait |
| The gateway does not start for a point, and `run.json` quotes its log | The arm's `[gateway]` table sets a key this gateway build does not know | Deploy a gateway build that has it, or fix the arm; `flatshare-bench e2 reset` restores `gateway.toml` |
| The gateway answers 429 `queue_full` or 503 `queue_timeout` | An `[admission]` policy's queue is full (`queue_limit`), its queues together hold `max_queued`, or a wait ran out (the policy's `queue_timeout_s`, or the class's) | Expected under the E2 policies, and counted in `flatshare_gateway_admission_refused_total` by reason |
| `flatshare-engine` (variant `make-room-serve`) fails at start, its journal saying "serving-check: N setting(s) differ from rtx-5060-ti-serve's: …" | The engine runs another configuration than the serving profile's: engine flags over the variant (`variant.sh use make-room-serve FLAG…`), a module or patch the profile does not name, another vLLM. The lines above the verdict name each setting and what the engine runs | `systemctl --user cat flatshare-engine`; `variant.sh use make-room-serve` without flags. A changed profile or patch is a new release: `make profiles UPDATE=1`, commit, `deploy.sh` |
| `deploy.sh` exits 1: "release ID fails its checks (…): nothing switched" | A check of step 2 failed: the lines before it say which (`FAIL files`, an environment's imports, a binary's version, the scheduler module, a drop-in the release lacks, a model or tool the units need) | The host runs what it ran. Fix it (a model: the bootstrap; a binary: `--gateway`) and deploy again; the release, prepared, is kept |
| `deploy.sh` exits 1: "release ID failed after the switch: …", then "the host runs OLD again" | A restarted unit did not stay active, or a check after the switch failed; the host switched back and restarted the same units on the release before | `journalctl --user -u UNIT -n 100` for the unit named; `deploy.sh HOST --status` |
| `deploy.sh` exits 3: "switching back to OLD failed too" | The release before does not come up either: something outside the releases changed (the GPU, a model, a configuration file in `~/.config`) | `systemctl --user list-units 'flatshare-*'`, the units' journals; `flatshare-release rollback --to ID` once fixed |
| `deploy.sh`: "the voice worker still has N call(s) after S s: nothing switched" | Serve mode's calls had not ended within `--drain-timeout`; the worker takes calls again | Deploy again later, with a longer `--drain-timeout`, or with `--force`, which cuts them |
| `deploy.sh`: "an E2 point (pid N) measures on this host" | A point or drill holds `~/flatshare/runs/.flatshare-bench-e2.lock` | Wait for it; `--force` releases anyway |
| `deploy.sh`: "is a deployment from before releases: deploy with --adopt" | The host has the layout from before releases, a real `~/flatshare/src` | `deploy.sh HOST --adopt` ([Deploy](#deploy)) |
| Callers get no agent while the worker runs serve mode; its journal says "refusing a call: …" | "2 of 2 calls in progress": the box is full, as the serving profile admits; "the engine does not serve: …": the gateway's `/ready` fails (the engine starts, is down, or the tunnel is) | `cat $XDG_RUNTIME_DIR/flatshare/voice-worker.json` (calls, whether it admits and why not); for the engine, the rows above |
| `flatshare-voice-worker` exits at start: "FLATSHARE_TURN_DETECTION=vad is the synthetic callers' fixed endpointing …" | The `serve` variant is linked beside `vad-turns` | Unlink `variant-vad-turns.conf` ([the voice plane](#the-voice-plane)) |
| `flatshare-engine` fails at start with "make_room.reserve_tokens leaves no room for a request of max_model_len tokens" | The `make-room` reserve and one `max_model_len` request do not fit in the KV pool together | Lower `make_room.reserve_tokens` below the pool less `max_model_len`: about 3,600 tokens on the frozen box |
| The voice worker logs "lifecycle events do not reach the control plane" | The gateway is down, or `FLATSHARE_LLM_UDS` is not the gateway's socket | `systemctl --user status flatshare-gateway`. The worker logs when events reach the gateway again; those lost meanwhile are not resent, and the gateway's session states time out after `state_timeout_s` |
| The engine's `/server_info` answers 500 | vLLM collects its environment with `uv pip list` and cannot find uv | The unit's `PATH` must include `~/.local/bin` and `VIRTUAL_ENV` must name the engine venv (as `flatshare-engine.service` does) |

## Pausing

Nothing should hold the shared GPU while idle. To pause the GPU node, run `systemctl --user stop flatshare.target flatshare-engine flatshare-stt flatshare-tts`, which returns once the units are down (stopping the target alone returns while they still stop); restart it with `systemctl --user start flatshare.target`. The target stays enabled, so a reboot brings the frozen box back. Gray is shared with other work: start it only for a booked window (`gpu-guard` refuses while another process holds the GPU) and stop it after; the engine takes about 20 s to stop, releasing its pinned CPU tier. Since 2026-10-23 another user's work has held Gray.
