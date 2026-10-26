# The A100 run, rehearsed on the box

`rehearse.sh` runs the A100 run's kit on the box, Phastform, one host acting as both GPU node and control host, as [HARDWARE.md](../../../docs/HARDWARE.md#running-on-another-gpu-step-by-step)'s "Running on another GPU, step by step" lays it out, before the A100's billed hours. It runs from the workstation and covers:

- `deploy/a100/push.sh` from the workstation;
- `bootstrap.sh stage`, `gpu` and `warm` on the host;
- `run.sh` over a short subset that still crosses each kind of step once:
  - block 0;
  - block 1's probes;
  - round 1's first E2 point (block 4), then the deadline's stop under `--until`;
- `deploy/a100/pull.sh`;
- `make reproduce-a100` over the pulled records.

All of it runs under `LH_SUPERVISOR=lh-svc` in a home of its own, `~/a100-rehearsal`, beside the box's deployment, which shares the account. The rehearsal leaves that deployment as it found it, and checks that it did. On 2026-10-25 its steps took 70 minutes on the box, the GPU in use for 57 of them, from `gpu` to `block4`; two runs before stopped at defects of the kit, which the third ran past once each was fixed ([the records](../../results/2026-10-25-a100-rehearsal/README.md)).

Two features of the kit exist for the rehearsal, and `check` refuses a checkout to push (`REHEARSAL_REPO`) without them:

- **`deploy/bin/in-private-net`** runs each GPU server in a network namespace of its own, so block 0's listener audit passes under lh-svc ([below](#block-0s-listener-audit)).
- **`run.sh --estimate BLOCK=MINUTES`.** Round 1's 29-minute block estimate covers three points. With it, no deadline can let block 4 start and still stop it after its first point: a deadline late enough to start the block leaves room for the second point. With `--estimate 4=15`, block 4 starts under a deadline 16 to 17 minutes away:
  - `campaign.sh`'s own budget lets the first point start: 13 minutes for W5's agent point plus 2 for the engine switch.
  - It then stops the block before the second point (voice alone: 7 minutes and a switch), as it would before any point that cannot end before the deadline.
  - `run.sh` exits 0 with the block left undone.

## Before

1. **The GPU is free.** FB's campaign is done, and nothing else of the box runs a point.
2. **The box's units are stopped, and you note which were active**, so that you can start them again after:

   ```sh
   pssh phastform 'systemctl --user list-units --state=active --no-legend --plain "lh-*" "leaseholder*"'   # note these
   pssh phastform 'systemctl --user stop leaseholder.target lh-engine.service lh-stt.service lh-tts.service lh-engine-proxy.socket lh-mps.service;
     systemctl --user list-units --all --no-legend --plain "lh-*" "leaseholder*"'                         # every one inactive
   ```

   `pssh` stands for the SSH command that reaches the host (`REHEARSAL_SSH`).

   `check` refuses while any of these units is active, activating or deactivating, and while any process holds the GPU.

   **Why the socket units matter:** a box unit left running would hold the GPU and the box's sockets. It would also count as "ours" in block 0's listener audit, which matches units by name.
3. **The checkout to push is at the revision to rehearse.** `REHEARSAL_REPO` defaults to the checkout this directory is in. `push.sh` records `git describe --always --dirty` as the host's `DEPLOYED`.

## The commands

From this directory on the workstation, in bash (macOS's 3.2 is enough); the rehearsal's own state, logs and records go to `REHEARSAL_WORK`, `~/leaseholder/rehearsal`. The workstation also needs `git`, `rsync`, `tar` and a `python3` that runs, for the `data` step's `long_traces.py`. On a Mac, `/usr/bin/git` and `/usr/bin/python3` need Xcode's developer tools, their license accepted; another `git` or `python3` first on `PATH` serves as well.

```sh
bash rehearse.sh all                       # every step not done yet, in order; stops at the first that fails
bash rehearse.sh status                    # each step: done or not; each host step: running, done STATUS
bash rehearse.sh STEP                      # one step; the steps before it must be done
bash rehearse.sh --redo STEP STEP          # a step again, although done (on the host too: a block's blocks/N.done goes)
bash rehearse.sh abort                     # stop what runs on the host, then cleanup
bash rehearse.sh --purge cleanup           # cleanup, again if done, and remove ~/a100-rehearsal on the host
```

**Resumable.** A step done leaves `state/STEP.done` and is skipped from then on.

**Steps on the host are detached from the SSH session.** These are `stage`, `gpu`, `warm`, `block0`, `block1` and `block4`:

- Each runs in a session of its own, in `~/a100-rehearsal/.rehearsal/steps/NAME.log` and `NAME.status`.
- A dropped connection does not stop the step. `rehearse.sh` waits for it again for up to 10 minutes. After that, running the same step again finds it still running and waits for it, or finds its exit status.

**Logs.** Each step's output goes to the terminal and to `state/logs/STEP.log`. `state/rehearse.log` has one line for each step's start and end.

**Environment:**

| Variable | Default | What it sets |
|---|---|---|
| `REHEARSAL_HOST` | `phastform` | the host |
| `REHEARSAL_SSH` | `LH_SSH`, else `ssh` | the ssh command, also `rsync -e`'s |
| `REHEARSAL_HOME` | `a100-rehearsal` | the rehearsal's home, one directory under the host's login home |
| `REHEARSAL_REPO` | the checkout this directory is in | the checkout pushed |
| `REHEARSAL_CYAN`, `REHEARSAL_PEACOCK` | `cyan`, `peacock` | where `data` and `reproduce` go |
| `REHEARSAL_CORPUS` | `~/leaseholder/data/weka-x32` | the workstation's corpus |
| `REHEARSAL_PEACOCK_CORPUS` | `leaseholder-data/weka-x32` | peacock's copy of it, under peacock's home |
| `REHEARSAL_GATEWAY_BIN` | unset | an `lh-gateway` for `push.sh`'s `LH_GATEWAY_BIN`; by default `bootstrap.sh` builds one on the host, as the A100's would when a copied binary does not run |
| `REHEARSAL_MIN_FREE_GIB` | `40` | the disk `check` asks for [estimate] |
| `REHEARSAL_WORK` | `~/leaseholder/rehearsal` | where `state/`, `data/`, `hosts/` and `reproduce/` go |

## Isolation: what keeps the box's deployment as it was

`init` writes `~/a100-rehearsal/rehearsal.env`. `bin/rehearsal-ssh` sources it before each command of the rehearsal, and is `push.sh`'s and `pull.sh`'s `LH_SSH`. Through it, the kit's `~/leaseholder` is `~/a100-rehearsal/leaseholder` with no change to the kit. The wrapper refuses where that home has no `rehearsal.env`, so nothing it runs can land in the login home. That matters most for `push.sh`'s `rsync --delete`, which elsewhere would delete what is in the box's `~/leaseholder/src`.

The environment it sets:

- **Its own home and directories.**
  - `HOME=~/a100-rehearsal`;
  - `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME` and `XDG_STATE_HOME` under it;
  - `TMPDIR` under it;
  - `XDG_RUNTIME_DIR=~/a100-rehearsal/run` (0700) for the engine's, the STT's, the TTS's and the gateway's sockets, which `host.py config` writes into `gateway.toml`. The box's sockets stay in `/run/user/UID/lh`.
- **Its own caches.**
  - uv's cache and its Python;
  - Rust's toolchain (`CARGO_HOME`, `RUSTUP_HOME`);
  - vLLM's compile caches: inductor's and Triton's, inside vLLM's compile cache;
  - DeepGEMM's cache;
  - FlashInfer's workspace (`%h/leaseholder/cache/flashinfer`).

  The units derive all of these from their `HOME`, which lh-svc gives as the rehearsal's.
- **lh-svc, and the box's user manager out of reach.**
  - `LH_SUPERVISOR=lh-svc`, which `env.sh` keeps.
  - The session bus is unset (`DBUS_SESSION_BUS_ADDRESS`), and the runtime directory is not `/run/user/UID`. So `systemctl --user` cannot reach the box's user manager. `init` checks this: in the rehearsal's environment it must answer that it cannot connect to a bus.
  - This matters because under `LH_SUPERVISOR=lh-svc`, on a host where a user manager answers, `bootstrap.sh`'s `shell_env` stops that manager's units of the kit's names (`stop_units systemd`). Here those would be the box's own `lh-engine`, `lh-stt` and `lh-tts`.
  - The preflight's "One supervisor" row cannot see the box's units from in here either; `check` reads them instead, in the login environment.
- **The box's hub cache, read offline.**
  - `HF_HUB_CACHE` is the box's, and `HF_HUB_OFFLINE=1`.
  - `bootstrap.sh`'s `hf download` therefore resolves the profile's revisions from the cache and downloads nothing. The vLLM units name the cache too, and run offline as on the box.
  - Kokoro-FastAPI and its model are copied from the box's `~/leaseholder/vendor`, so `bootstrap.sh` neither clones nor downloads them.
  - It still downloads uv, Python 3.12, the venvs' wheels, LiveKit's server, and rustup and crates for `lh-gateway`.
- **No sudo.** A `sudo` shim first on `PATH` refuses, so `bootstrap.sh` installs no system package. `check` finds the tools it would install already present: `cc`, `c++`, `ss`, `git`, `curl`, `rsync`, `tar`, and `python3` 3.10 or newer.
- **The GPU servers' own namespaces.** `in-private-net` starts each of the engine, the STT and the TTS in a network namespace of its own, made in an unprivileged user namespace, as `voicenet-holder` makes the voice plane's. Their loopback listeners (the engine cores' gloo groups) and CUDA's abstract sockets stay in those namespaces, out of reach of the host's other users and of the box's. Their own sockets are files in the rehearsal's runtime directory, reached from the host's namespace.
- **Ports.** The rehearsal listens in the host's network namespace on TCP 127.0.0.1:18080 (the gateway). LiveKit's TCP 17880 and UDP 17882 are the host's too, but only where user namespaces are refused. Ubuntu 22.04 allows them by default, and `check` reports whether Phastform does. `check` reads `ss -Hltn` and `ss -Hlun` read-only and refuses a listener on any of these ports, on any address.

  The kit has no setting to move these ports. A collision would need one, a kit change.

  The engine, the STT and the TTS serve on file sockets in the rehearsal's runtime directory.

**What it still shares, and what is watched.**

- The box's hub cache is read, never written: the units run offline and pin commits.
- `/tmp` gets vLLM's model-loading lock file and torch's default inductor directory, which vLLM replaces with its own before it compiles. Both are shared with the box's engine and are not part of its deployment.

`init` records the box's state in `.rehearsal/box-before.state`. That covers:

- every `lh-*` and `leaseholder*` unit, with its state and its enablement;
- the kind, mode, size and mtime of each entry, and each link's target, under:
  - `~/leaseholder`;
  - `~/.config/systemd/user`;
  - `~/.config/leaseholder`;
  - `~/.cache/vllm`;
  - `~/.cache/flashinfer`;
  - the hub cache.

`cleanup` records the same again and fails on any difference, naming the paths. It also fails on any process whose environment still names the rehearsal's home, a busy GPU, a held port, or a box unit that runs.

Some of the rehearsal's processes set their titles: AIPerf's controller and services, and vLLM's engine cores. That blanks their environment and the home named in it, so `cleanup` ends them by what they belong to:

- **lh-bench's tools.** AIPerf runs in a session of its own, which a step's stop does not reach. lh-bench names each tool's process group in `leaseholder/.lh-bench-e2-tools` until it stops the tool, and `cleanup` ends the groups named there with lh-bench's own `kill_left`, from the bench venv.
- **The groups and sessions of what it finds.** A process whose environment names the home takes with it the processes of its process group and session whose environment is blank, as a unit's server takes its engine core. A process whose environment can be read and lacks the home is never ended.

## LH_DATA: where the data comes from

The `data` step assembles `data/` (`LH_DATA`) on the workstation:

- **`weka-x32`** is linked from the workstation's `REHEARSAL_CORPUS` (393 traces, 1.7 GB). When that copy is absent, it is copied from cyan's `~/leaseholder/data/weka-x32`. `reproduce` uses peacock's copy, `REHEARSAL_PEACOCK_CORPUS`, as `CORPUS`, and copies the workstation's there when peacock has none.
- **`weka-x32-long`** comes from `benchmarks/e2/long_traces.py` over it: the 48 traces with the largest median input, as links, made by the same command that made cyan's. `push.sh` copies them as files: 1.0 GB on the host.
- **`callers`** is copied once from cyan's `~/leaseholder/data/callers`: `lh-voice render-callers`' clips and `manifest.json`. The repository holds only the voice tests' two clips, so cyan's copy is the only source.

## The steps

The minutes planned are measured where a measurement existed, and say so; the rest were estimates. The rehearsal of
2026-10-25 measured each step ([its records](../../results/2026-10-25-a100-rehearsal/README.md)): from the lab's network
and with the box's models in its hub cache, `push` and `stage` took a fraction of their plan.

| Step | Where | What it runs | Minutes planned | Measured |
|---|---|---|---:|---:|
| `check` | host, read-only | box units stopped, GPU idle, ports free, disk, the box's models at the profile's revisions in its hub cache, Kokoro, tools, user namespaces, logind's `KillUserProcesses` | 1 | 0.0 |
| `init` | host | `~/a100-rehearsal`, `rehearsal.env`, the sudo shim, Kokoro copied, the box's state recorded | 2 | 0.2 |
| `data` | workstation | `LH_DATA` (above) | 1, plus cyan's callers | 0.2 |
| `push` | workstation | `push.sh`: the source (15 MB) and the data (2.7 GB as files) into the rehearsal's home | 7 to 18 at 20 to 50 Mbit/s up | 1.9 |
| `stage` | host | `bootstrap.sh stage` | 45 to 60 | 9.1 |
| `gpu` | host | `bootstrap.sh gpu` | 5 | 1.4 |
| `warm` | host | `bootstrap.sh warm`: one engine start per compile configuration (`host.py`'s `WARM`) | 15 to 20 | 21.7 |
| `block0` | host | `run.sh 0`: `bootstrap.sh gpu`, the warm check, the frozen engine, `smoke.sh` and its listener audit, a 2-turn call | 5 to 10 | 4.1 |
| `block1` | host | `run.sh 1`: E0 with `--reps 3`; the KV case x3, the prefill case x3, the tail x2 | 25 | 16.1 |
| `block4` | host | `run.sh --until <17 min on> --estimate 4=15 4` | 16 to 17 | 14.2 |
| `pull` | workstation | `pull.sh --once` into `hosts/phastform` | 2 | 0.1 |
| `reproduce` | peacock | `make reproduce-a100 RECORDS=<the pulled records> CORPUS=<peacock's weka-x32>`; the report into `reproduce/` | 5 | 0.4 |
| `cleanup` | host | the rehearsal's units stopped, its processes ended, the box compared | 2 | 0.3 |

`warm` outran its plan: from the rehearsal's own empty caches the frozen configuration took 797.7 s to be ready (W8's
empty-cache start of the engine on the box: 531.8 s), and the next three 152 to 179 s each.

Where the planned minutes came from:

- **`stage`**:
  - the venvs' downloads (about 15 GB of wheels);
  - a Rust toolchain and `cargo build --release -p lh-gateway`;
  - FlashInfer's sampler: 138.7 s on peacock's 16 CPUs (`host.py jit-check`, 2026-10-25) [measured].
- **`warm`**: the box's first engine start with empty caches took 531.8 s, and a warm start 88.8 s (W8's start drills) [measured].
- **`block1`**:
  - E0 about 4 minutes;
  - each probe about 1 minute;
  - each tail run 4 minutes (240 s by its argument).
- **`block4`**:
  - the first point: W5's agent points took at most 11 min 34 s on the box, and its engine switches at most 1 min 41 s [measured, `campaign.sh`'s budgets];
  - the frontier report and `lh-bench e2 reset` after it.
- **`reproduce`**: the first `cargo build --release -p lh-sim` on peacock.

### What each step prints, and its check

**check.** One `check: NAME: ok|refused|note: ...` line per check, then `rehearse: check: done: <revision> on <host>`. Any `refused` stops it. The lines to expect on the box:

- `the box's units: ok: every lh-* and leaseholder* unit is inactive`
- `GPU: ok: idle, NVIDIA GeForce RTX 5060 Ti, <n> MiB, 16311 MiB`
- `ports: ok: TCP 18080, 17880 and UDP 17882 free`
- `hub cache: ok: /home/user/.cache/huggingface/hub`
- `model Qwen/Qwen3-4B-Instruct-2507-FP8: ok: 8591804019...`
- `model openai/whisper-large-v3-turbo: ok: 41f01f3fe8...`
- `user namespaces: ok: the voice plane and each GPU server in a network namespace of its own`
- `logind: ok: KillUserProcesses=no (the default)`
- `CPUs: note: 16 logical, 2 NUMA nodes`

**init.** It prints:

- `init: Kokoro-FastAPI copied from the box's ~/leaseholder/vendor (<size>)`;
- `init: the box's state recorded: <n> lines`;
- the rehearsal environment's values;
- `init: the box's user manager is out of the rehearsal's reach (systemctl --user there: Failed to connect to bus: ...)`.

It fails unless `systemctl --user`, from inside the rehearsal's environment, answers that it cannot connect (`Failed to connect to bus`, or from systemd 257 on `Failed to connect to user scope bus`) or `offline`. An answer that names a manager's state fails it, and so does none, as when the connection drops at the probe.

**data.** `long_traces.py`'s line (`393 traces, median input ...; the 48 largest: ...`), then `data: done: ... weka-x32 393 traces, weka-x32-long 48, callers <n> clips`. It checks for exactly 48 long traces and a `manifest.json`.

**push.** It prints `push: <host> has the source at <revision> and the data`, then the host's file counts, with no link left: `weka-x32 393 files 0 links; weka-x32-long 48 files 0 links; callers <n> files 0 links`. It checks that the rehearsal's `DEPLOYED` is the checkout's `git describe`, and that each of the host's three directories holds as many files as `LH_DATA`'s, and no link.

**stage.** Its output is `bootstrap.log`'s:

- `bootstrap stage: from ~/a100-rehearsal/leaseholder/src (<revision>)`;
- `uv 0.11.27 into ~/.local/bin`;
- `install path: cu130: driver ... runs CUDA 13.0: the box's lock`;
- `engine venv: cu130, ...`;
- `building lh-gateway`;
- `host: linked ...` and `host: wrote ...`;
- `host: built FlashInfer's sampler for 12.0 in <s> s`.

The node is rendered from the A100's profile (`a100-40gb`) with Phastform's facts: the card's memory and sm120, the CUDA 13.0 toolkit, the core plan from 16 CPUs with the engine on the GPU's NUMA node.

Then the preflight's table and `host: .../preflight.json: <n> ok, <n> degraded, 0 refused, <n> pending`. The rows expected degraded on the box:

- **systemd `--user` and lingering:** none, because the rehearsal hides the user manager, so lh-svc runs the units;
- **Cores:** 16 CPUs, so R7 halves the plan's wants;
- **Compile and JIT caches:** cold, until `warm`.

The rows that need an engine start stay pending.

The checks:

- `env.sh` keeps `LH_SUPERVISOR=lh-svc` and the box's hub cache;
- no preflight row is refused;
- the control plane (voicenet, LiveKit, the gateway, the worker) is active under lh-svc.

**gpu.** `bootstrap gpu: ...`, `vLLM parses the node's configs`, and the preflight with the card's checks. Its check: no refused row.

**warm.** `host.py warm`'s starts, then `verify: warm: frozen ok, ...`. Its check: every configuration in `cache/warm.json` is ok, and no preflight row is refused.

**block0.** `run.sh`'s `block 0: start`, then:

- `bootstrap.sh gpu` again;
- `block 0: every compile configuration warm already (cache/warm.json)`;
- `variant.sh use frozen`;
- `smoke.sh`'s stages: health, TTS, STT, LLM, TTS reply, GPU memory.

Then the listener audit's line, `listener audit  ok: the box's units hold <n> Unix sockets, all in ~/a100-rehearsal/run, and 1 TCP or UDP listeners, all allowed; lh-engine, lh-stt, lh-tts in network namespaces of their own, with <n> TCP or UDP listeners and <n> abstract sockets there; ...`, `smoke: ok`, and the call's summary.

The checks:

- `blocks/0.done` exists;
- `smoke.txt` ends in `smoke: ok`;
- the call's `call.txt` exists;
- the audit's line is `ok`, with the three GPU servers in network namespaces of their own. That is the line expected where the kernel allows unprivileged user namespaces, which `check` reports. Where it refuses them, the line expected is the degraded one: the GPU servers in the host's namespace (`LH_PRIVATE_NET=host`), with their loopback listeners and abstract sockets counted apart.

**block1.** `probes.sh`'s lines (`probes 1: engine: frozen`, `frozen-e0: lh-bench tiers --reps 3 (...)`, each probe's result line), `block 1 complete`, `block 1: done`. The checks: `blocks/1.done`, E0's `summary.json`, and the eight runs' `.txt` files.

**block4.** It prints:

- `rehearsal: block 4 until HH:MM UTC`;
- `run: blocks 4, until HH:MM UTC, estimates 4=15`;
- `block 4: start`;
- `campaign a100-r1: 3 points, ... until HH:MM UTC (epoch ...)`;
- the first point's lines: the engine switch, the measurement, its summary;
- `w5-a1-voice: <n> minutes would end after HH:MM UTC; stopping before it (--until)`;
- the frontier report;
- `a100-r1 stopped before its deadline (HH:MM UTC) with points left`.

The checks:

- `w5-disc-pr-ab48-s24/run.json` is complete;
- `w5-a1-voice` did not complete;
- no `blocks/4.done` exists;
- `run.sh` exited 0.

`block4` run again after a failure that came once its first point was complete resumes the campaign. Its first point is then skipped as done, and the second fits the deadline and runs, which the check reports as a second point.

**pull.** `pull: phastform up to date in .../hosts/phastform`. The check: the pulled copy has `blocks.log`, E0's summary, `preflight.json`, `warm.json`, the point's `run.json`, and the units' journal (`runs/journal/units.log`).

**reproduce.** `cargo build`, the analysis, then `reproduce: .../a100.md, its sections: ## 1. ... ## 5.`. The check: section 3 (the probes) and section 4 (E2 within the A100) are measured. Section 1, the step model, says `Not measured: no calibration point`, since block 5 is not rehearsed.

**cleanup.** It prints:

- `kill-tools: lh-bench's kill_left found nothing left`;
- `stop-units: inactive ...` for nine units;
- `leftovers: none`;
- `verify: the box's deployment as the rehearsal found it: <n> lines (units, files) alike`;
- `verify: the GPU is idle`;
- `check`'s lines again: the box's units still inactive, the GPU idle, the ports free.

## Block 0's listener audit

**Why the GPU servers need namespaces of their own.** The listener audit's first run on the box's sockets deployment found two kinds of socket of the GPU servers that every user of the host reaches:

- **gloo:** each vLLM engine core, the engine's and the STT's, keeps six TCP listeners for torch's gloo groups on 127.0.1.1, the host name's loopback alias;
- **CUDA UVM:** each GPU process keeps CUDA's abstract UVM socket, `@cuda-uvmfd-*`.

The audit counts them as the units' own: by their cgroups under systemd, and under lh-svc by the `LH_SVC_UNIT` that lh-svc puts in the environment of a unit's main process, which an engine core reads from its server's (it writes its title over the start of its own environment).

**What the kit does about them.**

- **The wrapper.** `deploy/bin/in-private-net` prefixes the GPU units' `ExecStart`. It runs the server in a network namespace of its own: `unshare --user --map-root-user --net` with the loopback up, the server keeping the process. Because it is part of the command, lh-svc runs it exactly as systemd does. systemd's own `PrivateNetwork=` cannot do this in systemd 249's user manager.
- **The audit.** It reads those namespaces' tables (`/proc/PID/net`, sockets matched by inode). It accepts the listeners there and judges their file sockets as before.
- **No user namespaces.** Where the kernel refuses unprivileged user namespaces, the server runs in the host's namespace, marked `LH_PRIVATE_NET=host`. The audit then accepts its loopback TCP listeners and abstract sockets and counts them apart, and the preflight's "User namespaces" row is degraded.

**What the rehearsal expects.**

- **On Phastform** (Ubuntu 22.04, user namespaces allowed): `block0`'s check wants the audit's `ok` line naming `lh-engine, lh-stt, lh-tts in network namespaces of their own`.
- **A refused audit, or the degraded line where namespaces are allowed:** `block0` fails, and the rehearsal stops there. That is a finding.
- **On the A100's Ubuntu 24.04, if it restricts them:** the degraded line is the one expected, and the preflight's row with it.

**What the box shows otherwise.** The box's own deployment runs the same wrapper under systemd, and its `smoke.sh` the same audit. The rehearsal's block 0 checks them under lh-svc, from the rehearsal's home.

## Abort and restore

1. **Abort.** `bash rehearse.sh abort` stops the step running on the host: TERM to its process group, then KILL to what is left of the group after 180 s. That is lh-bench's own stop: on TERM it gives its callers 30 s and AIPerf 120 s to stop, and AIPerf, in a session of its own, would outlive a KILL of the group. It then runs `cleanup`:
   - ends what lh-bench's tools left where a step was killed before lh-bench stopped them (above);
   - stops the rehearsal's units through its `env.sh` (lh-svc), and fails on one that is not inactive or failed after its stop;
   - ends any process whose environment still names `HOME=~/a100-rehearsal`, with the titled processes of its group and session;
   - compares the box's state with `init`'s;
   - checks the GPU and the ports.
2. **When `cleanup` fails.** It names what remains (a process, a held port, a busy GPU), or each path of the box's that changed (`diff` of the two states, `<` as found and `>` now). Nothing of the rehearsal writes outside `~/a100-rehearsal`, so a changed path is either a finding or a write by something else; the diff shows which file. `.rehearsal/box-before.state` and `box-after.state` stay on the host for reading.
3. **Restore.** The box's deployment is as it was once `cleanup` passes. Start the units you noted before, for example `systemctl --user start leaseholder.target` with any socket unit that was active, then check `systemctl --user list-units --state=active "lh-*" "leaseholder*"` against your note.
4. **Purge.** `bash rehearse.sh --purge cleanup` also removes `~/a100-rehearsal`, about 30 GB [estimate], once everything else passes. It runs `cleanup` again although it is done, so it also removes the home that an earlier `cleanup` kept. Keep it to read the rehearsal's logs on the host: `~/a100-rehearsal/leaseholder/node/bootstrap.log`, `runs/a100/run.log`, and `.rehearsal/steps/*.log`.

## What only the real run can show

The CPU test runs this logic against stand-ins. Only the run on the box can show:

- **`stage` on a real host.**
  - The downloads, which are uv's, Python's, the wheels', LiveKit's and the crates'.
  - Building the venvs from uv's own Python, including the engine's from the cu130 lock.
  - `lh-gateway` built on the host.
  - FlashInfer's sampler compiled for sm120 with the host's CUDA 13.0.
  - The models resolved offline from the box's hub cache.
  - The control plane started under lh-svc. This includes the voice plane in its own network namespace through `voicenet-holder`, and the gateway on 127.0.0.1:18080 and its socket in the rehearsal's runtime directory.
- **`gpu` and `warm`.**
  - The card's facts.
  - vLLM's parse.
  - Each compile configuration's start under lh-svc, with `wait-healthy` and MAINPID.
  - The preflight's GPU rows: compute mode, MIG, `gpu-guard`.
  - `NUMAPolicy` through `numactl`, where Phastform has it.
- **Block 0.**
  - `smoke.sh`'s turn on the rehearsal's sockets.
  - `in-private-net` under lh-svc: the three GPU servers each in a namespace of its own, `wait-healthy` reaching their file sockets from the host's namespace.
  - the audit over the real listeners there, through `/proc/PID/net`.
- **Blocks 1 and 4.**
  - The probes, E0, AIPerf, and the callers through LiveKit in the namespace.
  - The GPU sampler.
  - The measurement drop-in.
  - `campaign.sh --local`'s deadline stop with real durations.
- **`pull`.** The journal exported through lh-svc's `journalctl` shim.
- **`reproduce`.** The analysis over real records.
- **`cleanup`.** That the box's tree, units and hub cache are untouched by every one of those processes. This includes vLLM's and huggingface_hub's reads of the box's hub cache.
- **How long each step takes.** The rehearsal measures this for the A100 run's plan.

## What the rehearsal does not cover

These are the A100's paths that the box cannot take:

- **sm80's path:** Marlin and the BF16 rate.
- **Installs that depend on the driver:** the cu129 lock, `cuda-compat-13-0`, and CUDA 12.9.1's toolkit download.
- **Hardware the box does not have:** several GPUs and `LH_GPU_ID`; MIG; a compute mode other than Default.
- **Hosts without a usable user manager:** a container without systemd, and a host where lingering is refused.
- **systemd as the single host's supervisor**, including `bootstrap.sh`'s `stop_units systemd` when it moves units. On Phastform that would act on the box's units, so the rehearsal hides the user manager instead.
- **sudo and apt installs.**
- **A host without user namespaces**, where the voice plane and the GPU servers run in the host's network namespace (`LH_VOICENET=host`, `LH_PRIVATE_NET=host`), which the CPU test's degraded-line case stands in for.
- **The rest of the run:** blocks 2, 3 and 5 to 10, the optional arms, and a reboot mid-run.

## The CPU test

```sh
cd python/leaseholder_bench && uv run --extra test pytest -q tests/test_rehearse.py
```

It runs `rehearse.sh`'s every step in order against stand-ins, with the repository's own `push.sh` and `pull.sh`. The hosts are local directories: the repository's `fakes/campaign/ssh` runs commands in them, and the real rsync copies between them. The box's user manager, `nvidia-smi`, `ss` and `make` are stand-ins, and so are the kit's `bootstrap.sh`, `run.sh` and lh-svc shims.

It holds:

- **The checks:** `check` refuses a box unit that runs, a busy GPU, and a held TCP or UDP port.
- **Isolation and the environment:**
  - `push` leaves the box's tree as it was;
  - the box's user manager is asked only `list-units` and `list-unit-files`;
  - the kit runs in the rehearsal's environment: home, runtime directory, lh-svc, the box's hub cache offline, no session bus, sudo refused.
- **Failures:**
  - a command that fails ends its step, with its status;
  - `init` fails unless its isolation probe answers that `systemctl --user` cannot connect;
  - `push` fails on a file in the host's data that `LH_DATA` lacks;
  - `cleanup` fails where the host does not answer whether it has the rehearsal's home.
- **Resuming:**
  - a step done is not run again;
  - a step on the host outlives the call that started it, is not started twice, gives its exit status even when its command calls `exit`, and is ended by `stop`, which waits for its whole process group, not its leader alone;
  - a step's state is done once its status is written, and running only while its pid names a process with the step's mark (Linux); `wait` follows a log that grows as it is read; `start` fails, its command not run, when the step ends before its pid file is written.
- **Block 0:**
  - a refused audit stops the rehearsal;
  - the audit's degraded line fails block 0's check where the host allows user namespaces, and is the line expected where it refuses them;
  - a step that ran but failed its check is run again only under `--redo`, with the block's `blocks/0.done` removed on the host so that `run.sh` runs it again.
- **Block 4:** it is called with a deadline 16 to 17 minutes away and `--estimate 4=15`.
- **`pull` and `reproduce`:**
  - `pull` copies the rehearsal's records, not the box's;
  - `reproduce` runs `make reproduce-a100` on peacock over them;
- **`cleanup`:**
  - it stops lh-svc's units and passes on an untouched box, and fails on a unit left deactivating;
  - it ends a tool whose group lh-bench named, and a process in the group of one it finds, both titled (Linux);
  - it fails, naming the path, on a box file changed after `init`;
  - `--purge cleanup` after a `cleanup` removes the rehearsal's home.

On Linux the test also runs `cleanup`'s leftover-process check against `/proc`; macOS has no `/proc`, and there the check reports "not checked".
