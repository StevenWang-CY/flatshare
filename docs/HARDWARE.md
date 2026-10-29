# Running Flatshare on other hardware

Every measurement before 2026-10-26 ran on one RTX 5060 Ti 16 GB, the frozen box, with Gray, a second one, for the first make-room probes; that day an A100 SXM4 80 GB on Google Colab ran E2's core like for like ([what it measured](#what-it-measured-2026-10-26)). This page is for an operator who moves the deployment to another GPU or to a single host, and for a reader who checks the A100 numbers. It says what the deployment needs from a GPU node, how a node's files come from a hardware profile, what changes on a card or a host that is not the box's, how one host runs both the GPU servers and the control plane, and how the A100 run is laid out and what it measured.

A number here is measured (on the box, Gray or the A100, with its source), modeled (flatshare-sim's forecasts, [`forecast.md`](../benchmarks/a100/forecast.md) for the A100 40 GB and [`forecast-80gb.md`](../benchmarks/a100/forecast-80gb.md) for the A100 SXM4 80 GB) or an estimate, and says which.

## What runs where

What vLLM 0.32.0 runs for the box's checkpoint, Qwen3-4B-Instruct-2507-FP8 (block-scaled FP8 weights), by the card's compute capability, from vLLM's kernel selection order and its attention backends' priorities, and how far each is checked:

| Cards (compute capability) | Linear layers | Attention, KV layout | Status |
|---|---|---|---|
| RTX 5060 Ti (sm120): the box, Gray | FP8 W8A8 (DeepGEMM) | FlashAttention 2, LBHNC | **measured**: E0, E1's calibration, E2, the make-room probes |
| A100 40 GB or 80 GB, SXM4 or PCIe (sm80) | Marlin W8A16: the FP8 weights, BF16 activations, the BF16 tensor rate | FlashAttention 2, LBHNC | **measured** on an A100 SXM4 80 GB: E0, the step model, the make-room probes, E2 ([the A100 run](#what-it-measured-2026-10-26)); the 40 GB **modeled** ([the forecast](../benchmarks/a100/forecast.md)); the Marlin path was first run on the box ([its dry run](#the-marlin-dry-run-on-the-box)) |
| A10 (sm86), L4, L40S, RTX 4090 (sm89) | Marlin W8A16, as on the A100 (vLLM 0.32.0 sends this checkpoint to Marlin on sm89 too) | FlashAttention 2 | **modeled** (`flatshare-sim hardware`) |
| H100, H200 (sm90), B200 (sm100) | FP8 W8A8 | FlashAttention 3 (sm90), FlashInfer (sm100) | **modeled** |
| T4 (sm75) | Marlin with FP16 activations | Triton attention | **modeled**; the preflight labels both |
| Older than sm75, CPU-only, ROCm | — | — | not supported by this deployment: vLLM 0.32.0 needs sm75, and the CPU KV tier's platform differs off CUDA |

The KV layout follows the CPU tier's connector, not the card: vLLM's `OffloadingConnector` requires LBHNC, so every CUDA card with the tier runs it. What changes in what the results mean is in [the preflight](#the-preflight): within one card, the frozen engine against make-room compares as on the box; across cards, only in the regime's coordinates and each card's distance to its own SLO.

## The hardware profile

A GPU node's deployment is a function of a few inputs: the card (its memory, compute capability, driver), the host (CPUs by NUMA node, RAM, `/dev/shm`, systemd), the model and the rules below. [`deploy/profiles/`](../deploy/profiles) holds one profile per node type, and [`deploy/profile.py`](../deploy/profile.py) (Python's standard library only, so that a bare host runs it before any venv exists, from 3.11 for `tomllib`: on Ubuntu 22.04, whose `python3` is 3.10, uv's Python 3.12 runs it, as `bootstrap.sh` does) writes a node's directory from one:

```sh
uv run --no-project --python 3.12 deploy/profile.py detect --out host.json    # the host's facts: nvidia-smi, /sys, /proc, systemd, toolkits
uv run --no-project --python 3.12 deploy/profile.py render a100-40gb --host host.json --out ~/flatshare/node
uv run --no-project --python 3.12 deploy/profile.py check rtx-5060-ti         # the box's profile renders deploy/phastform byte for byte
~/flatshare/venvs/engine/bin/python deploy/profile.py check-vllm ~/flatshare/node   # on a GPU host: vLLM's own parser, no engine
```

| Command | What it does |
|---|---|
| `detect` | Writes the host's facts as JSON (below); with `--peer HOST`, also the TCP connect times to that host's SSH port |
| `render` | Writes a node's directory from a profile, and from the host's facts (`--host`) where the profile has "auto" values |
| `check` | Renders the profile into a scratch directory and prints each file's difference from its repository copy (the box's: `deploy/phastform`) |
| `update` | Writes the rendered files into the repository copy, after a template changes |
| `effective` | Prints a serving profile's `serving.json` ([the serving profile](../deploy/phastform/variants/README.md#the-serving-profile)) |
| `check-vllm` | Parses a node's `engine.yaml` and `stt.yaml` with vLLM's own parser and builds their engine configs, as `vllm serve` would, without starting an engine |

The host's facts, from `detect`:

- the GPU as `nvidia-smi` shows it: name, memory, compute capability, driver, PCIe link, MIG mode, power limit, compute mode and NUMA node;
- the CPUs and the RAM by NUMA node, `/dev/shm` and shmem huge pages;
- the kernel, a container or VM, root and sudo, systemd's user manager and lingering, and user namespaces;
- the CUDA toolkits, `cuda-compat-13-0`'s libraries, the C and C++ compilers, ninja and uv;
- the directory of the driver's `libcuda.so.1`, and whether the linker searches it ([single-host mode](#single-host-mode));
- the Hugging Face hub cache, where `bootstrap.sh` downloads the models and the units read them.

A node's directory holds `engine.yaml`, `stt.yaml`, the GPU units (`flatshare.target`, `flatshare-engine`, `flatshare-stt`, `flatshare-tts`, `flatshare-mps`), each server on its own Unix socket as the box's, the variants' drop-ins, the measurement drop-in (`measure/flatshare-engine.conf`, vLLM's development routes, which `variant.sh measure on` links), `variant.sh`, `install.sh`, and `node.json`: every value as resolved, the install path and the core plan, with the warnings.

[`deploy/templates/`](../deploy/templates) holds the box's files with their values as placeholders, which `render` fills from the profile; the comment lines that state a node's own facts are profile fields. `variant.sh`, `install.sh`, the measurement drop-in and the `mps` variant are copied from the box's directory as they are, since they name nothing of a node's own.

- **The box's profile** (`rtx-5060-ti`) holds the box's values only, and a test holds its rendering to `deploy/phastform`, so no measurement changes meaning. It has no "auto" values, so it renders without the host's facts. `make profiles` checks it and its serving profile; `make profiles UPDATE=1` writes both after a template, a profile or a file `serving.json` hashes changes.
- **When the box's own files change** (its units, say), `check` prints the lines to carry into `deploy/templates`. A file added to the box's directory has to be rendered, copied or named box-only before the tests pass.
- **The A100's profile** (`a100-40gb`) keeps the box's model, pool, step cap and CPU tier ratio and derives the rest from the host ("auto").
- **A profile with a base** takes the base's values under its own, key by key in each table: `a100-40gb-qwen3-14b` names the A100's profile in its `[profile]` table and changes the model, the pool and the directory.

## The rules

The profile's "auto" values, in `deploy/profile.py`:

- **R1, the pool.** Like-for-like, the box's 36,400 tokens (5 GiB of Qwen3-4B's KV), so that a card runs the box's comparison; or the card's own pool, what the box keeps beside its pool left to the model's KV in whole blocks (flatshare-sim's sizing, modeled, checked at the first start).
- **R2, the memory budgets.** vLLM refuses a start whose free memory is below `gpu-memory-utilization` × the card's memory even when `kv-cache-memory` fixes the pool, so the box's budgets port in GiB: 10.83 GiB for the engine and 2.32 GiB for Whisper, 0.28 and 0.06 of a 40 GiB card. Both vLLM servers fix their pools, Whisper's on every card at what the box's 0.15 gave it (1,400 blocks, 437.5 MiB, which vLLM counts as 4,020 tokens), since its need does not grow with the card. Neither then sizes its pool from the memory free at its start, and `flatshare.target` starts the three servers at once: each one's check passes in any order wherever starting the audio servers first passed, since the engine's worst case, the audio servers resident, is that order's, and the STT's, the engine resident, holds while the two budgets and the TTS's share fit the card together, the engine within its budget. A server starting still peaks above its steady state (vLLM's profiling pass, then its graph capture), which a check that meets it takes from the margins the budgets leave, and `Restart=on-failure` starts a server that lost again.

  An "auto" `gpu-memory-utilization` is the server's budget over the card's memory as `nvidia-smi` shows it, rounded up to 0.01. `profile.py` refuses a profile where the two budgets and the TTS's share take more than 0.95 of the card's memory as `nvidia-smi` shows it, the ceiling of one budget alone, since vLLM's total is the card less the driver's reserve; the TTS's share is Kokoro's footprint on the box, 1.08 GiB (`tts_gib`).

  On the box the three take 0.92 of the card, and the worst cases leave 12,250 MiB free against the engine's 11,096 and 4,360 against the STT's 2,378, the engine holding 10,388 MiB (vLLM's total of 15,852 MiB less the servers' footprints of 2026-10-25) [measured]; on a 40 GiB card they take 0.37. The 14B model of the optional arm C' ([below](#the-optional-arms)) outgrows its budget, its pool sized to fill the card up to the box's headroom [modeled], so the sum does not cover it, and its node starts the engine after the audio servers instead (the profile's `engine_order`): each audio server checks free memory before the 14B's allocation, and the engine's own check sees them resident.
- **R3, the CPU tier.** The box's ratio, four times the pool, bounded by what `/dev/shm` holds less 2 GiB and by the RAM a single host leaves after 25 GiB for its other processes [estimate], in whole GiB; refused below the pool's own size, since a tier no larger than the pool only mirrors it.
- **R4, the step cap.** The core keeps the box's 512/256. The time-equivalent cap is the largest multiple of 256 whose chunk takes no longer than the box's 512-token chunk at the scenario's median context, its long-prefill threshold half of it: 1,536 on the A100, central (2,048 low, 1,280 high) [modeled].
- **R5, the probes.** `probe.py` and `tail.py` size their requests from the engine's pool and context: on the box two KV-case requests of 17,999 tokens, three prefill-case requests of 10,000 and 12 tail agents.
- **R6, the scenario.** E2's corpus, its two callers and the sessions for the box's KV pressure; another load or model is an optional arm.
- **R7, the core plan.** On one host, the engine, the audio servers, the gateway, the voice plane, AIPerf and the callers share the CPUs 8:4:2:8:8:4 (34 give each its want), the engine on the GPU's NUMA node and the audio servers off it where there is another; below 17, half the wants, the plan still places every component and warns that the host checks the mechanics, not the numbers.
- **R8, the install path.** The box's lock (vLLM 0.32.0, torch 2.13.0+cu130) where the driver runs CUDA 13.0, natively from R580, or through `cuda-compat-13-0`, where it is installed, on a data-center card with R535 or R570 (the units then put its libraries on `LD_LIBRARY_PATH`); the cu129 lock ([`deploy/a100/requirements`](../deploy/a100/requirements): vLLM 0.32.0+cu129 with torch 2.13.0+cu129, every other package at the box's version) on any other driver from R525; refused below. FlashInfer's JIT compiles with a toolkit of the runtime's CUDA version, 13.0 or 12.9: the host's, else the venv's CUDA 13.0 (with `cache/cuda_lib`, which links it: the venv's `nvidia/cu13` has no `lib64` stubs), else CUDA 12.9.1's toolkit, which bootstrap installs into `~/flatshare/cuda-12.9` from NVIDIA's runfile without root (CUDA 12's wheels carry no `nvcc`).

## The preflight

`deploy/a100/host.py preflight NODE [--gpu]` checks what a campaign depends on and writes `NODE/preflight.json`; `bootstrap.sh` ends each stage with it and fails on a refused check, and `flatshare-bench e2 run` keeps every check that was not ok in each point's `run.json`. A check that needs the card or the engine's first start is pending until then.

| Check | Degraded, and what it changes | Refused |
|---|---|---|
| Compute capability and precision path | sm75-sm89: W8A16 through Marlin at the BF16 rate; the per-token cost is the BF16 path's, comparisons across cards only in regime coordinates. The row names the kernel the engine's first start selected (before it, the one expected): another than Marlin below sm90 is named as such | below sm75 |
| BF16 | sm75: FP16 activations; numerics differ from the box | |
| Attention backend and KV layout | not FlashAttention or FlashInfer on LBHNC (the engine log): attention's cost | |
| Driver and install path | cu130 with the compatibility libraries, or the cu129 lock: another CUDA runtime under the same vLLM and torch | the venv's build is not the one the driver needs; or the driver's libraries are in a directory the linker does not search and a GPU unit does not name it on `LD_LIBRARY_PATH` |
| torch on the GPU (`--gpu`) | | `torch.cuda.init()` fails with the engine's environment (errors 803, 804; or no driver, where the engine's environment does not reach the driver's `libcuda.so.1`) |
| FlashInfer's JIT | | its sampler does not build (the engine would fail at its first start; `VLLM_USE_FLASHINFER_SAMPLER=0` in a drop-in is the degraded way out), or was built for another compiler or card than the units name now (staged before the card was visible, the `gpu` stage can change both; it builds the sampler again: `host.py jit-check --if-stale`) |
| Host compilers | | no `cc` or `c++`: Triton builds its launchers with a C compiler, `nvcc` needs a C++ one |
| Python headers | | the engine venv's Python has no `Python.h`, which Triton compiles its launcher against at the engine's first start (a system Python 3.12 without its `-dev` package, Ubuntu 24.04's): `stage` makes the venvs from uv's own Python |
| Compile and JIT caches | cold (each compile configuration's first start compiles: 531.8 s on the box against 88.8 s warm, measured in W8's start drills), on tmpfs, or under 10 GiB free [estimate] | under 2 GiB free |
| CPU tier and `/dev/shm` | below four times the pool: fewer restores, more recompute; the ratio is in every record | another tier than `stage` rendered (`staged.json`): at the `gpu` stage something holds `/dev/shm` or the RAM that R3 bounds it by, and the core would run that tier for the whole campaign; `gpu` stops the GPU units before it reads the facts, so an engine of its own does not (R3 refuses at render a tier below the pool) |
| Pinned memory | `cudaHostRegister` failed: a pageable tier, slower restores | |
| Host RAM | the GPU's NUMA node short of the tier: a remote tier (the box measured one within 1% of pinned bandwidth) | |
| Async scheduling, block size | synchronous scheduling; another block size: make-room's token counts round to it | |
| Engine starts | | a start failed (warm.json keeps its log lines) |
| GPU | | several GPUs and none named, or `FLATSHARE_GPU_ID` naming none of them: the GPU units, `gpu-guard` and the GPU sampler use the lent GPU alone, so it must be named (`FLATSHARE_GPU_ID=INDEX\|UUID bootstrap.sh stage`; [single-host mode](#single-host-mode)) |
| GPU exclusive (`--gpu`) | | another process holds the lent GPU (`deploy/bin/gpu-guard`) |
| Compute mode | | anything but Default: under `Exclusive_Process`, common on shared HPC nodes, one CUDA context exists at a time, so the STT, the TTS and the engine cannot all run (`nvidia-smi -c DEFAULT` with root, or the `mps` variant, which is not the core's configuration) |
| MIG | one slice with `FLATSHARE_ALLOW_MIG=1` (at a stage; `env.sh` keeps it): a smaller card, on which the forecast does not hold, CUDA on the slice by its UUID | MIG on: one slice (the profile's budgets and the forecast are the whole card's), none or several |
| systemd `--user` and lingering | none, or `FLATSHARE_SUPERVISOR=flatshare-svc`: `deploy/bin/flatshare-svc` runs the units; nothing changes in the measurements | `env.sh` keeps systemd from an earlier stage, but lingering is off now: the units would stop with the last session |
| Socket paths | | the runtime directory (`%t`) too long for vLLM's ZMQ sockets, `%t/flatshare/engine.zmq/` and a uuid4, past the 107 bytes a socket's path may have: the engine would fail at its first start. Under flatshare-svc where the session has no `XDG_RUNTIME_DIR`, `%t` is `~/.local/state/flatshare-svc/run`, which leaves a home directory 18 characters |
| One supervisor | | units under the other supervisor too, an earlier stage's (a split control plane: the gateway's port held by one, its journal read from the other) |
| User namespaces | the kernel refuses them: the voice plane and the GPU servers run in the host's network namespace, under either supervisor, and `smoke.sh`'s listener audit accepts there what they listen on and nothing else ([single-host mode](#single-host-mode)); the host's public addresses are named, since they must drop inbound UDP. Allowing unprivileged user namespaces gives each its own (Ubuntu 24.04: `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`) | |
| Listener audit | | no `ss` (iproute2), which `smoke.sh`'s audit ([`deploy/bin/audit-listeners`](../deploy/bin/audit-listeners)) reads; `bootstrap.sh` installs it where apt works as root or with sudo |
| Cores | below 17: the warning of R7 | |
| Control plane, models, corpus and callers, binaries, disk | | down or missing (the models from the hub cache the facts recorded); a corpus with no traces, or one that does not resolve to a file this user can read (a copy of `long_traces.py`'s links, which `push.sh` copies as files) |

## Single-host mode

A host without a control host beside it, as the box has cyan, runs both: the GPU servers from its node directory and cyan's control plane, on the host's own Unix sockets and loopback, each on its core plan's cores. What differs from the box:

- **No tunnel.** The GPU servers serve on their own sockets, as the box's do (`$XDG_RUNTIME_DIR/flatshare/engine.sock`, `stt.sock` and `tts.sock`), and the control plane reaches them there: the gateway's upstream is the engine's socket, and a drop-in gives the voice worker the STT's and the TTS's, where cyan has the tunnel's `phastform-stt.sock` and `phastform-tts.sock`. No tunnel unit (`flatshare-tunnel@`) is linked; the units that want one start without it, as systemd starts them. E2's agents reach the engine through the gateway's TCP listener, as on cyan. AIPerf and `vllm bench serve` speak TCP only, so the calibration point and the batch arm, which go straight to the engine, reach it through [`deploy/bin/flatshare-forward`](../deploy/bin/flatshare-forward), from a loopback port to the engine's socket, on the load generator's cores, as the box's calibration points reached their engine through the tunnel's TCP forward.
- **One directory of units.** The control plane's units and drop-ins share `~/.config/systemd/user` with the GPU node's: `variant.sh` switches only the GPU units' variant drop-ins (`flatshare-engine`, `flatshare-stt`, `flatshare-tts`, the only units the box's and the node's variants change), and `flatshare-bench e2 run` keeps owning the control plane's (the worker's `vad-turns`, linked for good, and the arms' gateway overlay and worker switches), as on cyan.
- **One GPU of several.** On a host with more than one, the lent GPU is named once (`FLATSHARE_GPU_ID`, its index or UUID; `env.sh` keeps its UUID): the GPU units run on it alone (`CUDA_VISIBLE_DEVICES` by UUID, `CUDA_DEVICE_ORDER=PCI_BUS_ID`), its facts are the ones the profile derives from, and `gpu-guard` and `campaign.sh`'s sampler read it alone, so another tenant's work on the others neither stops a start nor enters a point's clocks.
- **No systemd user manager** (a container, or no lingering), or `FLATSHARE_SUPERVISOR=flatshare-svc` where there is one: [`deploy/bin/flatshare-svc`](../deploy/bin/flatshare-svc) runs the same unit files and drop-ins ([below](#flatshare-svc)), and `deploy/bin/shims/systemctl` and `journalctl`, first on `PATH` (`~/flatshare/node/env.sh`, which `deploy/a100/host.py env` writes), answer what `variant.sh`, `campaign.sh` and `flatshare-bench e2 run` ask.

  The first stage chooses the supervisor, enabling lingering where a user may enable its own and reading the facts again at once, and `env.sh` keeps the choice, flatshare-svc or systemd, for the later stages and every script that sources it; `FLATSHARE_SUPERVISOR=auto` chooses again from the host's facts, and it or `FLATSHARE_SUPERVISOR=flatshare-svc` moves the units, stopping them first under the supervisor they leave, so that none runs under both (the preflight refuses a split control plane).
- **No user namespaces** (Ubuntu 24.04 restricts them): the voice plane runs in the host's network namespace (`FLATSHARE_VOICENET=host`), under either supervisor; LiveKit's own candidates stay on 127.0.0.1, but the callers' and the worker's WebRTC clients gather theirs on the host's interfaces, so the preflight reports it as degraded and names the host's public addresses, which must drop inbound UDP. `in-voicenet` marks the processes it runs there `FLATSHARE_VOICENET=host`, and the listener audit of `smoke.sh` (block 0) accepts their UDP sockets, loopback TCP listeners and file sockets, which in their own namespace it would not see, and refuses anything else of ours, an abstract socket or a TCP listener off loopback among them. The GPU servers run in the host's too, by the `FLATSHARE_PRIVATE_NET=host` that the host's drop-ins set for `in-private-net`, which anywhere else fails a unit whose namespace cannot be made: there the audit accepts the loopback TCP listeners of vLLM's engine cores on ephemeral ports (torch's gloo groups) and CUDA's abstract sockets, which any user of the host reaches, and refuses anything else of theirs.
- **The driver's libraries outside the linker's search** (Colab's container keeps `libcuda.so.1` and the rest in `/usr/lib64-nvidia`, which `ldconfig -p` does not list and only its shells' `LD_LIBRARY_PATH` reaches, so `nvidia-smi` works there while torch in a unit finds no driver): the facts find the directory (the linker's cache, else the stages' `LD_LIBRARY_PATH`, else `/usr/lib64-nvidia` or `/usr/lib/x86_64-linux-gnu`; `driver_libs` in `host.json`), and where the linker does not search it every GPU unit names it on its own `LD_LIBRARY_PATH`, after `cuda-compat-13-0`'s where that is used, since a unit's environment has none of the operator's under either supervisor; the preflight's driver and torch rows name the case. A driver's libraries elsewhere are found by running the stages with their directory on `LD_LIBRARY_PATH`.
- **The measurement drop-in** ([`deploy/phastform/measure`](../deploy/phastform/measure)): the A100 run reads vLLM's development routes, as the box's measurements do (`/server_info`, `/reset_prefix_cache`): `campaign.sh` turns the drop-in on before each point, and `host.py warm` before its starts.
- **The campaign runs on the host** (`benchmarks/e2/campaign.sh --local`), so a dropped workstation link stops nothing, and [`deploy/a100/pull.sh`](../deploy/a100/pull.sh) copies each point, the probes, the node's records, the warm starts' and the units' logs (exported from the host's journal, or `flatshare-svc`'s, since the last round) to the workstation every two minutes.

None of these differs between the frozen engine and make-room on one host, so a single host's comparisons are made within it.

### flatshare-svc

[`deploy/bin/flatshare-svc`](../deploy/bin/flatshare-svc) runs the repository's systemd user units on a host without a user manager. It is Python's standard library alone, from 3.10 (Ubuntu 22.04's `python3`), since the shims run it before any venv exists. With `deploy/bin/shims` first on `PATH`, `systemctl` and `journalctl` are flatshare-svc, so `variant.sh`, `install.sh`, `campaign.sh` and `flatshare-bench e2 run` work unchanged.

It reads each unit and its drop-ins (`UNIT.d/*.conf`, in name order) from `~/.config/systemd/user`, where `install.sh` and `variant.sh` link them, afresh at every command. It implements the directives the repository's units use, as systemd does:

- `Type=notify` (the gateway): started at the main process's `READY=1` on `NOTIFY_SOCKET`, as `sd_notify` sends it. `Type=exec` or `simple` without `ExecStartPost`: started once the main process has been executed, even if it exits at once.
- `ExecStartPre`, `ExecStart`, `ExecStartPost`, `ExecStop` and `ExecStopPost`: an empty assignment resets the list, and a leading `-` ignores a command's failure.
- `Environment`, `EnvironmentFile` (`-` for an optional file) and `WorkingDirectory`.
- `CPUAffinity` through `taskset`, and `NUMAPolicy=preferred` with `NUMAMask` through `numactl`, where the host has them; otherwise the journal says it was not applied.
- `KillSignal`, `TimeoutStartSec` and `TimeoutStopSec`; `Restart` (`no`, `on-failure`, `always`), `RestartSec`, `StartLimitIntervalSec` and `StartLimitBurst`.
- `Wants`, `Requires`, `BindsTo`, `PartOf` and `After` among the units, a target's `Wants` included. A unit that `Wants=` names and that does not exist is left out, as systemd leaves it (a tunnel a single host has none of); one that `Requires=` or `BindsTo=` names fails the start.
- A socket unit (`flatshare-gateway.socket`) starts and stops as a target does, holding no listener: the gateway binds its own, as it does without `LISTEN_FDS`.
- In command lines, the specifiers `%n` and `%N` (the unit's name, and that name without its suffix), `%h`, `%t`, `%u`, `%U` and `%%`, and systemd's `$NAME` (a word of its own, split at whitespace) and `${NAME}` (in place).

A start refuses a unit with any other specifier in a command line before anything of the start runs, as systemd will not start a unit whose lines it cannot expand: a line that only a stop runs would otherwise fail at the stop, with the unit's processes running. `ExecReload`'s lines are checked too, although flatshare-svc runs no reload: its `systemctl` refuses `reload`, and `daemon-reload`, `enable`, `disable` and `import-environment` do nothing.

- **A start** takes a target's units one at a time, in the target's order, each after the one before it has started: the GPU servers start STT, TTS, then the engine, where systemd starts the three at once. It returns once a unit's `ExecStartPost` commands have succeeded, and fails as systemd's does when they cannot.
- **Each started unit has a runner** (`flatshare-svc run UNIT`, in a session of its own). It runs the unit's commands, relays their output into the unit's journal (each line with its Unix time), restarts the unit as `Restart` says and stops it on SIGTERM. The commands carry `FLATSHARE_SVC_UNIT=UNIT` where systemd would put them in the unit's cgroup: `gpu-guard` tells them from a foreign process by it, and `smoke.sh`'s listener audit tells whose listener is whose (the gateway's 127.0.0.1:18080 among them, which the gateway binds itself here).
- **A stop** runs a unit's `ExecStop` (the gateway's waits for its drain), then sends its `KillSignal` to the main process's group, SIGKILL to what still runs after `TimeoutStopSec`, and runs `ExecStopPost`. The signals and `ExecStopPost` run also when `ExecStop` fails or cannot run, which leaves the unit failed, as under systemd. A stop returns once every process it stops has exited: the unit's own, those of the units bound to it or part of it, and a target's units with the target.
- **A main process that exits on its own** has what is left of its group ended as a stop ends it, before `ExecStopPost` and any restart, as systemd ends what is left of a unit's cgroup: an engine core that outlived its server would otherwise hold the GPU, and the restart's `gpu-guard` would refuse to start beside it. The unit is deactivating meanwhile, and a start waits for it.
- **Orphans.** On Linux each runner is the subreaper of its unit's processes and reaps those whose parent exited before them, as systemd's manager does. Left to a container's init, which may reap none (`sleep infinity`), each would stay in its group as a zombie, and every restart and stop would wait out `TimeoutStopSec`.
- **State** is in `$XDG_STATE_HOME/flatshare-svc` (default `~/.local/state/flatshare-svc`): `UNIT.json` and the journals, `log/UNIT.log`.

## A GPU node without user namespaces (Gray)

Gray, the second RTX 5060 Ti, runs Ubuntu 24.04 with `kernel.apparmor_restrict_unprivileged_userns=1`, and its deploying user has no root to lift it, so an unprivileged user namespace is refused there (`unshare: write failed /proc/self/uid_map: Operation not permitted`, 2026-10-27): `in-private-net` cannot give a GPU server a network namespace of its own, and fails the unit of one that asks for it rather than run it without one. A two-host deployment handles it as the single host's preflight does, but by the node's install, explicitly and at every release:

- **Found at every install.** [`deploy/bin/private-net-check`](../deploy/bin/private-net-check) tries what `in-private-net` does (`unshare --user --map-root-user --net`, then the loopback up) and says `allowed`, `refused` with unshare's own error and the sysctl's value, or `unknown` where the host has no `unshare` or `ip`. [`deploy/gray/install.sh`](../deploy/gray/install.sh), which every release runs at its switch, links [`deploy/gray/private-net-host.conf`](../deploy/gray/private-net-host.conf) into the engine's, the STT's and the TTS's drop-ins where the kernel refuses, removes it where it allows, and changes nothing where the check cannot tell. Phastform, whose kernel allows them, has no such drop-in, and its servers keep namespaces of their own.
- **What the servers expose.** In the host's network namespace each server still serves only its Unix socket, in `$XDG_RUNTIME_DIR/flatshare` (mode 0700). What the host's other users can reach is what a namespace of its own would have hidden: the loopback TCP listeners of vLLM's engine cores on ephemeral ports (torch's gloo groups) and CUDA's abstract sockets. `deploy/bin/audit-listeners` accepts exactly those, by the servers' `FLATSHARE_PRIVATE_NET=host` mark, counts them apart, and fails on anything else of theirs, as on every host.
- **Recorded and checked by each release.** The manifest's `isolation` says where each server runs and the check's verdict ([the runbook](../ops/RUNBOOK.md#a-gpu-node-without-user-namespaces-gray)), and a release whose GPU server runs or would restart fails its checks before the switch if that server would ask for a namespace the kernel refuses, as an unmarked one on such a kernel would.
- **Undone the same way.** Allowing unprivileged user namespaces (`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, root's) and the next install remove the drop-in; each server takes its own namespace at its next start.

None of it changes how a server schedules or how fast it answers, only what else on the host can reach its loopback listeners, so a point measured on Gray compares with one on Phastform as one card with the other.

## Running on another GPU, step by step

From the workstation, with the repository at the revision to run and the corpora and the callers' audio (`weka-x32`, `weka-x32-long`, `callers`) in `FLATSHARE_DATA`:

```sh
deploy/a100/push.sh USER@HOST            # the source tree (with DEPLOYED), the data as files; FLATSHARE_GATEWAY_BIN to bring a flatshare-gateway
```

On the host, from the home directory:

```sh
bash flatshare/src/deploy/a100/bootstrap.sh stage     # no GPU needed: before the billed clock if the host is up earlier
bash flatshare/src/deploy/a100/bootstrap.sh gpu       # with the card (the GPU units stopped): the driver, the install path, vLLM's parse, the preflight
bash flatshare/src/deploy/a100/bootstrap.sh warm      # one start per compile configuration, so the campaign's are warm
nohup setsid bash flatshare/src/benchmarks/a100/run.sh --until HH:MM \
  >>flatshare/runs/a100/run.log 2>&1 </dev/null &  # the A100 run's core, block by block
```

On a host with several GPUs, `FLATSHARE_GPU_ID=INDEX|UUID bash flatshare/src/deploy/a100/bootstrap.sh stage` names the lent one (its index or UUID in `nvidia-smi -L`; [single-host mode](#single-host-mode) says what that changes). `FLATSHARE_SUPERVISOR=flatshare-svc bash flatshare/src/deploy/a100/bootstrap.sh stage` runs the units under `flatshare-svc` where a user manager could. `env.sh` keeps both choices for `gpu`, `warm` and the run.

`stage` takes uv and its own Python 3.12 (with its headers, where uv would prefer the system's), renders the node into `~/flatshare/node` (from the profile's own card when the GPU is not yet visible), builds the engine, TTS, voice and bench venvs from that Python (the engine's lock by R8), fetches Kokoro and the models at their pinned revisions (`FLATSHARE_MODELS=MODEL@REVISION` adds one, the 14B for C') into the Hugging Face hub cache the operator's environment names (`HF_HUB_CACHE`, or `HF_HOME`'s, a data volume's say; resolved once, at the first stage, kept in `env.sh` and named in the engine's and the STT's units, which load offline), installs LiveKit, builds `flatshare-gateway` when a copied binary does not run, links the units with this host's drop-ins, writes the control plane's configuration (`deploy/a100/host.py config`: new keys, gateway.toml from cyan's example, `e2-host.toml` for `flatshare-bench e2 run`), starts the control plane, and builds FlashInfer's sampler for the card's architecture into the engine's own workspace (`host.py jit-check`), which checks the compiler before a GPU minute is spent.

On the workstation meanwhile: `deploy/a100/pull.sh USER@HOST`, and after: `make reproduce-a100 RECORDS=~/flatshare/hosts/HOST CORPUS=<weka-x32>`.

**After a reboot** nothing of the run is up: the units are not enabled, and `flatshare-svc` is no boot-time service. Start `run.sh` again as above: it brings the control plane up first (`systemctl --user start flatshare-voice-worker.service`, through `env.sh`'s shims under `flatshare-svc`), each block starts the GPU units it needs, and the run resumes at the block it stopped in. `flatshare-svc` keeps each process's boot and start time beside its pid, so a state file from before the reboot, or a pid reused since, reads as a unit that is not running, which a start starts and a stop leaves alone.

## What to ask for at handover

The preflight finds the host's facts, but these decide the plan, and some need the provider. The first three are what `bootstrap.sh` must be told or cannot find out by itself:

1. SSH access: the user and the key, and whether the host has a public address and a firewall that drops inbound UDP (the voice plane, should it run without its own namespace).
2. Whether the host is a container (its `/dev/shm`: Docker's default 64 MB holds no CPU tier; `--shm-size` or `--ipc=host`), and whether there is root or sudo: build-essential, iproute2, git and `cuda-compat-13-0` install through apt as root (a container's root needs no sudo) or with `sudo -n`, as do the sysctl that allows user namespaces and any compute-mode change.
3. How many GPUs the host has and which one is ours, by its index or UUID in `nvidia-smi -L` (`FLATSHARE_GPU_ID=INDEX|UUID bash flatshare/src/deploy/a100/bootstrap.sh stage`; the preflight refuses several with none named), the card and its form factor (an A100 40 GB or 80 GB, SXM4 or PCIe), whether MIG is off (a slice is refused unless `FLATSHARE_ALLOW_MIG=1`: the forecast is the whole card's), and its compute mode (Default; `Exclusive_Process`, common on shared HPC nodes, lets one process at a time use the card and is refused).
4. The driver version (R580 or newer runs the box's lock as it is; R535 or R570 on a data-center card with `cuda-compat-13-0` installed, or root or sudo to install it; else the cu129 lock), and any CUDA toolkit installed (13.0 or 12.9 saves the toolkit download). In a container, also where the driver's libraries are when the linker's cache does not list them (`ldconfig -p | grep libcuda.so.1` shows nothing; Colab's are in `/usr/lib64-nvidia`): the stages find them on their `LD_LIBRARY_PATH` or there.
5. Whether the host is reachable before the billed clock, for `stage` and `warm` (block 0 is 44 minutes without them [estimate]; `warm` alone took 29 minutes on the box and 9 on the A100, measured).
6. Whether the home directory is on a volume that outlives the host's restarts: the compile caches and every record live there until `pull.sh` copies them. If `HF_HOME` or `HF_HUB_CACHE` is set, the models go where it says, a data volume's, say; the first stage keeps that choice.
7. The OS image, systemd with lingering (the first stage enables it where a user may enable their own), and whether unprivileged user namespaces are allowed (Ubuntu 24.04 restricts them unless sudo lifts it): without them the voice plane runs in the host's network namespace, degraded.
8. CPUs and their NUMA nodes (34 give the core plan its wants, 17 is the floor), RAM (the tier is 20 GiB for the core, 71 GiB for C' by R3), and free disk: the engine venv takes 6.1 GB on the box's lock and 12 GB on the cu129 lock, the CUDA 12.9 toolkit 9.4 GB (measured on a CPU host), the models about 7 GB and 16 GB more for C' [estimate].
9. Egress to PyPI, download.pytorch.org, GitHub, Hugging Face and NVIDIA's download site, for staging.
10. The billed window's length: the core was estimated at about 4.7 hours, and on the A100 it took 3 hours 13 minutes after 19 minutes of staging, `gpu` and `warm`.

## The A100 run

The A100 run asks whether make-room's result carries to a datacenter card; it ran on an A100 SXM4 80 GB on 2026-10-26 ([what it measured](#what-it-measured-2026-10-26)). Its core is like-for-like with the box: the box's model at its revision, its 5 GiB pool, 512/256, the 20 GiB CPU tier, 24 sessions of `weka-x32-long`, two callers, W5's three arm files and five rounds at seeds 11 to 15; only the card, its precision path and the host change. So a difference within the A100 is the change's, and a difference from the box is the card's.

A forecast written before the run gives every term the run measures, modeled, for each card the run could draw: [`forecast.md`](../benchmarks/a100/forecast.md) for the A100 40 GB, the card the run was planned on, and [`forecast-80gb.md`](../benchmarks/a100/forecast-80gb.md) for the A100 SXM4 80 GB (`make forecast-a100 GPU=a100-80gb`: the same compute, 2,039 GB/s of HBM2e against 1,555). The analysis compares a run against the forecast for the card its records name (`node/host.json`): `forecast-80gb.md` for an A100 80 GB, as on 2026-10-26, and `forecast.md` otherwise. For the A100 SXM4 80 GB the forecast gives the KV case's stock wait as 1,328–1,369 ms and make-room's first token as 126–145 ms. In the core scenario it gives a running share, the time running agent requests leave two calls short of KV, of 10% against the box's 17%, and the card 71% busy (its step time over the replay's) against the box's 100%. For the A100 40 GB the same four are 1,636–1,688 ms, 134–158 ms, 13% and 76%.

### The core, block by block

[`benchmarks/a100/run.sh`](../benchmarks/a100/run.sh) runs it on the host; each block is complete when it ends and is skipped once done, so a stop anywhere leaves whole comparisons and a rerun resumes. The plan's minutes are the box's measured ones where they exist and estimates otherwise; the last column is what each block took on the A100 on 2026-10-26 (`blocks.log` in [its records](../benchmarks/results/2026-10-26-a100/README.md)):

| Block | Work | Minutes (estimate) | Complete after it | Minutes on the A100 |
|---|---|---:|---|---:|
| 0 | `bootstrap.sh gpu` and `warm` (on the box's kernels in [the rehearsal](#the-kit-rehearsed-on-the-box), measured: the three servers' first start from empty caches 20.5 minutes to ready, each of the three other compile configurations, make-room's at the PRs' values among them, 2.5 to 2.9 minutes), the box's `smoke.sh` from the source tree (its listener audit included), one 2-turn call through LiveKit | 44 | the deployment, every compile configuration warm | 3 |
| 1 | the frozen engine: E0 (`flatshare-bench tiers --reps 3`), the KV case x3, the prefill case x3, the calls' tail x2 with whole 40-token replies ([`probes.sh`](../benchmarks/a100/probes.sh) 1) | 25 | the frozen side of the probes, E0's cells | 12 |
| 2 | stock at vLLM's default, 2,048/0: KV x3, tail x2; stock at the time-equivalent cap (`--cap`, 1,536): KV x1, tail x2 | 26 | the stock step-cap settings | 20 |
| 3 | the PRs' values (make-room, no reserve, a 48-token window): KV x3, prefill x3, tail x2 | 17 | the probe comparison on the A100 | 10 |
| 4 | E2 round 1 ([`a100-r1.campaign`](../benchmarks/a100/a100-r1.campaign)): the PRs' values, voice alone, frozen | 29 | one E2 round | 26 |
| 5 | the calibration point ([`calibration.sh`](../benchmarks/a100/calibration.sh)): AIPerf's replay of the whole corpus at 16 sessions straight to a fresh frozen engine (through `flatshare-forward`'s loopback port), 600 s + 60 s | 17 | the step model's data | 17 |
| 6-9 | rounds 2 to 5, the engine groups alternating so that no switch falls between rounds | 4 x 28 | three rounds by about 3.6 h, five by about 4.5 h | 4 x 26 |
| 10 | the GPU units stopped; the last pull | 10 | the core, about 4.7 h | 0.1 |

The rehearsal ran block 0 in 34 minutes on the box from a fresh home (`gpu` 1.4, `warm` 28.8, the smoke test and the call 4.1) and block 1 in 16. The estimates keep their margin for a host whose install path, CPUs and kernels are not the box's; `--estimate` replaces them once a host is measured. On the A100, `bootstrap.sh gpu` and `warm` ran before `run.sh`, in 3 and 9 minutes, so block 0 found every compile configuration warm, and the core's blocks took 3 hours 13 minutes against the 4.7 hours estimated.

`run.sh --until HH:MM` (UTC) starts no block whose estimate above would end after then, and hands that one deadline, as an epoch, to every campaign and to C', which start no point that cannot end before it: a point is budgeted by the longest of its kind measured on this host once one has run (`durations.tsv`), by W5's durations before. A campaign stopped so leaves its block undone and the run exits 0; any failure exits 1. `--estimate BLOCK=MINUTES` puts another estimate in the table's place for a block or an arm (a host measured faster or slower, or an E2 block run for its first point alone, which the campaign's own budget then stops at the deadline).

During every agent point `flatshare-bench e2 run` samples the engine's `/metrics` every 0.5 s, from which each point's summary gives the measured running share (the time free KV is under two calls' 2,048 tokens). W5's and FB's points on the box ran without this sampler; RS measured the box's afterwards on W5's samples, 8.1% beside the frozen engine against the forecast's 17% ([RS](BENCHMARKS.md#the-boxs-running-share-rs)).

### The optional arms

Each arm is complete in itself and runs after the core (`run.sh NAME`); the list is in the plan's order, and arm 8 was added during the run. On 2026-10-26 the batch, tuned vLLM, the PRs' values at tuned vLLM's cap, A' (one round) and FlashInfer's attention ran, in that order; C', the PRs' own code and preempt-resubmit did not ([what it measured](#what-it-measured-2026-10-26)).

1. **C'** ([`cprime.sh`](../benchmarks/a100/cprime.sh)): Qwen3-14B-FP8 at the A100 40 GB's own pool, 117,344 tokens (modeled), at 40 sessions, on its own node (`a100-40gb-qwen3-14b`), the agent's requests naming it (the `qwen3-14b` worker variant): its first starts, the pool checked, a smoke reply without `<think>`, E0 with one rep, its own voice alone, then three rounds of tuned vLLM (the step cap with the lowest calls' first-token p95 in blocks 1 and 2, [`probe_results.py`](../benchmarks/a100/probe_results.py) `pick-stock`) against the PRs' values. About 95 minutes.
2. **The PRs' own code** ([`branches.sh`](../benchmarks/a100/branches.sh)): vLLM at the PRs' tip with their switches on against vLLM at their base, two Python-only installs over precompiled binaries staged beforehand (`branches.sh stage BASE HEAD`), each through the probes: KV x3, prefill x2, tail x2. About 45 minutes.
3. **Preempt-resubmit** on the stock engine (`a100-resubmit-r3` to `-r5`): three points, seeds 13 to 15. About 36 minutes.
4. **The scheduler at a datacenter batch** ([`batch.sh`](../benchmarks/a100/batch.sh)): `vllm bench serve` keeping 256 requests in flight, 10% at priority 0, on frozen and on make-room with its switches off and on, each request small enough (48 prompt and 64 output tokens, 8 blocks at most) that all 256 hold their KV at once in the core's pool (2,048 of its 2,275 blocks), so the arm measures the scheduler's per-step cost at a batch of 256 rather than a queue behind the KV; the engine's running requests, sampled once a second, show the batch formed (`ENGINE-running.json`), and a pool too small for it is refused. The clients load their tokenizer from the engine's own snapshot, at the revision `engine.yaml` pins, since offline the model's name resolves to nothing where the download was by commit. About 15 minutes [estimate: four engine switches and two cold make-room compiles].
5. **A'** (`a100-aprime-r1`, `-r2`): the box's pool at 32 sessions, where the forecast places the A100 closest to the box's regime (a running share of 18%), each round the frozen engine and the PRs' values on one sample of the agents; voice alone, the SLO's reference, is the core's five points on the same card. About 56 minutes.
6. **Tuned vLLM** (`a100-tuned-r1`, `-r2`): the stock engine at the step cap whose calls' first-token p95 was the lowest in blocks 1 and 2 on this card (`probe_results.py pick-stock`), otherwise the core's frozen point, on the samples of the core's rounds 1 and 2, so that each pairs with that round's frozen point and the PRs' values. The core keeps the box's 512/256, the box's best, so that a difference within the A100 is the change's; this arm sets the change beside the stock engine at the A100's own best. About 28 minutes.
7. **FlashInfer's attention** (`run.sh backend`, [`probes.sh`](../benchmarks/a100/probes.sh) `backend`): the frozen engine with FlashInfer's attention in place of FlashAttention 2 (`--attention-backend FLASHINFER`) through the KV case x3, whose two 18k-token decodes take the same steps on either backend, so that its wait reads the decode step on each; the block stops before a run unless the engine's log at that start names FlashInfer. The calibration fitted the A100's decode attention at 42% of the card's bandwidth, where the box's reaches 77% of its own. About 12 minutes, more at the first start, which JIT-compiles FlashInfer's attention kernels for the card.
8. **The PRs' values at the A100's best stock step cap** (`run.sh prtuned`, `a100-prtuned-r1`, `-r2`): the PRs' 48-token configuration with tuned vLLM's 2,048/0 ([`a100-pr-ab48-2048-s24.toml`](../benchmarks/a100/arms/a100-pr-ab48-2048-s24.toml)), on the samples of the core's rounds 1 and 2, so that each pairs with tuned vLLM's round. Added during the run, when tuned vLLM's first round gave the agents 40% more tokens a second at 2,048/0 than the box's 512/256 on the same sample, the voice as far over its SLO: whether the PRs keep the voice under its SLO at the card's own best cap. About 24 minutes.

MIG (a 3g.20gb voice instance beside a 4g.20gb agent instance) needs administrator rights and a GPU reset and is not scripted.

### The analysis

`make reproduce-a100 RECORDS=<the pulled host directory> CORPUS=<weka-x32>` writes `build/reproduce-a100/a100.md` ([`analysis.py`](../benchmarks/a100/analysis.py)), each section from what the records hold: the step model fitted on the calibration point with the card's own E0, each term beside the forecast as a ratio; flatshare-sim with those costs against the measured point, as the box's calibration table; the probes beside the forecast and the box; `flatshare-bench e2 report` over the A100's points alone (the core's with tuned vLLM's and the PRs' values' at its cap, then A' with the core's voice alone), each round's points side by side and each configuration's measured running share; the batch arm's output tokens a second against the frozen engine's, each priority's latency beside them; the host around the card; and the two cards in regime coordinates only. The run of 2026-10-26 has its own as [`a100.md`](../benchmarks/results/2026-10-26-a100/a100.md) in its records.

The host is measured by [`bench_host.py`](../benchmarks/a100/bench_host.py), with the GPU units stopped (`flatshare/venvs/engine/bin/python flatshare/src/benchmarks/a100/bench_host.py --out flatshare/runs/a100/host`): its link from pinned host memory to the card through vLLM's own batched page copies, at the CPU tier's page sizes, each way, and its CPU at a fixed Python workload, the two of the host's costs the forecast takes from the box. The analysis reads them with E0's GPU hit (the host's work on a prompt, which grows with it) and its CPU restore beyond the GPU hit (the transfer).

`make check-a100-calibration EXPORTS=<the v0.3 calibration exports> CORPUS=<weka-x32>` runs the analysis's calibration code on the box's three W2 points with the box's E0, and checks that it gives the box's own table rows.

### On Google Colab

The run of 2026-10-26 had an A100 SXM4 80 GB on a Colab runtime with high RAM: a container (Ubuntu 24.04, root, no
systemd user manager) with 12 vCPUs of a 2.20 GHz Xeon (Cascade Lake), 167 GiB of RAM, an 82 GiB `/dev/shm`, driver
580.82.07 and CUDA 13.0's toolkit. Colab's command-line client
([google-colab-cli](https://github.com/googlecolab/google-colab-cli), its `colab` command, signed in once) makes the
runtime and reaches it:

```sh
colab new -s a100 --gpu A100 --high-mem   # the runtime, billed in compute units from here to `colab stop -s a100`
colab usage                               # the balance and the rate (6.77 units an hour for this runtime)
```

and an SSH host entry whose `ProxyCommand` is `colab ssh --proxy-mode -s a100` carries `push.sh`, `pull.sh` and every
command (`FLATSHARE_SSH="ssh -F <that file>"`). Four things differ from the hosts the kit was rehearsed on:

1. **One connection.** The bridge carries one SSH connection at a time and refuses a second (HTTP 429): the host entry
   multiplexes every command over one master connection (`ControlMaster auto`, `ControlPersist`), so that `pull.sh`'s
   copy every two minutes, a watch and a command typed meanwhile share it, and the run starts detached (`nohup setsid`),
   held by no connection.
2. **No user manager.** The first stage chose flatshare-svc.
3. **The driver's libraries outside the linker's search** (`/usr/lib64-nvidia`): `bootstrap.sh gpu` refused the host
   until the GPU units named that directory, which the kit now does ([single-host mode](#single-host-mode)); this run's
   host had it registered by hand (`/etc/ld.so.conf.d`, `ldconfig`) before the fix.
4. **Billing from creation to stop**, whatever the card does: staging ran before the GPU stages, the core's blocks back
   to back, and the optional arms were queued to start as the core ended, under one deadline the balance set; the
   records were pulled every two minutes, and the runtime stopped once the last pull was in. This run's runtime lived
   5 hours 5 minutes (04:07Z to 09:12Z) and used 34.35 compute units of 37.38, at 6.77 an hour; the GPU stages held the
   card from 04:28Z to 09:12Z, 93% of that time.

### What it measured (2026-10-26)

The run had an A100 SXM4 80 GB on Google Colab ([above](#on-google-colab)), so the analysis reads it against
[`forecast-80gb.md`](../benchmarks/a100/forecast-80gb.md). Staging, `gpu` and `warm` took 19 minutes (`warm` 9 of them,
the four compile configurations from empty caches on the host's 12 vCPUs), and the core's blocks 0 to 10 ran back to
back in 3 hours 13 minutes, each E2 round in 26 minutes. Its records are
[`benchmarks/results/2026-10-26-a100`](../benchmarks/results/2026-10-26-a100/README.md), with the analysis's report.

**E2, the core's five rounds** (W5's arm files and samples of the agents, seeds 11 to 15). Voice alone ran at a p95 of
1,419 ms (1,391–1,445), so the SLO is 1,669 ms:

| Two calls beside 24 coding-agent sessions | Voice p95, median (range) | Over the SLO | Agents' tokens/s | Agents' TTFT p50 / p99 | Running share |
|---|---:|---:|---:|---:|---:|
| Frozen, the box's flags (512/256) | 4,986 ms (4,163–5,710 ms) | +3,317 ms | 8,633 | 12 s / 26 s | 6% (4–9%) |
| The PRs' values (make-room, a 48-token window) | **1,634 ms** (1,609–1,663 ms) | −35 ms | 8,651 | 14 s / 26 s | 6% (2–7%) |

The PRs' values met the SLO in every round, by 7 to 60 ms; frozen missed it by 2.5 to 4.0 s. Round by round, the agents'
tokens/s over the calls' first 240 s with the PRs' values were 94–111% of frozen's (a median of 96%), their whole-call
tokens/s 95–103%, and no turn of the 750 failed. The forecast put the core's running share at 10% [modeled].

**The probes (blocks 1 to 3)**, against the forecast and the box. The box's column is the report's, the four runs of its
§2.4 on Phastform, against which [Table E1](REPORT.md#e-the-forecast-for-the-a100) compares the A100:

| Probe | A100 SXM4 80 GB | Forecast, central (low–high) [modeled] | The box (Table E1) |
|---|---:|---:|---:|
| KV case, stock 512/256: the voice turn's wait | 1,748 ms (1,721–1,750, 3 runs) | 1,328–1,369 ms (1,082–1,377) | 4,968 ms (4,900–4,971) |
| KV case, stock at vLLM's default 2,048/0 | 1,642 ms (1,620–1,645, 3 runs) | | |
| KV case, the PRs' values: the first token | 175 ms (150–176, 3 runs) | 126–145 ms (109–146) | 290 ms (216–294) |
| Prefill case, stock: the reply's token gap (1st to 16th) | 32 ms (3 runs) | 32.8 ms (25.7–35.4) | 60–66 ms |
| Prefill case, the PRs' values: the token gap | 7 ms (3 runs) | 5.6 ms (3.7–5.6) | 16–20 ms |
| The calls' first-token p95 beside 12 agents, 512/256 | 3,114 ms (3,102–3,126, 2 runs) | | 7,631 and 7,643 ms (the kit's rehearsal) |
| The same at 2,048/0, and at 1,536/768 | 2,526 ms (2,518–2,535) and 3,566 ms (3,446–3,687) | | |
| The same, the PRs' values | 173 ms (167–179, 2 runs) | | |

Of the stock step caps, vLLM's default 2,048/0 gave the A100 its lowest calls' tail, so tuned vLLM's arm ran it.

**The arms after the core**, in the order they ran: the batch, tuned vLLM, the PRs' values at tuned vLLM's cap (arm 8,
added once tuned vLLM's first round was in), A' (its first round) and FlashInfer's attention. C' (about 95 minutes),
the PRs' own code (45) and preempt-resubmit (36) did not run: the balance left after the core went to the five that set
the core beside the card's own best stock configuration.

- **The scheduler at a batch of 256** (9 minutes): with its switches off make-room made 1.018 of vLLM's output tokens a
  second (7,661 against 7,524), with 215 requests running at the median against 201, so its code path costs nothing this
  batch shows. With the PRs' values the priority 0 requests' TTFT p99 went from 574 to 177 ms and their TPOT p50 from
  28.6 to 6.9 ms, the engine's output to 1.153 of vLLM's, and the priority 2 requests' TTFT p99 from 1,086 to 7,971 ms
  (p50 341 to 283 ms): 26 priority 0 requests in closed loop keep first tokens due in most steps, and #63351 holds
  lower-priority prefill out of those steps ([BENCHMARKS.md](BENCHMARKS.md#the-a100-run-2026-10-26)).
- **Tuned vLLM, 2,048/0** (rounds 1 and 2, seeds 11 and 12): V2V p95 4,036 and 3,289 ms, against frozen's 4,163 and
  4,783 and the PRs' values' 1,663 and 1,658 on the same samples; the agents 15,082 and 13,990 tokens a second over the
  calls' first 240 s, 1.40 and 1.42 times frozen's and 1.49 and 1.29 times the PRs' values'. At the A100's own best
  stock cap the voice tail stays two to 2.4 times its SLO, while the box's 512/256 costs the agents about 30% of their
  tokens.
- **The PRs' values at 2,048/0** (arm 8, the same two samples): V2V p95 1,810 and 1,843 ms, 140 and 173 ms over the SLO,
  against tuned vLLM's 4,036 and 3,289 ms on the same samples; the agents at 100% and 101% of tuned vLLM's tokens a
  second (15,124 and 14,145), their TTFT p50 1.0 and 2.0 s and p99 18.4 and 11.2 s, against tuned vLLM's 1.0 and 2.0 s
  and 17.3 and 11.4 s; no turn of the 100 failed. The voice turn's stages are the 512-token PRs' but one: the LLM's
  first token, 412 and 414 ms at p95 against 183 and 216 ms at 512/256 on the same samples. A turn that arrives with
  steps in flight waits for them, two under asynchronous scheduling, and at 2,048 tokens a step of agent prefill carries
  four times the tokens; the hold acts only on the steps scheduled after the turn arrives. So the step cap trades the
  voice's tail against the agents' throughput, with the PRs as without: 512/256 meets the SLO at 70% of the card's best
  throughput, 2,048/0 keeps all of it 0.14–0.17 s over.
- **A' at 32 sessions** (round 1, seed 11; round 2 would have ended past the deadline with one point of two): frozen V2V
  p95 5,222 ms, the PRs' values 1,617 ms, 52 ms under the SLO; the agents 8,649 and 8,487 tokens a second (98%), their
  TTFT p50 24.5 and 25.9 s, 57 and 65 preemptions, no turn of the 100 failed. At the box's pool more agents make fewer
  tokens than 24 do on the same sample (10,749 and 10,125), waiting longer for KV, and two calls found KV short 6.5% of
  the time beside either engine, where the forecast placed 32 sessions at 18% (the fitted forecast at 15%): the
  simulator overstates how often this card runs short, here and in the core (10% forecast, 6% measured).
- **FlashInfer's attention** (the frozen engine at 512/256, the KV case x3): the voice turn's wait 1,580, 1,600 and
  1,607 ms against FlashAttention 2's 1,721–1,750, 8% shorter at the median. The two 18k-token decodes take the same
  steps on either backend, so the wait reads the decode step: FlashInfer's is shorter on these contexts, a backend's
  share of the tail that the rules leave as it is. The block refused its first start, since vLLM 0.32 logs FlashInfer's
  selection as `FlashInfer resolved query dtypes`, not in the line its check read; the check reads both since, and the
  block ran again.

**The step model (block 5)**, fitted on the calibration point (AIPerf's replay of the whole corpus at 16 sessions, 709
requests) with the card's own E0: a step's fixed cost 4.5 ms (the forecast's 3.88), 41.36 µs a token (41.33: Marlin's
W8A16 GEMMs as modeled), 4.07 ns a token pair (2.59) and 171.9 ns a KV token read (94.3). Decode attention is the miss,
1.82 times the forecast: 858 GB/s of KV, 42% of the card's bandwidth, where the box's FlashAttention 2 reaches 77% of
its own. The fit reproduces the point's decode steps to −3.0% at the median (−8.6/+8.0% at p10/p90) and E0's cells to
−0.7%.

**E0 (block 1)**, the box's fits beside it: a GPU hit 17 ms + 1.80 µs a token (the box's 35 ms + 3.8), a CPU restore 2.8
ms + 12.9 µs a token over it, 11.5 GB/s of KV (the box's 6.2), and a recompute 4.2 ms + 56.5 µs a token + 2.01 ns a
token squared (48 ms + 63 µs + 8.1 ns): at 16,384 tokens 46 ms, 268 ms and 1,471 ms against the box's 97, 509 and 3,264.
Restoring beats recomputing from 1,024 tokens, as on the box. Under the 512-token step cap a cold prefill's linear cost
is nearly the box's: without FP8 units the card runs the checkpoint's GEMMs through Marlin's weight-only path.

**The host** (`bench_host.py`, the GPU units stopped): Python's fixed workload in 227 ms at the median (the box's Xeon
E5-2609 v4, 381 ms), and the link at the CPU tier's 32 KiB pages 10.08 GB/s to the card and 10.61 back, one copy 12.32
and 12.86 GB/s. The card negotiates PCIe Gen4 x16, but the host's Xeon is a Cascade Lake, whose root ports are PCIe 3.0:
the copies run at a Gen3 x16 rate, 39% of the card's link (the box's PCIe 3.0 x8 reaches 5.44 GB/s at 32 KiB pages and
6.76 in one copy, 86% of its own). E0's GPU-hit slope, the host's work on a prompt, is half the box's (1.80 against 3.79
µs a token, by least squares), as Python's time is 0.60 of it. Placed on this host
([`forecast-80gb-on-host.md`](../benchmarks/a100/forecast-80gb-on-host.md), `make forecast-a100 GPU=a100-80gb
HOST_JSON=...`), the forecast's E0 cells come within 9% of the measured GPU hits and 10% of the CPU restores at the
mean, from 69% and 22% off before the run; recompute, which the host does not set, stays 17% off. The host leaves
the running share where it was: placed on it, the forecast gives the core 11% and A' 18%, against 6% and 6.5%
measured, so the host is not what the simulator misses there. It overstates the box's too: 17% against the 8.1%
RS measured on the same samples ([RS](BENCHMARKS.md#the-boxs-running-share-rs)).

## The kit, rehearsed on the box

[`benchmarks/a100/rehearsal`](../benchmarks/a100/rehearsal/README.md) runs the kit's every step on Phastform, the box's host standing in for the A100's, in a home of its own under flatshare-svc beside the box's deployment: `push.sh`, `bootstrap.sh stage`, `gpu` and `warm`, `run.sh` over block 0, block 1's probes and block 4's first point with the deadline's stop, `pull.sh` and `make reproduce-a100`, then a check that the box's deployment is as it found it.

On 2026-10-25 ([the records](../benchmarks/results/2026-10-25-a100-rehearsal/README.md)) it found two defects of the kit before the A100's hours would have: run as a script, `deploy/profile.py` stood in for the standard library's `profile` module when vLLM's import reached cProfile, which failed `bootstrap.sh gpu`; and under flatshare-svc, `gpu-guard` took an engine core for a foreign process once its title had blanked its /proc/PID/environ, so `warm` started no engine (the listener audit read the same environment).

The release's kit then ran every step in one pass from a fresh home on 2026-10-26 ([its records](../benchmarks/results/2026-10-26-a100-rehearsal/README.md)): 76 minutes, the GPU in use for 64 of them, 29 of those in `warm`, whose first start from empty caches took 20.5. Through the kit the box measured what it measures through its own deployment: the KV case's first token after 4,853–4,929 ms (Gray's: 4,923–5,077 ms), and the PRs' 48-token configuration at a V2V p95 of 2,333 ms (the box's two-host deployment: 2,132–2,397 ms over W5's fifteen runs), the control plane on the same host's 16 CPUs.

## The Marlin dry run on the box

Two sessions on Phastform tested the A100's code path before the A100 run: the FP8 checkpoint through `MarlinFP8ScaledMMLinearKernel` (W8A16), FlashAttention 2 on LBHNC, the CPU tier, make-room and the harness, on a real GPU. They measured function, not the A100's magnitudes: the box runs Marlin and BF16 at 47.4 TFLOPS with FP32 accumulation, so its linear layers on Marlin take at least about 153 µs a token against DeepGEMM's fitted 48 [derived].

- [`marlin-session1.sh`](../benchmarks/a100/marlin-session1.sh), about 25 minutes of GPU time: E0 on the frozen engine (DeepGEMM), then on Marlin (`--linear-backend marlin`, checked by the engine log and `/server_info`, with `smoke.sh`), then make-room's KV and prefill cases and the tail on Marlin, and the frozen engine back. `--fallback` selects Marlin by disabling the four block-FP8 kernels ahead of it (`VLLM_DISABLED_KERNELS`) instead of the flag.
- [`marlin-session2.sh`](../benchmarks/a100/marlin-session2.sh), about 30 minutes: [`bench_marlin.py`](../benchmarks/a100/bench_marlin.py) (Marlin, BF16 and DeepGEMM at Qwen3-4B's four linear shapes, M 1 to 1,024: Marlin's factor over BF16 on the box, one data point beside PR #61224's 1.42 on an L4, which does not say how it timed its GEMMs), the probes at a 3 GiB pool and a 16,384-token context (which size themselves from it), and one E2 point on Marlin ([`marlin-dry.campaign`](../benchmarks/a100/marlin-dry.campaign)), without the KV sampler, as W5's points it is read against ran.

`bench_marlin.py` gives each call's GPU time as the call runs inside a CUDA graph, which is where vLLM 0.32.0 runs these layers: it captures every step of up to 512 tokens by default, the box's step cap. Ten calls, each after an L2 flush, are captured in one graph and ten flushes in another; each graph is replayed between two CUDA events, and one call's time is the calls' median replay less the flushes', over ten. A call timed alone between two events measures the host's dispatch of its kernels instead, which the GPU waits for: the first bench timed that way, and session 2 keeps its `gemm.csv` as `gemm-events.csv` and runs the GEMMs again. Marlin's calls share one lock workspace, as the engine's do. The bench runs in the engine unit's environment (`deploy/a100/host.py with-engine-env deploy/phastform ...`), because DeepGEMM builds its kernels with nvcc at their first call and needs the unit's `CUDA_HOME` (CUDA 13.0 on the box; without it DeepGEMM takes `/usr/local/cuda`'s older nvcc, and its rows say "DeepGEMM needs nvcc: set CUDA_HOME").

Both run from the workstation with `FLATSHARE_SSH`, `FLATSHARE_GPU` and `FLATSHARE_CYAN` as for `campaign.sh`, keep each step's outputs on its host, skip a step done before, and copy the outputs to `FLATSHARE_OUT` (`~/flatshare/hosts/marlin-dry`).

### What it measured (2026-10-25)

Session 1 ran on the box's deployment of W5 (its servers on TCP; the same steps) and session 2's GEMM step on the corrected `bench_marlin.py`; then session 2 ran in full on the deployment of W6 (v0.3-191-g987fd09, the servers on sockets), its GEMM step giving each M's sum within 0.005 of the first run's. The records are in [`benchmarks/results/2026-10-25-marlin-dry`](../benchmarks/results/2026-10-25-marlin-dry).

- **The A100's path runs on a real GPU.** With `--linear-backend marlin` the engine selected `MarlinFP8ScaledMMLinearKernel` for the FP8 checkpoint, with FLASH_ATTN, FlashAttention 2 and the LBHNC layout; `/server_info` named `marlin`, and `smoke.sh` passed.
- **E0 on Marlin.** A prefill's per-token term rose from 65.4 µs on DeepGEMM to 175.7 µs, at the box's BF16 rate (at least about 153 µs a token for the linear layers, above); the GPU-pool hit (3.34 against 3.88 µs a token) and the CPU-tier restore (27.11 against 27.01 µs a token) did not move with the linear kernel.
- **make-room on Marlin, at the PRs' values.** The KV case's priority-0 first token came after 299 and 370 ms (W5's probes on DeepGEMM: 219–290 ms); in the prefill case the reply ran about 16 ms a token beside three 10,000-token prefills; the calls' tail over 240 s beside 12 agents: 73 turns, first-token p95 432 ms, none failed.
- **Marlin against BF16,** summed over Qwen3-4B's four linear shapes, each call inside a CUDA graph: 0.50 to 0.56 of BF16's time for M of 1 to 64, 0.97 at 128, 1.08 to 1.09 at 256 to 1,024. Weight-only FP8 halves the bytes of a memory-bound GEMV and costs about 9% where the GEMM is compute-bound and both run at the BF16 rate; DeepGEMM's W8A8, at the box's FP8 rate, takes 0.34 of BF16's time at M of 1,024. The forecast's A100 term was 29.2 µs a token times 1.0 to 1.42, and the box's 1.08 to 1.09 one point near the range's low end; the A100's calibration point then measured 41.36 µs a token, the range's high end ([what it measured](#what-it-measured-2026-10-26)).
- **The probes at a 3 GiB pool.** At 1,364 blocks and a 16,384-token context the probes sized their cases from the pool: the KV case's two background requests of 10,719 tokens left 22 blocks free, and the prefill case ran three of 6,000. make-room's priority-0 first token came after 173 and 248 ms, and the calls' tail over 120 s beside 13 agents ran 36 turns, first-token p95 332 ms, none failed.
- **E2 on Marlin.** The PRs' 48-token configuration in the discriminating scenario, on W5's round-1 sample of the agents, against FB's round 1 on DeepGEMM, the same code and sample ([`2026-10-25-e2-fb`](../benchmarks/results/2026-10-25-e2-fb)): the voice turn's p95 2,240 ms against 2,313 (W5's fifteen runs: 2,132–2,397 ms), none of 50 turns failed; its p50 2,028 against 1,883 ms, with the reply's first token 83 ms later at the median (348 against 265 ms). The calls' prompts were 96% cached, so that time is the agents' prefill in the steps a call joins, whose tokens take 2.7 times as long on Marlin (E0, above). The agents' tokens a second fell 28%, from 3,416 to 2,457. On the box Marlin and BF16 run at one rate, so the point shows the A100's path working end to end and the direction of its cost, not the A100's magnitudes.
