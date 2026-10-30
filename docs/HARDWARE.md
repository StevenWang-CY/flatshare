# Running Flatshare on other hardware

This page is for an operator who moves the deployment to another GPU or to a single host, where it runs as a measurement kit and not as a deployment: nothing of it starts at boot, and releases, rollback and the release's serving check reach only this lab's hosts, phastform, gray and cyan ([PRODUCTION.md](PRODUCTION.md#checking-it-on-your-hardware)). It says what the deployment needs from a GPU node, how a node's files come from a hardware profile, what changes on a card or a host that is not the box's, and how one host runs both the GPU servers and the control plane.

Every measurement before 2026-10-26 ran on one RTX 5060 Ti 16 GB, the frozen box, with Gray, a second one, for the first make-room probes; that day an A100 SXM4 80 GB on Google Colab ran E2's core like for like ([what it measured](BENCHMARKS.md#what-it-measured-2026-10-26)). How the A100 run was laid out and what it measured are in [BENCHMARKS.md](BENCHMARKS.md#the-a100-run-2026-10-26).

A number here is measured (on the box, Gray or the A100, with its source), modeled (flatshare-sim's forecasts, [`forecast.md`](../benchmarks/a100/forecast.md) for the A100 40 GB and [`forecast-80gb.md`](../benchmarks/a100/forecast-80gb.md) for the A100 SXM4 80 GB) or an estimate, and says which.

## What runs where

What vLLM 0.32.0 runs for the box's checkpoint, Qwen3-4B-Instruct-2507-FP8 (block-scaled FP8 weights), by the card's compute capability, from vLLM's kernel selection order and its attention backends' priorities, and how far each is checked:

| Cards (compute capability) | Linear layers | Attention, KV layout | Status |
|---|---|---|---|
| RTX 5060 Ti (sm120): the box, Gray | FP8 W8A8 (DeepGEMM) | FlashAttention 2, LBHNC | **measured**: E0, E1's calibration, E2, the make-room probes |
| A100 40 GB or 80 GB, SXM4 or PCIe (sm80) | Marlin W8A16: the FP8 weights, BF16 activations, the BF16 tensor rate | FlashAttention 2, LBHNC | **measured** on an A100 SXM4 80 GB: E0, the step model, the make-room probes, E2 ([the A100 run](BENCHMARKS.md#what-it-measured-2026-10-26)); the 40 GB **modeled** ([the forecast](../benchmarks/a100/forecast.md)); the Marlin path was first run on the box ([its dry run](BENCHMARKS.md#the-marlin-dry-run-on-the-box)) |
| A10 (sm86), L4, L40S, RTX 4090 (sm89) | Marlin W8A16, as on the A100 (vLLM 0.32.0 sends this checkpoint to Marlin on sm89 too) | FlashAttention 2 | **modeled** (`flatshare-sim hardware`) |
| H100, H200 (sm90), B200 (sm100) | FP8 W8A8 | FlashAttention 3 (sm90), FlashInfer (sm100) | **modeled** |
| T4 (sm75) | Marlin with FP16 activations | Triton attention | **modeled**; the preflight labels both |
| Older than sm75, CPU-only, ROCm | — | — | not supported by this deployment: vLLM 0.32.0 needs sm75, and the CPU KV tier's platform differs off CUDA |

The KV layout follows the CPU tier's connector, not the card: vLLM's `OffloadingConnector` requires LBHNC, so every CUDA card with the tier runs it. What changes in what the results mean is in [the preflight](#the-preflight): within one card, tuned vLLM against make-room compares as on the box; across cards, only in the regime's coordinates and each card's distance to its own SLO.

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

  On the box the three take 0.92 of the card, and the worst cases leave 12,250 MiB free against the engine's 11,096 and 4,360 against the STT's 2,378, the engine holding 10,388 MiB (vLLM's total of 15,852 MiB less the servers' footprints of 2026-10-25) [measured]; on a 40 GiB card they take 0.37. The 14B model of the optional arm C' ([the A100 run's arms](BENCHMARKS.md#the-optional-arms)) outgrows its budget, its pool sized to fill the card up to the box's headroom [modeled], so the sum does not cover it, and its node starts the engine after the audio servers instead (the profile's `engine_order`): each audio server checks free memory before the 14B's allocation, and the engine's own check sees them resident.
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

None of these differs between tuned vLLM and make-room on one host, so a single host's comparisons are made within it.

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

`stage` takes uv and its own Python 3.12 (with its headers, where uv would prefer the system's), then:

1. renders the node into `~/flatshare/node`, from the profile's own card when the GPU is not yet visible;
2. builds the engine, TTS, voice and bench venvs from that Python, the engine's lock by R8;
3. fetches Kokoro and the models at their pinned revisions (`FLATSHARE_MODELS=MODEL@REVISION` adds one, the 14B for C') into the Hugging Face hub cache the operator's environment names: `HF_HUB_CACHE`, or `HF_HOME`'s, a data volume's say, resolved once, at the first stage, kept in `env.sh` and named in the engine's and the STT's units, which load offline;
4. installs LiveKit, and builds `flatshare-gateway` when a copied binary does not run;
5. links the units with this host's drop-ins;
6. writes the control plane's configuration (`deploy/a100/host.py config`: new keys, gateway.toml from cyan's example, `e2-host.toml` for `flatshare-bench e2 run`) and starts the control plane;
7. builds FlashInfer's sampler for the card's architecture into the engine's own workspace (`host.py jit-check`), which checks the compiler before a GPU minute is spent.

`run.sh` then runs the core's blocks, which [BENCHMARKS.md](BENCHMARKS.md#the-core-block-by-block) lists with what each took on the A100.

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

<a id="the-a100-run"></a>

<a id="the-core-block-by-block"></a>

<a id="the-optional-arms"></a>

<a id="the-analysis"></a>

<a id="on-google-colab"></a>

<a id="what-it-measured-2026-10-26"></a>

<a id="the-kit-rehearsed-on-the-box"></a>

<a id="the-marlin-dry-run-on-the-box"></a>

<a id="what-it-measured-2026-10-25"></a>

## The A100 run and the runs that prepared it

These are measurements, so they are in BENCHMARKS.md with every other run:
- [the A100 run of 2026-10-26](BENCHMARKS.md#the-a100-run-2026-10-26): [what it measured](BENCHMARKS.md#what-it-measured-2026-10-26), and [how it was laid out](BENCHMARKS.md#how-the-run-was-laid-out), with [the core, block by block](BENCHMARKS.md#the-core-block-by-block), [the optional arms](BENCHMARKS.md#the-optional-arms), [the analysis](BENCHMARKS.md#the-analysis) and [the Colab runtime](BENCHMARKS.md#on-google-colab);
- [the kit, rehearsed on the box](BENCHMARKS.md#the-kit-rehearsed-on-the-box);
- [the Marlin dry run on the box](BENCHMARKS.md#the-marlin-dry-run-on-the-box), and [what it measured](BENCHMARKS.md#what-it-measured-2026-10-25).
