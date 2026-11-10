# Running Flatshare on other hardware

This page is for an operator who moves the deployment to another GPU or to a single host. Off this lab's hosts,
Phastform, Gray and cyan, the deployment is a measurement kit. Nothing of it starts at boot, and releases, rollback and
the serving check do not reach it ([PRODUCTION.md](PRODUCTION.md#checking-it-on-your-hardware)). Each number here is
marked as measured, as modeled by flatshare-sim's forecasts for [the A100 40 GB](../benchmarks/a100/forecast.md) and
[the 80 GB](../benchmarks/a100/forecast-80gb.md), or as an estimate.

## What runs where

What vLLM 0.32.0 runs for the box's checkpoint, Qwen3-4B-Instruct-2507-FP8, on each card, by vLLM's kernel selection
and its attention backends' priorities:

| Cards (compute capability) | Linear layers | Attention, KV layout | Status |
|---|---|---|---|
| RTX 5060 Ti (sm120): the box, Gray | FP8 W8A8 (DeepGEMM) | FlashAttention 2, LBHNC | **Measured**: E0, E1's calibration, E2, the make-room probes |
| A100 40 GB or 80 GB, SXM4 or PCIe (sm80) | Marlin W8A16: the FP8 weights, BF16 activations, the BF16 tensor rate | FlashAttention 2, LBHNC | **Measured** on an A100 SXM4 80 GB: E0, the step model, the probes and E2 ([the A100 run](BENCHMARKS.md#do-the-rules-carry-to-a-datacenter-card)); the 40 GB **modeled** |
| A10 (sm86), L4, L40S, RTX 4090 (sm89) | Marlin W8A16, as on the A100 (vLLM 0.32.0 sends this checkpoint to Marlin on sm89 too) | FlashAttention 2 | **Modeled** (`flatshare-sim hardware`) |
| H100, H200 (sm90), B200 (sm100) | FP8 W8A8 | FlashAttention 3 (sm90), FlashInfer (sm100) | **Modeled** |
| T4 (sm75) | Marlin with FP16 activations | Triton attention | **Modeled**; the preflight labels both |
| Older than sm75, CPU only, ROCm | — | — | Not supported: vLLM 0.32.0 needs sm75, and the CPU tier's platform differs off CUDA |

The KV layout is LBHNC on every card with the CPU tier, whose `OffloadingConnector` requires it. Within one card, tuned
vLLM and make-room compare as on the box, and across cards only in the regime's coordinates and each card's distance to
its own SLO.

The CPU tier's loads follow the card's link. On the box's PCIe 3.0 x8 link a load on the copy engine holds a concurrent
step's input copy until the load ends, and the slowest sampled step took 192–209 ms beside a 1.46 GB load. The serving
profile therefore loads the tier through vLLM's Triton copy kernel (`sm_load_programs = 4`), at 6.1 ms a step against
5.7 ms alone ([the probes](../benchmarks/results/2026-11-08-envelope/README.md#the-voice-tail-the-copy-engine-on-pcie-30-x8)).
No other link was measured, and on another card the record's probes say which path to take.

## What to ask for at handover

The preflight finds the host's facts, but these decide the plan, and `bootstrap.sh` cannot find out the first three.

1. **Access.** The SSH user and key, and whether the host has a public address and a firewall that drops inbound UDP.
2. **Container and root.** A container's `/dev/shm` must hold the CPU tier, so ask for `--shm-size` or `--ipc=host`.
   Root or sudo is needed for apt, the user-namespace sysctl and a compute-mode change.
3. **The GPU.** Which one is ours, by its index or UUID in `nvidia-smi -L`, its form factor, MIG off and the Default
   compute mode.
4. **The driver's version**, which picks the install path (R8), and in a container where the driver's libraries are if
   `ldconfig -p` does not list `libcuda.so.1`.
5. **Time before the billed clock** for `stage` and `warm`. `warm` alone took 29 minutes on the box and 9 on the A100,
   and the core 3 hours 13 minutes on the A100 [measured].
6. **A home on a volume that outlives restarts**, since the compile caches and every record live there until
   `pull.sh` copies them.
7. **The OS image**, with systemd and lingering, and whether unprivileged user namespaces are allowed.
8. **CPUs, RAM and disk.** 34 CPUs, 17 at least (R7), and RAM for the 20 GiB CPU tier (R3). The disk holds the engine
   venv (6.1 GB, or 12 GB on the cu129 lock) and CUDA 12.9's toolkit (9.4 GB) [measured], and the models (about 7 GB)
   [estimate].
9. **Egress** to PyPI, download.pytorch.org, GitHub, Hugging Face and NVIDIA's download site.

## Running on another GPU, step by step

From the workstation, with the repository at the revision to run and the corpora and the callers' audio
(`weka-x32`, `weka-x32-long`, `callers`) in `FLATSHARE_DATA`:

```sh
deploy/a100/push.sh USER@HOST            # the source tree (with DEPLOYED), the data as files;
                                         # FLATSHARE_GATEWAY_BIN to bring a flatshare-gateway
```

On the host, from the home directory:

```sh
bash flatshare/src/deploy/a100/bootstrap.sh stage     # no GPU needed: before the billed
                                                      # clock if the host is up earlier
bash flatshare/src/deploy/a100/bootstrap.sh gpu       # with the card (the GPU units
                                                      # stopped): the driver, the install
                                                      # path, vLLM's parse, the preflight
bash flatshare/src/deploy/a100/bootstrap.sh warm      # one start per compile configuration,
                                                      # so the campaign's are warm
nohup setsid bash flatshare/src/benchmarks/a100/run.sh --until HH:MM \
  >>flatshare/runs/a100/run.log 2>&1 </dev/null &  # the A100 run's core, block by block
```

On a host with several GPUs, `FLATSHARE_GPU_ID=INDEX|UUID` names the lent one at `stage`, and
`FLATSHARE_SUPERVISOR=flatshare-svc` chooses flatshare-svc even where a user manager could. `env.sh` keeps both.

`stage` installs uv and its own Python 3.12, and then:

1. renders the node into `~/flatshare/node`;
2. builds the venvs, the engine's from the lock R8 chooses;
3. fetches the models at their pinned revisions into the Hugging Face hub cache (`HF_HUB_CACHE` or `HF_HOME`);
4. installs LiveKit, and builds `flatshare-gateway` unless a copied binary runs;
5. links the units, and writes and starts the control plane;
6. builds FlashInfer's sampler for the card, which checks the compiler before a GPU minute is spent.

`run.sh` then runs the core's blocks, timed in
[the A100 run's record](../benchmarks/results/2026-10-26-a100/README.md#how-it-ran). Meanwhile
`deploy/a100/pull.sh USER@HOST` on the workstation copies the records, and afterwards
`make reproduce-a100 RECORDS=~/flatshare/hosts/HOST CORPUS=<weka-x32>` reads them. After a reboot, start `run.sh` again,
and it resumes at the block it stopped in.

<a id="the-a100-run"></a>

<a id="what-it-measured-2026-10-26"></a>

<a id="the-marlin-dry-run-on-the-box"></a>

On 2026-10-26 an A100 SXM4 80 GB on Google Colab ran E2's core like for like with this kit.
[BENCHMARKS.md](BENCHMARKS.md#do-the-rules-carry-to-a-datacenter-card) gives what it measured, and
[its record](../benchmarks/results/2026-10-26-a100/README.md) each block and the host. Before it,
[a rehearsal](../benchmarks/results/2026-10-26-a100-rehearsal/README.md) ran the kit end to end on the box, as
[benchmarks/a100/rehearsal](../benchmarks/a100/rehearsal/README.md) describes, and
[a dry run](../benchmarks/results/2026-10-25-marlin-dry/README.md) took the checkpoint through Marlin there.

## The hardware profile

A node's deployment is a function of the card, the host, the model and [the rules](#the-rules).
[`deploy/profiles/`](../deploy/profiles) holds one profile per node type, and [`deploy/profile.py`](../deploy/profile.py),
which needs Python 3.11's standard library alone, writes a node's directory from one.

```sh
# the host's facts: nvidia-smi, /sys, /proc, systemd, toolkits
uv run --no-project --python 3.12 deploy/profile.py detect --out host.json
uv run --no-project --python 3.12 deploy/profile.py render a100-40gb --host host.json \
  --out ~/flatshare/node
# the box's profile renders deploy/phastform byte for byte
uv run --no-project --python 3.12 deploy/profile.py check rtx-5060-ti
# on a GPU host: vLLM's own parser, no engine
~/flatshare/venvs/engine/bin/python deploy/profile.py check-vllm ~/flatshare/node
```

`detect` records the card, the CPUs and RAM by NUMA node, `/dev/shm`, the container, root, systemd, user namespaces,
the toolkits, the driver's library directory and the hub cache. `render` fills [`deploy/templates/`](../deploy/templates),
the box's files with placeholders, from the profile and, where it says "auto", from the host's facts. It writes the
node's configurations, units and drop-ins, and `node.json`, which keeps every resolved value and the warnings. `check`
prints each rendered file's difference from its repository copy, `update` writes it there, and `effective` prints a
serving profile's `serving.json` ([the serving profile](../deploy/phastform/variants/README.md#the-serving-profile)).

- **The box's profile** (`rtx-5060-ti`) holds the box's values alone, and a test holds its rendering to
  `deploy/phastform`, so no measurement changes meaning. `make profiles` checks it and its serving profile, and
  `make profiles UPDATE=1` writes both.
- **The A100's profile** (`a100-40gb`) keeps the box's model, pool, step cap and CPU tier ratio, and derives the rest
  from the host.
- **A profile with a base** takes the base's values under its own, key by key. `a100-40gb-qwen3-14b`, the optional arm
  C', changes the A100 profile's model, pool and directory.

## The rules

The profile's "auto" values, in `deploy/profile.py`:

- **R1, the pool.** Like for like, the box's 36,400 tokens (5 GiB of Qwen3-4B's KV), so that a card runs the box's
  comparison. Or the card's own pool, what is left for KV in whole blocks once the card keeps what the box keeps beside
  its pool [modeled].
- **R2, the memory budgets.** vLLM refuses to start when free memory is below `gpu-memory-utilization` × the card's
  memory, so the box's budgets port in GiB, 10.83 for the engine and 2.32 for Whisper, whose pool stays at 1,400 blocks.
  `profile.py` refuses a profile whose two budgets and the TTS's 1.08 GiB take more than 0.95 of the card. With both
  vLLM pools fixed, the three servers start at once, and each one's memory check passes in any order while the budgets
  fit the card together. On the box they take 0.92 of it, and the worst cases leave 12,250 MiB free against the
  engine's 11,096 and 4,360 against the STT's 2,378 [measured]. C''s 14B engine outgrows its budget [modeled], so its
  node starts the engine last (`engine_order`).
- **R3, the CPU tier.** The box's ratio, four times the pool, bounded by what `/dev/shm` holds less 2 GiB and by the
  RAM a single host leaves after 25 GiB for its other processes [estimate], in whole GiB. A tier no larger than the pool
  only mirrors it, so R3 refuses one.
- **R4, the step cap.** The core keeps the box's 512/256. The time-equivalent cap is the largest multiple of 256 whose
  chunk takes no longer than the box's 512-token chunk at the scenario's median context, with half of it as the
  long-prefill threshold. On the A100 it is 1,536, central (2,048 low, 1,280 high) [modeled].
- **R5, the probes.** `probe.py` and `tail.py` size their requests from the engine's pool and context. On the box that
  is two KV-case requests of 17,999 tokens, three prefill-case requests of 10,000 and 12 tail agents.
- **R6, the scenario.** E2's corpus, its two callers and the sessions for the box's KV pressure. Another load or model
  is an optional arm.
- **R7, the core plan.** On one host the engine, the audio servers, the gateway, the voice plane, AIPerf and the callers
  share the CPUs 8:4:2:8:8:4, and 34 give each its want, the engine on the GPU's NUMA node. Below 17 each gets half,
  and the plan warns that the host checks the mechanics, not the numbers.
- **R8, the install path.** The box's lock (vLLM 0.32.0, torch 2.13.0+cu130) where the driver runs CUDA 13.0, from R580
  or with `cuda-compat-13-0` on a data-center card with R535 or R570. On any other driver from R525, the cu129 lock
  ([`deploy/a100/requirements`](../deploy/a100/requirements)), which changes only vLLM's and torch's CUDA builds.
  Older drivers are refused. FlashInfer's JIT uses the host's CUDA toolkit or one `bootstrap.sh` installs without root.

## The preflight

`deploy/a100/host.py preflight NODE [--gpu]` checks what a campaign depends on and writes `NODE/preflight.json`. Each
`bootstrap.sh` stage ends with it and fails on a refused check, and each point's `run.json` keeps every check that was
not ok. A check that needs the card or the engine's first start is pending until then.

| Check | Degraded, and what it changes | Refused |
|---|---|---|
| Compute capability and precision path | sm75–sm89: Marlin's W8A16 at the BF16 rate, so cards compare only in regime coordinates (note 1) | Below sm75 |
| BF16 | sm75: FP16 activations, whose numerics differ from the box | — |
| Attention backend and KV layout | Not FlashAttention or FlashInfer on LBHNC: attention's cost | — |
| Driver and install path | cu130 through the compatibility libraries, or the cu129 lock: another CUDA runtime under the same vLLM and torch | The venv's build is not the driver's, or a GPU unit cannot reach its libraries (note 2) |
| torch on the GPU (`--gpu`) | — | `torch.cuda.init()` fails in the engine's environment |
| FlashInfer's JIT | — | Its sampler does not build, or was built for another compiler or card (note 3) |
| Host compilers | — | No `cc` or `c++`, which Triton and `nvcc` need |
| Python headers | — | The engine venv's Python has no `Python.h` for Triton (note 4) |
| Compile and JIT caches | Cold, on tmpfs, or under 10 GiB free [estimate] (note 5) | Under 2 GiB free |
| CPU tier and `/dev/shm` | Below four times the pool: fewer restores, more recompute | Another tier than `stage` rendered (note 6) |
| Pinned memory | `cudaHostRegister` failed: a pageable tier, slower restores | — |
| Host RAM | The GPU's NUMA node short of the tier: a remote tier, within 1% of pinned bandwidth on the box | — |
| Async scheduling, block size | Synchronous scheduling, or another block size, to which make-room's token counts round | — |
| Engine starts | — | A start failed (`warm.json` keeps its log lines) |
| GPU | — | Several GPUs and none named by `FLATSHARE_GPU_ID` |
| GPU exclusive (`--gpu`) | — | Another process holds the lent GPU (`deploy/bin/gpu-guard`) |
| Compute mode | — | Anything but Default (note 7) |
| MIG | One slice with `FLATSHARE_ALLOW_MIG=1`, a smaller card on which the forecast does not hold | MIG on otherwise |
| systemd `--user` and lingering | None, so flatshare-svc runs the units, which changes no measurement | Lingering off under the systemd an earlier stage chose |
| Socket paths | — | The runtime directory (`%t`) too long for vLLM's ZMQ sockets (note 8) |
| One supervisor | — | Units under the other supervisor too, a split control plane |
| User namespaces | Refused by the kernel, so the voice plane and the GPU servers share the host's network namespace ([below](#single-host-mode)) | — |
| Listener audit | — | No `ss` (iproute2), which [`audit-listeners`](../deploy/bin/audit-listeners) reads |
| Cores | Below 17: R7's warning | — |
| Control plane, models, corpus and callers, binaries, disk | — | Down or missing (note 9) |

1. The row names the kernel the engine's first start selected, or before that start the one expected.
2. They are in a directory the linker does not search, and no GPU unit names it on `LD_LIBRARY_PATH`.
3. `VLLM_USE_FLASHINFER_SAMPLER=0` in a drop-in is the degraded way out. A sampler staged before the card was visible is
   built again at the `gpu` stage.
4. Ubuntu 24.04's Python 3.12 has none without its `-dev` package, so `stage` makes the venvs from uv's Python.
5. Each compile configuration's first start compiles, 531.8 s on the box with cold caches against 88.8 s warm,
   measured in W8's start drills.
6. Something at the `gpu` stage holds `/dev/shm` or the RAM that bounds the tier (R3), and the core would run that
   smaller tier all campaign.
7. `Exclusive_Process`, common on shared HPC nodes, allows one CUDA context, so the three servers cannot all run. The
   way out is `nvidia-smi -c DEFAULT` with root, or the `mps` variant, which is not the core's configuration.
8. A socket's path may have 107 bytes. Under flatshare-svc without `XDG_RUNTIME_DIR`, `%t` is
   `~/.local/state/flatshare-svc/run`, which leaves a home directory 18 characters.
9. The models are checked in the hub cache the facts recorded. A copy of `long_traces.py`'s links does not resolve on
   another host, which is why `push.sh` copies them as files.

## Single-host mode

A host with no control host beside it runs both the GPU servers and cyan's control plane, on its own Unix sockets and
loopback, each on its core plan's cores. Nothing below differs between tuned vLLM and make-room, so a single host's
comparisons are made within it.

- **No tunnel.** The gateway and the voice worker reach the GPU servers' own sockets in `$XDG_RUNTIME_DIR/flatshare`.
  AIPerf and `vllm bench serve` speak TCP only, so the calibration point and the batch arm reach the engine through
  [`deploy/bin/flatshare-forward`](../deploy/bin/flatshare-forward).
- **One directory of units.** The control plane's units share `~/.config/systemd/user` with the GPU node's.
  `variant.sh` switches only the GPU units' drop-ins, and `flatshare-bench e2 run` owns the control plane's.
- **One GPU of several.** The lent GPU, named once by `FLATSHARE_GPU_ID`, is the only one the GPU units, the facts,
  `gpu-guard` and the sampler see, so another tenant's work on the others enters no point's clocks.
- **No systemd user manager.** In a container, without lingering, or with `FLATSHARE_SUPERVISOR=flatshare-svc`,
  [flatshare-svc](#flatshare-svc) runs the same units. The first stage chooses the supervisor and `env.sh` keeps it.
- **No user namespaces**, as on Ubuntu 24.04 by default. The voice plane and the GPU servers then share the host's
  network namespace, where the WebRTC clients gather candidates on its interfaces, so the preflight reports the mode as
  degraded and names the public addresses, which must drop inbound UDP. `smoke.sh`'s listener audit accepts there the
  voice plane's sockets and the GPU servers' loopback engine-core listeners and CUDA abstract sockets, which any local
  user can reach, and refuses anything else of ours.
- **Driver libraries outside the linker's search.** Colab's container keeps `libcuda.so.1` in `/usr/lib64-nvidia`,
  which only its shells' `LD_LIBRARY_PATH` reaches. The facts find the directory, and every GPU unit names it on its own
  `LD_LIBRARY_PATH`.
- **The campaign runs on the host** (`benchmarks/e2/campaign.sh --local`), so a dropped workstation link stops nothing.
  It opens vLLM's development routes before each point, as the box's measurements do
  ([`deploy/phastform/measure`](../deploy/phastform/measure)), and [`deploy/a100/pull.sh`](../deploy/a100/pull.sh)
  copies the records to the workstation every two minutes.

### flatshare-svc

[`deploy/bin/flatshare-svc`](../deploy/bin/flatshare-svc) runs the repository's systemd user units where there is no
user manager. It needs Python's standard library alone, from 3.10, and with `deploy/bin/shims` first on `PATH` it
answers as `systemctl` and `journalctl`, so the scripts work unchanged. At every command it reads each unit and its
drop-ins from `~/.config/systemd/user`, and it implements the directives the units use as systemd does:

- `Type=notify`, `exec` and `simple`;
- `ExecStartPre`, `ExecStart`, `ExecStartPost`, `ExecStop` and `ExecStopPost`, where an empty assignment resets the
  list and a leading `-` ignores a failure, and `Environment`, `EnvironmentFile` and `WorkingDirectory`;
- `CPUAffinity`, and `NUMAPolicy=preferred` with `NUMAMask`, through `taskset` and `numactl` where the host has them;
- `KillSignal`, `TimeoutStartSec`, `TimeoutStopSec`, `Restart`, `RestartSec`, `StartLimitIntervalSec` and
  `StartLimitBurst`;
- `Wants`, `Requires`, `BindsTo`, `PartOf` and `After`, where a missing unit that `Wants=` names is left out;
- a socket unit, which starts and stops as a target does and holds no listener;
- the specifiers `%n`, `%N`, `%h`, `%t`, `%u`, `%U` and `%%`, and `$NAME` and `${NAME}`.

Where it differs:

- **A start** refuses a unit with any other specifier before anything runs. It starts a target's units one at a time,
  so the GPU servers start STT, TTS, then the engine, where systemd starts the three at once.
- **Each unit has a runner** (`flatshare-svc run UNIT`) that runs its commands, writes its journal and restarts it as
  `Restart` says. Its processes carry `FLATSHARE_SVC_UNIT=UNIT` where systemd would use a cgroup.
- **A stop** runs `ExecStop`, signals the main process's group, kills what remains after `TimeoutStopSec` and runs
  `ExecStopPost`, and returns once every process it stops has exited. A main process that exits on its own has the rest
  of its group ended the same way, so an engine core cannot outlive its server and hold the GPU.
- **Orphans.** On Linux each runner is the subreaper of its unit's processes, since a container's init may reap none.
- **State** is in `$XDG_STATE_HOME/flatshare-svc`, by default `~/.local/state/flatshare-svc`. Each pid is kept with its
  boot and start time, so a state file from before a reboot reads as a unit that is not running.

## A GPU node without user namespaces (Gray)

Gray, the second RTX 5060 Ti, runs Ubuntu 24.04 with `kernel.apparmor_restrict_unprivileged_userns=1` and no root to
lift it, so `in-private-net` cannot give a GPU server a network namespace of its own. Gray's install handles this at
every release.

- **Found at every install.** [`deploy/bin/private-net-check`](../deploy/bin/private-net-check) tries what
  `in-private-net` does. Where the kernel refuses, [`deploy/gray/install.sh`](../deploy/gray/install.sh) links
  [`private-net-host.conf`](../deploy/gray/private-net-host.conf) into the three GPU servers' drop-ins, which puts them
  in the host's network namespace. Where it allows, the install removes the drop-in, and where the check cannot tell, it
  changes nothing.
- **What the servers expose.** Each still serves only its Unix socket (mode 0700), but the host's other users can reach
  the engine cores' loopback listeners and CUDA's abstract sockets. The listener audit accepts those and nothing else.
- **Recorded and checked by each release.** The manifest's `isolation` records where each server runs and why
  ([the runbook](../ops/RUNBOOK.md#a-gpu-node-without-user-namespaces-gray)). A release fails before its switch if a
  server would ask for a namespace the kernel refuses.
- **Undone the same way.** Allowing the namespaces (`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`)
  and the next install remove the drop-in.

None of it changes how a server schedules or how fast it answers, so a point on Gray compares with one on Phastform as
one card with another.
