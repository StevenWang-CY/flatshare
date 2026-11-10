# The A100 run, rehearsed on the box

`rehearse.sh` runs the A100 run's kit on the box, Phastform, before a rented GPU's billed hours, for an operator who
wants every step of [HARDWARE.md's run on another GPU](../../../docs/HARDWARE.md#running-on-another-gpu-step-by-step)
proven first. One host acts as both GPU node and control host. From the workstation the rehearsal pushes the kit and
runs `bootstrap.sh`'s three stages. It then runs a subset of the A100 run that crosses each kind of step once: block 0,
block 1's probes, and block 4's first point with the deadline's stop. Last it pulls the records and reproduces the
report from them ([the steps](#the-steps-and-their-checks)).

All of it runs under flatshare-svc in a home of its own, `~/a100-rehearsal`, beside the box's deployment, which shares
the account. The rehearsal leaves that deployment as it found it, and checks that it did. On 2026-10-26 the release's
kit ran every step in one pass from a fresh home, in 76 minutes, 64 of them with the GPU in use
([its record](../../results/2026-10-26-a100-rehearsal/README.md)). The A100 run followed the same day
([what it measured](../../../docs/BENCHMARKS.md#do-the-rules-carry-to-a-datacenter-card)).

Two features of the kit exist for the rehearsal, and `check` refuses a checkout without them:

- **`deploy/bin/in-private-net`** runs each GPU server in a network namespace of its own, so that block 0's listener
  audit passes under flatshare-svc ([below](#block-0s-listener-audit)).
- **`run.sh --estimate BLOCK=MINUTES`** lets block 4 stop after its first point. Under round 1's own 29-minute
  estimate of three points, no deadline both starts the block and stops it after one. With `--estimate 4=15` and a
  deadline 16 to 17 minutes away, the block starts, runs its first point and stops before the second, and `run.sh`
  exits 0 with the block undone.

## Before

1. **The GPU is free.** No campaign, drill or probe of the box runs, and no other process holds the GPU.
2. **The box's units are stopped, and you note which were active**, to start them again after. `pssh` stands for the
   SSH command that reaches the host (`REHEARSAL_SSH`):

   ```sh
   # note these
   pssh phastform 'systemctl --user list-units --state=active --no-legend --plain "flatshare*"'
   pssh phastform 'systemctl --user stop flatshare.target flatshare-engine.service \
       flatshare-stt.service flatshare-tts.service flatshare-engine-proxy.socket \
       flatshare-mps.service;
     systemctl --user list-units --all --no-legend --plain \
       "flatshare*"'                         # every one inactive
   ```

   A unit left running, a socket unit among them, would hold the GPU and the box's sockets, and count as the
   rehearsal's own in block 0's listener audit, which matches units by name. `check` refuses while any is active,
   activating or deactivating.
3. **The checkout to push is at the revision to rehearse.** `REHEARSAL_REPO` defaults to the checkout this directory is
   in, and `push.sh` records its `git describe --always --dirty` as the host's `DEPLOYED`.

## The commands

Run them from this directory on the workstation, in bash 3.2 or newer, with `git`, `rsync`, `tar` and a working
`python3`. On a Mac, `/usr/bin/git` and `/usr/bin/python3` need Xcode's developer tools with their licence accepted,
or another `git` or `python3` first on `PATH`.

```sh
bash rehearse.sh all                       # every step not done yet, in order;
                                           # stops at the first that fails
bash rehearse.sh status                    # each step: done or not; each
                                           # host step: running, done STATUS
bash rehearse.sh STEP                      # one step; the steps before it must be done
bash rehearse.sh --redo STEP STEP          # a step again, although done (on the
                                           # host too: a block's blocks/N.done goes)
bash rehearse.sh abort                     # stop what runs on the host, then cleanup
bash rehearse.sh --purge cleanup           # cleanup, again if done, and remove
                                           # ~/a100-rehearsal on the host
```

- **It resumes.** A step done leaves `state/STEP.done` and is skipped from then on.
- **Host steps outlive the SSH session.** The steps from `stage` to `block4` each run in a session of their own,
  logged in `~/a100-rehearsal/.rehearsal/steps/NAME.log`. After a dropped connection `rehearse.sh` waits for the step
  for up to 10 minutes, and running it again later finds it still running or finds its exit status.
- **Logs.** Each step's output goes to the terminal and to `state/logs/STEP.log`, and `state/rehearse.log` has a line
  for each step's start and end.

| Variable | Default | What it sets |
|---|---|---|
| `REHEARSAL_HOST` | `phastform`, the box | the host, by any SSH host name or alias |
| `REHEARSAL_SSH` | `FLATSHARE_SSH`, else `ssh` | the ssh command, also `rsync -e`'s |
| `REHEARSAL_HOME` | `a100-rehearsal` | the rehearsal's home, one directory under the host's login home |
| `REHEARSAL_REPO` | the checkout this directory is in | the checkout pushed |
| `REHEARSAL_CYAN`, `REHEARSAL_PEACOCK` | `cyan`, `peacock` | where `data` and `reproduce` go |
| `REHEARSAL_CORPUS` | `~/flatshare/data/weka-x32` | the workstation's corpus |
| `REHEARSAL_PEACOCK_CORPUS` | `flatshare-data/weka-x32` | peacock's copy of it, under peacock's home |
| `REHEARSAL_GATEWAY_BIN` | unset | a `flatshare-gateway` for `push.sh`'s `FLATSHARE_GATEWAY_BIN`; by default `bootstrap.sh` builds one on the host |
| `REHEARSAL_MIN_FREE_GIB` | `40` | the disk `check` asks for [estimate] |
| `REHEARSAL_WORK` | `~/flatshare/rehearsal` | where `state/`, `data/`, `hosts/` and `reproduce/` go |

## The steps and their checks

Each step ends in a check, and a failed check stops the rehearsal. On 2026-10-26 `warm` took 29 of the 76 minutes,
from empty compile caches, and every other step under 17
([the record's timings](../../results/2026-10-26-a100-rehearsal/README.md#how-it-ran)).

| Step | Where | What it runs | What its check requires |
|---|---|---|---|
| `check` | host, read-only | The box's units, the GPU, the ports, the disk, the models in the box's hub cache, Kokoro, the tools, user namespaces and logind | No line `refused` |
| `init` | host | `~/a100-rehearsal`, `rehearsal.env`, the sudo shim, Kokoro copied from the box, the box's state recorded | `systemctl --user` in the rehearsal's environment cannot connect to a bus ([isolation](#isolation)) |
| `data` | workstation | `FLATSHARE_DATA` ([below](#where-the-data-comes-from)) | 48 long traces and the callers' `manifest.json` |
| `push` | workstation | `push.sh`: the source, 15 MB, and the data, 2.7 GB as files | The host's `DEPLOYED` is the checkout's `git describe`, and each data directory holds as many files as `FLATSHARE_DATA`'s and no link |
| `stage` | host | `bootstrap.sh stage`, the node rendered from the A100's profile (`a100-40gb`) with Phastform's facts | No preflight row refused, `env.sh` keeps flatshare-svc and the box's hub cache, and the control plane is active |
| `gpu` | host | `bootstrap.sh gpu` | No preflight row refused |
| `warm` | host | `bootstrap.sh warm`, one engine start per compile configuration | Every configuration in `cache/warm.json` ok |
| `block0` | host | `run.sh 0`: `bootstrap.sh gpu`, the warm check, the frozen box's engine, `smoke.sh` and its listener audit, a 2-turn call | `blocks/0.done`, `smoke: ok`, the call's `call.txt`, and the audit's `ok` line ([below](#block-0s-listener-audit)) |
| `block1` | host | `run.sh 1`: E0 with `--reps 3`, the KV and prefill cases 3 times each, the tail twice | `blocks/1.done`, E0's `summary.json` and the eight runs' `.txt` files |
| `block4` | host | `run.sh --until <17 min on> --estimate 4=15 4` | The first point's `run.json` complete, the second point not, no `blocks/4.done`, and `run.sh` exited 0 |
| `pull` | workstation | `pull.sh --once` into `hosts/phastform` | `blocks.log`, E0's summary, `preflight.json`, `warm.json`, the point's `run.json` and the units' journal pulled |
| `reproduce` | peacock | `make reproduce-a100 RECORDS=<the pulled records> CORPUS=<peacock's weka-x32>`, into `reproduce/` | Sections 3 and 4 of `a100.md`, the probes and E2, measured |
| `cleanup` | host | The rehearsal's units stopped, its processes ended, the box compared with `init`'s record | The box as `init` found it, the GPU idle and the ports free |

- **Three preflight rows are degraded on the box, as expected.** The user manager is hidden so that flatshare-svc runs
  the units, the box's 16 CPUs make R7 halve the plan, and the compile caches are cold until `warm`.
- **`a100.md`'s step model is not measured**, since block 5 is not rehearsed.
- **A rerun of `block4` after its first point completed fails its check.** The campaign resumes, skips that point as
  done and runs the second, which then fits the deadline.

## Where the data comes from

The `data` step assembles `data/`, the run's `FLATSHARE_DATA`, on the workstation:

- **`weka-x32`** is linked from the workstation's `REHEARSAL_CORPUS`, 393 traces and 1.7 GB, or copied from cyan's
  `~/flatshare/data/weka-x32` where that is absent. `reproduce` uses peacock's copy, `REHEARSAL_PEACOCK_CORPUS`, and
  copies the workstation's there when peacock has none.
- **`weka-x32-long`** comes from `benchmarks/e2/long_traces.py` over it, as links to the 48 traces with the largest
  median input. `push.sh` copies them as files, 1.0 GB on the host.
- **`callers`** is copied once from cyan's `~/flatshare/data/callers`, the clips and `manifest.json` that
  `flatshare-voice render-callers` wrote. The repository holds only the voice tests' two clips, so cyan's copy is the
  only source.

## Isolation

`init` writes `~/a100-rehearsal/rehearsal.env`, and `bin/rehearsal-ssh`, which is `push.sh`'s and `pull.sh`'s
`FLATSHARE_SSH`, sources it before each command. Through it the kit's `~/flatshare` is `~/a100-rehearsal/flatshare`,
with no change to the kit. The wrapper refuses a home without `rehearsal.env`, so nothing it runs can land in the login
home, where `push.sh`'s `rsync --delete` would delete what is in the box's `~/flatshare/src`.

| What | How the rehearsal keeps it apart |
|---|---|
| Home | `HOME=~/a100-rehearsal`, with the XDG directories, `TMPDIR` and every cache the units derive from `HOME` under it, and its sockets in `~/a100-rehearsal/run` (0700) |
| Supervisor | flatshare-svc, with no session bus and the runtime directory outside `/run/user/UID`, so `systemctl --user` cannot reach the box's user manager (note 1) |
| Models | The box's hub cache, read offline and never written, and Kokoro-FastAPI copied from the box. The kit still downloads uv, Python, the wheels, LiveKit's server and the gateway's toolchain |
| Root | A `sudo` shim that refuses, so `bootstrap.sh` installs no system package, and `check` requires `cc`, `c++`, `ss`, `git`, `curl`, `rsync`, `tar` and `python3` 3.10 or newer |
| Network | Each GPU server in a network namespace of its own ([below](#block-0s-listener-audit)). In the host's namespace the rehearsal listens on the gateway's TCP 127.0.0.1:18080, and on LiveKit's TCP 17880 and UDP 17882 only where user namespaces are refused. `check` requires these ports free, since the kit cannot move them |

1. Under flatshare-svc, on a host where a user manager answers, `bootstrap.sh` stops that manager's units of the kit's
   names, which here are the box's own. `init`'s check guards against that, and `check` reads the box's units from the
   login environment.

The rehearsal still shares `/tmp` with the box's engine, for vLLM's model-loading lock and torch's default inductor
directory, neither of which is part of the box's deployment.

**How `cleanup` compares the box.** `init` records every `flatshare*` unit with its state and enablement, and each
entry's kind, mode, size, mtime and link target under `~/flatshare`, `~/.config/systemd/user`, `~/.config/flatshare`,
`~/.cache/vllm`, `~/.cache/flashinfer` and the hub cache. `cleanup` records the same again and fails on any
difference, naming the paths. It also fails on a busy GPU, a held port, a box unit that runs, or a process whose
environment still names the rehearsal's home. AIPerf and vLLM's engine cores blank that environment when they set
their titles, so `cleanup` ends them by their process group and session. It never ends a process whose environment can
be read and lacks the home.

## Block 0's listener audit

In the host's network namespace every user of the host reaches two kinds of socket of the GPU servers: the six TCP
listeners that each vLLM engine core keeps on 127.0.1.1 for torch's gloo groups, and CUDA's abstract socket,
`@cuda-uvmfd-*`. So `deploy/bin/in-private-net` prefixes the GPU units' `ExecStart` and runs each server in a network
namespace of its own, `unshare --user --map-root-user --net` with the loopback up. The audit reads those namespaces'
tables through `/proc/PID/net` and accepts the listeners there. Where the kernel refuses user namespaces, the servers
run in the host's namespace (`FLATSHARE_PRIVATE_NET=host`), and the audit accepts those sockets on a degraded line.

On Phastform, Ubuntu 22.04 with user namespaces allowed, `block0`'s check wants the audit's `ok` line naming
`flatshare-engine, flatshare-stt, flatshare-tts in network namespaces of their own`. A refused audit, or the degraded
line here, fails `block0` and stops the rehearsal, which is a finding. On a host that restricts user namespaces, as
Ubuntu 24.04's kernel does, the degraded line is the one expected. The A100's Colab container allowed them
([its block 0's `smoke.txt`](../../results/2026-10-26-a100/block0/smoke.txt)).

## Abort and restore

1. **Abort.** `bash rehearse.sh abort` sends TERM to the running host step's process group and KILL to what is left
   after 180 s, time for flatshare-bench to give its callers 30 s and AIPerf 120 s to stop. It then runs `cleanup`.
2. **When `cleanup` fails.** It names what remains, such as a process, a held port or a busy GPU, or each path of the
   box's that changed, as the `diff` of the two states (`<` as found, `>` now). Nothing of the rehearsal writes outside
   `~/a100-rehearsal`, so a changed path is a finding or another process's write. Both states stay in
   `~/a100-rehearsal/.rehearsal` for reading.
3. **Restore.** The box's deployment is as it was once `cleanup` passes. Start the units you noted, for example
   `systemctl --user start flatshare.target` with any socket unit that was active, then check
   `systemctl --user list-units --state=active "flatshare*"` against your note.
4. **Purge.** `bash rehearse.sh --purge cleanup` runs `cleanup` again and then removes `~/a100-rehearsal`, about 30 GB
   [estimate]. Keep the home to read the rehearsal's logs on the host: `~/a100-rehearsal/flatshare/node/bootstrap.log`,
   `runs/a100/run.log` and `.rehearsal/steps/*.log`.

## What it covers and what it does not

The CPU test below runs this logic against stand-ins, and only a run on the box exercises the real downloads, builds,
engine starts, namespaces, probes, AIPerf and callers, and the time each step takes. The box cannot take these paths of
the A100's:

- **sm80's path**, Marlin and the BF16 rate;
- **installs that depend on the driver**: the cu129 lock, `cuda-compat-13-0` and CUDA 12.9.1's toolkit download;
- **hardware the box lacks**: several GPUs and `FLATSHARE_GPU_ID`, MIG, and a compute mode other than Default;
- **the supervisors**: a host without a usable user manager, and systemd as the single host's supervisor, whose
  `stop_units systemd` would act on the box's units;
- **sudo and apt installs**, and a host without user namespaces, which the CPU test's degraded line stands in for;
- **the rest of the run**: blocks 2, 3 and 5 to 10, the optional arms, and a reboot mid-run.

The A100 run took some of these paths on 2026-10-26, in a Colab container.
[Its record](../../results/2026-10-26-a100/README.md#how-it-ran) gives what differed from the rehearsal's host, the
driver's libraries outside the linker's search among them.

## The CPU test

```sh
cd python/flatshare_bench && uv run --extra test pytest -q tests/test_rehearse.py
```

It runs every step of `rehearse.sh` in order, with the repository's own `push.sh` and `pull.sh`, against local
directories for the hosts and stand-ins for the box's user manager, `nvidia-smi`, `ss`, `make`, `bootstrap.sh`,
`run.sh` and flatshare-svc. Its docstring lists every case it holds. On Linux it also checks `cleanup`'s leftover
processes against `/proc`, which macOS lacks.
