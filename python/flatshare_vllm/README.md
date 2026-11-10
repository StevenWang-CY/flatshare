# flatshare-vllm

The change inside vLLM, for a developer who builds, tests or maintains it. The engine variant `make-room` runs the
scheduler diff of pull requests [#65282](https://github.com/vllm-project/vllm/pull/65282) and
[#65283](https://github.com/vllm-project/vllm/pull/65283), stacked on #65282, on vLLM 0.32.0, which the rounds measured,
and on 0.33.0. Both are proposed to vLLM, open and rebased, with no maintainer review as of 2026-11-09. The serving
engine's agent queue and CPU-tier copy are proposed in vLLM's own terms as the drafts
[#66558](https://github.com/vllm-project/vllm/pull/66558) and [#66527](https://github.com/vllm-project/vllm/pull/66527).
The variant `pr54530`, vLLM [#54530](https://github.com/vllm-project/vllm/pull/54530) ported to 0.32.0, is a baseline.
[ARCHITECTURE.md §2](../../docs/ARCHITECTURE.md#2-the-change) states the rules, and
[`deploy/phastform/variants`](../../deploy/phastform/variants/README.md) holds the systemd drop-ins that run them.

```text
src/flatshare_vllm/build.py      writes a variant's module at each engine start
src/flatshare_vllm/make-room/    the patches, 0.33.0's in v0.33.0/, and probe.py, tail.py and knee.sh
src/flatshare_vllm/pr54530/      the port of #54530 and its probe
src/flatshare_vllm/serve/        the serving variant's scheduler class and the parts it installs
```

## The build

```sh
python src/flatshare_vllm/build.py make-room OUT_DIR      # or pr54530
```

`build.py` reads the scheduler of the vLLM that the interpreter finds, applies the variant's patches in order, appends
the stock `AsyncScheduler` and writes `OUT_DIR/flatshare_make_room_scheduler.py`. The engine loads it with
`PYTHONPATH=OUT_DIR` and `--scheduler-cls flatshare_make_room_scheduler.AsyncScheduler`, and the installed vLLM is never
modified. The build needs only the standard library and finds vLLM without importing it, so the engine's own Python
runs it from this source tree.

- **Every hunk applies exactly or the build fails.** A hunk must apply at its stated line, with its context and removed
  lines as they are there, and the failure names the hunk.
- **Patches are per release.** The build reads the installed vLLM's release from its package metadata and refuses one
  that no patches are written against. `make-room` has 0.32.0's and 0.33.0's, and `pr54530` has 0.32.0's.
  `--source-tree` builds from a source checkout, whose release is the one whose patches apply to it. Exactly one must
  apply, or `--release R` names it.

`make-room` applies three patches in order:

- **`63350.patch` and `63351.patch`** hold the two pull requests' changes to `vllm/v1/core/sched/scheduler.py`,
  backported to v0.32.0. They are named for #63350 and #63351, the numbers the pull requests were first opened under.
  Each header says where 0.32.0 made the backport differ:
  - #65282 calls make-room after 0.32.0's long-prefill threshold, since 0.32.0 has no local prefill cadence;
  - the counters of make-room's preemptions and of the budget cap's held steps and tokens stay in the scheduler, since
    0.32.0's `SchedulerStats` has no fields for them;
  - #65283's check of the waiting queue for a lower priority is the scheduler's own, since the request queues stay
    stock;
  - #65282's prefix-cache hit keeps 0.32.0's bound, `num_tokens - 1`.
- **`variant.patch`** is what only the variant has. Its switches are keys of `--additional-config`'s `make_room` table,
  validated at engine start, where the pull requests have `SchedulerConfig` fields. It adds six switches, each off by
  default, and logs the rules' counts at most every 10 s.
  [The variants' switch table](../../deploy/phastform/variants/README.md#the-switches) lists every key and the pull
  requests' flag for each.

The serving variant, `make-room-serve`, loads `flatshare_make_room_serve.AsyncScheduler` from `src/flatshare_vllm/serve`
over make-room's module. Its constructor installs what the engine's additional config asks for, each part off at 0:

- `flatshare_jit`, which schedules the next batch just in time;
- `flatshare_queue`, the cache-aware queue, which orders the agents by the prefill each has left;
- `flatshare_copy`, which loads the CPU tier through the SM copy kernel.

It then writes the configuration the scheduler runs, with the SHA-256 of each module it builds on, to the file that
`FLATSHARE_ENGINE_EFFECTIVE` names. `deploy/bin/serving-check` holds that file to
[the serving profile](../../deploy/phastform/variants/README.md#the-serving-profile).

## One source for the change

The pull requests are the change's only source. A change is made on their branches on vLLM's main,
`priority-kv-preemption` and `priority-prefill-hold` on the fork
[StevenWang-CY/vllm](https://github.com/StevenWang-CY/vllm), and cherry-picked onto three branches of v0.32.0 that the
fork publishes:

- `v0.32.0-priority-kv-preemption`, with #65282's commit;
- `v0.32.0-priority-prefill-hold`, with #65283's on top;
- `v0.32.0-flatshare-make-room`, with the variant's commits on top.

In a clone of the fork with vLLM's `v0.32.0` tag fetched, `scripts/make-room-patches.sh CLONE` rewrites the three
patches from these branches. Each patch is the `scheduler.py` diff of one branch's commits, and its header names them.
Two checks keep the variant equal to the pull requests:

- **The trace** (`tests/test_trace.py`) builds the variant and a module of the pull requests' two patches alone, and
  runs both over the same request traces on vLLM's real scheduler. Every step's scheduled tokens by request, its
  preempted, new, cached and resumed requests and its free blocks must be equal. It covers each switch set E2 ran with
  the reserve off, synchronous and asynchronous, at watermarks 0 and 0.07. Its four workloads are the behavior suite's,
  KV pressure where make-room preempts under the budget cap and past two victims, and random ones with shared prefixes
  or with a KV connector's asynchronous loads. One-line mutants of the pull requests' module, its negative control,
  must each diverge. `TRACE_REFERENCE=<a generated module>` compares the variant with that module instead.
- **The backport against the pull requests' heads**, in `make compat-upstream` ([below](#fit-with-vllms-main)). The
  lines each pull request adds to `scheduler.py` must be its patch's, except those that `ADAPTED` in
  `scripts/compat_upstream.py` names under a commit whose "Backport:" paragraph says why. They are 39 lines of the
  pull requests' and 24 of the backport's, for the counters, the queue check and the hit's bound above and for three
  parts of main that v0.32.0 lacks. These are the queue of requests waiting to start an async KV load, the blocks' weak
  reference to their pool and the cache of parked KV loads' reservations.

## vLLM 0.33.0

make-room builds on vLLM 0.33.0, released 2026-11-02, from that release's own patches in `make-room/v0.33.0`.
`BASE=v0.33.0 scripts/make-room-patches.sh CLONE` writes them from three branches of v0.33.0 that the fork publishes,
`v0.33.0-priority-kv-preemption`, `v0.33.0-priority-prefill-hold` and `v0.33.0-flatshare-make-room`, which hold the same
12 commits. Every line they add to `scheduler.py` is one that 0.32.0's patches add, except for #65282's prefix-cache
hit, which takes the bound of v0.33.0's waiting loop.

- **What 0.33.0 changed under the patches.** Two places needed resolving, and the resolutions move lines without
  changing them. #65282's make-room call follows 0.33.0's local prefill cadence (#54627), as on main. The variant's
  short-load rule follows 0.33.0's cap of the external hit for auxiliary outputs, so that it judges the load that is
  left.
- **What checks it.** `make test-vllm VLLM_REF=v0.33.0` and `make test-make-room-ci VLLM_REF=v0.33.0` run the build,
  the scheduler suites and the trace on 0.33.0, the latter in the environment that `requirements-cpu-v0.33.0.txt`
  pins with CPU torch 2.14. `make compat-upstream` requires v0.33.0 to take its own patches. CI runs 0.32.0's.
- **What serves.** The serving profile stays on 0.32.0, which the rounds measured. On Phastform the repository's quick
  start on 0.33.0 gave the voice turn's first token in 167–207 ms with make-room, against 1,778–1,788 ms on stock
  0.33.0, and the serving variant's engine ran there
  ([the port's record](../../benchmarks/results/2026-11-02-port-v0.33.0/README.md)).

## Tests

`make test-vllm`, which CI runs, needs no GPU, torch or installed vLLM. On first use it makes a sparse checkout of
vLLM v0.32.0 in `~/.cache/flatshare/vllm-v0.32.0`, and `VLLM_SRC` points it at another checkout. The scheduler suites
need torch and vLLM's dependencies, so they skip there and run under their own targets:

```sh
make test-vllm                               # the build, the probes, the upkeep checks' own tests
make test-vllm VLLM_REF=v0.33.0              # the build on 0.33.0's source (make-room alone)
make test-make-room TORCH_PYTHON=PY          # the scheduler suites on vLLM's real scheduler, on CPU
make test-trace TORCH_PYTHON=PY              # the trace; TRACE_SEEDS=100 for 100 random seeds, not 20
make test-make-room-ci                       # both, as CI's make-room job runs them
```

In the package's directory, `VLLM_SRC=~/src/vllm-v0.32.0 uv run --extra test pytest` runs the first target's tests on a
checkout of your own. `TORCH_PYTHON` is an absolute path to a Python with CPU torch 2.13 and vLLM 0.32.0's Python
dependencies, and both targets fail if it cannot import them. `make test-make-room-ci` builds that environment in
`build/cpu-env` from `requirements-cpu.txt` (Python 3.12, Linux x86_64), and with `FLATSHARE_NO_SKIPS=1` a skipped test
fails the session.

On a GPU node the suites run against the engine environment with pytest beside it, kept off the GPU:

```sh
tools=~/flatshare/cache/testtools
[[ -d $tools/pytest ]] || ~/.local/bin/uv pip install -q --target "$tools" pytest
cd ~/flatshare/src/python/flatshare_vllm
VLLM_TARGET_DEVICE=cpu PYTHONPATH=$tools ~/flatshare/venvs/engine/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_make_room.py tests/test_trace.py
```

| Test | What it checks | Run by |
|---|---|---|
| `test_build.py` | Each module is its patches as `git apply` applies them, then the stock `AsyncScheduler`, and the serving profile's hash is the build's | `make test-vllm` |
| `test_make_room.py` | make-room's rules on vLLM's real v1 scheduler, with hand-made requests and no model | `make test-make-room` |
| `test_trace.py` | make-room against the pull requests, step for step ([above](#one-source-for-the-change)) | `make test-trace` |
| `test_serve.py`, `test_queue.py`, `test_copy.py` | The serving variant: its scheduling, the configuration it writes, its queue and its copy | `make test-make-room` |
| `test_gateway_caps.py` | The gateway's reply caps as vLLM's request models read them | `make test-make-room` |
| `test_jit.py` | Just-in-time scheduling on a stand-in engine core | `make test-vllm` |
| `test_probe.py`, `test_tail.py`, `test_knee.py` | The checks on a GPU ([PRODUCTION.md](../../docs/PRODUCTION.md#checking-it-on-your-hardware) runs the probes), against stand-ins such as `tests/fake_engine.py` | `make test-vllm` |
| `test_perf_schedule.py`, `test_perf_control.py` | The perf check's arithmetic and states, and its negative control | `make test-vllm` and `make test-make-room`; the control, `make perf-make-room` |
| `test_compat_upstream.py` | `compat_upstream.py` against a fake upstream | `make test-vllm` |
| `test_container.py` | [The container's](../../deploy/container/README.md) entrypoint against the serving drop-in | `make test-vllm` |

Each test file's docstring says what it holds.

## Upkeep: the change's cost and its fit upstream

Run both checks, which `make check` leaves out, before changing the engine's code or moving off vLLM 0.32.0.

### The cost of a scheduler step

```sh
make build/cpu-env/synced                                      # from the repository's root
make perf-make-room TORCH_PYTHON=$PWD/build/cpu-env/bin/python
```

`tests/perf_schedule.py` times `Scheduler.schedule()` for stock vLLM's scheduler and for make-room's module with its
switches off and at the serving profile's values. It covers the 13 steady states of the pull requests' cost tables,
and four states of the serving profile at `serving.json`'s settings with asynchronous scheduling:

- `idle-R2`, agents only;
- `hold-R2`, a reply's first tokens while an agent that fits waits;
- `nofit-R16`, a voice turn that two victims cannot make room for;
- `fire-R2`, the step in which make-room preempts for a voice turn.

A configuration's cost is its cheaper scheduler's 10th-percentile step, with two schedulers of each configuration
taking turns in one process, so that other work on the machine slows them alike. A run takes about 18 minutes on a
2.1 GHz Xeon, and a new baseline about 30. The tracked baseline, `benchmarks/perf/make-room-schedule.json`, was
measured on peacock in the environment above, and [the report's Figure 7](../../docs/REPORT.md#4-implementation) draws
it.

- **Two ratios a state.** off/stock − 1 is what the change costs with its switches off, which the pull requests
  measured on main. on/off − 1 is what it costs with them at the profile's values. Each is the median over the state's
  processes. On peacock off/stock is 0.0% to 1.8% in the pull requests' states, and 2.3% to 4.5% in the serving
  profile's, where the steps are short and the cost is 3 to 5 µs a step.
- **The verdict.** A state fails when a ratio exceeds the baseline's by more than its tolerance, 4%, after six more
  processes measure it again. The tolerance is twice the worst difference between two measurements of unchanged code
  on peacock, rounded up. A ratio that fell by more than the tolerance passes with a note that the baseline no longer
  describes the code. The table's "fails past" column gives each tolerance in µs.
- **Its negative control.** `tests/test_perf_control.py` then times three mutants of make-room, each with work added
  where the change avoids it, which must fail in the states where that work runs, and make-room itself, which must
  pass. On peacock the mutants fail by 17% to 87%.
- **What it cannot see.** The 10th-percentile step misses work added to fewer than about 9 steps in 10. A run whose
  1-minute load passes 1.5 times the baseline's highest prints a warning beside its verdict.
- **Another machine.** A different CPU weights the change's code differently. On an M5 Max on/off came within 1.3% of
  peacock's in 11 of the 17 states and 2.0% to 5.6% from it in the other six, and off/stock within 2.1% in all 17. So
  the check refuses a machine whose CPU, Python or torch differs from the baseline's, unless the script's
  `--any-machine` overrides it with a warning. Measure that machine's own baseline into a file of its own, then check
  against it:

  ```sh
  make perf-make-room TORCH_PYTHON=PY UPDATE=1 PERF_BASELINE=build/perf-baseline.json PERF_BESIDE="<what else runs>"
  make perf-make-room TORCH_PYTHON=PY PERF_BASELINE=build/perf-baseline.json
  ```

`tests/perf_schedule.py` exits 0 when every state is within its tolerance, 1 when one costs more, and 2 when it could
not measure, as with a baseline from another machine. The make target exits 2 for any failure, so a job that must tell
them apart runs the script itself.

```sh
cd python/flatshare_vllm && VLLM_TARGET_DEVICE=cpu PYTHONPATH=<VLLM_SRC> <TORCH_PYTHON> tests/perf_schedule.py check <baseline>
```

### Fit with vLLM's main

```sh
make compat-upstream [TORCH_PYTHON=<an interpreter for vLLM main's scheduler and tests>]
```

`scripts/compat_upstream.py` fetches vLLM's main, the pull requests' heads and the release tags into a blobless clone
in `~/.cache/flatshare/vllm-upstream`, and checks four things:

- **Each pull request against main.** Main may hold it already, as a squash-merged commit whose subject ends with its
  number, as a true merge, or as a merge that would change nothing. Otherwise it must merge cleanly
  (`git merge-tree --write-tree`, git 2.38 or newer, and 2.40 for #65283 once main holds #65282), or the report names
  the files that conflict.
- **Their tests.** Given an interpreter, it runs on CPU the test files each pull request adds or changes, on its merge.
  Each run has `--timeout` seconds, 900 by default, or its files fail.
- **make-room on newer releases.** It builds make-room from 0.32.0's patches on the newest release and on a newer
  release candidate if there is one, and on v0.33.0 from its own patches.
- **The backport against the pull requests' heads**, as [above](#one-source-for-the-change). It prints both sides of
  any line that differs without an adaptation, and notes an adapted line that no longer differs.

It exits 1 for a conflict, a failing test that main does not fail too, a backport line that differs without an
adaptation, or patches that do not apply to v0.32.0 or v0.33.0. It exits 2 when it could not check, as with no
network, an older git, or a pull request none of whose tests ran, and 0 otherwise. A newer release that 0.32.0's
patches do not fit still exits 0, with the first hunk that fails, since make-room is written against 0.32.0 until it
is ported.

**Reading a hunk's fit.** `build.py` takes a hunk only exactly at its line, so the report gives every hunk's fit as
`git apply` finds it, with all three lines of context (`-C3`) and with one (`-C1`). A hunk rejected at `-C3` is one a
rebase re-anchors by hand. One rejected at `-C1` too is one whose own lines, or the line beside them, changed.
