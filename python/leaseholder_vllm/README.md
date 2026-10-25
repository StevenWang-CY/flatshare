# leaseholder-vllm

The change on vLLM 0.32.0, the version E2 measured: vLLM [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351) (stacked on #63350); this package runs their scheduler diff on 0.32.0 as Leaseholder's `make-room` variant, beside `pr54530`, vLLM [#54530](https://github.com/vllm-project/vllm/pull/54530) ported to 0.32.0 as a baseline. A variant is a scheduler module that `build.py` writes at each engine start from the installed vLLM's own scheduler, which it never modifies. The variants' systemd drop-ins are in [`deploy/phastform/variants`](../../deploy/phastform/variants/README.md).

```text
src/leaseholder_vllm/build.py    writes a variant's module: its patches, applied exactly, and the stock AsyncScheduler
src/leaseholder_vllm/make-room/  63350.patch and 63351.patch, the PRs' scheduler diff backported to v0.32.0;
                                 variant.patch, what only the variant has; probe.py and tail.py, its GPU checks
src/leaseholder_vllm/pr54530/    pr54530.patch, #54530 ported to v0.32.0 by hand; probe.py
scripts/make-room-patches.sh     rewrites make-room's patches from the backport's branches in a vLLM clone
tests/                           the build, make-room's behavior, and make-room traced against the PRs
```

## The build

`python src/leaseholder_vllm/build.py make-room OUT_DIR` (or `pr54530`) reads the scheduler of the vLLM the interpreter finds, applies the variant's patches in order, appends the stock `AsyncScheduler`, which then extends the patched `Scheduler`, and writes `OUT_DIR/lh_make_room_scheduler.py`. The engine runs it with `PYTHONPATH=OUT_DIR` and `--scheduler-cls lh_make_room_scheduler.AsyncScheduler`. The build needs only the standard library and finds vLLM without importing it, so the engine's own Python runs it from this source tree, with no install. Every hunk must apply at its stated line, its context and removed lines as they are there: the build refuses any installed vLLM but 0.32.0, and a source the patches do not match fails it, naming the hunk. `--source-tree` builds from a v0.32.0 source checkout, which has no package metadata to check.

`make-room`'s patches apply in order:
- `63350.patch` and `63351.patch`: the two PRs' changes to `vllm/v1/core/sched/scheduler.py`, backported to v0.32.0 and adapted where 0.32.0 needs it, as each patch's header says under the commit: #63350 calls make-room after 0.32.0's long-prefill threshold, since 0.32.0 has no local prefill cadence; the PRs' counters of make-room's preemptions and of the budget cap's held steps and prefill tokens stay in the scheduler, since 0.32.0's `SchedulerStats` has no fields for them; and #63351's check of a waiting queue for a lower priority is the scheduler's own, since the request queues stay stock.
- `variant.patch`: what only the variant has. Its switches are keys of `--additional-config`'s `make_room` table, validated at engine start, instead of the PRs' `SchedulerConfig` fields; it adds the reserve, which requests of a positive priority value leave free, and logs the PRs' counters, with its own of the admissions the budget cap holds back, at most every 10 s ([`deploy/phastform/variants`](../../deploy/phastform/variants/README.md) has the keys).

## One source for the change

The PRs are the change's only source. A change is made on their branches, on vLLM main, and cherry-picked onto the backport: three branches of v0.32.0 in a vLLM clone, `up-make-room-kv-v0.32.0` (#63350's commits), `up-budget-cap-v0.32.0` (#63351's on top) and `up-make-room-v0.32.0` (the variant's commit on top). `scripts/make-room-patches.sh CLONE` then rewrites the three patches, each the `scheduler.py` diff of one branch's commits, with the commits named in its header.

`tests/test_trace.py` keeps the variant and the PRs from drifting:

- **What it compares.** It builds the variant, and a module from the PRs' two patches alone, their `SchedulerConfig` fields read from the variant's keys. Both run the same request traces on vLLM's real scheduler, and every step's scheduled tokens by request, preempted, new, cached and resumed requests and free blocks must be equal.
- **Where.** In each switch set E2 ran (with the reserve off), synchronous and asynchronous, at watermarks 0 and 0.07, over four workloads: the behavior suite's; KV pressure, where make-room preempts under the budget cap and past two victims; random ones with shared prefixes; random ones with a KV connector's asynchronous loads, "not yet" lookups and pending frees.
- **Its negative control.** One-line mutants of the PRs' module, each of which must diverge.
- **Against another module.** `TRACE_REFERENCE=<a generated module>` compares the variant with that module instead, the reserve and the counters included: before a change lands, the module it replaces against the new one. Before #63351 bounded a step's held tokens by its budget, a module counted each of the step's held chunks against the whole budget; against such a module (W5's), the variant's held tokens need only be at most the module's.

## Tests

`tests/test_build.py` builds both variants from a vLLM v0.32.0 source checkout's scheduler and checks that each module is the scheduler with its patches as `git apply` applies them, then the stock `AsyncScheduler`; that a source the patches do not match, or another installed vLLM, fails the build; and how hunks are matched. It needs no GPU, torch or installed vLLM. CI runs it with `make test-vllm`, which makes a sparse checkout in `~/.cache/leaseholder/vllm-v0.32.0` on first use; `VLLM_SRC` points it at another checkout, and its build tests skip without one:

```sh
VLLM_SRC=~/src/vllm-v0.32.0 uv run --extra test pytest      # in python/leaseholder_vllm
```

`tests/test_make_room.py` checks the `make-room` variant's behavior on vLLM's real v1 scheduler, on CPU, with hand-made requests and no model, as vLLM's own `tests/v1/core` do:

- **The KV preemption** for a waiting higher-priority request under KV pressure: the request chosen, the victims and their order, shared prefixes, the per-step cap, deferred block frees, LoRA adapters, a request the waiting loop skips.
- **The budget cap:** lower-priority prefill sits out a request's first tokens from the step it fits or its KV load is done, and the first steps of a load only, and not for a request that does not fit.
- **The reserve**, the switches' validation, and stock behavior with the switches off.

A stand-in KV connector loads a request's KV asynchronously or delays it. `tests/test_trace.py` is the trace above. The scheduler needs torch and vLLM's dependencies, which `make test-vllm`'s environment lacks, so there these tests skip; CI's `make-room` job runs them in a CPU environment of their own (below). On the GPU node they run against the engine environment with pytest beside it, and `VLLM_TARGET_DEVICE=cpu` keeps them off the GPU:

```sh
tools=~/leaseholder/cache/testtools
[[ -d $tools/pytest ]] || ~/.local/bin/uv pip install -q --target "$tools" pytest
cd ~/leaseholder/src/python/leaseholder_vllm
VLLM_TARGET_DEVICE=cpu PYTHONPATH=$tools ~/leaseholder/venvs/engine/bin/python -m pytest -q -p no:cacheprovider tests/test_make_room.py tests/test_trace.py
```

Elsewhere, an environment with CPU torch 2.13 and vLLM 0.32.0's Python dependencies runs them on the source checkout instead, from the repository's root: `make test-make-room TORCH_PYTHON=<its python>`, and `make test-trace TORCH_PYTHON=<its python>` (add `TRACE_SEEDS=100` for the random workloads at RV2's count). Both first check that the interpreter imports pytest, torch and vLLM's scheduler, and fail if it does not, rather than let pytest skip the suites. `make test-make-room-ci` runs both as CI's `make-room` job does. It runs them in the environment that `requirements-cpu.txt` pins: Python 3.12 on Linux x86_64, CPU torch 2.13.0, and vLLM's dependencies at the engine environment's versions (`requirements-cpu.in`, compiled by the command in its header). uv builds that environment in `build/cpu-env` (`CPU_ENV`), and the target sets `LH_NO_SKIPS=1`, under which any skipped test fails the session (`tests/conftest.py`).

The make-room probes' tests run in CI, with aiohttp from the test extra (the engine environment has it too), against `tests/fake_engine.py`, a stand-in for vLLM's server with a pool, a context and a fixed pace:

- `tests/test_probe.py`: `probe.py`'s requests sized from the engine's pool and context, the frozen box's sizes unchanged, and its result line as the committed outputs' check and the PR bodies' renderer parse it, on a loopback port and on a Unix socket.
- `tests/test_tail.py`: `tail.py`'s accounting of failed call turns and stopped agents, its agents and their prompts (the frozen box's unchanged, others from the pool and context), whole replies, a run's lines and an engine on its Unix socket.
- `tests/test_knee.py`: `knee.sh`, the budget cap's window sweep, against stand-ins for `variant.sh`, the engine, `journalctl` and `tail.py`.
