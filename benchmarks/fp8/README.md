# The FP8 KV cache crash: the harness

The box's engine died with an FP8 KV cache beside vLLM's CPU tier (the 2026-10-27 sweep's
[record](../results/2026-10-27-sweep/README.md#the-fp8-kv-cache)). This directory reproduces the crash on stock vLLM,
with no Flatshare code in the engine, and tests its cause, for a reader who reruns or extends those runs. FlashInfer's
`plan()` refills one page-locked buffer per wrapper and copies it to the GPU without a wait, so under asynchronous
scheduling the host can refill it before the previous step's copy has run (vLLM
[#61239](https://github.com/vllm-project/vllm/issues/61239), fixed by
[#64165](https://github.com/vllm-project/vllm/pull/64165)). What the runs found is in
[the record](../results/2026-10-31-fp8-crash/README.md).

| File | What it does |
|---|---|
| `run.sh NAME CONFIG SECONDS [FLAG...]` | Serves vLLM from `$FP8_VENV` (default the box's engine environment) with CONFIG and the flags on a private Unix socket, in a network namespace of its own (`deploy/bin/in-private-net`), drives it with `load.py`, and writes `runs/NAME` (note 1) |
| `load.py` | The load: 24 agent sessions that resend 8k–24k-token contexts growing each turn, with pauses, so the pool overflows into the CPU tier and resumed sessions load their prefixes back, beside two streams of short priority-0 turns. Prompts are token ids, from seeded generators. Exits 3 when the engine dies, 4 when it stalls (requests in flight and none finished for 180 s) |
| `offload.yaml`, `nooffload.yaml` | The frozen box's engine settings at v1.13 (a 5 GiB pool, priority scheduling, 512-token steps with a 256-token threshold, prefix caching) with and without the `OffloadingConnector` (20 GiB, write-through); runs add `--kv-cache-dtype fp8` |
| `patch_plan_wait.py VENV` | Applies #64165's three hunks to a scratch environment's `vllm/v1/attention/backends/flashinfer.py`, with counters: `FP8INV_PLAN_WAIT=block` waits as #64165 does, `count` only counts the plans that find the previous copy pending. It writes the file anew and renames it, so a hard link to uv's cache or another environment keeps the stock file |
| `check_record.py DIST...` | Checks that every file of the distributions hashes as its wheel's RECORD says: a stock install |
| `copy_contention.py OUT` | Times a 64 KiB pinned host-to-GPU copy alone and beside 1 GiB moving either way on another stream, and reads the card's copy-engine count |
| `dump_selftest.py DIR CUDA_GDB OUT` | The positive control for a stall's core dump: spins a kernel for about 20 s, requests a user-triggered GPU core dump while it runs, and passes if cuda-gdb names the kernel in it (note 2) |
| `diag/sitecustomize.py` | Loaded by every Python process of the server (`run.sh` puts it on `PYTHONPATH`): SIGUSR1 dumps every thread's stack, without ptrace |
| `queue-*.sh` | The runs of 2026-10-31 in the order they ran, each waiting for the GPU to be free, and each queue for the one before |
| `summarize.py RUNS_DIR` | The runs as one Markdown table, a row a run, from what each wrote; a run from before `load.py` detected stalls is held to its rule after the fact |

1. `runs/NAME` holds `result.txt` (versions, environment, verdict, first fault line), `server.log`, `requests.jsonl` and
   `summary.json`. The verdict is `CRASHED` (a CUDA fault in the log), `STALLED`, `DIED` (the server stopped answering
   with no fault line), `SURVIVED` or `NO-START`. On a stall `run.sh` asks the engine core for its stacks and, with
   user-triggered core dumps on, for a GPU core dump through the core's pipe.
2. Without `skip_abort` in the generation flags the driver aborts the process after the dump, as it did the self-test's
   child (exit -6), so r18 sets it.

The scripts are the ones that ran, formatted for the repository since. Two instruments came during the runs, and the
record says which runs had them: `load.py`'s stall detection, after r3, and `run.sh`'s `DIED` verdict and stall core
dump, before `queue-core.sh` began.

## Setup on the GPU host

Copy this directory there, with the box deployed (`deploy/phastform`). It writes `runs/`, its environments and its JIT
caches beside itself. The GPU must be idle, since `run.sh` refuses beside another process or 100 MiB in use.

```sh
# The fix: the engine's lock synced afresh into a scratch environment, then patched
uv venv -p 3.12 venv-fix
uv pip sync --python venv-fix/bin/python ~/flatshare/src/deploy/phastform/requirements/engine.txt \
  --torch-backend cu130
venv-fix/bin/python patch_plan_wait.py venv-fix
# vLLM 0.33.0rc3
uv venv -p 3.12 venv-rc3
uv pip install --python venv-rc3/bin/python "vllm==0.33.0rc3.dev3+g4ba787ec9" \
  --extra-index-url https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/cu130 \
  --torch-backend=cu130
# vLLM main: the nightly wheel of 564ac0de1 (0.33.0rc3's flashinfer.py, unchanged on main) into
# venv-main
./install-main.sh
```

`queue-main.sh` runs vLLM main with rc3's FlashInfer workspace.
