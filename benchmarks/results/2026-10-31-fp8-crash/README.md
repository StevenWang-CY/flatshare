# The FP8 KV cache crash (2026-10-31)

The 2026-10-27 sweep's FP8 KV cache died both times it ran beside vLLM's CPU tier
([its record](../2026-10-27-sweep/README.md#the-fp8-kv-cache)). These runs find why. On Phastform's RTX 5060 Ti,
[`benchmarks/fp8`](../../fp8/README.md) served stock vLLM, with none of the change's code, under a load of its own, and
varied one thing at a time: the CPU tier, asynchronous scheduling, the attention backend, the KV cache's type and the
decode kernel. It then tested the cause with the wait of vLLM [#64165](https://github.com/vllm-project/vllm/pull/64165),
with a counter of the plans that find their buffer's last copy still pending, on vLLM 0.33.0rc3 and main, with blocking
launches, and with a GPU core dump.

## The runs

How each run was set up, in the order the runs started:

| Run | vLLM | Build | KV cache | Attention | CPU tier |
|---|---|---|---|---|---|
| r1-fp8-offload | 0.32.0 | stock | FP8 | FLASHINFER | yes |
| r2-fp8-nooffload | 0.32.0 | stock | FP8 | FLASHINFER | no |
| r3-bf16-flashinfer-offload | 0.32.0 | stock | BF16 | FLASHINFER | yes |
| r9-fp8-offload-fix | 0.32.0 | + #64165 | FP8 | FLASHINFER | yes |
| r10-fp8-offload-count | 0.32.0 | + #64165, counting only | FP8 | FLASHINFER | yes |
| r12-bf16-flashinfer-offload-fix | 0.32.0 | + #64165 | BF16 | FLASHINFER | yes |
| r7-fp8-offload-noasync | 0.32.0 | stock, async scheduling off | FP8 | FLASHINFER | yes |
| r8-fp8-triton-offload | 0.32.0 | stock | FP8 | TRITON_ATTN | yes |
| r4-bf16-flashattn-offload | 0.32.0 | stock | BF16 | FLASH_ATTN | yes |
| r1b-fp8-offload | 0.32.0 | stock | FP8 | FLASHINFER | yes |
| r13w-rc3-warmup | 0.33.0rc3 | stock | FP8 | FLASHINFER | yes |
| r13-rc3-fp8-offload | 0.33.0rc3 | stock | FP8 | FLASHINFER | yes |
| r15-rc3-fp8-offload-fix | 0.33.0rc3 | + #64165 | FP8 | FLASHINFER | yes |
| r14-rc3-fp8-offload-count | 0.33.0rc3 | + #64165, counting only | FP8 | FLASHINFER | yes |
| r19-fp8-nooffload-count | 0.32.0 | + #64165, counting only | FP8 | FLASHINFER | no |
| r16-fp8-offload-xqa | 0.32.0 | stock, XQA decode | FP8 | FLASHINFER | yes |
| r17-fp8-offload-xqa-count | 0.32.0 | + #64165, counting only, XQA decode | FP8 | FLASHINFER | yes |
| r18-fp8-offload-coredump | 0.32.0 | stock, core dump | FP8 | FLASHINFER | yes |
| r5-fp8-offload-blocking | 0.32.0 | stock, blocking launches | FP8 | FLASHINFER | yes |
| r20w-main-warmup | main (564ac0de1) | stock | FP8 | FLASHINFER | yes |
| r20-main-fp8-offload | main (564ac0de1) | stock | FP8 | FLASHINFER | yes |

What each run did:

| Run | Verdict | Load planned, s | Answered / failed | Last answer, s | Plans finding the last copy pending |
|---|---|---:|---:|---:|---:|
| r1-fp8-offload | CRASHED at 124 s | 1,200 | 85 / 20 | 119 | — |
| r2-fp8-nooffload | SURVIVED | 900 | 405 / 0 | 898 | — |
| r3-bf16-flashinfer-offload | STALLED (by its requests; run.sh said SURVIVED) | 900 | 117 / 26 | 226 | — |
| r9-fp8-offload-fix | SURVIVED | 1,800 | 811 / 0 | 1,799 | 376 of 30,000 |
| r10-fp8-offload-count | STALLED at 345 s | 900 | 128 / 0 | 164 | 20 of 1,477 |
| r12-bf16-flashinfer-offload-fix | SURVIVED | 900 | 425 / 0 | 900 | 160 of 20,000 |
| r7-fp8-offload-noasync | SURVIVED | 900 | 432 / 0 | 900 | — |
| r8-fp8-triton-offload | SURVIVED | 900 | 432 / 0 | 898 | — |
| r4-bf16-flashattn-offload | SURVIVED | 900 | 404 / 0 | 895 | — |
| r1b-fp8-offload | CRASHED at 194 s | 600 | 85 / 30 | 189 | — |
| r13w-rc3-warmup | CRASHED at 128 s | 180 | 94 / 17 | 124 | — |
| r13-rc3-fp8-offload | CRASHED at 214 s | 900 | 158 / 24 | 209 | — |
| r15-rc3-fp8-offload-fix | SURVIVED | 900 | 449 / 0 | 897 | 223 of 15,000 |
| r14-rc3-fp8-offload-count | STALLED at 485 s | 900 | 209 / 0 | 304 | 117 of 5,000 |
| r19-fp8-nooffload-count | SURVIVED | 900 | 405 / 0 | 898 | 0 of 10,000 |
| r16-fp8-offload-xqa | STALLED at 339 s | 900 | 121 / 0 | 159 | — |
| r17-fp8-offload-xqa-count | STALLED at 309 s | 900 | 94 / 0 | 127 | 7 of 847 |
| r18-fp8-offload-coredump | CRASHED at 184 s | 900 | 139 / 24 | 180 | — |
| r5-fp8-offload-blocking | SURVIVED | 600 | 317 / 0 | 601 | — |
| r20w-main-warmup | SURVIVED (no answer in its last 67 s) | 180 | 82 / 0 | 114 | — |
| r20-main-fp8-offload | STALLED at 481 s | 900 | 202 / 0 | 300 | — |

r3 ran before `load.py` could detect a stall, so `run.sh` said it survived; no request was answered after 226 s of its
900, and its snapshot at 21:55:54 UTC ([`stall-snapshot.txt`](runs/r3-bf16-flashinfer-offload/stall-snapshot.txt))
shows the GPU at 100% and 43 W, the engine core waiting. [`summarize.py`](../../fp8/summarize.py) holds it to
`load.py`'s rule after the fact. r20w, main's 180 s warm-up, ended in 67 s without an answer, too short a silence for
that rule, with the GPU at 100% and 44 W as it ended (read by hand at 02:04:04 UTC); its row says so and leaves its
verdict. Times are from the load's start, a stalled run's from its last answer; "plans" counts
the builds of FlashInfer's plans that found the previous one's copy to the GPU still pending, as last logged. A run
named `w` was a warm-up, to compile the version's kernels before its measured run.

## What they show

- **The crash is stock vLLM's.** On 0.32.0, r1, r1b and r18 died at 124, 194 and 184 s with the 2026-10-27 signature: an
  illegal memory access raised at `async_utils.py:174` (`copy_event.synchronize()`), and the offload worker's "Failed
  to synchronize transfer end event". vLLM 0.33.0rc3 died the same way twice, at 128 s with the sweep's other error,
  "operation not supported on global/shared address space", and at 214 s. Main (the nightly build of 564ac0de1) stalled in
  both its runs, its warm-up 114 s in and its measured run 300 s in.
- **It needs the CPU tier and asynchronous scheduling.** Without the CPU tier (r2) or with `--no-async-scheduling` (r7)
  the same load ran its 900 s clean, and with blocking launches (r5), under which every copy is done before the host
  goes on, its 600 s. Were the crashes to come at a constant rate, the stock runs' mean of 168 s would make a clean
  900 s run a one-in-210 event, and 1,800 s one in 46,000.
- **It needs FlashInfer, not FP8.** FP8 on Triton's attention (r8) ran clean, and so did BF16 on FlashAttention (r4), the
  box's serving configuration. BF16 on FlashInfer (r3) stalled at 226 s: the same race can end in a kernel that never
  finishes instead of a fault, the GPU busy at 43 W and the engine core blocked at the same `synchronize()`. FP8 meets
  it because an FP8 cache moves this card's attention to FlashInfer, FlashAttention 2 having no FP8 KV cache here.
- **The kernel that faults.** A GPU core dump of a stock crash (r18, `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` with
  `skip_abort`, read with cuda-gdb 13.0:
  [`cuda-gdb.r18.2786334.1793496884.txt`](runs/r18-fp8-offload-coredump/cuda-gdb.r18.2786334.1793496884.txt)) names it: a warp's illegal address in `flashinfer::PersistentVariableLengthMergeStatesKernel`, which merges split-KV
  partial outputs at indices the plan gives. #61239's commenter found the same kernel in dumps on two RTX 3090s.
- **#64165's wait closes it,** on 0.32.0 and 0.33.0rc3. With the wait neither FP8 (r9, 1,800 s) nor BF16 (r12) crashed or
  stalled on 0.32.0, nor FP8 on rc3 (r15). Its counters: 376 of 30,000, 160 of 20,000 and 223 of 15,000 plans found the
  previous copy pending and waited, one in 80, 125 and 67.
- **Counting is not waiting.** With the same counter and no wait the engine stalled, on 0.32.0 (r10: 20 of its first
  1,477 plans pending, no answer after 164 s) and on rc3 (r14: 117 of 5,000, none after 304 s), its core blocked at the
  output copy's `synchronize()`.
- **The window needs the CPU tier.** Without the CPU tier the counter (r19) found not one plan pending in 10,000, and the
  run was clean. The card has two copy engines, one a direction ([`copy-contention.json`](runs/copy-contention.json)):
  a 64 KiB pinned copy to the GPU took 0.05 ms alone, 0.10 ms beside 1 GiB moving to the host, and 158.8 ms beside
  1 GiB moving to the GPU on another stream, the big copy's whole 158.9 ms. A plan's copy is such a small copy. The
  connector's loads go by DMA (`cuMemcpyBatchAsync`; FP8's 32 KiB pages are over the 28 KiB cutoff of vLLM's zero-copy
  kernel), and a resumed session's prefix, up to 1.8 GB in FP8 at 24k tokens, occupies the card's one host-to-GPU copy
  engine for up to 260 ms, many steps, while the host plans the next step into the same buffer.
- **XQA decode does not avoid it.** With decode on XQA (r16, `VLLM_HAS_FLASHINFER_CUBIN=1`), which takes no per-step
  decode plan, the engine stalled at 159 s all the same; the counter (r17) found 7 of its first 847 plans pending, the
  prefill wrapper's alone, about a fifth as many plans a second as with native decode, and stalled at 127 s.
- **The dump of a stall works too.** `dump_selftest.py` requested a GPU core dump of a kernel still running, and
  cuda-gdb named it ([`dump-selftest.json`](runs/dump-selftest.json)); without `skip_abort` the driver then aborted the
  process, which is why r18 set it. r18 ended in a fault, so it used the dump at the exception.

## The cause

FlashInfer's `plan()` writes a step's schedule (the request, tile and merge indices its kernels read) into one
page-locked buffer per wrapper and copies it to the GPU with `cudaMemcpyAsync`, without a wait
(`include/flashinfer/attention/scheduler.cuh`, lines 504 and 887 in 0.7.0.post1, the decode and prefill plans; FlashInfer
[#6533](https://github.com/flashinfer-ai/flashinfer/issues/6533)). vLLM keeps its wrappers, each with its own buffer,
for the engine's life, and plans them on nearly every step under this load. Until
[#57214](https://github.com/vllm-project/vllm/pull/57214) (merged 2026-09-27, in 0.32.0) its FlashInfer backend read
`seq_lens` back to the host each step, which drained the stream. Since then, under asynchronous scheduling, the host
plans step i+1 while step i's plan copy may still be queued; when it is, the buffer is overwritten first, and step i's
kernels run with step i+1's schedule. vLLM [#61239](https://github.com/vllm-project/vllm/issues/61239) tracks the race,
and #64165 records an event after the plans and waits on it before planning again.

What keeps the copy queued on this card is the CPU tier: its loads hold the one host-to-GPU copy engine for as long as
a prefix takes to move, and without them no plan found its copy pending (r19). That is why one GPU opens the window
here, about one plan in 80 under this load, where #64165's author counted 0 of 2,000 builds on one GPU without a CPU
tier. The kernel that faults is the one that reads the plan's merge indices.

The box's engine decodes on FlashInfer's native wrapper, which plans every step, rather than on XQA: vLLM 0.32.0 picks
XQA on this card only when it reaches NVIDIA's artifactory (`has_nvidia_artifactory()` in `vllm/utils/flashinfer.py`),
and the engine's network namespace has no route out. That is not why the box hits the race: with XQA on, the prefill
plans were enough (r16, r17).

## What it changes

The serving profile keeps its BF16 cache (REPORT §9): stock vLLM crashes or stalls beside the CPU tier on 0.32.0,
0.33.0rc3 and main, the fix is a pull request, open as of 2026-11-02, and an FP8 profile would ship an unreviewed patch
to the engine. vLLM 0.33.0 (2026-11-02) does not ship the wait; when a release does, an FP8 pool is the first
measurement on it. These runs are the data point given on #64165.

## How it ran

| | |
|---|---|
| The box | An RTX 5060 Ti 16 GB (sm_120), driver 580.173.02 (CUDA 13.0), on a PCIe 3.0 x8 link (6.8 GB/s to the GPU and 7.2 GB/s back, in [`copy-contention.json`](runs/copy-contention.json)) |
| The builds | vLLM 0.32.0 from PyPI, with torch 2.13.0+cu130 and FlashInfer 0.7.0.post1 (JIT), the box's engine environment; vLLM 0.33.0rc3 and main (the nightly build of 564ac0de1) with torch 2.14.0 and FlashInfer 0.7.2.post1 |
| The engine | Qwen3-4B-Instruct-2507-FP8 at the box's engine settings ([`offload.yaml`](../../fp8/offload.yaml)): a 5 GiB pool, priority scheduling, 512-token steps with a 256-token long-prefill threshold, prefix caching, asynchronous scheduling (vLLM's default) and the `OffloadingConnector` with 20 GiB, write-through |
| The load | [`load.py`](../../fp8/load.py): 24 agent sessions that resend 8k–24k-token contexts growing each turn, with pauses, so that the pool overflows into the CPU tier and resumed sessions load their prefixes back, beside two streams of short priority-0 turns |

## Files

Each run's directory, `runs/RUN/`, keeps what `run.sh` wrote:

| File | What it holds |
|---|---|
| `result.txt` | The versions, the environment, the verdict and the first fault line |
| `server.log` | The server's log; a stalled run's ends with the engine core's stacks (all but r20's: main's engine core left none, though the hook answers in main's interpreter) |
| `requests.jsonl`, `summary.json`, `load.out` | The load's requests, a line a request, its summary and its output |
| `stall-snapshot.txt` or `stall-nvidia-smi.txt` | A stalled run's GPU state at the stall |
| `cuda-gdb.*.txt` | r18's: cuda-gdb's reading of its dump (the 202 MB dump itself is not kept) |

Beside them, the `runs/queue-*.log` files give the order the runs ran in, `runs/copy-contention.json` the copy engines'
measurement and `runs/dump-selftest.json` the dump's positive control. Home directories are written as `~` and the
runtime directory as `$XDG_RUNTIME_DIR`. [`summarize.py`](../../fp8/summarize.py) prints the two tables:

```sh
python benchmarks/fp8/summarize.py benchmarks/results/2026-10-31-fp8-crash/runs
```
