# The quick start on the box (2026-10-29)

The README's [quick start](../../../README.md#quick-start), run as written on Phastform, the box, from a fresh
environment: vLLM 0.32.0 installed, make-room built, then the probe four times on stock vLLM and four times with
make-room.

<!-- table.py table -->
| Run | Stock vLLM | make-room |
|---:|---:|---:|
| 1 | 1,963 ms | 258 ms |
| 2 | 1,967 ms | 264 ms |
| 3 | 1,964 ms | 263 ms |
| 4 | 1,967 ms | 292 ms |

The voice turn's time to its first token. Every run sent the turn, 744–750 tokens, with 3 agent requests of 20,863
tokens running and 6 of the pool's 3,936 blocks free.
<!-- /table.py table -->

- **make-room answered the voice turn in 258–292 ms**, against 1,963–1,967 ms on stock vLLM, in four runs each.
- **The pool was the card's own, 62,992 tokens**, where the deployment fixes 36,400
  ([`engine.yaml`](../../../deploy/phastform/engine.yaml)): the quick start sets none, so vLLM sizes it from the card's
  memory. The probe sizes its load from the pool it finds, so stock vLLM's wait depends on the card;
  [BENCHMARKS](../../../docs/BENCHMARKS.md#the-kv-case-make-room) gives it on the deployment's pool.

## How it ran

| | |
|---|---|
| The box | Phastform: one RTX 5060 Ti 16 GB, driver 580.173.02, nothing else on the GPU |
| Software | vLLM 0.32.0, torch 2.13.0+cu130 (CUDA 13.0), nvcc 13.0, Python 3.12.13 from uv |
| The tree | v1.8, 5a7e31c |
| Settings | The serving profile's step cap, 256 tokens a step without a per-request threshold; make-room with the PRs' configuration, the KV preemption at 2 a step and the budget cap with a 48-token window, no reserve |
| One change | `--host 127.0.0.1` on each `vllm serve`, so that the shared host's network could not reach the engine |

## Why the install block has four lines

Each line answers what stopped an attempt before this run, on this box:

| Without | What stopped it |
|---|---|
| `uv venv --python 3.12`: the host's Python is 3.10 | `No matching distribution found for vllm==0.32.0`: vLLM 0.32.0 needs 3.11 to 3.14 |
| `CUDA_HOME` at the venv's CUDA: the host's is 12.9 | `#error "DeepGEMM SM120 kernels require CUDA 13.0 or newer"` |
| `cuda-toolkit[nvcc]==13.0.3.0`: pip took nvcc 13.4 | `#error "CUDA compiler and CUDA toolkit headers are incompatible…"`: torch's runtime is 13.0 |
| `FLASHINFER_EXTRA_LDFLAGS` and the `libcudart.so` link | `cannot find -lcudart`: FlashInfer links from `lib64/`, and pip's CUDA keeps `lib/libcudart.so.13` |

## Files

| File | What it holds |
|---|---|
| `stock-kv.log`, `make-room-kv.log` | probe.py's output, one line a run |
| `commands.sh` | The commands as run |
| `environment.txt` | The box, the versions, the tree, and each engine's settings and KV pool as it logged them |
| `table.py` | The table above (`python3 table.py`); `python/flatshare_bench/tests/test_quick_start_record.py` holds this README to it |
