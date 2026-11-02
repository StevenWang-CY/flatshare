# The quick start on Gray (2026-11-02)

The README's [quick start](../../../README.md#quick-start), run as written on Gray, an RTX 5060 Ti 16 GB as the box
is, from a fresh environment: vLLM 0.32.0 installed, make-room built, then the probe four times on stock vLLM and four
times with make-room in the configuration to deploy.

<!-- table.py table -->
| Run | Stock vLLM | make-room |
|---:|---:|---:|
| 1 | 2,236 ms | 307 ms |
| 2 | 2,222 ms | 300 ms |
| 3 | 2,217 ms | 317 ms |
| 4 | 2,226 ms | 301 ms |

The voice turn's time to its first token. Every run sent the turn, 746–751 tokens, with 3 agent requests of 22,335
tokens running and 8 of the pool's 4,214 blocks free.
<!-- /table.py table -->

- **make-room answered the voice turn in 300–317 ms**, against 2,217–2,236 ms on stock vLLM, in four runs each.
- **The pool was the card's own, 67,440 tokens**, where the deployment fixes 36,400
  ([`engine.yaml`](../../../deploy/phastform/engine.yaml)): the quick start sets none, so vLLM sizes it from the card's
  memory, and the probe sizes its load from the pool it finds.
- **Each run started 31 s after the last.** The configuration to deploy keeps a finished voice turn's prefix on the GPU
  for 30 s (`retain_s`), and the probe sizes its load for an empty pool: a run within 30 s of the last found the
  turn's blocks held, and its third agent request no longer fitted.
- **The module build.py wrote is the serving profile's**, `32db2c089548974d`
  ([`rtx-5060-ti-serve.toml`](../../../deploy/profiles/rtx-5060-ti-serve.toml)).

## How it ran

| | |
|---|---|
| The box | Gray: one RTX 5060 Ti 16 GB, driver 580.178.04, nothing else on the GPU |
| Software | vLLM 0.32.0, torch 2.13.0+cu130 (CUDA 13.0), nvcc 13.0, Python 3.12.3 from uv |
| The tree | v1.14-6-g4386143 |
| Settings | The serving profile's step cap, 256 tokens a step without a per-request threshold; make-room in the configuration to deploy: the KV preemption at 2 a step, the budget cap with a 48-token window, no reserve, the agents' reservations taken, the calls' prefixes kept within 3,584 tokens for 30 s, and a 400 ms hold for an announced turn |
| One change | `--host 127.0.0.1` on each `vllm serve`, so that the shared host's network could not reach the engine |

[The quick start of 2026-10-29](../2026-10-29-quick-start/README.md) ran the pull requests' configuration on
Phastform, at the card's pool of 62,992 tokens there: 258–292 ms against 1,963–1,967 ms.

## Files

| File | What it holds |
|---|---|
| `stock-kv.log`, `make-room-kv.log` | probe.py's output, one line a run |
| `commands.sh` | The commands as run |
| `environment.txt` | The box, the versions, the tree, and each engine's settings and KV pool as it logged them |
| `table.py` | The table above (`python3 table.py`); `python/flatshare_bench/tests/test_quick_start_record.py` holds this README to it |
