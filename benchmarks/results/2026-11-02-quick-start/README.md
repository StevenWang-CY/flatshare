# The quick start on Phastform (2026-11-02)

The README's [quick start](../../../README.md#quick-start), run as written on Phastform, the box's RTX 5060 Ti 16 GB,
from a fresh environment: vLLM 0.32.0 installed, make-room built, then the probe four times on stock vLLM and four
times with make-room in the configuration to deploy.

<!-- table.py table -->
| Run | Stock vLLM | make-room |
|---:|---:|---:|
| 1 | 1,545 ms | 272 ms |
| 2 | 1,544 ms | 274 ms |
| 3 | 1,547 ms | 277 ms |
| 4 | 1,546 ms | 276 ms |

The voice turn's time to its first token. Every run sent the turn, 745–752 tokens, with 3 agent requests of 22,367
tokens running and 5 of the pool's 4,218 blocks free.
<!-- /table.py table -->

- **make-room answered the voice turn in 272–277 ms**, against 1,544–1,547 ms on stock vLLM, in four runs each.
- **The decode hold is off.** With the agents' decodes held through the budget cap's window, as the serving profile had
  them until the follow-up rounds turned the hold down, the same probe gave 173–179 ms against 1,543–1,556 ms (this
  record at commit 1533fec): the turn of about 750 tokens prefills in three 256-token steps, which the hold kept clear
  of the three agents' long decodes.
- **The pool was the card's own, 67,504 tokens**, where the deployment fixes 36,400
  ([`engine.yaml`](../../../deploy/phastform/engine.yaml)): the quick start sets none, so vLLM sizes it from the card's
  memory, and the probe sizes its load from the pool it finds.
- **Both engines served twice from the one install.** The first time vLLM's start-up profile left stock vLLM 9.27 GiB
  for its pool and make-room 8.65 GiB, pools of 67,504 and 62,992 tokens, so the probe sent each a different load; the
  second time both had 67,504, and these are those runs. The first time's: 1,549–1,558 ms against 259–294 ms.
- **Each run started 31 s after the last.** The configuration to deploy keeps a finished voice turn's prefix on the GPU
  for 30 s (`retain_s`), and the probe sizes its load for an empty pool: a run within 30 s of the last would find the
  turn's blocks held, and its third agent request would no longer fit.
- **The module build.py wrote is the serving profile's**, `7fda82b068c057a1`
  ([`rtx-5060-ti-serve.toml`](../../../deploy/profiles/rtx-5060-ti-serve.toml)).

## How it ran

| | |
|---|---|
| The box | Phastform: one RTX 5060 Ti 16 GB, driver 580.173.02, nothing else on the GPU |
| Software | vLLM 0.32.0, torch 2.13.0+cu130 (CUDA 13.0), nvcc 13.0, Python 3.12.13 from uv |
| The tree | v1.14-11-g39e7637, with the make-room command as the README gives it from the next commit, which turns the decode hold off (`decode_hold_tokens 0`); nothing the quick start builds or runs differs between the two |
| Settings | The serving profile's step cap, 256 tokens a step without a per-request threshold; make-room in the configuration to deploy: the KV preemption at 2 a step, the budget cap with a 48-token window, no reserve, the agents' reservations taken, no decode hold, the calls' prefixes kept within 3,584 tokens for 30 s, and a 400 ms hold for an announced turn |
| One change | `--host 127.0.0.1` on each `vllm serve`, so that the shared host's network could not reach the engine |

[The quick start of 2026-10-29](../2026-10-29-quick-start/README.md) ran the pull requests' configuration on the same
card, whose pool came to 62,992 tokens then: 258–292 ms against 1,963–1,967 ms.

## Files

| File | What it holds |
|---|---|
| `stock-kv.log`, `make-room-kv.log` | probe.py's output, one line a run |
| `commands.sh` | The commands as run |
| `environment.txt` | The box, the versions, the tree, and each engine's settings and KV pool as it logged them |
| `table.py` | The table above (`python3 table.py`); `python/flatshare_bench/tests/test_quick_start_record.py` holds this README to it |
