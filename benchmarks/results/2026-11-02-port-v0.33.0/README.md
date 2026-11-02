# make-room on vLLM 0.33.0 (2026-11-02)

make-room ported to vLLM 0.33.0, which reached PyPI that morning: its three patches rebuilt on v0.33.0, checked on the
CPU against 0.33.0's real scheduler, and run on Phastform's RTX 5060 Ti, the README's quick start on `vllm==0.33.0`
and then the serving variant's own engine. The serving profile stays on 0.32.0, which the rounds measured.

<!-- table.py table -->
| Run | Stock vLLM | make-room |
|---:|---:|---:|
| 1 | 1,788 ms | 207 ms |
| 2 | 1,782 ms | 172 ms |
| 3 | 1,778 ms | 205 ms |
| 4 | 1,784 ms | 167 ms |

The voice turn's time to its first token. Every run sent the turn, 746–749 tokens, with 3 agent requests of 21,439
tokens running and 6 of the pool's 4,044 blocks free.
<!-- /table.py table -->

- **On vLLM 0.33.0 make-room answered the voice turn in 167–207 ms**, against 1,778–1,788 ms on stock vLLM 0.33.0, in
  four runs each, two at 167–172 ms and two at 205–207, with the serving profile's decode hold (version 1.17).
- **Without the hold**, as version 1.16 ran it at the same load, make-room gave 264–267 ms against 1,779–1,792 ms (this
  record at commit 9aa7257).
- **The pool was the card's own, 64,720 tokens**, which 0.33.0 sizes from the card's memory, where 0.32.0's was 67,504:
  the probe sizes its load from the pool, so [0.32.0's quick start](../2026-11-02-quick-start/README.md) sent another
  load (173–179 ms against 1,543–1,556 ms there), and the releases' numbers are no comparison of them.
- **The serving variant's engine ran on 0.33.0**: make-room-serve's own arguments, the box's `engine.yaml` with its
  fixed pool and CPU KV tier, the serve module and just in time, which logged its step model.
  [serving-check](serving-check.txt) found two settings that differ from the serving profile's, the ones 0.33.0 changes:
  the vLLM version and the scheduler module's SHA-256; the decode hold at 48 is the profile's. The probe answered in
  170–175 ms, with 2 agent requests of 17,999 tokens running and 18 of the pool's 2,274 blocks free, against 208–211 ms
  without the hold ([`serve-kv.log`](serve-kv.log)).
- **These are the second pass's runs.** In the first, after the install, vLLM's start-up profile left make-room a pool
  of 63,184 tokens against stock's 64,720, so the probe sent each a different load: stock 1,781–1,786 ms, make-room
  168–200 ms.
- **The module build.py wrote** from the installed vLLM 0.33.0 is `e4f7aba39d34e5c9`, the one it writes from the v0.33.0
  tag's source; the serving profile's, on 0.32.0, is still `7fda82b068c057a1`.

## The port

The backport's 26 commits, cherry-picked onto v0.33.0 on the fork's `v0.33.0-priority-make-room-kv`,
`v0.33.0-priority-prefill-budget-cap` and `v0.33.0-leaseholder-make-room`; `BASE=v0.33.0
python/flatshare_vllm/scripts/make-room-patches.sh` wrote
[`make-room/v0.33.0`](../../../python/flatshare_vllm/src/flatshare_vllm/make-room/v0.33.0) from them.
[`commits.txt`](commits.txt) compares each commit's change to `scheduler.py` with its sources' by patch ID:

- **The pull request's own.** #63350's first commit calls make-room after 0.33.0's local prefill cadence (#54627),
  which can defer a step's prefill, as on main: there it is the pull request's commit unchanged, where 0.32.0, without
  the cadence, needed it adapted. Nine more change `scheduler.py` as their pull request's commits do, and three
  change only tests or the docs.
- **Adapted as on 0.32.0.** Three, whose "Backport:" paragraphs now name 0.33.0: the stats that 0.33.0's
  `SchedulerStats` has no field for, and the queue method its stock request queues lack.
- **Adapted for 0.33.0.** The variant's short-load rule follows 0.33.0's new cap of the external hit for auxiliary
  outputs, so that it judges the load that is left. The variant's nine other commits are its 0.32.0 commits unchanged.
- **The lines.** The patches add the same 849 lines to `scheduler.py` as 0.32.0's, and remove the same 4.

## The checks

| Check | On vLLM 0.33.0 | On vLLM 0.32.0, the same tree |
|---|---|---|
| `make test-vllm` (the build, its patches, the upstream check's tests) | 118 passed; the 32 skipped are 0.32.0's own: `pr54530`, the serving profile's module, the pinned release's checks | 124 passed |
| `make test-make-room-ci` (the behavior suite, the perf check's arithmetic, the gateway's caps and the serve module on the real scheduler; then the trace against the pull requests' diff) | 570 and 1,081 passed, none skipped, in `requirements-cpu-v0.33.0.txt`'s environment (CPU torch 2.14) | 570 and 1,081 passed |
| `make compat-upstream` | v0.33.0 takes its own patches | both pull requests merge into main (30565c3937) cleanly |

## How it ran

| | |
|---|---|
| The box | Phastform: one RTX 5060 Ti 16 GB, driver 580.173.02, nothing else on the GPU |
| Software | vLLM 0.33.0, torch 2.14.0+cu130 (CUDA 13.0), nvcc 13.0, Python 3.12 from uv |
| The tree | v1.16-3-g541ff59, with the serving profile's decode hold: the files the passes ran the same by SHA-256 as at this record's commit |
| Settings | The quick start's: the serving profile's step cap, 256 tokens a step without a per-request threshold; make-room in the configuration to deploy: the KV preemption at 2 a step, the budget cap with a 48-token window, no reserve, the agents' reservations taken and their decodes held through the window, the voice reply's first 48 tokens, the calls' prefixes kept within 3,584 tokens for 30 s, and a 400 ms hold for an announced turn |
| The passes | 15:10–15:28Z the install and the first pass, 15:28–15:38Z the second (the table's), 15:38–15:43Z the serving variant's engine |
| Changes | `vllm==0.33.0` and CUDA toolkit 13.0, the one its torch was built with; the module in `~/make-room-033`; `--host 127.0.0.1` on each serve |

## Files

| File | What it holds |
|---|---|
| `stock-kv.log`, `make-room-kv.log` | probe.py's output on 0.33.0, one line a run |
| `serve-kv.log`, `serving-check.txt` | The serving variant's engine on 0.33.0: probe.py's output, and serving-check against serving.json |
| `commands.sh` | The commands as run |
| `environment.txt` | The box, the versions, the tree, and each engine's settings and KV pool as it logged them |
| `commits.txt` | The port's commits against their sources, by the patch ID of their change to `scheduler.py` |
| `table.py` | The table above (`python3 table.py`); `python/flatshare_bench/tests/test_port_record.py` holds this README to it |
