# W5 make-room probe records (2026-10-24)

The outputs behind [BENCHMARKS](../../../docs/BENCHMARKS.md#w5-the-fixed-make-room-on-phastform)' W5 probe tables and the GPU numbers vLLM [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351) cite. Every run was on Phastform's RTX 5060 Ti, the GPU every E2 point ran on, with the frozen box's units, pins and pool (vLLM 0.32.0, Qwen3-4B-Instruct-2507-FP8, 2,274 blocks of 16 tokens), with [`probe.py`](../../../python/leaseholder_vllm/src/leaseholder_vllm/make-room/probe.py) and [`tail.py`](../../../python/leaseholder_vllm/src/leaseholder_vllm/make-room/tail.py) (then in deploy/phastform/variants/make-room), on 2026-10-24 and 25. The earlier probes, on Gray before W5's fixes, are in [`2026-10-23-make-room-probes`](../2026-10-23-make-room-probes). Paths under the account's home directory are written from `~`.

`python3 check.py` prints each block's rows and checks them: that each block's engine came up, that it read every line of each probe log, and that each tail run's summary line agrees with its per-turn lines (within their 1 ms rounding). Its tables' percentiles are inclusive, as E2's report computes them; the summary lines were printed by tail.py before it took the inclusive method too (a3465a4), so it checks them by the exclusive one. It exits 1 if a check fails.

## `gpu-check/`: the fixed module beside v0.3's

The GPU check of make-room after W5's fixes, in the PRs' configuration: two preemptions a step, the budget cap with its 16-token window, no reserve. One block per module, each the KV case 3 times, the prefill case 4 times, tail.py fresh and growing twice each, and the engine's make-room counter lines; the make-room CPU suite ran first on the engine's own venv (`suite.log`, `final2-suite.log`). The drivers are `w5-final-probes.sh` (blocks final and v03), `w5-loadhold.sh` and `w5-final2.sh`; `run.log` is their log, with each block's `additional_config` as `/server_info` read it back.

| Block | Module (`build.py`) | What it is |
|---|---|---|
| final | `build-final.py` | Every W5 fix as first written (Lane A's e277e85): a KV load in flight holds nothing until it is done |
| v03 | v0.3's (the `make-room-v03` variant's) | The module E2 measured before W5 |
| loadhold | `build-loadhold.py`, which `make-loadhold.py` writes from `build-final.py` | final with a load in flight holding lower-priority prefill for its first `budget_cap_tokens` steps, and the scheduler's logger kept |
| final2 | 66deebb's `deploy/phastform/variants/make-room/build.py` | The module as merged (Lane A's 79163c6): the bounded load hold, the logger kept |

- **What it found.** final cost the calls 125–131 ms at p50 and 175–202 ms at p95 against v0.3, over tail.py's fresh and growing prompts; the bounded hold gave it back (loadhold, then final2), and still bounds a load that never completes.
- **The counter lines.** final's are missing: its generated module rebound the scheduler's logger, so its lines never reached the journal (fixed by the logger change loadhold tried and final2 carries). v0.3 logs none.
- **The ERROR lines** in `final-engine-make-room.log` and `final2-engine-make-room.log` are the engine instance that `variant.sh use` started, stopped by the block's restart before its probes, which rebuilds the module: another pid, before the block's ready line. The engines measured logged no error.

## `pr-probes/`: vLLM 0.32.0, #63350 alone and both PRs, one session

The probes the PRs cite, in one session on 2026-10-25 (02:36–03:47Z), after the PR campaign and on the make-room it ran (`DEPLOYED` in `run.log`): stock vLLM 0.32.0 (block stock, the frozen box), the KV preemption alone (block a: `budget_cap false`, `reserve_tokens 0`) and both PRs at the budget cap's default 16-token window (block ab: `reserve_tokens 0`), two preemptions a step in both. Each block ran the KV case 4 times, the prefill case 4 times and tail.py fresh and growing twice each, and kept the engine's make-room counter lines. The driver is `w5-pr-blocks.sh`; `run.log` is its log, with each block's scheduler class and `additional_config` as `/server_info` read them back.

- **The KV case.** A voice turn behind two 18k-token agent requests in a full pool has its first token after 4,900–4,971 ms on stock vLLM, 219–290 ms with #63350 alone and 216–294 ms with both. In one run with both, the reply ended before its 16th token (`16 tokens —`): the voice turn is a chat completion that may stop early, and this case measures its first token.
- **The prefill case.** Behind three agent prefills, the 16th token comes after 1,500 ms on stock vLLM, 1,350 ms with #63350 alone and 575 ms with both (medians of four), 60–66, 60–65 and 16–20 ms a token.
- **The load check.** Over two runs each, the calls' TTFT p95 is 6,830 and 7,158 ms on stock vLLM (fresh and growing prompts), 4,111 and 1,073 ms with #63350 alone, and 388 and 358 ms with both.
