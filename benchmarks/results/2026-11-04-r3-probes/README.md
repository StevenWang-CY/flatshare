# The pull requests' probes on their revision of 3 November (2026-11-04)

The outputs behind the GPU tables of vLLM [#65282](https://github.com/vllm-project/vllm/pull/65282) and
[#65283](https://github.com/vllm-project/vllm/pull/65283) as revised on 2026-11-03, and [BENCHMARKS](../../../docs/BENCHMARKS.md#the-probes-on-the-pull-requests-revision-of-3-november-2026-11-04)'
section on them: the KV case and the prefill case on Phastform's RTX 5060 Ti in one session, for stock vLLM 0.32.0 (the
control), the KV preemption alone and both pull requests at the budget cap's default 16-token window. They repeat the
[session of 2026-10-25](../2026-10-24-w5-probes/README.md) on the revision's code, without its tails.

## How it ran

| | |
|---|---|
| The engine | Phastform's RTX 5060 Ti with the frozen box's units, pins and pool: vLLM 0.32.0, Qwen3-4B-Instruct-2507-FP8, 2,274 blocks of 16 tokens |
| The code | Release 4c1d1f6 (`v1.20-1-g4c1d1f6` in `run.log`), v1.21's engine: make-room's patches carry the revision, #65282 361b2a81b and #65283 5aae512d0, backported to v0.32.0 |
| The probes | [`probe.py`](../../../python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py) through the engine's socket, on 2026-11-04 from 01:10:48Z to 01:18:48Z, driven by `r3-pr-blocks.sh` |
| The blocks | stock (the frozen box), a (`budget_cap false`, `reserve_tokens 0`) and ab (`reserve_tokens 0`), two preemptions a step in a and ab; each the KV case 4 times, then the prefill case 4 times |
| The check | `python3 check.py` prints each block's rows, and checks that each block's engine came up with its switches, that each log has its four runs and that no block logged an error; it exits 1 if a check fails |

A first attempt at 01:05Z addressed the engine at 127.0.0.1:8000, which the box no longer serves; it measured nothing
and is not kept.

## The rows

Medians of four runs, and their ranges:

| Block | KV case TTFT | Prefill case TTFT | Its 16th token |
|---|---:|---:|---:|
| stock, vLLM 0.32.0 | 4,976 ms (4,906–4,981) | 420 ms (325–541) | 1,374 ms (1,236–1,514) |
| a, #65282 | 284 ms (198–298) | 328 ms (257–470) | 1,242 ms (1,164–1,429) |
| ab, #65282 and #65283 | 274 ms (198–299) | 292 ms (232–325) | 560 ms (518–582) |

- **The control matches the session of 2026-10-25.** Stock vLLM gives the KV case's first token after 4,906–4,981 ms,
  against 4,900–4,971 ms then.
- **The revision schedules these probes as the earlier code did.** The KV preemption gets the turn its first token
  within 0.2–0.3 s instead of 4.9–5.0 s. With the budget cap the prefill case's 16th token comes after 0.52–0.58 s
  instead of 1.2–1.5 s, its tokens 17–20 ms apart instead of 60–66.

## Files

| File | What it holds |
|---|---|
| `run.log` | The session's driver log: each block's engine line, with its scheduler class and `additional_config` as `/server_info` read them back, and its done line |
| `stock-kv.log`, `a-kv.log`, `ab-kv.log` | The KV case's four probe lines in each block |
| `stock-prefill.log`, `a-prefill.log`, `ab-prefill.log` | The prefill case's four probe lines in each block |
| `*-engine-make-room.log` | The engine's make-room counter lines in each block (none on the frozen box) |
| `r3-pr-blocks.sh` | The driver |
| `check.py` | The rows' check |
