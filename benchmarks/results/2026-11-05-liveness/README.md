# The search for stalls of the budget cap and the KV preemption (2026-11-05)

A review of vLLM [#65283](https://github.com/vllm-project/vllm/pull/65283) found a stall on the real scheduler: a request
whose KV load is done held lower-priority prefill back without a bound, and when a request ahead of it failed its
allocation, the waiting loop never reached it, while the prefill it held back was what would have freed the blocks. A fresh
review of [#65282](https://github.com/vllm-project/vllm/pull/65282) and #65283 then found a prefill scheduled nothing doing
the same, and the KV preemption's gate waiting on a load. This record is the search for stalls of that kind, and the rows
the description of #65283 and
[BENCHMARKS](../../../docs/BENCHMARKS.md#the-search-for-stalls-2026-11-05) cite.

## How it ran

| | |
|---|---|
| The scheduler | vLLM's real v1 scheduler on CPU, stepped by the harness with a model's output faked, on a mock KV connector |
| The workloads | Pools of 12 to 48 blocks of 16 tokens, 3 to 10 requests of 3 priorities, half with a KV load of 1 to 14 steps, 30% sharing a prefix, 30% with a streaming-input session, with or without async scheduling and a watermark: [`liveness.py`](liveness.py)'s `base` profile |
| Its `ext` profile | Adds failed loads, aborts, encoder inputs on a small encoder cache, prefill throttling, draft tokens, a single running slot, budgets below the block size and longer outputs |
| The runs | Each workload with the options off, then on; `finished` means every request finished, and a workload stops at 600 steps that schedule nothing with nothing to wait for, or at 4,000 steps |
| The code | #65283's head a17c2cf3f; its plain rebase 2a02f0ee1, before the fixes, as the control; #65282's head 7f048ee9e alone; on the make-room module: this release's and v1.22's, the control, from [`liveness_module.py`](liveness_module.py), with the serving profile's switches |
| The features search | [`liveness_features.py`](liveness_features.py), 60,000 workloads over what those two profiles do not model: LoRA limits, connector lookups answered later, grammars, pause windows, cadenced prefill, sliding-window groups, deferred frees, and `--disable-chunked-mm-input` with a long-prefill threshold: #65283's head, and its revision bca930462 as the control |
| The probe | [`lv_probe.py`](lv_probe.py) repeats each workload the options stall, attributes it to one option, and turns both off at the stall: a stall the options cause then resumes in the next step, and a state the scheduler cannot leave stays stuck |
| Reproducing | `LV_PROFILE=base PYTHONHASHSEED=0 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD python liveness.py OUT.jsonl 0 200000` from a vLLM checkout at the head, with `liveness.py` beside the record; `SEARCH_CONTROL=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD python liveness_features.py 0 60000 > OUT.jsonl` for the features search; the raw rows are not kept, as each seed gives the same row again |

## The rows

| Code | Workloads | Stalls the options cause | Other stalls with the options on | Workloads stuck, options off / on |
|---|---:|---:|---:|---:|
| #65283 before the fix | 400,000 | 241 | 585 | 5,578 / 5,225 |
| #65283 | 400,000 | 0 | 682 | 5,578 / 5,287 |
| #65282 alone | 400,000 | 0 | 408 | 5,578 / 5,634 |
| The make-room module of v1.22 | 200,000 | 153 | 289 | 2,830 / 2,156 |
| The make-room module of v1.23 | 200,000 | 0 | 323 | 2,830 / 2,071 |

The features search: 100 stalls the options cause on the revision before, none on the head, 88 others, 6 of them with a running request at an image longer than the long prefill threshold, which stays there with the options off ([`features-summary.json`](features-summary.json)).

`python3 check.py` prints these from `summary.json`, `module-summary.json` and the probes' rows, and exits 1 if the heads
have a stall the options cause, if the control has none, or if a stall did not repeat.

## What it shows

- **The control finds the review's stall.** Each stall the options cause on the code before the fix has a running request
  held back, and ends when the options are turned off. None remains on the heads, and none on the module this release
  serves.
- **What remains is not the options'.** Every other stall of the two profiles has nothing running and stays stuck with the options off: a
  request at the head of the queue of requests holding KV blocks cannot get the blocks it needs, or start its load, and
  the requests behind it hold what it waits for, such as a streaming session's next chunk ahead of two finished KV loads.
  The options move some workloads into that state and others out of it: more leave it than enter it, as the last column
  shows. That state is vLLM main's own: [`lv_main.py`](lv_main.py) ran every workload stuck with the options off, and the
  first 20,000 of each profile, on main 0b3a608dd itself, and each one has the same outcome, step count and longest idle
  run as on the head with the options off ([`main-compare.txt`](main-compare.txt): the last line is its control, main
  against the head with the options on).
- **The GPU tables stand.** [`probe_replay.py`](probe_replay.py) runs both probes' cases on v1.22's module and this release's, on the
  card's pool, step cap and threshold, as new and with a KV load of 2, 8 or 32 steps: each schedules the same step for step,
  where a run without the switches differs ([`probe-replay.out`](probe-replay.out)).
- **What it cannot show.** The workloads are small and short, so this says which schedules can stall, not how often a real
  engine meets one.
