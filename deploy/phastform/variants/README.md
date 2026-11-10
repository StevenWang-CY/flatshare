# Engine variants of the frozen box

What the box's engine can run, for an operator who switches it and for a reader of E2's arms. The box serves
`make-room-serve`, which [the serving profile](#the-serving-profile) renders. An experiment runs the frozen box,
[`engine.yaml`](../engine.yaml) and the units in [`systemd/`](../systemd), changed in two ways that an arm file's
`[engine]` table names ([`benchmarks/e2`](../../../benchmarks/e2)):

- **a variant**, a directory here of systemd drop-ins, one `UNIT.conf` for each unit it changes, linked as
  `UNIT.service.d/variant-NAME.conf`;
- **engine flags**, which a generated drop-in, `flatshare-engine.service.d/variant-flags.conf`, adds to the engine's
  command line. vLLM takes the command line over `engine.yaml`, and the last of a repeated flag wins, so they beat a
  variant's own too.

```sh
deploy/phastform/variant.sh use NAME|frozen [FLAG...]   # run exactly that variant, with those flags
deploy/phastform/variant.sh on|off NAME                 # add or remove one variant, for work outside E2
deploy/phastform/variant.sh measure on|off              # the measurement drop-in
systemctl --user cat flatshare-engine                   # what the box runs
```

`variant.sh use` removes every other variant, links NAME's drop-ins, writes the flags and restarts what changed, which
for the engine alone takes about 2 minutes with warm caches. Before each point `benchmarks/e2/campaign.sh` calls it
with what `flatshare-bench e2 engine ARM` prints, and `flatshare-bench e2 run` checks the engine's own `/server_info`
against the arm. Every variant serves on the box's engine socket, `$XDG_RUNTIME_DIR/flatshare/engine.sock`.

## Variants

| Variant | Units | What it changes | E2 arm |
|---|---|---|---|
| `frozen` | none | Nothing: the frozen box | 1, and every arm that changes flags only |
| [`mps`](mps) | engine, STT, TTS | The three servers as clients of one CUDA MPS daemon instead of time-slicing (note 1) | the MPS factor |
| [`pr54530`](pr54530) | engine | vLLM [#54530](https://github.com/vllm-project/vllm/pull/54530) ported to 0.32.0: a waiting request preempts the lowest-priority running one once the running queue is full (note 2) | 7a, with the PR |
| [`sglang`](sglang) | engine | SGLang 0.5.22 in the engine's place, with the same model and step cap (note 3) | 7b |
| [`make-room`](make-room) | engine | The pull requests [#65282](https://github.com/vllm-project/vllm/pull/65282) and [#65283](https://github.com/vllm-project/vllm/pull/65283) proposed to vLLM for the KV case of [#40004](https://github.com/vllm-project/vllm/issues/40004), and a reserve of the variant's own (note 4) | 9, engine side |
| [`make-room-serve`](make-room-serve) | engine | The box's serving configuration ([below](#the-serving-profile)) | none |
| [`agents-only`](agents-only) | engine, STT, TTS | A card for the agents alone ([below](#a-card-for-the-agents-alone)) | the resource gate's agents-only points |
| [`make-room-v03`](make-room-v03) | engine | `make-room` as the v0.3 tag built it (note 5) | 9, W5's control (`9-pr-ab48-v03`) |

1. `mps` has not run since the servers moved into network namespaces of their own (`deploy/bin/in-private-net`). Run it
   once before measuring with it.
2. `pr54530` acts under priority scheduling. The engine loads the module that `build.py` writes at each start with
   `--scheduler-cls flatshare_pr54530_scheduler.AsyncScheduler`, and `--max-num-seqs 12` lets the queue fill, since the
   PR acts on the request count, never on KV pressure.
3. SGLang serves no Unix socket, so `sglang` listens on loopback port 8000, which the host's other users can reach
   while it runs, and a proxy serves the engine's socket from it. Its step cap is `--chunked-prefill-size 512`, its KV
   pool comes to 33,683–34,679 tokens of the 36,400 asked for, and its priority scheduling preempts on any priority
   gap. `install.sh` builds its environment once.
4. `make-room`'s unit runs arm 9's values: 2 preemptions a step, a 16-token window and a 2,048-token reserve. The
   engine loads the module that `build.py` writes at each start with
   `--scheduler-cls flatshare_make_room_scheduler.AsyncScheduler`, and `--max-num-seqs` stays the frozen box's.
   [`python/flatshare_vllm`](../../../python/flatshare_vllm/README.md) holds its patches and its checks on a GPU.
5. `make-room-v03` keeps the v0.3 tag's `build.py` byte for byte and its unit but for the paths and the names. Its
   module, `flatshare_make_room_v03_scheduler`, is named so that the engine's scheduler class tells it apart.

Under the frozen box's asynchronous scheduling, `pr54530` and `make-room` load the stock `AsyncScheduler` over their
patched `Scheduler`, so with and without each change differ by the change alone.

## The switches

make-room's switches are keys of vLLM's `--additional-config`, read once at engine start and each off by default. Its
[`flatshare-engine.conf`](make-room/flatshare-engine.conf) sets arm 9's values and says what each switch does. An arm
file ablates one with an engine flag, which comes after the variant's and wins:

```toml
flags = ["--additional-config.make_room.reserve_tokens", "0"]
```

| Variant key | What it does | The pull requests' flag |
|---|---|---|
| `make_room.max_preemptions_per_step` | The KV preemption, at most this many victims a step | `--max-priority-preemptions-per-step` ([#65282](https://github.com/vllm-project/vllm/pull/65282)) |
| `make_room.budget_cap`, `make_room.budget_cap_tokens` | The budget cap and its window, 16 tokens by default | `--priority-prefill-budget-cap`, `--priority-prefill-budget-cap-tokens` ([#65283](https://github.com/vllm-project/vllm/pull/65283)) |
| `make_room.reserve_tokens` | The reserve that requests of a positive priority value leave free | — |
| `make_room.retain_tokens`, `make_room.retain_s` | The calls' prefixes kept on the GPU between turns, for 30 s by default | — |
| `make_room.take_reservations` | A request leaves free only its own or higher priorities' reservations | — |
| `make_room.decode_hold_tokens` | The decode hold over a running request's first tokens | — |
| `make_room.recompute_tokens` | A realtime request's short CPU-tier load is recomputed | — |
| `make_room.hint_hold_ms` | A call's announced turn holds lower-priority prefill back | — |

The keys without a flag are the variant's only, and `take_reservations` and `decode_hold_tokens` need the budget cap.
The engine start fails on an unknown key, a bad value, or a switch under a policy other than priority. The engine
reports the switches under `additional_config` in `/server_info`, and `flatshare-bench e2 run` requires them to be
exactly what the variant's and the arm's flags build.

## The serving profile

The configuration to deploy is a serving profile's,
[`deploy/profiles/rtx-5060-ti-serve.toml`](../../profiles/rtx-5060-ti-serve.toml), the frozen box's profile with a
`[serving]` table. `deploy/profile.py` renders it into two files, which `make profiles` checks and
`make profiles UPDATE=1` writes again:

- [`make-room-serve/flatshare-engine.conf`](make-room-serve/flatshare-engine.conf), the variant that runs it, with every
  setting in its drop-in and the serve module's scheduler class
  ([`python/flatshare_vllm`](../../../python/flatshare_vllm/README.md#the-build));
- [`serving.json`](../serving.json), every setting the box must run, which `profile.py effective rtx-5060-ti-serve`
  prints, with the modules' SHA-256, the voice worker's two calls and the 24 agent sessions beside them.

Each engine value, which [PRODUCTION.md's engine table](../../../docs/PRODUCTION.md#the-engine) explains, is one that
a measured point ran with: `opt-disc-hold-s24` in
[the optimization rounds](../../../benchmarks/results/2026-11-02-opt-rounds/README.md#the-decode-hold-and-speculative-decoding),
and for the queue and the copy `queue-disc-s24` in
[the envelope rounds](../../../benchmarks/results/2026-11-08-envelope/README.md).
[`test_serving.py`](../../../python/flatshare_bench/tests/test_serving.py) holds the profile to these points, and to
the sweep's and W5's but for the settings added since.

**Each start holds the engine to the profile.** A production box serves no `/server_info`, so the engine says what it
runs. Once it has installed what the profile asks for, the serve module writes the engine's whole configuration to the
file that `FLATSHARE_ENGINE_EFFECTIVE` names, which the variant removes before each start. After the unit's wait for
`/health` it runs `deploy/bin/serving-check serving.json --engine FILE --pid $MAINPID`, which:

- prints every setting with the value the engine runs, where they differ;
- refuses a file that neither the unit's main process nor its engine core wrote;
- holds the release's patches to the profile's.

An engine that runs anything else, a flag over the profile's or another module, fails its start, and the journal names
each setting. A release runs the same check after it restarts the engine ([RUNBOOK.md](../../../ops/RUNBOOK.md#deploy)),
and holds the running gateway's `GET /nodes` to the 24 sessions (`serving-check --nodes`). To serve with it, on the
box:

```sh
deploy/phastform/variant.sh use make-room-serve     # the engine restarts; the
                                                    # unit's start runs the check
deploy/phastform/variant.sh measure off             # no development routes on a box that serves
python3 ~/flatshare/src/deploy/bin/serving-check ~/flatshare/src/deploy/phastform/serving.json \
  --engine $XDG_RUNTIME_DIR/flatshare/engine-effective.json \
  --pid "$(systemctl --user show -p MainPID --value flatshare-engine)"
```

## The measurement drop-in

`flatshare-engine.service` runs vLLM without its development mode, so a production box serves none of its development
or RL routes. A measured box needs two, `/server_info` for the engine's configuration and `/reset_prefix_cache`
between points, and [`measure/flatshare-engine.conf`](../measure/flatshare-engine.conf) turns them on with
`VLLM_SERVER_DEV_MODE=1`.

- **`variant.sh measure on|off`** links it as `flatshare-engine.service.d/measure.conf` or removes it, and restarts a
  running engine. `variant.sh use`, `on` and `off` keep it, `campaign.sh` turns it on before each point, and
  `flatshare-bench e2 run` refuses a box without it.
- **`make-room-serve` turns the routes off again** with `VLLM_SERVER_DEV_MODE=0`, so a box switched to serving after a
  campaign serves none of them even with the drop-in still linked.

What vLLM 0.32.0 serves (`vllm/entrypoints/launchers/api_server/routers.py`):

| Served on | Routes |
|---|---|
| Every box | The OpenAI API under `/v1`, `/inference/v1/generate`, `/health`, `/ping`, `/load`, `/version`, `/metrics`, `/tokenize`, `/detokenize`, `/invocations`, `/kv_event_sources`, the render routes and `/scale_elastic_ep` |
| Measured boxes: development | `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache`, `/collective_rpc`, `/server_info`, `/sleep`, `/wake_up`, `/is_sleeping`, `/release_kv_cache_memory` |
| Measured boxes: RL | `/pause`, `/resume`, `/is_paused`, `/abort_requests`, `/init_weight_transfer_engine`, `/shutdown_weight_transfer_engine`, `/start_weight_update`, `/start_draft_weight_update`, `/update_weights`, `/finish_weight_update`, `/update_weight_version`, `/weight_info`, `/weight_checker`, `/get_world_size` |

The `sglang` variant serves SGLang's own routes, `/server_info` and `/flush_cache` among them, with or without the
drop-in.

## A card for the agents alone

`agents-only` serves the agents and no call. It is the resource gate's best practical dedicated card, against which a
shared card's background work is weighed ([RESOURCE-GATE.md](../../../benchmarks/quality/RESOURCE-GATE.md)).

- **No speech servers.** Its STT and TTS drop-ins hold `ConditionPathExists=!/`, which never holds, so systemd skips
  both units without a failure, until a switch to another variant starts them again.
- **The pool in their memory.** `--kv-cache-memory 9019431321` adds the 3.4 GiB the speech servers hold
  ([ARCHITECTURE.md §5.1](../../../docs/ARCHITECTURE.md#51-the-frozen-box)) to the box's 5 GiB, 61,152 tokens where the
  box has 36,400. The engine then takes about 14.0 GiB of the card's 15.9, a budget of 0.88 under
  [R2's ceiling](../../profile.py) of 0.95. vLLM wants that share free at its start, so the engine refuses to start
  beside resident speech servers.
- **vLLM's scheduling for the card.** It runs first come, first served, with 2,048-token steps and no per-request
  prefill cap, since every request is the agents'. Without calls the box's 512-token steps only slow the agents, whose
  requests are 99.5% prompt tokens. In E2's discriminating scenario 1,024-token steps with no per-request cap gave them
  9% more than tuned vLLM, 3,888 against 3,552 tokens a second over the calls' first 240 s at the median of five runs
  ([W4's frontier](../../../benchmarks/results/2026-10-23-e2/report/frontier.md#workload-weka-x32-long-2-callers)).

An arm file names it as `[engine] variant = "agents-only"`, and `flatshare-bench e2 run` checks the pool by its bytes
([the E2 harness](../../../benchmarks/e2/README.md)).

## Engine flags of the E2 arms

[The variants' table](#variants) gives the arms' variants, and these are the flags the arms add. The frozen box runs
`--scheduling-policy priority --max-num-batched-tokens 512 --long-prefill-token-threshold 256`.

| Arm | `[engine]` | What it sets |
|---|---|---|
| 2, stock | `--scheduling-policy fcfs --max-num-batched-tokens 2048 --long-prefill-token-threshold 0` | vLLM's defaults on this GPU, with no priority sent (note 1) |
| 3, priority | `--max-num-batched-tokens 2048 --long-prefill-token-threshold 0` | Priority scheduling with vLLM's default chunking |
| 4, step cap | `--max-num-batched-tokens B --long-prefill-token-threshold T` | B in {512, 1024} and T in {256, 512, 1024}, five distinct cells, since a threshold at or above B never binds |
| 5, watermark | best arm 4 + `--watermark W` | W in {0.05, 0.1, 0.2}, the share of KV blocks kept free when the scheduler admits a request |
| 5, queued tokens | best arm 4 + `--max-num-queued-tokens N` | New requests of every class get 503, counted as `vllm:admission_rejections_total`, while the prompt tokens in prefill exceed N |
| 7a, without the PR | `--max-num-seqs 12` | The `pr54530` engine without the PR |
| 9, an ablation | variant `make-room` + `--additional-config.make_room.KEY VALUE`: `max_preemptions_per_step 0`, `budget_cap false` or `reserve_tokens 0` | The engine side without one mechanism; all three off is the frozen box's scheduling |
| 9, the change's configurations | variant `make-room` + `reserve_tokens 0`, with `budget_cap false` for #63350 alone, or `budget_cap_tokens 48` | #63350 alone, both rules with a 16-token window, or both with a 48-token window, the configuration to deploy (note 2) |

1. Arm 2 sends no priority (`upstream_priority = []`), and its 2,048-token step cap is the API server's default below
   70 GiB of GPU memory, so it is the engine out of the box. vLLM 0.32's FCFS scheduler would ignore a priority rather
   than reject it.
2. These are the change as #65282 and #65283 propose it, under their first numbers.
