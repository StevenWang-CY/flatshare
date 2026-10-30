# Engine variants of the frozen box

Every experiment runs on the frozen box ([`engine.yaml`](../engine.yaml) and the units in [`systemd/`](../systemd)). An E2 point changes it in two ways, both named by its arm file's `[engine]` table ([`benchmarks/e2`](../../../benchmarks/e2)):

- **a variant**, a directory here of systemd drop-ins, one `UNIT.conf` per unit it changes, linked as `UNIT.service.d/variant-NAME.conf`;
- **engine flags**, added to the engine's command line by a generated drop-in, `flatshare-engine.service.d/variant-flags.conf`, which sets `FLATSHARE_ENGINE_FLAGS`. vLLM takes the command line over `engine.yaml`, and the last of a repeated flag wins, so they beat a variant's own flags too.

`variant.sh use NAME|frozen [FLAG...]` makes the box run exactly that: it removes every other variant, links NAME's drop-ins, writes the flags and restarts what changed. When only the engine's drop-ins change and the audio servers run, only the engine restarts (about 2 minutes with warm caches); otherwise the whole box does. It does nothing when the box already runs NAME with those flags. `benchmarks/e2/campaign.sh` calls it before each point with what `flatshare-bench e2 engine ARM` prints, and `flatshare-bench e2 run` checks the result against the engine's own `/server_info`. `systemctl --user cat flatshare-engine` shows what the box runs.

`variant.sh on|off NAME` adds or removes one variant and restarts the box, for work outside E2.

Every engine serves on the box's engine socket, `$XDG_RUNTIME_DIR/flatshare/engine.sock`, in a directory only the box's user may enter: the frozen box's unit passes `--uds ${FLATSHARE_ENGINE_SOCKET}`, a variant that starts its own command passes it too, and the engine check leaves it out of a variant's flags, as it does `--config`.

## The measurement drop-in

`flatshare-engine.service` runs vLLM without its development mode, so a production box serves none of its development or RL routes. A measured box needs two of them: `flatshare-bench e2 run` reads the engine's configuration from `/server_info` and empties its prefix cache with `/reset_prefix_cache` between points, and the drills read `/server_info`. [`measure/flatshare-engine.conf`](../measure/flatshare-engine.conf) sets `VLLM_SERVER_DEV_MODE=1`; `variant.sh measure on|off` links it as `flatshare-engine.service.d/measure.conf` or removes it and restarts a running engine (only the engine), `variant.sh use`, `on` and `off` keep it, and `benchmarks/e2/campaign.sh` turns it on before each point. The serving variant, `make-room-serve`, sets `VLLM_SERVER_DEV_MODE=0` after it, so a box switched to serving after a campaign serves none of these routes even with the drop-in still linked; `variant.sh measure off` removes it. Without it, `flatshare-bench e2 run` fails at its engine check and names the command.

What vLLM 0.32.0 serves (`vllm/entrypoints/launchers/api_server/routers.py`, and the engine's own list of routes at start):

- **Production and measured boxes:** the OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, `/v1/messages` and the others under `/v1`), `/inference/v1/generate`, `/health`, `/ping`, `/load`, `/version`, `/metrics`, `/tokenize`, `/detokenize`, `/invocations`, `/kv_event_sources`, the render routes and `/scale_elastic_ep`.
- **Measured boxes only (the drop-in):**
  - Development: `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache`, `/collective_rpc`, `/server_info`, `/sleep`, `/wake_up`, `/is_sleeping`, `/release_kv_cache_memory`.
  - RL: `/pause`, `/resume`, `/is_paused`, `/abort_requests`, `/init_weight_transfer_engine`, `/shutdown_weight_transfer_engine`, `/start_weight_update`, `/start_draft_weight_update`, `/update_weights`, `/finish_weight_update`, `/update_weight_version`, `/weight_info`, `/weight_checker`, `/get_world_size`.

The `sglang` variant serves SGLang's own routes, `/server_info` and `/flush_cache` among them, with or without the drop-in.

## The serving profile

`make-room` is the experiments' variant: its unit keeps arm 9's values, and each E2 arm changes one with engine flags. The configuration to deploy is a serving profile's instead, [`deploy/profiles/rtx-5060-ti-serve.toml`](../../profiles/rtx-5060-ti-serve.toml), the frozen box's profile (`rtx-5060-ti`) with a `[serving]` table. `deploy/profile.py` renders it into this directory beside the frozen box's files, and `make profiles` keeps both current:

- [`variants/make-room-serve/flatshare-engine.conf`](make-room-serve/flatshare-engine.conf), the variant that runs it;
- [`serving.json`](../serving.json), the effective configuration the box must run, which `profile.py effective rtx-5060-ti-serve` prints:
  - **the engine:** the vLLM version (0.32.0), the model and its revision, the tokenizer and its revision (the model's, vLLM's default), the context;
  - **scheduling:** the policy (priority), the step cap (256) and long-prefill threshold (0, none), the profile's own, which the variant passes over `engine.yaml`'s, the frozen box's; the request slots and asynchronous scheduling vLLM sets on this card (256, on);
  - **memory:** the pool (2,275 blocks of 16 tokens, 36,400 tokens, from 5 GiB), the GPU budget (0.70), prefix caching, the CPU tier (`OffloadingConnector`, 20 GiB, written through);
  - **the change:** make-room's four switches, the scheduler class, the SHA-256 of the module `build.py` writes from make-room's patches on vLLM 0.32.0 and of each patch, and of the serve module;
  - **the voice worker's admission** in serve mode: the calls it takes at once (two) and its audio configuration;
  - **the agents' sessions** a node that carries the calls may run beside them (four, the card's envelope in [the rounds](../../../benchmarks/results/2026-10-28-rounds/README.md#beside-the-gate-the-margin)), which the gateway's `[[nodes]]` hold them to and a release holds the running gateway's `GET /nodes` to (`serving-check --nodes`, on cyan while its worker serves).

Each engine value is the one the sweep's point of the configuration to deploy at that step cap ran with (`sweep-disc-pr-ab48-b256-t0-s24` in [the sweep's screens](../../../benchmarks/results/2026-10-27-sweep/README.md)), which its `engine.json` records but for the GPU budget, which a record's reduction leaves out. W5's PR points ran the same but for the step cap, the frozen box's. A test holds both, and another that a point's record keeps every engine setting of `serving.json` but those it names, each with why ([`test_serving.py`](../../../python/flatshare_bench/tests/test_serving.py)); `test_build.py` holds the module's SHA-256 to a build from v0.32.0's source.

A production box serves no `/server_info`, so the engine says what it runs itself. The serve module's scheduler, once make-room's is built in the engine core, writes to the file `FLATSHARE_ENGINE_EFFECTIVE` names, whole: the switches as the scheduler holds them, its policy, step cap and pool, the model, revisions and context, the KV connector, the vLLM version, and the SHA-256 of the module it extends and of its own file.

The variant removes that file before each start, and after the unit's wait for `/health` runs `deploy/bin/serving-check serving.json --engine FILE --pid $MAINPID`, which:

- prints every setting with the value the engine runs where they differ;
- refuses a file the unit's main process or its engine core did not write;
- holds the release's patches to the profile's.

An engine that runs anything else, a flag over the profile's or another module, fails its start, and the journal names each setting. A release runs the same check once it has restarted the engine ([`ops/RUNBOOK.md`](../../../ops/RUNBOOK.md#deploy)). To serve with it, on the box:

```sh
deploy/phastform/variant.sh use make-room-serve     # the engine restarts; the unit's start runs the check
deploy/phastform/variant.sh measure off             # no development routes on a box that serves
python3 ~/flatshare/src/deploy/bin/serving-check ~/flatshare/src/deploy/phastform/serving.json \
  --engine $XDG_RUNTIME_DIR/flatshare/engine-effective.json --pid "$(systemctl --user show -p MainPID --value flatshare-engine)"
```

E2 does not run it: no arm names the variant, and `flatshare-bench e2 run` checks a point's engine against its arm's.

## Variants

| Variant | Units | What it changes | E2 arm |
|---|---|---|---|
| `frozen` | none | Nothing: the frozen box | 1, and every arm that changes flags only |
| [`mps`](mps) | engine, STT, TTS | The three servers run as clients of one CUDA MPS daemon (`flatshare-mps.service`) instead of time-slicing (note 1) | the MPS factor |
| [`pr54530`](pr54530) | engine | vLLM PR [#54530](https://github.com/vllm-project/vllm/pull/54530) ported to 0.32.0: a waiting request preempts the lowest-priority running one once the running queue is full (note 2) | 7a (with the PR) |
| [`sglang`](sglang) | engine | SGLang 0.5.22 in the engine's place, with the same model and step cap (note 3) | 7b |
| [`make-room`](make-room) | engine | The change for the KV case of vLLM [#40004](https://github.com/vllm-project/vllm/issues/40004), proposed to vLLM as pull requests [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351) without the reserve (note 4) | 9, engine side |
| [`make-room-serve`](make-room-serve) | engine | The box's serving configuration, rendered from the serving profile [`rtx-5060-ti-serve`](../../profiles/rtx-5060-ti-serve.toml) ([above](#the-serving-profile)) (note 5) | none: the configuration to deploy, which the sweep's `9-pr-ab48-b256-t0` point measured, and W5's `9-pr-ab48` points at the frozen box's step cap |
| [`agents-only`](agents-only) | engine, STT, TTS | A dedicated background card, the resource gate's best practical dedicated arrangement ([RESOURCE-GATE.md](../../../benchmarks/quality/RESOURCE-GATE.md)) (note 6) | the agents-only points of the resource gate |
| [`make-room-v03`](make-room-v03) | engine | W5's control: `make-room` as the v0.3 tag built it (note 7) | 9, W5's control (`9-pr-ab48-v03`) |

1. `mps` has not run since the servers moved into network namespaces of their own (`deploy/bin/in-private-net`), where each client is root in its user namespace and reaches the daemon through its pipe directory: run it once before measuring with it.
2. `pr54530` acts under priority scheduling. At each start [`build.py`](../../../python/flatshare_vllm) applies the port, a patch of 0.32.0's scheduler, the engine loads the module with `--scheduler-cls flatshare_pr54530_scheduler.AsyncScheduler`, and `--max-num-seqs 12` lets the queue fill, since the PR acts on the request count, never on KV pressure. [`probe.py`](../../../python/flatshare_vllm/src/flatshare_vllm/pr54530/probe.py) is the check.
3. `sglang` serves on its loopback port 8000 behind the engine's socket: SGLang serves no Unix socket, so `flatshare-engine-proxy.socket` proxies the socket to the port while the variant runs, and the host's other users can reach the port meanwhile. Its step cap is `--chunked-prefill-size 512`, without vLLM's 256-token threshold per request. It has a KV pool asked for at 36,400 tokens, which SGLang's static memory fraction brings to 33,683–34,679, and priority scheduling that preempts on any priority gap. Its environment is built once with `install.sh`.
4. `make-room`: the pull requests are open, with no maintainer review as of 2026-10-29.
   - **The rules.** Under priority scheduling, the highest-priority waiting request the scheduler can serve preempts running requests of a lower priority when it lacks KV (the KV preemption), lower-priority prefill sits out the steps of its first tokens (the budget cap), and lower classes leave a KV reserve free for priority 0. [ARCHITECTURE.md §2](../../../docs/ARCHITECTURE.md#2-the-change-63350-and-63351) states the rules.
   - **Its unit** runs arm 9's values: 2 preemptions a step, a 16-token window and a 2,048-token reserve.
   - **Its module.** At each start [`build.py`](../../../python/flatshare_vllm) applies the pull requests' scheduler diff, backported to 0.32.0, and the variant's own patch, and the engine loads the module with `--scheduler-cls flatshare_make_room_scheduler.AsyncScheduler`; `--max-num-seqs` stays the frozen box's.
   - **Its checks.** [`probe.py`](../../../python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py) checks a voice turn in a full KV pool (`kv`) and behind agent prefills (`prefill`), `tail.py` the calls' first-token tail beside closed-loop agents (both sized from the engine's pool and context), and [`knee.sh`](../../../python/flatshare_vllm/src/flatshare_vllm/make-room/knee.sh) sweeps the budget cap's window against whole replies.
5. `make-room-serve` is `make-room` as the report recommends deploying it, #63350 and #63351 as W5 measured them, at the step cap the sweep chose.
   - **In the drop-in**, rather than in engine flags over arm 9's values: its switches (2 preemptions a step, the budget cap with a 48-token window, no reserve) and its step cap (`--max-num-batched-tokens 256 --long-prefill-token-threshold 0`, over `engine.yaml`'s).
   - **Its scheduler** is `make-room`'s module, built from the same patches at every start, under [`flatshare_make_room_serve`](../../../python/flatshare_vllm/src/flatshare_vllm/serve/flatshare_make_room_serve.py)'s `AsyncScheduler`, which schedules as `make-room`'s and writes the configuration the engine runs to `$XDG_RUNTIME_DIR/flatshare/engine-effective.json`.
   - **Each start** then holds that to the profile's [`serving.json`](../serving.json) (`deploy/bin/serving-check`) and fails on any difference.
6. In `agents-only`, the STT and the TTS do not run: their drop-ins hold `ConditionPathExists=!/`, which never holds, so systemd skips them and `variant.sh` holds them stopped. The engine takes their memory, a KV pool of 8.4 GiB (9,019,431,321 bytes: the box's 5 GiB and the speech servers' 3.4, 61,152 tokens) at a budget of 0.88, which a card with the speech servers resident cannot start. It runs vLLM's own scheduling for the card, first come first served with a 2,048-token step cap and no per-request prefill cap ([below](#a-card-for-the-agents-alone)).
7. `make-room-v03` keeps the v0.3 tag's `build.py` byte for byte (the project's former names in it) and its unit verbatim but for the paths and the names, with its module renamed `flatshare_make_room_v03_scheduler` so that the engine's scheduler class tells it from `make-room`'s. W5 re-measured the fixed variant beside it.

The frozen box schedules asynchronously (vLLM's default here), so `pr54530` and `make-room` load the stock `AsyncScheduler` over their patched `Scheduler`: with and without each change differ by the change alone. `python/flatshare_vllm/tests/test_build.py` builds both modules from vLLM v0.32.0's source and checks that; `test_make_room.py` checks `make-room`'s behavior on the real scheduler, and `test_trace.py` that it schedules as the pull requests' own patches do, step for step.

`make-room`'s three switches are keys of vLLM's `--additional-config`, read once at engine start: `make_room.max_preemptions_per_step` (0 is off), `make_room.budget_cap` (with `make_room.budget_cap_tokens`, 16) and `make_room.reserve_tokens` (0 is off). Its `flatshare-engine.conf` sets arm 9's values, and an arm file ablates one with engine flags, which come after the variant's and win (vLLM warns of the duplicate key): `flags = ["--additional-config.make_room.reserve_tokens", "0"]`. The engine reports them under `additional_config` in `/server_info`, so each point's `engine.json` records them; `flatshare-bench e2 run` checks them: the engine's `additional_config` must be exactly what the variant's and the arm's flags build, and its scheduler class the one the variant loads.

The engine start fails on an unknown `make_room` key or a bad value (`budget_cap` takes true or false in any case, or 1 or 0; the others an integer, `budget_cap_tokens` at least 1), and on a switch under a policy other than priority; for an encoder-decoder model the room counts a request's cross-attention blocks. `build.py` refuses an installed vLLM other than 0.32.0. The pull requests name the same switches as engine flags:

| Variant key | The pull requests' flag |
|---|---|
| `make_room.max_preemptions_per_step` | `--max-priority-preemptions-per-step` ([#63350](https://github.com/vllm-project/vllm/pull/63350)) |
| `make_room.budget_cap`, `make_room.budget_cap_tokens` | `--priority-prefill-budget-cap`, `--priority-prefill-budget-cap-tokens` ([#63351](https://github.com/vllm-project/vllm/pull/63351)) |
| `make_room.reserve_tokens` | none: the reserve is the variant's only |

## A card for the agents alone

`agents-only` runs a card that serves the agents and no call: the resource gate's dedicated arrangement at its best practical (card B without the voice servers), against which a shared card's background work is weighed. What it changes, and why:

- **No speech servers.** Its STT and TTS drop-ins hold `ConditionPathExists=!/`, a condition that never holds: `flatshare.target`, and any start of either unit, skips them, which systemd counts as no failure. `variant.sh` reads that line in the drop-ins it links: with them linked, the box runs when the engine runs and the STT and the TTS are stopped, and a switch to any other variant (`variant.sh use frozen` among them) removes them, so the three start again.
- **The pool in their memory.** `--kv-cache-memory 9019431321`: the box's 5 GiB and the 3.4 GiB the speech servers hold on the card (STT about 2.4 GiB, TTS about 1.0, [ARCHITECTURE.md §5.1](../../../docs/ARCHITECTURE.md#31-the-frozen-box)), 3,822 blocks of 16 tokens, 61,152 tokens where the box has 36,400. The engine, about 10.6 GiB with the box's pool, takes about 14.0 GiB of the card's 15.9, a budget of 0.88 (`--gpu-memory-utilization`), under R2's ceiling of 0.95 for one budget ([the profile's rules](../../profile.py)). vLLM wants that share free at its start, which a card with the speech servers resident does not have, so the engine refuses to start beside them.
- **vLLM's scheduling for the card.** First come, first served, a step cap of 2,048 tokens (the API server's default below 70 GiB of GPU memory) and no per-request prefill cap. On a card without calls, the box's 512 tokens a step (256 a request), which keep a voice turn's steps short, only slow the agents: in E2's discriminating scenario, where their work bounds what they do, a cap of 1,024 with no threshold gave them 9% more than tuned vLLM over the calls' first 240 s (3,888 against 3,552 tokens a second, the medians of `disc-a4-b1024-t0-s24`'s and `disc-frozen-s24`'s five runs in [W4's records](../../../benchmarks/results/2026-10-23-e2/README.md), as the report's Table B3 gives them), and their requests are 99.5% prompt tokens, which a larger step prefills in fewer steps. Priority would order nothing: every request is the agents' class. The CPU tier and prefix caching stay `engine.yaml`'s.

An arm file names it as `[engine] variant = "agents-only"`, with flags over it like any variant's (`--max-num-batched-tokens 1024`, say, to measure E2's best cell on the card); `flatshare-bench e2 run` checks the pool by its bytes and the budget beside the scheduling. The card's controlled counterpart, the same engine as a shared card's with the speech servers idle, is that card's own variant with an agents-only point ([the E2 harness](../../../benchmarks/e2/README.md)).

## Engine flags of the E2 arms

Each flag was checked against vLLM v0.32.0 (`vllm/engine/arg_utils.py`, `vllm/config/scheduler.py`). The frozen box runs `--scheduling-policy priority --max-num-batched-tokens 512 --long-prefill-token-threshold 256`.

| Arm | `[engine]` | What it sets |
|---|---|---|
| 2, stock | `--scheduling-policy fcfs --max-num-batched-tokens 2048 --long-prefill-token-threshold 0` | vLLM's defaults on this GPU: first come, first served; a step cap of 2,048 tokens; no per-request prefill cap (note 1) |
| 3, priority | `--max-num-batched-tokens 2048 --long-prefill-token-threshold 0` | Priority scheduling with vLLM's default chunking |
| 4, step cap | `--max-num-batched-tokens B --long-prefill-token-threshold T` | B in {512, 1024}, T in {256, 512, 1024}: five distinct cells, 512/256 (the frozen box), 512/0, 1024/256, 1024/512 and 1024/0 (note 2) |
| 5, watermark | best arm 4 + `--watermark W` | W in {0.05, 0.1, 0.2}: the share of KV blocks kept free when the scheduler admits a waiting or preempted request |
| 5, queued tokens | best arm 4 + `--max-num-queued-tokens N` | New requests get 503 while the prompt tokens of the requests in prefill exceed N (note 3) |
| 7a, without the PR | `--max-num-seqs 12` | The `pr54530` engine without the PR |
| 7a, with the PR | variant `pr54530` | |
| 7b | variant `sglang` | |
| 9, engine side | variant `make-room` | The KV preemption at 2 a step, the budget cap and a reserve of 2,048 tokens, beside the gateway's policies |
| 9, an ablation | variant `make-room` + `--additional-config.make_room.max_preemptions_per_step 0`, `--additional-config.make_room.budget_cap false` or `--additional-config.make_room.reserve_tokens 0` | The engine side without one mechanism; all three off is the frozen box's scheduling, through the generated module |
| 9, the change's configurations | variant `make-room` + `--additional-config.make_room.reserve_tokens 0`, with `--additional-config.make_room.budget_cap false` for #63350 alone, or `--additional-config.make_room.budget_cap_tokens 48` for the 48-token window | #63350 alone, both with the budget cap's 16-token window, or both with a 48-token window, the configuration to deploy, all without the reserve (note 4) |

1. The step cap of 2,048 tokens is the API server's default below 70 GiB of GPU memory. Nothing signals priority (the gateway's `upstream_priority = []`).
2. A threshold at or above B never binds (each request's share of a step is capped by B anyway), so the grid has five distinct cells.
3. `--max-num-queued-tokens` applies to every class, voice included, and each 503 counts as `vllm:admission_rejections_total`.
4. These are the change as #63350 and #63351 propose it. The `make-room-serve` variant runs the configuration to deploy with its own switches, at the step cap the sweep chose ([the serving profile](#the-serving-profile)).

vLLM 0.32's FCFS scheduler ignores request priorities rather than rejecting them: nothing in 0.32.0 enforces the "will raise an error" of the OpenAI protocol's `priority` description. Arm 2 sends none so that it is the engine out of the box.
