# E2 campaigns

An E2 point is one arm file in [`arms/`](arms): the calls, the agent replay beside them, the gateway's configuration and the engine ([the arm-file reference](../../python/leaseholder_bench/README.md#e2-points)). `lh-bench e2 run` measures one point on cyan. A campaign file lists arm files, and `campaign.sh` measures them in order from a workstation with SSH to the hosts, the way `deploy/deploy.sh` deploys. A line `agents-seed N` in it sets AIPerf's seed for all of its points, in place of each arm file's `[agents] seed`, so that each round of a repeated campaign replays a workload sample of its own while its points stay paired. A line `kv-sample-s SECONDS` sets `lh-bench e2 run --kv-sample-s` for its points: how often an agent point samples the engine's KV use and requests for the measured running share (0.5 s by default, on one connection to the engine that it keeps for the calls); `kv-sample-s 0` samples nothing, as W5's points ran, the sampler having come after them, so that a campaign read against W5's matches its instrumentation (the Marlin dry run's point, `benchmarks/a100/marlin-dry.campaign`). A line `save-audio` has the callers save each call's audio, for [the demo video](#the-demo-video) (`lh-bench e2 run --save-audio`).

```sh
export LH_SSH="ssh -F $HOME/.ssh/penn_config"                  # your SSH command for the hosts, as for deploy.sh
benchmarks/e2/campaign.sh --dry-run benchmarks/e2/w2.campaign   # check every arm file; list the points and engines
benchmarks/e2/campaign.sh --pause benchmarks/e2/w2.campaign     # measure, then pause the GPU node
```

For each point whose run directory on cyan is not complete, `campaign.sh`:

1. switches the GPU node to the arm's engine with [`variant.sh use`](../../deploy/phastform/variants/README.md), which restarts only what changes and returns once the engine answers `/health`;
2. runs `lh-bench e2 run` on cyan, into `~/leaseholder/runs/e2/CAMPAIGN/POINT`, with the GPU node's deployed revision, its `~/leaseholder/src/DEPLOYED` read over SSH as the point starts, which `run.json` keeps beside cyan's (`--gpu-source`). Meanwhile it samples the GPU node's `nvidia-smi` once a second (SM and memory clocks, temperature, power and clock event reasons, timestamps in UTC, and the GPU's name, so that a record says which card it ran on), because a consumer card's boost clock follows its temperature, and puts the samples into that directory as `gpu.csv` when the point ends, failed or not;
3. fetches that directory into `$LH_E2_RESULTS/CAMPAIGN/POINT` (default `~/leaseholder/runs/e2`), without AIPerf's `inputs.json`.

A complete point is skipped, so running a stopped campaign again resumes it. The first failure stops the campaign, with the partial run directory fetched and its `run.json` saying what failed. At the end, `lh-bench e2 report` writes the frontier of the campaign's points (fetched into `CAMPAIGN/report`: `frontier.md`, `frontier.csv` and a figure per workload, `frontier.svg` for the arm files' own) and `lh-bench e2 reset` puts cyan's units back. `--pause` then stops the GPU node's units, which nothing should hold while idle. A campaign that fails once it has switched the node does both on its way out too, unless a point may still be running on cyan, which a reset or a pause would spoil: its lock is held, or the connection to cyan was lost (ssh's exit status 255, which `campaign.sh` reports as such rather than as the answer of the probe it cut). It then leaves cyan's units and the node as they are, and says so. `campaign.log` beside the points keeps every step.

Points that share an engine go together in a campaign file: a switch restarts the engine, about 2 minutes with warm caches. A point takes about 12 minutes: AIPerf builds its dataset for about 4 (its cache does not apply to this dataset type, see `configs/aiperf.env`), then a minute of warm-up and the calls.

**On one host.** `campaign.sh --local` runs a campaign on a single host that is cyan and the GPU node at once ([docs/HARDWARE.md](../../docs/HARDWARE.md)): every step runs there (`deploy/bin/lh-local` in place of ssh, so a dropped workstation link does not stop it), the node is `localhost`, its engine switches through `$LH_NODE_DIR/variant.sh` (a node rendered from a hardware profile has its own, `LH_NODE_DIR=leaseholder/node`), and the results go to `~/leaseholder/results/e2`, which `deploy/a100/pull.sh HOST` copies to the workstation every two minutes. `--until HH:MM` (UTC; tomorrow's once today's has passed), or `--until-epoch SECONDS`, an absolute deadline, which `benchmarks/a100/run.sh` hands every campaign of a run and `cprime.sh` (so that one ending after HH:MM hands the next no deadline a day away), starts no point that would end after then, by the longest point of its kind and engine switch measured on this host (`durations.tsv` beside the campaigns' results), and before one is, W5's durations: 13 minutes for a point with agents, 7 without, 2 more for an engine switch; the frontier and the reset follow as usual.

**Stopping.** Ctrl-C stops `campaign.sh` and its sampling of the GPU, but a point already running on cyan goes on to its end: closing its SSH session does not stop it. To stop it as well, send SIGINT to its pid, which `~/leaseholder/runs/.lh-bench-e2.lock` holds on cyan; `lh-bench e2 run` then stops AIPerf and the callers and marks the point failed. While a point runs, `campaign.sh` refuses to switch the engine under it.

## Before the first campaign

- Deploy both hosts (`deploy/deploy.sh`), then on cyan rerun `deploy/cyan/bootstrap.sh`: it builds `venvs/bench` and `venvs/smg`.
- cyan: the `vad-turns` worker variant is linked (`lh-bench e2 run` refuses to measure without it); `~/.config/leaseholder/gateway-keys.env` has `LH_GATEWAY_AGENTS_KEY`, the key of the `agents` tenant; `~/leaseholder/data/weka-x32` (`lh-bench scale-traces`) and `~/leaseholder/data/callers` (`lh-voice render-callers`) exist; `lh-gateway` is a build that knows every key the arms' `[gateway]` tables set (`upstream_priority`, `[admission]`).
- Phastform, for arm 7b: `bash ~/leaseholder/src/deploy/phastform/variants/sglang/install.sh` builds SGLang's environment once.

## Points

[`w2.campaign`](w2.campaign) holds the points that depend on no other arm's result:

| Arm | Points | Engine | Gateway |
|---|---|---|---|
| 1 | `a1-voice`: the calls alone | frozen | as configured |
| 2 | `a2-stock-s{8,16,24}` | FCFS, step cap 2,048, no long-prefill threshold | `upstream_priority = []` |
| 3 | `a3-priority-s{8,16,24}` | priority, step cap 2,048, no long-prefill threshold | as configured |
| 4 | `a4-b512-t256-s16` (the frozen box), `a4-b512-t0-s16`, `a4-b1024-t256-s16`, `a4-b1024-t512-s16`, `a4-b1024-t0-s16` | priority with each step cap B and long-prefill threshold T; T at or above B does not bind, so B × T has five distinct cells | as configured |
| 7a | `a7a-seqs12-s16`, `a7a-pr54530-s16` | priority, `--max-num-seqs 12`, without and with vLLM #54530 | as configured |
| 7b | `a7b-sglang-s{8,16}` | SGLang 0.5.22 | `upstream_priority = ["body"]` |

The gateway "as configured" is `~/.config/leaseholder/gateway.toml`, which sends each class's priority as `X-Vllm-Priority`. Every point runs 2 callers × 25 turns with barge-in on a quarter of them, beside AIPerf's agentic replay of the 32× corpus (think times capped at 120 s) through the gateway as the `agents` tenant.

The other arms take an earlier arm's winner, so their files are written from its results. Each starts from the winning arm file, with its own `arm` and `label`:

| Arm | Add to the winner's arm file |
|---|---|
| 4, best two at 8 and 24 sessions | `[agents] sessions = 8` (and 24) |
| 5, best arm 4 + headroom | `"--watermark", "0.05"` (0.1, 0.2) in `[engine] flags`; or `"--max-num-queued-tokens", "N"`, which answers new requests 503 while the prompt tokens in prefill exceed N, voice requests included. vLLM sizes N as the TTFT target times the prefill throughput |
| 6, best arm 5 + an operator's cap | `[gateway.admission]` with `policy = "static-cap"` and `background_max_in_flight = 2` (4, 8); or `policy = "token-reservation"` and `reserved_tokens` set to one voice turn's KV: the realtime class's prompt and completion tokens per completed request, from arm 1's `summary.json` (`gateway.counters`) |
| 8a, best stock + llm-d's flow control | `[gateway.admission]` with `policy = "flow-control"` and `ceilings = { realtime = …, interactive = …, background = … }` |
| 8b, best stock + SMG | `router = "smg"`, and `[gateway]` with `upstream = "unix:/run/user/1007/lh/smg.sock"` and `upstream_priority = ["smg-header"]`: SMG runs in the voice-plane namespace between two proxies (`deploy/cyan/smg.sh`), since it takes only http:// workers and listens on TCP only |
| 8c, best stock + preempt-resubmit | `[gateway.admission]` with `policy = "preempt-resubmit"` |
| factor, frozen and best stock | `[voice] preemptive_generation = false`, `arm = "<arm>-no-preemptive"` |
| calls, best stock | `[voice] callers = 4` (6), at 0 and 16 sessions, `arm = "<arm>-calls"` |
| discriminating scenario | `[agents] sessions = 24` and `data` naming a corpus with longer contexts (`lh-bench scale-traces` with a smaller factor) |

[`w2b.campaign`](w2b.campaign) holds W2's second batch, written from the first batch's results. At 16 sessions the best arm 4 cells, step cap 1024 with no long-prefill threshold and the frozen box (512/256), were within noise of each other (V2V p95 2,167 and 2,215 ms over 50 turns, background goodput within 4%), so both are measured again at 24 sessions, and the frozen box, the configuration every earlier result was measured on, is the engine of the other points: arm 5 (`--watermark 0.1`; `--max-num-queued-tokens 8192`, about 1.6 s of prefill at the step cap), arm 6 (caps of 2 and 8; a reservation of 2,048 tokens, two concurrent voice turns at their longest), arms 8a–8c (flow-control ceilings 6/6/6, so that background fills at most 6 and a realtime request sheds), preemptive generation off, and 4 and 6 callers. The admission arms wait at most 240 s, below the replay's 300 s request timeout.

**The discriminating scenario** replays only the traces with the largest contexts, so that agents fill the KV pool while voice turns keep arriving. `weka-x32-long` on cyan holds the 48 traces of `weka-x32` whose main agent has the largest median input (the median trace's median is 13,326 tokens against 4,784 for the whole corpus; per request, 13,927 tokens against 7,940; 23,848 main-agent requests), linked by [`long_traces.py`](long_traces.py): `python benchmarks/e2/long_traces.py ~/leaseholder/data/weka-x32 ~/leaseholder/data/weka-x32-long 48`. Its points are named `disc-*`.

[`w2c.campaign`](w2c.campaign) holds W2's third batch: arms 4 to 8 in the discriminating scenario, one run each (the arm files `disc-<arm>-s24.toml` are the 16- or 24-session arms with `sessions = 24` and `data = "weka-x32-long"`), and SGLang with its host tier (`--enable-hierarchical-cache --hicache-ratio 4 --hicache-write-policy write_through`) at 8 and 16 sessions, for parity with vLLM's write-through CPU tier. W4 repeats the best of each arm beside arm 9. [`a9.campaign`](a9.campaign) holds arm 9, Leaseholder, in the same scenario: the gateway's lifecycle policy (`[gateway.admission]` with `policy = "lifecycle"`, which E2's arm files spell `leaseholder`: 1,024 tokens of KV headroom per armed session, a 150 ms critical-window budget of predicted background prefill, a 16-token reply cap when the headroom was not kept), the engine's `make-room` variant and the voice worker's lifecycle events; its knob, the critical-window budget at 0 and 500 ms; and one run of each single-mechanism ablation (`disc-a9-no-*`, `trigger-request`, `gateway-only`, `engine-only`). A switched-off mechanism's keys are left out: the gateway refuses them.

[`r1.campaign`](r1.campaign) to [`r4.campaign`](r4.campaign) repeat the discriminating scenario in rounds, so that each headline configuration has five runs (w2c and a9 are round 0) and each of arm 9's ablations three (rounds 1 and 2). The headline configurations are the best of each arm in round 0: the frozen box and the best step-cap cell (`disc-a4-b1024-t0-s24`), the static reservation, preempt-resubmit, SGLang with its host tier, and arm 9. A round groups its points by engine, and the groups' order rotates from round to round, so that no configuration always runs at the same hour of a campaign that takes hours. Rounds 3 and 4 and both standard rounds (below) also measure voice alone (`a1-voice`), the reference every point's SLO is read against, which W2 measured once: arm 9's runs land just above that SLO, by about the spread of one point's p95, with degrade capping every reply, so the reference itself needs repeats.

[`a9-standard.campaign`](a9-standard.campaign) puts arm 9 in the standard scenario at 8, 16 and 24 agent sessions, as W2 measured arms 2 to 8, so that its frontier stands beside theirs. [`s1.campaign`](s1.campaign) and [`s2.campaign`](s2.campaign) repeat the realistic operating point, 16 agent sessions, for the frozen box and arm 9, so that each has three runs there.

[`a9-window.campaign`](a9-window.campaign) measures arm 9 without degrade and with make-room's first-tokens window at 48 tokens, after `s2`: without degrade, a reply's first sentence can outrun the default 16-token window, and degrade's 16-token cap, which keeps it inside, capped every reply in the discriminating scenario.

[`upstream.campaign`](upstream.campaign) measures the engine as the vLLM PRs propose it, without the reserve and behind the gateway without an admission policy: the KV preemption alone, with the budget cap, and with a 48-token window. [`upstream-r1.campaign`](upstream-r1.campaign) and [`upstream-r2.campaign`](upstream-r2.campaign) repeat these three and arm 9's 48-token window, so that each has three runs, as each of arm 9's ablations has: a configuration that misses the SLO moved by up to 1.9 s in p95 from round to round.

[`w5-r1.campaign`](w5-r1.campaign) to [`w5-r5.campaign`](w5-r5.campaign) re-measure, after W5's fixes to make-room, the PRs' configuration with a 48-token window (`arms/w5-disc-pr-ab48-s24.toml`) beside its control, the same on make-room as v0.3 built it (`w5-disc-pr-ab48-v03-s24.toml`, on the `make-room-v03` variant: v0.3's `build.py` and `lh-engine.conf`, its module renamed), and beside voice alone and the frozen box (`w5-a1-voice.toml`, `w5-disc-frozen-s24.toml`) in the same rounds. Each arm file is its W4 one with each call opening with an unmeasured warm-up turn (`[voice] warmup_turns = 1`). Each round replays a sample of the agents of its own (`agents-seed` 11 to 15) and runs the four in an order rotated by one from round to round, so that each runs in each place. The control's module is `lh_make_room_v03_scheduler`, so that `lh-bench e2 run`'s engine check, which reads the engine's scheduler class back, fails a point on the other module before it starts.

[`w5-pr-r1.campaign`](w5-pr-r1.campaign) to [`w5-pr-r5.campaign`](w5-pr-r5.campaign) re-measure the PRs' configurations in rounds of their own: the KV preemption alone (`arms/w5-disc-pr-a-s24.toml`) and both PRs at the budget cap's default 16-token window (`w5-disc-pr-ab-s24.toml`), beside the frozen box and the 48-token window of W5's campaign, so that every configuration the PRs cite has its runs in the same rounds. Round N replays w5-rN's sample of the agents and runs the four in an order rotated by one from round to round; the frozen box and the 48-token window, measured in both campaigns, show how far the two sessions differ.

[`w5-ext-r1.campaign`](w5-ext-r1.campaign) to [`w5-ext-r5.campaign`](w5-ext-r5.campaign) extend W5's re-measure over the same five samples of the agents: preempt-resubmit, the best arm without an engine change, beside the PRs' 48-token configuration in the discriminating scenario (`arms/w5-disc-a8c-resubmit-s24.toml`, `w5-disc-pr-ab48-s24.toml`), and the frozen box and the PRs' 48-token configuration in the standard scenario, 16 agent sessions of the whole corpus (`w5-std-frozen-s16.toml`, `w5-std-pr-ab48-s16.toml`), each call opening with a warm-up turn. Each round runs three engine groups in an order rotated by one group from round to round; the PRs' two points share their engine.

[`w5-cap-r1.campaign`](w5-cap-r1.campaign) to [`w5-cap-r3.campaign`](w5-cap-r3.campaign) measure calls per GPU: the PRs' 48-token configuration with 4 and with 6 callers beside the discriminating scenario (`arms/w5-disc-pr-ab48-c4-s24.toml`, `w5-disc-pr-ab48-c6-s24.toml`), and voice alone with as many callers, the SLO's references (`w5-a1-voice-c4.toml`, `w5-a1-voice-c6.toml`), over three of W5's samples of the agents. Each round swaps the order of its two engine groups, and of the caller counts within each.

[`fb-r1.campaign`](fb-r1.campaign) to [`fb-r5.campaign`](fb-r5.campaign) are FB's rounds: voice alone, the frozen box and the PRs' 48-token configuration again, on the final code, with W5's arm files, W5's five samples of the agents (seeds 11 to 15) and no KV sampler, as W5's points ran, so that each round compares with W5's of the same seed. Each round rotates the three by one place.

The `[admission]` keys are those of `lh-gateway`'s configuration ([its README](../../crates/lh-gateway/README.md)); `lh-bench e2 run` passes the `[gateway]` table through as it is.

`LH_PROFILE` names the GPU node's hardware profile ([`deploy/profiles`](../../deploy/profiles)) when it is not the frozen box's: `lh-bench e2 run --profile` then holds the engine to that profile's model and scheduler settings, and the report gives the points a workload of their own, by the GPU and the model their records name.

**SMG as shipped (arm 8b).** `lh-smg.service` runs SMG 1.13.0's priority scheduler with its defaults. A vLLM worker behind its `openai` backend reports no capacity, so SMG sizes its pool at 64 slots, and its built-in reservations (system 32, interactive at least 128, default 10%) do not fit: it scales them down to system 19, interactive 39 and default 4, which leaves 2 slots for the bulk class, where the gateway sends background requests. Measured against `lh-bench mock-server`, 40 concurrent bulk requests went through 2 at a time, and the others wait in SMG's bulk queue (1,024 deep, 503 after 300 s). As shipped, arm 8b therefore caps the agents at 2 requests in flight; reservations that fit the pool take a `--priority-scheduler-config` file.

## The headline configurations

The configurations the docs' tables report, by the arm file each ran from and the campaigns whose records hold its runs, in the discriminating scenario unless the row says otherwise. W4's records are in [`benchmarks/results/2026-10-23-e2`](../results/2026-10-23-e2/README.md) and W5's re-measure in [`benchmarks/results/2026-10-24-e2-w5`](../results/2026-10-24-e2-w5/README.md), each W5 arm file its W4 one with a warm-up turn opening each call.

| Configuration | W4: arm file, campaigns | W5: arm file, campaigns |
|---|---|---|
| Voice alone, the SLO's reference | `a1-voice`: w2, s1, s2, r3, r4 | `w5-a1-voice`: w5-r1 to w5-r5 |
| The frozen box | `disc-frozen-s24`: w2c, r1 to r4 | `w5-disc-frozen-s24`: w5-r1 to w5-r5, w5-pr-r1 to w5-pr-r5 |
| Step cap 1,024, no threshold | `disc-a4-b1024-t0-s24`: w2c, r1 to r4 | |
| Static reservation, 2,048 tokens | `disc-a6-reserve2048-s24`: w2c, r1 to r4 | |
| SGLang with HiCache | `disc-a7b-sglang-hicache-s24`: w2c, r1 to r4 | |
| Preempt-resubmit before the first byte | `disc-a8c-resubmit-s24`: w2c, r1 to r4 | `w5-disc-a8c-resubmit-s24`: w5-ext-r1 to w5-ext-r5 |
| Arm 9 | `disc-a9-s24`: a9, r1 to r4 | |
| #63350 alone (KV preemption) | `disc-pr-a-s24`: upstream, upstream-r1, upstream-r2 | `w5-disc-pr-a-s24`: w5-pr-r1 to w5-pr-r5 |
| #63350 + #63351 (the budget cap, 16 tokens) | `disc-pr-ab-s24`: upstream, upstream-r1, upstream-r2 | `w5-disc-pr-ab-s24`: w5-pr-r1 to w5-pr-r5 |
| #63350 + #63351, a 48-token window | `disc-pr-ab48-s24`: upstream, upstream-r1, upstream-r2 | `w5-disc-pr-ab48-s24`: w5-r1 to w5-r5, w5-pr-r1 to w5-pr-r5, w5-ext-r1 to w5-ext-r5 |
| The same on make-room as v0.3 built it (W5's control) | | `w5-disc-pr-ab48-v03-s24`: w5-r1 to w5-r5 |
| Standard scenario (16 agent sessions): the frozen box | `a4-b512-t256-s16`: w2, s1, s2 | `w5-std-frozen-s16`: w5-ext-r1 to w5-ext-r5 |
| Standard scenario: arm 9 | `a9-s16`: a9-standard, s1, s2 | |
| Standard scenario: #63350 + #63351, a 48-token window | | `w5-std-pr-ab48-s16`: w5-ext-r1 to w5-ext-r5 |
| 4 and 6 callers: #63350 + #63351, a 48-token window | | `w5-disc-pr-ab48-c4-s24`, `w5-disc-pr-ab48-c6-s24`: w5-cap-r1 to w5-cap-r3 |
| 4 and 6 callers: voice alone | | `w5-a1-voice-c4`, `w5-a1-voice-c6`: w5-cap-r1 to w5-cap-r3 |

The points on two GPU nodes (PLAN §3.4: a voice node and an agent node, against two pooled nodes under tuned stock and under Leaseholder) never ran: no second GPU is free before v1.0. Their arm files and campaigns, with the code that runs them (the gateway's `[nodes]` and `[placement]`, the voice worker's `LH_NODES`, `lh-bench e2 run`'s points on several nodes and `campaign.sh`'s several GPU nodes), are on the `two-node` branch.

## Results

`lh-bench e2 report RUN_DIR... --out DIR` builds a frontier from any set of run directories, across campaigns too. It compares points only within a workload: the agents' corpus and the number of callers, the GPU and the model where they are not the frozen box's, and the warm-up turns that open each call where the points differ in them (W4's records with W5's). The discriminating scenario (`weka-x32-long`) and the calls-per-GPU points (4 and 6 callers) each get their own section of `frontier.md`, with the voice SLO they are read against (250 ms above voice alone with as many callers and warm-up turns), and their own figure (`frontier-weka-x32-long.svg`, `frontier-4-callers.svg`), beside `frontier.svg` for the standard one ([the figures](../../python/leaseholder_bench/README.md#e2-points)). A point measured in several campaigns (W4's rounds of the discriminating scenario reuse w2c's and a9's arm files) is one configuration: the report keeps each run, gives the configuration's median and range, and places it on the frontier at its medians. `waterfall.svg` splits each configuration's voice turn into its stages (endpointing, the reply to its first sentence, TTS first audio, transport and playout), stacked to its mean V2V, with the median and p95 marked; `--waterfall POINT` picks the configurations it draws.

The repository keeps a compact record of each point, not its run directory: `lh-bench e2 export RUN_DIR... --out DIR` writes the few small files the report reads (about 15 kB a point, no credential and no home path) into `DIR/CAMPAIGN/POINT`, and the report gives the same tables and figures from them as from the run directories ([the compact record](../../python/leaseholder_bench/README.md#e2-points)). E2's records are in [`benchmarks/results/2026-10-23-e2`](../results/2026-10-23-e2/README.md), and `make reproduce-e2` regenerates the report from them into `build/reproduce-e2` (`E2_REPORT=DIR` to choose), with the bench venv:

```sh
python/leaseholder_bench/.venv/bin/lh-bench e2 export $LH_E2_RESULTS/w2c/* --out benchmarks/results/2026-10-23-e2
make reproduce-e2
```

The run directories themselves, with the raw records the report does not need (the gateway's request log, AIPerf's exports, the metric snapshots, the worker's journal, the GPU's samples, the tools' logs), are an asset of the v0.3 release.

## The demo video

Two calls side by side, one stuttering and one smooth, while background agents run: [`demo.py`](demo.py) renders them from two points measured with `lh-bench e2 run --save-audio`, which has the callers save each call's audio and the times of its turns ([`lh-voice bots --save-audio`](../../python/leaseholder_voice/README.md#measurement-callers)). [`demo.campaign`](demo.campaign) measures two such points with its `save-audio` line: the frozen box and the vLLM PRs' configuration with a 48-token window in the discriminating scenario, on W5's round-1 sample of the agents:

```sh
benchmarks/e2/campaign.sh benchmarks/e2/demo.campaign
```

Then, on a workstation with the two run directories fetched (`uv` installs the script's pinned numpy, matplotlib and imageio-ffmpeg, whose ffmpeg encodes):

```sh
uv run benchmarks/e2/demo.py $LH_E2_RESULTS/demo/w5-disc-frozen-s24 $LH_E2_RESULTS/demo/w5-disc-pr-ab48-s24 \
  --labels "vLLM 0.32, tuned" "With #63350 and #63351" --out demo.mp4
uv run benchmarks/e2/demo.py $LH_E2_RESULTS/demo/w5-disc-frozen-s24 $LH_E2_RESULTS/demo/w5-disc-pr-ab48-s24 \
  --labels "vLLM 0.32, tuned" "With #63350 and #63351" --first-turn 1 --seconds 30 --out turn1.mp4 \
  --gif docs/figures/e2-demo.gif --gif-seconds 15
```

The v0.3 release has the video rendered so, from two such points measured by hand on 2026-10-24 (`disc-frozen-s24` and `disc-pr-ab48-s24`, V2V p95 10,851 and 2,202 ms; they are not in the tables, and the release has their run directories). [`docs/figures/e2-demo.gif`](../../docs/figures/e2-demo.gif) is the second command's: the calls' second turn, the first 15 s.

The video is 1280×720 at 30 frames a second. Each side shows its point's first call (`--caller`): its scrolling waveform, the caller above and the agent below; the current turn's wait, from the end of the caller's speech to the agent's first audio, counting live; each reply's V2V as the callers measured it, in a bar that grows while the caller waits; and the background goodput, the point's over its calls and the last 10 s of AIPerf's completed requests on the wall clock. Both calls show the same turns from `--first-turn` (default 0): `--turns` of them, by default as many as fit in `--seconds` (default 90). Turn k of both starts at the same moment, so the same line is answered side by side, and the call that is ahead waits in silence before its next line: only a caller's pause between turns is stretched, never a measured wait. The soundtrack is one call at a time, the left one for the first half and the right one for the second, and the side playing is highlighted. `--gif` writes the first 10 s (`--gif-seconds`) as a 640×360 GIF at 10 frames a second, for a README, from frames of their own without the soundtrack's cues (no side marked as heard). A 90-second video takes two to three minutes to render.

`python/leaseholder_bench/tests/test_demo.py` runs the renderer on a synthetic pair of calls, tones in the files `--save-audio` writes: in CI the excerpt, the aligned turns, the soundtrack and the frames, with the bench venv's numpy and matplotlib; and, marked integration since it fetches the script's dependencies, the script itself through `uv run` to an MP4 and a GIF (`uv run --extra test pytest -m integration tests/test_demo.py` in `python/leaseholder_bench`).
