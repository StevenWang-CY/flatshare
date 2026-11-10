# E2 campaigns

How E2 is run, for a reader who measures a campaign or adds an arm. An E2 point is one arm file in [`arms/`](arms): the
calls, the agent replay beside them, the gateway's configuration and the engine
([the arm file](../../python/flatshare_bench/README.md#e2-points)). `flatshare-bench e2 run` measures one point on cyan.
A campaign file lists arm files, and `campaign.sh` measures them in order from a workstation with SSH to the hosts, as
`deploy/deploy.sh` deploys.

```sh
# your SSH command for the hosts, as for deploy.sh
export FLATSHARE_SSH="ssh -F $HOME/.ssh/flatshare_config"
# check every arm file; list the points and engines
benchmarks/e2/campaign.sh --dry-run benchmarks/e2/w2.campaign
# measure, then pause the GPU node
benchmarks/e2/campaign.sh --pause benchmarks/e2/w2.campaign
```

A line of a campaign file can also set, for all its points:

- **`agents-seed N`** replaces each arm file's AIPerf seed, so that each round of a repeated campaign replays a sample
  of the agents of its own while its points stay paired.
- **`kv-sample-s SECONDS`** sets how often an agent point samples the engine's KV use, 0.5 s by default, or 0 to match
  W5's instrumentation.
- **`save-audio`** has the callers save each call's audio, for [the demo video](#the-demo-video).

## How a campaign runs

For each point whose run directory on cyan is not complete, `campaign.sh`:

1. switches the GPU node to the arm's engine with [`variant.sh use`](../../deploy/phastform/variants/README.md), which
   restarts only what changes and returns once the engine answers `/health`;
2. runs `flatshare-bench e2 run` on cyan, into `~/flatshare/runs/e2/CAMPAIGN/POINT`, passing the GPU node's deployed
   revision, while it samples the node's `nvidia-smi` once a second into `gpu.csv`, since a consumer card's boost clock
   follows its temperature;
3. fetches the run directory into `$FLATSHARE_E2_RESULTS/CAMPAIGN/POINT` (default `~/flatshare/runs/e2`).

- **Resuming.** A complete point is skipped, so running a stopped campaign again resumes it.
- **Failing.** The first failure stops the campaign, with the partial run directory fetched and its `run.json` saying
  what failed.
- **The end.** `flatshare-bench e2 report` writes the campaign's frontier into `CAMPAIGN/report`, and
  `flatshare-bench e2 reset` puts cyan's units back. `--pause` then stops the GPU node's units. A campaign that fails
  after switching the node does both too, unless a point may still be running on cyan, and then says so.
- **The log.** `campaign.log` beside the points keeps every step.

Points that share an engine go together in a campaign file, since a switch restarts the engine, about 2 minutes with
warm caches. A point takes about 12 minutes, about 4 of them for AIPerf to build its dataset, then a minute of warm-up
and the calls. Later points that replay the same corpus with the same seed load the cached dataset instead. On peacock,
over the long traces, profiling then began 9 s after AIPerf's start rather than 143 s.

**On Gray.** `FLATSHARE_GPU` names the campaign's one GPU node, Phastform by default. With `FLATSHARE_GPU=gray` the
campaign switches and samples Gray, and passes `--node gray` to every point. A campaign's run directories hold one
node's points, so each node's rounds are campaign files of their own. Gray needs, once:

- its release (`deploy/deploy.sh gray`), which runs its servers in the host's network namespace
  ([the runbook](../../ops/RUNBOOK.md#a-gpu-node-without-user-namespaces-gray));
- its models (`bash ~/flatshare/src/deploy/gray/bootstrap.sh`) and cyan's tunnel keys
  (`deploy/authorize-tunnel.sh cyan gray`);
- on cyan, `~/.config/flatshare/tunnel-gray.env` with `FLATSHARE_TUNNEL_TARGET=<user>@<gray's host>`.

[`gray-smoke.campaign`](gray-smoke.campaign) checks Gray end to end, run as
`FLATSHARE_GPU=gray benchmarks/e2/campaign.sh --pause benchmarks/e2/gray-smoke.campaign`.

**Points without calls.** An arm file with `[voice] callers = 0` and agent sessions measures the agents alone, as a card
that serves no call runs them, over `[agents] window_s` after the warm-up. With `[engine] variant = "agents-only"` the
card runs without its speech servers, the engine's pool grown into their memory
([the variant](../../deploy/phastform/variants/README.md#a-card-for-the-agents-alone)). The report gives such points a
workload of their own, without a frontier or a figure.

**On one host.** `campaign.sh --local` runs a campaign on a single host that is both cyan and the GPU node
([HARDWARE.md](../../docs/HARDWARE.md#single-host-mode)). Every step runs there, so a dropped workstation link stops
nothing, and the results go to `~/flatshare/results/e2`, which `deploy/a100/pull.sh HOST` copies to the workstation.

**A deadline.** `--until HH:MM` (UTC) or `--until-epoch SECONDS` starts no point that would end after then. A point is
budgeted by the longest of its kind measured on the host (`durations.tsv`), and until one is, by W5's: 13 minutes with
agents, 7 without, and 2 more for an engine switch. A campaign stopped by its deadline exits 0, as a complete one does,
and `campaign.sh --points CAMPAIGN` lists its points to tell them apart.

**Stopping.** Ctrl-C stops `campaign.sh`, but a point already running on cyan goes on to its end. To stop it too, send
SIGINT to the pid in `~/flatshare/runs/.flatshare-bench-e2.lock` on cyan, and the point is marked failed.

## Before the first campaign

- Deploy both hosts (`deploy/deploy.sh`), which builds cyan's `venvs/bench` and `venvs/smg` among the environments
  ([the runbook](../../ops/RUNBOOK.md#deploy)).
- On cyan, the worker runs bench mode with the `vad-turns` variant linked, not `serve`, and
  `~/.config/flatshare/gateway-keys.env` has `FLATSHARE_GATEWAY_AGENTS_KEY`, the `agents` tenant's key.
  `~/flatshare/data/weka-x32` (`flatshare-bench scale-traces`) and `~/flatshare/data/callers`
  (`flatshare-voice render-callers`) exist, and the gateway build knows every key the arms' `[gateway]` tables set.
- On Phastform, for arm 7b, `bash ~/flatshare/src/deploy/phastform/variants/sglang/install.sh` builds SGLang once.

## A point on cyan

What a point changes on cyan, and how to undo it:

- **The gateway** runs from `~/.config/flatshare/gateway-e2.toml` through the `e2-gateway` drop-in, which is
  `gateway.toml` with the inference canary off, so that the engine serves only the point's requests, and the arm's
  `[gateway]` table merged in.
- **The worker.** `[voice] preemptive_generation = false` links the `no-preemptive-generation` drop-in,
  `turn_hints = true` the `turn-hints` one and `events = false` the `no-events` one.
- **Another GPU node.** A point on Gray starts its tunnel when it is not running, sets the gateway's `upstream` to
  `gray-engine.sock`, and gives the worker the drop-in `variant-e2-node.conf` with Gray's speech sockets.
- **SMG.** An arm with `router = "smg"` starts `flatshare-smg`, and every other arm stops it. SMG takes only http://
  workers and listens on TCP only, so it runs in the voice-plane namespace between two proxies (`deploy/cyan/smg.sh`).
- **Undo.** `~/flatshare/venvs/bench/bin/flatshare-bench e2 reset` puts the gateway back on `gateway.toml`, preemptive
  generation and the events on, the turn hints off, the worker back on Phastform's sockets, and SMG stopped. A tunnel a
  point started stays up, and a campaign resets at its end.

Each point keeps the gateway's request lines, the worker's journal and the GPU's samples
([the run directory](../../python/flatshare_bench/README.md#e2-points)).

## When a point fails

| Symptom | Likely cause | Action |
|---|---|---|
| `flatshare-bench e2 run` fails at its engine check | The GPU node runs another variant or other flags than the arm names | `variant.sh use` with what `flatshare-bench e2 engine ARM` prints; a campaign does this itself |
| `flatshare-bench e2 run` or `flatshare-bench drill cpu-tier`: "no /server_info" | The engine runs without the measurement drop-in, as a production box does, or runs `make-room-serve`, whose drop-in keeps the routes off even with it | `variant.sh measure on` (restarts the engine), and beside `make-room-serve` another variant (`variant.sh use NAME`); a campaign does this itself |
| `flatshare-bench e2 run`: "another flatshare-bench e2 run or reset holds" the lock | A point is still running, or one was interrupted | The pid is in `~/flatshare/runs/.flatshare-bench-e2.lock`; stop it with SIGINT, or wait |
| The gateway does not start for a point, and `run.json` quotes its log | The arm's `[gateway]` table sets a key this gateway build does not know | Deploy a gateway build that has it, or fix the arm; `flatshare-bench e2 reset` restores `gateway.toml` |

## Arms and campaigns

By default a point runs 2 callers × 25 turns with barge-in on a quarter of them, beside AIPerf's replay of the 32×
corpus with think times capped at 120 s, through the gateway as the `agents` tenant. The gateway "as configured" is
`~/.config/flatshare/gateway.toml`, which sends each class's priority as `X-Vllm-Priority`. [`w2.campaign`](w2.campaign)
holds the arms that depend on no other arm's result:

| Arm | Points | Engine | Gateway |
|---|---|---|---|
| 1 | `a1-voice`: the calls alone | frozen | as configured |
| 2 | `a2-stock-s{8,16,24}` | FCFS, step cap 2,048, no long-prefill threshold | `upstream_priority = []` |
| 3 | `a3-priority-s{8,16,24}` | priority, step cap 2,048, no long-prefill threshold | as configured |
| 4 | `a4-b512-t256-s16` (tuned vLLM), `a4-b512-t0-s16`, `a4-b1024-t256-s16`, `a4-b1024-t512-s16`, `a4-b1024-t0-s16` | priority with each step cap B and threshold T; T at or above B does not bind, so B × T has five distinct cells | as configured |
| 7a | `a7a-seqs12-s16`, `a7a-pr54530-s16` | priority, `--max-num-seqs 12`, without and with vLLM #54530 | as configured |
| 7b | `a7b-sglang-s{8,16}` | SGLang 0.5.22 | `upstream_priority = ["body"]` |

The other arms start from a winner's arm file, with their own `arm` and `label`, and add:

| Arm | Added to the winner's arm file |
|---|---|
| 5, headroom | `"--watermark", "0.1"` in `[engine] flags`, or `"--max-num-queued-tokens", "8192"`, which answers new requests 503 while the prefill exceeds it, voice included |
| 6, an operator's cap | `[gateway.admission]` with `policy = "static-cap"` and `background_max_in_flight` 2 or 8, or `policy = "token-reservation"` with `reserved_tokens = 2048`, two voice turns at their longest |
| 8a, llm-d's flow control | `[gateway.admission]` with `policy = "flow-control"` and `ceilings` 6/6/6 |
| 8b, SMG | `router = "smg"`, with the gateway's `upstream` at SMG's socket and `upstream_priority = ["smg-header"]` |
| 8c, preempt-resubmit | `[gateway.admission]` with `policy = "preempt-resubmit"` |
| A factor | `[voice] preemptive_generation = false`, or `[voice] callers = 4` (6) |
| The discriminating scenario | `[agents] sessions = 24` and `data = "weka-x32-long"` |

W2's second batch ([`w2b.campaign`](w2b.campaign)) takes tuned vLLM, within noise of the best step-cap cell at 16
sessions, as the engine of arms 5, 6 and 8. The admission arms wait at most 240 s, below the replay's 300 s request
timeout. SMG 1.13.0 as shipped sizes its pool at 64 slots behind a vLLM worker and leaves 2 for the bulk class, where
background requests go, so arm 8b caps the agents at 2 requests in flight.

**The discriminating scenario** replays only the 48 traces whose main agent has the largest median input, so that agents
fill the KV pool while voice turns keep arriving. Their median request holds 13,927 tokens, against 7,940 in the whole
corpus. [`long_traces.py`](long_traces.py) links them:

```sh
python benchmarks/e2/long_traces.py ~/flatshare/data/weka-x32 ~/flatshare/data/weka-x32-long 48
```

Each campaign file opens with what it measures, and
[EXPERIMENTS.md](../../docs/EXPERIMENTS.md#which-campaign-measured-what-on-which-code) gives the dates of the campaigns
from W2 to OS. By campaign:

| Campaign | Its files | What they measure | Records |
|---|---|---|---|
| W2 | `w2`, `w2b`, `w2c` | Every arm once in the standard scenario, then arms 4 to 8 in the discriminating one | [`2026-10-23-e2`](../results/2026-10-23-e2/README.md) |
| W3–W4 | `a9`, `r1` to `r4`, `a9-standard`, `s1`, `s2`, `a9-window`, `upstream`, `upstream-r1`, `upstream-r2` | Arm 9 and its ablations, rounds of the headline configurations, arm 9 in the standard scenario, and the engine as the pull requests propose it | [`2026-10-23-e2`](../results/2026-10-23-e2/README.md) |
| W5 | `w5-r1` to `w5-r5`, `w5-pr-r*`, `w5-ext-r*`, `w5-cap-r1` to `w5-cap-r3` | The configuration to deploy beside its control, the change's configurations, preempt-resubmit and the standard scenario, and 4 and 6 callers, each round on a sample of its own | [`2026-10-24-e2-w5`](../results/2026-10-24-e2-w5/README.md) |
| FB | `fb-r1` to `fb-r5` | Voice alone, tuned vLLM and the configuration to deploy on the code of 25 October | [`2026-10-25-e2-fb`](../results/2026-10-25-e2-fb/README.md) |
| RS | `rs-r1` to `rs-r5` | Tuned vLLM and the configuration to deploy on v1.2, with the KV sampler | [`2026-10-26-e2-rs`](../results/2026-10-26-e2-rs/README.md) |
| SW | `sweep-screen`, `sweep-screen2`, `fp8-repro` | The step cap, the window and the gateway's cap, screened | [`2026-10-27-sweep`](../results/2026-10-27-sweep/README.md) |
| RG | `rg-r1` to `rg-r5` | Two cards' arrangements, and the step cap confirmed | [`2026-10-28-rounds`](../results/2026-10-28-rounds/README.md) |
| OR, OS | `opt-r1` to `opt-r5`, `opt-s1`, `opt-s2` | The serving engine's additions, then the decode hold and speculative decoding | [`2026-11-02-opt-rounds`](../results/2026-11-02-opt-rounds/README.md) |
| The envelope | `env-r1` to `env-r3` | Version 1.24's envelope, at 4, 12 and 24 agent sessions | [`2026-11-08-envelope`](../results/2026-11-08-envelope/README.md) |
| — | `gray-smoke`, `demo` | Gray end to end, and the demo video's two points | — |

Rounds rotate their engine groups, so that no configuration always runs at the same hour. The A100 run's campaigns are
in [`benchmarks/a100`](../a100), its records in [`2026-10-26-a100`](../results/2026-10-26-a100/README.md).
`FLATSHARE_PROFILE` names the GPU node's hardware profile ([`deploy/profiles`](../../deploy/profiles)) when it is not
the frozen box's, and the report then gives the points a workload of their own.

## The headline configurations

The configurations the docs' tables report, by arm file and the campaigns whose records hold their runs, in the
discriminating scenario unless the row says otherwise. Each W5 arm file is its W4 one with a warm-up turn opening each
call.

| Configuration | W4: arm file, campaigns | W5: arm file, campaigns | FB and RS: campaigns, W5's arm files |
|---|---|---|---|
| Voice alone, the SLO's reference | `a1-voice`: w2, s1, s2, r3, r4 | `w5-a1-voice`: w5-r1 to w5-r5 | fb-r1 to fb-r5 |
| Tuned vLLM | `disc-frozen-s24`: w2c, r1 to r4 | `w5-disc-frozen-s24`: w5-r1 to w5-r5, w5-pr-r1 to w5-pr-r5 | fb-r1 to fb-r5, rs-r1 to rs-r5 |
| Step cap 1,024, no threshold | `disc-a4-b1024-t0-s24`: w2c, r1 to r4 | — | — |
| Static reservation, 2,048 tokens | `disc-a6-reserve2048-s24`: w2c, r1 to r4 | — | — |
| SGLang with HiCache | `disc-a7b-sglang-hicache-s24`: w2c, r1 to r4 | — | — |
| Preempt-resubmit before the first byte | `disc-a8c-resubmit-s24`: w2c, r1 to r4 | `w5-disc-a8c-resubmit-s24`: w5-ext-r1 to w5-ext-r5 | — |
| Arm 9 | `disc-a9-s24`: a9, r1 to r4 | — | — |
| #63350 alone (KV preemption) | `disc-pr-a-s24`: upstream, upstream-r1, upstream-r2 | `w5-disc-pr-a-s24`: w5-pr-r1 to w5-pr-r5 | — |
| #63350 + #63351 (the budget cap, 16 tokens) | `disc-pr-ab-s24`: upstream, upstream-r1, upstream-r2 | `w5-disc-pr-ab-s24`: w5-pr-r1 to w5-pr-r5 | — |
| #63350 + #63351, a 48-token window | `disc-pr-ab48-s24`: upstream, upstream-r1, upstream-r2 | `w5-disc-pr-ab48-s24`: w5-r1 to w5-r5, w5-pr-r1 to w5-pr-r5, w5-ext-r1 to w5-ext-r5 | fb-r1 to fb-r5, rs-r1 to rs-r5 |
| The same on make-room as v0.3 built it (W5's control) | — | `w5-disc-pr-ab48-v03-s24`: w5-r1 to w5-r5 | — |
| Standard scenario (16 agent sessions): tuned vLLM | `a4-b512-t256-s16`: w2, s1, s2 | `w5-std-frozen-s16`: w5-ext-r1 to w5-ext-r5 | — |
| Standard scenario: arm 9 | `a9-s16`: a9-standard, s1, s2 | — | — |
| Standard scenario: #63350 + #63351, a 48-token window | — | `w5-std-pr-ab48-s16`: w5-ext-r1 to w5-ext-r5 | — |
| 4 and 6 callers: #63350 + #63351, a 48-token window | — | `w5-disc-pr-ab48-c4-s24`, `w5-disc-pr-ab48-c6-s24`: w5-cap-r1 to w5-cap-r3 | — |
| 4 and 6 callers: voice alone | — | `w5-a1-voice-c4`, `w5-a1-voice-c6`: w5-cap-r1 to w5-cap-r3 | — |

W5's control runs on the `make-room-v03` variant, whose module name differs, so that the engine check fails a point on
the other module before it starts. No point fronts two GPU nodes at once. The points written for two nodes never ran,
and their arm files and code are on the `two-node` branch, while the main line's gateway places sessions on several
nodes itself ([Nodes](../../crates/flatshare-gateway/README.md#nodes)). The resource gate
([RESOURCE-GATE.md](../quality/RESOURCE-GATE.md)) measures each card's part of an arrangement as a point of its own, on
one node at a time, and the voice worker binds each call to its node
([the worker's nodes](../../python/flatshare_voice/README.md#nodes)).

## Results

`flatshare-bench e2 report RUN_DIR... --out DIR` builds a frontier from any set of run directories or records, across
campaigns too ([its rules](../../python/flatshare_bench/README.md#e2-points)). The discriminating scenario and the
calls-per-GPU points each get their own section of `frontier.md` and their own figure. W5's
[`figure.toml`](../results/2026-10-24-e2-w5/figure.toml) names the rows of [REPORT.md](../../docs/REPORT.md)'s headline
figures, and `--dots FILE` draws the rows such a file names. A figure of several GPUs gives each its rows under a
`[[group]]` table with a `title`.

The repository keeps a compact record of each point, about 15 kB with no credential and no home path, and the report
gives the same tables and figures from records as from run directories. `make reproduce-e2` regenerates the report of
W2–W4's records into `build/reproduce-e2`, `make reproduce-e2-w5` W5's and `make reproduce-e2-a100` the A100 run's:

```sh
python/flatshare_bench/.venv/bin/flatshare-bench e2 export $FLATSHARE_E2_RESULTS/w2c/* \
  --out benchmarks/results/2026-10-23-e2
make reproduce-e2
```

The run directories, with the raw records the report does not need, are release assets: W2–W4's in v0.3, W5's and FB's
in v1.0 (`leaseholder-raw-records-w5.tar.gz`, `leaseholder-raw-records-fb.tar.gz`, named before the project's rename),
and RS's in v1.3 (`flatshare-raw-records-rs.tar.gz`).

## The demo video

The same caller on two engines while coding agents run. [`demo.campaign`](demo.campaign) measures tuned vLLM and the
configuration to deploy in the discriminating scenario with its `save-audio` line, and [`demo.py`](demo.py) renders them
from the calls' audio and turn times
([`flatshare-voice bots --save-audio`](../../python/flatshare_voice/README.md#measurement-callers)). On a workstation,
with the two run directories fetched into one directory:

```sh
benchmarks/e2/campaign.sh benchmarks/e2/demo.campaign
make demo-video DEMO=$FLATSHARE_E2_RESULTS/demo
```

`make demo-video` renders the video to `build/flatshare-demo.mp4` (`DEMO_OUT`), and
[the repository README](../../README.md)'s animation to `docs/figures/e2-demo.webp` with its dark theme and phone
layout, in seven minutes on an Apple M5 Max. It reads W5's report for the voice SLO, and the A100's for the same two
configurations there. The v1.0 release's `leaseholder-raw-records-demo.tar.gz` holds the two points' run directories,
which are not in the tables.

**What it shows.** Each frame shows a panel per call, the baseline's in gray and the configuration to deploy's in blue.
A clock counts each wait against the SLO, voice alone's median V2V p95 plus 250 ms (2.03 s), bars give each reply's V2V,
and the agents' tokens a second stand at the foot. Both calls play the same turns from the same moment, and only a
caller's pause between turns is stretched, never a measured wait.

The frames follow the docs' figures, in STIX Two and their palette. The video is 1920×1080 at 60 frames a second, H.264
with AAC sound, and the animation is the first turn alone, silent and looping, as lossless WebP.
`python/flatshare_bench/tests/test_demo.py` checks the renderer on a synthetic pair of calls, and `test_figures.py`
beside it the four WebPs.

## Soaks

A soak loads the box as it serves for hours and keeps every attempt and every scrape.
[EXPERIMENTS.md](../../docs/EXPERIMENTS.md#the-soak-the-production-path-for-hours) says what it measures, and
[`flatshare-bench soak`](../../python/flatshare_bench/README.md#soaks) gives its options and files. It runs on cyan and
takes the lock of E2's points and the drills. It changes nothing on the box, and refuses to start on a box that does not
serve as deployed, saying what to fix.

**Before the first soak**, once:

- **The arrangement to deploy.** The voice worker in serve mode
  ([the voice plane](../../ops/RUNBOOK.md#the-voice-plane)): `variant-vad-turns.conf` unlinked, the `serve` drop-in
  linked as `variant-serve.conf`, and the worker restarted. To keep E2's fixed endpointing for the synthetic callers
  instead, leave `vad-turns` linked beside `serve` with a drop-in setting `FLATSHARE_AUDIO=synthetic`. The GPU node runs
  its serving profile (`make-room-serve`) with the measurement drop-in off (`variant.sh measure off`). With two nodes,
  `gateway.toml` has both `[[nodes]]` and `flatshare-tunnel@gray` runs
  ([Two GPU nodes](../../ops/RUNBOOK.md#two-gpu-nodes)). No E2 drop-in is left on cyan.
- **The guard key.** cyan reaches the GPU nodes only through its tunnel's keys. From a workstation,
  `deploy/authorize-guard.sh cyan phastform` (and `cyan gray`) makes cyan's `~/.ssh/flatshare_guard_ed25519` and lets it
  run one command on the node, `deploy/bin/node-report`, which lists the GPU's processes and the node's unit lines.
  `--guard NODE=off` leaves a node unguarded, which the record says.
- **The reference.** The SLO is voice alone's p95 with the same calls and worker configuration, plus 250 ms, and a
  two-hour voice-alone soak gives it (`--agents-sessions 0`).

**Run one** on cyan under a transient unit, so that it outlives the SSH session and stops cleanly:

```sh
bench=~/flatshare/venvs/bench/bin/flatshare-bench
soak() { systemd-run --user --unit flatshare-soak --collect -p KillMode=mixed \
  -p TimeoutStopSec=900 $bench soak run "$@"; }
# the reference, first
soak --hours 2 --agents-sessions 0 --out ~/flatshare/runs/soak/ref-voice-alone
# once it has ended: its p95 is P95
$bench soak report ~/flatshare/runs/soak/ref-voice-alone
# two nodes bounding agents at 4 + 12: a few under the bounds (the runbook's Two GPU nodes, Sizing)
soak --hours 24 --slo-reference P95 --calls-per-hour 20 --turns 10 --agents-sessions 14 \
  --agents-data weka-x32-long \
  --out ~/flatshare/runs/soak/NAME
# its progress: each tool's run, each hour's line
journalctl --user -u flatshare-soak -f
# ends it early, its summary written
systemctl --user stop flatshare-soak
```

While a guarded GPU holds another's process, or cannot be read three minutes in a row, the soak stops its load until
every GPU has been clear for five minutes. A 24-hour soak leaves about 40 MB on cyan, beside AIPerf's export.

**Read the summary** (`flatshare-bench soak report DIR`) per hour and overall:

- **calls** refused by `capacity` mean admission working, by `placement` the gateway placing a call on no GPU node, and
  `no_agent` a dispatch LiveKit lost;
- **replies** over the SLO should stay within 5% ([SLO.md](../../docs/SLO.md));
- **restarts** count the units' failures and restarts and the counters that went down, beside the guard's pauses.

An hour that differs from the rest points at `events.jsonl` and `units.jsonl` for the same hour. A GPU node's units are
named `NODE:UNIT` (`phastform:flatshare-engine.service`).

**Commit it** as a compact record with `flatshare-bench soak export DIR --out benchmarks/results/DATE-soak`, a few MB
for 24 hours, which `soak report` reads as it reads the directory.
