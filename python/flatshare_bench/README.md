# flatshare-bench

`flatshare-bench` is Flatshare's benchmark toolchain, for a reader who runs or changes a measurement. It replays the
SemiAnalysis Claude Code traces, 393 sessions and 98,827 requests with their subagents and think times, against a 16–32k
context with Flatshare's session metadata on every request. It measures the tier costs (E0), server-side voice turns,
E2's points and hours-long soaks. [ARCHITECTURE.md §6.3](../../docs/ARCHITECTURE.md#63-e2s-harness) places E2's harness
in the system.

| Piece | Module | What it does |
|---|---|---|
| `flatshare-bench scale-traces` | `traces.py` | Shrinks every request of the corpus by 32× or 64×, keeping its prefix structure |
| AIPerf loader `weka_trace_scaled` | `aiperf_plugin.py` | AIPerf's Weka loader, fixed to read each trace's own block size |
| AIPerf endpoint `flatshare_chat` | `aiperf_plugin.py` | Chat completions with Flatshare's session metadata |
| `configs/agentic_replay.yaml`, `configs/aiperf.env` | — | AIPerf's `agentic_replay`: steady-state agent sessions with recycling |
| `flatshare-bench mock-server` | `mock_server.py` | A recording OpenAI-compatible mock on 127.0.0.1, for dry runs |
| `flatshare-bench tiers` | `tiers.py` | E0: time to first token from the GPU pool, the CPU tier and recompute |
| `flatshare-bench voice-turn` | `voice_turn.py` | One server-side voice turn timed per stage |
| `flatshare-bench kv-metrics` | `kv_metrics.py` | vLLM's KV block lifecycle histograms, compared across a run |
| `flatshare-bench e2` | `e2/` | E2's points: one measured from its arm file (`run`), the frontier (`report`) and compact records (`export`) |
| `flatshare-bench drill` | `drills.py` | The client side of the failure drills in `ops/drills` |
| `flatshare-bench soak` | `soak/` | The production path for hours (`run`), its summary (`report`) and its compact record (`export`) |
| Background load | `load.py` | `prefill:<tokens>x<n>` (cold prefills) or `decode:<tokens>x<n>` (long decodes) beside a probe |
| `python -m flatshare_bench.quick_start` | `quick_start.py` | The repository README's quick-start result, from the probe's logs |

It needs Python 3.12, or 3.13 on macOS (`--python 3.13`), since `flatshare-bench e2 run` needs `os.waitid`, which
macOS has from 3.13. AIPerf, pinned to `aiperf==0.13.0`, is the `aiperf` extra, which the `test` extra includes. A
report does not need it, so `make reproduce-e2` syncs 28 packages, not 140. Installed with it, the package registers
both plugins through the `aiperf.plugins` entry point. On cyan, `deploy/cyan/bootstrap.sh` builds
`~/flatshare/venvs/bench` from `uv.lock`, and for development:

```sh
uv sync --python 3.12 --extra test      # in python/flatshare_bench: its own .venv, from uv.lock
```

## Quick start (cyan)

The traces, scaled, then 60 s of AIPerf's agentic replay against the mock server:

```sh
# Environment: the venv that deploy/cyan/bootstrap.sh builds
export PATH=$HOME/flatshare/venvs/bench/bin:$PATH

# Data
cd ~/flatshare/data
curl -sSL --fail -o cc-traces-weka-062126/traces.jsonl --create-dirs \
  https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl
flatshare-bench scale-traces cc-traces-weka-062126/traces.jsonl weka-x32 --factor 32
flatshare-bench scale-traces cc-traces-weka-062126/traces.jsonl weka-x64 --factor 64
# A quick-check subset: the 16 short traces of flatshare-sim's test data (8 with subagents)
ls ~/flatshare/src/crates/flatshare-sim/testdata/weka-x32-sub16 | sed 's/\.json$//' > sub16.ids
flatshare-bench scale-traces cc-traces-weka-062126/traces.jsonl weka-x32-sub16 --factor 32 \
  --ids sub16.ids

# Mock on loopback (keep clear of cyan's deployed ports, the gateway's 18080 and LiveKit's
# 17880-17882)
mkdir -p ~/flatshare/runs/w0-e2e && cd ~/flatshare/runs/w0-e2e
setsid nohup flatshare-bench mock-server --port 19180 \
  --log $PWD/mock-aiperf.jsonl < /dev/null > mock.out 2>&1 &
echo $! > mock.pid
ss -lntp | grep 19180          # 127.0.0.1:19180 only

# AIPerf agentic_replay, 60 s
cd ~/flatshare/src/python/flatshare_bench
. configs/aiperf.env
FLATSHARE_BENCH_URL=http://127.0.0.1:19180/v1/chat/completions \
FLATSHARE_BENCH_DATA=$HOME/flatshare/data/weka-x32-sub16 FLATSHARE_BENCH_SESSIONS=8 \
FLATSHARE_BENCH_DURATION=60 \
FLATSHARE_BENCH_GRACE=15 FLATSHARE_BENCH_THINK_CAP=3 FLATSHARE_BENCH_RAW=true \
FLATSHARE_BENCH_ARTIFACTS=$HOME/flatshare/runs/w0-e2e/aiperf \
  aiperf profile --config configs/agentic_replay.yaml

# What reached the server
python -c 'import json, sys
r = json.loads(open(sys.argv[1]).readlines()[-1])
print(r["headers"], r["body"])' ~/flatshare/runs/w0-e2e/mock-aiperf.jsonl
kill $(cat ~/flatshare/runs/w0-e2e/mock.pid)
```

On 2026-10-22 the 8 agent sessions sent 7 warm-up and 236 measured requests with no error. All 243 carried the four
fields of [the wire metadata](#wire-metadata), and `X-Session-ID` named the agent session's root on the 103 subagent
requests too.

## Trace scaling

Each request of the corpus is a list of 64-token prefix hash blocks, and the median main-agent prompt is 254k tokens.
AIPerf's own caps do not help, since `--synthesis-max-isl 32768` keeps 2% of main-agent requests and does not filter
subagents, and `--max-context-length 32768` keeps no trace. So `scale-traces` shrinks the tokens a block holds and
leaves the hash ids alone:

```text
block_size -> block_size / factor
in         -> len(hash_ids) * new block_size
out        -> ceil(out / factor)
```

Prefix identity, subagent forks and the cache hit rate are kept exactly. Every request shrinks by the same factor,
outputs included, so a cache miss keeps its cost relative to decode work, and since
`ceil(ceil(out / f) / (64 / f)) == ceil(out / 64)`, AIPerf splits each turn into the same blocks. Sessions, subagents
and timing are copied as recorded, with no length cap.

| Factor | Context | Largest `in + out` | Use |
|---|---|---:|---|
| 32 (`block_size` 2) | 32k | 31,144 | The default. The main agent's context is 7.9k tokens at the median, 20.3k at p90 |
| 64 (`block_size` 1) | 16k | 15,572 | Smaller KV pools |

```sh
flatshare-bench scale-traces traces.jsonl weka-x32 --factor 32    # the HF JSONL or a
                                                                  # directory of *.json
flatshare-bench scale-traces traces.jsonl weka-sub --ids ids.txt  # a subset; or --limit N
```

The full corpus takes about 22 s and 1.8 GB of RAM a factor on cyan, and each output is 1.8 GB.

## Wire metadata

The AIPerf endpoint renders each request's metadata through `flatshare_bench.protocol.SessionTagger`, whose names and
values mirror the Rust crate `flatshare-proto`:

| Field | Value |
|---|---|
| Header `X-Session-ID` | The agent session's root, which a parent agent and its subagents share: AIPerf's `root_correlation_id` for the session play |
| Header `X-Flatshare-Class` | `realtime`, `interactive` or `background` |
| Body `prompt_cache_key` | The session id |
| Body `priority` | vLLM's priority, which it schedules lowest first: realtime 0, interactive 1, background 2 |

AIPerf builds the endpoint itself in every worker, so the plugin reads its configuration from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `FLATSHARE_BENCH_CLASS` | `background` | The class of every session |
| `FLATSHARE_BENCH_SEND_PRIORITY` | `true` | `false` sends no `priority`, as a plain client would. vLLM 0.32 acts on it only under `--scheduling-policy priority` |

Keep `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID` off, since it overwrites the plugin's `X-Session-ID` with each
agent's own id. vLLM 0.32 reads `X-Session-ID` as the request's session and ignores `prompt_cache_key`, and the control
plane strips a client's `priority` and sets its own.

## AIPerf `agentic_replay`

`agentic_replay` keeps N agent sessions live. Each lane starts at a random point of a trace, replays it with its
recorded think times and subagents, then recycles a trace under a fresh cache-bust marker. AIPerf 0.13.0 selects it only
from YAML, so `configs/agentic_replay.yaml` is the template, and AIPerf fills its `${VAR:default}`:

| Variable | Default |
|---|---|
| `FLATSHARE_BENCH_URL` | `http://127.0.0.1:8000/v1/chat/completions` |
| `FLATSHARE_BENCH_API_KEY` | Empty, no `Authorization` header. Through the gateway, the trusted `agents` tenant's key |
| `FLATSHARE_BENCH_DATA` | `data/weka-x32`, a `scale-traces` output directory |
| `FLATSHARE_BENCH_MODEL`, `FLATSHARE_BENCH_TOKENIZER` | `Qwen/Qwen3-4B-Instruct-2507-FP8` |
| `FLATSHARE_BENCH_SESSIONS` | 16 live agent sessions |
| `FLATSHARE_BENCH_DURATION`, `FLATSHARE_BENCH_GRACE` | 1800 s and 120 s |
| `FLATSHARE_BENCH_THINK_CAP` | A 600 s cap on each think time |
| `FLATSHARE_BENCH_BURST_STARTS` | `true` |
| `FLATSHARE_BENCH_ARTIFACTS` | `runs/agentic-replay` |
| `FLATSHARE_BENCH_RAW` | `false`; `true` also exports every request with its headers and payload |
| `FLATSHARE_BENCH_SEED` | 11 |

Source `configs/aiperf.env` first, since AIPerf takes the settings it holds from the environment alone. It turns
AIPerf's dataset cache on, keeps the session-id override off, and raises the dataset timeouts, since the full corpus
takes about 300 s and 17–22 GB of RAM to load.

- **The dataset cache** keeps the dataset a run builds, under `~/.cache/aiperf/dataset_mmap`, for any later run of the
  same corpus, tokenizer, seed and settings. So an E2 round's points build it once. On peacock the 48 long traces'
  dataset took 138 to 165 s to build and 5 s to load. The plugin keeps its loader's datasets under a hash of its own
  code, so a changed loader builds anew. Nothing evicts an entry, 0.33 GB for the whole corpus, so delete the directory
  to reclaim it.
- **The `weka_trace_scaled` loader.** AIPerf 0.13.0 gives its Weka loader a block size of 64 that overrides a trace's
  own. On the 32× corpus the stock loader therefore reproduced about 77% of the shared prefix, with prompts 22% short at
  the median. The plugin's loader drops that override, and every shared prefix and prompt length then matches the trace
  (`tests/test_loader.py`).
- **Burst starts.** With spread starts each lane first waits out its recorded gap, which `interTurnDelayCapSeconds` does
  not cap. A 60 s check reached 2 of 8 agent sessions and 31 requests, against 8 of 8 and 236 with burst starts.
- **vLLM's flags.** Cached-token counts need `--enable-prompt-tokens-details`, and `priority` takes effect only under
  `--scheduling-policy priority`.
- **Assistant history.** AIPerf replays assistant content synthesized from hash blocks, never the model's reply, so the
  KV written while decoding turn k is never matched by turn k+1. The Weka loader hardcodes that context mode,
  `deltas_with_responses`.

## Latency probes

`flatshare-bench tiers` measures what resuming a session costs from each tier (E0). For each prompt length a unique
prompt of random token ids goes out three times, each asking for one token. The first is cold, so every token is
recomputed, the second a GPU prefix-cache hit, and the third, once other prompts have pushed it out of the pool, a
restore from the CPU tier. vLLM's `prefix_cache_hits` and `external_prefix_cache_hits` counters, read around each
request, verify where its tokens came from, and a sample whose eviction or restore went wrong is flagged invalid. The
run fits the GPU hit and the restore as `a + b·L` and the recompute as `a + b·L + c·L²`, the fits `flatshare-core`
reads, and exits 3 if any sample was invalid.

`flatshare-bench voice-turn` times one turn of a voice agent's cascade. It runs STT on the utterance, then the LLM up to
its first sentence of 20 characters, as LiveKit's tokenizer cuts one, then TTS to its first audio of that sentence.

Both take `--background prefill:8192x1` or `decode:1024x8` to measure under agent load, and write `run.json`, the
samples and `summary.json` to `--out`. A server's address is written as the gateway's `upstream` is, `http://host:port`
or `unix:` and a socket's path, and on cyan defaults to the GPU node's servers through the tunnel's sockets.

```sh
flatshare-bench tiers --reps 5 --out runs/e0-idle
flatshare-bench tiers --lengths 1024,4096,16384 --background decode:1024x8 --out runs/e0-decode
flatshare-bench voice-turn --audio turn.wav --reps 30 --background prefill:8192x1 \
  --out runs/voice-x1
```

## KV block lifecycle

`flatshare-bench kv-metrics` compares three of vLLM's KV block histograms, the reuse gap, the idle time before eviction
and the lifetime, between two snapshots of an engine's `/metrics`. The engine needs `--kv-cache-metrics`. Where evicted
blocks had been idle for less time than blocks typically wait between reuses, LRU is evicting KV that would have been
reused. Reuse gaps are recorded only for blocks that survived until reuse, so this is a signal, not a bound. W0's
reading is in [BENCHMARKS.md](../../docs/BENCHMARKS.md#how-do-the-agents-use-the-kv-cache).

```sh
engine=unix:$XDG_RUNTIME_DIR/flatshare/phastform-engine.sock  # on cyan: the engine
                                                              # through its tunnel
flatshare-bench kv-metrics snapshot $engine before.prom
# ... run the load ...
flatshare-bench kv-metrics snapshot $engine after.prom
flatshare-bench kv-metrics compare before.prom after.prom
```

## E2 points

`flatshare-bench e2 run ARM.toml --out DIR` measures one point of E2 on cyan, the measurement calls beside AIPerf's
agentic replay, through the gateway, on the engine the GPU node runs. [`benchmarks/e2`](../../benchmarks/e2/README.md)
holds the arm files and `campaign.sh`, which measures a list of them and switches the GPU node's engine between them.

```sh
flatshare-bench e2 run ~/flatshare/src/benchmarks/e2/arms/a1-voice.toml \
  --out ~/flatshare/runs/e2/w2/a1-voice
# the engine an arm needs, as variant.sh use takes it
flatshare-bench e2 engine a2-stock-s16.toml
# the frontier across points and campaigns; two configurations' waterfall
flatshare-bench e2 report ~/flatshare/runs/e2/{w2c,a9,r1,r2}/disc-* --out ~/flatshare/runs/e2/disc \
  --waterfall disc-frozen-s24 --waterfall disc-a9-s24
# compact records
flatshare-bench e2 export ~/flatshare/runs/e2/w2b/* --out benchmarks/results/2026-10-23-e2
# cyan's units as gateway.toml and the worker's defaults have them
flatshare-bench e2 reset
```

| `e2 run` option | Default | Meaning |
|---|---|---|
| `--node NODE` | the host's own | The GPU node, reached through its tunnel on cyan (`flatshare-tunnel@NODE`) with the gateway and the worker pointed at its sockets |
| `--gpu-source NODE=REVISION` | none | A GPU node's deployed revision, which `run.json` keeps beside cyan's |
| `--profile NAME` | the host's, else the frozen box's | The hardware profile the engine check holds the engine to |
| `--agents-seed N` | the arm file's | AIPerf's seed, as a campaign's `agents-seed` line sets it |
| `--kv-sample-s SECONDS` | 0.5 | How often an agent point samples the engine's KV use; 0 samples nothing, as W5's points ran |
| `--save-audio` | off | The callers save each call's audio for the demo video, which measures nothing differently |

### How a point runs

1. **The engine check.** The engine's `/server_info` must report the command line `variant.sh use` makes for the arm,
   down to vLLM's scheduler class, so that no engine passes for another. A mismatch fails the point before it starts,
   naming the command that fixes it. `/server_info` needs the measurement drop-in (`variant.sh measure on`).
2. **cyan's units.** The gateway runs on `~/.config/flatshare/gateway-e2.toml`, which is `gateway.toml` with the
   inference canary off, so that the engine serves only the point's requests, and the arm's `[gateway]` table merged in.
   Only a unit whose configuration changes restarts, and the worker must end turns with fixed endpointing (`vad-turns`).
3. **A cold start.** The engine's prefix cache and its CPU tier are emptied.
4. **The replay.** AIPerf starts on cores 16–23, and the calls wait for its profiling phase plus the warm-up.
5. **The window.** The gateway's and the engine's `/metrics` are saved, `flatshare-voice bots` runs in the voice-plane
   namespace on cores 24–31, and they are saved again. The calls are the window every metric covers, and a point without
   calls runs for `[agents] window_s` instead.
6. **The end.** AIPerf gets SIGINT and writes its exports, the window's gateway lines and worker journal are saved, and
   `summary.json` is written.

A component that fails fails the point, and `run.json` keeps the error and the tool's last lines. A closed SSH session
does not stop a point. A complete point is never overwritten, and an incomplete one is moved aside to
`DIR.incomplete-<UTC time>`. One `e2 run` or `e2 reset` runs at a time, under the lock
`~/flatshare/runs/.flatshare-bench-e2.lock`, which holds its pid.

cyan's addresses, cores and profile are the defaults. Another control host, or a single host that runs every part, sets
them in `~/.config/flatshare/e2-host.toml` ([HARDWARE.md](../../docs/HARDWARE.md#single-host-mode)).

### The arm file

Unknown keys and wrong types are errors, and the point is named after the file.

| Key | Default | Meaning |
|---|---|---|
| `arm` | required | E2's arm, which names the point's frontier: 1 to 10, then a letter for a sub-arm (`"7a"`) or a suffix for a factor (`"4-no-preemptive"`) |
| `label` | required | What the point is, for the report |
| `router` | none | `"smg"` starts `flatshare-smg` (arm 8b) |
| `[voice] callers`, `turns` | 2, 25 | Concurrent calls and turns per call; `callers = 0` is the agents alone |
| `[voice] barge_in`, `seed` | 0.25, 0 | The share of turns on which the caller interrupts the reply, and the callers' seed |
| `[voice] warmup_turns` | 0 | Turns that open each call, played but in no V2V sample or stage |
| `[voice] preemptive_generation`, `turn_hints`, `events` | `true`, `false`, `true` | LiveKit's preemptive generation, the worker's turn hints to the engine, and its lifecycle events to the gateway |
| `[agents] sessions` | 0 | Agent sessions AIPerf keeps live, through the gateway as the `agents` tenant; 0 is voice alone |
| `[agents] think_cap`, `warmup` | 120, 60 | The cap on each think time, and the profiling before the calls, in seconds |
| `[agents] data`, `seed` | `"weka-x32"`, 11 | The corpus under `~/flatshare/data`, and AIPerf's seed |
| `[agents] window_s` | 355 | A point without calls: its window in seconds, the median of W5's two-caller points' calls |
| `[gateway]` | none | Merged into `gateway.toml`, tables key by key: any of [the gateway's keys](../../crates/flatshare-gateway/README.md) but `[gateway.nodes]` and `[gateway.placement]`, which are refused |
| `[engine] variant` | `"frozen"` | A directory of [`deploy/phastform/variants`](../../deploy/phastform/variants/README.md), or `frozen` |
| `[engine] flags` | `[]` | Engine flags, each a plain word, added to its command line |

### The run directory

| File | Contents |
|---|---|
| `arm.toml`, `run.log` | The arm file as run, and each step, timestamped |
| `run.json` | `status` (`complete`, or `failed` with the `error`), the host and `node`, the times, the arm and options, the versions (cyan's and the GPU node's deployed revisions, the tools'), the engine and its checked flags, what cyan ran and the window |
| `engine.json` | The engine's whole configuration from `/server_info`, without credentials |
| `aiperf.log`, `aiperf/` | AIPerf's output and exports, one record per request |
| `bots.log`, `bots/` | The callers' output, `turns.jsonl` and `summary.json`, and with `--save-audio` each call's audio |
| `metrics/` | Both servers' `/metrics` at the window's start and end, and an agent point's KV samples, `engine-kv.csv` |
| `gateway.jsonl` | The gateway's JSON request lines for the window, with each first byte, outcome and tokens; a gateway that logs text fails the point |
| `voice-worker.log` | The worker's turn and session lines ([`flatshare-voice`](../flatshare_voice/README.md#lifecycle-events-and-turn-lines)) |
| `gpu.csv` | The GPU node's `nvidia-smi` samples once a second, which `campaign.sh` writes |
| `summary.json` | The point's E2 metrics, below |

### `summary.json`

Every metric covers the calls. A metric the gateway or the engine does not export is null, never zero, and a counter
that went down, its process restarted during the calls, fails the point.

| Key | What it holds |
|---|---|
| `window_s` | The calls' length |
| `voice` | From the callers: turns, errors, `v2v_ms` and `barge_in_to_silence_ms` (n, p50, p95, p99) and barge-ins |
| `worker` | From the worker's journal: sessions, turns, each stage's times, and the lifecycle events sent, dropped and failed |
| `background` | From AIPerf's records of the agent requests that ended in the calls: requests, errors, tokens a second, `ttft_ms` and `itl_ms`, and the same over the calls' first 240 s |
| `gateway` | Both classes' first byte, exact from `gateway.jsonl`, what admission shed, refused and resubmitted, and every counter that moved |
| `engine` | Preemptions, prompt tokens by source at first scheduling, queue-limit rejections, and the CPU tier's loads and stores |
| `kv` | The `running_share`, the share of samples in which running requests' KV left less than 2,048 tokens free, as flatshare-sim's running share counts it; free blocks; running and waiting requests |
| `gpu` | In a compact record only: the median SM clock, highest temperature and the GPU's name |

The agents' token rate comes from AIPerf's records, since the gateway's `completed` counts a response relayed in full
whatever its status. The gateway's own count of tokens a second stays beside it as a check.

### The report

`flatshare-bench e2 report RUN_DIR... --out DIR` writes `frontier.csv`, `frontier.md`, a frontier figure per workload
and `waterfall.svg`, from run directories or compact records alike.

- **Workloads.** A point's workload is the agents' corpus and the number of callers, with the GPU and the model off the
  frozen box, and the warm-up turns where points differ in them. Points of different workloads are never compared.
- **The frontier.** A point is on its arm's frontier when no other point of the arm and workload has at least as high
  an agents' token rate at a V2V p95 at least as low, one of the two strictly better. Arm 9's ablations and the
  change's configurations each count as their own arm, and labels naming the pull requests by letter get their numbers,
  #63350 and #63351.
- **Repeats.** Runs of one arm file in several campaigns are one configuration. `frontier.md` keeps a row per run, then
  the median and range, and the frontier places the configuration at its medians.
- **The fixed window.** Beside the agents' token rate over the whole calls, whose length differs by configuration, the
  report gives it over their first 240 s.
- **Failed turns.** A turn not answered within the callers' 30 s timeout is a sample censored at 30,000 ms, and `≥`
  marks a percentile it sets. A configuration with failed turns is placed on no frontier.
- **The figures** give each configuration its V2V p95 against the voice SLO, 250 ms above voice alone with as many
  callers on the same GPU, and the agents' token rate, each the median of its runs. The change's configurations are
  blue, the one to deploy saturated, and the rest gray. `waterfall.svg` splits each voice turn into endpointing, the
  reply to its first sentence, TTS's first audio and the rest, as means; `--waterfall POINT` picks the rows, and
  `--dots FILE` draws `voice-p95.svg` from a TOML list.
- **Columns some points have** appear when one does, among them the CPU tier's traffic in tokens, at the pool's bytes a
  token, 147,492 on the frozen box.

### The compact record

`flatshare-bench e2 export RUN_DIR... --out DIR` writes each complete point's record into `DIR/CAMPAIGN/POINT`: its
`run.json`, `arm.toml` and the callers' files, an `engine.json` reduced to which engine it ran, and a `summary.json`
that keeps what the report reads from the files left out. A record is about 15 kB, its home paths are written from `~`,
and it holds no credential.

The report gives the same tables and figures from records as from run directories, byte for byte, so the records are
what the repository keeps, under [`benchmarks/results/2026-10-23-e2`](../../benchmarks/results/2026-10-23-e2/README.md),
and `make reproduce-e2` regenerates the report from them. The raw run directories are the
[v0.3 release](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3)'s asset
`leaseholder-e2-runs-2026-10-24.tar.gz`, named before the project's rename.

## Soaks

`flatshare-bench soak run --hours H --out DIR` loads the box as it serves for hours, on cyan. Calls arrive on a schedule
against the voice worker in serve mode
([`flatshare-voice arrivals`](../flatshare_voice/README.md#arriving-calls-a-soaks)), beside AIPerf's agentic replay
through the gateway, with a scrape of the service every minute and a guard on the GPUs.
[EXPERIMENTS.md](../../docs/EXPERIMENTS.md#the-soak-the-production-path-for-hours) says what it measures, and
[E2's README](../../benchmarks/e2/README.md#soaks) how to run one.

```sh
# P95: the p95 of a voice-alone soak (--agents-sessions 0), as soak report prints it
flatshare-bench soak run --hours 24 --slo-reference P95 --agents-sessions 14 \
  --out ~/flatshare/runs/soak/day
# the hours' table, then the record to commit
flatshare-bench soak report ~/flatshare/runs/soak/day
flatshare-bench soak export ~/flatshare/runs/soak/day --out benchmarks/results/2026-10-28-soak
```

A soak first checks the box serves as deployed and refuses to start with what to fix: the worker in serve mode, no E2
drop-in on cyan, the gateway ready with the model, and every GPU clear. AIPerf starts as E2 starts it, and the window
waits for its warm-up. Through the `--hours` window the service is scraped and the GPUs checked, a tool that ends early
is started again after a backoff, and each hour's summary is written as it ends. A guard's pause stops the load until
every GPU has been clear for `--guard-clear-s`. After the window the calls in progress play their turns for at most
`--drain-s`. SIGINT or SIGTERM ends a soak the same way, and a second hangs the calls up.

| Option | Default | Meaning |
|---|---|---|
| `--hours` | required | The window, at most two weeks |
| `--slo-reference MS`, `--slo-margin MS` | none, 250 | Voice alone's V2V p95 with the same calls on the same box; the summary counts the replies over it plus the margin |
| `--calls-per-hour` | 20 | The calls' Poisson rate |
| `--turns` | 10 | Each call's measured turns |
| `--warmup-turns` | 1 | Unmeasured turns that open each call, since a serving worker's AEC warm-up ignores an interruption of its first reply |
| `--barge-in`, `--caller-seed` | 0.25, 0 | As `bots`' |
| `--dispatch-timeout` | 15 | How long a caller waits for the agent before the call counts as refused |
| `--max-open` | 8 | Attempts open at once; one due beyond them is skipped |
| `--drain-s` | 600 | How long the calls in progress at the window's end may play on |
| `--agents-sessions`, `--agents-data`, `--agents-seed`, `--think-cap`, `--agents-warmup` | 16, `weka-x32`, 11, 120, 60 | AIPerf's replay as E2's `[agents]` table sets it; 0 sessions is voice alone |
| `--agents-run-hours` | 0 | Start AIPerf again after this many hours of a run; 0 is one run for the window |
| `--model` | the serving profile's | The model AIPerf asks for |
| `--scrape-s`, `--guard-s`, `--guard-clear-s` | 60, 60, 300 | The scrapes' and the guard's intervals, and the guard's hold-off before a paused load resumes |
| `--guard NODE=METHOD` | `ssh` (`local` on a single host) | How a node's GPU and units are read through `deploy/bin/node-report`: `ssh`, `ssh:TARGET`, `local` or `off` |
| `--gpu-source NODE=REVISION` | none | A GPU node's deployed revision, which the record keeps |

**The soak directory** holds `soak.json`, with the options, the versions, how the box served, each tool's runs, the
pauses and how the soak ended. Each run of the callers keeps its `attempts.jsonl` and `turns.jsonl` under
`calls/run-NNN/`, and each run of AIPerf its exports under `agents/run-NNN/`. Every scrape (`series.jsonl`,
`scrapes/HOUR.gz`), unit journal line (`units.jsonl`) and guard check (`guard.jsonl`) is kept, beside `soak.log`,
`events.jsonl` and the summary per hour and overall. An attempt belongs to the hour it was due in, a turn to the hour it
ended in.

**The compact record** (`soak export`) keeps `soak.json`, the JSON lines, `turns.csv`, `series.csv`, `guard.csv` and the
summary, with home paths written from `~` and the login as `USER`. It holds no raw scrape, AIPerf export, request log or
credential, and `soak report` gives the same summary from it as from the directory.

## Mock server

`flatshare-bench mock-server --port P --log FILE` binds 127.0.0.1 only and streams chat completions. Each request
appends a JSON line to the log with its times, every header, every body field but `messages`, and a digest of the
messages. It also records `cached_tokens`, the longest prefix shared with one of the session's recent prompts, which
`usage` returns.

## Tests

The tests are offline and take four to five minutes, and CI runs them with `make test-python`. They cover the scaling,
both loaders through AIPerf's own reconstruction, the plugin's requests as AIPerf's transport sends them, the wire names
against `flatshare-proto`, and the probes against in-process fakes. The larger suites, whose module docstrings list what
each holds, test E2's points, report and records, `campaign.sh`, the soak and the demo video. The deployment's scripts
and the A100 run's kit have no package of their own, so their suites live here too, among them `test_deploy.py`,
`test_release.py` and `test_rehearse.py`.

```sh
uv run --extra test pytest      # in python/flatshare_bench
```
