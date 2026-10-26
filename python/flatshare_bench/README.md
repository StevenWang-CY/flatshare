# flatshare-bench

Flatshare's benchmark toolchain, for a reader who runs or changes a measurement: block-scaled agent
traces, AIPerf plugins, latency probes, E2's harness and report, and the failure drills' client. It
replays the SemiAnalysis Claude Code traces (`semianalysisai/cc-traces-weka-062126`: 393 sessions,
98,827 requests, subagents, think times) against a 16-32k context, with Flatshare's per-session
metadata on every request, and measures the tier cost model (E0), server-side voice turns and E2's
points.

| Piece | Module | Purpose |
|---|---|---|
| `flatshare-bench scale-traces` | `traces.py` | Shrinks every request of the corpus by 32x or 64x while keeping its prefix structure |
| AIPerf loader `weka_trace_scaled` | `aiperf_plugin.py` | AIPerf's Weka loader, fixed to read each trace's own block size |
| AIPerf endpoint `flatshare_chat` | `aiperf_plugin.py` | Chat completions plus `X-Session-ID`, `X-Flatshare-Class`, `prompt_cache_key` and `priority` |
| `configs/agentic_replay.yaml`, `configs/aiperf.env` | | AIPerf `agentic_replay` run: steady-state session trees with recycling |
| `flatshare-bench mock-server` | `mock_server.py` | Recording OpenAI-compatible mock on 127.0.0.1 for CPU-only dry runs |
| `flatshare-bench tiers` | `tiers.py` | E0: time to first token from the GPU pool, the CPU tier and recompute, verified per sample by vLLM's prefix-cache counters; optional background load |
| `flatshare-bench voice-turn` | `voice_turn.py` | One server-side voice turn timed per stage (STT, LLM first sentence, TTS first audio); optional background load |
| `flatshare-bench kv-metrics` | `kv_metrics.py` | KV block lifecycle histograms compared across a run |
| `flatshare-bench e2` | `e2/` (`arm.py`, `run.py`, `record.py`, `report.py`, `figures.py`) | E2 points: one measured from its arm file on cyan (`run`), the frontier across points (`report`), and their compact records (`export`) |
| `flatshare-bench drill` | `drills.py` | The client side of the failure drills (`ops/drills`): streams through the gateway while the engine dies or the gateway reloads, restarts or dies, the probe of both planes from the event's moment on (every attempt kept, each on a new connection), the engine's running requests sampled and stamped meanwhile, and the summary (`streams`, `report`); the CPU tier filled past its capacity (`cpu-tier`); a server's start timed from its journal (`start`) |
| background load | `load.py` | `prefill:<tokens>x<n>` (cold prefills) or `decode:<tokens>x<n>` (long decodes) beside a probe |

Python 3.12. AIPerf, pinned to `aiperf==0.13.0`, is the `aiperf` extra, which the `test` extra
includes: `flatshare-bench e2 run` and the plugins need it, a report does not, so `make reproduce-e2` syncs
28 packages, not the 140 of an environment with AIPerf. Installed with it, the package registers
both AIPerf plugins through the `aiperf.plugins` entry point; `aiperf plugins endpoint` lists
`flatshare_chat`. On cyan, `deploy/cyan/bootstrap.sh` builds `~/flatshare/venvs/bench` from
`uv.lock`, with the test extra; for development:

```bash
uv sync --python 3.12 --extra test      # in python/flatshare_bench: its own .venv, from uv.lock
```

On macOS, use Python 3.13 (`--python 3.13`): `flatshare-bench e2 run` needs `os.waitid`, which macOS has from
3.13 on. On 3.12 it refuses to start, and its tests skip with that reason.

[The quickstart](#quickstart-cyan) replays a subset of the traces through AIPerf to the mock server.

## Trace scaling

Each request in the corpus is a list of 64-token prefix hash blocks; the median main-agent
prompt is 254k tokens. AIPerf's own caps do not help: `--synthesis-max-isl 32768` keeps 2% of
main-agent requests and does not filter subagents, and `--max-context-length 32768` keeps no
trace. Instead, `scale-traces` shrinks the tokens per block and leaves the hash ids alone:

    block_size -> block_size / factor      in -> len(hash_ids) * new block_size      out -> ceil(out / factor)

Prefix identity, subagent forks and the cache hit rate are preserved exactly, and every request
shrinks by the same factor, outputs included, so a cache miss keeps its cost relative to decode
work. Because `ceil(ceil(out / f) / (64 / f)) == ceil(out / 64)`, AIPerf's split of each new turn
into assistant and user blocks is unchanged. Sessions, subagent entries and timing are copied
as recorded; there is no length cap.

| Factor | Context | Largest `in + out` | Use |
|---|---|---|---|
| 32 (`block_size` 2) | 32k | 31,144 | Default. Main-agent context p50 7.9k, p90 20.3k |
| 64 (`block_size` 1) | 16k | 15,572 | Smaller KV pools |

```bash
flatshare-bench scale-traces traces.jsonl weka-x32 --factor 32    # the HF JSONL or a directory of *.json
flatshare-bench scale-traces traces.jsonl weka-sub --ids ids.txt  # a subset; or --limit N
```

The full corpus takes about 22 s and 1.8 GB of RAM per factor on cyan; each output is 1.8 GB
(the hash ids are kept).

## Wire metadata

The AIPerf endpoint renders each request's metadata through
`flatshare_bench.protocol.SessionTagger`. Names and values mirror the Rust crate `flatshare-proto`.

| Field | Value |
|---|---|
| Header `X-Session-ID` | The session-tree root: a parent agent and all of its subagents share it. AIPerf mints one per session play (`root_correlation_id`) |
| Header `X-Flatshare-Class` | `realtime`, `interactive` or `background` |
| Body `prompt_cache_key` | The session id |
| Body `priority` | vLLM priority from `ENGINE_PRIORITY`: realtime 0, interactive 1, background 2. vLLM schedules lower values first |

Agent replay defaults to `background`. The plugin reads its configuration from the environment,
since AIPerf builds the endpoint itself, in every worker:

| Variable | Default | Meaning |
|---|---|---|
| `FLATSHARE_BENCH_CLASS` | `background` | Class of every session |
| `FLATSHARE_BENCH_SEND_PRIORITY` | `true` | `false` sends no `priority`, as a plain client would. vLLM 0.32 acts on it only under `--scheduling-policy priority`; its FCFS scheduler ignores it |

AIPerf also sends its own `X-Correlation-ID` and `X-Session-Affinity`, which identify each agent
rather than the tree. Keep `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID` off: when on, AIPerf
overwrites the plugin's `X-Session-ID` with the agent's own id. vLLM 0.32 reads `X-Session-ID`
as the request's session id and ignores `prompt_cache_key`; the control plane strips client
`priority` and sets its own.

## The `weka_trace_scaled` loader

AIPerf 0.13.0 registers `weka_trace` with `default_block_size: 64` and passes that value to the
loader as an override, so the stock loader ignores a trace file's own `block_size`. On a scaled
trace it decodes 64 tokens per block from the first 1/32 of each request's hash ids. On the 32x
corpus it reproduced about 77% of the shared prefix, with prompts 22% short at the median and
no assistant turns. `weka_trace_scaled` is the same loader class with that override dropped;
on the wire it reproduced the shared prefix to within 20 tokens. `tests/test_loader.py` runs
both loaders through AIPerf's own reconstruction. With the fix, every shared prefix and every
prompt length matches the trace exactly; without it they do not.

## AIPerf `agentic_replay`

`agentic_replay` keeps N session trees live. Each lane starts at a random point of a trace,
replays it with recorded think times and subagents, then recycles a trace with a fresh
cache-bust marker. AIPerf 0.13.0 selects it only from YAML. `configs/agentic_replay.yaml` is the
template. AIPerf substitutes `${VAR:default}` when it loads the file:

| Variable | Default |
|---|---|
| `FLATSHARE_BENCH_URL` | `http://127.0.0.1:8000/v1/chat/completions` |
| `FLATSHARE_BENCH_API_KEY` | empty: no `Authorization` header. Through `flatshare-gateway`, a tenant's key. Only a trusted tenant's `X-Session-ID` and `X-Flatshare-Class` are honored, so a replay that keeps its sessions and classes uses the `agents` key |
| `FLATSHARE_BENCH_DATA` | `data/weka-x32` (a `scale-traces` output directory) |
| `FLATSHARE_BENCH_MODEL`, `FLATSHARE_BENCH_TOKENIZER` | `Qwen/Qwen3-4B-Instruct-2507-FP8` |
| `FLATSHARE_BENCH_SESSIONS` | 16 live session trees |
| `FLATSHARE_BENCH_DURATION` | 1800 s |
| `FLATSHARE_BENCH_GRACE` | 120 s |
| `FLATSHARE_BENCH_THINK_CAP` | 600 s cap on each think time |
| `FLATSHARE_BENCH_BURST_STARTS` | `true` |
| `FLATSHARE_BENCH_ARTIFACTS` | `runs/agentic-replay` |
| `FLATSHARE_BENCH_RAW` | `false`; `true` also exports every request with its headers and payload |
| `FLATSHARE_BENCH_SEED` | 11 |

Source `configs/aiperf.env` first. It turns AIPerf's dataset cache on, keeps the session-id
override off, and raises the dataset timeouts that the full corpus (about 300 s to load and
17-22 GB of RAM) exceeds. AIPerf takes these settings only from the environment. The cache keeps
the memory-mapped dataset a run builds, under `~/.cache/aiperf/dataset_mmap`, for any later run
with the same corpus bytes, tokenizer, seed and dataset settings on the same AIPerf build, so the
points of an E2 round, which replay one corpus with one seed, build their dataset once: on
peacock the 48 long traces' dataset took 138 to 165 s to build and 5 s to load, byte for byte the
same. AIPerf 0.13.0 caches only its own trace types. The plugin adds `weka_trace_scaled`, and
keeps its datasets in a directory named by a hash of its own code and version
(`flatshare-<hash>`), which AIPerf's key does not cover, so a changed loader never loads a
dataset an older one built (`aiperf_plugin.cache_datasets`). As for AIPerf's own trace types, a
run then writes no `inputs.json`, which took 124 MB of each point's directory for the long
traces. Nothing evicts an entry, 0.14 GB for the long traces and 0.33 GB for the whole corpus:
delete the directory to reclaim them (the cache once filled a host's disk).

Things to know:

- **Burst starts.** With spread starts, each lane first waits out its recorded gap after its
  random start point, and `interTurnDelayCapSeconds` does not cap that wait. A 60 s check
  reached 2 of 8 session trees and 31 requests; with burst starts it reached 8 of 8 and 236.
  Set `FLATSHARE_BENCH_BURST_STARTS=false` to replay each lane's first arrival at its recorded gap.
- **Grace period.** AIPerf waits the whole grace period whenever session trees are still open,
  which in `agentic_replay` is nearly always.
- **vLLM flags.** Cached-token counts need `--enable-prompt-tokens-details`; the `priority` field
  takes effect only with `--scheduling-policy priority`.
- **Assistant history.** AIPerf replays assistant content synthesized from hash blocks, never the
  model's reply, so the KV the model writes while decoding turn k is never matched by turn k+1, as
  it is when a real agent resends its own output. AIPerf 0.13.0 threads live replies into history
  only in the `deltas_without_responses` context mode, and the Weka loader hardcodes
  `deltas_with_responses`.

## Latency probes (`flatshare-bench tiers`, `flatshare-bench voice-turn`)

`tiers` measures what resuming a session costs from each tier (E0). For each prompt length, a unique prompt of random token ids goes out three times, each asking for one token:
1. cold, so every token is recomputed;
2. again at once, a GPU prefix-cache hit;
3. after other prompts have pushed it out of the GPU pool, restored from the CPU tier.

vLLM's `prefix_cache_hits` and `external_prefix_cache_hits` counters, read around each request, verify where its tokens came from. A sample whose eviction or restore went wrong is flagged invalid instead of being mislabeled. The run fits GPU hit `a + b·L`, restore `a + b·L` and recompute `a + b·L + c·L²`; `flatshare-core` reads those fits.

`voice-turn` times one turn of the cascade a voice agent runs:
- STT on the utterance;
- the LLM's first sentence, counted once it reaches 20 characters as in LiveKit's tokenizer;
- TTS first audio for that sentence, sent while the reply is still streaming; the rest of its audio is read as the
  worker plays it, so the next turn reuses the connection (through the tunnel a new one is an SSH session).

Both take `--background prefill:8192x1` or `decode:1024x8` to measure under agent load. They write `run.json`, the samples and `summary.json` to `--out`; each summary cell has the count, the mean and p10, p50, p90, p95 (where the SLOs are set) and p99. `tiers` exits 3 if any sample was invalid.

Every command that reaches a server takes its address as the gateway's `upstream` is written: `http://host:port`, or `unix:` and the absolute path of its socket (`transport.py`). On cyan the defaults are the GPU node's servers through its tunnel's sockets, `unix:$XDG_RUNTIME_DIR/flatshare/phastform-engine.sock`, `-stt.sock` and `-tts.sock`; a request's path starts at the server's root (`/v1/...`, `/metrics`).

```bash
flatshare-bench tiers --reps 5 --out runs/e0-idle
flatshare-bench tiers --lengths 1024,4096,16384 --background decode:1024x8 --out runs/e0-decode
flatshare-bench voice-turn --audio turn.wav --reps 30 --background prefill:8192x1 --out runs/voice-x1
```

## KV block lifecycle (`flatshare-bench kv-metrics`)

Snapshot an engine's `/metrics` around a run, then compare vLLM's sampled KV block histograms over it. The engine needs `--kv-cache-metrics`. The histograms are `vllm:kv_block_reuse_gap_seconds`, `vllm:kv_block_idle_before_evict_seconds` and `vllm:kv_block_lifetime_seconds`.

```bash
engine=unix:$XDG_RUNTIME_DIR/flatshare/phastform-engine.sock  # on cyan: the engine through its tunnel
flatshare-bench kv-metrics snapshot $engine before.prom
# ... run the load ...
flatshare-bench kv-metrics snapshot $engine after.prom
flatshare-bench kv-metrics compare before.prom after.prom
```

On the frozen box, 10 minutes of the replay above with 16 sessions (2026-10-22):

```text
vllm:kv_block_reuse_gap_seconds: n=1763 p10=0.495s p50=2.54s p90=6.33s
vllm:kv_block_idle_before_evict_seconds: n=4930 p10=1.32s p50=6.22s p90=27.3s
vllm:kv_block_lifetime_seconds: n=4930 p10=1.52s p50=7.91s p90=46.3s
evicted blocks idle no longer than the reuse-gap p50 (2.54s): 23.8%
evicted blocks idle no longer than the reuse-gap p90 (6.33s): 50.4%
```

When evicted blocks were idle for less time than blocks typically wait between reuses, LRU is evicting KV that would have been reused. Reuse gaps are recorded only for blocks that survived until reuse, so the comparison is a signal, not a bound.

## E2 points

`flatshare-bench e2 run ARM.toml --out DIR` measures one point of E2 on cyan: the measurement calls beside the AIPerf agentic replay, through the gateway, on the engine the GPU node runs. [`benchmarks/e2`](../../benchmarks/e2/README.md) holds the arm files and `campaign.sh`, which measures a list of them from a workstation and switches the GPU node's engine between them.

```bash
flatshare-bench e2 run ~/flatshare/src/benchmarks/e2/arms/a1-voice.toml --out ~/flatshare/runs/e2/w2/a1-voice
flatshare-bench e2 run ~/flatshare/src/benchmarks/e2/arms/disc-a9-s24.toml --out ~/flatshare/runs/e2/demo/disc-a9-s24 \
  --save-audio                               # the callers save each call's audio, for the demo video
flatshare-bench e2 engine a2-stock-s16.toml  # the engine an arm needs, as variant.sh use takes it
flatshare-bench e2 report ~/flatshare/runs/e2/w2/{a1-voice,a2-stock-s16} --out ~/flatshare/runs/e2/first
flatshare-bench e2 report ~/flatshare/runs/e2/{w2c,a9,r1,r2}/disc-* --out ~/flatshare/runs/e2/disc \
  --waterfall disc-frozen-s24 --waterfall disc-a9-s24   # repeats across rounds; two configurations' waterfall
flatshare-bench e2 export ~/flatshare/runs/e2/w2b/* --out benchmarks/results/2026-10-23-e2   # compact records
flatshare-bench e2 reset                     # cyan's units as gateway.toml and the worker's defaults have them
```

A point runs in this order:
1. **The engine.** Its `/server_info` (on vLLM it needs the development mode of the measurement drop-in, `deploy/phastform/variant.sh measure on`, which a campaign runs) gives the server, its version, the KV pool and its settings. They must be those of the engine's command line as `variant.sh use` makes it, or the point fails before anything starts and names the `variant.sh use` command that fixes it: on vLLM what the GPU node's `engine.yaml` sets by its hardware profile ([`deploy/profiles`](../../deploy/profiles); the frozen box's unless `--profile NAME` or the host's settings name another): the policy, the step cap and the long-prefill threshold, the model, its revision and the context; then the flags of the command the variant's `flatshare-engine.conf` starts, then the arm's `[engine]` flags, the last of a repeated flag winning. vLLM's settings are its cache, scheduler and kernel configuration's (`--linear-backend` among them) and its model's, by its Hugging Face id where offline mode reports its path in the cache. On vLLM the scheduler class (`scheduler_cls`) must be the one `--scheduler-cls` names, vLLM's own without it, and `additional_config` exactly what the `--additional-config.KEY` flags build (make-room's switches), so that a frozen, make-room or pr54530 engine, or make-room with another switch, cannot pass for another. A flag whose setting the engine does not report is recorded as unchecked.
2. **cyan's units.** SMG runs only for `router = "smg"`. With a `[gateway]` table the gateway runs on `~/.config/flatshare/gateway-e2.toml` (`gateway.toml` with the table merged in, mode 600) through the `e2-gateway` drop-in, and without one on `gateway.toml`. The voice worker's preemptive generation is switched with the `no-preemptive-generation` drop-in, and its lifecycle events with `no-events`. Only a unit whose configuration changes restarts, and the worker must register again within 90 s. The worker must end turns with fixed endpointing (`vad-turns`), or nothing runs.
3. **A cold start.** The engine's prefix cache, and vLLM's CPU tier, are emptied.
4. **The replay.** With agent sessions, AIPerf starts on cores 16–23, asking for the engine's model (the profile's, or the arm's `--model`), and the calls wait for its profiling phase plus the warm-up.
5. **The window.** The gateway's and the engine's `/metrics` are saved, `flatshare-voice bots` runs in the voice-plane namespace on cores 24–31, and they are saved again. The calls are the window every metric covers; AIPerf must outlast them. Through them an agent point samples the engine's `/metrics` every 0.5 s for its KV use and its running and waiting requests (`metrics/engine-kv.csv`), on one connection, kept alive for the window and opened again only after a failed request (through cyan's tunnel a new connection is a new SSH session on the GPU node); `--kv-sample-s SECONDS` sets another interval and 0 samples nothing, as W5's points ran (the sampler came after them). With `--save-audio`, a run option rather than the arm's, the callers also save each call's audio for the demo video (`flatshare-voice bots --save-audio`), which measures nothing differently: they write it once their calls have ended, and the window and the second `/metrics` end with the calls, at the callers' summary, not with their exit.
6. **The end.** AIPerf gets SIGINT, cancels its requests and writes its exports. The gateway's request lines for the window go to `gateway.jsonl`, and the voice worker's journal from the start of the calls to `voice-worker.log`, once every call's session has logged its end (at most 30 s). Then `summary.json` is written.

**The host's settings.** cyan's addresses, cores and the GPU node's profile are `flatshare-bench`'s defaults. Another control host, a single host running every part, sets them in `~/.config/flatshare/e2-host.toml`, strings keyed by `Host`'s fields in `e2/run.py`: `gateway_url`, `gateway_socket`, `engine_url` (each an address: `http://host:port`, or `unix:` and a socket's path), `aiperf_cores`, `caller_cores`, `profile` and `preflight`, whose not-ok checks each point's `run.json` keeps (`deploy/a100/host.py config` writes them from the profile's core plan, [`docs/HARDWARE.md`](../../docs/HARDWARE.md#single-host-mode)); any other key is an error.

**The arm file.** Unknown keys and wrong types are errors. The point is named after the file.

| Key | Default | Meaning |
|---|---|---|
| `arm` | required | E2's arm, which names the point's frontier: 1 to 10, then a letter for a sub-arm (`"7a"`) or a suffix for a factor (`"4-no-preemptive"`) |
| `label` | required | What the point is, for the report |
| `router` | none | `"smg"` starts `flatshare-smg` (arm 8b); the `[gateway]` table points the gateway at it |
| `[voice] callers`, `turns` | 2, 25 | Concurrent calls, and turns per call |
| `[voice] barge_in` | 0.25 | Share of turns on which the caller interrupts the reply |
| `[voice] seed` | 0 | The callers' seed |
| `[voice] warmup_turns` | 0 | Turns that open each call before its `turns`, played but not measured (`flatshare-voice bots --warmup-turns`): marked `"warmup": true` in `bots/turns.jsonl`, they are in no V2V sample, and the worker's lines for them, each session's first, are in no stage |
| `[voice] preemptive_generation` | `true` | LiveKit's preemptive generation in the voice worker (`FLATSHARE_PREEMPTIVE_GENERATION`) |
| `[voice] events` | `true` | The voice worker's lifecycle events to the gateway (`FLATSHARE_EVENTS`); `false` measures a point without them |
| `[agents] sessions` | 0 | Session trees AIPerf keeps live, through the gateway as the `agents` tenant with `FLATSHARE_GATEWAY_AGENTS_KEY` from `~/.config/flatshare/gateway-keys.env`; 0 is voice alone |
| `[agents] think_cap` | 120 | Cap on each recorded think time, in seconds |
| `[agents] warmup` | 60 | Seconds of profiling before the calls start |
| `[agents] data` | `"weka-x32"` | The corpus, a `scale-traces` directory under `~/flatshare/data` |
| `[agents] seed` | 11 | AIPerf's seed, which a campaign's `agents-seed` line replaces for its points (`flatshare-bench e2 run --agents-seed`) |
| `[gateway]` | none | Merged into `gateway.toml` as it is, tables key by key: `upstream`, `upstream_priority`, `[gateway.admission]` and any other key of [`flatshare-gateway`'s configuration](../../crates/flatshare-gateway/README.md). `[gateway.nodes]` and `[gateway.placement]`, for points on several GPU nodes, are refused: those points are on the `two-node` branch |
| `[engine] variant` | `"frozen"` | A directory of [`deploy/phastform/variants`](../../deploy/phastform/variants/README.md), or `frozen` |
| `[engine] flags` | `[]` | Engine flags, each a plain word, added to its command line |

**The run directory.**

| File | Contents |
|---|---|
| `arm.toml` | The arm file as run |
| `run.json` | `status` (`running`, then `complete`, or `failed` with the `error`); the host; start and end; the arm as read; `profile`, the hardware profile the engine check held the engine to; `save_audio`, the run option; `agents_seed`, the seed AIPerf replayed with (the campaign's `--agents-seed`, else the arm file's; null for voice alone); `kv_sample_s`, the KV sampler's interval (null for voice alone or `--kv-sample-s 0`); versions (cyan's deployed commit from `~/flatshare/src/DEPLOYED`, the GPU node's beside it as `gpu_source` (by node, from `--gpu-source NODE=REVISION`, which `campaign.sh` reads from the node's own `DEPLOYED` as the point starts; null when nothing named it), `flatshare-bench`, AIPerf, `flatshare-gateway`); the engine (server, version, KV pool, flags checked and unchecked); what cyan ran (which units restarted, the gateway's configuration without key digests); the window; AIPerf's exit status |
| `run.log` | Each step, timestamped |
| `engine.json` | The engine's whole configuration from `/server_info`, without credentials |
| `aiperf.log`, `aiperf/` | AIPerf's output and artifacts: `profile_export.jsonl` holds one record per request |
| `bots.log`, `bots/` | The callers' output and run directory (`turns.jsonl`, `summary.json`, and with `--save-audio` each call's `call-<i>.wav` and `call-<i>.json`) |
| `metrics/` | `gateway-start.prom`, `gateway-end.prom`, `engine-start.prom`, `engine-end.prom`. An agent point's `engine-kv.csv`: a line per sample of the engine's `/metrics` every 0.5 s through the calls, its time, `vllm:kv_cache_usage_perc`, and the running and waiting requests |
| `gateway.jsonl` | The gateway's request lines for the window (`journalctl --user -u flatshare-gateway --since --until`, the JSON lines whose message is `request`): class, status, outcome, `first_byte_ms`, `total_ms`, `admission_wait_ms` and tokens. The first bytes come from them: a gateway that logs text (`log_format` other than `"json"`) logs none, and the point fails when it is summarized |
| `voice-worker.log` | The voice worker's journal from the start of the calls: a line per turn with its stage times, a line per session at its end with its lifecycle-event counts ([`flatshare-voice`](../flatshare_voice/README.md#lifecycle-events-and-turn-lines)) |
| `gpu.csv` | Written by `campaign.sh`, not by `e2 run`: the GPU node's `nvidia-smi` samples, once a second while the point ran (timestamps in UTC, SM and memory clocks, temperature, power, clock event reasons and, since 2026-10-25, the GPU's name) |
| `summary.json` | The point's E2 metrics |

A component that fails fails the point: `run.json` keeps `failed` and the error, and a tool's error gives its exit status and the last lines of its log. AIPerf and the callers run in process groups of their own and are stopped through them, with SIGINT and then SIGKILL for whatever is left, also when the point fails or `flatshare-bench e2 run` gets SIGINT or SIGTERM. Closing the SSH session that started it does not stop a point. A complete point is never overwritten; an incomplete one is moved aside to `DIR.incomplete-<UTC time>`, and a directory that holds anything else is refused. One `e2 run` or `e2 reset` runs at a time; the lock file, `~/flatshare/runs/.flatshare-bench-e2.lock`, holds its pid. Every process a tool starts carries the run's mark in its environment, under its root (`~/flatshare`), and a point's start kills what a run under the same root left, never the tools of a run under another, such as a test session's beside another's.

**`summary.json`.** Every metric covers the calls. A metric the gateway or the engine does not export is null, never zero, and a counter that went down (its process restarted during the calls) fails the point.

| Key | Contents |
|---|---|
| `window_s` | Length of the calls |
| `voice` | From the callers: turns, errors, `v2v_ms` and `barge_in_to_silence_ms` (n, p50, p95, p99) and barge-ins |
| `worker` | Null without the worker's lines. From `voice-worker.log`: `sessions` that logged their end, `turns` (without each session's first `[voice] warmup_turns` turn lines, its replies to the warm-up turns), the `waterfall` (each stage of the turn lines: `transcript_ms`, `end_of_turn_ms`, `llm_ttft_ms`, `llm_first_sentence_ms`, `turn_to_first_sentence_ms`, `tts_first_audio_ms`, `e2e_ms`, with n, mean, p10, p50, p90, p95, p99) and the lifecycle `events` its sessions sent, dropped and failed to deliver (null when it sent none: `events = false`, or the LLM bypassed the gateway) |
| `background` | Null for voice alone. From AIPerf's records of the agent requests that ended during the calls: requests, errors (failed and cancelled requests count as lost), requests/s, prompt, completion and total tokens/s from the server's usage counts, `ttft_ms`, and `itl_ms`, AIPerf's inter-token latency per request (the time after its first token over the tokens after it, for requests of two tokens or more), the step time a decoding agent sees; and the same goodput over the calls' first 240 s (`GOODPUT_WINDOW_S`), the same span for every point, where the calls' own length differs by configuration (`fixed_window`: its `seconds`, fewer for calls that lasted less, `requests`, `requests_per_s` and `tokens_per_s`) |
| `gateway` | `realtime_first_byte_ms` and `background_first_byte_ms` (n, mean, p50, p95, p99): exact, from `gateway.jsonl` (`"source": "log"`); what its admission point shed, refused and resubmitted, and the prompt tokens of the attempts it dropped; `background_tokens_per_s`, its own count from response usage; and every counter that moved |
| `engine` | vLLM's preemptions; its prompt tokens by source at their first scheduling (local compute, GPU-pool and CPU-tier hits; vLLM 0.32 counts none at a preempted request's resume); queue-limit rejections; and the CPU tier's own loads into the GPU pool and stores from it, every one of them, in bytes (`kv_offload_load_bytes`, `kv_offload_store_bytes`, from `vllm:kv_offload_load_bytes_total` and `vllm:kv_offload_store_bytes_total`) |
| `kv` | Null for voice alone. From `metrics/engine-kv.csv` and the pool in `engine.json`: the `samples` in the calls, the `running_share`, the share of them in which the KV that running requests hold left less than two calls' KV free (2,048 tokens, flatshare-sim's realtime tier: the measured counterpart of its `realtime_blocked_share`; vLLM counts a cached block whose requests are gone as free), `free_blocks` (p50, min), and `running` and `waiting` (p50, max) |
| `gpu` | Only in a compact record (`flatshare-bench e2 export`): from `gpu.csv`, the GPU's median SM clock (`sm_clock_p50_mhz`) and highest temperature (`temperature_max_c`) during the calls, and its `name` where the samples carry it |

The background goodput comes from AIPerf's records rather than from the gateway's request outcomes, whose `completed` counts a response relayed in full whatever its status: it cannot tell a request the engine refused from one it served. The gateway's token rate stays beside it as a check. A node's share, which AIPerf cannot tell apart, counts the gateway's request lines that completed with status 200.

**The report.** `flatshare-bench e2 report RUN_DIR... --out DIR` writes `frontier.csv`, `frontier.md`, a frontier figure per workload and `waterfall.svg`, SVG drawn in the docs' figure style by `figure_style.py`, which embeds a subset of the STIX Two fonts in `fonts/` (fontTools, pinned to 4.67.0, makes the subsets, so the figures come out as the committed bytes). A point's workload is the agents' corpus (`[agents] data`) and the number of callers (`frontier.csv`'s `workload`, such as `weka-x32, 2 callers`), with the GPU and the model where they are not the frozen box's (`weka-x32-long, 2 callers, NVIDIA A100-SXM4-80GB`, as the A100 run's records name it; a record whose samples name no GPU ran on the box) and the warm-up turns that open its calls where the report's points differ in them (`weka-x32, 2 callers, 1 warm-up turn`; a record set has one count, W4's none and W5's one), and points of different workloads are never compared: a point is on its arm's frontier when no other point of the arm and workload harvests at least as much background goodput at a V2V p95 at least as low, and one of the two strictly better. `frontier.md` has a section per workload, which opens with its voice SLO and the name of its figure, with a table per arm in it, ordered by background goodput, where • marks the arm's frontier when the arm has more than one configuration (a lone configuration is its own). Run directories without a complete point are named and left out.
- **The frontier figures.** `frontier.svg` draws the arm file's own workload (`weka-x32`, 2 callers); another workload's figure is named after its corpus and its callers where they differ (`frontier-weka-x32-long.svg`, `frontier-4-callers.svg`), after its GPU and model off the box (`frontier-weka-x32-long-nvidia-a100-sxm4-80gb.svg`), and after its warm-up turns where its name gives them (`frontier-1-warm-up-turn.svg`). A figure gives each configuration a row under its family's heading, in E2's order (stock vLLM, arms 1–5; operator answers, 6; engines, 7; routers, 8; make-room, 9; MPS, 10), named by its label without the clauses its workload gives, and two bars from zero: its voice-to-voice p95, against the voice SLO, and the agents' goodput over the calls, each the median of its runs. The SLO is a dashed rule 250 ms above voice alone (arm 1) with as many callers on the same GPU and model and as many warm-up turns, from whichever corpus, since voice alone replays none; a workload without that point says so. The engine change's configurations are blue, the one to deploy (`9-pr-ab48`) saturated and the others pale, and every other configuration gray; an open bar is a configuration off its arm's frontier. A configuration past its workload's fence for far outliers (three interquartile ranges above the upper quartile of the V2V p95s) has its bar cut square at the scale's end, its value printed, so that one collapse does not flatten the rest.
- **Configurations of their own.** Arm 9's ablations and the PRs' configurations say `arm = 9` in their arm files, as run, but each is a configuration of its own: the report gives each its own arm (`9-no-headroom`, `9-pr-ab48`; `CONFIGURATION_ARMS` in `e2/report.py`), so that none is placed on arm 9's frontier or joined to it. Where their labels name the PRs by letter (PR A, PRs A and B), the report names them by number, #63350 and #63351, as the docs do (`PR_LABELS`). The records stay as run.
- **Repeats.** Points measured from the same arm file in several campaigns, under the same point name, are repeats of one configuration. `frontier.md` keeps a row per run, named by its campaign (`frontier.csv`'s `campaign`: the directory that holds the run directory), then gives the configuration's median and range (min–max) of V2V p50/p95/p99, background tokens/s (over the whole calls and over their first 240 s), the agents' TTFT p50 and p99, the realtime first byte p95 and failed turns, with its run count. The frontier places the configuration at its medians, and its figure draws its medians, its run count under its name. A configuration measured once is shown as before.
- **The goodput over a fixed window.** Beside each point's background goodput over its whole calls, whose length differs by configuration, `frontier.md` gives its goodput over their first 240 s (`frontier.csv`'s `fixed_window_tokens_per_s`, with its `fixed_window_s`), and for a repeated configuration their median and range; † marks one over a point's whole calls, which lasted less. The frontier stays placed by the goodput over the whole calls.
- **Failed turns.** V2V comes from the callers' measured turns in `bots/turns.jsonl` (not the warm-up turns that open each call under `[voice] warmup_turns`), where a turn the agent did not answer within the callers' reply timeout counts as a sample censored at 30,000 ms, in every percentile: `frontier.md` marks a percentile such a sample sets with `≥`. A configuration with failed turns is not placed on the frontier and keeps no other point off it; its figure cuts its bar at the scale's end with its censored value (`≥30.00`) and names its failed turns under its name, and the stage waterfall leaves its runs out.
- **The realtime first byte** is exact from the run directory's `gateway.jsonl`, even one fetched after the run (from the journal, for the window in `run.json`), and otherwise the summary's, which took it from the same log; `realtime_first_byte_source` says so (`log`, in every record).
- **The stage waterfall** (`waterfall.svg`, and its table in `frontier.md`) splits each configuration's voice turn into stages that follow one another, as means over the replies of its runs: endpointing (`end_of_turn_ms`), the reply to its first sentence (`turn_to_first_sentence_ms`) and TTS to first audio (`tts_first_audio_ms`), from the worker's turn lines, then the rest of the callers' V2V, transport and playout (the mean V2V less the three). So a bar is as long as the mean V2V, which it prints at its end; the reply to its first sentence, the LLM's stage, is its one hue among grays, and each run of rows from one workload sits under the workload's name. The table beside it gives the median and p95 of the callers' samples in `bots/turns.jsonl`, and the LLM's own time to first token (`llm_ttft_ms`), from its request: under preemptive generation the request starts at the final transcript, before the end of the turn, so it lies across the first two stages. `--waterfall POINT` (repeatable) draws only those configurations, in that order; by default every one with turn lines.
- **Columns only some points have** appear when one of the points has them: the worker's stage times (p50 and p95 of each, in a table of their own in `frontier.md`, which counts the points without the worker's turn lines rather than listing them) and event counts; from `gpu.csv` (or a compact record's summary), the GPU's median SM clock and highest temperature during the calls; and under the gateway's lifecycle policy (`policy = "lifecycle"`; `leaseholder` in the records from before the project's rename), from its counters, what the policy did: the background requests it deferred by what held them back when they arrived (`deferred_headroom`, `deferred_critical`) and the realtime replies it capped (`degraded_replies` of `realtime_replies`, `degraded_share`), in a table of their own in `frontier.md`. A mechanism that deferred or capped nothing in a point did not act there. On vLLM, the engine's prompt tokens by source at their first scheduling (`first_schedule_computed_tokens`, `first_schedule_gpu_hit_tokens`, `first_schedule_cpu_hit_tokens`) and the CPU tier's loads and stores of every scheduling, resumes included (`cpu_tier_loaded_tokens`, `cpu_tier_stored_tokens`), in tokens at the bytes of KV a token takes in the engine's pool, its KV cache's bytes over its tokens from the point's `engine.json` (`kv_bytes_per_token`, 147,492 on the frozen box against the model's exact 147,456), in a table of their own in `frontier.md` with what the CPU tier loaded beyond first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls.

**The compact record.** `flatshare-bench e2 export RUN_DIR... --out DIR` writes each complete point's record into `DIR/CAMPAIGN/POINT`, where its run directory sits in its campaign: `run.json` (with `versions.gpu_source` null where the run did not record it), `arm.toml`, the callers' `summary.json` and `turns.jsonl`, a `summary.json` that keeps what the report takes from the files the record leaves out (the first bytes of both classes exactly from `gateway.jsonl`, `"source": "log"`, from `gpu.csv` the GPU's clock and temperature, `gpu`, from AIPerf's records the goodput over the calls' first 240 s, and from the engine's metric snapshots the CPU tier's loads and stores, both of which a point measured before its summary kept them gains this way), and an `engine.json` reduced to what says which engine the point ran on. For vLLM that is the model by its Hugging Face id and revision, `max_model_len`, dtype and quantization; from `cache_config` the block size, prefix caching, the KV dtype and the pool (`kv_cache_memory_bytes`, `num_gpu_blocks`); from `scheduler_config` the policy, `max_num_batched_tokens`, `long_prefill_token_threshold`, `max_num_seqs`, async scheduling, `watermark`, `max_num_queued_tokens` and the scheduler class (`scheduler_cls`, null for vLLM's own), the settings `e2 run` checks the arms' flags against; the CPU tier's connector with its `kv_connector_extra_config`; and `additional_config`. For SGLang it is the same of its server arguments (`SGLANG_ENGINE` in `e2/record.py`). Any path under a home directory is written from `~`. A record is about 15 kB and holds no credential: the callers' `run.json`, which names LiveKit's API key, stays out. The report gives the same tables and figures from the records as from the run directories, byte for byte (its figures carry no date), so the records are what goes into the repository, under [`benchmarks/results/2026-10-23-e2`](../../benchmarks/results/2026-10-23-e2/README.md), and `make reproduce-e2` regenerates the report from them with this venv. The raw run directories (the gateway's request log, AIPerf's exports, the metric snapshots, the worker's journal, the GPU's samples, the tools' logs) are an asset of the [v0.3 release](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3), `leaseholder-e2-runs-2026-10-24.tar.gz` (named before the project's rename).

## Mock server

`flatshare-bench mock-server --port P --log FILE` binds 127.0.0.1 only and streams chat completions.
Each request appends one JSON line to the log: arrival and end times, every header, every body
field except `messages`, a role/size digest of the messages, and `cached_tokens` (the longest
prefix shared with one of the session's recent prompts, also returned in `usage`).

## Quickstart (cyan)

```bash
# Environment: the venv that deploy/cyan/bootstrap.sh builds
export PATH=$HOME/flatshare/venvs/bench/bin:$PATH

# Data
cd ~/flatshare/data
curl -sSL --fail -o cc-traces-weka-062126/traces.jsonl --create-dirs \
  https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl
flatshare-bench scale-traces cc-traces-weka-062126/traces.jsonl weka-x32 --factor 32
flatshare-bench scale-traces cc-traces-weka-062126/traces.jsonl weka-x64 --factor 64
# A quick-check subset; sub16.ids lists 16 short traces, one id per line (8 with subagents)
flatshare-bench scale-traces cc-traces-weka-062126/traces.jsonl weka-x32-sub16 --factor 32 --ids sub16.ids

# Mock on loopback (keep clear of cyan's deployed ports, the gateway's 18080 and LiveKit's 17880-17882)
mkdir -p ~/flatshare/runs/w0-e2e && cd ~/flatshare/runs/w0-e2e
setsid nohup flatshare-bench mock-server --port 19180 --log $PWD/mock-aiperf.jsonl < /dev/null > mock.out 2>&1 &
echo $! > mock.pid
ss -lntp | grep 19180          # 127.0.0.1:19180 only

# AIPerf agentic_replay, 60 s
cd ~/flatshare/src/python/flatshare_bench
. configs/aiperf.env
FLATSHARE_BENCH_URL=http://127.0.0.1:19180/v1/chat/completions \
FLATSHARE_BENCH_DATA=$HOME/flatshare/data/weka-x32-sub16 FLATSHARE_BENCH_SESSIONS=8 FLATSHARE_BENCH_DURATION=60 \
FLATSHARE_BENCH_GRACE=15 FLATSHARE_BENCH_THINK_CAP=3 FLATSHARE_BENCH_RAW=true \
FLATSHARE_BENCH_ARTIFACTS=$HOME/flatshare/runs/w0-e2e/aiperf \
  aiperf profile --config configs/agentic_replay.yaml

# What reached the server
python -c 'import json,sys; r=json.loads(open(sys.argv[1]).readlines()[-1]); print(r["headers"], r["body"])' \
  ~/flatshare/runs/w0-e2e/mock-aiperf.jsonl
kill $(cat ~/flatshare/runs/w0-e2e/mock.pid)
```

Results on 2026-10-22, against the mock:

| Run | Requests | Every request carries the four fields | `X-Session-ID` is the tree root |
|---|---|---|---|
| AIPerf, 8 session trees, 60 s | 7 warmup + 236, 0 errors | 243/243 | 243/243 (103 subagent requests) |

## Tests

The tests are offline and take a few seconds. CI runs them with `make test-python`:

```bash
uv run --extra test pytest      # in python/flatshare_bench
```

They cover the scaling math and schema, the loader's prefix reproduction (both loaders, through
AIPerf's own reconstruction with a tokenizer-free prompt generator), the plugin's headers and
body as AIPerf's transport sends them, the wire names and class-to-priority mapping against
`flatshare-proto`, the YAML template and the probes against in-process fake servers.
`tests/test_dataset_cache.py` builds an E2 point's dataset with AIPerf's own dataset manager on
four committed traces: a hit replays a miss's dataset and an uncached build's byte for byte, and
a changed plugin misses. `tests/test_e2.py`
runs whole E2 points against a fake gateway and engine on
loopback, a recording `systemctl` and stand-ins for AIPerf and the callers that run as real
processes (`tests/fakes/`): the sequence, the units each arm restarts, the engine check, failures
at each step with every process stopped, the point directory's rules, the committed arm files, and
the logs a point keeps: the gateway's request lines and the worker's journal, waiting for its
sessions to end only when they will. `tests/test_e2_results.py` covers the summary over the window,
the exact first bytes from a request log, the Pareto marking within each arm and workload, each
workload's section, voice SLO and figure (its families, the arms' names and what it leaves off its
scale), repeated points placed at their medians with every run kept, the stage waterfall's means
and V2V marks, the report's columns that only some points have, compact records that give the
report of their run directories byte for byte, the committed records (complete, without
credentials, exported again to themselves, and giving the report committed beside them, every file byte for byte), the names of the lines the voice
worker writes, and the series the record and the report read from the gateway: each in its checked-in exposition
(`crates/flatshare-gateway/metrics.txt`), and each count the sum of the series it is made of. `tests/test_campaign.py` runs `benchmarks/e2/campaign.sh` against stand-ins for the hosts (an ssh that runs each command in a directory per host, rsync, cyan's `flatshare-bench` and the GPU node's `variant.sh`): what each point is measured with. `tests/test_demo.py` runs the demo
video's renderer on a synthetic pair of calls ([`benchmarks/e2`](../../benchmarks/e2/README.md#the-demo-video));
its encoding test is marked `integration` and left out of CI.
