# The agentic-api HTTP baseline (2026-10-22)

agentic-api's own `response_provider` benchmark (W1), run against the frozen box as the HTTP baseline for
[vllm-project/agentic-api#367](https://github.com/vllm-project/agentic-api/issues/367): agentic-api over WebSocket,
HTTP/SSE and JSON, and direct vLLM over HTTP/SSE and JSON. All five arms start at one barrier and run at
once, so each row of a table saw the same engine load. The benchmark sweep for #367 was opened as
[vllm-project/agentic-api#478](https://github.com/vllm-project/agentic-api/pull/478) on 2026-10-23 and merged on
2026-10-26.

The transport workload, 4 tool rounds a turn and 10 turns a session: each turn's latency, p50 / p95 in ms.

| Sessions per arm | agentic-api WS | agentic-api HTTP/SSE | agentic-api JSON | vLLM HTTP/SSE | vLLM JSON | Paired median, agentic-api minus vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,173 / 4,194 | 3,173 / 4,191 | 3,175 / 4,194 | 3,176 / 4,183 | 3,187 / 4,272 | 0 ms |
| 5 | 4,488 / 7,925 | 4,489 / 7,686 | 4,490 / 7,837 | 4,482 / 7,890 | 4,500 / 7,772 | +3 ms |
| 10 | 6,099 / 10,949 | 6,102 / 10,953 | 6,094 / 10,977 | 6,072 / 10,945 | 6,097 / 10,975 | +1 ms |

- **Task correctness was 100%, 98% and 99% on every arm** at 1, 5 and 10 sessions, except vLLM JSON at 90% with one
  session.
- **A turn's request is 3.6–3.8 KB through agentic-api**, which holds the state, against 8.4 KB direct.
- **agentic-api used about 5.5 ms of CPU a Responses round**: 8.6 s for about 1,500 rounds in 71 s at 10 sessions per
  arm, 12% of one core. Its peak RSS was 44–54 MiB (`gateway_usage.py` over `server-usage.txt`).
- **History rehydration behaved the same on every arm.** At depths 1, 5, 10 and 25 success is 100%, 40%, 20% and 8% on
  each, as every arm fails the recall at turn 2 with this 4B model. That confirms equal behavior, and saying anything
  about depth needs a larger model.

## How it ran

| | |
|---|---|
| agentic-api | a7dc7b5 (0.9.0), built on peacock with a static OpenSSL 3.0.15; HTTP adapter; SQLite store; `--gateway-host 127.0.0.1`; pinned to cores 8–15 of cyan (2× Xeon Silver 4310, 48 threads) |
| Engine | vLLM 0.32.0, Qwen3-4B-Instruct-2507-FP8, the frozen box on Phastform (RTX 5060 Ti 16 GB) with `--enable-auto-tool-choice --tool-call-parser hermes`; both kinds of arm reach it through the same SSH local forward (about +0.45 ms a small request) |
| Benchmark | `response_provider --provider all --seed 20261022` on cores 16–23 of cyan; `run.sh` is the exact script |
| Workloads | Transport at 1, 5 and 10 sessions an arm; history rehydration at depths 1, 5, 10 and 25, 5 sessions each |

## Files

| File | What it holds |
|---|---|
| `transport-s1.summary.md`, `transport-s5.summary.md`, `transport-s10.summary.md` | The benchmark's summary of the transport workload at 1, 5 and 10 sessions an arm, as it wrote it |
| `history-s5.depth_summary.md` | Its summary of the history workload by depth, as it wrote it |
| `campaign.log` | The campaign's log: each run's start and exit status |
| `run.sh` | The script as run |
| `server-usage.txt`, `gateway_usage.py` | agentic-api's CPU and memory samples, and the script that gives its resource numbers above from them (`python gateway_usage.py .`) |

The per-turn `turns.csv` and `run.json` files (8 MB) are kept with the run archive.
