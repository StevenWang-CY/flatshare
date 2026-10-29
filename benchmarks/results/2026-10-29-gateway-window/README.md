# The gateway's cost on the deployed build (2026-10-29)

What the gateway adds to a voice request, measured on the build the deployment runs: version 1.8 (5a7e31c), on
2026-10-29 at 19:30Z. cyan's gateway served one GPU node, and Phastform's engine and TTS server were reached through
the tunnel's sockets. No call was in progress.

<!-- table.py table -->
| Stage | Through the gateway minus direct, median | p95 | Pairs |
|---|---:|---:|---:|
| First token | +0.12 ms | +1.22 ms | 80 |
| First sentence | +0.06 ms | +0.84 ms | 80 |
| End of the reply | +0.20 ms | +1.08 ms | 80 |

| Path | First token | First sentence | End of the reply | Gap between tokens |
|---|---:|---:|---:|---:|
| Direct, the engine's socket | 43.7 ms | 150.6 ms | 429.9 ms | 12.1 ms |
| Through the gateway | 43.8 ms | 150.5 ms | 430.0 ms | 12.1 ms |

| Requests through the gateway | Over 2 ms in its own leg | The 99th percentile |
|---:|---:|---:|
| 82 | 0 | ≤ 0.2 ms |

| Text | Its audio | First audio, median | p95 | Requests |
|---|---:|---:|---:|---:|
| A clause | 1.13 s | 124 ms | 127 ms | 40 |
| A sentence | 2.89 s | 135 ms | 137 ms | 40 |
| Two sentences | 5.06 s | 148 ms | 150 ms | 40 |
<!-- /table.py table -->

- **The gateway adds +0.12 ms to the first token at the median**, over 80 pairs of the same request through the gateway
  and on the engine's own socket (p95 +1.22 ms). The reply's pace is the same on both paths: 12.1 ms between tokens.
- **No request spent over 2 ms in the gateway's own leg** (0 of 82), and the 99th percentile was at most 0.2 ms, inside
  the overhead objective ([SLO.md](../../../docs/SLO.md#the-production-objectives)).
- **A clause's first audio comes about 10 ms before a sentence's** (10.3 ms between the medians, 124 and 135 ms), so
  speaking a clause first would save about that much of a 2,034 ms SLO.
- **W6's 0.29 ms** ([its record](../2026-10-25-w6-transport/README.md)) was its own build's measurement, v0.3-191.

## How it ran

| Part | What | How |
|---|---|---|
| A | The voice agent's LLM request, greedy, 60 tokens | Once through the gateway's voice socket and once on the engine's socket each round, alternating; 80 rounds after 2 warm-ups, as one realtime session, one request at a time |
| B | The gateway's overhead histogram, `flatshare_gateway_overhead_seconds` | Scraped from the voice socket's `/metrics` before and after A |
| C | The TTS server's first audio: Kokoro, `af_heart`, raw PCM, streamed | Three texts interleaved, 40 requests each after 2 warm-ups: a clause, the voice agent's typical first sentence, and two sentences |

Each part has its own time bound, and the run stops at 20 minutes. The script refuses to start while a call is on, and
it changes no unit, configuration or release.

## Files

- `results.json`: the run's output as written.
- `window.py`: the script as run. It was written for version 1.7's window, as `rv3_window.py`; the run used version 1.8.
- `table.py`: the tables above, from `results.json` (`python3 table.py`).
