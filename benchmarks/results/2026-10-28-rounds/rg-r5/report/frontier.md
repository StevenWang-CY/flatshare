# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32, 2 callers

The voice SLO allows a V2V p95 of 2,092 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,842 ms). Figure: `frontier.svg`.

### Arm 1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call |  | 0 | 0 | — | 0.00 | 1,641 | 1,842 | 1,898 | 0 of 50 | — / — | 64 | 0 | — / 0 / — | 2,580 | 57 |

## Workload: weka-x32-long, 2 callers

The voice SLO allows a V2V p95 of 2,092 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,842 ms). Figure: `frontier-weka-x32-long.svg`.

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call |  | 24 | 3,043 | 3,322 | 0.27 | 4,385 | 12,309 | 12,727 | 0 of 50 | 90,272 / 150,658 | 8,221 | 25 | — / 0 / — | 2,730 | 68 |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call |  | 24 | 2,969 | 3,060 | 0.26 | 1,934 | 2,246 | 2,344 | 0 of 50 | 89,626 / 148,099 | 396 | 39 | — / 0 / — | 2,730 | 68 |

### Arm 9-pr-ab48-b256-t0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn |  | 24 | 2,831 | 2,922 | 0.23 | 1,850 | 2,064 | 2,167 | 0 of 50 | 100,950 / 155,076 | 290 | 0 | — / 0 / — | 2,730 | 67 |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w5-a1-voice | 740 / 801 | 745 / 806 | 79 / 100 | 300 / 470 | 303 / 473 | 195 / 273 | 1,262 / 1,500 | 249 / 0 / 0 |
| w5-disc-frozen-s24 | 808 / 831 | 813 / 836 | 1,356 / 8,298 | 3,032 / 10,985 | 3,042 / 10,991 | 209 / 286 | 4,102 / 12,012 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 | 808 / 856 | 813 / 861 | 260 / 416 | 510 / 814 | 517 / 817 | 249 / 295 | 1,589 / 1,910 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-b256-t0-s24 | 813 / 840 | 819 / 847 | 236 / 322 | 442 / 653 | 451 / 655 | 212 / 280 | 1,493 / 1,716 | 248 / 0 / 0 |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice | 0 | 1,650 | 40,128 | 0 | — | — | 2,543 |
| w5-disc-frozen-s24 | 25 | 1,287,131 | 20,832 | 58,160 | 271,054 | 212,894 | 1,293,668 |
| w5-disc-pr-ab48-s24 | 39 | 1,005,303 | 17,952 | 53,888 | 146,268 | 92,380 | 1,031,156 |
| sweep-disc-pr-ab48-b256-t0-s24 | 0 | 940,248 | 23,648 | 46,272 | 46,261 | -11 | 937,131 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call | 1 | 59 | 752 | 326 | 207 | 342 | 1,627 | 1,641 | 1,842 | 79 |
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call | 1 | 59 | 813 | 3,738 | 225 | 305 | 5,081 | 4,385 | 12,309 | 2,453 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call | 1 | 59 | 820 | 541 | 248 | 361 | 1,970 | 1,934 | 2,246 | 270 |
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn | 1 | 59 | 825 | 460 | 226 | 349 | 1,860 | 1,850 | 2,064 | 221 |
