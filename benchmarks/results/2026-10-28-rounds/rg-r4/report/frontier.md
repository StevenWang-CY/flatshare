# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32, 2 callers

The voice SLO allows a V2V p95 of 2,080 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,830 ms). Figure: `frontier.svg`.

### Arm 1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call |  | 0 | 0 | — | 0.00 | 1,613 | 1,830 | 1,900 | 0 of 50 | — / — | 64 | 0 | — / 0 / — | 2,584 | 68 |

## Workload: weka-x32-long, 2 callers

The voice SLO allows a V2V p95 of 2,080 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,830 ms). Figure: `frontier-weka-x32-long.svg`.

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call |  | 24 | 3,336 | 3,479 | 0.32 | 4,877 | 12,010 | 12,870 | 0 of 50 | 68,657 / 109,415 | 8,690 | 25 | — / 0 / — | 2,565 | 89 |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call |  | 24 | 3,288 | 3,337 | 0.31 | 1,957 | 2,264 | 2,898 | 0 of 50 | 77,701 / 114,820 | 405 | 48 | — / 0 / — | 2,546 | 90 |

### Arm 9-pr-ab48-b256-t0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn |  | 24 | 3,106 | 3,067 | 0.29 | 1,843 | 2,093 | 2,184 | 0 of 50 | 84,794 / 120,179 | 273 | 1 | — / 0 / — | 2,632 | 88 |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w5-a1-voice | 769 / 797 | 775 / 804 | 82 / 102 | 302 / 474 | 314 / 478 | 150 / 233 | 1,245 / 1,483 | 248 / 0 / 0 |
| w5-disc-frozen-s24 | 843 / 857 | 846 / 862 | 2,476 / 8,739 | 3,562 / 10,558 | 3,528 / 10,569 | 181 / 276 | 4,514 / 11,701 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 | 839 / 861 | 843 / 866 | 294 / 439 | 574 / 792 | 582 / 794 | 177 / 240 | 1,606 / 1,811 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-b256-t0-s24 | 838 / 858 | 843 / 863 | 222 / 308 | 436 / 679 | 450 / 683 | 188 / 242 | 1,473 / 1,742 | 248 / 0 / 0 |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice | 0 | 1,685 | 40,384 | 0 | — | — | 2,607 |
| w5-disc-frozen-s24 | 25 | 1,439,899 | 20,432 | 38,528 | 269,502 | 230,974 | 1,446,783 |
| w5-disc-pr-ab48-s24 | 48 | 1,046,092 | 15,168 | 36,752 | 129,696 | 92,944 | 1,061,101 |
| sweep-disc-pr-ab48-b256-t0-s24 | 1 | 966,025 | 19,536 | 29,488 | 31,192 | 1,704 | 973,890 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call | 1 | 59 | 777 | 326 | 157 | 352 | 1,612 | 1,613 | 1,830 | 82 |
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call | 1 | 59 | 847 | 4,411 | 187 | 409 | 5,853 | 4,877 | 12,010 | 3,284 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call | 1 | 59 | 842 | 595 | 187 | 357 | 1,982 | 1,957 | 2,264 | 329 |
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn | 1 | 59 | 843 | 462 | 191 | 362 | 1,859 | 1,843 | 2,093 | 218 |
