# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32, 2 callers

The voice SLO allows a V2V p95 of 2,001 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,751 ms). Figure: `frontier.svg`.

### Arm 1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call |  | 0 | 0 | — | 0.00 | 1,625 | 1,751 | 1,942 | 0 of 50 | — / — | 65 | 0 | — / 0 / — | 2,580 | 51 |

## Workload: weka-x32-long, 2 callers

The voice SLO allows a V2V p95 of 2,001 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,751 ms). Figure: `frontier-weka-x32-long.svg`.

### Arm 4-b1024-t0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-b1024-t0-s24: priority, step cap 1024, long-prefill threshold 0, discriminating scenario, a warm-up turn opening each call |  | 24 | 4,096 | 3,956 | 0.34 | 4,934 | 9,115 | 9,806 | 0 of 50 | 57,068 / 85,953 | 6,696 | 0 | — / 0 / — | 2,730 | 69 |

### Arm 6-cap4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-frozen-cap4-s24: frozen box + static cap, 4 background in flight, discriminating scenario, a warm-up turn opening each call |  | 24 | 3,781 | 3,587 | 0.32 | 4,126 | 10,884 | 11,380 | 0 of 50 | 62,862 / 95,610 | 7,447 | 27 | 0 / 0 / 0 | 2,722 | 68 |

### Arm 6-cap8

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-frozen-cap8-s24: frozen box + static cap, 8 background in flight, discriminating scenario, a warm-up turn opening each call |  | 24 | 3,739 | 3,644 | 0.31 | 4,386 | 9,228 | 12,414 | 0 of 50 | 63,630 / 100,994 | 5,987 | 37 | 0 / 0 / 0 | 2,722 | 68 |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call |  | 24 | 3,488 | 3,355 | 0.28 | 1,945 | 2,236 | 2,296 | 0 of 50 | 71,195 / 100,311 | 375 | 49 | — / 0 / — | 2,730 | 68 |

### Arm 9-pr-ab48-b1024-t0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-b1024-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 1024, threshold 0, discriminating scenario, warm-up turn |  | 24 | 3,890 | 3,832 | 0.31 | 2,058 | 2,420 | 2,707 | 0 of 50 | 63,216 / 89,599 | 661 | 2 | — / 0 / — | 2,737 | 70 |

### Arm 9-pr-ab48-b1024-t256

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-b1024-t256-s24: vLLM #63350 and #63351, 48-token window, step cap 1024, threshold 256, discriminating scenario, warm-up turn |  | 24 | 3,522 | 3,442 | 0.28 | 2,015 | 3,505 | 4,837 | 0 of 50 | 69,942 / 98,499 | 1,995 | 88 | — / 0 / — | 2,730 | 69 |

### Arm 9-pr-ab48-b256-t0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn |  | 24 | 3,293 | 3,158 | 0.27 | 1,897 | 2,128 | 2,303 | 0 of 50 | 78,636 / 107,665 | 296 | 0 | — / 0 / — | 2,722 | 66 |

### Arm 9-pr-ab48-cap4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-cap4-s24: vLLM #63350 and #63351, 48-token window, static cap 4 background in flight, discriminating scenario, warm-up turn |  | 24 | 3,495 | 3,421 | 0.28 | 1,964 | 2,194 | 2,326 | 0 of 50 | 71,907 / 101,979 | 361 | 44 | 0 / 0 / 0 | 2,730 | 68 |

### Arm 9-pr-ab48-cap8

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-cap8-s24: vLLM #63350 and #63351, 48-token window, static cap 8 background in flight, discriminating scenario, warm-up turn |  | 24 | 3,511 | 3,454 | 0.28 | 1,965 | 2,230 | 2,362 | 0 of 50 | 70,877 / 98,983 | 384 | 38 | 0 / 0 / 0 | 2,722 | 68 |

### Arm 9-pr-ab96-b1024-t0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab96-b1024-t0-s24: vLLM #63350 and #63351, 96-token window, step cap 1024, threshold 0, discriminating scenario, warm-up turn |  | 24 | 3,846 | 3,767 | 0.31 | 2,065 | 2,415 | 2,609 | 0 of 50 | 63,286 / 88,677 | 695 | 2 | — / 0 / — | 2,737 | 69 |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w5-a1-voice | 742 / 772 | 747 / 777 | 78 / 101 | 297 / 418 | 299 / 420 | 226 / 269 | 1,261 / 1,406 | 249 / 0 / 0 |
| sweep-disc-b1024-t0-s24 | 810 / 833 | 815 / 839 | 2,784 / 6,097 | 3,584 / 7,771 | 3,597 / 7,789 | 210 / 294 | 4,619 / 8,831 | 248 / 0 / 0 |
| sweep-disc-frozen-cap4-s24 | 812 / 833 | 816 / 840 | 1,502 / 7,506 | 2,714 / 9,156 | 2,713 / 8,724 | 253 / 299 | 3,760 / 9,795 | 248 / 0 / 0 |
| sweep-disc-frozen-cap8-s24 | 810 / 847 | 815 / 851 | 1,410 / 6,209 | 2,831 / 7,690 | 2,767 / 7,808 | 239 / 327 | 3,853 / 8,885 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 | 808 / 861 | 814 / 870 | 293 / 413 | 541 / 787 | 550 / 791 | 210 / 311 | 1,600 / 1,893 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-b1024-t0-s24 | 811 / 834 | 816 / 840 | 426 / 683 | 702 / 984 | 707 / 986 | 208 / 303 | 1,743 / 2,045 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-b1024-t256-s24 | 808 / 837 | 813 / 843 | 307 / 2,051 | 596 / 2,198 | 592 / 1,933 | 216 / 326 | 1,647 / 2,914 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-b256-t0-s24 | 808 / 834 | 813 / 838 | 229 / 322 | 474 / 687 | 476 / 691 | 212 / 261 | 1,506 / 1,740 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-cap4-s24 | 808 / 839 | 813 / 844 | 274 / 392 | 535 / 834 | 540 / 837 | 207 / 272 | 1,580 / 1,869 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-cap8-s24 | 811 / 856 | 815 / 860 | 283 / 410 | 514 / 743 | 517 / 748 | 212 / 292 | 1,560 / 1,860 | 248 / 0 / 0 |
| sweep-disc-pr-ab96-b1024-t0-s24 | 810 / 831 | 814 / 835 | 422 / 722 | 694 / 999 | 695 / 1,004 | 204 / 249 | 1,725 / 2,041 | 248 / 0 / 0 |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice | 0 | 1,654 | 39,568 | 0 | — | — | 2,527 |
| sweep-disc-b1024-t0-s24 | 0 | 1,802,643 | 12,976 | 71,376 | 71,359 | -17 | 1,798,217 |
| sweep-disc-frozen-cap4-s24 | 27 | 1,693,777 | 31,760 | 46,864 | 176,229 | 129,365 | 1,688,148 |
| sweep-disc-frozen-cap8-s24 | 37 | 1,619,649 | 33,712 | 44,688 | 284,874 | 240,186 | 1,596,666 |
| w5-disc-pr-ab48-s24 | 49 | 1,145,504 | 42,176 | 39,040 | 141,981 | 102,941 | 1,146,664 |
| sweep-disc-pr-ab48-b1024-t0-s24 | 2 | 1,327,555 | 22,224 | 58,112 | 58,338 | 226 | 1,333,546 |
| sweep-disc-pr-ab48-b1024-t256-s24 | 88 | 1,282,279 | 18,704 | 54,480 | 411,579 | 357,099 | 1,288,149 |
| sweep-disc-pr-ab48-b256-t0-s24 | 0 | 1,098,702 | 40,256 | 40,208 | 40,454 | 246 | 1,105,362 |
| sweep-disc-pr-ab48-cap4-s24 | 44 | 1,162,135 | 27,456 | 38,144 | 144,381 | 106,237 | 1,160,517 |
| sweep-disc-pr-ab48-cap8-s24 | 38 | 1,141,141 | 31,168 | 38,624 | 124,434 | 85,810 | 1,140,857 |
| sweep-disc-pr-ab96-b1024-t0-s24 | 2 | 1,297,531 | 23,296 | 56,560 | 56,546 | -14 | 1,303,218 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call | 1 | 59 | 751 | 304 | 221 | 349 | 1,625 | 1,625 | 1,751 | 80 |
| sweep-disc-b1024-t0-s24: priority, step cap 1024, long-prefill threshold 0, discriminating scenario, a warm-up turn opening each call | 1 | 59 | 817 | 3,858 | 230 | 342 | 5,248 | 4,934 | 9,115 | 2,704 |
| sweep-disc-frozen-cap4-s24: frozen box + static cap, 4 background in flight, discriminating scenario, a warm-up turn opening each call | 1 | 59 | 817 | 3,377 | 251 | 461 | 4,907 | 4,126 | 10,885 | 2,387 |
| sweep-disc-frozen-cap8-s24: frozen box + static cap, 8 background in flight, discriminating scenario, a warm-up turn opening each call | 1 | 59 | 820 | 3,387 | 242 | 389 | 4,838 | 4,386 | 9,228 | 2,340 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call | 1 | 59 | 821 | 569 | 231 | 356 | 1,976 | 1,945 | 2,236 | 298 |
| sweep-disc-pr-ab48-b1024-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 1024, threshold 0, discriminating scenario, warm-up turn | 1 | 59 | 821 | 702 | 226 | 346 | 2,095 | 2,058 | 2,420 | 420 |
| sweep-disc-pr-ab48-b1024-t256-s24: vLLM #63350 and #63351, 48-token window, step cap 1024, threshold 256, discriminating scenario, warm-up turn | 1 | 59 | 817 | 778 | 228 | 391 | 2,213 | 2,015 | 3,505 | 513 |
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn | 1 | 59 | 817 | 491 | 217 | 365 | 1,890 | 1,897 | 2,128 | 224 |
| sweep-disc-pr-ab48-cap4-s24: vLLM #63350 and #63351, 48-token window, static cap 4 background in flight, discriminating scenario, warm-up turn | 1 | 59 | 818 | 563 | 214 | 360 | 1,955 | 1,964 | 2,194 | 284 |
| sweep-disc-pr-ab48-cap8-s24: vLLM #63350 and #63351, 48-token window, static cap 8 background in flight, discriminating scenario, warm-up turn | 1 | 59 | 821 | 542 | 229 | 363 | 1,955 | 1,965 | 2,230 | 278 |
| sweep-disc-pr-ab96-b1024-t0-s24: vLLM #63350 and #63351, 96-token window, step cap 1024, threshold 0, discriminating scenario, warm-up turn | 1 | 59 | 817 | 699 | 209 | 367 | 2,092 | 2,065 | 2,415 | 433 |
