# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32-long, 2 callers, NVIDIA A100-SXM4-80GB

No point of voice alone with 2 callers sets a voice SLO. Figure: `frontier-weka-x32-long-nvidia-a100-sxm4-80gb.svg`.

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a100-aprime-frozen-s32: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call, 32 sessions |  | 32 | 8,614 | 8,649 | 0.79 | 2,670 | 5,222 | 5,527 | 0 of 50 | 24,479 / 55,810 | 3,014 | 57 | — / 0 / — | 1,395 | 68 |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a100-aprime-pr-ab48-s32: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call, 32 sessions |  | 32 | 8,547 | 8,487 | 0.75 | 1,462 | 1,617 | 1,650 | 0 of 50 | 25,881 / 52,273 | 180 | 65 | — / 0 / — | 1,395 | 67 |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| a100-aprime-frozen-s32 | 728 / 765 | 730 / 767 | 783 / 3,027 | 1,488 / 3,948 | 1,488 / 3,949 | 136 / 214 | 2,358 / 4,897 | 248 / 0 / 0 |
| a100-aprime-pr-ab48-s32 | 731 / 764 | 733 / 766 | 141 / 196 | 228 / 321 | 229 / 323 | 140 / 205 | 1,113 / 1,245 | 249 / 0 / 0 |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| a100-aprime-frozen-s32 | 57 | 3,114,718 | 32,736 | 56,320 | 428,007 | 371,687 | 3,131,683 |
| a100-aprime-pr-ab48-s32 | 65 | 2,771,737 | 20,368 | 67,088 | 245,780 | 178,692 | 2,793,046 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a100-aprime-frozen-s32: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call, 32 sessions | 1 | 59 | 730 | 1,809 | 147 | 343 | 3,029 | 2,670 | 5,222 | 1,278 |
| a100-aprime-pr-ab48-s32: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call, 32 sessions | 1 | 59 | 736 | 240 | 145 | 338 | 1,459 | 1,462 | 1,617 | 142 |
