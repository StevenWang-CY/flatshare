# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. A point measured in several campaigns lists each run by its campaign, then the median and range (min–max) over its runs, where the frontier places it. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32, 2 callers, NVIDIA A100-SXM4-80GB

The voice SLO allows a V2V p95 of 1,669 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,419 ms). Figure: `frontier-nvidia-a100-sxm4-80gb.svg`.

### Arm 1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice (a100-r1) |  | 0 | 0 | — | 0.00 | 1,285 | 1,445 | 1,481 | 0 of 50 | — / — | 35 | 0 | — / 0 / — | 1,155 | 39 |
| w5-a1-voice (a100-r2) |  | 0 | 0 | — | 0.00 | 1,283 | 1,414 | 1,436 | 0 of 50 | — / — | 36 | 0 | — / 0 / — | 1,155 | 42 |
| w5-a1-voice (a100-r3) |  | 0 | 0 | — | 0.00 | 1,271 | 1,419 | 1,448 | 0 of 50 | — / — | 35 | 0 | — / 0 / — | 1,155 | 40 |
| w5-a1-voice (a100-r4) |  | 0 | 0 | — | 0.00 | 1,262 | 1,391 | 1,428 | 0 of 50 | — / — | 36 | 0 | — / 0 / — | 1,155 | 48 |
| w5-a1-voice (a100-r5) |  | 0 | 0 | — | 0.00 | 1,277 | 1,424 | 1,457 | 0 of 50 | — / — | 34 | 0 | — / 0 / — | 1,155 | 40 |
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call; median (range) of 5 runs |  | 0 | 0 (0–0) | — | | 1,277 (1,262–1,285) | 1,419 (1,391–1,445) | 1,448 (1,428–1,481) | 0 (0–0) | — / — | 35 (34–36) | | | | |

## Workload: weka-x32-long, 2 callers, NVIDIA A100-SXM4-80GB

The voice SLO allows a V2V p95 of 1,669 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,419 ms). Figure: `frontier-weka-x32-long-nvidia-a100-sxm4-80gb.svg`.

### Arm 3

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a100-tuned-s24 (a100-tuned-r1) |  | 24 | 15,215 | 15,082 | 1.66 | 1,850 | 4,036 | 5,331 | 0 of 50 | 970 / 17,317 | 2,388 | 1 | — / 0 / — | 1,395 | 67 |
| a100-tuned-s24 (a100-tuned-r2) |  | 24 | 14,085 | 13,990 | 1.73 | 1,613 | 3,289 | 3,766 | 0 of 50 | 2,009 / 11,357 | 1,617 | 0 | — / 0 / — | 1,410 | 66 |
| a100-tuned-s24: priority, default chunking (the A100's best stock step cap), discriminating scenario, a warm-up turn opening each call; median (range) of 2 runs |  | 24 | 14,650 (14,085–15,215) | 14,536 (13,990–15,082) | | 1,732 (1,613–1,850) | 3,663 (3,289–4,036) | 4,548 (3,766–5,331) | 0 (0–0) | 1,489 (970–2,009) / 14,337 (11,357–17,317) | 2,003 (1,617–2,388) | | | | |

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-frozen-s24 (a100-r1) |  | 24 | 10,900 | 10,749 | 1.25 | 1,983 | 4,163 | 5,718 | 0 of 50 | 6,520 / 25,848 | 2,304 | 39 | — / 0 / — | 1,410 | 66 |
| w5-disc-frozen-s24 (a100-r2) |  | 24 | 11,543 | 9,820 | 1.46 | 2,175 | 4,783 | 9,614 | 0 of 50 | 5,796 / 22,077 | 2,554 | 29 | — / 0 / — | 1,410 | 66 |
| w5-disc-frozen-s24 (a100-r3) |  | 24 | 9,529 | 8,398 | 0.91 | 1,984 | 4,986 | 5,415 | 0 of 50 | 11,932 / 37,970 | 3,141 | 38 | — / 0 / — | 1,410 | 66 |
| w5-disc-frozen-s24 (a100-r4) |  | 24 | 9,828 | 8,633 | 1.02 | 2,452 | 5,169 | 5,758 | 0 of 50 | 15,270 / 24,311 | 3,332 | 44 | — / 0 / — | 1,410 | 66 |
| w5-disc-frozen-s24 (a100-r5) |  | 24 | 8,213 | 8,312 | 0.81 | 3,219 | 5,710 | 6,746 | 0 of 50 | 18,950 / 36,666 | 3,996 | 75 | — / 0 / — | 1,395 | 66 |
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call; median (range) of 5 runs |  | 24 | 9,828 (8,213–11,543) | 8,633 (8,312–10,749) | | 2,175 (1,983–3,219) | 4,986 (4,163–5,710) | 5,758 (5,415–9,614) | 0 (0–0) | 11,932 (5,796–18,950) / 25,848 (22,077–37,970) | 3,141 (2,304–3,996) | | | | |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-s24 (a100-r1) |  | 24 | 10,841 | 10,125 | 1.22 | 1,484 | 1,663 | 1,691 | 0 of 50 | 5,335 / 26,077 | 174 | 48 | — / 0 / — | 1,410 | 66 |
| w5-disc-pr-ab48-s24 (a100-r2) |  | 24 | 11,914 | 10,851 | 1.50 | 1,444 | 1,658 | 1,770 | 0 of 50 | 5,151 / 19,993 | 204 | 31 | — / 0 / — | 1,410 | 65 |
| w5-disc-pr-ab48-s24 (a100-r3) |  | 24 | 9,310 | 8,025 | 0.89 | 1,450 | 1,611 | 1,669 | 0 of 50 | 13,771 / 37,973 | 155 | 55 | — / 0 / — | 1,410 | 65 |
| w5-disc-pr-ab48-s24 (a100-r4) |  | 24 | 9,302 | 8,651 | 0.96 | 1,451 | 1,634 | 1,936 | 0 of 50 | 15,280 / 25,115 | 274 | 76 | — / 0 / — | 1,410 | 67 |
| w5-disc-pr-ab48-s24 (a100-r5) |  | 24 | 8,139 | 7,922 | 0.81 | 1,439 | 1,609 | 1,664 | 0 of 50 | 19,806 / 41,596 | 163 | 78 | — / 0 / — | 1,410 | 66 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call; median (range) of 5 runs | • | 24 | 9,310 (8,139–11,914) | 8,651 (7,922–10,851) | | 1,450 (1,439–1,484) | 1,634 (1,609–1,663) | 1,691 (1,664–1,936) | 0 (0–0) | 13,771 (5,151–19,806) / 26,077 (19,993–41,596) | 174 (155–274) | | | | |
| a100-pr-ab48-2048-s24 (a100-prtuned-r1) |  | 24 | 15,229 | 15,124 | 1.69 | 1,429 | 1,810 | 1,874 | 0 of 50 | 1,042 / 18,392 | 399 | 5 | — / 0 / — | 1,395 | 68 |
| a100-pr-ab48-2048-s24 (a100-prtuned-r2) |  | 24 | 14,320 | 14,145 | 1.79 | 1,536 | 1,843 | 1,907 | 0 of 50 | 2,025 / 11,202 | 399 | 2 | — / 0 / — | 1,410 | 67 |
| a100-pr-ab48-2048-s24: vLLM #63350 and #63351 with a 48-token window at default chunking (the A100's best stock step cap), no gateway policy, a warm-up turn opening each call; median (range) of 2 runs | • | 24 | 14,774 (14,320–15,229) | 14,635 (14,145–15,124) | | 1,482 (1,429–1,536) | 1,826 (1,810–1,843) | 1,891 (1,874–1,907) | 0 (0–0) | 1,534 (1,042–2,025) / 14,797 (11,202–18,392) | 399 (399–399) | | | | |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w5-a1-voice (a100-r1) | 675 / 732 | 678 / 732 | 40 / 50 | 130 / 188 | 132 / 191 | 116 / 169 | 936 / 1,026 | 249 / 0 / 0 |
| w5-a1-voice (a100-r2) | 675 / 689 | 677 / 692 | 41 / 50 | 142 / 191 | 146 / 192 | 104 / 164 | 935 / 1,030 | 249 / 0 / 0 |
| w5-a1-voice (a100-r3) | 674 / 712 | 676 / 714 | 43 / 49 | 130 / 204 | 133 / 205 | 96 / 153 | 915 / 1,023 | 249 / 0 / 0 |
| w5-a1-voice (a100-r4) | 676 / 690 | 678 / 693 | 42 / 50 | 125 / 188 | 129 / 189 | 102 / 155 | 913 / 1,018 | 249 / 0 / 0 |
| w5-a1-voice (a100-r5) | 674 / 689 | 677 / 692 | 39 / 46 | 137 / 196 | 138 / 198 | 99 / 151 | 921 / 996 | 249 / 0 / 0 |
| a100-tuned-s24 (a100-tuned-r1) | 727 / 760 | 730 / 763 | 325 / 2,403 | 530 / 2,812 | 531 / 2,813 | 183 / 215 | 1,477 / 3,749 | 249 / 0 / 0 |
| a100-tuned-s24 (a100-tuned-r2) | 714 / 764 | 716 / 766 | 212 / 1,660 | 386 / 2,083 | 387 / 2,084 | 128 / 198 | 1,302 / 2,966 | 249 / 0 / 0 |
| w5-disc-frozen-s24 (a100-r1) | 724 / 762 | 727 / 764 | 342 / 2,341 | 785 / 2,816 | 786 / 2,817 | 132 / 199 | 1,663 / 3,678 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (a100-r2) | 719 / 759 | 721 / 762 | 331 / 2,854 | 897 / 3,590 | 898 / 3,591 | 130 / 218 | 1,799 / 4,504 | 249 / 0 / 0 |
| w5-disc-frozen-s24 (a100-r3) | 721 / 765 | 724 / 767 | 348 / 3,175 | 806 / 3,758 | 808 / 3,759 | 128 / 161 | 1,652 / 4,649 | 249 / 0 / 0 |
| w5-disc-frozen-s24 (a100-r4) | 726 / 761 | 729 / 764 | 548 / 3,289 | 1,265 / 3,821 | 1,266 / 3,822 | 132 / 199 | 2,091 / 4,703 | 249 / 0 / 0 |
| w5-disc-frozen-s24 (a100-r5) | 728 / 768 | 730 / 770 | 1,402 / 4,021 | 2,040 / 4,503 | 2,041 / 4,504 | 134 / 198 | 2,904 / 5,380 | 249 / 0 / 0 |
| w5-disc-pr-ab48-s24 (a100-r1) | 722 / 758 | 724 / 760 | 132 / 183 | 233 / 338 | 236 / 339 | 183 / 211 | 1,140 / 1,267 | 249 / 0 / 0 |
| w5-disc-pr-ab48-s24 (a100-r2) | 720 / 767 | 722 / 769 | 129 / 216 | 231 / 335 | 233 / 337 | 125 / 185 | 1,086 / 1,312 | 249 / 0 / 0 |
| w5-disc-pr-ab48-s24 (a100-r3) | 725 / 788 | 727 / 791 | 127 / 169 | 222 / 304 | 224 / 306 | 129 / 205 | 1,097 / 1,209 | 249 / 0 / 0 |
| w5-disc-pr-ab48-s24 (a100-r4) | 731 / 769 | 732 / 772 | 142 / 292 | 231 / 390 | 232 / 391 | 123 / 164 | 1,108 / 1,262 | 249 / 0 / 0 |
| w5-disc-pr-ab48-s24 (a100-r5) | 726 / 761 | 728 / 764 | 140 / 181 | 232 / 319 | 233 / 320 | 131 / 164 | 1,094 / 1,224 | 249 / 0 / 0 |
| a100-pr-ab48-2048-s24 (a100-prtuned-r1) | 718 / 747 | 720 / 749 | 114 / 412 | 233 / 510 | 234 / 511 | 124 / 194 | 1,096 / 1,402 | 249 / 0 / 0 |
| a100-pr-ab48-2048-s24 (a100-prtuned-r2) | 709 / 760 | 711 / 762 | 132 / 414 | 292 / 584 | 298 / 585 | 128 / 212 | 1,170 / 1,507 | 249 / 0 / 0 |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice (a100-r1) | 0 | 1,682 | 38,736 | 0 | — | — | 2,415 |
| w5-a1-voice (a100-r2) | 0 | 1,742 | 39,088 | 0 | 0 | 0 | 2,527 |
| w5-a1-voice (a100-r3) | 0 | 1,713 | 37,888 | 0 | — | — | 2,447 |
| w5-a1-voice (a100-r4) | 0 | 1,713 | 38,288 | 0 | 0 | 0 | 2,447 |
| w5-a1-voice (a100-r5) | 0 | 1,687 | 38,864 | 0 | — | — | 2,495 |
| a100-tuned-s24 (a100-tuned-r1) | 1 | 1,897,823 | 255,056 | 3,036,256 | 3,035,563 | -693 | 1,811,798 |
| a100-tuned-s24 (a100-tuned-r2) | 0 | 1,777,346 | 334,000 | 2,531,088 | 2,530,022 | -1,066 | 1,658,491 |
| w5-disc-frozen-s24 (a100-r1) | 39 | 2,413,743 | 139,984 | 1,273,376 | 1,444,143 | 170,767 | 2,348,834 |
| w5-disc-frozen-s24 (a100-r2) | 29 | 2,134,460 | 185,792 | 1,935,632 | 2,115,707 | 180,075 | 2,039,998 |
| w5-disc-frozen-s24 (a100-r3) | 38 | 2,292,783 | 84,176 | 897,920 | 1,172,722 | 274,802 | 2,254,234 |
| w5-disc-frozen-s24 (a100-r4) | 44 | 2,863,281 | 71,008 | 765,792 | 1,071,162 | 305,370 | 2,880,817 |
| w5-disc-frozen-s24 (a100-r5) | 75 | 3,022,637 | 21,872 | 203,488 | 698,070 | 494,582 | 3,044,313 |
| w5-disc-pr-ab48-s24 (a100-r1) | 48 | 2,181,244 | 122,976 | 1,267,264 | 1,457,068 | 189,804 | 2,136,182 |
| w5-disc-pr-ab48-s24 (a100-r2) | 31 | 1,989,211 | 156,448 | 1,900,960 | 1,972,910 | 71,950 | 1,851,180 |
| w5-disc-pr-ab48-s24 (a100-r3) | 55 | 2,332,161 | 69,744 | 815,200 | 1,047,744 | 232,544 | 2,287,937 |
| w5-disc-pr-ab48-s24 (a100-r4) | 76 | 2,634,058 | 36,928 | 615,024 | 865,037 | 250,013 | 2,616,337 |
| w5-disc-pr-ab48-s24 (a100-r5) | 78 | 2,476,072 | 40,368 | 178,096 | 410,236 | 232,140 | 2,474,036 |
| a100-pr-ab48-2048-s24 (a100-prtuned-r1) | 5 | 1,903,757 | 239,504 | 2,885,664 | 2,895,117 | 9,453 | 1,796,969 |
| a100-pr-ab48-2048-s24 (a100-prtuned-r2) | 2 | 1,700,688 | 323,632 | 2,794,000 | 2,776,090 | -17,910 | 1,595,194 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call | 5 | 295 | 680 | 139 | 111 | 351 | 1,281 | 1,274 | 1,416 | 41 |
| a100-tuned-s24: priority, default chunking (the A100's best stock step cap), discriminating scenario, a warm-up turn opening each call | 2 | 118 | 720 | 842 | 154 | 342 | 2,059 | 1,761 | 3,695 | 534 |
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call | 5 | 295 | 728 | 1,489 | 138 | 348 | 2,704 | 2,238 | 5,054 | 1,049 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call | 5 | 295 | 729 | 243 | 142 | 347 | 1,461 | 1,454 | 1,646 | 142 |
| a100-pr-ab48-2048-s24: vLLM #63350 and #63351 with a 48-token window at default chunking (the A100's best stock step cap), no gateway policy, a warm-up turn opening each call | 2 | 118 | 715 | 300 | 140 | 341 | 1,495 | 1,500 | 1,826 | 191 |
