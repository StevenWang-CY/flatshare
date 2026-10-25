# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. A point measured in several campaigns lists each run by its campaign, then the median and range (min–max) over its runs, where the frontier places it. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32, 2 callers

The voice SLO allows a V2V p95 of 2,017 ms, 250 ms above voice alone with as many callers (a1-voice: 1,767 ms). Figure: `frontier.svg`.

### Arm 1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a1-voice (r3) |  | 0 | 0 | — | 0.00 | 1,622 | 1,805 | 1,858 | 0 of 50 | — / — | 66 | 0 | 0 / 0 / 0 | 2,580 | 56 |
| a1-voice (r4) |  | 0 | 0 | — | 0.00 | 1,578 | 1,751 | 1,798 | 0 of 50 | — / — | 67 | 0 | 0 / 0 / 0 | 2,640 | 59 |
| a1-voice (s1) |  | 0 | 0 | — | 0.00 | 1,609 | 1,770 | 1,799 | 0 of 50 | — / — | 59 | 0 | 0 / 0 / 0 | 2,580 | 54 |
| a1-voice (s2) |  | 0 | 0 | — | 0.00 | 1,614 | 1,767 | 1,861 | 0 of 50 | — / — | 60 | 0 | 0 / 0 / 0 | 2,580 | 57 |
| a1-voice (w2) |  | 0 | 0 | — | 0.00 | 1,611 | 1,744 | 1,784 | 0 of 50 | — / — | 96 | 0 | 0 / 0 / 0 | — | — |
| a1-voice: frozen box, voice alone; median (range) of 5 runs |  | 0 | 0 (0–0) | — | | 1,611 (1,578–1,622) | 1,767 (1,744–1,805) | 1,799 (1,784–1,861) | 0 (0–0) | — / — | 66 (59–96) | | | | |

### Arm 2

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a2-stock-s8: stock vLLM: FCFS, default chunking | • | 8 | 2,800 | 2,602 | 0.85 | 1,667 | 1,886 | 1,959 | 0 of 50 | 80 / 576 | 114 | 0 | 0 / 0 / 0 | — | — |
| a2-stock-s16: stock vLLM: FCFS, default chunking | • | 16 | 5,159 | 5,489 | 0.98 | 1,727 | 2,406 | 5,232 | 0 of 50 | 137 / 2,750 | 446 | 0 | 0 / 0 / 0 | — | — |
| a2-stock-s24: stock vLLM: FCFS, default chunking | • | 24 | 9,562 | 9,901 | 2.41 | 1,952 | 3,008 | 4,253 | 0 of 50 | 135 / 2,581 | 1,280 | 1 | 0 / 0 / 0 | — | — |

### Arm 3

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a3-priority-s8: priority, default chunking | • | 8 | 2,702 | 2,566 | 0.81 | 1,670 | 1,871 | 2,038 | 0 of 50 | 77 / 620 | 80 | 0 | 0 / 0 / 0 | — | — |
| a3-priority-s16: priority, default chunking | • | 16 | 5,168 | 5,403 | 1.04 | 1,719 | 2,400 | 2,898 | 0 of 50 | 121 / 3,312 | 333 | 0 | 0 / 0 / 0 | — | — |
| a3-priority-s24: priority, default chunking | • | 24 | 9,224 | 9,635 | 2.31 | 1,887 | 4,360 | 4,458 | 0 of 50 | 136 / 2,969 | 2,243 | 0 | 0 / 0 / 0 | — | — |

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a4-b1024-t512-s16: priority, step cap 1024, long-prefill threshold 512 |  | 16 | 5,015 | 5,420 | 0.95 | 1,703 | 2,510 | 2,684 | 0 of 50 | 129 / 3,123 | 273 | 0 | 0 / 0 / 0 | — | — |
| a4-b1024-t256-s16: priority, step cap 1024, long-prefill threshold 256 |  | 16 | 5,032 | 5,412 | 0.96 | 1,724 | 2,257 | 2,504 | 0 of 50 | 129 / 3,563 | 178 | 0 | 0 / 0 / 0 | — | — |
| a4-b512-t0-s16: priority, step cap 512, no long-prefill threshold |  | 16 | 5,097 | 5,555 | 0.96 | 1,718 | 2,402 | 2,574 | 0 of 50 | 128 / 2,994 | 224 | 1 | 0 / 0 / 0 | — | — |
| a4-b512-t256-s16 (s1) |  | 16 | 5,167 | 5,351 | 1.05 | 1,685 | 2,323 | 2,536 | 0 of 50 | 119 / 3,852 | 179 | 0 | 0 / 0 / 0 | 2,790 | 63 |
| a4-b512-t256-s16 (s2) |  | 16 | 5,176 | 5,498 | 1.01 | 1,723 | 2,530 | 2,942 | 0 of 50 | 126 / 3,840 | 190 | 0 | 0 / 0 / 0 | 2,790 | 63 |
| a4-b512-t256-s16 (w2) |  | 16 | 5,068 | 5,552 | 0.96 | 1,736 | 2,215 | 2,925 | 0 of 50 | 139 / 3,616 | 224 | 0 | 0 / 0 / 0 | — | — |
| a4-b512-t256-s16: priority, step cap 512, long-prefill threshold 256 (frozen box); median (range) of 3 runs |  | 16 | 5,167 (5,068–5,176) | 5,498 (5,351–5,552) | | 1,723 (1,685–1,736) | 2,323 (2,215–2,530) | 2,925 (2,536–2,942) | 0 (0–0) | 126 (119–139) / 3,840 (3,616–3,852) | 190 (179–224) | | | | |
| a4-b1024-t0-s16: priority, step cap 1024, no long-prefill threshold | • | 16 | 5,246 | 5,492 | 1.02 | 1,687 | 2,167 | 2,479 | 0 of 50 | 124 / 2,778 | 144 | 0 | 0 / 0 / 0 | — | — |
| a4-b512-t256-s24: priority, step cap 512, long-prefill threshold 256 (frozen box) |  | 24 | 9,371 | 9,837 | 2.27 | 1,986 | 3,865 | 4,883 | 0 of 50 | 155 / 3,834 | 542 | 0 | 0 / 0 / 0 | — | — |
| a4-b1024-t0-s24: priority, step cap 1024, long-prefill threshold 0 | • | 24 | 9,498 | 9,939 | 2.35 | 1,910 | 3,683 | 4,178 | 0 of 50 | 138 / 2,754 | 1,539 | 0 | 0 / 0 / 0 | — | — |

### Arm 4-no-preemptive

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| factor-no-preemptive-s16: frozen box, preemptive generation off |  | 16 | 5,121 | 5,580 | 0.98 | 1,687 | 2,278 | 2,953 | 0 of 50 | 144 / 3,601 | 294 | 1 | 0 / 0 / 0 | — | — |

### Arm 5

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a5-queued8192-s16: frozen box + max-num-queued-tokens 8192 |  | 16 | 4,454 | 4,375 | 0.90 | 1,703 | 2,133 | ≥30,000 | 2 of 50 | 108 / 1,659 | 137 | 0 | 0 / 43 / 0 | — | — |
| a5-watermark10-s16: frozen box + watermark 0.1 | • | 16 | 5,001 | 5,486 | 0.95 | 1,703 | 2,495 | 2,986 | 0 of 50 | 133 / 2,782 | 216 | 0 | 0 / 0 / 0 | — | — |

### Arm 6

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a6-reserve2048-s16: frozen box + token reservation, 2,048 tokens |  | 16 | 4,922 | 5,340 | 0.92 | 1,688 | 2,328 | 3,068 | 0 of 50 | 142 / 2,754 | 276 | 0 | 0 / 0 / 0 | — | — |
| a6-cap8-s16: frozen box + static cap, 8 background in flight |  | 16 | 5,296 | 5,702 | 1.04 | 1,683 | 3,065 | 3,647 | 0 of 50 | 128 / 2,845 | 510 | 1 | 0 / 0 / 0 | — | — |
| a6-cap2-s16: frozen box + static cap, 2 background in flight | • | 16 | 5,569 | 6,497 | 1.10 | 1,755 | 2,208 | 2,729 | 0 of 50 | 203 / 4,069 | 236 | 0 | 0 / 0 / 0 | — | — |

### Arm 7a

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a7a-pr54530-s16: priority, max-num-seqs 12, with vLLM #54530 |  | 16 | 4,897 | 5,348 | 0.92 | 1,728 | 2,736 | 3,103 | 0 of 50 | 166 / 2,387 | 461 | 0 | 0 / 0 / 0 | — | — |
| a7a-seqs12-s16: priority, max-num-seqs 12, without vLLM #54530 | • | 16 | 4,981 | 5,339 | 0.94 | 1,710 | 2,104 | 2,984 | 0 of 50 | 164 / 2,435 | 161 | 0 | 0 / 0 / 0 | — | — |

### Arm 7b

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a7b-sglang-hicache-s8: SGLang 0.5.22 with HiCache 4x (write-through) | • | 8 | 2,764 | 2,463 | 0.85 | 1,660 | 1,822 | 2,011 | 0 of 50 | 86 / 461 | 97 | — | 0 / 0 / 0 | 2,790 | 55 |
| a7b-sglang-s8: SGLang 0.5.22: priority scheduling with preemption |  | 8 | 2,770 | 2,363 | 0.78 | 1,681 | 3,228 | 4,538 | 0 of 50 | 93 / 4,883 | 1,440 | — | 0 / 0 / 0 | — | — |
| a7b-sglang-hicache-s16: SGLang 0.5.22 with HiCache 4x (write-through) | • | 16 | 4,924 | 5,207 | 0.94 | 1,720 | 2,218 | 4,088 | 0 of 50 | 99 / 3,849 | 249 | — | 0 / 0 / 0 | 2,790 | 62 |
| a7b-sglang-s16: SGLang 0.5.22: priority scheduling with preemption | • | 16 | 5,543 | 6,296 | 1.20 | 1,882 | 8,477 | 9,458 | 0 of 50 | 602 / 10,030 | 2,284 | — | 0 / 0 / 0 | — | — |

### Arm 8a

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a8a-flow-s16: frozen box + flow control (llm-d, emulated), ceilings 6/6/6 |  | 16 | 5,413 | 5,910 | 1.04 | 1,768 | 2,468 | 2,821 | 0 of 50 | 182 / 3,002 | 216 | 2 | 1 / 0 / 0 | — | — |

### Arm 8b

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a8b-smg-s16: frozen box behind SMG 1.13.0, as shipped |  | 16 | 5,545 | 6,484 | 1.10 | 1,723 | 2,202 | 3,255 | 0 of 50 | 220 / 4,173 | 237 | 0 | 0 / 0 / 0 | — | — |

### Arm 8c

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a8c-resubmit-s16: frozen box + preempt-resubmit before the first byte |  | 16 | 5,090 | 5,528 | 0.96 | 1,678 | 2,520 | 2,711 | 0 of 50 | 142 / 4,773 | 235 | 0 | 0 / 0 / 11 | — | — |

### Arm 9

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| a9-s8: Leaseholder: timed headroom, critical-window throttle, degrade; make-room | • | 8 | 2,675 | 2,463 | 0.81 | 1,745 | 2,060 | 2,090 | 0 of 50 | 79 / 620 | 85 | 0 | 0 / 0 / 0 | 2,790 | 56 |
| a9-s16 (a9-standard) |  | 16 | 5,244 | 5,605 | 1.05 | 1,707 | 2,193 | 2,542 | 0 of 50 | 127 / 2,171 | 211 | 0 | 0 / 0 / 0 | 2,790 | 63 |
| a9-s16 (s1) |  | 16 | 4,990 | 5,429 | 0.94 | 1,684 | 2,446 | 2,532 | 0 of 50 | 137 / 2,774 | 211 | 0 | 0 / 0 / 0 | 2,790 | 63 |
| a9-s16 (s2) |  | 16 | 5,271 | 5,516 | 1.08 | 1,693 | 2,196 | 2,714 | 0 of 50 | 123 / 2,584 | 196 | 0 | 0 / 0 / 0 | 2,790 | 63 |
| a9-s16: Leaseholder: timed headroom, critical-window throttle, degrade; make-room; median (range) of 3 runs | • | 16 | 5,244 (4,990–5,271) | 5,516 (5,429–5,605) | | 1,693 (1,684–1,707) | 2,196 (2,193–2,446) | 2,542 (2,532–2,714) | 0 (0–0) | 127 (123–137) / 2,584 (2,171–2,774) | 211 (196–211) | | | | |
| a9-s24: Leaseholder: timed headroom, critical-window throttle, degrade; make-room | • | 24 | 9,606 | 10,028 | 2.42 | 1,848 | 2,375 | 2,897 | 0 of 50 | 166 / 3,871 | 357 | 0 | 0 / 0 / 0 | 2,782 | 65 |

## Workload: weka-x32, 4 callers

The voice SLO allows a V2V p95 of 2,267 ms, 250 ms above voice alone with as many callers (calls4-s0: 2,017 ms). Figure: `frontier-4-callers.svg`.

### Arm 1-calls

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| calls4-s0: frozen box, 4 callers, voice alone |  | 0 | 0 | — | 0.00 | 1,655 | 2,017 | 2,110 | 0 of 100 | — / — | 154 | 0 | 0 / 0 / 0 | — | — |

### Arm 4-calls

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| calls4-s16: frozen box, 4 callers |  | 16 | 5,296 | 5,688 | 1.04 | 1,747 | 2,599 | 3,135 | 0 of 100 | 155 / 3,979 | 428 | 0 | 0 / 0 / 0 | — | — |

## Workload: weka-x32, 6 callers

No point of voice alone with 6 callers sets a voice SLO. Figure: `frontier-6-callers.svg`.

### Arm 4-calls

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| calls6-s16: frozen box, 6 callers |  | 16 | 5,067 | 5,680 | 0.96 | 1,735 | 2,670 | 3,471 | 0 of 150 | 137 / 2,923 | 413 | 0 | 0 / 0 / 0 | — | — |

## Workload: weka-x32-long, 2 callers

The voice SLO allows a V2V p95 of 2,017 ms, 250 ms above voice alone with as many callers (a1-voice: 1,767 ms). Figure: `frontier-weka-x32-long.svg`.

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-frozen-s24 (r1) |  | 24 | 3,710 | 3,499 | 0.31 | 4,306 | 10,062 | 12,263 | 0 of 50 | 63,873 / 93,399 | 7,772 | 32 | 0 / 0 / 0 | 2,722 | 70 |
| disc-frozen-s24 (r2) |  | 24 | 3,726 | 3,552 | 0.30 | 5,066 | 9,257 | 10,677 | 0 of 50 | 65,040 / 99,378 | 6,865 | 32 | 0 / 0 / 0 | 2,722 | 68 |
| disc-frozen-s24 (r3) |  | 24 | 3,745 | 3,556 | 0.31 | 4,520 | 10,673 | 12,110 | 0 of 50 | 64,290 / 101,249 | 7,834 | 31 | 0 / 0 / 0 | 2,722 | 68 |
| disc-frozen-s24 (r4) |  | 24 | 3,688 | 3,501 | 0.30 | 5,256 | 10,138 | 10,897 | 0 of 50 | 64,691 / 99,066 | 7,885 | 29 | 0 / 0 / 0 | 2,722 | 69 |
| disc-frozen-s24 (w2c) |  | 24 | 3,703 | 3,566 | 0.30 | 4,270 | 10,811 | 12,427 | 0 of 50 | 66,641 / 104,254 | 8,556 | 31 | 0 / 0 / 0 | 2,722 | 69 |
| disc-frozen-s24: frozen box, discriminating scenario (large contexts); median (range) of 5 runs |  | 24 | 3,710 (3,688–3,745) | 3,552 (3,499–3,566) | | 4,520 (4,270–5,256) | 10,138 (9,257–10,811) | 12,110 (10,677–12,427) | 0 (0–0) | 64,691 (63,873–66,641) / 99,378 (93,399–104,254) | 7,834 (6,865–8,556) | | | | |
| disc-a4-b1024-t0-s24 (r1) |  | 24 | 4,078 | 3,876 | 0.33 | 4,228 | 8,161 | 9,682 | 0 of 50 | 58,318 / 79,201 | 5,807 | 0 | 0 / 0 / 0 | 2,730 | 70 |
| disc-a4-b1024-t0-s24 (r2) |  | 24 | 4,081 | 3,840 | 0.33 | 4,348 | 8,432 | 10,589 | 0 of 50 | 57,266 / 90,237 | 5,924 | 0 | 0 / 0 / 0 | 2,722 | 71 |
| disc-a4-b1024-t0-s24 (r3) |  | 24 | 4,166 | 3,888 | 0.35 | 4,975 | 9,490 | 11,936 | 0 of 50 | 56,895 / 85,102 | 7,006 | 1 | 0 / 0 / 0 | 2,730 | 70 |
| disc-a4-b1024-t0-s24 (r4) |  | 24 | 4,131 | 3,888 | 0.34 | 4,244 | 9,268 | 9,580 | 0 of 50 | 57,262 / 85,131 | 7,161 | 0 | 0 / 0 / 0 | 2,730 | 69 |
| disc-a4-b1024-t0-s24 (w2c) |  | 24 | 4,097 | 3,928 | 0.34 | 4,816 | 8,789 | 10,024 | 0 of 50 | 56,254 / 81,923 | 6,523 | 0 | 0 / 0 / 0 | 2,730 | 69 |
| disc-a4-b1024-t0-s24: priority, step cap 1024, long-prefill threshold 0, discriminating scenario; median (range) of 5 runs | • | 24 | 4,097 (4,078–4,166) | 3,888 (3,840–3,928) | | 4,348 (4,228–4,975) | 8,789 (8,161–9,490) | 10,024 (9,580–11,936) | 0 (0–0) | 57,262 (56,254–58,318) / 85,102 (79,201–90,237) | 6,523 (5,807–7,161) | | | | |

### Arm 5

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a5-watermark10-s24: frozen box + watermark 0.1, discriminating scenario | • | 24 | 3,712 | 3,649 | 0.30 | 4,843 | 10,647 | 13,583 | 0 of 50 | 67,288 / 101,550 | 8,201 | 24 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a5-queued8192-s24: frozen box + max-num-queued-tokens 8192, discriminating scenario |  | 24 | 5,597 | 5,584 | 0.48 | ≥30,000 | ≥30,000 | ≥30,000 | 48 of 50 | 609 / 8,976 | 56 | 0 | 0 / 1,132 / 0 | 2,745 | 67 |

### Arm 6

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a6-reserve2048-s24 (r1) |  | 24 | 3,608 | 3,500 | 0.29 | 3,127 | 5,671 | 6,931 | 0 of 50 | 68,803 / 101,480 | 3,190 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a6-reserve2048-s24 (r2) |  | 24 | 3,559 | 3,446 | 0.28 | 3,205 | 4,906 | 6,298 | 0 of 50 | 70,368 / 104,132 | 2,419 | 0 | 0 / 0 / 0 | 2,722 | 68 |
| disc-a6-reserve2048-s24 (r3) |  | 24 | 3,543 | 3,551 | 0.28 | 2,878 | 5,288 | 7,085 | 0 of 50 | 68,867 / 102,717 | 2,935 | 0 | 0 / 0 / 0 | 2,722 | 68 |
| disc-a6-reserve2048-s24 (r4) |  | 24 | 3,564 | 3,570 | 0.29 | 3,047 | 5,798 | 6,681 | 0 of 50 | 69,160 / 101,757 | 3,300 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a6-reserve2048-s24 (w2c) |  | 24 | 3,633 | 3,551 | 0.29 | 3,311 | 5,706 | 6,317 | 0 of 50 | 68,051 / 99,411 | 3,631 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a6-reserve2048-s24: frozen box + token reservation, 2,048 tokens, discriminating scenario; median (range) of 5 runs | • | 24 | 3,564 (3,543–3,633) | 3,551 (3,446–3,570) | | 3,127 (2,878–3,311) | 5,671 (4,906–5,798) | 6,681 (6,298–7,085) | 0 (0–0) | 68,867 (68,051–70,368) / 101,757 (99,411–104,132) | 3,190 (2,419–3,631) | | | | |
| disc-a6-cap2-s24: frozen box + static cap, 2 background in flight, discriminating scenario | • | 24 | 3,566 | 3,446 | 0.29 | 3,363 | 8,573 | 10,385 | 0 of 50 | 69,506 / 105,445 | 6,320 | 8 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a6-cap8-s24: frozen box + static cap, 8 background in flight, discriminating scenario | • | 24 | 3,785 | 3,586 | 0.31 | 5,713 | 9,876 | 11,369 | 0 of 50 | 63,414 / 96,158 | 7,502 | 27 | 0 / 0 / 0 | 2,722 | 69 |

### Arm 7a

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a7a-pr54530-s24: priority, max-num-seqs 12, with vLLM #54530, discriminating scenario |  | 24 | 3,718 | 3,624 | 0.30 | 4,687 | 10,930 | 11,870 | 0 of 50 | 66,369 / 98,431 | 7,964 | 30 | 0 / 0 / 0 | 2,722 | 68 |

### Arm 7b

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a7b-sglang-s24: SGLang 0.5.22: priority scheduling with preemption, discriminating scenario |  | 24 | 3,678 | 3,577 | 0.29 | 5,367 | 12,051 | 17,937 | 0 of 50 | 70,580 / 101,569 | 6,451 | — | 0 / 0 / 0 | 2,767 | 67 |
| disc-a7b-sglang-hicache-s24 (r1) |  | 24 | 4,693 | 4,152 | 0.40 | 5,257 | 9,394 | 17,868 | 0 of 50 | 47,773 / 80,772 | 6,242 | — | 0 / 0 / 0 | 2,767 | 67 |
| disc-a7b-sglang-hicache-s24 (r2) |  | 24 | 4,723 | 4,318 | 0.41 | 4,705 | 12,663 | 19,230 | 0 of 50 | 45,937 / 86,878 | 8,555 | — | 0 / 0 / 0 | 2,767 | 68 |
| disc-a7b-sglang-hicache-s24 (r3) |  | 24 | 4,785 | 4,445 | 0.41 | 5,236 | 16,702 | 18,063 | 0 of 50 | 44,554 / 82,135 | 8,553 | — | 0 / 0 / 0 | 2,767 | 65 |
| disc-a7b-sglang-hicache-s24 (r4) |  | 24 | 4,598 | 4,330 | 0.39 | 4,467 | 11,948 | 18,888 | 0 of 50 | 50,205 / 83,373 | 8,012 | — | 0 / 0 / 0 | 2,767 | 66 |
| disc-a7b-sglang-hicache-s24 (w2c) |  | 24 | 4,908 | 4,590 | 0.43 | 4,274 | 10,497 | 13,193 | 0 of 50 | 42,680 / 81,674 | 5,000 | — | 0 / 0 / 0 | 2,767 | 66 |
| disc-a7b-sglang-hicache-s24: SGLang 0.5.22 with HiCache 4x (write-through), discriminating scenario; median (range) of 5 runs | • | 24 | 4,723 (4,598–4,908) | 4,330 (4,152–4,590) | | 4,705 (4,274–5,257) | 11,948 (9,394–16,702) | 18,063 (13,193–19,230) | 0 (0–0) | 45,937 (42,680–50,205) / 82,135 (80,772–86,878) | 8,012 (5,000–8,555) | | | | |

### Arm 8a

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a8a-flow-s24: frozen box + flow control (llm-d, emulated), ceilings 6/6/6, discriminating scenario |  | 24 | 4,006 | 3,684 | 0.37 | 4,221 | 10,992 | 11,277 | 0 of 50 | 40,530 / 80,817 | 8,172 | 29 | 60 / 0 / 0 | 2,722 | 69 |

### Arm 8b

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a8b-smg-s24: frozen box behind SMG 1.13.0, as shipped, discriminating scenario |  | 24 | 3,557 | 3,446 | 0.29 | 3,511 | 11,050 | 13,839 | 0 of 50 | 69,582 / 105,041 | 8,783 | 8 | 0 / 0 / 0 | 2,722 | 69 |

### Arm 8c

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a8c-resubmit-s24 (r1) |  | 24 | 3,707 | 3,432 | 0.39 | 2,024 | 2,625 | 2,710 | 0 of 50 | 30,689 / 160,309 | 415 | 6 | 0 / 0 / 1,103 | 2,722 | 69 |
| disc-a8c-resubmit-s24 (r2) |  | 24 | 3,462 | 3,593 | 0.38 | 1,921 | 2,393 | 2,633 | 0 of 50 | 26,910 / 199,039 | 343 | 5 | 0 / 0 / 1,126 | 2,722 | 68 |
| disc-a8c-resubmit-s24 (r3) |  | 24 | 3,630 | 3,691 | 0.37 | 1,880 | 2,110 | 2,214 | 0 of 50 | 34,626 / 157,449 | 334 | 6 | 0 / 0 / 1,130 | 2,722 | 68 |
| disc-a8c-resubmit-s24 (r4) |  | 24 | 3,398 | 3,346 | 0.34 | 2,027 | 2,663 | 3,038 | 0 of 50 | 40,004 / 175,337 | 387 | 7 | 0 / 0 / 1,105 | 2,722 | 68 |
| disc-a8c-resubmit-s24 (w2c) |  | 24 | 3,901 | 3,670 | 0.41 | 1,960 | 2,591 | 2,684 | 0 of 50 | 28,103 / 140,442 | 374 | 8 | 0 / 0 / 1,080 | 2,722 | 68 |
| disc-a8c-resubmit-s24: frozen box + preempt-resubmit before the first byte, discriminating scenario; median (range) of 5 runs |  | 24 | 3,630 (3,398–3,901) | 3,593 (3,346–3,691) | | 1,960 (1,880–2,027) | 2,591 (2,110–2,663) | 2,684 (2,214–3,038) | 0 (0–0) | 30,689 (26,910–40,004) / 160,309 (140,442–199,039) | 374 (334–415) | | | | |

### Arm 9

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-s24 (a9) |  | 24 | 3,411 | 3,308 | 0.27 | 1,836 | 2,075 | 2,172 | 0 of 50 | 75,081 / 105,287 | 290 | 0 | 0 / 0 / 0 | 2,730 | 68 |
| disc-a9-s24 (r1) |  | 24 | 3,394 | 3,355 | 0.27 | 1,825 | 2,134 | 2,274 | 0 of 50 | 76,651 / 98,076 | 346 | 0 | 0 / 0 / 0 | 2,730 | 70 |
| disc-a9-s24 (r2) |  | 24 | 3,387 | 3,402 | 0.27 | 1,776 | 2,057 | 2,105 | 0 of 50 | 74,993 / 98,098 | 282 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-s24 (r3) |  | 24 | 3,438 | 3,381 | 0.27 | 1,809 | 2,003 | 2,139 | 0 of 50 | 75,420 / 103,312 | 275 | 0 | 0 / 0 / 0 | 2,722 | 68 |
| disc-a9-s24 (r4) |  | 24 | 3,427 | 3,352 | 0.27 | 1,834 | 1,980 | 2,078 | 0 of 50 | 73,092 / 104,642 | 306 | 0 | 0 / 0 / 0 | 2,722 | 68 |
| disc-a9-s24: Leaseholder: timed headroom, critical-window throttle, degrade; make-room; median (range) of 5 runs |  | 24 | 3,411 (3,387–3,438) | 3,355 (3,308–3,402) | | 1,825 (1,776–1,836) | 2,057 (1,980–2,134) | 2,139 (2,078–2,274) | 0 (0–0) | 75,081 (73,092–76,651) / 103,312 (98,076–105,287) | 290 (275–346) | | | | |

### Arm 9-budget0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-budget0-s24 (a9) |  | 24 | 3,443 | 3,406 | 0.27 | 1,823 | 2,166 | 2,263 | 0 of 50 | 72,857 / 100,661 | 317 | 0 | 0 / 0 / 0 | 2,730 | 68 |
| disc-a9-budget0-s24 (r1) |  | 24 | 3,387 | 3,420 | 0.28 | 1,848 | 2,094 | 2,223 | 0 of 50 | 72,247 / 104,432 | 284 | 0 | 0 / 0 / 0 | 2,730 | 70 |
| disc-a9-budget0-s24 (r2) |  | 24 | 3,370 | 3,307 | 0.27 | 1,784 | 2,109 | 2,192 | 0 of 50 | 74,490 / 106,918 | 304 | 0 | 0 / 0 / 0 | 2,722 | 71 |
| disc-a9-budget0-s24: Leaseholder, critical-window budget 0 ms; median (range) of 3 runs |  | 24 | 3,387 (3,370–3,443) | 3,406 (3,307–3,420) | | 1,823 (1,784–1,848) | 2,109 (2,094–2,166) | 2,223 (2,192–2,263) | 0 (0–0) | 72,857 (72,247–74,490) / 104,432 (100,661–106,918) | 304 (284–317) | | | | |

### Arm 9-budget500

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-budget500-s24 (a9) |  | 24 | 3,265 | 3,354 | 0.26 | 1,847 | 2,060 | 2,112 | 0 of 50 | 76,107 / 101,836 | 321 | 0 | 0 / 0 / 0 | 2,730 | 68 |
| disc-a9-budget500-s24 (r1) |  | 24 | 3,370 | 3,380 | 0.27 | 1,831 | 2,080 | 2,294 | 0 of 50 | 75,822 / 104,106 | 287 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-budget500-s24 (r2) |  | 24 | 3,325 | 3,308 | 0.26 | 1,780 | 1,990 | 2,108 | 0 of 50 | 77,491 / 109,576 | 301 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-budget500-s24: Leaseholder, critical-window budget 500 ms; median (range) of 3 runs |  | 24 | 3,325 (3,265–3,370) | 3,354 (3,308–3,380) | | 1,831 (1,780–1,847) | 2,060 (1,990–2,080) | 2,112 (2,108–2,294) | 0 (0–0) | 76,107 (75,822–77,491) / 104,106 (101,836–109,576) | 301 (287–321) | | | | |

### Arm 9-engine-only

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-engine-only-s24 (a9) |  | 24 | 3,386 | 3,366 | 0.27 | 1,931 | 4,034 | 5,178 | 0 of 50 | 74,821 / 97,479 | 295 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a9-engine-only-s24 (r1) |  | 24 | 3,418 | 3,409 | 0.27 | 1,949 | 2,984 | 5,095 | 0 of 50 | 75,410 / 102,069 | 319 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-engine-only-s24 (r2) |  | 24 | 3,477 | 3,406 | 0.28 | 2,189 | 4,062 | 5,823 | 0 of 50 | 74,870 / 104,157 | 304 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a9-engine-only-s24: Make-room alone, without the gateway policy; median (range) of 3 runs |  | 24 | 3,418 (3,386–3,477) | 3,406 (3,366–3,409) | | 1,949 (1,931–2,189) | 4,034 (2,984–4,062) | 5,178 (5,095–5,823) | 0 (0–0) | 74,870 (74,821–75,410) / 102,069 (97,479–104,157) | 304 (295–319) | | | | |

### Arm 9-gateway-only

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-gateway-only-s24 (a9) |  | 24 | 3,690 | 3,556 | 0.30 | 4,452 | 11,958 | 13,557 | 0 of 50 | 64,809 / 98,579 | 9,121 | 29 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a9-gateway-only-s24 (r1) |  | 24 | 3,679 | 3,633 | 0.30 | 4,601 | 10,028 | 11,340 | 0 of 50 | 65,646 / 103,011 | 7,607 | 40 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-gateway-only-s24 (r2) |  | 24 | 3,697 | 3,653 | 0.30 | 4,396 | 10,993 | 11,990 | 0 of 50 | 67,249 / 98,498 | 8,673 | 29 | 0 / 0 / 0 | 2,722 | 68 |
| disc-a9-gateway-only-s24: Leaseholder's gateway policy on the frozen engine; median (range) of 3 runs |  | 24 | 3,690 (3,679–3,697) | 3,633 (3,556–3,653) | | 4,452 (4,396–4,601) | 10,993 (10,028–11,958) | 11,990 (11,340–13,557) | 0 (0–0) | 65,646 (64,809–67,249) / 98,579 (98,498–103,011) | 8,673 (7,607–9,121) | | | | |

### Arm 9-no-budget-cap

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-no-budget-cap-s24 (a9) |  | 24 | 3,582 | 3,532 | 0.29 | 3,069 | 5,295 | 6,841 | 0 of 50 | 71,998 / 103,693 | 2,835 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-no-budget-cap-s24 (r1) |  | 24 | 3,522 | 3,465 | 0.29 | 2,635 | 5,063 | 6,610 | 0 of 50 | 74,402 / 107,519 | 2,900 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-no-budget-cap-s24 (r2) |  | 24 | 3,445 | 3,448 | 0.28 | 2,950 | 6,115 | 7,734 | 0 of 50 | 71,374 / 97,482 | 3,970 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-no-budget-cap-s24: Leaseholder, make-room without the prefill budget cap; median (range) of 3 runs |  | 24 | 3,522 (3,445–3,582) | 3,465 (3,448–3,532) | | 2,950 (2,635–3,069) | 5,295 (5,063–6,115) | 6,841 (6,610–7,734) | 0 (0–0) | 71,998 (71,374–74,402) / 103,693 (97,482–107,519) | 2,900 (2,835–3,970) | | | | |

### Arm 9-no-degrade

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-no-degrade-s24 (a9) |  | 24 | 3,506 | 3,454 | 0.28 | 2,126 | 4,079 | 6,299 | 0 of 50 | 72,423 / 99,908 | 279 | 0 | 0 / 0 / 0 | 2,730 | 68 |
| disc-a9-no-degrade-s24 (r1) |  | 24 | 3,456 | 3,307 | 0.28 | 1,968 | 2,890 | 5,200 | 0 of 50 | 73,893 / 105,202 | 354 | 0 | 0 / 0 / 0 | 2,722 | 71 |
| disc-a9-no-degrade-s24 (r2) |  | 24 | 3,489 | 3,336 | 0.28 | 2,040 | 3,837 | 6,513 | 0 of 50 | 73,248 / 97,440 | 326 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a9-no-degrade-s24: Leaseholder without degrade; median (range) of 3 runs |  | 24 | 3,489 (3,456–3,506) | 3,336 (3,307–3,454) | | 2,040 (1,968–2,126) | 3,837 (2,890–4,079) | 6,299 (5,200–6,513) | 0 (0–0) | 73,248 (72,423–73,893) / 99,908 (97,440–105,202) | 326 (279–354) | | | | |

### Arm 9-no-events

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-no-events-s24 (a9) |  | 24 | 3,393 | 3,402 | 0.27 | 1,850 | 2,048 | 2,199 | 0 of 50 | 75,269 / 98,194 | 301 | 0 | 0 / 0 / 0 | 2,722 | 68 |
| disc-a9-no-events-s24 (r1) |  | 24 | 3,424 | 3,307 | 0.27 | 1,823 | 2,005 | 2,105 | 0 of 50 | 74,603 / 105,803 | 295 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-no-events-s24 (r2) |  | 24 | 3,338 | 3,326 | 0.27 | 1,817 | 2,036 | 2,124 | 0 of 50 | 76,923 / 102,213 | 292 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a9-no-events-s24: Leaseholder without lifecycle events; median (range) of 3 runs |  | 24 | 3,393 (3,338–3,424) | 3,326 (3,307–3,402) | | 1,823 (1,817–1,850) | 2,036 (2,005–2,048) | 2,124 (2,105–2,199) | 0 (0–0) | 75,269 (74,603–76,923) / 102,213 (98,194–105,803) | 295 (292–301) | | | | |

### Arm 9-no-headroom

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-no-headroom-s24 (a9) |  | 24 | 3,389 | 3,339 | 0.27 | 1,837 | 2,071 | 2,211 | 0 of 50 | 74,637 / 97,899 | 299 | 0 | 0 / 0 / 0 | 2,730 | 68 |
| disc-a9-no-headroom-s24 (r1) |  | 24 | 3,380 | 3,326 | 0.27 | 1,832 | 2,033 | 2,181 | 0 of 50 | 75,921 / 102,263 | 300 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-no-headroom-s24 (r2) |  | 24 | 3,292 | 3,362 | 0.27 | 1,815 | 1,947 | 2,067 | 0 of 50 | 76,692 / 107,828 | 302 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a9-no-headroom-s24: Leaseholder without the turn-imminent headroom; median (range) of 3 runs |  | 24 | 3,380 (3,292–3,389) | 3,339 (3,326–3,362) | | 1,832 (1,815–1,837) | 2,033 (1,947–2,071) | 2,181 (2,067–2,211) | 0 (0–0) | 75,921 (74,637–76,692) / 102,263 (97,899–107,828) | 300 (299–302) | | | | |

### Arm 9-no-kv-makeroom

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-no-kv-makeroom-s24 (a9) |  | 24 | 3,290 | 3,321 | 0.26 | 1,827 | 2,037 | 2,153 | 0 of 50 | 76,773 / 102,147 | 306 | 0 | 0 / 0 / 0 | 2,726 | 69 |
| disc-a9-no-kv-makeroom-s24 (r1) |  | 24 | 3,360 | 3,326 | 0.27 | 1,814 | 1,990 | 2,164 | 0 of 50 | 75,501 / 102,279 | 317 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a9-no-kv-makeroom-s24 (r2) |  | 24 | 3,377 | 3,327 | 0.27 | 1,828 | 2,039 | 2,154 | 0 of 50 | 75,973 / 98,643 | 274 | 0 | 0 / 0 / 0 | 2,722 | 69 |
| disc-a9-no-kv-makeroom-s24: Leaseholder, make-room without KV make-room; median (range) of 3 runs |  | 24 | 3,360 (3,290–3,377) | 3,326 (3,321–3,327) | | 1,827 (1,814–1,828) | 2,037 (1,990–2,039) | 2,154 (2,153–2,164) | 0 (0–0) | 75,973 (75,501–76,773) / 102,147 (98,643–102,279) | 306 (274–317) | | | | |

### Arm 9-no-reserve

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-no-reserve-s24 (a9) |  | 24 | 3,419 | 3,406 | 0.27 | 1,884 | 2,115 | 2,214 | 0 of 50 | 73,269 / 106,476 | 349 | 39 | 0 / 0 / 0 | 2,730 | 69 |
| disc-a9-no-reserve-s24 (r1) |  | 24 | 3,422 | 3,485 | 0.28 | 1,876 | 2,116 | 2,191 | 0 of 50 | 72,387 / 95,339 | 391 | 38 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-no-reserve-s24 (r2) |  | 24 | 3,508 | 3,438 | 0.28 | 1,897 | 2,155 | 2,275 | 0 of 50 | 72,389 / 95,741 | 373 | 27 | 0 / 0 / 0 | 2,722 | 68 |
| disc-a9-no-reserve-s24: Leaseholder, make-room without the realtime KV reserve; median (range) of 3 runs |  | 24 | 3,422 (3,419–3,508) | 3,438 (3,406–3,485) | | 1,884 (1,876–1,897) | 2,116 (2,115–2,155) | 2,214 (2,191–2,275) | 0 (0–0) | 72,389 (72,387–73,269) / 95,741 (95,339–106,476) | 373 (349–391) | | | | |

### Arm 9-no-throttle

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-no-throttle-s24 (a9) |  | 24 | 3,378 | 3,424 | 0.26 | 1,817 | 2,049 | 2,140 | 0 of 50 | 76,041 / 97,414 | 307 | 0 | 0 / 0 / 0 | 2,730 | 68 |
| disc-a9-no-throttle-s24 (r1) |  | 24 | 3,390 | 3,333 | 0.27 | 1,832 | 1,995 | 2,197 | 0 of 50 | 77,208 / 104,776 | 316 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-no-throttle-s24 (r2) |  | 24 | 3,420 | 3,357 | 0.27 | 1,814 | 2,081 | 2,178 | 0 of 50 | 77,361 / 108,519 | 331 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-no-throttle-s24: Leaseholder without the critical-window throttle; median (range) of 3 runs |  | 24 | 3,390 (3,378–3,420) | 3,357 (3,333–3,424) | | 1,817 (1,814–1,832) | 2,049 (1,995–2,081) | 2,178 (2,140–2,197) | 0 (0–0) | 77,208 (76,041–77,361) / 104,776 (97,414–108,519) | 316 (307–331) | | | | |

### Arm 9-pr-a

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-pr-a-s24 (upstream) |  | 24 | 3,576 | 3,572 | 0.29 | 3,041 | 5,612 | 8,932 | 0 of 50 | 64,966 / 96,675 | 2,517 | 37 | 0 / 0 / 0 | 2,722 | 68 |
| disc-pr-a-s24 (upstream-r1) |  | 24 | 3,660 | 3,628 | 0.29 | 3,394 | 6,063 | 8,110 | 0 of 50 | 67,310 / 98,091 | 2,785 | 40 | 0 / 0 / 0 | 2,722 | 68 |
| disc-pr-a-s24 (upstream-r2) |  | 24 | 3,633 | 3,514 | 0.30 | 3,246 | 5,773 | 6,177 | 0 of 50 | 66,198 / 96,205 | 3,172 | 37 | 0 / 0 / 0 | 2,722 | 68 |
| disc-pr-a-s24: vLLM #63350: make-room's KV preemption alone, no gateway policy; median (range) of 3 runs |  | 24 | 3,633 (3,576–3,660) | 3,572 (3,514–3,628) | | 3,246 (3,041–3,394) | 5,773 (5,612–6,063) | 8,110 (6,177–8,932) | 0 (0–0) | 66,198 (64,966–67,310) / 96,675 (96,205–98,091) | 2,785 (2,517–3,172) | | | | |

### Arm 9-pr-ab

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-pr-ab-s24 (upstream) |  | 24 | 3,620 | 3,485 | 0.30 | 2,213 | 4,772 | 5,274 | 0 of 50 | 69,819 / 96,355 | 375 | 32 | 0 / 0 / 0 | 2,730 | 68 |
| disc-pr-ab-s24 (upstream-r1) |  | 24 | 3,519 | 3,540 | 0.29 | 1,986 | 3,229 | 4,135 | 0 of 50 | 71,343 / 100,148 | 336 | 41 | 0 / 0 / 0 | 2,730 | 68 |
| disc-pr-ab-s24 (upstream-r2) |  | 24 | 3,568 | 3,541 | 0.29 | 1,976 | 3,220 | 3,729 | 0 of 50 | 72,326 / 101,116 | 338 | 38 | 0 / 0 / 0 | 2,722 | 68 |
| disc-pr-ab-s24: vLLM #63350 and #63351: KV preemption and the budget cap, no gateway policy; median (range) of 3 runs |  | 24 | 3,568 (3,519–3,620) | 3,540 (3,485–3,541) | | 1,986 (1,976–2,213) | 3,229 (3,220–4,772) | 4,135 (3,729–5,274) | 0 (0–0) | 71,343 (69,819–72,326) / 100,148 (96,355–101,116) | 338 (336–375) | | | | |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-pr-ab48-s24 (upstream) |  | 24 | 3,428 | 3,362 | 0.28 | 1,978 | 2,168 | 2,241 | 0 of 50 | 72,284 / 96,050 | 372 | 37 | 0 / 0 / 0 | 2,730 | 68 |
| disc-pr-ab48-s24 (upstream-r1) |  | 24 | 3,493 | 3,362 | 0.28 | 1,943 | 2,286 | 2,341 | 0 of 50 | 71,729 / 97,408 | 357 | 38 | 0 / 0 / 0 | 2,730 | 68 |
| disc-pr-ab48-s24 (upstream-r2) |  | 24 | 3,519 | 3,355 | 0.28 | 1,922 | 2,193 | 2,281 | 0 of 50 | 70,216 / 96,986 | 377 | 43 | 0 / 0 / 0 | 2,730 | 69 |
| disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy; median (range) of 3 runs |  | 24 | 3,493 (3,428–3,519) | 3,362 (3,355–3,362) | | 1,943 (1,922–1,978) | 2,193 (2,168–2,286) | 2,281 (2,241–2,341) | 0 (0–0) | 71,729 (70,216–72,284) / 96,986 (96,050–97,408) | 372 (357–377) | | | | |

### Arm 9-trigger-request

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-trigger-request-s24 (a9) |  | 24 | 3,386 | 3,339 | 0.27 | 1,836 | 2,008 | 2,128 | 0 of 50 | 72,595 / 98,697 | 294 | 0 | 0 / 0 / 0 | 2,730 | 68 |
| disc-a9-trigger-request-s24 (r1) |  | 24 | 3,401 | 3,402 | 0.27 | 1,806 | 2,165 | 2,209 | 0 of 50 | 75,912 / 98,193 | 356 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-trigger-request-s24 (r2) |  | 24 | 3,345 | 3,307 | 0.27 | 1,852 | 2,124 | 2,192 | 0 of 50 | 74,750 / 106,011 | 318 | 0 | 0 / 0 / 0 | 2,722 | 70 |
| disc-a9-trigger-request-s24: Leaseholder, headroom armed by the realtime request; median (range) of 3 runs |  | 24 | 3,386 (3,345–3,401) | 3,339 (3,307–3,402) | | 1,836 (1,806–1,852) | 2,124 (2,008–2,165) | 2,192 (2,128–2,209) | 0 (0–0) | 74,750 (72,595–75,912) / 98,697 (98,193–106,011) | 318 (294–356) | | | | |

### Arm 9-window48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| disc-a9-window48-s24 (a9-window) |  | 24 | 3,402 | 3,352 | 0.28 | 1,837 | 2,098 | 2,207 | 0 of 50 | 75,978 / 96,319 | 327 | 0 | 0 / 0 / 0 | 2,722 | 68 |
| disc-a9-window48-s24 (upstream-r1) |  | 24 | 3,422 | 3,343 | 0.28 | 1,942 | 2,205 | 2,310 | 0 of 50 | 75,867 / 96,587 | 277 | 0 | 0 / 0 / 0 | 2,722 | 68 |
| disc-a9-window48-s24 (upstream-r2) |  | 24 | 3,362 | 3,307 | 0.27 | 1,866 | 2,289 | 2,506 | 0 of 50 | 74,942 / 106,072 | 318 | 0 | 0 / 0 / 0 | 2,730 | 68 |
| disc-a9-window48-s24: Leaseholder without degrade, a 48-token first-tokens window; median (range) of 3 runs |  | 24 | 3,402 (3,362–3,422) | 3,343 (3,307–3,352) | | 1,866 (1,837–1,942) | 2,205 (2,098–2,289) | 2,310 (2,207–2,506) | 0 (0–0) | 75,867 (74,942–75,978) / 96,587 (96,319–106,072) | 318 (277–327) | | | | |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver. 30 points without the worker's turn lines are left out.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| a1-voice (r3) | 744 / 806 | 748 / 810 | 78 / 112 | 310 / 427 | 312 / 430 | 196 / 248 | 1,272 / 1,442 | 241 / 0 / 0 |
| a1-voice (r4) | 748 / 802 | 752 / 805 | 77 / 95 | 284 / 366 | 286 / 380 | 182 / 242 | 1,234 / 1,367 | 241 / 0 / 0 |
| a1-voice (s1) | 748 / 799 | 753 / 805 | 80 / 97 | 313 / 407 | 314 / 410 | 190 / 240 | 1,272 / 1,378 | 241 / 0 / 0 |
| a1-voice (s2) | 746 / 791 | 751 / 797 | 79 / 96 | 274 / 399 | 280 / 405 | 194 / 240 | 1,237 / 1,373 | 240 / 0 / 0 |
| a4-b512-t256-s16 (s1) | 767 / 849 | 772 / 855 | 92 / 228 | 280 / 825 | 293 / 843 | 215 / 275 | 1,299 / 1,841 | 240 / 0 / 0 |
| a4-b512-t256-s16 (s2) | 768 / 846 | 774 / 851 | 94 / 217 | 320 / 1,160 | 328 / 1,164 | 225 / 285 | 1,327 / 2,191 | 240 / 0 / 0 |
| a7b-sglang-hicache-s8 | 756 / 847 | 761 / 854 | 100 / 144 | 291 / 441 | 295 / 443 | 201 / 244 | 1,266 / 1,483 | 241 / 0 / 0 |
| a7b-sglang-hicache-s16 | 798 / 886 | 803 / 892 | 101 / 286 | 322 / 825 | 332 / 862 | 221 / 336 | 1,359 / 1,943 | 240 / 0 / 0 |
| a9-s8 | 786 / 831 | 792 / 836 | 90 / 116 | 347 / 604 | 349 / 607 | 230 / 299 | 1,370 / 1,721 | 241 / 0 / 0 |
| a9-s16 (a9-standard) | 794 / 835 | 796 / 841 | 92 / 239 | 333 / 811 | 349 / 814 | 203 / 271 | 1,341 / 1,892 | 241 / 0 / 0 |
| a9-s16 (s1) | 796 / 855 | 802 / 860 | 94 / 232 | 345 / 969 | 350 / 976 | 213 / 261 | 1,336 / 2,049 | 240 / 0 / 0 |
| a9-s16 (s2) | 796 / 837 | 802 / 842 | 97 / 256 | 332 / 672 | 337 / 678 | 222 / 297 | 1,348 / 1,719 | 240 / 0 / 0 |
| a9-s24 | 811 / 843 | 816 / 849 | 188 / 393 | 417 / 997 | 433 / 999 | 212 / 257 | 1,458 / 2,047 | 240 / 0 / 0 |
| disc-frozen-s24 (r1) | 816 / 852 | 821 / 860 | 2,027 / 7,800 | 2,921 / 8,610 | 3,019 / 8,687 | 226 / 316 | 4,055 / 9,800 | 240 / 0 / 0 |
| disc-frozen-s24 (r2) | 813 / 841 | 816 / 845 | 2,401 / 6,899 | 3,280 / 7,672 | 3,457 / 7,688 | 231 / 376 | 4,531 / 8,852 | 240 / 0 / 0 |
| disc-frozen-s24 (r3) | 808 / 851 | 811 / 857 | 1,679 / 7,860 | 3,069 / 9,275 | 3,041 / 9,351 | 224 / 360 | 4,106 / 10,425 | 240 / 0 / 0 |
| disc-frozen-s24 (r4) | 815 / 840 | 818 / 845 | 2,432 / 7,920 | 3,424 / 8,812 | 3,601 / 8,818 | 234 / 289 | 4,675 / 9,852 | 240 / 0 / 0 |
| disc-frozen-s24 (w2c) | 820 / 850 | 825 / 857 | 1,572 / 8,585 | 2,772 / 9,376 | 2,778 / 9,397 | 264 / 403 | 3,888 / 10,485 | 240 / 0 / 0 |
| disc-a4-b1024-t0-s24 (r1) | 819 / 868 | 824 / 873 | 1,999 / 5,834 | 2,819 / 6,693 | 2,785 / 6,722 | 234 / 307 | 3,830 / 7,763 | 240 / 0 / 0 |
| disc-a4-b1024-t0-s24 (r2) | 816 / 858 | 821 / 862 | 2,338 / 5,962 | 2,951 / 7,017 | 2,915 / 7,061 | 221 / 310 | 3,961 / 8,169 | 240 / 0 / 0 |
| disc-a4-b1024-t0-s24 (r3) | 818 / 845 | 824 / 849 | 2,839 / 7,074 | 3,458 / 8,155 | 3,424 / 8,171 | 231 / 287 | 4,464 / 9,239 | 240 / 0 / 0 |
| disc-a4-b1024-t0-s24 (r4) | 817 / 846 | 822 / 849 | 2,259 / 7,227 | 2,884 / 7,840 | 2,840 / 7,853 | 236 / 306 | 3,874 / 8,938 | 240 / 0 / 0 |
| disc-a4-b1024-t0-s24 (w2c) | 813 / 840 | 817 / 845 | 2,289 / 6,554 | 3,248 / 7,379 | 3,601 / 7,425 | 239 / 336 | 4,691 / 8,490 | 240 / 0 / 0 |
| disc-a5-watermark10-s24 | 814 / 860 | 821 / 866 | 2,366 / 8,231 | 3,425 / 9,238 | 3,488 / 9,302 | 228 / 332 | 4,543 / 10,347 | 240 / 0 / 0 |
| disc-a5-queued8192-s24 | 764 / 835 | 778 / 845 | 85 / 156 | 421 / 612 | 424 / 615 | 224 / 274 | 1,483 / 1,691 | 32 / 0 / 0 |
| disc-a6-reserve2048-s24 (r1) | 818 / 929 | 823 / 933 | 649 / 3,220 | 1,702 / 4,315 | 1,714 / 4,338 | 240 / 329 | 2,831 / 5,381 | 240 / 0 / 0 |
| disc-a6-reserve2048-s24 (r2) | 812 / 876 | 817 / 880 | 688 / 2,432 | 1,814 / 3,593 | 1,817 / 3,596 | 233 / 312 | 2,856 / 4,669 | 240 / 0 / 0 |
| disc-a6-reserve2048-s24 (r3) | 811 / 850 | 818 / 857 | 585 / 2,964 | 1,400 / 3,907 | 1,413 / 3,936 | 218 / 263 | 2,472 / 5,001 | 240 / 0 / 0 |
| disc-a6-reserve2048-s24 (r4) | 817 / 846 | 823 / 851 | 634 / 3,326 | 1,694 / 4,339 | 1,738 / 4,354 | 234 / 292 | 2,784 / 5,421 | 240 / 0 / 0 |
| disc-a6-reserve2048-s24 (w2c) | 815 / 921 | 820 / 926 | 734 / 3,665 | 1,897 / 4,285 | 1,937 / 4,298 | 241 / 331 | 3,052 / 5,386 | 240 / 0 / 0 |
| disc-a6-cap2-s24 | 817 / 866 | 822 / 871 | 1,002 / 6,347 | 1,927 / 7,156 | 1,919 / 7,188 | 254 / 374 | 2,994 / 8,260 | 240 / 0 / 0 |
| disc-a6-cap8-s24 | 816 / 856 | 820 / 862 | 3,118 / 7,524 | 4,293 / 8,437 | 4,255 / 8,447 | 240 / 308 | 5,317 / 9,510 | 240 / 0 / 0 |
| disc-a7a-pr54530-s24 | 816 / 850 | 820 / 855 | 1,639 / 7,992 | 2,934 / 9,463 | 3,084 / 9,485 | 242 / 324 | 4,147 / 10,563 | 240 / 0 / 0 |
| disc-a7b-sglang-s24 | 810 / 833 | 815 / 838 | 1,979 / 6,513 | 4,022 / 10,541 | 4,095 / 10,617 | 217 / 355 | 5,104 / 11,661 | 240 / 0 / 0 |
| disc-a7b-sglang-hicache-s24 (r1) | 808 / 844 | 813 / 849 | 1,551 / 6,257 | 3,892 / 8,041 | 3,898 / 8,096 | 208 / 311 | 4,934 / 9,098 | 240 / 0 / 0 |
| disc-a7b-sglang-hicache-s24 (r2) | 805 / 830 | 810 / 835 | 2,227 / 8,582 | 3,375 / 11,323 | 3,377 / 11,356 | 222 / 327 | 4,397 / 12,380 | 240 / 0 / 0 |
| disc-a7b-sglang-hicache-s24 (r3) | 806 / 838 | 810 / 844 | 1,538 / 8,577 | 3,854 / 15,401 | 3,856 / 15,404 | 216 / 317 | 4,902 / 16,490 | 241 / 0 / 0 |
| disc-a7b-sglang-hicache-s24 (r4) | 807 / 839 | 813 / 844 | 979 / 8,073 | 3,034 / 10,562 | 2,860 / 9,690 | 214 / 317 | 3,867 / 10,711 | 240 / 0 / 0 |
| disc-a7b-sglang-hicache-s24 (w2c) | 811 / 894 | 815 / 898 | 1,565 / 5,033 | 2,818 / 9,110 | 2,833 / 9,129 | 236 / 376 | 3,924 / 10,192 | 240 / 0 / 0 |
| disc-a8a-flow-s24 | 814 / 839 | 821 / 846 | 1,414 / 8,209 | 2,780 / 9,646 | 2,798 / 9,653 | 236 / 329 | 3,859 / 10,712 | 240 / 0 / 0 |
| disc-a8b-smg-s24 | 810 / 840 | 814 / 845 | 884 / 8,798 | 2,059 / 9,569 | 2,167 / 9,685 | 230 / 328 | 3,201 / 10,698 | 240 / 0 / 0 |
| disc-a8c-resubmit-s24 (r1) | 822 / 842 | 828 / 848 | 291 / 429 | 589 / 1,135 | 597 / 1,138 | 237 / 301 | 1,677 / 2,213 | 240 / 0 / 0 |
| disc-a8c-resubmit-s24 (r2) | 810 / 854 | 813 / 860 | 261 / 392 | 528 / 1,051 | 532 / 1,054 | 212 / 281 | 1,572 / 2,101 | 240 / 0 / 0 |
| disc-a8c-resubmit-s24 (r3) | 816 / 851 | 820 / 857 | 256 / 371 | 491 / 702 | 497 / 706 | 203 / 268 | 1,524 / 1,749 | 240 / 0 / 0 |
| disc-a8c-resubmit-s24 (r4) | 819 / 916 | 824 / 920 | 283 / 409 | 598 / 1,241 | 601 / 1,244 | 224 / 317 | 1,685 / 2,373 | 240 / 0 / 0 |
| disc-a8c-resubmit-s24 (w2c) | 815 / 861 | 820 / 863 | 251 / 394 | 551 / 1,220 | 560 / 1,224 | 206 / 297 | 1,598 / 2,277 | 240 / 0 / 0 |
| disc-a9-s24 (a9) | 817 / 850 | 821 / 854 | 226 / 308 | 434 / 574 | 435 / 576 | 215 / 284 | 1,478 / 1,637 | 240 / 0 / 0 |
| disc-a9-s24 (r1) | 816 / 877 | 822 / 882 | 228 / 383 | 417 / 678 | 424 / 680 | 213 / 302 | 1,483 / 1,792 | 240 / 0 / 0 |
| disc-a9-s24 (r2) | 812 / 855 | 817 / 858 | 220 / 310 | 390 / 581 | 394 / 605 | 212 / 254 | 1,430 / 1,666 | 240 / 0 / 0 |
| disc-a9-s24 (r3) | 813 / 857 | 818 / 862 | 205 / 299 | 415 / 545 | 423 / 547 | 212 / 262 | 1,468 / 1,634 | 240 / 0 / 0 |
| disc-a9-s24 (r4) | 817 / 863 | 822 / 864 | 222 / 335 | 414 / 540 | 418 / 543 | 212 / 260 | 1,468 / 1,608 | 240 / 0 / 0 |
| disc-a9-budget0-s24 (a9) | 813 / 869 | 818 / 875 | 201 / 329 | 416 / 619 | 418 / 623 | 215 / 280 | 1,453 / 1,702 | 240 / 0 / 0 |
| disc-a9-budget0-s24 (r1) | 813 / 871 | 818 / 876 | 227 / 312 | 445 / 580 | 449 / 588 | 216 / 270 | 1,503 / 1,691 | 240 / 0 / 0 |
| disc-a9-budget0-s24 (r2) | 811 / 869 | 816 / 875 | 213 / 326 | 401 / 590 | 406 / 594 | 208 / 248 | 1,451 / 1,615 | 240 / 0 / 0 |
| disc-a9-budget500-s24 (a9) | 814 / 870 | 817 / 876 | 225 / 343 | 423 / 564 | 430 / 566 | 212 / 303 | 1,482 / 1,632 | 240 / 0 / 0 |
| disc-a9-budget500-s24 (r1) | 808 / 858 | 814 / 863 | 237 / 321 | 452 / 612 | 455 / 614 | 206 / 246 | 1,490 / 1,675 | 240 / 0 / 0 |
| disc-a9-budget500-s24 (r2) | 808 / 844 | 813 / 848 | 228 / 318 | 393 / 531 | 404 / 533 | 208 / 239 | 1,427 / 1,567 | 240 / 0 / 0 |
| disc-a9-engine-only-s24 (a9) | 812 / 842 | 817 / 846 | 241 / 322 | 467 / 2,631 | 491 / 2,637 | 243 / 295 | 1,556 / 3,711 | 240 / 0 / 0 |
| disc-a9-engine-only-s24 (r1) | 817 / 862 | 821 / 868 | 250 / 353 | 510 / 1,564 | 525 / 1,626 | 220 / 284 | 1,576 / 2,727 | 240 / 0 / 0 |
| disc-a9-engine-only-s24 (r2) | 814 / 929 | 818 / 935 | 229 / 331 | 713 / 2,672 | 753 / 2,695 | 238 / 315 | 1,887 / 3,744 | 240 / 0 / 0 |
| disc-a9-gateway-only-s24 (a9) | 811 / 887 | 816 / 892 | 1,818 / 9,150 | 3,011 / 10,560 | 3,158 / 10,575 | 212 / 313 | 4,237 / 11,609 | 240 / 0 / 0 |
| disc-a9-gateway-only-s24 (r1) | 828 / 928 | 833 / 929 | 2,289 / 7,629 | 3,015 / 8,611 | 3,109 / 8,661 | 219 / 338 | 4,175 / 9,739 | 240 / 0 / 0 |
| disc-a9-gateway-only-s24 (r2) | 814 / 853 | 819 / 858 | 1,471 / 8,712 | 3,039 / 9,622 | 3,205 / 9,657 | 216 / 335 | 4,288 / 10,727 | 240 / 0 / 0 |
| disc-a9-no-budget-cap-s24 (a9) | 813 / 864 | 819 / 867 | 629 / 2,851 | 1,723 / 3,896 | 1,727 / 3,905 | 214 / 278 | 2,762 / 4,985 | 240 / 0 / 0 |
| disc-a9-no-budget-cap-s24 (r1) | 822 / 879 | 829 / 885 | 596 / 2,928 | 1,282 / 3,676 | 1,303 / 3,713 | 215 / 261 | 2,336 / 4,737 | 240 / 0 / 0 |
| disc-a9-no-budget-cap-s24 (r2) | 812 / 881 | 818 / 886 | 573 / 3,999 | 1,504 / 4,760 | 1,544 / 4,801 | 213 / 329 | 2,604 / 5,837 | 240 / 0 / 0 |
| disc-a9-no-degrade-s24 (a9) | 815 / 865 | 819 / 871 | 226 / 310 | 656 / 2,704 | 664 / 2,746 | 240 / 330 | 1,725 / 3,814 | 240 / 0 / 0 |
| disc-a9-no-degrade-s24 (r1) | 812 / 853 | 817 / 855 | 246 / 364 | 545 / 1,449 | 587 / 1,554 | 212 / 297 | 1,651 / 2,653 | 240 / 0 / 0 |
| disc-a9-no-degrade-s24 (r2) | 820 / 878 | 825 / 884 | 247 / 338 | 574 / 2,422 | 597 / 2,424 | 235 / 323 | 1,658 / 3,474 | 240 / 0 / 0 |
| disc-a9-no-events-s24 (a9) | 815 / 854 | 818 / 859 | 231 / 330 | 438 / 610 | 446 / 613 | 210 / 248 | 1,492 / 1,686 | — / — / — |
| disc-a9-no-events-s24 (r1) | 814 / 846 | 817 / 851 | 217 / 325 | 416 / 575 | 420 / 578 | 211 / 260 | 1,471 / 1,611 | — / — / — |
| disc-a9-no-events-s24 (r2) | 818 / 849 | 823 / 855 | 221 / 320 | 400 / 569 | 402 / 571 | 215 / 252 | 1,459 / 1,624 | — / — / — |
| disc-a9-no-headroom-s24 (a9) | 813 / 865 | 818 / 867 | 235 / 346 | 443 / 564 | 447 / 568 | 218 / 317 | 1,489 / 1,704 | 240 / 0 / 0 |
| disc-a9-no-headroom-s24 (r1) | 811 / 852 | 816 / 858 | 234 / 338 | 430 / 609 | 431 / 612 | 205 / 239 | 1,457 / 1,647 | 240 / 0 / 0 |
| disc-a9-no-headroom-s24 (r2) | 815 / 863 | 819 / 869 | 246 / 324 | 399 / 537 | 403 / 540 | 210 / 256 | 1,451 / 1,593 | 240 / 0 / 0 |
| disc-a9-no-kv-makeroom-s24 (a9) | 816 / 844 | 821 / 850 | 233 / 324 | 422 / 605 | 423 / 607 | 215 / 255 | 1,479 / 1,644 | 240 / 0 / 0 |
| disc-a9-no-kv-makeroom-s24 (r1) | 812 / 844 | 817 / 849 | 220 / 344 | 428 / 576 | 431 / 578 | 211 / 255 | 1,476 / 1,626 | 240 / 0 / 0 |
| disc-a9-no-kv-makeroom-s24 (r2) | 808 / 843 | 811 / 851 | 221 / 297 | 432 / 585 | 431 / 589 | 208 / 251 | 1,469 / 1,631 | 240 / 0 / 0 |
| disc-a9-no-reserve-s24 (a9) | 816 / 844 | 821 / 851 | 269 / 381 | 475 / 617 | 482 / 633 | 210 / 259 | 1,515 / 1,692 | 240 / 0 / 0 |
| disc-a9-no-reserve-s24 (r1) | 820 / 863 | 825 / 869 | 289 / 421 | 472 / 649 | 478 / 651 | 214 / 258 | 1,523 / 1,698 | 240 / 0 / 0 |
| disc-a9-no-reserve-s24 (r2) | 813 / 914 | 820 / 916 | 296 / 406 | 487 / 666 | 491 / 685 | 218 / 299 | 1,533 / 1,795 | 240 / 0 / 0 |
| disc-a9-no-throttle-s24 (a9) | 811 / 841 | 816 / 845 | 224 / 333 | 416 / 596 | 432 / 598 | 212 / 267 | 1,453 / 1,656 | 241 / 0 / 0 |
| disc-a9-no-throttle-s24 (r1) | 809 / 851 | 814 / 856 | 223 / 350 | 422 / 614 | 425 / 617 | 208 / 255 | 1,450 / 1,662 | 240 / 0 / 0 |
| disc-a9-no-throttle-s24 (r2) | 812 / 837 | 816 / 843 | 227 / 348 | 418 / 599 | 427 / 602 | 209 / 263 | 1,464 / 1,656 | 240 / 0 / 0 |
| disc-pr-a-s24 (upstream) | 818 / 897 | 821 / 905 | 583 / 2,553 | 1,634 / 4,224 | 1,656 / 4,244 | 239 / 321 | 2,738 / 5,317 | 240 / 0 / 0 |
| disc-pr-a-s24 (upstream-r1) | 817 / 854 | 822 / 856 | 611 / 2,886 | 1,926 / 4,698 | 1,928 / 4,707 | 241 / 338 | 3,003 / 5,729 | 241 / 0 / 0 |
| disc-pr-a-s24 (upstream-r2) | 815 / 868 | 819 / 873 | 641 / 3,190 | 1,846 / 4,266 | 1,863 / 4,278 | 232 / 289 | 2,928 / 5,323 | 240 / 0 / 0 |
| disc-pr-ab-s24 (upstream) | 812 / 859 | 816 / 866 | 265 / 406 | 771 / 3,327 | 773 / 3,331 | 244 / 298 | 1,856 / 4,415 | 240 / 0 / 0 |
| disc-pr-ab-s24 (upstream-r1) | 821 / 867 | 826 / 872 | 265 / 366 | 550 / 1,787 | 557 / 1,812 | 244 / 331 | 1,624 / 2,921 | 240 / 0 / 0 |
| disc-pr-ab-s24 (upstream-r2) | 821 / 847 | 827 / 852 | 262 / 377 | 542 / 1,799 | 549 / 1,804 | 222 / 321 | 1,616 / 2,910 | 240 / 0 / 0 |
| disc-pr-ab48-s24 (upstream) | 813 / 838 | 818 / 844 | 269 / 395 | 534 / 710 | 544 / 715 | 235 / 298 | 1,608 / 1,785 | 240 / 0 / 0 |
| disc-pr-ab48-s24 (upstream-r1) | 817 / 857 | 821 / 862 | 276 / 386 | 510 / 798 | 512 / 801 | 262 / 349 | 1,597 / 1,915 | 240 / 0 / 0 |
| disc-pr-ab48-s24 (upstream-r2) | 815 / 839 | 819 / 845 | 264 / 409 | 494 / 686 | 503 / 690 | 232 / 312 | 1,564 / 1,813 | 240 / 0 / 0 |
| disc-a9-trigger-request-s24 (a9) | 811 / 855 | 817 / 856 | 224 / 328 | 445 / 572 | 448 / 595 | 213 / 256 | 1,486 / 1,643 | 240 / 0 / 0 |
| disc-a9-trigger-request-s24 (r1) | 805 / 838 | 810 / 844 | 223 / 379 | 427 / 660 | 430 / 663 | 205 / 272 | 1,456 / 1,775 | 240 / 0 / 0 |
| disc-a9-trigger-request-s24 (r2) | 812 / 879 | 819 / 885 | 228 / 347 | 437 / 602 | 443 / 608 | 212 / 258 | 1,501 / 1,688 | 240 / 0 / 0 |
| disc-a9-window48-s24 (a9-window) | 818 / 853 | 823 / 859 | 239 / 356 | 426 / 664 | 430 / 670 | 227 / 295 | 1,496 / 1,760 | 240 / 0 / 0 |
| disc-a9-window48-s24 (upstream-r1) | 818 / 845 | 823 / 852 | 218 / 303 | 480 / 705 | 484 / 708 | 249 / 317 | 1,593 / 1,826 | 240 / 0 / 0 |
| disc-a9-window48-s24 (upstream-r2) | 816 / 926 | 820 / 931 | 232 / 351 | 427 / 729 | 433 / 735 | 236 / 387 | 1,486 / 1,986 | 240 / 0 / 0 |

## The lifecycle policy

What the gateway's lifecycle policy (`policy = "lifecycle"`; E2's arm files spell it `leaseholder`) did in each point that ran it: the background requests its admission point deferred, by what held them back when they arrived (the KV headroom first, then the critical window's prefill budget), and the realtime replies it capped at `degrade_max_tokens` because the headroom was not there, of the realtime requests it admitted. A mechanism that deferred or capped nothing did not act in that point.

| Point | Deferred by the headroom | Deferred by the critical window | Degraded replies |
|---|---:|---:|---:|
| a9-s8 | 0 | 1 | 0 of 60 (0%) |
| a9-s16 (a9-standard) | 8 | 2 | 0 of 60 (0%) |
| a9-s16 (s1) | 5 | 4 | 0 of 60 (0%) |
| a9-s16 (s2) | 10 | 0 | 0 of 60 (0%) |
| a9-s24 | 41 | 27 | 1 of 60 (2%) |
| disc-a9-s24 (a9) | 41 | 0 | 60 of 60 (100%) |
| disc-a9-s24 (r1) | 31 | 0 | 60 of 60 (100%) |
| disc-a9-s24 (r2) | 41 | 0 | 60 of 60 (100%) |
| disc-a9-s24 (r3) | 44 | 0 | 60 of 60 (100%) |
| disc-a9-s24 (r4) | 45 | 0 | 60 of 60 (100%) |
| disc-a9-budget0-s24 (a9) | 39 | 0 | 60 of 60 (100%) |
| disc-a9-budget0-s24 (r1) | 42 | 0 | 60 of 60 (100%) |
| disc-a9-budget0-s24 (r2) | 49 | 0 | 60 of 60 (100%) |
| disc-a9-budget500-s24 (a9) | 39 | 0 | 60 of 60 (100%) |
| disc-a9-budget500-s24 (r1) | 48 | 0 | 60 of 60 (100%) |
| disc-a9-budget500-s24 (r2) | 44 | 0 | 60 of 60 (100%) |
| disc-a9-gateway-only-s24 (a9) | 86 | 0 | 60 of 60 (100%) |
| disc-a9-gateway-only-s24 (r1) | 83 | 0 | 60 of 60 (100%) |
| disc-a9-gateway-only-s24 (r2) | 71 | 0 | 60 of 60 (100%) |
| disc-a9-no-budget-cap-s24 (a9) | 61 | 0 | 60 of 60 (100%) |
| disc-a9-no-budget-cap-s24 (r1) | 55 | 0 | 60 of 60 (100%) |
| disc-a9-no-budget-cap-s24 (r2) | 56 | 0 | 60 of 60 (100%) |
| disc-a9-no-degrade-s24 (a9) | 50 | 0 | 0 of 60 (0%) |
| disc-a9-no-degrade-s24 (r1) | 55 | 0 | 0 of 60 (0%) |
| disc-a9-no-degrade-s24 (r2) | 44 | 0 | 0 of 60 (0%) |
| disc-a9-no-events-s24 (a9) | 4 | 0 | 60 of 60 (100%) |
| disc-a9-no-events-s24 (r1) | 6 | 0 | 60 of 60 (100%) |
| disc-a9-no-events-s24 (r2) | 4 | 0 | 60 of 60 (100%) |
| disc-a9-no-headroom-s24 (a9) | 0 | 3 | 60 of 60 (100%) |
| disc-a9-no-headroom-s24 (r1) | 0 | 2 | 60 of 60 (100%) |
| disc-a9-no-headroom-s24 (r2) | 0 | 1 | 60 of 60 (100%) |
| disc-a9-no-kv-makeroom-s24 (a9) | 36 | 0 | 60 of 60 (100%) |
| disc-a9-no-kv-makeroom-s24 (r1) | 46 | 0 | 60 of 60 (100%) |
| disc-a9-no-kv-makeroom-s24 (r2) | 43 | 0 | 60 of 60 (100%) |
| disc-a9-no-reserve-s24 (a9) | 51 | 0 | 60 of 60 (100%) |
| disc-a9-no-reserve-s24 (r1) | 46 | 0 | 60 of 60 (100%) |
| disc-a9-no-reserve-s24 (r2) | 44 | 0 | 60 of 60 (100%) |
| disc-a9-no-throttle-s24 (a9) | 52 | 0 | 60 of 60 (100%) |
| disc-a9-no-throttle-s24 (r1) | 43 | 0 | 60 of 60 (100%) |
| disc-a9-no-throttle-s24 (r2) | 38 | 0 | 60 of 60 (100%) |
| disc-a9-trigger-request-s24 (a9) | 9 | 0 | 60 of 60 (100%) |
| disc-a9-trigger-request-s24 (r1) | 10 | 0 | 60 of 60 (100%) |
| disc-a9-trigger-request-s24 (r2) | 11 | 0 | 60 of 60 (100%) |
| disc-a9-window48-s24 (a9-window) | 48 | 0 | 0 of 60 (0%) |
| disc-a9-window48-s24 (upstream-r1) | 50 | 0 | 0 of 60 (0%) |
| disc-a9-window48-s24 (upstream-r2) | 41 | 0 | 0 of 60 (0%) |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| a1-voice (r3) | 0 | 1,615 | 35,728 | 0 | — | — | 2,351 |
| a1-voice (r4) | 0 | 1,609 | 35,424 | 0 | — | — | 2,335 |
| a1-voice (s1) | 0 | 1,574 | 36,912 | 0 | — | — | 2,479 |
| a1-voice (s2) | 0 | 1,553 | 35,264 | 0 | — | — | 2,319 |
| a1-voice (w2) | 0 | 1,581 | 37,168 | 0 | — | — | 2,479 |
| a2-stock-s8 | 0 | 40,139 | 695,696 | 185,392 | 185,347 | -45 | 47,860 |
| a2-stock-s16 | 0 | 120,642 | 1,080,784 | 375,872 | 375,780 | -92 | 124,706 |
| a2-stock-s24 | 1 | 369,411 | 1,782,832 | 981,600 | 977,937 | -3,663 | 389,233 |
| a3-priority-s8 | 0 | 40,600 | 733,424 | 209,520 | 209,469 | -51 | 48,228 |
| a3-priority-s16 | 0 | 128,349 | 1,204,576 | 404,416 | 404,317 | -99 | 131,824 |
| a3-priority-s24 | 0 | 367,408 | 1,742,400 | 986,560 | 986,319 | -241 | 385,666 |
| a4-b1024-t512-s16 | 0 | 121,456 | 1,057,712 | 394,464 | 394,368 | -96 | 126,881 |
| a4-b1024-t256-s16 | 0 | 117,541 | 1,075,136 | 366,304 | 366,215 | -89 | 125,617 |
| a4-b512-t0-s16 | 1 | 109,604 | 1,069,120 | 419,296 | 415,898 | -3,398 | 116,899 |
| a4-b512-t256-s16 (s1) | 0 | 129,783 | 1,243,872 | 425,408 | 425,304 | -104 | 138,190 |
| a4-b512-t256-s16 (s2) | 0 | 117,742 | 1,192,944 | 417,632 | 417,530 | -102 | 128,193 |
| a4-b512-t256-s16 (w2) | 0 | 117,065 | 1,067,840 | 418,672 | 418,570 | -102 | 121,714 |
| a4-b1024-t0-s16 | 0 | 122,599 | 1,150,160 | 414,736 | 414,635 | -101 | 130,880 |
| a4-b512-t256-s24 | 0 | 418,988 | 1,955,920 | 1,106,784 | 1,087,974 | -18,810 | 424,984 |
| a4-b1024-t0-s24 | 0 | 374,930 | 1,853,872 | 1,032,544 | 1,028,869 | -3,675 | 395,967 |
| factor-no-preemptive-s16 | 1 | 109,046 | 1,078,816 | 418,560 | 419,130 | 570 | 115,652 |
| a5-queued8192-s16 | 0 | 140,202 | 1,190,336 | 269,312 | 269,246 | -66 | 128,801 |
| a5-watermark10-s16 | 0 | 117,233 | 1,044,256 | 442,368 | 442,260 | -108 | 121,842 |
| a6-reserve2048-s16 | 0 | 115,361 | 1,035,088 | 456,000 | 452,450 | -3,550 | 123,954 |
| a6-cap8-s16 | 1 | 140,918 | 1,175,824 | 472,000 | 471,885 | -115 | 142,445 |
| a6-cap2-s16 | 0 | 152,898 | 1,351,600 | 452,848 | 452,737 | -111 | 147,100 |
| a7a-pr54530-s16 | 0 | 115,513 | 1,059,936 | 378,864 | 378,772 | -92 | 123,650 |
| a7a-seqs12-s16 | 0 | 116,147 | 1,042,800 | 383,520 | 382,259 | -1,261 | 123,762 |
| a8a-flow-s16 | 2 | 127,636 | 1,097,088 | 448,704 | 449,874 | 1,170 | 128,241 |
| a8b-smg-s16 | 0 | 153,757 | 1,354,992 | 467,088 | 466,974 | -114 | 147,804 |
| a8c-resubmit-s16 | 0 | 104,842 | 1,130,624 | 395,632 | 445,731 | 50,099 | 142,717 |
| a9-s8 | 0 | 40,487 | 717,872 | 210,176 | 210,125 | -51 | 48,340 |
| a9-s16 (a9-standard) | 0 | 121,921 | 1,189,360 | 444,704 | 439,957 | -4,747 | 127,425 |
| a9-s16 (s1) | 0 | 127,466 | 1,024,112 | 417,424 | 417,322 | -102 | 132,240 |
| a9-s16 (s2) | 0 | 139,531 | 1,255,664 | 494,480 | 493,543 | -937 | 145,309 |
| a9-s24 | 0 | 387,513 | 1,750,288 | 1,104,736 | 1,104,466 | -270 | 399,502 |
| calls4-s0 | 0 | 3,300 | 71,728 | 0 | 0 | 0 | 4,623 |
| calls4-s16 | 0 | 132,851 | 1,208,992 | 460,560 | 460,959 | 399 | 138,238 |
| calls6-s16 | 0 | 116,586 | 1,116,640 | 507,200 | 507,076 | -124 | 124,786 |
| disc-frozen-s24 (r1) | 32 | 1,567,708 | 16,704 | 62,816 | 233,335 | 170,519 | 1,565,026 |
| disc-frozen-s24 (r2) | 32 | 1,537,324 | 39,184 | 37,776 | 249,443 | 211,667 | 1,544,151 |
| disc-frozen-s24 (r3) | 31 | 1,503,170 | 38,752 | 31,824 | 196,656 | 164,832 | 1,489,108 |
| disc-frozen-s24 (r4) | 29 | 1,554,413 | 42,192 | 35,648 | 259,009 | 223,361 | 1,543,239 |
| disc-frozen-s24 (w2c) | 31 | 1,476,557 | 16,848 | 59,712 | 219,674 | 159,962 | 1,475,864 |
| disc-a4-b1024-t0-s24 (r1) | 0 | 1,582,067 | 20,752 | 56,848 | 57,634 | 786 | 1,577,455 |
| disc-a4-b1024-t0-s24 (r2) | 0 | 1,588,012 | 34,992 | 41,760 | 42,550 | 790 | 1,601,417 |
| disc-a4-b1024-t0-s24 (r3) | 1 | 1,673,959 | 45,040 | 32,544 | 33,096 | 552 | 1,678,838 |
| disc-a4-b1024-t0-s24 (r4) | 0 | 1,663,501 | 48,000 | 30,704 | 30,697 | -7 | 1,675,047 |
| disc-a4-b1024-t0-s24 (w2c) | 0 | 1,610,189 | 8,560 | 71,152 | 71,135 | -17 | 1,615,525 |
| disc-a5-watermark10-s24 | 24 | 1,498,128 | 29,680 | 45,776 | 188,594 | 142,818 | 1,516,638 |
| disc-a5-queued8192-s24 | 0 | 1,949,452 | 630,448 | 2,247,264 | 2,246,715 | -549 | 1,874,342 |
| disc-a6-reserve2048-s24 (r1) | 0 | 1,319,436 | 40,000 | 37,328 | 37,463 | 135 | 1,316,367 |
| disc-a6-reserve2048-s24 (r2) | 0 | 1,265,643 | 21,472 | 55,408 | 55,394 | -14 | 1,267,466 |
| disc-a6-reserve2048-s24 (r3) | 0 | 1,178,780 | 29,888 | 44,000 | 44,181 | 181 | 1,182,991 |
| disc-a6-reserve2048-s24 (r4) | 0 | 1,308,548 | 30,784 | 47,392 | 47,380 | -12 | 1,287,430 |
| disc-a6-reserve2048-s24 (w2c) | 0 | 1,357,646 | 31,728 | 44,912 | 45,109 | 197 | 1,336,554 |
| disc-a6-cap2-s24 | 8 | 1,387,537 | 35,344 | 33,072 | 67,472 | 34,400 | 1,376,800 |
| disc-a6-cap8-s24 | 27 | 1,559,034 | 34,544 | 42,224 | 229,336 | 187,112 | 1,558,739 |
| disc-a7a-pr54530-s24 | 30 | 1,583,713 | 29,488 | 40,896 | 240,965 | 200,069 | 1,583,661 |
| disc-a8a-flow-s24 | 29 | 1,621,836 | 23,776 | 217,216 | 440,644 | 223,428 | 1,629,090 |
| disc-a8b-smg-s24 | 8 | 1,357,574 | 22,576 | 47,440 | 73,150 | 25,710 | 1,358,580 |
| disc-a8c-resubmit-s24 (r1) | 6 | 461,657 | 127,552 | 629,088 | 879,929 | 250,841 | 1,115,856 |
| disc-a8c-resubmit-s24 (r2) | 5 | 519,211 | 159,424 | 504,288 | 632,358 | 128,070 | 1,147,928 |
| disc-a8c-resubmit-s24 (r3) | 6 | 565,889 | 123,888 | 410,096 | 664,942 | 254,846 | 1,066,476 |
| disc-a8c-resubmit-s24 (r4) | 7 | 594,038 | 113,952 | 466,288 | 713,282 | 246,994 | 1,150,439 |
| disc-a8c-resubmit-s24 (w2c) | 8 | 645,172 | 70,160 | 619,792 | 821,847 | 202,055 | 1,114,720 |
| disc-a9-s24 (a9) | 0 | 939,122 | 38,896 | 22,672 | 22,858 | 186 | 917,456 |
| disc-a9-s24 (r1) | 0 | 908,602 | 38,368 | 23,952 | 24,138 | 186 | 914,257 |
| disc-a9-s24 (r2) | 0 | 903,571 | 35,184 | 27,904 | 28,089 | 185 | 908,386 |
| disc-a9-s24 (r3) | 0 | 898,908 | 40,400 | 21,920 | 21,915 | -5 | 888,583 |
| disc-a9-s24 (r4) | 0 | 939,343 | 35,152 | 27,136 | 27,321 | 185 | 923,119 |
| disc-a9-budget0-s24 (a9) | 0 | 912,987 | 27,456 | 34,688 | 34,871 | 183 | 901,556 |
| disc-a9-budget0-s24 (r1) | 0 | 906,573 | 37,472 | 24,608 | 24,602 | -6 | 906,739 |
| disc-a9-budget0-s24 (r2) | 0 | 912,933 | 36,816 | 25,088 | 25,082 | -6 | 920,911 |
| disc-a9-budget500-s24 (a9) | 0 | 904,948 | 37,024 | 25,504 | 25,498 | -6 | 908,290 |
| disc-a9-budget500-s24 (r1) | 0 | 916,326 | 37,424 | 24,960 | 25,130 | 170 | 910,066 |
| disc-a9-budget500-s24 (r2) | 0 | 897,679 | 37,584 | 23,664 | 23,690 | 26 | 910,274 |
| disc-a9-engine-only-s24 (a9) | 0 | 1,097,758 | 43,984 | 33,040 | 33,032 | -8 | 1,116,079 |
| disc-a9-engine-only-s24 (r1) | 0 | 1,047,877 | 45,824 | 28,864 | 28,857 | -7 | 1,037,427 |
| disc-a9-engine-only-s24 (r2) | 0 | 1,199,747 | 29,536 | 48,816 | 48,804 | -12 | 1,182,767 |
| disc-a9-gateway-only-s24 (a9) | 29 | 1,445,761 | 27,392 | 45,552 | 231,879 | 186,327 | 1,434,914 |
| disc-a9-gateway-only-s24 (r1) | 40 | 1,457,712 | 27,088 | 39,632 | 262,448 | 222,816 | 1,433,762 |
| disc-a9-gateway-only-s24 (r2) | 29 | 1,394,652 | 37,008 | 30,224 | 196,480 | 166,256 | 1,399,274 |
| disc-a9-no-budget-cap-s24 (a9) | 0 | 1,096,657 | 27,840 | 42,848 | 43,029 | 181 | 1,110,497 |
| disc-a9-no-budget-cap-s24 (r1) | 0 | 1,090,811 | 39,296 | 31,216 | 31,400 | 184 | 1,095,796 |
| disc-a9-no-budget-cap-s24 (r2) | 0 | 1,098,686 | 41,024 | 30,912 | 30,904 | -8 | 1,103,299 |
| disc-a9-no-degrade-s24 (a9) | 0 | 1,154,967 | 44,992 | 31,344 | 31,544 | 200 | 1,151,623 |
| disc-a9-no-degrade-s24 (r1) | 0 | 1,117,896 | 34,464 | 39,472 | 39,606 | 134 | 1,120,622 |
| disc-a9-no-degrade-s24 (r2) | 0 | 1,113,855 | 45,104 | 31,312 | 31,496 | 184 | 1,125,277 |
| disc-a9-no-events-s24 (a9) | 0 | 929,796 | 34,000 | 29,584 | 29,769 | 185 | 925,422 |
| disc-a9-no-events-s24 (r1) | 0 | 941,797 | 37,632 | 24,768 | 24,762 | -6 | 947,721 |
| disc-a9-no-events-s24 (r2) | 0 | 910,675 | 38,048 | 24,592 | 24,778 | 186 | 910,226 |
| disc-a9-no-headroom-s24 (a9) | 0 | 894,993 | 34,128 | 27,648 | 27,641 | -7 | 911,889 |
| disc-a9-no-headroom-s24 (r1) | 0 | 898,871 | 22,368 | 39,680 | 39,862 | 182 | 897,029 |
| disc-a9-no-headroom-s24 (r2) | 0 | 897,481 | 28,256 | 34,736 | 34,919 | 183 | 927,869 |
| disc-a9-no-kv-makeroom-s24 (a9) | 0 | 930,478 | 39,296 | 22,832 | 23,018 | 186 | 928,733 |
| disc-a9-no-kv-makeroom-s24 (r1) | 0 | 900,329 | 23,936 | 38,224 | 38,407 | 183 | 899,876 |
| disc-a9-no-kv-makeroom-s24 (r2) | 0 | 904,833 | 38,768 | 23,344 | 23,530 | 186 | 892,742 |
| disc-a9-no-reserve-s24 (a9) | 39 | 938,806 | 18,096 | 43,472 | 78,685 | 35,213 | 932,668 |
| disc-a9-no-reserve-s24 (r1) | 38 | 931,816 | 17,776 | 44,432 | 109,877 | 65,445 | 950,584 |
| disc-a9-no-reserve-s24 (r2) | 27 | 936,752 | 31,552 | 29,872 | 77,245 | 47,373 | 946,681 |
| disc-a9-no-throttle-s24 (a9) | 0 | 898,196 | 36,576 | 26,000 | 25,994 | -6 | 919,535 |
| disc-a9-no-throttle-s24 (r1) | 0 | 878,592 | 37,840 | 24,720 | 24,714 | -6 | 896,565 |
| disc-a9-no-throttle-s24 (r2) | 0 | 930,418 | 35,760 | 26,672 | 26,665 | -7 | 925,662 |
| disc-pr-a-s24 (upstream) | 37 | 1,277,712 | 27,504 | 49,056 | 156,202 | 107,146 | 1,311,344 |
| disc-pr-a-s24 (upstream-r1) | 40 | 1,446,837 | 18,192 | 61,248 | 220,234 | 158,986 | 1,448,190 |
| disc-pr-a-s24 (upstream-r2) | 37 | 1,266,297 | 24,720 | 51,984 | 144,653 | 92,669 | 1,298,163 |
| disc-pr-ab-s24 (upstream) | 32 | 1,300,337 | 18,560 | 55,408 | 111,013 | 55,605 | 1,290,997 |
| disc-pr-ab-s24 (upstream-r1) | 41 | 1,271,058 | 22,880 | 55,920 | 176,085 | 120,165 | 1,303,970 |
| disc-pr-ab-s24 (upstream-r2) | 38 | 1,136,945 | 19,184 | 56,336 | 122,578 | 66,242 | 1,144,025 |
| disc-pr-ab48-s24 (upstream) | 37 | 1,138,238 | 42,032 | 33,056 | 103,399 | 70,343 | 1,137,002 |
| disc-pr-ab48-s24 (upstream-r1) | 38 | 1,154,539 | 24,672 | 50,912 | 143,837 | 92,925 | 1,168,403 |
| disc-pr-ab48-s24 (upstream-r2) | 43 | 1,192,074 | 30,512 | 37,808 | 92,441 | 54,633 | 1,202,394 |
| disc-a9-trigger-request-s24 (a9) | 0 | 906,307 | 33,056 | 29,200 | 29,193 | -7 | 908,914 |
| disc-a9-trigger-request-s24 (r1) | 0 | 907,293 | 33,680 | 30,160 | 30,345 | 185 | 920,639 |
| disc-a9-trigger-request-s24 (r2) | 0 | 909,229 | 36,640 | 25,712 | 25,898 | 186 | 916,416 |
| disc-a9-window48-s24 (a9-window) | 0 | 1,034,011 | 36,048 | 37,248 | 37,383 | 135 | 1,051,871 |
| disc-a9-window48-s24 (upstream-r1) | 0 | 1,114,347 | 45,632 | 32,128 | 32,264 | 136 | 1,136,219 |
| disc-a9-window48-s24 (upstream-r2) | 0 | 1,071,820 | 47,616 | 27,152 | 27,289 | 137 | 1,077,993 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a1-voice: frozen box, voice alone | 4 | 236 | 757 | 297 | 199 | 352 | 1,605 | 1,608 | 1,770 | 80 |
| a4-b512-t256-s16: priority, step cap 512, long-prefill threshold 256 (frozen box) | 2 | 118 | 788 | 407 | 219 | 361 | 1,775 | 1,709 | 2,476 | 117 |
| a7b-sglang-hicache-s8: SGLang 0.5.22 with HiCache 4x (write-through) | 1 | 59 | 777 | 309 | 204 | 355 | 1,645 | 1,660 | 1,822 | 102 |
| a7b-sglang-hicache-s16: SGLang 0.5.22 with HiCache 4x (write-through) | 1 | 59 | 799 | 439 | 227 | 347 | 1,812 | 1,720 | 2,218 | 164 |
| a9-s8: Leaseholder: timed headroom, critical-window throttle, degrade; make-room | 1 | 59 | 785 | 378 | 231 | 355 | 1,749 | 1,745 | 2,060 | 91 |
| a9-s16: Leaseholder: timed headroom, critical-window throttle, degrade; make-room | 3 | 177 | 792 | 399 | 214 | 353 | 1,759 | 1,694 | 2,248 | 119 |
| a9-s24: Leaseholder: timed headroom, critical-window throttle, degrade; make-room | 1 | 59 | 812 | 498 | 217 | 350 | 1,877 | 1,848 | 2,375 | 200 |
| disc-frozen-s24: frozen box, discriminating scenario (large contexts) | 5 | 295 | 822 | 3,801 | 244 | 343 | 5,211 | 4,526 | 10,424 | 2,760 |
| disc-a4-b1024-t0-s24: priority, step cap 1024, long-prefill threshold 0, discriminating scenario | 5 | 295 | 826 | 3,475 | 239 | 370 | 4,909 | 4,374 | 9,145 | 2,601 |
| disc-a5-watermark10-s24: frozen box + watermark 0.1, discriminating scenario | 1 | 59 | 827 | 4,243 | 239 | 300 | 5,609 | 4,843 | 10,647 | 3,261 |
| disc-a6-reserve2048-s24: frozen box + token reservation, 2,048 tokens, discriminating scenario | 5 | 295 | 827 | 1,967 | 238 | 346 | 3,378 | 3,127 | 5,693 | 935 |
| disc-a6-cap2-s24: frozen box + static cap, 2 background in flight, discriminating scenario | 1 | 59 | 829 | 2,733 | 262 | 372 | 4,197 | 3,363 | 8,573 | 1,862 |
| disc-a6-cap8-s24: frozen box + static cap, 8 background in flight, discriminating scenario | 1 | 59 | 824 | 4,243 | 241 | 379 | 5,688 | 5,713 | 9,876 | 3,381 |
| disc-a7a-pr54530-s24: priority, max-num-seqs 12, with vLLM #54530, discriminating scenario | 1 | 59 | 823 | 4,040 | 252 | 344 | 5,459 | 4,687 | 10,930 | 2,827 |
| disc-a7b-sglang-s24: SGLang 0.5.22: priority scheduling with preemption, discriminating scenario | 1 | 59 | 816 | 4,812 | 238 | 283 | 6,149 | 5,367 | 12,051 | 2,370 |
| disc-a7b-sglang-hicache-s24: SGLang 0.5.22 with HiCache 4x (write-through), discriminating scenario | 5 | 295 | 816 | 4,144 | 234 | 383 | 5,576 | 4,829 | 12,779 | 2,483 |
| disc-a8a-flow-s24: frozen box + flow control (llm-d, emulated), ceilings 6/6/6, discriminating scenario | 1 | 59 | 820 | 3,756 | 251 | 311 | 5,137 | 4,221 | 10,992 | 2,579 |
| disc-a8b-smg-s24: frozen box behind SMG 1.13.0, as shipped, discriminating scenario | 1 | 59 | 819 | 3,446 | 239 | 308 | 4,812 | 3,511 | 11,050 | 2,494 |
| disc-a8c-resubmit-s24: frozen box + preempt-resubmit before the first byte, discriminating scenario | 5 | 295 | 826 | 622 | 222 | 353 | 2,023 | 1,960 | 2,584 | 274 |
| disc-a9-s24: Leaseholder: timed headroom, critical-window throttle, degrade; make-room | 5 | 295 | 824 | 429 | 222 | 354 | 1,830 | 1,815 | 2,075 | 221 |
| disc-a9-budget0-s24: Leaseholder, critical-window budget 0 ms | 3 | 177 | 826 | 431 | 222 | 354 | 1,833 | 1,820 | 2,152 | 217 |
| disc-a9-budget500-s24: Leaseholder, critical-window budget 500 ms | 3 | 177 | 821 | 426 | 216 | 360 | 1,824 | 1,818 | 2,054 | 223 |
| disc-a9-engine-only-s24: Make-room alone, without the gateway policy | 3 | 177 | 825 | 850 | 239 | 352 | 2,266 | 1,991 | 4,044 | 234 |
| disc-a9-gateway-only-s24: Leaseholder's gateway policy on the frozen engine | 3 | 177 | 830 | 4,020 | 231 | 309 | 5,390 | 4,473 | 11,417 | 3,110 |
| disc-a9-no-budget-cap-s24: Leaseholder, make-room without the prefill budget cap | 3 | 177 | 828 | 1,784 | 225 | 340 | 3,177 | 2,896 | 5,562 | 938 |
| disc-a9-no-degrade-s24: Leaseholder without degrade | 3 | 177 | 827 | 888 | 239 | 345 | 2,299 | 2,048 | 3,980 | 238 |
| disc-a9-no-events-s24: Leaseholder without lifecycle events | 3 | 177 | 823 | 432 | 217 | 353 | 1,825 | 1,831 | 2,036 | 223 |
| disc-a9-no-headroom-s24: Leaseholder without the turn-imminent headroom | 3 | 177 | 825 | 432 | 218 | 358 | 1,833 | 1,827 | 2,039 | 233 |
| disc-a9-no-kv-makeroom-s24: Leaseholder, make-room without KV make-room | 3 | 177 | 821 | 432 | 217 | 357 | 1,827 | 1,820 | 2,036 | 223 |
| disc-a9-no-reserve-s24: Leaseholder, make-room without the realtime KV reserve | 3 | 177 | 827 | 492 | 221 | 358 | 1,898 | 1,890 | 2,131 | 277 |
| disc-a9-no-throttle-s24: Leaseholder without the critical-window throttle | 3 | 177 | 820 | 437 | 216 | 358 | 1,830 | 1,822 | 2,060 | 230 |
| disc-pr-a-s24: vLLM #63350: make-room's KV preemption alone, no gateway policy | 3 | 177 | 826 | 2,082 | 241 | 346 | 3,496 | 3,228 | 5,846 | 959 |
| disc-pr-ab-s24: vLLM #63350 and #63351: KV preemption and the budget cap, no gateway policy | 3 | 177 | 826 | 914 | 245 | 353 | 2,336 | 2,053 | 4,108 | 265 |
| disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy | 3 | 177 | 824 | 526 | 249 | 353 | 1,953 | 1,939 | 2,209 | 270 |
| disc-a9-trigger-request-s24: Leaseholder, headroom armed by the realtime request | 3 | 177 | 822 | 451 | 217 | 358 | 1,848 | 1,839 | 2,125 | 231 |
| disc-a9-window48-s24: Leaseholder without degrade, a 48-token first-tokens window | 3 | 177 | 828 | 473 | 241 | 357 | 1,899 | 1,877 | 2,207 | 231 |
