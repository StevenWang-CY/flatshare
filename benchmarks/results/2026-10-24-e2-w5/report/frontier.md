# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. A point measured in several campaigns lists each run by its campaign, then the median and range (min–max) over its runs, where the frontier places it. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32, 2 callers

The voice SLO allows a V2V p95 of 2,034 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,784 ms). Figure: `frontier.svg`.

### Arm 1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice (w5-r1) |  | 0 | 0 | — | 0.00 | 1,655 | 1,810 | 1,916 | 0 of 50 | — / — | 60 | 0 | 0 / 0 / 0 | 2,580 | 56 |
| w5-a1-voice (w5-r2) |  | 0 | 0 | — | 0.00 | 1,608 | 1,774 | 1,808 | 0 of 50 | — / — | 59 | 0 | 0 / 0 / 0 | 2,580 | 58 |
| w5-a1-voice (w5-r3) |  | 0 | 0 | — | 0.00 | 1,628 | 1,819 | 1,892 | 0 of 50 | — / — | 60 | 0 | 0 / 0 / 0 | 2,580 | 56 |
| w5-a1-voice (w5-r4) |  | 0 | 0 | — | 0.00 | 1,612 | 1,764 | 1,798 | 0 of 50 | — / — | 62 | 0 | 0 / 0 / 0 | 2,580 | 55 |
| w5-a1-voice (w5-r5) |  | 0 | 0 | — | 0.00 | 1,619 | 1,784 | 1,938 | 0 of 50 | — / — | 74 | 0 | 0 / 0 / 0 | 2,580 | 57 |
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call; median (range) of 5 runs |  | 0 | 0 (0–0) | — | | 1,619 (1,608–1,655) | 1,784 (1,764–1,819) | 1,892 (1,798–1,938) | 0 (0–0) | — / — | 60 (59–74) | | | | |

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-std-frozen-s16 (w5-ext-r1) |  | 16 | 5,137 | 5,330 | 1.06 | 1,714 | 2,147 | 2,895 | 0 of 50 | 113 / 3,649 | 243 | 0 | 0 / 0 / 0 | 2,790 | 63 |
| w5-std-frozen-s16 (w5-ext-r2) |  | 16 | 10,415 | 12,763 | 1.54 | 2,039 | 3,832 | 5,696 | 0 of 50 | 533 / 10,618 | 729 | 6 | 0 / 0 / 0 | 2,782 | 65 |
| w5-std-frozen-s16 (w5-ext-r3) |  | 16 | 6,153 | 7,131 | 1.08 | 1,937 | 3,067 | 3,579 | 0 of 50 | 172 / 6,092 | 453 | 1 | 0 / 0 / 0 | 2,782 | 66 |
| w5-std-frozen-s16 (w5-ext-r4) |  | 16 | 6,055 | 7,263 | 1.25 | 1,731 | 2,300 | 2,814 | 0 of 50 | 117 / 1,383 | 266 | 0 | 0 / 0 / 0 | 2,782 | 66 |
| w5-std-frozen-s16 (w5-ext-r5) |  | 16 | 3,978 | 4,954 | 0.80 | 1,668 | 2,516 | 5,081 | 0 of 50 | 129 / 3,892 | 218 | 0 | 0 / 0 / 0 | 2,782 | 64 |
| w5-std-frozen-s16: priority, step cap 512, long-prefill threshold 256 (frozen box), a warm-up turn opening each call; median (range) of 5 runs |  | 16 | 6,055 (3,978–10,415) | 7,131 (4,954–12,763) | | 1,731 (1,668–2,039) | 2,516 (2,147–3,832) | 3,579 (2,814–5,696) | 0 (0–0) | 129 (113–533) / 3,892 (1,383–10,618) | 266 (218–729) | | | | |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-std-pr-ab48-s16 (w5-ext-r1) |  | 16 | 5,295 | 5,570 | 1.07 | 1,702 | 2,038 | 2,280 | 0 of 50 | 122 / 4,392 | 205 | 1 | 0 / 0 / 0 | 2,782 | 63 |
| w5-std-pr-ab48-s16 (w5-ext-r2) |  | 16 | 10,620 | 12,223 | 1.57 | 2,037 | 2,419 | 2,739 | 0 of 50 | 581 / 11,006 | 385 | 9 | 0 / 0 / 0 | 2,782 | 66 |
| w5-std-pr-ab48-s16 (w5-ext-r3) |  | 16 | 6,194 | 6,815 | 1.09 | 1,799 | 2,122 | 2,394 | 0 of 50 | 167 / 4,924 | 261 | 1 | 0 / 0 / 0 | 2,782 | 65 |
| w5-std-pr-ab48-s16 (w5-ext-r4) |  | 16 | 6,220 | 7,523 | 1.29 | 1,721 | 2,050 | 2,222 | 0 of 50 | 129 / 2,169 | 206 | 0 | 0 / 0 / 0 | 2,782 | 65 |
| w5-std-pr-ab48-s16 (w5-ext-r5) |  | 16 | 3,827 | 4,667 | 0.77 | 1,681 | 2,021 | 2,113 | 0 of 50 | 132 / 4,739 | 208 | 1 | 0 / 0 / 0 | 2,782 | 65 |
| w5-std-pr-ab48-s16: vLLM #63350 and #63351 with a 48-token window, no gateway policy, standard scenario, a warm-up turn opening each call; median (range) of 5 runs |  | 16 | 6,194 (3,827–10,620) | 6,815 (4,667–12,223) | | 1,721 (1,681–2,037) | 2,050 (2,021–2,419) | 2,280 (2,113–2,739) | 0 (0–0) | 132 (122–581) / 4,739 (2,169–11,006) | 208 (205–385) | | | | |

## Workload: weka-x32, 4 callers

The voice SLO allows a V2V p95 of 2,044 ms, 250 ms above voice alone with as many callers (w5-a1-voice-c4: 1,794 ms). Figure: `frontier-4-callers.svg`.

### Arm 1-calls

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice-c4 (w5-cap-r1) |  | 0 | 0 | — | 0.00 | 1,682 | 1,894 | 2,128 | 0 of 100 | — / — | 66 | 0 | 0 / 0 / 0 | 2,782 | 56 |
| w5-a1-voice-c4 (w5-cap-r2) |  | 0 | 0 | — | 0.00 | 1,647 | 1,794 | 1,856 | 0 of 100 | — / — | 73 | 0 | 0 / 0 / 0 | 2,782 | 56 |
| w5-a1-voice-c4 (w5-cap-r3) |  | 0 | 0 | — | 0.00 | 1,622 | 1,780 | 1,936 | 0 of 100 | — / — | 66 | 0 | 0 / 0 / 0 | 2,782 | 57 |
| w5-a1-voice-c4: frozen box, 4 callers, voice alone, a warm-up turn opening each call; median (range) of 3 runs |  | 0 | 0 (0–0) | — | | 1,647 (1,622–1,682) | 1,794 (1,780–1,894) | 1,936 (1,856–2,128) | 0 (0–0) | — / — | 66 (66–73) | | | | |

## Workload: weka-x32, 6 callers

The voice SLO allows a V2V p95 of 2,135 ms, 250 ms above voice alone with as many callers (w5-a1-voice-c6: 1,885 ms). Figure: `frontier-6-callers.svg`.

### Arm 1-calls

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice-c6 (w5-cap-r1) |  | 0 | 0 | — | 0.00 | 1,637 | 1,885 | 1,996 | 0 of 150 | — / — | 86 | 0 | 0 / 0 / 0 | 2,790 | 60 |
| w5-a1-voice-c6 (w5-cap-r2) |  | 0 | 0 | — | 0.00 | 1,674 | 1,885 | 2,094 | 0 of 150 | — / — | 82 | 0 | 0 / 0 / 0 | 2,790 | 62 |
| w5-a1-voice-c6 (w5-cap-r3) |  | 0 | 0 | — | 0.00 | 1,635 | 1,878 | 2,028 | 0 of 150 | — / — | 78 | 0 | 0 / 0 / 0 | 2,790 | 59 |
| w5-a1-voice-c6: frozen box, 6 callers, voice alone, a warm-up turn opening each call; median (range) of 3 runs |  | 0 | 0 (0–0) | — | | 1,637 (1,635–1,674) | 1,885 (1,878–1,885) | 2,028 (1,996–2,094) | 0 (0–0) | — / — | 82 (78–86) | | | | |

## Workload: weka-x32-long, 2 callers

The voice SLO allows a V2V p95 of 2,034 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,784 ms). Figure: `frontier-weka-x32-long.svg`.

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-frozen-s24 (w5-pr-r1) |  | 24 | 3,750 | 3,511 | 0.32 | 5,172 | 9,946 | 11,677 | 0 of 50 | 63,008 / 98,747 | 7,040 | 36 | 0 / 0 / 0 | 2,715 | 70 |
| w5-disc-frozen-s24 (w5-pr-r2) |  | 24 | 3,299 | 3,083 | 0.37 | 4,382 | 10,133 | 14,663 | 0 of 50 | 62,002 / 104,489 | 7,941 | 24 | 0 / 0 / 0 | 2,730 | 68 |
| w5-disc-frozen-s24 (w5-pr-r3) |  | 24 | 3,325 | 3,304 | 0.26 | 4,238 | 10,810 | 12,466 | 0 of 50 | 76,510 / 128,811 | 7,786 | 34 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-frozen-s24 (w5-pr-r4) |  | 24 | 3,721 | 3,713 | 0.37 | 5,134 | 12,115 | 13,964 | 0 of 50 | 62,282 / 106,807 | 8,965 | 30 | 0 / 0 / 0 | 2,730 | 68 |
| w5-disc-frozen-s24 (w5-pr-r5) |  | 24 | 3,244 | 3,216 | 0.27 | 5,814 | 10,523 | 11,375 | 0 of 50 | 80,440 / 120,696 | 7,786 | 28 | 0 / 0 / 0 | 2,730 | 68 |
| w5-disc-frozen-s24 (w5-r1) |  | 24 | 3,739 | 3,562 | 0.31 | 4,772 | 10,576 | 12,724 | 0 of 50 | 62,695 / 95,930 | 7,561 | 31 | 0 / 0 / 0 | 2,722 | 70 |
| w5-disc-frozen-s24 (w5-r2) |  | 24 | 3,409 | 3,235 | 0.38 | 4,263 | 10,770 | 11,361 | 0 of 50 | 62,798 / 105,673 | 6,942 | 31 | 0 / 0 / 0 | 2,730 | 70 |
| w5-disc-frozen-s24 (w5-r3) |  | 24 | 3,375 | 3,301 | 0.27 | 3,411 | 8,686 | 10,282 | 0 of 50 | 76,617 / 125,343 | 5,374 | 22 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-frozen-s24 (w5-r4) |  | 24 | 3,756 | 3,634 | 0.38 | 4,873 | 9,782 | 10,153 | 0 of 50 | 62,494 / 97,490 | 6,912 | 32 | 0 / 0 / 0 | 2,722 | 70 |
| w5-disc-frozen-s24 (w5-r5) |  | 24 | 3,322 | 3,306 | 0.29 | 4,953 | 10,432 | 12,291 | 0 of 50 | 76,597 / 113,974 | 8,082 | 37 | 0 / 0 / 0 | 2,730 | 70 |
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call; median (range) of 10 runs |  | 24 | 3,392 (3,244–3,756) | 3,305 (3,083–3,713) | | 4,822 (3,411–5,814) | 10,477 (8,686–12,115) | 11,984 (10,153–14,663) | 0 (0–0) | 62,903 (62,002–80,440) / 106,240 (95,930–128,811) | 7,673 (5,374–8,965) | | | | |

### Arm 8c

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-a8c-resubmit-s24 (w5-ext-r1) |  | 24 | 3,552 | 3,597 | 0.36 | 1,990 | 2,516 | 2,779 | 0 of 50 | 33,336 / 161,907 | 349 | 10 | 0 / 0 / 1,129 | 2,722 | 68 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r2) |  | 24 | 3,437 | 3,285 | 0.52 | 2,064 | 2,860 | 3,147 | 0 of 50 | 28,202 / 160,331 | 351 | 10 | 0 / 0 / 1,251 | 2,722 | 68 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r3) |  | 24 | 3,181 | 2,798 | 0.35 | 1,968 | 2,522 | 2,919 | 0 of 50 | 32,395 / 199,981 | 341 | 3 | 0 / 0 / 1,192 | 2,722 | 69 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r4) |  | 24 | 3,756 | 3,703 | 0.50 | 1,968 | 2,522 | 2,678 | 0 of 50 | 26,622 / 192,812 | 312 | 7 | 0 / 0 / 1,210 | 2,722 | 69 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r5) |  | 24 | 3,058 | 3,332 | 0.40 | 2,076 | 2,868 | 3,111 | 0 of 50 | 35,732 / 232,405 | 358 | 6 | 0 / 0 / 1,371 | 2,730 | 68 |
| w5-disc-a8c-resubmit-s24: frozen box + preempt-resubmit before the first byte, discriminating scenario, a warm-up turn opening each call; median (range) of 5 runs |  | 24 | 3,437 (3,058–3,756) | 3,332 (2,798–3,703) | | 1,990 (1,968–2,076) | 2,522 (2,516–2,868) | 2,919 (2,678–3,147) | 0 (0–0) | 32,395 (26,622–35,732) / 192,812 (160,331–232,405) | 349 (312–358) | | | | |

### Arm 9-pr-a

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-a-s24 (w5-pr-r1) |  | 24 | 3,653 | 3,515 | 0.30 | 3,630 | 6,261 | 7,107 | 0 of 50 | 66,728 / 98,455 | 2,407 | 66 | 0 / 0 / 0 | 2,722 | 70 |
| w5-disc-pr-a-s24 (w5-pr-r2) |  | 24 | 3,311 | 3,103 | 0.37 | 3,351 | 5,376 | 5,743 | 0 of 50 | 65,136 / 111,737 | 2,120 | 39 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-a-s24 (w5-pr-r3) |  | 24 | 3,378 | 3,316 | 0.27 | 3,131 | 4,693 | 5,362 | 0 of 50 | 78,139 / 130,502 | 1,602 | 60 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-a-s24 (w5-pr-r4) |  | 24 | 3,680 | 3,719 | 0.37 | 3,191 | 4,970 | 5,970 | 0 of 50 | 61,726 / 98,968 | 1,737 | 50 | 0 / 0 / 0 | 2,730 | 68 |
| w5-disc-pr-a-s24 (w5-pr-r5) |  | 24 | 3,276 | 3,414 | 0.28 | 3,154 | 5,085 | 6,661 | 0 of 50 | 78,873 / 121,931 | 1,539 | 51 | 0 / 0 / 0 | 2,730 | 68 |
| w5-disc-pr-a-s24: vLLM #63350: make-room's KV preemption alone, no gateway policy, a warm-up turn opening each call; median (range) of 5 runs |  | 24 | 3,378 (3,276–3,680) | 3,414 (3,103–3,719) | | 3,191 (3,131–3,630) | 5,085 (4,693–6,261) | 5,970 (5,362–7,107) | 0 (0–0) | 66,728 (61,726–78,873) / 111,737 (98,455–130,502) | 1,737 (1,539–2,407) | | | | |

### Arm 9-pr-ab

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab-s24 (w5-pr-r1) |  | 24 | 3,512 | 3,552 | 0.29 | 2,096 | 3,022 | 4,705 | 0 of 50 | 70,885 / 99,415 | 377 | 47 | 0 / 0 / 0 | 2,722 | 70 |
| w5-disc-pr-ab-s24 (w5-pr-r2) |  | 24 | 3,282 | 3,120 | 0.38 | 1,974 | 3,100 | 3,530 | 0 of 50 | 66,245 / 105,048 | 350 | 47 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab-s24 (w5-pr-r3) |  | 24 | 3,204 | 3,108 | 0.25 | 1,938 | 2,868 | 3,885 | 0 of 50 | 84,061 / 130,170 | 343 | 38 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab-s24 (w5-pr-r4) |  | 24 | 3,657 | 3,497 | 0.36 | 1,983 | 3,192 | 4,858 | 0 of 50 | 62,963 / 104,866 | 364 | 52 | 0 / 0 / 0 | 2,730 | 68 |
| w5-disc-pr-ab-s24 (w5-pr-r5) |  | 24 | 3,186 | 3,298 | 0.27 | 2,067 | 3,094 | 3,746 | 0 of 50 | 80,230 / 124,399 | 383 | 50 | 0 / 0 / 0 | 2,730 | 68 |
| w5-disc-pr-ab-s24: vLLM #63350 and #63351: KV preemption and the budget cap, no gateway policy, a warm-up turn opening each call; median (range) of 5 runs |  | 24 | 3,282 (3,186–3,657) | 3,298 (3,108–3,552) | | 1,983 (1,938–2,096) | 3,094 (2,868–3,192) | 3,885 (3,530–4,858) | 0 (0–0) | 70,885 (62,963–84,061) / 105,048 (99,415–130,170) | 364 (343–383) | | | | |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-s24 (w5-ext-r1) |  | 24 | 3,457 | 3,425 | 0.28 | 2,018 | 2,293 | 2,467 | 0 of 50 | 72,171 / 104,474 | 398 | 55 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-s24 (w5-ext-r2) |  | 24 | 3,099 | 3,097 | 0.35 | 1,941 | 2,257 | 2,347 | 0 of 50 | 67,227 / 111,121 | 353 | 49 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-s24 (w5-ext-r3) |  | 24 | 3,130 | 3,130 | 0.25 | 1,940 | 2,132 | 2,197 | 0 of 50 | 85,097 / 131,915 | 335 | 45 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-s24 (w5-ext-r4) |  | 24 | 3,597 | 3,634 | 0.35 | 1,918 | 2,162 | 2,300 | 0 of 50 | 65,923 / 106,191 | 363 | 53 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-s24 (w5-ext-r5) |  | 24 | 3,146 | 3,203 | 0.27 | 1,925 | 2,145 | 2,254 | 0 of 50 | 81,374 / 119,912 | 367 | 52 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-s24 (w5-pr-r1) |  | 24 | 3,457 | 3,456 | 0.28 | 2,001 | 2,174 | 2,379 | 0 of 50 | 73,553 / 95,625 | 344 | 43 | 0 / 0 / 0 | 2,722 | 69 |
| w5-disc-pr-ab48-s24 (w5-pr-r2) |  | 24 | 3,170 | 3,033 | 0.35 | 1,970 | 2,301 | 2,349 | 0 of 50 | 69,251 / 115,301 | 341 | 40 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-s24 (w5-pr-r3) |  | 24 | 3,202 | 3,236 | 0.25 | 1,922 | 2,189 | 2,552 | 0 of 50 | 81,280 / 129,286 | 403 | 53 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-s24 (w5-pr-r4) |  | 24 | 3,603 | 3,611 | 0.37 | 1,996 | 2,232 | 2,332 | 0 of 50 | 66,109 / 94,304 | 376 | 47 | 0 / 0 / 0 | 2,730 | 68 |
| w5-disc-pr-ab48-s24 (w5-pr-r5) |  | 24 | 3,136 | 3,173 | 0.27 | 1,982 | 2,397 | 2,967 | 0 of 50 | 82,552 / 122,284 | 443 | 46 | 0 / 0 / 0 | 2,730 | 67 |
| w5-disc-pr-ab48-s24 (w5-r1) |  | 24 | 3,584 | 3,403 | 0.29 | 1,944 | 2,236 | 2,352 | 0 of 50 | 70,284 / 98,331 | 405 | 50 | 0 / 0 / 0 | 2,722 | 69 |
| w5-disc-pr-ab48-s24 (w5-r2) |  | 24 | 3,169 | 3,108 | 0.36 | 1,971 | 2,175 | 2,882 | 0 of 50 | 68,011 / 109,844 | 377 | 39 | 0 / 0 / 0 | 2,730 | 70 |
| w5-disc-pr-ab48-s24 (w5-r3) |  | 24 | 3,168 | 3,156 | 0.24 | 1,967 | 2,298 | 2,338 | 0 of 50 | 86,754 / 130,484 | 375 | 32 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-s24 (w5-r4) |  | 24 | 3,583 | 3,436 | 0.36 | 1,928 | 2,250 | 2,384 | 0 of 50 | 66,536 / 98,610 | 351 | 46 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-s24 (w5-r5) |  | 24 | 3,082 | 3,115 | 0.27 | 1,985 | 2,307 | 2,445 | 0 of 50 | 83,919 / 124,827 | 376 | 45 | 0 / 0 / 0 | 2,730 | 70 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call; median (range) of 15 runs |  | 24 | 3,170 (3,082–3,603) | 3,203 (3,033–3,634) | | 1,967 (1,918–2,018) | 2,236 (2,132–2,397) | 2,352 (2,197–2,967) | 0 (0–0) | 72,171 (65,923–86,754) / 111,121 (94,304–131,915) | 375 (335–443) | | | | |

### Arm 9-pr-ab48-v03

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-v03-s24 (w5-r1) |  | 24 | 3,463 | 3,518 | 0.28 | 1,992 | 2,273 | 2,530 | 0 of 50 | 73,831 / 94,731 | 343 | 36 | 0 / 0 / 0 | 2,730 | 70 |
| w5-disc-pr-ab48-v03-s24 (w5-r2) |  | 24 | 3,139 | 3,132 | 0.35 | 1,941 | 2,333 | 2,356 | 0 of 50 | 67,158 / 113,090 | 381 | 50 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-v03-s24 (w5-r3) |  | 24 | 3,174 | 3,129 | 0.25 | 1,921 | 2,134 | 2,387 | 0 of 50 | 86,449 / 134,131 | 382 | 32 | 0 / 0 / 0 | 2,730 | 70 |
| w5-disc-pr-ab48-v03-s24 (w5-r4) |  | 24 | 3,470 | 3,521 | 0.35 | 1,961 | 2,241 | 2,331 | 0 of 50 | 66,882 / 106,335 | 386 | 42 | 0 / 0 / 0 | 2,730 | 70 |
| w5-disc-pr-ab48-v03-s24 (w5-r5) |  | 24 | 3,069 | 3,124 | 0.27 | 1,924 | 2,214 | 2,293 | 0 of 50 | 83,838 / 122,430 | 383 | 41 | 0 / 0 / 0 | 2,730 | 70 |
| w5-disc-pr-ab48-v03-s24: vLLM #63350 and #63351 with a 48-token window as v0.3 built them, no gateway policy, a warm-up turn opening each call; median (range) of 5 runs |  | 24 | 3,174 (3,069–3,470) | 3,132 (3,124–3,521) | | 1,941 (1,921–1,992) | 2,241 (2,134–2,333) | 2,356 (2,293–2,530) | 0 (0–0) | 73,831 (66,882–86,449) / 113,090 (94,731–134,131) | 382 (343–386) | | | | |

## Workload: weka-x32-long, 4 callers

The voice SLO allows a V2V p95 of 2,044 ms, 250 ms above voice alone with as many callers (w5-a1-voice-c4: 1,794 ms). Figure: `frontier-weka-x32-long-4-callers.svg`.

### Arm 9-pr-ab48-calls

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-c4-s24 (w5-cap-r1) |  | 24 | 3,238 | 3,059 | 0.26 | 1,993 | 2,313 | 2,476 | 0 of 100 | 78,025 / 108,696 | 370 | 69 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-c4-s24 (w5-cap-r2) |  | 24 | 3,060 | 3,055 | 0.35 | 2,017 | 2,415 | 2,920 | 0 of 100 | 73,373 / 100,786 | 385 | 60 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-c4-s24 (w5-cap-r3) |  | 24 | 2,912 | 2,923 | 0.22 | 1,973 | 2,250 | 2,512 | 0 of 100 | 94,393 / 140,642 | 395 | 65 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-c4-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, 4 callers, a warm-up turn opening each call; median (range) of 3 runs |  | 24 | 3,060 (2,912–3,238) | 3,055 (2,923–3,059) | | 1,993 (1,973–2,017) | 2,313 (2,250–2,415) | 2,512 (2,476–2,920) | 0 (0–0) | 78,025 (73,373–94,393) / 108,696 (100,786–140,642) | 385 (370–395) | | | | |

## Workload: weka-x32-long, 6 callers

The voice SLO allows a V2V p95 of 2,135 ms, 250 ms above voice alone with as many callers (w5-a1-voice-c6: 1,885 ms). Figure: `frontier-weka-x32-long-6-callers.svg`.

### Arm 9-pr-ab48-calls

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-c6-s24 (w5-cap-r1) |  | 24 | 3,104 | 2,921 | 0.25 | 2,036 | 2,392 | 2,502 | 0 of 150 | 83,499 / 109,724 | 379 | 70 | 0 / 0 / 0 | 2,730 | 69 |
| w5-disc-pr-ab48-c6-s24 (w5-cap-r2) |  | 24 | 2,844 | 2,707 | 0.33 | 2,016 | 2,360 | 2,533 | 0 of 150 | 77,716 / 124,556 | 368 | 73 | 0 / 0 / 0 | 2,737 | 69 |
| w5-disc-pr-ab48-c6-s24 (w5-cap-r3) |  | 24 | 2,759 | 2,716 | 0.21 | 2,015 | 2,468 | 2,753 | 0 of 150 | 100,979 / 145,377 | 411 | 69 | 0 / 0 / 0 | 2,737 | 69 |
| w5-disc-pr-ab48-c6-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, 6 callers, a warm-up turn opening each call; median (range) of 3 runs |  | 24 | 2,844 (2,759–3,104) | 2,716 (2,707–2,921) | | 2,016 (2,015–2,036) | 2,392 (2,360–2,468) | 2,533 (2,502–2,753) | 0 (0–0) | 83,499 (77,716–100,979) / 124,556 (109,724–145,377) | 379 (368–411) | | | | |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w5-a1-voice (w5-r1) | 747 / 795 | 750 / 800 | 81 / 98 | 298 / 482 | 303 / 488 | 228 / 292 | 1,285 / 1,501 | 249 / 0 / 0 |
| w5-a1-voice (w5-r2) | 743 / 789 | 748 / 792 | 78 / 96 | 291 / 429 | 293 / 431 | 193 / 251 | 1,238 / 1,411 | 249 / 0 / 0 |
| w5-a1-voice (w5-r3) | 739 / 780 | 745 / 785 | 78 / 94 | 291 / 453 | 293 / 456 | 208 / 262 | 1,238 / 1,497 | 249 / 0 / 0 |
| w5-a1-voice (w5-r4) | 746 / 800 | 752 / 806 | 81 / 98 | 280 / 414 | 284 / 417 | 206 / 268 | 1,244 / 1,438 | 249 / 0 / 0 |
| w5-a1-voice (w5-r5) | 747 / 799 | 753 / 805 | 80 / 99 | 296 / 412 | 298 / 422 | 190 / 247 | 1,265 / 1,419 | 248 / 0 / 0 |
| w5-std-frozen-s16 (w5-ext-r1) | 766 / 847 | 771 / 852 | 94 / 213 | 312 / 682 | 318 / 691 | 243 / 300 | 1,368 / 1,810 | 248 / 0 / 0 |
| w5-std-frozen-s16 (w5-ext-r2) | 806 / 846 | 810 / 852 | 225 / 750 | 643 / 2,321 | 648 / 2,332 | 235 / 299 | 1,715 / 3,367 | 248 / 0 / 0 |
| w5-std-frozen-s16 (w5-ext-r3) | 807 / 851 | 811 / 852 | 175 / 471 | 522 / 1,624 | 519 / 1,640 | 222 / 277 | 1,588 / 2,735 | 248 / 0 / 0 |
| w5-std-frozen-s16 (w5-ext-r4) | 792 / 852 | 795 / 857 | 105 / 306 | 369 / 915 | 372 / 916 | 224 / 265 | 1,376 / 1,984 | 249 / 0 / 0 |
| w5-std-frozen-s16 (w5-ext-r5) | 766 / 847 | 771 / 854 | 95 / 262 | 323 / 1,070 | 326 / 1,072 | 205 / 271 | 1,330 / 2,154 | 249 / 0 / 0 |
| w5-std-pr-ab48-s16 (w5-ext-r1) | 777 / 838 | 782 / 843 | 92 / 236 | 304 / 540 | 306 / 548 | 227 / 298 | 1,316 / 1,725 | 248 / 0 / 0 |
| w5-std-pr-ab48-s16 (w5-ext-r2) | 808 / 890 | 810 / 895 | 236 / 421 | 542 / 965 | 555 / 969 | 240 / 328 | 1,654 / 2,069 | 248 / 0 / 0 |
| w5-std-pr-ab48-s16 (w5-ext-r3) | 807 / 844 | 811 / 849 | 145 / 301 | 399 / 621 | 417 / 626 | 220 / 275 | 1,451 / 1,744 | 248 / 0 / 0 |
| w5-std-pr-ab48-s16 (w5-ext-r4) | 805 / 856 | 810 / 861 | 104 / 239 | 353 / 604 | 356 / 606 | 228 / 287 | 1,382 / 1,657 | 248 / 0 / 0 |
| w5-std-pr-ab48-s16 (w5-ext-r5) | 783 / 852 | 787 / 857 | 94 / 231 | 299 / 586 | 302 / 589 | 213 / 260 | 1,302 / 1,682 | 248 / 0 / 0 |
| w5-a1-voice-c4 (w5-cap-r1) | 749 / 826 | 754 / 831 | 81 / 102 | 297 / 432 | 304 / 438 | 201 / 259 | 1,283 / 1,420 | 508 / 0 / 0 |
| w5-a1-voice-c4 (w5-cap-r2) | 746 / 810 | 750 / 816 | 82 / 102 | 294 / 443 | 303 / 447 | 205 / 271 | 1,276 / 1,437 | 510 / 0 / 0 |
| w5-a1-voice-c4 (w5-cap-r3) | 749 / 819 | 753 / 825 | 79 / 103 | 294 / 436 | 300 / 439 | 198 / 251 | 1,263 / 1,426 | 510 / 0 / 0 |
| w5-a1-voice-c6 (w5-cap-r1) | 756 / 826 | 762 / 831 | 84 / 121 | 279 / 458 | 285 / 461 | 194 / 305 | 1,257 / 1,486 | 779 / 0 / 0 |
| w5-a1-voice-c6 (w5-cap-r2) | 751 / 828 | 756 / 834 | 82 / 115 | 303 / 455 | 307 / 460 | 217 / 291 | 1,287 / 1,518 | 777 / 0 / 0 |
| w5-a1-voice-c6 (w5-cap-r3) | 751 / 864 | 756 / 869 | 83 / 109 | 276 / 423 | 284 / 428 | 194 / 283 | 1,259 / 1,495 | 777 / 0 / 0 |
| w5-disc-frozen-s24 (w5-pr-r1) | 814 / 862 | 819 / 868 | 2,696 / 7,078 | 3,732 / 8,500 | 3,873 / 8,552 | 268 / 349 | 4,970 / 9,686 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (w5-pr-r2) | 811 / 845 | 816 / 849 | 1,825 / 8,024 | 2,940 / 8,711 | 2,903 / 8,756 | 240 / 303 | 4,017 / 9,789 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (w5-pr-r3) | 809 / 845 | 814 / 848 | 1,460 / 7,844 | 2,868 / 9,466 | 2,926 / 9,483 | 236 / 338 | 3,971 / 10,503 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (w5-pr-r4) | 812 / 858 | 817 / 863 | 2,675 / 9,116 | 3,543 / 10,708 | 3,742 / 10,739 | 240 / 330 | 4,865 / 11,789 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (w5-pr-r5) | 813 / 871 | 818 / 876 | 3,265 / 7,787 | 4,541 / 8,806 | 4,507 / 8,821 | 238 / 330 | 5,540 / 9,865 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (w5-r1) | 814 / 855 | 819 / 857 | 1,831 / 7,683 | 3,239 / 8,957 | 3,226 / 9,029 | 254 / 353 | 4,320 / 10,282 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (w5-r2) | 815 / 848 | 819 / 851 | 1,792 / 7,097 | 2,926 / 9,260 | 3,090 / 9,280 | 246 / 326 | 4,134 / 10,448 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (w5-r3) | 819 / 892 | 823 / 898 | 853 / 5,527 | 2,042 / 7,187 | 2,117 / 7,229 | 245 / 364 | 3,150 / 8,387 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (w5-r4) | 815 / 852 | 819 / 857 | 2,058 / 6,955 | 3,424 / 8,275 | 3,522 / 8,281 | 238 / 290 | 4,578 / 9,299 | 248 / 0 / 0 |
| w5-disc-frozen-s24 (w5-r5) | 813 / 867 | 818 / 871 | 1,984 / 8,113 | 3,420 / 9,047 | 3,520 / 9,062 | 239 / 365 | 4,578 / 10,140 | 248 / 0 / 0 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r1) | 813 / 859 | 818 / 863 | 259 / 368 | 552 / 978 | 554 / 985 | 256 / 300 | 1,640 / 2,088 | 248 / 0 / 0 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r2) | 814 / 895 | 818 / 896 | 275 / 388 | 606 / 1,476 | 630 / 1,482 | 230 / 286 | 1,714 / 2,547 | 248 / 0 / 0 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r3) | 813 / 843 | 816 / 848 | 286 / 366 | 534 / 1,092 | 536 / 1,097 | 220 / 321 | 1,570 / 2,204 | 248 / 0 / 0 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r4) | 809 / 835 | 815 / 842 | 250 / 332 | 535 / 1,198 | 550 / 1,202 | 212 / 259 | 1,608 / 2,244 | 248 / 0 / 0 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r5) | 816 / 867 | 822 / 873 | 286 / 389 | 610 / 1,354 | 614 / 1,365 | 226 / 300 | 1,685 / 2,473 | 248 / 0 / 0 |
| w5-disc-pr-a-s24 (w5-pr-r1) | 814 / 845 | 818 / 847 | 703 / 2,529 | 2,174 / 4,674 | 2,167 / 4,711 | 258 / 354 | 3,257 / 5,983 | 248 / 0 / 0 |
| w5-disc-pr-a-s24 (w5-pr-r2) | 808 / 842 | 813 / 847 | 716 / 2,143 | 1,908 / 4,008 | 1,928 / 4,013 | 244 / 306 | 3,050 / 5,056 | 248 / 0 / 0 |
| w5-disc-pr-a-s24 (w5-pr-r3) | 812 / 858 | 817 / 866 | 587 / 1,679 | 1,676 / 3,293 | 1,723 / 3,299 | 238 / 336 | 2,790 / 4,394 | 248 / 0 / 0 |
| w5-disc-pr-a-s24 (w5-pr-r4) | 814 / 844 | 818 / 848 | 667 / 1,785 | 1,751 / 3,702 | 1,776 / 3,707 | 234 / 293 | 2,871 / 4,762 | 248 / 0 / 0 |
| w5-disc-pr-a-s24 (w5-pr-r5) | 809 / 833 | 814 / 838 | 521 / 1,559 | 1,694 / 3,667 | 1,773 / 3,684 | 249 / 335 | 2,881 / 4,757 | 248 / 0 / 0 |
| w5-disc-pr-ab-s24 (w5-pr-r1) | 814 / 876 | 819 / 883 | 289 / 402 | 665 / 1,502 | 671 / 1,513 | 247 / 323 | 1,702 / 2,712 | 248 / 0 / 0 |
| w5-disc-pr-ab-s24 (w5-pr-r2) | 808 / 839 | 812 / 846 | 268 / 385 | 546 / 1,680 | 556 / 1,693 | 234 / 286 | 1,593 / 2,767 | 248 / 0 / 0 |
| w5-disc-pr-ab-s24 (w5-pr-r3) | 811 / 853 | 817 / 859 | 293 / 382 | 518 / 1,484 | 524 / 1,520 | 214 / 256 | 1,558 / 2,587 | 248 / 0 / 0 |
| w5-disc-pr-ab-s24 (w5-pr-r4) | 813 / 866 | 817 / 871 | 267 / 406 | 551 / 1,831 | 555 / 1,871 | 241 / 286 | 1,634 / 2,924 | 248 / 0 / 0 |
| w5-disc-pr-ab-s24 (w5-pr-r5) | 812 / 856 | 817 / 857 | 301 / 409 | 626 / 1,618 | 642 / 1,642 | 238 / 272 | 1,680 / 2,709 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-ext-r1) | 812 / 858 | 817 / 864 | 269 / 418 | 564 / 825 | 570 / 828 | 246 / 314 | 1,645 / 1,955 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-ext-r2) | 813 / 874 | 816 / 878 | 272 / 372 | 533 / 808 | 542 / 812 | 241 / 325 | 1,581 / 1,947 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-ext-r3) | 810 / 841 | 814 / 847 | 271 / 364 | 501 / 755 | 515 / 758 | 239 / 289 | 1,583 / 1,817 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-ext-r4) | 815 / 843 | 820 / 848 | 276 / 388 | 487 / 687 | 489 / 690 | 219 / 315 | 1,540 / 1,780 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-ext-r5) | 813 / 850 | 818 / 854 | 292 / 390 | 542 / 705 | 545 / 707 | 234 / 267 | 1,610 / 1,770 | 249 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-pr-r1) | 812 / 852 | 816 / 857 | 280 / 371 | 538 / 751 | 548 / 758 | 246 / 320 | 1,629 / 1,832 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-pr-r2) | 807 / 848 | 812 / 852 | 246 / 376 | 547 / 853 | 552 / 857 | 242 / 311 | 1,605 / 1,977 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-pr-r3) | 810 / 829 | 815 / 836 | 262 / 440 | 506 / 760 | 510 / 767 | 231 / 280 | 1,556 / 1,815 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-pr-r4) | 819 / 929 | 824 / 930 | 276 / 410 | 535 / 757 | 542 / 760 | 239 / 331 | 1,636 / 1,860 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-pr-r5) | 810 / 842 | 814 / 847 | 283 / 512 | 539 / 1,002 | 550 / 1,007 | 241 / 299 | 1,621 / 2,065 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-r1) | 811 / 845 | 815 / 849 | 266 / 425 | 528 / 788 | 524 / 793 | 249 / 306 | 1,596 / 1,850 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-r2) | 814 / 858 | 818 / 863 | 284 / 397 | 528 / 729 | 535 / 735 | 241 / 307 | 1,589 / 1,824 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-r3) | 812 / 847 | 816 / 851 | 280 / 399 | 530 / 781 | 543 / 784 | 244 / 320 | 1,596 / 1,959 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-r4) | 814 / 848 | 819 / 853 | 268 / 380 | 503 / 759 | 509 / 767 | 237 / 270 | 1,565 / 1,847 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 (w5-r5) | 819 / 857 | 824 / 861 | 277 / 407 | 531 / 779 | 528 / 782 | 233 / 309 | 1,596 / 1,863 | 248 / 0 / 0 |
| w5-disc-pr-ab48-v03-s24 (w5-r1) | 816 / 857 | 821 / 862 | 285 / 377 | 538 / 802 | 547 / 818 | 247 / 316 | 1,620 / 1,953 | 248 / 0 / 0 |
| w5-disc-pr-ab48-v03-s24 (w5-r2) | 810 / 842 | 815 / 847 | 291 / 416 | 526 / 749 | 531 / 753 | 239 / 279 | 1,583 / 1,831 | 248 / 0 / 0 |
| w5-disc-pr-ab48-v03-s24 (w5-r3) | 816 / 879 | 820 / 884 | 264 / 408 | 503 / 715 | 505 / 718 | 236 / 305 | 1,574 / 1,813 | 248 / 0 / 0 |
| w5-disc-pr-ab48-v03-s24 (w5-r4) | 811 / 902 | 816 / 907 | 279 / 416 | 526 / 766 | 531 / 787 | 243 / 291 | 1,602 / 1,881 | 248 / 0 / 0 |
| w5-disc-pr-ab48-v03-s24 (w5-r5) | 816 / 865 | 821 / 871 | 269 / 401 | 482 / 774 | 485 / 779 | 242 / 311 | 1,537 / 1,914 | 248 / 0 / 0 |
| w5-disc-pr-ab48-c4-s24 (w5-cap-r1) | 818 / 862 | 823 / 867 | 274 / 408 | 547 / 770 | 554 / 776 | 240 / 326 | 1,632 / 1,916 | 507 / 0 / 0 |
| w5-disc-pr-ab48-c4-s24 (w5-cap-r2) | 821 / 926 | 826 / 931 | 265 / 414 | 562 / 963 | 578 / 972 | 246 / 381 | 1,676 / 2,075 | 507 / 0 / 0 |
| w5-disc-pr-ab48-c4-s24 (w5-cap-r3) | 816 / 883 | 822 / 889 | 274 / 434 | 521 / 745 | 532 / 760 | 239 / 336 | 1,597 / 1,885 | 507 / 0 / 0 |
| w5-disc-pr-ab48-c6-s24 (w5-cap-r1) | 819 / 947 | 824 / 950 | 263 / 403 | 531 / 782 | 534 / 785 | 244 / 421 | 1,640 / 1,972 | 774 / 0 / 0 |
| w5-disc-pr-ab48-c6-s24 (w5-cap-r2) | 822 / 941 | 826 / 945 | 260 / 396 | 527 / 759 | 533 / 762 | 255 / 455 | 1,642 / 1,983 | 775 / 0 / 0 |
| w5-disc-pr-ab48-c6-s24 (w5-cap-r3) | 828 / 965 | 833 / 967 | 258 / 441 | 514 / 832 | 518 / 838 | 241 / 507 | 1,620 / 2,094 | 774 / 0 / 0 |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice (w5-r1) | 0 | 1,647 | 39,536 | 0 | — | — | 2,559 |
| w5-a1-voice (w5-r2) | 0 | 1,626 | 38,720 | 0 | — | — | 2,463 |
| w5-a1-voice (w5-r3) | 0 | 1,715 | 39,408 | 0 | — | — | 2,527 |
| w5-a1-voice (w5-r4) | 0 | 1,662 | 39,744 | 0 | — | — | 2,591 |
| w5-a1-voice (w5-r5) | 0 | 1,690 | 39,232 | 0 | — | — | 2,511 |
| w5-std-frozen-s16 (w5-ext-r1) | 0 | 129,631 | 1,290,560 | 440,672 | 440,564 | -108 | 138,670 |
| w5-std-frozen-s16 (w5-ext-r2) | 6 | 375,266 | 1,794,496 | 1,842,624 | 1,856,747 | 14,123 | 393,888 |
| w5-std-frozen-s16 (w5-ext-r3) | 1 | 267,579 | 1,279,616 | 686,464 | 689,496 | 3,032 | 250,659 |
| w5-std-frozen-s16 (w5-ext-r4) | 0 | 153,776 | 1,531,424 | 435,888 | 435,782 | -106 | 144,989 |
| w5-std-frozen-s16 (w5-ext-r5) | 0 | 172,434 | 872,960 | 301,408 | 301,334 | -74 | 177,957 |
| w5-std-pr-ab48-s16 (w5-ext-r1) | 1 | 126,447 | 1,256,144 | 445,776 | 445,891 | 115 | 131,712 |
| w5-std-pr-ab48-s16 (w5-ext-r2) | 9 | 377,210 | 1,843,408 | 1,773,232 | 1,782,381 | 9,149 | 383,762 |
| w5-std-pr-ab48-s16 (w5-ext-r3) | 1 | 244,726 | 1,258,320 | 604,624 | 594,959 | -9,665 | 226,473 |
| w5-std-pr-ab48-s16 (w5-ext-r4) | 0 | 154,163 | 1,521,360 | 464,032 | 463,919 | -113 | 146,252 |
| w5-std-pr-ab48-s16 (w5-ext-r5) | 1 | 201,192 | 797,552 | 314,304 | 314,739 | 435 | 177,461 |
| w5-a1-voice-c4 (w5-cap-r1) | 0 | 3,338 | 82,896 | 0 | 0 | 0 | 5,215 |
| w5-a1-voice-c4 (w5-cap-r2) | 0 | 3,423 | 85,136 | 0 | — | — | 5,423 |
| w5-a1-voice-c4 (w5-cap-r3) | 0 | 3,337 | 81,120 | 0 | — | — | 5,119 |
| w5-a1-voice-c6 (w5-cap-r1) | 0 | 5,261 | 119,344 | 0 | 0 | 0 | 7,358 |
| w5-a1-voice-c6 (w5-cap-r2) | 0 | 5,173 | 123,792 | 0 | — | — | 7,662 |
| w5-a1-voice-c6 (w5-cap-r3) | 0 | 5,159 | 118,624 | 0 | — | — | 7,246 |
| w5-disc-frozen-s24 (w5-pr-r1) | 36 | 1,746,634 | 28,416 | 58,224 | 310,468 | 252,244 | 1,736,424 |
| w5-disc-frozen-s24 (w5-pr-r2) | 24 | 1,386,636 | 19,280 | 51,632 | 210,061 | 158,429 | 1,397,179 |
| w5-disc-frozen-s24 (w5-pr-r3) | 34 | 1,482,188 | 5,808 | 40,416 | 282,651 | 242,235 | 1,490,804 |
| w5-disc-frozen-s24 (w5-pr-r4) | 30 | 1,657,621 | 20,544 | 43,312 | 289,625 | 246,313 | 1,665,369 |
| w5-disc-frozen-s24 (w5-pr-r5) | 28 | 1,411,874 | 15,056 | 61,232 | 246,452 | 185,220 | 1,420,037 |
| w5-disc-frozen-s24 (w5-r1) | 31 | 1,642,376 | 37,008 | 46,464 | 271,086 | 224,622 | 1,650,797 |
| w5-disc-frozen-s24 (w5-r2) | 31 | 1,556,293 | 17,440 | 72,096 | 272,733 | 200,637 | 1,523,004 |
| w5-disc-frozen-s24 (w5-r3) | 22 | 1,420,684 | 4,432 | 42,208 | 198,032 | 155,824 | 1,424,564 |
| w5-disc-frozen-s24 (w5-r4) | 32 | 1,638,250 | 22,112 | 49,088 | 244,884 | 195,796 | 1,647,950 |
| w5-disc-frozen-s24 (w5-r5) | 37 | 1,514,406 | 34,768 | 47,264 | 341,613 | 294,349 | 1,522,572 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r1) | 10 | 585,804 | 113,088 | 557,136 | 830,597 | 273,461 | 1,241,985 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r2) | 10 | 670,043 | 114,240 | 555,184 | 830,213 | 275,029 | 1,163,860 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r3) | 3 | 624,542 | 181,264 | 485,552 | 793,870 | 308,318 | 1,347,975 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r4) | 7 | 609,611 | 80,448 | 649,232 | 805,611 | 156,379 | 1,263,036 |
| w5-disc-a8c-resubmit-s24 (w5-ext-r5) | 6 | 518,850 | 139,008 | 494,288 | 731,725 | 237,437 | 1,142,569 |
| w5-disc-pr-a-s24 (w5-pr-r1) | 66 | 1,452,256 | 26,960 | 55,664 | 227,400 | 171,736 | 1,458,220 |
| w5-disc-pr-a-s24 (w5-pr-r2) | 39 | 1,230,839 | 14,240 | 64,880 | 162,504 | 97,624 | 1,246,880 |
| w5-disc-pr-a-s24 (w5-pr-r3) | 60 | 1,329,896 | 10,656 | 33,680 | 229,544 | 195,864 | 1,317,310 |
| w5-disc-pr-a-s24 (w5-pr-r4) | 50 | 1,512,610 | 11,824 | 52,896 | 207,917 | 155,021 | 1,505,904 |
| w5-disc-pr-a-s24 (w5-pr-r5) | 51 | 1,276,695 | 37,744 | 44,816 | 222,986 | 178,170 | 1,281,687 |
| w5-disc-pr-ab-s24 (w5-pr-r1) | 47 | 1,156,612 | 25,648 | 53,616 | 158,809 | 105,193 | 1,178,608 |
| w5-disc-pr-ab-s24 (w5-pr-r2) | 47 | 1,065,356 | 13,312 | 71,232 | 184,707 | 113,475 | 1,073,322 |
| w5-disc-pr-ab-s24 (w5-pr-r3) | 38 | 1,134,658 | 9,936 | 33,168 | 165,911 | 132,743 | 1,142,489 |
| w5-disc-pr-ab-s24 (w5-pr-r4) | 52 | 1,303,911 | 18,672 | 41,264 | 158,313 | 117,049 | 1,262,844 |
| w5-disc-pr-ab-s24 (w5-pr-r5) | 50 | 1,085,937 | 31,904 | 44,512 | 172,182 | 127,670 | 1,069,371 |
| w5-disc-pr-ab48-s24 (w5-ext-r1) | 55 | 1,205,336 | 34,880 | 45,200 | 194,560 | 149,360 | 1,214,439 |
| w5-disc-pr-ab48-s24 (w5-ext-r2) | 49 | 1,031,266 | 14,064 | 48,080 | 130,480 | 82,400 | 1,034,531 |
| w5-disc-pr-ab48-s24 (w5-ext-r3) | 45 | 1,125,902 | 8,096 | 37,728 | 159,753 | 122,025 | 1,123,790 |
| w5-disc-pr-ab48-s24 (w5-ext-r4) | 53 | 1,226,032 | 13,472 | 44,752 | 111,941 | 67,189 | 1,238,865 |
| w5-disc-pr-ab48-s24 (w5-ext-r5) | 52 | 1,052,656 | 35,392 | 39,568 | 179,572 | 140,004 | 1,029,892 |
| w5-disc-pr-ab48-s24 (w5-pr-r1) | 43 | 1,160,252 | 34,464 | 42,240 | 116,803 | 74,563 | 1,161,012 |
| w5-disc-pr-ab48-s24 (w5-pr-r2) | 40 | 1,068,760 | 13,344 | 61,088 | 178,132 | 117,044 | 1,064,972 |
| w5-disc-pr-ab48-s24 (w5-pr-r3) | 53 | 1,149,753 | 8,592 | 35,632 | 246,420 | 210,788 | 1,132,619 |
| w5-disc-pr-ab48-s24 (w5-pr-r4) | 47 | 1,255,949 | 21,856 | 40,560 | 157,162 | 116,602 | 1,270,922 |
| w5-disc-pr-ab48-s24 (w5-pr-r5) | 46 | 1,053,565 | 24,656 | 54,128 | 213,340 | 159,212 | 1,017,719 |
| w5-disc-pr-ab48-s24 (w5-r1) | 50 | 1,200,683 | 42,688 | 37,280 | 192,017 | 154,737 | 1,196,044 |
| w5-disc-pr-ab48-s24 (w5-r2) | 39 | 1,076,655 | 19,664 | 49,024 | 147,996 | 98,972 | 1,073,674 |
| w5-disc-pr-ab48-s24 (w5-r3) | 32 | 1,101,283 | 8,944 | 36,000 | 117,763 | 81,763 | 1,090,358 |
| w5-disc-pr-ab48-s24 (w5-r4) | 46 | 1,198,003 | 21,552 | 38,352 | 122,018 | 83,666 | 1,191,213 |
| w5-disc-pr-ab48-s24 (w5-r5) | 45 | 1,066,153 | 37,504 | 40,368 | 186,786 | 146,418 | 1,068,443 |
| w5-disc-pr-ab48-v03-s24 (w5-r1) | 36 | 1,132,825 | 36,528 | 41,456 | 125,713 | 84,257 | 1,162,484 |
| w5-disc-pr-ab48-v03-s24 (w5-r2) | 50 | 1,041,407 | 16,128 | 47,360 | 148,268 | 100,908 | 1,053,615 |
| w5-disc-pr-ab48-v03-s24 (w5-r3) | 32 | 1,070,580 | 5,632 | 36,560 | 159,753 | 123,193 | 1,072,170 |
| w5-disc-pr-ab48-v03-s24 (w5-r4) | 42 | 1,176,630 | 15,760 | 44,112 | 167,495 | 123,383 | 1,201,435 |
| w5-disc-pr-ab48-v03-s24 (w5-r5) | 41 | 1,113,134 | 24,608 | 52,816 | 201,727 | 148,911 | 1,088,262 |
| w5-disc-pr-ab48-c4-s24 (w5-cap-r1) | 69 | 1,193,877 | 33,408 | 89,408 | 201,087 | 111,679 | 1,201,707 |
| w5-disc-pr-ab48-c4-s24 (w5-cap-r2) | 60 | 1,037,310 | 28,256 | 107,152 | 161,145 | 53,993 | 1,048,224 |
| w5-disc-pr-ab48-c4-s24 (w5-cap-r3) | 65 | 1,069,458 | 11,264 | 72,416 | 225,705 | 153,289 | 1,093,701 |
| w5-disc-pr-ab48-c6-s24 (w5-cap-r1) | 70 | 1,086,000 | 39,488 | 121,200 | 204,702 | 83,502 | 1,088,726 |
| w5-disc-pr-ab48-c6-s24 (w5-cap-r2) | 73 | 995,173 | 34,160 | 104,704 | 178,069 | 73,365 | 1,010,393 |
| w5-disc-pr-ab48-c6-s24 (w5-cap-r3) | 69 | 969,551 | 27,984 | 97,808 | 180,356 | 82,548 | 969,619 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call | 5 | 295 | 756 | 303 | 211 | 352 | 1,623 | 1,623 | 1,796 | 80 |
| w5-std-frozen-s16: priority, step cap 512, long-prefill threshold 256 (frozen box), a warm-up turn opening each call | 5 | 295 | 794 | 581 | 228 | 354 | 1,957 | 1,778 | 3,033 | 213 |
| w5-std-pr-ab48-s16: vLLM #63350 and #63351 with a 48-token window, no gateway policy, standard scenario, a warm-up turn opening each call | 5 | 295 | 798 | 416 | 229 | 357 | 1,801 | 1,751 | 2,247 | 149 |
| w5-a1-voice-c4: frozen box, 4 callers, voice alone, a warm-up turn opening each call | 3 | 357 | 760 | 304 | 209 | 379 | 1,651 | 1,647 | 1,840 | 82 |
| w5-a1-voice-c6: frozen box, 6 callers, voice alone, a warm-up turn opening each call | 3 | 546 | 772 | 302 | 212 | 377 | 1,663 | 1,653 | 1,885 | 84 |
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call | 10 | 590 | 825 | 3,881 | 250 | 343 | 5,299 | 4,608 | 10,609 | 2,795 |
| w5-disc-a8c-resubmit-s24: frozen box + preempt-resubmit before the first byte, discriminating scenario, a warm-up turn opening each call | 5 | 295 | 822 | 673 | 232 | 357 | 2,084 | 2,006 | 2,717 | 275 |
| w5-disc-pr-a-s24: vLLM #63350: make-room's KV preemption alone, no gateway policy, a warm-up turn opening each call | 5 | 295 | 820 | 1,990 | 250 | 356 | 3,415 | 3,312 | 5,290 | 771 |
| w5-disc-pr-ab-s24: vLLM #63350 and #63351: KV preemption and the budget cap, no gateway policy, a warm-up turn opening each call | 5 | 295 | 822 | 738 | 236 | 355 | 2,151 | 1,992 | 3,090 | 289 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call | 15 | 885 | 822 | 551 | 242 | 355 | 1,971 | 1,957 | 2,268 | 282 |
| w5-disc-pr-ab48-v03-s24: vLLM #63350 and #63351 with a 48-token window as v0.3 built them, no gateway policy, a warm-up turn opening each call | 5 | 295 | 824 | 537 | 241 | 356 | 1,958 | 1,952 | 2,243 | 276 |
| w5-disc-pr-ab48-c4-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, 4 callers, a warm-up turn opening each call | 3 | 357 | 831 | 569 | 252 | 367 | 2,018 | 1,994 | 2,330 | 279 |
| w5-disc-pr-ab48-c6-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, 6 callers, a warm-up turn opening each call | 3 | 546 | 846 | 541 | 269 | 386 | 2,042 | 2,024 | 2,395 | 267 |
