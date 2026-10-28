# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32-long, no calls

Points without calls: the agents alone on the card, each over a fixed window (`[agents] window_s`) where a point with calls is measured over its calls. No voice SLO applies, nothing is placed on a frontier and no figure is drawn; the background goodput is the measure.

### Arm 2-agents-only

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s24: the agents alone on a dedicated card, its pool in the speech servers' memory, vLLM's scheduling for the card |  | 24 | 3,815 | 3,750 | 0.30 | — | — | — | — | 74,714 / 130,348 | — | 1 | — / 0 / — | 2,730 | 70 |

### Arm 2-agents-only-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s12: the agents alone on a dedicated card at 12 session trees, its pool in the speech servers' memory |  | 12 | 7,538 | 6,909 | 0.48 | — | — | — | — | 1,854 / 37,324 | — | 1 | — / 0 / — | 2,782 | 69 |

### Arm 2-agents-only-s14

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s14: the agents alone on a dedicated card at 14 session trees, its pool in the speech servers' memory |  | 14 | 5,459 | 4,710 | 0.36 | — | — | — | — | 14,754 / 69,422 | — | 1 | — / 0 / — | 2,745 | 70 |

### Arm 2-agents-only-s16

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s16: the agents alone on a dedicated card at 16 session trees, its pool in the speech servers' memory |  | 16 | 3,804 | 3,829 | 0.27 | — | — | — | — | 44,556 / 85,236 | — | 1 | — / 0 / — | 2,730 | 70 |

### Arm 2-agents-only-s20

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s20: the agents alone on a dedicated card at 20 session trees, its pool in the speech servers' memory |  | 20 | 3,874 | 3,733 | 0.32 | — | — | — | — | 58,881 / 105,646 | — | 1 | — / 0 / — | 2,730 | 70 |

### Arm 4-agents-only

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-frozen-s24: the agents alone on the frozen box, its speech servers idle |  | 24 | 3,065 | 2,960 | 0.23 | — | — | — | — | 101,907 / 163,786 | — | 23 | — / 0 / — | 2,730 | 68 |

### Arm 4-agents-only-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-frozen-s12: the agents alone on the frozen box at 12 session trees, its speech servers idle |  | 12 | 6,700 | 6,041 | 0.42 | — | — | — | — | 5,091 / 66,809 | — | 8 | — / 0 / — | 2,782 | 68 |

### Arm 9-pr-ab48-b256-agents-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-pr-ab48-b256-s12: the agents alone on the serving card's engine at step cap 256, the speech servers idle, 12 session trees |  | 12 | 6,255 | 6,054 | 0.41 | — | — | — | — | 5,406 / 65,358 | — | 0 | — / 0 / — | 2,775 | 66 |

### Arm 9-pr-ab48-b256-agents-s16

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-pr-ab48-b256-s16: the agents alone on the serving card's engine at step cap 256, the speech servers idle, 16 session trees |  | 16 | 3,030 | 2,955 | 0.21 | — | — | — | — | 65,593 / 119,347 | — | 1 | — / 0 / — | 2,730 | 66 |

## Workload: weka-x32, 2 callers, 1 warm-up turn

The voice SLO allows a V2V p95 of 2,010 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,760 ms). Figure: `frontier-1-warm-up-turn.svg`.

### Arm 1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call |  | 0 | 0 | — | 0.00 | 1,592 | 1,760 | 1,826 | 0 of 50 | — / — | 68 | 0 | — / 0 / — | 2,580 | 56 |

## Workload: weka-x32, 1 caller, 1 warm-up turn

The voice SLO allows a V2V p95 of 1,982 ms, 250 ms above voice alone with as many callers (rg-voice-c1: 1,732 ms). Figure: `frontier-1-callers-1-warm-up-turn.svg`.

### Arm 1-c1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-voice-c1: voice alone, one caller, a warm-up turn opening each call |  | 0 | 0 | — | 0.00 | 1,589 | 1,732 | 1,781 | 0 of 25 | — / — | 59 | 0 | — / 0 / — | 2,580 | 53 |

## Workload: weka-x32-long, 1 caller, 1 warm-up turn

The voice SLO allows a V2V p95 of 1,982 ms, 250 ms above voice alone with as many callers (rg-voice-c1: 1,732 ms). Figure: `frontier-weka-x32-long-1-callers-1-warm-up-turn.svg`.

### Arm 4-c1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-c1-s12: frozen box, one caller beside 12 session trees on weka-x32-long, a warm-up turn opening each call |  | 12 | 6,688 | 5,976 | 0.42 | 2,264 | 3,622 | 5,462 | 0 of 25 | 4,711 / 66,995 | 1,387 | 7 | — / 0 / — | 2,782 | 68 |

### Arm 9-pr-ab48-b256-c1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-c1-s12: vLLM #63350 and #63351, 48-token window, step cap 256, one caller beside 12 session trees, warm-up turn |  | 12 | 6,054 | 5,687 | 0.40 | 1,912 | 2,255 | 2,314 | 0 of 25 | 5,216 / 70,233 | 377 | 1 | — / 0 / — | 2,775 | 66 |

## Workload: weka-x32-long, 2 callers, 1 warm-up turn

The voice SLO allows a V2V p95 of 2,010 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,760 ms). Figure: `frontier-weka-x32-long-1-warm-up-turn.svg`.

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call |  | 24 | 3,137 | 3,066 | 0.24 | 4,904 | 12,171 | 15,927 | 0 of 50 | 97,155 / 167,744 | 8,948 | 32 | — / 0 / — | 2,730 | 69 |

### Arm 4-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s12: frozen box, two callers beside 12 session trees on weka-x32-long, a warm-up turn opening each call |  | 12 | 6,820 | 6,344 | 0.42 | 2,115 | 6,794 | 8,112 | 0 of 50 | 4,168 / 55,881 | 4,196 | 4 | — / 0 / — | 2,782 | 67 |

### Arm 4-s2

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s2: frozen box, two callers beside 2 session trees on weka-x32-long, a warm-up turn opening each call |  | 2 | 2,037 | 1,524 | 0.30 | 1,623 | 1,875 | 1,936 | 0 of 50 | 111 / 864 | 89 | 0 | — / 0 / — | 2,790 | 59 |

### Arm 4-s4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s4: frozen box, two callers beside 4 session trees on weka-x32-long, a warm-up turn opening each call |  | 4 | 3,593 | 3,598 | 0.15 | 1,634 | 2,444 | 2,917 | 0 of 50 | 597 / 6,048 | 579 | 0 | — / 0 / — | 2,790 | 58 |

### Arm 4-s8

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s8: frozen box, two callers beside 8 session trees on weka-x32-long, a warm-up turn opening each call |  | 8 | 6,642 | 5,836 | 0.43 | 1,882 | 3,806 | 4,420 | 0 of 50 | 680 / 7,407 | 777 | 1 | — / 0 / — | 2,782 | 63 |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call |  | 24 | 2,949 | 2,843 | 0.22 | 1,967 | 2,249 | 2,363 | 0 of 50 | 115,597 / 175,399 | 382 | 50 | — / 0 / — | 2,730 | 68 |

### Arm 9-pr-ab48-b256-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s12: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 12 session trees, warm-up turn |  | 12 | 5,719 | 5,371 | 0.38 | 1,852 | 2,203 | 2,287 | 0 of 50 | 7,456 / 63,934 | 408 | 3 | — / 0 / — | 2,767 | 67 |

### Arm 9-pr-ab48-b256-s2

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s2: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 2 session trees, warm-up turn |  | 2 | 2,037 | 1,503 | 0.30 | 1,623 | 1,874 | 1,934 | 0 of 50 | 112 / 4,807 | 103 | 0 | — / 0 / — | 2,790 | 56 |

### Arm 9-pr-ab48-b256-s4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s4: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 4 session trees, warm-up turn |  | 4 | 3,615 | 3,598 | 0.16 | 1,655 | 1,886 | 1,991 | 0 of 50 | 595 / 5,130 | 213 | 0 | — / 0 / — | 2,790 | 58 |

### Arm 9-pr-ab48-b256-s8

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s8: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 8 session trees, warm-up turn |  | 8 | 6,540 | 5,682 | 0.43 | 1,797 | 2,244 | 2,535 | 0 of 50 | 675 / 6,641 | 400 | 0 | — / 0 / — | 2,790 | 61 |

### Arm 9-pr-ab48-b256-t0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn |  | 24 | 2,824 | 2,725 | 0.20 | 1,866 | 2,272 | 2,314 | 0 of 50 | 126,340 / 181,725 | 292 | 4 | — / 0 / — | 2,730 | 67 |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver. 9 points without the worker's turn lines are left out.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w5-a1-voice | 742 / 804 | 747 / 810 | 79 / 97 | 290 / 432 | 293 / 434 | 197 / 257 | 1,233 / 1,418 | 249 / 0 / 0 |
| rg-voice-c1 | 740 / 753 | 743 / 758 | 79 / 96 | 284 / 401 | 289 / 414 | 202 / 234 | 1,228 / 1,384 | 121 / 0 / 0 |
| rg-disc-frozen-c1-s12 | 807 / 819 | 813 / 825 | 452 / 1,457 | 958 / 2,296 | 978 / 2,299 | 210 / 275 | 1,978 / 3,306 | 121 / 0 / 0 |
| rg-disc-pr-ab48-b256-c1-s12 | 805 / 835 | 807 / 840 | 228 / 414 | 490 / 859 | 492 / 861 | 248 / 294 | 1,535 / 1,947 | 121 / 0 / 0 |
| w5-disc-frozen-s24 | 810 / 833 | 815 / 838 | 2,974 / 9,023 | 3,462 / 10,676 | 3,464 / 10,677 | 213 / 295 | 4,475 / 11,822 | 248 / 0 / 0 |
| rg-disc-frozen-s12 | 811 / 885 | 816 / 892 | 284 / 4,253 | 675 / 5,516 | 675 / 5,536 | 207 / 308 | 1,697 / 6,579 | 248 / 0 / 0 |
| rg-disc-frozen-s2 | 752 / 822 | 757 / 827 | 84 / 125 | 311 / 472 | 313 / 474 | 192 / 236 | 1,258 / 1,501 | 249 / 0 / 0 |
| rg-disc-frozen-s4 | 743 / 815 | 748 / 821 | 92 / 614 | 289 / 1,132 | 292 / 1,135 | 194 / 222 | 1,239 / 2,142 | 249 / 0 / 0 |
| rg-disc-frozen-s8 | 796 / 830 | 802 / 836 | 189 / 808 | 484 / 2,424 | 505 / 2,426 | 201 / 256 | 1,518 / 3,485 | 249 / 0 / 0 |
| w5-disc-pr-ab48-s24 | 810 / 836 | 815 / 841 | 256 / 415 | 526 / 730 | 536 / 752 | 252 / 298 | 1,612 / 1,895 | 248 / 0 / 0 |
| rg-disc-pr-ab48-b256-s12 | 808 / 830 | 810 / 837 | 223 / 452 | 476 / 890 | 484 / 894 | 209 / 283 | 1,502 / 1,893 | 248 / 0 / 0 |
| rg-disc-pr-ab48-b256-s2 | 756 / 818 | 760 / 826 | 81 / 129 | 306 / 467 | 308 / 469 | 196 / 247 | 1,261 / 1,501 | 249 / 0 / 0 |
| rg-disc-pr-ab48-b256-s4 | 749 / 816 | 754 / 821 | 91 / 242 | 324 / 529 | 326 / 531 | 195 / 240 | 1,285 / 1,572 | 249 / 0 / 0 |
| rg-disc-pr-ab48-b256-s8 | 766 / 832 | 773 / 838 | 145 / 442 | 411 / 818 | 413 / 828 | 203 / 280 | 1,418 / 1,881 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-b256-t0-s24 | 809 / 837 | 811 / 844 | 198 / 325 | 464 / 782 | 473 / 786 | 215 / 300 | 1,522 / 1,932 | 248 / 0 / 0 |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| rg-disc-agents-only-s24 | 1 | 1,295,451 | 32,288 | 8,576 | 8,590 | 14 | 1,282,919 |
| rg-disc-agents-only-s12 | 1 | 494,086 | 380,448 | 1,726,912 | 1,726,586 | -326 | 465,918 |
| rg-disc-agents-only-s14 | 1 | 986,018 | 37,520 | 890,800 | 890,726 | -74 | 915,057 |
| rg-disc-agents-only-s16 | 1 | 1,284,402 | 34,288 | 10,736 | 10,829 | 93 | 1,290,693 |
| rg-disc-agents-only-s20 | 1 | 1,337,685 | 15,008 | 30,304 | 30,313 | 9 | 1,334,826 |
| rg-disc-agents-frozen-s24 | 23 | 1,079,675 | 19,072 | 16,256 | 197,840 | 181,584 | 1,081,000 |
| rg-disc-agents-frozen-s12 | 8 | 576,177 | 88,368 | 1,705,120 | 1,731,769 | 26,649 | 512,947 |
| rg-disc-agents-pr-ab48-b256-s12 | 0 | 607,587 | 72,880 | 1,507,760 | 1,507,392 | -368 | 602,973 |
| rg-disc-agents-pr-ab48-b256-s16 | 1 | 1,033,093 | 15,808 | 21,536 | 21,691 | 155 | 1,037,395 |
| w5-a1-voice | 0 | 1,687 | 40,032 | 0 | — | — | 2,591 |
| rg-voice-c1 | 0 | 849 | 20,624 | 0 | — | — | 1,424 |
| rg-disc-frozen-c1-s12 | 7 | 554,518 | 93,792 | 1,806,352 | 1,835,376 | 29,024 | 513,699 |
| rg-disc-pr-ab48-b256-c1-s12 | 1 | 570,152 | 94,832 | 1,407,264 | 1,406,920 | -344 | 586,353 |
| w5-disc-frozen-s24 | 32 | 1,478,259 | 32,960 | 45,408 | 281,083 | 235,675 | 1,480,582 |
| rg-disc-frozen-s12 | 4 | 568,014 | 68,896 | 2,006,960 | 2,030,448 | 23,488 | 516,194 |
| rg-disc-frozen-s2 | 0 | 37,774 | 642,512 | 33,568 | 33,560 | -8 | 44,261 |
| rg-disc-frozen-s4 | 0 | 26,627 | 604,832 | 636,864 | 636,709 | -155 | 30,313 |
| rg-disc-frozen-s8 | 1 | 190,145 | 554,880 | 1,754,272 | 1,760,066 | 5,794 | 156,522 |
| w5-disc-pr-ab48-s24 | 50 | 984,090 | 33,904 | 38,736 | 163,256 | 124,520 | 1,006,154 |
| rg-disc-pr-ab48-b256-s12 | 3 | 580,747 | 46,592 | 1,379,552 | 1,379,903 | 351 | 547,018 |
| rg-disc-pr-ab48-b256-s2 | 0 | 37,684 | 641,648 | 34,816 | 34,808 | -8 | 44,245 |
| rg-disc-pr-ab48-b256-s4 | 0 | 26,700 | 637,712 | 606,288 | 606,140 | -148 | 30,409 |
| rg-disc-pr-ab48-b256-s8 | 0 | 116,458 | 540,656 | 1,600,352 | 1,599,961 | -391 | 99,784 |
| sweep-disc-pr-ab48-b256-t0-s24 | 4 | 939,043 | 29,856 | 40,432 | 40,422 | -10 | 935,244 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call | 1 | 59 | 752 | 297 | 201 | 351 | 1,602 | 1,592 | 1,760 | 79 |
| rg-voice-c1: voice alone, one caller, a warm-up turn opening each call | 1 | 29 | 745 | 285 | 206 | 350 | 1,586 | 1,589 | 1,732 | 78 |
| rg-disc-frozen-c1-s12: frozen box, one caller beside 12 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 29 | 803 | 1,206 | 212 | 340 | 2,561 | 2,264 | 3,622 | 598 |
| rg-disc-pr-ab48-b256-c1-s12: vLLM #63350 and #63351, 48-token window, step cap 256, one caller beside 12 session trees, warm-up turn | 1 | 29 | 807 | 532 | 241 | 345 | 1,926 | 1,912 | 2,255 | 243 |
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call | 1 | 59 | 816 | 4,907 | 226 | 353 | 6,302 | 4,904 | 12,171 | 3,727 |
| rg-disc-frozen-s12: frozen box, two callers beside 12 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 812 | 1,393 | 216 | 368 | 2,790 | 2,115 | 6,794 | 714 |
| rg-disc-frozen-s2: frozen box, two callers beside 2 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 770 | 318 | 197 | 349 | 1,633 | 1,623 | 1,875 | 87 |
| rg-disc-frozen-s4: frozen box, two callers beside 4 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 762 | 417 | 193 | 358 | 1,729 | 1,634 | 2,444 | 157 |
| rg-disc-frozen-s8: frozen box, two callers beside 8 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 789 | 761 | 208 | 345 | 2,102 | 1,882 | 3,806 | 283 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call | 1 | 59 | 819 | 547 | 248 | 358 | 1,972 | 1,967 | 2,249 | 288 |
| rg-disc-pr-ab48-b256-s12: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 12 session trees, warm-up turn | 1 | 59 | 807 | 502 | 222 | 352 | 1,884 | 1,852 | 2,203 | 238 |
| rg-disc-pr-ab48-b256-s2: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 2 session trees, warm-up turn | 1 | 59 | 769 | 311 | 200 | 353 | 1,633 | 1,623 | 1,874 | 86 |
| rg-disc-pr-ab48-b256-s4: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 4 session trees, warm-up turn | 1 | 59 | 764 | 341 | 198 | 358 | 1,661 | 1,655 | 1,886 | 111 |
| rg-disc-pr-ab48-b256-s8: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 8 session trees, warm-up turn | 1 | 59 | 783 | 463 | 216 | 363 | 1,824 | 1,797 | 2,244 | 197 |
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn | 1 | 59 | 817 | 490 | 228 | 358 | 1,894 | 1,866 | 2,272 | 210 |
