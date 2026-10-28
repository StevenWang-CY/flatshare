# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32-long, no calls

Points without calls: the agents alone on the card, each over a fixed window (`[agents] window_s`) where a point with calls is measured over its calls. No voice SLO applies, nothing is placed on a frontier and no figure is drawn; the background goodput is the measure.

### Arm 2-agents-only

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s24: the agents alone on a dedicated card, its pool in the speech servers' memory, vLLM's scheduling for the card |  | 24 | 4,339 | 4,232 | 0.41 | — | — | — | — | 47,875 / 86,611 | — | 0 | — / 0 / — | 2,722 | 71 |

### Arm 2-agents-only-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s12: the agents alone on a dedicated card at 12 session trees, its pool in the speech servers' memory |  | 12 | 7,169 | 7,213 | 0.78 | — | — | — | — | 197 / 6,092 | — | 1 | — / 0 / — | 2,782 | 67 |

### Arm 2-agents-only-s14

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s14: the agents alone on a dedicated card at 14 session trees, its pool in the speech servers' memory |  | 14 | 9,307 | 8,282 | 0.82 | — | — | — | — | 640 / 14,742 | — | 0 | — / 0 / — | 2,782 | 67 |

### Arm 2-agents-only-s16

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s16: the agents alone on a dedicated card at 16 session trees, its pool in the speech servers' memory |  | 16 | 6,293 | 5,624 | 0.67 | — | — | — | — | 5,857 / 59,142 | — | 0 | — / 0 / — | 2,737 | 70 |

### Arm 2-agents-only-s20

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s20: the agents alone on a dedicated card at 20 session trees, its pool in the speech servers' memory |  | 20 | 4,385 | 4,203 | 0.41 | — | — | — | — | 42,093 / 71,822 | — | 2 | — / 0 / — | 2,722 | 71 |

### Arm 4-agents-only

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-frozen-s24: the agents alone on the frozen box, its speech servers idle |  | 24 | 3,756 | 3,451 | 0.34 | — | — | — | — | 65,651 / 112,125 | — | 19 | — / 0 / — | 2,722 | 68 |

### Arm 4-agents-only-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-frozen-s12: the agents alone on the frozen box at 12 session trees, its speech servers idle |  | 12 | 6,856 | 7,008 | 0.74 | — | — | — | — | 422 / 13,226 | — | 3 | — / 0 / — | 2,782 | 65 |

### Arm 9-pr-ab48-b256-agents-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-pr-ab48-b256-s12: the agents alone on the serving card's engine at step cap 256, the speech servers idle, 12 session trees |  | 12 | 7,048 | 7,189 | 0.71 | — | — | — | — | 479 / 14,596 | — | 0 | — / 0 / — | 2,782 | 65 |

### Arm 9-pr-ab48-b256-agents-s16

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-pr-ab48-b256-s16: the agents alone on the serving card's engine at step cap 256, the speech servers idle, 16 session trees |  | 16 | 3,533 | 3,347 | 0.29 | — | — | — | — | 51,104 / 82,752 | — | 0 | — / 0 / — | 2,730 | 66 |

## Workload: weka-x32, 2 callers, 1 warm-up turn

The voice SLO allows a V2V p95 of 2,000 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,750 ms). Figure: `frontier-1-warm-up-turn.svg`.

### Arm 1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call |  | 0 | 0 | — | 0.00 | 1,622 | 1,750 | 1,846 | 0 of 50 | — / — | 59 | 0 | — / 0 / — | 2,640 | 54 |

## Workload: weka-x32, 1 caller, 1 warm-up turn

The voice SLO allows a V2V p95 of 1,988 ms, 250 ms above voice alone with as many callers (rg-voice-c1: 1,738 ms). Figure: `frontier-1-callers-1-warm-up-turn.svg`.

### Arm 1-c1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-voice-c1: voice alone, one caller, a warm-up turn opening each call |  | 0 | 0 | — | 0.00 | 1,608 | 1,738 | 1,796 | 0 of 25 | — / — | 62 | 0 | — / 0 / — | 2,580 | 57 |

## Workload: weka-x32-long, 1 caller, 1 warm-up turn

The voice SLO allows a V2V p95 of 1,988 ms, 250 ms above voice alone with as many callers (rg-voice-c1: 1,738 ms). Figure: `frontier-weka-x32-long-1-callers-1-warm-up-turn.svg`.

### Arm 4-c1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-c1-s12: frozen box, one caller beside 12 session trees on weka-x32-long, a warm-up turn opening each call |  | 12 | 6,862 | 6,616 | 0.69 | 1,866 | 3,297 | 3,456 | 0 of 25 | 492 / 12,339 | 592 | 3 | — / 0 / — | 2,782 | 65 |

### Arm 9-pr-ab48-b256-c1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-c1-s12: vLLM #63350 and #63351, 48-token window, step cap 256, one caller beside 12 session trees, warm-up turn |  | 12 | 6,798 | 6,569 | 0.69 | 1,832 | 2,300 | 2,312 | 0 of 25 | 584 / 14,526 | 522 | 1 | — / 0 / — | 2,782 | 64 |

## Workload: weka-x32-long, 2 callers, 1 warm-up turn

The voice SLO allows a V2V p95 of 2,000 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,750 ms). Figure: `frontier-weka-x32-long-1-warm-up-turn.svg`.

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call |  | 24 | 3,697 | 3,399 | 0.35 | 5,276 | 9,000 | 12,009 | 0 of 50 | 64,623 / 102,275 | 6,724 | 24 | — / 0 / — | 2,722 | 69 |

### Arm 4-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s12: frozen box, two callers beside 12 session trees on weka-x32-long, a warm-up turn opening each call |  | 12 | 6,797 | 6,706 | 0.69 | 1,844 | 4,086 | 4,596 | 0 of 50 | 448 / 14,474 | 770 | 1 | — / 0 / — | 2,782 | 65 |

### Arm 4-s2

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s2: frozen box, two callers beside 2 session trees on weka-x32-long, a warm-up turn opening each call |  | 2 | 2,522 | 2,702 | 0.40 | 1,630 | 1,838 | 2,059 | 0 of 50 | 106 / 617 | 86 | 0 | — / 0 / — | 2,797 | 56 |

### Arm 4-s4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s4: frozen box, two callers beside 4 session trees on weka-x32-long, a warm-up turn opening each call |  | 4 | 3,312 | 3,374 | 0.19 | 1,629 | 1,978 | 2,538 | 0 of 50 | 250 / 5,771 | 258 | 0 | — / 0 / — | 2,790 | 60 |

### Arm 4-s8

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s8: frozen box, two callers beside 8 session trees on weka-x32-long, a warm-up turn opening each call |  | 8 | 5,817 | 5,291 | 0.67 | 1,883 | 4,405 | 5,425 | 0 of 50 | 149 / 8,373 | 865 | 1 | — / 0 / — | 2,782 | 64 |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call |  | 24 | 3,504 | 3,314 | 0.32 | 1,888 | 2,232 | 2,528 | 0 of 50 | 74,013 / 116,670 | 382 | 44 | — / 0 / — | 2,722 | 69 |

### Arm 9-pr-ab48-b256-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s12: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 12 session trees, warm-up turn |  | 12 | 7,053 | 6,826 | 0.71 | 1,811 | 2,102 | 2,279 | 0 of 50 | 516 / 14,409 | 536 | 0 | — / 0 / — | 2,782 | 65 |

### Arm 9-pr-ab48-b256-s2

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s2: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 2 session trees, warm-up turn |  | 2 | 2,528 | 2,701 | 0.40 | 1,597 | 1,838 | 1,932 | 0 of 50 | 108 / 699 | 82 | 0 | — / 0 / — | 2,797 | 55 |

### Arm 9-pr-ab48-b256-s4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s4: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 4 session trees, warm-up turn |  | 4 | 3,334 | 3,374 | 0.19 | 1,603 | 1,951 | 2,189 | 0 of 50 | 245 / 6,783 | 206 | 0 | — / 0 / — | 2,790 | 60 |

### Arm 9-pr-ab48-b256-s8

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s8: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 8 session trees, warm-up turn |  | 8 | 5,760 | 5,256 | 0.64 | 1,813 | 2,213 | 2,341 | 0 of 50 | 187 / 8,660 | 281 | 0 | — / 0 / — | 2,782 | 63 |

### Arm 9-pr-ab48-b256-t0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn |  | 24 | 3,191 | 3,185 | 0.28 | 1,851 | 2,098 | 2,340 | 0 of 50 | 82,557 / 115,531 | 338 | 4 | — / 0 / — | 2,722 | 68 |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver. 9 points without the worker's turn lines are left out.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w5-a1-voice | 745 / 768 | 749 / 774 | 77 / 96 | 290 / 437 | 294 / 440 | 219 / 243 | 1,254 / 1,428 | 249 / 0 / 0 |
| rg-voice-c1 | 742 / 762 | 744 / 766 | 77 / 98 | 277 / 421 | 279 / 424 | 204 / 254 | 1,237 / 1,426 | 121 / 0 / 0 |
| rg-disc-frozen-c1-s12 | 803 / 825 | 806 / 830 | 179 / 625 | 495 / 1,952 | 509 / 1,966 | 203 / 293 | 1,456 / 2,984 | 121 / 0 / 0 |
| rg-disc-pr-ab48-b256-c1-s12 | 802 / 829 | 806 / 835 | 191 / 558 | 416 / 958 | 421 / 968 | 220 / 272 | 1,476 / 1,998 | 121 / 0 / 0 |
| w5-disc-frozen-s24 | 816 / 844 | 822 / 849 | 2,451 / 6,858 | 3,792 / 7,667 | 3,761 / 7,334 | 210 / 303 | 4,884 / 8,369 | 248 / 0 / 0 |
| rg-disc-frozen-s12 | 802 / 831 | 808 / 837 | 208 / 802 | 521 / 2,714 | 523 / 2,716 | 202 / 272 | 1,521 / 3,800 | 249 / 0 / 0 |
| rg-disc-frozen-s2 | 757 / 811 | 762 / 812 | 84 / 119 | 308 / 446 | 310 / 468 | 194 / 280 | 1,269 / 1,497 | 249 / 0 / 0 |
| rg-disc-frozen-s4 | 746 / 807 | 751 / 811 | 91 / 254 | 294 / 593 | 302 / 596 | 210 / 269 | 1,286 / 1,657 | 249 / 0 / 0 |
| rg-disc-frozen-s8 | 791 / 825 | 795 / 828 | 162 / 885 | 521 / 2,956 | 523 / 2,983 | 247 / 308 | 1,545 / 4,067 | 249 / 0 / 0 |
| w5-disc-pr-ab48-s24 | 809 / 828 | 813 / 833 | 269 / 417 | 504 / 678 | 510 / 686 | 250 / 285 | 1,557 / 1,802 | 248 / 0 / 0 |
| rg-disc-pr-ab48-b256-s12 | 802 / 831 | 807 / 835 | 171 / 578 | 423 / 752 | 424 / 754 | 203 / 280 | 1,457 / 1,754 | 249 / 0 / 0 |
| rg-disc-pr-ab48-b256-s2 | 751 / 818 | 756 / 823 | 81 / 115 | 293 / 441 | 295 / 443 | 190 / 251 | 1,250 / 1,444 | 249 / 0 / 0 |
| rg-disc-pr-ab48-b256-s4 | 745 / 820 | 749 / 825 | 87 / 241 | 287 / 629 | 296 / 631 | 202 / 246 | 1,246 / 1,634 | 248 / 0 / 0 |
| rg-disc-pr-ab48-b256-s8 | 800 / 832 | 804 / 836 | 143 / 325 | 390 / 737 | 390 / 741 | 240 / 282 | 1,436 / 1,848 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-b256-t0-s24 | 811 / 835 | 815 / 840 | 215 / 376 | 428 / 672 | 430 / 674 | 215 / 375 | 1,460 / 1,743 | 248 / 0 / 0 |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| rg-disc-agents-only-s24 | 0 | 1,506,865 | 12,064 | 8,688 | 8,686 | -2 | 1,517,102 |
| rg-disc-agents-only-s12 | 1 | 340,570 | 1,339,456 | 815,200 | 815,017 | -183 | 349,259 |
| rg-disc-agents-only-s14 | 0 | 471,131 | 889,472 | 1,888,992 | 1,888,531 | -461 | 451,170 |
| rg-disc-agents-only-s16 | 0 | 1,097,306 | 183,680 | 927,808 | 927,581 | -227 | 1,048,592 |
| rg-disc-agents-only-s20 | 2 | 1,497,833 | 14,336 | 39,440 | 39,782 | 342 | 1,503,473 |
| rg-disc-agents-frozen-s24 | 19 | 1,297,814 | 12,928 | 3,056 | 119,443 | 116,387 | 1,315,199 |
| rg-disc-agents-frozen-s12 | 3 | 342,300 | 637,584 | 1,442,224 | 1,459,100 | 16,876 | 349,947 |
| rg-disc-agents-pr-ab48-b256-s12 | 0 | 360,419 | 502,032 | 1,629,280 | 1,628,882 | -398 | 336,958 |
| rg-disc-agents-pr-ab48-b256-s16 | 0 | 1,155,152 | 9,872 | 96,896 | 96,872 | -24 | 1,144,680 |
| w5-a1-voice | 0 | 1,707 | 39,952 | 0 | — | — | 2,511 |
| rg-voice-c1 | 0 | 843 | 21,328 | 0 | — | — | 1,472 |
| rg-disc-frozen-c1-s12 | 3 | 339,018 | 485,600 | 1,539,552 | 1,543,943 | 4,391 | 347,467 |
| rg-disc-pr-ab48-b256-c1-s12 | 1 | 359,885 | 525,056 | 1,431,328 | 1,416,934 | -14,394 | 358,281 |
| w5-disc-frozen-s24 | 24 | 1,639,188 | 17,184 | 43,568 | 231,543 | 187,975 | 1,640,351 |
| rg-disc-frozen-s12 | 1 | 361,339 | 574,304 | 1,477,776 | 1,480,742 | 2,966 | 348,811 |
| rg-disc-frozen-s2 | 0 | 25,668 | 802,624 | 33,968 | 33,960 | -8 | 31,640 |
| rg-disc-frozen-s4 | 0 | 27,680 | 798,992 | 367,232 | 367,142 | -90 | 30,712 |
| rg-disc-frozen-s8 | 1 | 194,767 | 1,020,512 | 1,022,656 | 1,029,125 | 6,469 | 172,726 |
| w5-disc-pr-ab48-s24 | 44 | 1,146,881 | 17,968 | 36,544 | 124,546 | 88,002 | 1,164,548 |
| rg-disc-pr-ab48-b256-s12 | 0 | 329,878 | 527,760 | 1,560,864 | 1,560,483 | -381 | 338,429 |
| rg-disc-pr-ab48-b256-s2 | 0 | 25,726 | 793,200 | 34,144 | 34,136 | -8 | 31,576 |
| rg-disc-pr-ab48-b256-s4 | 0 | 27,497 | 754,928 | 327,024 | 326,944 | -80 | 30,025 |
| rg-disc-pr-ab48-b256-s8 | 0 | 207,240 | 929,872 | 959,424 | 980,960 | 21,536 | 173,478 |
| sweep-disc-pr-ab48-b256-t0-s24 | 4 | 986,030 | 12,272 | 36,544 | 36,791 | 247 | 1,004,971 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call | 1 | 59 | 753 | 300 | 213 | 351 | 1,617 | 1,622 | 1,750 | 78 |
| rg-voice-c1: voice alone, one caller, a warm-up turn opening each call | 1 | 29 | 748 | 285 | 212 | 351 | 1,596 | 1,608 | 1,738 | 78 |
| rg-disc-frozen-c1-s12: frozen box, one caller beside 12 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 29 | 794 | 694 | 217 | 343 | 2,048 | 1,866 | 3,297 | 236 |
| rg-disc-pr-ab48-b256-c1-s12: vLLM #63350 and #63351, 48-token window, step cap 256, one caller beside 12 session trees, warm-up turn | 1 | 29 | 790 | 493 | 224 | 337 | 1,845 | 1,832 | 2,300 | 229 |
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call | 1 | 59 | 821 | 3,812 | 226 | 447 | 5,306 | 5,276 | 9,000 | 2,951 |
| rg-disc-frozen-s12: frozen box, two callers beside 12 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 793 | 818 | 213 | 351 | 2,175 | 1,844 | 4,086 | 275 |
| rg-disc-frozen-s2: frozen box, two callers beside 2 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 771 | 316 | 209 | 355 | 1,651 | 1,630 | 1,838 | 88 |
| rg-disc-frozen-s4: frozen box, two callers beside 4 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 759 | 353 | 222 | 348 | 1,682 | 1,629 | 1,978 | 117 |
| rg-disc-frozen-s8: frozen box, two callers beside 8 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 789 | 848 | 247 | 352 | 2,236 | 1,883 | 4,405 | 291 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call | 1 | 59 | 815 | 517 | 244 | 354 | 1,931 | 1,888 | 2,232 | 281 |
| rg-disc-pr-ab48-b256-s12: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 12 session trees, warm-up turn | 1 | 59 | 797 | 453 | 213 | 348 | 1,811 | 1,811 | 2,102 | 208 |
| rg-disc-pr-ab48-b256-s2: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 2 session trees, warm-up turn | 1 | 59 | 767 | 304 | 198 | 353 | 1,621 | 1,597 | 1,838 | 84 |
| rg-disc-pr-ab48-b256-s4: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 4 session trees, warm-up turn | 1 | 59 | 758 | 321 | 205 | 358 | 1,642 | 1,603 | 1,951 | 102 |
| rg-disc-pr-ab48-b256-s8: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 8 session trees, warm-up turn | 1 | 59 | 792 | 429 | 237 | 373 | 1,830 | 1,813 | 2,213 | 184 |
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn | 1 | 59 | 820 | 455 | 231 | 362 | 1,868 | 1,851 | 2,098 | 226 |
