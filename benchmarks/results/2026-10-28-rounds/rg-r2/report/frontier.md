# E2 frontier

Background goodput (prompt and completion tokens per second of the agent requests that completed during the calls) against voice-to-voice latency in ms, by workload: the agents' corpus (`[agents] data`) and the number of callers, since points of different workloads are not compared. • marks the points on their arm's frontier in their workload, where the arm has more than one configuration. The realtime first byte is the gateway's, from its request log. Refused counts the gateway's admission point and the engine's queue limits; — is a count the point's gateway or engine does not keep. A failed turn, one the agent did not answer within the callers' reply timeout, counts as a V2V sample censored at 30,000 ms, so that ≥ marks a percentile it sets; a configuration with failed turns is not placed on the frontier. Beside the goodput over each point's whole calls, whose length differs by configuration, is its goodput over their first 240 s, the same span for every point; † marks one over a point's whole calls, which lasted less. SM clock (MHz) and temperature (°C) are the GPU's during the calls.

## Workload: weka-x32-long, no calls

Points without calls: the agents alone on the card, each over a fixed window (`[agents] window_s`) where a point with calls is measured over its calls. No voice SLO applies, nothing is placed on a frontier and no figure is drawn; the background goodput is the measure.

### Arm 2-agents-only

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s24: the agents alone on a dedicated card, its pool in the speech servers' memory, vLLM's scheduling for the card |  | 24 | 4,234 | 4,294 | 0.34 | — | — | — | — | 47,096 / 79,289 | — | 0 | — / 0 / — | 2,527 | 90 |

### Arm 2-agents-only-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s12: the agents alone on a dedicated card at 12 session trees, its pool in the speech servers' memory |  | 12 | 8,237 | 7,372 | 0.55 | — | — | — | — | 1,065 / 18,411 | — | 0 | — / 0 / — | 2,700 | 88 |

### Arm 2-agents-only-s14

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s14: the agents alone on a dedicated card at 14 session trees, its pool in the speech servers' memory |  | 14 | 7,827 | 7,893 | 0.54 | — | — | — | — | 5,102 / 40,977 | — | 0 | — / 0 / — | 2,685 | 89 |

### Arm 2-agents-only-s16

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s16: the agents alone on a dedicated card at 16 session trees, its pool in the speech servers' memory |  | 16 | 6,932 | 6,489 | 0.54 | — | — | — | — | 4,861 / 43,794 | — | 0 | — / 0 / — | 2,625 | 89 |

### Arm 2-agents-only-s20

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-only-s20: the agents alone on a dedicated card at 20 session trees, its pool in the speech servers' memory |  | 20 | 5,819 | 4,780 | 0.48 | — | — | — | — | 18,495 / 50,604 | — | 0 | — / 0 / — | 2,505 | 89 |

### Arm 4-agents-only

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-frozen-s24: the agents alone on the frozen box, its speech servers idle |  | 24 | 3,505 | 3,499 | 0.27 | — | — | — | — | 66,580 / 104,865 | — | 33 | — / 0 / — | 2,580 | 90 |

### Arm 4-agents-only-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-frozen-s12: the agents alone on the frozen box at 12 session trees, its speech servers idle |  | 12 | 6,886 | 6,302 | 0.47 | — | — | — | — | 4,412 / 31,007 | — | 5 | — / 0 / — | 2,685 | 88 |

### Arm 9-pr-ab48-b256-agents-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-pr-ab48-b256-s12: the agents alone on the serving card's engine at step cap 256, the speech servers idle, 12 session trees |  | 12 | 6,924 | 6,715 | 0.47 | — | — | — | — | 3,653 / 30,140 | — | 0 | — / 0 / — | 2,692 | 87 |

### Arm 9-pr-ab48-b256-agents-s16

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-agents-pr-ab48-b256-s16: the agents alone on the serving card's engine at step cap 256, the speech servers idle, 16 session trees |  | 16 | 5,327 | 4,135 | 0.42 | — | — | — | — | 17,954 / 58,899 | — | 1 | — / 0 / — | 2,625 | 88 |

## Workload: weka-x32, 2 callers, 1 warm-up turn

The voice SLO allows a V2V p95 of 1,972 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,722 ms). Figure: `frontier-1-warm-up-turn.svg`.

### Arm 1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call |  | 0 | 0 | — | 0.00 | 1,573 | 1,722 | 1,840 | 0 of 50 | — / — | 65 | 0 | — / 0 / — | 2,587 | 65 |

## Workload: weka-x32, 1 caller, 1 warm-up turn

The voice SLO allows a V2V p95 of 1,943 ms, 250 ms above voice alone with as many callers (rg-voice-c1: 1,693 ms). Figure: `frontier-1-callers-1-warm-up-turn.svg`.

### Arm 1-c1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-voice-c1: voice alone, one caller, a warm-up turn opening each call |  | 0 | 0 | — | 0.00 | 1,621 | 1,693 | 1,744 | 0 of 25 | — / — | 64 | 0 | — / 0 / — | 2,580 | 59 |

## Workload: weka-x32-long, 1 caller, 1 warm-up turn

The voice SLO allows a V2V p95 of 1,943 ms, 250 ms above voice alone with as many callers (rg-voice-c1: 1,693 ms). Figure: `frontier-weka-x32-long-1-callers-1-warm-up-turn.svg`.

### Arm 4-c1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-c1-s12: frozen box, one caller beside 12 session trees on weka-x32-long, a warm-up turn opening each call |  | 12 | 6,921 | 6,594 | 0.47 | 2,392 | 4,460 | 5,800 | 0 of 25 | 4,689 / 29,220 | 2,753 | 4 | — / 0 / — | 2,685 | 88 |

### Arm 9-pr-ab48-b256-c1

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-c1-s12: vLLM #63350 and #63351, 48-token window, step cap 256, one caller beside 12 session trees, warm-up turn |  | 12 | 6,437 | 6,055 | 0.44 | 1,764 | 2,140 | 2,310 | 0 of 25 | 4,778 / 37,218 | 482 | 1 | — / 0 / — | 2,685 | 86 |

## Workload: weka-x32-long, 2 callers, 1 warm-up turn

The voice SLO allows a V2V p95 of 1,972 ms, 250 ms above voice alone with as many callers (w5-a1-voice: 1,722 ms). Figure: `frontier-weka-x32-long-1-warm-up-turn.svg`.

### Arm 4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call |  | 24 | 3,477 | 3,486 | 0.28 | 5,213 | 11,167 | 11,753 | 0 of 50 | 66,877 / 108,658 | 8,479 | 32 | — / 0 / — | 2,550 | 90 |

### Arm 4-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s12: frozen box, two callers beside 12 session trees on weka-x32-long, a warm-up turn opening each call |  | 12 | 6,605 | 6,299 | 0.45 | 2,227 | 4,510 | 6,700 | 0 of 50 | 5,942 / 25,947 | 2,640 | 3 | — / 0 / — | 2,685 | 88 |

### Arm 4-s2

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s2: frozen box, two callers beside 2 session trees on weka-x32-long, a warm-up turn opening each call |  | 2 | 2,540 | 2,158 | 0.33 | 1,608 | 1,853 | 2,362 | 0 of 50 | 116 / 786 | 118 | 0 | — / 0 / — | 2,707 | 74 |

### Arm 4-s4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s4: frozen box, two callers beside 4 session trees on weka-x32-long, a warm-up turn opening each call |  | 4 | 2,584 | 2,703 | 0.17 | 1,612 | 2,338 | 2,588 | 0 of 50 | 279 / 5,120 | 376 | 0 | — / 0 / — | 2,707 | 78 |

### Arm 4-s8

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-frozen-s8: frozen box, two callers beside 8 session trees on weka-x32-long, a warm-up turn opening each call |  | 8 | 7,060 | 6,374 | 0.43 | 1,887 | 2,919 | 3,555 | 0 of 50 | 679 / 5,345 | 598 | 1 | — / 0 / — | 2,700 | 83 |

### Arm 9-pr-ab48

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call |  | 24 | 3,217 | 3,303 | 0.25 | 1,966 | 2,243 | 2,477 | 0 of 50 | 74,143 / 106,532 | 368 | 44 | — / 0 / — | 2,565 | 90 |

### Arm 9-pr-ab48-b256-s12

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s12: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 12 session trees, warm-up turn |  | 12 | 5,792 | 5,988 | 0.40 | 1,842 | 2,248 | 2,292 | 0 of 50 | 5,094 / 42,139 | 373 | 3 | — / 0 / — | 2,685 | 88 |

### Arm 9-pr-ab48-b256-s2

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s2: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 2 session trees, warm-up turn |  | 2 | 2,464 | 2,158 | 0.33 | 1,679 | 1,911 | 2,153 | 0 of 50 | 118 / 821 | 183 | 0 | — / 0 / — | 2,715 | 74 |

### Arm 9-pr-ab48-b256-s4

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s4: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 4 session trees, warm-up turn |  | 4 | 2,616 | 2,703 | 0.18 | 1,605 | 1,925 | 1,973 | 0 of 50 | 259 / 5,235 | 216 | 0 | — / 0 / — | 2,707 | 77 |

### Arm 9-pr-ab48-b256-s8

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| rg-disc-pr-ab48-b256-s8: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 8 session trees, warm-up turn |  | 8 | 6,816 | 6,459 | 0.42 | 1,744 | 2,071 | 2,241 | 0 of 50 | 662 / 5,726 | 486 | 2 | — / 0 / — | 2,700 | 82 |

### Arm 9-pr-ab48-b256-t0

| Point | | Sessions | Background tokens/s | First 240 s | Requests/s | V2V p50 | V2V p95 | V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions | Shed / refused / resubmitted | SM clock p50 | Max temp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn |  | 24 | 3,079 | 3,182 | 0.23 | 1,844 | 2,186 | 2,350 | 0 of 50 | 81,238 / 116,379 | 310 | 0 | — / 0 / — | 2,572 | 89 |

## Voice turn stages

The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request to its first token and to the first sentence handed to TTS; from the end-of-turn decision to that sentence; from that sentence to its first audio; and from the end of speech to the reply's first audio frame on the track. Events are those the worker sent the gateway, dropped from its queue and failed to deliver. 9 points without the worker's turn lines are left out.

| Point | Transcript | End of turn | LLM first token | LLM first sentence | Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w5-a1-voice | 770 / 789 | 775 / 795 | 80 / 100 | 285 / 440 | 297 / 442 | 142 / 176 | 1,217 / 1,377 | 248 / 0 / 0 |
| rg-voice-c1 | 770 / 786 | 774 / 791 | 80 / 99 | 259 / 420 | 270 / 424 | 158 / 186 | 1,207 / 1,374 | 121 / 0 / 0 |
| rg-disc-frozen-c1-s12 | 830 / 850 | 835 / 856 | 416 / 2,311 | 1,009 / 3,094 | 1,011 / 3,096 | 170 / 223 | 2,029 / 4,119 | 121 / 0 / 0 |
| rg-disc-pr-ab48-b256-c1-s12 | 822 / 845 | 825 / 850 | 208 / 503 | 433 / 840 | 435 / 843 | 168 / 214 | 1,432 / 1,862 | 121 / 0 / 0 |
| w5-disc-frozen-s24 | 842 / 866 | 848 / 876 | 2,929 / 8,493 | 3,826 / 9,719 | 3,825 / 9,746 | 170 / 256 | 4,844 / 10,857 | 248 / 0 / 0 |
| rg-disc-frozen-s12 | 830 / 849 | 834 / 857 | 326 / 2,678 | 851 / 3,150 | 870 / 3,180 | 163 / 240 | 1,861 / 4,204 | 248 / 0 / 0 |
| rg-disc-frozen-s2 | 775 / 839 | 779 / 846 | 91 / 152 | 305 / 482 | 310 / 487 | 144 / 179 | 1,246 / 1,435 | 248 / 0 / 0 |
| rg-disc-frozen-s4 | 772 / 833 | 777 / 840 | 96 / 414 | 331 / 909 | 333 / 913 | 149 / 198 | 1,260 / 1,912 | 249 / 0 / 0 |
| rg-disc-frozen-s8 | 817 / 845 | 820 / 850 | 217 / 631 | 529 / 1,530 | 539 / 1,536 | 158 / 211 | 1,496 / 2,537 | 248 / 0 / 0 |
| w5-disc-pr-ab48-s24 | 839 / 861 | 844 / 866 | 286 / 403 | 546 / 848 | 554 / 850 | 194 / 264 | 1,585 / 1,906 | 248 / 0 / 0 |
| rg-disc-pr-ab48-b256-s12 | 821 / 842 | 826 / 848 | 247 / 418 | 512 / 765 | 514 / 777 | 166 / 229 | 1,499 / 1,790 | 248 / 0 / 0 |
| rg-disc-pr-ab48-b256-s2 | 790 / 849 | 795 / 854 | 95 / 200 | 338 / 537 | 336 / 520 | 171 / 242 | 1,316 / 1,525 | 248 / 0 / 0 |
| rg-disc-pr-ab48-b256-s4 | 768 / 834 | 773 / 839 | 100 / 245 | 312 / 534 | 326 / 537 | 150 / 202 | 1,241 / 1,563 | 248 / 0 / 0 |
| rg-disc-pr-ab48-b256-s8 | 802 / 830 | 807 / 836 | 164 / 522 | 424 / 722 | 438 / 725 | 159 / 198 | 1,407 / 1,695 | 248 / 0 / 0 |
| sweep-disc-pr-ab48-b256-t0-s24 | 843 / 860 | 849 / 866 | 234 / 333 | 473 / 768 | 476 / 772 | 168 / 230 | 1,500 / 1,819 | 248 / 0 / 0 |

## The engine's prompt tokens and its CPU tier

vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts every load into the GPU pool and every store from it, resumes included, in bytes, given here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls (one falls before or after them, or the request is aborted in between). How much of a resumed request's prompt was computed again, no counter shows.

| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling | CPU-tier hits at first scheduling | Loaded from the CPU tier | Loaded beyond first scheduling | Stored to the CPU tier |
|---|---:|---:|---:|---:|---:|---:|---:|
| rg-disc-agents-only-s24 | 0 | 1,418,162 | 29,280 | 69,264 | 69,247 | -17 | 1,423,844 |
| rg-disc-agents-only-s12 | 0 | 412,879 | 676,448 | 1,788,560 | 1,788,123 | -437 | 386,770 |
| rg-disc-agents-only-s14 | 0 | 705,880 | 394,992 | 1,673,040 | 1,672,632 | -408 | 639,428 |
| rg-disc-agents-only-s16 | 0 | 848,843 | 269,888 | 1,313,664 | 1,313,343 | -321 | 825,366 |
| rg-disc-agents-only-s20 | 0 | 1,141,144 | 94,848 | 824,816 | 824,615 | -201 | 1,122,078 |
| rg-disc-agents-frozen-s24 | 33 | 1,208,630 | 27,632 | 2,288 | 214,972 | 212,684 | 1,220,550 |
| rg-disc-agents-frozen-s12 | 5 | 475,525 | 226,576 | 1,732,016 | 1,751,332 | 19,316 | 416,378 |
| rg-disc-agents-pr-ab48-b256-s12 | 0 | 430,754 | 268,128 | 1,727,488 | 1,727,066 | -422 | 424,264 |
| rg-disc-agents-pr-ab48-b256-s16 | 1 | 856,288 | 17,568 | 989,968 | 989,870 | -98 | 829,973 |
| w5-a1-voice | 0 | 1,668 | 39,664 | 0 | — | — | 2,543 |
| rg-voice-c1 | 0 | 820 | 20,704 | 0 | — | — | 1,392 |
| rg-disc-frozen-c1-s12 | 4 | 426,682 | 243,152 | 1,848,480 | 1,869,815 | 21,335 | 417,818 |
| rg-disc-pr-ab48-b256-c1-s12 | 1 | 439,967 | 187,152 | 1,545,792 | 1,554,005 | 8,213 | 378,196 |
| w5-disc-frozen-s24 | 32 | 1,593,423 | 27,936 | 48,096 | 250,787 | 202,691 | 1,601,401 |
| rg-disc-frozen-s12 | 3 | 424,167 | 240,304 | 1,752,976 | 1,766,161 | 13,185 | 394,976 |
| rg-disc-frozen-s2 | 0 | 37,635 | 695,264 | 119,824 | 119,795 | -29 | 43,061 |
| rg-disc-frozen-s4 | 0 | 28,973 | 544,672 | 370,288 | 370,198 | -90 | 33,736 |
| rg-disc-frozen-s8 | 1 | 87,666 | 425,552 | 2,071,344 | 2,071,174 | -170 | 74,270 |
| w5-disc-pr-ab48-s24 | 44 | 1,173,332 | 5,088 | 50,976 | 122,626 | 71,650 | 1,156,182 |
| rg-disc-pr-ab48-b256-s12 | 3 | 514,523 | 213,472 | 1,258,624 | 1,258,525 | -99 | 486,489 |
| rg-disc-pr-ab48-b256-s2 | 0 | 38,140 | 731,104 | 120,496 | 120,467 | -29 | 43,957 |
| rg-disc-pr-ab48-b256-s4 | 0 | 28,599 | 522,816 | 342,592 | 363,479 | 20,887 | 33,128 |
| rg-disc-pr-ab48-b256-s8 | 2 | 79,674 | 448,032 | 1,719,280 | 1,719,612 | 332 | 65,936 |
| sweep-disc-pr-ab48-b256-t0-s24 | 0 | 1,009,493 | 33,376 | 33,584 | 33,576 | -8 | 1,023,398 |

## Stage waterfall

Each configuration's voice turn as stages that follow one another, in ms, the mean over the replies of its runs: the end-of-turn decision after the end of speech (endpointing), the reply's first sentence after that decision and its first audio from TTS, as the voice worker times them, then the rest of the callers' V2V, transport and playout. The four add up to the mean V2V. The LLM's own first token is timed from its request, which under preemptive generation starts at the final transcript, before the end of the turn, so it is not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, where the tables above give the median of the runs' own. waterfall.svg stacks the four and marks the pooled median and p95.

| Configuration | Runs | Replies | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 | LLM first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w5-a1-voice: frozen box, voice alone, a warm-up turn opening each call | 1 | 59 | 778 | 303 | 139 | 358 | 1,578 | 1,573 | 1,722 | 80 |
| rg-voice-c1: voice alone, one caller, a warm-up turn opening each call | 1 | 29 | 774 | 303 | 158 | 346 | 1,581 | 1,621 | 1,693 | 80 |
| rg-disc-frozen-c1-s12: frozen box, one caller beside 12 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 29 | 834 | 1,352 | 172 | 353 | 2,711 | 2,392 | 4,460 | 631 |
| rg-disc-pr-ab48-b256-c1-s12: vLLM #63350 and #63351, 48-token window, step cap 256, one caller beside 12 session trees, warm-up turn | 1 | 29 | 817 | 476 | 171 | 347 | 1,811 | 1,764 | 2,140 | 217 |
| w5-disc-frozen-s24: frozen box, discriminating scenario (large contexts), a warm-up turn opening each call | 1 | 59 | 853 | 4,437 | 180 | 404 | 5,874 | 5,213 | 11,167 | 3,516 |
| rg-disc-frozen-s12: frozen box, two callers beside 12 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 823 | 1,280 | 173 | 339 | 2,615 | 2,227 | 4,510 | 641 |
| rg-disc-frozen-s2: frozen box, two callers beside 2 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 791 | 349 | 146 | 356 | 1,642 | 1,608 | 1,853 | 107 |
| rg-disc-frozen-s4: frozen box, two callers beside 4 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 790 | 409 | 149 | 350 | 1,698 | 1,612 | 2,338 | 137 |
| rg-disc-frozen-s8: frozen box, two callers beside 8 session trees on weka-x32-long, a warm-up turn opening each call | 1 | 59 | 806 | 641 | 159 | 361 | 1,967 | 1,887 | 2,919 | 269 |
| w5-disc-pr-ab48-s24: vLLM #63350 and #63351 with a 48-token window, no gateway policy, a warm-up turn opening each call | 1 | 59 | 846 | 580 | 201 | 359 | 1,987 | 1,966 | 2,243 | 283 |
| rg-disc-pr-ab48-b256-s12: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 12 session trees, warm-up turn | 1 | 59 | 820 | 531 | 168 | 353 | 1,873 | 1,842 | 2,248 | 254 |
| rg-disc-pr-ab48-b256-s2: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 2 session trees, warm-up turn | 1 | 59 | 800 | 344 | 178 | 365 | 1,687 | 1,679 | 1,911 | 106 |
| rg-disc-pr-ab48-b256-s4: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 4 session trees, warm-up turn | 1 | 59 | 787 | 339 | 153 | 356 | 1,636 | 1,605 | 1,925 | 116 |
| rg-disc-pr-ab48-b256-s8: vLLM #63350 and #63351, 48-token window, step cap 256, two callers beside 8 session trees, warm-up turn | 1 | 59 | 796 | 461 | 157 | 355 | 1,768 | 1,744 | 2,071 | 212 |
| sweep-disc-pr-ab48-b256-t0-s24: vLLM #63350 and #63351, 48-token window, step cap 256, threshold 0, discriminating scenario, warm-up turn | 1 | 59 | 851 | 504 | 179 | 348 | 1,882 | 1,844 | 2,186 | 238 |
