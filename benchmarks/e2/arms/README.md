# Arm files

One file for each configuration at one load, a point of E2. [The E2 README](../README.md#arms-and-campaigns) lists the
arms and the campaigns that run them, and [the bench README](../../../python/flatshare_bench/README.md#e2-points) an arm
file's tables and keys. A record names its points after these files, so a file keeps its name once a record carries it.
The parts of a name:

| Part | Means |
|---|---|
| `a1` to `a9` | The arm's number ([the arms](../../../docs/EXPERIMENTS.md#e2-voice-under-agent-load)): 1 voice alone, 2 stock defaults, 3 priority scheduling, 4 a step cap and threshold, 5 vLLM's own knobs, 6 static answers, 7 what engines ship, 8 what routers ship, 9 make-room with the lifecycle policy |
| `disc`, `std` | The discriminating scenario, 24 sessions of the 48 traces with the largest contexts, and the standard one, the whole corpus |
| `frozen` | Tuned vLLM, the engine of the frozen box |
| `pr-a`, `pr-ab`, `pr-ab48` | The first pull request alone, both, and both with a 48-token window |
| `bN`, `tN` | A step cap of N tokens, and a long-prefill threshold of N |
| `sN`, `cN` | N agent sessions, and N callers |
| `w5-` | Written for W5: the earlier arm with a warm-up turn opening each call. Later rounds and the A100 run replay these |
| `sweep-`, `rg-` | Written for the sweep of the step cap and the window, and for the two-card comparison |
| `opt-`, `queue-` | Written for the serving engine's additions, and for the cache-aware queue of the envelope rounds |
