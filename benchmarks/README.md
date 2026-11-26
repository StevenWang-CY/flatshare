# Benchmarks

What ran each measurement and what it recorded, for a reader who checks a number or repeats a run.
[BENCHMARKS.md](../docs/BENCHMARKS.md) gives every number by the question it answers, and
[EXPERIMENTS.md](../docs/EXPERIMENTS.md) each experiment's design and
[which campaign measured what](../docs/EXPERIMENTS.md#which-campaign-measured-what-on-which-code).

| Directory | What it holds |
|---|---|
| [`results/`](results/README.md) | The committed records, one dated directory for each measurement, with a README that says what ran, on which revision, and the command that recomputes its tables. A record is frozen once a release carries it |
| [`e2/`](e2/README.md) | E2, voice under agent load: an arm file for each configuration at one load, the campaign files that list them, and `campaign.sh`, which measures a campaign from a workstation |
| [`a100/`](a100/README.md) | The same comparison on a rented A100: the run's blocks (`run.sh`), its campaigns and arms, its analysis, the simulator's forecasts made before it, and [its rehearsal on the box](a100/rehearsal/README.md) |
| [`voice/`](voice) | Measures of the voice pipeline alone: the endpointing sweep, the length of the agent's first sentences, and speech recognition and synthesis on the CPU |
| [`quality/`](quality/GATE.md) | The quality gate, whether the served model can do a useful pair of background jobs, and [the two-card gate](quality/RESOURCE-GATE.md) |
| [`fp8/`](fp8/README.md) | The harness that reproduced the engine's crash with an FP8 KV cache on stock vLLM, and tested its cause |
| [`perf/`](perf) | The tracked baseline of the change's cost per scheduler step (`make perf-make-room`) |

The tools the scripts call are in [`python/flatshare_bench`](../python/flatshare_bench/README.md), and the probes of
the change itself in [`python/flatshare_vllm`](../python/flatshare_vllm/README.md). `make reproduce-e2`,
`make reproduce-e2-w5` and `make reproduce-e2-a100` rebuild E2's reports from the records, with no GPU.
