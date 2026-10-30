# One GPU, two clocks: a voice agent beside coding agents

This page told the measurements' story from one to the next, and repeated the report. Each of its parts now has one home, which the links below give: [REPORT.md](REPORT.md) has the reasoning and the results, [EXPERIMENTS.md](EXPERIMENTS.md) each experiment's design, and [BENCHMARKS.md](BENCHMARKS.md) every run.

<a id="the-box"></a>

<a id="what-an-agent-turn-costs-to-resume-e0"></a>

<a id="voice-under-agent-load-e2"></a>

<a id="how-a-point-is-measured"></a>

<a id="the-arms"></a>

<a id="where-the-pool-has-room-the-standard-scenario"></a>

<a id="where-agent-kv-fills-the-pool-the-discriminating-scenario"></a>

<a id="arm-9-make-room-with-the-lifecycle-policy"></a>

<a id="the-results"></a>

<a id="what-lifecycle-timing-could-and-could-not-do"></a>

<a id="what-transfers-to-other-hardware"></a>

<a id="two-gpus-dedicate-or-pool"></a>

<a id="how-the-runs-were-made"></a>

<a id="limitations"></a>

## Where each part went

- **The box**: [the report's §5.1](REPORT.md#51-hardware-and-software) and [the frozen box](BENCHMARKS.md#the-frozen-box).
- **What an agent turn costs to resume (E0)**: [the tier cost model](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0), and [the report's §2.2](REPORT.md#22-coding-agents-as-background-work).
- **Voice under agent load (E2)**: how a point is measured, the arms and arm 9 in [EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load); the results in [the report's §6](REPORT.md#6-results); every run in [BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2), with W4's configurations and their two charts in [its W4 section](BENCHMARKS.md#the-discriminating-scenario-in-rounds-with-arm-9-and-the-change-w4).
- **What lifecycle timing could and could not do**: [the report's §6.5](REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work).
- **What transfers to other hardware**: [the report's §6.7](REPORT.md#67-the-rules-carry-to-an-a100) and [its Appendix E](REPORT.md#e-the-forecast-for-the-a100).
- **Two GPUs: dedicate or pool**: [the report's §6.8](REPORT.md#68-on-two-cards-consolidating-both-calls-on-one-does-34-more-background-work-most-of-it-without-the-change), and the two-node design planned for E2 in [EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load).
- **How the runs were made**: [EXPERIMENTS.md](EXPERIMENTS.md#how-runs-are-made) and [the report's Appendix A](REPORT.md#a-reproducing-the-results).
- **Limitations**: [the report's §9](REPORT.md#9-limitations).
