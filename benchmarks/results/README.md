# Records

The committed record of every measurement, for a reader who checks a number against what was measured. Each directory
is one measurement, named for the day it began, and its README says what ran, on which revision, and the command that
recomputes its tables. [BENCHMARKS.md](../../docs/BENCHMARKS.md) gives the numbers by the question each answers, and
[the campaign map](../../docs/EXPERIMENTS.md#which-campaign-measured-what-on-which-code) which campaign measured what.
A record is frozen once a release carries it ([CONTRIBUTING.md](../../CONTRIBUTING.md#changes)).

## Voice under agent load (E2)

| Record | What it holds |
|---|---|
| [`2026-10-23-e2`](2026-10-23-e2/README.md) | W2 to W4: 129 points, every arm an operator can deploy, the lifecycle policy and the change |
| [`2026-10-24-e2-w5`](2026-10-24-e2-w5/README.md) | W5: 72 points of the fixed make-room and the configuration to deploy, in rounds |
| [`2026-10-25-e2-fb`](2026-10-25-e2-fb/README.md) | FB: the configuration to deploy on the code of 25 October, five rounds |
| [`2026-10-26-e2-rs`](2026-10-26-e2-rs/README.md) | RS: how often running agents leave two calls short of KV, five rounds |
| [`2026-10-27-sweep`](2026-10-27-sweep/README.md) | SW: the step cap, the long-prefill threshold and the window, screened once each |
| [`2026-10-28-rounds`](2026-10-28-rounds/README.md) | RG: the arrangements of two cards, and the step cap confirmed in five rounds |
| [`2026-11-02-opt-rounds`](2026-11-02-opt-rounds/README.md) | OR and OS: the serving engine's additions against v1.14's configuration |
| [`2026-11-08-envelope`](2026-11-08-envelope/README.md) | The cache-aware queue and the SM copy, and how many sessions one card takes beside both calls |

## The change in isolation

| Record | What it holds |
|---|---|
| [`2026-10-23-make-room-probes`](2026-10-23-make-room-probes/README.md) | The KV case, the prefill case and the calls' first-token tail on Gray, before W5's fixes |
| [`2026-10-24-w5-probes`](2026-10-24-w5-probes/README.md) | The same probes on Phastform after W5's fixes |
| [`2026-10-25-window-knee`](2026-10-25-window-knee/README.md) | The budget cap's window swept from 16 to 64 tokens |
| [`2026-10-27-first-sentence`](2026-10-27-first-sentence/README.md) | How many tokens the voice agent's first sentences take |
| [`2026-11-01-engine-points`](2026-11-01-engine-points/README.md) | What each of the serving engine's additions does to a call's reply, traced step by step |
| [`2026-11-04-r3-probes`](2026-11-04-r3-probes/README.md) | The probes on the pull requests' revision of 3 November |
| [`2026-11-05-liveness`](2026-11-05-liveness/README.md) | The search for stalls of the budget cap and the KV preemption |
| [`2026-10-27-compat`](2026-10-27-compat/README.md) | Whether the pull requests still merge into vLLM's main, and the patches' fit to its next release |
| [`2026-11-02-port-v0.33.0`](2026-11-02-port-v0.33.0/README.md) | make-room ported to vLLM 0.33.0 and checked on the box |
| [`2026-11-02-quick-start`](2026-11-02-quick-start/README.md) | The README's quick start, run as written on the box |
| [`2026-10-29-quick-start`](2026-10-29-quick-start/README.md) | The quick start as version 1.8 gave it, which the run above replaces |

## Another GPU

| Record | What it holds |
|---|---|
| [`2026-10-26-a100`](2026-10-26-a100/README.md) | The A100 run: the main comparison like for like on a rented A100 |
| [`2026-10-25-marlin-dry`](2026-10-25-marlin-dry/README.md) | The A100's code path, dry-run on the box before the rented hours |
| [`2026-10-25-a100-rehearsal`](2026-10-25-a100-rehearsal/README.md) | The A100 kit rehearsed end to end on the box |
| [`2026-10-26-a100-rehearsal`](2026-10-26-a100-rehearsal/README.md) | The same rehearsal on the release's kit |

## The deployment in operation

| Record | What it holds |
|---|---|
| [`2026-10-25-drills`](2026-10-25-drills/README.md) | The failure drills: an engine kill, and a gateway reload, restart and kill |
| [`2026-10-28-node-failover`](2026-10-28-node-failover/README.md) | The two-node failover drills |
| [`2026-10-28-two-node-smoke`](2026-10-28-two-node-smoke/README.md) | Two 15-minute soaks of the two-node arrangement before the long one |
| [`2026-10-28-soak`](2026-10-28-soak/README.md) | The 24-hour soak of the arrangement to deploy |
| [`2026-10-29-status-send`](2026-10-29-status-send/README.md) | A free line offered to the next call at once: the worker's status sent when a call ends, before and after |
| [`2026-10-25-w6-transport`](2026-10-25-w6-transport/README.md) | The gateway and the transport in front of the engine: Unix sockets through an SSH tunnel, against TCP forwards |
| [`2026-10-29-gateway-window`](2026-10-29-gateway-window/README.md) | What the gateway adds to a voice request on the deployed build |
| [`2026-10-22-agentic-api-http-baseline`](2026-10-22-agentic-api-http-baseline/README.md) | What an API layer adds to an agent's turn: agentic-api's own benchmark against the box |

## The voice pipeline and the model

| Record | What it holds |
|---|---|
| [`2026-10-28-endpointing`](2026-10-28-endpointing/README.md) | The endpointing sweep: shorter silence against lines answered too soon |
| [`2026-10-27-cpu-audio`](2026-10-27-cpu-audio/README.md) | Speech recognition and synthesis on the CPU, and what that would cost the calls |
| [`2026-10-27-quality`](2026-10-27-quality/README.md) | The quality gate's screen of the served model on a useful pair of jobs |
| [`2026-10-31-fp8-crash`](2026-10-31-fp8-crash/README.md) | Why the engine crashed with an FP8 KV cache beside the CPU tier |
