<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/logo-dark.svg">
    <img src="docs/brand/logo.svg" alt="Flatshare" width="400">
  </picture>
</p>

<h3 align="center">A voice agent and coding agents on one GPU</h3>

<p align="center">
  <a href="https://github.com/StevenWang-CY/flatshare/actions/workflows/ci.yml"><img src="https://github.com/StevenWang-CY/flatshare/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/StevenWang-CY/flatshare/releases/latest"><img src="https://img.shields.io/github/v/release/StevenWang-CY/flatshare?color=57606a" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-57606a" alt="License: Apache-2.0"></a>
</p>

<p align="center">
  <a href="docs/REPORT.md"><b>Technical report</b></a> |
  <a href="docs/BENCHMARKS.md"><b>Benchmarks</b></a> |
  <a href="docs/PRODUCTION.md"><b>Deployment</b></a> |
  <a href="docs/ARCHITECTURE.md"><b>Architecture</b></a> |
  <a href="https://github.com/StevenWang-CY/flatshare/releases/tag/v1.2"><b>Demo video</b></a> |
  <a href="docs/README.md"><b>Docs</b></a>
</p>

Flatshare runs a self-hosted voice agent (speech to text, an LLM, text to speech) and long-context coding agents on
the same GPU. The voice agent needs the GPU for a fraction of each turn, and the coding agents can use the rest.
On vLLM 0.32 with 24 coding-agent sessions on one RTX 5060 Ti, the voice agent's p95 voice-to-voice latency rises from
1.78 s, with voice alone, to 10.5 s. The time is lost in vLLM's scheduler: a waiting voice turn cannot take KV cache
from running agent requests, and once it runs, agent prefill shares every step of its reply.

Two scheduler changes, proposed to vLLM as pull requests [#63350](https://github.com/vllm-project/vllm/pull/63350) and
[#63351](https://github.com/vllm-project/vllm/pull/63351) (open as of 2026-10-26), bring that p95 to 2.24 s, 0.20 s over
the voice agent's SLO, while the agents keep 97% of their token rate. A third of the replies still take longer than the
SLO allows, where tuned vLLM leaves 90%: the changes narrow the gap without closing it. On an A100 SXM4 80 GB with the
same model, KV pool and sessions, they take it from 4.99 s to 1.63 s, under that card's SLO in each of five rounds,
while the agents keep their throughput. This repository holds the two changes as a variant of vLLM 0.32's scheduler, the
harness that measured them end to end, the deployment they ran on, the committed records and a technical report.

<picture>
  <source media="(max-width: 767px) and (prefers-color-scheme: dark)" srcset="docs/figures/e2-demo-narrow-dark.webp">
  <source media="(max-width: 767px)" srcset="docs/figures/e2-demo-narrow.webp">
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/e2-demo-dark.webp">
  <img src="docs/figures/e2-demo.webp" width="100%" alt="The same caller's first measured turn on two engines beside 24 coding-agent sessions on one RTX 5060 Ti, in two panels side by side as it plays. Each panel shows the wait's clock and the wait against the 2.03 s SLO, the reply's voice-to-voice as a bar against the SLO, the caller's line and the agent's reply as waveforms on the turn's time axis with the wait between them, and the agents' tokens a second over the calls' first 240 s. Tuned vLLM 0.32 answers after 9.6 s, over the SLO; with #63350 and #63351 the reply comes after 1.9 s, within it, and the agents make 3,355 tokens a second against 3,437, 98%">
</picture>

*The same caller's first measured turn on each engine, beside the same agents (recorded 2026-10-25), as it plays.
Tuned vLLM answers 9.6 s after the caller's line ends, over the SLO. With the two changes the reply comes after 1.9 s,
within it, and the agents keep 98% of their tokens a second. Over both calls of each run the V2V p95 was 11.2 and
2.2 s. The video of ten turns, with sound and a closing card for both GPUs, is attached to the
[v1.2 release](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.2).*

## Results

<picture>
  <source media="(max-width: 767px)" srcset="docs/figures/e2-voice-p95-gpus-narrow.svg">
  <img src="docs/figures/e2-voice-p95-gpus.svg" width="100%" alt="Two panels, rows in three groups beside 24 coding-agent sessions, each group against its card's SLO: the voice-to-voice p95, median of its runs, and the agents' tokens a second with each one's share of its group's first row. RTX 5060 Ti 16 GB, SLO 2.03 s: tuned vLLM 10.48 s at 3,305 tokens/s, a router that aborts and resubmits 2.52 s at 101%, #63350 and #63351 2.24 s at 97%. A100 at 512-token steps, SLO 1.67 s: tuned vLLM with the RTX 5060 Ti's tuning 4.99 s at 8,633 tokens/s, the PRs 1.63 s at 100%. A100 at 2,048-token steps: tuned vLLM, the card's best, 3.66 s at 14,536 tokens/s, the PRs 1.83 s at 101%">
</picture>

| Two calls beside 24 coding-agent sessions that fill the KV pool of one RTX 5060 Ti | Voice p95 | Over the SLO | Replies over the SLO | Agents' tokens/s | Agents' TTFT p50 |
|---|---:|---:|---:|---:|---:|
| Voice alone, the SLO's reference | 1,784 ms | — | 0.3% | — | — |
| vLLM 0.32, tuned: priority scheduling, a 512-token step cap, a CPU KV tier | 10,477 ms | +8,443 ms | 90% | 3,305 | 63 s |
| Tuned, with a router that aborts agent requests before their first byte and resubmits them | 2,522 ms | +488 ms | 44% | 3,332 | 32 s |
| + #63350: a waiting request preempts lower-priority work for KV | 5,085 ms | +3,051 ms | 89% | 3,414 | 67 s |
| + #63351: agent prefill sits out the reply's first 16 tokens | 3,094 ms | +1,060 ms | 43% | 3,298 | 71 s |
| + #63351 with a 48-token window: **the configuration to deploy** | **2,236 ms** | +202 ms | 32% | 3,203 | 72 s |

Each value is the median of five rounds. Each round replays its own sample of the agents' traces and runs every
configuration in a rotated order (10 runs for tuned vLLM and 15 for the last row, which more than one campaign ran).
The SLO is the voice-alone p95 plus 250 ms and allows 5% of the replies over it; the replies over it are counted over
all of a configuration's runs. The agents' tokens/s count the calls' first 240 s of a replay of coding agents' requests,
and nearly all of them are prompt tokens: the replay runs no task. These agents wait
about a minute for a first token, which is the regime the change is for
([is yours?](docs/PRODUCTION.md#is-your-deployment-in-this-regime)). Every run and range is in
[BENCHMARKS.md](docs/BENCHMARKS.md#voice-under-agent-load-e2).

- **The variant and the PRs.** The rows with #63350 and #63351 ran as Flatshare's make-room variant of vLLM 0.32.0
  with the PRs' switch values. It schedules as the PRs' diff did before their revision of 2026-10-25, which ends the
  window's hold one step sooner for replies at least the window long
  ([the revision](docs/BENCHMARKS.md#after-w5-the-first-tokens-in-flight)).
- **On the final code**, five more rounds on the same samples gave the last row 2.27 s, between 45 ms lower and 77 ms
  higher than the same round in the table ([the records](benchmarks/results/2026-10-25-e2-fb)).

| The same two calls and 24 sessions on an A100 SXM4 80 GB | Voice p95 | Over the SLO | Agents' tokens/s | Agents' TTFT p50 |
|---|---:|---:|---:|---:|
| Voice alone, the SLO's reference | 1,419 ms | — | — | — |
| vLLM 0.32 with the RTX 5060 Ti's tuning, 512-token steps | 4,986 ms | +3,317 ms | 8,633 | 12 s |
| + #63350 and #63351 | **1,634 ms** | −35 ms | 8,651 | 14 s |
| vLLM 0.32 at its default 2,048-token steps, the card's best | 3,663 ms | +1,994 ms | 14,536 | 1.5 s |
| + #63350 and #63351 | 1,826 ms | +157 ms | 14,635 | 1.5 s |

The A100 ran the RTX 5060 Ti's model, KV pool, CPU tier and agents on the same five samples of the agents' traces.
Values are medians of five rounds at 512-token steps and of two rounds, on the first two samples, at 2,048-token steps
([the A100 run](docs/HARDWARE.md#what-it-measured-2026-10-26)).

- **At 512-token steps** the PRs' p95 was under the card's SLO in every round, and the agents' tokens/s were 94–111%
  of tuned vLLM's round by round. With 32 sessions at the same pool, the p95 was 1.62 s against 5.22 s, in one round.
- **At vLLM's default 2,048-token steps**, which give this card's agents 1.4 times the tokens a second on the same
  samples, the PRs kept all of that throughput and halved the p95, to 0.14 and 0.17 s over the SLO. The larger steps
  add a wait for the steps already in flight when a voice turn arrives, which no scheduling rule takes back.

## How it works

<picture>
  <source media="(max-width: 767px)" srcset="docs/figures/method-narrow.svg">
  <img src="docs/figures/method.svg" width="100%" alt="The method in three panels. (a) A caller's speech goes to speech to text, then to vLLM 0.32's scheduler with #63350 and #63351, then to text to speech and back to the caller; 24 coding-agent sessions send their requests to the same scheduler at priority 2, the voice turn's at priority 0, and the scheduler allocates from the KV pool beneath it. (b) #63350: in a full pool of 12 KV blocks, two agent requests hold 5 each and 2 are free; a voice turn that needs 4 waits beside it on stock vLLM, and with #63350 the latest agent request, agent B, is preempted, its KV kept in the CPU tier for its resume, and the turn takes its blocks. (c) #63351, the engine's steps in time: without it, each step of the reply carries an agent's prefill chunk beside its token, 14 tokens in the time #63351, holding agent prefill, gives the reply its first 48; agent prefill resumes after the window">
</picture>

*The voice agent's turn and the coding agents share vLLM's scheduler and its KV pool (a). #63350 lets a waiting turn
take KV from the latest agent request (b), and #63351 keeps agent prefill out of the steps of the reply's first tokens
(c). The blocks and steps are illustrative. A step's length is drawn in the ratio of the gaps measured between the
reply's tokens beside agent prefill and with it held (60–65 and 16–20 ms).*

1. **vLLM 0.32 cannot preempt for a waiting request.** Its scheduler serves running requests first and preempts only
   so that one of them can grow, and its waiting loop stops at the first request it cannot allocate. Behind two
   18k-token agent requests in a full pool, a 750-token voice turn waits 4.9–5.0 s for its first token, and 219–290 ms
   with #63350 ([the KV case](docs/BENCHMARKS.md#the-kv-case-make-room)).
2. **Admitting the turn is not enough.** A step cap bounds a step's tokens, not its time. Beside three agent prefills
   the reply's tokens arrive 60–65 ms apart with #63350 alone and 16–20 ms apart with #63351, and #63350 alone leaves
   the voice p95 at 5.1 s.
3. **The hold must cover the reply's first sentence**, because the voice agent hands text to speech its first complete
   sentence rather than its first token. This agent's replies run 23 tokens at the median and 52 at most: with a
   16-token window the p95 is 3.1 s, and with 48 tokens it is 2.24 s.
4. **A control plane in front of the engine can make room only by aborting work.** A gateway can hold back new
   requests and abort admitted ones, but only the engine can take KV from a request that is already decoding.
   Aborting and resubmitting agent requests reaches 2.52 s at the agents' expense, their TTFT p99 rising to 193 s from
   tuned vLLM's 106 s. A lifecycle policy timed by the conversation never found headroom to give and left the p95 at
   11.0 s in an earlier campaign ([what it could not do](docs/METHODS.md#what-lifecycle-timing-could-and-could-not-do)).

## Limits and costs

- **A pool with room.** At a realistic load of the whole corpus (16 sessions), where the pool has room, #63350 has
  little to do (0–7 preemptions a run) and the cap holds agent prefill in 3–13% of the steps. The PRs' p95 was lower
  in every one of five rounds, by 109–1,413 ms, but met the SLO in only one: 2,021–2,419 ms against tuned vLLM's
  2,147–3,832 ms.
- **Many calls.** #63351 holds agent prefill while any call's first tokens are due, so overlapping replies hold it for
  longer. With 4 and 6 callers, each count against its own voice-alone SLO, the PRs' p95 exceeded it in every round, by
  0.27 s and 0.26 s at the median, against 0.20 s with two ([how it fails](docs/PRODUCTION.md#how-it-fails)). At a
  datacenter batch on the A100, 256 requests in flight with 10% at priority 0, the hold took the priority-2 requests'
  TTFT p99 from 1.1 s to 8.0 s while their median fell.
- **The agents pay through the prefill hold.** Their TTFT p50 rises by 9 s, from 63 to 72 s, and the cap holds their
  prefill in 33% of the engine's steps during the calls. The preemptions cost no measurable throughput. No vLLM 0.32
  counter shows what a preempted request recomputes when it resumes
  ([what the change takes from the agents](docs/BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)).
- **A larger step on a faster card.** The 512-token step cap that suits the RTX 5060 Ti costs the A100's agents 30%
  of their throughput. At the card's own best, 2,048-token steps, the PRs keep all of it but leave the voice p95
  0.14–0.17 s over the SLO: the LLM's first token waits for the larger steps in flight (412 and 414 ms at p95, against
  183 and 216 ms at 512-token steps on the same samples).
- **Other GPUs.** The mechanisms carry to the A100, but their costs do not scale with the card's peak figures. A
  forecast written before that run placed the KV case's first token at 1.33–1.37 s on stock vLLM and 126–145 ms with
  the PRs. The card measured 1.75 s and 175 ms, because its decode attention reads KV at 42% of its bandwidth, where
  the RTX 5060 Ti's reaches 77% ([the A100 run](docs/HARDWARE.md#what-it-measured-2026-10-26)). Beside tuned vLLM at
  the RTX 5060 Ti's pool, two calls found KV short 8% of the time on the RTX 5060 Ti and 6% on the A100, where the
  simulator gives 17% and 10% ([the measurement](docs/BENCHMARKS.md#the-boxs-running-share-rs)). Other cards are
  placed by the same model ([what transfers](docs/METHODS.md#what-transfers-to-other-hardware)).

## Quick start

The make-room variant builds vLLM 0.32.0's scheduler with the two PRs, backported, from an installed vLLM, without
modifying the installation. In an environment with vLLM 0.32.0 and a GPU it supports:

```sh
git clone https://github.com/StevenWang-CY/flatshare && cd flatshare
python python/flatshare_vllm/src/flatshare_vllm/build.py make-room ~/make-room
PYTHONPATH=~/make-room vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --scheduling-policy priority --max-num-batched-tokens 512 --long-prefill-token-threshold 256 \
  --scheduler-cls flatshare_make_room_scheduler.AsyncScheduler \
  --additional-config.make_room.max_preemptions_per_step 2 \
  --additional-config.make_room.budget_cap true --additional-config.make_room.budget_cap_tokens 48 \
  --additional-config.make_room.reserve_tokens 0
python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv   # a voice turn behind two long agent requests
```

Send `priority` with each request, 0 for voice and 2 for background work, and add the CPU KV tier from
[`engine.yaml`](deploy/phastform/engine.yaml). `probe.py` sizes its requests from the engine's KV pool and context
length, so the check runs unchanged on another card. [docs/PRODUCTION.md](docs/PRODUCTION.md) has the whole
configuration (the CPU tier, the priorities, what to watch and how it fails) and how to check it on your hardware.

## How it was measured

- **The RTX 5060 Ti** (the docs call this machine "the box"). One RTX 5060 Ti 16 GB runs vLLM 0.32 with
  Qwen3-4B-Instruct-2507-FP8 and a 36,400-token KV pool, Whisper and Kokoro on the same GPU, a 20 GiB CPU KV tier, and
  NUMA-pinned units.
- **The A100.** The same deployment, rendered from a hardware profile onto an A100 SXM4 80 GB on Google Colab, with
  the control plane on the same host and the RTX 5060 Ti's KV pool kept. A forecast of each measurement was written
  before the run ([the A100 run](docs/HARDWARE.md#the-a100-run)).
- **Voice-to-voice from waveforms.** Two LiveKit callers speak scripted lines, synthesized in six voices, to a stock
  LiveKit voice agent, and time the end of their speech to the agent's first audio. The agent logs each turn's stages.
- **The agents.** AIPerf replays the 48 longest-context Claude Code sessions of SemiAnalysis' traces, block-scaled 32×,
  with their think times, through the gateway at a lower priority.
- **Rounds.** Every configuration runs in each round, in a rotated order, on the round's own sample of the agents, and
  each call opens with an unmeasured warm-up turn. Baselines are configured as their documentation recommends.
- **Reproducibility.** `make reproduce-e2` rebuilds every table and figure from the committed records. The gateway
  adds 0.29 ms to a voice request's first token, paired against the engine's own socket on the final transport
  ([the gateway's cost](docs/BENCHMARKS.md#the-control-plane-flatshare-gateway-in-front-of-the-engine)).

## Documentation

[docs/README.md](docs/README.md) maps every document. The most used:

| To | Read |
|---|---|
| read the work as one paper: the problem, the change, the results and their limits | [docs/REPORT.md](docs/REPORT.md), the technical report |
| deploy the change and watch it | [docs/PRODUCTION.md](docs/PRODUCTION.md) |
| see how it works in vLLM's scheduler | [docs/ARCHITECTURE.md §2](docs/ARCHITECTURE.md#2-the-change-63350-and-63351), then the two PR diffs |
| see every number, with its runs and ranges | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| know how each experiment was designed and run | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and [docs/METHODS.md](docs/METHODS.md) |
| compare with llm-d, Dynamo, SMG, SGLang and published work | [docs/PRIOR-ART.md](docs/PRIOR-ART.md) |
| run it on other hardware or on one host | [docs/HARDWARE.md](docs/HARDWARE.md) |
| operate the hosts, and read what the failure drills showed | [ops/RUNBOOK.md](ops/RUNBOOK.md) |
| look up a term (the box, arm 9, the window, W0 to W8) | [docs/GLOSSARY.md](docs/GLOSSARY.md) |

## Related systems

llm-d, Dynamo, SMG and vllm-project/router act in front of the engine, and the voice turn loses its time inside the
engine's scheduler. The end-to-end experiment, E2, measured SMG, llm-d's flow control (emulated), and vLLM's and
SGLang's own priority preemption: none meets the SLO, and most leave the p95 between 8.8 and 12.1 s. What Flatshare
adds is the change to vLLM's scheduler, its end-to-end measurement against every deployable alternative, and two
measured negative results: a gateway policy timed by the conversation (E2) and agent-side KV prefetch (E1). A separate
draft for SGLang, [#45686](https://github.com/sgl-project/sglang/pull/45686), proposes the counterpart of #63350.
`flatshare-gateway` is a small Rust harness for these experiments, not a fleet router.
[docs/PRIOR-ART.md](docs/PRIOR-ART.md) has the sources and where each result went upstream.

## Repository layout

```text
crates/flatshare-proto/   wire types and their JSON Schemas
crates/flatshare-core/    cost and step-time models, KV geometry, the admission policies E2 compares
crates/flatshare-gateway/ the gateway that sets priorities and runs E2's admission policies
crates/flatshare-sim/     the KV cache simulator behind E1
python/flatshare_vllm/    the change on vLLM 0.32: the PRs as patches, the build, probes and tests
python/flatshare_bench/   the E2 harness, trace conversion, probes, drills, figures and the logo
python/flatshare_voice/   the voice agent and the measurement callers
python/flatshare_stats/   the statistics the measurements publish, each defined once
benchmarks/               E2's arm files and campaigns, the committed records, and the A100 run
deploy/                   per-host bootstrap, hardware profiles, systemd units, the engine variants' drop-ins, tunnels,
                          single-host mode
ops/                      the runbook, the failure drills, Prometheus rules and a Grafana dashboard
docs/                     the documents above, ADRs, figures and the logo
```

## Development

Development needs Rust 1.98, uv, Python 3.12 and GNU Make 3.81 or newer (macOS's own will do), and no GPU.

```sh
make check                           # lint and the tests of CI's main job
make test-make-room-ci               # CI's make-room job (Linux x86-64): the change's behavior suite and its trace
                                     # against the PRs' patches, on vLLM's real scheduler
make perf-make-room TORCH_PYTHON=PY  # the change's cost a scheduler step on CPU, its switches off against stock vLLM
                                     # and on against off, against peacock's baseline (PY: an absolute path)
make compat-upstream                 # the PRs against vLLM's main, make-room's patches against vLLM's newest release
make reproduce-e2                    # E2's tables and figures from the committed records (W4)
make reproduce-e2-w5                 # the same for W5's records
make reproduce-e2-a100               # the same for the A100 run's records
make reproduce-a100 RECORDS=DIR CORPUS=DIR
                                     # the A100 run's analysis from its raw records (v1.1's asset)
make reproduce-e1 CORPUS=DIR         # E1's tables with flatshare-sim, from the 32x corpus
make forecast-a100 CORPUS=DIR        # flatshare-sim's forecast for the A100 run, benchmarks/a100/forecast.md
make figures brand                   # the docs' figures, and the logo and social preview, from their generators
```

## Acknowledgments

Flatshare builds on the work of others:

- [vLLM](https://github.com/vllm-project/vllm) serves its LLM and its speech-to-text model, and its scheduler is what
  the two changes modify.
- [LiveKit](https://github.com/livekit/livekit) and [LiveKit Agents](https://github.com/livekit/agents) carry the calls
  and run the voice agent.
- [Qwen3-4B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8),
  [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) and
  [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), served by
  [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI), are the models it runs.
- [AIPerf](https://github.com/ai-dynamo/aiperf) replays the coding agents from
  [SemiAnalysis' Claude Code traces](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126).
- The [STIX Two](https://github.com/stipub/stixfonts) fonts, under the SIL Open Font License, set its figures and logo.

## Citation

If you use Flatshare or its measurements, please cite the repository ([CITATION.cff](CITATION.cff)):

```bibtex
@software{wang_flatshare_2026,
  author  = {Wang, Chuyue},
  title   = {Flatshare},
  year    = {2026},
  version = {1.5.0},
  url     = {https://github.com/StevenWang-CY/flatshare}
}
```

## Name and license

The project was called Leaseholder through v1.1, after its first design, in which the gateway held leases on sessions'
KV. E1 and E2 showed that the engine had to carry the realtime tier, and that design was dropped
([ADR-0006](docs/adr/0006-the-engine-change-carries-the-realtime-tier.md)). Records from before the rename keep the old
names ([docs/GLOSSARY.md](docs/GLOSSARY.md)).

Flatshare is an independent project by Chuyue (Steven) Wang. It is not affiliated with, endorsed by, or sponsored by
the vLLM project, LiveKit, or any other project or organization named in this repository. Their names identify only
the software Flatshare runs on and measures. Flatshare is released under the [Apache License 2.0](LICENSE).
