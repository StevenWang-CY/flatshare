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

Two scheduler rules, proposed to vLLM as pull requests [#63350](https://github.com/vllm-project/vllm/pull/63350) and
[#63351](https://github.com/vllm-project/vllm/pull/63351) (open, with no maintainer review as of 2026-10-29), bring that
p95 to 2.24 s against an SLO of 2.03 s. The replies over the SLO fall from 90% to 32% (286 of 885), and the agents keep
97% of their token rate.

- **At the 256-token step cap the deployment runs**, 44 of 295 replies were over their round's SLO in five rounds on
  new samples of the agents, against 94 of 295 at 512 tokens and 261 of 295 on tuned vLLM.
- **On an A100** at the same KV pool and 512-token steps, the configuration to deploy takes the p95 from 4.99 s to
  1.63 s, under that card's SLO in each of five rounds.

<picture>
  <source media="(max-width: 767px) and (prefers-color-scheme: dark)" srcset="docs/figures/e2-demo-narrow-dark.webp">
  <source media="(max-width: 767px)" srcset="docs/figures/e2-demo-narrow.webp">
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/e2-demo-dark.webp">
  <img src="docs/figures/e2-demo.webp" width="100%" alt="The same caller's first measured turn on two engines beside 24 coding-agent sessions on one RTX 5060 Ti, in two panels side by side as it plays. Each panel shows the wait's clock and the wait against the 2.03 s SLO, the reply's voice-to-voice as a bar against the SLO, the caller's line and the agent's reply as waveforms on the turn's time axis with the wait between them, and the agents' tokens a second over the calls' first 240 s. Tuned vLLM answers after 9.6 s, over the SLO; with the configuration to deploy the reply comes after 1.9 s, within it, and the agents make 3,355 tokens a second against 3,437, 98%">
</picture>

*The same caller's first measured turn on each engine, beside the same agents (recorded 2026-10-25). Tuned vLLM answers
after 9.6 s, and the two changes after 1.9 s
([the video](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.2)).*

## Results

<picture>
  <source media="(max-width: 767px)" srcset="docs/figures/e2-voice-p95-gpus-narrow.svg">
  <img src="docs/figures/e2-voice-p95-gpus.svg" width="100%" alt="Three panels, rows in three groups beside 24 coding-agent sessions, each group against its card's SLO, under the claim that the configuration to deploy brings the voice tail within 0.21 s of each card's SLO: the voice-to-voice p95, median of its runs, the agents' tokens a second with each one's share of its group's first row, and the agents' time to first token at the 99th percentile. RTX 5060 Ti 16 GB, SLO 2.03 s: tuned vLLM 10.48 s at 3,305 tokens/s and a TTFT p99 of 106 s, preempt-resubmit, which aborts agent requests before their first byte and resubmits them, 2.52 s at 101% and 193 s, the configuration to deploy, #63350 and #63351 with a 48-token window, 2.24 s at 97% and 111 s. A100 at 512-token steps, SLO 1.67 s: tuned vLLM with the RTX 5060 Ti's tuning 4.99 s at 8,633 tokens/s and 26 s, the configuration to deploy 1.63 s at 100% and 26 s. A100 at 2,048-token steps: tuned vLLM, the card's best, 3.66 s at 14,536 tokens/s and 14 s, the configuration to deploy 1.83 s at 101% and 15 s">
</picture>

| Two calls beside 24 coding-agent sessions that fill one RTX 5060 Ti's KV pool | Voice p95 | Over the SLO | Replies over the SLO | Agents' tokens/s | Agents' TTFT p50 | Agents' TTFT p99 |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone, the SLO's reference | 1,784 ms | — | 0.3% | — | — | — |
| vLLM 0.32, tuned: priority scheduling, a 512-token step cap, a CPU KV tier | 10,477 ms | +8,443 ms | 90% | 3,305 | 63 s | 106 s |
| Tuned, with a router that aborts agent requests before their first byte and resubmits them | 2,522 ms | +488 ms | 44% | 3,332 | 32 s | 193 s |
| + #63350: a waiting request preempts lower-priority work for KV | 5,085 ms | +3,051 ms | 89% | 3,414 | 67 s | 112 s |
| + #63351: agent prefill sits out the reply's first 16 tokens | 3,094 ms | +1,060 ms | 43% | 3,298 | 71 s | 105 s |
| + #63351 with a 48-token window: **the PRs' configuration** | **2,236 ms** | +202 ms | 32% | 3,203 | 72 s | 111 s |

Medians of five runs, or 10 for tuned vLLM and 15 for the last row
([the runs](docs/BENCHMARKS.md#the-discriminating-scenario-after-w5s-fixes-w5)). The SLO is voice alone's p95 plus
250 ms and allows 5% of the replies over it. The agents' tokens/s count the calls' first 240 s and are nearly all prompt
tokens.

- **At the 256-token step cap**, in the five rounds on new samples, the median p95 of the configuration to deploy was
  2,098 ms, against 2,246 ms at 512 tokens and 12,010 ms on tuned vLLM
  ([§6.9](docs/REPORT.md#69-a-256-token-step-cap-halves-the-changes-replies-over-the-slo-beside-24-sessions)).
- **On the A100**, at the RTX 5060 Ti's KV pool and 512-token step cap, the p95 of the configuration to deploy was
  7–60 ms under that card's SLO of 1.67 s, with the agents at 8,651 tokens a second against tuned vLLM's 8,633
  ([§6.7](docs/REPORT.md#67-the-rules-carry-to-an-a100)).
- **On two RTX 5060 Ti cards**, both calls and four agent sessions on one card with the configuration to deploy, and
  agents alone on the other, did 34% more background work than dedicated roles, with 5 of 177 replies over the SLO. On
  tuned vLLM the same arrangement, with two sessions, did 27% more, so most of the gain is the arrangement's
  ([the rounds](benchmarks/results/2026-10-28-rounds/README.md); [the deployment](docs/PRODUCTION.md#two-cards)).

## How it works

<picture>
  <source media="(max-width: 767px)" srcset="docs/figures/method-narrow.svg">
  <img src="docs/figures/method.svg" width="100%" alt="The method in three panels. (a) A caller's speech goes to speech to text, then to vLLM 0.32's scheduler with #63350 and #63351, then to text to speech and back to the caller; 24 coding-agent sessions send their requests to the same scheduler at priority 2, the voice turn's at priority 0, and the scheduler allocates from the KV pool beneath it. (b) #63350: in a full pool of 12 KV blocks, two agent requests hold 5 each and 2 are free; a voice turn that needs 4 waits beside it on stock vLLM, and with #63350 the latest agent request, agent B, is preempted, its KV kept in the CPU tier for its resume, and the turn takes its blocks. (c) #63351, the engine's steps in time: without it, each step of the reply carries an agent's prefill chunk beside its token, 14 tokens in the time #63351, holding agent prefill, gives the reply its first 48; agent prefill resumes after the window">
</picture>

*The voice turn and the agents share vLLM's scheduler and KV pool (a). #63350 makes room in the pool (b), and #63351
in the reply's first steps (c). Blocks and steps are illustrative.*

1. **vLLM 0.32 cannot preempt for a waiting request.** Its scheduler preempts only so that a running request can grow,
   and its waiting loop stops at the first request it cannot allocate. Behind two 18k-token agent requests in a full
   pool, a 750-token voice turn waits 4.9–5.0 s for its first token, and 219–290 ms with #63350
   ([the KV case](docs/BENCHMARKS.md#the-prs-configuration-in-one-session-w5)).
2. **Admitting the turn is not enough.** A step cap bounds a step's tokens, not its time. Beside three agent prefills
   the reply's tokens arrive 60–65 ms apart with #63350 alone and 16–20 ms apart with #63351.
3. **The hold must cover the reply's first sentence**, because the voice agent hands text to speech its first complete
   sentence rather than its first token. This agent's replies run 23 tokens at the median and 52 at most, so a 16-token
   window leaves the p95 at 3.1 s.
4. **A control plane in front of the engine can make room only by aborting work.** Only the engine can take KV from a
   decoding request, so preempt-resubmit, which aborts and resubmits agent requests, reaches 2.52 s at the agents'
   expense.

## Limits and costs

- **What the shared model can do.** The agents' replay needs no answer to be right, so the results say what sharing
  costs the call, not whether the background work is worth doing. On a useful pair of jobs, questions about private
  documents in a call (QASPER) beside enriching new documents into records (Re-DocRED), the served model failed 9 of the
  11 criteria of a quality gate: 50.6% of the questions it answered were right, and 23.9% of the facts in its records,
  against 80% for each. Qwen3-8B-FP8 on a card of its own failed too ([the gate](benchmarks/quality/GATE.md);
  [its record](benchmarks/results/2026-10-27-quality/README.md)). A deployment that shares a card for background work
  has to check first that the model can do that work.
- **Still above the SLO's 5%.** Beside 24 sessions the configuration to deploy left 32% of the replies over the SLO at
  512-token steps and 14.9% at 256.
- **A pool with room.** At 16 sessions of the whole corpus, where the pool has room, the configuration to deploy met the
  SLO in one of five rounds. The serving profile's 256-token steps leave the pool room beside 24 sessions too: the
  engine preempted 0–4 requests a run there and none beside four, so at that cap the budget cap and the step cap do the
  work ([§6.9](docs/REPORT.md#69-a-256-token-step-cap-halves-the-changes-replies-over-the-slo-beside-24-sessions)).
- **Many calls.** Overlapping replies hold agent prefill for longer, and with 4 and 6 callers the p95 of the
  configuration to deploy exceeded each count's SLO in every round ([how it fails](docs/PRODUCTION.md#how-it-fails)).
- **The agents pay through the budget cap**, which keeps their prefill out of 33% of the engine's steps during the
  calls, and the 256-token steps cost them 7% of their token rate against 512. No vLLM 0.32 counter shows what a
  preempted request recomputes when it resumes
  ([the agents' cost](docs/BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)).
- **Other GPUs.** On the A100 the 512-token step cap costs the agents 30% of their throughput, and at the card's own
  best, 2,048-token steps, the configuration to deploy leaves the voice p95 0.14–0.17 s over the SLO
  ([§6.7](docs/REPORT.md#67-the-rules-carry-to-an-a100)). Other cards are placed only by the simulator, whose costs
  scaled from published peaks missed the A100's decode attention by 1.8 times
  ([what transfers](docs/METHODS.md#what-transfers-to-other-hardware)).

## Quick start

The make-room variant builds vLLM 0.32.0's scheduler with the two pull requests, backported, from an installed vLLM,
without modifying the installation. On a host with a GPU that vLLM 0.32.0 supports, install it and build the variant
([uv](https://docs.astral.sh/uv/) fetches Python 3.12 where the host's is older):

```sh
uv venv --python 3.12 --seed ~/vllm-0.32 && . ~/vllm-0.32/bin/activate   # vLLM 0.32.0 needs Python 3.11 to 3.14
pip install vllm==0.32.0 'cuda-toolkit[nvcc]==13.0.3.0'   # CUDA 13.0's compiler, matching torch's runtime headers
export CUDA_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13   # vLLM compiles its FP8 kernels at start with it
export FLASHINFER_EXTRA_LDFLAGS=-L$CUDA_HOME/lib && ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so   # FlashInfer links -lcudart
git clone https://github.com/StevenWang-CY/flatshare && cd flatshare
python python/flatshare_vllm/src/flatshare_vllm/build.py make-room ~/make-room
```

Serve stock vLLM with the serving profile's settings. Once it is up, run the probe from a second shell in the clone: it
sends a voice turn behind long agent requests that fill the KV pool.

```sh
# Without --max-model-len vLLM takes the model's 262,144 tokens, which no 16 GB card holds, and without --max-num-seqs
# 1,024 request slots on a card of 70 GiB or more whose name lacks "a100", more than the step cap, which vLLM refuses.
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 256 \
  --scheduling-policy priority --max-num-batched-tokens 256 --long-prefill-token-threshold 0
. ~/vllm-0.32/bin/activate && python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv
```

Stop the engine, then serve the same settings with make-room and run the probe again:

```sh
PYTHONPATH=~/make-room vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 256 \
  --scheduling-policy priority --max-num-batched-tokens 256 --long-prefill-token-threshold 0 \
  --scheduler-cls flatshare_make_room_scheduler.AsyncScheduler \
  --additional-config.make_room.max_preemptions_per_step 2 \
  --additional-config.make_room.budget_cap true --additional-config.make_room.budget_cap_tokens 48 \
  --additional-config.make_room.reserve_tokens 0
. ~/vllm-0.32/bin/activate && python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv
```

<!-- quick-start probe -->
The probe prints the voice turn's time to its first token as `priority-0 TTFT`. On the box, with the card's own KV pool,
it was 1,963–1,967 ms on stock vLLM and 258–292 ms with make-room, in four runs each
([stock](benchmarks/results/2026-10-29-quick-start/stock-kv.log),
[make-room](benchmarks/results/2026-10-29-quick-start/make-room-kv.log),
[the record](benchmarks/results/2026-10-29-quick-start/README.md)).
<!-- /quick-start probe -->

Both ran at the serving profile's step cap, 256 tokens a step without a per-request threshold, and make-room in the
configuration to deploy: the KV preemption at 2 a step and the budget cap with a 48-token window, no reserve. `probe.py`
sizes its requests from the engine's KV pool and context length, so the check runs unchanged on another card.

To serve beside agents, send `priority` with each request, 0 for voice and 2 for background work, and add the CPU KV
tier from [`engine.yaml`](deploy/phastform/engine.yaml), which holds the box's 32k context and fixed KV pool too. The
sweep chose that step cap for the change, and the two-card results ran it. The Results table ran 512 with a 256-token
threshold. Then read:

- [docs/PRODUCTION.md](docs/PRODUCTION.md) to configure and size it, to check that your deployment is in its regime, and
  for what to watch and how it fails;
- [ops/RUNBOOK.md](ops/RUNBOOK.md) to operate the hosts and run the failure drills;
- [docs/HARDWARE.md](docs/HARDWARE.md) to run it on another GPU or on a single host.

## How it was measured

- **The cards.** One RTX 5060 Ti 16 GB (the docs call this machine "the box") runs vLLM 0.32 with
  Qwen3-4B-Instruct-2507-FP8 and a 36,400-token KV pool, Whisper and Kokoro on the same GPU, a 20 GiB CPU KV tier, and
  NUMA-pinned units. An A100 SXM4 80 GB on Google Colab ran the same deployment, rendered from a hardware profile, with
  the control plane on the same host and the RTX 5060 Ti's KV pool ([the A100 run](docs/HARDWARE.md#the-a100-run)).
- **Voice-to-voice from waveforms.** Two LiveKit callers speak scripted lines, synthesized in six voices, to a stock
  LiveKit voice agent, and time the end of their speech to the agent's first audio. The agent logs each turn's stages.
- **The agents.** AIPerf replays the 48 longest-context Claude Code sessions of SemiAnalysis' traces, block-scaled 32×,
  with their think times, through the gateway at a lower priority.
- **Rounds.** Every configuration runs in each round, in a rotated order, on the round's own sample of the agents, and
  each call opens with an unmeasured warm-up turn. Baselines are configured as their documentation recommends.
- **Reproducibility.** `make reproduce-e2-w5` and `make reproduce-e2-a100` rebuild the table above and the A100's
  results from the committed records, the rounds' `gate.py` the step cap's and the two cards', and `make figures` every
  figure. Measured on v1.8 (5a7e31c), the gateway adds +0.12 ms to a voice request's first token at the median, paired
  against the engine's own socket over 80 requests, and none of 82 requests spent over 2 ms in its own leg
  ([the record](benchmarks/results/2026-10-29-gateway-window/README.md)).

## Documentation

[docs/README.md](docs/README.md) orders the documents for a first reading, with the question each answers. The most
used:

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
| know what a release may change, and what to change when you upgrade | [CONTRIBUTING.md](CONTRIBUTING.md#versions-and-compatibility), then [CHANGELOG.md](CHANGELOG.md) |
| look up a term | [docs/GLOSSARY.md](docs/GLOSSARY.md) |

## Related systems

- **llm-d, Dynamo, SMG and vllm-project/router act in front of the engine**, and the voice turn loses its time inside
  the engine's scheduler. In the end-to-end experiment, E2, none of SMG, llm-d's flow control (emulated), or vLLM's and
  SGLang's own priority preemption meets the SLO, and most configurations without the change leave the p95 between 8.6
  and 12.1 s ([§6.5](docs/REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work)).
- **Flatshare adds the change to vLLM's scheduler**, measured end to end against the configurations an operator can
  deploy without it: vLLM 0.32, SGLang 0.5.22, SMG and llm-d's flow control (emulated). It also measured two negative
  results, a gateway policy timed by the conversation (E2) and agent-side KV prefetch (E1). For SGLang, pull request
  [#45686](https://github.com/sgl-project/sglang/pull/45686) proposes the counterpart of #63350.
- **`flatshare-gateway` is the serving control plane** the deployment runs: priorities, bounded intake, deadlines,
  failure-inclusive accounting, and two-node placement and failover, with E2's admission policies as its harness. It is
  not a fleet router ([ADR-0007](docs/adr/0007-the-gateway-is-the-serving-control-plane.md)).

[docs/PRIOR-ART.md](docs/PRIOR-ART.md) has the sources and where each result went upstream.

## Repository layout

```text
crates/flatshare-proto/   wire types and their JSON Schemas
crates/flatshare-core/    cost and step-time models, KV geometry, the admission policies E2 compares
crates/flatshare-gateway/ the serving control plane, with E2's admission policies as its harness
crates/flatshare-sim/     the KV cache simulator behind E1
python/flatshare_vllm/    the change on vLLM 0.32: the pull requests as patches, the build, probes and tests
python/flatshare_bench/   the E2 harness, trace conversion, probes, drills, figures and the logo
python/flatshare_voice/   the voice agent and the measurement callers
python/flatshare_stats/   the statistics the measurements publish, each defined once
benchmarks/e2/            E2's arm files and campaigns, and the demo video's renderer
benchmarks/a100/          the A100 run's campaigns, scripts, analysis, forecast and rehearsal
benchmarks/quality/       the quality gate with its scripts, and the two-card resource gate
benchmarks/endpointing/   the sweep of the turn's endpointing
benchmarks/replies/       how many tokens the voice agent's first sentences take
benchmarks/audio/         speech to text and text to speech on the CPU
benchmarks/perf/          the baseline of the scheduler step's cost on CPU
benchmarks/results/       the committed records, one directory per campaign or probe
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
                                     # against the pull requests' patches, on vLLM's real scheduler
make perf-make-room TORCH_PYTHON=PY  # the change's cost a scheduler step on CPU, its switches off against stock vLLM
                                     # and on against off, against benchmarks/perf's baseline (PY: an absolute path)
make compat-upstream                 # the pull requests against vLLM's main and the backport against their heads,
                                     # make-room's patches against vLLM's newest release, hunk by hunk
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
  version = {1.8.0},
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
the software Flatshare runs on and measures. Flatshare is released under the [Apache License 2.0](LICENSE); the software
and models it runs keep their own licences ([THIRD-PARTY.md](THIRD-PARTY.md)).
