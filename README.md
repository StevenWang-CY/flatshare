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

Flatshare runs a self-hosted voice agent (speech to text, an LLM, text to speech) beside long-context coding agents
on one GPU, which the voice agent needs for only a fraction of each turn. On vLLM 0.32 with 24 coding-agent sessions on
one RTX 5060 Ti, the voice agent's p95 voice-to-voice latency rises from 1.78 s alone to 10.5 s. The time is lost in
vLLM's scheduler. A waiting voice turn cannot take KV cache from running agent requests, and once it runs, agent
prefill shares every step of its reply.

Two scheduler rules, proposed to vLLM as pull requests [#65282](https://github.com/vllm-project/vllm/pull/65282) and
[#65283](https://github.com/vllm-project/vllm/pull/65283) (open, with no maintainer review as of 2026-11-09), bring that
p95 to 2.24 s against an SLO of 2.03 s. The replies over the SLO fall from 90% to 32% (286 of 885), and the agents keep
97% of their token rate.

<picture>
  <source media="(max-width: 767px) and (prefers-color-scheme: dark)" srcset="docs/figures/e2-demo-narrow-dark.webp">
  <source media="(max-width: 767px)" srcset="docs/figures/e2-demo-narrow.webp">
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/e2-demo-dark.webp">
  <img src="docs/figures/e2-demo.webp" width="100%" alt="The same caller's first measured turn on two engines beside 24 coding-agent sessions on one RTX 5060 Ti, in two panels side by side as it plays. Each panel shows the wait's clock and the wait against the 2.03 s SLO, the reply's voice-to-voice as a bar against the SLO, the caller's line and the voice agent's reply as waveforms on the turn's time axis with the wait between them, and the agents' tokens a second over the calls' first 240 s. Tuned vLLM answers after 9.6 s, over the SLO; with the configuration to deploy the reply comes after 1.9 s, within it, and the agents make 3,355 tokens a second, 98% of tuned vLLM's 3,437">
</picture>

*The same caller's first measured turn on each engine, beside the same 24 agent sessions on one RTX 5060 Ti, recorded
2026-10-25. Tuned vLLM answers after 9.6 s, and the configuration to deploy after 1.9 s, its agents at 98% of tuned
vLLM's token rate in this run ([the video](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.2)).*

## Results

<picture>
  <source media="(max-width: 767px)" srcset="docs/figures/report-headline-narrow.svg">
  <img src="docs/figures/report-headline.svg" width="100%" alt="Each run's voice-to-voice p95 on a log scale against the SLO of 2.03 s, and the agents' tokens a second over the calls' first 240 s, with each configuration's median: voice alone, no agents: 1.78 s at the median of 5 runs (1.76 to 1.82 s); tuned vLLM, the baseline: 10.48 s at the median of 10 runs (8.69 to 12.12 s), its agents at 3,305 tokens/s, 100% of tuned vLLM's; preempt-resubmit, before the first byte: 2.52 s at the median of 5 runs (2.52 to 2.87 s), its agents at 3,332 tokens/s, 101% of tuned vLLM's; the KV preemption, #63350: 5.09 s at the median of 5 runs (4.69 to 6.26 s), its agents at 3,414 tokens/s, 103% of tuned vLLM's; the change, 16-token window: 3.09 s at the median of 5 runs (2.87 to 3.19 s), its agents at 3,298 tokens/s, 100% of tuned vLLM's; and the configuration to deploy, 48-token window: 2.24 s at the median of 15 runs (2.13 to 2.40 s), its agents at 3,203 tokens/s, 97% of tuned vLLM's.">
</picture>

*With two callers beside 24 coding-agent sessions on one RTX 5060 Ti, the two rules bring the voice tail to 0.20 s
over the SLO, where tuned vLLM is 8.4 s over, and the agents keep 97% of their token rate. Each tick is a run and each
marker the median of a configuration's runs. The dashed line is the SLO, voice alone's p95 plus 250 ms
([the report's Figure 8](docs/REPORT.md#61-the-two-rules-bring-the-voice-tail-to-224-s-while-the-agents-keep-97-of-their-throughput)).*

The KV preemption's row runs the first pull request alone and the change's rows run both; the measurements name them by
their first numbers, #63350 and #63351. Each row is the median of five runs, or 10 for tuned vLLM and 15 for the last
([the runs](docs/BENCHMARKS.md#how-close-does-each-configuration-bring-the-voice-tail-to-its-slo)). Tuned vLLM runs priority scheduling, a
512-token step cap and a CPU KV tier. The SLO allows 5% of the replies over it. The configuration's agents waited 72 s
for their first token at the median, against tuned vLLM's 63 s ([every
measure](docs/REPORT.md#61-the-two-rules-bring-the-voice-tail-to-224-s-while-the-agents-keep-97-of-their-throughput)).

The card serves Qwen3-4B-Instruct-2507-FP8 with a 36,400-token KV pool and a 20 GiB CPU KV tier, beside Whisper and
Kokoro. LiveKit callers read voice-to-voice latency from the waveforms, and AIPerf replays 24 sessions of SemiAnalysis'
Claude Code traces, whose contexts fill the pool, through the gateway. The gateway adds 0.12 ms to a voice request's
first token at the median ([the record](benchmarks/results/2026-10-29-gateway-window/README.md)), and
`make reproduce-e2-w5` rebuilds these numbers from the committed records ([the setup](docs/REPORT.md#5-evaluation-setup)).

The deployment's serving profile adds to the two rules, measured in rounds on new samples of the agents:

- **At its 256-token step cap**, 44 of 295 replies were over the SLO in five rounds, against 94 of 295 at 512 tokens
  and 261 of 295 on tuned vLLM
  ([§6.9](docs/REPORT.md#69-a-256-token-step-cap-halves-the-changes-replies-over-the-slo-beside-24-sessions)).
- **With the engine's additions and version 1.24's agent queue and copy kernel**, one card holds 24 agent sessions
  beside both calls within the SLO's allowance. In three rounds 3 of 177 replies were over the SLO, and the agents
  made 2.2 times the tokens a second of the four sessions the card held before
  ([§6.10](docs/REPORT.md#610-the-serving-engines-additions-cut-the-replies-over-the-slo-by-more-than-half),
  [§6.11](docs/REPORT.md#611-ordering-the-agents-by-what-they-have-left-raises-the-cards-envelope-from-4-sessions-to-24)).
  The queue and the copy kernel are proposed to vLLM as drafts,
  [#66558](https://github.com/vllm-project/vllm/pull/66558) and [#66527](https://github.com/vllm-project/vllm/pull/66527).
- **On an A100** at the same KV pool and 512-token steps, the configuration to deploy takes the p95 from 4.99 s to
  1.63 s, under that card's SLO in each of five rounds, with the agents at 70% of their throughput at vLLM's default
  2,048-token steps ([§6.7](docs/REPORT.md#67-the-rules-carry-to-an-a100)).

## How it works

<picture>
  <source media="(max-width: 767px)" srcset="docs/figures/method-narrow.svg">
  <img src="docs/figures/method.svg" width="100%" alt="Panel a: a caller's speech goes to speech to text, then to vLLM 0.32's scheduler, which carries #63350 and #63351, then to text to speech and back to the caller; 24 coding-agent sessions send their requests to the same scheduler at priority 2, the voice turn's at priority 0; the scheduler allocates from the KV pool beneath it. Panel b, the KV preemption, #63350: in a pool of 12 KV blocks, two agent requests hold 5 each and 2 are free, fewer than the 4 a waiting voice turn needs; on vLLM 0.32 the turn keeps waiting, while the KV preemption preempts the latest agent request, agent B, and the turn takes 4 of its blocks. Panel c, the budget cap, #63351: the engine's steps in time, one box a step, from the reply's first token: on vLLM 0.32 each step of the reply also carries an agent's prefill chunk, so in the time the budget cap gives the reply its first 48 tokens the reply gets 14 without it; after the window, agent prefill resumes. Blocks and steps are illustrative.">
</picture>

*The voice turn and the agents share vLLM's scheduler and KV pool (a). The KV preemption, #63350, makes room in the pool
(b), and the budget cap, #63351, in the reply's first steps (c). Blocks and steps are illustrative.*

1. **vLLM 0.32 cannot preempt for a waiting request.** It preempts only so that a running request can grow. Behind two
   18k-token agent requests in a full pool, a 750-token voice turn waits 4.9–5.0 s for its first token, and 219–290 ms
   with #63350 ([the KV case](docs/BENCHMARKS.md#what-does-each-rule-remove)).
2. **Admitting the turn is not enough.** Beside three agent prefills the reply's tokens arrive 60–65 ms apart with
   #63350 alone, and 16–20 ms apart once #63351 holds the prefill back
   ([the prefill case](docs/BENCHMARKS.md#what-does-each-rule-remove)).
3. **The budget cap's window must cover the reply's first sentence**, which the voice agent hands to text to speech
   whole. Its first sentences run 15 tokens at the median and 39 at most. A 16-token window leaves the p95 at 3.1 s, and
   48 tokens cover them all ([the first sentences](benchmarks/results/2026-10-27-first-sentence/README.md)).
4. **A control plane in front of the engine can make room only by aborting work.** llm-d, Dynamo, SMG and
   vllm-project/router act there. Preempt-resubmit, which aborts and resubmits agent requests, comes nearest the SLO at
   2.52 s, at the cost of the agents' tail
   ([§6.5](docs/REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work)).

## Limits

- **The two rules help most where agents' KV fills the pool.** Beside 16 sessions of the whole trace corpus, where the
  pool has room, the configuration to deploy met the SLO in one of five rounds
  ([§6.6](docs/REPORT.md#66-the-gain-shrinks-where-the-pool-has-room-and-the-tail-grows-as-calls-overlap)).
- **The served model failed the quality gate** for a useful pair of background jobs, so a deployment has to check its
  own model first ([the gate](benchmarks/quality/GATE.md)).
- **The two rules alone leave 32% of the replies over the SLO**, whose allowance is 5%, and meeting it takes the
  serving profile's additions.
- **The agents pay in their tail**, with a time to first token of 133 s at p99 beside 24 sessions.
- **Two GPU models and up to six callers were measured**; the simulator places other cards by a model.

[The report's limitations](docs/REPORT.md#9-limitations) give each in full.

## Quick start

On a host with a GPU that vLLM 0.32.0 supports, install it and build the make-room variant, vLLM 0.32.0's scheduler
with the two pull requests backported, without modifying the installation ([uv](https://docs.astral.sh/uv/) fetches
Python 3.12 where the host's is older):

```sh
# vLLM 0.32.0 needs Python 3.11 to 3.14
uv venv --python 3.12 --seed ~/vllm-0.32 && . ~/vllm-0.32/bin/activate
# CUDA 13.0's compiler, matching torch's headers: vLLM compiles its FP8 kernels with it at start
pip install vllm==0.32.0 'cuda-toolkit[nvcc]==13.0.3.0'
export CUDA_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13
# FlashInfer's JIT links -lcudart
export FLASHINFER_EXTRA_LDFLAGS=-L$CUDA_HOME/lib && \
  ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
git clone https://github.com/StevenWang-CY/flatshare && cd flatshare
python python/flatshare_vllm/src/flatshare_vllm/build.py make-room ~/make-room
```

Serve stock vLLM with the serving profile's settings. Once it is up, run the probe from a second shell in the clone. It
sends a voice turn behind long agent requests that fill the KV pool.

```sh
# Without --max-model-len vLLM takes the model's 262,144 tokens, which no 16 GB card holds; without
# --max-num-seqs, 1,024 request slots on a card of 70 GiB or more, more than the step cap allows.
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 256 \
  --scheduling-policy priority --max-num-batched-tokens 256 --long-prefill-token-threshold 0
# In a second shell, in the clone, once the engine is up:
. ~/vllm-0.32/bin/activate && \
  python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv
```

Stop the engine, then serve the same settings with make-room and run the probe again:

```sh
PYTHONPATH=~/make-room vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --max-model-len 32768 --max-num-seqs 256 \
  --scheduling-policy priority --max-num-batched-tokens 256 --long-prefill-token-threshold 0 \
  --scheduler-cls flatshare_make_room_scheduler.AsyncScheduler \
  --additional-config.make_room.max_preemptions_per_step 2 \
  --additional-config.make_room.budget_cap true --additional-config.make_room.budget_cap_tokens 48 \
  --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.take_reservations true \
  --additional-config.make_room.decode_hold_tokens 48 \
  --additional-config.make_room.retain_tokens 3584 --additional-config.make_room.retain_s 30 \
  --additional-config.make_room.hint_hold_ms 400
. ~/vllm-0.32/bin/activate && \
  python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv
```

To run the probe again on make-room, wait 30 s first, since the engine keeps the last voice turn's prefix on the GPU
that long (`retain_s`) and the probe sizes its load for an empty pool.

<!-- quick-start probe -->
The probe prints the voice turn's time to its first token as `priority-0 TTFT`. On Phastform, with the card's own KV
pool, it was 1,543–1,556 ms on stock vLLM and 173–179 ms with make-room, in four runs each
([stock](benchmarks/results/2026-11-02-quick-start/stock-kv.log),
[make-room](benchmarks/results/2026-11-02-quick-start/make-room-kv.log),
[the record](benchmarks/results/2026-11-02-quick-start/README.md)).
<!-- /quick-start probe -->

make-room runs the configuration to deploy here ([its switches](docs/PRODUCTION.md#the-engine)), and it builds on vLLM
0.33.0 too ([the port's record](benchmarks/results/2026-11-02-port-v0.33.0/README.md)). To serve beside agents, send
`priority` with each request, 0 for voice and 2 for background work, and add the fixed KV pool and the CPU KV tier of
[`engine.yaml`](deploy/phastform/engine.yaml).

## Documentation

[docs/README.md](docs/README.md) maps every document to the question it answers. The most used:

| To | Read |
|---|---|
| read the work as one paper, with its limits | [docs/REPORT.md](docs/REPORT.md) |
| deploy the change, size it and watch it | [docs/PRODUCTION.md](docs/PRODUCTION.md) |
| see how the system is built | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#at-a-glance) |
| see every number, with its runs and records | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| run it on another GPU or on one host | [docs/HARDWARE.md](docs/HARDWARE.md) |
| compare it with llm-d, Dynamo, SMG, SGLang and published work | [docs/PRIOR-ART.md](docs/PRIOR-ART.md) |
| build and test it, with `make check` and no GPU | [CONTRIBUTING.md](CONTRIBUTING.md#build-and-test) |

## Acknowledgments

Flatshare builds on [vLLM](https://github.com/vllm-project/vllm), [LiveKit](https://github.com/livekit/livekit) and
[LiveKit Agents](https://github.com/livekit/agents), and runs
[Qwen3-4B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8),
[Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) and
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI).
[AIPerf](https://github.com/ai-dynamo/aiperf) replays
[SemiAnalysis' Claude Code traces](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126), and the
[STIX Two](https://github.com/stipub/stixfonts) fonts set the figures and logo.

## Citation

If you use Flatshare or its measurements, please cite the repository ([CITATION.cff](CITATION.cff)):

```bibtex
@software{wang_flatshare_2026,
  author  = {Wang, Chuyue},
  title   = {Flatshare},
  year    = {2026},
  version = {1.25.0},
  url     = {https://github.com/StevenWang-CY/flatshare}
}
```

## Licence

Flatshare is an independent project by Chuyue (Steven) Wang. It is not affiliated with, endorsed by, or sponsored by
the vLLM project, LiveKit, or any other project or organization named in this repository. Their names identify only
the software Flatshare runs on and measures. Flatshare is released under the [Apache License 2.0](LICENSE); the software
and models it runs keep their own licences ([THIRD-PARTY.md](THIRD-PARTY.md)).
