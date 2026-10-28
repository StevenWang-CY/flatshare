# Benchmarks

Every measurement behind the other documents, run by run, with its range and the records it comes from: the home of each number, for a reader checking a claim. It covers W0 to W8 ([the stages](GLOSSARY.md#experiments-and-how-they-are-counted)) on one RTX 5060 Ti 16 GB (sm_120, PCIe Gen3 x8) in Phastform, a host with two Xeon E5-2609 v4 at 1.7 GHz and 188 GiB of RAM, with vLLM 0.32.0, and the run of 2026-10-26 on an A100 SXM4 80 GB ([its own section](#the-a100-run-2026-10-26)). The change measured here is pull requests #63350 and #63351, proposed to vLLM and under review, run as the make-room variant ([GLOSSARY.md](GLOSSARY.md#the-change-and-its-parts)). The numbers come from the frozen box ([`deploy/phastform`](../deploy/phastform)), the `flatshare-bench` probes and the E2 harness in this repository, run from cyan through the SSH tunnel as the control plane reaches the engine; the first make-room probes (2026-10-23) ran on Gray, the identical second box, and the later ones on Phastform. Earlier probe numbers are kept only where W1 has not re-measured them, and are marked. A record names the revision it ran on as `git describe` of the history before it was rewritten on 2026-10-27; [REVISIONS.md](REVISIONS.md) gives each one's commit now.

**Contents.**
- The box and its costs: [the frozen box](#the-frozen-box), [E0, the tier cost model](#tier-cost-model-resuming-an-agent-turn-e0), [voice on the shared GPU](#voice-on-the-shared-gpu).
- [E2, voice under agent load](#voice-under-agent-load-e2): W2's [standard](#the-standard-scenario-w2) and [discriminating](#the-discriminating-scenario-w2) scenarios; W4's [discriminating scenario in rounds](#the-discriminating-scenario-in-rounds-with-arm-9-and-the-change-w4) and [standard scenario repeated](#the-standard-scenario-repeated-w4); W5's [standard scenario in five rounds](#the-standard-scenario-in-five-rounds-w5), [discriminating scenario after its fixes](#the-discriminating-scenario-after-w5s-fixes-w5) and [calls per GPU](#calls-per-gpu-w5); [the final code](#the-final-code-against-w5-fb) (FB); [the box's running share](#the-boxs-running-share-rs) (RS); [E3, what make-room takes from the agents](#e3-what-make-room-takes-from-the-agents).
- The change's probes: [the KV case and the prefill case](#the-kv-case-make-room), [W5's modules](#w5-the-fixed-make-room-on-phastform), [the PRs' configuration](#the-prs-configuration-in-one-session-w5), [the first tokens in flight](#after-w5-the-first-tokens-in-flight), [the window's knee](#the-windows-knee).
- What sits in front of the engine: [flatshare-gateway](#the-control-plane-flatshare-gateway-in-front-of-the-engine), [agentic-api](#the-api-layer-agentic-api-in-front-of-the-engine).
- The agents and the simulator: [agent replay on the frozen box](#agent-replay-on-the-frozen-box-a-first-retention-headroom-signal), [at three loads, with the simulator's calibration](#agent-replay-at-three-loads-and-the-simulator-calibrated-on-it-w2), [E1 in the simulator](#e1-parking-in-the-simulator).
- [The A100 run](#the-a100-run-2026-10-26), [failure drills](#failure-drills), [load generation](#load-generation), and [the engine plugins removed after v0.3](#engine-plugins).

## The frozen box

| | Value |
|---|---|
| GPU KV pool | 36,400 tokens, fixed at 5 GiB (32k context) |
| CPU tier | 20 GiB in `/dev/shm`: 17.35 GiB on NUMA node 1 (GPU-local), 2.65 GiB spilled to node 0. In the earlier probe, a remote NUMA node cost at most 1% of pinned bandwidth |
| GPU memory, all three servers idle | 14,408 / 16,311 MiB |
| GPU memory, stress peak (LLM c=16, 30k prefill, STT c=4, TTS c=4) | 14,086 / 16,311 MiB in the earlier probe; no OOM, no preemption |
| Cold start, first boot (includes a one-time FlashInfer JIT) | 7.5 min for all three servers |
| Engine restart with warm caches | about 2 min |

The two start rows are W0's (2026-10-22). On the final build the engine started in 88.8 s on warm caches and 531.8 s on empty ones, and a restart of all three servers took 102.7 s ([the drills](#failure-drills)).

## Tier cost model: resuming an agent turn (E0)

`flatshare-bench tiers` sends a unique prompt of L random token ids three times, each asking for one token: cold (recomputed), again at once (a GPU prefix-cache hit), and after other prompts pushed it out of the GPU pool (restored from the CPU tier). vLLM's prefix-cache counters, read around each request, verify where its tokens came from; no sample of these runs was invalid. Time to first token (TTFT), p50 of 5 samples, frozen box (Qwen3-4B-FP8, audio servers resident, 36,400-token pool, 20 GiB write-through CPU tier, step cap 512/256), 2026-10-22:

| Prompt tokens | Recomputed | GPU hit | Restored from host RAM |
|---|---|---|---|
| 1,024 | 135 ms | 45 ms | 64 ms |
| 2,048 | 198 ms | 40 ms | 95 ms |
| 4,096 | 437 ms | 48 ms | 157 ms |
| 8,192 | 1,114 ms | 64 ms | 281 ms |
| 16,384 | **3,264 ms** | **97 ms** | **509 ms** |
| 24,576 | 6,506 ms | 128 ms | 722 ms |
| 31,744 | 10,239 ms | 156 ms | 903 ms |

**Fitted model** (`flatshare-core`'s defaults):
- GPU hit ≈ 35 ms + 3.8 µs/token;
- a CPU restore adds ≈ 10 ms + 23.6 µs/token over a GPU hit (about 6.2 GB/s of KV at 144 KiB/token), the same slope as the earlier BF16 probe;
- recompute ≈ 48 ms + 63 µs/token + 8.1 ns/token², the quadratic term being attention. The 512-token step cap that protects decode latency makes cold prefill slower than an uncapped engine.

Restoring beats recomputing at every measured length (from 1,024 tokens). In the dimensionless terms E1 uses, `flatshare-sim` takes the restore time from these fits as what a restore adds over a GPU hit. At the 8k median prompt, ρ = restore / recompute ≈ 0.18. π = the median pause / restore depends on which pause:
- Copilot's median tool call (166 ms) gives about 0.8;
- the W0 replay window's median pause, over the turns that pause (2.4 s, at its 3.4k median prompt), gives about 26.

**Under contention** (p50; same probe while background load runs):

| Prompt tokens | Background | Recomputed | GPU hit | Restored |
|---|---|---|---|---|
| 1,024 | 8 decode streams (1k prompt, 1k output) | 200 ms | 131 ms | 173 ms |
| 16,384 | 8 decode streams | 4,528 ms | 186 ms | 592 ms |
| 1,024 | one stream of cold 8k prefills | 342 ms | 283 ms | 263 ms |
| 16,384 | one stream of cold 8k prefills | 5,230 ms | 387 ms | 728 ms |

- **Decode load** adds a roughly constant ~85 ms to a GPU hit and to a restore (the probe shares steps with the decode batch); the restore slope is unchanged, so PCIe is not the bottleneck.
- **Prefill load** is worse. Even a GPU-cached 1k prompt waits ~250 ms behind the running chunked prefill, and recompute slows 1.6–2.5×. This is the compute-interference path that a critical-window prefill throttle targets.

The earlier probe (2026-10-21, BF16 weights, no audio servers) measured 0.18 s / 0.57 s / 4.75 s at 16k for GPU hit / restore / recompute.

## Voice on the shared GPU

`flatshare-bench voice-turn` runs the cascade a voice agent runs, against the frozen box through the tunnel:
1. the caller's utterance goes to STT (LibriSpeech clips of 2.0 and 3.2 s);
2. the transcript is streamed through the LLM at the realtime priority;
3. the first sentence goes to TTS as soon as it is complete. A sentence counts once it reaches 20 characters, LiveKit's default.

The turn runs from sending the audio to the first byte of TTS audio. That is the server-side share of voice-to-voice latency: endpointing, the network and playout come on top. 30 turns per row, p50 / p90 in ms (W1, 2026-10-22):

| Background agent load | STT | LLM first token | LLM first sentence | TTS first audio | **Turn** | Background prefill |
|---|---|---|---|---|---|---|
| None | 160 / 168 | 49 / 51 | 251 / 319 | 176 / 187 | **589 / 665** | — |
| One stream of cold 8k-token prefills | 233 / 240 | 172 / 197 | 807 / 937 | 226 / 251 | **1,261 / 1,413** | 6,220 tokens/s |
| Two streams | 233 / 248 | 565 / 1,207 | 1,802 / 2,113 | 231 / 254 | **2,279 / 2,593** | 6,966 tokens/s |

- **The LLM's first sentence carries most of the damage.** Decode steps for the voice request share the GPU with 512-token prefill chunks, so the inter-token gap stretches. The co-located STT and TTS slow by up to 46% and 31% (time-slicing between processes).
- **MPS does not fix it.** With the three servers as clients of one MPS daemon (variant `mps`):
  - idle turns are unchanged (606 / 669 ms);
  - STT and TTS gain 7–8% under load;
  - beside one prefill stream, the turn gets worse (1,403 / 2,061 ms) and background prefill throughput drops 13%;
  - beside two streams it is mixed (2,042 / 2,870 ms).

  The interference that matters is inside the engine, so the frozen box stays on time-slicing, and E2 did not run the MPS split it had planned as arm 10.

**End to end, from waveforms.** `flatshare-voice bots` places calls through LiveKit (2 callers, 25 turns each, a barge-in on 25% of turns) to the stock voice agent (`flatshare-voice-worker`). Voice-to-voice (V2V) runs from the end of the caller's speech to the agent's first audio, read at the caller on one clock (W1, 2026-10-22; the LLM path goes straight to the engine):

| Setup | V2V p50 | p95 | p99 | Barge-in to silence p50 / p95 |
|---|---|---|---|---|
| Voice alone, LiveKit's end-of-turn model | 1,646 ms | 3,226 ms | 3,444 ms | 637 / 686 ms |
| Voice alone, fixed endpointing (experiments) | **1,614 ms** | **1,760 ms** | **1,889 ms** | 634 / 696 ms |
| Beside the agent replay (16 agent sessions, 1.3 requests/s), fixed endpointing | 1,664 ms | **2,312 ms** | 3,013 ms | 563 / 687 ms |

- **Endpointing dominates voice alone.** About 0.75 s of each turn is endpointing: the VAD's 0.55 s of silence and about 0.2 s more, LiveKit's 0.5 s minimum delay never binding (the endpointing sweep below); the server-side turn and the transport are the rest.
- **The endpointing sweep** (2026-10-28, [its record](../benchmarks/results/2026-10-28-endpointing/README.md)): one caller playing the corpus's 40 lines in order, voice alone on the serving profile, at four settings of the VAD's silence and the delay. The worker ended each turn about 0.2 s after the silence ran out at every delay tried (747, 753, 587 and 492 ms at the median with 0.55/0.5, 0.55/0.2, 0.4/0.3 and 0.3/0.2 s), so lowering the delay alone changed nothing, and 0.3 s of silence brought the V2V p50 from 1,612 to 1,346 ms. At 0.4 and 0.3 s one line of 40 was answered before it had ended, its pause after a first sentence taken for its end. Every E2 point keeps 0.55/0.5.
- **The end-of-turn model adds a 3 s mode.** It judged about one line in six from the synthesized callers unfinished and waited for its 3 s maximum. Experiments therefore end turns a fixed delay after speech (the `vad-turns` variant), the same for every arm.
- **Agent load breaks the voice SLO.** Beside a realistic agent replay, V2V p95 rises by 552 ms and p99 by 1.1 s, against an SLO of at most +250 ms. Agent TTFT stayed 149 ms p50 and 11.2 s p99.
- **Nothing ran out of memory.** GPU memory held at 14.05 GiB, and neither the calls (50 turns) nor the replay (507 requests) had an error.

**A barge-in closes the LLM stream** (W0: LiveKit Agents 1.8.8 with mock STT, LLM and TTS, in the voice-plane namespace on cyan, [`python/flatshare_voice`](../python/flatshare_voice), 10 runs). Measured from the caller's barge-in onset to the LLM server seeing the disconnect:
- 2.19 s median with the stock handling, which pauses the reply audio and interrupts it on the barge-in's final transcript;
- 0.58 s with `resume_false_interruption=False`, which interrupts after 0.5 s of speech.

In stock mode the reply keeps streaming for the whole barge-in.

**Earlier probe** (2026-10-21/22, 8-character first sentence):
- the server-side turn took 415 ms idle and 876 ms median under 8k prefills;
- Whisper large-v3-turbo on vLLM took 144 / 166 / 221 ms for 2 / 5 / 10 s utterances at concurrency 1;
- Kokoro-82M reached first audio for one sentence in 154–179 ms.

## Voice under agent load (E2)

Each point: two callers (`flatshare-voice bots`, 25 turns each, a barge-in on a quarter of the turns) through LiveKit to the stock voice agent, beside the AIPerf replay of the 32× Claude Code corpus at a number of agent sessions, through `flatshare-gateway` ([EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load) has the arms, [`benchmarks/e2`](../benchmarks/e2) the arm files and campaigns). V2V is read from waveforms at the callers, with fixed endpointing. Background goodput is the prompt and completion tokens per second of the agent requests that completed during the calls. The realtime first byte is the gateway's, exact from its request log. A turn the agent did not answer within the callers' 30 s reply timeout counts as a V2V sample censored at 30,000 ms; ≥ marks a percentile it sets, and a configuration with failed turns is not placed on the frontier.

The voice SLO is V2V p95 at most 250 ms above voice alone: 1,744 ms alone in W2's batch, so 1,994 ms for W2's tables, which are one run per point. With 50 turns a point, a p95 lies between the third- and fourth-largest turn, so single runs differ by a few hundred milliseconds at p95. W4 repeats the discriminating scenario and measures voice alone four more times, on the W3 build (with W2's run, a median of 1,767 ms, so an SLO of 2,017 ms), [below the W2 tables](#the-discriminating-scenario-in-rounds-with-arm-9-and-the-change-w4).

### The standard scenario (W2)

The whole corpus, 2026-10-22/23, on the W2 build (SGLang with HiCache on the W3 build, from w2c). V2V p95 and, in brackets, the realtime first byte p95, in ms:

| Agent sessions | Background tokens/s | Arm 2: stock defaults (FCFS, 2,048-token chunks) | Arm 3: + priority | Arm 4: + step cap 512/256 (the frozen box) | Arm 7b: SGLang 0.5.22 |
|---|---:|---:|---:|---:|---:|
| 0 (voice alone) | 0 | 1,744 (96), frozen box | | | |
| 8 | 2,700–2,800 | 1,886 (114) | 1,871 (80) | — | 3,228 (1,440); with HiCache 1,822 (97) |
| 16 | 4,900–5,600 | 2,406 (446) | 2,400 (333) | 2,215 (224) | 8,477 (2,284); with HiCache 2,218 (249) |
| 24 | 9,200–9,600 | 3,008 (1,280) | 4,360 (2,243) | 3,865 (542) | — |

At 16 agent sessions, every arm (V2V p50 / p95 / p99 in ms; the realtime first byte p95; background tokens/s):

| Arm | Point | V2V p50 / p95 / p99 | First byte p95 | Background tokens/s | Notes |
|---|---|---:|---:|---:|---|
| 4 | frozen box: step cap 512, long-prefill threshold 256 | 1,736 / 2,215 / 2,925 | 224 | 5,068 | |
| 4 | step cap 1,024, no threshold | 1,687 / 2,167 / 2,479 | 144 | 5,246 | the best cell; within noise of the frozen box |
| 4 | other cells (512/0, 1,024/256, 1,024/512) | 1,703–1,724 / 2,257–2,510 / 2,504–2,684 | 178–273 | 5,015–5,097 | |
| 5 | `--watermark 0.1` | 1,703 / 2,495 / 2,986 | 216 | 5,001 | |
| 5 | `--max-num-queued-tokens 8192` | 1,703 / 2,133 / ≥30,000 | 137 | 4,454 | 2 of 50 voice turns failed, counted at the callers' 30 s reply timeout: the limit refuses realtime requests too (43 refusals) |
| 6 | static cap, 2 background in flight | 1,755 / 2,208 / 2,729 | 236 | 5,569 | the highest goodput |
| 6 | static cap, 8 | 1,683 / 3,065 / 3,647 | 510 | 5,296 | |
| 6 | static reservation, 2,048 tokens | 1,688 / 2,328 / 3,068 | 276 | 4,922 | |
| 7a | `--max-num-seqs 12` | 1,710 / 2,104 / 2,984 | 161 | 4,981 | |
| 7a | `--max-num-seqs 12` with #54530 | 1,728 / 2,736 / 3,103 | 461 | 4,897 | |
| 7b | SGLang 0.5.22, priority preemption | 1,882 / 8,477 / 9,458 | 2,284 | 5,543 | no host tier: half its prompt tokens hit, against about 90% for vLLM with its CPU tier |
| 7b | SGLang with HiCache (4×, written through) | 1,720 / 2,218 / 4,088 | 249 | 4,924 | the host tier, for parity with vLLM's |
| 8a | llm-d flow control, emulated (ceilings 6/6/6) | 1,768 / 2,468 / 2,821 | 216 | 5,413 | 1 background request shed |
| 8b | SMG 1.13.0, as shipped | 1,723 / 2,202 / 3,255 | 237 | 5,545 | |
| 8c | preempt-resubmit before the first byte | 1,678 / 2,520 / 2,711 | 235 | 5,090 | 11 resubmissions |
| factor | frozen box, preemptive generation off | 1,687 / 2,278 / 2,953 | 294 | 5,121 | |
| calls | frozen box, 4 callers / 6 callers | 1,747 / 2,599 / 3,135 · 1,735 / 2,670 / 3,471 | 428 · 413 | 5,296 · 5,067 | 4 callers alone: 1,655 / 2,017 / 2,110 |

What the standard scenario says:
- **At 8 agent sessions every vLLM arm measured in W2 holds the voice SLO** (1,871–1,886 ms against 1,994). SGLang does not (3,228 ms).
- **At 16, the realistic operating point, none clearly does, and the arms are within noise of each other.** The frozen box, the best stock cell, `--max-num-seqs 12`, SMG and the static cap of 2 land at 2,104–2,215 ms, 110–221 ms above the SLO; `--max-num-queued-tokens` refuses work, voice turns included (2 of 50 failed), and lands above it too (2,133 ms). No gateway policy or engine knob separates itself from tuned vLLM here.
- **The step cap shows in the realtime first byte, not in V2V.** At 24 agent sessions, priority scheduling alone gives a first-byte p95 of 2,243 ms and the 512/256 step cap 542 ms, but V2V p95 is 4,360 against 3,865 ms: the rest of the turn, decoding to the first sentence and TTS, still shares the GPU with agent prefills. No arm holds the SLO at 24.
- **SGLang's deficit was its missing host tier.** Without one it is the worst arm at every load; with HiCache, written through like vLLM's CPU tier, it matches vLLM: 1,822 ms at 8 agent sessions and 2,218 ms at 16 (on the W3 build), against the frozen box's 2,215 ms (on the W2 build).
- **Preemptive generation off** changes nothing measurable (2,278 against 2,215 ms), and **more callers** cost what they cost alone and some: four callers beside 16 agent sessions reach 2,599 ms against 2,017 ms for four callers alone (their SLO 2,267 ms).

### The discriminating scenario (W2)

W2's third batch, 2026-10-23, on the W3 build like everything after it: 24 agent sessions of the 48 traces with the largest contexts (`weka-x32-long`, 13.9k tokens at the median request), beside the same two callers, one run per point. The pool holds two or three of these contexts, and the agents' working set is far beyond both tiers: under the frozen box about 95% of their prompt tokens are computed, and agents wait 40–71 s for a first token at the median under every arm but preempt-resubmit (28 s) and the queued-token limit, which refuses them. It is the regime where running agent KV fills the pool. V2V p50 / p95 / p99 and the realtime first byte p95 in ms; background tokens/s:

| Arm | Point | V2V p50 / p95 / p99 | First byte p95 | Background tokens/s | Notes |
|---|---|---:|---:|---:|---|
| 4 | frozen box | 4,270 / 10,811 / 12,427 | 8,556 | 3,703 | |
| 4 | step cap 1,024, no threshold | 4,816 / 8,789 / 10,024 | 6,523 | 4,097 | |
| 5 | `--watermark 0.1` | 4,843 / 10,647 / 13,583 | 8,201 | 3,712 | |
| 5 | `--max-num-queued-tokens 8192` | ≥30,000 / ≥30,000 / ≥30,000 | 56 | 5,597 | 48 of 50 voice turns failed, counted at the 30 s reply timeout: 1,132 requests refused, voice turns among them |
| 6 | static cap, 2 / 8 background in flight | 3,363 / 8,573 / 10,385 · 5,713 / 9,876 / 11,369 | 6,320 · 7,502 | 3,566 · 3,785 | |
| 6 | static reservation, 2,048 tokens | 3,311 / 5,706 / 6,317 | 3,631 | 3,633 | |
| 7a | `--max-num-seqs 12` with #54530 | 4,687 / 10,930 / 11,870 | 7,964 | 3,718 | |
| 7b | SGLang 0.5.22, priority preemption | 5,367 / 12,051 / 17,937 | 6,451 | 3,678 | |
| 7b | SGLang with HiCache (4×, written through) | 4,274 / 10,497 / 13,193 | 5,000 | 4,908 | |
| 8a | llm-d flow control, emulated | 4,221 / 10,992 / 11,277 | 8,172 | 4,006 | 60 background requests shed |
| 8b | SMG 1.13.0 | 3,511 / 11,050 / 13,839 | 8,783 | 3,557 | |
| 8c | preempt-resubmit before the first byte | 1,960 / 2,591 / 2,684 | 374 | 3,901 | 1,080 resubmissions; agent TTFT p99 140 s |

The voice turn by stage (means over the replies, ms; `waterfall.svg` draws them):

| Point | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | LLM first token |
|---|---:|---:|---:|---:|---:|---:|
| frozen box | 827 | 3,651 | 276 | 326 | 5,080 | 2,701 |
| static reservation, 2,048 tokens | 830 | 2,173 | 250 | 351 | 3,604 | 1,068 |
| preempt-resubmit | 827 | 627 | 223 | 358 | 2,034 | 265 |

What the discriminating scenario says:
- **Every arm but one collapses, and the loss is the wait for KV.** V2V p95 is 5.7–12.1 s. Endpointing and TTS do not move; the turn loses its seconds between the end of the turn and the first sentence, most of them before the LLM's first token.
- **Engine priority does not help here**, in any form measured: vLLM's priority scheduling with any step cap, #54530 (it preempts at the request-count cap, which does not bind here), SGLang's priority preemption, with or without its host tier. SGLang's host tier lifts agent goodput by a third, not the voice tail.
- **Static answers fall short.** The 2,048-token reservation halves the tail (5.7 s), still 3.7 s over the SLO; concurrency caps do not help, because two contexts fill the pool. vLLM's queue limit refuses the voice turns with the rest (48 of 50 failed).
- **Preempt-resubmit holds the tail** at 2.6 s p95, 0.6 s over the SLO, at the frozen box's goodput: it aborts the background requests that have no first byte yet, which frees their KV at once, and their resubmissions restore the prefixes already written through to the CPU tier rather than recompute them (at their first scheduling, the engine computed 0.65M prompt tokens and restored 0.62M, against 1.48M and 0.06M under the frozen box; the CPU tier's own counter, which sees a resume too, loaded 0.82M tokens against 0.22M). Its cost is the agents' tail. It cannot reach a request that is already decoding, which an engine-side preemption can (arm 9).
- **An instrument note.** vLLM counts a preemption when the victim's next output reaches the frontend, so the per-point preemption counts are lower bounds ([the KV case](#the-kv-case-make-room)).

### The discriminating scenario in rounds, with arm 9 and the change (W4)

W4, 2026-10-23/24, in rounds: w2c and a9 are round 0, `r1`–`r4` repeat it, each round grouping its points by engine and rotating the groups' order, so that no configuration always runs at the same hour, except that the PRs' configurations and arm 9's 48-token window ran after the rounds, in a session of their own with no frozen-box, arm-9 or voice-alone point (10-23 23:16Z – 10-24 01:35Z), and arm 9's ablations always ran in one order. Every headline configuration has five runs, each of arm 9's ablations and each configuration of the PRs three, and voice alone, the SLO's reference, five (1,744–1,805 ms, median 1,767 ms, so the SLO is 2,017 ms). All on the W3 build but one run of voice alone, W2's 1,744 ms (the four on the W3 build give 1,768.5 ms); [`benchmarks/e2`](../benchmarks/e2/README.md) lists the campaigns. The median of the runs, and for V2V p95 its range:

| Configuration | Runs | V2V p50 / p95 / p99 | p95 range | First byte p95 | Background tokens/s | Agent TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 5 | 1,611 / 1,767 / 1,799 | 1,744–1,805 | 66 | — | — |
| The frozen box | 5 | 4,520 / 10,138 / 12,110 | 9,257–10,811 | 7,834 | 3,710 | 65 / 99 s |
| Step cap 1,024, no threshold | 5 | 4,348 / 8,789 / 10,024 | 8,161–9,490 | 6,523 | 4,097 | 57 / 85 s |
| Static reservation, 2,048 tokens | 5 | 3,127 / 5,671 / 6,681 | 4,906–5,798 | 3,190 | 3,564 | 69 / 102 s |
| SGLang with HiCache | 5 | 4,705 / 11,948 / 18,063 | 9,394–16,702 | 8,012 | 4,723 | 46 / 82 s |
| Preempt-resubmit before the first byte | 5 | 1,960 / 2,591 / 2,684 | 2,110–2,663 | 374 | 3,630 | 31 / 160 s |
| **Arm 9**, every reply capped at 16 tokens | 5 | 1,825 / **2,057** / 2,139 | 1,980–2,134 | 290 | 3,411 | 75 / 103 s |
| #63350 alone (KV preemption) | 3 | 3,246 / 5,773 / 8,110 | 5,612–6,063 | 2,785 | 3,633 | 66 / 97 s |
| #63350 + #63351 (the budget cap, 16 tokens) | 3 | 1,986 / 3,229 / 4,135 | 3,220–4,772 | 338 | 3,568 | 71 / 100 s |
| **#63350 + #63351, a 48-token window** | 3 | 1,943 / **2,193** / 2,281 | 2,168–2,286 | 372 | 3,493 | 72 / 97 s |

The PR configurations ran the make-room variant on vLLM 0.32.0 with the PRs' switch values (two preemptions a step, the budget cap, no reserve), to which the PRs' diff, ported to 0.32.0, schedules identically ([GLOSSARY.md](GLOSSARY.md#the-change-and-its-parts)), behind the gateway without an admission policy and with full replies. Arm 9 without one part at a time, three runs each (V2V p95, median and range, ms):

| Arm 9 without | V2V p95 | Range | | Arm 9 without | V2V p95 | Range |
|---|---:|---:|---|---|---:|---:|
| the critical-window throttle | 2,049 | 1,995–2,081 | | degrade | 3,837 | 2,890–4,079 |
| the headroom | 2,033 | 1,947–2,071 | | degrade, with a 48-token window | 2,205 | 2,098–2,289 |
| the lifecycle events | 2,036 | 2,005–2,048 | | the gateway policy (engine only) | 4,034 | 2,984–4,062 |
| (headroom armed by the request) | 2,124 | 2,008–2,165 | | KV make-room | 2,037 | 1,990–2,039 |
| (budget 0 ms / 500 ms) | 2,109 / 2,060 | 2,094–2,166 / 1,990–2,080 | | the reserve | 2,116 | 2,115–2,155 |
| | | | | the budget cap | 5,295 | 5,063–6,115 |
| | | | | the change (gateway only) | 10,993 | 10,028–11,958 |

What the gateway's policy did in each run, as its counters at the end of the calls (60 realtime requests a run, preemptive generation included):

| Configuration | Runs | Headroom deferrals | Critical-window deferrals | Realtime requests degraded |
|---|---:|---:|---:|---:|
| Arm 9 | 5 | 31–45 | 0 | 60 |
| Without the throttle | 3 | 38–52 | — | 60 |
| Throttle budget 0 / 500 ms | 3 each | 39–49 / 39–48 | 0 / 0 | 60 |
| Without the headroom | 3 | — | 1–3 | 60 |
| Headroom armed by the request | 3 | 9–11 | 0 | 60 |
| Without the lifecycle events | 3 | 4–6 | 0 | 60 |
| Without degrade, a 48-token window | 3 | 41–50 | 0 | — |
| The gateway only (stock engine) | 3 | 71–86 | 0 | 60 |
| Standard scenario, 16 / 24 agent sessions | 3 / 1 | 5–10 / 41 | 0–4 / 27 | 0 / 1 |

The voice turn by stage (means over each configuration's replies, ms):

| Configuration | Endpointing | Reply to first sentence | TTS first audio | Transport and playout | Mean V2V | LLM first token |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 757 | 297 | 199 | 352 | 1,605 | 80 |
| The frozen box | 822 | 3,801 | 244 | 343 | 5,211 | 2,760 |
| Static reservation | 827 | 1,967 | 238 | 346 | 3,378 | 935 |
| Preempt-resubmit | 826 | 622 | 222 | 353 | 2,023 | 274 |
| Arm 9 | 824 | 429 | 222 | 354 | 1,830 | 221 |
| #63350 + #63351, 16 tokens | 826 | 914 | 245 | 353 | 2,336 | 265 |
| #63350 + #63351, 48 tokens | 824 | 526 | 249 | 353 | 1,953 | 270 |

![The voice turn by stage: stock vLLM loses its seconds in the reply's first sentence](figures/e2-waterfall.svg)

The same stages as bars (`make e2-figures`), a row per point of the records: `a1-voice` is voice alone (four runs: its W2 run predates the worker's stage times), `disc-frozen-s24` the frozen box, `a6-reserve2048` the static reservation, `a8c-resubmit` preempt-resubmit, `a9` arm 9 and `pr-ab48` the PRs with a 48-token window. The violet stage is the reply to its first sentence; the LLM's first token, from its request, is in the table above.

What the repeats say:
- **The change brings the voice tail to within 0.2 s of the SLO.** With full replies and a 48-token window, the PRs' configuration reaches 2,193 ms (2,168–2,286), 176 ms over the SLO, against the frozen box's 10,138 and preempt-resubmit's 2,591. Arm 9, whose gateway capped every reply at 16 tokens, reaches 2,057 ms: two of its five runs are under the SLO, and its median is 40 ms over it, within one point's spread. No configuration without the change has a run under the SLO; vLLM's queued-token limit refused 48 of its 50 voice turns, which count at the reply timeout.
- **Its budget cap is necessary, and either KV preemption or the reserve makes the room.** Without the budget cap the p95 stays at 5.3 s even with KV preemption and the reserve. With full replies the two forms gave 2,193 ms (the KV preemption, as the PRs) and 2,205 ms (the reserve, with arm 9's policy); with capped replies the reserve alone gave 2,037 ms and the KV preemption alone 2,116 ms (2,115–2,155, every run above arm 9's median), and with both the reserve kept the pool from filling, so the KV preemption never fired (0 preemptions in five runs). #63350 alone halves the frozen box's tail (5.8 s); with #63351 the turn's first token comes in 338 ms at p95 (frozen 7,834).
- **The reply has to fit the window.** With the 16-token window a full reply (20–22 tokens at the median and up to 40, in the gateway's log) outruns it and decodes beside agent prefill again: 3.2–4.0 s whether the gateway runs or not. Degrade, which caps the reply at 16 tokens when the headroom is not there, capped every reply here; a 48-token window covers full replies instead, at 2,193 ms for the PRs alone and 2,205 ms with arm 9's gateway policy and reserve, about 140 ms over arm 9's capped replies: the price of a full reply's longer first sentence (526 against 429 ms).
- **The gateway's lifecycle-timed mechanisms could not act here** (their counters are in the table above). Running agent requests held the pool, which the headroom can protect only by deferring new requests, so it never had its room: degrade capped 60 of 60 realtime requests in every run of arm 9. The critical-window throttle, checked after the headroom, deferred nothing while the headroom ran, even at a 0 ms budget. Room for a voice turn came from the engine alone: without the change, the policy leaves voice where stock vLLM does (10,993 ms). So E2 shows that headroom kept by deferring new requests cannot make room while running work holds the KV; a control plane can make room only by aborting admitted work, as preempt-resubmit does, at the agents' tail. Whether lifecycle timing helps where its mechanisms engage (in the standard scenario at 24 agent sessions the throttle deferred 27 requests) was not isolated.
- **The cost to the agents.** Arm 9 keeps 92% of the frozen box's background goodput and the PRs alone 94%, with the agents' TTFT p99 unchanged (97–103 s against 99 s) and their median TTFT 11–16% longer (71.7–75.1 s against 64.7 s). Preempt-resubmit keeps 98% but its resubmissions take the agents' p99 to 160 s. Over the calls' first 240 s, a window of the same length for every configuration, the shares are 94.5% (arm 9), 94.7% (the PRs) and 101.2% (preempt-resubmit) of the frozen box's 3,552 tokens/s.
- **The first turn does not set this tail.** Without each call's first turn, the medians of the runs' p95 are 2,058 ms for arm 9, 2,194 for the PRs' configuration and 10,224 for the frozen box, against 2,057, 2,193 and 10,138 with it.
- **Spread.** A configuration near the SLO moves by about 150 ms in p95 from run to run; one that misses it moves by up to 1.9 s (the gateway policy on stock vLLM, 10,028–11,958 ms) and SGLang by 7 s. One run cannot rank configurations that miss.

### The standard scenario, repeated (W4)

The whole corpus on 2026-10-23, three runs of each at the realistic load of 16 agent sessions (the frozen box's third run is W2's): arm 9's p95 is 2,196 ms (2,193–2,446) against the frozen box's 2,323 (2,215–2,530), both over the SLO and within noise of each other, at 5,244 and 5,167 tokens/s. Each call's first turn is slower under load (a mean of 2.4–2.5 s against 1.8 s for the later turns) and sets part of this p95: without it, 2,037 and 2,186 ms. Where the pool is not full the KV preemption has nothing to do, and the gateway rarely degrades a reply here, so a full reply's first sentence outruns the 16-token window as in the discriminating scenario. One run each at 8 and 24 agent sessions gives 2,060 and 2,375 ms; W2's single runs there, on the W2 build, were 1,871 (priority alone) and 3,865 ms (the frozen box), so the 1.5 s at 24, where the pool fills, is indicative only.

### The standard scenario in five rounds (W5)

W5's extension, 2026-10-25: the PRs' 48-token configuration and the frozen box at 16 agent sessions of the whole corpus, in each of five rounds on the round's own sample of it (agent seeds 11–15), every call opening with a warm-up turn. The PRs' configuration had the lower p95 in every round, by 109, 1,413, 945, 250 and 496 ms (median 496): 2,050 ms (2,021–2,419) against the frozen box's 2,516 (2,147–3,832), at 6,815 and 7,131 agents' tokens/s over the first 240 s (round by round −287 at the median, from −540 to +260). It met W5's SLO of 2,034 ms in one round, the frozen box in none. Were there no difference, five rounds would agree in sign one time in sixteen: they resolve the direction, not the size, which follows the sample (the frozen box's p95 moves by 1.7 s across them, the PRs' by 0.4 s). Here make-room preempted 0–7 requests a run and the budget cap held agent prefill back in 3–13% of the steps, against 26–39% in the discriminating scenario ([E3](#e3-what-make-room-takes-from-the-agents)). W4's three runs above, on one sample, had found arm 9, whose 16-token window a full reply outruns, within noise of the frozen box.

### Instrument notes (W2 to W4)

Background goodput counts the agent requests that completed during the calls, so its window is the calls' duration, which differs by configuration (about 280 s for arm 9, whose calls are fastest, to 430 s for the frozen box) against agent requests of about 75 s: edge effects differ by a few percent. The report also gives the goodput over the calls' first 240 s, a window of the same length for every point; every point with agents ran its calls at least that long.

vLLM counts a preemption when the victim next produces output, so per-point preemption counts are lower bounds. The prompt-token sources (computed, GPU hit, CPU hit) are counted at a request's first scheduling only; the CPU tier's own counter also sees a resume's loads. In W4's discriminating scenario it loaded, beyond first scheduling, 160–223k tokens a run on the frozen box (29–32 counted preemptions), 55–93k on the PRs' configuration with the 48-token window (37–43), 93–159k on #63350 alone (37–40) and 35–65k for arm 9 without the reserve (27–39). In that scenario a run without a counted preemption loads −0.5k to +0.8k tokens beyond first scheduling, so these loads are the victims' resumes; in the standard scenario the same difference spans −19k to +50k without one, so there it measures nothing. What a resume recomputes is not measured.

SGLang's KV pool came out at 33,683–34,679 tokens against vLLM's 36,400, 5–7% smaller: its static memory fraction sets it.

### The discriminating scenario after W5's fixes (W5)

W5, 2026-10-24/25. After W5's fixes to make-room ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351)) and to this harness, three campaigns of five rounds re-measured the configurations the PRs and the README cite, on the fixed make-room, round N of each on the agents' sample of seed 10 + N ([EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load) has the rounds' design):
- W5's campaign: voice alone, the frozen box and the PRs' 48-token configuration, and as its control the same configuration on make-room as v0.3 built it (`make-room-v03`), which E2 measured before W5;
- the PR campaign: the frozen box, the KV preemption alone and both PRs at 16 and at 48 tokens;
- the extension: preempt-resubmit (arm 8c, a router that aborts agent requests before their first byte and resubmits them) and the PRs' 48-token configuration again, and the standard scenario's pair ([above](#the-standard-scenario-in-five-rounds-w5)).

Voice alone gave 1,764–1,819 ms (median 1,784 ms), so W5's SLO is 2,034 ms. The records are in [`benchmarks/results/2026-10-24-e2-w5`](../benchmarks/results/2026-10-24-e2-w5) (`make reproduce-e2-w5`; its `compare.py` gives these tables). The median of the five runs, and for V2V p95 its range:

| Configuration | Campaign | V2V p50 / p95 / p99 | p95 range | First byte p95 | Agents' tokens/s, first 240 s | Agent TTFT p50 / p99 |
|---|---|---:|---:|---:|---:|---:|
| Voice alone | W5 | 1,619 / 1,784 / 1,892 | 1,764–1,819 | 60 | — | — |
| The frozen box | W5 | 4,772 / 10,432 / 11,361 | 8,686–10,770 | 6,942 | 3,306 | 63 / 106 s |
| The frozen box | PRs | 5,134 / 10,523 / 12,466 | 9,946–12,115 | 7,786 | 3,304 | 63 / 107 s |
| Preempt-resubmit (arm 8c) | extension | 1,990 / 2,522 / 2,919 | 2,516–2,868 | 349 | 3,332 | 32 / 193 s |
| #63350 alone (KV preemption) | PRs | 3,191 / 5,085 / 5,970 | 4,693–6,261 | 1,737 | 3,414 | 67 / 112 s |
| #63350 + #63351 (the budget cap, 16 tokens) | PRs | 1,983 / 3,094 / 3,885 | 2,868–3,192 | 364 | 3,298 | 71 / 105 s |
| **#63350 + #63351, a 48-token window** | W5 | 1,967 / **2,250** / 2,384 | 2,175–2,307 | 376 | 3,156 | 70 / 110 s |
| **#63350 + #63351, a 48-token window** | PRs | 1,982 / **2,232** / 2,379 | 2,174–2,397 | 376 | 3,236 | 74 / 115 s |
| **#63350 + #63351, a 48-token window** | extension | 1,940 / **2,162** / 2,300 | 2,132–2,293 | 363 | 3,203 | 72 / 111 s |
| The same on v0.3's module | W5 | 1,941 / 2,241 / 2,356 | 2,134–2,333 | 382 | 3,132 | 74 / 113 s |

Differences between configurations that ran the same sample of the agents' traces, round by round (rounds 1 to 5), and their median:

| Pair | V2V p95 (ms) | Median | First byte p95, median | Agents' tokens/s over 240 s, median |
|---|---|---:|---:|---:|
| The fixed make-room less v0.3's (W5's campaign) | −37, −158, +164, +8, +93 | +8 | −6 | −24 |
| The frozen box, the PR campaign less W5's | −630, −637, +2,124, +2,333, +92 | +92 | +1,000 | −51 |
| The 48-token window, the PR campaign less W5's | −62, +126, −109, −17, +91 | −17 | +26 | +58 |
| The 48-token window, the extension less W5's | +57, +82, −166, −88, −161 | −88 | −10 | +22 |
| The 48-token window less preempt-resubmit (the extension) | −223, −603, −390, −360, −723 | −390 | +9 | −130 |

What W5's re-measure says:
- **W5's fixes cost nothing measurable.** Round by round, the fixed module differs from v0.3's by −158 to +164 ms in V2V p95 (median +8 ms), −6 ms in the first byte and −24 tokens/s of the agents' goodput: within the run-to-run spread of either.
- **The result holds over five samples of the agents' traces.** Over the campaigns the frozen box's p95 moves from 8.7 to 12.1 s from run to run; the 48-token window's stays at 2.13–2.40 s, 98–363 ms over W5's SLO of 2,034 ms (202 ms at the median of its fifteen runs). Reply by reply that is 286 of its 885 replies over the SLO (32%), where the SLO allows 5%, against 532 of the frozen box's 590 (90%), and the p95 of the fifteen runs' replies pooled is 2,268 ms (`flatshare_bench.report.data`, `Configuration.over` and `pooled_p95`). The five samples of the agents are the same in the three campaigns, with the same callers: the fifteen runs are five workload samples measured three times. In every round of the PR campaign the configurations rank the same: the frozen box, #63350 alone, both PRs at 16 tokens, both at 48.
- **The campaigns agree.** Round by round, the 48-token window's p95 differs between the PR campaign and W5's by −109 to +126 ms (median −17 ms) and between the extension and W5's by −166 to +82 ms (median −88 ms); the frozen box's differs between the first two by −637 to +2,333 ms (median +92 ms), its tail varying by seconds from run to run within either campaign.
- **Aborting agent requests also makes room, at the agents' tail.** Preempt-resubmit reaches 2,522 ms (2,516–2,868), 488 ms over the SLO, and the 48-token window is lower in every round, by 223 to 723 ms (median 390 ms). Its agents keep their throughput (3,332 tokens/s over the first 240 s), but their TTFT p99 is 193 s (160–232 s) against 106 s for the frozen box and 111 s for the 48-token window.
- **Nearly every whole reply fits the 48-token window.** W5's gateway logged 5,571 voice replies over its campaigns: 23 tokens at the median and 52 at most; 79% of them run past 16 tokens and one past 48. The window has to cover only a reply's first sentence, which the voice agent hands TTS whole.
- **The cost to the agents comes from the budget cap.** Over the calls' first 240 s they keep 98% of their throughput at 48 tokens and all of it at 16 (3,236 and 3,298 against 3,304 tokens/s in the PR campaign; at 48 tokens 97% over the fifteen runs, 3,203 against the frozen box's 3,305); with #63350 alone they gain 3%. In the PR campaign their median TTFT rises from 63 to 74 s at 48 tokens (from 63 to 72 s over the pooled runs), and their p99 stays within the runs' spread (105–115 s against 107 s). [E3](#e3-what-make-room-takes-from-the-agents) says where it goes.

### Calls per GPU (W5)

W5, 2026-10-25: voice alone and the PRs' 48-token configuration with 4 and with 6 callers, beside the same 24 agent sessions of the discriminating scenario, in three rounds on the round's own sample of the agents' traces (agent seeds 11–13), every call opening with a warm-up turn. Each configuration's SLO is voice alone's p95 with as many callers plus 250 ms ([SLO.md](SLO.md#the-slos-reference-campaign-by-campaign)).

| Configuration | Callers | V2V p50 / p95 / p99 | p95 range | First byte p95 | Agents' tokens/s, first 240 s | Agent TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 4 | 1,647 / 1,794 / 1,936 | 1,780–1,894 | 66 | — | — |
| **#63350 + #63351, a 48-token window** | 4 | 1,993 / 2,313 / 2,512 | 2,250–2,415 | 385 | 3,055 | 78 / 109 s |
| Voice alone | 6 | 1,637 / 1,885 / 2,028 | 1,878–1,885 | 82 | — | — |
| **#63350 + #63351, a 48-token window** | 6 | 2,016 / 2,392 / 2,533 | 2,360–2,468 | 379 | 2,716 | 83 / 125 s |

- **More calls, about as far over the SLO.** The PRs' p95 is over its SLO in every round, by 269 ms (206–370) with 4 callers and 257 ms (226–333) with 6, against 202 ms at the median with two (the fifteen runs above). #63351 holds agent prefill while any call's first tokens are due, so the calls' overlapping replies hold it for longer: in 53% and 64% of the engine's steps over the calls, against 33% with two ([E3](#e3-what-make-room-takes-from-the-agents)), and the agents' throughput over the first 240 s falls to 3,055 and 2,716 tokens/s, from 3,203 with two.

### The final code against W5 (FB)

FB, 2026-10-25: voice alone, the frozen box and the PRs' 48-token configuration on the final code, both hosts at v0.3-191-g987fd09: W6's transport (the GPU servers on Unix sockets behind cyan's tunnel of one SSH session a connection, the gateway on its socket unit), the budget cap counting a reply's tokens in flight as #63351 does since its revision 9013ed433, and the harness's fixes. Each round replays the sample of the agents of W5's round of the same number (agent seed 10 + N) with W5's arm files, every call opening with a warm-up turn, so each configuration compares with W5's campaign round by round. V2V p95 in ms, the median of the five rounds and their range, and the agents' tokens/s over the calls' first 240 s:

| Configuration | Rounds | W5 V2V p95 | FB V2V p95 | FB − W5, round by round | Agents' tokens/s, W5 / FB |
|---|---:|---:|---:|---|---:|
| Voice alone | 5 | 1,784 (1,764–1,819) | 1,796 (1,768–1,819) | −4, +18, −51, +55, +11 | — |
| The frozen box | 5 | 10,432 (8,686–10,770) | 10,399 (8,374–10,866) | −177, −2,396, +1,706, +761, +435 | 3,306 / 3,368 |
| #63350 + #63351, a 48-token window | 5 | 2,250 (2,175–2,307) | 2,268 (2,246–2,343) | +77, +71, −45, +19, +36 | 3,156 / 3,252 |

- **W5's numbers stand on the final code.** The PRs' configuration came out at 2,268 ms against 2,250 in the same five rounds of W5's campaign, higher in four rounds and lower in one, within what two sets of runs of it on the same samples show (W5's fifteen runs of it, whose median of 2,236 ms the tables above and the README give, span 2,132–2,397 ms). Its V2V p50 and its realtime first byte came out lower in four of the five rounds, by a median of 26 ms and 20 ms: the sockets and the pooled connections take a little off each turn and leave the tail where W5 measured it.
- **The references moved as they move between runs.** Voice alone by −51 to +55 ms, inside W5's own range; the frozen box by −2,396 to +1,706 ms, as its p95 moves from run to run on the same sample (W5's ten runs of it span 8,686–12,115 ms). No turn failed in any of the 15 points. The records are in [`benchmarks/results/2026-10-25-e2-fb`](../benchmarks/results/2026-10-25-e2-fb), whose `compare.py` gives this table from them and W5's, with one row a round.

### The box's running share (RS)

RS, 2026-10-26: the frozen box and the PRs' 48-token configuration on v1.2, with W5's arm files and the engine's free KV sampled every 0.5 s, which W5's and FB's points ran without. Each round replays the sample of the agents of W5's round of the same number (agent seed 10 + N), as the A100 run's core did, so each configuration compares with the A100's round by round on the same agents. The running share is the share of the samples in which free KV was under two calls' 2,048 tokens while the agents ran; beside it, the box's V2V p95 in ms and the agents' tokens/s over the calls' first 240 s. The median of the five rounds and their range:

| Configuration | Rounds | Running share, the box | Running share, the A100 | V2V p95 on the box, ms | Agents' tokens/s on the box |
|---|---:|---:|---:|---:|---:|
| The frozen box | 5 | 8.1% (7.1–8.9%) | 5.7% (4.2–9.1%) | 9,818 (9,200–11,636) | 3,356 (3,040–3,690) |
| #63350 + #63351, a 48-token window | 5 | 6.1% (5.4–7.4%) | 6.3% (2.4–6.9%) | 2,169 (2,143–2,277) | 3,160 (2,903–3,496) |

- **About half the modeled share, on both cards.** Two calls found KV short 8.1% of the time beside the frozen engine on the box and 5.7% on the A100, on the same five samples of the agents (the box higher in four of the five rounds), and 6.1% and 6.3% beside the PRs' configuration. The simulator places the box at 17% on its measured costs and the A100 at 10% [modeled], 2.1 and 1.8 times what each card measured beside the frozen engine. What sets the cards' voice tails apart is how long a turn waits once KV is short: behind two 18k-token agent requests in a full pool, stock vLLM's wait is 4.9–5.0 s on the box ([W5's probes](#the-prs-configuration-in-one-session-w5)) and 1.72–1.75 s on the A100.
- **The rest stays within W5's runs.** With the sampler on, the frozen box's p95 of 9,818 ms and the PRs' 2,169 ms lie within W5's runs of each (8,686–12,115 and 2,132–2,397 ms), and the agents made 3,356 and 3,160 tokens/s against W5's 3,305 and 3,203. No turn of the 500 failed. The records are in [`benchmarks/results/2026-10-26-e2-rs`](../benchmarks/results/2026-10-26-e2-rs), whose `running_share.py` gives this table from them and the A100 core's, with one row a round.

### E3: what make-room takes from the agents

The cost to the agents by its mechanism, over W5's points ([EXPERIMENTS.md](EXPERIMENTS.md#e3-the-background-tiers-cost-of-make-room) has the counters and their limits). The median of each configuration's runs:

| Configuration | Runs | Engine steps over the calls | Steps the cap held agent prefill back in (share) | Prefill tokens held back (an upper bound) | make-room's preemptions | vLLM's preemptions | Tokens loaded beyond first scheduling | Agents' tokens/s, first 240 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| The frozen box | 10 | 5,778 | — | — | — | 31 | 212,630 | 3,305 |
| Preempt-resubmit (arm 8c) | 5 | 5,390 | — | — | — | 7 | 273,461 | 3,332 |
| #63350 alone | 5 | 5,430 | 0 (0%) | 0 | 34 | 51 | 171,736 | 3,414 |
| #63350 + #63351, 16 tokens | 5 | 5,161 | 1,310 (25%) | 325,689 | 32 | 47 | 117,049 | 3,298 |
| **#63350 + #63351, 48 tokens** | 15 | 5,404 | 1,778 (33%) | 382,720 | 31 | 46 | 117,044 | 3,203 |
| The same on v0.3's module | 5 | 4,939 | — | — | — | 41 | 123,193 | 3,132 |
| Standard scenario: the frozen box | 5 | 10,755 | — | — | — | 0 | −74 | 7,131 |
| Standard scenario: 48 tokens | 5 | 10,988 | 505 (4%) | 31,520 | 1 | 1 | 115 | 6,815 |
| 4 callers: 48 tokens | 3 | 6,349 | 3,321 (53%) | 747,879 | 54 | 65 | 111,679 | 3,055 |
| 6 callers: 48 tokens | 3 | 6,727 | 4,332 (64%) | 1,067,013 | 64 | 70 | 82,548 | 2,716 |

- **The agents pay through the budget cap.** At 48 tokens the cap held lower-priority prefill back in a third of the engine's steps over the calls (33%; 25% at 16 tokens). #63350 alone holds nothing, and there the agents' goodput is 3% above the frozen box's.
- **The KV preemption's victims are few.** make-room preempted about 31 requests a run at 48 tokens and 34 with #63350 alone, against a pool the agents fill; vLLM's own count of every preemption, the frozen box's growth preemptions included, is 46 against the frozen box's 31. The tokens loaded from the CPU tier beyond the requests' first scheduling, the victims' restores among them, are 117,044 at 48 tokens against the frozen box's 212,630 and preempt-resubmit's 273,461.
- **Where the pool has room** (the standard scenario), make-room preempted 1 request a run at the median (0–7) and the cap held agent prefill back in 4% of the steps (3–13%); the loads beyond first scheduling are within the window's edges of zero (−74 and 115 tokens).
- **The held prefill tokens are an upper bound.** The module W5's points ran counted each held chunk against the step's remaining budget, so a step that held several lower-priority prefills counts each of them. With `--long-prefill-token-threshold 256` a held chunk counts at most 256 of a step's 512 tokens, so a step overstates only when it held three or more; the 48-token runs count 215 tokens a held step at the median (171–260 over the fifteen runs), about one chunk's worth. The PRs bound a step's count by its budget since #63351's revision d1399e208, and so does the variant built from them.
- **What a resume recomputes is not measured**: vLLM 0.32 counts a request's computed prompt tokens at its first scheduling only ([EXPERIMENTS.md](EXPERIMENTS.md#e3-the-background-tiers-cost-of-make-room)).

### The step cap, screened and confirmed (SW, RG)

SW, 2026-10-27: two screens of the PRs' configuration over the step cap, the per-request threshold, the window and the gateway's static cap on agent requests in flight, and of the frozen box at two step caps, one run a setting on W5's tuning sample (agent seed 11) beside the 24 sessions of the discriminating scenario ([the sweep's record](../benchmarks/results/2026-10-27-sweep/README.md), whose `table.py` gives every point). The step cap moved the PRs most: 9 of 59 replies over W5's SLO at 256 without the threshold, 24 at 512 and 33 at 1,024; the static caps did less, and the frozen box missed at every setting. RG's five rounds (agent seeds 21–25, Phastform and Gray in turn, below) then ran the three configurations side by side, each round against its own SLO:

| Configuration | Rounds | V2V p95: median of the rounds / pooled | Replies over the SLO | By round | Agents' tokens/s, first 240 s | Agent TTFT p50 / p99 |
|---|---:|---:|---:|---|---:|---:|
| The frozen box | 5 | 12,010 / 11,766 | 261 / 295 (88.5%) | 53, 55, 55, 54, 44 | 3,399 | 69 / 109 s |
| #63350 + #63351, a 48-token window, 512/256 (W5's) | 5 | 2,246 / 2,259 | 94 / 295 (31.9%) | 16, 29, 25, 9, 15 | 3,303 | 78 / 117 s |
| **#63350 + #63351, a 48-token window, 256 without a threshold** | 5 | **2,098 / 2,151** | **44 / 295 (14.9%)** | 10, 17, 10, 4, 3 | 3,067 | 85 / 120 s |

- **The screens' choice stands.** At 256 tokens a step without a threshold the PRs had fewer replies over the SLO than at W5's 512/256 in each of the five rounds, half as many pooled, and the serving profile runs it. The agents pay 7% of their token rate for it (3,067 against 3,303 tokens/s) and their median TTFT rises from 78 to 85 s.
- **Still over the requirement at 24 sessions.** 14.9% of the replies were over, where the SLO allows 5%; the two-card comparison below finds the shared card within it at four sessions. Each round's SLO is in [SLO.md](SLO.md#the-slos-reference-campaign-by-campaign), and the record's `gate.py` gives this table.

### Two cards (RG)

RG, 2026-10-28: the resource gate's comparison of the arrangements of two RTX 5060 Ti cards ([RESOURCE-GATE.md](../benchmarks/quality/RESOURCE-GATE.md), fixed before the runs), in three rounds on new samples of the agents' traces (agent seeds 21–23), on Phastform, Gray and Phastform in turn, the hosts and cyan at v1.6-27-gea7efd0. Each card's part of an arrangement is a point of its own, beside the discriminating scenario's sessions at the counts given; the change runs the serving profile, the PRs' switches with a 48-token window at a step cap of 256 without a per-request threshold. Each round's SLO is its voice alone's p95 plus 250 ms: 2,000, 1,972 and 2,010 ms with two callers, 1,988, 1,943 and 1,982 ms with one. Replies over the SLO are pooled over the rounds, and the agents' tokens/s over the calls' first 240 s are summed over the cards, the median of the rounds with each round's sum:

| Arrangement | Sessions (shared + second card) | Replies over the SLO | Agents' tokens/s, first 240 s | Rounds | Agent TTFT p50 / p99 |
|---|---:|---:|---:|---|---:|
| Dedicated roles, controlled (the second card the frozen box, its speech servers idle) | 0 + 12 | 0 / 177 | 6,302 | 7,008, 6,302, 6,041 | 4.4 / 31.0 s |
| **Dedicated roles, best practical** (the second card agents alone: a 61,152-token pool, FCFS, 2,048-token steps) | 0 + 14 | 0 / 177 | **7,893** | 8,282, 7,893, 4,710 | 5.1 / 41.0 s |
| Stock pooling | 12 + 12 | 53 / 87 | 13,188 | 13,233, 13,188, 11,951 | 4.7 / 29.2 s |
| Flatshare pooling | 12 + 12 | 24 / 87 | 12,111 | 13,138, 12,111, 11,374 | 4.8 / 37.2 s |
| Consolidation, stock vLLM | 24 + 14 | 163 / 177 | 11,378 | 11,681, 11,378, 7,776 | 39.7 / 85.8 s |
| Consolidation, stock vLLM | 12 + 14 | 95 / 177 | 14,191 | 14,987, 14,191, 11,055 | 4.6 / 33.5 s |
| Consolidation, stock vLLM | 8 + 14 | 69 / 177 | 13,573 | 13,573, 14,267, 10,547 | 0.7 / 11.6 s |
| Consolidation, stock vLLM | 4 + 14 | 18 / 177 | 10,596 | 11,655, 10,596, 8,308 | 0.6 / 10.4 s |
| Consolidation, stock vLLM | 2 + 14 | 4 / 177 | 10,051 | 10,984, 10,051, 6,234 | 0.4 / 7.8 s |
| Consolidation, the change | 24 + 14 | 37 / 177 | 11,075 | 11,466, 11,075, 7,435 | 48.0 / 92.5 s |
| Consolidation, the change | 12 + 14 | 33 / 177 | 13,880 | 15,107, 13,880, 10,082 | 5.1 / 41.6 s |
| Consolidation, the change | 8 + 14 | 31 / 177 | 13,537 | 13,537, 14,352, 10,392 | 0.7 / 11.7 s |
| **Consolidation, the change** | 4 + 14 | 5 / 177 | **10,596** | 11,655, 10,596, 8,308 | 0.6 / 10.8 s |
| Consolidation, the change | 2 + 14 | 2 / 177 | 10,050 | 10,982, 10,050, 6,213 | 0.4 / 9.8 s |
| Consolidation, the change at W5's configuration | 24 + 14 | 70 / 177 | 11,196 | 11,596, 11,196, 7,553 | 44.4 / 88.0 s |

- **The verdict.** An arrangement qualifies with at most 5% of its replies over the SLO. Consolidation on the change qualifies with four sessions on the shared card (2.8%) and not with eight (17.5%), and at four it did 34.2% more than the best qualifying dedicated arrangement, the agents' card at 14 sessions: past the gate's 20% hurdle. Pooling qualifies on neither engine.
- **Most of it is the arrangement's.** Stock vLLM's shared card qualifies at two sessions, and that arrangement did 27.3% more than the best dedicated one; the change's own value, its envelope against stock's, is 5.4%, short of the hurdle. At two and four sessions both engines completed the same agents' requests in the calls' first 240 s in every round (at four, 44, 41 and 37), so there the shared card's part is the sessions' own pace, which the traces' think times set.
- **The dedicated card's best count.** The agents' card did the most at 14 sessions (12 in the third round); at 24 its pool thrashes, the agents' median TTFT is 48 s and it did 4,232 tokens/s, 46% below its best.
- **The margin.** With 500 ms over voice alone in place of 250, the change's shared card would qualify at every count measured, 8 of 177 replies over at 24 sessions, and stock vLLM's still only at two (12 of 177 at four).
- **Where the margin goes.** From eight sessions on the transcript comes 51–71 ms later than with the calls alone on either engine (745 ms at the median), the speech server's share of the GPU, and on stock vLLM the first sentence's p95 grows to 909, 2,424 and 9,719 ms at 4, 8 and 24 sessions, where the change holds it to 534, 737 and 768 (437 alone).
- **A warm spare**, the second card on the change's engine with its speech servers resident, did the most at 12 sessions, 6,715 tokens/s; with consolidation at four sessions, 9,652 tokens/s, 22.3% over the best dedicated arrangement, which has no standby.

The records are in [`benchmarks/results/2026-10-28-rounds`](../benchmarks/results/2026-10-28-rounds), whose `gate.py` gives these tables, the margins, the agents' requests in the window and the stages from them.

## The control plane: flatshare-gateway in front of the engine

On cyan, the voice agent and the agent replay reach the engine through `flatshare-gateway`:
- the voice agent on a private Unix socket, as tenant `voice` (realtime class);
- the agent replay over loopback TCP with a key, as tenant `agents` (background class).

The gateway forwards over the SSH tunnel to the frozen box. Measured W1, 2026-10-22.

**What it adds to a voice request, paired.** The voice agent's own LLM request ran 40 rounds, alternating between the engine's tunnel socket and the gateway's voice socket, with greedy decoding so both paths generate the same text:

| Path | First token p50 | First sentence p50 | End p50 | Gap between chunks, p50 / p99 |
|---|---|---|---|---|
| Engine socket | 43.7 ms | 223.0 ms | 370.1 ms | 12.13 / 12.64 ms |
| Through flatshare-gateway | 44.1 ms | 223.3 ms | 370.3 ms | 12.14 / 12.69 ms |
| **Paired median difference** | **+0.38 ms** | **+0.26 ms** | **+0.25 ms** | — |

The same probe on the final transport (W6, 2026-10-25: every server on a Unix socket, the gateway's upstream the
tunnel's engine socket), 40 rounds again, the same text on both paths ([records](../benchmarks/results/2026-10-25-w6-transport)):

| Path | First token p50 | First sentence p50 | End p50 | Gap between chunks, p50 / p99 |
|---|---|---|---|---|
| Engine socket | 43.3 ms | 162.1 ms | 344.1 ms | 12.08 / 12.53 ms |
| Through flatshare-gateway | 43.8 ms | 162.7 ms | 344.7 ms | 12.09 / 12.52 ms |
| **Paired median difference** | **+0.29 ms** | **+0.40 ms** | **+0.45 ms** | — |

**The final transport against W5's** (the transport A/B, 2026-10-25): `flatshare-bench voice-turn` (30 turns of the two LibriSpeech clips) and `flatshare-bench tiers` (5 reps) from cyan on the frozen engine, with the same clips, reps and cores, once through W5's tunnel, one SSH connection's TCP forwards (at v0.3-90), and once through the final build's, an SSH session of its own for each connection to each socket (at v0.3-274), p50 / p90 in ms ([records](../benchmarks/results/2026-10-25-w6-transport), whose `ab.py` prints this table):

| Stage | TCP forwards (W5's tunnel) | Sockets (flatshare-tunnel) | Sockets − TCP, p50 |
|---|---:|---:|---:|
| STT | 152.6 / 161.3 | 152.5 / 160.9 | −0.1 |
| LLM first token | 49.1 / 50.5 | 40.7 / 41.5 | −8.4 |
| LLM first sentence | 252.5 / 317.9 | 249.9 / 316.6 | −2.6 |
| TTS first audio | 176.3 / 187.6 | 175.6 / 186.7 | −0.8 |
| **Turn** | 584.6 / 655.2 | 581.0 / 655.1 | −3.6 |
| LLM whole reply | 303.0 / 369.2 | 263.3 / 330.0 | −39.7 |

- **The turn is the same:** −3.6 ms on the sockets at the median, the STT, the first sentence and the first audio −0.1, −2.6 and −0.8 ms, the LLM's first token −8.4 ms. The TCP side's probe (before d4dd9d9) opened a connection for each TTS request, which through W5's forwards is a channel on the SSH connection: `GET /health` took 2.4–3.2 ms on a new connection against 2.1–2.7 ms on a kept one (p50 of 50 for each server; the TTS's new ones 44.7 ms at p95, the tail below), so the TCP side's first audio holds about 0.4 ms at the median that the socket side's, on a kept connection, does not.
- **The sockets drop a 40 ms tail.** On the TCP side a stream ended 51 ms after its reply's text was complete, at the median, against 13 ms on the sockets, with the same text on both, so the LLM's whole reply, which the turn does not wait for, is 39.7 ms shorter; and `tiers` had 2 of its 21 cells' medians 29 and 40 ms longer over TCP, the other 19 within 10 ms. 40 ms is Linux's shortest delayed acknowledgment, which a small last write held back by Nagle's algorithm waits for; these records do not say which TCP connection of the path held it.

**Its own time, in place** (`flatshare_gateway_overhead_seconds`: from the end of a request's body to the upstream write). The voice agent's turns are seconds apart:

| Upstream keep-alive (engine / gateway) | Requests | Mean | p99 | Within 0.5 ms | First byte p99 |
|---|---|---|---|---|---|
| 5 s / 4 s (vLLM's default) | 60 | 0.24 ms | 0.50 ms | 100% | 205 ms |
| 60 s / 55 s (the frozen box from W1) | 60 | 0.13 ms | 0.44 ms | 100% | 99 ms |
| 60 s / 55 s, beside the 16-session agent replay (both classes) | 603 | 0.16 ms | 0.58 ms | 98.5% (all within 2 ms) | 700 ms (realtime) |

- **A client's upload is not overhead.** The first in-place reading was 1.8 ms on average, against about 0.03 ms in process. Of that, 1.57 ms was the voice worker's event loop sending the request body after its headers, which a direct call waits for too. The overhead metric therefore starts at the end of the body, and the upload is timed separately (`flatshare_gateway_request_body_seconds`: 1.35–1.57 ms mean for the voice agent).
- **Connections must outlive the turn gap.**
  - vLLM drops idle connections after 5 s, and voice turns are often further apart. So most voice requests opened a new connection through the tunnel.
  - A new connection costs about 1 ms, and now and then 40 ms more. One-token requests 6 s apart show it:

    | | Gateway's own time | First byte mean | First byte p99 |
    |---|---|---|---|
    | 5 s keep-alive | 0.26 ms | 41.8 ms | about 90 ms |
    | 60 s keep-alive | 0.08 ms | 38.5 ms | 49.8 ms |

  - The engine now keeps idle connections for 60 s (`VLLM_HTTP_TIMEOUT_KEEP_ALIVE`) and the gateway for 55 s.
  - Through the final build's tunnel a new connection is a new SSH session: `GET /health` took 13.1–13.9 ms
    on a new connection against 2.2–2.8 ms on a kept one, for each of the three sockets (p50 of 50,
    cyan to Phastform), so about 11 ms, where through W5's forwards it was a channel, about 0.4 ms
    (above). The servers keep idle connections 60 s for it, and the voice worker's clients 55 s.
- **End to end it is inside the run-to-run spread.**
  - Voice alone through the gateway: V2V p50 1,684, 1,600 and 1,633 ms over three runs.
  - Straight to the engine: 1,614 and 1,579 ms over two runs, every run with 0 errors.
  - Beside the agent replay: 1,727 / 2,361 / 3,141 and 1,710 / 2,361 / 2,872 ms (p50 / p95 / p99) over two runs through the gateway, against 1,664 / 2,312 / 3,013 ms direct.
- **In process** (`cargo run --release -p flatshare-gateway --example overhead` on cyan, [`flatshare-gateway` README](../crates/flatshare-gateway/README.md#overhead)), the gateway's own time is 32 µs on average. Against a direct call it adds 226 µs at p50 for one client, and at most 1.92 ms at p99, with 64 concurrent requests sharing the process with their client and engine.
- **Under load it carried both classes without an error.** Over the two runs:
  - the replay sent 578 and 543 background requests and the calls 60 realtime ones each;
  - every request completed except one realtime stream, which the voice agent closed early, as it does on a barge-in or when it replaces a preemptive reply;
  - there were no upstream errors;
  - agent TTFT was 141 and 146 ms p50, and 9.9 and 9.2 s p99.

## The API layer: agentic-api in front of the engine

A coding agent such as Claude Code or Codex reaches a self-hosted engine through an API layer that serves the APIs it speaks, so that layer's cost is part of every agent turn. This section measures one, the vLLM project's [agentic-api](https://github.com/vllm-project/agentic-api), with that project's own `response_provider` benchmark, which compares its WebSocket, HTTP/SSE and JSON paths with direct vLLM, all five arms at once against the frozen box with tool calling on (agentic-api a7dc7b5, 2026-10-22). E2's agents, replayed by AIPerf, reach the engine through `flatshare-gateway` alone. Per-turn latency for a 4-round tool loop, p50 in ms:

| Sessions per arm | agentic-api HTTP/SSE | vLLM HTTP/SSE | Paired median difference |
|---|---|---|---|
| 1 | 3,173 | 3,176 | 0 ms |
| 5 | 4,489 | 4,482 | +3 ms |
| 10 | 6,102 | 6,072 | +1 ms |

- **Latency.** agentic-api's latency cost is below the noise at up to 50 concurrent tool loops; the inference server dominates.
- **Resources.** It used about 5.5 ms of CPU per Responses round and 44–54 MiB of memory.
- **Request size.** Clients send 3.6–3.8 KB per turn through it against 8.4 KB direct, because it holds the conversation state.

The method, all five arms, correctness and the history-rehydration runs are in [`benchmarks/results/2026-10-22-agentic-api-http-baseline`](../benchmarks/results/2026-10-22-agentic-api-http-baseline/).

## Engine plugins

The lease-aware plugins of v0.3 ([its tree](https://github.com/StevenWang-CY/flatshare/tree/v0.3/python/leaseholder_vllm), where the package has its former name, `leaseholder_vllm`), measured in W1 and never used; they were removed after v0.3 ([ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md)).

`LeasePolicy` at a full-size tier of 18,204 chunks (40 GiB of Qwen3-4B BF16 KV), against vLLM's real offload manager on CPU:

| Operation | LeasePolicy | Stock LRU |
|---|---|---|
| `prepare_store` plus evicting 128 chunks, mostly leased | 0.30 ms | 0.21 ms |
| Finishing a 1,011-chunk session: renewal | 0.39 ms | — |
| Finishing a 1,011-chunk session: class change | 0.28 ms | — |

**Live check on the frozen box** ([its tests in the v0.3 tree](https://github.com/StevenWang-CY/flatshare/tree/v0.3/python/leaseholder_vllm/tests/live)):
- the lease hint reached both plugins;
- `X-Session-ID` appeared in GPU-tier KV events and in no CPU-tier one, and `LeaseConnector` mirrored it into the hint;
- a 7,921-token prompt was written through to the CPU tier. After the GPU prefix cache was reset, the same prompt was restored with 7,920 tokens from the CPU tier, as reported by both the hit receipt and `usage.prompt_tokens_details.cached_tokens`.

## The KV case: make room

`python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py` reproduces the KV case of vLLM [#40004](https://github.com/vllm-project/vllm/issues/40004) on the frozen box's image, on Gray (the second RTX 5060 Ti, with the same units, pins and pool), on 2026-10-23. Two priority-2 requests of 17,999 prompt tokens, each decoding up to 4,096 tokens, fill the 36,400-token pool. Once the engine reports both running and at most 24 blocks free, a priority-0 chat request of about 750 tokens is sent, and its time to first token is measured. The outputs of this section's runs, with each run's revision and switches, are in [`benchmarks/results/2026-10-23-make-room-probes`](../benchmarks/results/2026-10-23-make-room-probes), whose `check.py` recomputes the three tables.

The rows name `make-room`'s switches: the KV preemption (at most 2 victims a step), the budget cap and the 2,048-token reserve. Pull request [#63350](https://github.com/vllm-project/vllm/pull/63350), proposed to vLLM and under review, carries the KV preemption, and [#63351](https://github.com/vllm-project/vllm/pull/63351), stacked on it, the budget cap, without the reserve. The KV case runs without the reserve, which would keep the second priority-2 request out of the full pool; 22 of 2,274 blocks were free when the request was sent, in every run on a full pool.

| Scheduler | Runs | Priority-0 TTFT | Then |
|---|---:|---|---|
| Stock vLLM 0.32.0, priority scheduling | 3 | 4,923, 5,069, 5,077 ms | 2 running, 1 waiting |
| `make-room`, the KV preemption alone (#63350) | 2 | 247, 321 ms | 2 running, 1 waiting |
| `make-room`, the KV preemption and the budget cap (#63350 and #63351) | 2 | 225, 297 ms | 2 running, 1 waiting |
| Either, on an idle pool | 4 | 107–137 ms | — |

- **Why stock waits.** A KV timeline of the same load on the stock scheduler (`/metrics` every half second) shows the two prefills filling the pool to 98.7% at 7.7 s, then decoding into its last free blocks until 12.6 s, when one of them cannot grow and vLLM preempts the later one, and use falls to 50%. A priority-0 request that arrives in between waits: the waiting loop stops at its failed allocation, and running requests are preempted only for each other's growth. Here it waited 4.9 to 5.1 s.
- **What make-room does.** It preempts one priority-2 request at once, through the stock preemption path; with the OffloadingConnector the victim's KV is in the CPU tier for its resume. Both runs end in the same state, one agent request waiting, but the voice turn's first token comes 4.6 to 4.9 s sooner. The budget cap changes nothing here, since the priority-2 requests are decoding rather than prefilling: the 16th token came after 535 and 610 ms with the KV preemption alone, and 514 and 586 ms with the budget cap.
- **An instrument note.** `vllm:num_preemptions_total` stayed 0 through the stock timeline's preemption: vLLM counts a preemption when the victim's next output reaches the frontend, so a victim still waiting is not counted yet. E2's per-point engine counters therefore miss victims that resume after a point's window, and the probe reads the running and waiting counts instead.

**The prefill case.** The probe's `prefill` case sends three priority-2 requests of 10,000 prompt tokens, which prefill beside each other with room left in the pool, and the voice turn once all three run. It times the turn's first token and its 16th, the end of the budget cap's window (on Gray, 2026-10-23; the median and range where a row has four runs or more):

| Scheduler | Runs | Priority-0 TTFT | 16th token |
|---|---:|---|---|
| Stock vLLM 0.32.0, priority scheduling | 6 | 361 ms (290–596) | 1,395 ms (1,226–1,683) |
| `make-room` without the budget cap (`budget_cap_tokens 0`), with the reserve | 2 | 324, 339 ms | 1,248, 1,277 ms |
| `make-room`, the KV preemption alone (#63350) | 4 | 381 ms (329–589) | 1,423 ms (1,370–1,674) |
| `make-room`, the KV preemption and the budget cap (#63350 and #63351) | 4 | 320 ms (263–328) | 573 ms (564–618) |
| `make-room` as arm 9 runs it, with the reserve | 2 | 282, 354 ms | 578, 651 ms |

- **Why.** Every step of the reply carried the agents' prefill chunks, up to 512 tokens, so its tokens came 62 to 73 ms apart, on stock vLLM as with the KV preemption alone, which finds KV to spare here and changes nothing; with the budget cap they come 16 to 20 ms apart. The first token still waits for the step in flight when the turn arrives. E2's first arm-9 point, on the first module, showed the same loss at scale: its waterfall put it in the reply's first sentence (1,335 ms at the median against 536 ms to the first token).
- **Spread.** The case varies with how far the three prefills have got when the turn arrives: stock vLLM's first two runs gave 1,226 ms each, four later ones 1,367–1,683 ms. With two runs a row, the KV preemption alone would have looked slower than stock.

**The calls' first-token tail.** `python/flatshare_vllm/src/flatshare_vllm/make-room/tail.py` runs 12 closed-loop agent sessions at priority 2 beside two calls at priority 0 for 240 s and gives the calls' time to first token. The agents' prompts are 6–30k tokens, either fresh each turn, as E2's discriminating replay computes about 95% of its prompt tokens, or growing by a tool result each turn, so that they reuse the CPU tier; a call turns every 4–8 s with 16 output tokens, the budget cap's window, so neither this check nor the prefill case shows a reply that outruns the window. On Gray, 2026-10-23:

| Scheduler | Agents' prompts | Call turns | TTFT p50 / p90 / p95 / max |
|---|---|---:|---|
| Stock vLLM 0.32.0, priority scheduling (the frozen box) | fresh | 47 | 3,081 / 6,529 / 8,239 / 11,094 ms |
| Stock vLLM 0.32.0, priority scheduling (the frozen box) | growing | 50 | 1,359 / 6,816 / 7,258 / 8,692 ms |
| `make-room`, the KV preemption alone (#63350) | fresh | 60 | 643 / 1,715 / 2,730 / 6,298 ms |
| `make-room`, the KV preemption alone (#63350) | growing | 60 | 540 / 1,102 / 1,950 / 3,325 ms |
| `make-room` with the reserve, before its KV-load rule | fresh | 68 | 361 / 1,546 / 2,812 / 4,120 ms |
| `make-room` with the reserve, before its KV-load rule | growing | 70 | 425 / 605 / 2,395 / 3,544 ms |
| `make-room`, the KV preemption and the budget cap (#63350 and #63351) | fresh | 73 | 257 / 346 / 353 / 370 ms |
| `make-room`, the KV preemption and the budget cap (#63350 and #63351) | growing | 73 | 261 / 387 / 434 / 604 ms |
| `make-room` as arm 9 runs it, with the reserve | fresh | 73 | 202 / 283 / 316 / 398 ms |
| `make-room` as arm 9 runs it, with the reserve | growing | 73 | 202 / 289 / 324 / 678 ms |

The rows before the KV-load rule ran with the scheduler's state logged for each call that waited over 0.3 s; the stock rows' p95 is near that of E2's frozen box in the discriminating scenario (the realtime first byte, 8,556 ms). Each row is one run; the percentiles are inclusive, as E2's report computes them, from the run's per-turn lines.

- **Why.** Between a call's turns, the agents' prefills cycle the GPU's free blocks, so the turn's prefix is evicted and loads back from the CPU tier, asynchronously. Every slow turn was in `WAITING_FOR_REMOTE_KVS` (the scheduler's state logged each step), with KV to spare. Once its load is done, a request waits for the waiting loop to promote it, and the waiting loop runs only on budget the running loop leaves; the agents' 512-token chunks left none. The budget cap now holds lower-priority prefill while a higher-priority request loads, as while it prefills.
- **Each switch's share.** The KV preemption alone takes the calls' p95 from 7.26–8.24 s to 1.95–2.73 s, since a turn no longer waits for an agent request to free its blocks; the budget cap takes it under 0.5 s. With the reserve as well, the median is lower (202 against 257 and 261 ms).
- **E2's points show the same tail:** disc-a9-s24 on the module before this rule had its first token after 334 ms at the median and 2,146 ms at p95, against preempt-resubmit's 251 and 394 ms; its turns loaded about 40k tokens from the CPU tier, about their own prefixes.

### W5: the fixed make-room on Phastform

W5 fixed make-room's rules ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351)) and checked each module on Phastform's RTX 5060 Ti, the GPU of every E2 point, with the frozen box's units, pins and pool, on 2026-10-24. Every block ran make-room in the PRs' configuration (the KV preemption at 2 victims a step, the budget cap with its 16-token window, no reserve): the KV case 3 times, the prefill case 4 times, and `tail.py` with fresh and with growing prompts twice each, one block per module. The outputs are in [`benchmarks/results/2026-10-24-w5-probes`](../benchmarks/results/2026-10-24-w5-probes), whose `check.py` recomputes the table (the median and range where a cell has four runs; a tail cell pools both runs' 145–146 turns, none failed, and its percentiles are inclusive, as E2's report computes them):

| Module | KV case TTFT | Prefill case TTFT / 16th token | Calls' TTFT, fresh prompts, p50 / p95 | Growing prompts, p50 / p95 |
|---|---|---|---:|---:|
| v0.3's, which E2 measured before W5 | 212, 286, 287 ms | 322 (302–365) / 619 (599–661) ms | 227 / 365 ms | 236 / 388 ms |
| W5's fixes as first written: a KV load in flight holds nothing until it is done | 215, 289, 290 ms | 368 (355–395) / 664 (652–692) ms | 358 / 567 ms | 361 / 563 ms |
| The same, with a load in flight holding lower-priority prefill for its first 16 steps | 217, 290, 291 ms | 316 (311–323) / 611 (560–620) ms | 248 / 371 ms | 251 / 393 ms |
| As merged: the bounded load hold | 216, 290, 291 ms | 309 (301–318) / 606 (597–614) ms | 232 / 356 ms | 238 / 382 ms |

- **What the first fixes cost, and what gave it back.** Holding nothing while a load is in flight lets the agents' prefill run through a call's KV load: the calls' first token came 125–131 ms later at the median and 175–202 ms later at p95 than on v0.3's module. Holding prefill for the load's first 16 steps gives it back, and bounds a load that never completes, which v0.3's unbounded hold did not; the merged module is within 5 ms of v0.3's at the median and 6–9 ms below it at p95.
- **The KV case and the prefill case do not move** across the four modules: the voice turn's first token comes 212–291 ms after it is sent into a full pool (stock vLLM waits 4.9–5.1 s), and its 16th 606–664 ms after it is sent behind three agent prefills.
- **The engine's make-room lines.** The first fixes' generated module rebound the scheduler's logger, so its counter lines never reached the journal; the merged module keeps vLLM's logger (v0.3's logged none).

### The PRs' configuration in one session (W5)

The probes #63350 and #63351 cite, on Phastform on 2026-10-25, after the PR campaign and on the make-room it ran: stock vLLM 0.32.0 (the frozen box), the KV preemption alone, and both PRs at the budget cap's default 16-token window, two preemptions a step and no reserve. Each block ran the KV case and the prefill case four times and `tail.py` with fresh and with growing prompts twice ([`pr-probes/`](../benchmarks/results/2026-10-24-w5-probes)); a tail cell pools both runs' turns, none failed, at inclusive percentiles:

| Engine | KV case TTFT | Prefill case TTFT / 16th token | Calls' TTFT, fresh prompts, p50 / p95 | Growing prompts, p50 / p95 |
|---|---|---|---:|---:|
| Stock vLLM 0.32.0, priority scheduling | 4,968 ms (4,900–4,971) | 529 (313–558) / 1,500 (1,220–1,529) ms | 2,244 / 6,830 ms | 1,324 / 7,158 ms |
| #63350 alone (the KV preemption) | 290 ms (219–290) | 416 (304–556) / 1,350 (1,211–1,528) ms | 716 / 4,111 ms | 615 / 1,073 ms |
| #63350 and #63351 (the budget cap, 16 tokens) | 290 ms (216–294) | 304 (292–318) / 575 (536–615) ms | 236 / 388 ms | 258 / 358 ms |

- **The KV case is #63350's.** The voice turn has its first token after 0.22–0.29 s instead of 4.9–5.0 s, and the budget cap changes nothing there. In one run with both, the reply ended before its 16th token; this case measures the first.
- **The prefill case and the tail are #63351's.** With #63350 alone, which finds KV to spare in the prefill case, the reply's tokens still come 60–65 ms apart behind the agents' chunks and the calls' p95 stays at 1.1–4.1 s; with the budget cap they come 16–20 ms apart and the p95 is under 0.4 s.


### After W5: the first tokens in flight

The frozen box schedules asynchronously: vLLM schedules a step before it has processed the output of the step before it. W5's make-room, like #63351 until its revision of 2026-10-25 ([9013ed433](https://github.com/vllm-project/vllm/pull/63351)), counted a request's first tokens from the outputs processed, which lag a step behind those scheduled, so the budget cap held lower-priority prefill back one step past its window. Both now count the tokens in flight too (vLLM's output placeholders); a test on vLLM's own scheduler holds a 4-token window as many steps under asynchronous scheduling as without, and fails on the earlier count. Traced against the module W5 ran, the fixed make-room schedules alike in all 6,040 cases without the budget cap or with synchronous scheduling, and differently in 2,224 of the 2,712 asynchronous cases with it.

Only a reply of at least the window's tokens meets that step, since a shorter one ends the hold when it ends. At the 16-token window, each such reply held agent prefill back a 17th step: 81% (248 of 305) of the replies in W5's 16-token points. At the 48-token window, 1 of the 2,469 replies in W5's 48-token points reached 48 tokens, so their measurements stand but for that one. The probes above measure the first token and the 16th, both before the step.

### The window's knee

`knee.sh` on Phastform on 2026-10-25, on the box's units and pool: make-room with the PRs' switches at a budget-cap window of 16, 24, 32, 48 and 64 tokens, each read back from the engine, and for each two 240 s runs of `tail.py` beside the box's closed-loop agents, with fresh prompts and call replies of 40 tokens timed to their first, 16th and last token ([the records](../benchmarks/results/2026-10-25-window-knee/README.md)). The medians of each window's two runs, in ms; no turn failed:

| Window (tokens) | First token p95 | 16th token p95 | 40th token p95 | Agents' prompt tokens/s |
|---:|---:|---:|---:|---:|
| 16 | 448 | 726 | 3,759 | 3,483 |
| 24 | 387 | 648 | 2,642 | 3,443 |
| 32 | 326 | 572 | 1,854 | 3,300 |
| 48 | 347 | 616 | 1,037 | 3,205 |
| 64 | 366 | 612 | 1,018 | 3,246 |

The reply's tail falls until the window covers the reply, 3.8 s at 16 tokens to 1.0 s at 48, and moves no further at 64; the first token, which the window does not touch, stays at 326–448 ms. The agents' prompt throughput falls 8% from 16 to 48 and no further, since a hold ends with its reply: a window longer than the replies costs no more than one just long enough ([choosing N](PRODUCTION.md#the-replys-first-tokens)).

## Agent replay on the frozen box: a first retention-headroom signal

Measured in W0: AIPerf `agentic_replay` of the 32×-scaled Claude Code corpus, 16 agent sessions, think times capped at 120 s, `ignore_eos`, 10 minutes, against the frozen box through the tunnel. `flatshare-bench kv-metrics` compared vLLM's sampled KV block histograms over the run.

**Load.** 688 requests, 0 errors.

| | p50 | p90 | p99 |
|---|---|---|---|
| TTFT | 135 ms | 815 ms | 8,501 ms |
| Inter-token latency | 15.6 ms | 40.9 ms | 59.1 ms |

**Where prompt tokens came from.** 93.4% were served from cache:

| Source | Share of prompt tokens |
|---|---|
| GPU prefix cache | about 57% |
| Restored from the CPU tier | about 37% |
| Computed | 6.6% |

`flatshare-sim`'s replay of this run splits the computed share (6.1% there) into new context each turn adds (3.15 points), recompute after eviction from both tiers (2.05) and cold first turns (0.91).

**KV block lifecycle.**

| Histogram | p50 | p90 |
|---|---|---|
| Reuse gap | 2.54 s | 6.33 s |
| Idle before eviction | 6.22 s | 27.3 s |

**Reading.**
- Half of the evicted blocks (50.4%) had been idle no longer than the 90th-percentile reuse gap, so LRU evicts sessions that are coming back.
- With the 4× write-through CPU tier, those misses cost a restore, not a recompute. W0 read the headroom on this configuration as GPU placement and prefetch timing, not CPU-tier retention; E1 has since superseded that reading: in the simulator even perfect knowledge of the returns buys under 9% of the median resume ([below](#agent-replay-at-three-loads-and-the-simulator-calibrated-on-it-w2)).
- The reuse-gap histogram counts only blocks that survived until reuse, so it is biased short. This is a signal, and the simulator's clairvoyant reference is the measurement.

## Agent replay at three loads, and the simulator calibrated on it (W2)

AIPerf `agentic_replay` of the 32× corpus against the frozen box (straight to the engine, think times capped at 120 s, 10 minutes plus a 60 s grace), each on a freshly started engine, 2026-10-22. TTFT is AIPerf's profiling phase; the prompt-token split is vLLM's `prompt_tokens_by_source` counters over the whole run, warm-up included. `make reproduce-calibration` recomputes the table, and the simulator's figures below, from the runs' exports, an asset of the [v0.3 release](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3) ([`crates/flatshare-sim`](../crates/flatshare-sim/README.md#calibration-against-the-gpu-operating-points)).

| Agent sessions | Requests | TTFT p50 / p90 / p99 | ITL p50 | Prompt tokens: GPU / CPU tier / computed | Preemptions |
|---|---|---|---|---|---|
| 8 | 408 | 84 / 376 / 1,554 ms | 13.2 ms | 64.3 / 28.0 / 7.7% | 0 |
| 16 | 686 | 132 / 787 / 9,198 ms | 15.6 ms | 55.2 / 34.6 / 10.2% | 3 |
| 24 | 1,568 | 175 / 1,415 / 15,605 ms | 22.0 ms | 51.4 / 33.4 / 15.2% | 2 |

- **Repeatability.** Three 16-session runs (W0's, one on a warm engine and the fresh one above) agree within 1.5 points on the split and range from 787 to 1,023 ms at TTFT p90. At 24 sessions a run on a warm engine was slower (TTFT p90 1,900 ms).
- **The simulator against them.** `flatshare-sim` with the W2 contention model (the step loop and a fitted step time; [its README](../crates/flatshare-sim/README.md#the-contention-model)) reproduces resume TTFT p50 / p90 within 4–8% at 8 and 16 sessions, also when each point is left out of the fit, and request counts within 1%. At 24 sessions it is 18% / 30% fast, because restores take 163 ms longer there than modeled; the likely cause is store traffic, which the model omits.

**E1 in the simulator: does knowing when a session returns help?** Resume TTFT gain over LRU + the CPU tier, in the realistic regime (the traces' think times, the CPU tier at 4×, 20 seeds): the mean of the per-seed gains, and in parentheses, where it differs by more than 2 points, the gain of the seeds' mean latencies:

| Agent sessions (KV pressure) | `lru-cpu` resume p50 / p90 | Oracle prefetch p50 / p90 | Clairvoyant reference p50 / p90 |
|---|---|---|---|
| 8 (1.6) | 86 / 267 ms | +3.4% / +28.0% | +1.4% / +15.5% |
| 12 (2.4) | 103 / 299 ms | +3.3% / +28.8% | +0.5% / +8.0% |
| 16 (3.2) | 139 / 895 ms | +8.2% (+10%) / +13.2% (−3%) | +1.7% / +9.4% |
| 24 (4.4, outside the calibration) | 208 / 1,645 ms | −2.4% (−24%) / −16.9% (−24%) | +2.3% / +10.9% |

Even perfect knowledge buys under 9% of the median resume, which is already a GPU hit or a short restore. At the tail it buys 28% at low pressure (about 80–90 ms of the p90), more than the clairvoyant reference, because a prefetch hides a restore inside the pause; session history, the knowledge a deployment has, buys nothing. Read at p50, this is G-sim's negative branch ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)): the CPU-tier policy plugin stays off. The oracle's rows were corrected on 2026-10-24: its simulated prefetch had fired late for the turns AIPerf's idle cap sends early. The full regime map, with TraceLab's pauses, the history-level arms and the 2× and 8× tiers, is [below](#e1-parking-in-the-simulator). So is the placement of an L4, an A100 and an H100 on the same map, from costs modeled on their published peaks and the kernel path vLLM takes on each ([Other hardware, modeled](../crates/flatshare-sim/README.md#other-hardware-modeled)); none of its numbers is measured, and one card has since been: [the A100 run](#the-a100-run-2026-10-26).

## E1: parking, in the simulator

`flatshare-sim` ([its README](../crates/flatshare-sim/README.md) has the model, its calibration and its validation)
replays the 32× corpus through a model of vLLM 0.32's two-tier KV cache and step loop, with each parking arm in place of
the engine's own LRU and CPU offload ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking) has the design).

### The regime map

`flatshare-sim regime DATA/weka-x32`, at its default seeds 11, 12 and 13 (`make reproduce-e1` writes this as `regime.md`): resume TTFT gain over `lru-cpu` (1 − arm / `lru-cpu`) at p50 / p90, the mean over the seeds, with the CPU tier at 4× the pool. ρ is 0.25–0.27 in every cell. The Markdown has the 2× and 8× tiers too, and the CSV the other arms, avoidable recompute and PCIe bytes.

| Pauses | Sessions | Pressure | π | `oracle-prefetch` | `history-prefetch` | `tokencake` | `ttl-30` | `clairvoyant` |
|---|---|---|---|---|---|---|---|---|
| trace | 5 | 1.3 | 27.0 | +2 / +13% | −1 / +0% | −5 / −9% | −1 / +2% | +1 / +7% |
| trace | 8 | 1.8 | 29.6 | +3 / +40% | +1 / +11% | −6 / −8% | +1 / +4% | +1 / +27% |
| trace | 12 | 2.6 | 21.4 | +5 / +32% | −1 / −6% | −10 / −21% | +0 / −2% | +0 / +7% |
| trace | 16 | 3.6 | 23.4 | +15 / +10% | −11 / −27% | −27 / −22% | +2 / −5% | +2 / +8% |
| trace | 24 | 4.7 | 23.9 | −25 / −14% | −31 / −22% | −60 / −23% | +5 / −3% | +3 / +8% |
| trace | 32 | 6.2 | 23.5 | −48 / −24% | −40 / −17% | −37 / −11% | −7 / +1% | +3 / +2% |
| trace | 40 | 7.5 | 22.9 | −55 / −27% | −46 / −29% | −47 / −22% | +4 / +6% | +1 / +1% |
| tracelab | 5 | 1.2 | 3.2 | −0 / +1% | −0 / −2% | −5 / −7% | −1 / +1% | +0 / +1% |
| tracelab | 8 | 1.8 | 2.9 | +1 / +6% | −2 / −3% | −6 / −14% | +0 / +3% | +1 / +4% |
| tracelab | 12 | 2.4 | 2.7 | +2 / +7% | −3 / −9% | −13 / −18% | +0 / +0% | +0 / +7% |
| tracelab | 16 | 3.4 | 2.6 | −29 / −51% | −39 / −42% | −30 / −41% | +7 / −2% | +4 / +2% |
| tracelab | 24 | 4.7 | 2.9 | −81 / −25% | −80 / −27% | −86 / −27% | +13 / +1% | +2 / +0% |
| tracelab | 32 | 6.1 | 3.3 | −30 / −21% | −30 / −26% | −32 / −27% | −0 / −0% | +1 / +1% |
| tracelab | 40 | 7.3 | 2.8 | −11 / −4% | −20 / −3% | −14 / −4% | +0 / +0% | +1 / +1% |

With trace pauses up to 24 sessions, `mori` loses 1–15% at p50 and 10–24% at p90 against `lru-cpu` (ranking by noisy predicted returns evicts sessions that come back), and `infercept` 18–36% at p50 (its swaps turn GPU hits into restores, and the pool has no use for the memory they free).

**The realistic regime.** The frozen box with the traces' think times and the CPU tier at 4× (π about 23), over 20 seeds (11–30; `realistic.md`): gain over `lru-cpu`, the mean of the per-seed gains, their median (in parentheses) and range [in brackets], and the oracle's gain on the seeds' mean latencies:

| Sessions | Pressure | π | `lru-cpu` resume p50 / p90 | `oracle-prefetch` p50 | p90 | `oracle-prefetch` on mean latencies, p50 / p90 | `clairvoyant` p50 / p90 | Best history arm, p90 |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.6 | 28.6 | 86 / 267 ms | +3.4% (+3.2) [−3, +11] | +28.0% (+28.0) [+12, +48] | +3.5 / +29.3% | +1.4 / +15.5% | `history-prefetch` +3.6% |
| 12 | 2.4 | 20.6 | 103 / 299 ms | +3.3% (+3.0) [−1, +9] | +28.8% (+29.4) [+16, +40] | +3.4 / +29.6% | +0.5 / +8.0% | `history-prefetch` −9.5% |
| 16 | 3.2 | 21.3 | 139 / 895 ms | +8.2% (+6.5) [−0, +25] | +13.2% (+21.9) [−43, +50] | +10.4 / −2.9% | +1.7 / +9.4% | `infercept` −9.8% |
| 24 | 4.4 | 23.3 | 208 / 1,645 ms | −2.4% (+5.9) [−103, +20] | −16.9% (−15.2) [−61, +21] | −24.4 / −23.7% | +2.3 / +10.9% | `infercept` −7.0% |

**G-sim.** The gate is read at p50. There the oracle level closes under 9% of resume TTFT across the realistic regime by the mean of the per-seed gains, and 10% at W0's 16 sessions by the gain of the seeds' mean latencies: at the gate, not clearly under it. Below, gains are means of the per-seed gains, with the gain of the mean latencies in parentheses where the two differ by more than two points.
- **p50.** +3% at 8 and 12 sessions, +8% at 16 (+10%: 139 to 125 ms) and −2% at 24 (−24%). The median resumed turn is a GPU hit or a short restore, and a restore turned into a hit saves only its transfer and the two steps it waits after it.
- **p90.** +28% at 8 and 12 sessions (pressure 1.6–2.4: 78 and 88 ms of `lru-cpu`'s 267 and 299 ms) and +13% at W0's 16 sessions (−3%), whose seed range is wider than the effect (−43 to +50%). At 24 sessions (pressure 4.4) it is −17% (−24%): its loads queue with the demand restores on the one load stream and stall the step loop as they do.
- **The reference.** The clairvoyant gap itself is 0.5–2.3% at p50 and 8–16% at p90. The oracle passes it at 8 and 12 sessions at both percentiles: a prefetch hides a restore inside the pause, which no eviction order can.
- **The history level**, the information a deployment has without tools that report when they will return, closes nothing: its best arm gains 3.6% at p90 at 8 sessions and loses 7–10% from 12 sessions on, its prefetches mistimed by heavy-tailed pauses.
- **TraceLab's tool pauses** (π about 3, a pause about three restores long): the oracle gains at most 7% (p90, 12 sessions) and loses from 16 sessions on.
- **Caveat.** The model under-predicts restores at 24 sessions (above), and a dearer restore at high pressure would raise both what prefetching saves and what its loads cost.

### Miss-ratio curves

`flatshare-sim mrc` computes curves for the GPU tier alone on the reference arm's admission sequence: LRU at 41 capacities from 1/16 to 64 times the pool in one pass of Mattson stack distances, and MIN at chosen capacities. Each request looks up its blocks, then its blocks become the most recent in the order vLLM frees them. Hits are token weighted, per block and per prefix (a block counts only when its whole prefix hits). MIN's per-block hit ratio is a bound: no policy hits more blocks of this sequence at that capacity. Its prefix hit ratio is a reference. `--markdown` writes the table below: the per-block miss ratios, at `--min-at`'s multiples of the pool, of the replay and of the same order at the original token counts; `make reproduce-e1` writes it with the idle service.

Three sequences: the 32× replay; the same admission order with every prompt cut at its original token counts (64-token hash blocks, outputs ×32, capacities ×32); and the replay run again at the original token counts. Miss ratios on the W0 configuration, with the idle service:

| Pool multiple | LRU, 32× | LRU, original tokens, same order | MIN, 32× | MIN, original tokens, same order |
|---|---|---|---|---|
| 0.25 | 0.718 | 0.718 | 0.503 | 0.503 |
| 0.5 | 0.541 | 0.540 | 0.341 | 0.340 |
| 1 | 0.356 | 0.355 | 0.229 | 0.228 |
| 2 | 0.217 | 0.216 | 0.108 | 0.107 |
| 4 | 0.058 | 0.057 | 0.041 | 0.039 |
| 8 | 0.041 | 0.039 | 0.041 | 0.039 |

- **Scaling keeps the reuse structure.** On the same order the curves agree within 0.2 points at every capacity: cutting blocks 32 times coarser loses almost no sharing.
- **Scaling changes the timing.** Replayed at original token counts with the box's E0 fit, a cold prefill of a 100k-token prompt is priced at 88 s, and the 10-minute run completes 10 requests. The 32× replay's pacing, and with it its KV pressure and π, is a property of the scaling. So E1 states its results in the regime coordinates (KV pressure, π, ρ), and other hardware is placed on that map from cost fits built from published specifications, labeled as modeled ([below](../crates/flatshare-sim/README.md#other-hardware-modeled)).
- **Prefixes stay whole.** LRU's per-block and per-prefix ratios are equal: tail-first eviction never leaves a hole in a cached prefix. MIN's are equal to three decimals.
- **Headroom.** At the pool's size MIN misses 22.9% of prompt tokens where LRU misses 35.6%. MIN is down to the cold misses (4.1%) at four times the pool, LRU at eight.

## The A100 run (2026-10-26)

An A100 SXM4 80 GB on Google Colab (12 vCPUs of a 2.20 GHz Cascade Lake Xeon, whose PCIe 3.0 root ports set the link to
host memory), the release's kit ([HARDWARE.md](HARDWARE.md#the-a100-run)): E2's core like for like, the box's model at
its revision, its 5 GiB pool, 512/256 and 20 GiB CPU tier, 24 sessions of the long traces and two callers, W5's arm
files and samples of the agents; only the card, its precision path (Marlin's W8A16, no FP8 units) and the host change.
Records: [`benchmarks/results/2026-10-26-a100`](../benchmarks/results/2026-10-26-a100/README.md); the analysis against
the forecast written before the run is its [`a100.md`](../benchmarks/results/2026-10-26-a100/a100.md).

**E2, by round.** The V2V p95 (ms) and the agents' tokens a second over the calls' first 240 s, each round on W5's
sample of the agents for its number; voice alone sets the SLO at 1,419 + 250 = 1,669 ms. No turn of the 750 failed.

| Agents' sample (seed) | Voice alone | Frozen, the box's 512/256 | The PRs' values | The PRs' tokens/s, of frozen's |
|---|---:|---:|---:|---:|
| 11 | 1,445 | 4,163 / 10,749 | 1,663 / 10,125 | 94% |
| 12 | 1,414 | 4,783 / 9,820 | 1,658 / 10,851 | 111% |
| 13 | 1,419 | 4,986 / 8,398 | 1,611 / 8,025 | 96% |
| 14 | 1,391 | 5,169 / 8,633 | 1,634 / 8,651 | 100% |
| 15 | 1,424 | 5,710 / 8,312 | 1,609 / 7,922 | 95% |
| Median | 1,419 | 4,986 / 8,633 | 1,634 / 8,651 | 96% |

- **Every round under the SLO with the PRs' values**, by 7 to 60 ms; frozen 2.5 to 4.0 s over. The agents' whole-call
  tokens/s with the PRs' values were 95–103% of frozen's round by round, and their TTFT p50 moved by −1.2 to +1.8 s
  (frozen's 5.8–18.9 s across the rounds, p99 22–38 s).
- **Less often short of KV than modeled, as the box is.** Two calls found KV short (the measured running share) 4–9%
  of the time beside frozen and 2–7% beside the PRs' values, against the forecast's 10% for this card [modeled]. On
  the same samples RS measured 7–9% and 5–7% on the box, against the simulator's 17%
  ([RS](#the-boxs-running-share-rs)). Frozen preempted 29–75 agent requests a point, the PRs' values 31–78.

**At vLLM's default 2,048/0**, the card's best stock cap in the probes, on rounds 1 and 2's samples: tuned vLLM, then
the PRs' values at that cap (arm 8), the V2V p95 (ms) / the agents' tokens a second over the calls' first 240 s, with
the LLM's first token at p95 from the voice worker beside the 512/256 points':

| Agents' sample (seed) | Tuned vLLM, 2,048/0 | The PRs' values, 2,048/0 | Of tuned's tokens/s | LLM first token p95: 2,048/0 PRs, 512/256 PRs |
|---|---:|---:|---:|---:|
| 11 | 4,036 / 15,082 | 1,810 / 15,124 | 100% | 412 ms, 183 ms |
| 12 | 3,289 / 13,990 | 1,843 / 14,145 | 101% | 414 ms, 216 ms |

No turn of the 200 failed. Tuned vLLM made 1.40 and 1.42 times the agents' tokens of the 512/256 points on the same
samples, its TTFT p50 1.0 and 2.0 s against frozen's 6.5 and 5.8 s; the PRs' values at 2,048/0 kept them all, 140 and
173 ms over the SLO, the rest of their tail the LLM's first token behind the larger steps in flight.

**A', 32 sessions at the box's pool** (round 1, seed 11; one point a configuration):

| Configuration | V2V p95 | Agents' tokens/s, first 240 s | Agents' TTFT p50 / p99 | Preemptions | Running share |
|---|---:|---:|---:|---:|---:|
| Frozen, the box's 512/256 | 5,222 ms | 8,649 | 24.5 / 55.8 s | 57 | 6.5% |
| The PRs' values | **1,617 ms** | 8,487 (98%) | 25.9 / 52.3 s | 65 | 6.5% |

No turn of the 100 failed. With 24 sessions on the same sample the agents made 10,749 and 10,125 tokens a second; the
forecast placed 32 sessions at a running share of 18% (fitted: 15%) [modeled].

**The probes** (blocks 1 to 3; each run's numbers in [`a100.md`](../benchmarks/results/2026-10-26-a100/a100.md)): the KV
case's wait on stock vLLM 1,721–1,750 ms at 512/256 and 1,620–1,645 ms at vLLM's default 2,048/0, and the first token
with the PRs' values 150–176 ms; the prefill case's token gap, 1st to 16th, 32 ms on stock vLLM and 7 ms with the PRs'
values; the calls' first-token p95 beside 12 agents 3,102–3,126 ms at 512/256, 2,518–2,535 ms at 2,048/0, 3,446–3,687 ms
at 1,536/768 and 167–179 ms with the PRs' values, over 57–75 turns a run, none failed. With FlashInfer's attention in
place of FlashAttention 2 (the backend arm, after the core), the KV case's wait on stock vLLM at 512/256 was
1,580–1,607 ms, 8% shorter at the median.

**E0** (block 1, 3 repetitions, p50 in ms):

| Prompt tokens | Recomputed | GPU hit | Restored from host RAM |
|---|---:|---:|---:|
| 1,024 | 64 | 18 | 32 |
| 2,048 | 128 | 21 | 49 |
| 4,096 | 269 | 25 | 81 |
| 8,192 | 602 | 32 | 139 |
| 16,384 | **1,471** | **46** | **268** |
| 24,576 | 2,607 | 61 | 379 |
| 31,744 | 3,825 | 75 | 483 |

Fitted: a GPU hit 17 ms + 1.80 µs/token, a CPU restore 2.8 ms + 12.9 µs/token over it (11.5 GB/s of KV), a recompute 4.2
ms + 56.5 µs/token + 2.01 ns/token². Restoring beats recomputing from 1,024 tokens.

**The step model** (the calibration point: AIPerf's replay of the whole corpus at 16 sessions, 709 requests, fitted with
`crates/flatshare-sim/scripts/fit_step_cost.py` and E0): a step's fixed cost 4.5 ms, 41.36 µs a token, 4.07 ns a token
pair, 171.9 ns a KV token read; the forecast's 3.88 ms, 41.33 µs, 2.59 ns and 94.3 ns. It reproduces the point's 18,380
decode steps to −3.0% at the median (p10/p90 −8.6/+8.0%).

**The scheduler at a batch of 256** (`batch.sh`: `vllm bench serve` keeping 26 requests at priority 0 and 230 at
priority 2 in flight, 48 prompt and 64 output tokens each, so that all 256 hold their KV at once; one run an engine):

| Engine | Output tokens/s | Priority 0: TTFT p50 / p99 | Priority 0: TPOT p50 | Priority 2: TTFT p50 / p99 | Priority 2: TPOT p50 | Running p50 / max |
|---|---:|---:|---:|---:|---:|---:|
| vLLM 0.32.0 (frozen) | 7,524 | 166 / 574 ms | 28.6 ms | 341 / 1,086 ms | 29.1 ms | 201 / 254 |
| make-room, its switches off | 7,661 | 133 / 510 ms | 30.9 ms | 214 / 1,212 ms | 30.8 ms | 215 / 256 |
| make-room, the PRs' values | 8,672 | 90 / 177 ms | 6.9 ms | 283 / 7,971 ms | 28.6 ms | 154 / 230 |

- **The code path costs nothing at the batch.** With its switches off make-room schedules as vLLM does, and it made
  1.018 of vLLM's output tokens a second, with more requests running at once (215 at the median against 201), which is
  also why each one's TPOT is longer.
- **The PRs' values turn the priorities' latencies around.** The priority 0 requests' first tokens come 3.2 times as
  soon at p99 and their tokens 4.1 times as fast, since #63351 keeps priority 2 prefill out of their first 48 tokens'
  steps; closed loop at a concurrency of 26, the priority 0 client then completes its requests at 2.8 times the rate,
  and the engine makes 1.153 of vLLM's tokens a second. The priority 2 requests pay at their tail: TTFT p99 7.3 times
  frozen's, its median lower, as a steady stream of priority 0 first tokens holds their prefill back.

**The host** (`bench_host.py`): Python's fixed workload 227 ms at the median (the box's 381); pinned host memory to the
card 10.08 GB/s at the CPU tier's 32 KiB pages, 11.09 at 64 KiB, 12.28 at 2,304 KiB (a 16-token block of all 36 layers)
and 12.32 in one copy (the box's 5.44, 6.03, 6.74, 6.76), and back to the host 10.61, 11.62, 12.80 and 12.86.

## Failure drills

The deployed box on the configuration to deploy (make-room with the PRs' switches and a 48-token window), on 2026-10-25, from `ops/drills` ([RUNBOOK.md](../ops/RUNBOOK.md#drills) runs them): the final build (v0.3-295 on both hosts), and W8's, before W6's hardening, beside it. Each drill opens two realtime streams through the gateway's voice-plane socket and four background ones through its TCP port, each decoding up to 2,000 tokens, then breaks something once every stream has its first byte; from the event on it probes both planes every second, each probe on a new connection, until the streams have ended and a request succeeds. The records, with W6's and an intermediate build's, are in [`benchmarks/results/2026-10-25-drills`](../benchmarks/results/2026-10-25-drills/README.md).

| Drill | What the six streams saw | What the new requests got | W8, before W6 |
|---|---|---|---|
| The engine killed (SIGKILL to every process of `flatshare-engine`, mid-decode) | Each got vLLM's error event, `upstream_unavailable`, 8–11 ms after the kill | 210 of 212 probes failed at once (2 answered 502, then 503 `upstream_not_ready` without reaching for the engine), until one succeeded 106.2 s after the kill | Every body cut 54–57 ms after the kill, with no error event; a request succeeded 112.3 s after it |
| The gateway reloaded (`systemctl --user reload flatshare-gateway`, a deploy's) | The realtime streams finished whole, 20.9 s after; the background ones ended 5.0 s after, with the `draining` error event | None of 42 probes failed or waited: the slowest was answered in 0.13 s | — (a deploy restarted the gateway) |
| The gateway restarted (`systemctl --user restart flatshare-gateway`), 30 s after the reload | As the reload's: the realtime streams whole, 20.7 s after; the background ones `draining`, 5.0 s after | None of 42 failed; those sent during the drain waited in the socket's backlog and were answered once the new gateway started, 20.7 s after the restart (the slowest in 19.8 s) | Every stream finished, 22.4–23.2 s after, while the gateway refused new connections; the probe ran after the drain |
| The gateway killed (SIGKILL; systemd starts it again a second later, its socket unit holding the listeners) | All broke 11–14 ms after | None of 4 failed; they waited for the gateway's start, the slowest 1.3 s | — |

- **Where an engine's recovery goes.** The unit's last process exited 7.6 s after the kill (the GPU's teardown), systemd started the unit 5.1 s after that (`RestartSec`, 15 s in W8), and its warm start listened 92.1 s later. Until its `/health` answers, the gateway answers 503 with `Retry-After: 1` and does not reach for the engine.
- **No KV or host memory outlives a killed engine.** The engine ran six requests before the kill and none after its restart, and `/dev/shm` held the CPU tier's 20.0 GiB before and after (an unlinked 21.5 GB file there, made again by each start); each kill, W8's and the final build's, left one 32-byte semaphore.
- **A deploy without a gap.** Until W6 a deploy restarted the gateway, and new requests failed for as long as its longest stream took to finish, up to 30 s. Since W6 a deploy reloads it: the gateway hands its listening sockets to a successor that serves at once, and drains ([PRODUCTION.md](PRODUCTION.md#how-it-fails)). A restart drains too since the unit's `ExecStop` waits for the drain itself (v0.3-191's restart after a reload cut its streams: systemd 249 does not wait for a main process it did not start), and the requests sent meanwhile, queued in the socket's backlog, are answered once the new gateway serves: on v0.3-274, 5 of the restart's 42 probes got 502, where cyan's tunnel let the new gateway's burst of upstream connections pass a master's sessions and the node's sshd dropped them; the tunnel now waits for a master with room.
- **The CPU tier overfilled (W8).** `flatshare-bench drill cpu-tier` stored 43.5 GB, 2.0 times the tier's 20 GiB, in 18 requests of 16,384 tokens (3.30 s each at the median); the engine stayed healthy, the latest 8,192-token prompt restored from the tier in 294 ms against 1,112 ms recomputed, and the first had been evicted (recomputed, no tier hit).

Each server's start, from its journal (`ops/drills/cold-start.sh`):

| Start | Unit start to ready | Where the time goes |
|---|---:|---|
| The engine, warm caches | 88.8 s | The API server up at 16.5 s, the engine process at 38.4 s, the weights loaded at 54.1 s, the KV cache at 55.4 s, the CUDA graphs captured at 84.1 s, listening at 87.9 s; torch.compile from its cache in 0.3 s |
| The engine, empty caches, as on a new host | 531.8 s | The weights at 86.1 s; torch.compile 84 s (Dynamo 23 s, Inductor 43 s) and the profiling run 17 s, so the KV cache at 189.3 s; 337 s more to the CUDA graphs (526.6 s), of which the capture is 13 s and the rest kernel warmup and JIT builds, FlashInfer's sampling module among them |
| The box, the final build (`systemctl --user restart flatshare.target`) | 102.7 s | The units stopped within 8.1 s, then the three started at once: the TTS ready 22.9 s after its start, the STT 64.0 s, the engine 94.6 s (on v0.3-191 the engine waited for the STT to be ready, and the box took 161.4 s) |

The engine's warm start is most of the recovery from an engine failure; on a new host, with empty caches, the first start takes six times as long.

**Two GPU nodes** (2026-10-28, [the record](../benchmarks/results/2026-10-28-node-failover/README.md)): the serving arrangement, both calls on Phastform and Gray a warm spare (`calls = "fill"`), with `ops/drills/node-failover.sh` placing two calls as the voice worker does and running a 32-token turn of each every 3 s beside two agents' streams, then breaking Phastform's engine after five whole turns of each call:

| Drill | Turns before / after the event, whole | Calls moved to Gray | Turns failed after the first move | Phastform passed over / unhealthy / available again | Gray covered |
|---|---|---:|---:|---|---:|
| SIGKILL to the engine; systemd starts it again | 10 of 10 / 148 of 148 | +0.7 s | 0 | +0.10 s / +2.1 s / +101.0 s | 238 s |
| The engine stopped to the drill's end | 10 of 10 / 152 of 152 | +1.9 s | 0 | +1.3 s / +3.3 s / down to the end | 244 s |

- **The calls moved within a turn,** and none of the 300 turns after the events failed; no turn was in flight at either event, so none was cut. Gray then covered to the drills' end, as the moved calls stayed on it, and the agents' requests that began meanwhile waited out their class's 120 s queue timeout and got 503 `queue_timeout`. A first run found the drill's own client sending a wrong model id, every request 404 before the event as after it; the runs above are the first after its fix.

## Load generation

- AIPerf 0.13.0 `agentic_replay` over the SemiAnalysis Claude Code traces: 99.4% of turns arrive within 50 ms of their recorded delay.
- Block scaling makes all 98,827 requests fit:
  - 32× fits a 32k context, with main-agent context p50 7.9k and p90 20.3k tokens;
  - 64× fits 16k.
- Load generators run on cyan, on pinned cores; another host considered for them had a p99 timing noise of 200–350 ms, which ruled it out.
