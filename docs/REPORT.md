# Making room in the engine: a realtime voice tier beside coding agents on one GPU

*Technical report v1. Leaseholder, StevenWang-CY/leaseholder. Every number comes from the committed records
([BENCHMARKS.md](BENCHMARKS.md) lists every run); E2's are W5's, measured on 2026-10-24 and 25.*

## Abstract

A self-hosted voice agent leaves its GPU idle between turns, and background LLM work such as coding agents could use
those gaps. On vLLM today the voice tier pays for it. Beside 24 coding-agent sessions of long contexts on one RTX 5060
Ti, the voice-to-voice p95 of two calls grows from 1.78 s alone to 10.5 s on vLLM tuned for priority. Two scheduler
mechanisms cause the loss: a waiting voice turn cannot take KV cache from running background requests, and once it runs,
background prefill shares every step of its reply. We change vLLM's priority scheduling in two rules. A waiting request
preempts lower-priority running work when it lacks KV, and the victims' KV is already written through to the CPU tier.
Lower-priority prefill sits out the steps of the request's first tokens. Proposed upstream as two stacked pull requests
and run on vLLM 0.32 as a scheduler built from them, the change brings the p95 to 2.24 s, 0.20 s over an SLO of voice
alone plus 250 ms, while the agents keep 97% of their throughput. Each rule is necessary: the KV preemption alone leaves
5.1 s, and the prefill rule's window must cover the reply's first sentence (3.1 s with 16 tokens). A router that aborts
background requests and resubmits them reaches 2.52 s, at the agents' expense; a gateway that holds back background work
by the conversation's state leaves the tail where tuned vLLM does. A control plane makes room only by aborting work, and
only the engine takes KV from a request already decoding. For an A100 40 GB the simulator calibrated on this box
forecasts the KV case's first token at 1.6–1.7 s on vLLM 0.32 and 134–158 ms with the KV rule, written before the run
that will check it.

## 1. Introduction

Voice agents are judged by how soon they answer, so a self-hosted cascade (speech to text, an LLM, text to speech) is
provisioned for latency, and the GPU it runs on idles most of the time: the LLM is busy only while a reply decodes.
Background work fills such gaps elsewhere in serving, and coding agents are the background work of the moment. They send
long prompts that reuse long prefixes, tolerate a minute to the first token, and run for hours.

The two do not share a GPU well on today's engines. We measure a voice agent and 24 Claude Code sessions on one RTX 5060
Ti with vLLM 0.32, tuned as an operator would tune it: priority scheduling with the voice tier at the highest priority,
a 512-token step cap that bounds how long a decode waits behind a prefill chunk, and a 20 GiB CPU KV tier written
through as tokens are computed. The voice tail grows 5.9-fold. The loss is not in speech recognition or synthesis, which
barely move, but in the LLM's first sentence: 3.88 s of a 5.30 s mean turn, against 0.30 s alone. Every configuration an
operator can deploy today misses the SLO: vLLM's pending request-count preemption, SGLang's priority preemption with its
host tier, a static KV reservation and two routers by seconds (W4), and a router that aborts background work before its
first byte and resubmits it by 0.49 s, with the agents' TTFT p99 at 193 s against 106 s.

Two mechanisms in the scheduler explain the loss, and both are independent of the card. First, priority scheduling in
vLLM preempts among running requests, never for a waiting one. When agents' running requests hold the KV pool, a voice
turn waits for one of them to finish: 4.9–5.0 s for a 750-token turn behind two 18k-token agent requests. Second, once
the turn runs, every step of its reply also carries agents' prefill chunks, so its tokens arrive 60–65 ms apart instead
of 16–20. A control plane in front of the engine can do little about either. It can delay new background requests and
abort admitted ones, but it cannot take KV from a request that is decoding, and it cannot reach into a step.

We change the scheduler instead. The first rule lets the highest-priority waiting request preempt lower-priority running
requests when it lacks KV, through vLLM's own preemption path and in vLLM's own victim order, at most a few a step. The
victims' KV is already in the CPU tier, which a resume reloads from where the tier still holds it. The second rule holds
lower-priority prefill out of the steps of the request's first tokens, a budget cap counted in output tokens, while
decodes continue. Neither needs a model of the hardware: both count blocks, tokens and steps.

This report makes four contributions:
- **It measures the failure end to end.** Voice-to-voice latency is read from waveforms at the caller, beside a replay
  of real coding-agent traces, against every deployable alternative, in rotated rounds over five samples of the agents'
  traces (§4, §5.1).
- **It changes vLLM's priority scheduling** for a full KV cache and for prefill interference, in two rules proposed
  upstream as two stacked pull requests and built into vLLM 0.32's scheduler without modifying the installed vLLM (§3).
- **It shows each rule necessary and prices it.** The KV preemption alone halves the tail; the prefill rule closes most
  of the rest once its window covers the reply's first sentence; the agents keep 97% of their throughput and pay through
  the prefill rule's hold, not the preemptions (§5.2-5.4).
- **It bounds what a control plane can do.** Headroom, a prefill throttle and degraded replies timed by the conversation
  leave the tail where tuned vLLM does when agents' KV fills the pool, and aborting admitted work comes within 0.49 s of
  the SLO only by discarding the agents' prefill (§5.5).

## 2. Why the voice turn waits

The voice tier's request is small: a system prompt, a few turns of dialog and a reply of a few dozen tokens, about 750
tokens of KV. Agents' requests are large: in the traces we replay, a median request of 13.9k tokens in the
longest-context sessions, and a 16k-token resume costs 3.26 s to recompute on this card against 509 ms to restore from
the CPU tier and 97 ms from the GPU pool ([E0](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0)). The KV pool
beside the audio models holds 36,400 tokens: two or three agent contexts.

**A waiting request cannot take KV.** vLLM's scheduler serves running requests first, then admits waiting ones while the
step's token budget and KV last. Under priority scheduling it preempts the lowest-priority running request when a
running request cannot grow, which protects decodes from each other, but the waiting loop stops at its first failed
allocation: nothing preempts on a waiting request's behalf (vLLM issue #40004). A probe that fills the pool with two
18k-token priority-2 requests and then sends a 750-token priority-0 request measures the gap directly: the voice-sized
request's first token comes after 4.9–5.0 s on vLLM 0.32, when one agent request is preempted for the other's growth and
its KV frees, and after 219–290 ms with the first rule ([the KV case](BENCHMARKS.md#the-kv-case-make-room)).

**Prefill shares the reply's steps.** Chunked prefill keeps steps short, but it spreads an agent's prefill over many
steps, and each of them also carries the voice reply's decode. With three agent prefills of 10,000 tokens running, the
voice reply's tokens come 60–65 ms apart with the first rule alone, and its 16th token after 1,211–1,528 ms; with
lower-priority prefill held out of its first steps, 16–20 ms apart and its 16th token after 536–615 ms. A step cap
bounds a step's tokens, not its time. The voice agent hands text to speech its first complete sentence, and this agent's
replies run 23 tokens at the median, so the interference lands exactly where the caller waits.

**What the caller hears.** In the end-to-end measurement the two compound. On tuned vLLM the turn's mean is 5.30 s: 0.82
s of endpointing, 3.88 s from the end of the turn to the LLM's first sentence, 0.25 s to text to speech's first audio
and 0.34 s of transport and playout. With voice alone the same stages take 0.76, 0.30, 0.21 and 0.35 s.

## 3. The engine change

The change adds two rules to vLLM's priority scheduler; our variant keeps a third. Each is a switch, off by default, and
each counts blocks, tokens and steps rather than time.

**KV make-room ([#63350](https://github.com/vllm-project/vllm/pull/63350)).** At the start of a step, the scheduler
finds the highest-priority waiting request that the waiting loop could schedule in this step: not one loading its KV
from another tier, compiling a grammar, paused for streaming input or awaiting output from before a preemption, nor one
whose LoRA adapter cannot join the running ones. If that request lacks KV for its whole sequence, net of its
prefix-cache hit, the scheduler preempts running requests of a strictly lower priority, before the running loop and
through vLLM's own preemption path, in vLLM's own order: the lowest priority first, then the latest arrival. It preempts
as few as make the request fit, at most N a step, and none if N would not suffice. Three details make the count right. A
block is freed only when every request that holds it is a victim, since prefix caching shares blocks. The request's own
prefix-hit blocks never count as freed. And the running decodes keep growing until the victims' frees land, which under
asynchronous scheduling, with frees deferred by a step, means two steps of growth. The scheduler makes no room again for
a request the waiting loop has not tried since, and a victim waits until prefills already in flight have the blocks they
still need, so preemptions cannot cascade. With vLLM's CPU offload connector, a victim's pending stores are flushed and
its KV stays in the CPU tier: it resumes by a restore.

**The budget cap ([#63351](https://github.com/vllm-project/vllm/pull/63351)).** Lower-priority prefill sits out the
steps of a high-priority request's first tokens: from the step in which the highest-priority waiting request fits, while
it prefills, until it has its first T output tokens, counting those still in flight under asynchronous scheduling.
Running lower-priority prefill chunks are held, waiting lower-priority requests are not admitted, and a held request
that already holds its KV is skipped so that the requests behind it are still considered. Decodes are never cut, and
nothing is held while every running slot is taken, so the cap cannot stall the engine. A voice turn's prefix is usually
evicted from the GPU between turns under agent load and loads back from the CPU tier; the load holds lower-priority
prefill for its first T steps, and again from the step in which it completes, so a load that never completes holds
nothing after its first T steps.

**The reserve.** Our variant also keeps a reserve: requests of a lower priority are admitted only while their whole
sequences leave R tokens free. It is a class-aware watermark. With full replies it does the KV preemption's job by other
means (2,205 ms against the KV preemption's 2,193 ms in W4), so the pull requests leave it out and the configuration to
deploy sets it to 0.

**How it is built.** The make-room variant applies the PRs' scheduler diff, backported to vLLM 0.32.0, and a patch of
its own (its switches, the reserve and its counters) to a copy of the installed vLLM's scheduler at each engine start,
and the engine loads it with `--scheduler-cls`; it refuses any vLLM other than 0.32.0. Its tests run vLLM's real
scheduler on CPU torch, and a trace test runs the variant and the PRs' own patches on the same request traces and
requires every step to be equal. The PRs carry vLLM's own tests of each rule.

**What it costs per step.** The rules run in the scheduler, on the host. On vLLM 0.32.0's real scheduler on a CPU (a
Xeon Silver 4110), `Scheduler.schedule()` with the switches off takes what stock vLLM's does (within 1.1% at every state
measured); at the PRs' values it takes 58–149 µs longer a step while only lower-priority work runs (5–18% more, at 16 to
256 running requests), 0.54–0.63 ms longer when a high-priority request waits that two victims cannot make room for, 1.7
ms longer when that request's prefix hit is 999 blocks long, and 160–247 µs when every running slot is taken; a decode
step on the box takes 16–20 ms. `make perf-make-room` measures it again against that baseline, and fails a change that
costs more beyond the measurement's noise.

## 4. How it is measured

**The box.** One RTX 5060 Ti 16 GB on a PCIe 3.0 x8 link runs the whole voice cascade beside the LLM: Whisper
large-v3-turbo on vLLM for speech to text, Kokoro-82M for text to speech, and vLLM 0.32.0 serving Qwen3-4B-Instruct-2507
in FP8 with a 5 GiB KV pool (36,400 tokens), a 20 GiB CPU tier written through, priority scheduling and a 512-token step
cap with a 256-token threshold per request. A second host runs the control plane, the media server and the load
generators. Every configuration we compare runs on this box and differs from it only in engine flags or the scheduler
module.

**The voice tier.** Two scripted callers place calls through LiveKit to a stock LiveKit voice agent, 25 measured turns
each after one unmeasured warm-up turn, with barge-in on a quarter of the turns. Voice-to-voice latency runs from the
end of the caller's speech to the agent's first audio, read from the waveforms at the caller on one clock. Endpointing
is fixed at the agent so that the turn detector's variance does not enter the comparison. A turn not answered within 30
s counts at 30 s.

**The agents.** AIPerf replays the SemiAnalysis Claude Code traces with their think times, sessions and subagents,
block-scaled 32× so that their reuse structure fits the box. The discriminating scenario runs 24 agent sessions of the
48 traces with the longest contexts, a regime where running agents' KV fills the pool; the standard scenario runs 16
agent sessions of the whole corpus, where it does not. The agents reach the engine through the gateway at priority 2,
the voice agent's requests at priority 0. The agents' throughput is the prompt and completion tokens per second of the
requests that complete in the calls' first 240 s.

**The SLO and the repeats.** The SLO is the voice tail alone plus 250 ms: voice alone's V2V p95, 1.78 s, so 2.03 s. Each
configuration runs in five rounds; each round replays its own sample of the agents' traces and runs its configurations
in an order rotated by one from round to round. W5 ran three campaigns of five rounds on the same five samples: the
headline configurations, the PRs' rules one by one, and an extension with the router and the standard scenario; a
fourth, of three rounds, ran 4 and 6 callers. A configuration that more than one campaign ran is pooled over them: 10
runs of tuned vLLM and 15 of the configuration to deploy. We report each configuration's median over its runs and their
range.

## 5. Results

### 5.1 The headline

![Voice-to-voice p95 of each configuration over its runs, against the SLO](figures/e2-voice-p95.svg)

| Two calls beside 24 agent sessions that fill the KV pool | Voice p95, median (range) | Over the SLO | Agents' tokens/s | Agents' TTFT p50 / p99 |
|---|---:|---:|---:|---:|
| Voice alone, the SLO's reference | 1,784 ms (1,764–1,819 ms) | — | — | — |
| vLLM 0.32, tuned: priority scheduling, a 512-token step cap, a CPU KV tier | 10,477 ms (8,686–12,115 ms) | +8,443 ms | 3,305 | 63 s / 106 s |
| Tuned, with a router that aborts agent requests before their first byte and resubmits them | 2,522 ms (2,516–2,868 ms) | +488 ms | 3,332 | 32 s / 193 s |
| + #63350: a waiting request preempts lower-priority work for KV | 5,085 ms (4,693–6,261 ms) | +3,051 ms | 3,414 | 67 s / 112 s |
| + #63351: agent prefill sits out the reply's first 16 tokens | 3,094 ms (2,868–3,192 ms) | +1,060 ms | 3,298 | 71 s / 105 s |
| + #63351 with a 48-token window: **the configuration to deploy** | **2,236 ms** (2,132–2,397 ms) | +202 ms | 3,203 | 72 s / 111 s |

On tuned vLLM the voice p95 is 10.5 s, and its fastest run is still 6.7 s over the SLO. With both rules and a 48-token
window it is 2.24 s over 15 runs, 0.20 s over the SLO; the SLO's 250 ms is a strict budget, and none of the 15 runs met
it (2.13–2.40 s). Of the configurations an operator can deploy today, the router that aborts and resubmits comes
nearest, 0.49 s over. Round by round, on the same sample of the agents, the PRs' configuration was the faster in all
five rounds, by 223–723 ms (a median of 390 ms), and the router's agents waited 193 s at p99 for a first token, against
111 s (§5.4). Five more rounds on the final code, W6's transport and #63351's revision among its changes, replayed W5's
own samples with its arm files: the PRs' configuration gave 2.27 s, round by round −45 to +77 ms from W5's, voice alone
moved by −51 to +55 ms, and no turn failed, so W5's numbers stand for the code to deploy
([FB](../benchmarks/results/2026-10-25-e2-fb)).

The PRs' rows ran as the make-room variant with the PRs' switch values, which schedules as the PRs' diff, ported to vLLM
0.32.0, did before their revision of 2026-10-25. The revision counts a request's first tokens still in flight under
asynchronous scheduling, so the window's hold ends a step sooner; only a reply at least the window long reaches that
step, most of the replies with a 16-token window and one with 48 ([after
W5](BENCHMARKS.md#after-w5-the-first-tokens-in-flight)).

### 5.2 Each rule's share

The PR campaign ran the rules one at a time on one set of five samples. Tuned vLLM's p95 was 10.5 s there; the KV
preemption alone brought it to 5.1 s, the budget cap with a 16-token window to 3.1 s, and with 48 tokens to 2.23 s. The
voice turn's stages show where each rule acts (means over every reply of each configuration's runs, in seconds):

| Configuration | Endpointing | End of turn to first sentence | TTS first audio | Transport and playout | Mean V2V | LLM first token |
|---|---:|---:|---:|---:|---:|---:|
| Voice alone | 0.76 | 0.30 | 0.21 | 0.35 | 1.62 | 80 ms |
| Tuned vLLM | 0.82 | 3.88 | 0.25 | 0.34 | 5.30 | 2,795 ms |
| + #63350 | 0.82 | 1.99 | 0.25 | 0.36 | 3.42 | 771 ms |
| + #63351, 16 tokens | 0.82 | 0.74 | 0.24 | 0.35 | 2.15 | 289 ms |
| + #63351, 48 tokens | 0.82 | 0.55 | 0.24 | 0.35 | 1.97 | 282 ms |
| The router that aborts and resubmits | 0.82 | 0.67 | 0.23 | 0.36 | 2.08 | 275 ms |

Endpointing, text to speech and transport move by at most 69 ms from voice alone's in any configuration; the seconds are
all between the end of the turn and the LLM's first sentence. The KV preemption takes the wait for KV out of the LLM's
first token (2,795 ms to 771 ms at the mean), which the KV probe isolates: 4.9–5.0 s to 219–290 ms. What it leaves is
the reply decoding beside agent prefill, 60–65 ms a token in the prefill probe, which the budget cap takes out (16–20
ms). The KV preemption alone stays 3.05 s over the SLO, and both rules with a 16-token window 1.06 s.

### 5.3 The window and the reply

The budget cap's window is counted in output tokens, and the voice agent does not speak its first token: it hands text
to speech its first complete sentence. Over W5's 5,571 replies, a reply ran 23 tokens at the median and 52 at most. A
16-token window ends before 79% of the replies do, and with it the p95 stays at 3.1 s; a 48-token window outlasts all
but one of them, at 2.23 s on the same samples. The window has to cover the reply's first sentence, what text to speech
needs before it can speak, not the whole reply. A sweep of the window against 40-token replies beside the box's agents
places the knee where the window covers the reply: the 40th token's p95 fell from 3.8 s at 16 tokens to 1.04 s at 48 and
moved no further at 64 (1.02 s), and the agents paid 8% of their prompt throughput for it, nothing more past the reply's
length, since a hold ends with its reply ([the knee](BENCHMARKS.md#the-windows-knee)).

### 5.4 What the agents pay

With the configuration to deploy, the agents completed 3,203 tokens a second in the calls' first 240 s, against 3,305 on
tuned vLLM: 97%. Their first tokens came later: the TTFT p50 rose by 9 s, from 63 to 72 s, and the p99 was 111 s against
106 s. The cost is the budget cap's hold, not the preemptions. Over the calls, the cap held lower-priority prefill in
33% of the engine's steps with a 48-token window (25% with 16), while the KV preemption preempted a median of 31 agent
requests a run, each resumed from the CPU tier where it still held the victim's KV. How much a resume reloaded and how
much it recomputed, no vLLM 0.32 counter shows ([E3](BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)).

The router that aborts and resubmits pays differently. It keeps the agents' throughput (101% of tuned vLLM's), and their
median first token came after 32 s against 63 s, but it discards the prefill of every request it aborts, so a request
aborted more than once waits for its prefill again each time: the agents' TTFT p99 reached 193 s.

### 5.5 What a control plane cannot do

E2's gateway, `lh-gateway`, ran the control-plane mechanisms an operator could add in front of the engine: KV headroom
held by deferring background requests while a call is armed, a throttle on background prefill during the critical window
of a turn, degraded replies when the pool is short, all timed by the voice agent's lifecycle events. On tuned vLLM in
the discriminating scenario this lifecycle policy left the voice p95 at 10,993 ms, no better than tuned vLLM's 10,138 ms
(both W4): deferring new requests cannot make room while running work holds the KV. On top of the engine change, the
same policy, or arm 9 without its headroom, its throttle or its events, landed within the spread of its own runs (W4):
the room comes from the engine. Of the routers, SMG's preemption before the first byte and llm-d's flow control
(emulated) left 11,050 and 10,992 ms (W4); the abort-and-resubmit policy reaches 2.52 s only by discarding admitted
prefill (§5.4). A control plane sees requests, not KV blocks or steps: it can hold back what has not started and abort
what has, and only the engine can take KV from a request that is decoding or keep agent prefill out of a reply's step.

### 5.6 Where it does not help

**A pool with room.** In the standard scenario, the whole corpus at 16 agent sessions, the pool has room and the KV
preemption has little to do (0–7 preemptions a run); the cap holds agent prefill in 3–13% of the steps. The PRs' p95 was
the lower in every one of five rounds, by 109–1,413 ms, but met the SLO in only one: 2,021–2,419 ms against tuned vLLM's
2,147–3,832 ms. What remains is again in the LLM's first sentence: 0.42 s from the end of the turn at the mean, against
0.58 s on tuned vLLM and 0.30 s with voice alone.

**Many calls.** The budget cap holds lower-priority prefill while any call's first tokens are due, so overlapping
replies hold it for longer. With 4 and 6 callers, each against its own SLO (voice alone with as many callers plus 250
ms: 2,044 ms and 2,135 ms), the PRs' p95 was 2,250–2,415 ms and 2,360–2,468 ms, over the SLO in every round by a median
of 0.27 s and 0.26 s, against 0.20 s with two. The hold grew to 53% and 64% of the engine's steps, and the agents'
throughput fell to 3,055 and 2,716 tokens a second, against 3,203 with two calls. A model on vLLM's real scheduler
places the limit near one realtime reply a second in aggregate, where the cap holds the background tier nearly always
([how it fails](PRODUCTION.md#how-it-fails)).

### 5.7 Another GPU

The mechanisms are not specific to the card; what each costs is. Before the run on an A100 40 GB, `lh-sim` forecast the
KV case's first token there at 1.6–1.7 s on vLLM 0.32 and 134–158 ms with #63350, at the box's pool ([the
forecast](../benchmarks/a100/forecast.md)). The run that checks it, a campaign of about 7 GPU-hours whose kit was
rehearsed on this box ([the A100 run](HARDWARE.md#the-a100-run)), is this report's next version.

## 6. Deploying it

The configuration to deploy is vLLM 0.32 with priority scheduling, a 512-token step cap and a 256-token threshold per
request, the CPU KV tier written through, and the two rules: `make_room.max_preemptions_per_step 2`, the budget cap on
with a 48-token window, and no reserve. Voice requests are sent at priority 0 and background work at 2.
[PRODUCTION.md](PRODUCTION.md) has the whole configuration, whether a deployment is in the regime where the change pays
([is yours?](PRODUCTION.md#is-your-deployment-in-this-regime)), what to watch and how it fails.

**What to watch.** The voice tier's V2V p95 per call and its time to first token at p95; the engine's KV usage, running
and waiting requests and preemptions; the agents' completed tokens a second and their TTFT p99, where a preemption
policy moves the cost. The repository ships these as Prometheus rules, the SLO's burn-rate alerts on the realtime first
byte, and a Grafana dashboard.

**How it fails.** A voice turn that N preemptions cannot make room for waits as on stock vLLM; only realtime work leaves
nothing to preempt; many overlapping calls hold the background tier (§5.6). The failure drills on the deployed box ([the
drills](BENCHMARKS.md#failure-drills)): a killed engine breaks every stream it serves at once, and the next request
succeeds when the engine's warm restart completes; a restarted gateway lets its streams finish while it drains; a CPU
tier overfilled to twice its size keeps the engine healthy and restores what it still holds. On the final build each
stream of a killed engine got a typed error event within 11 ms, and a request succeeded again 106 s after the kill, 92 s
of it the engine's warm start; a reload, as a deploy takes a new gateway, failed none of 42 requests, and a restart none
either, holding those sent during its 21 s drain until it ended. The box's three GPU servers start together in 103 s on
warm caches; the engine's first start on empty caches takes 532 s.

## 7. Related work

**Co-serving a latency class with best-effort work** is established. ConServe, HyGen, BROS and Niyama schedule online
requests beside offline ones in one engine, and Valve colocates them in production on 8,054 GPUs with sub-millisecond
compute preemption and dynamic memory reservation. We do not claim the goal. We measure it for a workload pair these
systems do not report together, a cascaded voice agent beside long-context coding agents on one GPU, and change an
existing engine's scheduler rather than build one.

**Engines.** vLLM preempts among running requests under priority scheduling, never for a waiting one (#40004), and its
open fixes (#54530 and the ones before it) preempt at the request-count cap, which does not bind when KV fills. SGLang
ships priority preemption, but its admission loop stops when KV memory runs out; draft #45686 adds preemption on KV
exhaustion with a best-effort keep of the victim's KV, the SGLang counterpart of our first rule. Llumnix holds memory
headroom for a high-priority request and migrates others away; DeepInfer admits prefill only below a 90% KV watermark.
Sarathi-Serve's chunked prefill bounds how long a decode waits behind a prefill; our second rule takes lower-priority
chunks out of a high-priority reply's first steps altogether.

**Routers and gateways.** llm-d's flow control (priority bands, holdback ceilings, SLO-deadline ordering), SMG (reserved
slots per class, preemption before the first byte), Dynamo and AIBrix (priority queues mapped onto engine priority) and
vllm-project/router (program admission against logical KV) act before a request reaches the engine. llm-d's flow control
(emulated), SMG and an abort-and-resubmit policy are baselines here; none can take KV from a running request, which is
where the voice turn waits.

**KV across agents' pauses.** InferCept, CachedAttention, Pensieve, KVFlow, Tokencake, Continuum, MORI, CacheWise and
CacheScout park, retain or prefetch agents' KV around tool calls, and "When Fancy Eviction Fails" finds LRU hard to beat
on production prefix-cache traces. Our simulator agrees for agent-side retention: session history buys nothing, and an
oracle of the tools' return times buys under 9% of median resume latency.

**Voice under contention.** LiveKit's preemptive generation, Endpoint Anticipation and PredGen start the LLM earlier in
the turn. Unmute and NVIDIA's single-GPU voice agent document the co-location cost, and Metronome shows full-duplex
models collapsing when session KV fills the pool. We add the voice tier's tail beside a background agent class in one
engine, against the configurations an operator can deploy.

## 8. Limitations

**One model on one card, end to end.** Every E2 result is Qwen3-4B on one RTX 5060 Ti, whose 36,400-token pool the
agents' contexts fill. An A100 40 GB is forecast (§5.7), not yet measured. Other cards are placed by the simulator, from
costs scaled by published peaks, and the coordinates that place them do not carry a configuration's gain from one card
to another.

**A few calls.** The headline runs two callers, and with 4 and 6 the configuration misses each one's SLO in every round,
by a median of 0.27 s and 0.26 s (§5.6): many calls per GPU is its known limit.

**Synthesized callers.** The callers speak scripted lines synthesized in six Kokoro voices, with the voice agent's
endpointing fixed. Its end-of-turn wait is about 0.82 s of every configuration's V2V, the same for all of them; real
callers vary in pace, pauses and barge-ins.

**Block-scaled traces.** The agents replay SemiAnalysis' Claude Code traces scaled 32× in blocks of tokens, so that a
session's context fits the box's pool; the original traces fit no GPU (their median prompt is 142k tokens). The scaling
sets KV pressure, which is what the result depends on.

**What a resume recomputes.** A preempted request's KV is in the CPU tier, and on resume vLLM reloads what the tier
still holds. How much it reloads and how much it recomputes, no vLLM 0.32 counter shows.

**The variant, not the branches.** E2 ran the change as the make-room variant on vLLM 0.32.0 with the PRs' switch
values. The trace test holds the variant and the PRs' diff, ported to 0.32.0, to the same steps, but the PR branches
themselves, on vLLM's main, ran only on CPU.

**What five runs resolve.** Each comparison rests on five runs a configuration (15 for the configuration to deploy).
Differences of a few hundred milliseconds between configurations near the SLO are within the spread of their runs, and
we say so where they arise rather than test them; the pairs by round (§5.1, §5.6) compare configurations on the same
sample of the agents.

## 9. Conclusion

On one GPU, a voice tier beside coding agents loses its seconds inside the engine's scheduler, in two places: a waiting
turn cannot take KV from running background requests, and a running reply shares its steps with background prefill. Two
counting rules in vLLM's priority scheduler, a preemption for the waiting request and a hold on lower-priority prefill
through the reply's first sentence, bring the voice p95 from 10.5 s to 2.24 s while the agents keep 97% of their
throughput. That is still 0.20 s over a strict SLO; the nearest a control plane in front of the engine comes, by
aborting and resubmitting the agents' requests, is 0.49 s over, with the agents' TTFT p99 at 193 s. Where the pool has
room or many calls overlap, the change helps less. For a voice agent that shares its GPU with agents whose KV fills the
pool, the fix belongs in the engine: preempt for the waiting voice turn, and keep background prefill out of its first
sentence.
