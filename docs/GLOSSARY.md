# Glossary

The terms the other documents use, each with where it is defined in full. Code identifiers and the records' labels
(arm files, point names, `policy = "leaseholder"`) keep their names; prose uses the names below.

## The change and its parts

- **The change**: the two vLLM pull requests that change priority scheduling for a full KV cache,
  [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351).
  [ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351) states their rules.
- **The KV preemption** (#63350): a waiting request of a higher priority that lacks KV preempts running requests of a
  lower priority, at most N a step (`--max-priority-preemptions-per-step`).
- **The budget cap** (#63351): lower-priority prefill sits out the steps of a higher-priority request's first tokens.
  **Its window** is how many first tokens it covers (`--priority-prefill-budget-cap-tokens`, 16 by default; 48 in the
  configuration to deploy).
- **make-room**: the engine variant that builds vLLM 0.32.0's scheduler at each start from the PRs' scheduler diff,
  backported to 0.32.0, and a patch of its own (its switches, the reserve and its counters), and loads it with
  `--scheduler-cls` ([python/leaseholder_vllm](../python/leaseholder_vllm/README.md)). E2 measured the PRs'
  configuration as make-room with the PRs' switch values. Ported to 0.32.0, the PRs' diff schedules identically to it:
  `tests/test_trace.py` runs the variant and the PRs' own patches on the same request traces and requires every step
  to be equal. Run against the module W5's points loaded, it finds them equal but for two things: the held-token
  count, which that module overstated ([E3](BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)), and, under
  asynchronous scheduling with the budget cap, the hold's last step, which that module, like #63351 before its
  revision of 2026-10-25, held a step past the window
  ([after W5](BENCHMARKS.md#after-w5-the-first-tokens-in-flight)).
- **The reserve**: make-room's third switch, which keeps KV free for priority 0 by holding back lower-priority
  admissions. The PRs leave it out.
- **The configuration to deploy**: the KV preemption at 2 a step and the budget cap with a 48-token window, no reserve
  ([PRODUCTION.md](PRODUCTION.md#the-configuration)).

## The box and the hosts

- **The frozen box**: the one pinned configuration every experiment runs on: vLLM 0.32.0 with
  Qwen3-4B-Instruct-2507-FP8, priority scheduling, a 512-token step cap and a 20 GiB CPU KV tier, beside Whisper (STT)
  and Kokoro (TTS) on the same RTX 5060 Ti ([ARCHITECTURE.md §3.1](ARCHITECTURE.md#31-the-frozen-box)). **Tuned vLLM** in the headline
  means the frozen box; **stock vLLM** means vLLM without the change.
- **The CPU tier**: vLLM's `OffloadingConnector` writing each request's KV through to host memory as it is computed, so
  that an evicted or preempted request can reload it.
- **Phastform** and **Gray**: the two GPU nodes, each one RTX 5060 Ti 16 GB. **cyan**: the control host (the gateway,
  LiveKit, the voice agent and the callers, AIPerf). **peacock**: a CPU host for builds and tests.

## Classes and priorities

- **Realtime** (voice turns), **interactive** and **background** (coding agents, batch work) are the gateway's
  classes; they map to vLLM priorities 0, 1 and 2, and in vLLM a lower value is served first
  ([SLO.md](SLO.md)).

## Experiments and how they are counted

- **E0**: the tier cost model, what resuming an agent turn costs from the GPU cache, from host memory and by recompute.
- **E1**: the simulator's study of agent-side KV retention and prefetch; its result is negative.
- **E2**: the voice SLO measured end to end beside background agents, against every deployable alternative.
- **E3**: what make-room takes from the agents, from the engine's own counters.
- **Arm**: one engine and gateway configuration of E2, numbered by family (1: voice alone; 2 to 5: vLLM's own knobs;
  6: operator answers; 7: other engines; 8: routers; 9: make-room with the lifecycle policy; 10: MPS)
  ([EXPERIMENTS.md](EXPERIMENTS.md)).
- **Point**: one arm at one load, as one arm file names it. **Run**: one measurement of a point. **Round**: one run of
  each point of a campaign, in a rotated order, on the round's own sample of the agents' traces. **Campaign**: a set of
  rounds.
- **The standard scenario**: the whole corpus of agent traces at a realistic load. **The discriminating scenario**:
  the 48 traces with the longest contexts, whose KV fills the pool.
- **Agent session**: one replayed Claude Code session with its subagents, the replay's unit (a "session tree" in
  AIPerf).
- **Stages W0 to W8**: the project's build and measurement stages. W0 (2026-10-22): the frozen box and its first replay.
  W1 (2026-10-22): the gateway and the voice path. W2 (2026-10-22 to 23): E2's first batches. W3 (2026-10-23): the
  build with the voice agent's lifecycle events. W4 (2026-10-23 to 24): the discriminating scenario in rounds. W5
  (2026-10-24 to 25): the fixed make-room and the PRs' configuration in five rounds. W6 to W8: production hardening,
  refactors and failure drills.

## Measures

- **V2V (voice-to-voice)**: from the end of the caller's speech to the agent's first audio, read from the waveforms at
  the caller. A turn not answered within 30 s counts at 30 s.
- **The SLO**: voice V2V p95 at most 250 ms above voice alone with as many callers. Its reference is measured per
  campaign ([SLO.md](SLO.md)).
- **Goodput**: the agents' prompt and completion tokens per second over requests that complete; **the first 240 s**
  (the fixed window) counts it over the calls' first 240 s, so that configurations whose calls last longer compare
  fairly.
- **Percentiles** are inclusive (Python's `statistics.quantiles(..., method="inclusive")`) everywhere: the E2 report,
  the probes and the simulator.

## The lifecycle policy and the simulator's coordinates

- **The lifecycle policy** (`leaseholder` in arm files): the gateway policy timed by the conversation, which E2 measured
  as a negative result. Its **headroom** keeps one voice turn of KV free from speech onset by deferring new background
  requests; its **throttle** budgets background prefill from the end of a turn to the reply's first audio; **degrade**
  caps a reply at 16 tokens when the headroom was not there; **aging** admits deferred work after 10 s
  ([METHODS.md](METHODS.md#what-lifecycle-timing-could-and-could-not-do)).
- **KV pressure**: the agents' live working set over the KV pool. **π**: the median pause between an agent's turns over
  the time to restore its context. **ρ**: restore time over recompute time at the median prompt. They locate a card and
  a workload on E1's map ([METHODS.md](METHODS.md#what-transfers-to-other-hardware)).
