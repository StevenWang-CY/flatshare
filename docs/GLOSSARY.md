# Glossary

The terms the other documents use, each with where it is defined in full, sorted within each section. Code
identifiers and the records' labels (point names, arm labels) keep their names; prose uses the names below. The project
was called Leaseholder through v1.1, and the records keep its former names: [Former names](#former-names) maps them to
the current ones.

## The change and its parts

- **The budget cap** (#63351): lower-priority prefill sits out the steps of a higher-priority request's first tokens.
  **Its window** is how many first tokens it covers (`--priority-prefill-budget-cap-tokens`, 16 by default; 48 in the
  configuration to deploy).
- **The change**: the two vLLM pull requests that change priority scheduling,
  [#63350](https://github.com/vllm-project/vllm/pull/63350) for a full KV cache and
  [#63351](https://github.com/vllm-project/vllm/pull/63351) for prefill beside a priority request's first tokens.
  [ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351) states their rules.
- **The configuration to deploy**: the frozen box's engine with the KV preemption at 2 a step, the budget cap with a
  48-token window and no reserve ([PRODUCTION.md](PRODUCTION.md#the-configuration)).
- **The KV preemption** (#63350; **KV make-room** in make-room's switches): a waiting request of a higher priority that
  lacks KV preempts running requests of a lower priority, at most N a step (`--max-priority-preemptions-per-step`).
- **make-room**: the engine variant that builds vLLM 0.32.0's scheduler at each start from the PRs' scheduler diff,
  backported to 0.32.0, and a patch of its own (its switches, the reserve and its counters), and loads it with
  `--scheduler-cls` ([python/flatshare_vllm](../python/flatshare_vllm/README.md)). E2 measured the PRs' configuration
  as make-room with the PRs' switch values, to which the PRs' diff, ported to 0.32.0, schedules identically:
  `tests/test_trace.py` runs both on the same request traces and requires every step to be equal, and finds the module
  W5's points loaded equal but for its held-token count, which it overstated
  ([E3](BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)), and for the hold's last step under asynchronous
  scheduling with the budget cap, which it held a step past the window, as #63351 did before its revision of
  2026-10-25 ([after W5](BENCHMARKS.md#after-w5-the-first-tokens-in-flight)).
- **The reserve**: make-room's third switch, which keeps KV free for priority 0 by holding back lower-priority
  admissions. The PRs leave it out.

## The box and the hosts

- **The box**: the RTX 5060 Ti 16 GB host and the frozen box it runs: Phastform, the GPU node of every E2 point but
  the A100 run's.
- **The CPU tier**: vLLM's `OffloadingConnector` writing each request's KV through to host memory as it is computed,
  so that an evicted or preempted request can reload it; 20 GiB on the box.
- **cyan**: the control host, CPU only: the gateway, LiveKit, the voice agent and the callers, and AIPerf.
- **The frozen box**: the one pinned configuration every experiment on the box runs: vLLM 0.32.0 with
  Qwen3-4B-Instruct-2507-FP8, priority scheduling, a 512-token step cap and a 20 GiB CPU KV tier, beside Whisper (STT)
  and Kokoro (TTS) on the same GPU ([ARCHITECTURE.md §3.1](ARCHITECTURE.md#31-the-frozen-box)).
- **Gray**: the second GPU node, identical to Phastform, on which the first make-room probes ran (2026-10-23).
- **peacock**: a CPU host for builds and tests, and the machine of the change's per-step cost baseline
  (`make perf-make-room`).
- **Phastform**: the box's host: one RTX 5060 Ti 16 GB, two Xeon E5-2609 v4 and 188 GiB of RAM.
- **Stock vLLM**: vLLM 0.32.0 without the change, whatever its flags.
- **Tuned vLLM**: stock vLLM as an operator would tune it. On the box it is the frozen box; on the A100, the stock
  engine at that card's best step cap, 2,048/0 ([the A100 run](HARDWARE.md#what-it-measured-2026-10-26)).

## Classes and priorities

- **Realtime** (voice turns), **interactive** and **background** (coding agents, batch work) are the gateway's
  classes; they map to vLLM priorities 0, 1 and 2, and in vLLM a lower value is served first
  ([SLO.md](SLO.md)).

## Experiments and how they are counted

- **The A100 run** (A1, 2026-10-26): the release's kit on an A100 SXM4 80 GB on Google Colab. Its **core** repeated
  E2's discriminating scenario like for like (the box's model, pool, 512/256 and CPU tier, W5's arm files and five
  samples) with E0, the probes and a calibration point, and its **arms** after the core were the batch (256 requests
  in flight), tuned vLLM at 2,048/0, the PRs' values at that cap, A' (32 sessions) and FlashInfer's attention
  ([HARDWARE.md](HARDWARE.md#what-it-measured-2026-10-26)).
- **Agent session**: one replayed Claude Code session with its subagents, the replay's unit (a "session tree" in
  AIPerf).
- **Arm**: one engine and gateway configuration of E2, numbered by family: 1, voice alone; 2 to 5, stock vLLM and its
  own knobs; 6, static answers in the gateway; 7, what engines ship; 8, what routers ship; 9, make-room with the
  lifecycle policy; 10, MPS, planned and not run ([EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load)).
- **Campaign**: a set of rounds.
- **The discriminating scenario**: 24 agent sessions of the 48 traces with the longest contexts, whose KV fills the
  pool.
- **E0**: the tier cost model, what resuming an agent turn costs from the GPU cache, from host memory and by
  recompute.
- **E1**: the simulator's study of agent-side KV retention and prefetch; its result is negative.
- **E2**: the voice SLO measured end to end beside background agents, against every deployable alternative.
- **E3**: what make-room takes from the agents, from the engine's own counters.
- **FB** (2026-10-25): the final benchmark, five rounds of W5's headline configurations (voice alone, the frozen box
  and the PRs' 48-token configuration) on the final code, each round on the agents' sample of W5's round of the same
  number ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2); the records are in
  [`benchmarks/results/2026-10-25-e2-fb`](../benchmarks/results/2026-10-25-e2-fb)).
- **The KV case**: the probe in which a voice-sized request of about 750 tokens arrives at priority 0 while two
  18k-token priority-2 requests fill the KV pool, timed to its first token
  ([BENCHMARKS.md](BENCHMARKS.md#the-kv-case-make-room)).
- **Point**: one arm at one load, as one arm file names it.
- **Preempt-resubmit** (arm 8c): a gateway policy that aborts the background requests that have no first byte yet and
  resubmits them; of the configurations an operator can deploy without the change, the nearest to the SLO.
- **The prefill case**: the probe in which the voice-sized request arrives behind three 10,000-token priority-2
  prefills, with room in the pool, timed to its first and 16th tokens.
- **Round**: one run of each point of a campaign, in a rotated order, on the round's own sample of the agents'
  traces.
- **Run**: one measurement of a point.
- **The standard scenario**: the whole corpus of agent traces at 16 agent sessions, the realistic load (and at 8 and
  24 in W2 and W4).
- **W0 to W8**: the project's build and measurement stages. W0 (2026-10-22): the frozen box and its first replay; W1
  (2026-10-22): the gateway and the voice path; W2 (2026-10-22 to 23): E2's first batches; W3 (2026-10-23): the build
  with the voice agent's lifecycle events; W4 (2026-10-23 to 24): the discriminating scenario in rounds; W5
  (2026-10-24 to 25): the fixed make-room and the PRs' configuration in five rounds; W6 to W8 (2026-10-25): production
  hardening, refactors and failure drills.

## Measures

- **Goodput**: the agents' prompt and completion tokens per second over the requests that complete during the calls;
  **the first 240 s** (the fixed window) counts it over the calls' first 240 s, so that configurations whose calls
  last longer compare fairly.
- **Percentiles** are inclusive everywhere, in the E2 report, the probes and the simulator: Python's
  `statistics.quantiles(..., method="inclusive")`, defined once in
  [`flatshare_stats`](../python/flatshare_stats/README.md).
- **Running share**: the share of the time in which free KV is under two calls' 2,048 tokens. The A100 run measured it
  from the engine's metrics, sampled every 0.5 s, and flatshare-sim models it
  ([HARDWARE.md](HARDWARE.md#the-core-block-by-block)).
- **The SLO**: voice V2V p95 at most 250 ms above voice alone with as many callers. Its reference is measured per
  campaign ([SLO.md](SLO.md#the-slos-reference-campaign-by-campaign)).
- **V2V (voice-to-voice)**: from the end of the caller's speech to the agent's first audio, read from the waveforms at
  the caller. A turn not answered within 30 s counts at 30 s.

## The lifecycle policy and the simulator's coordinates

- **KV pressure**: the agents' live working set over the KV pool, one of the coordinates that locate a card and a
  workload on E1's map ([METHODS.md](METHODS.md#what-transfers-to-other-hardware)).
- **The lifecycle policy** (`policy = "lifecycle"`): the gateway policy timed by the conversation, which E2 measured as
  a negative result. Its **headroom** keeps one voice turn of KV free from speech onset by deferring new background
  requests, its **throttle** budgets background prefill from the end of a turn to the reply's first audio, **degrade**
  caps a reply at 16 tokens when the headroom was not there, and **aging** admits deferred work after 10 s
  ([METHODS.md](METHODS.md#what-lifecycle-timing-could-and-could-not-do)).
- **π**: the median pause between an agent's turns over the time to restore its context; a coordinate of E1's map.
- **ρ**: restore time over recompute time at the median prompt; a coordinate of E1's map.

## Former names

The project was called Leaseholder through v1.1 and was renamed Flatshare on 2026-10-26. The records committed before
the rename, under [`benchmarks/results`](../benchmarks/results), keep the former names, as do the tags up to v1.1 and
their release assets (`leaseholder-*.tar.gz`):

| Former name | Name now |
|---|---|
| Leaseholder, the project; `Leaseholder` in arm 9's labels and the units' descriptions | Flatshare; `Arm 9` |
| `lh-*` crates, tools and systemd units; `leaseholder.target` | `flatshare-*`; `flatshare.target` |
| `leaseholder_*` Python packages and loggers | `flatshare_*` |
| `lh_*_scheduler`, the engine variants' scheduler modules | `flatshare_*_scheduler` |
| `lh_gateway_*` metrics; `lh:*` recording rules | `flatshare_gateway_*`; `flatshare:*` |
| `LH_*` variables | `FLATSHARE_*` |
| `X-LH-*` headers | `X-Flatshare-*` |
| `~/leaseholder`, `~/.config/leaseholder` and `$XDG_RUNTIME_DIR/lh` | `~/flatshare`, `~/.config/flatshare` and `$XDG_RUNTIME_DIR/flatshare` |
| `policy = "leaseholder"`, the lifecycle policy | `policy = "lifecycle"` |

The code that reads records accepts the former names of what it reads: the gateway's metrics, the voice worker's
loggers, the scheduler modules and the policy's name, which the gateway's configuration still accepts too. The report
shows arm 9's labels as its arm files give them now, `Arm 9` for `Leaseholder`.
