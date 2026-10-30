# Glossary

The terms the other documents use, for a reader who meets one they do not know, each with where it is defined in full,
sorted within each section. [One name for each concept](#one-name-for-each-concept) gives the name the documents and the
figures use for each concept, and the names it replaces. Code identifiers and the records' labels (point names, arm
labels) keep their names. The project was called Leaseholder through v1.1, and the records keep its former names:
[Former names](#former-names) maps them to the current ones.

## One name for each concept

Each concept has one name in the documents and in the figures. The figures take theirs from one table in the code,
[`flatshare_bench.labels`](../python/flatshare_bench/src/flatshare_bench/labels.py), and this table gives the same
names. Where a retired name stays for another concept, as make-room does for the engine variant, the Never column says
which use it retires.

| Concept | Use | Never |
|---|---|---|
| #63350 and #63351 together, as a change to vLLM's priority scheduling | the change | the PRs, the engine change, the scheduler change |
| The two proposals on GitHub: their diff, branches and state | the pull requests, or #63350 and #63351 | the PRs |
| #63350's rule | the KV preemption | make-room (for the rule), KV make-room |
| #63351's rule, and how many first tokens it covers | the budget cap, and its window | the prefill hold, the hold |
| The engine variant that runs the change on vLLM 0.32.0 | make-room | the generated scheduler |
| Both rules at the budget cap's default window of 16 tokens | the change at its 16-token window | make-room and the hold |
| Both rules at two preemptions a step, a 48-token window and no reserve | the configuration to deploy | the PRs' configuration, the PRs' values |
| Stock vLLM with the box's tuning, the baseline of a comparison | tuned vLLM | the frozen box (for the baseline), the frozen engine, frozen |
| A policy that aborts agent requests with no first byte and resubmits them | preempt-resubmit | abort-and-resubmit, the router |
| One replayed Claude Code session with its subagents | an agent session | session tree |
| The agents' prompt and completion tokens a second | the agents' token rate | goodput |
| One card for the calls and another for the agents | dedicated roles | a card per workload |
| Two cards, both calls on the first and a warm spare | the arrangement to deploy | the serving arrangement |

## The change and its parts

- **The budget cap** (#63351): lower-priority prefill sits out the steps of a higher-priority request's first tokens.
  **Its window** is how many first tokens it covers (`--priority-prefill-budget-cap-tokens`, 16 by default; 48 in the
  configuration to deploy), which the technical report writes T. The name is the flag's, and #63351's title says what it
  does: "hold lower-priority prefill for first tokens".
- **The change**: the two pull requests proposed to vLLM (open, with no maintainer review as of 2026-10-29) that change
  priority scheduling, [#63350](https://github.com/vllm-project/vllm/pull/63350) for a full KV cache and
  [#63351](https://github.com/vllm-project/vllm/pull/63351) for prefill beside a priority request's first tokens.
  [ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change-63350-and-63351) states their rules. **The pull requests** are the
  proposals themselves, their diff, branches and state. In a figure *the change* is both rules at the budget cap's
  default 16-token window.
- **The configuration to deploy**: the change's switches, the KV preemption at 2 a step and the budget cap with a
  48-token window and no reserve, at the serving profile's step cap of 256 tokens without a per-request threshold
  ([PRODUCTION.md](PRODUCTION.md#the-configuration)). W5's evaluation ran the same switches on the frozen box's engine,
  at 512 tokens a step with a 256-token threshold.
- **The KV preemption** (#63350): a waiting request of a higher priority that lacks KV preempts running requests of a
  lower priority, at most N a step (`--max-priority-preemptions-per-step`; in make-room,
  `make_room.max_preemptions_per_step`).
- **make-room**: the engine variant that builds vLLM 0.32.0's scheduler at each start from the pull requests' scheduler
  diff, backported to 0.32.0, and a patch of its own (its switches, the reserve and its counters), and loads it with
  `--scheduler-cls` ([python/flatshare_vllm](../python/flatshare_vllm/README.md)). The name is the variant's only:
  #63350's rule is the KV preemption. E2 measured the change as make-room with the pull requests' switch values, to
  which their diff, ported to 0.32.0, schedules identically.
  - `tests/test_trace.py` runs both on the same request traces and requires every step to be equal. It finds the module
    W5's points loaded equal but for its held-token count, which it overstated
    ([E3](BENCHMARKS.md#e3-what-make-room-takes-from-the-agents)), and for the budget cap's last held step under
    asynchronous scheduling, a step past the window, as #63351 held before its revision of 2026-10-25
    ([after W5](BENCHMARKS.md#after-w5-the-first-tokens-in-flight)).
  - It finds the module that the sweep's and the rounds' points loaded, built on 2026-10-27, equal in every step to
    today's, which differs from it only in counting the blocks a preemption releases in one pass.
- **The reserve**: make-room's third switch, which keeps KV free for priority 0 by holding back lower-priority
  admissions. The pull requests leave it out.

## The box and the hosts

- **The box**: the RTX 5060 Ti 16 GB host and the frozen box it runs: Phastform, and Gray, deployed the same way.
- **The CPU tier**: vLLM's `OffloadingConnector` writing each request's KV through to host memory as it is computed,
  so that an evicted or preempted request can reload it; 20 GiB on the box.
- **cyan**: the control host, CPU only: the gateway, LiveKit, the voice agent and the callers, and AIPerf.
- **The frozen box**: the one pinned configuration every experiment on the box runs: vLLM 0.32.0 with
  Qwen3-4B-Instruct-2507-FP8, priority scheduling, a 512-token step cap and a 20 GiB CPU KV tier, beside Whisper (STT)
  and Kokoro (TTS) on the same GPU ([ARCHITECTURE.md §5.1](ARCHITECTURE.md#31-the-frozen-box)). Beside other
  configurations its engine is tuned vLLM, which the A100 run's records label *frozen* on whatever card it runs.
- **Gray**: the second GPU node, identical to Phastform: the probes' booked windows (the first make-room probes ran on
  it on 2026-10-23), the second card of the two-card rounds and the arrangement to deploy's warm spare.
- **peacock**: a CPU host for builds and tests, and the machine of the change's per-step cost baseline
  (`make perf-make-room`).
- **Phastform**: the box's host: one RTX 5060 Ti 16 GB, two Xeon E5-2609 v4 and 188 GiB of RAM.
- **Stock vLLM**: vLLM 0.32.0 without the change, whatever its flags.
- **Tuned vLLM**: stock vLLM as an operator would tune it. On the box it is the frozen box's engine. On the A100 the
  technical report names its step cap: at 512-token steps it runs the box's tuning, as the core compares, and at 2,048/0
  the card's best stock step cap ([the A100 run](BENCHMARKS.md#what-it-measured-2026-10-26)).

## Classes and priorities

- **Realtime** (voice turns), **interactive** and **background** (coding agents, batch work) are the gateway's
  classes; they map to vLLM priorities 0, 1 and 2, and in vLLM a lower value is served first
  ([SLO.md](SLO.md)).

## Experiments and how they are counted

- **A' and C'**: two of the A100 run's optional arms: A', the box's pool at 32 agent sessions (one round ran), and C',
  Qwen3-14B-FP8 at an A100 40 GB's own pool (not run) ([BENCHMARKS.md](BENCHMARKS.md#the-optional-arms)).
- **Agent session**: one replayed Claude Code session with its subagents, the replay's unit (a "session tree" in
  AIPerf).
- **Arm**: one engine and gateway configuration of E2, numbered by family: 1, voice alone; 2 to 5, stock vLLM and its
  own knobs; 6, static answers in the gateway; 7, what engines ship; 8, what routers ship; 9, make-room with the
  lifecycle policy; 10, MPS, planned and not run ([EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load)).
- **Campaign**: a set of rounds, or of runs, under one code: W2, W3–W4, W5, FB, the A100 run (A1), RS, SW, RG and the
  soak. [The campaign map](EXPERIMENTS.md#which-campaign-measured-what-on-which-code) gives each one's code, what it
  measured, when it ran, its rounds and its samples of the agents' traces, and the code it ran.
- **The discriminating scenario**: 24 agent sessions of the 48 traces with the longest contexts, whose KV fills the
  pool.
- **E0**: the tier cost model, what resuming an agent turn costs from the GPU cache, from host memory and by
  recompute.
- **E1**: the simulator's study of agent-side KV retention and prefetch; its result is negative.
- **E2**: the voice SLO measured end to end beside background agents, against the configurations of vLLM 0.32, SGLang
  0.5.22, SMG and llm-d's flow control (emulated) that an operator can deploy without the change.
- **E3**: what make-room takes from the agents, from the engine's own counters.
- **G-sim**: E1's gate: if even the oracle closes under about 10% of resume TTFT over LRU plus the CPU tier in the
  realistic regime, no lease-aware policy runs on the GPU. Read at p50, E1 took this negative branch
  ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)).
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
  with the voice agent's lifecycle events; W4 (2026-10-23 to 24): the discriminating scenario in rounds; W5 (2026-10-24
  to 25): the fixed make-room and the configuration to deploy in five rounds; W6 to W8 (2026-10-25): production
  hardening, refactors and failure drills.
- **W5's campaigns**: *W5's campaign* (voice alone, tuned vLLM and the configuration to deploy, with make-room as v0.3
  built it as its control), *the PR campaign* (tuned vLLM, the KV preemption alone, the change at its 16-token window
  and the configuration to deploy), *the extension* (preempt-resubmit, the configuration to deploy again, and the
  standard scenario's pair), each of five rounds, and *calls per GPU*, of three
  ([EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load)).

## Measures

- **The agents' token rate** (`goodput` in the records and the code): the agents' prompt and completion tokens per
  second over the requests that complete during the calls; **the first 240 s** (the fixed window) counts it over the
  calls' first 240 s, so that configurations whose calls last longer are counted over the same span. Nearly all of it
  is prompt tokens (the generated share is about 0.47% in W5), and it is not goodput in the serving literature's sense,
  the requests that meet their SLO.
- **Percentiles** are inclusive everywhere, in the E2 report, the probes and the simulator: Python's
  `statistics.quantiles(..., method="inclusive")`, defined once in
  [`flatshare_stats`](../python/flatshare_stats/README.md).
- **Realtime first byte**: the gateway's time from receiving a realtime request to the first byte of the engine's
  response, from its request log, exact to the millisecond ([SLO.md](SLO.md#indicators)).
- **Running share**: the share of the time in which free KV is under two calls' 2,048 tokens. The A100 run and RS
  measured it from the engine's metrics, sampled every 0.5 s, and flatshare-sim models it
  ([BENCHMARKS.md](BENCHMARKS.md#the-core-block-by-block)).
- **The SLO**: voice V2V p95 at most 250 ms above voice alone with as many callers. Its reference is measured per
  campaign ([SLO.md](SLO.md#the-slos-reference-campaign-by-campaign)).
- **TTFT** (time to first token): for the agents, from AIPerf's per-request records; for a probe, from sending its
  request to the first token streamed back.
- **V2V (voice-to-voice)**: from the end of the caller's speech to the agent's first audio, read from the waveforms at
  the caller. A turn not answered within 30 s counts at 30 s.

## The lifecycle policy and the simulator's coordinates

- **The critical window**: a voice turn's span from the end of the caller's turn to the reply's first audio, in which
  the lifecycle policy's throttle budgets background prefill.
- **KV pressure**: the agents' live working set over the KV pool, one of the coordinates that locate a card and a
  workload on E1's map ([REPORT.md §6.7](REPORT.md#67-the-rules-carry-to-an-a100)).
- **The lifecycle policy** (`policy = "lifecycle"`): the gateway policy timed by the conversation, which E2 measured as
  a negative result. Its **headroom** keeps one voice turn of KV free from speech onset by deferring new background
  requests, its **throttle** budgets background prefill from the end of a turn to the reply's first audio, **degrade**
  caps a reply at 16 tokens when the headroom was not there, and **aging** admits deferred work after 10 s
  ([REPORT.md §6.5](REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work)).
- **The realistic regime** (E1): the simulator's replay with the traces' own think times and the CPU tier at four times
  the pool, the box's ratio, over 20 seeds; E1's gate is read in it
  ([BENCHMARKS.md](BENCHMARKS.md#e1-parking-in-the-simulator)).
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
loggers, the scheduler modules and the policy's name; the gateway's configuration refuses `policy = "leaseholder"` since
v1.8. The report
shows arm 9's labels as its arm files give them now, `Arm 9` for `Leaseholder`.
