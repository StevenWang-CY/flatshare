# Glossary

The terms the other documents use, each with where it is defined in full, for a reader who meets one they do not know.

## One name for each concept

The documents and the figures give each concept one name, the figures from
[`flatshare_bench.labels`](../python/flatshare_bench/src/flatshare_bench/labels.py).

| Concept | Use | Never |
|---|---|---|
| #63350 and #63351 together | the change | the PRs, the engine change, the scheduler change |
| The two proposals on GitHub: their diff, branches and state | the pull requests, or #65282 and #65283 (#63350 and #63351 until 2026-11-03) | the PRs |
| #63350's rule | the KV preemption | make-room (for the rule), KV make-room |
| #63351's rule, and how many first tokens it covers | the budget cap, and its window | the prefill hold, the hold |
| The engine variant that runs the change | make-room | the generated scheduler |
| Both rules at the budget cap's default window of 16 tokens | the change at its 16-token window | make-room and the hold |
| Both rules at two preemptions a step, a 48-token window and no reserve | the configuration to deploy | the PRs' configuration, the PRs' values |
| Stock vLLM with the box's tuning, as a baseline | tuned vLLM | the frozen box (for the baseline), the frozen engine, frozen |
| A policy that aborts agent requests with no first byte and resubmits them | preempt-resubmit | abort-and-resubmit, the router |
| One replayed Claude Code session with its subagents | an agent session | session tree |
| The agents' prompt and completion tokens a second | the agents' token rate | goodput |
| One card for the calls and another for the agents | dedicated roles | a card per workload |
| Two cards, both calls on the first and a warm spare | the arrangement to deploy | the serving arrangement |

## The change and its parts

- **The change**: the two pull requests proposed to vLLM, [#65282](https://github.com/vllm-project/vllm/pull/65282)
  and [#65283](https://github.com/vllm-project/vllm/pull/65283), opened as #63350 and #63351, the numbers the records
  use ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-change)).
- **The KV preemption** (#63350): a higher-priority waiting request that lacks KV preempts lower-priority running
  requests, at most N a step (`--max-priority-preemptions-per-step`).
- **The budget cap** (#63351): lower-priority prefill sits out the steps of a higher-priority request's first tokens.
  **Its window** (T, `--priority-prefill-budget-cap-tokens`) is how many, 16 by default and 48 in the configuration to
  deploy. The name is the flag's, where #63351's title calls the rule a hold.
- **make-room**: the engine variant that builds vLLM 0.32.0's or 0.33.0's scheduler from the pull requests' diff at each
  start, with a reserve of its own ([python/flatshare_vllm](../python/flatshare_vllm/README.md)). A trace test holds it
  to the pull requests' diff on 0.32.0, step for step (`tests/test_trace.py`). The rounds of the report's §6.1–§6.10
  ran it as it was before the pull requests' revisions of 3 and 5 November, which change how it schedules
  ([§4](REPORT.md#4-implementation)). W5's module also overstated its held tokens and, scheduling asynchronously, held
  one step past the window ([BENCHMARKS.md](BENCHMARKS.md#do-w5s-numbers-hold-on-the-code-of-25-october)).
- **The reserve**: make-room's third switch, KV kept free for priority 0 by holding back lower-priority admissions. The
  pull requests leave it out.
- **The configuration to deploy**: the KV preemption at 2 a step and the budget cap at 48 tokens, with no reserve, at
  the serving profile's 256-token step cap without a per-request threshold
  ([PRODUCTION.md](PRODUCTION.md#the-configuration)), and in W5 at 512/256.
- **The serving profile**: the configuration the deployment serves,
  [`rtx-5060-ti-serve.toml`](../deploy/profiles/rtx-5060-ti-serve.toml). It holds the frozen box with the configuration
  to deploy at 256-token steps, the engine's additions, the cache-aware queue and the SM copy, for two calls and up to
  24 agent sessions a card ([ARCHITECTURE.md §3.1](ARCHITECTURE.md#31-the-serving-profile)).
- **The engine's additions** (version 1.15), five switches the serving profile sets around the two rules
  ([the report's §6.10](REPORT.md#610-the-serving-engines-additions-cut-the-replies-over-the-slo-by-more-than-half)):
  - **retention** keeps the calls' prefixes on the GPU between turns;
  - **just in time** schedules the next batch when the step in flight is due to end;
  - **the announced turn** holds agent prefill from the caller's end of speech;
  - **the reservations taken** let a call free only its own class's block reservations;
  - **the decode hold** (version 1.17) keeps lower-priority decodes out of a reply's first tokens.
- **The cache-aware queue** (version 1.24): within a priority, the request with the least prefill left after the GPU
  pool's and the CPU tier's hits goes first, and one that has waited 120 s goes ahead
  ([the report's §6.11](REPORT.md#611-ordering-the-agents-by-what-they-have-left-raises-the-cards-envelope-from-4-sessions-to-24)).
- **The SM copy** (version 1.24): the CPU tier's loads on vLLM's Triton copy kernel, where the copy engine would hold
  the steps beside each load.
- **The envelope**: the most agent sessions a card holds beside both calls by the
  [resource gate's rule](../benchmarks/quality/RESOURCE-GATE.md), the serving profile's `max_agents`: 24 on one card
  from version 1.24, 4 before, and 4 and 12 on two cards.

## The box, the hosts and the classes

- **The box**: Phastform's RTX 5060 Ti 16 GB and the frozen box it runs, and Gray, deployed the same way.
- **The CPU tier**: vLLM's `OffloadingConnector`, which writes each request's KV through to host memory for an evicted
  or preempted request to reload, 20 GiB on the box.
- **cyan**: the CPU-only control host of the gateway, LiveKit, the voice agent, the callers and AIPerf.
- **The frozen box**: the configuration every experiment on the box runs, vLLM 0.32.0 with Qwen3-4B-Instruct-2507-FP8,
  priority scheduling, a 512-token step cap and a 20 GiB CPU tier, beside Whisper and Kokoro
  ([ARCHITECTURE.md §5.1](ARCHITECTURE.md#51-the-frozen-box)). Records label its engine *frozen*.
- **Gray**: the second GPU node, identical to Phastform, the two-card rounds' second card and the warm spare.
- **peacock**: a CPU host for builds, tests and the change's per-step cost baseline.
- **Phastform**: the box's host, with one RTX 5060 Ti 16 GB, two Xeon E5-2609 v4 and 188 GiB of RAM.
- **Stock vLLM**: vLLM 0.32.0 without the change, whatever its flags.
- **Tuned vLLM**: stock vLLM as an operator would tune it, the frozen box's engine, which on the A100 also ran at the
  card's best step cap, 2,048/0 ([the A100 run](BENCHMARKS.md#do-the-rules-carry-to-a-datacenter-card)).
- **Realtime** (voice turns), **interactive** and **background** (coding agents, batch work) are the gateway's classes,
  at vLLM priorities 0, 1 and 2, where a lower value is served first ([SLO.md](SLO.md)).

## Experiments and how they are counted

- **A' and C'**: optional arms of the A100 run. A', 32 agent sessions at the box's pool, ran one round
  ([the A100 run](BENCHMARKS.md#do-the-rules-carry-to-a-datacenter-card)). C', Qwen3-14B-FP8 at an A100 40 GB's pool,
  was only [forecast](../benchmarks/a100/forecast.md).
- **Agent session**: one replayed Claude Code session with its subagents, AIPerf's "session tree".
- **Arm**: one engine and gateway configuration of E2, numbered 1 to 10 by family
  ([the arms](EXPERIMENTS.md#e2-voice-under-agent-load)).
- **Campaign**: a set of rounds or runs under one code, such as W5 or RG, which
  [the campaign map](EXPERIMENTS.md#which-campaign-measured-what-on-which-code) lists.
- **The discriminating scenario**: 24 agent sessions of the 48 traces with the longest contexts, whose KV fills the
  pool.
- **E0**: the tier cost model, of resuming an agent turn from the GPU cache, from host memory or by recompute.
- **E1**: the simulator's study of agent-side KV retention and prefetch, whose result is negative.
- **E2**: the voice SLO measured end to end beside background agents, against the configurations deployable without the
  change.
- **E3**: what make-room takes from the agents, from the engine's counters.
- **G-sim**: E1's gate, under which no lease-aware policy runs if even the oracle gains under about 10% of resume TTFT
  over LRU in the realistic regime. Read at p50, E1 took this branch ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)).
- **The KV case**: a probe that times the first token of a priority-0 request of about 750 tokens while two 18k-token
  priority-2 requests fill the KV pool ([BENCHMARKS.md](BENCHMARKS.md#why-does-a-voice-turn-wait-beside-agents)).
- **Point**: one arm at one load, as one arm file names it.
- **Preempt-resubmit** (arm 8c): a gateway policy that aborts the background requests with no first byte and resubmits
  them.
- **The prefill case**: a probe that times the same request's first and 16th tokens behind three 10,000-token
  priority-2 prefills, with room in the pool.
- **Round**: one run of each point of a campaign, in rotated order, on the round's sample of the agents.
- **Run**: one measurement of a point.
- **Sample of the agents**: where each agent session enters its trace, drawn by AIPerf's seed (`agents-seed`). Samples
  differ in start points, not in which traces run.
- **The standard scenario**: the whole corpus of agent traces at 16 sessions, the realistic load, or 8 and 24 in W2
  and W4.
- **W0 to W8**: the project's build and measurement stages, from 2026-10-22 to 2026-10-25:
  - W0, the frozen box and its first replay;
  - W1, the gateway and the voice path;
  - W2, E2's first batches;
  - W3, the voice agent's lifecycle events;
  - W4, the discriminating scenario in rounds;
  - W5, the fixed make-room and the configuration to deploy;
  - W6 to W8, production hardening, refactors and failure drills.
- **W5's campaigns**: W5's campaign, the PR campaign and the extension, of five rounds each
  ([BENCHMARKS.md](BENCHMARKS.md#how-close-does-each-configuration-bring-the-voice-tail-to-its-slo) gives their
  configurations), and calls per GPU, three rounds at 4 and 6 callers.

## Measures

- **The agents' token rate** (`goodput` in records and code): the prompt and completion tokens a second of the agent
  requests that complete during the calls, or during **the first 240 s** of them, the same span for every
  configuration. It is not goodput in the serving literature's sense, and 0.47% of it was generated in W5.
- **Percentiles** are inclusive everywhere, Python's `statistics.quantiles(..., method="inclusive")`, defined once in
  [`flatshare_stats`](../python/flatshare_stats/README.md).
- **Realtime first byte**: the gateway's time from a realtime request's receipt to the engine's first response byte
  ([SLO.md](SLO.md#indicators)).
- **Running share**: the share of time with free KV under two calls' 2,048 tokens, sampled every 0.5 s and modeled by
  flatshare-sim ([BENCHMARKS.md](BENCHMARKS.md#how-often-do-running-agents-leave-two-calls-short-of-kv)).
- **The SLO**: voice V2V p95 at most 250 ms above voice alone with as many callers, its reference measured per campaign
  ([SLO.md](SLO.md#the-slos-reference-campaign-by-campaign)).
- **TTFT** (time to first token): from AIPerf's records for the agents, and from sending to the first token back for a
  probe.
- **V2V (voice-to-voice)**: the end of the caller's speech to the agent's first audio, from the waveforms at the
  caller. A turn unanswered within 30 s counts at 30 s.

## The lifecycle policy and the simulator's coordinates

- **The critical window**: a voice turn's span from the end of the caller's turn to the reply's first audio.
- **KV pressure**: the agents' live working set over the KV pool
  ([REPORT.md §6.7](REPORT.md#67-the-rules-carry-to-an-a100)).
- **The lifecycle policy** (`policy = "lifecycle"`): the gateway policy timed by the conversation, a measured negative
  result ([REPORT.md §6.5](REPORT.md#65-a-control-plane-in-front-of-the-engine-makes-room-only-by-aborting-agent-work)).
  Its **headroom** defers new background requests from speech onset to keep a voice turn's KV free, and its
  **throttle** budgets background prefill in the critical window. **Degrade** caps a reply at 16 tokens without
  headroom, and **aging** admits deferred work after 10 s.
- **The realistic regime** (E1): the replay with the traces' own think times and the CPU tier at four times the pool,
  over 20 seeds, where E1's gate is read
  ([BENCHMARKS.md](BENCHMARKS.md#does-knowing-when-an-agent-returns-help)).
- **π**: the median pause between an agent's turns over the time to restore its context.
- **ρ**: restore time over recompute time at the median prompt. With π and KV pressure it places a regime on E1's map.

## Former names

The project was Leaseholder through v1.1, renamed Flatshare on 2026-10-26. The records, tags and release assets from
before keep the former names, which the code that reads records accepts.

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
| `policy = "leaseholder"`, the lifecycle policy | `policy = "lifecycle"`; the gateway refuses the former since v1.8 |
