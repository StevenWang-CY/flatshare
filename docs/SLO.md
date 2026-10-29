# Service levels

This page defines the service classes, the indicators measured for each, the voice SLO with its reference in each campaign, and the production objectives, for a reader interpreting an SLO figure anywhere in the documents. Three classes share the GPU, matching the tiers API providers sell as priority, standard and flex. A gateway maps each tenant to a class, capped by the tenant's `max_class`, and sends the class's priority to the engine, where a lower value is served first.

| Class | Traffic | Engine priority | Indicators |
|---|---|---|---|
| `realtime` | Voice turns | 0 | Voice V2V, the voice turn's stages, the realtime first byte |
| `interactive` | Agent loops with a person waiting | 1 | None measured: no E2 arm runs this class |
| `background` | Batch agents, trace replays | 2 | Background goodput and the agents' time to first token |

The deployed configuration sends the priority and nothing else ([ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md)). Under the lifecycle policy (`policy = "lifecycle"`), an experiment arm, a realtime request is never queued or refused ([ADR-0003](adr/0003-realtime-is-never-shed.md)).

## Indicators

E2's report computes these for every point (`flatshare-bench e2`; [EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load) gives the method), except two: the server-side voice turn, from a W1 probe, and the control-plane overhead, the gateway's own histogram. The voice targets are stated against voice alone on the same box: experiments end a turn a fixed delay after the end of speech, and endpointing takes about 0.8 s of every turn (0.76–0.82 s at the mean in W5; [REPORT.md §6.2](REPORT.md#62-make-room-removes-the-wait-for-kv-and-the-hold-removes-agent-prefill-from-the-turns-steps)), so the target is the degradation that agent load may add.

| Indicator | Definition | Target, or what is reported |
|---|---|---|
| Voice V2V | The end of the caller's speech to the agent's first audio, from waveforms at the caller (`flatshare-voice bots`), with fixed endpointing; a turn not answered within 30 s counts at 30 s | **The voice SLO: p95 at most 250 ms above voice alone with as many callers.** 250 ms is about four times voice alone's own spread from run to run (1,744–1,805 ms over W3–W4's five runs), and about one gap between turns in conversation: their mean is 208 ms over ten languages ([Stivers et al., PNAS 2009](https://doi.org/10.1073/pnas.0903616106)). Each campaign states it against its own voice alone with as many callers ([below](#the-slos-reference-campaign-by-campaign)) |
| Replies over the SLO | The share of a configuration's measured replies, pooled over its runs, whose V2V is above the SLO's threshold; the p95 of the same replies pooled beside it | **At most 5%**, the same SLO read reply by reply: a configuration meets it when its pooled p95 is within the threshold. The median of the runs' p95 can be within a few hundred ms of the threshold while a third of the replies are over it (W5's 48-token configuration: 2,236 ms at the median, 286 of 885 replies over) |
| The voice turn's stages | Endpointing, the reply's first sentence, TTS's first audio, transport and playout, from the voice worker's line per reply | Reported per point; the reply's first sentence dominates under load |
| The reply's end | When the agent has finished: its state (`lk.agent.state`, which LiveKit Agents sets) leaves `speaking`, and its last audio has arrived (`end_by: "state"` in the callers' `turns.jsonl`, from v1.6); from an agent that publishes no state, a second of quiet (`end_by: "silence"`, how every caller before v1.6 ended a reply) | Reported: `voice.replies` in a point's summary, with each reply's audio length. A pause of a second or more inside a reply ended it early under the old rule, and the caller's next line talked over the rest |
| Pauses inside a reply | The quiet stretches of 300 ms or more between the spans of a reply's speech, from its first audio to its end: the gaps a caller hears while the agent has not finished, natural pauses between sentences among them | Reported: the replies with a pause over 500 ms and the longest pause's percentiles, against voice alone's, which gives the natural pauses |
| Concealed audio | The caller's WebRTC receiver's count of the samples it played over a reply without a packet (`concealed_samples` less `silent_concealed_samples`, from the agent track's statistics), as a share of the samples it played | Reported, where the receiver gives its statistics |
| The agent's playout | From the voice worker (its turn lines, from v1.7): each time a reply's output ran out of audio before the reply was over, 10 ms or longer, and the least audio a reply had ready ahead of its playhead once synthesis fell behind it (`worker.playout` in a point's summary) | Reported: the replies with a stall, the stalls' lengths, and the low points of the inventory. A pause the caller hears with no stall behind it is in the synthesized audio, in a pause of the session's own or on the way to the caller, not a wait for the LLM or TTS |
| Realtime first byte | The gateway's request log, exact to the millisecond | Reported per point |
| Barge-in to silence | The caller starts speaking during the agent's speech, until the agent's audio falls below threshold: how long a cancelled reply keeps playing | Reported per point |
| Server-side voice turn | The utterance sent to STT, until the first byte of TTS audio (`flatshare-bench voice-turn`): the part serving controls | p90 ≤ 800 ms (voice alone: 589 ms p50 and 665 ms p90, W1) |
| Background goodput | The prompt and completion tokens per second of the agent requests that complete during the calls | Reported: the frontier's x-axis, over the calls' duration, which differs by configuration, and over the calls' first 240 s, the same span for every point, which the headline tables give |
| Agents' time to first token | AIPerf's per-request records | p50 and p99, reported |
| The gateway's decisions | Under an admission policy: deferrals by reason, degraded replies and resubmissions (`flatshare_gateway_admission_*`) | Reported per point. The gateway also counts the headroom's false alarms, which no E2 run had: every utterance of the callers is a turn |
| Realtime first byte, in production | The gateway's service level for the realtime class, failures included: `flatshare_gateway_sli_good_total` over `flatshare_gateway_sli_valid_total` (`class="realtime"`), where a valid request is good when it completes with its first content within 250 ms of its receipt, and one that fails, is refused for capacity, times out or hangs is a miss (the first byte's histogram never sees those; [the gateway's README](../crates/flatshare-gateway/README.md#accounting)) | 95% good, first content within 250 ms: voice alone's p95 (66 ms, W4) plus most of the 250 ms the voice SLO allows a turn. The shared configurations of W4 all miss it, as they miss the voice SLO (the PRs with a 48-token window 372 ms, arm 9 290 ms), and voice alone holds it. It is the first-byte objective, which the production alerts burn on, since V2V needs the callers' waveforms ([below](#the-production-objectives)) |
| Control-plane overhead | `flatshare_gateway_overhead_seconds`, on every request: from the end of the request body to the upstream write, with an admission policy's decision but not the wait it imposes | p99 ≤ 2 ms per request. On cyan with the W1 build: p99 0.44 ms for voice alone and 0.58 ms beside the agent replay. Paired against the engine's own socket, the gateway adds 0.29 ms to a voice request's first token on the final transport (W6; 0.38 ms on the W1 build) ([BENCHMARKS.md](BENCHMARKS.md#the-control-plane-flatshare-gateway-in-front-of-the-engine)). On an Apple M5 Max, an admission policy adds 6–24 µs of its own time on average, `lifecycle` 9 µs ([`flatshare-gateway` README](../crates/flatshare-gateway/README.md#overhead)) |

## The SLO's reference, campaign by campaign

| Campaign | Callers | Voice alone's V2V p95 | From | The SLO |
|---|---:|---:|---|---:|
| W2 | 2 | 1,744 ms | one run, the W2 build | 1,994 ms |
| W3–W4 | 2 | 1,767 ms (1,744–1,805) | the median of five runs; the four on the W3 build alone give 1,768.5 ms | 2,017 ms |
| W5 | 2 | 1,784 ms (1,764–1,819) | the median of five rounds | 2,034 ms |
| W5, calls per GPU | 4 | 1,794 ms (1,780–1,894) | the median of three rounds | 2,044 ms |
| W5, calls per GPU | 6 | 1,885 ms (1,878–1,885) | the median of three rounds | 2,135 ms |
| The A100 run (an A100 SXM4 80 GB) | 2 | 1,419 ms (1,391–1,445) | the median of five rounds | 1,669 ms |
| RG, the two-card rounds: round 1, Phastform | 2 / 1 | 1,750 / 1,738 ms | the round's own run, each round held to its own | 2,000 / 1,988 ms |
| RG: round 2, Gray | 2 / 1 | 1,722 / 1,693 ms | the round's own run | 1,972 / 1,943 ms |
| RG: round 3, Phastform | 2 / 1 | 1,760 / 1,732 ms | the round's own run | 2,010 / 1,982 ms |
| RG: round 4, Gray (the sweep's confirmation only) | 2 | 1,830 ms | the round's own run | 2,080 ms |
| RG: round 5, Phastform (the sweep's confirmation only) | 2 | 1,842 ms | the round's own run | 2,092 ms |
| The 24-hour soak, serve mode | 2 at once, 20 an hour | 1,726 ms | two hours of voice alone with the same calls | 1,976 ms |

Not computed: the interactive class's resume target, and the cache-contract honor rate, which went with leases. The plan's continuity indicator, audio gaps of 100 ms or more, is read from v1.6 and v1.7 as the pauses inside a reply, the concealed audio and the agent's playout above.

## The production objectives

The production alerts burn on two objectives, both read in the gateway: **the first-byte objective**, 95% of the realtime class's valid requests good (their first content within 250 ms, a failure, a refusal or a timeout counted as a miss), and **the overhead objective**, 99% of the requests within 2 ms in the gateway. [`ops/prometheus/rules.yml`](../ops/prometheus/rules.yml) records their error ratios over five windows (the first-byte one as 1 − good/valid) and alerts on their burn rates over a 30-day budget (`FlatshareRealtimeFirstByteBudgetBurn` and `FlatshareRealtimeFirstByteBudgetSlowBurn`, `FlatshareOverheadBudgetBurn` and `FlatshareOverheadBudgetSlowBurn`), which `rules.test.yml`'s promtool tests check (`make check-rules`), and [`ops/grafana/flatshare.json`](../ops/grafana/flatshare.json) shows them beside the engine's state ([the runbook](../ops/RUNBOOK.md#monitoring)).

The first-byte objective is not the voice SLO: it times a turn's LLM stage alone. Where agents' KV fills the pool, the regime the alerts are for, it agreed with the voice SLO's verdict in every such configuration of the records. It cannot see endpointing, STT, TTS or the audio's way to the caller, which can take a turn over the voice SLO while it holds.
