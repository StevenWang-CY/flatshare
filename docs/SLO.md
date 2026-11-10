# Service levels

This page defines the service classes, the indicators measured for each, the voice SLO with its reference in each
campaign, and the production objectives, for a reader interpreting an SLO figure anywhere in the documents. Three
classes share the GPU. A gateway maps each tenant to a class, capped by the tenant's `max_class`, and sends the class's
priority to the engine, where a lower value is served first.

| Class | Traffic | Engine priority | Indicators |
|---|---|---:|---|
| `realtime` | Voice turns | 0 | Voice V2V, the voice turn's stages, the realtime first byte |
| `interactive` | Agent loops with a person waiting | 1 | None measured: no E2 arm runs this class |
| `background` | Batch agents, trace replays | 2 | The agents' token rate and their time to first token |

The deployed configuration runs no admission policy. The gateway sends the class's priority, bounds each class's work
and deadlines, and places calls and agents' sessions on the nodes
([the service contract](PRODUCTION.md#the-service-contract), [ADR-0007](adr/0007-the-gateway-is-the-serving-control-plane.md)).
Under the lifecycle policy, an experiment arm, the policy never queues or refuses a realtime request
([ADR-0003](adr/0003-realtime-is-never-shed.md)), and the gateway's own bounds still apply.

## Indicators

E2's report computes these for every point ([EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load)), except the
server-side voice turn and the control-plane overhead. The voice targets are stated against voice alone on the same box,
since endpointing alone takes about 0.8 s of every turn
([REPORT.md §6.2](REPORT.md#62-the-kv-preemption-removes-the-wait-for-kv-and-the-budget-cap-removes-agent-prefill-from-the-turns-steps)).
The target is what agent load may add.

| Indicator | What it measures, from where | Target, or what is reported |
|---|---|---|
| Voice V2V | The end of the caller's speech to the agent's first audio, from the waveforms at the caller (`flatshare-voice bots`); a turn unanswered within 30 s counts at 30 s | **The voice SLO: p95 at most 250 ms above voice alone with as many callers** |
| Replies over the SLO | The share of a configuration's replies, pooled over its runs, above the SLO's threshold | **At most 5%** |
| The voice turn's stages | Endpointing, the reply's first sentence and TTS's first audio from the voice worker's line per reply, and transport and playout, the rest of V2V | Per point |
| The reply's end | The agent's state leaving `speaking` with its last audio in (`end_by: "state"`), or a second of quiet from an agent with no state, from the callers' `turns.jsonl` | Per point, with each reply's audio length |
| Pauses inside a reply | Quiet stretches of 300 ms or more inside a reply's speech, at the caller | The replies with a pause over 500 ms, and the longest pause's percentiles, against voice alone's |
| Concealed audio | The share of samples the caller's WebRTC receiver played without a packet | Where the receiver gives it |
| The agent's playout | Each time a reply's output ran out of audio for 10 ms or more, and the least audio it had ready (`worker.playout`) | The replies with a stall, and the stalls' lengths |
| Realtime first byte | The gateway's request log, to the millisecond | Per point |
| Barge-in to silence | The caller speaking over the agent until the agent's audio falls below threshold | Per point |
| Server-side voice turn | The utterance sent to STT until TTS's first audio byte (`flatshare-bench voice-turn`) | p90 ≤ 800 ms (voice alone: 589 ms p50 and 665 ms p90, W1) |
| The agents' token rate | The agent requests that complete during the calls | Over the calls' duration, the frontier's axis, and over their first 240 s, the headline tables' |
| Agents' time to first token | AIPerf's per-request records | p50 and p99 |
| The gateway's decisions | Deferrals by reason, degraded replies, resubmissions and the headroom's false alarms (`flatshare_gateway_admission_*`) | Per point, under an admission policy |
| Realtime first byte, in production | `flatshare_gateway_sli_good_total` over `flatshare_gateway_sli_valid_total` (`class="realtime"`), failures included | **95% good, first content within 250 ms** |
| Control-plane overhead | `flatshare_gateway_overhead_seconds`, from the request body's end to the upstream write | **p99 ≤ 2 ms per request** |

- **The voice SLO's 250 ms** is about four times voice alone's spread from run to run, 1,744–1,805 ms over W3–W4's
  five runs. It is also about one gap between turns in conversation, 208 ms at the mean over ten languages
  ([Stivers et al., PNAS 2009](https://doi.org/10.1073/pnas.0903616106)).
- **Replies over the SLO.** A configuration meets the SLO when its replies' pooled p95 is within the threshold. The
  median of its runs' p95s can sit within a few hundred ms of the threshold while a third of the replies are over it,
  as for W5's 48-token configuration, at 2,236 ms with 286 of 885 replies over.
- **Realtime first byte, in production.** A valid request is good when its first content, the response body's first
  frame, comes within 250 ms of its receipt ([the gateway's README](../crates/flatshare-gateway/README.md#accounting)).
  One that fails, is refused, times out or hangs is a miss, which the first byte's histogram never sees. The 95% target
  is voice alone's p95 of 66 ms in W4 plus most of the voice SLO's 250 ms. W4's shared configurations miss it as they
  miss the voice SLO, the configuration to deploy at 372 ms and arm 9 at 290 ms, while voice alone holds it.
- **Control-plane overhead.** On v1.8, 0 of 82 requests took over 2 ms, the p99 was at most 0.2 ms, and the gateway
  added 0.12 ms to a voice request's first token at the median, paired against the engine's own socket ([the
  record](../benchmarks/results/2026-10-29-gateway-window/README.md)). On an Apple M5 Max an admission policy adds 6–24
  µs to the gateway's own time on average, `lifecycle` 9 µs ([the gateway's
  README](../crates/flatshare-gateway/README.md#overhead)).

## The SLO's reference, campaign by campaign

Each campaign holds its configurations to its own voice alone with as many callers, plus 250 ms.

| Campaign | Callers | Voice alone's V2V p95 | From | The SLO |
|---|---:|---:|---|---:|
| W2 | 2 | 1,744 ms | one run | 1,994 ms |
| W3–W4 | 2 | 1,767 ms (1,744–1,805) | the median of five runs | 2,017 ms |
| W5 | 2 | 1,784 ms (1,764–1,819) | the median of five rounds | 2,034 ms |
| W5, calls per GPU | 4 | 1,794 ms (1,780–1,894) | the median of three rounds | 2,044 ms |
| W5, calls per GPU | 6 | 1,885 ms (1,878–1,885) | the median of three rounds | 2,135 ms |
| The A100 run (an A100 SXM4 80 GB) | 2 | 1,419 ms (1,391–1,445) | the median of five rounds | 1,669 ms |
| RG, the two-card rounds | 2 or 1 | 1,693–1,842 ms | each round's own run, on its card ([the record](../benchmarks/results/2026-10-28-rounds/README.md)) | 1,943–2,092 ms |
| The 24-hour soak, serve mode | 2 at once, 20 an hour | 1,726 ms | two hours of voice alone with the same calls | 1,976 ms |
| OR: the serving engine's additions | 2 / 4 / 6 | — | each round's own run; the SLO's median over five rounds | 2,010 / 2,065 / 2,119 ms |
| OS: the decode hold | 2 / 4 / 6 | — | each round's own run ([the record](../benchmarks/results/2026-11-02-opt-rounds/README.md)) | — |
| The envelope rounds, version 1.24 | 2 | 1,714–1,817 ms | each round's own run, on its card ([the record](../benchmarks/results/2026-11-08-envelope/README.md)) | 1,964–2,067 ms |

## The production objectives

The production alerts burn on two objectives, both read in the gateway over a 30-day budget
([`ops/prometheus/rules.yml`](../ops/prometheus/rules.yml)). Each pages when both windows of a pair burn the budget
faster than the pair's rate, 1 h and 5 min at 14.4 times or 6 h and 30 min at 6 times, which spend 2% and 5% of it. It
opens a ticket when 3 d and 6 h both burn faster than the budget's own rate.

| Objective | Target | Alerts, page and ticket |
|---|---:|---|
| First byte: the realtime class's valid requests good | 95% | `FlatshareRealtimeFirstByteBudgetBurn`, `FlatshareRealtimeFirstByteBudgetSlowBurn` |
| Overhead: the requests within 2 ms in the gateway | 99% | `FlatshareOverheadBudgetBurn`, `FlatshareOverheadBudgetSlowBurn` |

- **The first-byte objective's** error ratio is 1 − good/valid, with good and valid as [above](#indicators), and its
  metrics are [the gateway's](../crates/flatshare-gateway/README.md#metrics-and-logs).
- **Where they are checked and shown.** `rules.yml` records both error ratios over five windows, its promtool tests
  check the alerts (`make check-rules`), and [`ops/grafana/flatshare.json`](../ops/grafana/flatshare.json) shows them
  beside the engine's state ([the runbook](../ops/RUNBOOK.md#monitoring)).

The first-byte objective is not the voice SLO, since it times a turn's LLM stage alone, and the alerts burn on it
because V2V needs the callers' waveforms. Where agents' KV fills the pool, the regime the alerts are for, it agreed with
the voice SLO's verdict in every such configuration of the records. It cannot see endpointing, STT, TTS or the audio's
way to the caller, which can take a turn over the voice SLO while it holds.
