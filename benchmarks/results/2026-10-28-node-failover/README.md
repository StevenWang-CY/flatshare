# The two-node failover drills (2026-10-28)

The serving arrangement on two GPU nodes ([PRODUCTION.md](../../../docs/PRODUCTION.md#two-cards)): both calls on
Phastform beside at most four agents' sessions, and Gray a warm spare on the same engine, its speech servers resident,
with at most 12 agents' sessions; `[placement] calls = "fill"` and `idle_s = 150`. With the hosts at v1.6-56-g885a874,
[`ops/drills/node-failover.sh`](../../../ops/drills/node-failover.sh), at a8b896b, placed two calls with `POST
/v1/sessions`, as the voice worker does, ran a streaming turn of each every 3 s (at most 32 tokens) beside two agents'
streams, and after five whole turns of each call broke Phastform's engine:

- `node-kill-1`: SIGKILL to every process of the engine's unit, as a crash would end it, and systemd started it again;
- `node-stop-1`: `systemctl --user stop flatshare-engine`, the engine down until the drill's end.

| Drill | Turns before the event, whole | After, whole | Calls moved to Gray | Turns failed after the first move | Phastform passed over / unhealthy / available again | Gray covered |
|---|---:|---:|---:|---:|---|---:|
| kill | 10 of 10 | 148 of 148 | +0.7 s | 0 | +0.10 s / +2.1 s / +101.0 s | 238 s |
| stop | 10 of 10 | 152 of 152 | +1.9 s | 0 | +1.3 s / +3.3 s / down to the end | 244 s |

Times are from the event. Each directory keeps the drill's client (`client.sh`) and what it recorded (`turns.tsv` and
`agents.tsv`, a line a request: when it began, the session, the node before and after it, the status, the first byte
and the total in seconds, and how its stream ended; `nodes.jsonl`, each `/nodes` answer, stamped), the event's time,
the gateway's journal and the failed node's engine journal from the event on, the last response bodies, and the report
the drill printed (`report.txt`, and `drill-output.log` with the drill's own log). Home directories are written as `~`.

What they show:

- **The calls moved within a turn.** Both calls' next turn went to Gray 0.7 s after the kill and 1.9 s after the stop,
  and no turn failed after the move. No turn was in flight at either event, the calls being between turns, so none was
  cut.
- **The agents waited.** Gray covered from the move to the drills' end, as the moved calls stayed on it. The agents'
  requests that began after it covered waited out their class's `queue_timeout_s` and got 503 `queue_timeout`, 120 s
  later, twice each; a few that reached Gray in the second before the move completed. The agent stream in flight on
  Phastform at the kill broke with it, and at the stop one got 500 from the stopping engine and the next 502.
- **The killed engine was back in 101 s,** systemd's restart and its warm start; from then new calls would go to
  Phastform again, while the moved calls stayed on Gray to their end.

A first run of the kill drill, on the same arrangement, found the drill's client reading the engine's model id wrong:
every request it sent got 404, before the event as after it, and the drill broke the node all the same. a8b896b
fixed the client and made the event wait for whole turns; these are the first runs after it.
