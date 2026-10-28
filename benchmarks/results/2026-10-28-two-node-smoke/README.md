# The two-node smokes (2026-10-28)

Two 15-minute soaks of the serving arrangement before its 24-hour soak: both calls on Phastform beside at most four
agents' sessions and Gray a warm spare with at most 12 ([PRODUCTION.md](../../../docs/PRODUCTION.md#two-cards)),
`[placement] calls = "fill"`, the voice worker in serve mode with the rounds' synthetic audio and fixed endpointing, and
16 of the discriminating scenario's session trees for the agents (AIPerf, seed 26), the hosts at v1.6-56-g885a874.
Each directory is a compact soak record (`flatshare-bench soak export`;
[the runbook's Soaks](../../../ops/RUNBOOK.md#soaks) says what each file holds), `flatshare-bench soak report DIR`
gives its summary, and [`table.py`](table.py) gives this table, which
`python/flatshare_bench/tests/test_two_node_smoke_record.py` holds to the records:

<!-- table.py -->
| Placement `idle_s` | Calls (accepted, refused, failed) | Turns (failed) | Replies p95, ms | Realtime SLI | Background requests: offered / completed / refused for an agent slot | Background tokens/s | Phastform / Gray: prompt tokens/s, requests running |
|---:|---:|---:|---:|---:|---:|---:|---|
| 300 s | 7 (6, 1, 0) | 60 (0) | 1,803 | 1.000 | 1,002 / 845 / 165 | 11,226 | 1,940, 0.27 / 9,296, 0.93 |
| 150 s | 7 (6, 1, 0) | 60 (0) | 2,017 | 0.964 | 1,348 / 1,272 / 76 | 15,456 | 7,903, 0.67 / 7,540, 0.40 |
<!-- /table.py -->

What they show:

- **The arrangement places as configured.** Every call went to Phastform, and the worker refused the one that arrived
  while two were in progress, for capacity; the agents' sessions filled Phastform's four slots and Gray's twelve; no
  node covered, and no turn failed.
- **The bound counts sessions, and the replay does not end its own.** AIPerf starts a new session tree as one ends,
  without ending the old session, so a finished tree holds its agent slot until it has gone `idle_s` without a
  request, and meanwhile the new tree's requests get 503 `no_agent_slot`, counted here as refused. With `idle_s` at
  300 s, 165 of the smoke's 1,002 background requests were refused, and Phastform's slots were held by finished trees
  often enough that its engine did 1,940 prompt tokens a second. At 150 s, just above the replay's 120 s cap on think
  times, so that a live tree keeps its slot, 76 of 1,348 were refused, and Phastform did 7,903.
- **The shared card then runs nearer its envelope.** With Phastform fuller, the replies' p95 rose from 1,803 to
  2,017 ms and the realtime service level fell from 1.000 to 0.964, one run of each: what the 24-hour soak measures at
  length. The serving arrangement keeps `idle_s` at 150 s, and the 24-hour soak offers 14 trees, so that two slots
  hold the trees that have ended until they idle out.
