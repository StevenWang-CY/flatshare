# The 24-hour soak (2026-10-28)

The serving arrangement for 24 hours, every attempt counted ([PRODUCTION.md](../../../docs/PRODUCTION.md#two-cards)):
both calls on Phastform beside at most four agents' sessions and Gray a warm spare with at most 12,
`[placement] calls = "fill"` with `idle_s` at 150 s, the voice worker in serve mode with the rounds' synthetic audio and
fixed endpointing, and 14 of the discriminating scenario's session trees for the agents (AIPerf, seed 26), the hosts at
v1.6-56-g885a874. Calls arrived as a Poisson process at 20 an hour, each with ten measured turns after a warm-up turn
and a quarter of the turns interrupted, as in [the smokes](../2026-10-28-two-node-smoke/README.md). Before it, two hours
of voice alone with the same calls set its SLO: their replies' p95, 1,726 ms, plus 250 ms. How it would be read was
fixed before it started: after four hours, more than 7.5% of the replies over the SLO would have stopped it and lowered
Phastform's bound to three sessions for another 24 hours. Of the first four hours' 827 replies 5 were over (0.6%), and
it ran its 24 hours.

Each directory is a compact soak record (`flatshare-bench soak export`;
[the runbook's Soaks](../../../ops/RUNBOOK.md#soaks) says what each file holds), and `flatshare-bench soak report DIR`
gives its summary, hour by hour. [`make-room.tsv`](make-room.tsv) holds each GPU node's make-room totals as its engine
journaled them every 10 s while they moved
(`journalctl --user -u flatshare-engine -o short-unix --grep 'make-room: budget cap held'`), a line each with only its
numbers: the time, the engine's process, the steps the budget cap held lower-priority prefill back, the prefill tokens
and admissions held, and the preemptions to make room, each cumulative over the process.
[`worker-events.tsv`](worker-events.tsv) holds the voice worker's events that tell what LiveKit knew of it, from its
journal
(`journalctl --user -u flatshare-voice-worker -o short-unix --grep 'marking as (un)?available|"process exiting"'`,
LiveKit Agents 1.8.8's own lines), a line each with the time and the worker's process: each status it sent LiveKit when
it changed, `full` or `available`, with the load it was set from, and each exit of a call's job process (`exit`).
[`table.py`](table.py) gives these four tables, which `python/flatshare_bench/tests/test_soak_record.py` holds to the
records:

<!-- table.py table -->
| Run | Hours | Calls (accepted, refused, failed) | Turns (failed) | Replies | Over the SLO | Replies p95, ms | Realtime SLI | Background tokens/s | Restarts | Guard pauses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Voice alone | 2 | 35 (31, 4, 0) | 310 (0) | 395 | the reference | 1,726 | 1.000 | 0 | 0 | 0 |
| The serving arrangement | 24 | 476 (402, 74, 0) | 4,020 (0) | 5,058 | 74 (1.5%) | 1,828 | 0.996 | 3,529 | 0 | 0 |
<!-- /table.py table -->

<!-- table.py hours -->
| Of the 24 hours | Replies | Over the SLO | Replies p95, ms | Realtime SLI | Background tokens/s |
|---|---:|---:|---:|---:|---:|
| Least | 132 | 0.0% | 1,721 | 0.983 | 582 |
| Most | 296 | 5.5% | 1,989 | 1.000 | 9,300 |
<!-- /table.py hours -->

<!-- table.py spare -->
| Scrapes | Calls on Phastform, the most at a scrape | Calls on Gray, the most at a scrape | Scrapes with Gray covering |
|---:|---:|---:|---:|
| 1,440 | 2 | 0 | 0 |
<!-- /table.py spare -->

<!-- table.py rules -->
| Card | Steps the budget cap held prefill back | Prefill tokens held | Admissions held | Make-room's preemptions | vLLM's preemptions | KV pool used, mean / most |
|---|---:|---:|---:|---:|---:|---:|
| Phastform, the calls | 4,853 | 154,562 | 4,456 | 0 | 0 | 4.9% / 96.8% |
| Gray, the spare | 0 | 0 | 0 | 0 | 3 | 5.9% / 99.3% |
<!-- /table.py rules -->

What it shows:

- **The calls kept the SLO over the day.** 74 of the 5,058 replies were over the SLO, 1.5%, inside the contract's 5%
  ([SLO.md](../../../docs/SLO.md)). The worst hour, hour 4, put 14 of 254 over (5.5%); 30 calls arrived in it, and 9 of
  them got no agent, both lines being busy. Of the day's 476 calls the worker answered 402 and 74 got no agent: 72 with
  both lines busy and 2 while a line was free (below); none failed, dropped or was cut, and none of the 4,020 turns
  failed.
- **Two calls found no agent with a line free.** Calls 262 and 386 came 1.0 and 1.2 s after a call's process had exited,
  while the last status the worker had sent LiveKit was still full ([`worker-events.tsv`](worker-events.tsv)), so the
  server dispatched them to no worker, and it dispatches a room once; the worker sent `available` 1.5 and 0.8 s after
  they came. LiveKit Agents 1.8.8 reads the worker's load every 0.5 s and sends the server its status only every 2.5 s:
  over the day LiveKit learned of a line freed from a full worker 1.6 s after the call's process exited, at the median
  of the 173 such exits, and 3.0 s at most. The soak counts 386 as `no_agent` and 262 as refused for capacity, as its
  reason reads the worker's status file, which showed both lines busy when its caller gave up 15 s later.
- **Nothing restarted, paused or went unread.** No unit failed or restarted, the guard never paused the load, and none
  of the 1,440 scrapes failed.
- **The spare stayed a spare.** Every call was placed on Phastform, which held both at once at most, and Gray carried
  none and never covered (the third table, at each scrape of the gateway).
- **What the rules did.** On Phastform, beside the calls, the budget cap held lower-priority prefill back in 4,853 steps
  over the day (154,562 prefill tokens and 4,456 admissions held, summed over the steps), and make-room never had to
  preempt: vLLM counted no preemption there; its KV pool held 4.9% of its blocks on average and 96.8% at most (the
  fourth table). On Gray, which carried no call, neither rule acted, and vLLM's own scheduler preempted 3 of its agents'
  requests, its pool at 99.3% at most.
- **The agents ran at their traces' own pace.** They did 3,529 tokens a second over the day, 27,273 requests completed
  and 0 failed, 9,300 in hour 0 against 582–5,402 in each hour after: AIPerf resumes its 14 trajectory sessions in one
  burst at its start ([its log](aiperf-start.txt)), and the replay then keeps its traces' recorded pace, two thirds of
  its turns waiting on others of their tree. The gateway held 11 agents' sessions at once in hour 0, at the median of
  its scrapes, and 1–10 in each hour after (1–2 in the last four), while AIPerf kept its 14 trees open to the end
  ([its last lines](aiperf-end.txt)). The rounds' windows, the calls' first 240 s a minute into the replay
  ([the rounds' record](../2026-10-28-rounds/README.md)), sit in that start, so their agents' rates are the replay's
  first minutes, not a day's.
