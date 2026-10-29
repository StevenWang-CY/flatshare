# A free line offered at once (2026-10-29)

After a call ended, LiveKit kept a full voice worker marked full until LiveKit Agents 1.8.8 sent its next status, every
2.5 s, and it dispatches a room only once: a call that came in that gap got no agent although a line was free. The
24-hour soak turned 2 of its 476 calls away so ([its record](../2026-10-28-soak/README.md)).

Version 1.8's worker sends its status as soon as a call's process closes (`send_status_when_calls_end` in
[`worker.py`](../../../python/flatshare_voice/src/flatshare_voice/worker.py)). These two runs measure it on the box,
with calls arriving faster than two lines take them, so that calls end with the worker full and the next come within
seconds.

<!-- table.py table -->
| Run | The worker sends LiveKit | Calls | Accepted | Refused, both lines busy | Refused, a line free | Failed |
|---|---|---:|---:|---:|---:|---:|
| `before` | v1.7: its status every 2.5 s | 233 | 51 | 172 | 10 | 0 |
| `after` | v1.8: and as each call's process closes | 233 | 53 | 180 | 0 | 0 |
<!-- /table.py table -->

- **Before: 1.5 s.** LiveKit learned of a free line 1.5 s after a call's process exited (the median of 44 exits from a
  full worker; 2.6 s at most), at the SDK's next status round. 10 of the 233 calls came in that gap and were turned
  away, 0.1–2.1 s after the exit.
- **After: 44 ms.** It learned of it 44 ms after (the median of 43 exits; 60 ms at most), and no call was turned away
  with a line free. 53 of the 233 calls got an agent, against 51 before.

## The runs

| | Both runs |
|---|---|
| Load | Voice alone on Phastform's card, in the serving profile; Gray paused (its engine's scrapes fail in both records) |
| Calls | 720 an hour for 20 minutes, each 2 measured turns after 1 warm-up turn; the callers' seed the same |
| Skipped | The soak tool dials no call while 8 are open, and a call no agent takes waits out its 15 s dispatch timeout: 10 in each run, left out of the table |
| `before` | v1.6-56-g885a874, the 24-hour soak's build (version 1.7's worker) |
| `after` | v1.7-53-g3cd2659, the build whose worker version 1.8 ships |

## The files

Each directory is a compact soak record (`flatshare-bench soak export`;
[the runbook's Soaks](../../../ops/RUNBOOK.md#soaks) says what each file holds). `worker-events-RUN.tsv` holds the
voice worker's events over its run, as [the soak record](../2026-10-28-soak/README.md) keeps them: each status it sent
LiveKit when it changed, `full` or `available` with the load, and each exit of a call's job process (`exit`).

[`table.py`](table.py) gives the table, and `python/flatshare_bench/tests/test_status_send_record.py` holds it to the
records. A refusal counts as made with a line free when the last status the worker had sent LiveKit was `full` and a
call's process had exited since: LiveKit dispatches no call to a worker it holds full, so its calls could only fall.
