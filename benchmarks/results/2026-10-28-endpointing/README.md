# The endpointing sweep (2026-10-28)

Most of a voice turn's floor on this box is endpointing: the voice agent ends a caller's turn once Silero VAD has heard
`SILENCE` seconds of silence, and then waits LiveKit's endpointing delay, `DELAY`, before the LLM gets the transcript,
0.55 s and 0.5 s by default. A shorter setting answers sooner, and takes a pause inside a line for its end more often.
These records measure both, for the synthetic callers every E2 point uses.

## How it ran

[`benchmarks/endpointing/sweep.sh`](../../endpointing/sweep.sh) ran four settings on 2026-10-28, from 13:29 to 14:04
UTC, with cyan and Phastform at v1.6-56-g885a874 (each `setting.json` names the release, 885a87483ccc). For each setting
it restarted the voice worker with `FLATSHARE_VAD_MIN_SILENCE` and `FLATSHARE_ENDPOINTING_MIN_DELAY` set, in bench mode
with E2's fixed endpointing, and one caller then played every line of the caller corpus once, in order: 40 lines after a
warm-up turn, with no interruptions, voice alone on Phastform's serving profile. Each setting's directory keeps
`setting.json` (the two values, the worker's start, the release), the caller's `bots/` (`turns.jsonl`, `summary.json`)
and `bots.log`, and `voice-worker.log`, the worker's journal while the caller ran; `sweep.log` is the sweep's own log.
Home directories are written as `~`.

[`summary.py`](../../endpointing/summary.py) gives the table from these records (`cd python/flatshare_bench && uv run
python ../../benchmarks/endpointing/summary.py ../../benchmarks/results/2026-10-28-endpointing`), and
`python/flatshare_bench/tests/test_endpointing_record.py` fails while this README's table is not what it gives. The
endpointing column is the worker's own time from the end of speech to its end-of-turn decision (its turn lines'
`end_of_turn_ms`); the last column names the lines whose answer began before they had ended (the caller's
`spoke_during_turn`).

<!-- summary.py -->
| VAD silence, s | Endpointing delay, s | Turns (failed) | V2V p50 / p95, ms | Endpointing p50 / p95, ms | Lines answered before they ended |
|---:|---:|---:|---:|---:|---|
| 0.55 | 0.5 | 40 (0) | 1,612 / 1,764 | 747 / 765 | 0 |
| 0.55 | 0.2 | 40 (0) | 1,644 / 1,825 | 753 / 774 | 0 |
| 0.4 | 0.3 | 40 (0) | 1,446 / 1,558 | 587 / 605 | 1: turn-22 |
| 0.3 | 0.2 | 40 (0) | 1,346 / 1,502 | 492 / 510 | 1: turn-22 |
<!-- /summary.py -->

## What it says

- **The VAD's silence is the lever, not the delay.** The worker ended each turn about 0.2 s after the VAD's silence had
  run out, at every delay tried: 747 ms at the median with 0.55 s of silence and a 0.5 s delay, 753 with 0.55 and 0.2,
  587 with 0.4 and 0.3, and 492 with 0.3 and 0.2. In all four settings the delay was no longer than the silence, and it
  never bound: lowering it alone, from 0.5 to 0.2 s, moved neither the endpointing (747 to 753 ms) nor the V2V (1,612 to
  1,644 ms at the median), one run each.
- **The silence it waits is the reply's delay, about one for one.** With 0.4 s of silence the V2V was 1,446 ms at the
  median, 166 ms sooner than the default's 1,612, and with 0.3 s 1,346 ms, 266 ms sooner; the p95s were 1,558 and
  1,502 ms against 1,764.
- **The cost is a line cut short.** At 0.4 and at 0.3 s one of the 40 lines was answered before it had ended, both times
  `turn-22`, "Do you have anything without nuts? My son has an allergy.", whose pause after its first sentence the VAD
  took for the end of the turn. At 0.55 s no line was.
- **What it changes.** Nothing in the measured configuration: every E2 point keeps the default 0.55 s and 0.5 s, so that
  the comparisons share one floor, and the serving profile's real callers get LiveKit's end-of-turn model instead,
  which this sweep does not measure. Fixed endpointing at 0.3 to 0.4 s of silence would take 0.17 to 0.27 s off the
  median reply for callers who do not pause inside a request; these synthetic lines pause less than people do.
