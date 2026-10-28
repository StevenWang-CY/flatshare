"""A stand-in for ``flatshare-voice arrivals`` in the soak tests: it takes the soak's options and, every
``FAKE_ARRIVALS_GAP`` seconds (0.2) until ``--duration`` has passed, makes an attempt: in turn accepted (its
``--warmup-turns`` and ``--turns`` turns, ``FAKE_ARRIVALS_TURN`` s each, 0.02), refused by capacity, accepted, and
failed to connect. Each accepted call's turns answer in 1500 + 10 k ms for its k-th turn, the last with a pause of
600 ms. SIGINT ends it: no more attempts, the call in progress cut after its turn. It writes attempts.jsonl and
turns.jsonl as it goes and summary.json at its end, its argv to args.json and its pid to arrivals.pid in its run
directory. ``FAKE_ARRIVALS=crash`` on its run 0 exits 1 at once, as a run that cannot start would."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
for option in (
    "--corpus",
    "--calls-per-hour",
    "--duration",
    "--turns",
    "--warmup-turns",
    "--barge-in",
    "--seed",
    "--run",
    "--first-call",
    "--dispatch-timeout",
    "--max-open",
    "--worker-status",
    "--out",
):
    parser.add_argument(option, required=True)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "args.json").write_text(json.dumps(sys.argv[1:]))
(out / "arrivals.pid").write_text(str(os.getpid()))
run, first = int(args.run), int(args.first_call)
if os.environ.get("FAKE_ARRIVALS") == "crash" and run == 0:
    print("could not start", flush=True)
    sys.exit(1)
stopping = []
signal.signal(signal.SIGINT, lambda *_: stopping.append(time.time()))
gap = float(os.environ.get("FAKE_ARRIVALS_GAP", "0.2"))
turn_s = float(os.environ.get("FAKE_ARRIVALS_TURN", "0.02"))
turns_per_call = int(args.warmup_turns) + int(args.turns)
start, duration = time.time(), float(args.duration)
counts = {"accepted": 0, "refused": 0, "failed": 0}
with (out / "attempts.jsonl").open("a") as attempts, (out / "turns.jsonl").open("a") as turns:
    number = first
    while not stopping and time.time() - start < duration:
        due = time.time()
        kind = ("accepted", "refused", "accepted", "failed")[(number - first) % 4]
        record = {
            "run": run,
            "call": number,
            "due": due,
            "started": due,
            "late_ms": 0.0,
            "outcome": kind,
        }
        if kind == "refused":
            record |= {"reason": "capacity", "detail": "2 of 2 calls"}
        elif kind == "failed":
            record |= {"reason": "connect", "detail": "ConnectError: refused"}
        else:
            played = 0
            for turn in range(turns_per_call):
                if stopping:
                    break
                time.sleep(turn_s)
                row = {"run": run, "caller": number, "turn": turn, "clip": f"turn-{turn:02d}"}
                row |= {"v2v_ms": 1500.0 + 10 * turn, "barge_in": False, "end_by": "state"}
                row |= {
                    "reply_audio_ms": 2000.0,
                    "reply_pauses_ms": [600.0] if turn == turns_per_call - 1 else [],
                }
                if turn < int(args.warmup_turns):
                    row["warmup"] = True
                turns.write(json.dumps(row | {"at": time.time()}) + "\n")
                turns.flush()
                played += 1
            ended = "completed" if played == turns_per_call else "cut"
            record |= {"ended": ended, "turns": played, "errors": 0}
            if ended == "cut":
                record["reason"] = "SIGINT"
        counts[kind] += 1
        attempts.write(json.dumps(record | {"ended_at": time.time()}) + "\n")
        attempts.flush()
        number += 1
        time.sleep(gap)
summary = {"attempts": sum(counts.values()), "outcomes": counts, "turns": 0, "errors": 0}
(out / "summary.json").write_text(json.dumps(summary))
