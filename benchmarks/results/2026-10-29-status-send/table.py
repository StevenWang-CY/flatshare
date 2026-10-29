"""The two back-to-back runs' table, which README.md gives and python/flatshare_bench/tests/test_status_send_record.py
holds to the records: each run's calls as the soak tool counts them (its summary.json), and its refusals by what
LiveKit knew of the worker (worker-events-RUN.tsv, the voice worker's journal as the soak record keeps it): with both
lines busy, or with a line free, LiveKit holding the worker full by the last status it had been sent while a call's
process had exited since."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNS = {"before": "v1.7: its status every 2.5 s", "after": "v1.8: and as each call's process closes"}


def attempts(run: str) -> list[dict[str, Any]]:
    lines = (HERE / run / "attempts.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def events(run: str) -> list[list[str]]:
    return [line.split("\t") for line in (HERE / f"worker-events-{run}.tsv").read_text().splitlines()[1:]]


def line_free(attempt: dict[str, Any], events: list[list[str]]) -> tuple[float, float | None] | None:
    """For a call LiveKit turned away while the worker had a line free, the seconds since a call's process had exited
    and until the worker sent "available"; None for any other. A line was free when the worker's gate was open (its
    status file's reason "N of M calls"), the last status it had sent LiveKit was full and a call's process had exited
    since: LiveKit dispatches a worker it holds full no call, so its calls can only fall from that status on."""
    worker = attempt.get("worker") or {}
    if attempt["outcome"] != "refused" or not re.fullmatch(r"\d+ of \d+ calls", str(worker.get("reason"))):
        return None
    mine = [(float(at), event) for at, pid, event, _ in events if pid == str(worker["pid"])]
    sent = [(at, event) for at, event in mine if event != "exit" and at <= attempt["started"]]
    exits = [at for at, event in mine if event == "exit" and sent and sent[-1][0] < at < attempt["started"]]
    if not sent or sent[-1][1] != "full" or not exits:
        return None
    after = [at for at, event in mine if event != "exit" and at > attempt["started"]]
    return attempt["started"] - exits[0], (after[0] - attempt["started"]) if after else None


def counts(run: str) -> dict[str, int]:
    calls = json.loads((HERE / run / "summary.json").read_text())["overall"]["calls"]
    run_events = events(run)
    free = sum(1 for a in attempts(run) if line_free(a, run_events) is not None)
    return {
        "attempted": calls["attempted"],
        "accepted": calls["accepted"],
        "busy": calls["refused"] - free,
        "free": free,
        "failed": calls["failed"],
    }


def table() -> str:
    rows = [
        "| Run | The worker sends LiveKit | Calls | Accepted | Refused, both lines busy | Refused, a line free | Failed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run, sends in RUNS.items():
        c = counts(run)
        rows.append(
            f"| `{run}` | {sends} | {c['attempted']:,} | {c['accepted']:,} | {c['busy']:,} | {c['free']:,} "
            f"| {c['failed']:,} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    print(table())
