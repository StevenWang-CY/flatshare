"""Write e3.tsv: for each W5 point with agents, the engine's steps over the calls and make-room's own counts over them,
which the compact records do not keep. It ran once, on the raw run directories (which keep the engine's metrics at the
calls' start and end) and an export of Phastform's engine journal, where make-room logs its running totals, at most
every 10 s and when they move: python3 e3_counters.py RUNS JOURNAL > e3.tsv, with JOURNAL from
journalctl --user -u lh-engine -o short-unix. A point's count is the totals of the last line at or before the calls'
end, less those of the same engine process's last line at or before their start (none: zero, since make-room logs its
first move at once), so either edge may miss up to 10 s of moves. The frozen box runs no make-room, and v0.3's module
logs no totals: their make-room columns are empty."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

LINE = re.compile(
    r"^(\d+\.\d+) \S+ vllm\[\d+\]: \(EngineCore pid=(\d+)\) INFO .* make-room: budget cap held lower-priority "
    r"prefill back in (\d+) steps \((\d+) prefill tokens and (\d+) admissions, summed over the steps\); "
    r"(\d+) preemptions to make room$"
)
STEPS = "vllm:iteration_tokens_total_count"
COLUMNS = (
    "campaign",
    "point",
    "agents_seed",
    "calls_s",
    "engine_steps",
    "held_steps",
    "held_prefill_tokens",
    "held_admissions",
    "make_room_preemptions",
)


def counter(exposition: Path, name: str) -> float:
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in exposition.read_text().splitlines()
        if line.startswith((name + "{", name + " "))
    )


def main(runs: Path, journal: Path) -> int:
    totals = [
        (float(m[1]), m[2], tuple(int(x) for x in m.groups()[2:]))
        for line in journal.read_text().splitlines()
        if (m := LINE.match(line))
    ]
    print("\t".join(COLUMNS))
    for path in sorted(runs.glob("w5*/*/run.json")):
        run = json.loads(path.read_text())
        if run.get("status") != "complete" or "aiperf" not in run:
            continue  # voice alone has no agents
        start, end = run["window"]["start"], run["window"]["end"]
        metrics = path.parent / "metrics"
        steps = counter(metrics / "engine-end.prom", STEPS) - counter(metrics / "engine-start.prom", STEPS)
        variant = tomllib.loads((path.parent / "arm.toml").read_text()).get("engine", {}).get("variant")
        counts = [""] * 4
        mine = [t for t in totals if run["started"] - 900 <= t[0] <= end]
        if variant == "make-room" and not mine:
            print(f"{path.parent}: no make-room line in the journal", file=sys.stderr)
        elif variant == "make-room":
            pid = mine[-1][1]
            mine = [t for t in mine if t[1] == pid]
            before = [t for t in mine if t[0] <= start]
            counts = [a - b for a, b in zip(mine[-1][2], before[-1][2] if before else (0, 0, 0, 0), strict=True)]
        cells = [path.parent.parent.name, path.parent.name, run["agents_seed"], round(end - start), round(steps)]
        print("\t".join(str(c) for c in [*cells, *counts]))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
