"""Recompute the make-room probe tables of docs/BENCHMARKS.md from the outputs in this directory.

Each row names the files (and, in a driver's log, the section) its runs are in. A call-tail row computes the run's
percentiles from its per-turn lines by the inclusive method, as E2's report does, and checks the run's summary line
against them by the exclusive method tail.py used when these runs were made; the per-turn lines are rounded to the
millisecond and the summary is not, so the two may differ by 1 ms.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DASH = "\u2013"  # the docs' range dash

KV = [
    (
        "Stock vLLM 0.32.0, priority scheduling",
        [("logs/gray-makeroom-probe3.log", "probe: frozen box"), ("pr-probes/stock-kv.out", None)],
    ),
    ("make-room, the KV preemption alone (#63350)", [("pr-probes/a-kv.out", None)]),
    (
        "make-room, the KV preemption and the budget cap (#63350 and #63351)",
        [("final-probes/makeroom-kv.out", None)],
    ),
    (
        "Either, on an idle pool",
        [("logs/gray-makeroom.log", None), ("logs/gray-makeroom-probe.log", None)],
    ),
]
PREFILL = [
    (
        "Stock vLLM 0.32.0, priority scheduling",
        [
            ("logs/gray-firsttokens.log", "probe prefill: frozen box"),
            ("pr-probes/stock-prefill.out", None),
        ],
    ),
    (
        "make-room without the first-tokens rule, with the reserve",
        [("logs/gray-firsttokens.log", "first-tokens rule off")],
    ),
    ("make-room, the KV preemption alone (#63350)", [("pr-probes/a-prefill.out", None)]),
    (
        "make-room, the KV preemption and the budget cap (#63350 and #63351)",
        [("pr-probes/ab-prefill.out", None)],
    ),
    ("make-room as arm 9 runs it, with the reserve", [("final-probes/makeroom-prefill.out", None)]),
]
TAIL = [
    ("Stock vLLM 0.32.0, priority scheduling", "fresh", "final-probes/frozen-tail-fresh.out"),
    ("Stock vLLM 0.32.0, priority scheduling", "growing", "final-probes/frozen-tail-grow.out"),
    ("make-room, the KV preemption alone (#63350)", "fresh", "pr-probes/a-tail-fresh.out"),
    ("make-room, the KV preemption alone (#63350)", "growing", "pr-probes/a-tail-grow.out"),
    ("make-room with the reserve, before its KV-load rule", "fresh", "tail/v2b-fresh.out"),
    ("make-room with the reserve, before its KV-load rule", "growing", "tail/tailprobe.out"),
    (
        "make-room, the KV preemption and the budget cap (#63350 and #63351)",
        "fresh",
        "pr-probes/ab-tail-fresh.out",
    ),
    (
        "make-room, the KV preemption and the budget cap (#63350 and #63351)",
        "growing",
        "pr-probes/ab-tail-grow.out",
    ),
    (
        "make-room as arm 9 runs it, with the reserve",
        "fresh",
        "final-probes/makeroom-tail-fresh.out",
    ),
    (
        "make-room as arm 9 runs it, with the reserve",
        "growing",
        "final-probes/makeroom-tail-grow.out",
    ),
]

# probe.py's result line, "[kv: |prefill: ]priority-0 TTFT 247 ms[, 16 tokens 535 ms], for ... free; after it, ...",
# and its first form, "priority-0 TTFT 117 ms for ..."
RUN = re.compile(
    r"^(?:(?:kv|prefill): )?priority-0 TTFT (\d+) ms(?:, 16 tokens (\d+) ms)?,? for \d+ prompt tokens,? "
    r"(?:sent )?with .*? and (\d+) of 2274 blocks free(?:; after it, (\d+ running and \d+ waiting))?"
)
TURN = re.compile(r" call \d ttft +(\S+) ms$")
SUMMARY = re.compile(r"^calls: (\d+) turns, TTFT p50 (\d+) p90 (\d+) p95 (\d+) max (\d+) ms$")


def lines(path: str, section: str | None) -> list[str]:
    """The file's lines, or those of the driver's section whose "[HH:MM:SSZ] ..." heading contains `section`."""
    out, inside = [], section is None
    for line in (HERE / path).read_text().splitlines():
        if section is not None and line.startswith("["):
            inside = section in line
        elif inside:
            out.append(line)
    return out


def runs(sources: list[tuple[str, str | None]]) -> list[tuple[int, int | None, int, str | None]]:
    found = []
    for path, section in sources:
        for line in lines(path, section):
            if m := RUN.match(line):
                found.append((int(m[1]), int(m[2]) if m[2] else None, int(m[3]), m[4]))
    return found


def half_up(x: float) -> int:
    return int(x + 0.5)


def tail(ok: list[float], method: str) -> tuple[int, float, float, float, float]:
    """The sorted turns' count, p50, p90, p95 and max, by the given percentile method."""
    q = statistics.quantiles(ok, n=20, method=method)
    return len(ok), statistics.median(ok), q[17], q[18], ok[-1]


def spread(values: list[int]) -> str:
    """As the tables give a row: each run, or the median and range from four runs on."""
    v = sorted(values)
    if len(v) >= 4:
        return f"{half_up(statistics.median(v)):,} ms ({v[0]:,}{DASH}{v[-1]:,})"
    return ", ".join(f"{x:,}" for x in v) + " ms"


def main() -> int:
    failures = 0
    print("The KV case (a full pool unless the row says otherwise)")
    for name, sources in KV:
        r = runs(sources)
        ttft = sorted(x[0] for x in r)
        free = sorted({x[2] for x in r})
        after = sorted({x[3] for x in r if x[3]})
        shown = f"{ttft[0]}{DASH}{ttft[-1]} ms" if "idle" in name else spread(ttft)
        print(
            f"  {name}: {len(r)} runs, TTFT {shown}; free blocks at the send {free}; then {after or '—'}"
        )
        sixteenth = [x[1] for x in r if x[1] is not None]
        if sixteenth:
            print(f"    16th token: {', '.join(f'{x:,}' for x in sorted(sixteenth))} ms")
    print("The prefill case")
    for name, sources in PREFILL:
        r = runs(sources)
        gaps = sorted((x[1] - x[0]) / 15 for x in r)
        print(
            f"  {name}: {len(r)} runs, TTFT {spread([x[0] for x in r])}, 16th token {spread([x[1] for x in r])}; "
            f"the reply's tokens {gaps[0]:.1f}{DASH}{gaps[-1]:.1f} ms apart"
        )
    print("The calls' first-token tail (p50 / p90 / p95 / max)")
    for name, prompts, path in TAIL:
        text = (HERE / path).read_text().splitlines()
        turns = [float(m[1]) for line in text if (m := TURN.search(line))]
        ok = sorted(t for t in turns if t == t)
        summary = next(
            tuple(int(g) for g in m.groups()) for line in text if (m := SUMMARY.match(line))
        )
        logged = tail(ok, "exclusive")
        agrees = logged[0] == summary[0] and all(
            abs(logged[i] - summary[i]) <= 1 for i in range(1, 5)
        )
        failures += not agrees
        failed = len(turns) - len(ok)
        check = (
            "its summary line agrees"
            if agrees
            else f"its summary line {summary} DISAGREES: {logged}"
        )
        shown = tail(ok, "inclusive")
        print(
            f"  {name}, {prompts}: {shown[0]} turns ({failed} failed), {' / '.join(f'{half_up(x):,}' for x in shown[1:])} ms; {check}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
