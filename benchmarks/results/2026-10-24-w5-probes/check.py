"""Recompute W5's make-room probe tables (docs/BENCHMARKS.md, and the GPU numbers of vLLM #63350 and #63351) from
the outputs in this directory, and check them: each block's engine came up (its driver's log has its ready line), and
each tail run's summary line agrees with its per-turn lines (within their 1 ms rounding) by the exclusive method tail.py
used when these runs were made. The tables' percentiles are inclusive, as E2's report computes them. Prints each block's
rows; exits 1 if a check fails. python3 check.py, from anywhere."""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DASH = "\u2013"  # the docs' range dash

BLOCKS = {"gpu-check": ("final", "v03", "loadhold", "final2"), "pr-probes": ("stock", "a", "ab")}
"""Each directory's blocks, as its driver's log names them."""

PROBE = re.compile(
    r"^(kv|prefill): priority-0 TTFT (\d+) ms, 16 tokens (?:(\d+) ms|\u2014), for (\d+) prompt tokens, sent with (\d+) "
    r"priority-2 requests of (\d+) tokens running and (\d+) of (\d+) blocks free; after it, (\d+ running and "
    r"\d+ waiting)$"
)
TURN = re.compile(r" call \d (?:ttft +(\d+) ms|failed)$")
SUMMARY = re.compile(
    r"^calls: (\d+) turns, (\d+) failed \(counted at 30,000 ms\), TTFT p50 (\d+) p90 (\d+) p95 (\d+) max (\d+) ms; "
    r"agents: (\d+) of (\d+) stopped early$"
)
TIMEOUT_MS = 30_000.0  # tail.py's: a failed turn counts at it


def spread(values: list[int]) -> str:
    """As the tables give a row: each run, or the median and range from four runs on."""
    v = sorted(values)
    if len(v) >= 4:
        return f"{statistics.median(v):,.0f} ms ({v[0]:,}{DASH}{v[-1]:,})"
    return ", ".join(f"{x:,}" for x in v) + " ms"


def tail_numbers(turns: list[float], method: str) -> tuple[float, float, float, float]:
    """p50 / p90 / p95 / max: the median and statistics.quantiles(n=20)'s 18th and 19th cut points by `method`."""
    q = statistics.quantiles(turns, n=20, method=method)
    return statistics.median(turns), q[17], q[18], max(turns)


def main() -> int:
    failures = 0
    for directory, blocks in BLOCKS.items():
        log = (HERE / directory / "run.log").read_text()
        print(f"== {directory}")
        for name in blocks:
            path = HERE / directory
            ready = re.search(rf"\b{name}: engine ready; (.*?)(?:; GPU|$)", log, re.M)
            print(f"  {name}: engine {ready[1] if ready else 'NOT READY'}")
            failures += ready is None
            for case in ("kv", "prefill"):
                lines = (path / f"{name}-{case}.log").read_text().splitlines()
                runs = [m for line in lines if (m := PROBE.match(line))]
                if unread := [line for line in lines if line.strip() and not PROBE.match(line)]:
                    print(f"    {case}: UNREAD LINES {unread}")
                    failures += 1
                ttft = [int(m[2]) for m in runs]
                timed = [(int(m[2]), int(m[3])) for m in runs if m[3]]  # a reply can end before 16 tokens
                free = sorted({int(m[7]) for m in runs})
                gaps = sorted((b - a) / 15 for a, b in timed)
                short = f", {len(runs) - len(timed)} of them ending before it" if len(timed) < len(runs) else ""
                print(
                    f"    {case}: {len(runs)} runs, TTFT {spread(ttft)}, 16th token {spread([b for _, b in timed])}"
                    f"{short} ({gaps[0]:.1f}{DASH}{gaps[-1]:.1f} ms a token); {sorted({int(m[6]) for m in runs})} "
                    f"tokens a request, {free} blocks free at the send; then {sorted({m[9] for m in runs})}"
                )
            for mode in ("fresh", "grow"):
                pooled: list[float] = []
                for run in sorted(path.glob(f"{name}-tail-{mode}-*.log")):
                    text = run.read_text().splitlines()
                    turns = [
                        TIMEOUT_MS if m[1] is None else float(m[1])
                        for line in text
                        if (m := TURN.search(line))
                    ]
                    [summary] = [m for line in text if (m := SUMMARY.match(line))]
                    agrees = int(summary[1]) == len(turns) and all(
                        abs(a - int(b)) <= 1
                        for a, b in zip(tail_numbers(turns, "exclusive"), summary.groups()[2:6], strict=True)
                    )
                    failures += not agrees
                    pooled += turns
                    print(
                        f"    tail {mode} {run.stem[-1]}: {summary[1]} turns, {summary[2]} failed, "
                        f"{' / '.join(f'{x:,.0f}' for x in tail_numbers(turns, 'inclusive'))} ms, {summary[7]} of "
                        f"{summary[8]} agents stopped; its summary line {'agrees' if agrees else 'DISAGREES'}"
                    )
                print(
                    f"    tail {mode}, both runs: {len(pooled)} turns, "
                    f"{' / '.join(f'{x:,.0f}' for x in tail_numbers(pooled, 'inclusive'))} ms (p50 / p90 / p95 / max)"
                )
    print("all checks pass" if not failures else f"{failures} checks FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
