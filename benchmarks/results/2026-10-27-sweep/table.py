"""The sweep's screens' table (README.md) from this record's points.

For each point: its runs, the median of their p95s and the p95 of their replies pooled, the replies over W5's SLO (W5's
voice alone with two callers, the median of its five rounds' p95s, plus 250 ms, from W5's records) and over the
screen's own (its voice alone's p95 plus 250 ms), the agents' tokens a second over the calls' first 240 s and their
generated tokens a second over the calls, and their time to first token at p50 and p99, each the median over the
runs. A reply that never came counts at the reply timeout, as in every V2V percentile.

    cd python/flatshare_bench && uv run python ../../benchmarks/results/2026-10-27-sweep/table.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from flatshare_bench.e2.report import load_points
from flatshare_stats import quantiles

HERE = Path(__file__).resolve().parent
W5 = HERE.parent / "2026-10-24-e2-w5"
SLO_MARGIN_MS = 250.0
SCREENS = ("sweep-screen", "sweep-screen2")


def points(root: Path, campaigns: tuple[str, ...]) -> dict[str, list[dict]]:
    """Each complete point's rows of E2's report under ``root``'s campaigns, by point."""
    dirs = sorted(
        run.parent
        for campaign in campaigns
        for run in (root / campaign).glob("*/run.json")
        if (run.parent / "summary.json").exists()
    )
    rows, _ = load_points(dirs)
    found: dict[str, list[dict]] = {}
    for row in rows:
        row["dir"] = root / row["campaign"] / row["point"]
        found.setdefault(row["point"], []).append(row)
    return found


def w5_slo() -> float:
    """W5's SLO: its voice alone's p95 with two callers, the median over its five rounds, plus the margin."""
    alone = points(W5, tuple(f"w5-r{n}" for n in range(1, 6)))["w5-a1-voice"]
    return round(statistics.median(r["v2v_p95_ms"] for r in alone) + SLO_MARGIN_MS, 1)


def table(root: Path = HERE, campaigns: tuple[str, ...] = SCREENS) -> str:
    by_point = points(root, campaigns)
    w5 = w5_slo()
    alone = [r["v2v_p95_ms"] for r in by_point.get("w5-a1-voice", [])]
    own = round(statistics.median(alone) + SLO_MARGIN_MS, 1)
    lines = [
        f"W5's SLO {w5:,.1f} ms; the screen's own {own:,.1f} ms (its voice alone's p95 "
        f"{statistics.median(alone):,.1f} ms plus {SLO_MARGIN_MS:.0f})",
        "",
        "| Point | Runs | Median run p95, ms | Pooled p95, ms | Over W5's SLO | Over the screen's | "
        "Agents, tokens/s (240 s) | Generated tokens/s | Agents' TTFT p50/p99, s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point, rows in sorted(by_point.items()):
        samples = sorted(s for r in rows for s in r["v2v_samples"])
        n = len(samples)
        over = sum(s > w5 for s in samples)
        over_own = sum(s > own for s in samples)
        median_p95 = statistics.median(r["v2v_p95_ms"] for r in rows)
        head = (
            f"| {point} | {len(rows)} | {median_p95:,.0f} | {quantiles(samples)[94]:,.0f} | "
            f"{over}/{n} ({100 * over / n:.0f}%) | {over_own}/{n} ({100 * over_own / n:.0f}%) |"
        )
        agents = [r for r in rows if r["sessions"]]
        if not agents:
            lines.append(head + " - | - | - |")
            continue
        generated = [
            json.loads((r["dir"] / "summary.json").read_text())["background"][
                "completion_tokens_per_s"
            ]
            for r in agents
        ]
        ttft = (
            statistics.median(r["agent_ttft_p50_ms"] for r in agents) / 1000,
            statistics.median(r["agent_ttft_p99_ms"] for r in agents) / 1000,
        )
        tokens = statistics.median(r["fixed_window_tokens_per_s"] for r in agents)
        lines.append(
            head + f" {tokens:,.0f} | {statistics.median(generated):.1f} | "
            f"{ttft[0]:.0f}/{ttft[1]:.0f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE))
