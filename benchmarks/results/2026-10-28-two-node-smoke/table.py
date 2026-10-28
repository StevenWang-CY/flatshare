"""The two-node smokes' table from their records: each smoke's placement idle time, as the gateway's configuration in its
soak.json gives it, beside its summary's calls, replies, realtime service level and background work, and each engine's
prompt tokens a second and requests running.

    cd python/flatshare_bench && uv run python ../../benchmarks/results/2026-10-28-two-node-smoke/table.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SMOKES = ("idle-300", "idle-150")


def row(smoke: Path) -> str:
    record = json.loads((smoke / "soak.json").read_text())
    overall = json.loads((smoke / "summary.json").read_text())["overall"]
    idle = record["serving"]["gateway"]["placement"]["idle_s"]
    calls, turns, gateway = overall["calls"], overall["turns"], overall["gateway"]
    background, engines = gateway["background"], overall["engines"]
    refused = gateway.get("placements_refused", {}).get("no_agent_slot", 0)
    nodes = " / ".join(
        f"{engines[n]['prompt_tokens_per_s']:,.0f}, {engines[n]['running']['mean']:.2f}"
        for n in ("phastform", "gray")
    )
    return (
        f"| {idle:g} s | {calls['attempted']} ({calls['accepted']}, {calls['refused']}, {calls['failed']}) | "
        f"{turns['measured']} ({turns['failed']}) | {turns['replies']['p95']:,.0f} | "
        f"{gateway['realtime_sli']['good_share']:.3f} | {background['offered']:,} / {background['completed']:,} / "
        f"{refused:,} | {background['tokens_per_s']:,.0f} | {nodes} |"
    )


def table(root: Path = HERE) -> str:
    lines = [
        "| Placement `idle_s` | Calls (accepted, refused, failed) | Turns (failed) | Replies p95, ms | Realtime SLI | "
        "Background requests: offered / completed / refused for an agent slot | Background tokens/s | "
        "Phastform / Gray: prompt tokens/s, requests running |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines += [row(root / smoke) for smoke in SMOKES]
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
