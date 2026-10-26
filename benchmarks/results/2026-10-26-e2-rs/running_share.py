"""RS's five rounds on the box beside the A100 core's five, from the records. Round N of each replays the same sample
of the agents (agent seed 10 + N) with W5's arm files, and every agent point sampled the engine's free KV every
0.5 s, so each configuration's measured running share (the share of samples in which free KV was under two calls'
2,048 tokens while agents ran) compares across the two cards round by round. Prints BENCHMARKS' RS table (each
configuration's running share on both cards, and its V2V p95 and the agents' tokens a second over the calls' first
240 s on the box), then one row a round and configuration.

python running_share.py [RS] [A100]: RS defaults to this directory, A100 to the A100 run's E2 records beside it
(../2026-10-26-a100/e2); both hold CAMPAIGN/POINT/summary.json, as records or as raw run directories."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
A100 = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "2026-10-26-a100" / "e2"
POINTS = {
    "w5-disc-frozen-s24": "The frozen box",
    "w5-disc-pr-ab48-s24": "#63350 + #63351, a 48-token window",
}
ROUNDS = range(1, 6)
DASH = chr(0x2013)  # the docs' range dash


def point(path: Path) -> dict | None:
    summary = path / "summary.json"
    if not summary.is_file():
        return None
    data = json.loads(summary.read_text())
    kv = data.get("kv") or {}
    window = (data.get("background") or {}).get("fixed_window") or {}
    return {
        "share": kv.get("running_share"),
        "samples": kv.get("samples"),
        "p95": data["voice"]["v2v_ms"]["p95"],
        "failed": data["voice"]["errors"],
        "tput": window.get("tokens_per_s"),
    }


def spread(values: list[float], fmt: str, unit: str = "") -> str:
    """The median and the range of ``values``, each formatted with ``fmt`` and followed by ``unit``."""
    if not values:
        return "—"
    middle, low, high = statistics.median(values), min(values), max(values)
    return f"{middle:{fmt}}{unit} ({low:{fmt}}{DASH}{high:{fmt}}{unit})"


rows = {
    name: [(r, point(RS / f"rs-r{r}" / name), point(A100 / f"a100-r{r}" / name)) for r in ROUNDS]
    for name in POINTS
}
print(
    "| Configuration | Rounds | Running share, the box | Running share, the A100 | V2V p95 on the box, ms "
    "| Agents' tokens/s on the box |"
)
print("|---|---:|---:|---:|---:|---:|")
for name, label in POINTS.items():
    box = [b for _, b, _ in rows[name] if b and b["share"] is not None]
    a100 = [a for _, _, a in rows[name] if a and a["share"] is not None]
    print(
        f"| {label} | {len(box)} | {spread([100 * b['share'] for b in box], '.1f', '%')} "
        f"| {spread([100 * a['share'] for a in a100], '.1f', '%')} | {spread([b['p95'] for b in box], ',.0f')} "
        f"| {spread([b['tput'] for b in box if b['tput']], ',.0f')} |"
    )

print()
print(
    "| Round | Configuration | Box share | Box samples | A100 share | Box V2V p95 | Box tokens/s "
    "| Box failed turns |"
)
print("|---:|---|---:|---:|---:|---:|---:|---:|")
for name, label in POINTS.items():
    for r, box, a100 in rows[name]:
        if not box:
            continue
        a100_share = f"{100 * a100['share']:.1f}%" if a100 and a100["share"] is not None else "—"
        tput = f"{box['tput']:,.0f}" if box["tput"] else "—"
        print(
            f"| {r} | {label} | {100 * box['share']:.1f}% | {box['samples']:,} | {a100_share} "
            f"| {box['p95']:,.0f} | {tput} | {box['failed']} |"
        )
