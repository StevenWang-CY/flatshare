"""FB's five rounds against W5's, round by round, from the records: round N of each replays the same sample of the
agents (agent seed 10 + N) with the same arm files, so each configuration's V2V p95 and the agents' tokens a second
over the calls' first 240 s compare pairwise. Prints BENCHMARKS' FB table (each configuration's median and range in
both, and the differences round by round), then one row a round and configuration.

python compare.py [FB] [W5]: FB defaults to this directory, W5 to W5's records beside it (../2026-10-24-e2-w5); both
hold CAMPAIGN/POINT/summary.json, as records or as raw run directories."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FB = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
W5 = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "2026-10-24-e2-w5"
POINTS = {
    "w5-a1-voice": "Voice alone",
    "w5-disc-frozen-s24": "The frozen box",
    "w5-disc-pr-ab48-s24": "#63350 + #63351, a 48-token window",
}
ROUNDS = range(1, 6)
DASH, MINUS = "\u2013", "\u2212"  # the docs' range dash and minus sign


def point(path: Path) -> dict | None:
    summary = path / "summary.json"
    if not summary.is_file():
        return None
    data = json.loads(summary.read_text())
    window = (data.get("background") or {}).get("fixed_window") or {}
    return {
        "p95": data["voice"]["v2v_ms"]["p95"],
        "failed": data["voice"]["errors"],
        "tput": window.get("tokens_per_s"),
    }


def spread(values: list[float]) -> str:
    return f"{statistics.median(values):,.0f} ({min(values):,.0f}{DASH}{max(values):,.0f})"


def signed(value: float) -> str:
    return f"{value:+,.0f}".replace("-", MINUS)


rows = {
    name: [(r, point(W5 / f"w5-r{r}" / name), point(FB / f"fb-r{r}" / name)) for r in ROUNDS]
    for name in POINTS
}
print(
    f"| Configuration | Rounds | W5 V2V p95 | FB V2V p95 | FB {MINUS} W5, round by round "
    "| Agents' tokens/s, W5 / FB |"
)
print("|---|---:|---:|---:|---|---:|")
for name, label in POINTS.items():
    pairs = [(w5, fb) for _, w5, fb in rows[name] if w5 and fb]
    if not pairs:
        continue
    w5s, fbs = [w5["p95"] for w5, _ in pairs], [fb["p95"] for _, fb in pairs]
    deltas = ", ".join(signed(fb["p95"] - w5["p95"]) for w5, fb in pairs)
    tputs = [(w5["tput"], fb["tput"]) for w5, fb in pairs if w5["tput"] and fb["tput"]]
    tput = (
        f"{statistics.median(t for t, _ in tputs):,.0f} / {statistics.median(t for _, t in tputs):,.0f}"
        if tputs
        else "—"
    )
    print(f"| {label} | {len(pairs)} | {spread(w5s)} | {spread(fbs)} | {deltas} | {tput} |")

print()
print(
    f"| Round | Configuration | W5 p95 | FB p95 | FB {MINUS} W5 | W5 tokens/s | FB tokens/s "
    "| FB failed turns |"
)
print("|---:|---|---:|---:|---:|---:|---:|---:|")
for name, label in POINTS.items():
    for r, w5, fb in rows[name]:
        if not fb:
            continue
        cells = [
            f"{w5['p95']:,.0f}" if w5 else "—",
            f"{fb['p95']:,.0f}",
            signed(fb["p95"] - w5["p95"]) if w5 else "—",
            f"{w5['tput']:,.0f}" if w5 and w5["tput"] else "—",
            f"{fb['tput']:,.0f}" if fb["tput"] else "—",
            str(fb["failed"]),
        ]
        print(f"| {r} | {label} | " + " | ".join(cells) + " |")
