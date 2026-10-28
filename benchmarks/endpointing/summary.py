"""The endpointing sweep's table (sweep.sh) from its records: for each setting of Silero VAD's silence and LiveKit's
endpointing delay, the measured turns and those that failed, the caller's voice-to-voice latency, the worker's
endpointing delay (from the end of speech to the end-of-turn decision, its turn lines' ``end_of_turn_ms``), and the
lines whose answer began before they had ended (the caller's ``spoke_during_turn``), named.

    cd python/flatshare_bench && uv run python ../../benchmarks/endpointing/summary.py SWEEP_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from flatshare_stats import percentiles, v2v_samples

TURNS_LOGGER = "flatshare_voice.turns"
REPLY_TIMEOUT_MS = 30_000.0
"""The callers' reply timeout, at which a reply that never came counts in every V2V percentile."""


def setting_rows(sweep: Path) -> list[dict[str, Any]]:
    """Each setting's row, the longest silence and delay first."""
    rows = []
    for here in sorted(p for p in sweep.iterdir() if (p / "setting.json").exists()):
        setting = json.loads((here / "setting.json").read_text())
        turns = [
            json.loads(line) for line in (here / "bots" / "turns.jsonl").read_text().splitlines()
        ]
        measured = [t for t in turns if not t.get("warmup")]
        samples, censored = v2v_samples(measured, REPLY_TIMEOUT_MS)
        worker = []
        for line in (here / "voice-worker.log").read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # The warm-up turn's reply is the session's first turn line.
            if record.get("name") == TURNS_LOGGER and record.get("turn", 0) >= 1:
                worker.append(record)
        rows.append(
            {
                **setting,
                "turns": len(measured),
                "failed": sum(1 for t in measured if t.get("error")),
                "v2v_ms": percentiles(samples, censored),
                "end_of_turn_ms": percentiles(
                    [w["end_of_turn_ms"] for w in worker if "end_of_turn_ms" in w]
                ),
                "cut": sorted(t["clip"] for t in measured if t.get("spoke_during_turn")),
            }
        )
    rows.sort(key=lambda r: (-r["vad_min_silence"], -r["endpointing_min_delay"]))
    return rows


def table(sweep: Path) -> str:
    lines = [
        "| VAD silence, s | Endpointing delay, s | Turns (failed) | V2V p50 / p95, ms | "
        "Endpointing p50 / p95, ms | Lines answered before they ended |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for r in setting_rows(sweep):
        v, e = r["v2v_ms"], r["end_of_turn_ms"]
        cut = f"{len(r['cut'])}: {', '.join(r['cut'])}" if r["cut"] else "0"
        lines.append(
            f"| {r['vad_min_silence']:g} | {r['endpointing_min_delay']:g} | {r['turns']} ({r['failed']}) | "
            f"{v.get('p50', 0):,.0f} / {v.get('p95', 0):,.0f} | {e.get('p50', 0):,.0f} / {e.get('p95', 0):,.0f} | {cut} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(Path(sys.argv[1])))
