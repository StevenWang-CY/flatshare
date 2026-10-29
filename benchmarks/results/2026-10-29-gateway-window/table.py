"""The README's tables, from results.json: `python3 table.py` prints them, and the record's test holds the README to them.

A pairs each voice request through the gateway with the same request on the engine's socket, so its differences are per
pair. B is the gateway's own overhead histogram, the difference of two scrapes around A. C is the TTS server's first
audio for three text lengths.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = {"first_ms": "First token", "sentence_ms": "First sentence", "end_ms": "End of the reply"}
TEXTS = {"clause": "A clause", "sentence": "A sentence", "two_sentences": "Two sentences"}
LIMIT_S = 0.002
"""The overhead objective's line: a request over 2 ms in the gateway's own leg spends its budget."""


def results() -> dict:
    return json.loads((HERE / "results.json").read_text())


def overhead(record: dict) -> list[tuple[float, float]]:
    """B's histogram: the cumulative count by upper bound added between the scrapes, summed over the routes."""
    before, after = record["b"]["before"], record["b"]["after"]
    sums: dict[float, float] = {}
    for key, value in after.items():
        if not key.startswith("flatshare_gateway_overhead_seconds_bucket"):
            continue
        le = re.search(r'le="([^"]+)"', key).group(1)
        bound = float("inf") if le == "+Inf" else float(le)
        sums[bound] = sums.get(bound, 0.0) + value - before.get(key, 0.0)
    return sorted(sums.items())


def _ms(value: float, signed: bool = False) -> str:
    return f"{value:+.2f} ms" if signed else f"{value:.1f} ms"


def table() -> str:
    record = results()
    paired = record["a"]["paired_gateway_minus_direct_ms"]
    summary = record["a"]["summary"]
    a = [
        "| Stage | Through the gateway minus direct, median | p95 | Pairs |",
        "|---|---:|---:|---:|",
        *(
            f"| {name} | {_ms(paired[key]['p50'], True)} | {_ms(paired[key]['p95'], True)} | {paired[key]['n']} |"
            for key, name in STAGES.items()
        ),
    ]
    paths = [
        "| Path | First token | First sentence | End of the reply | Gap between tokens |",
        "|---|---:|---:|---:|---:|",
        *(
            f"| {name} | "
            + " | ".join(_ms(summary[path][key]["p50"]) for key in (*STAGES, "gaps_ms"))
            + " |"
            for path, name in (
                ("direct", "Direct, the engine's socket"),
                ("gateway", "Through the gateway"),
            )
        ),
    ]
    buckets = overhead(record)
    total = buckets[-1][1]
    within = next(count for bound, count in buckets if bound >= LIMIT_S)
    p99 = next(bound for bound, count in buckets if count >= 0.99 * total)
    b = [
        "| Requests through the gateway | Over 2 ms in its own leg | The 99th percentile |",
        "|---:|---:|---:|",
        f"| {total:.0f} | {total - within:.0f} | ≤ {p99 * 1000:g} ms |",
    ]
    tts = record["c"]["summary"]
    c = [
        "| Text | Its audio | First audio, median | p95 | Requests |",
        "|---|---:|---:|---:|---:|",
        *(
            f"| {name} | {tts[key]['audio_s']['p50']:.2f} s | {tts[key]['first_audio_ms']['p50']:.0f} ms | "
            f"{tts[key]['first_audio_ms']['p95']:.0f} ms | {tts[key]['first_audio_ms']['n']} |"
            for key, name in TEXTS.items()
        ),
    ]
    return "\n\n".join("\n".join(part) for part in (a, paths, b, c))


if __name__ == "__main__":
    print(table())
