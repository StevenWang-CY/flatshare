"""The configurations E2's records measured, and the orders and stages its report and figures share."""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

REPEATED = (
    "background_tokens_per_s",
    "fixed_window_tokens_per_s",
    "v2v_p50_ms",
    "v2v_p95_ms",
    "v2v_p99_ms",
    "agent_ttft_p50_ms",
    "agent_ttft_p99_ms",
    "realtime_first_byte_p95_ms",
    "voice_errors",
)
"""The columns a configuration measured in several campaigns takes the median of over its runs, and
frontier.md gives with their range."""


def configuration(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """What makes the runs of one configuration: its workload, its arm and its arm file's name."""
    return row["workload"], row["arm"], row["point"]


def configurations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The configurations the rows measured, in the order of their first runs: each arm file's runs,
    one per campaign that measured it, with the median of each REPEATED column over the runs that
    measured it (the run's own value for a configuration measured once), and the failed turns and
    the turns of all its runs."""
    runs: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        runs.setdefault(configuration(row), []).append(row)
    return [
        {"workload": workload, "arm": arm, "point": point, "runs": measured}
        | {"frontier": all(r["frontier"] for r in measured)}
        | {column: median(r[column] for r in measured) for column in REPEATED}
        | {"failed_turns": sum(r["voice_errors"] for r in measured)}
        | {"turns": sum(r["voice_turns"] for r in measured)}
        for (workload, arm, point), measured in runs.items()
    ]


def median(values: Iterable[float | None]) -> float | None:
    measured = [v for v in values if v is not None]
    return statistics.median(measured) if measured else None


def arm_order(arm: str) -> tuple[int, str]:
    """Arms in E2's order: 2 before 10, 7a before 7b."""
    number = re.match(r"\d+", arm)
    return (int(number[0]) if number else 1000, arm)


STACKED = (
    ("end_of_turn_ms", "Endpointing"),
    ("turn_to_first_sentence_ms", "Reply to first sentence"),
    ("tts_first_audio_ms", "TTS first audio"),
)
"""The worker's stages that follow one another from the end of the caller's speech, as the stage
waterfall stacks them; the rest of the callers' V2V is transport and playout."""
