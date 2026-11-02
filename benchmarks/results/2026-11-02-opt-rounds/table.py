"""The optimization rounds' tables (README.md), from this record's points: rounds opt-r1 to opt-r5, odd ones on
Phastform and even ones on Gray, each a fresh sample of the agents.

Each scenario pairs v1.14's configuration to deploy (the change at the serving profile's step cap of 256) with the same
plus the program's optimizations (the calls' prefixes kept on the GPU, the next batch just in time, each turn announced,
the calls taking the agents' reservations), in the same round. A reply is over the SLO when its voice-to-voice time
exceeds that round's voice alone with as many callers, plus 250 ms (benchmarks/quality/RESOURCE-GATE.md's rule). A
row's figures are the median over the rounds, with their range; the replies over the SLO are pooled over the rounds.

    cd python/flatshare_bench && uv run python ../../benchmarks/results/2026-11-02-opt-rounds/table.py NAME
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

from flatshare_bench.e2.report import load_points

HERE = Path(__file__).resolve().parent
ROUNDS = ("opt-r1", "opt-r2", "opt-r3", "opt-r4", "opt-r5")
MARGIN_MS = 250.0
VOICE_ALONE = {2: "w5-a1-voice", 4: "w5-a1-voice-c4", 6: "w5-a1-voice-c6"}
SCENARIOS = (
    # (the scenario, v1.14's point, the first four additions' point)
    ("24 agent sessions, 2 callers", "sweep-disc-pr-ab48-b256-t0-s24", "opt-disc-ret-s24"),
    ("24 agent sessions, 4 callers", "opt-disc-b256-c4-s24", "opt-disc-ret-c4-s24"),
    ("24 agent sessions, 6 callers", "opt-disc-b256-c6-s24", "opt-disc-ret-c6-s24"),
    ("16 agent sessions of the whole corpus, 2 callers", "opt-std-b256-s16", "opt-std-ret-s16"),
)
CONFIGURATIONS = ("v1.14", "The first four additions")
STAGES = (
    ("transcript_ms", "Transcript"),
    ("llm_ttft_ms", "LLM first token"),
    ("llm_first_sentence_ms", "LLM first sentence"),
    ("tts_first_audio_ms", "TTS first audio"),
)
FOLLOW_ROUNDS = ("opt-s1", "opt-s2")
"""The follow-up rounds, Phastform then Gray, each a fresh sample of the agents (seeds 36 and 37)."""
FOLLOW = (
    # (the scenario, the first four additions' point, + the decode hold, + the hold and speculative decoding)
    ("24 agent sessions, 2 callers", "opt-disc-ret-s24", "opt-disc-hold-s24", "opt-disc-spec-s24"),
    ("24 agent sessions, 4 callers", "opt-disc-ret-c4-s24", "opt-disc-hold-c4-s24", "opt-disc-spec-c4-s24"),
    ("24 agent sessions, 6 callers", "opt-disc-ret-c6-s24", "opt-disc-hold-c6-s24", "opt-disc-spec-c6-s24"),
)
FOLLOW_CONFIGURATIONS = ("The first four additions", "+ the decode hold", "+ the hold and speculation")
TABLES = ("slo", "voice", "agents", "stages", "follow", "follow-agents")


def points(rounds: tuple[str, ...] = ROUNDS) -> dict[tuple[str, str], dict]:
    """Each complete point's row of E2's report, by (round, point)."""
    dirs = sorted(
        run.parent
        for name in rounds
        for run in (HERE / name).glob("*/run.json")
        if (run.parent / "summary.json").exists()
    )
    found, _ = load_points(dirs)
    return {(r["campaign"], r["point"]): r for r in found}


def ran(rows: dict, rounds: tuple[str, ...] = ROUNDS) -> tuple[str, ...]:
    return tuple(name for name in rounds if any(key[0] == name for key in rows))


def spread(values: list[float], fmt: str = "{:,.0f}") -> str:
    """The median of values and, for more than one, their range: an en dash between two ends, or "to" where an end
    is signed, so that a range of negatives reads "\u22124.3% to \u22120.5%"; a minus sign, not a hyphen."""
    if not values:
        return "\u2014"
    def show(value: float) -> str:
        return fmt.format(value).replace("-", "\u2212")
    mid = show(statistics.median(values))
    if len(values) == 1:
        return mid
    between = " to " if "+" in fmt else "\u2013"
    return f"{mid} ({show(min(values))}{between}{show(max(values))})"


def slo_ms(rows: dict, name: str, callers: int) -> float | None:
    alone = rows.get((name, VOICE_ALONE[callers]))
    return None if alone is None else alone["v2v_p95_ms"] + MARGIN_MS


def by_round(point: str, rows: dict | None = None) -> list[tuple[int, int, int, float]]:
    """A point's rounds as (round, replies over that round's SLO, replies, the agents' tokens a second over the calls'
    first 240 s), in the rounds' order: the report's figure of the rounds reads these."""
    rows = points() if rows is None else rows
    found = []
    for name in ran(rows):
        row = rows.get((name, point))
        threshold = None if row is None else slo_ms(rows, name, row["callers"])
        if threshold is None:
            continue
        over = sum(s > threshold for s in row["v2v_samples"])
        found.append((int(name.removeprefix("opt-r")), over, len(row["v2v_samples"]), row["background_tokens_per_s"]))
    return found


def slo() -> str:
    """Each scenario's replies over their round's SLO, pooled, and the V2V p95 against the SLO."""
    rows = points()
    out = [
        "| Scenario | Configuration | Rounds | Replies over the SLO | V2V p50, ms | V2V p95 | The SLO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, *pair in SCENARIOS:
        for config, point in zip(CONFIGURATIONS, pair, strict=True):
            over = total = 0
            p50, p95, slos = [], [], []
            for name in ran(rows):
                row = rows.get((name, point))
                if row is None:
                    continue
                threshold = slo_ms(rows, name, row["callers"])
                if threshold is None:
                    continue
                over += sum(s > threshold for s in row["v2v_samples"])
                total += len(row["v2v_samples"])
                p50.append(row["v2v_p50_ms"])
                p95.append(row["v2v_p95_ms"])
                slos.append(threshold)
            share = f"{over} of {total} ({100 * over / total:.1f}%)" if total else "\u2014"
            out.append(
                f"| {scenario} | {config} | {len(p95)} | {share} | {spread(p50)} | {spread(p95)} | {spread(slos)} |"
            )
    return "\n".join(out)


def voice() -> str:
    """The voice turn's LLM part: its first token and first sentence, p50 and p95."""
    rows = points()
    out = [
        "| Scenario | Configuration | LLM first token p50, ms | p95 | First sentence p50 | p95 | E2E p95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, *pair in SCENARIOS:
        for config, point in zip(CONFIGURATIONS, pair, strict=True):
            got = [rows[name, point] for name in ran(rows) if (name, point) in rows]
            out.append(
                f"| {scenario} | {config} | {spread([r['llm_ttft_p50_ms'] for r in got])} | "
                f"{spread([r['llm_ttft_p95_ms'] for r in got])} | "
                f"{spread([r['llm_first_sentence_p50_ms'] for r in got])} | "
                f"{spread([r['llm_first_sentence_p95_ms'] for r in got])} | "
                f"{spread([r['e2e_p95_ms'] for r in got])} |"
            )
    return "\n".join(out)


def agents() -> str:
    """The agents' work beside the calls: tokens a second over the calls' first 240 s, and their time to first token."""
    rows = points()
    out = [
        "| Scenario | Configuration | Agents' tokens/s | Against v1.14 | Agents' TTFT p50, s | p99 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario, before, after in SCENARIOS:
        base = {name: rows[name, before] for name in ran(rows) if (name, before) in rows}
        for config, point in zip(CONFIGURATIONS, (before, after), strict=True):
            got = {name: rows[name, point] for name in ran(rows) if (name, point) in rows}
            ratios = [
                got[name]["background_tokens_per_s"] / base[name]["background_tokens_per_s"] - 1
                for name in got
                if name in base and point == after
            ]
            out.append(
                f"| {scenario} | {config} | {spread([r['background_tokens_per_s'] for r in got.values()])} | "
                f"{spread([100 * x for x in ratios], '{:+.1f}%') if ratios else '\u2014'} | "
                f"{spread([r['agent_ttft_p50_ms'] / 1e3 for r in got.values()], '{:.1f}')} | "
                f"{spread([r['agent_ttft_p99_ms'] / 1e3 for r in got.values()], '{:.1f}')} |"
            )
    return "\n".join(out)


def stages() -> str:
    """The voice turn's mean stage times, over every reply of each configuration's rounds."""
    rows = points()
    out = [
        "| Scenario | Configuration | " + " | ".join(label for _, label in STAGES) + " |",
        "|---|---|" + "---:|" * len(STAGES),
    ]
    for scenario, *pair in SCENARIOS:
        for config, point in zip(CONFIGURATIONS, pair, strict=True):
            got = [rows[name, point] for name in ran(rows) if (name, point) in rows]
            cells = []
            for key, _ in STAGES:
                pairs = [r["stage_means"][key] for r in got if key in r["stage_means"]]
                n = sum(count for count, _ in pairs)
                cells.append(f"{sum(count * mean for count, mean in pairs) / n:,.0f}" if n else "\u2014")
            out.append(f"| {scenario} | {config} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def followed() -> list[tuple[str, str, list[dict], str, list[float]]]:
    """The follow-up rounds: the first four additions, with the decode hold, and with speculative decoding on top. Each
    scenario and configuration: its rounds' rows, its replies over their round's SLO pooled over the rounds, and the
    agents' token rate against the first four additions' in each round, in percent."""
    rows = points(FOLLOW_ROUNDS)
    found = []
    for scenario, *trio in FOLLOW:
        base = {name: rows[name, trio[0]] for name in ran(rows, FOLLOW_ROUNDS) if (name, trio[0]) in rows}
        for config, point in zip(FOLLOW_CONFIGURATIONS, trio, strict=True):
            over = total = 0
            by_round = {}
            for name in ran(rows, FOLLOW_ROUNDS):
                row = rows.get((name, point))
                threshold = None if row is None else slo_ms(rows, name, row["callers"])
                if threshold is None:
                    continue
                over += sum(s > threshold for s in row["v2v_samples"])
                total += len(row["v2v_samples"])
                by_round[name] = row
            ratios = [
                100 * (row["background_tokens_per_s"] / base[name]["background_tokens_per_s"] - 1)
                for name, row in by_round.items()
                if name in base and point != trio[0]
            ]
            share = f"{over} of {total} ({100 * over / total:.1f}%)" if total else "\u2014"
            found.append((scenario, config, list(by_round.values()), share, ratios))
    return found


def follow() -> str:
    """The follow-up rounds' voice turns: the replies over the SLO, the V2V p95 and the LLM's part."""
    out = [
        "| Scenario | Configuration | Rounds | Replies over the SLO | V2V p95, ms | LLM first token p95 | First sentence"
        " p50 | p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, config, got, share, _ in followed():
        out.append(
            f"| {scenario} | {config} | {len(got)} | {share} | {spread([r['v2v_p95_ms'] for r in got])} | "
            f"{spread([r['llm_ttft_p95_ms'] for r in got])} | "
            f"{spread([r['llm_first_sentence_p50_ms'] for r in got])} | "
            f"{spread([r['llm_first_sentence_p95_ms'] for r in got])} |"
        )
    return "\n".join(out)


def follow_agents() -> str:
    """The follow-up rounds' agents: their tokens a second over the calls' first 240 s, and against the first four
    additions in the same round."""
    out = [
        "| Scenario | Configuration | Agents' tokens/s | Against the first four additions |",
        "|---|---|---:|---:|",
    ]
    for scenario, config, got, _, ratios in followed():
        out.append(
            f"| {scenario} | {config} | {spread([r['background_tokens_per_s'] for r in got])} | "
            f"{spread(ratios, '{:+.1f}%') if ratios else '\u2014'} |"
        )
    return "\n".join(out)


def table(name: str) -> str:
    return {
        "slo": slo,
        "voice": voice,
        "agents": agents,
        "stages": stages,
        "follow": follow,
        "follow-agents": follow_agents,
    }[name]()


RANGED = re.compile(r"^([\u2212+]?[\d,.]+%?) \(([\u2212+]?[\d,.]+%?(?:\u2013| to )[\u2212+]?[\d,.]+%?)\)$")
"""A cell that holds a median and its range: "1,744 (1,709\u20131,800)"."""


def unbroken(span: str) -> str:
    """A range kept on one line: word joiners (U+2060) around its en dash, no-break spaces around its "to"."""
    return span.replace("\u2013", "\u2060\u2013\u2060").replace(" to ", "\u00a0to\u00a0")


def compact(text: str) -> str:
    """A table of text with each median-and-range cell set on two lines, the range under the median in small type:
    the copy the README and BENCHMARKS.md print, where a page-wide table of ranges would wrap each cell over three
    lines. The delimiter row, which opens with no space, stays as it is."""
    rows = []
    for row in text.split("\n"):
        if row.startswith("| ") and row.endswith(" |"):
            cells = (RANGED.sub(lambda m: f"{m[1]}<br><sub>{unbroken(m[2])}</sub>", c) for c in row[2:-2].split(" | "))
            row = f"| {' | '.join(cells)} |"
        rows.append(row)
    return "\n".join(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TABLES:
        sys.exit(f"usage: table.py {'|'.join(TABLES)}")
    print(table(sys.argv[1]))
