"""The two-card comparison's verdict (benchmarks/quality/RESOURCE-GATE.md) and the sweep's confirmation, from the
rounds' records (rg-r1 to rg-r5).

The gate, as fixed before the rounds ran:

- each arrangement serves both calls, and qualifies only if, over its runs pooled, at most 5% of the calls' replies
  take longer than the SLO for their card's number of callers: voice alone on that card with as many callers, plus
  250 ms, measured in the same campaign (here, the same round);
- its background measure is the agents' tokens a second over the calls' first 240 s, summed over its cards, the
  median over its runs (rounds), with the agents' time to first token beside it;
- sharing has resource value when a qualifying shared arrangement (Flatshare pooling, or consolidation at the most
  agents one card runs beside both calls while it qualifies) does at least 20% more background work than the best
  qualifying dedicated arrangement; Flatshare's own value is the same comparison against stock pooling and against
  consolidation on stock vLLM.

A dedicated card's agents run at the number of session trees at which that card does the most (the highest median over
the rounds), of those the rounds measured: how many agents a card runs at once is part of the scheduling that gives
them the most, and at 24 its pool thrashes. Consolidation's second card is dedicated roles' best practical card.

Each card's part of an arrangement is a point of its own, measured on one host a round, the hosts swapped from round to
round. A pooled arrangement's two cards run the same point, so its replies are that point's and its agents twice its
own.

    cd python/flatshare_bench && uv run python ../../benchmarks/results/2026-10-28-rounds/gate.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from flatshare_bench.e2.report import load_points
from flatshare_stats import quantiles

HERE = Path(__file__).resolve().parent
ROUNDS = ("rg-r1", "rg-r2", "rg-r3")
SWEEP_ROUNDS = (*ROUNDS, "rg-r4", "rg-r5")
MARGIN_MS = 250.0
WIDER_MARGIN_MS = 500.0
"""Beside the gate, never in its verdict: the envelopes had the SLO allowed twice the margin it fixed."""
REQUIREMENT = 0.05
HURDLE = 0.20
VOICE_ALONE = {2: "w5-a1-voice", 1: "rg-voice-c1"}

AGENTS_ALONE = {
    # The dedicated card's agents at each number of session trees the rounds measured, most first.
    "Dedicated roles, controlled": (
        (24, "rg-disc-agents-frozen-s24"),
        (12, "rg-disc-agents-frozen-s12"),
    ),
    "Dedicated roles, best practical": (
        (24, "rg-disc-agents-only-s24"),
        (20, "rg-disc-agents-only-s20"),
        (16, "rg-disc-agents-only-s16"),
        (14, "rg-disc-agents-only-s14"),
        (12, "rg-disc-agents-only-s12"),
    ),
}
BEST_PRACTICAL = "Dedicated roles, best practical"
SPARE = (
    # Beside the gate: a warm spare's agents, alone on the serving card's engine with its speech servers idle.
    (16, "rg-disc-agents-pr-ab48-b256-s16"),
    (12, "rg-disc-agents-pr-ab48-b256-s12"),
)
POOLING = {
    "Stock pooling": (("rg-disc-frozen-c1-s12", 2),),
    "Flatshare pooling": (("rg-disc-pr-ab48-b256-c1-s12", 2),),
}
CONSOLIDATION = {
    # The consolidated card at each number of session trees the rounds measured, most first.
    "Consolidation, the change at a step cap of 256": (
        (24, "sweep-disc-pr-ab48-b256-t0-s24"),
        (12, "rg-disc-pr-ab48-b256-s12"),
        (8, "rg-disc-pr-ab48-b256-s8"),
        (4, "rg-disc-pr-ab48-b256-s4"),
        (2, "rg-disc-pr-ab48-b256-s2"),
    ),
    "Consolidation, stock vLLM": (
        (24, "w5-disc-frozen-s24"),
        (12, "rg-disc-frozen-s12"),
        (8, "rg-disc-frozen-s8"),
        (4, "rg-disc-frozen-s4"),
        (2, "rg-disc-frozen-s2"),
    ),
    "Consolidation, the change at W5's configuration": ((24, "w5-disc-pr-ab48-s24"),),
}
SWEEP = ("w5-disc-frozen-s24", "w5-disc-pr-ab48-s24", "sweep-disc-pr-ab48-b256-t0-s24")


def rows(root: Path, campaigns: tuple[str, ...]) -> dict[tuple[str, str], dict]:
    """Each complete point's row of E2's report, by (campaign, point)."""
    dirs = sorted(
        run.parent
        for campaign in campaigns
        for run in (root / campaign).glob("*/run.json")
        if (run.parent / "summary.json").exists()
    )
    found, _ = load_points(dirs)
    return {(r["campaign"], r["point"]): r for r in found}


def nodes(root: Path, campaign: str) -> str:
    """The GPU node a round ran on, as its points' run.json name it (versions.gpu_source)."""
    found = {
        node
        for run in (root / campaign).glob("*/run.json")
        for node in json.loads(run.read_text())["versions"].get("gpu_source", {})
    }
    return ", ".join(sorted(found)) or "?"


def slos(
    points: dict, campaigns: tuple[str, ...], margin: float = MARGIN_MS
) -> dict[tuple[str, int], float]:
    """Each round's SLO by its number of callers: its voice alone's p95 plus the margin."""
    return {
        (c, callers): points[c, alone]["v2v_p95_ms"] + margin
        for c in campaigns
        for callers, alone in VOICE_ALONE.items()
        if (c, alone) in points
    }


def foreground(points: dict, slo: dict, cards, campaigns) -> tuple[int, int]:
    """The replies over their round's SLO, and the replies, over the cards with calls, pooled over the rounds."""
    over = total = 0
    for c in campaigns:
        for point, _ in cards:
            row = points.get((c, point))
            if row is None or not row["callers"]:
                continue
            threshold = slo[c, row["callers"]]
            over += sum(s > threshold for s in row["v2v_samples"])
            total += len(row["v2v_samples"])
    return over, total


def background(points: dict, cards, campaigns) -> tuple[float | None, list[float]]:
    """The agents' tokens a second over the first 240 s summed over the cards, the median over the rounds that
    measured every card, and each such round's sum."""
    sums = []
    for c in campaigns:
        if all((c, point) in points for point, _ in cards):
            sums.append(
                sum(
                    n * points[c, point]["fixed_window_tokens_per_s"]
                    for point, n in cards
                    if points[c, point]["sessions"]
                )
            )
    return (statistics.median(sums) if sums else None), sums


def busiest(points: dict, sizes, campaigns) -> tuple[int, str] | None:
    """Of an agents-alone card's numbers of session trees (most first), the one at which it does the most: the highest
    median over the rounds of its agents' tokens a second, the most trees on a tie."""
    found = []
    for sessions, point in sizes:
        rates = [
            points[c, point]["fixed_window_tokens_per_s"] for c in campaigns if (c, point) in points
        ]
        if rates:
            found.append((statistics.median(rates), sessions, point))
    if not found:
        return None
    _, sessions, point = max(found, key=lambda f: f[0])
    return sessions, point


def envelope(points: dict, slo: dict, sizes, campaigns) -> tuple[int, str] | None:
    """Of a consolidated card's numbers of session trees beside both calls (most first), the most at which it meets
    the foreground requirement over its rounds pooled: the profile's envelope, as far as the rounds measured it."""
    for sessions, point in sizes:
        over, total = foreground(points, slo, ((point, 1),), campaigns)
        if total and over <= REQUIREMENT * total:
            return sessions, point
    return None


def ttft(points: dict, cards, campaigns) -> str:
    agents = [points[c, p] for c in campaigns for p, _ in cards if (c, p) in points]
    agents = [r for r in agents if r["sessions"]]
    if not agents:
        return "-"
    p50 = statistics.median(r["agent_ttft_p50_ms"] for r in agents) / 1000
    p99 = statistics.median(r["agent_ttft_p99_ms"] for r in agents) / 1000
    return f"{p50:.1f}/{p99:.1f}"


def gate(root: Path = HERE) -> str:
    points = rows(root, ROUNDS)
    slo = slos(points, ROUNDS)
    lines = ["Each round's SLO, ms (voice alone's p95 plus 250):", ""]
    lines += ["| Round | Node | Two callers | One caller |", "|---|---|---:|---:|"]
    for c in ROUNDS:
        node = nodes(root, c)
        two, one = slo.get((c, 2)), slo.get((c, 1))
        lines.append(
            f"| {c} | {node} | {two:,.1f} | {one:,.1f} |"
            if two and one
            else f"| {c} | {node} | - | - |"
        )
    chosen = {name: busiest(points, sizes, ROUNDS) for name, sizes in AGENTS_ALONE.items()}
    lines += [
        "",
        "The agents alone on a dedicated card, by session trees; each arrangement runs the card at its busiest:",
        "",
        "| Card | Session trees | Agents, tokens/s (240 s): median | Rounds | Agents' TTFT p50/p99, s |",
        "|---|---:|---:|---|---:|",
    ]
    for name, sizes in AGENTS_ALONE.items():
        for sessions, point in sizes:
            median, sums = background(points, ((point, 1),), ROUNDS)
            if median is None:
                continue
            busiest_here = " (busiest)" if chosen[name] == (sessions, point) else ""
            lines.append(
                f"| {name.removeprefix('Dedicated roles, ')}: {point} | {sessions}{busiest_here} | {median:,.0f} | "
                f"{', '.join(f'{s:,.0f}' for s in sums)} | {ttft(points, ((point, 1),), ROUNDS)} |"
            )
    lines += [
        "",
        "| Arrangement | Cards | Replies over the SLO | Qualifies | Agents, tokens/s (240 s): median | Rounds | "
        "Agents' TTFT p50/p99, s |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    measured: dict[str, tuple[bool, float | None]] = {}

    def add(name: str, cards) -> None:
        over, total = foreground(points, slo, cards, ROUNDS)
        qualifies = total > 0 and over <= REQUIREMENT * total
        median, sums = background(points, cards, ROUNDS)
        measured[name] = (qualifies, median)
        shown = " + ".join(f"{n} x {p}" if n > 1 else p for p, n in cards)
        share = f"{over}/{total} ({100 * over / total:.1f}%)" if total else "-"
        lines.append(
            f"| {name} | {shown} | {share} | {'yes' if qualifies else 'no'} | "
            f"{median:,.0f} | {', '.join(f'{s:,.0f}' for s in sums)} | {ttft(points, cards, ROUNDS)} |"
            if median is not None
            else f"| {name} | {shown} | {share} | {'yes' if qualifies else 'no'} | - | - | - |"
        )

    for name, card in chosen.items():
        if card is not None:
            add(f"{name}, {card[0]} session trees", (("w5-a1-voice", 1), (card[1], 1)))
    for name, cards in POOLING.items():
        add(name, cards)
    second = chosen[BEST_PRACTICAL]
    within: dict[str, str | None] = {}
    envelopes: dict[str, tuple[int, str] | None] = {}
    for name, sizes in CONSOLIDATION.items():
        for sessions, point in sizes:
            if second is not None:
                add(f"{name}, {sessions} session trees", ((point, 1), (second[1], 1)))
        envelopes[name] = envelope(points, slo, sizes, ROUNDS)
        found = envelopes[name]
        within[name] = f"{name}, {found[0]} session trees" if found and second is not None else None

    def best(names) -> tuple[str, float] | None:
        found = [
            (n, measured[n][1]) for n in names if measured[n][0] and measured[n][1] is not None
        ]
        return max(found, key=lambda x: x[1]) if found else None

    def versus(name: str | None, against: tuple[str, float] | None) -> str:
        if name is None or not measured[name][0]:
            return "does not qualify"
        value = measured[name][1]
        if against is None:
            return f"qualifies, {value:,.0f} tokens/s, where its counterpart does not"
        gain = value / against[1] - 1
        verdict = "meets" if gain >= HURDLE else "does not meet"
        return f"{gain:+.1%} against {against[0]} ({against[1]:,.0f}): {verdict} the 20% hurdle"

    dedicated = best(
        f"{name}, {card[0]} session trees" for name, card in chosen.items() if card is not None
    )
    lines += [
        "",
        f"The best qualifying dedicated arrangement: {dedicated[0]}, {dedicated[1]:,.0f} tokens/s."
        if dedicated
        else "No dedicated arrangement qualifies.",
    ]
    change = within["Consolidation, the change at a step cap of 256"]
    stock = within["Consolidation, stock vLLM"]
    lines += [
        "",
        "Sharing's resource value:",
        f"- Flatshare pooling: {versus('Flatshare pooling', dedicated)}",
        f"- Consolidation on the change, at its envelope ({change or 'none'}): {versus(change, dedicated)}",
        "- Beside the criterion, whose consolidated card runs the serving profile: consolidation on stock vLLM, at its "
        f"envelope ({stock or 'none'}): {versus(stock, dedicated)}",
        "",
        "Flatshare's own value:",
        f"- Flatshare pooling against stock pooling: "
        f"{versus('Flatshare pooling', best(['Stock pooling']))}",
        f"- Consolidation on the change against consolidation on stock vLLM ({stock or 'none qualifies'}): "
        f"{versus(change, best([stock]) if stock else None)}",
    ]
    spare = busiest(points, SPARE, ROUNDS)
    if spare is not None:
        lines += [
            "",
            "Beside the gate, which counts no spare as a resource benefit: the second card as a warm spare, on the "
            "serving card's engine with its speech servers idle, so that it can take the calls (busiest at "
            f"{spare[0]} session trees, {spare[1]}):",
        ]
        spared = {
            "Dedicated roles with a warm spare": (("w5-a1-voice", 1), (spare[1], 1)),
        }
        found = envelopes["Consolidation, the change at a step cap of 256"]
        if found is not None:
            spared[
                f"Consolidation on the change at {found[0]} session trees, with a warm spare"
            ] = (
                (found[1], 1),
                (spare[1], 1),
            )
        for name, cards in spared.items():
            over, total = foreground(points, slo, cards, ROUNDS)
            median, sums = background(points, cards, ROUNDS)
            if median is None:
                continue
            versus_dedicated = (
                f", {median / dedicated[1] - 1:+.1%} against {dedicated[0]}" if dedicated else ""
            )
            lines.append(
                f"- {name} ({' + '.join(p for p, _ in cards)}): {over}/{total} replies over the SLO, "
                f"{median:,.0f} tokens/s ({', '.join(f'{s:,.0f}' for s in sums)}){versus_dedicated}, agents' "
                f"TTFT p50/p99 {ttft(points, cards, ROUNDS)} s"
            )
    return "\n".join(lines)


def margins(root: Path = HERE) -> str:
    """Beside the gate: each consolidated card's replies over the SLO by its number of session trees, its rounds
    pooled, with the gate's 250 ms margin and with 500, and the envelope each margin gives."""
    points = rows(root, ROUNDS)
    by_margin = {m: slos(points, ROUNDS, m) for m in (MARGIN_MS, WIDER_MARGIN_MS)}
    names = ("Consolidation, the change at a step cap of 256", "Consolidation, stock vLLM")
    heads = [f"{n.removeprefix('Consolidation, ')}, +{m:.0f} ms" for n in names for m in by_margin]
    lines = [
        "| Session trees | " + " | ".join(heads) + " |",
        "|---:|" + "---:|" * len(heads),
    ]
    sizes = {name: dict(CONSOLIDATION[name]) for name in names}
    for sessions in sorted({s for name in names for s in sizes[name]}, reverse=True):
        cells = []
        for name in names:
            for slo in by_margin.values():
                point = sizes[name].get(sessions)
                over, total = foreground(points, slo, ((point, 1),), ROUNDS) if point else (0, 0)
                cells.append(f"{over}/{total} ({100 * over / total:.1f}%)" if total else "-")
        lines.append(f"| {sessions} | " + " | ".join(cells) + " |")
    found = []
    for name in names:
        for margin, slo in by_margin.items():
            within = envelope(points, slo, CONSOLIDATION[name], ROUNDS)
            found.append(
                f"{name.removeprefix('Consolidation, ')}, +{margin:.0f} ms: "
                + (f"{within[0]} session trees" if within else "none measured")
            )
    lines += [
        "",
        "The envelope, the most session trees whose rounds pooled qualify: "
        + "; ".join(found)
        + ".",
    ]
    return "\n".join(lines)


def pace(root: Path = HERE) -> str:
    """Beside the gate: the consolidated card's agents' requests that completed in the calls' first 240 s, by round, on
    either engine. Where both engines complete the same requests, the card's part of the background measure is the
    sessions' own pace, which the traces' think times set, not what the card could do."""
    names = ("Consolidation, the change at a step cap of 256", "Consolidation, stock vLLM")
    sizes = {name: dict(CONSOLIDATION[name]) for name in names}
    heads = [f"{n.removeprefix('Consolidation, ')}: requests by round" for n in names]
    lines = ["| Session trees | " + " | ".join(heads) + " |", "|---:|" + "---:|" * len(heads)]
    for sessions in sorted({s for name in names for s in sizes[name]}, reverse=True):
        cells = []
        for name in names:
            found = []
            for c in ROUNDS:
                summary = root / c / sizes[name][sessions] / "summary.json"
                window = (
                    json.loads(summary.read_text())["background"]["fixed_window"]
                    if summary.exists()
                    else {}
                )
                found.append(f"{window['requests']:,}" if "requests" in window else "-")
            cells.append(", ".join(found))
        lines.append(f"| {sessions} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


STAGES = (
    ("transcript_ms", "p50", "Transcript"),
    ("llm_ttft_ms", "p50", "LLM first token"),
    ("llm_ttft_ms", "p95", "LLM first token, p95"),
    ("llm_first_sentence_ms", "p50", "First sentence"),
    ("llm_first_sentence_ms", "p95", "First sentence, p95"),
    ("tts_first_audio_ms", "p50", "TTS first audio"),
)
"""The voice worker's stage timings (summary.json's worker.waterfall), each a percentile of a point's turns: the
transcript from the end of the caller's speech, the LLM's first token and first sentence from its request, and TTS's
first audio from that sentence."""


def stages(root: Path = HERE) -> str:
    """Beside the gate: where a consolidated card's replies spend their time, as the voice worker times its stages,
    each the median over the rounds of the points' percentile, for voice alone and for each engine by session trees."""
    names = ("Consolidation, the change at a step cap of 256", "Consolidation, stock vLLM")
    cards = [("Voice alone", "w5-a1-voice")] + [
        (f"{name.removeprefix('Consolidation, ')}, {sessions} trees", point)
        for name in names
        for sessions, point in sorted(CONSOLIDATION[name])
    ]
    lines = [
        "| Card | " + " | ".join(head for *_, head in STAGES) + " |",
        "|---|" + "---:|" * len(STAGES),
    ]
    for label, point in cards:
        waterfalls = [
            json.loads(summary.read_text())["worker"]["waterfall"]
            for campaign in ROUNDS
            if (summary := root / campaign / point / "summary.json").exists()
        ]
        if not waterfalls:
            continue
        cells = []
        for stage, quantile, _ in STAGES:
            values = [w[stage][quantile] for w in waterfalls if stage in w]
            cells.append(f"{statistics.median(values):,.0f}" if values else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sweep(root: Path = HERE) -> str:
    """The sweep's confirmation over its five rounds: each configuration at two calls beside 24 session trees, its
    replies over its round's SLO pooled, its median and pooled p95, and its agents' token rate."""
    points = rows(root, SWEEP_ROUNDS)
    slo = slos(points, SWEEP_ROUNDS)
    lines = [
        "| Configuration | Runs | Median run p95, ms | Pooled p95, ms | Replies over the SLO | "
        "Agents, tokens/s (240 s): median | Agents' TTFT p50/p99, s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for point in SWEEP:
        runs = [points[c, point] for c in SWEEP_ROUNDS if (c, point) in points]
        if not runs:
            continue
        over, total = foreground(points, slo, ((point, 1),), SWEEP_ROUNDS)
        samples = sorted(s for r in runs for s in r["v2v_samples"])
        lines.append(
            f"| {point} | {len(runs)} | {statistics.median(r['v2v_p95_ms'] for r in runs):,.0f} | "
            f"{quantiles(samples)[94]:,.0f} | {over}/{total} ({100 * over / total:.1f}%) | "
            f"{statistics.median(r['fixed_window_tokens_per_s'] for r in runs):,.0f} | "
            f"{ttft(points, ((point, 1),), SWEEP_ROUNDS)} |"
        )
    lines += [
        "",
        "Replies over the round's SLO, by round:",
        "",
        "| Configuration | " + " | ".join(SWEEP_ROUNDS) + " |",
        "|---|" + "---:|" * len(SWEEP_ROUNDS),
        "| The round's SLO, ms | "
        + " | ".join(f"{slo[c, 2]:,.1f}" if (c, 2) in slo else "-" for c in SWEEP_ROUNDS)
        + " |",
    ]
    for point in SWEEP:
        cells = []
        for c in SWEEP_ROUNDS:
            over, total = foreground(points, slo, ((point, 1),), (c,))
            cells.append(f"{over}/{total}" if total else "-")
        lines.append(f"| {point} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    print(gate(root))
    print()
    print(margins(root))
    print()
    print(pace(root))
    print()
    print(stages(root))
    print()
    print(sweep(root))
