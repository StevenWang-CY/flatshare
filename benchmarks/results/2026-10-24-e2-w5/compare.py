"""W5's E2 tables from these records (docs/BENCHMARKS.md): each configuration's median and range over its five runs,
campaign by campaign; the frozen box and the 48-token window, which several campaigns ran, pooled over them; and the
paired differences by round, all of whose rounds replay the same sample of the agents' traces: the fixed make-room
against v0.3's in W5's campaign, the 48-token window against preempt-resubmit and, in the standard scenario, against the
frozen box in the extension, and a configuration from one campaign to another; and calls per GPU against their SLO,
voice alone's with as many callers, round by round. Then E3's table: what make-room takes from the agents, from each
point's engine counters (summary.json) and make-room's own (e3.tsv, from e3_counters.py). make reproduce-e2-w5 gives the
per-point tables. python3 compare.py."""

from __future__ import annotations

import csv
import functools
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DASH = "\u2013"  # the docs' range dash
CAMPAIGNS = {
    "w5": "W5's campaign (w5-r1 to w5-r5)",
    "w5-pr": "the PR campaign (w5-pr-r1 to w5-pr-r5)",
    "w5-ext": "the extension (w5-ext-r1 to w5-ext-r5)",
    "w5-cap": "calls per GPU (w5-cap-r1 to w5-cap-r3)",
}
POOLED = ("w5-disc-frozen-s24", "w5-disc-pr-ab48-s24")
"""The configurations more than one campaign ran, over the same five samples: their runs pooled."""
METRICS = {
    "V2V p50": ("voice", "v2v_ms", "p50"),
    "V2V p95": ("voice", "v2v_ms", "p95"),
    "V2V p99": ("voice", "v2v_ms", "p99"),
    "realtime first byte p95": ("gateway", "realtime_first_byte_ms", "p95"),
    "agents' tokens/s, first 240 s": ("background", "fixed_window", "tokens_per_s"),
    "agents' tokens/s, whole calls": ("background", "tokens_per_s"),
    "agent TTFT p50": ("background", "ttft_ms", "p50"),
    "agent TTFT p99": ("background", "ttft_ms", "p99"),
}


def value(summary: dict, path: tuple[str, ...]) -> float | None:
    for key in path:
        if not isinstance(summary, dict) or summary.get(key) is None:
            return None
        summary = summary[key]
    return float(summary)


def runs(campaign: str) -> dict[str, list[tuple[int, dict]]]:
    """Each point of the campaign's rounds: (the round's agents seed, its summary.json), in round order."""
    out: dict[str, list[tuple[int, dict]]] = {}
    rounds = sorted(HERE.glob(f"{campaign}-r[0-9]"), key=lambda d: int(d.name.rsplit("-r", 1)[1]))
    for directory in rounds:
        for record in sorted(directory.iterdir()):
            run = json.loads((record / "run.json").read_text())
            assert run["status"] == "complete", record
            seed = run["agents_seed"]  # e2 run records it since W5
            summary = json.loads((record / "summary.json").read_text())
            out.setdefault(record.name, []).append((seed, summary))
    return out


def spread(values: list[float]) -> str:
    return f"{statistics.median(values):,.0f} ({min(values):,.0f}{DASH}{max(values):,.0f})"


def main() -> int:
    data = {campaign: runs(campaign) for campaign in CAMPAIGNS}
    for campaign, title in CAMPAIGNS.items():
        print(f"== {title}: median (range) of the runs")
        for point, points in data[campaign].items():
            failed = sum(int(s["voice"]["errors"]) for _, s in points)
            cells = []
            for name, path in METRICS.items():
                values = [v for _, s in points if (v := value(s, path)) is not None]
                cells.append(f"{name} {spread(values)}" if values else f"{name} -")
            print(f"  {point} ({len(points)} runs, {failed} failed turns): " + "; ".join(cells))
    print("== the SLO by the number of callers: voice alone's median V2V p95 + 250 ms")
    for campaign in CAMPAIGNS:
        for point, points in data[campaign].items():
            if point.startswith("w5-a1-voice"):
                p95 = statistics.median(value(s, METRICS["V2V p95"]) for _, s in points)
                print(f"  {point} ({len(points)} runs): {p95:,.0f} + 250 = {p95 + 250:,.0f} ms")
    print(
        "== calls per GPU against their SLO, round by round: the PRs' 48-token p95 over voice alone's median + 250"
    )
    cap = data.get("w5-cap", {})
    for callers in (4, 6):
        alone, prs = cap.get(f"w5-a1-voice-c{callers}"), cap.get(f"w5-disc-pr-ab48-c{callers}-s24")
        if not alone or not prs:
            print(f"  {callers} callers: not in these records yet")
            continue
        slo = statistics.median(value(s, METRICS["V2V p95"]) for _, s in alone) + 250
        over = [value(s, METRICS["V2V p95"]) - slo for _, s in prs]
        each = ", ".join(f"{d:+,.0f}" for d in over)
        met = sum(d <= 0 for d in over)
        median = statistics.median(over)
        print(
            f"  {callers} callers, the SLO {slo:,.0f} ms: {each}; median {median:+,.0f}; met in {met} of {len(over)}"
        )
    print("== pooled over the campaigns that ran them: median (range)")
    for point in POOLED:
        points = [p for c in CAMPAIGNS for p in data[c].get(point, [])]
        cells = []
        for name, path in METRICS.items():
            values = [v for _, s in points if (v := value(s, path)) is not None]
            cells.append(f"{name} {spread(values)}" if values else f"{name} -")
        print(f"  {point} ({len(points)} runs): " + "; ".join(cells))
    print(
        "== paired by round (the same agents' sample): differences, each round's and their median"
    )
    pairs = [
        (
            "the fixed make-room - v0.3's, W5's campaign",
            "w5",
            "w5-disc-pr-ab48-s24",
            "w5",
            "w5-disc-pr-ab48-v03-s24",
        ),
        (
            "the frozen box, PR campaign - W5's",
            "w5-pr",
            "w5-disc-frozen-s24",
            "w5",
            "w5-disc-frozen-s24",
        ),
        (
            "the 48-token window, PR campaign - W5's",
            "w5-pr",
            "w5-disc-pr-ab48-s24",
            "w5",
            "w5-disc-pr-ab48-s24",
        ),
        (
            "the 48-token window - preempt-resubmit, the extension",
            "w5-ext",
            "w5-disc-pr-ab48-s24",
            "w5-ext",
            "w5-disc-a8c-resubmit-s24",
        ),
        (
            "the 48-token window - the frozen box, standard scenario, the extension",
            "w5-ext",
            "w5-std-pr-ab48-s16",
            "w5-ext",
            "w5-std-frozen-s16",
        ),
        (
            "the 48-token window, the extension - W5's",
            "w5-ext",
            "w5-disc-pr-ab48-s24",
            "w5",
            "w5-disc-pr-ab48-s24",
        ),
    ]
    for title, ca, pa, cb, pb in pairs:
        if pa not in data.get(ca, {}) or pb not in data.get(cb, {}):
            print(f"  {title}: not in these records yet")
            continue
        a, b = data[ca][pa], data[cb][pb]
        seeds_a, seeds_b = [s for s, _ in a], [s for s, _ in b]
        if seeds_a != seeds_b:
            print(f"  {title}: the rounds' seeds differ ({seeds_a} against {seeds_b})")
            return 1
        print(f"  {title}:")
        for name, path in METRICS.items():
            diffs = [value(x, path) - value(y, path) for (_, x), (_, y) in zip(a, b, strict=True)]
            each = ", ".join(f"{d:+,.0f}" for d in diffs)
            print(f"    {name}: {each}; median {statistics.median(diffs):+,.0f}")
    e3()
    return 0


def e3_rows() -> dict[tuple[str, str], list[dict]]:
    """E3's counts of each run, by (campaign, point): the engine's steps over the calls and those in which the budget
    cap held lower-priority prefill back, the prefill tokens held back (summed over the steps), make-room's preemptions
    and vLLM's count of every preemption, the tokens loaded from the CPU tier beyond the requests' first scheduling (the
    victims' resumes, give or take the window's edges), and the agents' tokens/s over the first 240 s."""
    counts = {
        (r["campaign"], r["point"]): r
        for r in csv.DictReader((HERE / "e3.tsv").open(), delimiter="\t")
    }
    rows: dict[tuple[str, str], list[dict]] = {}
    for (campaign, point), c in counts.items():
        record = HERE / campaign / point
        summary = json.loads((record / "summary.json").read_text())
        cache = json.loads((record / "engine.json").read_text())["cache_config"]
        per_token = cache["kv_cache_memory_bytes"] / (cache["num_gpu_blocks"] * cache["block_size"])
        engine = summary["engine"]
        beyond = (
            engine["kv_offload_load_bytes"] / per_token
            - engine["prompt_tokens_by_source"]["external_kv_transfer"]
        )
        held = int(c["held_steps"]) if c["held_steps"] else None
        rows.setdefault((campaign.rsplit("-r", 1)[0], point), []).append(
            {
                "steps": int(c["engine_steps"]),
                "held": held,
                "share": None if held is None else held / int(c["engine_steps"]),
                "held_tokens": int(c["held_prefill_tokens"]) if c["held_prefill_tokens"] else None,
                "make_room": int(c["make_room_preemptions"])
                if c["make_room_preemptions"]
                else None,
                "vllm": engine["preemptions"],
                "beyond": beyond,
                "goodput": summary["background"]["fixed_window"]["tokens_per_s"],
            }
        )
    return rows


def median_of(runs_: list[dict], key: str, fmt: str = ",.0f") -> str:
    """The median of `key` over the runs that have it, formatted; "-" when none has."""
    values = [r[key] for r in runs_ if r[key] is not None]
    return format(statistics.median(values), fmt) if values else "-"


def e3() -> None:
    """E3's table: each configuration's medians over its runs (e3_rows)."""
    if not (HERE / "e3.tsv").exists():
        print("== E3: no e3.tsv yet (e3_counters.py writes it)")
        return
    print("== E3: the medians of each configuration's runs")
    for (campaign, point), runs_ in sorted(e3_rows().items()):
        med = functools.partial(median_of, runs_)
        print(
            f"  {point} ({CAMPAIGNS[campaign]}, {len(runs_)} runs): steps {med('steps')}; held {med('held')} "
            f"({med('share', '.1%')}); prefill tokens held back {med('held_tokens')}; make-room preemptions "
            f"{med('make_room')}; vLLM's preemptions {med('vllm')}; loaded beyond first scheduling {med('beyond')}; "
            f"agents' tokens/s over 240 s {med('goodput')}"
        )


if __name__ == "__main__":
    sys.exit(main())
