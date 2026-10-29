"""The 24-hour soak's tables from its records: voice alone's two hours, whose replies' p95 sets the SLO, beside the
serving arrangement's 24 hours, each from the summary the soak tool made of its record (calls, turns, replies and the
share over the SLO, the realtime service level, the agents' work, restarts and the guard's pauses), and the least and
the most of the 24 hours' figures, the spare's part: the calls each node carried and Gray's covering, at every
scrape, and what the change's rules did on each card.

    cd python/flatshare_bench && uv run python ../../benchmarks/results/2026-10-28-soak/table.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNS = (("Voice alone", "voice-alone"), ("The serving arrangement", "serving-24h"))


def _summary(run: str, root: Path) -> dict[str, Any]:
    return json.loads((root / run / "summary.json").read_text())


def _over(replies: dict[str, Any]) -> str:
    share = replies.get("over_slo_share")
    return "the reference" if share is None else f"{replies['over_slo']} ({share:.1%})"


def _row(label: str, cell: dict[str, Any], hours: str) -> str:
    calls, turns, gateway = cell["calls"], cell["turns"], cell["gateway"]
    replies = turns["replies"]
    # As the summary's own table counts them: a unit's failures and the restarts systemd scheduled.
    restarts = sum(
        c.get("failed", 0) + c.get("restart_scheduled", 0)
        for c in cell["restarts"]["units"].values()
    )
    return (
        f"| {label} | {hours} | {calls['attempted']:,} ({calls['accepted']:,}, {calls['refused']:,}, {calls['failed']:,}) | "
        f"{turns['measured']:,} ({turns['failed']:,}) | {replies['n']:,} | {_over(replies)} | {replies['p95']:,.0f} | "
        f"{gateway['realtime_sli']['good_share']:.3f} | {gateway['background']['tokens_per_s']:,.0f} | "
        f"{restarts} | {cell['guard']['pauses']} |"
    )


def table(root: Path = HERE) -> str:
    lines = [
        "| Run | Hours | Calls (accepted, refused, failed) | Turns (failed) | Replies | Over the SLO | Replies p95, ms | "
        "Realtime SLI | Background tokens/s | Restarts | Guard pauses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, run in RUNS:
        summary = _summary(run, root)
        lines.append(_row(label, summary["overall"], f"{summary['window']['hours']:g}"))
    return "\n".join(lines)


def hours(root: Path = HERE) -> str:
    """The least and the most, over the serving arrangement's hours, of each hour's replies, share over the SLO,
    replies' p95, realtime service level and background tokens a second."""
    cells = _summary("serving-24h", root)["hours"]
    columns = (
        ("Replies", lambda h: h["turns"]["replies"]["n"], "{:,.0f}"),
        ("Over the SLO", lambda h: h["turns"]["replies"]["over_slo_share"], "{:.1%}"),
        ("Replies p95, ms", lambda h: h["turns"]["replies"]["p95"], "{:,.0f}"),
        (
            "Realtime SLI",
            lambda h: h["gateway"]["realtime_sli"]["good_share"],
            "{:.3f}",
        ),
        (
            "Background tokens/s",
            lambda h: h["gateway"]["background"]["tokens_per_s"],
            "{:,.0f}",
        ),
    )
    lines = [
        f"| Of the {len(cells)} hours | " + " | ".join(name for name, _, _ in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for label, pick in (("Least", min), ("Most", max)):
        lines.append(
            f"| {label} | "
            + " | ".join(form.format(pick(get(h) for h in cells)) for _, get, form in columns)
            + " |"
        )
    return "\n".join(lines)


def spare(root: Path = HERE) -> str:
    """At each of the serving arrangement's scrapes of the gateway (series.csv), as its summary counts them, from the
    window's start through the last scrape after its end, the calls placed on each node and whether Gray covered: the
    most calls either node held at a scrape, and the scrapes with Gray covering."""
    window = _summary("serving-24h", root)["window"]
    with (root / "serving-24h" / "series.csv").open() as f:
        rows = list(csv.reader(f))
    head = rows[0]
    body = [r for r in rows[1:] if float(r[0]) > window["start"]]  # summary.py's overall period

    def column(name: str) -> list[float]:
        i = head.index(f"gateway|{name}")
        return [float(r[i]) for r in body if r[i] != ""]

    calls = {
        node: column(f'flatshare_gateway_node_sessions{{class="realtime",node="{node}"}}')
        for node in ("phastform", "gray")
    }
    covering = column('flatshare_gateway_node_covering{node="gray"}')
    return "\n".join(
        [
            "| Scrapes | Calls on Phastform, the most at a scrape | Calls on Gray, the most at a scrape | "
            "Scrapes with Gray covering |",
            "|---:|---:|---:|---:|",
            f"| {len(covering):,} | {max(calls['phastform']):g} | {max(calls['gray']):g} | "
            f"{sum(1 for v in covering if v > 0):,} |",
        ]
    )


MAKE_ROOM = ("held_steps", "held_prefill_tokens", "held_admissions", "preemptions")


def make_room(root: Path = HERE) -> dict[str, dict[str, int]]:
    """Each GPU node's make-room totals over the serving arrangement's window, from make-room.tsv (its engine's
    journal): the last totals in the window less the last before it, of the engine process that logged the last; none
    for a node whose engine logged none, since the scheduler logs its totals only while they move."""
    window = _summary("serving-24h", root)["window"]
    with (root / "make-room.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    totals = {}
    for node in ("phastform", "gray"):
        logged = [r for r in rows if r["node"] == node and float(r["at"]) <= window["end"]]
        if not logged:
            totals[node] = dict.fromkeys(MAKE_ROOM, 0)
            continue
        last = logged[-1]
        before = [r for r in logged if r["pid"] == last["pid"] and float(r["at"]) < window["start"]]
        totals[node] = {k: int(last[k]) - (int(before[-1][k]) if before else 0) for k in MAKE_ROOM}
    return totals


def rules(root: Path = HERE) -> str:
    """What the change's rules did on each card over the window: the steps in which the budget cap held lower-priority
    prefill back and what it held, and make-room's preemptions to make room (make-room.tsv), beside vLLM's own count of
    preemptions and the KV pool's use at the scrapes (the summary's engines)."""
    engines = _summary("serving-24h", root)["overall"]["engines"]
    held = make_room(root)
    lines = [
        "| Card | Steps the budget cap held prefill back | Prefill tokens held | Admissions held | "
        "Make-room's preemptions | vLLM's preemptions | KV pool used, mean / most |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for node, label in (
        ("phastform", "Phastform, the calls"),
        ("gray", "Gray, the spare"),
    ):
        h, e = held[node], engines[node]
        kv = e["kv_cache_usage"]
        lines.append(
            f"| {label} | {h['held_steps']:,} | {h['held_prefill_tokens']:,} | {h['held_admissions']:,} | "
            f"{h['preemptions']:,} | {e['preemptions']:,} | {kv['mean']:.1%} / {kv['max']:.1%} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
    print()
    print(hours())
    print()
    print(spare())
    print()
    print(rules())
