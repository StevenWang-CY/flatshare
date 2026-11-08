"""The envelope rounds' tables (README.md), from this record's points.

Version 1.24's rounds (env-r1 to r3, Phastform, Gray, Phastform, each a fresh sample of the agents, seeds 21, 22 and
23) put the serving profile through version 1.23 at its envelope, 4 session trees beside the two calls, against the same
with the serve module's cache-aware queue and SM copy (the profile's queue_max_wait_s = 120 and sm_load_programs = 4) at
12 and 24 trees, each round from the release's own tree. The trial's round on Gray (opt2-env-r2, seed 22) ran the same
configurations from the trial's modules, before a review found that its count of the CPU tier's hit could overcount;
its table stands beside the rounds, outside the gate.
A reply is over the SLO when its voice-to-voice time exceeds that round's voice alone plus 250 ms, and a configuration
qualifies when at most 5% of its replies, pooled over the rounds, are over (benchmarks/quality/RESOURCE-GATE.md's
rule). A row's other figures are the median over the rounds, with their range.

The Gray windows before them (opt2-w1g, opt2-w1g-fix, opt2-w2g, opt2-i5g; one seed, 36, beside 24 trees) found the
queue's gain and the cause of its voice tail, each point against the control of its own window.

    cd python/flatshare_bench && uv run python ../../benchmarks/results/2026-11-08-envelope/table.py NAME
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from flatshare_bench.e2.report import load_points

HERE = Path(__file__).resolve().parent
ROUNDS = ("env-r1", "env-r2", "env-r3")
TRIAL = "opt2-env-r2"
ALONE = "w5-a1-voice"
MARGIN_MS = 250.0
SHARE = 0.05
GATE = (
    # (the configuration, session trees, the point)
    ("The serving profile through v1.23", 4, "opt-disc-hold-s4"),
    ("The queue and the SM copy", 12, "queue-disc-s12"),
    ("The queue and the SM copy", 24, "queue-disc-s24"),
)
TRIAL_GATE = (
    ("The serving profile through v1.23", 4, "opt-disc-hold-s4"),
    ("The trial's queue and SM copy", 12, "opt2-disc-srpfsm-s12"),
    ("The trial's queue and SM copy", 24, "opt2-disc-srpfsm-s24"),
)
WINDOWS = (
    # (the window, its control, then each point with what it adds to the control)
    (
        "Gray, window 1 (11-07)",
        "opt2-w1g",
        (
            ("opt2-w1g", "opt-disc-hold-s24", "v1.23, the control"),
            ("opt2-w1g", "opt2-disc-t30-s24", "a 30 GiB CPU tier"),
            ("opt2-w1g", "opt2-disc-srpf-s24", "the queue, every count refreshed each step"),
            ("opt2-w1g-fix", "opt2-disc-srpf-s24", "the queue, two counts a step, none beside a call"),
            ("opt2-w1g", "opt2-disc-t30srpf-s24", "the queue (every count) and a 30 GiB tier"),
        ),
    ),
    (
        "Gray, window 2 (11-08)",
        "opt2-w2g",
        (
            ("opt2-w2g", "opt-disc-hold-s24", "v1.23, the control"),
            ("opt2-w2g", "opt2-disc-srpfsm-s24", "the queue and the SM copy"),
            ("opt2-w2g", "opt2-disc-t30srpfsm-s24", "the queue, the SM copy and a 30 GiB tier"),
        ),
    ),
    (
        "Gray, the whole points (11-08)",
        "opt2-i5g",
        (
            ("opt2-i5g", "opt-disc-hold-s24", "v1.23, the control"),
            ("opt2-i5g", "opt2-disc-srpfsm-s24", "the queue and the SM copy"),
            ("opt2-i5g", "opt-disc-hold-ep-s24", "endpointing 0.2/0.65 s alone (not in the profile)"),
            ("opt2-i5g", "opt2-disc-srpfsm-ep-s24", "the queue, the SM copy and endpointing 0.2/0.65 s"),
            ("opt2-i5g", "opt-disc-hold-c4-s24", "v1.23, 4 callers"),
            ("opt2-i5g", "opt2-disc-srpfsm-ep-c4-s24", "the queue, the SM copy and endpointing 0.2/0.65 s, 4 callers"),
        ),
    ),
)
PAIR = ("opt2-env-r2", "opt-disc-hold-s24", "opt-disc-hold-jit-s24")
"""Round 2's pair for flatshare_jit's early finish: v1.23 against the same with it on, adjacent, one seed."""
TABLES = ("gate", "rounds", "trial", "windows", "jit")
DASH = "\u2014"
MINUS = "\u2212"


def points() -> dict[tuple[str, str], dict]:
    """Each complete point's row of E2's report, by (campaign, point)."""
    dirs = sorted(run.parent for run in HERE.glob("*/*/run.json") if (run.parent / "summary.json").exists())
    found, _ = load_points(dirs)
    return {(r["campaign"], r["point"]): r for r in found}


def slos(rows: dict, rounds: tuple[str, ...] = ROUNDS) -> dict[str, float]:
    """Each round's SLO: its voice alone's V2V p95 plus 250 ms."""
    return {c: rows[c, ALONE]["v2v_p95_ms"] + MARGIN_MS for c in rounds if (c, ALONE) in rows}


def computed(row: dict) -> float:
    """The share of the agents' and calls' prompt tokens computed at their first schedule, not found on the GPU or in
    the CPU tier, in percent."""
    parts = (row["first_schedule_computed_tokens"], row["first_schedule_gpu_hit_tokens"], row["first_schedule_cpu_hit_tokens"])
    return 100 * parts[0] / (sum(parts) or 1)


def loaded(row: dict) -> float:
    """What the point loaded from the CPU tier, in GB: the tokens loaded times the KV's bytes a token."""
    return row["cpu_tier_loaded_tokens"] * row["kv_bytes_per_token"] / 1e9


def spread(values: list[float], fmt: str = "{:,.0f}") -> str:
    """The median of values and, for more than one, their range beneath it."""
    if not values:
        return "—"
    mid = fmt.format(statistics.median(values))
    if len(values) == 1:
        return mid
    return f"{mid}<br><sub>{fmt.format(min(values))}⁠–⁠{fmt.format(max(values))}</sub>"


def gate(rows: dict, rounds: tuple[str, ...] = ROUNDS, configurations: tuple = GATE) -> list[str]:
    slo = slos(rows, rounds)
    out = [
        "| Configuration | Session trees | Rounds | Replies over the SLO | Qualifies | V2V p50, ms | V2V p95 | Agents' tokens/s | Agents' TTFT p50, s | p99 | Computed |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, trees, point in configurations:
        found = [(c, rows[c, point]) for c in rounds if (c, point) in rows and c in slo]
        over = sum(sum(v > slo[c] for v in row["v2v_samples"]) for c, row in found)
        total = sum(len(row["v2v_samples"]) for _, row in found)
        share = over / total if total else 0.0
        out.append(
            f"| {name} | {trees} | {len(found)} | {over} of {total} ({100 * share:.1f}%) | "
            f"{'yes' if total and share <= SHARE else 'no'} | "
            f"{spread([r['v2v_p50_ms'] for _, r in found])} | {spread([r['v2v_p95_ms'] for _, r in found])} | "
            f"{spread([r['fixed_window_tokens_per_s'] for _, r in found])} | "
            f"{spread([r['agent_ttft_p50_ms'] / 1e3 for _, r in found], '{:.1f}')} | "
            f"{spread([r['agent_ttft_p99_ms'] / 1e3 for _, r in found], '{:.0f}')} | "
            f"{spread([computed(r) for _, r in found], '{:.1f}%')} |"
        )
    return out


def trial(rows: dict) -> list[str]:
    """The trial's round on Gray, by the gate's rule, alone."""
    return gate(rows, (TRIAL,), TRIAL_GATE)


def rounds(rows: dict) -> list[str]:
    slo = slos(rows)
    out = [
        "| Round | Node | Voice alone V2V p95, ms | The SLO | " + " | ".join(f"{name}, {trees} trees" for name, trees, _ in GATE) + " |",
        "|---|---|---:|---:|" + "---:|" * len(GATE),
    ]
    for c in ROUNDS:
        if c not in slo:
            continue
        cells = []
        for _, _, point in GATE:
            row = rows.get((c, point))
            cells.append(
                "—" if row is None
                else f"{sum(v > slo[c] for v in row['v2v_samples'])} of {len(row['v2v_samples'])}, "
                f"{row['fixed_window_tokens_per_s']:,.0f} tokens/s"
            )
        out.append(f"| {c} | {rows[c, ALONE]['node']} | {rows[c, ALONE]['v2v_p95_ms']:,.0f} | {slo[c]:,.0f} | " + " | ".join(cells) + " |")
    return out


def windows(rows: dict) -> list[str]:
    out = [
        "| Window | Point | Agents' tokens/s | Against the control | Computed | Loaded from the CPU tier, GB | Agents' TTFT p50, s | Calls' first byte p95, ms | LLM first sentence p95 | V2V p50 | V2V p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window, home, entries in WINDOWS:
        control = rows[home, "opt-disc-hold-s24"]
        for campaign, point, about in entries:
            row = rows[campaign, point]
            base = control if row["callers"] == control["callers"] else rows[home, "opt-disc-hold-c4-s24"]
            gain = row["fixed_window_tokens_per_s"] / base["fixed_window_tokens_per_s"] - 1
            against = DASH if row is base else f"{100 * gain:+.0f}%".replace("-", MINUS)
            out.append(
                f"| {window} | {about} | {row['fixed_window_tokens_per_s']:,.0f} | "
                f"{against} | {computed(row):.1f}% | {loaded(row):,.1f} | "
                f"{row['agent_ttft_p50_ms'] / 1e3:.1f} | {row['realtime_first_byte_p95_ms']:,.0f} | "
                f"{row['llm_first_sentence_p95_ms']:,.0f} | {row['v2v_p50_ms']:,.0f} | {row['v2v_p95_ms']:,.0f} |"
            )
    return out


def jit(rows: dict) -> list[str]:
    campaign, off, on = PAIR
    out = [
        "| flatshare_jit's early finish | Agents' tokens/s | Agents' TTFT p50, s | LLM first token p95, ms | LLM first sentence p95 | V2V p50 | V2V p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, point in (("off (v1.23, as measured)", off), ("on (Model Runner V2's event read)", on)):
        row = rows[campaign, point]
        out.append(
            f"| {label} | {row['fixed_window_tokens_per_s']:,.0f} | {row['agent_ttft_p50_ms'] / 1e3:.1f} | "
            f"{row['llm_ttft_p95_ms']:,.0f} | {row['llm_first_sentence_p95_ms']:,.0f} | {row['v2v_p50_ms']:,.0f} | "
            f"{row['v2v_p95_ms']:,.0f} |"
        )
    return out


def table(name: str, rows: dict | None = None) -> str:
    rows = points() if rows is None else rows
    return "\n".join({"gate": gate, "rounds": rounds, "trial": trial, "windows": windows, "jit": jit}[name](rows))


if __name__ == "__main__":
    for name in sys.argv[1:] or TABLES:
        print(f"<!-- table.py {name} -->\n{table(name)}\n<!-- /table.py {name} -->\n")
