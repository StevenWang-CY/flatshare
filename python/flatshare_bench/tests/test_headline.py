"""The headline tables against the committed records: every number of README.md's table and of docs/PRODUCTION.md's
"What it buys" table, as printed, from the medians of the W5 runs its row reports, pooled over the campaigns that ran
it as benchmarks/results/2026-10-24-e2-w5/compare.py pools them; and BENCHMARKS.md's FB, RS, transport and
window-sweep tables, from those records' own compare.py, running_share.py, ab.py and table.py. New records, or a table
edited by hand, cannot leave a stale headline: the commit that changes either fails here."""

from __future__ import annotations

import importlib.util
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from flatshare_bench.report import data

REPO = Path(__file__).resolve().parents[3]
W5_RESULTS = REPO / "benchmarks" / "results" / "2026-10-24-e2-w5"
_spec = importlib.util.spec_from_file_location("w5_compare", W5_RESULTS / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

ALONE = ("w5:w5-a1-voice",)
FROZEN = ("w5:w5-disc-frozen-s24", "w5-pr:w5-disc-frozen-s24")
RESUBMIT = ("w5-ext:w5-disc-a8c-resubmit-s24",)
AB48 = ("w5:w5-disc-pr-ab48-s24", "w5-pr:w5-disc-pr-ab48-s24", "w5-ext:w5-disc-pr-ab48-s24")
ROWS = {
    "Voice alone": ALONE,
    "vLLM 0.32, tuned": FROZEN,
    "Shared, tuned vLLM 0.32": FROZEN,
    "Tuned, with a router": RESUBMIT,
    "Shared, abort-and-resubmit": RESUBMIT,
    "+ #63350: a waiting request": ("w5-pr:w5-disc-pr-a-s24",),
    "+ #63351: agent prefill": ("w5-pr:w5-disc-pr-ab-s24",),
    "+ #63351 with a 48-token window": AB48,
    "Shared, this configuration": AB48,
}
"""Each headline row, by the start of its first cell, and the points (campaign:point) whose runs it reports."""
NONE = "\u2014"  # an empty cell's dash
GOODPUT = "agents' tokens/s, first 240 s"
MINUS = "\u2212"  # a negative difference's sign, as the docs print it


def summaries(points: Sequence[str]) -> list[dict]:
    out: list[dict] = []
    for point in points:
        campaign, name = point.split(":")
        found = compare.runs(campaign).get(name)
        assert found, f"{point}: not in the records"
        out += [summary for _, summary in found]
    return out


def median(runs: Sequence[dict], metric: str) -> float:
    return statistics.median(compare.value(s, compare.METRICS[metric]) for s in runs)


def rows_of(path: Path, header: str) -> list[list[str]]:
    """The body rows of the Markdown table whose header row starts with ``header``, each cell stripped and without
    bold."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    body = []
    for line in lines[start + 2 :]:  # past the header and its rule
        if not line.startswith("|"):
            break
        body.append([cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")])
    assert body, f"{path}: no rows under {header!r}"
    return body


def points_of(label: str) -> tuple[str, ...]:
    found = [points for start, points in ROWS.items() if label.startswith(start)]
    assert len(found) == 1, f"no single headline configuration for the row {label!r}"
    return found[0]


def slo() -> float:
    return median(summaries(ALONE), "V2V p95") + 250


def over_slo(label: str) -> tuple[int, int]:
    """How many of the row's replies took longer than the SLO, and of how many: every measured turn of its runs, as
    :meth:`flatshare_bench.report.data.Configuration.over` counts them over the same campaigns."""
    points = points_of(label)
    [name] = {point.split(":")[1] for point in points}
    c = data.configuration(data.w5_rows(), name)
    assert sorted(f"{run.campaign.rsplit('-r', 1)[0]}:{name}" for run in c.runs) == sorted(
        point for point in points for _ in range(5)
    ), label
    return c.over(data.slo_ms(data.w5_rows())), len(c.samples())


def share(over: int, replies: int) -> str:
    """A share of replies as the docs print it: to a tenth of a percent under 1%, else to the percent."""
    percent = 100 * over / replies
    return f"{percent:.1f}%" if percent < 1 else f"{percent:.0f}%"


def test_the_readmes_headline_table_is_the_records_medians() -> None:
    rows = rows_of(REPO / "README.md", "| Two calls beside 24 coding-agent sessions")
    assert len(rows) == 6
    for label, p95, over, replies, goodput, ttft in rows:
        runs = summaries(points_of(label))
        voice = median(runs, "V2V p95")
        assert p95 == f"{voice:,.0f} ms", label
        assert replies == share(*over_slo(label)), label
        if points_of(label) == ALONE:
            assert (over, goodput, ttft) == (NONE, NONE, NONE), label
            continue
        assert over == f"{voice - slo():+,.0f} ms".replace("-", MINUS), label
        assert goodput == f"{median(runs, GOODPUT):,.0f}", label
        assert ttft == f"{median(runs, 'agent TTFT p50') / 1000:.0f} s", label


def test_productions_headline_table_is_the_records_medians() -> None:
    path = REPO / "docs" / "PRODUCTION.md"
    assert f"the SLO is voice alone plus 250 ms at p95, {slo():,.0f} ms" in path.read_text()
    rows = rows_of(path, "| Configuration | Runs | V2V p95 |")
    assert len(rows) == 4
    for label, count, p95, replies, goodput, ttft in rows:
        runs = summaries(points_of(label))
        p95s = [compare.value(s, compare.METRICS["V2V p95"]) for s in runs]
        assert count == str(len(runs)), label
        assert replies == "{} / {}".format(*over_slo(label)), label
        assert (
            p95
            == f"{statistics.median(p95s):,.0f} ms ({min(p95s):,.0f}{compare.DASH}{max(p95s):,.0f})"
        ), label
        if points_of(label) == ALONE:
            assert (goodput, ttft) == (NONE, NONE), label
            continue
        assert goodput == f"{median(runs, GOODPUT):,.0f}", label
        assert ttft == f"{median(runs, 'agent TTFT p99') / 1000:.0f} s", label


KNEE = REPO / "benchmarks" / "results" / "2026-10-25-window-knee"


def test_the_window_knees_tables_are_its_records_medians() -> None:
    # The window sweep's table in BENCHMARKS.md and in its records' README, row by row, from its runs' outputs
    # through the records' own table.py.
    spec = importlib.util.spec_from_file_location("knee_table", KNEE / "table.py")
    table = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(table)
    medians = table.medians(KNEE)
    assert sorted(medians) == [16, 24, 32, 48, 64]
    for doc in (REPO / "docs" / "BENCHMARKS.md", KNEE / "README.md"):
        text = doc.read_text()
        for window, m in medians.items():
            row = (
                f"| {window} | {m['ttft_p95']:,.0f} | {m['t16_p95']:,.0f} | {m['t40_p95']:,.0f} "
                f"| {m['prompt_tps']:,.0f} |"
            )
            assert row in text, (doc.name, row)


FB = REPO / "benchmarks" / "results" / "2026-10-25-e2-fb"


def test_fbs_table_is_its_records_against_w5s() -> None:
    # BENCHMARKS.md's FB table is what FB's records' compare.py prints first, from FB's records and W5's.
    printed = subprocess.run(
        [sys.executable, str(FB / "compare.py")], capture_output=True, text=True, check=True
    ).stdout
    first = printed.split("\n\n")[0].strip()
    assert first.count("\n") == 4  # the header, its rule and three configurations
    assert first in (REPO / "docs" / "BENCHMARKS.md").read_text()


RS = REPO / "benchmarks" / "results" / "2026-10-26-e2-rs"


def test_rss_table_is_its_records_beside_the_a100s() -> None:
    # BENCHMARKS.md's RS table is what RS's records' running_share.py prints first, from RS's records and the A100
    # core's, every round of both counted.
    printed = subprocess.run(
        [sys.executable, str(RS / "running_share.py")], capture_output=True, text=True, check=True
    ).stdout
    first = printed.split("\n\n")[0].strip()
    assert first.count("\n") == 3  # the header, its rule and two configurations
    # Each configuration's five rounds on the box.
    assert all(row.split(" | ")[1] == "5" for row in first.splitlines()[2:])
    assert first in (REPO / "docs" / "BENCHMARKS.md").read_text()


TRANSPORT = REPO / "benchmarks" / "results" / "2026-10-25-w6-transport"


def test_the_transport_ab_table_is_its_records() -> None:
    # BENCHMARKS.md's transport A/B table is what the records' ab.py prints first, from both sides' summaries.
    printed = subprocess.run(
        [sys.executable, str(TRANSPORT / "ab.py")], capture_output=True, text=True, check=True
    ).stdout
    first = printed.split("\n\n")[0].strip()
    assert first.count("\n") == 7  # the header, its rule and six stages
    assert first in (REPO / "docs" / "BENCHMARKS.md").read_text()
