"""The headline tables against the committed records: every number of README.md's table and of docs/PRODUCTION.md's
"What it buys" table, as printed, from the medians of the W5 runs its row reports, pooled over the campaigns that ran
it as benchmarks/results/2026-10-24-e2-w5/compare.py pools them; the headline numbers of the README's opening and of
docs/REPORT.md's abstract, from the W5, A100 and two-card records, and the one date both give the pull requests' state;
the report's Tables 6 and 8 and §6.9's comparison, from the two-card record's own gate.py; and BENCHMARKS.md's FB, RS,
transport and window-sweep tables, from those records' own compare.py, running_share.py, ab.py and table.py. New
records, or a table edited by hand, cannot leave a stale headline: the commit that changes either fails here."""

from __future__ import annotations

import functools
import importlib.util
import re
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from flatshare_bench.report import data

REPO = Path(__file__).resolve().parents[3]
REPORT = REPO / "docs" / "REPORT.md"
W5_RESULTS = REPO / "benchmarks" / "results" / "2026-10-24-e2-w5"
_spec = importlib.util.spec_from_file_location("w5_compare", W5_RESULTS / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)
ROUNDS_GATE = data.two_cards_gate()
"""The two-card record's own gate.py (benchmarks/results/2026-10-28-rounds), whose functions count the rounds behind
§6.8 and §6.9."""

ALONE = ("w5:w5-a1-voice",)
FROZEN = ("w5:w5-disc-frozen-s24", "w5-pr:w5-disc-frozen-s24")
RESUBMIT = ("w5-ext:w5-disc-a8c-resubmit-s24",)
AB48 = ("w5:w5-disc-pr-ab48-s24", "w5-pr:w5-disc-pr-ab48-s24", "w5-ext:w5-disc-pr-ab48-s24")
ROWS = {
    "Voice alone": ALONE,
    "vLLM 0.32, tuned": FROZEN,
    "Shared, tuned vLLM 0.32": FROZEN,
    "Tuned, with a router": RESUBMIT,
    "Shared, preempt-resubmit": RESUBMIT,
    "+ #63350: a waiting request": ("w5-pr:w5-disc-pr-a-s24",),
    "+ #63351: agent prefill": ("w5-pr:w5-disc-pr-ab-s24",),
    "+ #63351 with a 48-token window": AB48,
    "Shared, this configuration": AB48,
}
"""Each headline row, by the start of its first cell, and the points (campaign:point) whose runs it reports."""
README_COLUMNS = [
    "Voice p95",
    "Over the SLO",
    "Replies over the SLO",
    "Agents' tokens/s",
    "Agents' TTFT p50",
    "Agents' TTFT p99",
]
"""The README's headline table's columns after the configuration's, in the order the test reads them."""
NONE = "\u2014"  # an empty cell's dash
REPORT_NONE = "\u2013"  # an empty cell's dash in the report's tables
GOODPUT = "agents' tokens/s, first 240 s"
MINUS = "\u2212"  # a negative difference's sign, as the docs print it
STEP_CAPS = {
    "Tuned vLLM, 512-token steps": "w5-disc-frozen-s24",
    "The configuration to deploy, 512-token steps": "w5-disc-pr-ab48-s24",
    "The configuration to deploy, 256-token steps": "sweep-disc-pr-ab48-b256-t0-s24",
}
"""Table 8's rows (§6.9), by their first cell, and the point of the two-card record each reports (gate.py's SWEEP)."""
VOICE_CARD = ("w5-a1-voice", 1)  # the calls alone on a card
AGENTS_CARD = ("rg-disc-agents-only-s14", 1)  # the agents alone, the best practical dedicated card
TWO_CARDS = {
    # Table 6's rows (§6.8), by their first two cells, and the cards each arrangement runs as gate.py counts them:
    # the card with the calls first; a pooled arrangement runs its one point on both cards. The rounds' stock engine
    # ran the frozen box's flags, tuned vLLM in GLOSSARY's terms, as Figure 14 names it.
    ("Dedicated roles, controlled", "0 + 12"): (VOICE_CARD, ("rg-disc-agents-frozen-s12", 1)),
    ("Dedicated roles, best practical", "0 + 14"): (VOICE_CARD, AGENTS_CARD),
    ("Pooling, tuned vLLM", "12 + 12"): (("rg-disc-frozen-c1-s12", 2),),
    ("Pooling, the change", "12 + 12"): (("rg-disc-pr-ab48-b256-c1-s12", 2),),
    ("Consolidation, tuned vLLM", "4 + 14"): (("rg-disc-frozen-s4", 1), AGENTS_CARD),
    ("Consolidation, tuned vLLM", "2 + 14"): (("rg-disc-frozen-s2", 1), AGENTS_CARD),
    ("Consolidation, the change", "8 + 14"): (("rg-disc-pr-ab48-b256-s8", 1), AGENTS_CARD),
    ("Consolidation, the change", "4 + 14"): (("rg-disc-pr-ab48-b256-s4", 1), AGENTS_CARD),
    ("Consolidation, the change, with a warm spare", "4 + 12"): (
        ("rg-disc-pr-ab48-b256-s4", 1),
        ("rg-disc-agents-pr-ab48-b256-s12", 1),
    ),
}
WORDS = {2: "two", 4: "four", 5: "five"}
"""The counts the openings spell out."""


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


def columns_of(path: Path, header: str) -> list[str]:
    """The cells of the Markdown table header row that starts with ``header``."""
    [line] = [line for line in path.read_text().splitlines() if line.startswith(header)]
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def passage(path: Path, start: str, end: str) -> str:
    """The text of ``path`` from the line that starts with ``start`` to the next line that starts with ``end``, its
    lines joined with single spaces."""
    lines = path.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith(start))
    last = next(i for i in range(first + 1, len(lines)) if lines[i].startswith(end))
    return " ".join(" ".join(lines[first:last]).split())


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


def tenths(over: int, replies: int) -> str:
    """A share of replies to a tenth of a percent, as §6.9 prints it."""
    return f"{100 * over / replies:.1f}%"


@functools.cache
def rounds(campaigns: tuple[str, ...]) -> tuple[dict, dict]:
    """The two-card record's points over ``campaigns``, by (campaign, point), and each round's SLO by its callers, as
    its gate.py reads them."""
    points = ROUNDS_GATE.rows(data.TWO_CARDS, campaigns)
    return points, ROUNDS_GATE.slos(points, campaigns)


@functools.cache
def step_caps() -> dict[str, dict]:
    """§6.9's three configurations beside 24 sessions, over the sweep's five rounds, as the two-card record's gate.py
    counts them: the replies over each round's SLO, pooled and by round, the median and the pooled p95 V2V in ms, and
    the agents' median tokens a second and TTFT p50 and p99 in s."""
    gate, campaigns = ROUNDS_GATE, ROUNDS_GATE.SWEEP_ROUNDS
    points, slo = rounds(campaigns)
    found = {}
    for point in gate.SWEEP:
        runs, cards = [points[c, point] for c in campaigns], ((point, 1),)
        found[point] = {
            "replies": gate.foreground(points, slo, cards, campaigns),
            "by round": [gate.foreground(points, slo, cards, (c,))[0] for c in campaigns],
            "p95": statistics.median(run["v2v_p95_ms"] for run in runs),
            "pooled": gate.quantiles(sorted(s for run in runs for s in run["v2v_samples"]))[94],
            "tokens": statistics.median(run["fixed_window_tokens_per_s"] for run in runs),
            "ttft": tuple(
                statistics.median(run[f"agent_ttft_{q}_ms"] for run in runs) / 1000
                for q in ("p50", "p99")
            ),
        }
    assert sorted(found) == sorted(STEP_CAPS.values())
    return found


def test_the_readmes_headline_table_is_the_records_medians() -> None:
    header = "| Two calls beside 24 coding-agent sessions"
    assert columns_of(REPO / "README.md", header)[1:] == README_COLUMNS
    rows = rows_of(REPO / "README.md", header)
    assert len(rows) == 6
    for label, p95, over, replies, goodput, p50, p99 in rows:
        runs = summaries(points_of(label))
        voice = median(runs, "V2V p95")
        assert p95 == f"{voice:,.0f} ms", label
        assert replies == share(*over_slo(label)), label
        if points_of(label) == ALONE:
            assert (over, goodput, p50, p99) == (NONE,) * 4, label
            continue
        assert over == f"{voice - slo():+,.0f} ms".replace("-", MINUS), label
        assert goodput == f"{median(runs, GOODPUT):,.0f}", label
        assert p50 == f"{median(runs, 'agent TTFT p50') / 1000:.0f} s", label
        assert p99 == f"{median(runs, 'agent TTFT p99') / 1000:.0f} s", label


def test_the_readmes_opening_and_the_reports_abstract_are_the_records() -> None:
    # The headline numbers both openings state, each from its record: W5's medians and replies over the SLO, the step
    # cap's five rounds (§6.9), the A100's (§6.7) and the two cards' (§6.8).
    alone, tuned, router, change = (
        median(summaries(points), "V2V p95") / 1000 for points in (ALONE, FROZEN, RESUBMIT, AB48)
    )
    tuned_over = over_slo("vLLM 0.32, tuned")
    change_over = over_slo("+ #63351 with a 48-token window")
    kept = median(summaries(AB48), GOODPUT) / median(summaries(FROZEN), GOODPUT)
    caps = {label: "{} of {}".format(*step_caps()[p]["replies"]) for label, p in STEP_CAPS.items()}
    a100_tuned, a100_change = data.a100()[512]
    a100_slo = data.slo_ms(data.a100_rows())
    assert all(run.p95 < a100_slo for run in a100_change.runs)
    cards = data.two_cards()
    gains = {
        a.engine: a.tokens / cards.best.tokens - 1
        for a in cards.arrangements
        if a.kind == "consolidation" and not a.spare and a.sessions[0] == cards.envelopes[a.engine]
    }
    both = [
        f"{alone:.2f} s",
        f"{change:.2f} s",
        f"{slo() / 1000:.2f} s",
        share(*tuned_over),
        share(*change_over),
        "{} of {}".format(*change_over),
        f"{kept:.0%} of their token rate",
        caps["The configuration to deploy, 256-token steps"],
        caps["The configuration to deploy, 512-token steps"],
        f"from {a100_tuned.p95 / 1000:.2f} s to {a100_change.p95 / 1000:.2f} s, under that card's SLO in each of "
        f"{WORDS[len(a100_change.runs)]} rounds",
    ]
    readme = passage(REPO / "README.md", "Flatshare runs a self-hosted", "<picture>")
    for phrase in [*both, f"to {tuned:.1f} s", caps["Tuned vLLM, 512-token steps"]]:
        assert phrase in readme, phrase
    abstract = passage(REPORT, "## Abstract", "## Contents")
    for phrase in [
        *both,
        f"to {tuned:.2f} s",
        f"{router:.2f} s at a {median(summaries(RESUBMIT), 'agent TTFT p99') / 1000:.0f} s agent p99",
        f"{WORDS[cards.envelopes['change']]} agent sessions",
        f"{gains['change']:.0%} more",
        f"consolidated at {WORDS[cards.envelopes['stock']]} sessions, does {gains['stock']:.0%} more",
    ]:
        assert phrase in abstract, phrase
    # The pull requests' state in one wording and under one date: the README's opening, the report's note and its two
    # references.
    state = r"open, with no maintainer review as of (\d{4}-\d{2}-\d{2})"
    [date] = set(re.findall(rf"\({state}\)", readme))
    text = REPORT.read_text()
    [noted] = re.findall(rf"(?m)^> .*\b{state}\.", text)
    listed = re.findall(rf"pull request #6335[01]\. .*?, {state}\.", text)
    assert noted == date
    assert listed == [date, date]


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


def test_the_reports_two_card_table_is_the_rounds_records() -> None:
    # Table 6 (§6.8), cell by cell, as the two-card record's gate.py counts each arrangement over its three rounds.
    gate, campaigns = ROUNDS_GATE, ROUNDS_GATE.ROUNDS
    points, slo = rounds(campaigns)
    rows = rows_of(REPORT, "| Arrangement | Agent sessions |")
    assert [(label, sessions) for label, sessions, *_ in rows] == list(TWO_CARDS)

    def qualifies(cards: tuple[tuple[str, int], ...]) -> bool:
        over, replies = gate.foreground(points, slo, cards, campaigns)
        return over <= gate.REQUIREMENT * replies

    best = max(
        gate.background(points, cards, campaigns)[0]
        for (label, _), cards in TWO_CARDS.items()
        if label.startswith("Dedicated") and qualifies(cards)
    )
    for label, sessions, replies, tokens, gain, ttft in rows:
        cards = TWO_CARDS[label, sessions]
        work, _ = gate.background(points, cards, campaigns)
        each = [str(points[campaigns[0], p]["sessions"]) for p, n in cards for _ in range(n)]
        assert sessions == " + ".join(each), label
        assert replies == "{} / {}".format(*gate.foreground(points, slo, cards, campaigns)), label
        assert tokens == f"{work:,.0f}", label
        if label.startswith("Dedicated"):
            assert gain == REPORT_NONE, label
        elif qualifies(cards):
            assert gain == f"{work / best - 1:+.1%}", label
        else:
            assert gain == "does not qualify", label
        assert ttft == gate.ttft(points, cards, campaigns).replace("/", " / ") + " s", label


def test_the_reports_step_cap_table_and_its_comparison_are_the_rounds_records() -> None:
    # Table 8 (§6.9), cell by cell, and §6.9's comparison of the step caps, as the two-card record's gate.py counts
    # the sweep's five rounds.
    rows = rows_of(REPORT, "| Configuration | Median p95 V2V |")
    assert [label for label, *_ in rows] == list(STEP_CAPS)
    for label, p95, pooled, replies, by_round, tokens, ttft in rows:
        cap = step_caps()[STEP_CAPS[label]]
        assert (p95, pooled) == (f"{cap['p95']:,.0f} ms", f"{cap['pooled']:,.0f} ms"), label
        assert replies == "{} / {}".format(*cap["replies"]), label
        assert by_round == ", ".join(map(str, cap["by round"])), label
        assert tokens == f"{cap['tokens']:,.0f}", label
        assert ttft == "{:.0f} / {:.0f} s".format(*cap["ttft"]), label
    tuned, at512, at256 = (step_caps()[point] for point in STEP_CAPS.values())
    # Fewer replies over the SLO at 256 than at 512 in each of the five rounds.
    assert all(a < b for a, b in zip(at256["by round"], at512["by round"], strict=True))
    prose = passage(REPORT, "### 6.9", "**Table 8.")
    for phrase in (
        "{} of {} against {}".format(*at256["replies"], at512["replies"][0]),
        f"({tenths(*at256['replies'])} against {tenths(*at512['replies'])})",
        f"pooled p95 of {at256['pooled'] / 1000:.2f} s against {at512['pooled'] / 1000:.2f} s",
        "tuned vLLM had {} of {} over".format(*tuned["replies"]),
        f"{1 - at256['tokens'] / at512['tokens']:.0%} of their token rate",
        f"{at256['tokens']:,.0f} tokens a second against {at512['tokens']:,.0f}",
        f"from {at512['ttft'][0]:.0f} s to {at256['ttft'][0]:.0f} s",
    ):
        assert phrase in prose, phrase


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
