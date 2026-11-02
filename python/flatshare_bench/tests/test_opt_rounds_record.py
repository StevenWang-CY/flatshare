"""The optimization rounds' record (benchmarks/results/2026-11-02-opt-rounds): its README's tables are what its table.py
makes of the rounds' points, and every round ran every point its campaign lists."""

from __future__ import annotations

import importlib.util
import re
import statistics
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-11-02-opt-rounds"


def _table() -> ModuleType:
    spec = importlib.util.spec_from_file_location("opt_rounds_table", RECORD / "table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_readmes_tables_are_what_the_records_give() -> None:
    table = _table()
    readme = (RECORD / "README.md").read_text()
    for name in table.TABLES:
        found = re.search(
            rf"<!-- table.py {name} -->\n(.*?)\n<!-- /table.py {name} -->", readme, re.S
        )
        assert found is not None, name
        assert found.group(1) == table.compact(table.table(name))


def test_every_round_ran_its_campaigns_points() -> None:
    table = _table()
    rows = table.points(table.ROUNDS + table.FOLLOW_ROUNDS)
    for name in table.ROUNDS + table.FOLLOW_ROUNDS:
        listed = re.findall(
            r"^arms/(\S+)\.toml$", (REPO / "benchmarks/e2" / f"{name}.campaign").read_text(), re.M
        )
        assert sorted(point for campaign, point in rows if campaign == name) == sorted(listed), name


def test_the_record_names_home_directories_as_a_tilde() -> None:
    # The export writes home directories as ~; a campaign's log, copied beside it, names its report under the home
    # directory of the host that ran it, which the record must not carry.
    leaked = [
        path.relative_to(RECORD)
        for path in RECORD.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and re.search(r"/(home|Users)/[^/\s]+", path.read_text(errors="replace"))
    ]
    assert not leaked, leaked


def test_benchmarks_gives_the_rounds_tables_as_the_record_does() -> None:
    # BENCHMARKS.md's ledger of the rounds is the record's own tables of them, row for row.
    table = _table()
    benchmarks = (REPO / "docs" / "BENCHMARKS.md").read_text()
    for name in ("slo", "voice", "agents", "follow", "follow-agents"):
        assert table.compact(table.table(name)) in benchmarks, name


def test_production_gives_the_two_callers_rounds_as_the_record_does() -> None:
    # PRODUCTION.md's rows of the rounds beside 24 sessions with two callers: each configuration's replies over the SLO,
    # pooled, and the medians over the rounds of the agents' token rate and their TTFT p50 and p99.
    table = _table()
    rows = table.points()
    production = (REPO / "docs" / "PRODUCTION.md").read_text()
    labels = (
        "| v1.14's engine beside 24 sessions, five more rounds on new samples | 24 |",
        "| **The same with v1.15's first four additions** | 24 |",
    )
    [(_, before, after)] = [s for s in table.SCENARIOS if s[0] == "24 agent sessions, 2 callers"]
    for label, point in zip(labels, (before, after), strict=True):
        rounds = table.by_round(point, rows)
        over, replies = sum(r[1] for r in rounds), sum(r[2] for r in rounds)
        got = [rows[name, point] for name in table.ran(rows) if (name, point) in rows]
        tokens = statistics.median(r["background_tokens_per_s"] for r in got)
        p50 = statistics.median(r["agent_ttft_p50_ms"] for r in got) / 1e3
        p99 = statistics.median(r["agent_ttft_p99_ms"] for r in got) / 1e3
        row = (
            f"{label} {over} / {replies} ({100 * over / replies:.1f}%) | {tokens:,.0f} | "
            f"{p50:.0f} / {p99:.0f} s |"
        )
        assert row in production, row


WORDS = ("none", "one", "two", "three", "four", "five", "six")


def test_the_docs_give_the_follow_up_rounds_verdicts_as_the_record_does() -> None:
    # The follow-up rounds' rules, fixed before they ran: the decode hold stays with no more replies over the SLO than
    # the first four additions, pooled over the six rounds and caller counts, for at most 3% of the agents' token rate
    # at the median of the six; speculative decoding comes in with fewer than the hold, pooled and in five of the six.
    # The verdicts follow from the record, and the docs that give them give its counts.
    table = _table()
    rows = table.points(table.FOLLOW_ROUNDS)
    pairs = []  # each round and caller count: (replies over the SLO, replies, agents' tokens a second) a configuration
    for _, *trio in table.FOLLOW:
        for name in table.ran(rows, table.FOLLOW_ROUNDS):
            pair = []
            for point in trio:
                row = rows[name, point]
                threshold = table.slo_ms(rows, name, row["callers"])
                over = sum(s > threshold for s in row["v2v_samples"])
                pair.append((over, len(row["v2v_samples"]), row["background_tokens_per_s"]))
            pairs.append(pair)
    assert len(pairs) == 6
    four, hold, spec = (sum(pair[k][0] for pair in pairs) for k in range(3))
    replies = sum(pair[0][1] for pair in pairs)
    hold_fewer = sum(pair[1][0] < pair[0][0] for pair in pairs)
    spec_fewer = sum(pair[2][0] < pair[1][0] for pair in pairs)
    cost = -100 * statistics.median(pair[1][2] / pair[0][2] - 1 for pair in pairs)
    # The rule turns the hold down, its cost to the agents past 3%; version 1.17 serves it anyway.
    assert hold <= four and cost > 3
    assert not (spec < hold and spec_fewer >= 5)  # speculative decoding not adopted
    phrases = {
        "docs/REPORT.md": (
            f"The hold put {hold} of {replies} replies over the SLO against {four}, fewer in "
            f"{WORDS[hold_fewer]} of the six",
            f"cost the agents {cost:.1f}% of their token rate at the median of the six",
            f"Speculative decoding put {spec} over against the hold's {hold}, fewer in {WORDS[spec_fewer]} of "
            "the six where the rule asked for five",
        ),
        "docs/BENCHMARKS.md": (
            f"**The decode hold put {hold} of {replies} replies over the SLO against {four}**, fewer in "
            f"{WORDS[hold_fewer]} of the six",
            f"at {cost:.1f}% of the agents' token rate at the median of the six",
            f"**Speculative decoding on top put {spec} over against the hold's {hold}**, fewer in "
            f"{WORDS[spec_fewer]} of the six",
        ),
        "benchmarks/results/2026-11-02-opt-rounds/README.md": (
            f"{hold} of {replies} against {four} for the first four additions",
            f"fewer in {WORDS[hold_fewer]} of the six",
            f"It cost the agents {cost:.1f}% of their token rate at the median of the six",
            f"put {spec} of {replies} over against the hold's {hold}, fewer in {WORDS[spec_fewer]} of the six",
        ),
        "docs/PRODUCTION.md": (
            f"it put {hold} of {replies} replies over the SLO against {four} but cost the agents {cost:.1f}% of their "
            "token rate at the median",
        ),
        "deploy/profiles/rtx-5060-ti-serve.toml": (
            f"put {hold} of {replies} replies over the SLO beside 24 agent sessions against {four} without it, but "
            f"cost the agents {cost:.1f}% of their token rate at the median",
        ),
    }
    for path, expected in phrases.items():
        text = " ".join((REPO / path).read_text().replace("\n#", "\n").split())
        for phrase in expected:
            assert phrase in text, (path, phrase)
