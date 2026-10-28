"""The two-card comparison's record (benchmarks/results/2026-10-28-rounds): its README against gate.py, and gate.py's
rules on points made by hand, so that a verdict follows the gate as RESOURCE-GATE.md fixed it."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "benchmarks" / "results" / "2026-10-28-rounds"


def _gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rounds_gate", RECORD / "gate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_readmes_verdict_and_tables_are_what_the_records_give() -> None:
    gate = _gate()
    readme = (RECORD / "README.md").read_text()
    for name, made in (
        ("gate.py", gate.gate()),
        ("gate.py margins", gate.margins()),
        ("gate.py pace", gate.pace()),
        ("gate.py stages", gate.stages()),
        ("gate.py sweep", gate.sweep()),
    ):
        found = re.search(rf"<!-- {name} -->\n(.*?)\n<!-- /{name} -->", readme, re.S)
        assert found is not None, name
        assert found.group(1) == made


def point(callers: int, sessions: int, samples: list[float], tokens: float = 0.0) -> dict:
    return {
        "callers": callers,
        "sessions": sessions,
        "v2v_samples": samples,
        "v2v_p95_ms": max(samples) if samples else None,
        "fixed_window_tokens_per_s": tokens,
        "agent_ttft_p50_ms": 1000.0,
        "agent_ttft_p99_ms": 2000.0,
    }


def test_a_card_is_held_to_its_rounds_slo_for_its_callers() -> None:
    gate = _gate()
    points = {
        ("r1", "w5-a1-voice"): point(2, 0, [1000.0, 1750.0]),
        ("r1", "rg-voice-c1"): point(1, 0, [900.0, 1600.0]),
        ("r1", "pooled"): point(1, 12, [1849.0, 1851.0, 1500.0, 1700.0], 1000.0),
        ("r1", "consolidated"): point(2, 12, [1999.0, 2001.0, 1500.0, 1700.0], 1500.0),
    }
    slo = gate.slos(points, ("r1",))
    assert slo == {("r1", 2): 2000.0, ("r1", 1): 1850.0}
    # One caller's card against 1,850 ms, two callers' against 2,000: one reply over each.
    assert gate.foreground(points, slo, (("pooled", 2),), ("r1",)) == (1, 4)
    assert gate.foreground(points, slo, (("consolidated", 1),), ("r1",)) == (1, 4)
    # A pooled arrangement's two cards run the same point: its agents count twice, its replies once.
    assert gate.background(points, (("pooled", 2),), ("r1",)) == (2000.0, [2000.0])
    both = (("consolidated", 1), ("w5-a1-voice", 1))
    assert gate.background(points, both, ("r1",)) == (1500.0, [1500.0])


@pytest.mark.parametrize(("over", "qualifies"), [(5, True), (6, False)])
def test_an_arrangement_qualifies_with_at_most_5_percent_of_its_replies_over(
    over: int, qualifies: bool
) -> None:
    gate = _gate()
    samples = [3000.0] * over + [1000.0] * (100 - over)
    points = {("r1", "w5-a1-voice"): point(2, 0, [1750.0]), ("r1", "card"): point(2, 12, samples)}
    slo = gate.slos(points, ("r1",))
    got_over, total = gate.foreground(points, slo, (("card", 1),), ("r1",))
    assert (got_over, total) == (over, 100)
    assert (got_over <= gate.REQUIREMENT * total) is qualifies


def test_a_dedicated_card_runs_the_number_of_agents_at_which_it_does_the_most() -> None:
    gate = _gate()
    rounds = ("r1", "r2", "r3")
    rates = {
        24: (4000.0, 4400.0, 4300.0),
        16: (6100.0, 5900.0, 7000.0),
        12: (6000.0, 6050.0, 9000.0),
    }
    points = {
        (c, f"agents-s{n}"): point(0, n, [], rate)
        for n, by_round in rates.items()
        for c, rate in zip(rounds, by_round, strict=True)
    }
    sizes = tuple((n, f"agents-s{n}") for n in rates)
    # The medians are 4,300, 6,100 and 6,050: 16 trees, though 12 had the single busiest round.
    assert gate.busiest(points, sizes, rounds) == (16, "agents-s16")
    # A tie goes to the most trees, the first listed.
    tied = {k: point(0, 0, [], 5000.0) for k in points}
    assert gate.busiest(tied, sizes, rounds) == (24, "agents-s24")
    assert gate.busiest({}, sizes, rounds) is None


def test_the_envelope_is_the_most_session_trees_whose_rounds_pooled_qualify() -> None:
    gate = _gate()
    rounds = ("r1", "r2")
    points = {(c, "w5-a1-voice"): point(2, 0, [1750.0]) for c in rounds}
    # At 12, 3 of 40 replies over the 2,000 ms SLO (7.5%); at 8, 2 of 40 (5%); at 4, none.
    over = {12: (2, 1), 8: (1, 1), 4: (0, 0)}
    for n, by_round in over.items():
        for c, k in zip(rounds, by_round, strict=True):
            points[c, f"s{n}"] = point(2, n, [2500.0] * k + [1500.0] * (20 - k), 1000.0)
    slo = gate.slos(points, rounds)
    sizes = ((12, "s12"), (8, "s8"), (4, "s4"))
    assert gate.envelope(points, slo, sizes, rounds) == (8, "s8")
    # A count the rounds did not measure is no envelope.
    assert gate.envelope(points, slo, ((16, "s16"),), rounds) is None
