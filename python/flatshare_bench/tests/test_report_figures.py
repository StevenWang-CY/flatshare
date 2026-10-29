"""The technical report's figures (flatshare_bench.report): each draws from the committed records and passes the style's
checks, its text names the configurations as the report does, and the values it plots, which the report quotes, are
the records' own: the medians, ranges and counts their reports (frontier.csv), summaries and the campaigns' own scripts
give, and docs/BENCHMARKS.md states. That every report-*.svg under docs/figures is what its generator draws today,
test_figures.py checks with every other figure."""

from __future__ import annotations

import csv
import html
import importlib.util
import itertools
import json
import re
import statistics
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import pytest

from flatshare_bench.report import charts, data, figures

REPO = Path(__file__).resolve().parents[3]
SVG = "{http://www.w3.org/2000/svg}"
CODES = re.compile(r"\bW\d\b|\bFB\b|\bRS\b|\barms? \d|\bw5-|\bdisc-|\ba100-|frozen|\bE[0-3]\b")
"""What figure text never names: campaigns (W4, W5, FB, RS), arms, experiments, points and the records' names for
tuned vLLM; the report's terms name configurations instead."""


def frontier_rows(results: Path) -> dict[tuple[str, str], Mapping[str, str]]:
    """A record set's committed frontier.csv, by (campaign, point)."""
    with (results / "report" / "frontier.csv").open() as rows:
        return {(r["campaign"], r["point"]): r for r in csv.DictReader(rows)}


def summaries(results: Path, point: str, campaigns: tuple[str, ...]) -> list[dict]:
    """The summary.json of each run of ``point`` in the campaigns named by their rounds' prefix (w5-r, w5-pr-r)."""
    found = []
    for campaign in campaigns:
        for path in sorted(results.glob(f"{campaign}[0-9]/{point}/summary.json")):
            found.append(json.loads(path.read_text()))
    assert found, (point, campaigns)
    return found


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("report")
    return {path.name: path for path in figures.main(out)}


# The figures --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("figure", figures.FIGURES, ids=lambda f: f.__name__)
def test_each_figure_draws_and_passes_the_styles_checks(
    figure: Callable[[Path], list[Path]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Draft mode would write a figure that fails its checks; nothing but a person laying one out sets it.
    monkeypatch.delenv("FLATSHARE_FIGURE_DRAFT", raising=False)
    for path in figure(tmp_path):
        root = ET.parse(path).getroot()
        narrow = path.stem.endswith("-narrow")
        assert path.name.startswith("report-"), path.name
        assert float(root.get("width")) == (420 if narrow else 1120), path.name
        title, description = root.find(f"{SVG}title").text, root.find(f"{SVG}desc").text
        assert title and description, path.name
        # The alt text: a name, and one sentence that says what the figure shows.
        assert not title.endswith("."), title
        assert description.endswith(".") and not re.search(r"\. [A-Z]", description), description


def test_the_report_has_its_figures_and_the_phone_layouts_it_shows(drawn: dict[str, Path]) -> None:
    assert sorted(drawn) == sorted(
        f"report-{name}.svg"
        for name in (
            "stages",
            "stages-narrow",
            "kv-timeline",
            "kv-timeline-narrow",
            "isolation",
            "isolation-narrow",
            "tail",
            "tail-narrow",
            "window",
            "window-narrow",
            "step-cost",
            "step-cost-narrow",
            "headline",
            "headline-narrow",
            "cdf",
            "cdf-narrow",
            "agents",
            "agents-narrow",
            "frontier",
            "frontier-narrow",
            "margin",
            "margin-narrow",
            "a100",
            "a100-narrow",
            "cards",
            "cards-narrow",
            "step-cap",
            "step-cap-narrow",
            "soak",
            "soak-narrow",
            "starts",
            "starts-narrow",
        )
    )


def text_of(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    found = ["".join(t.itertext()) for t in root.iter(f"{SVG}text")]
    return [*found, root.find(f"{SVG}title").text, root.find(f"{SVG}desc").text]


def test_figure_text_names_configurations_as_the_report_does(drawn: dict[str, Path]) -> None:
    for name, path in drawn.items():
        for text in text_of(path):
            assert not CODES.search(text), (name, text)


def test_a_log_scale_says_so_on_its_axis(drawn: dict[str, Path]) -> None:
    logarithmic = (
        "report-headline",
        "report-cdf",
        "report-frontier",
        "report-step-cost",
        "report-tail",
    )
    for name, path in drawn.items():
        text = " ".join(text_of(path)[:-2])  # the art's text, not its title and description
        assert ("log scale" in text) == name.startswith(logarithmic), name


# The chart parts --------------------------------------------------------------------------------------------------


def chart() -> charts.Chart:
    return charts.Chart(400, 200, "A chart", "What it shows.")


def test_a_label_over_a_mark_is_refused() -> None:
    s = chart()
    s.mark(110, 40, "blue")
    s.text(100, 46, "a label")
    with pytest.raises(ValueError, match="a label covers a mark: 'a label'"):
        s.render()
    with pytest.raises(ValueError, match="a mark's hue is one of"):
        chart().mark(10, 10, "red")
    with pytest.raises(ValueError, match="a stage is one of"):
        chart().stack(10, 10, [(40, "queue")], 20)


def test_a_rule_may_run_under_a_label_only_where_a_segment_painted_over_it_hides_it() -> None:
    s = chart()
    s.vrule(150, 10, 190, "grid")
    s.stack(100, 60, [(120, "wait")], 30)
    s.text(160, 81, "2.81 s", size=18, anchor="middle")
    s.render()
    # A rule painted after the segment shows across the label.
    s = chart()
    s.stack(100, 60, [(120, "wait")], 30)
    s.vrule(150, 10, 190, "grid")
    s.text(160, 81, "2.81 s", size=18, anchor="middle")
    with pytest.raises(ValueError, match=r"a rule crosses the text '2\.81 s'"):
        s.render()


def test_a_halo_lets_a_label_sit_on_a_gridline_and_on_no_other_rule() -> None:
    s = chart()
    s.vrule(150, 10, 190, "grid")
    s.text(140, 81, "a value", size=18, halo=True)
    s.render()
    for draw in (
        lambda s: s.vrule(150, 10, 190, "slo"),
        lambda s: s.vrule(150, 10, 190, "axis"),
        lambda s: s.ref((10, 76), (390, 76), "ink trace dotted"),
    ):
        s = chart()
        draw(s)
        s.text(140, 81, "a value", size=18, halo=True)
        with pytest.raises(ValueError, match="a rule crosses the text 'a value'"):
            s.render()


def test_a_charts_style_holds_its_parts_and_both_themes() -> None:
    svg = chart().render()
    assert svg.count("<style>") == 1 and ".mark{" in svg and ".s-wait{" in svg
    # Each color the charts add has its dark value under the dark scheme.
    light, dark = svg.split("@media(prefers-color-scheme:dark)")[-2:]
    for name, (day, night) in charts.PALETTE.items():
        assert f"--{name}:{day}" in light and f"--{name}:{night}" in dark, name


def test_a_lines_key_is_a_level_stroke_in_the_lines_classes() -> None:
    s = chart()
    last = charts.legend(
        s, 10, 30, [(charts.line_key("blue trace thick"), "the change")], width=300
    )
    svg = s.render()
    assert last == 30 and 'd="M10 24.24H42" class="blue trace thick"' in svg and "the change" in svg


def test_a_log_scale_and_an_ecdf() -> None:
    x = charts.Scale(1000, 100000, 0, 200, log=True)
    assert (x(1000), x(10000), x(100000)) == (0, 100, 200)
    y = charts.Scale(0, 1, 100, 0)
    steps = charts.ecdf([3000, 1000, 1000, 50000], x, y, cap=30000)
    # Up a quarter for each sample, both of the ties at once, the last capped.
    assert [round(py, 6) for _, py in steps] == [100, 100, 50, 50, 25, 25, 0]
    assert steps[-1][0] == pytest.approx(x(30000))


# The values -------------------------------------------------------------------------------------------------------


def pooled(runs: list[dict], stage: str) -> float:
    cells = [run["worker"]["waterfall"][stage] for run in runs]
    return sum(c["n"] * c["mean"] for c in cells) / sum(c["n"] for c in cells)


STAGE_CAMPAIGNS = {
    data.VOICE_ALONE: ("w5-r",),
    data.TUNED: ("w5-r", "w5-pr-r"),
    data.MAKE_ROOM: ("w5-pr-r",),
    data.CHANGE: ("w5-r", "w5-pr-r", "w5-ext-r"),
}


def test_the_stages_are_the_summaries_pooled_over_each_configurations_replies() -> None:
    expected = {
        # endpointing, the wait for the LLM's first token, the decode, TTS, transport; the mean V2V; runs, replies
        data.VOICE_ALONE: ((756, 85, 218, 211, 352), 1623, 5, 295),
        data.TUNED: ((825, 2815, 1066, 250, 343), 5299, 10, 590),
        data.MAKE_ROOM: ((820, 783, 1207, 250, 356), 3415, 5, 295),
        data.CHANGE: ((822, 289, 262, 242, 355), 1971, 15, 885),
    }
    for cell in data.stages():
        parts, v2v, runs, replies = expected[cell.point]
        assert (tuple(round(p) for p in cell.parts), round(cell.v2v), cell.runs, cell.replies) == (
            parts,
            v2v,
            runs,
            replies,
        )
        # Count-weighted over the summaries' stage means, and adding up to the mean V2V.
        records = summaries(data.W5, cell.point, STAGE_CAMPAIGNS[cell.point])
        decode = pooled(records, "llm_first_sentence_ms") - pooled(records, "llm_ttft_ms")
        assert cell.decode == pytest.approx(decode)
        assert cell.wait + cell.decode == pytest.approx(
            pooled(records, "turn_to_first_sentence_ms")
        )
        assert cell.endpointing == pytest.approx(pooled(records, "end_of_turn_ms"))
        assert sum(cell.parts) == pytest.approx(cell.v2v)
        # The worker's own end-to-end mean is its stages', to within a millisecond of their counts' differences.
        assert abs(sum(cell.parts[:4]) - pooled(records, "e2e_ms")) < 2


def test_the_probes_are_their_logs_runs() -> None:
    kv, prefill = data.probes("kv"), data.probes("prefill")
    assert {e: [r.ttft for r in runs] for e, runs in kv.items()} == {
        "stock": [4900, 4971, 4969, 4966],
        "a": [219, 289, 290, 290],
        "ab": [216, 290, 294, 289],
    }
    assert {e: statistics.median(r.ttft for r in runs) for e, runs in kv.items()} == {
        "stock": 4967.5,
        "a": 289.5,
        "ab": 289.5,
    }
    assert {e: [r.t16 for r in runs] for e, runs in prefill.items()} == {
        "stock": [1529, 1501, 1220, 1500],
        "a": [1528, 1211, 1214, 1485],
        "ab": [615, 536, 541, 609],
    }
    # docs/BENCHMARKS.md: the 16th token at 1,211-1,528 ms with make-room alone against 536-615 ms with the hold, the
    # reply's tokens 60-66, 60-65 and 16-20 ms apart.
    gaps = {e: [(r.t16 - r.ttft) / 15 for r in runs] for e, runs in prefill.items()}
    assert {e: (round(min(g)), round(max(g))) for e, g in gaps.items()} == {
        "stock": (60, 66),
        "a": (60, 65),
        "ab": (16, 20),
    }
    # Each case's turn was sent beside what its panel says.
    assert {(r.agents, r.agent_tokens) for runs in kv.values() for r in runs} == {(2, 17999)}
    assert {(r.agents, r.agent_tokens) for runs in prefill.values() for r in runs} == {(3, 10000)}
    assert {r.prompt_tokens // 10 for runs in (*kv.values(), *prefill.values()) for r in runs} <= {
        74,
        75,
    }


def test_a_probe_line_the_parser_cannot_read_is_an_error(tmp_path: Path) -> None:
    for engine in data.ENGINES:
        (tmp_path / f"{engine}-kv.log").write_text(
            "kv: priority-0 TTFT 290 ms, 16 tokens —, for 745 prompt tokens, sent with 2 priority-2 requests "
            "of 17999 tokens running and 22 of 2274 blocks free; after it, 1 running and 1 waiting\n"
        )
    assert data.probes("kv", tmp_path)["ab"][0].t16 is None
    (tmp_path / "a-kv.log").write_text("kv: something else\n")
    with pytest.raises(ValueError, match=r"a-kv\.log: an unread line"):
        data.probes("kv", tmp_path)


def test_the_kv_timeline_is_its_logs_block() -> None:
    samples = data.kv_timeline()
    assert len(samples) == 46
    first, last = samples[0], samples[-1]
    assert (first.t, first.usage, first.running, first.waiting, first.tokens) == (
        0.1,
        0,
        0,
        0,
        (0, 0),
    )
    assert (last.t, last.usage, last.running, last.waiting, last.tokens) == (
        24.5,
        0.517,
        1,
        1,
        (812, 192),
    )
    # Read about every 0.55 s.
    assert {round(b.t - a.t, 1) for a, b in itertools.pairwise(samples)} == {0.5, 0.6}
    m = data.kv_moments(samples)
    assert (m.full.t, m.full.usage) == (7.7, 0.987)
    # The first read with both agent requests running and at most 24 of the pool's 2,274 blocks free.
    assert (m.send.t, m.send.running, m.send.free_blocks) == (8.3, 2, 20)
    # Nothing frees a block from then to the preemption.
    held = [p.usage for p in samples if m.send.t <= p.t <= m.before.t]
    assert held == sorted(held) and held[-1] == 1
    # The step: from the whole pool with both requests running to half of it, one running and agent B waiting,
    # stopped at 192 tokens while agent A decodes on.
    assert (m.before.t, m.before.usage, m.before.running, m.before.waiting, m.before.tokens) == (
        12.6,
        1,
        2,
        0,
        (191, 189),
    )
    assert (m.after.t, m.after.usage, m.after.running, m.after.waiting, m.after.tokens) == (
        13.1,
        0.501,
        1,
        1,
        (218, 192),
    )
    assert {p.tokens[1] for p in samples if p.t >= m.after.t} == {192}
    # The engine's preemption counter reads 0 throughout, so the figure marks the preemption by the requests running
    # and waiting, not by it.
    assert {p.preemptions for p in samples} == {0}


def test_a_timeline_line_the_parser_cannot_read_is_an_error(tmp_path: Path) -> None:
    log = tmp_path / "timeline.log"
    read = "  0.1s usage 0.000 running 0 waiting 0 preemptions 0 tokens [0, 0]"
    log.write_text(f"[05:25:42Z] timeline\n{read}\n[05:26:07Z] engine log\nanything\n")
    assert len(data.kv_timeline(log)) == 1
    log.write_text(f"[05:25:42Z] timeline\n{read}\n  0.6s usage high\n")
    with pytest.raises(ValueError, match="an unread line in the timeline"):
        data.kv_timeline(log)
    log.write_text(f"{read}\n")
    with pytest.raises(ValueError, match="0 timeline blocks"):
        data.kv_timeline(log)


TAILS = {
    # turns, then p50 and p95 in ms as docs/BENCHMARKS.md's table of the PRs' probes prints them
    ("fresh", "stock"): (96, "2,244", "6,830"),
    ("fresh", "a"): (117, "716", "4,111"),
    ("fresh", "ab"): (146, "236", "388"),
    ("grow", "stock"): (102, "1,324", "7,158"),
    ("grow", "a"): (125, "615", "1,073"),
    ("grow", "ab"): (146, "258", "358"),
}


def test_the_tails_are_each_blocks_two_runs_turns() -> None:
    tails = data.tails()
    for (mode, engine), (turns, p50, p95) in TAILS.items():
        t = tails[mode][engine]
        assert (t.runs, len(t.turns), t.failed) == (2, turns, 0), (mode, engine)
        assert (f"{t.percentile(50):,.0f}", f"{t.percentile(95):,.0f}") == (p50, p95), (
            mode,
            engine,
        )
        # The inclusive percentiles, as the probes' check.py computes its tables'.
        q = statistics.quantiles(t.turns, n=20, method="inclusive")
        assert (t.percentile(50), t.percentile(95)) == (statistics.median(t.turns), q[18])
    # The figure's axis, 40 ms to 15 s, holds every turn: the fastest took 43 ms, the slowest 11.8 s.
    every = [v for engines in tails.values() for t in engines.values() for v in t.turns]
    assert (min(every), max(every)) == (43, 11804)


def test_a_tail_run_whose_turns_disagree_with_its_summary_is_an_error(tmp_path: Path) -> None:
    summary = (
        "calls: {n} turns, 0 failed (counted at 30,000 ms), TTFT p50 236 p90 342 p95 388 max 516 ms; "
        "agents: 0 of 12 stopped early"
    )
    for engine in data.ENGINES:
        for mode in data.MODES:
            (tmp_path / f"{engine}-tail-{mode}-1.log").write_text(
                "03:26:12.611 call 1 ttft    236 ms\n" + summary.format(n=1) + "\n"
            )
    assert data.tails(tmp_path)["fresh"]["ab"].turns == (236,)
    (tmp_path / "ab-tail-grow-1.log").write_text(
        "03:26:12.611 call 1 ttft    236 ms\n" + summary.format(n=2) + "\n"
    )
    with pytest.raises(ValueError, match=r"ab-tail-grow-1\.log: its turns do not give"):
        data.tails(tmp_path)
    (tmp_path / "ab-tail-grow-1.log").write_text("03:26:12.611 call 1 slow\n" + summary.format(n=0))
    with pytest.raises(ValueError, match=r"ab-tail-grow-1\.log: an unread line"):
        data.tails(tmp_path)


def test_the_replies_and_the_window_sweep() -> None:
    tokens = data.reply_tokens()
    assert (len(tokens), statistics.median(tokens), min(tokens), max(tokens)) == (5571, 23, 6, 52)
    # "79% run past 16 tokens; one runs past 48."
    assert (
        sum(t <= 16 for t in tokens) == 1181 and f"{1 - data.share_within(tokens, 16):.0%}" == "79%"
    )
    assert sum(t > 48 for t in tokens) == 1
    runs = data.knee()
    assert [(r.window, r.run) for r in runs] == [
        (w, k) for w in (16, 24, 32, 48, 64) for k in (1, 2)
    ]
    assert not any(r.failed for r in runs)
    assert data.knee_medians(runs, "t40_p95") == {
        16: 3759,
        24: 2642,
        32: 1853.5,
        48: 1037,
        64: 1018,
    }
    assert data.knee_medians(runs, "ttft_p95") == {16: 448.5, 24: 387, 32: 326.5, 48: 347, 64: 366}
    prompt = data.knee_medians(runs, "prompt_tps")
    assert prompt == {16: 3483, 24: 3443, 32: 3300, 48: 3205, 64: 3246}
    assert f"{1 - prompt[48] / prompt[16]:.0%}" == "8%"


def test_the_step_costs_are_the_baselines_cells() -> None:
    costs = {c.key: c for c in data.step_costs()}
    baseline = json.loads(data.STEP_COSTS.read_text())["cells"]
    assert list(costs) == list(baseline)
    for key, cell in baseline.items():
        c = costs[key]
        assert (c.stock, c.off, c.on) == (cell["stock_us"], cell["off_us"], cell["on_us"])
        assert c.added == pytest.approx(
            cell["change_us"], abs=0.11
        )  # the cells round each to 0.1 µs
    # Switched off, the change costs what stock vLLM does, within 1.3% in the PRs' states and 3.1% in the serving
    # profile's; deployed, it adds tens to hundreds of µs a step, 1.5 ms where a voice turn too large for two victims
    # waits behind a 999-block hit, and it takes 30 µs off the step in which the hold keeps an agent's prefill back.
    assert max(c.off_ratio for c in costs.values() if not c.serving) < 0.013
    assert max(c.off_ratio for c in costs.values() if c.serving) < 0.031
    assert max(c.added for c in costs.values() if c.scenario == "A") == pytest.approx(165.5)
    assert sorted(round(c.added) for c in costs.values()) == [
        -30, 17, 36, 50, 62, 72, 89, 158, 166, 166, 260, 320, 364, 441, 463, 468, 1539,
    ]  # fmt: skip
    assert (costs["B64-hit999"].variant, costs["B256-W10"].variant) == (
        "a 999-block prefix hit",
        "10 waiting",
    )
    serving = [(c.scenario, c.running, c.variant) for c in costs.values() if c.serving]
    assert serving == [("idle", 2, ""), ("hold", 2, ""), ("nofit", 16, ""), ("fire", 2, "")]


def test_the_step_costs_stand_beside_the_boxs_cpu_and_steps(tmp_path: Path) -> None:
    """The box's CPU takes 1.68 times as long as the host's at bench_host.py's workload, by its median against the
    median of the host's as the baseline's rounds ended; a reply's step on the box alone is W6's median token gap, and
    the agents' at 24 session trees their mean token gap at the median of the three rounds that the two-card verdict
    counts (38.07, 38.23 and 30.49 ms; the sweep's other two rounds measured 37.26 and 34.61)."""
    box = json.loads(data.BOX_HOST.read_text())["python_ms"]["p50"]
    host = [load["python_ms"]["p50"] for load in json.loads(data.STEP_COSTS.read_text())["load"]]
    assert data.box_cpu() == box / statistics.median(host) and box == pytest.approx(381.1, abs=0.01)
    assert f"{data.box_cpu():.2f}" == "1.68"
    # Its top axis says it is a scaling of the host's, in both layouts.
    scaled = "the box's CPU (ms), scaled: the host's time \N{MULTIPLICATION SIGN} 1.68"
    for path in figures.step_cost(tmp_path):
        assert scaled in text_of(path), path.name
    assert data.voice_step_ms() == 12.08
    assert data.agents_step_ms() == 38.07


HEADLINE = {
    # runs, the median V2V p95 and its range, the agents' tokens a second over the first 240 s (ms, tokens/s)
    data.VOICE_ALONE: (5, 1784.3, 1764.2, 1818.7, None),
    data.TUNED: (10, 10477.35, 8685.9, 12115.2, 3305.15),
    data.ROUTER: (5, 2521.9, 2516.3, 2868.4, 3332.4),
    data.MAKE_ROOM: (5, 5085.2, 4693.3, 6261.1, 3413.8),
    data.HOLD_16: (5, 3093.8, 2868.5, 3191.5, 3298.0),
    data.CHANGE: (15, 2236.0, 2132.0, 2397.4, 3202.7),
}


def test_the_headline_is_each_configurations_runs_in_frontier_csv() -> None:
    rows = frontier_rows(data.W5)
    configurations = data.headline()
    assert [c.point for c in configurations] == list(HEADLINE)
    for c in configurations:
        assert (
            len(c.runs),
            round(c.p95, 2),
            min(r.p95 for r in c.runs),
            max(r.p95 for r in c.runs),
        ) == (HEADLINE[c.point][:4]), c.point
        assert c.tokens == HEADLINE[c.point][4], c.point
        for run in c.runs:
            row = rows[(run.campaign, c.point)]
            assert run.p95 == float(row["v2v_p95_ms"]), (run.campaign, c.point)
            if run.tokens is not None:
                assert run.tokens == float(row["fixed_window_tokens_per_s"]), (
                    run.campaign,
                    c.point,
                )
    # The SLO, voice alone's median p95 plus 250 ms; the change 0.20 s over it, at 97% of tuned vLLM's tokens.
    slo = data.slo_ms(data.w5_rows())
    assert slo == pytest.approx(2034.3)
    change, tuned = configurations[-1], configurations[1]
    assert round(change.p95 - slo) == 202 and f"{change.tokens / tuned.tokens:.0%}" == "97%"
    assert all(run.p95 > slo for run in change.runs)


def test_the_replies_over_the_slo_and_the_pooled_p95() -> None:
    # Beside each configuration's median run p95, how many of its replies missed the SLO and the p95 of all of them
    # taken together: a third of the change's replies missed it, which its 202 ms over the SLO at the median hides.
    rows = data.w5_rows()
    found = {}
    two = [(2, c.point) for c in data.headline(rows)] + [(2, point) for point in data.STANDARD]
    more = [
        (callers, point)
        for callers, points in data.CALLS.items()
        if callers > 2
        for point in points
    ]
    for callers, point in two + more:
        c = data.configuration(rows, point)
        found[point] = (
            c.over(data.slo_ms(rows, callers)),
            len(c.samples()),
            round(c.pooled_p95, 2),
        )
    assert found == {
        data.VOICE_ALONE: (1, 295, 1796.03),
        data.TUNED: (532, 590, 10609.19),
        data.ROUTER: (130, 295, 2717.44),
        data.MAKE_ROOM: (263, 295, 5290.11),
        "w5-disc-pr-ab-s24": (128, 295, 3090.42),
        data.CHANGE: (286, 885, 2267.76),
        "w5-a1-voice-c4": (4, 357, 1840.14),
        "w5-disc-pr-ab48-c4-s24": (142, 357, 2330.44),
        "w5-a1-voice-c6": (2, 546, 1884.9),
        "w5-disc-pr-ab48-c6-s24": (165, 546, 2395.25),
        "w5-std-frozen-s16": (72, 295, 3033.49),
        "w5-std-pr-ab48-s16": (46, 295, 2246.54),
    }


def test_the_agents_tokens_are_a_replay_s_prompts() -> None:
    # The change's 97% is tuned vLLM's agents' token rate over the calls' first 240 s kept in each campaign that ran
    # both on the same sample: ten pairs. The rate is the replay's prompt tokens nearly all; it generates few.
    retained = data.paired_retention(data.w5_rows(), data.CHANGE, data.TUNED)
    assert len(retained) == 10
    assert [round(100 * f(retained.values()), 2) for f in (statistics.median, min, max)] == [
        96.66,
        94.24,
        98.67,
    ]
    for point, share, generated in ((data.TUNED, 0.4655, 17.4), (data.CHANGE, 0.4703, 16.0)):
        runs = data.agent_tokens(point)
        assert round(100 * statistics.median(r.generated_share for r in runs), 4) == share, point
        assert round(statistics.median(r.completion for r in runs), 1) == generated, point


def test_every_turn_is_each_runs_measured_turns() -> None:
    rows = data.w5_rows()
    slo = data.slo_ms(rows)
    shares, slowest = {}, {}
    for c in data.headline(rows):
        samples = c.samples()
        campaigns = {r.campaign.rsplit("-r", 1)[0] + "-r" for r in c.runs}
        records = summaries(data.W5, c.point, tuple(sorted(campaigns)))
        # Each run's turns but its warm-up turns, a barge-in's second reply among them, as its summary counts them.
        assert len(samples) == sum(r["voice"]["v2v_ms"]["n"] for r in records), c.point
        assert max(samples) < 30000, c.point  # no turn went unanswered
        shares[c.point] = round(sum(v <= slo for v in samples) / len(samples), 3)
        slowest[c.point] = round(max(samples))
    assert {p: shares[p] for p in figures.CDF_LINES} == {
        data.VOICE_ALONE: 0.997,
        data.TUNED: 0.098,
        data.ROUTER: 0.559,
        data.CHANGE: 0.677,
    }
    assert {p: slowest[p] for p in figures.CDF_LINES} == {
        data.VOICE_ALONE: 2043,
        data.TUNED: 18624,
        data.ROUTER: 3366,
        data.CHANGE: 3711,
    }


AGENTS = {
    # runs; the median of the agents' TTFT p99 in s, of the steps the hold held agent prefill in (none without the
    # hold's counter) and of make-room's own preemptions (none without make-room)
    data.TUNED: (10, "106", None, None),
    data.ROUTER: (5, "193", None, None),
    data.MAKE_ROOM: (5, "112", "0%", 34),
    data.HOLD_16: (5, "105", "25%", 32),
    data.CHANGE: (15, "111", "33%", 31),
}


def test_what_the_agents_pay_is_each_runs_frontier_row_and_e3_counts() -> None:
    rows, counts = frontier_rows(data.W5), data.e3_counts()
    runs = data.agent_runs()
    assert list(runs) == list(AGENTS)
    for point, cells in runs.items():
        shares = [r.held_share for r in cells if r.held_share is not None]
        preemptions = [r.preemptions for r in cells if r.preemptions is not None]
        assert len(shares) in (0, len(cells)) and len(preemptions) in (0, len(cells)), point
        assert (
            len(cells),
            f"{statistics.median(r.ttft_p99 for r in cells) / 1000:.0f}",
            f"{statistics.median(shares):.0%}" if shares else None,
            statistics.median(preemptions) if preemptions else None,
        ) == AGENTS[point], point
        for run in cells:
            row, count = rows[(run.campaign, point)], counts[(run.campaign, point)]
            assert (run.ttft_p50, run.ttft_p99, run.tokens) == tuple(
                float(row[k])
                for k in ("agent_ttft_p50_ms", "agent_ttft_p99_ms", "fixed_window_tokens_per_s")
            ), (run.campaign, point)
            assert (run.steps, run.held, run.preemptions) == tuple(
                int(count[k]) if count[k] else None
                for k in ("engine_steps", "held_steps", "make_room_preemptions")
            ), (run.campaign, point)
    # The report's reading: the agents' median TTFT from 63 s on tuned vLLM to 72 s with the change, whose p99 lies
    # within the spread of tuned vLLM's runs (96-129 s).
    tuned, change = runs[data.TUNED], runs[data.CHANGE]
    assert [round(statistics.median(r.ttft_p50 for r in c) / 1000) for c in (tuned, change)] == [
        63,
        72,
    ]
    spread = [round(f(r.ttft_p99 for r in tuned) / 1000) for f in (min, max)]
    assert spread == [96, 129]
    assert (
        min(r.ttft_p99 for r in tuned)
        < statistics.median(r.ttft_p99 for r in change)
        < max(r.ttft_p99 for r in tuned)
    )


def test_the_frontier_is_every_deployable_configuration_of_w4s_discriminating_scenario() -> None:
    points = data.frontier()
    rows = frontier_rows(data.W4)
    assert len(points) == 18
    for p in points:
        runs = [r for (_, point), r in rows.items() if point == p.point]
        assert sorted(p.p95s) == sorted(float(r["v2v_p95_ms"]) for r in runs), p.point
        # The agents' tokens a second over the calls' first 240 s, a window every run filled.
        tokens = sorted(float(r["fixed_window_tokens_per_s"]) for r in runs)
        assert sorted(p.tokens_per_run) == tokens, p.point
        assert {r["fixed_window_s"] for r in runs} == {"240.0"}, p.point
        assert {r["arm"] for r in runs} == {p.arm}, p.point
    by_point = {p.point: p for p in points}
    # The figure names every configuration, under its family's heading.
    assert set(figures.FRONTIER_NAMES) == set(by_point)
    assert sorted(
        p.point for _, members in figures._frontier_rows(points) for p in members
    ) == sorted(by_point)
    named = (
        "disc-frozen-s24",
        "disc-a8b-smg-s24",
        "disc-a8a-flow-s24",
        "disc-a7b-sglang-hicache-s24",
        "disc-a8c-resubmit-s24",
        "disc-pr-ab48-s24",
        "disc-a9-gateway-only-s24",
        "disc-a5-queued8192-s24",
    )
    assert {k: (round(by_point[k].p95), round(by_point[k].tokens)) for k in named} == {
        "disc-frozen-s24": (10138, 3552),
        "disc-a8b-smg-s24": (11050, 3446),
        "disc-a8a-flow-s24": (10992, 3684),
        "disc-a7b-sglang-hicache-s24": (11948, 4330),
        "disc-a8c-resubmit-s24": (2591, 3593),
        "disc-pr-ab48-s24": (2193, 3362),
        "disc-a9-gateway-only-s24": (10993, 3633),
        "disc-a5-queued8192-s24": (30000, 5584),
    }
    assert by_point["disc-a5-queued8192-s24"].failed_turns == 48
    families = {}
    for p in points:
        families.setdefault(p.family, []).append(p.arm)
    assert families == {
        "tuned": ["4", "4", "5", "5"],
        "operator": ["6", "6", "6"],
        "engine": ["7a", "7b", "7b"],
        "router": ["8a", "8b"],
        "resubmit": ["8c"],
        "lifecycle-change": ["9"],
        "lifecycle": ["9-gateway-only"],
        "change": ["9-pr-a", "9-pr-ab", "9-pr-ab48"],
    }
    # Nothing without the engine change, and nothing but the router that aborts agent requests, is under 5.6 s.
    assert sorted(round(p.p95) for p in points if p.family not in ("change", "lifecycle-change"))[
        :2
    ] == [2591, 5671]
    assert data.slo_ms(data.w4_rows()) == pytest.approx(2017.3)


def test_the_margin_is_the_standard_scenarios_rounds_and_the_calls_per_gpu() -> None:
    rows = data.w5_rows()
    rounds = data.standard_rounds(rows)
    assert [r for r, _, _ in rounds] == [1, 2, 3, 4, 5]
    assert [round(change - tuned) for _, tuned, change in rounds] == [-109, -1413, -945, -250, -496]
    # Panel a's axis starts at 1.5 s, under every round's p95 and the SLO.
    assert min(min(tuned, change) for _, tuned, change in rounds) > figures.MARGIN_FLOOR
    assert data.slo_ms(rows) > figures.MARGIN_FLOOR
    counts = data.e3_counts()
    # Every caller count on the samples the four- and six-caller rounds used, the first three.
    seeds = {r["agents_seed"] for (_, p), r in counts.items() if p == data.CALLS[4][1]}
    assert seeds == {"11", "12", "13"}
    for c in data.calls(rows):
        # Each count's own SLO, and the change over it in every run.
        assert (c.callers, round(c.slo, 1)) in {(2, 2034.3), (4, 2044.5), (6, 2134.6)}
        assert min(c.over) > 0
        point = data.CALLS[c.callers][1]
        held = sorted(
            int(r["held_steps"]) / int(r["engine_steps"])
            for (_, p), r in counts.items()
            if p == point and r["agents_seed"] in seeds
        )
        assert sorted(c.held) == held, c.callers
    summary = {
        c.callers: (
            len(c.over),
            round(statistics.median(c.over)),
            round(min(c.over)),
            round(max(c.over)),
            f"{statistics.median(c.held):.0%}",
            f"{min(c.held):.0%}",
            f"{max(c.held):.0%}",
            round(statistics.median(c.tokens)),
        )
        for c in data.calls(rows)
    }
    assert summary == {
        # Two callers' nine runs on these samples; over all 15 the median is the same 202 ms and 33%, at 3,203 tokens/s.
        2: (9, 202, 98, 267, "33%", "26%", "38%", 3156),
        4: (3, 269, 206, 370, "53%", "44%", "54%", 3055),
        6: (3, 257, 226, 333, "64%", "57%", "68%", 2716),
    }


def test_the_a100_is_its_core_and_its_step_caps_rounds() -> None:
    rows = data.a100_rows()
    csv_rows = frontier_rows(data.A100)
    pairs = data.a100(rows)
    slo = data.slo_ms(rows)
    assert slo == pytest.approx(1669.4)
    tuned, change = pairs[512]
    assert [round(r.p95) for r in tuned.runs] == [4163, 4783, 4986, 5169, 5710]
    assert [round(r.p95) for r in change.runs] == [1663, 1658, 1611, 1634, 1609]
    # Under the card's SLO in every round, by 6 to 60 ms.
    assert (
        round(slo - max(r.p95 for r in change.runs)),
        round(slo - min(r.p95 for r in change.runs)),
    ) == (7, 60)
    assert (round(tuned.p95), round(change.p95)) == (4986, 1634)
    tuned, change = pairs[2048]
    assert [round(r.p95) for r in tuned.runs] == [4036, 3289]
    assert [round(r.p95) for r in change.runs] == [1810, 1843]
    assert [round(r.tokens) for r in (*tuned.runs, *change.runs)] == [15082, 13990, 15124, 14145]
    for cap, (t, c) in pairs.items():
        for config in (t, c):
            for run in config.runs:
                row = csv_rows[(run.campaign, config.point)]
                assert (run.p95, run.tokens) == (
                    float(row["v2v_p95_ms"]),
                    float(row["fixed_window_tokens_per_s"]),
                ), (cap, run.campaign)


def rounds_gate() -> ModuleType:
    """The two-card record's own gate.py, loaded from its file."""
    spec = importlib.util.spec_from_file_location("rounds_gate", data.TWO_CARDS / "gate.py")
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


ARRANGEMENTS = {
    # The kind, the engine and the agent sessions on each card, the calls' first: the replies over the SLO and the
    # replies, whether they meet the requirement, and the agents' tokens a second over the calls' first 240 s summed
    # over both cards, at the median of the rounds.
    ("dedicated", "stock", (0, 14)): (0, 177, True, "7,893"),
    ("dedicated", "stock", (0, 12)): (0, 177, True, "6,302"),
    ("consolidation", "stock", (2, 14)): (4, 177, True, "10,051"),
    ("consolidation", "change", (4, 14)): (5, 177, True, "10,596"),
    ("consolidation", "change", (4, 12)): (5, 177, True, "9,652"),  # with a warm spare
    ("consolidation", "change", (8, 14)): (31, 177, False, "13,537"),
    ("pooling", "stock", (12, 12)): (53, 87, False, "13,188"),
    ("pooling", "change", (12, 12)): (24, 87, False, "12,111"),
}


def test_the_two_cards_are_their_gates_verdict(drawn: dict[str, Path]) -> None:
    gate = rounds_gate()
    cards = data.two_cards()
    assert cards.rounds == gate.ROUNDS == ("rg-r1", "rg-r2", "rg-r3")
    assert (cards.requirement, cards.hurdle) == (gate.REQUIREMENT, gate.HURDLE) == (0.05, 0.20)
    # (a) Each engine's consolidated card as gate.py's own functions count it, its rounds pooled, and the envelope
    # that gives: two sessions on stock vLLM, four with the change.
    points = gate.rows(data.TWO_CARDS, gate.ROUNDS)
    slo = gate.slos(points, gate.ROUNDS)
    for engine, name in data.CONSOLIDATED.items():
        sizes = gate.CONSOLIDATION[name]
        cells = cards.consolidated[engine]
        assert [(c.sessions, c.over, c.replies) for c in cells] == [
            (sessions, *gate.foreground(points, slo, ((point, 1),), gate.ROUNDS))
            for sessions, point in sorted(sizes)
        ], engine
        assert cards.envelopes[engine] == gate.envelope(points, slo, sizes, gate.ROUNDS)[0]
    shares = {
        engine: [(c.sessions, f"{c.share:.1%}") for c in cells]
        for engine, cells in cards.consolidated.items()
    }
    assert shares == {
        "stock": [(2, "2.3%"), (4, "10.2%"), (8, "39.0%"), (12, "53.7%"), (24, "92.1%")],
        "change": [(2, "1.1%"), (4, "2.8%"), (8, "17.5%"), (12, "18.6%"), (24, "20.9%")],
    }
    assert {c.replies for cells in cards.consolidated.values() for c in cells} == {177}
    assert cards.envelopes == {"stock": 2, "change": 4}
    # (b) Each arrangement, in the figure's order, as the tables gate.py prints give it: its replies over the SLO,
    # whether it qualifies, its median and its rounds; the warm spare's, beside the gate, from its own line.
    found = {
        (a.kind, a.engine, a.sessions): (a.over, a.replies, a.qualifies, f"{a.tokens:,.0f}")
        for a in cards.arrangements
    }
    assert list(found.items()) == list(ARRANGEMENTS.items())
    assert [a.spare for a in cards.arrangements] == [False] * 4 + [True] + [False] * 3
    printed = gate.gate(data.TWO_CARDS).splitlines()
    for a in cards.arrangements:
        rounds = ", ".join(f"{r:,.0f}" for r in a.rounds)
        assert len(a.rounds) == len(gate.ROUNDS) and a.tokens == statistics.median(a.rounds), a.name
        assert a.qualifies == (a.over <= gate.REQUIREMENT * a.replies), a.name
        if a.spare:
            [line] = [line for line in printed if line.startswith(f"- {a.name} (")]
            replies = f"{a.over}/{a.replies} replies over the SLO"
            assert f": {replies}, {a.tokens:,.0f} tokens/s ({rounds})" in line, line
            continue
        [row] = [line for line in printed if line.startswith(f"| {a.name} |")]
        share = f"{a.over}/{a.replies} ({100 * a.over / a.replies:.1f}%)"
        verdict = "yes" if a.qualifies else "no"
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert cells[2:6] == [share, verdict, f"{a.tokens:,.0f}", rounds], row
    # The hurdle, 20% over the best qualifying dedicated arrangement, which each qualifying shared arrangement here
    # passes; gate.py's verdict says so of both envelopes.
    best = f"The best qualifying dedicated arrangement: {cards.best.name}, {cards.best.tokens:,.0f} tokens/s."
    assert best in printed and cards.best is cards.arrangements[0]
    assert f"{cards.threshold:,.0f}" == "9,471"  # 1.2 times 7,892.8
    shared = [a for a in cards.arrangements if a.kind != "dedicated" and a.qualifies]
    assert [a.tokens > cards.threshold for a in shared] == [True, True, True]
    assert sum("meets the 20% hurdle" in line for line in printed) == 2
    # Panel b's scale holds every round; the drawing hatches the three that miss the requirement, each in its engine's
    # hue, and prints every median.
    assert max(r for a in cards.arrangements for r in a.rounds) == pytest.approx(14352.2)
    assert max(r for a in cards.arrangements for r in a.rounds) < figures.CARD_TOKENS.hi
    svg, texts = drawn["report-cards.svg"].read_text(), set(text_of(drawn["report-cards.svg"]))
    assert (svg.count('class="gray hatch"'), svg.count('class="blue hatch"')) == (1, 2)
    assert {f"{a.tokens:,.0f}" for a in cards.arrangements} <= texts


def test_the_two_cards_read_only_the_rounds_their_gate_counts(tmp_path: Path) -> None:
    # The record's later rounds and logs change nothing the figure plots: a copy of the record with one more round
    # reads the same.
    (tmp_path / "gate.py").symlink_to(data.TWO_CARDS / "gate.py")
    for campaign in rounds_gate().ROUNDS:
        (tmp_path / campaign).symlink_to(data.TWO_CARDS / campaign, target_is_directory=True)
    (tmp_path / "rg-r4").symlink_to(data.TWO_CARDS / "rg-r1", target_is_directory=True)
    assert data.two_cards(tmp_path) == data.two_cards()


# The deployed result: the step cap, the soak and the engine's starts -------------------------------------------------


def test_the_step_cap_is_its_rounds_as_gate_py_counts_them(drawn: dict[str, Path]) -> None:
    gate = rounds_gate()
    caps = data.step_caps()
    assert [c.point for c in caps] == list(gate.SWEEP) == list(data.STEP_CAP_POINTS)
    assert all(c.point.endswith("-s24") for c in caps)  # beside 24 agent sessions
    assert [(c.engine, c.cap) for c in caps] == [("tuned", 512), ("change", 512), ("change", 256)]
    assert figures.REQUIREMENT == gate.REQUIREMENT == 0.05
    # Each configuration's lines in gate.py's sweep: its replies over the SLO pooled and by round, and its agents'
    # median over the rounds.
    printed = gate.sweep(data.TWO_CARDS).splitlines()
    for c in caps:
        [line, by_round] = [line for line in printed if line.startswith(f"| {c.point} |")]
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert cells[4] == f"{c.over}/{c.replies} ({100 * c.over / c.replies:.1f}%)", line
        assert cells[5] == f"{c.tokens:,.0f}", line
        rounds = " | ".join(f"{r.over}/{r.replies}" for r in c.rounds)
        assert by_round == f"| {c.point} | {rounds} |", by_round
        assert c.tokens == statistics.median(r.tokens for r in c.rounds)
    assert [r.round for c in caps for r in c.rounds] == [1, 2, 3, 4, 5] * 3
    assert {r.replies for c in caps for r in c.rounds} == {59}
    assert [(c.over, c.replies) for c in caps] == [(261, 295), (94, 295), (44, 295)]
    assert [[r.over for r in c.rounds] for c in caps] == [
        [53, 55, 55, 54, 44],
        [16, 29, 25, 9, 15],
        [10, 17, 10, 4, 3],
    ]
    assert [f"{c.tokens:,.0f}" for c in caps] == ["3,399", "3,303", "3,067"]
    # The claim its captions open with, true of every round: the change had fewer replies over the SLO at 256 tokens
    # a step than at 512 (and tuned vLLM more than either), and no configuration was within the requirement.
    tuned, at512, at256 = caps
    rounds = zip(tuned.rounds, at512.rounds, at256.rounds, strict=True)
    assert all(a.over > b.over > c.over for a, b, c in rounds)
    assert all(r.share > figures.REQUIREMENT for c in caps for r in c.rounds)
    assert {f"{c.tokens:,.0f}" for c in caps} <= set(text_of(drawn["report-step-cap.svg"]))


def soak_table() -> ModuleType:
    """The soak record's own table.py, loaded from its file."""
    spec = importlib.util.spec_from_file_location("soak_table", data.SOAK.parents[1] / "table.py")
    assert spec is not None and spec.loader is not None
    table = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(table)
    return table


def test_the_soak_is_its_summarys_hours(drawn: dict[str, Path]) -> None:
    soak = data.soak()
    table = soak_table()
    assert [h.hour for h in soak.hours] == list(range(24))
    assert sum(h.replies for h in soak.hours) == soak.replies == 5058
    assert sum(h.over for h in soak.hours) == soak.over == 74
    assert (f"{soak.share:.1%}", f"{soak.tokens:,.0f}", f"{soak.slo:,.0f}") == (
        "1.5%",
        "3,529",
        "1,976",
    )
    # The record's own tables: the day's row, and the least and the most of the hours.
    [day] = [
        line
        for line in table.table().splitlines()
        if line.startswith("| The serving arrangement |")
    ]
    assert f"| {soak.replies:,} | {soak.over} ({soak.share:.1%}) |" in day, day
    assert f"| {soak.tokens:,.0f} |" in day, day
    rows = {
        line.split("|")[1].strip(): line
        for line in table.hours().splitlines()
        if line.startswith("| ")
    }
    for name, pick in (("Least", min), ("Most", max)):
        cells = [cell.strip() for cell in rows[name].strip("|").split("|")]
        assert cells[2] == f"{pick(h.share for h in soak.hours):.1%}", rows[name]
        assert cells[5] == f"{pick(h.tokens for h in soak.hours):,.0f}", rows[name]
    # The claim its captions open with: one hour over the requirement, hour 4, 14 of its 254 replies.
    assert [(h.hour, h.over, h.replies) for h in soak.hours if h.share > figures.REQUIREMENT] == [
        (4, 14, 254)
    ]
    assert f"{max(h.share for h in soak.hours):.1%}" == "5.5%"
    # The agents: 9,300 tokens a second in hour 0, when AIPerf resumes its 14 sessions at once, 582 to 5,402 after.
    first, *rest = soak.hours
    assert f"{first.tokens:,.0f}" == "9,300"
    assert (f"{min(h.tokens for h in rest):,.0f}", f"{max(h.tokens for h in rest):,.0f}") == (
        "582",
        "5,402",
    )
    aiperf = (data.SOAK.parents[1] / "aiperf-start.txt").read_text()
    assert "resuming 14 trajectory sessions (burst" in aiperf
    texts = set(text_of(drawn["report-soak.svg"]))
    assert {"5.5%", "9,300", "the day, 1.5%", "the day, 3,529", figures.ALLOWED} <= texts


def test_the_starts_are_the_drills_journals(drawn: dict[str, Path]) -> None:
    starts, kill = data.starts(), data.kill()
    # Each start's phases as its record's report.txt prints them, from the same journal.
    for kind, name in data.STARTS.items():
        printed = re.findall(
            r"^ +([\d.]+) s  (.+)$", (data.DRILLS / name / "report.txt").read_text(), re.M
        )
        assert [(phase, f"{at:.1f}") for phase, at in starts[kind].phases] == [
            (p, t) for t, p in printed
        ], kind
    warm, cold = starts["warm"], starts["cold"]
    assert (f"{warm.at(figures.READY):.1f}", f"{cold.at(figures.READY):.1f}") == ("88.8", "531.8")
    # The kill, as its record's report gives it and as BENCHMARKS and the report do: the unit's last process gone
    # 7.6 s after it, systemd's restart 5.1 s later, the warm start listening 92.1 s after that, and a request
    # answered 106.2 s after the kill.
    report = (data.DRILLS / data.KILL / "report.txt").read_text()
    assert f"the first success answered {kill.first_success:.1f} s after the engine kill" in report
    assert (f"{kill.gone:.1f}", f"{kill.restarted - kill.gone:.1f}") == ("7.6", "5.1")
    assert f"{kill.start.at('the server listens'):.1f}" == "92.1"
    assert f"{kill.first_success:.1f}" == "106.2"
    assert len(kill.streams) == 6 and max(kill.streams) < 0.012
    # The claim its captions open with: the warm start most of the recovery, and a start on empty caches six times
    # a warm one.
    assert (kill.first_success - kill.restarted) / kill.first_success > 0.5
    assert round(cold.at(figures.READY) / warm.at(figures.READY)) == 6
    texts = set(text_of(drawn["report-starts.svg"]))
    assert {
        "a request answered, 106.2 s",
        "ready, 88.8 s",
        "ready, 531.8 s",
        "down 12.7 s, then a warm start",
    } <= texts


REPORT = REPO / "docs" / "REPORT.md"
BENCHMARKS = REPO / "docs" / "BENCHMARKS.md"


def shown(doc: Path, name: str) -> tuple[str, str]:
    """Where ``doc`` shows the figure ``name`` with its phone layout: its alt text, and the caption under it."""
    found = re.search(
        rf'<source media="\(max-width: 767px\)" srcset="figures/{name}-narrow\.svg">\n'
        rf'  <img src="figures/{name}\.svg" width="100%" alt="([^"]*)">\n</picture>\n\n(.+?)\n\n',
        doc.read_text(),
        re.S,
    )
    assert found, (doc.name, name)
    return html.unescape(found[1]), " ".join(found[2].split())


def test_the_deployed_results_figures_say_what_their_records_do(drawn: dict[str, Path]) -> None:
    # REPORT (Figures 15, D1 and D2) and BENCHMARKS show each chart with the drawing's own description as its alt
    # text, and a caption that opens with its claim, in the numbers the records give (the tests above hold the
    # claims against the records).
    caps, soak, starts = data.step_caps(), data.soak(), data.starts()
    sessions = {int(c.point.rsplit("-s", 1)[1]) for c in caps}
    [per_round] = {r.replies for c in caps for r in c.rounds}
    worst = max(soak.hours, key=lambda h: h.share)
    aiperf = (data.SOAK.parents[1] / "aiperf-start.txt").read_text()
    [resumed] = re.findall(r"resuming (\d+) trajectory sessions \(burst", aiperf)
    times = {6: "six"}[round(starts["cold"].at(figures.READY) / starts["warm"].at(figures.READY))]
    requirement = f"{figures.REQUIREMENT:.0%}"
    figures_shown = {
        "report-step-cap": (
            "Figure 15",
            "At 256 tokens a step the change had fewer replies over the SLO than at 512 in every round, and no "
            f"configuration met the {requirement} requirement in any.",
            f"(a) Each round's share of its {per_round} replies over that round's SLO, beside "
            f"{sessions.pop()} coding-agent sessions; the dashed line is the requirement.",
        ),
        "report-starts": (
            "Figure D1",
            "The engine's recovery from a kill is mostly its warm start, and on empty caches a start takes "
            f"{times} times as long.",
            "the dashed box is the killed engine down, until systemd starts it again.",
        ),
        "report-soak": (
            "Figure D2",
            f"Over the {len(soak.hours)} hours {soak.share:.1%} of the replies were over the SLO, and only hour "
            f"{worst.hour}, at {worst.share:.1%}, went over the {requirement} requirement.",
            f"{soak.hours[0].tokens:,.0f} in hour 0, as AIPerf resumes its {resumed} sessions at once.",
        ),
    }
    report = REPORT.read_text()
    for name, (number, claim, key) in figures_shown.items():
        description = ET.parse(drawn[f"{name}.svg"]).getroot().find(f"{SVG}desc").text
        for doc, opening in ((REPORT, f"**{number}. {claim}** "), (BENCHMARKS, f"**{claim}** ")):
            alt, caption = shown(doc, name)
            assert alt == description, (doc.name, name)
            assert caption.startswith(opening), (doc.name, caption)
            assert key.lower() in caption.lower(), (doc.name, caption)
        # Besides its caption, the report's text and its Table A1 name each.
        assert len(re.findall(rf"{number}\b", report)) >= 3, number
