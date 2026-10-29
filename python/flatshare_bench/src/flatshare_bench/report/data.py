"""What the report's figures plot, read from the repository's committed records into plain values.

Each value is computed the way the records' own reports compute it, so that a figure agrees with the tables of
docs/BENCHMARKS.md: E2's points through :func:`flatshare_bench.e2.report.load_points` (V2V percentiles from the callers'
turns, a warm-up turn left out, a stage's mean pooled over the replies of a configuration's runs), the probes from the
lines probe.py and tail.py printed, as the campaigns' check.py and table.py read them, the scheduler's cost per step
from the baseline perf_schedule.py measured, and the two-card comparison through its record's own gate.py. A
configuration measured in several campaigns pools its runs, as the report's tables do, and its median is the median
over its runs.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from flatshare_bench.e2.report import (
    configurations,
    load_points,
    read_dots,
    stage_waterfall,
    voice_slo,
)
from flatshare_stats import quantiles

REPO = Path(__file__).resolve().parents[5]
"""The repository, whose committed records the figures read: the package runs from its source tree (make and the
bench's tests run it editable)."""
RESULTS = REPO / "benchmarks" / "results"
W4 = RESULTS / "2026-10-23-e2"
W5 = RESULTS / "2026-10-24-e2-w5"
A100 = RESULTS / "2026-10-26-a100"
PROBES = RESULTS / "2026-10-24-w5-probes" / "pr-probes"
KV_TIMELINE = RESULTS / "2026-10-23-make-room-probes" / "logs" / "gray-kvtimeline.log"
KNEE = RESULTS / "2026-10-25-window-knee"
STEP_COSTS = REPO / "benchmarks" / "perf" / "make-room-schedule.json"
BOX_HOST = REPO / "crates" / "flatshare-sim" / "testdata" / "box-host.json"
"""The box's host as bench_host.py measured it: its CPU's time at the Python workload of the scheduler's kind."""
VOICE_PATHS = RESULTS / "2026-10-25-w6-transport" / "llm-paths.json"
"""W6's voice replies on the box, straight to the engine and through the gateway: the gaps between a reply's tokens."""
TWO_CARDS = RESULTS / "2026-10-28-rounds"
"""The two-card comparison's rounds. Its gate.py is the comparison's single source of truth, and the figure reads only
the rounds that gate.py's verdict counts (its ROUNDS), whatever else the record holds."""

# W5's configurations in the discriminating scenario, by their points' names: each pools its runs over the campaigns
# that ran it (tuned vLLM's W5 and PR campaigns, the change's W5, PR and extension campaigns).
VOICE_ALONE = "w5-a1-voice"
TUNED = "w5-disc-frozen-s24"
ROUTER = "w5-disc-a8c-resubmit-s24"
MAKE_ROOM = "w5-disc-pr-a-s24"
HOLD_16 = "w5-disc-pr-ab-s24"
CHANGE = "w5-disc-pr-ab48-s24"
STANDARD = ("w5-std-frozen-s16", "w5-std-pr-ab48-s16")
"""Tuned vLLM and the change in the standard scenario, 16 agent sessions of the whole corpus (the extension)."""
CALLS = {2: (VOICE_ALONE, CHANGE)} | {
    n: (f"w5-a1-voice-c{n}", f"w5-disc-pr-ab48-c{n}-s24") for n in (4, 6)
}
"""Calls per GPU: for each number of callers, the voice alone that sets its SLO and the change beside the 24 agent
sessions."""
A100_CAMPAIGNS = ("a100-r", "a100-tuned-r", "a100-prtuned-r")
"""The A100's core rounds and the two rounds at vLLM's default 2,048-token step cap (A' is a workload of its own)."""
A100_POINTS = {
    512: ("w5-disc-frozen-s24", "w5-disc-pr-ab48-s24"),
    2048: ("a100-tuned-s24", "a100-pr-ab48-2048-s24"),
}
"""On the A100, by the step cap: tuned vLLM's point and the change's."""
DISCRIMINATING = "weka-x32-long, 2 callers"
"""W4's workload of the frontier figure: the 48 traces with the largest contexts, 24 sessions, two callers."""


def _median(values: Iterable[float]) -> float:
    return statistics.median(list(values))


def _round(campaign: str) -> int:
    """A campaign's round, its directory's suffix: w5-pr-r3 is round 3, which replays the agents' sample of seed 13."""
    return int(campaign.rsplit("-r", 1)[1])


def _points(pattern: str, root: Path) -> list[dict[str, Any]]:
    rows, incomplete = load_points(sorted(path.parent for path in root.glob(pattern)))
    if incomplete or not rows:
        raise ValueError(f"incomplete records under {root}: {incomplete}")
    return rows


def w5_rows(root: Path = W5) -> list[dict[str, Any]]:
    """Every point of W5's records, as E2's report reads them."""
    return _points("*/*/run.json", root)


def _runs(rows: Sequence[Mapping[str, Any]], point: str) -> list[Mapping[str, Any]]:
    runs = [row for row in rows if row["point"] == point]
    if not runs:
        raise ValueError(f"no run of {point}")
    return sorted(
        runs, key=lambda row: (row["campaign"].rsplit("-r", 1)[0], _round(row["campaign"]))
    )


def slo_ms(rows: Sequence[Mapping[str, Any]], callers: int = 2) -> float:
    """The voice SLO of a set of records: their voice alone's median V2V p95 with as many callers, plus 250 ms."""
    found = voice_slo(configurations(rows), callers, rows[0].get("hardware", ""))
    if found is None:
        raise ValueError(f"no voice alone with {callers} callers")
    return found[0]


# The voice turn by stage --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stages:
    """A configuration's voice turn from the end of the caller's speech, each stage's mean in ms over the replies of
    its runs: endpointing, the wait for the LLM's first token (the time to the reply's first sentence less its
    decode), the decode of the first sentence (from the LLM's first token to the sentence), TTS's first audio, and
    transport and playout, the rest of the mean V2V, which they add up to."""

    point: str
    runs: int
    replies: int
    endpointing: float
    wait: float
    decode: float
    tts: float
    transport: float
    v2v: float

    @property
    def parts(self) -> tuple[float, float, float, float, float]:
        return (self.endpointing, self.wait, self.decode, self.tts, self.transport)


def _pooled(runs: Iterable[Mapping[str, Any]], stage: str) -> float:
    """A stage's mean over the replies of several runs: each run's mean weighted by its count."""
    counted = [run["stage_means"][stage] for run in runs]
    return sum(n * mean for n, mean in counted) / sum(n for n, _ in counted)


def stages(rows: Sequence[Mapping[str, Any]] | None = None) -> list[Stages]:
    """Voice alone, tuned vLLM, make-room and the change, each pooled over its runs as the stage waterfall pools them
    (:func:`flatshare_bench.e2.report.stage_waterfall`)."""
    rows = w5_rows() if rows is None else rows
    found = []
    for point in (VOICE_ALONE, TUNED, MAKE_ROOM, CHANGE):
        runs = _runs(rows, point)
        [cell] = stage_waterfall(runs)
        decode = _pooled(runs, "llm_first_sentence_ms") - _pooled(runs, "llm_ttft_ms")
        found.append(
            Stages(
                point,
                cell["runs"],
                cell["replies"],
                cell["end_of_turn_ms"],
                cell["turn_to_first_sentence_ms"] - decode,
                decode,
                cell["tts_first_audio_ms"],
                cell["transport_ms"],
                cell["v2v_mean_ms"],
            )
        )
    return found


# The two waits in isolation -----------------------------------------------------------------------------------------

PROBE = re.compile(
    r"^(?P<case>kv|prefill): priority-0 TTFT (?P<ttft>\d+) ms, 16 tokens (?:(?P<t16>\d+) ms|—), "
    r"for (?P<prompt>\d+) prompt tokens, sent with (?P<agents>\d+) priority-2 requests of (?P<agent_tokens>\d+) "
    r"tokens running and (?P<free>\d+) of (?P<blocks>\d+) blocks free; after it, \d+ running and \d+ waiting$"
)
"""A line of probe.py's KV or prefill case (as the probes' check.py reads it). A reply that ended before its 16th
token prints a dash there."""
ENGINES = ("stock", "a", "ab")
"""The probe blocks, in the figure's order: vLLM 0.32.0, make-room alone (#63350) and both rules (#63351 too), each
at the budget cap's default 16-token window."""


@dataclass(frozen=True)
class Probe:
    """One run of a probe case: the voice turn's first token and 16th token in ms from its request (None where the
    reply ended before it), and what it was sent beside."""

    ttft: int
    t16: int | None
    prompt_tokens: int
    agents: int
    agent_tokens: int
    free_blocks: int
    blocks: int


def probes(case: str, root: Path = PROBES) -> dict[str, list[Probe]]:
    """The runs of the KV case (``kv``) or the prefill case (``prefill``) by engine, from the probe blocks' logs. A
    line the pattern does not read is an error, as in check.py."""
    found = {}
    for engine in ENGINES:
        runs = []
        for line in (root / f"{engine}-{case}.log").read_text().splitlines():
            if not line.strip():
                continue
            match = PROBE.match(line)
            if not match or match["case"] != case:
                raise ValueError(f"{engine}-{case}.log: an unread line: {line!r}")
            runs.append(
                Probe(
                    int(match["ttft"]),
                    int(match["t16"]) if match["t16"] else None,
                    *(
                        int(match[k])
                        for k in ("prompt", "agents", "agent_tokens", "free", "blocks")
                    ),
                )
            )
        found[engine] = runs
    return found


# The KV pool over time ----------------------------------------------------------------------------------------------

POOL_BLOCKS = 2274
"""The frozen box's KV pool: 2,275 blocks of 16 tokens less vLLM's null block, as probe.py counts them (its lines
print them)."""
KV_FREE_BLOCKS = 24
"""probe.py's FREE_BLOCKS: the KV case sends its voice turn once its agent requests all run with at most this many
blocks free."""
POOL_FULL = 0.98
"""The pool counts as full from 98% of its blocks in use: the KV case's two prompts of 17,999 tokens alone take 2,250
of its 2,274 blocks."""
SAMPLE = re.compile(
    r"^ *(?P<t>\d+\.\d)s usage (?P<usage>\d\.\d+) running (?P<running>\d+) waiting (?P<waiting>\d+) "
    r"preemptions (?P<preemptions>\d+) tokens \[(?P<a>\d+), (?P<b>\d+)\]$"
)
"""A line of the timeline: the seconds since the agent requests were sent, the share of the pool's blocks in use, the
requests running and waiting, the engine's preemption counter, and each agent request's output tokens."""


@dataclass(frozen=True)
class PoolSample:
    """One read of the engine's metrics while the KV case's two agent requests run: the seconds since they were
    sent, the share of the pool's blocks in use, the requests running and waiting, the engine's preemption counter,
    and each request's output tokens (agent A's, agent B's)."""

    t: float
    usage: float
    running: int
    waiting: int
    preemptions: int
    tokens: tuple[int, int]

    @property
    def free_blocks(self) -> int:
        """The blocks free, as probe.py derives them from the usage."""
        return round((1 - self.usage) * POOL_BLOCKS)


def kv_timeline(path: Path = KV_TIMELINE) -> list[PoolSample]:
    """The samples of the block after the log's ``timeline`` line, up to its next bracketed line: the frozen box under
    the KV case's load. A line in the block the pattern does not read is an error."""
    lines = path.read_text().splitlines()
    starts = [k for k, line in enumerate(lines) if re.fullmatch(r"\[[\d:]+Z\] timeline", line)]
    if len(starts) != 1:
        raise ValueError(f"{path.name}: {len(starts)} timeline blocks, not one")
    samples = []
    for line in itertools.takewhile(lambda line: not line.startswith("["), lines[starts[0] + 1 :]):
        match = SAMPLE.match(line)
        if not match:
            raise ValueError(f"{path.name}: an unread line in the timeline: {line!r}")
        samples.append(
            PoolSample(
                float(match["t"]),
                float(match["usage"]),
                int(match["running"]),
                int(match["waiting"]),
                int(match["preemptions"]),
                (int(match["a"]), int(match["b"])),
            )
        )
    if not samples:
        raise ValueError(f"{path.name}: an empty timeline")
    return samples


@dataclass(frozen=True)
class PoolMoments:
    """The timeline's moments: the first sample with the pool full, the first at which the KV case would send its
    voice turn (every agent request running and at most KV_FREE_BLOCKS free), and the samples either side of the
    preemption, the last with both requests running and the first with one."""

    full: PoolSample
    send: PoolSample
    before: PoolSample
    after: PoolSample


def kv_moments(samples: Sequence[PoolSample]) -> PoolMoments:
    agents = max(sample.running for sample in samples)
    full = next(sample for sample in samples if sample.usage >= POOL_FULL)
    send = next(
        sample
        for sample in samples
        if sample.running == agents and sample.free_blocks <= KV_FREE_BLOCKS
    )
    before, after = next((a, b) for a, b in itertools.pairwise(samples) if b.running < a.running)
    return PoolMoments(full, send, before, after)


# Both waits beside closed-loop agents ---------------------------------------------------------------------------------

TURN = re.compile(r"^\d\d:\d\d:\d\d\.\d{3} call \d (?:ttft +(?P<ttft>\d+) ms|failed)$")
TAIL_SUMMARY = re.compile(
    r"^calls: (?P<turns>\d+) turns, (?P<failed>\d+) failed \(counted at 30,000 ms\), TTFT p50 \d+ p90 \d+ p95 \d+ "
    r"max \d+ ms; agents: \d+ of \d+ stopped early$"
)
"""tail.py's lines: a call turn's time to its first token (or its failure), and the run's summary (as the probes'
check.py reads them)."""
TAIL_TIMEOUT_MS = 30_000.0
"""tail.py's: a failed turn counts at it."""
MODES = ("fresh", "grow")
"""tail.py's agent prompts: fresh each turn, or each grown by a tool result."""


@dataclass(frozen=True)
class Tail:
    """The calls' turns beside the closed-loop agents of one probe block with one kind of agent prompt, pooled over
    its runs: each turn's time to its first token in ms, sorted, a failed turn at tail.py's 30 s."""

    engine: str
    mode: str
    runs: int
    turns: tuple[float, ...]
    failed: int

    def percentile(self, p: int) -> float:
        """The inclusive percentile, as E2's report and the probes' tables compute it."""
        return quantiles(self.turns)[p - 1]


def tails(root: Path = PROBES) -> dict[str, dict[str, Tail]]:
    """By prompt mode, then by engine, the turns of tail.py's runs. A line neither a turn nor a summary, or a summary
    whose counts its turns do not give, is an error."""
    found: dict[str, dict[str, Tail]] = {}
    for mode in MODES:
        found[mode] = {}
        for engine in ENGINES:
            paths = sorted(root.glob(f"{engine}-tail-{mode}-[0-9].log"))
            if not paths:
                raise ValueError(f"no tail run of {engine} with {mode} prompts")
            turns: list[float] = []
            failures = 0
            for path in paths:
                run, summaries = [], []
                for line in path.read_text().splitlines():
                    if match := TURN.match(line):
                        run.append(match["ttft"])
                    elif match := TAIL_SUMMARY.match(line):
                        summaries.append(match)
                    elif line.strip():
                        raise ValueError(f"{path.name}: an unread line: {line!r}")
                failed = sum(ttft is None for ttft in run)
                if len(summaries) != 1 or (
                    int(summaries[0]["turns"]),
                    int(summaries[0]["failed"]),
                ) != (
                    len(run),
                    failed,
                ):
                    raise ValueError(
                        f"{path.name}: its turns do not give its summary line's counts"
                    )
                turns += [TAIL_TIMEOUT_MS if ttft is None else float(ttft) for ttft in run]
                failures += failed
            found[mode][engine] = Tail(engine, mode, len(paths), tuple(sorted(turns)), failures)
    return found


# Sizing the window ------------------------------------------------------------------------------------------------


def reply_tokens(root: Path = W5) -> list[int]:
    """The output tokens of every voice reply W5's gateway completed (replies.tsv, from the raw records' request
    logs), the warm-up turns' among them."""
    with (root / "replies.tsv").open() as tsv:
        return [int(row["completion_tokens"]) for row in csv.DictReader(tsv, delimiter="\t")]


def share_within(tokens: Sequence[int], window: int) -> float:
    """The share of the replies that end within ``window`` tokens."""
    return sum(t <= window for t in tokens) / len(tokens)


REPLIES = re.compile(
    r"^replies: \d+ turns, 16 tokens p50 \d+ p90 \d+ p95 (?P<t16>\d+) max \d+ ms, "
    r"40 tokens p50 (?P<t40_p50>\d+) p90 \d+ p95 (?P<t40>\d+) max \d+ ms$"
)
AGENTS = re.compile(
    r"^agents: \d+ requests in \d+ s, [\d.]+ a second, with (?P<prompt>[\d,]+) prompt and "
)
CALLS_LINE = re.compile(
    r"^calls: (?P<turns>\d+) turns, (?P<failed>\d+) failed .*TTFT p50 \d+ p90 \d+ p95 (?P<ttft>\d+) max \d+ ms"
)
"""tail.py's summary lines in the window sweep's outputs, as its table.py reads them."""


@dataclass(frozen=True)
class KneeRun:
    """One 240 s run of the window sweep: the calls' 40-token replies at a budget-cap window of ``window`` tokens,
    their first token's, 16th's and 40th's p95 in ms, and the agents' prompt tokens a second beside them."""

    window: int
    run: int
    ttft_p95: int
    t16_p95: int
    t40_p95: int
    prompt_tps: int
    turns: int
    failed: int


def knee(root: Path = KNEE) -> list[KneeRun]:
    """Every run of the window sweep (wWINDOW-RUN.out), by window then run."""
    found = []
    for path in sorted(
        root.glob("w*-[0-9].out"), key=lambda p: tuple(map(int, p.stem[1:].split("-")))
    ):
        window, run = map(int, path.stem[1:].split("-"))
        values: dict[str, int] = {}
        for line in path.read_text().splitlines():
            for pattern in (REPLIES, AGENTS, CALLS_LINE):
                if match := pattern.match(line):
                    values |= {k: int(v.replace(",", "")) for k, v in match.groupdict().items()}
        if not {"t16", "t40", "prompt", "ttft"} <= values.keys():
            raise ValueError(f"{path.name}: no summary line")
        found.append(
            KneeRun(
                window,
                run,
                values["ttft"],
                values["t16"],
                values["t40"],
                values["prompt"],
                values["turns"],
                values["failed"],
            )
        )
    return found


def knee_medians(runs: Sequence[KneeRun], field: str) -> dict[int, float]:
    """A field's median over each window's runs, by window."""
    windows = sorted({run.window for run in runs})
    return {w: _median(getattr(run, field) for run in runs if run.window == w) for w in windows}


# The rules' cost per step -----------------------------------------------------------------------------------------

SCENARIOS = {
    "A": "only lower-priority work",
    "B": "a waiting turn needs more room than two victims free",
    "C": "every running slot taken",
    "idle": "only agents",
    "hold": "a reply's first tokens, an agent waiting that fits",
    "nofit": "a waiting turn needs more room than two victims free",
    "fire": "the step that preempts for a waiting turn",
}
"""perf_schedule.py's states, as the figure names them (python/flatshare_vllm/tests/perf_schedule.py): the steady
states of the PRs' cost tables, A to C, then the serving profile's."""


@dataclass(frozen=True)
class StepCost:
    """A state's cost of one call of Scheduler.schedule() on the host, in µs: stock vLLM 0.32.0's scheduler,
    make-room's with its switches off, and make-room as deployed (two victims a step, the hold with its 48-token
    window, no reserve)."""

    key: str
    scenario: str
    running: int
    variant: str
    stock: float
    off: float
    on: float

    @property
    def added(self) -> float:
        """What the deployed switches add to the step, in µs (the baseline's ``change_us``)."""
        return self.on - self.off

    @property
    def off_ratio(self) -> float:
        """What the switches off cost against stock vLLM, as a fraction of its step."""
        return self.off / self.stock - 1

    @property
    def serving(self) -> bool:
        """Whether the state is the serving profile's, which schedules asynchronously in 256-token steps, rather than
        one of the PRs' cost tables' states."""
        return self.scenario not in ("A", "B", "C")


STATE = re.compile(
    r"^(?P<scenario>[ABC]|idle|hold|nofit|fire)(?:-R)?(?P<running>\d+)(?:-(?P<variant>.+))?$"
)
VARIANTS = {
    "": "",
    "hit999": "a 999-block prefix hit",
    "nohit": "no prefix hit",
    "W10": "10 waiting",
}
"""A state's variant, from its key's suffix: the waiting head's prefix hit (127 blocks unless named) and the waiting
requests (1,000 unless named; the hit states have 100)."""


def box_cpu(path: Path = STEP_COSTS, box: Path = BOX_HOST) -> float:
    """How many times as long the box's CPU takes as the host that measured the step costs, at bench_host.py's Python
    workload: the box's median time there (box-host.json) over the host's, the median of its times as each of the
    baseline's rounds ended."""
    box_ms = json.loads(box.read_text())["python_ms"]["p50"]
    return box_ms / _median(
        load["python_ms"]["p50"] for load in json.loads(path.read_text())["load"]
    )


def voice_step_ms(path: Path = VOICE_PATHS) -> float:
    """A voice reply's decode step on the box with nothing beside it: the median gap between its tokens, straight to
    the engine (W6)."""
    return json.loads(path.read_text())["summary"]["direct"]["gap_p50_ms"]


AGENTS_STEP_POINT = "sweep-disc-pr-ab48-b256-t0-s24"
"""The serving profile's configuration beside both calls and 24 session trees, in each of the sweep's rounds."""


def agents_step_ms(root: Path = TWO_CARDS) -> float:
    """The agents' step on the box in the serving profile beside both calls at 24 session trees: the mean gap between
    their tokens, the median over the rounds that gate.py's verdict counts (its ROUNDS)."""
    gaps = []
    for name in two_cards_gate(root).ROUNDS:
        summary = json.loads((root / name / AGENTS_STEP_POINT / "summary.json").read_text())
        gaps.append(summary["background"]["itl_ms"]["mean"])
    return _median(gaps)


def step_costs(path: Path = STEP_COSTS) -> list[StepCost]:
    """The baseline's cells in its order: each state's cost a step, stock, off and on."""
    found = []
    for key, cell in json.loads(path.read_text())["cells"].items():
        match = STATE.match(key)
        if not match:
            raise ValueError(f"an unread state: {key}")
        found.append(
            StepCost(
                key,
                match["scenario"],
                int(match["running"]),
                VARIANTS[match["variant"] or ""],
                cell["stock_us"],
                cell["off_us"],
                cell["on_us"],
            )
        )
    return found


# The headline, and every turn ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One run of a configuration: its campaign and round, its V2V p95 in ms, the agents' tokens a second over the
    calls' first 240 s (None without agents), and the V2V samples of its measured turns, in ms."""

    campaign: str
    round: int
    p95: float
    tokens: float | None
    samples: tuple[float, ...]


@dataclass(frozen=True)
class Configuration:
    point: str
    runs: tuple[Run, ...]

    @property
    def p95(self) -> float:
        """The median of the runs' V2V p95, as frontier.csv's configurations take it."""
        return _median(run.p95 for run in self.runs)

    @property
    def tokens(self) -> float | None:
        measured = [run.tokens for run in self.runs if run.tokens is not None]
        return _median(measured) if measured else None

    def samples(self) -> list[float]:
        """Every measured turn's V2V over the runs, sorted."""
        return sorted(sample for run in self.runs for sample in run.samples)

    def over(self, threshold: float) -> int:
        """How many of the runs' measured turns took longer than ``threshold`` ms: with the SLO's threshold, the
        replies that missed it."""
        return sum(sample > threshold for sample in self.samples())

    @property
    def pooled_p95(self) -> float:
        """The p95 of the runs' measured turns taken together, by the method the records' own p95 uses
        (:func:`flatshare_stats.quantiles`), beside :attr:`p95`, the median of the runs' p95."""
        return quantiles(self.samples())[94]


def configuration(rows: Sequence[Mapping[str, Any]], point: str) -> Configuration:
    return Configuration(
        point,
        tuple(
            Run(
                row["campaign"],
                _round(row["campaign"]),
                row["v2v_p95_ms"],
                row["fixed_window_tokens_per_s"] if row["sessions"] else None,
                tuple(row["v2v_samples"]),
            )
            for row in _runs(rows, point)
        ),
    )


def headline(
    rows: Sequence[Mapping[str, Any]] | None = None, root: Path = W5
) -> list[Configuration]:
    """Voice alone, then the rows of W5's figure.toml in its order."""
    rows = w5_rows(root) if rows is None else rows
    [group] = read_dots(root / "figure.toml")
    return [
        configuration(rows, point) for point in (VOICE_ALONE, *(r["point"] for r in group["rows"]))
    ]


def paired_retention(
    rows: Sequence[Mapping[str, Any]], point: str, against: str
) -> dict[str, float]:
    """``point``'s agents' tokens a second over the calls' first 240 s as a share of ``against``'s, in each campaign
    that ran both on the same sample of the agents, by campaign."""
    mine = {run.campaign: run.tokens for run in configuration(rows, point).runs}
    theirs = {run.campaign: run.tokens for run in configuration(rows, against).runs}
    return {
        campaign: mine[campaign] / theirs[campaign]
        for campaign in sorted(mine.keys() & theirs.keys())
        if mine[campaign] is not None and theirs[campaign]
    }


@dataclass(frozen=True)
class AgentTokens:
    """The agents' tokens of one run over the whole call, a second, from its summary: the prompt tokens the engine
    processed for them and the tokens it generated."""

    campaign: str
    prompt: float
    completion: float

    @property
    def generated_share(self) -> float:
        """The generated tokens' share of all the agents' tokens."""
        return self.completion / (self.prompt + self.completion)


def agent_tokens(point: str, root: Path = W5) -> list[AgentTokens]:
    """Each run of ``point``'s agents' tokens over the whole call."""
    found = []
    for path in sorted(root.glob(f"*/{point}/summary.json")):
        background = json.loads(path.read_text())["background"]
        found.append(
            AgentTokens(
                path.parts[-3],
                background["prompt_tokens_per_s"],
                background["completion_tokens_per_s"],
            )
        )
    if not found:
        raise ValueError(f"no run of {point}")
    return found


# The frontier -------------------------------------------------------------------------------------------------------

FAMILIES = {
    "tuned": ("4", "5"),
    "operator": ("6",),
    "engine": ("7a", "7b"),
    "router": ("8a", "8b"),
    "resubmit": ("8c",),
    "lifecycle": ("9-gateway-only",),
    "change": ("9-pr-a", "9-pr-ab", "9-pr-ab48"),
    "lifecycle-change": ("9",),
}
"""The frontier figure's families of W4's arms (docs/EXPERIMENTS.md), by the arms E2's report gives the points: vLLM
tuned by its own settings (arm 4, priority scheduling with a step cap, the frozen box's 512/256 among them, and arm 5,
vLLM's other knobs); the operator's static answers in the gateway (6); what engines ship (7: vLLM with #54530, SGLang
with and without HiCache); the routers (8a, llm-d's flow control emulated; 8b, SMG), the abort-and-resubmit router
(8c); the lifecycle policy in front of tuned vLLM (arm 9's gateway-only ablation); the change as the PRs propose it
(make-room alone, both rules at 16 and at 48 tokens, all on W4's v0.3 variant); and arm 9, the lifecycle policy over
that variant with its KV reserve on, a 16-token window and every reply capped at 16 tokens. Arms 1-3
ran no point in the discriminating scenario (voice alone, which sets the SLO, ran without agents). Arm 9's other
ablations, each arm 9 less one mechanism, are not configurations to deploy and are left out."""
FAMILY = {arm: key for key, arms in FAMILIES.items() for arm in arms}


@dataclass(frozen=True)
class FrontierPoint:
    """A configuration of W4's discriminating scenario: its arm and family, its runs' V2V p95 and the agents' tokens a
    second over the calls' first 240 s (the measure every figure of the report gives, over a window of the same length
    for every configuration), and its failed turns, which censor its V2V at the reply timeout."""

    arm: str
    point: str
    family: str
    p95s: tuple[float, ...]
    tokens_per_run: tuple[float, ...]
    failed_turns: int

    @property
    def p95(self) -> float:
        return _median(self.p95s)

    @property
    def tokens(self) -> float:
        return _median(self.tokens_per_run)


def w4_rows(root: Path = W4) -> list[dict[str, Any]]:
    return _points("*/*/run.json", root)


def frontier(rows: Sequence[Mapping[str, Any]] | None = None) -> list[FrontierPoint]:
    """Every configuration of W4's discriminating scenario that FAMILIES places, in the order of E2's report."""
    rows = w4_rows() if rows is None else rows
    return [
        FrontierPoint(
            group["arm"],
            group["point"],
            FAMILY[group["arm"]],
            tuple(run["v2v_p95_ms"] for run in group["runs"]),
            tuple(run["fixed_window_tokens_per_s"] for run in group["runs"]),
            group["failed_turns"],
        )
        for group in configurations(rows)
        if group["workload"] == DISCRIMINATING and group["arm"] in FAMILY
    ]


# Where the margin narrows -------------------------------------------------------------------------------------------


def standard_rounds(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[int, float, float]]:
    """The standard scenario's rounds: each round, tuned vLLM's V2V p95 and the change's, in ms."""
    rows = w5_rows() if rows is None else rows
    tuned, change = (configuration(rows, point) for point in STANDARD)
    by_round = {run.round: run.p95 for run in change.runs}
    return [(run.round, run.p95, by_round[run.round]) for run in tuned.runs]


@dataclass(frozen=True)
class Calls:
    """The change with ``callers`` calls beside the 24 agent sessions: its SLO (that many callers' voice alone, plus
    250 ms), and per run its V2V p95 over the SLO, the share of the engine's steps in which the hold held agent
    prefill back (e3.tsv), and the agents' tokens a second over the calls' first 240 s."""

    callers: int
    slo: float
    over: tuple[float, ...]
    held: tuple[float, ...]
    tokens: tuple[float, ...]


def e3_counts(root: Path = W5) -> dict[tuple[str, str], Mapping[str, str]]:
    """e3.tsv's counts of each point with agents, by (campaign, point)."""
    with (root / "e3.tsv").open() as tsv:
        return {(r["campaign"], r["point"]): r for r in csv.DictReader(tsv, delimiter="\t")}


def calls(rows: Sequence[Mapping[str, Any]] | None = None, root: Path = W5) -> list[Calls]:
    """Two, four and six callers, each against its own voice alone: four and six from the calls-per-GPU rounds, and two
    from the discriminating campaigns' runs of the change on the same samples of the agents (e3.tsv's agents_seed), so
    that the caller counts compare on the same agents."""
    rows = w5_rows(root) if rows is None else rows
    counts = e3_counts(root)
    groups = configurations(rows)
    seeds = {
        counts[(row["campaign"], point)]["agents_seed"]
        for callers, (_, point) in CALLS.items()
        if callers > 2
        for row in _runs(rows, point)
    }
    found = []
    for callers, (alone, point) in CALLS.items():
        reference, points = voice_slo(groups, callers) or (None, [])
        if points != [alone]:
            raise ValueError(f"{callers} callers: the SLO's voice alone is {points}, not {alone}")
        runs = [
            row
            for row in _runs(rows, point)
            if counts[(row["campaign"], point)]["agents_seed"] in seeds
        ]
        held = []
        for run in runs:
            count = counts[(run["campaign"], point)]
            held.append(int(count["held_steps"]) / int(count["engine_steps"]))
        found.append(
            Calls(
                callers,
                reference,
                tuple(run["v2v_p95_ms"] - reference for run in runs),
                tuple(held),
                tuple(run["fixed_window_tokens_per_s"] for run in runs),
            )
        )
    return found


# What the agents pay -----------------------------------------------------------------------------------------------

AGENT_POINTS = (TUNED, ROUTER, MAKE_ROOM, HOLD_16, CHANGE)
"""The headline's configurations that run agents, in its order."""


@dataclass(frozen=True)
class AgentRun:
    """One run of a configuration beside the 24 agent sessions: the agents' TTFT p50 and p99 in ms and their tokens a
    second over the calls' first 240 s (as frontier.csv gives them), and e3.tsv's counts over the calls: the engine's
    steps, the steps in which the prefill hold kept agent prefill out (None without the hold's counter) and
    make-room's own preemptions (None without make-room)."""

    campaign: str
    ttft_p50: float
    ttft_p99: float
    tokens: float
    steps: int
    held: int | None
    preemptions: int | None

    @property
    def held_share(self) -> float | None:
        """The share of the engine's steps the hold held agent prefill in."""
        return None if self.held is None else self.held / self.steps


def _count(cell: str) -> int | None:
    return int(cell) if cell else None


def agent_runs(
    rows: Sequence[Mapping[str, Any]] | None = None, root: Path = W5
) -> dict[str, list[AgentRun]]:
    """By configuration, its runs' agents' TTFT and tokens joined with their e3.tsv counts by campaign and point."""
    rows = w5_rows(root) if rows is None else rows
    counts = e3_counts(root)
    found = {}
    for point in AGENT_POINTS:
        runs = []
        for row in _runs(rows, point):
            count = counts[(row["campaign"], point)]
            runs.append(
                AgentRun(
                    row["campaign"],
                    row["agent_ttft_p50_ms"],
                    row["agent_ttft_p99_ms"],
                    row["fixed_window_tokens_per_s"],
                    int(count["engine_steps"]),
                    _count(count["held_steps"]),
                    _count(count["make_room_preemptions"]),
                )
            )
        found[point] = runs
    return found


# The A100 -----------------------------------------------------------------------------------------------------------


def a100_rows(root: Path = A100) -> list[dict[str, Any]]:
    """The A100's core rounds, and tuned vLLM's and the change's at the 2,048-token step cap."""
    rows, incomplete = load_points(
        sorted(
            path.parent
            for campaign in A100_CAMPAIGNS
            for path in root.glob(f"e2/{campaign}[0-9]/*/run.json")
        )
    )
    if incomplete or not rows:
        raise ValueError(f"incomplete records under {root}: {incomplete}")
    return rows


def a100(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[int, tuple[Configuration, Configuration]]:
    """By the step cap, tuned vLLM's configuration and the change's on the A100."""
    rows = a100_rows() if rows is None else rows
    return {
        cap: (configuration(rows, tuned), configuration(rows, change))
        for cap, (tuned, change) in A100_POINTS.items()
    }


# The two cards ------------------------------------------------------------------------------------------------------

CONSOLIDATED = {
    "stock": "Consolidation, stock vLLM",
    "change": "Consolidation, the change at a step cap of 256",
}
"""gate.py's consolidated card (its CONSOLIDATION) by engine: stock vLLM, and the change at the serving profile's step
cap of 256 tokens."""
POOLED = {"stock": "Stock pooling", "change": "Flatshare pooling"}
"""gate.py's pooling (its POOLING) by engine: one call a card beside agents on each, on stock vLLM and on the change."""


def two_cards_gate(root: Path = TWO_CARDS) -> ModuleType:
    """The record's gate.py, loaded from its file as a module: the rounds its verdict counts, its arrangements' cards
    and the functions that give its verdict."""
    spec = importlib.util.spec_from_file_location("rounds_gate", root / "gate.py")
    if spec is None or spec.loader is None:
        raise ValueError(f"no gate.py under {root}")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


@dataclass(frozen=True)
class Consolidated:
    """The consolidated card beside both calls at one number of agent sessions, on one engine: how many of its replies
    took longer than their round's SLO, and how many it made, over the rounds pooled."""

    engine: str
    sessions: int
    over: int
    replies: int

    @property
    def share(self) -> float:
        """The share of its replies over the SLO, which the requirement bounds."""
        return self.over / self.replies


@dataclass(frozen=True)
class Arrangement:
    """One arrangement of the two cards as gate.py measures it: its name in gate.py's tables, the key of its kind of
    card there (an AGENTS_ALONE, CONSOLIDATION or POOLING name), its kind ("dedicated", "consolidation" or "pooling")
    and engine ("stock" or "change"), the agent sessions on each card, the card with the calls first, whether the
    second card is a warm spare, its replies over their round's SLO and its replies, over the rounds pooled, whether
    those meet the requirement, and the agents' tokens a second over the calls' first 240 s summed over both cards, in
    each round in gate.py's order and at their median."""

    name: str
    key: str
    kind: str
    engine: str
    sessions: tuple[int, int]
    spare: bool
    over: int
    replies: int
    qualifies: bool
    rounds: tuple[float, ...]
    tokens: float


@dataclass(frozen=True)
class TwoCards:
    """The comparison: the rounds its verdict counts, its requirement (the share of the replies over the SLO an
    arrangement may have at most) and its hurdle (how much more background work than the best qualifying dedicated
    arrangement a qualifying shared one must do), each engine's consolidated card by its sessions, fewest first, and
    its envelope (the most sessions at which it qualifies), the best qualifying dedicated arrangement, and the
    arrangements in the figure's order."""

    rounds: tuple[str, ...]
    requirement: float
    hurdle: float
    consolidated: Mapping[str, tuple[Consolidated, ...]]
    envelopes: Mapping[str, int]
    best: Arrangement
    arrangements: tuple[Arrangement, ...]

    @property
    def threshold(self) -> float:
        """The background work a qualifying shared arrangement must reach for sharing to have resource value: the
        hurdle over the best dedicated arrangement's."""
        return (1 + self.hurdle) * self.best.tokens


def _sessions(
    points: Mapping[tuple[str, str], Mapping[str, Any]], rounds: Sequence[str], point: str
) -> int:
    """A point's agent sessions, which every round ran it with."""
    found = {points[c, point]["sessions"] for c in rounds if (c, point) in points}
    if len(found) != 1:
        raise ValueError(f"{point}: agent sessions {sorted(found)} over the rounds, not one number")
    return found.pop()


def two_cards(root: Path = TWO_CARDS) -> TwoCards:
    """The comparison over the rounds gate.py counts, by gate.py's own functions: each engine's consolidated card at
    every number of sessions the rounds measured; then the dedicated arrangements, each card of agents at its busiest
    and the best practical first; consolidation at each engine's envelope beside the best practical card, the
    change's also with a warm spare as its second card and at the fewest sessions past its envelope; and pooling on
    each engine. The dedicated cards run stock vLLM, at its defaults or as the shared card runs it."""
    gate = two_cards_gate(root)
    rounds = gate.ROUNDS
    points = gate.rows(root, rounds)
    slo = gate.slos(points, rounds)

    def arrangement(
        name: str,
        key: str,
        kind: str,
        engine: str,
        cards: Sequence[tuple[str, int]],
        *,
        spare: bool = False,
    ) -> Arrangement:
        over, replies = gate.foreground(points, slo, cards, rounds)
        median, sums = gate.background(points, cards, rounds)
        if median is None or len(sums) != len(rounds):
            raise ValueError(f"{name}: a round did not measure each of its cards")
        calls, agents = (_sessions(points, rounds, point) for point, n in cards for _ in range(n))
        qualifies = replies > 0 and over <= gate.REQUIREMENT * replies
        return Arrangement(
            name,
            key,
            kind,
            engine,
            (calls, agents),
            spare,
            over,
            replies,
            qualifies,
            tuple(sums),
            median,
        )

    def busiest(key: str, sizes: Sequence[tuple[int, str]]) -> tuple[int, str]:
        found = gate.busiest(points, sizes, rounds)
        if found is None:
            raise ValueError(f"{key}: no round measured it")
        return found

    dedicated = {}
    for key, sizes in gate.AGENTS_ALONE.items():
        sessions, point = busiest(key, sizes)
        dedicated[key] = arrangement(
            f"{key}, {sessions} session trees",
            key,
            "dedicated",
            "stock",
            ((gate.VOICE_ALONE[2], 1), (point, 1)),
        )
    # gate.py's best, the first of the most where two tie, in its order.
    best = max((a for a in dedicated.values() if a.qualifies), key=lambda a: a.tokens)
    second = busiest(gate.BEST_PRACTICAL, gate.AGENTS_ALONE[gate.BEST_PRACTICAL])[1]
    consolidated, envelopes, shared = {}, {}, []
    for engine, key in CONSOLIDATED.items():
        sizes = gate.CONSOLIDATION[key]
        consolidated[engine] = tuple(
            Consolidated(engine, sessions, *gate.foreground(points, slo, ((point, 1),), rounds))
            for sessions, point in sorted(sizes)
        )
        found = gate.envelope(points, slo, sizes, rounds)
        if found is None:
            raise ValueError(f"{key}: it qualifies at no number of sessions the rounds measured")
        envelopes[engine] = found[0]
        shared.append(
            arrangement(
                f"{key}, {found[0]} session trees",
                key,
                "consolidation",
                engine,
                ((found[1], 1), (second, 1)),
            )
        )
    on_change = CONSOLIDATED["change"]
    change, within = dict(gate.CONSOLIDATION[on_change]), envelopes["change"]
    spare = busiest("the warm spare", gate.SPARE)[1]
    shared.append(
        arrangement(
            f"Consolidation on the change at {within} session trees, with a warm spare",
            on_change,
            "consolidation",
            "change",
            ((change[within], 1), (spare, 1)),
            spare=True,
        )
    )
    past = min((s for s in change if s > within), default=None)
    if past is None:
        raise ValueError(f"{on_change}: the rounds measured no more sessions than its envelope's")
    shared.append(
        arrangement(
            f"{on_change}, {past} session trees",
            on_change,
            "consolidation",
            "change",
            ((change[past], 1), (second, 1)),
        )
    )
    pooled = [
        arrangement(name, name, "pooling", engine, gate.POOLING[name])
        for engine, name in POOLED.items()
    ]
    arrangements = (
        dedicated[gate.BEST_PRACTICAL],
        *(a for k, a in dedicated.items() if k != gate.BEST_PRACTICAL),
        *shared,
        *pooled,
    )
    return TwoCards(
        tuple(rounds), gate.REQUIREMENT, gate.HURDLE, consolidated, envelopes, best, arrangements
    )
