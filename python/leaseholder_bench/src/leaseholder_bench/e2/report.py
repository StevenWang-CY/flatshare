"""E2's frontier across points (``lh-bench e2 report``).

:func:`write_report` places the points on E2's frontier: background goodput on x, voice-to-voice
p95 on y, one frontier per arm and workload (the agents' corpus and the number of callers, the GPU
and the model where they are not the frozen box's, and the warm-up turns that open each call where
the points differ in them, since points of different workloads are never compared). A point is on
its arm's frontier when no other point of the arm and workload harvests at least as much background
goodput at a V2V p95 at least as low, and one of the two strictly better. Points measured from the
same arm file in several campaigns are repeats of one configuration, which the frontier places at
their medians; the tables keep every run. The stage waterfall splits each configuration's voice turn
into the stages that add up to its mean V2V.

It reads each point as its compact record keeps it (:func:`.record.compact`): a record, or a run
directory, whose raw files only :mod:`.record` reads.
"""

from __future__ import annotations

import csv
import re
import statistics
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import orjson

from leaseholder_bench.e2.record import (
    GOODPUT_WINDOW_S,
    REPLY_TIMEOUT_MS,
    WATERFALL,
    compact,
    engine_model,
    kv_token_bytes,
    measured_sum,
    point_engine,
    turn_samples,
)
from leaseholder_stats import percentiles, summarize

COLUMNS = (
    "arm",
    "workload",
    "point",
    "campaign",
    "label",
    "frontier",
    "callers",
    "sessions",
    "preemptive_generation",
    "engine",
    "router",
    "background_tokens_per_s",
    "background_requests_per_s",
    "v2v_p50_ms",
    "v2v_p95_ms",
    "v2v_p99_ms",
    "voice_turns",
    "voice_errors",
    "barge_in_to_silence_p95_ms",
    "agent_ttft_p50_ms",
    "agent_ttft_p99_ms",
    "realtime_first_byte_p95_ms",
    "realtime_first_byte_source",
    "engine_preemptions",
    "shed",
    "refused",
    "resubmitted",
    "gateway_background_tokens_per_s",
)
"""frontier.csv's columns; goodput is in tokens and requests per second, latencies in ms. A point's
workload is the agents' corpus and the number of callers ("weka-x32, 2 callers"). A run's campaign
is the directory that holds its run directory (campaign.sh measures a campaign's points into
CAMPAIGN/POINT). The realtime first byte is the gateway's, exactly from its request log (its
source, "log" in every record). Refused adds up the gateway's admission point and the
engine's queue limits; the gateway's own count of the background tokens it served checks AIPerf's
goodput."""
CONFIGURATION_ARMS = {
    # Arm 9 in the discriminating scenario without one of its parts, or with one factor changed.
    "disc-a9-no-headroom-s24": "9-no-headroom",
    "disc-a9-no-throttle-s24": "9-no-throttle",
    "disc-a9-budget0-s24": "9-budget0",
    "disc-a9-budget500-s24": "9-budget500",
    "disc-a9-no-degrade-s24": "9-no-degrade",
    "disc-a9-window48-s24": "9-window48",
    "disc-a9-no-events-s24": "9-no-events",
    "disc-a9-trigger-request-s24": "9-trigger-request",
    "disc-a9-engine-only-s24": "9-engine-only",
    "disc-a9-gateway-only-s24": "9-gateway-only",
    "disc-a9-no-kv-makeroom-s24": "9-no-kv-makeroom",
    "disc-a9-no-reserve-s24": "9-no-reserve",
    "disc-a9-no-budget-cap-s24": "9-no-budget-cap",
    # The engine as vLLM #63350 and #63351 propose it: no reserve, no gateway policy.
    "disc-pr-a-s24": "9-pr-a",
    "disc-pr-ab-s24": "9-pr-ab",
    "disc-pr-ab48-s24": "9-pr-ab48",
}
"""The arm the report gives the points of an arm file whose ``arm`` names another configuration's:
arm 9's ablations and the vLLM PRs' configurations say ``arm = 9`` as run, but each is a
configuration of its own, never a point on arm 9's frontier. The records stay as run."""
PR_LABELS = (("vLLM PRs A and B", "vLLM #63350 and #63351"), ("vLLM PR A", "vLLM #63350"))
"""The vLLM PRs as the records' labels name them, and as the report shows them: by their numbers,
as the docs do. The records keep their labels as run."""
STANDARD_WORKLOAD = ("weka-x32", 2)
"""The corpus and callers of an arm file that names neither (e2's Agents.data and Voice.callers):
the workload drawn in frontier.svg."""
BOX_GPU = "NVIDIA GeForce RTX 5060 Ti"
BOX_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
"""The frozen box's GPU, as nvidia-smi names it in campaign.sh's samples, and its model: a point's
workload names its GPU and its model only where they are not these. A record that names no GPU ran
on the box, as every point did before the samples named it."""
WORKER_COLUMNS = (
    *(f"{stage.removesuffix('_ms')}_{q}_ms" for stage in WATERFALL for q in ("p50", "p95")),
    "events_sent",
    "events_dropped",
    "events_failed",
)
GPU_COLUMNS = ("gpu_sm_clock_p50_mhz", "gpu_temperature_max_c")
ADMISSION_COLUMNS = (
    "deferred_headroom",
    "deferred_critical",
    "degraded_replies",
    "realtime_replies",
    "degraded_share",
)
LIFECYCLE_POLICY = ("lifecycle", "leaseholder")
"""The names the gateway's ``[admission] policy`` gives the lifecycle policy: its own, and the one
E2's arm files spell."""
WINDOW_COLUMNS = ("fixed_window_tokens_per_s", "fixed_window_s")
"""The background goodput over the calls' first GOODPUT_WINDOW_S, and that window's seconds: fewer for a
point whose calls lasted less, which frontier.md marks with †."""
FIRST_SCHEDULE = {
    "local_compute": "first_schedule_computed_tokens",
    "local_cache_hit": "first_schedule_gpu_hit_tokens",
    "external_kv_transfer": "first_schedule_cpu_hit_tokens",
}
"""vLLM's prompt-token sources (``vllm:prompt_tokens_by_source_total``), which it counts at a request's
first scheduling only, and their columns."""
CPU_TIER_COLUMNS = (
    *FIRST_SCHEDULE.values(),
    "cpu_tier_loaded_tokens",
    "cpu_tier_stored_tokens",
    "kv_bytes_per_token",
)
"""Columns that only some points have, each group shown when one of them has it: the voice worker's
stage times and lifecycle events; the GPU's median SM clock and highest temperature during the calls;
under the gateway's lifecycle policy, the background requests it deferred by what held
them back and the realtime replies it capped, with their share of its realtime requests; and on vLLM
its prompt tokens by source at their first scheduling and the CPU tier's loads and stores of every
scheduling, in tokens at the bytes of KV a token takes in the engine's pool."""


def load_points(run_dirs: Iterable[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    """One row per complete point, with its arm's frontier in its workload marked, and the run
    directories whose point did not complete. The rows go by workload, the arm file's own first,
    then by arm. The runs of a configuration measured in several campaigns go together, and the
    frontier places the configuration at their medians."""
    rows, incomplete = [], []
    for run_dir in run_dirs:
        if (point := compact(run_dir)) is None:
            incomplete.append(run_dir)
            continue
        rows.append(_row(run_dir, *point))
    # A record set opens every call with as many warm-up turns (W4's none, W5's one), and its
    # workloads leave them unnamed; points that differ in them are of different workloads.
    named = len({r["warmup_turns"] for r in rows}) > 1
    for row in rows:
        warmup = row["warmup_turns"] if named else 0
        row["workload"] = _workload(row["data"], row["callers"], row["hardware"], warmup)
        row["figure"] = figure_name(row["data"], row["callers"], row["hardware"], warmup)
    groups = configurations(rows)
    mark_frontier(groups)
    placed = {_configuration(g): g for g in groups}
    for row in rows:
        row["frontier"] = placed[_configuration(row)]["frontier"]

    def order(r: Mapping[str, Any]) -> tuple[Any, ...]:
        goodput = placed[_configuration(r)]["background_tokens_per_s"]
        # The box's workloads first; among them the arm file's own, then the rest of its corpus, then
        # the other corpora.
        data, callers = r["data"], r["callers"]
        first = (
            r.get("hardware", ""),
            r["warmup_turns"],
            (data, callers) != STANDARD_WORKLOAD,
            data != STANDARD_WORKLOAD[0],
            data,
            callers,
        )
        return (*first, arm_order(r["arm"]), goodput, r["point"], r["campaign"])

    rows.sort(key=order)
    return rows, incomplete


def _configuration(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """What makes the runs of one configuration: its workload, its arm and its arm file's name."""
    return row["workload"], row["arm"], row["point"]


def _workload(data: str, callers: int, on: str = "", warmup_turns: int = 0) -> str:
    """A workload's name, as frontier.csv and frontier.md give it; ``on`` is :func:`hardware`'s, and
    ``warmup_turns`` the warm-up turns its calls open with, where the report names them."""
    name = f"{data}, {callers} caller{'' if callers == 1 else 's'}" + (f", {on}" if on else "")
    return name + (f", {_warmups(warmup_turns)}" if warmup_turns else "")


def _warmups(turns: int) -> str:
    return f"{turns} warm-up turn{'' if turns == 1 else 's'}"


def _shown(label: str) -> str:
    """A record's label as the report shows it (PR_LABELS)."""
    for kept, shown in PR_LABELS:
        label = label.replace(kept, shown)
    return label


def hardware(gpu_name: str | None, model: str | None) -> str:
    """What a point's workload names beside its corpus and callers: its GPU and its model where they
    are not the frozen box's (BOX_GPU, BOX_MODEL); nothing on the box."""
    return ", ".join(
        name for name, box in ((gpu_name, BOX_GPU), (model, BOX_MODEL)) if name and name != box
    )


def _row(run_dir: Path, run: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    """A point's row of frontier.csv but for its workload, which load_points names."""
    arm, voice = run["arm"], summary["voice"]
    background = summary["background"] or {}
    gateway, engine = summary["gateway"], summary["engine"]
    first_byte = gateway["realtime_first_byte_ms"]
    worker = summary.get("worker") or {}
    waterfall, events = worker.get("waterfall") or {}, worker.get("events") or {}
    # V2V from the callers' turns, where a failed turn counts at the reply timeout.
    samples = v2v_samples(run_dir / "bots" / "turns.jsonl")
    v2v = percentiles(samples)
    settings = point_engine(run_dir)
    on = hardware((summary.get("gpu") or {}).get("name"), engine_model(settings))
    row = {
        "arm": CONFIGURATION_ARMS.get(run["name"], str(arm["arm"])),
        "point": run["name"],
        "campaign": run_dir.absolute().parent.name,
        "label": _shown(arm["label"]),
        "frontier": False,
        "callers": arm["voice"]["callers"],
        "sessions": arm["agents"]["sessions"],
        "preemptive_generation": arm["voice"]["preemptive_generation"],
        "engine": " ".join([arm["engine"]["variant"], *arm["engine"]["flags"]]),
        "router": arm["router"] or "",
        "background_tokens_per_s": background.get("tokens_per_s", 0.0),
        "background_requests_per_s": background.get("requests_per_s", 0.0),
        "v2v_p50_ms": v2v.get("p50"),
        "v2v_p95_ms": v2v.get("p95"),
        "v2v_p99_ms": v2v.get("p99"),
        "voice_turns": voice["turns"],
        "voice_errors": voice["errors"],
        "barge_in_to_silence_p95_ms": voice["barge_in_to_silence_ms"].get("p95"),
        "agent_ttft_p50_ms": background.get("ttft_ms", {}).get("p50"),
        "agent_ttft_p99_ms": background.get("ttft_ms", {}).get("p99"),
        "realtime_first_byte_p95_ms": first_byte.get("p95"),
        "realtime_first_byte_source": first_byte["source"],
        "engine_preemptions": engine["preemptions"],
        "shed": gateway["shed"],
        "refused": measured_sum(gateway["refused"], *(engine["refused"] or {}).values()),
        "resubmitted": gateway["resubmitted"],
        "gateway_background_tokens_per_s": gateway["background_tokens_per_s"],
    }
    row |= {
        f"{stage.removesuffix('_ms')}_{q}_ms": waterfall.get(stage, {}).get(q)
        for stage in WATERFALL
        for q in ("p50", "p95")
    }
    row |= {f"events_{kind}": events.get(kind) for kind in ("sent", "dropped", "failed")}
    policy = ((arm.get("gateway") or {}).get("admission") or {}).get("policy")
    row |= (
        lifecycle_counts(gateway.get("counters") or {})
        if policy in LIFECYCLE_POLICY
        else dict.fromkeys(ADMISSION_COLUMNS)
    )
    # What frontier.csv leaves out: the corpus, the hardware and the warm-up turns, from which
    # load_points names the workload, and the stage waterfall's inputs, the count and mean of each
    # of the worker's stages and the callers' V2V samples. An arm from before 2c2c0dc names no
    # warm-up turns: it ran none.
    row["data"] = arm["agents"]["data"]
    row["hardware"] = on
    row["warmup_turns"] = arm["voice"].get("warmup_turns", 0)
    row["stage_means"] = {s: (c["n"], c["mean"]) for s, c in waterfall.items() if "mean" in c}
    row["v2v_samples"] = samples
    row |= cpu_tier(engine, settings)
    window = background.get("fixed_window") or {}
    row |= {
        "fixed_window_tokens_per_s": window.get("tokens_per_s"),
        "fixed_window_s": window.get("seconds"),
    }
    return row | _gpu(summary)


def cpu_tier(
    engine: Mapping[str, Any], settings: Mapping[str, Any] | None
) -> dict[str, int | float | None]:
    """A point's prompt tokens by source at their first scheduling, and the tokens its CPU tier loaded
    and stored (``engine``'s counts in bytes, at the bytes of KV a token takes in the pool of the
    engine ``settings`` describe), with those bytes."""
    sources = engine.get("prompt_tokens_by_source")
    row: dict[str, int | float | None] = {
        column: None if sources is None else sources.get(source)
        for source, column in FIRST_SCHEDULE.items()
    }
    per_token = kv_token_bytes(settings)

    def tokens(key: str) -> int | None:
        count = engine.get(key)
        return None if count is None or per_token is None else round(count / per_token)

    row["cpu_tier_loaded_tokens"] = tokens("kv_offload_load_bytes")
    row["cpu_tier_stored_tokens"] = tokens("kv_offload_store_bytes")
    row["kv_bytes_per_token"] = None if per_token is None else round(per_token, 1)
    return row


def lifecycle_counts(counters: Mapping[str, int]) -> dict[str, int | float | None]:
    """What the lifecycle policy did over the window, from the gateway's counters (those that moved):
    the background requests it deferred, by what held them back when they arrived (the KV headroom,
    then the critical window's prefill budget), and the realtime replies it capped because the
    headroom was not there, with their share of the realtime requests it admitted."""

    def count(series: str) -> int:
        return counters.get(series, 0)

    degraded = count("lh_gateway_admission_degraded_total")
    realtime = count('lh_gateway_admission_admitted_total{class="realtime"}')
    return {
        "deferred_headroom": count('lh_gateway_admission_deferred_total{reason="headroom"}'),
        "deferred_critical": count('lh_gateway_admission_deferred_total{reason="critical"}'),
        "degraded_replies": degraded,
        "realtime_replies": realtime,
        "degraded_share": round(degraded / realtime, 3) if realtime else None,
    }


def v2v_samples(turns: Path) -> list[float]:
    """The callers' voice-to-voice samples in their turns.jsonl (:func:`turn_samples`), a failed
    turn's censored at REPLY_TIMEOUT_MS."""
    return turn_samples(map(orjson.loads, filter(str.strip, turns.read_text().splitlines())))[0]


def _gpu(kept: Mapping[str, Any]) -> dict[str, int | None]:
    """The GPU columns from what the summary keeps of the GPU's samples, ``gpu``; None without
    them."""
    clocks = kept.get("gpu") or {}
    return {column: clocks.get(column.removeprefix("gpu_")) for column in GPU_COLUMNS}


REPEATED = (
    "background_tokens_per_s",
    "fixed_window_tokens_per_s",
    "v2v_p50_ms",
    "v2v_p95_ms",
    "v2v_p99_ms",
    "agent_ttft_p50_ms",
    "agent_ttft_p99_ms",
    "realtime_first_byte_p95_ms",
    "voice_errors",
)
"""The columns a configuration measured in several campaigns takes the median of over its runs, and
frontier.md gives with their range."""


def configurations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The configurations the rows measured, in the order of their first runs: each arm file's runs,
    one per campaign that measured it, with the median of each REPEATED column over the runs that
    measured it (the run's own value for a configuration measured once), and the failed turns and
    the turns of all its runs."""
    runs: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        runs.setdefault(_configuration(row), []).append(row)
    return [
        {"workload": workload, "arm": arm, "point": point, "runs": measured}
        | {"frontier": all(r["frontier"] for r in measured)}
        | {column: _median(r[column] for r in measured) for column in REPEATED}
        | {"failed_turns": sum(r["voice_errors"] for r in measured)}
        | {"turns": sum(r["voice_turns"] for r in measured)}
        for (workload, arm, point), measured in runs.items()
    ]


def _median(values: Iterable[float | None]) -> float | None:
    measured = [v for v in values if v is not None]
    return statistics.median(measured) if measured else None


def mark_frontier(rows: Sequence[dict[str, Any]]) -> None:
    """Mark the points that no other point of their arm and workload dominates. A point with failed
    turns is left off: its V2V is censored, not measured, so it neither takes a place on the
    frontier nor keeps another point off it."""

    def dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        ax, ay = a["background_tokens_per_s"], a["v2v_p95_ms"]
        bx, by = b["background_tokens_per_s"], b["v2v_p95_ms"]
        return ax >= bx and ay <= by and (ax > bx or ay < by)

    placed = [r for r in rows if r["v2v_p95_ms"] is not None and not r.get("failed_turns")]
    for row in placed:
        rivals = (
            r
            for r in placed
            if r is not row and (r["arm"], r["workload"]) == (row["arm"], row["workload"])
        )
        row["frontier"] = not any(dominates(rival, row) for rival in rivals)


SLO_MARGIN_MS = 250.0
"""The voice V2V target (docs/SLO.md): p95 at most 250 ms above voice alone."""


def voice_slo(
    groups: Sequence[Mapping[str, Any]],
    callers: int,
    on: str = "",
    warmup_turns: int | None = None,
) -> tuple[float, list[str]] | None:
    """The V2V p95 that the voice SLO allows with ``callers`` callers on the GPU and model ``on``
    names (:func:`hardware`; the box's when empty), in calls that open with ``warmup_turns``
    warm-up turns (any when None), and the configurations of voice alone it is read against: arm
    1's with as many callers on the same hardware and as many warm-up turns, from whichever corpus,
    since voice alone replays none. Their median V2V p95 plus SLO_MARGIN_MS; None without one."""
    alone = [
        g
        for g in groups
        if arm_order(g["arm"])[0] == 1
        and g["runs"][0]["callers"] == callers
        and g["runs"][0].get("hardware", "") == on
        and (warmup_turns is None or g["runs"][0]["warmup_turns"] == warmup_turns)
        and g["v2v_p95_ms"] is not None
    ]
    if not alone:
        return None
    p95 = statistics.median(g["v2v_p95_ms"] for g in alone)
    return p95 + SLO_MARGIN_MS, [g["point"] for g in alone]


def arm_order(arm: str) -> tuple[int, str]:
    """Arms in E2's order: 2 before 10, 7a before 7b."""
    number = re.match(r"\d+", arm)
    return (int(number[0]) if number else 1000, arm)


STACKED = (
    ("end_of_turn_ms", "Endpointing"),
    ("turn_to_first_sentence_ms", "Reply to first sentence"),
    ("tts_first_audio_ms", "TTS first audio"),
)
"""The worker's stages that follow one another from the end of the caller's speech, as the stage
waterfall stacks them; the rest of the callers' V2V is transport and playout."""


def stage_waterfall(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each configuration's voice turn, over the runs that have turn lines: the mean of each STACKED
    stage over their turns, the rest of the mean V2V (``transport_ms``: the callers' V2V less the
    worker's stages), the LLM's mean time to first token and when its request started (at the final
    transcript under preemptive generation, otherwise at the end of the turn), and the mean, median
    and p95 of the callers' V2V samples. Configurations without turn lines are left out, and so are
    runs with failed turns, whose replies the worker's stages do not cover."""
    cells = []
    for group in configurations(rows):
        runs = [
            r
            for r in group["runs"]
            if r["stage_means"] and r["v2v_samples"] and not r["voice_errors"]
        ]
        stages = {stage: _pooled(runs, stage) for stage, _ in STACKED}
        if not runs or None in stages.values():
            continue
        v2v = summarize(sample for r in runs for sample in r["v2v_samples"])
        request = _pooled(runs, "transcript_ms") if runs[0]["preemptive_generation"] else None
        cells.append(
            {
                "point": group["point"],
                "workload": group["workload"],
                "label": runs[0]["label"],
                "runs": len(runs),
                "replies": v2v["n"],
                **stages,
                "transport_ms": v2v["mean"] - sum(stages.values()),
                "llm_request_ms": stages["end_of_turn_ms"] if request is None else request,
                "llm_ttft_ms": _pooled(runs, "llm_ttft_ms"),
                "v2v_mean_ms": v2v["mean"],
                "v2v_p50_ms": v2v["p50"],
                "v2v_p95_ms": v2v["p95"],
            }
        )
    return cells


def _pooled(runs: Iterable[Mapping[str, Any]], stage: str) -> float | None:
    """A stage's mean over the turns of several runs: each run's mean weighted by its count."""
    counted = [r["stage_means"][stage] for r in runs if stage in r["stage_means"]]
    turns = sum(n for n, _ in counted)
    return sum(n * mean for n, mean in counted) / turns if turns else None


DOT_KEYS = ("point", "label", "note", "change", "featured")
"""The keys of a row of figure.toml: the point's name and the row's label, a note the row's second
line gives before its runs, whether it runs the engine change, and whether it is the configuration
to deploy."""


def read_dots(path: Path) -> list[dict[str, Any]]:
    """The groups of voice-p95.svg from a TOML file, in the file's order: the ``[[row]]`` tables
    at its top as one group, or ``[[group]]`` tables, each with a ``title`` and ``[[group.row]]``
    tables, for a figure of several GPUs, and the ``gpu`` its points ran on where the GPUs ran the
    same arm files (as :func:`gpu_name` gives it: "A100-SXM4-80GB"). A row names its point
    (``point``) and its label (``label``), and may give a ``note``, whether it runs the engine
    change (``change``) and whether it is the configuration to deploy (``featured``, at most one a
    group; it runs the change), both false by default. A ValueError names what is wrong."""
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if "row" in data and "group" in data:
        raise ValueError(f"{path}: [[row]] tables and [[group]] tables together")
    groups = data.get("group") or ([{"title": None, "row": data["row"]}] if "row" in data else [])
    if not groups or not all(g.get("row") for g in groups):
        raise ValueError(f"{path}: no [[row]] tables")
    for group in groups:
        if set(group) - {"title", "gpu", "row"}:
            raise ValueError(f"{path}: a group with keys other than title, gpu and row: {group}")
        if not isinstance(group.get("gpu", ""), str):
            raise ValueError(f"{path}: a group whose gpu is not a string: {group}")
        if len(groups) > 1 and not isinstance(group.get("title"), str):
            raise ValueError(f"{path}: a group without a title, among several")
        for row in group["row"]:
            if not isinstance(row.get("point"), str) or not isinstance(row.get("label"), str):
                raise ValueError(f"{path}: a row without a point or a label: {row}")
            if set(row) - set(DOT_KEYS):
                keys = ", ".join(DOT_KEYS[:-1]) + f" and {DOT_KEYS[-1]}"
                raise ValueError(f"{path}: a row with keys other than {keys}: {row}")
            if not isinstance(row.get("note", ""), str):
                raise ValueError(f"{path}: a row whose note is not a string: {row}")
            for key in ("change", "featured"):
                if not isinstance(row.get(key, False), bool):
                    raise ValueError(f"{path}: a row whose {key} is not true or false: {row}")
            if row.get("featured") and not row.get("change"):
                raise ValueError(f"{path}: a featured row that does not run the change: {row}")
        if sum(bool(row.get("featured")) for row in group["row"]) > 1:
            raise ValueError(f"{path}: a group with more than one featured row")
    return [{"title": g.get("title"), "gpu": g.get("gpu"), "rows": g["row"]} for g in groups]


def write_report(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    waterfall: Sequence[str] = (),
    dots: Sequence[Mapping[str, Any]] = (),
) -> list[Path]:
    """frontier.csv, frontier.md, a frontier figure per workload (:func:`figure_name`) and
    waterfall.svg: the stage waterfall of the points ``waterfall`` names, in its order, or of every
    configuration with turn lines. With ``dots`` (:func:`read_dots`'s groups), voice-p95.svg and
    its phone layout voice-p95-narrow.svg: the V2V p95 and the agents' tokens/s of the
    configurations of the points each group names, in its order, against its workload's SLO (see
    :func:`~leaseholder_bench.e2.figures.plot_voice`). Naming a point without turn lines, a dots
    point the report lacks or that two workloads measured, or a group whose points are of
    different workloads, is a ValueError."""
    from leaseholder_bench.e2.figures import (  # figures.py builds on this module
        plot,
        plot_voice,
        plot_waterfall,
    )

    cells = stage_waterfall(rows)
    if waterfall:
        missing = [name for name in waterfall if name not in {c["point"] for c in cells}]
        if missing:
            raise ValueError(f"the report has no turn lines of {', '.join(missing)}")
        cells = [c for name in waterfall for c in cells if c["point"] == name]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "frontier.csv", out_dir / "frontier.md"]
    with written[0].open("w", newline="") as f:
        columns = [*COLUMNS, *(c for group in _measured(rows) for c in group)]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    written[1].write_text(markdown(rows))
    groups = configurations(rows)
    for members in _by_workload(rows):
        first = members[0]
        slo = voice_slo(groups, first["callers"], first["hardware"], first["warmup_turns"])
        written.append(plot(members, out_dir / first["figure"], None if slo is None else slo[0]))
    if cells:
        written.append(plot_waterfall(cells, out_dir / "waterfall.svg"))
    if dots:
        figure = [_dot_group(group, groups) for group in dots]
        if len({(g["callers"], g["sessions"]) for g in figure}) > 1:
            raise ValueError(
                "the figure's groups differ in their callers or their agents' sessions"
            )
        written += plot_voice(figure, out_dir / "voice-p95.svg", out_dir / "voice-p95-narrow.svg")
    return written


def _dot_group(
    group: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """A group of voice-p95.svg: the configuration of each point its rows name, with the row, and
    their workload's voice SLO, GPU, callers and agents' sessions; with the group's ``gpu``, the
    configurations measured on that GPU."""
    gpu = group.get("gpu")
    chosen = []
    for dot in group["rows"]:
        found = [
            g
            for g in configurations
            if g["point"] == dot["point"] and gpu in (None, gpu_name(g["runs"][0]["hardware"]))
        ]
        if len(found) != 1:
            on = f" on {gpu}" if gpu else ""
            raise ValueError(
                f"{len(found)} configurations of {dot['point']}{on} in the report, not 1"
            )
        chosen.append((found[0], dot))
    workloads = {c["workload"] for c, _ in chosen}
    if len(workloads) > 1:
        raise ValueError(
            f"a group of the figure's rows from {len(workloads)} workloads: {workloads}"
        )
    first = chosen[0][0]["runs"][0]
    slo = voice_slo(configurations, first["callers"], first["hardware"], first["warmup_turns"])
    return {
        "title": group["title"],
        "rows": chosen,
        "slo": None if slo is None else slo[0],
        "gpu": gpu_name(first["hardware"]),
        "callers": first["callers"],
        "sessions": first["sessions"],
    }


def gpu_name(on: str) -> str:
    """The GPU a point's :func:`hardware` names, as a figure gives it: the box's when it names none
    (a model's name, which has its organization's slash, is not a GPU's), without the vendor's
    prefixes ("RTX 5060 Ti", "A100-SXM4-40GB")."""
    gpu = next((part for part in on.split(", ") if part and "/" not in part), BOX_GPU)
    return re.sub(r"^NVIDIA (GeForce )?", "", gpu)


def _by_workload(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """The rows of each workload, in the rows' order."""
    members: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        members.setdefault(row["workload"], []).append(row)
    return list(members.values())


def figure_name(data: str, callers: int, on: str = "", warmup_turns: int = 0) -> str:
    """A workload's frontier figure: frontier.svg for STANDARD_WORKLOAD on the box, else named after
    the corpus when it is another (frontier-weka-x32-long.svg), the callers when they are, the GPU
    and model off the box (frontier-weka-x32-long-nvidia-a100-sxm4-40gb.svg), and the warm-up turns
    where the workload's name gives them (frontier-1-warm-up-turn.svg)."""
    name = "frontier"
    if data != STANDARD_WORKLOAD[0]:
        name += "-" + data.replace("/", "-")
    if callers != STANDARD_WORKLOAD[1]:
        name += f"-{callers}-callers"
    if on:
        name += "-" + re.sub(r"[^a-z0-9]+", "-", on.lower()).strip("-")
    if warmup_turns:
        name += "-" + _warmups(warmup_turns).replace(" ", "-")
    return name + ".svg"


def _measured(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    """The optional column groups that some point has."""
    groups = (
        WORKER_COLUMNS,
        GPU_COLUMNS,
        ADMISSION_COLUMNS,
        CPU_TIER_COLUMNS,
        WINDOW_COLUMNS,
    )
    return [group for group in groups if any(r[c] is not None for r in rows for c in group)]


def markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """The frontier as a section per workload, with its voice SLO, and in it a table per arm,
    ordered by background goodput; then the voice worker's stage times when the points have them,
    and the stage waterfall of the configurations with turn lines."""
    measured = _measured(rows)
    gpu_columns = GPU_COLUMNS in measured
    window_columns = WINDOW_COLUMNS in measured
    groups = {_configuration(g): g for g in configurations(rows)}
    intro = (
        "Background goodput (prompt and completion tokens per second of the agent requests that "
        "completed during the calls) against voice-to-voice latency in ms, by workload: the agents' "
        "corpus (`[agents] data`) and the number of callers, since points of different workloads "
        "are not compared. • marks the points on their arm's frontier in their workload, where "
        "the arm has more than one configuration. The realtime first byte is the gateway's, from "
        "its request log. Refused counts the gateway's "
        "admission point and the engine's queue limits; — is a count the point's gateway or engine "
        "does not keep. A failed turn, one the agent did not answer within the callers' reply "
        f"timeout, counts as a V2V sample censored at {REPLY_TIMEOUT_MS:,.0f} ms, so that ≥ marks a "
        "percentile it sets; a configuration with failed turns is not placed on the frontier."
    )
    if any(len(g["runs"]) > 1 for g in groups.values()):
        intro += (
            " A point measured in several campaigns lists each run by its campaign, then the median "
            "and range (min\N{EN DASH}max) over its runs, where the frontier places it."
        )
    if window_columns:
        intro += (
            f" Beside the goodput over each point's whole calls, whose length differs by "
            f"configuration, is its goodput over their first {GOODPUT_WINDOW_S:.0f} s, the same span "
            f"for every point; † marks one over a point's whole calls, which lasted less."
        )
    if gpu_columns:
        intro += " SM clock (MHz) and temperature (°C) are the GPU's during the calls."
    lines = ["# E2 frontier", "", intro]
    window = f" First {GOODPUT_WINDOW_S:.0f} s |" if window_columns else ""
    header = (
        f"| Point | | Sessions | Background tokens/s |{window} Requests/s | V2V p50 | V2V p95 "
        "| V2V p99 | Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions "
        "| Shed / refused / resubmitted |"
    )
    align = "|---|---|---:|---:|" + ("---:|" if window_columns else "")
    align += "---:|---:|---:|---:|---:|---:|---:|---:|---|"
    if gpu_columns:
        header += " SM clock p50 | Max temp |"
        align += "---:|---:|"
    for members in _by_workload(rows):
        lines += ["", f"## Workload: {members[0]['workload']}", "", _slo_line(members[0], groups)]
        for arm in sorted({r["arm"] for r in members}, key=arm_order):
            lines += ["", f"### Arm {arm}", "", header, align]
            lines += _arm_rows(
                [r for r in members if r["arm"] == arm], groups, gpu_columns, window_columns
            )
    if WORKER_COLUMNS in measured:
        lines += _stages(rows)
    if ADMISSION_COLUMNS in measured:
        lines += _admission(rows)
    if CPU_TIER_COLUMNS in measured:
        lines += _cpu_tier(rows)
    if cells := stage_waterfall(rows):
        lines += _waterfall_table(cells)
    return "\n".join(lines) + "\n"


def _slo_line(
    row: Mapping[str, Any], groups: Mapping[tuple[str, str, str], Mapping[str, Any]]
) -> str:
    """The sentence that opens the section of ``row``'s workload: its voice SLO, and its figure."""
    callers, figure = row["callers"], row["figure"]
    slo = voice_slo(list(groups.values()), callers, row["hardware"], row["warmup_turns"])
    if slo is None:
        return (
            f"No point of voice alone with {callers} callers sets a voice SLO. Figure: `{figure}`."
        )
    allowed, alone = slo
    return (
        f"The voice SLO allows a V2V p95 of {allowed:,.0f} ms, {SLO_MARGIN_MS:.0f} ms above voice "
        f"alone with as many callers ({', '.join(alone)}: {allowed - SLO_MARGIN_MS:,.0f} ms). "
        f"Figure: `{figure}`."
    )


def _window(row: Mapping[str, Any], value: str) -> str:
    """A fixed-window goodput cell, † when the point's calls lasted less than the window."""
    short = row["fixed_window_s"] is not None and row["fixed_window_s"] < GOODPUT_WINDOW_S
    return f"{value} †" if short and value != "—" else value


def _arm_rows(
    rows: Sequence[Mapping[str, Any]],
    groups: Mapping[tuple[str, str, str], Mapping[str, Any]],
    gpu_columns: bool,
    window_columns: bool,
) -> list[str]:
    """An arm's table rows in a workload: a line per run, and a repeated configuration's medians.
    An arm with one configuration marks no frontier: it is its own."""
    lines = []
    several = len({_configuration(r) for r in rows}) > 1
    for r in rows:
        group = groups[_configuration(r)]
        counts = " / ".join(_cell(r[k]) for k in ("shed", "refused", "resubmitted"))
        ttft = f"{_cell(r['agent_ttft_p50_ms'])} / {_cell(r['agent_ttft_p99_ms'])}"
        first_byte = _cell(r["realtime_first_byte_p95_ms"])
        # A repeated configuration's frontier mark goes on its medians' line.
        name, mark = f"{r['point']}: {r['label']}", "•" if r["frontier"] and several else ""
        if len(group["runs"]) > 1:
            name, mark = f"{r['point']} ({r['campaign']})", ""
        window = f" {_window(r, _cell(r['fixed_window_tokens_per_s']))} |" if window_columns else ""
        line = (
            f"| {name} | {mark} | {r['sessions']} "
            f"| {_cell(r['background_tokens_per_s'])} |{window} "
            f"{_cell(r['background_requests_per_s'], 2)} "
            f"| {_v2v(r['v2v_p50_ms'])} | {_v2v(r['v2v_p95_ms'])} | {_v2v(r['v2v_p99_ms'])} "
            f"| {r['voice_errors']} of {r['voice_turns']} | {ttft} | {first_byte} "
            f"| {_cell(r['engine_preemptions'])} | {counts} |"
        )
        if gpu_columns:
            line += f" {_cell(r['gpu_sm_clock_p50_mhz'])} | {_cell(r['gpu_temperature_max_c'])} |"
        lines.append(line)
        if len(group["runs"]) > 1 and r is group["runs"][-1]:
            lines.append(_spread(group, gpu_columns, window_columns, several))
    return lines


def _spread(
    group: Mapping[str, Any], gpu_columns: bool, window_columns: bool, several: bool
) -> str:
    """A repeated configuration's line in its arm's table: the median and range of each REPEATED
    column over its runs, and its frontier mark when its arm has ``several`` configurations."""
    runs = group["runs"]

    def spread(column: str) -> str:
        values = [r[column] for r in runs if r[column] is not None]
        if not values:
            return "—"
        median = group[column]
        if column.startswith("v2v_"):
            return f"{_v2v(median)} ({_v2v(min(values))}\N{EN DASH}{_v2v(max(values))})"
        # A median between two counts of failed turns keeps its half.
        digits = 1 if column == "voice_errors" and not float(median).is_integer() else 0
        return f"{_cell(median, digits)} ({_cell(min(values))}\N{EN DASH}{_cell(max(values))})"

    first_byte = spread("realtime_first_byte_p95_ms")
    first = runs[0]
    window = ""
    if window_columns:
        short = [r for r in runs if r["fixed_window_s"] is not None]
        short = [r for r in short if r["fixed_window_s"] < GOODPUT_WINDOW_S]
        window = f" {spread('fixed_window_tokens_per_s')}{' †' if short else ''} |"
    line = (
        f"| {first['point']}: {first['label']}; median (range) of {len(runs)} runs "
        f"| {'•' if group['frontier'] and several else ''} | {first['sessions']} "
        f"| {spread('background_tokens_per_s')} |{window} | {spread('v2v_p50_ms')} "
        f"| {spread('v2v_p95_ms')} | {spread('v2v_p99_ms')} | {spread('voice_errors')} "
        f"| {spread('agent_ttft_p50_ms')} / {spread('agent_ttft_p99_ms')} | {first_byte} | | |"
    )
    return line + (" | |" if gpu_columns else "")


def _stages(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The voice worker's stage times of each point it timed, p50 / p95; the points it did not are
    counted."""
    timed = [r for r in rows if any(r[column] is not None for column in WORKER_COLUMNS)]
    intro = (
        "The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's "
        "speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request "
        "to its first token and to the first sentence handed to TTS; from the end-of-turn decision "
        "to that sentence; from that sentence to its first audio; and from the end of speech to the "
        "reply's first audio frame on the track. Events are those the worker sent the gateway, "
        "dropped from its queue and failed to deliver."
    )
    if untimed := len(rows) - len(timed):
        intro += (
            f" {untimed} point{'' if untimed == 1 else 's'} without the worker's turn lines "
            f"{'is' if untimed == 1 else 'are'} left out."
        )
    lines = [
        "",
        "## Voice turn stages",
        "",
        intro,
        "",
        "| Point | Transcript | End of turn | LLM first token | LLM first sentence "
        "| Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    runs = Counter(map(_configuration, rows))
    for r in timed:
        stages = [
            " / ".join(_cell(r[f"{stage.removesuffix('_ms')}_{q}_ms"]) for q in ("p50", "p95"))
            for stage in WATERFALL
        ]
        events = " / ".join(_cell(r[f"events_{kind}"]) for kind in ("sent", "dropped", "failed"))
        name = f"{r['point']} ({r['campaign']})" if runs[_configuration(r)] > 1 else r["point"]
        lines.append(f"| {name} | {' | '.join(stages)} | {events} |")
    return lines


def _admission(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """What the lifecycle policy did in each point that ran it."""
    lines = [
        "",
        "## The lifecycle policy",
        "",
        "What the gateway's lifecycle policy (`policy = \"lifecycle\"`; E2's arm files spell it "
        "`leaseholder`) did in each point that ran it: the background requests "
        "its admission point deferred, by what held them back when they arrived (the KV headroom "
        "first, then the critical window's prefill budget), and the realtime replies it capped at "
        "`degrade_max_tokens` because the headroom was not there, of the realtime requests it "
        "admitted. A mechanism that deferred or capped nothing did not act in that point.",
        "",
        "| Point | Deferred by the headroom | Deferred by the critical window | Degraded replies |",
        "|---|---:|---:|---:|",
    ]
    runs = Counter(map(_configuration, rows))
    for r in rows:
        if r["deferred_headroom"] is None:
            continue
        name = f"{r['point']} ({r['campaign']})" if runs[_configuration(r)] > 1 else r["point"]
        degraded = f"{_cell(r['degraded_replies'])} of {_cell(r['realtime_replies'])}"
        if r["degraded_share"] is not None:
            degraded += f" ({r['degraded_share']:.0%})"
        lines.append(
            f"| {name} | {_cell(r['deferred_headroom'])} | {_cell(r['deferred_critical'])} "
            f"| {degraded} |"
        )
    return lines


def _cpu_tier(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each vLLM point's prompt tokens by source at their first scheduling, and its CPU tier's loads
    and stores of every scheduling."""
    lines = [
        "",
        "## The engine's prompt tokens and its CPU tier",
        "",
        "vLLM's own counts over the calls, in tokens. vLLM 0.32 counts a request's prompt tokens by "
        "where they came from only at its first scheduling: computed, hit in the GPU pool, or loaded "
        "from the CPU tier; a preempted request's resume is in none of them. The CPU tier counts "
        "every load into the GPU pool and every store from it, resumes included, in bytes, given "
        "here in tokens at the bytes of KV a token takes in the engine's pool (its KV cache's bytes "
        "over its tokens; `kv_bytes_per_token` in frontier.csv). Loaded beyond first scheduling is "
        "the CPU tier's loads less its hits at first scheduling: the loads of requests resumed after "
        "a preemption, give or take those of requests whose load and first scheduling are not both "
        "within the calls (one falls before or after them, or the request is aborted in between). "
        "How much of a resumed request's prompt was computed again, no counter shows.",
        "",
        "| Point | Preemptions | Computed at first scheduling | GPU-pool hits at first scheduling "
        "| CPU-tier hits at first scheduling | Loaded from the CPU tier "
        "| Loaded beyond first scheduling | Stored to the CPU tier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    runs = Counter(map(_configuration, rows))
    for r in rows:
        if all(r[column] is None for column in CPU_TIER_COLUMNS):
            continue
        name = f"{r['point']} ({r['campaign']})" if runs[_configuration(r)] > 1 else r["point"]
        loaded, hits = r["cpu_tier_loaded_tokens"], r["first_schedule_cpu_hit_tokens"]
        beyond = None if loaded is None or hits is None else loaded - hits
        counts = (
            r["engine_preemptions"],
            *(r[column] for column in FIRST_SCHEDULE.values()),
            loaded,
            beyond,
            r["cpu_tier_stored_tokens"],
        )
        lines.append(f"| {name} | {' | '.join(_cell(count) for count in counts)} |")
    return lines


def _waterfall_table(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """The stage waterfall of each configuration with turn lines, as waterfall.svg draws it."""
    lines = [
        "",
        "## Stage waterfall",
        "",
        "Each configuration's voice turn as stages that follow one another, in ms, the mean over the "
        "replies of its runs: the end-of-turn decision after the end of speech (endpointing), the "
        "reply's first sentence after that decision and its first audio from TTS, as the voice "
        "worker times them, then the rest of the callers' V2V, transport and playout. The four add "
        "up to the mean V2V. The LLM's own first token is timed from its request, which under "
        "preemptive generation starts at the final transcript, before the end of the turn, so it is "
        "not a stage of its own. Its V2V p50 and p95 pool the replies of the configuration's runs, "
        "where the tables above give the median of the runs' own. waterfall.svg stacks the four and "
        "marks the pooled median and p95.",
        "",
        "| Configuration | Runs | Replies | Endpointing | Reply to first sentence "
        "| TTS first audio | Transport and playout | Mean V2V | Pooled V2V p50 | Pooled V2V p95 "
        "| LLM first token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stages = (*(key for key, _ in STACKED), "transport_ms")
    columns = (*stages, "v2v_mean_ms", "v2v_p50_ms", "v2v_p95_ms")
    for c in cells:
        times = " | ".join(_cell(c[key]) for key in columns)
        lines.append(
            f"| {c['point']}: {c['label']} | {c['runs']} | {c['replies']} | {times} "
            f"| {_cell(c['llm_ttft_ms'])} |"
        )
    return lines


def _cell(value: Any, digits: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _v2v(value: float | None) -> str:
    """A V2V percentile's cell: ≥ marks one that a failed turn sets, censored at the reply timeout."""
    return ("≥" if value is not None and value >= REPLY_TIMEOUT_MS else "") + _cell(value)
