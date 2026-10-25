"""E2's results: a point's summary from its run directory, and the frontier across points."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import re
import tomllib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from leaseholder_bench.cli import main
from leaseholder_bench.e2.arm import Agents, Voice, flag_values
from leaseholder_bench.e2.figures import load_sweep
from leaseholder_bench.e2.record import (
    DISCARDED,
    OUTCOMES,
    RECORD,
    REFUSED,
    RESUBMITS,
    SESSIONS_LOGGER,
    SGLANG_ENGINE,
    TURNS_LOGGER,
    USAGE,
    WATERFALL,
    background,
    compact,
    engine_counts,
    engine_settings,
    export_point,
    gateway_counts,
    gateway_requests,
    kv_samples,
    kv_values,
    logged_first_byte,
    summarize_point,
)
from leaseholder_bench.e2.record import gpu as gpu_samples
from leaseholder_bench.e2.report import (
    ADMISSION_COLUMNS,
    COLUMNS,
    CONFIGURATION_ARMS,
    CPU_TIER_COLUMNS,
    GPU_COLUMNS,
    STACKED,
    STANDARD_WORKLOAD,
    WINDOW_COLUMNS,
    WORKER_COLUMNS,
    configurations,
    figure_name,
    hardware,
    lifecycle_counts,
    load_points,
    mark_frontier,
    read_dots,
    stage_waterfall,
    voice_slo,
    write_report,
)
from leaseholder_bench.e2.run import BUSY, SCHEDULER, SECRETS, variant_flags
from leaseholder_stats import metric_samples

T0 = 1_800_000_000.0  # the calls run from T0 to T0 + 100 s
REPO = Path(__file__).resolve().parents[3]
VOICE = REPO / "python" / "leaseholder_voice"
VOICE_EVENTS = VOICE / "src" / "leaseholder_voice" / "events.py"
GATEWAY_METRICS = REPO / "crates" / "lh-gateway" / "metrics.txt"
"""The gateway's exposition under the lifecycle policy, every family it registers (``make metrics``
writes it, and the gateway's tests check it is current)."""
# The committed E2 records: W4's, then W5's re-measure.
RESULTS = REPO / "benchmarks" / "results" / "2026-10-23-e2"
W5_RESULTS = REPO / "benchmarks" / "results" / "2026-10-24-e2-w5"


def record(end: float, *, phase: str = "profiling", ttft: float = 100.0, **extra: Any) -> str:
    metrics = {
        "time_to_first_token": {"value": ttft},
        "usage_prompt_tokens": {"value": 1000},
        "usage_completion_tokens": {"value": 20},
    }
    meta = {"benchmark_phase": phase, "request_end_ns": int(end * 1e9)}
    return json.dumps({"metadata": meta | extra.pop("meta", {}), "metrics": metrics} | extra)


def test_background_goodput_counts_the_requests_completed_during_the_calls(tmp_path: Path) -> None:
    lines = [
        record(T0 - 1),  # before the calls
        record(T0 + 1, phase="warmup"),
        *(record(T0 + i, ttft=100.0 + i) for i in range(1, 101)),
        record(T0 + 50, error={"code": 503, "message": "queue full"}),
        record(T0 + 60, meta={"was_cancelled": True}),
        record(T0 + 101),  # after
    ]
    path = tmp_path / "profile_export.jsonl"
    path.write_text("\n".join(lines) + "\n")
    cell = background(path, T0, T0 + 100)
    assert (cell["requests"], cell["errors"], cell["without_usage"]) == (100, 2, 0)
    assert cell["requests_per_s"] == 1.0
    assert (cell["prompt_tokens_per_s"], cell["completion_tokens_per_s"]) == (1000.0, 20.0)
    assert cell["tokens_per_s"] == 1020.0
    assert cell["ttft_ms"]["n"] == 100 and cell["ttft_ms"]["p50"] == 150.5
    assert cell["itl_ms"] == {"n": 0}  # these requests streamed one token each
    with pytest.raises(ValueError, match="completed no request"):
        background(path, T0 + 200, T0 + 300)
    with pytest.raises(ValueError, match="no per-request records"):
        background(tmp_path / "missing.jsonl", T0, T0 + 100)


def test_the_fixed_window_counts_the_requests_completed_in_the_calls_first_240_s(
    tmp_path: Path,
) -> None:
    # 300 s of calls, a request a second: 2,020 tokens each for the first 100 s, 1,020 after.
    lines = [
        record(T0 + i).replace('"value": 1000}', '"value": 2000}' if i <= 100 else '"value": 1000}')
        for i in range(1, 301)
    ]
    lines += [record(T0 + 120, error={"code": 500}), record(T0 + 250, meta={"was_cancelled": True})]
    path = tmp_path / "profile_export.jsonl"
    path.write_text("\n".join(lines) + "\n")
    cell = background(path, T0, T0 + 300)
    assert (cell["requests"], cell["errors"], cell["tokens_per_s"]) == (300, 2, 1353.3)
    # The first 240 s: 100 requests of 2,020 tokens and 140 of 1,020; the failed one is lost.
    assert cell["fixed_window"] == {
        "seconds": 240.0,
        "requests": 240,
        "requests_per_s": 1.0,
        "tokens_per_s": 1436.7,
    }
    # Calls that lasted less than the window give their whole goodput, over their own seconds.
    short = background(path, T0, T0 + 200.0004)
    assert short["fixed_window"] == {
        "seconds": 200.0,
        "requests": 200,
        "requests_per_s": 1.0,
        "tokens_per_s": 1520.0,
    }


GATEWAY_BEFORE = """\
lh_gateway_first_byte_seconds_bucket{class="realtime",le="0.1"} 10
lh_gateway_first_byte_seconds_bucket{class="realtime",le="0.5"} 10
lh_gateway_first_byte_seconds_bucket{class="realtime",le="+Inf"} 10
lh_gateway_first_byte_seconds_sum{class="realtime"} 0.5
lh_gateway_first_byte_seconds_bucket{class="background",le="0.1"} 0
lh_gateway_first_byte_seconds_bucket{class="background",le="0.5"} 0
lh_gateway_first_byte_seconds_bucket{class="background",le="+Inf"} 50
lh_gateway_request_outcomes_total{class="background",outcome="completed"} 40
lh_gateway_request_outcomes_total{class="background",outcome="shed"} 0
lh_gateway_usage_tokens_total{class="background",kind="prompt"} 1000
lh_gateway_usage_tokens_total{class="background",kind="completion"} 10
lh_gateway_usage_tokens_total{class="background",kind="cached"} 0
lh_gateway_usage_tokens_total{class="realtime",kind="prompt"} 5
lh_gateway_admission_refused_total{class="background",reason="queue_full"} 1
lh_gateway_admission_refused_total{class="background",reason="timeout"} 0
lh_gateway_admission_resubmits_total{class="background"} 0
lh_gateway_admission_discarded_tokens_total{class="background"} 0
"""
GATEWAY_AFTER = """\
lh_gateway_first_byte_seconds_bucket{class="realtime",le="0.1"} 18
lh_gateway_first_byte_seconds_bucket{class="realtime",le="0.5"} 20
lh_gateway_first_byte_seconds_bucket{class="realtime",le="+Inf"} 20
lh_gateway_first_byte_seconds_sum{class="realtime"} 1.4
lh_gateway_first_byte_seconds_bucket{class="background",le="0.1"} 0
lh_gateway_first_byte_seconds_bucket{class="background",le="0.5"} 0
lh_gateway_first_byte_seconds_bucket{class="background",le="+Inf"} 90
lh_gateway_request_outcomes_total{class="background",outcome="completed"} 140
lh_gateway_request_outcomes_total{class="background",outcome="shed"} 3
lh_gateway_usage_tokens_total{class="background",kind="prompt"} 101000
lh_gateway_usage_tokens_total{class="background",kind="completion"} 2010
lh_gateway_usage_tokens_total{class="background",kind="cached"} 50000
lh_gateway_usage_tokens_total{class="realtime",kind="prompt"} 505
lh_gateway_admission_refused_total{class="background",reason="queue_full"} 3
lh_gateway_admission_refused_total{class="background",reason="timeout"} 4
lh_gateway_admission_resubmits_total{class="background"} 5
lh_gateway_admission_discarded_tokens_total{class="background"} 2500
"""


def test_the_gateway_counts_cover_the_window(tmp_path: Path) -> None:
    (tmp_path / "gateway.jsonl").write_text(REQUESTS + "\n")
    requests = gateway_requests(tmp_path, T0, T0 + 100)
    cell = gateway_counts(GATEWAY_BEFORE, GATEWAY_AFTER, 100.0, requests)
    # The first bytes exactly from the request lines, whatever the histogram says.
    assert cell["realtime_first_byte_ms"] == logged_first_byte(requests, "realtime")
    assert cell["background_first_byte_ms"] == {
        "n": 1,
        "mean": 2500.0,
        "p50": 2500.0,
        "p95": 2500.0,
        "p99": 2500.0,
        "source": "log",
    }
    assert (cell["shed"], cell["refused"], cell["resubmitted"]) == (3, 6, 5)
    assert cell["discarded_tokens"] == 2500
    assert cell["background_tokens_per_s"] == 1020.0  # prompt and completion, background only
    assert (
        cell["counters"]['lh_gateway_request_outcomes_total{class="background",outcome="shed"}']
        == 3
    )
    assert (
        'lh_gateway_admission_refused_total{class="background",reason="timeout"}'
        in cell["counters"]
    )
    assert len(cell["counters"]) == 10  # every counter series that moved
    restarted = GATEWAY_AFTER.replace('completed"} 140', 'completed"} 2')
    with pytest.raises(ValueError, match="went down during the calls: the gateway restarted"):
        gateway_counts(GATEWAY_BEFORE, restarted, 100.0, requests)


def test_a_gateway_without_an_admission_point_counts_none_of_its_outcomes() -> None:
    before, after = (
        "\n".join(
            line for line in text.splitlines() if "admission" not in line and "shed" not in line
        )
        for text in (GATEWAY_BEFORE, GATEWAY_AFTER)
    )
    cell = gateway_counts(before, after, 100.0, [])
    assert (cell["shed"], cell["refused"], cell["resubmitted"], cell["discarded_tokens"]) == (
        None,
        None,
        None,
        None,
    )
    assert cell["background_tokens_per_s"] == 1020.0
    empty = gateway_counts("", "", 100.0, [])
    assert empty["realtime_first_byte_ms"] == {"n": 0, "source": "log"}
    assert empty["background_tokens_per_s"] is None


def logged(at: float, slo_class: str | None, first_byte_ms: float | None, **fields: Any) -> str:
    """One of the gateway's JSON request lines, for a request that ended at ``at``."""
    stamp = datetime.fromtimestamp(at, UTC).isoformat(timespec="microseconds")
    line = {
        "timestamp": stamp.replace("+00:00", "Z"),
        "level": "INFO",
        "message": "request",
        "class": slo_class,
        "status": 200,
        "outcome": "completed",
        "first_byte_ms": first_byte_ms,
    }
    return json.dumps({k: v for k, v in (line | fields).items() if v is not None})


# The calls' realtime requests had their first byte at 100, 200, ... 1000 ms.
REQUESTS = "\n".join(
    [
        logged(T0 - 1, "realtime", 9999.0),  # before the calls
        *(logged(T0 + n, "realtime", 100.0 * n) for n in range(1, 11)),
        logged(T0 + 50, "realtime", None, outcome="cancelled"),  # left before its first byte
        logged(T0 + 60, "realtime", 5.0, outcome=None),  # never went upstream
        logged(T0 + 70, "background", 2500.0),
        '{"timestamp": "2027-01-15T08:00:00.000000Z", "message": "lifecycle event"}',
        "a line that is not JSON",
    ]
)


def test_the_first_bytes_come_exactly_from_the_request_log(tmp_path: Path) -> None:
    (tmp_path / "gateway.jsonl").write_text(REQUESTS + "\n")
    requests = gateway_requests(tmp_path, T0, T0 + 100)
    assert requests is not None and len(requests) == 13
    assert logged_first_byte(requests, "realtime") == {
        "n": 10,
        "mean": 550.0,
        "p50": 550.0,
        "p95": 955.0,
        "p99": 991.0,
        "source": "log",
    }
    assert logged_first_byte(requests, "background")["p50"] == 2500.0
    assert gateway_requests(tmp_path / "missing", T0, T0 + 100) is None
    (tmp_path / "gateway.jsonl").write_text("lh-gateway serving\n")  # a gateway that logs text
    assert gateway_requests(tmp_path, T0, T0 + 100) is None


ENGINE = """\
vllm:num_preemptions_total{{engine="0",model_name="m"}} {preemptions}
vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="local_compute"}} {computed}
vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="local_cache_hit"}} {gpu}
vllm:admission_rejections_total{{model_name="m",reason="max_num_queued_tokens"}} {refused}
vllm:kv_offload_load_bytes_total{{engine="0",model_name="m"}} {loaded}
vllm:kv_offload_store_bytes_total{{engine="0",model_name="m"}} {stored}
"""


def test_the_engine_counts_cover_the_window_and_null_what_it_does_not_export() -> None:
    before = ENGINE.format(
        preemptions=2, computed=100, gpu=1000, refused=0, loaded="1.47456e+08", stored=0
    )
    after = ENGINE.format(
        preemptions=5, computed=400, gpu=9000, refused=7, loaded="4.42368e+08", stored=2.9491e9
    )
    assert engine_counts(before, after) == {
        "preemptions": 3,
        "prompt_tokens_by_source": {"local_compute": 300, "local_cache_hit": 8000},
        "refused": {"max_num_queued_tokens": 7},
        # The CPU tier's own counts, in bytes: 2,000 tokens loaded, about 20,000 stored.
        "kv_offload_load_bytes": 294_912_000,
        "kv_offload_store_bytes": 2_949_100_000,
    }
    sglang = "sglang:num_running_reqs 3\n"  # SGLang exports none of vLLM's counters
    assert engine_counts(sglang, sglang) == {
        "preemptions": None,
        "prompt_tokens_by_source": None,
        "refused": None,
        "kv_offload_load_bytes": None,
        "kv_offload_store_bytes": None,
    }


def point(
    root: Path,
    name: str,
    arm: str,
    tokens_per_s: float | None,
    p95: float,
    status: str = "complete",
    *,
    callers: int = 2,
    data: str = "weka-x32",
    warmup_turns: int | None = None,
) -> Path:
    """A run directory as lh-bench e2 run leaves it; without ``warmup_turns``, its arm names none,
    as the arms run before 2c2c0dc."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    settings = {
        "arm": arm,
        "label": f"{name} label",
        "router": None,
        "voice": {"callers": callers, "preemptive_generation": True},
        "agents": {"sessions": 0 if tokens_per_s is None else 16, "data": data},
        "engine": {"variant": "frozen", "flags": ["--max-num-seqs", "12"]},
    }
    if warmup_turns is not None:
        settings["voice"]["warmup_turns"] = warmup_turns
    run = {"kind": "e2-point", "name": name, "status": status, "arm": settings}
    run["window"] = {"start": T0, "end": T0 + 100}
    (run_dir / "run.json").write_text(json.dumps(run))
    bg = None
    if tokens_per_s is not None:
        bg = {
            "tokens_per_s": tokens_per_s,
            "requests_per_s": tokens_per_s / 1000,
            "ttft_ms": {"p50": 140.0, "p99": 8000.0},
        }
    summary = {
        "name": name,
        "arm": arm,
        "label": f"{name} label",
        "voice": {
            "turns": 50,
            "errors": 0,
            "v2v_ms": {"n": 60, "p50": 1600.0, "p95": p95, "p99": p95 + 500},
            "barge_in_to_silence_ms": {"n": 9, "p95": 650.0},
        },
        "background": bg,
        "gateway": {
            "realtime_first_byte_ms": {"n": 60, "p95": 120.0, "source": "log"},
            "shed": None,
            "refused": 2,
            "resubmitted": None,
            "background_tokens_per_s": tokens_per_s,
        },
        "engine": {"preemptions": 4, "refused": {"max_num_queued_tokens": 1}},
    }
    if status == "complete":
        (run_dir / "summary.json").write_text(json.dumps(summary))
        # The callers' turns: of 101 samples, the inclusive quantiles are the 51st, 96th and 100th.
        v2v = [1600.0] * 51 + [p95] * 45 + [p95 + 500] * 5
        (run_dir / "bots").mkdir()
        (run_dir / "bots" / "turns.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in answered(*v2v))
        )
    return run_dir


def test_each_arm_has_its_own_frontier_in_each_workload(tmp_path: Path) -> None:
    dirs = [
        point(tmp_path, "a1-voice", "1", None, 1760),
        point(tmp_path, "a2-s8", "2", 3000, 2000),
        point(tmp_path, "a2-s16", "2", 5000, 2600),
        point(tmp_path, "a2-s24", "2", 4500, 3000),  # less goodput at a worse p95
        point(tmp_path, "a7a-seqs12", "7a", 4000, 3100),  # beaten by arm 2, but on its own frontier
        point(tmp_path, "a2-failed", "2", None, 0, status="failed"),
        # Arm 2 again, each beating every point above, but measured on another workload.
        point(tmp_path, "disc-a2", "2", 6000, 1900, data="weka-x32-long"),
        point(tmp_path, "calls4-a2", "2", 5500, 1950, callers=4),
    ]
    rows, incomplete = load_points(dirs)
    assert incomplete == [tmp_path / "a2-failed"]
    assert [(r["point"], r["workload"], r["frontier"]) for r in rows] == [
        ("a1-voice", "weka-x32, 2 callers", True),
        ("a2-s8", "weka-x32, 2 callers", True),
        ("a2-s24", "weka-x32, 2 callers", False),
        ("a2-s16", "weka-x32, 2 callers", True),
        ("a7a-seqs12", "weka-x32, 2 callers", True),
        ("calls4-a2", "weka-x32, 4 callers", True),  # the arm file's own corpus first
        ("disc-a2", "weka-x32-long, 2 callers", True),
    ]
    assert rows[0]["background_tokens_per_s"] == 0.0 and rows[0]["agent_ttft_p50_ms"] is None
    assert rows[1]["engine"] == "frozen --max-num-seqs 12"


def test_ties_are_not_dominated() -> None:
    rows = [
        {"arm": "4", "workload": "w", "background_tokens_per_s": 100.0, "v2v_p95_ms": 2000.0},
        {"arm": "4", "workload": "w", "background_tokens_per_s": 100.0, "v2v_p95_ms": 2000.0},
        {"arm": "4", "workload": "w", "background_tokens_per_s": 50.0, "v2v_p95_ms": None},
    ]
    mark_frontier(rows)
    assert [r.get("frontier") for r in rows] == [True, True, None]


def test_each_workload_has_its_section_its_voice_slo_and_its_figure(tmp_path: Path) -> None:
    dirs = [
        point(tmp_path, "a1-voice", "1", None, 1760),
        point(tmp_path, "a2-s16", "2", 5000, 2600),
        point(tmp_path, "calls4-s16", "4-calls", 5000, 2700, callers=4),
        point(tmp_path, "disc-a4", "4", 4000, 9000, data="weka-x32-long"),
    ]
    rows, _ = load_points(dirs)
    out = tmp_path / "report"
    written = write_report(rows, out)
    with (out / "frontier.csv").open() as f:
        assert [r["workload"] for r in csv.DictReader(f)] == [
            "weka-x32, 2 callers",
            "weka-x32, 2 callers",
            "weka-x32, 4 callers",
            "weka-x32-long, 2 callers",
        ]
    table = (out / "frontier.md").read_text()
    sections = [line for line in table.splitlines() if line.startswith("## Workload: ")]
    assert sections == [
        "## Workload: weka-x32, 2 callers",
        "## Workload: weka-x32, 4 callers",
        "## Workload: weka-x32-long, 2 callers",
    ]
    assert "### Arm 4-calls" in table and "### Arm 2" in table
    # Voice alone replays no corpus: with as many callers, it sets every corpus's SLO.
    slo = (
        "The voice SLO allows a V2V p95 of 2,010 ms, 250 ms above voice alone with as many callers"
    )
    assert f"{slo} (a1-voice: 1,760 ms). Figure: `frontier.svg`." in table
    assert f"{slo} (a1-voice: 1,760 ms). Figure: `frontier-weka-x32-long.svg`." in table
    assert "No point of voice alone with 4 callers sets a voice SLO." in table
    assert (Agents.data, Voice.callers) == STANDARD_WORKLOAD  # an arm file that names neither
    assert figure_name("weka-x32", 4) == "frontier-4-callers.svg"
    assert figure_name("traces/long", 6) == "frontier-traces-long-6-callers.svg"
    if importlib.util.find_spec("matplotlib"):
        figures = ["frontier.svg", "frontier-4-callers.svg", "frontier-weka-x32-long.svg"]
        assert [p.name for p in written if p.suffix == ".svg"] == figures
        texts = {name: (out / name).read_text() for name in figures}
        assert "E2 frontier: weka-x32, 2 callers" in texts["frontier.svg"]
        assert (
            "the voice SLO, 2,010 ms: voice alone + 250 ms" in texts["frontier-weka-x32-long.svg"]
        )
        assert "a1-voice" not in texts["frontier-weka-x32-long.svg"]  # only its workload's points
        assert "No point of voice alone with 4 callers" in texts["frontier-4-callers.svg"]


def test_the_report_writes_the_table_and_the_csv(tmp_path: Path, capsys) -> None:
    dirs = [
        point(tmp_path / "runs", "a2-s8", "2", 3000, 2000),
        point(tmp_path / "runs", "a10-mps", "10", 2000, 2100),
        point(tmp_path / "runs", "a2-failed", "2", None, 0, status="failed"),
    ]
    out = tmp_path / "report"
    assert main(["e2", "report", *map(str, dirs), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "a2-failed: the point did not complete" in captured.err
    assert str(out / "frontier.md") in captured.out
    with (out / "frontier.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(COLUMNS)  # no point has the optional columns
    assert [r["point"] for r in rows] == ["a2-s8", "a10-mps"]  # arm 2 before arm 10
    assert rows[0]["v2v_p95_ms"] == "2000.0" and rows[0]["frontier"] == "True"
    assert rows[0]["refused"] == "3"  # 2 by the gateway's admission point, 1 by the engine
    assert rows[0]["gateway_background_tokens_per_s"] == "3000"
    assert (rows[0]["realtime_first_byte_p95_ms"], rows[0]["realtime_first_byte_source"]) == (
        "120.0",
        "log",
    )
    table = (out / "frontier.md").read_text()
    assert "\n## Workload: weka-x32, 2 callers\n" in table
    assert "\n### Arm 2\n" in table and "\n### Arm 10\n" in table
    # Each arm has one configuration here, on its own frontier: no mark says so.
    assert "| a2-s8: a2-s8 label |  | 16 | 3,000 | 3.00 | 1,600 | 2,000 | 2,500 |" in table
    assert "| 0 of 50 | 140 / 8,000 | 120 | 4 | — / 3 / — |" in table
    assert "•" not in table.split("## Workload")[1] and "~" not in table
    assert "Voice turn stages" not in table and "SM clock" not in table
    assert main(["e2", "report", str(dirs[2]), "--out", str(out)]) == 1


def test_the_report_gives_the_fixed_window_beside_the_whole_calls(tmp_path: Path) -> None:
    def windowed(run_dir: Path, tokens_per_s: float, seconds: float) -> Path:
        window = {
            "seconds": seconds,
            "requests": 1,
            "requests_per_s": 1.0,
            "tokens_per_s": tokens_per_s,
        }
        rewrite(run_dir, lambda s: s["background"].update(fixed_window=window))
        return run_dir

    dirs = [
        windowed(point(tmp_path / "r1", "disc-frozen", "4", 3710, 10000), 3580, 240.0),
        windowed(point(tmp_path / "r2", "disc-frozen", "4", 3726, 9000), 3600, 240.0),
        windowed(point(tmp_path / "r1", "disc-pr", "9-pr-ab48", 3430, 2200), 3400, 230.0),
        point(tmp_path / "r1", "a1-voice", "1", None, 1767),
    ]
    rows, _ = load_points(dirs)
    table = write_report(rows, tmp_path / "report")[1].read_text()
    assert "goodput over their first 240 s, the same span for every point; † marks one" in table
    assert "| Background tokens/s | First 240 s | Requests/s |" in table
    assert "| disc-frozen (r1) |  | 16 | 3,710 | 3,580 | 3.71 |" in table
    assert "| disc-pr: disc-pr label |  | 16 | 3,430 | 3,400 † | 3.43 |" in table
    assert "| a1-voice: a1-voice label |  | 0 | 0 | — | 0.00 |" in table
    medians = f"| {spread('3,718', '3,710', '3,726')} | {spread('3,590', '3,580', '3,600')} | |"
    assert medians in table
    with (tmp_path / "report" / "frontier.csv").open() as f:
        written = {r["point"]: r for r in csv.DictReader(f)}
    assert (
        written["disc-pr"]["fixed_window_tokens_per_s"],
        written["disc-pr"]["fixed_window_s"],
    ) == (
        "3400",
        "230.0",
    )


def test_the_report_takes_a_request_log_fetched_after_the_run(tmp_path: Path) -> None:
    run_dir = point(tmp_path, "a2-s8", "2", 3000, 2000)
    (run_dir / "gateway.jsonl").write_text(REQUESTS + "\n")  # backfilled from the journal
    [row], _ = load_points([run_dir])
    assert (row["realtime_first_byte_p95_ms"], row["realtime_first_byte_source"]) == (955.0, "log")
    assert "| 140 / 8,000 | 955 | 4 |" in write_report([row], tmp_path / "report")[1].read_text()


def smi(at: float, clock: str, temperature: str) -> str:
    """One sample of campaign.sh's nvidia-smi query, timestamped in UTC."""
    stamp = datetime.fromtimestamp(at, UTC).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    return f"{stamp}, {clock}, 13801 MHz, {temperature}, 150.21 W, 0x0000000000000000"


def test_the_report_shows_stages_and_clocks_only_for_points_that_have_them(tmp_path: Path) -> None:
    dirs = [point(tmp_path, "a2-s8", "2", 3000, 2000), point(tmp_path, "a2-s16", "2", 5000, 2600)]
    summary = json.loads((dirs[1] / "summary.json").read_text())
    stages = {s: {"n": 50, "p50": 100.0 * n, "p95": 150.0 * n} for n, s in enumerate(WATERFALL, 1)}
    events = {"sent": 300, "dropped": 0, "failed": 1}
    summary["worker"] = {"sessions": 2, "turns": 50, "waterfall": stages, "events": events}
    (dirs[1] / "summary.json").write_text(json.dumps(summary))
    samples = [
        smi(T0 - 10, "1530 MHz", "40"),  # before the calls: not counted
        smi(T0 + 10, "2805 MHz", "61"),
        smi(T0 + 20, "2745 MHz", "70"),
        smi(T0 + 30, "2685 MHz", "71"),
        smi(T0 + 40, "[N/A]", "[N/A]"),
        "2027/01/15 08:0",  # cut short when the sampling stopped
    ]
    (dirs[1] / "gpu.csv").write_text("\n".join(samples) + "\n")
    rows, _ = load_points(dirs)
    out = tmp_path / "report"
    write_report(rows, out)
    with (out / "frontier.csv").open() as f:
        written = {r["point"]: r for r in csv.DictReader(f)}
    assert list(written["a2-s8"]) == [*COLUMNS, *WORKER_COLUMNS, *GPU_COLUMNS]
    assert {c: written["a2-s8"][c] for c in GPU_COLUMNS} == dict.fromkeys(GPU_COLUMNS, "")
    sixteen = written["a2-s16"]
    assert (sixteen["transcript_p50_ms"], sixteen["e2e_p95_ms"], sixteen["events_failed"]) == (
        "100.0",
        "1050.0",
        "1",
    )
    assert (sixteen["gpu_sm_clock_p50_mhz"], sixteen["gpu_temperature_max_c"]) == ("2745", "71")
    table = (out / "frontier.md").read_text()
    assert "| SM clock p50 | Max temp |" in table and "| — / 3 / — | 2,745 | 71 |" in table
    assert "| — / 3 / — | — | — |" in table  # a2-s8, sampled by no campaign
    assert "| a2-s16 | 100 / 150 | 200 / 300 | 300 / 450 |" in table
    assert "| 700 / 1,050 | 300 / 0 / 1 |" in table
    # A point the worker logged no turn of has no row of stage times, only a count.
    stages = table.split("## Voice turn stages")[1].split("\n## ")[0]
    assert "| a2-s8 |" not in stages and "1 point without the worker's turn lines" in stages


def test_the_kv_samples_give_the_time_running_requests_leave_two_calls_short(
    tmp_path: Path,
) -> None:
    exposition = (
        'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.9\n'
        'vllm:num_requests_running{engine="0",model_name="m"} 3\n'
        'vllm:num_requests_waiting{engine="0",model_name="m"} 1\n'
        'vllm:num_requests_waiting{engine="1",model_name="m"} 2\n'
    )
    assert kv_values(exposition) == (0.9, 3.0, 3.0)
    assert kv_values("vllm:num_requests_running 3\n") is None  # no KV use: not vLLM's metrics
    path = tmp_path / "engine-kv.csv"
    rows = [
        (T0 - 1, 1.0, 9, 9),  # before the calls
        *((T0 + i, 0.97, 4, 2) for i in range(1, 4)),  # 68 of 2,274 blocks free: short of 128
        *((T0 + i, 0.5, 2, 0) for i in range(4, 8)),
    ]
    lines = ["time,kv_cache_usage,running,waiting", *(",".join(map(str, row)) for row in rows)]
    path.write_text("\n".join([*lines, f"{T0 + 9},0.4"]) + "\n")  # the last cut short
    engine = {"cache_config": {"block_size": 16, "num_gpu_blocks": 2275}}
    assert kv_samples(path, {"start": T0, "end": T0 + 100}, engine) == {
        "samples": 7,
        "running_share": 0.4286,
        "free_blocks": {"p50": 1137, "min": 68},
        "running": {"p50": 2.0, "max": 4.0},
        "waiting": {"p50": 0.0, "max": 2.0},
    }
    assert kv_samples(path, {"start": T0 + 50, "end": T0 + 100}, engine) is None  # none inside
    assert kv_samples(path, {"start": T0, "end": T0 + 100}, {"cache_config": {}}) is None
    assert kv_samples(tmp_path / "none.csv", {"start": T0, "end": T0 + 100}, engine) is None


def named(at: float, clock: str, gpu: str) -> str:
    """A sample of campaign.sh's query from 2026-10-25 on, which ends in the GPU's name."""
    return f"{smi(at, clock, '60')}, {gpu}"


def test_a_point_off_the_box_is_a_workload_of_its_own(tmp_path: Path) -> None:
    a100, box = "NVIDIA A100-SXM4-40GB", "NVIDIA GeForce RTX 5060 Ti"
    assert hardware(None, None) == hardware(box, "Qwen/Qwen3-4B-Instruct-2507-FP8") == ""
    assert hardware(a100, "Qwen/Qwen3-14B-FP8") == f"{a100}, Qwen/Qwen3-14B-FP8"
    dirs = {
        "a1-voice": point(tmp_path / "box", "a1-voice", "1", None, 2000),
        "a2-s16": point(tmp_path / "box", "a2-s16", "2", 5000, 2600),
        "a2-s8": point(
            tmp_path / "box", "a2-s8", "2", 3000, 2300
        ),  # its samples name the box's GPU
        "a100-voice": point(tmp_path / "a100", "a1-voice", "1", None, 1700),
        "a100-a2": point(tmp_path / "a100", "a2-s16", "2", 9000, 1900),
        "14b-a2": point(tmp_path / "14b", "a2-s16", "2", 7000, 2100),
    }
    for name, gpu in (("a2-s8", box), ("a100-voice", a100), ("a100-a2", a100), ("14b-a2", a100)):
        samples = [named(T0 + 10, "1410 MHz", gpu), named(T0 + 20, "1395 MHz", gpu)]
        (dirs[name] / "gpu.csv").write_text("\n".join(samples) + "\n")
    fourteen = {"model_config": {"model": "Qwen/Qwen3-14B-FP8", "revision": "abc"}}
    (dirs["14b-a2"] / "engine.json").write_text(json.dumps(fourteen))
    rows, _ = load_points(list(dirs.values()))
    workloads = {(r["campaign"], r["point"]): r["workload"] for r in rows}
    assert workloads == {
        ("box", "a1-voice"): "weka-x32, 2 callers",
        ("box", "a2-s16"): "weka-x32, 2 callers",
        ("box", "a2-s8"): "weka-x32, 2 callers",
        ("a100", "a1-voice"): f"weka-x32, 2 callers, {a100}",
        ("a100", "a2-s16"): f"weka-x32, 2 callers, {a100}",
        ("14b", "a2-s16"): f"weka-x32, 2 callers, {a100}, Qwen/Qwen3-14B-FP8",
    }
    assert [r["hardware"] for r in rows][:3] == ["", "", ""]  # the box's first
    # Each GPU reads its voice SLO against its own voice alone.
    groups = configurations(rows)
    assert voice_slo(groups, 2) == (2250.0, ["a1-voice"])
    assert voice_slo(groups, 2, a100) == (1950.0, ["a1-voice"])
    assert voice_slo(groups, 2, f"{a100}, Qwen/Qwen3-14B-FP8") is None  # C' needs its own
    write_report(rows, tmp_path / "report")
    table = (tmp_path / "report" / "frontier.md").read_text()
    assert f"## Workload: weka-x32, 2 callers, {a100}\n" in table
    assert (
        "allows a V2V p95 of 1,950 ms" in table and "`frontier-nvidia-a100-sxm4-40gb.svg`" in table
    )
    assert figure_name("weka-x32", 2) == "frontier.svg"


def test_points_that_differ_in_warm_up_turns_are_workloads_of_their_own(tmp_path: Path) -> None:
    # W5's calls open with a warm-up turn that no caller measures; W4's arms name none and open with
    # none. One report over both must neither compare their points nor read them against one SLO.
    dirs = [
        point(tmp_path / "w4", "a1-voice", "1", None, 1760),
        point(tmp_path / "w4", "a2-s8", "2", 3000, 2000),
        point(tmp_path / "w4", "a2-s16", "2", 5000, 2600),
        point(tmp_path / "w5", "w5-a1-voice", "1", None, 1700, warmup_turns=1),
        point(tmp_path / "w5", "w5-a2-s16", "2", 5500, 1950, warmup_turns=1),  # beats both of W4's
    ]
    rows, _ = load_points(dirs)
    warm = "weka-x32, 2 callers, 1 warm-up turn"
    assert [(r["point"], r["workload"], r["frontier"]) for r in rows] == [
        ("a1-voice", "weka-x32, 2 callers", True),
        ("a2-s8", "weka-x32, 2 callers", True),
        ("a2-s16", "weka-x32, 2 callers", True),
        ("w5-a1-voice", warm, True),
        ("w5-a2-s16", warm, True),
    ]
    groups = configurations(rows)
    assert voice_slo(groups, 2, warmup_turns=0) == (2010.0, ["a1-voice"])
    assert voice_slo(groups, 2, warmup_turns=1) == (1950.0, ["w5-a1-voice"])
    written = write_report(rows, tmp_path / "report")
    table = (tmp_path / "report" / "frontier.md").read_text()
    assert f"## Workload: {warm}\n" in table
    assert "(a1-voice: 1,760 ms). Figure: `frontier.svg`." in table
    assert "(w5-a1-voice: 1,700 ms). Figure: `frontier-1-warm-up-turn.svg`." in table
    if importlib.util.find_spec("matplotlib"):
        figures = [p.name for p in written if p.suffix == ".svg"]
        assert figures == ["frontier.svg", "frontier-1-warm-up-turn.svg"]
    # A report of one record set names no warm-up turns, as each committed report reads one.
    w5, _ = load_points(dirs[3:])
    assert [r["workload"] for r in w5] == ["weka-x32, 2 callers"] * 2
    assert voice_slo(configurations(w5), 2, warmup_turns=1) == (1950.0, ["w5-a1-voice"])
    assert figure_name("weka-x32", 2, warmup_turns=2) == "frontier-2-warm-up-turns.svg"


def test_a_records_engine_json_keeps_the_linear_kernel_where_the_engine_reported_it() -> None:
    kernel = {"linear_backend": "marlin", "moe_backend": "auto"}
    assert engine_settings(VLLM_CONFIG | {"kernel_config": kernel})["kernel_config"] == {
        "linear_backend": "marlin"
    }
    assert "kernel_config" not in engine_settings(VLLM_CONFIG)  # as records before it


def rewrite(run_dir: Path, change: Callable[[dict[str, Any]], None]) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    change(summary)
    (run_dir / "summary.json").write_text(json.dumps(summary))


def spread(median: str, low: str, high: str) -> str:
    """A repeated configuration's cell in frontier.md: its median, then its runs' range."""
    return f"{median} ({low}\N{EN DASH}{high})"


def test_repeats_of_a_point_are_one_configuration_placed_at_their_medians(tmp_path: Path) -> None:
    # Three runs of disc-frozen; one of them alone would beat disc-wide, their medians do not.
    dirs = [
        point(tmp_path / "w2c", "disc-frozen", "4", 5000, 2000),
        point(tmp_path / "r1", "disc-frozen", "4", 3000, 4000),
        point(tmp_path / "r2", "disc-frozen", "4", 4000, 3000),
        point(tmp_path / "w2c", "disc-wide", "4", 4500, 2500),
        point(tmp_path / "w2c", "disc-cap2", "6", 2000, 2200),
    ]
    rewrite(dirs[1], lambda s: s["voice"].update(errors=3))
    rewrite(dirs[2], lambda s: s["voice"].update(errors=2))
    stages = {s: {"n": 50, "mean": 100.0, "p50": 100.0, "p95": 150.0} for s in WATERFALL}
    rewrite(dirs[0], lambda s: s.update(worker={"turns": 50, "waterfall": stages}))
    rows, _ = load_points(dirs)
    assert [(r["point"], r["campaign"], r["frontier"]) for r in rows] == [
        ("disc-frozen", "r1", False),  # the configuration's runs together, at its median goodput
        ("disc-frozen", "r2", False),
        ("disc-frozen", "w2c", False),
        ("disc-wide", "w2c", True),
        ("disc-cap2", "w2c", True),
    ]
    [frozen, wide, cap2] = configurations(rows)
    assert (frozen["background_tokens_per_s"], frozen["v2v_p95_ms"], frozen["voice_errors"]) == (
        4000,
        3000,
        2,
    )
    assert (len(wide["runs"]), wide["v2v_p95_ms"], cap2["frontier"]) == (1, 2500, True)
    out = tmp_path / "report"
    write_report(rows, out)
    with (out / "frontier.csv").open() as f:
        assert [r["campaign"] for r in csv.DictReader(f)] == ["r1", "r2", "w2c", "w2c", "w2c"]
    table = (out / "frontier.md").read_text()
    assert "lists each run by its campaign, then the median and range" in table
    # Each run by its campaign, every one of them kept, then the configuration's medians and range.
    assert "| disc-frozen (r1) |  | 16 | 3,000 | 3.00 | 1,600 | 4,000 | 4,500 | 3 of 50 |" in table
    assert "| disc-frozen (w2c) |  | 16 | 5,000 | 5.00 | 1,600 | 2,000 | 2,500 | 0 of 50 |" in table
    medians = (
        f"| disc-frozen: disc-frozen label; median (range) of 3 runs |  | 16 "
        f"| {spread('4,000', '3,000', '5,000')} | | {spread('1,600', '1,600', '1,600')} "
        f"| {spread('3,000', '2,000', '4,000')} | {spread('3,500', '2,500', '4,500')} "
        f"| {spread('2', '0', '3')} | {spread('140', '140', '140')} / {spread('8,000', '8,000', '8,000')} "
        f"| {spread('120', '120', '120')} | | |"
    )
    lines = table.splitlines()
    assert lines[lines.index(medians) - 1].startswith("| disc-frozen (w2c) |")  # after its runs
    assert "| disc-wide: disc-wide label | • | 16 | 4,500 |" in table  # measured once: as before
    assert "| disc-frozen (w2c) | 100 / 150 |" in table and "| disc-wide | — / — |" not in table
    assert "4 points without the worker's turn lines are left out." in table  # the rest


def test_a_median_between_two_runs_keeps_its_half(tmp_path: Path) -> None:
    dirs = [point(tmp_path / c, "disc-frozen", "4", 4000, 3000) for c in ("w2c", "r1")]
    rewrite(dirs[1], lambda s: s["voice"].update(errors=1))
    rows, _ = load_points(dirs)
    table = write_report(rows, tmp_path / "report")[1].read_text()
    assert f"| {spread('0.5', '0', '1')} |" in table
    # A run's failed turn keeps its configuration off the frontier.
    assert f"|  | 16 | {spread('4,000', '4,000', '4,000')} |" in table


def test_a_failed_turn_is_censored_and_keeps_its_configuration_off_the_frontier(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    stages = {"end_of_turn_ms": 800, "turn_to_first_sentence_ms": 300, "tts_first_audio_ms": 200}
    ok = point(tmp_path, "a5-ok-s24", "5", 4000, 2000)
    turn_lines(ok, stages, 3, answered(1900, 2000, 2100))
    # vLLM's queued-token limit refused 48 of the 50 turns: they count at the reply timeout.
    refusing = point(tmp_path, "a5-queued-s24", "5", 5600, 2013)
    failed = [
        {"caller": 0, "turn": n, "error": "the agent did not start speaking"} for n in range(48)
    ]
    turn_lines(refusing, stages, 2, [*answered(1797, 2013), *failed])
    rewrite(refusing, lambda s: s["voice"].update(errors=48))
    rows, _ = load_points([point(tmp_path, "a1-voice", "1", None, 1760), ok, refusing])
    by = {r["point"]: r for r in rows}
    assert (by["a5-queued-s24"]["v2v_p50_ms"], by["a5-queued-s24"]["v2v_p95_ms"]) == (
        30_000,
        30_000,
    )
    # Its V2V is censored, not measured: off the frontier, and no rival of the measured point.
    assert not by["a5-queued-s24"]["frontier"] and by["a5-ok-s24"]["frontier"]
    assert [c["point"] for c in stage_waterfall(rows)] == ["a5-ok-s24"]
    out = tmp_path / "report"
    write_report(rows, out)
    table = (out / "frontier.md").read_text()
    assert "| a5-queued-s24: a5-queued-s24 label |  | 16 | 5,600 |" in table
    assert "| ≥30,000 | ≥30,000 | ≥30,000 | 48 of 50 |" in table
    figure = (out / "frontier.svg").read_text()
    assert (
        "Failed turns: a5-queued-s24, 48 of 50, its V2V censored at 30 s, off the scale" in figure
    )


@pytest.mark.parametrize(
    "name", ["leaseholder", "lifecycle"]
)  # as E2's arm files spell it, and its own
def test_the_lifecycle_policy_shows_what_it_deferred_and_what_it_capped(
    tmp_path: Path, name: str
) -> None:
    a9 = point(tmp_path, "disc-a9-s24", "9", 3400, 2057)
    run = json.loads((a9 / "run.json").read_text())
    run["arm"]["gateway"] = {"admission": {"policy": name, "headroom_tokens": 1024}}
    (a9 / "run.json").write_text(json.dumps(run))
    counters = {
        'lh_gateway_admission_admitted_total{class="realtime"}': 60,
        'lh_gateway_admission_deferred_total{reason="headroom"}': 41,
        "lh_gateway_admission_degraded_total": 60,
    }  # a counter that did not move is not kept: the critical window deferred nothing
    rewrite(a9, lambda s: s["gateway"].update(counters=counters))
    rows, _ = load_points([a9, point(tmp_path, "disc-frozen-s24", "4", 3700, 10138)])
    by = {r["point"]: r for r in rows}
    assert {column: by["disc-a9-s24"][column] for column in ADMISSION_COLUMNS} == {
        "deferred_headroom": 41,
        "deferred_critical": 0,
        "degraded_replies": 60,
        "realtime_replies": 60,
        "degraded_share": 1.0,
    }
    assert all(by["disc-frozen-s24"][column] is None for column in ADMISSION_COLUMNS)
    out = tmp_path / "report"
    write_report(rows, out)
    section = (out / "frontier.md").read_text().split("## The lifecycle policy")[1]
    assert "| disc-a9-s24 | 41 | 0 | 60 of 60 (100%) |" in section
    assert "disc-frozen-s24" not in section  # a point without the policy has no line
    with (out / "frontier.csv").open() as f:
        shares = {r["point"]: r["degraded_share"] for r in csv.DictReader(f)}
    assert shares == {"disc-a9-s24": "1.0", "disc-frozen-s24": ""}


def test_arm_9s_ablations_and_the_prs_are_configurations_of_their_own(tmp_path: Path) -> None:
    # As run, their arm files say arm 9; the report gives each its own arm, so none is placed on arm
    # 9's frontier or joined to it, and the records stay as they are.
    dirs = [
        point(tmp_path, "disc-a9-s24", "9", 3400, 2057),
        point(tmp_path, "disc-pr-ab48-s24", "9", 3500, 2193),
        point(tmp_path, "disc-a9-gateway-only-s24", "9", 3700, 10993),
    ]
    rows, _ = load_points(dirs)
    arms = {r["point"]: (r["arm"], r["frontier"]) for r in rows}
    assert arms == {
        "disc-a9-s24": ("9", True),
        "disc-pr-ab48-s24": ("9-pr-ab48", True),
        "disc-a9-gateway-only-s24": ("9-gateway-only", True),
    }
    assert all(arm.startswith("9-") for arm in CONFIGURATION_ARMS.values())


def test_the_report_names_the_prs_by_their_numbers(tmp_path: Path) -> None:
    # The records' labels name vLLM #63350 and #63351 "PR A" and "PRs A and B", as their arm files
    # did; the report shows the numbers, as the docs name them, and the records stay as run.
    labels = {
        "disc-pr-a-s24": "vLLM PR A: make-room's KV preemption alone, no gateway policy",
        "disc-pr-ab48-s24": "vLLM PRs A and B with a 48-token window, no gateway policy",
    }
    dirs = []
    for name, label in labels.items():
        run_dir = point(tmp_path, name, "9", 3500, 2193)
        run = json.loads((run_dir / "run.json").read_text())
        run["arm"]["label"] = label
        (run_dir / "run.json").write_text(json.dumps(run))
        dirs.append(run_dir)
    rows, _ = load_points(dirs)
    assert {r["point"]: r["label"] for r in rows} == {
        "disc-pr-a-s24": "vLLM #63350: make-room's KV preemption alone, no gateway policy",
        "disc-pr-ab48-s24": "vLLM #63350 and #63351 with a 48-token window, no gateway policy",
    }
    table = write_report(rows, tmp_path / "report")[1].read_text()
    assert "PR A" not in table and "| disc-pr-a-s24: vLLM #63350: make-room's" in table
    assert json.loads((dirs[0] / "run.json").read_text())["arm"]["label"] == labels["disc-pr-a-s24"]


def test_a_line_joins_an_arms_frontier_only_across_load_levels() -> None:
    def at(*sessions: int) -> list[dict[str, Any]]:
        return [{"runs": [{"sessions": n}]} for n in sessions]

    assert load_sweep(at(8, 16, 24)) and not load_sweep(at(24, 24)) and not load_sweep(at(24))


def turn_lines(run_dir: Path, stages: dict[str, float], turns: int, v2v: list[dict]) -> None:
    """A run directory's turn lines: the worker's stage means over ``turns`` turns, and the callers'
    turns.jsonl."""
    waterfall = {stage: {"n": turns, "mean": mean} for stage, mean in stages.items()}
    rewrite(run_dir, lambda s: s.update(worker={"turns": turns, "waterfall": waterfall}))
    (run_dir / "bots").mkdir(exist_ok=True)
    (run_dir / "bots" / "turns.jsonl").write_text("".join(json.dumps(t) + "\n" for t in v2v))


def answered(*v2v_ms: float) -> list[dict[str, Any]]:
    return [
        {"caller": 0, "turn": n, "v2v_ms": ms, "barge_in": False} for n, ms in enumerate(v2v_ms)
    ]


def test_the_warm_up_turns_are_in_no_voice_to_voice_sample(tmp_path: Path) -> None:
    run_dir = point(tmp_path, "a9-warm", "9", 3400, 2000)
    stages = {"end_of_turn_ms": 800, "turn_to_first_sentence_ms": 400, "tts_first_audio_ms": 200}
    warm = [
        {"caller": c, "turn": 0, "v2v_ms": 9000.0, "barge_in": False, "warmup": True}
        for c in (0, 1)
    ]
    turn_lines(run_dir, stages, 4, [*warm, *answered(1700, 1800, 1900, 2000)])
    [row], _ = load_points([run_dir])
    assert row["v2v_samples"] == [1700, 1800, 1900, 2000] and row["v2v_p95_ms"] == 1985.0
    [cell] = stage_waterfall([row])
    assert (cell["replies"], cell["v2v_mean_ms"]) == (4, 1850)


def test_the_stage_waterfall_stacks_each_configurations_stages_to_its_mean_v2v(
    tmp_path: Path, capsys
) -> None:
    frozen = [point(tmp_path / c, "disc-frozen", "4", 3700, 10800) for c in ("w2c", "r1")]
    barged = {"caller": 1, "turn": 0, "v2v_ms": 5000.0, "barge_in": True}
    turn_lines(
        frozen[0],
        {"transcript_ms": 790, "end_of_turn_ms": 800, "llm_ttft_ms": 2500}
        | {"turn_to_first_sentence_ms": 3000, "tts_first_audio_ms": 300},
        4,
        [*answered(4000, 4400), barged | {"v2v_after_barge_in_ms": 4600}],
    )
    turn_lines(
        frozen[1],
        {"transcript_ms": 810, "end_of_turn_ms": 820, "llm_ttft_ms": 2100}
        | {"turn_to_first_sentence_ms": 2600, "tts_first_audio_ms": 280},
        4,
        answered(3800, 4200, 4100, 4300),
    )
    cap2 = point(tmp_path / "w2c", "disc-cap2", "6", 3500, 8500)
    turn_lines(
        cap2,
        {"transcript_ms": 820, "end_of_turn_ms": 830, "llm_ttft_ms": 1500}
        | {"turn_to_first_sentence_ms": 2000, "tts_first_audio_ms": 250},
        2,
        answered(3300, 3500),
    )
    run = json.loads((cap2 / "run.json").read_text())
    run["arm"]["voice"]["preemptive_generation"] = False
    (cap2 / "run.json").write_text(json.dumps(run))
    voice = point(tmp_path / "w2", "a1-voice", "1", None, 1760)  # no turn lines
    rows, _ = load_points([*frozen, cap2, voice])
    [disc, capped] = stage_waterfall(rows)
    # Each stage's mean over both runs' turns; the V2V samples pooled as the callers pool them.
    assert {k: disc[k] for k in ("runs", "replies", *(s for s, _ in STACKED))} == {
        "runs": 2,
        "replies": 8,
        "end_of_turn_ms": 810,
        "turn_to_first_sentence_ms": 2800,
        "tts_first_audio_ms": 290,
    }
    assert (disc["v2v_mean_ms"], disc["v2v_p50_ms"], disc["v2v_p95_ms"]) == (4300, 4250, 4860)
    assert disc["transport_ms"] == 400  # the rest of the mean V2V
    # Under preemptive generation the LLM request starts at the final transcript.
    assert (disc["llm_request_ms"], disc["llm_ttft_ms"]) == (800, 2300)
    assert (capped["llm_request_ms"], capped["transport_ms"]) == (830, 320)
    out = tmp_path / "report"
    write_report(rows, out)
    table = (out / "frontier.md").read_text()
    assert (
        "| disc-frozen: disc-frozen label | 2 | 8 | 810 | 2,800 | 290 | 400 | 4,300 | 4,250 | 4,860 "
        "| 2,300 |"
    ) in table
    assert "| disc-cap2: disc-cap2 label | 1 | 2 | 830 | 2,000 | 250 | 320 | 3,400 |" in table
    assert "a1-voice: a1-voice label | 1 |" not in table
    with pytest.raises(ValueError, match="no turn lines of a1-voice"):
        write_report(rows, out, ["disc-cap2", "a1-voice"])
    args = ["e2", "report", *map(str, [*frozen, cap2, voice]), "--out", str(out)]
    assert main([*args, "--waterfall", "a1-voice"]) == 1
    assert "lh-bench e2: the report has no turn lines of a1-voice" in capsys.readouterr().err
    if importlib.util.find_spec("matplotlib"):
        assert main([*args, "--waterfall", "disc-cap2"]) == 0
        figure = (out / "waterfall.svg").read_text()
        assert "disc-cap2" in figure and "disc-frozen" not in figure
        write_report(rows, out)
        figure = (out / "waterfall.svg").read_text()
        assert "disc-frozen (2 runs)" in figure and "a1-voice" not in figure


SNAPSHOT = (
    "/home/alice/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/snapshots/859180"
)
VLLM_CONFIG = {
    "model_config": {"model": SNAPSHOT, "tokenizer": SNAPSHOT, "revision": None, "hf_token": None}
    | {"max_model_len": 32768, "dtype": "torch.bfloat16", "quantization": "fp8"},
    "cache_config": {"block_size": 16, "enable_prefix_caching": True, "cache_dtype": "auto"}
    | {"kv_cache_memory_bytes": 5368709120, "num_gpu_blocks": 2275, "gpu_memory_utilization": 0.7},
    "scheduler_config": {"policy": "priority", "max_num_batched_tokens": 512, "max_num_seqs": 256}
    | {"scheduler_cls": "lh_make_room_scheduler.AsyncScheduler"}
    | {"long_prefill_token_threshold": 256, "async_scheduling": True, "watermark": 0.1}
    | {"max_num_queued_tokens": None, "enable_chunked_prefill": True},
    "kv_transfer_config": {"kv_connector": "OffloadingConnector", "kv_role": "kv_both"}
    | {"kv_connector_extra_config": {"cpu_bytes_to_use": 21474836480, "spill": "/home/alice/kv"}},
    "additional_config": {"make_room": {"max_preemptions_per_step": 2}},
    "compilation_config": {"cache_dir": "/home/alice/.cache/vllm/torch_compile_cache"},
    "lora_config": None,
}
"""A vLLM engine's configuration as /server_info gives it, in a few of its 30-odd sections."""


def test_a_records_engine_json_says_what_the_point_ran_on() -> None:
    assert engine_settings(VLLM_CONFIG) == {
        "model_config": {"model": "Qwen/Qwen3-4B-Instruct-2507-FP8", "revision": "859180"}
        | {"max_model_len": 32768, "dtype": "torch.bfloat16", "quantization": "fp8"},
        "cache_config": {"block_size": 16, "enable_prefix_caching": True, "cache_dtype": "auto"}
        | {"kv_cache_memory_bytes": 5368709120, "num_gpu_blocks": 2275},
        "scheduler_config": {"policy": "priority", "max_num_batched_tokens": 512}
        | {"long_prefill_token_threshold": 256, "max_num_seqs": 256, "async_scheduling": True}
        | {"watermark": 0.1, "max_num_queued_tokens": None}
        | {"scheduler_cls": "lh_make_room_scheduler.AsyncScheduler"},
        "kv_transfer_config": {"kv_connector": "OffloadingConnector"}
        | {"kv_connector_extra_config": {"cpu_bytes_to_use": 21474836480, "spill": "~/kv"}},
        "additional_config": {"make_room": {"max_preemptions_per_step": 2}},
    }
    bare = engine_settings({"model_config": {"model": "Qwen/Qwen3-4B"}, "kv_transfer_config": None})
    assert bare["model_config"]["model"] == "Qwen/Qwen3-4B" and bare["kv_transfer_config"] is None
    sglang = {"model_path": "/Users/bob/models/qwen", "context_length": 32768, "api_key": None}
    settings = engine_settings(sglang | {"chunked_prefill_size": 512, "port": 8000})
    assert list(settings) == list(SGLANG_ENGINE) and settings["model_path"] == "~/models/qwen"
    assert (settings["chunked_prefill_size"], settings["page_size"]) == (512, None)


def whole(run_dir: Path) -> Path:
    """A run directory with everything lh-bench e2 run and campaign.sh leave in one."""
    turn_lines(
        run_dir,
        {"transcript_ms": 790, "end_of_turn_ms": 800, "llm_ttft_ms": 2500}
        | {"turn_to_first_sentence_ms": 3000, "tts_first_audio_ms": 300, "e2e_ms": 4110},
        2,
        answered(4400, 4500),
    )
    (run_dir / "gateway.jsonl").write_text(REQUESTS + "\n")  # exact first bytes; p95 955 ms
    samples = [smi(T0 + 10, "2805 MHz", "61"), smi(T0 + 20, "2745 MHz", "71")]
    (run_dir / "gpu.csv").write_text("\n".join(samples) + "\n")
    (run_dir / "voice-worker.log").write_text("registered worker\n")
    (run_dir / "arm.toml").write_text('arm = 4\nlabel = "x"\n')
    (run_dir / "engine.json").write_text(json.dumps(VLLM_CONFIG))
    run = json.loads((run_dir / "run.json").read_text())  # as e2 run wrote it before gpu_source
    run["versions"] = {"source": "03af728", "leaseholder_bench": "0.3.0", "aiperf": "0.13.0"}
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (run_dir / "bots" / "summary.json").write_text(json.dumps({"turns": 2, "errors": 0}))
    callers = {"config": {"api_key": "APIexample000000", "api_secret": "<redacted>"}}
    (run_dir / "bots" / "run.json").write_text(json.dumps(callers))
    for name in ("metrics/gateway-end.prom", "bots.log"):
        (run_dir / name).parent.mkdir(exist_ok=True)
        (run_dir / name).write_text("raw\n")
    # AIPerf's records: 50 agent requests of 1,020 tokens over the calls' 100 s, and one after them.
    (run_dir / "aiperf").mkdir()
    lines = [record(T0 + 2 * n) for n in range(1, 51)] + [record(T0 + 101)]
    (run_dir / "aiperf" / "profile_export.jsonl").write_text("\n".join(lines) + "\n")
    return run_dir


def test_a_compact_record_gives_the_report_of_its_run_directory(tmp_path: Path, capsys) -> None:
    runs = tmp_path / "runs"
    dirs = [
        whole(point(runs / "w2c", "disc-frozen", "4", 3700, 10800)),
        whole(point(runs / "r1", "disc-frozen", "4", 3500, 9800)),
        point(runs / "w2c", "disc-cap2", "6", 3400, 8500),  # a summary from before the logs
        point(runs / "w2c", "disc-cap8", "6", None, 0, status="failed"),
    ]
    out = tmp_path / "records"
    assert main(["e2", "export", *map(str, dirs), "--out", str(out)]) == 0
    records = [out / "w2c" / "disc-frozen", out / "r1" / "disc-frozen", out / "w2c" / "disc-cap2"]
    printed = capsys.readouterr()
    assert printed.out.split() == [str(record) for record in records]  # CAMPAIGN/POINT
    assert "disc-cap8: the point did not complete" in printed.err
    record = records[0]
    kept = sorted(str(p.relative_to(record)) for p in record.rglob("*") if p.is_file())
    assert kept == sorted([*RECORD, "summary.json", "engine.json"])
    assert json.loads((record / "engine.json").read_text()) == engine_settings(VLLM_CONFIG)
    # The run's run.json, the GPU node's revision null beside cyan's: it was not recorded then.
    run = json.loads((dirs[0] / "run.json").read_text())
    recorded = json.loads((record / "run.json").read_text())
    assert recorded["versions"] == {
        "source": "03af728",
        "gpu_source": None,
        "leaseholder_bench": "0.3.0",
        "aiperf": "0.13.0",
    }
    assert {k: v for k, v in recorded.items() if k != "versions"} == {
        k: v for k, v in run.items() if k != "versions"
    }
    run["versions"]["gpu_source"] = {"phastform": "v0.3-30-g6851e13"}  # a run that recorded it
    (dirs[0] / "run.json").write_text(json.dumps(run))
    again = export_point(dirs[0], tmp_path / "again")
    assert json.loads((again / "run.json").read_text())["versions"] == run["versions"]
    written = [p.read_text() for p in out.rglob("*") if p.is_file()]
    assert not any("APIexample000000" in text or "/home/" in text for text in written)
    summary = json.loads((record / "summary.json").read_text())
    assert summary["gateway"]["realtime_first_byte_ms"]["p95"] == 955.0  # exact, from the log
    assert summary["gateway"]["background_first_byte_ms"]["source"] == "log"
    assert summary["gpu"] == {"sm_clock_p50_mhz": 2775, "temperature_max_c": 71}
    # From AIPerf's records, over the calls' 100 s, which are shorter than the fixed window.
    assert summary["background"]["fixed_window"] == {
        "seconds": 100.0,
        "requests": 50,
        "requests_per_s": 0.5,
        "tokens_per_s": 510.0,
    }
    full, _ = load_points(dirs[:3])
    compact, _ = load_points(records)
    assert compact == full
    assert full[0]["campaign"] == "r1" and full[0]["gpu_sm_clock_p50_mhz"] == 2775
    reports = [write_report(rows, tmp_path / name) for name, rows in (("a", full), ("b", compact))]
    for a, b in zip(*reports, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name  # the figures too: same bytes every time
    assert main(["e2", "export", str(dirs[3]), "--out", str(out)]) == 1
    assert "no complete point to export" in capsys.readouterr().err


def test_a_record_names_no_account_where_the_run_named_a_file_in_its_home(tmp_path: Path) -> None:
    # The gateway's configuration in run.json names its cache-salt secret file, under the home
    # directory of cyan's account; the record names it from ~, as it does the engine's paths.
    run_dir = point(tmp_path / "runs" / "w5-r1", "disc-frozen", "4", 3700, 10800)
    run = json.loads((run_dir / "run.json").read_text())
    secret = ".config/leaseholder/cache-salt-secret"
    run["cyan"] = {"gateway": {"config": {"cache_salt_secret_file": f"/home/someone/{secret}"}}}
    (run_dir / "run.json").write_text(json.dumps(run))
    record = export_point(run_dir, tmp_path / "records")
    recorded = json.loads((record / "run.json").read_text())
    assert recorded == run | {
        "cyan": {"gateway": {"config": {"cache_salt_secret_file": f"~/{secret}"}}}
    }
    assert not any(
        re.search(r"/(home|Users|root)/", p.read_text()) for p in record.rglob("*") if p.is_file()
    )


def test_the_report_gives_the_cpu_tiers_loads_and_stores_in_tokens(tmp_path: Path) -> None:
    # A point measured before the summary kept the CPU tier's counts: its snapshots have them.
    run_dir = whole(point(tmp_path / "runs" / "r1", "disc-a9-s24", "9", 3400, 2057))
    sources = {"local_compute": 931_816, "local_cache_hit": 17_776, "external_kv_transfer": 44_432}
    rewrite(run_dir, lambda s: s["engine"].update(prompt_tokens_by_source=sources, preemptions=38))
    per_token = 5368709120 / (2275 * 16)  # VLLM_CONFIG's pool: 147,492.0 bytes a token
    for when, loaded, stored in (("start", 0, 0), ("end", 100_000, 1_000_000)):
        counts = {"preemptions": 0, "computed": 0, "gpu": 0, "refused": 0}
        engine = ENGINE.format(
            **counts, loaded=round(loaded * per_token), stored=stored * per_token
        )
        (run_dir / "metrics" / f"engine-{when}.prom").write_text(engine)
    old = json.loads((run_dir / "summary.json").read_text())
    _, summary = compact(run_dir)
    assert summary["engine"] == old["engine"] | {
        "kv_offload_load_bytes": round(100_000 * per_token),
        "kv_offload_store_bytes": round(1_000_000 * per_token),
    }  # the summary's own counts as they were, the CPU tier's after them
    sglang = point(tmp_path / "runs" / "r1", "disc-a7b-sglang-s24", "7b", 4700, 9000)
    rows, _ = load_points([run_dir, sglang])
    a9 = next(r for r in rows if r["point"] == "disc-a9-s24")
    assert {column: a9[column] for column in CPU_TIER_COLUMNS} == {
        "first_schedule_computed_tokens": 931_816,
        "first_schedule_gpu_hit_tokens": 17_776,
        "first_schedule_cpu_hit_tokens": 44_432,
        "cpu_tier_loaded_tokens": 100_000,
        "cpu_tier_stored_tokens": 1_000_000,
        "kv_bytes_per_token": 147_492.0,
    }
    out = tmp_path / "report"
    write_report(rows, out)
    with (out / "frontier.csv").open() as f:
        assert list(next(csv.DictReader(f))) == [
            *COLUMNS,
            *GPU_COLUMNS,
            *CPU_TIER_COLUMNS,
            *WINDOW_COLUMNS,
        ]
    section = (
        (out / "frontier.md").read_text().split("## The engine's prompt tokens and its CPU tier")[1]
    )
    assert "| Computed at first scheduling | GPU-pool hits at first scheduling |" in section
    # 100,000 tokens loaded, 44,432 of them at first scheduling: 55,568 at resumes.
    assert (
        "| disc-a9-s24 | 38 | 931,816 | 17,776 | 44,432 | 100,000 | 55,568 | 1,000,000 |" in section
    )
    assert "disc-a7b-sglang-s24" not in section.split("## Stage waterfall")[0]  # SGLang keeps none
    # The record keeps them, and gives the same report.
    record = export_point(run_dir, tmp_path / "records")
    assert json.loads((record / "summary.json").read_text())["engine"] == summary["engine"]
    again, _ = load_points([record, sglang])
    assert again == rows


def unredacted(data: Any) -> list[str]:
    """The credential settings (e2.SECRETS) in a JSON document that hold anything but a redaction."""
    if isinstance(data, list):
        return [key for item in data for key in unredacted(item)]
    if not isinstance(data, dict):
        return []
    found = [
        key
        for key, value in data.items()
        if key in SECRETS and value and value not in ("<redacted>", ["<redacted>"] * len(value))
    ]
    return found + [key for value in data.values() for key in unredacted(value)]


@pytest.mark.parametrize("results", [RESULTS, W5_RESULTS], ids=lambda p: p.name)
def test_the_committed_records_are_complete_and_hold_no_credential(results: Path) -> None:
    assert unredacted({"tenants": {"a": {"keys_sha256": ["<sha256>"]}}}) == ["keys_sha256"]
    dirs = sorted(run.parent for run in results.glob("*/*/run.json"))  # as make reproduce-e2 does
    assert dirs
    rows, incomplete = load_points(dirs)
    assert len(rows) == len(dirs) and not incomplete
    for run_dir in dirs:
        kept = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())
        assert kept == sorted([*RECORD, "summary.json", "engine.json"]), run_dir
        for path in run_dir.rglob("*.json"):
            assert not unredacted(json.loads(path.read_text())), path
        for path in (p for p in run_dir.rglob("*") if p.is_file()):
            assert not re.search(r"/(home|Users|root)/", path.read_text()), path  # no account


def test_the_committed_records_give_their_committed_report(tmp_path: Path) -> None:
    # Each set of records keeps the report it gives in report/, every file make reproduce-e2 writes
    # (make reproduce-e2 E2_REPORT=<its report/> rewrites it), so that a change to the report's code
    # shows in what it changes there.
    reports = sorted(REPO.glob("benchmarks/results/*/report"))
    assert {RESULTS / "report", W5_RESULTS / "report"} <= set(reports)
    for committed in reports:
        dirs = sorted(run.parent for run in committed.parent.glob("*/*/run.json"))
        rows, incomplete = load_points(dirs)
        assert dirs and not incomplete
        out = tmp_path / committed.parent.name
        written = write_report(rows, out)
        assert sorted(p.name for p in written) == sorted(p.name for p in committed.iterdir())
        for path in written:
            assert path.read_bytes() == (committed / path.name).read_bytes(), committed / path.name


@pytest.mark.parametrize("results", [RESULTS, W5_RESULTS], ids=lambda p: p.name)
def test_a_committed_record_exports_to_itself(tmp_path: Path, results: Path) -> None:
    # A record leaves out the files compact() reads beside run.json and summary.json, so exporting it
    # gives it back byte for byte: lh-bench e2 report reads records and run directories alike.
    records = sorted(run.parent for run in results.glob("*/*/run.json"))
    assert records
    for record in records:
        again = export_point(record, tmp_path)
        assert again is not None and again.relative_to(tmp_path) == record.relative_to(results)
        files = sorted(p.relative_to(record) for p in record.rglob("*") if p.is_file())
        assert sorted(p.relative_to(again) for p in again.rglob("*") if p.is_file()) == files
        for name in files:
            assert (again / name).read_bytes() == (record / name).read_bytes(), record / name


def test_each_w5_record_ran_the_scheduler_its_variant_loads_on_the_revision_it_names() -> None:
    # W5's harness checked the engine's scheduler class before each point, so that make-room as W5
    # fixed it and as v0.3 built it (make-room-v03) cannot pass for one another, and each record names
    # the revision deployed on the GPU node: the one cyan ran.
    runs = sorted(W5_RESULTS.glob("*/*/run.json"))
    assert len(runs) == 72
    for path in runs:
        run = json.loads(path.read_text())
        variant = tomllib.loads((path.parent / "arm.toml").read_text()).get("engine", {})
        loads = flag_values(variant_flags(variant.get("variant", "frozen"))).get(SCHEDULER)
        engine = json.loads((path.parent / "engine.json").read_text())
        assert engine["scheduler_config"]["scheduler_cls"] == loads, path
        assert (SCHEDULER in run["engine"]["flags"]["checked"]) == (loads is not None), path
        assert run["versions"]["gpu_source"] == {"phastform": run["versions"]["source"]}, path


def test_the_report_reads_the_lines_the_voice_worker_writes() -> None:
    source = VOICE_EVENTS.read_text()
    [stages] = [
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "STAGES" for t in node.targets)
    ]
    assert stages == WATERFALL
    assert f'logging.getLogger("{TURNS_LOGGER}")' in source
    # The sessions' end lines come from the module's own logger.
    assert SESSIONS_LOGGER == "leaseholder_voice.events" and "logging.getLogger(__name__)" in source
    assert '"session ended"' in source


def test_a_point_waits_for_the_gateway_by_series_it_exports() -> None:
    # e2 run counts the requests the gateway ended by their outcomes and waits for it to go idle by
    # its gauges: a gauge it read under a name the gateway does not export would read as idle at once.
    exported = {name for name, _, _ in metric_samples(GATEWAY_METRICS.read_text())}
    assert {OUTCOMES, *BUSY} <= exported


def test_the_record_and_the_report_read_series_the_gateway_exports() -> None:
    # Each counter series of the gateway's exposition moves by a count of its own over the window, so
    # that each count the record and the report take is the sum of the series they read: a family or
    # a label the gateway renamed leaves the series below unexported, and one the record or the
    # report reads under another name leaves its count short.
    before = GATEWAY_METRICS.read_text()
    moves: dict[tuple[str, frozenset[tuple[str, str]]], int] = {}
    after = []
    for line in before.splitlines():
        for name, labels, _ in metric_samples(line):
            if name.endswith("_total"):
                moves[name, frozenset(labels.items())] = len(moves) + 1
                line = f"{line.rsplit(' ', 1)[0]} {len(moves)}"
        after.append(line)

    def moved(name: str, labels: Mapping[str, str] | None = None) -> int:
        """What the series of ``name`` that carry ``labels`` moved by; 0 where none is exported."""
        wanted = (labels or {}).items()
        return sum(n for (series, held), n in moves.items() if series == name and wanted <= held)

    cell = gateway_counts(before, "\n".join(after) + "\n", 1.0, [])
    assert cell["shed"] == moved(OUTCOMES, {"outcome": "shed"}) > 0
    assert cell["refused"] == moved(REFUSED) > 0
    assert cell["resubmitted"] == moved(RESUBMITS) > 0
    assert cell["discarded_tokens"] == moved(DISCARDED) > 0
    usage = [
        moved(USAGE, {"class": "background", "kind": kind}) for kind in ("prompt", "completion")
    ]
    assert cell["background_tokens_per_s"] == sum(usage) and all(usage)
    counts = lifecycle_counts(cell["counters"])
    deferred = "lh_gateway_admission_deferred_total"
    assert counts["deferred_headroom"] == moved(deferred, {"reason": "headroom"}) > 0
    assert counts["deferred_critical"] == moved(deferred, {"reason": "critical"}) > 0
    assert counts["degraded_replies"] == moved("lh_gateway_admission_degraded_total") > 0
    admitted = "lh_gateway_admission_admitted_total"
    assert counts["realtime_replies"] == moved(admitted, {"class": "realtime"}) > 0


def test_the_figure_needs_matplotlib(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    dirs = [
        point(tmp_path, "a1-voice", "1", None, 1760),
        point(tmp_path, "a2-s8", "2", 3000, 2000),
        point(tmp_path, "a2-s16", "2", 5000, 2600),
    ]
    rows, _ = load_points(dirs)
    written = write_report(rows, tmp_path / "report")
    assert (tmp_path / "report" / "frontier.svg") in written
    figure = (tmp_path / "report" / "frontier.svg").read_text()
    assert "<svg" in figure and "whiskers" not in figure
    assert "stock vLLM" not in figure  # one family: the arms' names, no legend
    repeated = point(tmp_path / "r1", "a2-s16", "2", 4000, 3000)
    rows, _ = load_points([*dirs, repeated])
    write_report(rows, tmp_path / "report")
    assert "whiskers span repeated runs" in (tmp_path / "report" / "frontier.svg").read_text()


def test_the_figure_draws_families_and_leaves_a_collapse_off_its_scale(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    dirs = [
        point(tmp_path, "a1-voice", "1", None, 1760),
        point(tmp_path, "a2-s8", "2", 3000, 2000),
        point(tmp_path, "a3-s8", "3", 3000, 2100),
        point(tmp_path, "a2-s16", "2", 5000, 2600),
        point(tmp_path, "a8a-s16", "8a", 5000, 2500),
        point(tmp_path, "a7b-s16", "7b", 5000, 9000),  # past the far outliers' fence, 4,225 ms
    ]
    rows, _ = load_points(dirs)
    write_report(rows, tmp_path / "report")
    figure = (tmp_path / "report" / "frontier.svg").read_text()
    # One color per family, named in the legend; the arms by their names.
    legend = ("stock vLLM (arms 1\N{EN DASH}5)", "engines (arm 7)", "routers (arm 8)")
    assert all(f"<!-- {family} -->" in figure for family in legend)
    assert all(f"<!-- {arm} -->" in figure for arm in ("1", "2", "3", "8a"))
    assert "Off the scale: a7b-s16, 9,000 ms at 5,000 tokens/s." in figure
    assert "<!-- 7b -->" not in figure  # its only point is off the scale
    assert "The dashed line is the voice SLO, 2,010 ms" in figure


def test_the_dot_figure_draws_the_named_configurations_in_order(tmp_path: Path, capsys) -> None:
    pytest.importorskip("matplotlib")
    dirs = [
        point(tmp_path / "r1", "w5-a1-voice", "1", None, 1760),
        point(tmp_path / "r1", "w5-frozen", "2", 3300, 10000),
        point(tmp_path / "r2", "w5-frozen", "2", 3400, 11000),
        point(tmp_path / "r1", "w5-pr", "9", 3200, 2200),
        point(tmp_path / "r2", "w5-pr", "9", 3100, 2300),
    ]
    for run_dir in dirs[
        1:
    ]:  # the agents' tokens/s over the fixed window, as W5's records have them
        summary = json.loads((run_dir / "summary.json").read_text())
        bg = summary["background"]
        bg["fixed_window"] = {"tokens_per_s": bg["tokens_per_s"], "seconds": 240.0}
        (run_dir / "summary.json").write_text(json.dumps(summary))
    rows, _ = load_points(dirs)
    figure_toml = tmp_path / "figure.toml"
    figure_toml.write_text(
        '[[row]]\npoint = "w5-frozen"\nlabel = "Tuned vLLM"\n\n'
        '[[row]]\npoint = "w5-pr"\nlabel = "Both PRs"\nchange = true\n'
    )
    out = tmp_path / "report"
    assert out / "voice-p95.svg" in write_report(rows, out, dots=read_dots(figure_toml))
    figure = (out / "voice-p95.svg").read_text()
    # The rows in the file's order with their runs, each one's goodput against the first row's
    # (3,150 against 3,350 tokens/s), and the SLO that voice alone sets.
    assert figure.index("<!-- Tuned vLLM (2 runs) -->") < figure.index("<!-- Both PRs (2 runs) -->")
    assert "<!-- 100% -->" in figure and "<!-- 94% -->" in figure
    assert "<!-- the SLO: voice alone (1.76 s) + 250 ms -->" in figure
    write_report(rows, tmp_path / "again", dots=read_dots(figure_toml))
    assert (tmp_path / "again" / "voice-p95.svg").read_bytes() == figure.encode()
    # A point the report lacks, or a row the file gets wrong, names what is wrong.
    figure_toml.write_text('[[row]]\npoint = "w5-gone"\nlabel = "Gone"\n')
    with pytest.raises(ValueError, match="0 configurations of w5-gone in the report, not 1"):
        write_report(rows, out, dots=read_dots(figure_toml))
    for bad, error in (
        ('[[row]]\npoint = "w5-pr"\n', "a row without a point or a label"),
        ('[[row]]\npoint = "w5-pr"\nlabel = "PR"\ncolor = "blue"\n', "keys other than"),
        ('[[row]]\npoint = "w5-pr"\nlabel = "PR"\nchange = "yes"\n', "not true or false"),
        ("", "no \\[\\[row\\]\\] tables"),
    ):
        figure_toml.write_text(bad)
        with pytest.raises(ValueError, match=error):
            read_dots(figure_toml)
    args = ["e2", "report", *map(str, dirs), "--out", str(out), "--dots", str(figure_toml)]
    assert main(args) == 1
    assert "no [[row]] tables" in capsys.readouterr().err


def test_a_point_summary_reads_its_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "a1"
    (run_dir / "bots").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    bots = {
        "turns": 50,
        "errors": 1,
        "v2v_ms": {"n": 59, "p50": 1600.0, "p95": 1760.0, "p99": 1890.0},
        "barge_in_to_silence_ms": {"n": 9, "p50": 630.0, "p95": 690.0, "p99": 700.0},
        "barge_ins": 9,
        "caller_max_lateness_ms": 0.0,
    }
    (run_dir / "bots" / "summary.json").write_text(json.dumps(bots))
    for when, text in (("start", GATEWAY_BEFORE), ("end", GATEWAY_AFTER)):
        (run_dir / "metrics" / f"gateway-{when}.prom").write_text(text)
        (run_dir / "metrics" / f"engine-{when}.prom").write_text("")
    # Without the gateway's request lines of the calls, the point has no first byte: it fails.
    with pytest.raises(ValueError, match="the gateway logged no request line for the calls"):
        summarize_point(run_dir, T0, T0 + 321.04, agents=False)
    (run_dir / "gateway.jsonl").write_text("lh-gateway serving\n")  # a gateway that logs text
    with pytest.raises(ValueError, match="the gateway logged no request line for the calls"):
        summarize_point(run_dir, T0, T0 + 321.04, agents=False)
    (run_dir / "gateway.jsonl").write_text(REQUESTS + "\n")
    summary = summarize_point(run_dir, T0, T0 + 321.04, agents=False)
    assert summary["gateway"]["realtime_first_byte_ms"]["p95"] == 955.0
    assert summary["window_s"] == 321.0 and summary["background"] is None
    assert summary["gateway"]["background_tokens_per_s"] == 317.7  # 102,000 tokens over the window
    assert summary["voice"] == {k: bots[k] for k in summary["voice"]}
    assert summary["engine"]["preemptions"] is None
    bots["v2v_ms"] = {"n": 0}
    (run_dir / "bots" / "summary.json").write_text(json.dumps(bots))
    with pytest.raises(ValueError, match="measured no voice-to-voice latency"):
        summarize_point(run_dir, T0, T0 + 321, agents=False)


def test_the_gpu_columns_read_the_lent_gpu_of_several_alone(tmp_path: Path) -> None:
    # A sampler file from a host with four GPUs, the point's at 1,410 MHz and three idle at 210: the file's header
    # names the lent one, whose rows alone count; with no header the GPUs are not mixed (nothing is kept), and a
    # file of one GPU, a UUID or not (the box's records), reads as before.
    start = datetime(2026, 10, 25, 9, 0, 0, tzinfo=UTC).timestamp()
    window = {"start": start, "end": start + 60}
    rows = []
    for second in range(3):
        for index, (clock, temperature) in enumerate(((1410, 61), (210, 34), (210, 34), (210, 35))):
            rows.append(
                f"2026/10/25 09:00:0{second}.000, {clock} MHz, 1215 MHz, {temperature}, 55.00 W, "
                f"0x0000000000000000, NVIDIA A100-SXM4-40GB, GPU-{index}"
            )
    named = tmp_path / "named.csv"
    named.write_text("# gpu GPU-0\n" + "\n".join(rows) + "\n")
    assert gpu_samples(named, window) == {
        "gpu_sm_clock_p50_mhz": 1410,
        "gpu_temperature_max_c": 61,
        "gpu_name": "NVIDIA A100-SXM4-40GB",
    }
    unnamed = tmp_path / "unnamed.csv"
    unnamed.write_text("\n".join(rows) + "\n")
    assert gpu_samples(unnamed, window) == {
        "gpu_sm_clock_p50_mhz": None,
        "gpu_temperature_max_c": None,
    }
    one = tmp_path / "one.csv"
    one.write_text("\n".join(row.rsplit(", GPU-", 1)[0] for row in rows[::4]) + "\n")
    assert gpu_samples(one, window)["gpu_sm_clock_p50_mhz"] == 1410
    # The sampler before it named the GPU in each row: several rows a second are several GPUs, none of them named.
    old = tmp_path / "old.csv"
    old.write_text("\n".join(row.rsplit(", GPU-", 1)[0] for row in rows) + "\n")
    assert gpu_samples(old, window) == {
        "gpu_sm_clock_p50_mhz": None,
        "gpu_temperature_max_c": None,
    }
