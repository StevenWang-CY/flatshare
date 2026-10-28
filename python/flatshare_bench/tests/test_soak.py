"""``flatshare-bench soak``: the scraper's series against the gateway's committed expositions and vLLM's metric names,
the summary's numbers on a synthetic record and the compact record that gives them back, the guard's checks and its
pauses, and whole soaks against a fake cyan (a gateway on a Unix socket, a recording user manager, stand-ins for AIPerf
and the callers that run as real processes): their tools, scrapes and record, SIGINT, the guard's pause, a tool that
fails, the lock and the checks that refuse a start."""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
from servers import UnixServer, serve, socket_file, stop

from flatshare_bench.cli import main
from flatshare_bench.e2 import tools
from flatshare_bench.e2.arm import Agents, Arm, E2Error
from flatshare_bench.e2.cyan import Host, exclusive
from flatshare_bench.soak import guard as guard_module
from flatshare_bench.soak import run as soak_run
from flatshare_bench.soak.guard import Check, Guard, NodeGuard, node_guard, parse_guards, parse_list
from flatshare_bench.soak.run import (
    SoakOptions,
    Tools,
    aiperf_env,
    gateway_nodes,
    read_worker_status,
    run_soak,
)
from flatshare_bench.soak.scrape import (
    ENGINE_KEPT,
    GATEWAY_KEPT,
    RawScrapes,
    Scraper,
    SoakSystemd,
    kept_series,
    manager_lines,
    own_lines,
    parse_key,
    read_raw,
    series_key,
)
from flatshare_bench.soak.summary import (
    _Counts,
    _increase,
    _paused,
    export,
    load,
    read_series_csv,
    read_turns_csv,
    render_hour,
    summarize,
    write_series_csv,
    write_turns_csv,
)
from flatshare_stats import metric_samples

REPO = Path(__file__).resolve().parents[3]
FAKES = Path(__file__).parent / "fakes"
GATEWAY_FIXTURE = REPO / "crates" / "flatshare-gateway" / "metrics.txt"
NODES_FIXTURE = REPO / "crates" / "flatshare-gateway" / "metrics-nodes.txt"
VLLM_METRICS = REPO / "ops" / "metrics" / "vllm-0.32.0.txt"
EXAMPLE = REPO / "deploy" / "cyan" / "gateway.example.toml"
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
needs_waitid = pytest.mark.skipif(not hasattr(os, "waitid"), reason=tools.NEEDS_WAITID)


@pytest.fixture(autouse=True)
def _quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "POLL_S", 0.02)
    monkeypatch.setattr(soak_run, "POLL_S", 0.05)
    for name in ("FAKE_AIPERF", "FAKE_ARRIVALS", "FAKE_ARRIVALS_GAP", "FAKE_ARRIVALS_TURN"):
        monkeypatch.delenv(name, raising=False)


# The scraper's series


def test_the_scraper_keeps_every_series_of_the_gateways_exposition() -> None:
    # Every family the soak keeps is one the gateway exports (metrics.txt, and metrics-nodes.txt for [[nodes]]):
    # a name it does not export would be a count the summary reads as absent.
    exported = {
        name
        for path in (GATEWAY_FIXTURE, NODES_FIXTURE)
        for name, _, _ in metric_samples(path.read_text())
    }
    assert set(GATEWAY_KEPT) <= exported, sorted(set(GATEWAY_KEPT) - exported)
    series = kept_series(GATEWAY_FIXTURE.read_text(), GATEWAY_KEPT)
    for key in (
        'flatshare_gateway_sli_valid_total{class="realtime"}',
        'flatshare_gateway_sli_good_total{class="background"}',
        'flatshare_gateway_requests_offered_total{class="background",route="/v1/chat/completions",tenant="agents"}',
        'flatshare_gateway_requests_terminal_total{class="background",outcome="completed",'
        'route="/v1/chat/completions",tenant="agents"}',
        'flatshare_gateway_request_outcomes_total{class="realtime",outcome="timeout_total"}',
        'flatshare_gateway_engine_state{state="serving"}',
        'flatshare_gateway_usage_tokens_total{class="background",kind="prompt"}',
        'flatshare_gateway_in_flight{class="realtime"}',
        "flatshare_gateway_node_ready",
        'flatshare_gateway_admission_refused_total{class="realtime",reason="queue_full"}',
    ):
        assert series.get(key) == 0.0, key
    # Nothing but the kept families: the histograms' buckets stay in the raw scrapes.
    assert {parse_key(key)[0] for key in series} <= set(GATEWAY_KEPT)
    assert not any("_bucket" in key for key in series)


def test_the_scraper_reads_every_nodes_family_from_its_exposition() -> None:
    # With [[nodes]] the engine's families carry a node label, and the nodes add their own (OpenMetrics, # EOF).
    series = kept_series(NODES_FIXTURE.read_text(), GATEWAY_KEPT)
    for node in ("gray", "phastform"):
        assert series[f'flatshare_gateway_node_healthy{{node="{node}"}}'] == 0.0
        assert series[f'flatshare_gateway_engine_state{{node="{node}",state="serving"}}'] == 0.0
        assert series[f'flatshare_gateway_node_in_flight{{class="realtime",node="{node}"}}'] == 0.0
        assert (
            series[
                f'flatshare_gateway_placements_total{{class="realtime",node="{node}",reason="moved"}}'
            ]
            == 0
        )
        assert (
            series[f'flatshare_gateway_node_component_healthy{{component="tts",node="{node}"}}']
            == 0
        )
    assert (
        series['flatshare_gateway_placements_refused_total{class="realtime",reason="no_call_slot"}']
        == 0
    )
    families = {parse_key(key)[0] for key in series}
    assert families == {name for name in GATEWAY_KEPT if name in NODES_FIXTURE.read_text()}


def test_the_scraper_keeps_vllms_series_without_the_models_path() -> None:
    # vLLM names its model by its path in the Hugging Face cache, under the home directory: the kept series leave the
    # label out, and every family kept is one of vLLM 0.32's.
    names = {
        line.split()[0] for line in VLLM_METRICS.read_text().splitlines() if line and line[0] != "#"
    }
    assert {name.removesuffix("_total") for name in ENGINE_KEPT} <= names
    model = (
        'model_name="/home/alice/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/85918040"'
    )
    exposition = (
        f'vllm:num_preemptions_total{{engine="0",{model}}} 54.0\n'
        f'vllm:kv_cache_usage_perc{{engine="0",{model}}} 0.29991204925241866\n'
        f'vllm:prompt_tokens_by_source_total{{engine="0",{model},source="local_compute"}} 1.833485e+06\n'
        f'vllm:request_success_total{{engine="0",finished_reason="length",{model}}} 141.0\n'
        f'vllm:num_preemptions_created{{engine="0",{model}}} 1.79e9\n'  # not kept
        f'vllm:iteration_tokens_total_bucket{{engine="0",le="1.0",{model}}} 7\n'
    )
    assert kept_series(exposition, ENGINE_KEPT) == {
        'vllm:num_preemptions_total{engine="0"}': 54.0,
        'vllm:kv_cache_usage_perc{engine="0"}': 0.29991204925241866,
        'vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"}': 1833485.0,
        'vllm:request_success_total{engine="0",finished_reason="length"}': 141.0,
    }


def test_a_series_key_gives_its_metric_and_labels_back() -> None:
    labels = {"b": 'say "hi"', "a": "C:\\dir\\", "c": "two\nlines", "model_name": "/home/x"}
    key = series_key("m_total", labels)
    assert key == 'm_total{a="C:\\\\dir\\\\",b="say \\"hi\\"",c="two\\nlines"}'
    assert parse_key(key) == ("m_total", {k: v for k, v in labels.items() if k != "model_name"})
    assert series_key("m", {}) == "m" and parse_key("m") == ("m", {})


def test_the_raw_scrapes_keep_every_source_whole_by_the_hour(tmp_path: Path) -> None:
    raw = RawScrapes(tmp_path / "scrapes", start=1000.0)
    raw.write(
        1060.0, {"gateway": ("ok", "a_total 1\n"), "engine/phastform": ("HTTP 502: down", "")}
    )
    raw.write(4700.0, {"gateway": ("ok", "a_total 2")})
    assert sorted(p.name for p in (tmp_path / "scrapes").iterdir()) == ["000.gz", "001.gz"]
    assert list(read_raw(tmp_path / "scrapes")) == [
        (1060.0, "gateway", "ok", "a_total 1\n"),
        (1060.0, "engine/phastform", "HTTP", ""),
        (4700.0, "gateway", "ok", "a_total 2\n"),
    ]
    with gzip.open(tmp_path / "scrapes" / "000.gz", "rt") as text:
        assert text.readline() == "# soak-scrape 1060.000 gateway ok\n"


def test_the_managers_journal_lines_are_kept_and_the_targets_hidden() -> None:
    def entry(at: float, unit: str, message: str) -> str:
        return json.dumps(
            {"__REALTIME_TIMESTAMP": str(int(at * 1e6)), "USER_UNIT": unit, "MESSAGE": message}
        )

    text = "\n".join(
        [
            entry(
                10.5,
                "flatshare-gateway.service",
                "flatshare-gateway.service: Main process exited, code=killed",
            ),
            entry(
                11.0, "flatshare-tunnel@gray.service", "master connected to alice@gray.example.org"
            ),
            entry(12.0, "other.service", "Started other.service"),
            "not json",
        ]
    )
    lines = manager_lines(text, ["alice@gray.example.org", "gray.example.org"])
    assert lines == [
        {"at": 10.5, "unit": "flatshare-gateway.service",
         "message": "flatshare-gateway.service: Main process exited, code=killed"},
        {"at": 11.0, "unit": "flatshare-tunnel@gray.service", "message": "master connected to <hidden>"},
    ]  # fmt: skip
    own = own_lines('panicked at src/main.rs\n{"message":"request","status":200}\n\nexit 101\n')
    assert own == ["panicked at src/main.rs", "exit 101"]


# The summary of a synthetic record

START = 1_800_000_000.0
"""The synthetic soak's window start: two hours, a scrape a minute."""


def minute(n: int) -> float:
    return START + 60 * n


def gateway_at(n: int, reset: int | None = None) -> dict[str, float]:
    """The gateway's kept series at minute ``n``: each counter grows by its step a minute, from zero again at
    ``reset`` (its process restarted a minute before)."""
    k = n - reset + 1 if reset is not None and n >= reset else n
    offered = 'tenant="agents",route="/v1/chat/completions",class="background"'
    return {
        series_key(name, dict(parse_key("x{" + labels + "}")[1]) | extra): float(step * k)
        for name, labels, extra, step in (
            ("flatshare_gateway_requests_offered_total", offered, {}, 60),
            ("flatshare_gateway_requests_admitted_total", offered, {"admission": "none"}, 58),
            ("flatshare_gateway_requests_terminal_total", offered, {"outcome": "completed"}, 55),
            ("flatshare_gateway_requests_terminal_total", offered, {"outcome": "cancelled"}, 1),
            ("flatshare_gateway_requests_terminal_total", offered, {"outcome": "timeout_total"}, 2),
            (
                "flatshare_gateway_usage_tokens_total",
                'class="background",kind="prompt"',
                {},
                60_000,
            ),
            (
                "flatshare_gateway_usage_tokens_total",
                'class="background",kind="completion"',
                {},
                600,
            ),
            ("flatshare_gateway_sli_valid_total", 'class="realtime"', {}, 10),
            ("flatshare_gateway_sli_good_total", 'class="realtime"', {}, 9),
            ("flatshare_gateway_upstream_errors_total", 'kind="timeout"', {}, 2),
        )
    } | {
        'flatshare_gateway_engine_state{state="serving"}': 1.0,
        'flatshare_gateway_engine_state{state="warming"}': 0.0,
        'flatshare_gateway_in_flight{class="background"}': 10.0 if n % 2 else 20.0,
    }


def engine_at(n: int) -> dict[str, float]:
    return {
        'vllm:num_preemptions_total{engine="0"}': 2.0 * n,
        'vllm:kv_cache_usage_perc{engine="0"}': 0.7 if n % 2 else 0.5,
        'vllm:num_requests_running{engine="0"}': 6.0,
        'vllm:num_requests_waiting{engine="0"}': 1.0 if n % 2 else 3.0,
        'vllm:prompt_tokens_total{engine="0"}': 60_000.0 * n,
        'vllm:generation_tokens_total{engine="0"}': 600.0 * n,
        'vllm:prefix_cache_queries_total{engine="0"}': 60.0 * n,
        'vllm:prefix_cache_hits_total{engine="0"}': 30.0 * n,
    }


def turn(caller: int, number: int, at: float, v2v: float | None, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"run": 0, "caller": caller, "turn": number, "clip": f"turn-{number:02d}"}
    if v2v is not None:
        row["v2v_ms"] = v2v
    row |= {"barge_in": False, "end_by": "state", "reply_audio_ms": 2000.0, "reply_pauses_ms": []}
    return row | extra | {"at": at}


def synthetic_soak(directory: Path) -> Path:
    """A soak directory as soak run leaves it: two hours, a scrape a minute (the gateway restarting at minute 90, the
    engine's scrape failing at minute 100), its calls' attempts and turns, the user manager's lines of the gateway's
    restart, and a guard that found the GPU clear every minute."""
    directory.mkdir()
    record = {
        "kind": "soak",
        "status": "complete",
        "ended_by": "hours",
        "host": "cyan",
        "started": START - 120,
        "finished": START + 7300,
        "window": {"start": START, "end": START + 7200, "hours": 2.0},
        "options": {"slo_reference_ms": 1784.0, "turns": 3},
        "serving": {"nodes": ["phastform"], "with_nodes": False,
                    "serving_profile": {"profile": "rtx-5060-ti-serve"},
                    "worker": {"FLATSHARE_SERVING_PROFILE": "/home/alice/flatshare/src/deploy/phastform/serving.json"}},
        "calls_runs": [{"run": 0, "exit_status": 0}],
        "agents_runs": [{"run": 0, "exit_status": 0}],
        "pauses": [],
    }  # fmt: skip
    (directory / "soak.json").write_text(json.dumps(record))
    calls = directory / "calls" / "run-000"
    calls.mkdir(parents=True)
    attempts = [
        {"call": 0, "due": minute(1), "outcome": "accepted", "ended": "completed"},
        {"call": 1, "due": minute(5), "outcome": "refused", "reason": "capacity", "detail": "2 of 2 calls"},
        {"call": 2, "due": minute(9), "outcome": "accepted", "ended": "completed"},
        {"call": 3, "due": minute(20), "outcome": "failed", "reason": "connect",
         "detail": "ConnectError: /home/alice/flatshare/x"},
        {"call": 4, "due": minute(30), "outcome": "refused", "reason": "engine"},
        {"call": 5, "due": minute(40), "outcome": "skipped", "reason": "max_open"},
        {"call": 6, "due": minute(65), "outcome": "accepted", "ended": "dropped", "reason": "agent_left"},
        {"call": 7, "due": minute(70), "outcome": "refused", "reason": "capacity"},
        {"call": 8, "due": minute(119), "outcome": "accepted", "ended": "completed"},
    ]  # fmt: skip
    (calls / "attempts.jsonl").write_text(
        "".join(json.dumps({"run": 0} | a) + "\n" for a in attempts)
    )
    turns = [
        turn(0, 0, minute(2), 9000.0, warmup=True),
        turn(0, 1, minute(2) + 10, 1800.0),
        turn(0, 2, minute(2) + 20, 2100.0, barge_in=True, v2v_after_barge_in_ms=1900.0),
        turn(0, 3, minute(2) + 30, 2200.0),
        turn(2, 1, minute(10), 1700.0),
        turn(2, 2, minute(10) + 10, None, error="the agent did not start speaking"),
        turn(2, 3, minute(10) + 20, 2500.0, reply_pauses_ms=[320.0, 650.0],
             received={"samples": 48_000, "concealed": 960, "silent_concealed": 480}),
        turn(6, 1, minute(66), 2000.0),
        # The last call plays its turns after the window's end: they belong to its last hour.
        turn(8, 1, minute(120) + 20, 1600.0, reply_pauses_ms=[300.0, 400.0]),
    ]  # fmt: skip
    for row in turns:
        if "error" in row:
            for key in ("end_by", "reply_audio_ms", "reply_pauses_ms"):
                row.pop(key)
    (calls / "turns.jsonl").write_text("".join(json.dumps(row) + "\n" for row in turns))
    lines = []
    for n in range(121):
        line: dict[str, Any] = {"at": minute(n), "scrape": n, "gateway": gateway_at(n, reset=90)}
        line["engines"] = {"phastform": None if n == 100 else engine_at(n)}
        line["errors"] = {"engine/phastform": "HTTP 502: engine down"} if n == 100 else {}
        line["worker"] = {
            "active_calls": float(n % 3),
            "max_calls": 2.0,
            "admitting": 1.0,
            "refused": n // 30,
        }
        if n == 0:
            line["label"] = "window"
        lines.append(line)
    (directory / "series.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    events = [
        {"at": minute(90), "event": "counter_reset", "source": "gateway", "series": 10},
        {
            "at": minute(100),
            "event": "scrape_failed",
            "source": "engine/phastform",
            "error": "HTTP 502",
        },
        {"at": minute(101), "event": "scrape_answered", "source": "engine/phastform"},
    ]
    (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    unit = "flatshare-gateway.service"
    units = [
        {"at": minute(89) + 30, "unit": unit, "message": f"{unit}: Main process exited, code=killed, status=9/KILL",
         "own_lines": ["thread 'main' panicked at /home/alice/src/main.rs"]},
        {"at": minute(89) + 30.1, "unit": unit, "message": f"{unit}: Failed with result 'signal'."},
        {"at": minute(89) + 31, "unit": unit, "message": f"{unit}: Scheduled restart job, restart counter is at 1."},
        {"at": minute(89) + 32, "unit": unit, "message": "Started flatshare-gateway.service - Flatshare gateway."},
    ]  # fmt: skip
    (directory / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    checks = [{"node": "phastform", "verdict": "clear", "at": minute(n) + 1} for n in range(120)]
    (directory / "guard.jsonl").write_text("".join(json.dumps(c) + "\n" for c in checks))
    return directory


def test_the_summary_gives_each_hours_calls_replies_and_the_services_own_counts(
    tmp_path: Path,
) -> None:
    summary = summarize(load(synthetic_soak(tmp_path / "soak")), slo_reference_ms=1784.0)
    assert summary["slo"] == {
        "reference_p95_ms": 1784.0,
        "margin_ms": 250.0,
        "threshold_ms": 2034.0,
    }
    assert summary["window"]["hours"] == 2.0 and len(summary["hours"]) == 2
    first, second = summary["hours"]
    # Calls by when they were due: what the service was offered, and why it refused; a skipped call was never made.
    assert first["calls"] | {"ended": None} == {
        "attempted": 5, "accepted": 2, "refused": 2, "refused_by": {"capacity": 1, "engine": 1},
        "failed": 1, "failed_by": {"connect": 1}, "skipped": 1, "abandoned": 0, "ended": None,
        "refused_share": 0.4,
    }  # fmt: skip
    assert first["calls"]["ended"] == {"completed": 2, "dropped": 0, "cut": 0}
    assert (second["calls"]["attempted"], second["calls"]["ended"]["dropped"]) == (3, 1)
    # The replies: each turn's and the interruption's, the failed turn's censored at 30 s, warm-up turns left out.
    # Hour 0's seven: 1,700, 1,800, 1,900, 2,100, 2,200, 2,500 and 30,000 (censored).
    replies = first["turns"]["replies"]
    assert (first["turns"]["measured"], first["turns"]["failed"], first["turns"]["warmup"]) == (
        6,
        1,
        1,
    )
    assert replies == {
        "n": 7, "censored": 1, "p50": 2100.0, "p95": 21750.0, "p99": 28350.0, "over_slo": 4,
        "over_slo_share": round(4 / 7, 4),
    }  # fmt: skip
    heard = first["turns"]["heard"]
    assert (heard["n"], heard["with_a_pause_over_500_ms"], heard["pause_over_500_ms_share"]) == (
        5,
        1,
        0.2,
    )
    assert heard["concealed_share"] == round(480 / 48_000, 5)
    # The last hour has the calls that played on after the window's end.
    assert second["turns"]["replies"] == {"n": 2, "p50": 1800.0, "p95": 1980.0, "p99": 1996.0, "over_slo": 0,
                                          "over_slo_share": 0.0}  # fmt: skip
    for hour in (first, second):
        gateway = hour["gateway"]
        # A minute's counts sixty times: across the gateway's restart at minute 90, its counters count from zero
        # again, so the hour loses nothing.
        assert gateway["span_s"] == 3600.0
        assert gateway["realtime_sli"] == {"valid": 600, "good": 540, "good_share": 0.9}
        assert gateway["background"] == {
            "offered": 3600, "admitted": 3480, "ended": 3480, "completed": 3300, "cancelled": 60, "failed": 120,
            "outcomes": {"cancelled": 60, "completed": 3300, "timeout_total": 120},
            "prompt_tokens_per_s": 1000.0, "completion_tokens_per_s": 10.0, "tokens_per_s": 1010.0,
        }  # fmt: skip
        assert gateway["upstream_errors"] == {"timeout": 120}
        assert gateway["engine_state"] == {"phastform": {"serving": 60}}
        assert gateway["in_flight"]["background"] == {"n": 60, "mean": 15.0, "max": 20.0}
        engine = hour["engines"]["phastform"]
        assert (engine["preemptions"], engine["prefix_cache_hit_share"]) == (120, 0.5)
        assert (engine["prompt_tokens_per_s"], engine["generation_tokens_per_s"]) == (1000.0, 10.0)
    assert first["engines"]["phastform"]["kv_cache_usage"] == {"n": 60, "mean": 0.6, "max": 0.7}
    # The engine's scrape at minute 100 failed: its next one counts both minutes, and the hour has one gauge fewer.
    assert second["engines"]["phastform"]["scrapes"] == 59
    assert second["scrapes"] == {"n": 60, "failed": {"engine/phastform": 1}}
    restarts = second["restarts"]
    assert restarts["units"] == {
        "flatshare-gateway.service": {
            "exited": 1,
            "failed": 1,
            "restart_scheduled": 1,
            "started": 1,
        }
    }
    assert restarts["counter_resets"] == {"gateway": 1} and restarts["scrapes_failed"] == {
        "engine/phastform": 1
    }
    assert first["restarts"]["units"] == {} and first["guard"]["checks"] == {"clear": 60}
    assert first["worker"] == {"active_calls": {"n": 60, "mean": 1.0, "max": 2.0}, "refused": 2}
    overall = summary["overall"]
    assert overall["calls"]["attempted"] == 8 and overall["turns"]["replies"]["n"] == 9
    assert overall["gateway"]["realtime_sli"] == {"valid": 1200, "good": 1080, "good_share": 0.9}
    assert overall["gateway"]["background"]["tokens_per_s"] == 1010.0
    assert "replies p95 21,750 ms, 57.1% over the SLO" in render_hour(first)
    # Without a reference, the share over the SLO is not computed.
    bare = summarize(load(tmp_path / "soak"))
    assert bare["slo"] is None and "over_slo" not in bare["hours"][0]["turns"]["replies"]


def test_with_two_nodes_the_summary_gives_each_nodes_engine_health_and_placements(
    tmp_path: Path,
) -> None:
    # A gateway with [[nodes]] labels its engines' families by node and adds the nodes' own: each node's engine state
    # and health over the scrapes, its placements and retries, and the sessions no node took.
    def gateway(n: int) -> dict[str, float]:
        series = {}
        for node, healthy in (("phastform", 1.0), ("gray", 0.0 if 2 <= n <= 4 else 1.0)):
            state = "degraded" if node == "gray" and 2 <= n <= 4 else "serving"
            for name in ("serving", "degraded"):
                series[f'flatshare_gateway_engine_state{{node="{node}",state="{name}"}}'] = float(
                    name == state
                )
            series[f'flatshare_gateway_node_healthy{{node="{node}"}}'] = healthy
            series[f'flatshare_gateway_node_covering{{node="{node}"}}'] = float(
                node == "phastform" and 3 <= n <= 4
            )
            series[
                f'flatshare_gateway_placements_total{{class="realtime",node="{node}",reason="new"}}'
            ] = float(n)
        series[
            'flatshare_gateway_placements_total{class="realtime",node="phastform",reason="moved"}'
        ] = float(min(n, 3) - 1 if n >= 3 else 0)
        series['flatshare_gateway_node_retries_total{node="phastform"}'] = 1.0 if n >= 3 else 0.0
        series[
            'flatshare_gateway_placements_refused_total{class="realtime",reason="no_call_slot"}'
        ] = float(n // 3)
        return series

    soak = tmp_path / "soak"
    soak.mkdir()
    record = {"status": "complete", "ended_by": "hours", "window": {"start": START, "end": START + 360, "hours": 0.1},
              "serving": {"nodes": ["phastform", "gray"], "with_nodes": True}, "pauses": []}  # fmt: skip
    (soak / "soak.json").write_text(json.dumps(record))
    lines = [
        {"at": minute(n), "gateway": gateway(n), "errors": {}, "worker": None,
         "engines": {node: engine_at(n) for node in ("phastform", "gray")}}
        for n in range(7)
    ]  # fmt: skip
    (soak / "series.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    overall = summarize(load(soak))["overall"]
    gateway_summary = overall["gateway"]
    assert gateway_summary["engine_state"] == {
        "phastform": {"serving": 6},
        "gray": {"degraded": 3, "serving": 3},
    }
    assert gateway_summary["nodes"] == {
        "phastform": {
            "healthy_scrapes": 6,
            "covering_scrapes": 2,
            "placements": {"moved": 2, "new": 6},
            "retries": 1,
        },
        "gray": {
            "healthy_scrapes": 3,
            "covering_scrapes": 0,
            "placements": {"new": 6},
            "retries": 0,
        },
    }
    assert gateway_summary["placements_refused"] == {"no_call_slot": 2}
    assert (
        set(overall["engines"]) == {"phastform", "gray"}
        and overall["engines"]["gray"]["preemptions"] == 12
    )


def test_with_two_nodes_the_scraper_asks_each_engine_by_its_name(tmp_path: Path) -> None:
    gateway = FakeGateway()
    gateway.nodes = True
    try:
        scraper = Scraper(
            tmp_path,
            gateway.url,
            ["phastform", "gray"],
            True,
            lambda: None,
            FakeSystemd({}),
            START,
            (),
        )
        line, events = scraper.scrape(START + 1.0, "window")
    finally:
        stop(gateway)
    assert sorted(gateway.queries) == [
        "/metrics/engine?node=gray",
        "/metrics/engine?node=phastform",
    ]
    assert set(line["engines"]) == {"phastform", "gray"} and line["nodes"] == {
        "gray": {"verdict": "unhealthy", "available": False, "covering": False, "sessions": {}, "in_flight": {}},
        "phastform": {"verdict": "healthy", "available": True, "covering": True, "sessions": {"realtime": 2},
                      "in_flight": {"realtime": 1}},
    }  # fmt: skip
    # The fake's engine answers its n-th /metrics/engine with n preemptions: gray's is the second asked.
    assert line["engines"]["gray"] == {'vllm:num_preemptions_total{engine="0"}': 2.0,
                                       'vllm:kv_cache_usage_perc{engine="0"}': 0.5}  # fmt: skip
    assert events == [] and line["label"] == "window"
    assert [source for _, source, _, _ in read_raw(tmp_path / "scrapes")] == [
        "gateway", "engine/phastform", "engine/gray", "nodes", "worker",
    ]  # fmt: skip


def test_a_scrape_that_fails_and_a_counter_that_goes_down_are_events(tmp_path: Path) -> None:
    gateway = FakeGateway()
    try:
        scraper = Scraper(
            tmp_path, gateway.url, ["phastform"], False, lambda: None, FakeSystemd({}), START, ()
        )
        scraper.scrape(START)
        scraper.scrape(START + 60)
        gateway.down.add("/metrics/engine")
        first, failed = scraper.scrape(START + 120)
        gateway.down.clear()
        gateway.hits.clear()  # the gateway "restarted": its counters start again
        _, back = scraper.scrape(START + 180)
    finally:
        stop(gateway)
    assert first["engines"] == {"phastform": None}
    assert first["errors"] == {"engine/phastform": "HTTP 503: engine warming"}
    assert failed == [
        {
            "event": "scrape_failed",
            "source": "engine/phastform",
            "error": "HTTP 503: engine warming",
        }
    ]
    kinds = {(e["event"], e["source"]) for e in back}
    assert kinds == {("scrape_answered", "engine/phastform"), ("counter_reset", "gateway"),
                     ("counter_reset", "engine/phastform")}  # fmt: skip


def test_a_compact_record_gives_the_same_summary_and_names_no_home(tmp_path: Path) -> None:
    soak = synthetic_soak(tmp_path / "soak")
    record = export(soak, tmp_path / "record")
    assert sorted(p.name for p in record.iterdir()) == sorted(
        ["soak.json", "summary.json", "summary.md", "attempts.jsonl", "turns.csv", "series.csv",
         "events.jsonl", "units.jsonl", "guard.csv"]
    )  # fmt: skip
    raw, compact = load(soak), load(record)
    assert compact.turns == raw.turns  # turns.csv gives each line back
    assert compact.scrapes == [
        {
            k: v
            for k, v in s.items()
            if k in ("at", "scrape", "label", "gateway", "engines", "worker", "errors")
        }
        for s in raw.scrapes
    ]
    # The soak's own reference, from its options, in both.
    assert summarize(compact, 1784.0) == summarize(raw, 1784.0)
    assert json.loads((record / "summary.json").read_text()) == summarize(raw, 1784.0)
    for path in record.iterdir():
        text = path.read_text()
        assert "/home/" not in text and "alice" not in text, path.name
    assert "~/flatshare/src/deploy/phastform/serving.json" in (record / "soak.json").read_text()
    table = (record / "summary.md").read_text()
    assert table.startswith("# Soak summary") and "| all | 8 | 4 | 3 | 1 | 8 | 1 |" in table


def test_the_compact_tables_give_their_rows_back() -> None:
    turns = [
        {"run": 1, "caller": 7, "turn": 0, "at": 1.25, "warmup": True, "v2v_ms": 9000.0},
        {"run": 1, "caller": 7, "turn": 1, "at": 2.5, "v2v_ms": 1712.3, "barge_in": True,
         "interrupted_speech": False, "v2v_after_barge_in_ms": 1500.0, "reply_pauses_ms": [],
         "interrupted_reply_pauses_ms": [412.5], "received": {"samples": 9, "concealed": 1, "silent_concealed": 0},
         "end_by": "state", "clip": "turn-03", "barge_in_clip": "barge_in-41", "new_key": {"x": [1, 2]}},
        {"run": 1, "caller": 7, "turn": 2, "at": 3.0, "error": 'the agent did not "start", speaking',
         "end_by": None},
    ]  # fmt: skip
    assert read_turns_csv(write_turns_csv(turns)) == turns
    scrapes = [
        {"at": 10.0, "scrape": 0, "label": "window", "gateway": {"a_total": 1.0, 'b{x="1,2"}': 0.25},
         "engines": {"phastform": {"vllm:x_total": 3.0}}, "worker": {"active_calls": 1.0}, "errors": {}},
        {"at": 70.5, "scrape": 1, "gateway": None, "engines": {"phastform": {"vllm:x_total": 4e12 + 0.5}},
         "worker": None, "errors": {"gateway": "ConnectError: refused"}},
    ]  # fmt: skip
    assert read_series_csv(write_series_csv(scrapes)) == scrapes


def test_counters_count_from_zero_after_a_restart_and_steps_go_to_the_period_they_end_in() -> None:
    assert _increase({"a_total": 10.0, "g": 5.0}, {"a_total": 12.0, "g": 1.0, "b_total": 3.0}) == {
        "a_total": 2.0,
        "b_total": 3.0,  # new to the later scrape: from zero
    }
    assert _increase({"a_total": 10.0}, {"a_total": 4.0}) == {
        "a_total": 4.0
    }  # restarted: from zero
    scrapes = [
        {"at": t, "gateway": {"a_total": v}, "engines": {}, "worker": None, "errors": {}}
        for t, v in ((0.0, 0.0), (60.0, 5.0), (120.0, 9.0), (150.0, 2.0))
    ]
    counts = _Counts(scrapes)
    assert counts.increase("gateway", 0.0, 60.0) == ({"a_total": 5.0}, 60.0)
    assert counts.increase("gateway", 60.0, math.inf) == ({"a_total": 6.0}, 90.0)
    assert (
        _paused([{"start": 50.0, "end": 130.0}, {"start": 200.0, "end": None}], 0.0, 100.0, 1000.0)
        == 50.0
    )
    assert _paused([{"start": 200.0, "end": None}], 0.0, math.inf, 260.0) == 60.0


# The guard


def completed(
    status: int, stdout: str = "", stderr: str = ""
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, status, stdout, stderr)

    return run


def report(status: int, listed: str = "", said: str = "", journal: list[Any] | None = None) -> str:
    """deploy/bin/node-report's output: gpu-guard --list's status, its lines and errors, and the journal asked."""
    found: dict[str, Any] = {"gpu_guard": {"status": status, "stdout": listed, "stderr": said}}
    return json.dumps(found | ({"journal": journal} if journal is not None else {}))


def test_a_check_reads_the_nodes_report_and_tells_whose_process_holds_the_gpu() -> None:
    mine = "ours\t2211\talice\t13990\tpython3\n"
    theirs = "other\t9001\tbob\t5120\ttrain.py\nother\t9002\talice\t[N/A]\tnotebook\n"
    assert parse_list(mine + theirs + "garbage\n", "alice") == (
        {"whose": "ours", "pid": 2211, "owner": "self", "memory_mib": 13990, "name": "python3"},
        {"whose": "other", "pid": 9001, "owner": "another", "memory_mib": 5120, "name": "train.py"},
        {"whose": "other", "pid": 9002, "owner": "self", "memory_mib": "[N/A]", "name": "notebook"},
    )
    target = "alice@gray.example.org"
    hidden = (target, "gray.example.org")

    def guard(run: Callable[..., subprocess.CompletedProcess[str]]) -> NodeGuard:
        return NodeGuard("gray", "ssh", ["ssh", target], hidden, run, "alice")

    clear = guard(completed(0, report(0, mine))).check(1.0)
    assert clear is not None and (clear.verdict, len(clear.processes), clear.journal) == (
        "clear",
        1,
        (),
    )
    taken = guard(completed(1, report(1, mine + theirs))).check(2.0)
    assert taken is not None and taken.verdict == "taken" and len(taken.others()) == 2
    assert (
        taken.describe()
        == "pid 9001 (another's train.py, 5120 MiB), pid 9002 (self's notebook, [N/A] MiB)"
    )
    # No owner's name and no target in what a record keeps.
    assert "bob" not in json.dumps(taken.record()) and target not in json.dumps(taken.record())
    # No report: SSH failed, its error naming the node, not its target.
    refused = "ssh: connect to host gray.example.org port 22: Connection timed out"
    blind = guard(completed(255, "", refused)).check(3.0)
    assert blind is not None and blind.verdict == "blind"
    assert (
        blind.error
        == "exit status 255: ssh: connect to host gray's target port 22: Connection timed out"
    )
    said = "gpu-guard: nvidia-smi could not list the GPU's processes"
    smi = guard(completed(2, report(2, "", said))).check(4.0)
    assert smi is not None and smi.verdict == "blind" and smi.error == f"exit status 2: {said}"

    def timeout(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 30)

    hung = guard(timeout).check(5.0)
    assert hung is not None and hung.verdict == "blind" and target not in str(hung.error)
    assert NodeGuard("gray", "off", None).check() is None


def test_a_check_reads_the_nodes_journal_from_where_the_last_one_read_it() -> None:
    # Each check asks the node's units' journal from a little before the last check (the hosts' clocks may differ),
    # as since=EPOCH, which the key's forced command reads from the session's command.
    asked: list[list[str]] = []
    engine = "flatshare-engine.service"
    lines = [
        {"at": 1_800_000_010.5, "unit": engine, "message": f"{engine}: Main process exited, code=killed",
         "own_lines": ["connection to alice@gray.example.org lost"]},
        {"at": 1_800_000_011.0, "unit": engine, "message": f"Started {engine}"},
        "not a line",
    ]  # fmt: skip

    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        asked.append(command)
        return subprocess.CompletedProcess(command, 0, report(0, "", "", lines), "")

    target = "alice@gray.example.org"
    guard = NodeGuard("gray", "ssh", ["ssh", target], (target,), run, "alice")
    assert guard.check(1_800_000_000.0) is not None and asked[-1][-1] == target  # none asked
    guard.since = 1_800_000_000.0
    check = guard.check(1_800_000_060.0)
    assert check is not None and asked[-1][-1] == "since=1800000000.000"
    assert check.journal == (
        {"at": 1_800_000_010.5, "unit": engine, "message": f"{engine}: Main process exited, code=killed",
         "own_lines": ["connection to gray's target lost"]},
        {"at": 1_800_000_011.0, "unit": engine, "message": f"Started {engine}"},
    )  # fmt: skip
    assert guard.since == 1_800_000_060.0 - guard_module.OVERLAP_S
    guard.check(1_800_000_120.0)
    assert asked[-1][-1] == f"since={1_800_000_060.0 - guard_module.OVERLAP_S:.3f}"


def test_the_guard_pauses_while_a_gpu_is_taken_and_resumes_once_clear_for_its_hold_off() -> None:
    nodes = {name: NodeGuard(name, "ssh", ["ssh"]) for name in ("phastform", "gray")}
    guard = Guard(nodes, clear_s=120.0, blind_after=3)

    def check(node: str, verdict: str, at: float) -> Check:
        processes = (
            {"whose": "other", "pid": 7, "owner": "another", "memory_mib": 900, "name": "train.py"},
        )
        return Check(node, verdict, at, processes if verdict == "taken" else ())

    assert guard.observe([check("phastform", "clear", 0), check("gray", "clear", 0)]) == []
    events = guard.observe([check("phastform", "clear", 60), check("gray", "taken", 60)])
    assert [e["event"] for e in events] == ["guard_taken"] and guard.paused
    assert guard.why() == "gray: its GPU is in use by pid 7 (another's train.py, 900 MiB)"
    assert (
        guard.observe([check("gray", "taken", 120)]) == []
    )  # still taken: one event for the episode
    assert (
        guard.observe([check("gray", "clear", 180)]) == [] and guard.paused
    )  # clear, not yet for 120 s
    assert (
        guard.observe([check("gray", "blind", 240)])[0]["event"] == "guard_failed"
    )  # the hold-off starts over
    assert guard.observe([check("gray", "clear", 300)]) == [] and guard.paused
    assert guard.observe([check("gray", "clear", 360)]) == [] and guard.paused
    events = guard.observe([check("gray", "clear", 420)])
    assert [e["event"] for e in events] == ["guard_clear"] and not guard.paused
    assert events[0]["clear_s"] == 120.0
    # A GPU it cannot read three times in a row holds the load back too, as gpu-guard refuses when it cannot tell.
    assert [e["event"] for e in guard.observe([check("phastform", "blind", 480)])] == [
        "guard_failed"
    ]
    assert guard.observe([check("phastform", "blind", 540)]) == [] and not guard.paused
    assert [e["event"] for e in guard.observe([check("phastform", "blind", 600)])] == [
        "guard_blind"
    ]
    assert guard.paused and "could not be read 3 times in a row" in guard.why()


def test_a_nodes_guard_goes_to_its_tunnels_target_with_the_guard_key(tmp_path: Path) -> None:
    host = Host(home=tmp_path)
    with pytest.raises(E2Error, match="names none"):
        node_guard("gray", "ssh", host)
    host.config.mkdir(parents=True)
    (host.config / "tunnel-gray.env").write_text("FLATSHARE_TUNNEL_TARGET=alice@gray.example.org\n")
    with pytest.raises(E2Error, match=re.escape("deploy/authorize-guard.sh cyan gray")):
        node_guard("gray", "ssh", host)
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "flatshare_guard_ed25519").write_text("key\n")
    guard = node_guard("gray", "ssh", host)
    assert guard.command is not None and guard.command[:3] == [
        "ssh",
        "-i",
        str(tmp_path / guard_module.GUARD_KEY),
    ]
    assert guard.command[-1] == "alice@gray.example.org" and "BatchMode=yes" in guard.command
    assert guard.hidden == ("alice@gray.example.org", "gray.example.org")
    assert guard.record() == {"node": "gray", "method": "ssh"}  # never its target
    assert node_guard("gray", "ssh:bob@other", host).command[-1] == "bob@other"  # type: ignore[index]
    local = node_guard("localhost", "local", host).command
    assert local == ["python3", str(host.root / "src" / "deploy" / "bin" / "node-report")]
    assert node_guard("gray", "off", host).command is None
    assert parse_guards(["gray=off", "phastform=ssh:a@b"]) == {
        "gray": "off",
        "phastform": "ssh:a@b",
    }
    for wrong in (["gray"], ["gray=maybe"], ["gray=off", "gray=ssh"]):
        with pytest.raises(E2Error):
            parse_guards(wrong)


# The soak's settings


def test_the_soaks_agents_run_as_e2s_do(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The same variables from the same files as an E2 point's replay (e2 run's _aiperf_env), but for how long it runs
    # and where it writes.
    from flatshare_bench.e2.run import _aiperf_env

    host = Host(home=tmp_path, gateway_url="http://127.0.0.1:18080")
    host.config.mkdir(parents=True)
    (host.config / "gateway-keys.env").write_text("FLATSHARE_GATEWAY_AGENTS_KEY=agents-key\n")
    (host.root / "data" / "weka-x32").mkdir(parents=True)
    options = SoakOptions(hours=1, agents_sessions=24, agents_seed=13, think_cap_s=90.0)
    arm = Arm("a", "9", "x", agents=Agents(sessions=24, seed=13, think_cap=90.0, data="weka-x32"))
    soak = aiperf_env(host, tmp_path / "out", options, 13, MODEL, 3900)
    e2 = _aiperf_env(arm, host, tmp_path / "out", 13, MODEL)
    differ = {k for k in soak.keys() | e2.keys() if soak.get(k) != e2.get(k)}
    assert differ == {"FLATSHARE_BENCH_DURATION"}
    assert (
        soak["FLATSHARE_BENCH_DURATION"] == "3900" and soak["FLATSHARE_BENCH_CLASS"] == "background"
    )


def test_the_default_rate_finds_the_worker_full_now_and_then() -> None:
    # docs/EXPERIMENTS.md: calls of a warm-up and 10 turns at E2's pace (calls of 26 turns in about 280 to 430 s) last
    # 118 to 182 s, so 20 calls an hour offer the worker's two slots 0.66 to 1.01 Erlang, and Erlang B gives the
    # share of calls that find both taken: about one in nine to one in five.
    def erlang_b(servers: int, offered: float) -> float:
        blocked = 1.0
        for n in range(1, servers + 1):
            blocked = offered * blocked / (n + offered * blocked)
        return blocked

    options = SoakOptions(hours=24)
    turns = options.warmup_turns + options.turns
    loads = [options.calls_per_hour * turns * (seconds / 26) / 3600 for seconds in (280, 430)]
    assert [round(load, 2) for load in loads] == [0.66, 1.01]
    assert [round(1 / erlang_b(2, load), 1) for load in loads] == [8.7, 4.9]


def test_the_nodes_are_the_gateways() -> None:
    assert gateway_nodes({"upstream": "unix:/run/user/1/flatshare/phastform-engine.sock"}) == (
        ["phastform"],
        False,
    )
    assert gateway_nodes({"upstream": "unix:/run/user/1/flatshare/engine.sock"}) == (
        ["localhost"],
        False,
    )
    assert gateway_nodes({"nodes": [{"name": "phastform"}, {"name": "gray"}]}) == (
        ["phastform", "gray"],
        True,
    )


def test_a_soak_refuses_settings_it_cannot_run(capsys: pytest.CaptureFixture[str]) -> None:
    for wrong, says in [
        ({"hours": 0.0}, "hours must be above 0"),
        ({"hours": 1000.0}, "at most two weeks"),
        ({"turns": 0}, "a call needs a measured turn"),
        ({"barge_in": 1.5}, "barge_in is a share"),
        ({"scrape_s": 0.5}, "scrape_s and guard_s are 1 s or more"),
    ]:
        with pytest.raises(E2Error, match=says):
            SoakOptions(**({"hours": 1.0} | wrong))
    with pytest.raises(SystemExit):
        main(["soak", "run", "--hours", "0", "--out", "x"])
    assert "must be above 0" in capsys.readouterr().err


# Whole soaks on a fake cyan


class _Handler(BaseHTTPRequestHandler):
    server: FakeGateway

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        self.server.hits[path] += 1
        status, body = self.server.answer(path, query)
        data = (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


class FakeGateway(UnixServer):
    """The gateway's socket: /metrics with its counters grown at every scrape, an engine's through /metrics/engine,
    /ready and /v1/models; ``down`` answers its paths 503."""

    def __init__(self) -> None:
        self.hits: Counter[str] = Counter()
        self.queries: list[str] = []
        self.down: set[str] = set()
        self.nodes = False
        self.path = socket_file("flatshare-soak-")
        super().__init__(str(self.path), _Handler)
        serve(self)

    @property
    def url(self) -> str:
        return f"unix:{self.path}"

    def answer(self, path: str, query: str) -> tuple[int, Any]:
        if query:
            self.queries.append(f"{path}?{query}")
        if path in self.down:
            return 503, "engine warming"
        n = self.hits[path]
        if path == "/nodes" and self.nodes:
            healthy = {"verdict": "healthy", "available": True, "covering": True}
            down = {"verdict": "unhealthy", "available": False, "covering": False}
            return 200, {
                "nodes": [
                    {
                        "name": "phastform",
                        **healthy,
                        "sessions": {"realtime": 2},
                        "in_flight": {"realtime": 1},
                    },
                    {"name": "gray", **down, "sessions": {}, "in_flight": {}, "max_calls": 2},
                ],
                "sessions": 2,
            }
        if path == "/metrics":
            return 200, (
                f'flatshare_gateway_sli_valid_total{{class="realtime"}} {10 * n}\n'
                f'flatshare_gateway_sli_good_total{{class="realtime"}} {9 * n}\n'
                f'flatshare_gateway_requests_offered_total{{tenant="agents",class="background"}} {5 * n}\n'
                f'flatshare_gateway_requests_terminal_total{{class="background",outcome="completed"}} {4 * n}\n'
                f'flatshare_gateway_usage_tokens_total{{class="background",kind="prompt"}} {1000 * n}\n'
                'flatshare_gateway_engine_state{state="serving"} 1\n'
                f'flatshare_gateway_first_byte_seconds_bucket{{class="realtime",le="+Inf"}} {n}\n'
            )
        if path == "/metrics/engine":
            return 200, (
                f'vllm:num_preemptions_total{{engine="0",model_name="/home/alice/m"}} {n}\n'
                'vllm:kv_cache_usage_perc{engine="0",model_name="/home/alice/m"} 0.5\n'
            )
        if path == "/ready":
            return 200, "ready"
        if path == "/v1/models":
            return 200, {"data": [{"id": MODEL}]}
        return 404, "not found"


class FakeTools(Tools):
    """The soak's tools as stand-ins that run as real processes: AIPerf's and the callers'."""

    def aiperf(self) -> list[str]:
        return [sys.executable, str(FAKES / "fake_aiperf.py")]

    def calls(self) -> list[str]:
        return [sys.executable, str(FAKES / "fake_arrivals.py")]


class FakeSystemd(SoakSystemd):
    """cyan's user manager: the worker's environment, the units' states (``mains`` their MainPIDs) and the manager's
    journal lines (``lines``: when, unit, message)."""

    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.mains = {"flatshare-gateway.service": 100, "flatshare-voice-worker.service": 200}
        self.lines: list[tuple[float, str, str]] = []
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> str:
        self.calls.append(args)
        return ""

    def environment(self, unit: str) -> dict[str, str]:
        return dict(self.env)

    def units(self) -> dict[str, dict[str, str]]:
        return {
            unit: {
                "ActiveState": "active",
                "SubState": "running",
                "NRestarts": "0",
                "MainPID": str(pid),
            }
            for unit, pid in self.mains.items()
        }

    def manager_journal(self, units: Any, since: float, until: float) -> str:
        return "\n".join(
            json.dumps(
                {"__REALTIME_TIMESTAMP": str(int(at * 1e6)), "USER_UNIT": unit, "MESSAGE": message}
            )
            for at, unit, message in self.lines
            if since <= at < until
        )

    def unit_tail(self, unit: str, since: float, until: float, lines: int = 20) -> str:
        return 'it broke\n{"message":"request","status":200}\n'


@dataclass
class Cyan:
    host: Host
    systemd: FakeSystemd
    gateway: FakeGateway
    tools: FakeTools
    status: Path
    checks: list[str]
    """The guard's answers, one a check (clear, taken, blind), the last one again once they run out."""
    node_lines: list[dict[str, Any]] = field(default_factory=list)
    """The GPU node's journal lines about its units, which its report gives from the time a check asks."""

    def guard(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        """node-report on the node, over SSH: each check's verdict in turn, and the node's journal from the time the
        command asks (``node_lines``)."""
        verdict = self.checks.pop(0) if len(self.checks) > 1 else self.checks[0]
        ours = "ours\t2211\talice\t13990\tpython3\n"
        asked = command[-1].removeprefix("since=") if command[-1].startswith("since=") else None
        journal = None if asked is None else [x for x in self.node_lines if x["at"] >= float(asked)]
        if verdict == "taken":
            listed = ours + "other\t9001\tbob\t5120\ttrain.py\n"
            return subprocess.CompletedProcess(command, 1, report(1, listed, "", journal), "")
        if verdict == "blind":
            refused = "ssh: connect to host gpu.example.org: refused"
            return subprocess.CompletedProcess(command, 255, "", refused)
        return subprocess.CompletedProcess(command, 0, report(0, ours, "", journal), "")


@pytest.fixture
def cyan(tmp_path: Path) -> Iterator[Cyan]:
    if not hasattr(os, "waitid"):
        pytest.skip(tools.NEEDS_WAITID)  # every soak runs its tools
    home = tmp_path / "home"
    config = home / ".config" / "flatshare"
    config.mkdir(parents=True)
    (config / "gateway.toml").write_text(EXAMPLE.read_text())
    (config / "gateway-keys.env").write_text("FLATSHARE_GATEWAY_AGENTS_KEY=agents-key\n")
    (config / "livekit.env").write_text("LIVEKIT_API_KEY=k\nLIVEKIT_API_SECRET=s\n")
    (config / "tunnel-phastform.env").write_text("FLATSHARE_TUNNEL_TARGET=alice@gpu.example.org\n")
    (home / ".ssh").mkdir()
    (home / ".ssh" / "flatshare_guard_ed25519").write_text("key\n")
    (home / "flatshare" / "data" / "callers").mkdir(parents=True)
    (home / "flatshare" / "data" / "callers" / "manifest.json").write_text("{}")
    (home / "flatshare" / "data" / "weka-x32").mkdir()
    status = tmp_path / "voice-worker.json"
    status.write_text(json.dumps({"schema": 1, "updated": time.time() + 86400, "mode": "serve", "max_calls": 2,
                                  "active_calls": 1, "admitting": True, "reason": "1 of 2 calls", "refused": 0,
                                  "audio": "synthetic", "profile": "rtx-5060-ti-serve"}))  # fmt: skip
    profile = tmp_path / "serving.json"
    profile.write_text(json.dumps({"profile": "rtx-5060-ti-serve", "variant": "make-room-serve",
                                   "engine": {"model": {"model": MODEL}}, "voice": {"max_calls": 2}}))  # fmt: skip
    env = {
        "FLATSHARE_VOICE_MODE": "serve",
        "FLATSHARE_SERVING_PROFILE": str(profile),
        "FLATSHARE_VOICE_STATUS": str(status),
        "FLATSHARE_AUDIO": "synthetic",
        "FLATSHARE_LLM_UDS": "/run/user/1/flatshare/gateway.sock",
    }
    gateway = FakeGateway()
    host = Host(home=home, gateway_url="http://127.0.0.1:9", gateway_socket=gateway.url)
    yield Cyan(host, FakeSystemd(env), gateway, FakeTools(host), status, ["clear"])
    stop(gateway)


def soak(cyan: Cyan, name: str = "s", **options: Any) -> dict[str, Any]:
    settings = {
        "hours": 3 / 3600,
        "scrape_s": 1.0,
        "guard_s": 1.0,
        "agents_sessions": 4,
        "agents_warmup_s": 0.1,
        "drain_s": 5.0,
        "slo_reference_ms": 1784.0,
        "guards": {"phastform": "ssh"},
    } | options
    return run_soak(
        SoakOptions(**settings),
        cyan.host.root / "runs" / name,
        cyan.host,
        cyan.systemd,
        cyan.tools,
        cyan.guard,
    )


def record_of(cyan: Cyan, name: str = "s") -> dict[str, Any]:
    return json.loads((cyan.host.root / "runs" / name / "soak.json").read_text())


def events_of(cyan: Cyan, name: str = "s") -> list[dict[str, Any]]:
    path = cyan.host.root / "runs" / name / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@needs_waitid
def test_a_soak_runs_its_calls_and_agents_scrapes_the_service_and_writes_its_summary(
    cyan: Cyan,
) -> None:
    cyan.systemd.lines.append(
        (time.time() + 1.0, "flatshare-gateway.service", "Started flatshare-gateway.service")
    )
    engine = "flatshare-engine.service"
    failed = {
        "at": time.time() + 1.5,
        "unit": engine,
        "message": f"{engine}: Failed with result 'oom-kill'.",
    }
    cyan.node_lines.append(failed | {"own_lines": ["torch.OutOfMemoryError: CUDA out of memory"]})
    summary = soak(cyan)
    out = cyan.host.root / "runs" / "s"
    record = record_of(cyan)
    assert (record["status"], record["ended_by"]) == ("complete", "hours")
    assert record["serving"]["nodes"] == ["phastform"] and record["serving"]["model"] == MODEL
    assert record["serving"]["guards"] == [{"node": "phastform", "method": "ssh"}]
    assert record["serving"]["worker"]["FLATSHARE_VOICE_MODE"] == "serve"
    assert "alice@gpu.example.org" not in (out / "soak.json").read_text()
    # One run of each tool for the window: the callers' ended by themselves once the window had, AIPerf was stopped.
    [calls] = record["calls_runs"]
    [agents] = record["agents_runs"]
    assert (calls["exit_status"], calls["how"], calls["first_call"]) == (0, "planned", 0)
    assert (agents["exit_status"], agents["how"], agents["seed"], agents["sessions"]) == (
        0,
        "stopped",
        11,
        4,
    )
    assert agents["profiling"] <= record["window"]["start"]
    arguments = json.loads((out / "calls" / "run-000" / "args.json").read_text())
    options = dict(zip(arguments[::2], arguments[1::2], strict=True))
    assert (
        options["--calls-per-hour"] == "20"
        and options["--turns"] == "10"
        and options["--warmup-turns"] == "1"
    )
    assert options["--worker-status"] == str(cyan.status) and options["--run"] == "0"
    env = json.loads((out / "agents" / "run-000" / "aiperf" / "env.json").read_text())
    assert env["FLATSHARE_BENCH_SESSIONS"] == "4" and env["FLATSHARE_BENCH_MODEL"] == MODEL
    assert not Path(env["TMPDIR"]).exists()  # the run's temp directory went with it
    # Every scrape: before the load, at the window's start, a minute's... and after; whole in scrapes/, kept in
    # series.jsonl, never anything but /metrics, /metrics/engine, /ready and /v1/models.
    series = [json.loads(line) for line in (out / "series.jsonl").read_text().splitlines()]
    labels = [s.get("label") for s in series]
    assert labels[:2] == ["before", "window"] and labels[-1] == "after" and len(series) >= 5
    assert all(s["gateway"] is not None and s["engines"]["phastform"] is not None for s in series)
    assert set(cyan.gateway.hits) == {"/metrics", "/metrics/engine", "/ready", "/v1/models"}
    assert 'vllm:num_preemptions_total{engine="0"}' in series[-1]["engines"]["phastform"]
    assert len({at for at, _, _, _ in read_raw(out / "scrapes")}) == len(series)
    units = [json.loads(line) for line in (out / "units.jsonl").read_text().splitlines()]
    # cyan's units, and the GPU node's through its report, each line once although the checks overlap.
    assert sorted((u["unit"], u["message"]) for u in units) == [
        ("flatshare-gateway.service", "Started flatshare-gateway.service"),
        (f"phastform:{engine}", f"{engine}: Failed with result 'oom-kill'."),
    ]
    assert summary["overall"]["restarts"]["units"][f"phastform:{engine}"] == {"failed": 1}
    checks = [json.loads(line) for line in (out / "guard.jsonl").read_text().splitlines()]
    assert {c["verdict"] for c in checks} == {"clear"} and len(checks) >= 3
    kinds = [e["event"] for e in events_of(cyan)]
    assert (
        kinds[:3] == ["agents_started", "window_started", "calls_started"]
        and "window_ended" in kinds
    )
    # The summary: the fake callers' attempts (accepted, refused, accepted, failed, ...) and turns, and the
    # gateway's counts (a scrape's worth of each counter a second).
    attempts = [
        json.loads(line)
        for line in (out / "calls" / "run-000" / "attempts.jsonl").read_text().splitlines()
    ]
    overall = summary["overall"]
    assert overall["calls"]["attempted"] == len(attempts) >= 4
    assert overall["calls"]["refused_by"] == {
        "capacity": sum(1 for a in attempts if a["outcome"] == "refused")
    }
    assert overall["turns"]["measured"] == 10 * overall["calls"]["ended"]["completed"]
    assert overall["gateway"]["realtime_sli"]["good_share"] == 0.9
    assert json.loads((out / "summary.json").read_text()) == summary
    assert (out / "summary.md").read_text().startswith("# Soak summary")
    assert dead(
        int((out / "calls" / "run-000" / "arrivals.pid").read_text())
    )  # stopped, and reaped


@needs_waitid
def test_each_hours_summary_is_written_as_the_hour_ends(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Hours of a second: the soak logs each one as it ends and writes the summary then, so that a soak that dies
    # keeps the hours before; the summary's periods are those hours.
    from flatshare_bench.soak import summary as summaries

    monkeypatch.setattr(summaries, "HOUR_S", 1.0)
    summary = soak(cyan, hours=3.5 / 3600)
    log = (cyan.host.root / "runs" / "s" / "soak.log").read_text()
    assert "hour 0: calls " in log and "hour 1: calls " in log
    assert [h["hour"] for h in summary["hours"]] == [0, 1, 2, 3]
    assert (
        sum(h["calls"]["attempted"] for h in summary["hours"])
        == summary["overall"]["calls"]["attempted"]
    )
    assert (
        sum(h["turns"]["measured"] for h in summary["hours"])
        == summary["overall"]["turns"]["measured"]
    )


@needs_waitid
def test_sigint_ends_a_soak_cleanly_with_its_summary_written(cyan: Cyan) -> None:
    # An hour's soak, interrupted two seconds into its window: the calls end with their turns, AIPerf stops, a last
    # scrape is taken, and the record says SIGINT ended it, its summary up to then.
    out = cyan.host.root / "runs" / "s"

    def interrupt() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with_window = (out / "soak.json").exists() and "window" in (
                out / "soak.json"
            ).read_text()
            if with_window:
                time.sleep(2.0)
                os.kill(os.getpid(), signal.SIGINT)
                return
            time.sleep(0.05)

    started, handler = time.monotonic(), signal.getsignal(signal.SIGINT)
    threading.Thread(target=interrupt, daemon=True).start()
    summary = soak(cyan, hours=1.0)
    assert time.monotonic() - started < 30
    record = record_of(cyan)
    assert (record["status"], record["ended_by"]) == ("complete", "SIGINT")
    [calls] = record["calls_runs"]
    assert calls["how"] == "stopped" and calls["exit_status"] == 0
    assert summary["window"]["hours"] < 0.01 and len(summary["hours"]) == 1
    assert summary["overall"]["calls"]["attempted"] >= 1
    assert json.loads((out / "summary.json").read_text()) == summary
    series = (out / "series.jsonl").read_text().splitlines()
    assert json.loads(series[-1])["label"] == "after"
    assert signal.getsignal(signal.SIGINT) is handler  # the soak's handlers are gone


@needs_waitid
def test_the_guard_pauses_the_load_while_another_holds_the_gpu(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The GPU is clear at the start, taken for two checks, then clear: the soak stops its tools, records the episode,
    # and starts them again once the GPU has been clear for its hold-off.
    monkeypatch.setattr(soak_run, "LEFT_S", 0.5)  # a window of seconds still has time for the tools
    cyan.checks[:] = ["clear", "clear", "taken", "taken", "clear", "clear", "clear", "clear"]
    summary = soak(cyan, hours=9 / 3600, guard_clear_s=1.0)
    record = record_of(cyan)
    assert record["status"] == "complete"
    [pause] = record["pauses"]
    assert (
        pause["end"] > pause["start"] and "pid 9001 (another's train.py, 5120 MiB)" in pause["why"]
    )
    kinds = [e["event"] for e in events_of(cyan)]
    for kind in ("guard_taken", "load_paused", "guard_clear", "load_resumed"):
        assert kind in kinds, kind
    assert kinds.index("load_paused") < kinds.index("load_resumed")
    assert [r["how"] for r in record["calls_runs"][:1]] == ["stopped"]
    assert len(record["calls_runs"]) == 2 and len(record["agents_runs"]) == 2
    # The second run's calls are numbered after the first's.
    first = record["calls_runs"][0]
    out = cyan.host.root / "runs" / "s"
    numbers = [
        json.loads(line)["call"]
        for line in (out / "calls" / "run-000" / "attempts.jsonl").read_text().splitlines()
    ]
    assert record["calls_runs"][1]["first_call"] == max(numbers) + 1 and first["first_call"] == 0
    assert (
        summary["overall"]["guard"]["pauses"] == 1 and summary["overall"]["guard"]["paused_s"] > 0
    )
    assert summary["overall"]["guard"]["checks"]["taken"] == 2
    taken = next(e for e in events_of(cyan) if e["event"] == "guard_taken")
    assert taken["processes"][1] == {"whose": "other", "pid": 9001, "owner": "another", "memory_mib": 5120,
                                     "name": "train.py"}  # fmt: skip


@needs_waitid
def test_a_tool_that_fails_is_started_again(cyan: Cyan, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(soak_run, "RESTART_BACKOFF_S", (0.2, 0.2, 0.2, 0.2))
    monkeypatch.setattr(soak_run, "LEFT_S", 0.5)
    monkeypatch.setenv("FAKE_ARRIVALS", "crash")  # the callers' first run cannot start
    soak(cyan, hours=4 / 3600)
    record = record_of(cyan)
    failed, again = record["calls_runs"][:2]
    assert (failed["exit_status"], failed["how"]) == (1, "ended") and "could not start" in failed[
        "log_tail"
    ]
    assert again["exit_status"] == 0 and again["run"] == 1
    ended = next(e for e in events_of(cyan) if e["event"] == "calls_ended")
    assert ended == ended | {"run": 0, "exit_status": 1, "restart_in_s": 0.2}
    summary = json.loads((cyan.host.root / "runs" / "s" / "summary.json").read_text())
    assert summary["overall"]["restarts"]["calls_runs_ended"] == 1


@needs_waitid
def test_a_soak_does_not_start_while_an_e2_point_holds_cyan(cyan: Cyan) -> None:
    with exclusive(cyan.host), pytest.raises(E2Error, match="holds"):
        soak(cyan)
    assert not (cyan.host.root / "runs" / "s").exists()


@needs_waitid
@pytest.mark.parametrize(
    ("change", "says"),
    [
        (lambda c: c.systemd.env.update(FLATSHARE_VOICE_MODE="bench"), "runs bench mode"),
        (
            lambda c: c.status.write_text(json.dumps({"updated": time.time() - 60})),
            "its last is 60",
        ),
        (lambda c: c.gateway.down.add("/ready"), "/ready answered 503"),
        (
            lambda c: c.systemd.env.update(FLATSHARE_SERVING_PROFILE="/nonexistent"),
            "serving profile",
        ),
        (lambda c: c.checks.__setitem__(0, "taken"), "in use by pid 9001"),
        (lambda c: c.checks.__setitem__(0, "blind"), "deploy/authorize-guard.sh cyan phastform"),
        (
            lambda c: (
                (c.host.units / "flatshare-gateway.service.d").mkdir(parents=True)
                or (
                    c.host.units / "flatshare-gateway.service.d" / "variant-e2-gateway.conf"
                ).write_text("")
            ),
            "flatshare-bench e2 reset removes them",
        ),
    ],
)
def test_a_soak_refuses_a_box_that_does_not_serve_as_deployed(
    cyan: Cyan, change: Any, says: str
) -> None:
    change(cyan)
    with pytest.raises(E2Error, match=re.escape(says)):
        soak(cyan)
    record = record_of(cyan)
    assert record["status"] == "failed" and says.split()[0] in record["error"]
    assert record["calls_runs"] == [] and record["agents_runs"] == []  # nothing started


@needs_waitid
def test_a_soak_can_be_voice_alone_and_unguarded(cyan: Cyan) -> None:
    # Voice alone, the SLO's reference measured the same way: no AIPerf; and a node whose guard is off, said so.
    soak(cyan, agents_sessions=0, guards={"phastform": "off"})
    record = record_of(cyan)
    assert record["status"] == "complete" and record["agents_runs"] == []
    assert record["serving"]["guards"] == [{"node": "phastform", "method": "off"}]
    assert not (cyan.host.root / "runs" / "s" / "guard.jsonl").exists()
    assert "no guard on phastform" in (cyan.host.root / "runs" / "s" / "soak.log").read_text()


def test_the_cli_reports_and_exports_a_soak(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    soak_dir = synthetic_soak(tmp_path / "soak")
    assert main(["soak", "report", str(soak_dir), "--slo-reference", "1784"]) == 0
    table = capsys.readouterr().out
    assert "| 0 | 5 | 2 | 2 | 1 | 6 | 1 | 21,750 | 57.1% |" in table
    assert main(["soak", "report", str(soak_dir), "--write"]) == 0  # the soak's own reference
    assert json.loads((soak_dir / "summary.json").read_text())["slo"]["reference_p95_ms"] == 1784.0
    capsys.readouterr()
    assert main(["soak", "export", str(soak_dir), "--out", str(tmp_path / "record")]) == 0
    assert (tmp_path / "record" / "series.csv").exists()


def test_the_worker_status_is_read_with_its_age(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    assert read_worker_status(path) is None
    path.write_text(
        json.dumps({"updated": 100.0, "mode": "serve", "active_calls": 2, "max_calls": 2})
    )
    status = read_worker_status(path, now=101.25)
    assert status is not None and (status["age_s"], status["active_calls"]) == (1.2, 2)
    # What the worker runs, as it describes itself: its endpointing and whether it binds calls to nodes.
    described = {"audio": "synthetic", "turn_detection": "vad", "node_binding": True}
    path.write_text(json.dumps({"updated": 100.0, "mode": "serve"} | described))
    status = read_worker_status(path, now=101.25)
    assert status is not None and {key: status[key] for key in described} == described


def test_a_complete_soak_is_never_written_over(tmp_path: Path) -> None:
    out = tmp_path / "s"
    out.mkdir()
    (out / "soak.json").write_text(json.dumps({"status": "complete"}))
    with pytest.raises(E2Error, match="holds a complete soak"):
        soak_run.fresh(out)
    (out / "soak.json").write_text(json.dumps({"status": "failed"}))
    soak_run.fresh(out)  # an incomplete one goes aside
    assert out.exists() and not (out / "soak.json").exists()
    assert [p.name.startswith("s.incomplete-") for p in tmp_path.iterdir() if p.name != "s"] == [
        True
    ]
    other = tmp_path / "other"
    other.mkdir()
    (other / "x").write_text("")
    with pytest.raises(E2Error, match="holds no soak"):
        soak_run.fresh(other)
