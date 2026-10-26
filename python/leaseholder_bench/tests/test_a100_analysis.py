"""benchmarks/a100/analysis.py on synthetic records: the forecast's costs read back, each fitted term set beside them,
the probes beside the forecast and the box, the running shares by configuration, and a report that says what the
records lack. (Its calibration code runs on the box's W2 exports in `make check-a100-calibration`, and the whole
report on the CPU dry run's records in test_a100_dry_run.py.)"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
A100 = REPO / "benchmarks" / "a100"


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = _module("lh_a100_analysis", A100 / "analysis.py")
DASH = "\u2013"  # the range dash
PROBE = (
    "{case}: priority-0 TTFT {ttft} ms, 16 tokens {t16} ms, for 761 prompt tokens, sent with 2 priority-2 requests of"
    " 17999 tokens running and 24 of 2274 blocks free; after it, 2 running and 0 waiting\n"
)
TAIL = (
    "calls: 40 turns, 0 failed (counted at 30,000 ms), TTFT p50 110 p90 300 p95 {p95} max 900 ms; agents: 0 of 12"
    " stopped early\n"
)


def test_the_forecasts_costs_read_back() -> None:
    costs = analysis.forecast_costs(analysis.FORECAST.read_text())
    assert costs["central"] == {
        "step_ms": 4.55,
        "us_per_token": 41.33,
        "ns_per_pair": 2.59,
        "ns_per_context": 123.7,
    }
    assert costs["low"]["us_per_token"] == 29.19 and costs["high"]["ns_per_pair"] == 3.78
    with pytest.raises(ValueError):
        analysis.forecast_costs("## Step and tier costs, Qwen3-4B-Instruct-2507-FP8\n\nno table\n")


def test_each_fitted_term_stands_beside_its_forecast() -> None:
    forecast = analysis.forecast_costs(analysis.FORECAST.read_text())
    steps = {"step_ms": 5.0, "us_per_token": 60.0, "ns_per_pair": 2.6, "ns_per_context": 120.0}
    lines = analysis.step_model_lines(steps, forecast)
    assert (
        f"| Step: fixed | 5 ms | 4.55 ms | 3.33{DASH}4.55 ms | 1.10, outside the range |" in lines
    )
    assert f"| per pair | 2.6 ns | 2.59 ns | 2.15{DASH}3.78 ns | 1.00 |" in lines
    # 60 µs a token is 1.45 times the forecast: the largest miss, and what it points at.
    assert lines[-1].startswith(
        "The largest miss is the per token term, 1.45 times the forecast (outside its range)"
    )
    assert "Marlin" in lines[-1]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def probe_runs(runs: Path) -> None:
    for i, (ttft, t16) in enumerate(((1600, 1700), (1700, 1800), (1650, 1750)), 1):
        write(runs / "block1" / f"frozen-kv-{i}.txt", PROBE.format(case="kv", ttft=ttft, t16=t16))
    for i, ttft in enumerate((140, 160, 150), 1):
        write(
            runs / "block3" / f"pr-kv-{i}.txt", PROBE.format(case="kv", ttft=ttft, t16=ttft + 120)
        )
    for i, (ttft, t16) in enumerate(((300, 800), (320, 830)), 1):  # 33.3 and 34.0 ms a token
        write(
            runs / "block1" / f"frozen-prefill-{i}.txt",
            PROBE.format(case="prefill", ttft=ttft, t16=t16),
        )
    write(
        runs / "block3" / "pr-prefill-1.txt", PROBE.format(case="prefill", ttft=200, t16=290)
    )  # 6.0 ms
    for prefix, block, p95s in (
        ("frozen-tail", "block1", (1800, 1900)),
        ("stock2048-tail", "block2", (1500, 1700)),
        ("stock1536-768-tail", "block2", (1400, 1600)),
        ("pr-tail", "block3", (300, 340)),
    ):
        for i, p95 in enumerate(p95s, 1):
            write(runs / block / f"{prefix}-{i}.txt", TAIL.format(p95=p95))


def test_the_probes_stand_beside_the_forecast_and_the_box(tmp_path: Path) -> None:
    probe_runs(tmp_path)
    lines = analysis.probe_lines(tmp_path)
    text = "\n".join(lines)
    assert (
        f"| KV case, stock: the voice turn's wait | 1,650 ms (1,600{DASH}1,700, 3 runs) | 1,636{DASH}1,688 ms"
        in text
    )
    assert (
        f"| KV case, the PRs' values: first token | 150 ms (140{DASH}160, 3 runs) | 134{DASH}158 ms"
        in text
    )
    assert (
        f"| Prefill case, stock: the reply's token gap (1st to 16th) | 34 ms (33{DASH}34, 2 runs) |"
        in text
    )
    assert "| Prefill case, the PRs' values: the token gap | 6 ms | 6.6 ms" in text
    assert "| KV case, stock at 2,048/0 | not measured |" in text
    assert (
        "The calls' first-token p95 beside the agents (tail.py): frozen, 512/256 1,850 ms "
        f"(1,800{DASH}1,900, 2 runs); stock, 2,048/0 1,600 ms (1,500{DASH}1,700, 2 runs); stock, 1,536/768 1,500 ms "
        f"(1,400{DASH}1,600, 2 runs); the PRs' values 320 ms (300{DASH}340, 2 runs)."
    ) in text


def test_running_shares_by_configuration(tmp_path: Path) -> None:
    for name, label, share in (
        ("a", "frozen", 0.12),
        ("b", "frozen", 0.15),
        ("c", "PRs", 0.05),
        ("d", "voice", None),
        ("e", "vLLM PRs A and B with a 48-token window", 0.06),
    ):
        arm = {"label": label} | ({"name": "w5-disc-pr-ab48-s24"} if name == "e" else {})
        write(tmp_path / name / "run.json", json.dumps({"arm": arm}))
        kv = {"running_share": share} if share is not None else None
        write(tmp_path / name / "summary.json", json.dumps({"kv": kv}))
    points = sorted(tmp_path.iterdir())
    # A point of the A100's arm files goes by its name in the rounds' table, as the docs give it.
    assert analysis.running_shares(points) == {
        "frozen": [0.12, 0.15],
        "PRs": [0.05],
        "The PRs' values": [0.06],
    }


def test_a_report_says_what_its_records_lack(tmp_path: Path) -> None:
    host = tmp_path / "host"
    probe_runs(host / "runs" / "a100")
    out = tmp_path / "out"
    # No calibration point and no E2 point: lh-sim and lh-bench are never called.
    path = analysis.report(host, tmp_path / "corpus", tmp_path / "no-lh-sim", out, None)
    text = path.read_text()
    assert "## 1. The step model\n\nNot measured: no calibration point" in text
    assert "## 2. The calibration table\n\nNot measured." in text
    assert "| block1/frozen-kv-1 | kv | 1600 | 1700 |" in text
    assert "## 4. E2 within the A100\n\nNot measured: no E2 point" in text
    assert "## 5. A': the box's pool, more agents\n\nNot measured: no A' point" in text
    assert "## 6. The scheduler at a batch of 256\n\nNot measured: no batch arm" in text
    assert "## 7. The host around the card\n\nNot measured: no host check" in text
    assert "## 8. Across cards (different hardware)" in text


def test_the_core_section_reads_the_core_rounds_alone(tmp_path: Path) -> None:
    # §4 is the core's E2 within the A100: its rounds a100-r1 to -r5 (and tuned vLLM's), not the other optional arms'
    # campaigns, whose names also begin a100-r (a100-resubmit-r3 to -r5); A' (a100-aprime-*) has its own section, and
    # a point not complete (one in progress, or stopped) is read nowhere.
    host = tmp_path / "host"
    probe_runs(host / "runs" / "a100")
    for campaign, label, share in (
        ("a100-r1", "frozen", 0.12),
        ("a100-r2", "frozen", 0.14),
        ("a100-resubmit-r3", "stock, preempt-resubmit", 0.50),
        ("a100-aprime-r1", "the box's pool", 0.18),
        ("a100-r3", "in progress", 0.30),
    ):
        point = host / "results" / "e2" / campaign / "w5-disc-frozen-s24"
        status = "running" if label == "in progress" else "complete"
        write(point / "run.json", json.dumps({"status": status, "arm": {"label": label}}))
        write(point / "summary.json", json.dumps({"kv": {"running_share": share}}))
    # Voice alone, the SLO's reference, which A' reads from the core's rounds for its report alone.
    voice = host / "results" / "e2" / "a100-r1" / "w5-a1-voice"
    write(
        voice / "run.json",
        json.dumps(
            {
                "status": "complete",
                "arm": {"name": "w5-a1-voice", "label": "voice alone"},
                "agents_seed": 11,
            }
        ),
    )
    write(voice / "summary.json", json.dumps({"voice": {"v2v_ms": {"p95": 1445.0}}}))
    text = analysis.report(
        host, tmp_path / "corpus", tmp_path / "no-lh-sim", tmp_path / "out", None
    ).read_text()
    section = text.split("## 4. E2 within the A100", 1)[1].split("## 5.", 1)[0]
    assert "- frozen: 12%, 14%" in section
    assert "preempt-resubmit" not in section and "the box's pool" not in section
    assert "in progress" not in text
    aprime = text.split("## 5. A'", 1)[1].split("## 6.", 1)[0]
    assert (
        "- the box's pool: 18%" in aprime
        and "against the forecast's 18% (central) for A'" in aprime
    )
    # Voice alone has its round's row in §4; in A' it is the report's SLO reference alone, with no row or share.
    assert "| 11 | 1,445 |" in section
    assert "oice alone" not in aprime


def test_the_tail_line_names_each_stock_setting_by_its_cap_and_threshold(tmp_path: Path) -> None:
    # --cap 2048, the forecast's low value: its runs (2,048/1,024) beside vLLM's default 2,048/0, each its own.
    probe_runs(tmp_path)
    for i, p95 in enumerate((1100, 1300), 1):
        write(tmp_path / "block2" / f"stock2048-1024-tail-{i}.txt", TAIL.format(p95=p95))
    text = "\n".join(analysis.probe_lines(tmp_path))
    assert f"stock, 2,048/0 1,600 ms (1,500{DASH}1,700, 2 runs)" in text
    assert f"stock, 1,536/768 1,500 ms (1,400{DASH}1,600, 2 runs)" in text
    assert f"stock, 2,048/1,024 1,200 ms (1,100{DASH}1,300, 2 runs)" in text


def host_json(host: Path, gpu: str) -> None:
    """The node record pull.sh copies, with the card as nvidia-smi names it."""
    write(host / "node" / "host.json", json.dumps({"hostname": "a100", "gpu": {"name": gpu}}))


@pytest.mark.parametrize(
    ("gpu", "forecast", "card"),
    [
        ("NVIDIA A100-SXM4-80GB", "forecast-80gb.md", "A100 SXM4 80 GB"),
        ("NVIDIA A100 80GB PCIe", "forecast-80gb.md", "A100 SXM4 80 GB"),
        ("NVIDIA A100-SXM4-40GB", "forecast.md", "A100 40 GB"),
        ("NVIDIA A100-PCIE-40GB", "forecast.md", "A100 40 GB"),
        ("NVIDIA H100 80GB HBM3", "forecast.md", "A100 40 GB"),
        (None, "forecast.md", "A100 40 GB"),
    ],
)
def test_the_forecast_is_the_one_for_the_card_the_records_name(
    tmp_path: Path, gpu: str | None, forecast: str, card: str
) -> None:
    host = tmp_path / "host"
    if gpu is not None:
        host_json(host, gpu)
    assert analysis.gpu_name(host) == gpu
    path = analysis.forecast_for(analysis.gpu_name(host))
    assert path == A100 / forecast
    assert analysis.read_forecast(path).card == card


def test_each_forecast_reads_back_its_own_cards_numbers() -> None:
    f40, f80 = (
        analysis.read_forecast(analysis.FORECAST),
        analysis.read_forecast(analysis.FORECAST_80GB),
    )
    # The 80 GB card's bandwidth terms are smaller and its compute terms the 40 GB card's.
    assert f80.costs["central"] == {
        "step_ms": 3.88,
        "us_per_token": 41.33,
        "ns_per_pair": 2.59,
        "ns_per_context": 94.3,
    }
    assert f80.costs["low"]["step_ms"] == 2.54 and f40.costs["low"]["step_ms"] == 3.33
    assert f80.probes == {
        "kv_stock": f"1,328{DASH}1,369 ms (1,082{DASH}1,377)",
        "kv_pr": f"126{DASH}145 ms (109{DASH}146)",
        "prefill_stock": f"32.8 ms (25.7{DASH}35.4)",
        "prefill_pr": f"5.6 ms (3.7{DASH}5.6)",
    }
    assert f40.probes["kv_stock"] == f"1,636{DASH}1,688 ms (1,411{DASH}1,695)"
    assert f80.core == {"running": "10%", "busy": "71%", "stalled": "50%"}
    assert f40.core == {"running": "13%", "busy": "76%", "stalled": "57%"}
    with pytest.raises(ValueError):
        analysis.forecast_core(analysis.FORECAST.read_text(), "A100 SXM4 80 GB")
    # A forecast that places the card on a host (lh-sim forecast --host) names its cells so.
    text = analysis.FORECAST_80GB.read_text()
    on_host = text.replace(" (central) |", " (central, on this host) |")
    assert on_host != text
    assert analysis.forecast_core(on_host, f80.card) == f80.core
    assert analysis.forecast_core(on_host, f80.card, analysis.APRIME) == f80.aprime


@pytest.mark.parametrize(
    ("gpu", "forecast"),
    [("NVIDIA A100-SXM4-80GB", "forecast-80gb.md"), ("NVIDIA A100-SXM4-40GB", "forecast.md")],
)
def test_a_report_compares_against_its_cards_forecast_and_says_which(
    tmp_path: Path, gpu: str, forecast: str
) -> None:
    host = tmp_path / "host"
    host_json(host, gpu)
    probe_runs(host / "runs" / "a100")
    point = host / "results" / "e2" / "a100-r1" / "w5-disc-frozen-s24"
    write(point / "run.json", json.dumps({"status": "complete", "arm": {"label": "frozen"}}))
    write(point / "summary.json", json.dumps({"kv": {"running_share": 0.11}}))
    f = analysis.read_forecast(A100 / forecast)
    text = analysis.report(
        host, tmp_path / "corpus", tmp_path / "no-lh-sim", tmp_path / "out", None
    ).read_text()
    assert text.startswith(f"# The {f.card} run against its forecast\n")
    assert f"It compares against benchmarks/a100/{forecast}, the forecast for the {f.card}" in text
    assert f"node/host.json names the card {gpu}" in text
    assert f"| Case | {f.card}, measured |" in text
    assert (
        f"| KV case, stock: the voice turn's wait | 1,650 ms (1,600{DASH}1,700, 3 runs) | {f.probes['kv_stock']} |"
        in text
    )
    assert f"| 150 ms (140{DASH}160, 3 runs) | {f.probes['kv_pr']} |" in text
    assert f"against the forecast's {f.core['running']} (central) for the core" in text
    assert (
        f"the forecast puts the {f.card} at {f.core['busy']} busy, {f.core['stalled']} stalled and "
        f"{f.core['running']} [modeled]"
    ) in text
    # Neither card's report quotes the other's forecast.
    other = analysis.read_forecast(
        analysis.FORECAST if forecast == "forecast-80gb.md" else analysis.FORECAST_80GB
    )
    assert other.probes["kv_stock"] not in text and other.card not in text


def test_records_without_a_card_compare_against_the_40gb_forecast(tmp_path: Path) -> None:
    host = tmp_path / "host"
    probe_runs(host / "runs" / "a100")
    text = analysis.report(
        host, tmp_path / "corpus", tmp_path / "no-lh-sim", tmp_path / "out", None
    ).read_text()
    assert text.startswith("# The A100 40 GB run against its forecast\n")
    assert (
        "It compares against benchmarks/a100/forecast.md, the forecast for the A100 40 GB: the records have no "
        "node/host.json naming their card."
    ) in text


def e2_point(host: Path, campaign: str, arm: str, seed: int, p95: float, rate: float) -> None:
    """A complete E2 point as campaign.sh leaves it: its arm file's name and agents seed, and its summary's voice
    p95 and the agents' tokens a second over the calls' first 240 s."""
    point = host / "results" / "e2" / campaign / arm
    write(
        point / "run.json",
        json.dumps({"status": "complete", "arm": {"name": arm, "label": arm}, "agents_seed": seed}),
    )
    write(
        point / "summary.json",
        json.dumps(
            {
                "voice": {"v2v_ms": {"p95": p95}},
                "background": {"fixed_window": {"tokens_per_s": rate}},
            }
        ),
    )


def test_tuned_vllm_pairs_with_the_cores_rounds_by_the_agents_sample(tmp_path: Path) -> None:
    # Tuned vLLM's rounds replay the samples of the core's first two: each round's row sets the configurations side by
    # side, each stock engine beside the PRs' values at its step cap, and a configuration a round lacks leaves its cell
    # empty.
    host = tmp_path / "host"
    for campaign, arm, seed, p95, rate in (
        ("a100-r1", "w5-disc-pr-ab48-s24", 11, 1663.4, 10125.2),
        ("a100-r1", "w5-a1-voice", 11, 1412.0, 0),
        ("a100-r1", "w5-disc-frozen-s24", 11, 4120.0, 10900.0),
        ("a100-tuned-r1", "a100-tuned-s24", 11, 2950.0, 11800.0),
        ("a100-r2", "w5-disc-frozen-s24", 12, 4400.0, 10800.0),
    ):
        e2_point(host, campaign, arm, seed, p95, rate)
    lines = analysis.rounds_table(
        analysis.complete_points(host, "a100-r[0-9]*")
        + analysis.complete_points(host, "a100-tuned-r*")
    )
    assert lines == [
        "| Agents' sample (seed) | Voice alone | Frozen, 512/256 | The PRs' values | Tuned vLLM, 2,048/0 |",
        "|---|---|---|---|---|",
        "| 11 | 1,412 | 4,120 / 10,900 | 1,663 / 10,125 | 2,950 / 11,800 |",
        "| 12 |  | 4,400 / 10,800 |  |  |",
    ]


def test_the_batch_section_sets_make_rooms_output_beside_the_frozen_engines(tmp_path: Path) -> None:
    # More requests running at once lengthen each one's time per token while the engine makes more tokens a second,
    # so the section reads the code path's cost from the output (the A100's "off" engine: 215 running against 201,
    # TPOT 6% longer, output 2% higher).
    batch = tmp_path / "batch"
    for name, tpot, output, ttft99 in (
        ("frozen", 29.0, 9000.0, 1100.0),
        ("off", 30.8, 9180.0, 1200.0),
        ("on", 28.6, 10350.0, 8000.0),
    ):
        for priority, share, p99 in ((0, 0.1, 500.0), (2, 0.9, ttft99)):
            write(
                batch / f"{name}-p{priority}.json",
                json.dumps(
                    {
                        "output_throughput": output * share,
                        "median_tpot_ms": tpot,
                        "p99_tpot_ms": tpot * 1.5,
                        "median_ttft_ms": 80.0,
                        "p99_ttft_ms": p99,
                    }
                ),
            )
        running = {"frozen": 201.0, "off": 215.0, "on": 154.0}[name]
        write(
            batch / f"{name}-running.json",
            json.dumps({"running_p50": running, "running_max": 256.0}),
        )
    lines = analysis.batch_lines(batch)
    assert (
        "| frozen | 9,000 | 80 / 500 | 29.00 / 43.50 | 80 / 1,100 | 29.00 / 43.50 | 201 / 256 |"
        in lines
    )
    assert (
        "| make-room, the PRs' values | 10,350 | 80 / 500 | 28.60 / 42.90 | 80 / 8,000 |"
        in lines[-3]
    )
    assert lines[-1] == (
        "Output tokens a second at the batch, against the frozen engine's: make-room, switches off 1.020; "
        "make-room, the PRs' values 1.150. With its switches off make-room schedules as vLLM does, so a ratio "
        "under 1.000 there is its code path's cost per step."
    )
    assert analysis.batch_lines(tmp_path / "none") == [
        "Not measured: no batch arm (runs/a100/batch)."
    ]


def test_the_host_section_reads_e0_against_the_link_and_the_cpu_against_the_boxs(
    tmp_path: Path,
) -> None:
    # E0's GPU hit grows with the prompt by the host's own work on it, and the CPU restore beyond the GPU hit is the
    # transfer, each a least-squares slope over the prompt lengths: here 1.83 and 12.84 µs a token, so 11.49 GB/s
    # of 144 KiB a token, against the box's seven cells in the forecast's E0 table (3.79 and 23.60) and the
    # forecast's own link.
    e0 = {
        "gpu": {"1024": 18.27, "31744": 74.57},
        "cpu": {"1024": 32.09, "31744": 482.72},
    }
    record = {
        "cpu": "Intel(R) Xeon(R) CPU @ 2.20GHz",
        "python_ms": {"p50": 190.0},
        "gpu": {"pcie.link.gen.current": "4", "pcie.link.width.current": "16"},
        "link": [
            {"direction": "h2d", "page_bytes": 32768, "gb_per_s": 10.5},
            {"direction": "h2d", "page_bytes": None, "gb_per_s": 12.1},
            {"direction": "d2h", "page_bytes": None, "gb_per_s": 12.9},
        ],
    }
    lines = analysis.host_lines(record, e0, analysis.FORECAST_80GB.read_text())
    text = "\n".join(lines)
    assert (
        "in 190.0 ms, the box's Xeon E5-2609 v4 at 1.70 GHz in 381.1: 2.01 times as fast." in text
    )
    assert "by 1.83 ms every 1,000 tokens on this host and 3.79 on the box (2.07 times)" in text
    assert (
        "takes 12.84 µs a token on this host, 11.49 GB/s of KV, and 23.60 on the box (6.25 GB/s)"
        in text
    )
    assert "the forecast took 5.91 µs a token (25.0 GB/s)" in text
    assert "(PCIe Gen4 x16 as the card negotiates it; the box's PCIe 3.0 x8), in GB/s:" in text
    assert "| pages of 32 KiB | 10.50 | 5.44 |" in lines and "| one copy | 12.10 | 6.76 |" in lines
    assert lines[-1] == (
        "One copy reached 12.10 GB/s, 38% of the 31.5 GB/s each way that the card's link carries; the box's reached "
        "6.76, 86% of its link's 7.9. A rate well under the card's own link is set by a narrower hop between the "
        "card and the host's memory."
    )
    assert analysis.host_lines({"link_error": "RuntimeError: no GPU"}, None, "") == [
        "- **The link** was not measured: RuntimeError: no GPU"
    ]
