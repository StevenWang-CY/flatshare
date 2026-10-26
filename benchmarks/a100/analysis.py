"""The A100 run's analysis (docs/HARDWARE.md, the analysis), on a CPU, from a single host's records as
deploy/a100/pull.sh copies them.

    python3 benchmarks/a100/analysis.py report HOST_DIR --corpus WEKA_X32 --flatshare-sim FLATSHARE_SIM --out OUT
        [--flatshare-bench BIN]
    python3 benchmarks/a100/analysis.py calibration-check EXPORTS --corpus WEKA_X32 --flatshare-sim FLATSHARE_SIM
        --out OUT

`report` writes OUT/a100.md (`make reproduce-a100`), each section from what HOST_DIR holds and saying what it lacks,
against the forecast for the card the records name (node/host.json's GPU): benchmarks/a100/forecast-80gb.md for an
A100 80 GB, benchmarks/a100/forecast.md (the A100 40 GB's, the card the kit was built for) otherwise; a100.md says
which.
1. The step model: crates/flatshare-sim/scripts/fit_step_cost.py on the calibration point (runs/a100/calibration-s16)
   with the card's own E0 (runs/a100/block1/frozen-e0), each fitted term beside the forecast's central value and range
   as a ratio, and the largest miss with what it is likely to mean.
2. The calibration table: flatshare-sim with those costs (and E0's tier fits) against the measured point, as the box's
   (crates/flatshare-sim/scripts/calibration.py, whose functions it uses): requests, resume TTFT p50 / p90 / p99 and the
   GPU, CPU and computed shares of the prompt tokens.
3. The probes (probe_results.py): every run, then the KV case's wait and the prefill case's token gap on stock and on
   the PRs' values, and the calls' tail at each stock setting, beside the forecast and the box.
4. E2 within the A100: `flatshare-bench e2 report` over the core's points, tuned vLLM's and the PRs' values at its step
   cap (results/e2/a100-r1 to -r5, a100-tuned-r1 and -r2, a100-prtuned-r1 and -r2) into OUT/e2, its frontier quoted, the
   points by round (each round one sample of the agents, which the two arms' rounds share with the core's first two),
   and each configuration's measured running share against the forecast's.
5. A': the same over its points (results/e2/a100-aprime-r1, -r2) and the core's voice alone, the SLO's reference on
   this card, into OUT/e2-aprime, against the forecast's A' cell.
6. The scheduler at a batch of 256 (runs/a100/batch, batch.sh): each engine's throughput, TPOT and TTFT by priority,
   the batch it held, and each make-room engine's output tokens a second against the frozen engine's.
7. The host around the card (runs/a100/host/host.json, bench_host.py): its CPU and its link against the box's, read
   with E0's GPU hit (the host's work on a prompt) and its CPU restore beyond it (the transfer).
8. Across cards: the regime coordinates the forecast gives the box, on its measured costs, and the A100, beside the
   A100's measured running share, labeled different hardware.

`calibration-check` is P-1's prep 6: report's calibration code on the box's three W2 operating points (the
calibration exports, calibration.py's EXPORTS) with the box's own E0 gives calibration.py report's rows for them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "crates" / "flatshare-sim" / "scripts"
FORECAST = HERE / "forecast.md"
"""The forecast for the A100 40 GB, the card the kit was built for (make forecast-a100)."""
FORECAST_80GB = HERE / "forecast-80gb.md"
"""The forecast for the A100 SXM4 80 GB (make forecast-a100 GPU=a100-80gb)."""
DASH = "\u2013"  # the range dash, as calibration.py writes it
CORE = "A: the core, the box's pool"
APRIME = "A': the box's pool, more agents"
"""The forecast's scenarios the run measures: the core and A', by their names in its table."""


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibration = _module("flatshare_calibration", SCRIPTS / "calibration.py")
results = _module("flatshare_a100_probe_results", HERE / "probe_results.py")

TERMS = (
    ("step_ms", "Step: fixed", "ms", "the step's launches and kernel tails beyond the weight read"),
    (
        "us_per_token",
        "per token",
        "µs",
        "the linear layers' rate on this card's path (Marlin W8A16 on sm80)",
    ),
    ("ns_per_pair", "per pair", "ns", "FlashAttention 2's efficiency at these shapes"),
    ("ns_per_context", "per KV token read", "ns", "decode attention's bandwidth"),
)
"""fit_step_cost.py's terms, the forecast table's columns, their unit, and what a miss in each points at."""
BOX = {
    "kv_stock_ms": f"4,923{DASH}5,077 ms (Gray, the second 5060 Ti)",
    "kv_pr_ms": "225 and 297 ms (Gray)",
    "prefill_stock_gap_ms": "69 ms (Gray)",
    "prefill_pr_gap_ms": "17 ms (Gray)",
}
"""The box's probe results, measured (docs/BENCHMARKS.md; benchmarks/results/2026-10-23-make-room-probes)."""
BOX_ROW = "RTX 5060 Ti 16 GB (measured)"
"""The forecasts' calibration row: flatshare-sim on the box's measured costs, the box's regime coordinates (W5's points
did not sample its running share)."""
BOX_HOST = {
    "cpu": "Xeon E5-2609 v4 at 1.70 GHz",
    "python_ms": 381.1,
    "link": "PCIe 3.0 x8",
    "lanes": (3, 8),
    "h2d_gb_per_s": {32 * 1024: 5.44, 64 * 1024: 6.03, 16 * 144 * 1024: 6.74, None: 6.76},
}
"""The box's host, measured by bench_host.py on Phastform (2026-10-26, its GPU units stopped): the CPU at the fixed
Python workload and the link from pinned host memory to the card, by page size (None: the bytes as one copy)."""
PCIE_LANE_GB_PER_S = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}
"""A PCIe lane's rate each way by generation, in GB/s, after its line code (8b/10b to Gen2, 128b/130b from Gen3)."""
KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2
"""Qwen3-4B's KV a token at BF16: 36 layers of 8 KV heads of 128, K and V (144 KiB)."""
POINTS = {
    "w5-a1-voice": "Voice alone",
    "w5-disc-frozen-s24": "Frozen, 512/256",
    "w5-disc-pr-ab48-s24": "The PRs' values",
    "a100-tuned-s24": "Tuned vLLM, 2,048/0",
    "a100-pr-ab48-2048-s24": "The PRs' values, 2,048/0",
    "a100-aprime-frozen-s32": "Frozen, 512/256",
    "a100-aprime-pr-ab48-s32": "The PRs' values",
}
"""E2's points on the A100, by their arm file's name, in the order the tables give them."""


def gpu_name(host: Path) -> str | None:
    """The card the records name: node/host.json's GPU, as nvidia-smi gives it ("NVIDIA A100-SXM4-80GB")."""
    try:
        return json.loads((host / "node" / "host.json").read_text())["gpu"]["name"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def forecast_for(gpu: str | None) -> Path:
    """The forecast for the card: forecast-80gb.md for an A100 80 GB, forecast.md (the A100 40 GB's) otherwise."""
    name = (gpu or "").upper().replace(" ", "")
    return FORECAST_80GB if "A100" in name and "80GB" in name else FORECAST


def forecast_card(text: str) -> str:
    """The card a forecast is for, as its title names it ("A100 40 GB")."""
    title = re.match(r"# Forecast: the (.+) against the frozen box\n", text)
    if title is None:
        raise ValueError("the forecast has no title naming its card")
    return title.group(1)


def forecast_costs(text: str) -> dict[str, dict[str, float]]:
    """The forecast's costs on its card for the core's model, by setting (low, central, high), in the fit's units."""
    card = forecast_card(text)
    section = text.split("## Step and tier costs, Qwen3-4B-Instruct-2507-FP8", 1)[1].split(
        "\n## ", 1
    )[0]
    out: dict[str, dict[str, float]] = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0] == card and cells[1] in ("low", "central", "high"):
            values = [float(re.match(r"[\d.]+", c).group().replace(",", "")) for c in cells[2:6]]
            out[cells[1]] = dict(zip((name for name, *_ in TERMS), values, strict=True))
    if set(out) != {"low", "central", "high"}:
        raise ValueError(f"the forecast has no low, central and high cost rows for the {card}")
    return out


def forecast_probes(text: str, card: str) -> dict[str, str]:
    """The forecast's probe cases on its card, each "central ms (low-high)" as the forecast writes them."""
    on = rf"On the {re.escape(card)}: "
    kv = re.search(
        on
        + rf"(\S+) ms \(low{DASH}high (\S+)\)\. make-room's first token, [^:]*: (\S+) ms \((\S+)\)\.",
        text,
    )
    prefill = re.search(on + r"(\S+) ms \((\S+)\) and (\S+) ms \((\S+)\), with the chunks", text)
    if kv is None or prefill is None:
        raise ValueError(f"the forecast has no KV case or prefill case for the {card}")

    def pair(m: re.Match[str], i: int) -> str:
        return f"{m.group(i)} ms ({m.group(i + 1)})"

    return {
        "kv_stock": pair(kv, 1),
        "kv_pr": pair(kv, 3),
        "prefill_stock": pair(prefill, 1),
        "prefill_pr": pair(prefill, 3),
    }


def forecast_core(
    text: str, card: str, scenario: str = CORE, row: str | None = None
) -> dict[str, str]:
    """A scenario's cell on the forecast's card, central (the core's by default), on the host the forecast places the
    card on if it names one (flatshare-sim forecast --host), or on the row named: its running share, busy and stalled
    shares."""
    labels = (row,) if row else (f"{card} (central)", f"{card} (central, on this host)")
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 11 and cells[0] == scenario and cells[1] in labels:
            return {"running": cells[8].split(" / ")[0], "busy": cells[9], "stalled": cells[10]}
    raise ValueError(f"the forecast has no cell of {scenario!r} on {' or '.join(labels)}")


@dataclass(frozen=True)
class Forecast:
    """What the report sets beside the measurements, from one forecast."""

    path: Path
    card: str
    costs: dict[str, dict[str, float]]
    probes: dict[str, str]
    core: dict[str, str]
    aprime: dict[str, str]
    box: dict[str, str]


def read_forecast(path: Path) -> Forecast:
    text = path.read_text()
    card = forecast_card(text)
    return Forecast(
        path,
        card,
        forecast_costs(text),
        forecast_probes(text, card),
        forecast_core(text, card),
        forecast_core(text, card, APRIME),
        forecast_core(text, card, row=BOX_ROW),
    )


def forecast_line(gpu: str | None, forecast: Forecast) -> str:
    """Which forecast the report compares against, and why."""
    which = f"It compares against {_shown(forecast.path)}, the forecast for the {forecast.card}"
    if gpu is None:
        return which + ": the records have no node/host.json naming their card."
    if forecast.path == FORECAST_80GB:
        return which + f", as node/host.json names the card {gpu}."
    return which + f", as node/host.json names the card {gpu}, not an A100 80 GB."


def _shown(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def fit(export: Path, log: Path, e0: Path | None, out: Path) -> tuple[dict[str, Any], str]:
    """fit_step_cost.py on one run: the fitted costs and the fit's report."""
    args = [sys.executable, str(SCRIPTS / "fit_step_cost.py")]
    if e0 is not None:
        args += ["--e0", str(e0)]
    done = subprocess.run(
        [*args, str(export / calibration.FILES[0]), str(log)],
        check=True,
        capture_output=True,
        text=True,
    )
    out.write_text(done.stdout)
    return json.loads(done.stdout), done.stderr


def step_model_lines(steps: dict[str, Any], forecast: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        f"| Term | Fitted | Forecast, central | Range (low{DASH}high) | Fitted over central |",
        "|---|---:|---:|---:|---:|",
    ]
    misses = []
    for name, label, unit, cause in TERMS:
        value, central = steps[name], forecast["central"][name]
        low, high = sorted((forecast["low"][name], forecast["high"][name]))
        ratio = value / central
        inside = low <= value <= high
        misses.append((abs(math.log(ratio)), label, ratio, cause, inside))
        lines.append(
            f"| {label} | {value:g} {unit} | {central:g} {unit} | {low:g}{DASH}{high:g} {unit} | {ratio:.2f}"
            + ("" if inside else ", outside the range")
            + " |"
        )
    _, label, ratio, cause, inside = max(misses)
    lines += [
        "",
        f"The largest miss is the {label.lower().removeprefix('step: ')} term, {ratio:.2f} times the forecast"
        + (" (within its range)" if inside else " (outside its range)")
        + f": {cause}.",
    ]
    return lines


def calibration_lines(
    sim: Any, export: Path, sessions: int, options: list[str], name: str
) -> tuple[list[str], list[float]]:
    """The measured point and flatshare-sim's contention and idle models of it, as calibration.py's table rows."""
    m = calibration.measured(export)
    lines = [
        f"| {sessions} sessions | measured | {calibration.n(m['requests'])} | "
        f"{' / '.join(map(calibration.n, m['resume']))} | {calibration.shares(m['shares'])} |"
    ]
    contention: list[float] = []
    for model, extra in (("idle (v0)", ["--service", "idle"]), ("contention", [])):
        s = sim.run(f"{name}-{model.split()[0]}", "--concurrency", str(sessions), *options, *extra)
        requests, resume, sim_shares = calibration.sim_row(s)
        lines.append(
            f"| | {model} | {calibration.n(requests)} "
            f"({calibration.signed(100 * (requests / m['requests'] - 1), 1)}%) | "
            f"{' / '.join(map(calibration.n, resume))} ({calibration.errors(resume, m['resume'])}%) | "
            f"{calibration.shares(sim_shares)} |"
        )
        if model == "contention":
            contention = resume
    return lines, contention


TABLE_HEAD = [
    "| Point | | Requests | Resume TTFT p50 / p90 / p99 (ms) | GPU / CPU / computed (%) |",
    "|---|---|---|---|---|",
]


def median_range(values: list[float]) -> str:
    if not values:
        return "not measured"
    if len(values) == 1:
        return f"{values[0]:,.0f} ms"
    return f"{statistics.median(values):,.0f} ms ({min(values):,.0f}{DASH}{max(values):,.0f}, {len(values)} runs)"


def probe_lines(runs: Path, forecast: Forecast | None = None) -> list[str]:
    """The probes' runs, and the cases beside the forecast (the A100 40 GB's unless given) and the box."""
    forecast = forecast or read_forecast(FORECAST)
    f = forecast.probes

    def probes(block: str, prefix: str) -> list[Any]:
        return [
            r for r in results.runs(runs, block, prefix).values() if isinstance(r, results.Probe)
        ]

    def gap(run: Any) -> float | None:
        t16 = run.marks.get(16)
        return None if t16 is None else (t16 - run.ttft_ms) / 15

    lines = [results.markdown(runs).rstrip("\n"), ""]
    kv_stock = [r.ttft_ms for r in probes("block1", "frozen-kv")]
    kv_pr = [r.ttft_ms for r in probes("block3", "pr-kv")]
    kv_2048 = [r.ttft_ms for r in probes("block2", "stock2048-kv")]
    kv_flashinfer = [r.ttft_ms for r in probes("backend", "backend-kv")]
    pf_stock = [g for r in probes("block1", "frozen-prefill") if (g := gap(r)) is not None]
    pf_pr = [g for r in probes("block3", "pr-prefill") if (g := gap(r)) is not None]
    lines += [
        f"| Case | {forecast.card}, measured | Forecast, central (low{DASH}high) [modeled] | The box [measured] |",
        "|---|---|---|---|",
        f"| KV case, stock: the voice turn's wait | {median_range(kv_stock)} | {f['kv_stock']} "
        f"| {BOX['kv_stock_ms']} |",
        f"| KV case, stock at 2,048/0 | {median_range(kv_2048)} | | |",
        *(
            [f"| KV case, stock on FlashInfer's attention | {median_range(kv_flashinfer)} | | |"]
            if kv_flashinfer
            else []
        ),
        f"| KV case, the PRs' values: first token | {median_range(kv_pr)} | {f['kv_pr']} | {BOX['kv_pr_ms']} |",
        f"| Prefill case, stock: the reply's token gap (1st to 16th) | {median_range(pf_stock)} | "
        f"{f['prefill_stock']} | {BOX['prefill_stock_gap_ms']} |",
        f"| Prefill case, the PRs' values: the token gap | {median_range(pf_pr)} | {f['prefill_pr']} "
        f"| {BOX['prefill_pr_gap_ms']} |",
    ]
    # The time-equivalent caps' runs, by cap and threshold (stock1536-768-tail-N); vLLM's default is stock2048-.
    caps = sorted(
        (int(cap), int(threshold))
        for p in (runs / "block2").glob("stock*-*-tail-1.txt")
        if (parts := p.stem.removeprefix("stock").split("-"))[0].isdigit()
        and len(parts) == 4
        and (cap := parts[0]).isdigit()
        and (threshold := parts[1]).isdigit()
    )
    tails = []
    for label, block, prefix in (
        ("frozen, 512/256", "block1", "frozen-tail"),
        ("stock, 2,048/0", "block2", "stock2048-tail"),
        *(
            (f"stock, {cap:,}/{threshold:,}", "block2", f"stock{cap}-{threshold}-tail")
            for cap, threshold in caps
        ),
        ("the PRs' values", "block3", "pr-tail"),
    ):
        p95 = [
            r.ttft["p95"][0]
            for r in results.runs(runs, block, prefix).values()
            if isinstance(r, results.Tail) and r.ttft
        ]
        if p95:
            tails.append(f"{label} {median_range(p95)}")
    if tails:
        lines += [
            "",
            "The calls' first-token p95 beside the agents (tail.py): " + "; ".join(tails) + ".",
        ]
    return lines


def running_shares(points: list[Path]) -> dict[str, list[float]]:
    """Each configuration's measured running shares (the points' summary.json kv), by its name in POINTS, as the
    rounds' table gives it, or its arm's label."""
    shares: dict[str, list[float]] = {}
    for point in points:
        try:
            run = json.loads((point / "run.json").read_text())
            summary = json.loads((point / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        kv = summary.get("kv") or {}
        if "running_share" in kv:
            arm = run["arm"]
            shares.setdefault(POINTS.get(arm.get("name", ""), arm["label"]), []).append(
                kv["running_share"]
            )
    return shares


def complete_points(host: Path, campaigns: str) -> list[Path]:
    """The complete E2 points of the campaigns the glob names (run.json's status), as campaign.sh leaves them: a point
    in progress, or stopped, has a run.json too."""
    points = []
    for run in sorted(host.glob(f"results/e2/{campaigns}/*/run.json")):
        try:
            if json.loads(run.read_text()).get("status") == "complete":
                points.append(run.parent)
        except (OSError, json.JSONDecodeError):
            continue
    return points


def e2_lines(
    points: list[Path],
    out: Path,
    flatshare_bench: Path | None,
    forecast_running: str,
    scenario: str,
    reference: list[Path] | None = None,
) -> list[str]:
    """E2's points: flatshare-bench e2 report's frontier over them (with flatshare-bench), the points by round, and each
    configuration's measured running share against the forecast's for the scenario. `reference` (voice alone's
    points, the SLO's reference) enters the report alone."""
    lines = []
    if flatshare_bench is not None:
        subprocess.run(
            [
                str(flatshare_bench),
                "e2",
                "report",
                *map(str, [*(reference or []), *points]),
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        lines += [(out / "frontier.md").read_text().strip(), ""]
    if table := rounds_table(points):
        lines += [
            "Each round's points, the V2V p95 (ms) / the agents' tokens a second over the calls' first 240 s:",
            "",
            *table,
            "",
        ]
    lines += [
        "The measured running share (the time free KV was under two calls' 2,048 tokens while agents ran), "
        f"against the forecast's {forecast_running} (central) for {scenario} [modeled]:",
        "",
    ]
    lines += [
        f"- {label}: {', '.join(f'{100 * s:.0f}%' for s in values)}"
        for label, values in running_shares(points).items()
    ] or ["- Not measured: no point's summary.json has a running share (kv.running_share)."]
    return lines


def rounds_table(points: list[Path]) -> list[str]:
    """The points by round, each configuration's V2V p95 and the agents' tokens a second over the calls' first 240 s
    (summary.json), a round being one sample of the agents (run.json's agents seed): the pairs E2's design compares."""
    rounds: dict[int, dict[str, str]] = {}
    for point in points:
        try:
            run = json.loads((point / "run.json").read_text())
            summary = json.loads((point / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        label, seed = POINTS.get(run.get("arm", {}).get("name", "")), run.get("agents_seed")
        p95 = ((summary.get("voice") or {}).get("v2v_ms") or {}).get("p95")
        if label is None or seed is None or p95 is None:
            continue
        rate = ((summary.get("background") or {}).get("fixed_window") or {}).get("tokens_per_s")
        rounds.setdefault(seed, {})[label] = f"{p95:,.0f}" + (f" / {rate:,.0f}" if rate else "")
    labels = [
        label
        for label in dict.fromkeys(POINTS.values())
        if any(label in r for r in rounds.values())
    ]
    if not rounds:
        return []
    return [
        "| Agents' sample (seed) | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
        *(
            f"| {seed} | " + " | ".join(rounds[seed].get(label, "") for label in labels) + " |"
            for seed in sorted(rounds)
        ),
    ]


def batch_lines(batch: Path) -> list[str]:
    """The batch arm (batch.sh): each engine's two clients at 256 requests in flight, from vllm bench serve's results
    and the engine's sampled running requests, and each make-room engine's output tokens a second against the frozen
    engine's. With its switches off make-room schedules as vLLM does, so its ratio is its code path's cost per step; with
    the PRs' values the schedule itself changes, which each priority's latency shows."""
    names = (
        ("frozen", "frozen"),
        ("off", "make-room, switches off"),
        ("on", "make-room, the PRs' values"),
    )
    rows, output = [], {}
    for name, label in names:
        try:
            p0, p2 = (json.loads((batch / f"{name}-p{p}.json").read_text()) for p in (0, 2))
            running = json.loads((batch / f"{name}-running.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        output[name] = p0["output_throughput"] + p2["output_throughput"]
        rows.append(
            f"| {label} | {output[name]:,.0f} | "
            + " | ".join(
                f"{c['median_ttft_ms']:,.0f} / {c['p99_ttft_ms']:,.0f} | "
                f"{c['median_tpot_ms']:.2f} / {c['p99_tpot_ms']:.2f}"
                for c in (p0, p2)
            )
            + f" | {running['running_p50']:g} / {running['running_max']:g} |"
        )
    if not rows:
        return ["Not measured: no batch arm (runs/a100/batch)."]
    lines = [
        "| Engine | Output tokens/s | Priority 0: TTFT p50 / p99 (ms) | Priority 0: TPOT p50 / p99 (ms) | "
        "Priority 2: TTFT p50 / p99 (ms) | Priority 2: TPOT p50 / p99 (ms) | Running p50 / max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
    ]
    if "frozen" in output and len(output) > 1:
        lines += [
            "",
            "Output tokens a second at the batch, against the frozen engine's: "
            + "; ".join(
                f"{dict(names)[n]} {output[n] / output['frozen']:.3f}"
                for n in output
                if n != "frozen"
            )
            + ". With its switches off make-room schedules as vLLM does, so a ratio under 1.000 there is its code "
            "path's cost per step.",
        ]
    return lines


def e0_slopes(cells: dict[str, dict[str, float]]) -> dict[str, float]:
    """E0's per-token terms, in µs a prompt token, as least-squares slopes over its prompt lengths: the GPU hit's
    (the host's work on a prompt, tokenizing and hashing it, with no prefill), and the CPU restore's beyond it (the
    transfer)."""
    tokens = sorted(int(t) for t in cells["gpu"])
    thousands = [t / 1e3 for t in tokens]
    gpu = [cells["gpu"][str(t)] for t in tokens]
    beyond = [cells["cpu"][str(t)] - hit for t, hit in zip(tokens, gpu, strict=True)]
    return {
        "gpu_hit": statistics.linear_regression(thousands, gpu).slope,
        "transfer": statistics.linear_regression(thousands, beyond).slope,
    }


def forecast_e0_box(text: str) -> dict[str, dict[str, float]]:
    """The box's measured E0 cells (ms) as the forecast's E0 table gives them, by kind and prompt tokens."""
    section = text.split("## E0's cells", 1)[1].split("\n## ", 1)[0]
    cells: dict[str, dict[str, float]] = {"gpu": {}, "cpu": {}}
    for line in section.splitlines():
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(parts) == 7 and parts[0].replace(",", "").isdigit():
            tokens = parts[0].replace(",", "")
            cells["gpu"][tokens] = float(parts[1].replace(",", ""))
            cells["cpu"][tokens] = float(parts[3].replace(",", ""))
    return cells


def host_lines(record: dict[str, Any], e0: dict[str, Any] | None, forecast_text: str) -> list[str]:
    """The host around the card against the box's: the CPU at bench_host.py's Python workload and in E0's GPU hit,
    and the link, at the CPU tier's page sizes, against E0's restore and the forecast's link."""
    lines = []
    if python := record.get("python_ms"):
        ratio = BOX_HOST["python_ms"] / python["p50"]
        lines.append(
            f"- **The CPU.** {record.get('cpu', 'This host')} runs bench_host.py's Python workload in "
            f"{python['p50']:.1f} ms, the box's {BOX_HOST['cpu']} in {BOX_HOST['python_ms']:.1f}: "
            f"{ratio:.2f} times as fast."
        )
    if e0 is not None:
        ours, box = e0_slopes(e0), e0_slopes(forecast_e0_box(forecast_text))
        lines.append(
            f"- **E0's GPU hit grows with the prompt** by {ours['gpu_hit']:.2f} ms every 1,000 tokens on this host "
            f"and {box['gpu_hit']:.2f} on the box ({box['gpu_hit'] / ours['gpu_hit']:.2f} times): the host's work on "
            "a prompt (tokenizing it, hashing its blocks), with no prefill, which the forecast scaled with the card."
        )
        per_token = ours["transfer"]  # µs a token: ms every 1,000 tokens
        card = forecast_card(forecast_text)
        modeled = re.search(
            rf"\| {re.escape(card)} \| central \| [^\n]*? \| [\d.]+ ms \+ ([\d.]+) µs/token \|",
            forecast_text,
        )
        lines.append(
            f"- **E0's CPU restore beyond the GPU hit** takes {per_token:.2f} µs a token on this host, "
            f"{KV_BYTES_PER_TOKEN / per_token / 1e3:.2f} GB/s of KV, and {box['transfer']:.2f} on the box "
            f"({KV_BYTES_PER_TOKEN / box['transfer'] / 1e3:.2f} GB/s)"
            + (
                f"; the forecast took {float(modeled[1]):.2f} µs a token "
                f"({KV_BYTES_PER_TOKEN / float(modeled[1]) / 1e3:.1f} GB/s), the card's link as its "
                "datasheet gives it [modeled]."
                if modeled
                else "."
            )
        )
    link = {row["page_bytes"]: row for row in record.get("link", []) if row["direction"] == "h2d"}
    if link:
        gpu = record.get("gpu", {})
        gen, width = gpu.get("pcie.link.gen.current", "?"), gpu.get("pcie.link.width.current", "?")
        lines += [
            "",
            f"The link, pinned host memory to the card (PCIe Gen{gen} x{width} as the card negotiates it; "
            f"the box's {BOX_HOST['link']}), in GB/s:",
            "",
            "| Copies | This host | The box |",
            "|---|---:|---:|",
            *(
                f"| {'one copy' if page is None else f'pages of {page // 1024:,} KiB'} | "
                f"{row['gb_per_s']:.2f} | {BOX_HOST['h2d_gb_per_s'].get(page, float('nan')):.2f} |"
                for page, row in sorted(link.items(), key=lambda kv: kv[0] or 1 << 40)
            ),
        ]
        one = link.get(None)
        if one and str(gen).isdigit() and str(width).isdigit() and int(gen) in PCIE_LANE_GB_PER_S:
            rate = PCIE_LANE_GB_PER_S[int(gen)] * int(width)
            box_gen, box_width = BOX_HOST["lanes"]
            box_rate = PCIE_LANE_GB_PER_S[box_gen] * box_width
            box_one = BOX_HOST["h2d_gb_per_s"][None]
            lines += [
                "",
                f"One copy reached {one['gb_per_s']:.2f} GB/s, {one['gb_per_s'] / rate:.0%} of the {rate:.1f} GB/s "
                f"each way that the card's link carries; the box's reached {box_one:.2f}, "
                f"{box_one / box_rate:.0%} of its link's {box_rate:.1f}. A rate well under the card's own link is set "
                "by a narrower hop between the card and the host's memory.",
            ]
    elif record.get("link_error"):
        lines.append(f"- **The link** was not measured: {record['link_error']}")
    return lines


def report(host: Path, corpus: Path, binary: Path, out: Path, flatshare_bench: Path | None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    sim = calibration.Sim(binary, corpus, out)
    runs = host / "runs" / "a100"
    gpu = gpu_name(host)
    forecast = read_forecast(forecast_for(gpu))
    core, box = forecast.core, forecast.box
    lines = [
        f"# The {forecast.card} run against its forecast",
        "",
        f"From {host}. Every A100 number here is measured on the A100 unless it says modeled (the forecast, "
        f"{_shown(forecast.path)}) or names the box. {forecast_line(gpu, forecast)}",
        "",
        "## 1. The step model",
        "",
    ]
    export, e0 = runs / "calibration-s16", runs / "block1" / "frozen-e0" / "summary.json"
    steps_path = None
    measured_point = (export / "after.prom").exists() and e0.exists()
    if measured_point and not binary.exists():
        lines.append(
            f"Not computed: no flatshare-sim at {binary} (cargo build --release --locked -p flatshare-sim)."
        )
    elif measured_point:
        log = out / "calibration-s16-replay-log.csv"
        sim.replay_log("calibration-s16-replay-log", export)
        steps, fitted = fit(export, log, e0, out / "steps.json")
        steps_path = out / "steps.json"
        lines += step_model_lines(steps, forecast.costs)
        lines += ["", "The fit's own account:", "", "```", fitted.strip(), "```"]
    else:
        lines.append(
            "Not measured: no calibration point (runs/a100/calibration-s16) or no E0 (block 1's)."
        )
    lines += ["", "## 2. The calibration table", ""]
    if steps_path is not None:
        rows, _ = calibration_lines(
            sim, export, 16, ["--costs", str(e0), "--step-costs", str(steps_path)], "a100-s16"
        )
        lines += [*TABLE_HEAD, *rows]
    else:
        lines.append("Not computed." if measured_point else "Not measured.")
    lines += ["", "## 3. The probes", ""]
    if any((runs / block).is_dir() for block in ("block1", "block2", "block3")):
        lines += probe_lines(runs, forecast)
    else:
        lines.append("Not measured: no probe block (runs/a100/block1-3).")
    lines += ["", "## 4. E2 within the A100", ""]
    # The core's rounds (a100-r1 to -r5), and tuned vLLM's and the PRs' values at its step cap (a100-tuned-r1, -r2,
    # a100-prtuned-r1, -r2), which replay the samples of the core's rounds 1 and 2; the other optional arms' campaigns
    # begin a100-r too (a100-resubmit-r3 ...), and A' has its own.
    core_points = complete_points(host, "a100-r[0-9]*")
    points = (
        core_points
        + complete_points(host, "a100-tuned-r*")
        + complete_points(host, "a100-prtuned-r*")
    )
    if points:
        lines += e2_lines(points, out / "e2", flatshare_bench, core["running"], "the core")
    else:
        lines.append("Not measured: no E2 point (results/e2/a100-r1 to -r5).")
    lines += ["", "## 5. A': the box's pool, more agents", ""]
    aprime = complete_points(host, "a100-aprime-r*")
    if aprime:
        # The SLO's reference is voice alone on this card: the core's points of it.
        alone = [p for p in core_points if p.name == "w5-a1-voice"]
        lines += e2_lines(
            aprime,
            out / "e2-aprime",
            flatshare_bench,
            forecast.aprime["running"],
            "A', 32 sessions",
            reference=alone,
        )
    else:
        lines.append("Not measured: no A' point (results/e2/a100-aprime-r1, -r2).")
    lines += ["", "## 6. The scheduler at a batch of 256", ""]
    lines += batch_lines(runs / "batch")
    lines += ["", "## 7. The host around the card", ""]
    try:
        record = json.loads((runs / "host" / "host.json").read_text())
    except (OSError, json.JSONDecodeError):
        record = None
    if record is None:
        lines.append("Not measured: no host check (runs/a100/host/host.json, bench_host.py).")
    else:
        cells = json.loads(e0.read_text())["p50_ms"] if e0.exists() else None
        lines += host_lines(record, cells, forecast.path.read_text())
    lines += [
        "",
        "## 8. Across cards (different hardware)",
        "",
        "The core scenario's regime coordinates, the forecast's [modeled]: on the box's measured costs (W5's points "
        f"did not sample its running share) a saturated card, {box['busy']} busy, {box['stalled']} of agent requests "
        f"waiting for blocks and a running share of {box['running']}; the {forecast.card} at {core['busy']} busy, "
        f"{core['stalled']} stalled and {core['running']}, where §4 gives the running share it measured. Each card's "
        "distance to its own SLO (its voice alone + 250 ms) is in §4 for the A100 and in docs/BENCHMARKS.md for the "
        "box; the two are compared in these coordinates only.",
    ]
    path = out / "a100.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def calibration_check(exports: Path, corpus: Path, binary: Path, out: Path) -> list[str]:
    """report's calibration code on the box's W2 points with the box's E0, against calibration.py report's rows:
    the rows that differ (none when it reproduces them)."""
    out.mkdir(parents=True, exist_ok=True)
    expected = calibration.report(exports, corpus, binary, out / "box").splitlines()
    sim = calibration.Sim(binary, corpus, out)
    pairs = []
    for _, _, run in calibration.POINTS:
        sim.replay_log(f"{run}-replay-log", exports / run)
        pairs.append((exports / run / calibration.FILES[0], out / f"{run}-replay-log.csv"))
    steps, _ = calibration.fit(pairs, out / "steps.json")
    differ = []
    for label, sessions, run in calibration.POINTS:
        rows, _ = calibration_lines(sim, exports / run, sessions, ["--step-costs", str(steps)], run)
        ours = [rows[0].replace(f"| {sessions} sessions |", f"| {label} |"), *rows[1:]]
        for row in ours:
            if row not in expected:
                differ.append(row)
    return differ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysis.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("report", "calibration-check"):
        p = commands.add_parser(name)
        p.add_argument(
            "dir", type=Path, help="the host's records (report) or the box's exports (check)"
        )
        p.add_argument("--corpus", type=Path, required=True, help="the whole 32x corpus, weka-x32")
        p.add_argument(
            "--flatshare-sim", type=Path, required=True, help="flatshare-sim's release binary"
        )
        p.add_argument("--out", type=Path, required=True)
        if name == "report":
            p.add_argument(
                "--flatshare-bench",
                type=Path,
                help="flatshare-bench, for E2's report (the bench venv's)",
            )
    args = parser.parse_args(argv)
    if args.command == "report":
        print(report(args.dir, args.corpus, args.flatshare_sim, args.out, args.flatshare_bench))
        return 0
    differ = calibration_check(args.dir, args.corpus, args.flatshare_sim, args.out)
    for row in differ:
        print(f"analysis: not in calibration.py's table: {row}", file=sys.stderr)
    print(
        f"analysis: {'reproduces' if not differ else 'does not reproduce'} the box's calibration rows"
    )
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
