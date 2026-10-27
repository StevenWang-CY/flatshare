"""A point's results from its run directory (:mod:`.run`): its summary and its compact record.

:func:`summarize_point` reads one run directory of ``flatshare-bench e2 run``: the callers' summary, the
per-request records of the AIPerf replay, the gateway's request lines of the calls, the ``/metrics``
snapshots taken when the calls started and ended and, on an agent point, the engine's KV samples
through them, and the voice worker's journal of the calls. Every metric covers that window. A metric
that the gateway or the engine does not export is null, never zero, and a counter that went down
(its process restarted during the calls) is an error.

:func:`export_point` writes a point's compact record (``flatshare-bench e2 export``): the few small files
the report reads. :func:`compact` gives the record's run.json and summary.json from a run directory
or from a record alike, so the report gives the same tables from either.
"""

from __future__ import annotations

import re
import shutil
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from flatshare_bench.kv_metrics import Labels, metric_series
from flatshare_stats import metric_samples, summarize, v2v_samples

OUTCOMES = "flatshare_gateway_request_outcomes_total"
"""By class and outcome; ``shed`` is a request the admission point gave up for a stricter class."""
REFUSED = "flatshare_gateway_admission_refused_total"
RESUBMITS = "flatshare_gateway_admission_resubmits_total"
DISCARDED = "flatshare_gateway_admission_discarded_tokens_total"
"""The prompt tokens of the attempts preempt-resubmit dropped: at most the prefill they discarded."""
USAGE = "flatshare_gateway_usage_tokens_total"
"""Tokens from response usage, by class and kind, counted as each response ends."""
GATEWAY_PREFIX = "flatshare_gateway_"
"""The prefix of the gateway's metric names."""

TURNS_LOGGER = "flatshare_voice.turns"
SESSIONS_LOGGER = "flatshare_voice.events"
"""The voice worker's loggers (flatshare_voice.events): a line per turn, and one per session as it
ends ("session ended")."""
FORMER_PREFIXES = {"lh_": "flatshare_", "leaseholder_": "flatshare_"}
"""The prefixes that the records from before the project's rename (docs/GLOSSARY.md) give what is
named ``flatshare_`` now: the gateway's metrics (lh_gateway_), the voice worker's loggers
(leaseholder_voice.) and the engine variants' scheduler modules (lh_make_room_scheduler)."""
WATERFALL = (
    "transcript_ms",
    "end_of_turn_ms",
    "llm_ttft_ms",
    "llm_first_sentence_ms",
    "turn_to_first_sentence_ms",
    "tts_first_audio_ms",
    "e2e_ms",
)
"""The stage times of the worker's turn lines, in the order of a turn (its ``STAGES``)."""
GPU_TIME = "%Y/%m/%d %H:%M:%S.%f"
"""nvidia-smi's timestamp, which campaign.sh has it write in UTC."""
GOODPUT_WINDOW_S = 240.0
"""The background goodput's fixed window, beside the one over the whole calls: their first 240 s, the
same span for every point, where the calls' own length differs by configuration (276 to 435 s for
the discriminating scenario's headline configurations) and the agents' rate drifts over a run. A
point whose calls lasted less gives its goodput over its whole calls instead."""
REPLY_TIMEOUT_MS = 30_000.0
"""How long ``flatshare-voice bots`` waits for a reply (its ``reply_timeout``), after which the turn fails: a
reply that never came counts as a sample censored at this value, in every V2V percentile."""
KV_GAUGES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting")
"""What ``flatshare-bench e2 run`` samples from the engine's ``/metrics`` during an agent point's calls,
into metrics/engine-kv.csv, one line a sample: its time, then these."""
REALTIME_TIER_TOKENS = 2 * 1024
"""flatshare-sim's realtime tier, two calls of 1,024 tokens (crates/flatshare-sim, ``hardware::TOKENS_PER_CALL``):
the running share is the time the KV that running requests hold leaves less than this free."""


def summarize_point(
    run_dir: Path,
    start: float,
    end: float,
    *,
    agents: bool,
    warmup_turns: int = 0,
    calls: bool = True,
) -> dict[str, Any]:
    """The E2 metrics of the window ``start``..``end`` (epoch seconds) from a run directory. Raises
    ``ValueError`` when the point measured nothing to place on the frontier. The ``warmup_turns`` that
    open each call are measured by nothing: the callers leave them out, and so does the worker's
    account of the turns. The first bytes come from the gateway's request lines of the calls
    (gateway.jsonl), without which the point fails. A point without ``calls`` (``[voice] callers = 0``,
    the agents alone over a fixed window) has no callers' summary and no worker journal: its ``voice``
    and ``worker`` are None."""
    voice = None
    if calls:
        bots = orjson.loads((run_dir / "bots" / "summary.json").read_bytes())
        if bots["v2v_ms"].get("n", 0) <= bots["v2v_ms"].get("censored", 0):
            raise ValueError(
                f"the calls measured no voice-to-voice latency ({bots['errors']} failed)"
            )
        keys = ("turns", "errors", "v2v_ms", "barge_in_to_silence_ms", "barge_ins")
        # The replies' end, length and pauses, from callers that read the agent's state (from v1.6).
        voice = {key: bots[key] for key in keys} | (
            {"replies": bots["replies"]} if "replies" in bots else {}
        )
    elif not agents:
        raise ValueError("a point without calls measures the agents, and this one had none")
    metrics = run_dir / "metrics"
    records = run_dir / "aiperf" / "profile_export.jsonl"
    requests = gateway_requests(run_dir, start, end)
    if requests is None:
        raise ValueError(
            "the gateway logged no request line for the calls (gateway.jsonl): the first bytes come "
            'from them, with its log_format = "json"'
        )
    summary = {
        "window_s": round(end - start, 1),
        "voice": voice,
        "worker": worker(run_dir / "voice-worker.log", warmup_turns) if calls else None,
        "background": background(records, start, end) if agents else None,
        "gateway": gateway_counts(*_snapshots(metrics, "gateway"), end - start, requests),
        "engine": engine_counts(*_snapshots(metrics, "engine")),
    }
    # The engine's KV use while the calls ran, which e2 run samples on agent points.
    window = {"start": start, "end": end}
    summary["kv"] = kv_samples(metrics / "engine-kv.csv", window, point_engine(run_dir))
    return summary


def kv_values(exposition: str) -> tuple[float, float, float] | None:
    """An engine's KV use (0 to 1, the mean over its series) and its running and waiting requests
    (summed) in a ``/metrics`` exposition, as KV_GAUGES orders them; None when it lacks one (SGLang)."""
    found: dict[str, list[float]] = {name: [] for name in KV_GAUGES}
    for name, _, value in metric_samples(exposition, KV_GAUGES):
        found[name].append(value)
    if not all(found.values()):
        return None
    usage, running, waiting = (found[name] for name in KV_GAUGES)
    return statistics.fmean(usage), sum(running), sum(waiting)


def kv_samples(
    path: Path, window: Mapping[str, float], engine: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """An engine's KV samples of the calls (``path``, as KV_GAUGES describes it): how many, and the
    share of them in which the KV that running requests hold left less than REALTIME_TIER_TOKENS free
    (the measured running share, flatshare-sim's ``realtime_blocked_share``: vLLM's usage counts a cached
    block whose requests are gone as free, since a request that needs it may evict it), with the free
    blocks' median and least and the running and waiting requests' median and most. None without
    samples in the window, or without the pool's size in the engine's record (``engine``)."""
    cache = (engine or {}).get("cache_config") or {}
    blocks, block_size = cache.get("num_gpu_blocks"), cache.get("block_size")
    if not path.exists() or not blocks or not block_size:
        return None
    pool = blocks - 1  # vLLM's usage leaves out its null block (BlockPool.get_usage)
    rows = []
    for line in path.read_text(errors="replace").splitlines()[1:]:
        try:
            at, usage, running, waiting = (float(field) for field in line.split(","))
        except ValueError:
            continue  # a line cut short when the sampling stopped
        if window["start"] <= at <= window["end"]:
            rows.append(((1 - usage) * pool, running, waiting))
    if not rows:
        return None
    free, running, waiting = (sorted(column) for column in zip(*rows, strict=True))
    short = REALTIME_TIER_TOKENS / block_size
    return {
        "samples": len(rows),
        "running_share": round(sum(f < short for f in free) / len(rows), 4),
        "free_blocks": {"p50": round(statistics.median(free)), "min": round(free[0])},
        "running": {"p50": statistics.median(running), "max": running[-1]},
        "waiting": {"p50": statistics.median(waiting), "max": waiting[-1]},
    }


def _snapshots(metrics: Path, name: str) -> tuple[str, str]:
    """The ``/metrics`` of ``name`` (``gateway`` or ``engine``) as the calls started and ended."""
    start, end = ((metrics / f"{name}-{when}.prom").read_text() for when in ("start", "end"))
    return start, end


def measured_turns(turns: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The callers' measured turns (bots/turns.jsonl): all but the warm-up turns that open each call,
    whose lines say ``"warmup": true`` (``[voice] warmup_turns``)."""
    return [turn for turn in turns if not turn.get("warmup")]


def turn_samples(turns: Iterable[Mapping[str, Any]]) -> tuple[list[float], int]:
    """The voice-to-voice samples of the callers' measured turns (bots/turns.jsonl) and how many are
    censored: each turn's reply and, after a barge-in, the reply to the interruption, as ``flatshare-voice
    bots`` pools them; a reply that never came (a failed turn) stands at REPLY_TIMEOUT_MS."""
    return v2v_samples(measured_turns(turns), REPLY_TIMEOUT_MS)


def _json_lines(text: str) -> Iterator[dict[str, Any]]:
    """The JSON objects among the lines of a log."""
    for line in text.splitlines():
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def worker(journal: Path, warmup_turns: int = 0) -> dict[str, Any] | None:
    """The voice worker's own account of the calls, from its journal (voice-worker.log): each stage
    of the waterfall over the turns, and the lifecycle events its sessions sent, dropped and failed
    to deliver. None without a turn or session line in it; the events are None when no session sent
    any (``FLATSHARE_EVENTS=off``, or an LLM that bypasses the gateway). The first ``warmup_turns`` turn
    lines of each session are the replies to the call's warm-up turns, which have no interruption to
    add a reply, and are left out."""
    if not journal.exists():
        return None
    turns, sessions = worker_lines(journal.read_text(errors="replace"))
    if not turns and not sessions:
        return None
    turns = [t for t in turns if t.get("turn", warmup_turns) >= warmup_turns]
    counted = [s for s in sessions if "events_sent" in s]
    events = {
        kind: sum(s[f"events_{kind}"] for s in counted) for kind in ("sent", "dropped", "failed")
    }
    return {
        "sessions": len(sessions),
        "turns": len(turns),
        "waterfall": {
            stage: summarize(t[stage] for t in turns if stage in t) for stage in WATERFALL
        },
        "events": events if counted else None,
    }


def worker_lines(journal: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The voice worker's turn lines and session-end lines among its journal's JSON lines, under its
    loggers' names before the project's rename too."""
    turns, sessions = [], []
    for record in _json_lines(journal):
        logger = current_name(str(record.get("name")))
        if logger == TURNS_LOGGER:
            turns.append(record)
        elif logger == SESSIONS_LOGGER and record.get("message") == "session ended":
            sessions.append(record)
    return turns, sessions


def current_name(name: str) -> str:
    """A metric's, logger's or scheduler module's name, or a series of a metric, as it is now
    (FORMER_PREFIXES)."""
    for former, current in FORMER_PREFIXES.items():
        if name.startswith(former):
            return current + name.removeprefix(former)
    return name


def current_exposition(exposition: str) -> str:
    """A ``/metrics`` exposition with the gateway's metrics under their names now (FORMER_PREFIXES):
    its samples and its HELP and TYPE lines."""
    return re.sub(r"(?m)^(# (?:HELP|TYPE) )?lh_gateway_", rf"\g<1>{GATEWAY_PREFIX}", exposition)


def gateway_request_lines(journal: str) -> list[str]:
    """The gateway's "request" lines among its journal's lines: one JSON object per request that
    ended, with its class, status, outcome and times (``log_format = "json"``, as on cyan)."""
    return [line for line in journal.splitlines() if _request(line) is not None]


def logged_requests(journal: str) -> list[tuple[float, dict[str, Any]]]:
    """The gateway's request lines in its journal, each with when its request ended."""
    return [request for request in map(_request, journal.splitlines()) if request is not None]


def gateway_requests(run_dir: Path, start: float, end: float) -> list[dict[str, Any]] | None:
    """The requests the gateway logged as ending within ``start``..``end``, from the run's
    gateway.jsonl; None without one, or with one that holds no request (the calls make some)."""
    path = run_dir / "gateway.jsonl"
    if not path.exists():
        return None
    logged = [r for r in map(_request, path.read_text(errors="replace").splitlines()) if r]
    if not logged:
        return None
    return [record for ended, record in logged if start <= ended <= end]


def _request(line: str) -> tuple[float, dict[str, Any]] | None:
    """A request line of the gateway's, with when the request ended; None for any other line."""
    try:
        record = orjson.loads(line)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("message") != "request":
        return None
    try:
        return datetime.fromisoformat(record["timestamp"]).timestamp(), record
    except (KeyError, TypeError, ValueError):
        return None


def background(records: Path, start: float, end: float) -> dict[str, Any]:
    """The agent requests AIPerf completed during the window: the background goodput, which counts
    failed and cancelled requests as lost, and their time to first token; and the goodput over the
    window's first GOODPUT_WINDOW_S (``fixed_window``)."""
    ended = profiling_requests(records, start, end)
    ok = [r for r in ended if not r.get("error") and not r["metadata"].get("was_cancelled")]
    if not ok:
        raise ValueError(f"AIPerf completed no request during the calls ({len(ended)} failed)")
    usage = [r["metrics"] for r in ok if "usage_prompt_tokens" in r["metrics"]]
    prompt = sum(m["usage_prompt_tokens"]["value"] for m in usage)
    completion = sum(m.get("usage_completion_tokens", {}).get("value", 0) for m in usage)
    seconds = end - start
    ttft = [
        r["metrics"]["time_to_first_token"]["value"]
        for r in ok
        if "time_to_first_token" in r["metrics"]
    ]
    return {
        "requests": len(ok),
        "errors": len(ended) - len(ok),
        "without_usage": len(ok) - len(usage),
        "requests_per_s": round(len(ok) / seconds, 3),
        "prompt_tokens_per_s": round(prompt / seconds, 1),
        "completion_tokens_per_s": round(completion / seconds, 1),
        "tokens_per_s": round((prompt + completion) / seconds, 1),
        "ttft_ms": summarize(ttft),
        "itl_ms": agent_itl(ok),
        "fixed_window": fixed_window(ended, start, end),
    }


def agent_itl(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """The agent requests' inter-token latency in ms, AIPerf's per request (the time after the first
    token over the tokens after it), over the requests that streamed two tokens or more: the step time
    a decoding agent sees (P-1's per-step overhead at the point's batch)."""
    return summarize(
        r["metrics"]["inter_token_latency"]["value"]
        for r in records
        if "inter_token_latency" in r["metrics"]
    )


def profiling_requests(records: Path, start: float, end: float) -> list[dict[str, Any]]:
    """AIPerf's records (profile_export.jsonl) of the agent requests of its profiling phase that
    ended within ``start``..``end``."""
    if not records.exists():
        raise ValueError(f"AIPerf wrote no per-request records ({records})")
    ended = []
    for line in records.read_bytes().splitlines():
        record = orjson.loads(line)
        meta = record["metadata"]
        if meta.get("benchmark_phase") != "profiling":
            continue
        if start <= meta["request_end_ns"] / 1e9 <= end:
            ended.append(record)
    return ended


def fixed_window(
    ended: Iterable[Mapping[str, Any]], start: float, end: float
) -> dict[str, float | int]:
    """The background goodput over the first GOODPUT_WINDOW_S of the calls, counted as
    :func:`background` counts it over all of them, from the records of the requests that ended
    during the calls: its ``seconds``, less than GOODPUT_WINDOW_S only for calls that lasted less, the
    requests that completed in it, and their rate and tokens a second."""
    seconds = min(GOODPUT_WINDOW_S, round(end - start, 3))
    ok = [
        r
        for r in ended
        if r["metadata"]["request_end_ns"] / 1e9 <= start + seconds
        and not r.get("error")
        and not r["metadata"].get("was_cancelled")
    ]
    tokens = sum(
        r["metrics"].get(kind, {}).get("value", 0)
        for r in ok
        for kind in ("usage_prompt_tokens", "usage_completion_tokens")
    )
    return {
        "seconds": seconds,
        "requests": len(ok),
        "requests_per_s": round(len(ok) / seconds, 3),
        "tokens_per_s": round(tokens / seconds, 1),
    }


def gateway_counts(
    before: str, after: str, seconds: float, requests: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The gateway's first byte by class, exactly from ``requests``, the window's lines of its request
    log; what its admission point shed, refused and resubmitted, and its own count of the background
    tokens served over the window of ``seconds``; every counter that moved is listed under
    ``counters``. The admission counts are null for a gateway without an admission point, the
    background rate for one without usage counters. A gateway from before the project's rename named
    its metrics otherwise (FORMER_PREFIXES); they are counted under their names now."""
    then, now = (metric_series(current_exposition(text)) for text in (before, after))
    counters = {
        name: deltas(then, now, name, "gateway") or {}
        for name in sorted(now)
        if name.startswith(GATEWAY_PREFIX) and name.endswith("_total")
    }

    def total(name: str, match: Mapping[str, str]) -> int | None:
        if name not in counters:
            return None
        return round(sum(v for k, v in counters[name].items() if dict(k).items() >= match.items()))

    # A gateway with an admission point exports its series from its start, at zero; one without has
    # none, and its shed count, from a family every gateway has, is null rather than zero.
    admission = REFUSED in counters
    prompt = total(USAGE, {"class": "background", "kind": "prompt"})
    completion = total(USAGE, {"class": "background", "kind": "completion"})
    return {
        "realtime_first_byte_ms": logged_first_byte(requests, "realtime"),
        "background_first_byte_ms": logged_first_byte(requests, "background"),
        "shed": total(OUTCOMES, {"outcome": "shed"}) if admission else None,
        "refused": total(REFUSED, {}),
        "resubmitted": total(RESUBMITS, {}),
        "discarded_tokens": total(DISCARDED, {}),
        "background_tokens_per_s": (
            None
            if prompt is None or completion is None
            else round((prompt + completion) / seconds, 1)
        ),
        "counters": {
            _series(name, labels): round(value)
            for name, series in counters.items()
            for labels, value in series.items()
            if value
        },
    }


CPU_TIER = {
    "kv_offload_load_bytes": "vllm:kv_offload_load_bytes_total",
    "kv_offload_store_bytes": "vllm:kv_offload_store_bytes_total",
}
"""The CPU tier's own counts (vLLM's OffloadingConnector): the bytes of every load into the GPU pool and
of every store from it, a preempted request's resume included, which the prompt-token sources leave
out."""


def engine_counts(before: str, after: str) -> dict[str, Any]:
    """vLLM's preemptions, prompt tokens by source, refused requests and the CPU tier's loads and
    stores in bytes over the window. vLLM 0.32 counts a request's prompt tokens by source only at its
    first scheduling, so a preempted request's resume is in none of them; the CPU tier's counts have
    every load and store."""
    then, now = metric_series(before), metric_series(after)

    def by(label: str, name: str) -> dict[str, int] | None:
        series = deltas(then, now, name, "engine")
        if series is None:
            return None
        totals: dict[str, int] = {}
        for labels, value in series.items():
            key = dict(labels).get(label, "")
            totals[key] = totals.get(key, 0) + round(value)
        return totals

    def total(name: str) -> int | None:
        series = deltas(then, now, name, "engine")
        return None if series is None else round(sum(series.values()))

    return {
        "preemptions": total("vllm:num_preemptions_total"),
        "prompt_tokens_by_source": by("source", "vllm:prompt_tokens_by_source_total"),
        "refused": by("reason", "vllm:admission_rejections_total"),
        **{key: total(name) for key, name in CPU_TIER.items()},
    }


def deltas(
    before: Mapping[str, Mapping[Labels, float]],
    after: Mapping[str, Mapping[Labels, float]],
    name: str,
    source: str,
) -> dict[Labels, float] | None:
    """How much each series of counter ``name`` grew from ``before`` to ``after``, two expositions'
    :func:`~flatshare_bench.kv_metrics.metric_series`; None when ``after`` has no such counter. A
    counter that went down means its process restarted during the window, which voids it."""
    now = after.get(name)
    if not now:
        return None
    then = before.get(name, {})
    grown = {}
    for labels, value in now.items():
        grown[labels] = value - then.get(labels, 0.0)
        if grown[labels] < 0:
            raise ValueError(f"{name} went down during the calls: the {source} restarted")
    return grown


def logged_first_byte(requests: Iterable[Mapping[str, Any]], slo_class: str) -> dict[str, Any]:
    """Count, mean and p50/p95/p99 of a class's first byte at the gateway in ms, exactly from its
    request lines (``source`` "log"): the requests that had a first byte from the upstream, as its
    first-byte histogram counts them (a request refused or timed out before going upstream has an
    outcome and no first byte)."""
    first_bytes = [
        r["first_byte_ms"]
        for r in requests
        if r.get("class") == slo_class and "outcome" in r and "first_byte_ms" in r
    ]
    cell = summarize(first_bytes)
    return {key: cell[key] for key in ("n", "mean", "p50", "p95", "p99") if key in cell} | {
        "source": "log"
    }


def _series(name: str, labels: Labels) -> str:
    return name + ("{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else "")


def render(summary: Mapping[str, Any]) -> str:
    """One line of a point's headline numbers."""
    voice = summary["voice"]
    if voice is None:
        line = f"{summary['name']}: no calls, the agents alone for {summary['window_s']:g} s"
    else:
        v2v = "/".join(str(voice["v2v_ms"].get(p)) for p in ("p50", "p95", "p99"))
        line = f"{summary['name']}: V2V p50/p95/p99 {v2v} ms"
        line += f", {voice['errors']} of {voice['turns']} turns failed"
    if background := summary["background"]:
        ttft = "/".join(str(background["ttft_ms"].get(p)) for p in ("p50", "p99"))
        line += f"; background {background['requests_per_s']} requests/s"
        line += f", {background['tokens_per_s']:,.0f} tokens/s, agent TTFT p50/p99 {ttft} ms"
    return line


def gpu(path: Path, window: Mapping[str, float] | None) -> dict[str, Any]:
    """The GPU's median SM clock (MHz) and highest temperature (°C) during the calls, from the samples
    campaign.sh takes once a second on the GPU node (gpu.csv); None without them. Where the samples
    name the GPU, also its name (``gpu_name``). A file whose rows carry several GPUs' UUIDs (a host's
    other GPUs) counts the rows of the one its ``# gpu UUID`` line names, the point's, alone, and none
    where it names none; one from before the rows named their GPU with several rows a second (several
    GPUs) counts none: the GPUs are not mixed."""
    clocks: list[float] = []
    temperatures: list[float] = []
    names: list[str] = []
    for fields in _gpu_rows(path, window):
        if (clock := _number(fields, 1)) is not None:
            clocks.append(clock)
        if (temperature := _number(fields, 3)) is not None:
            temperatures.append(temperature)
        if len(fields) > 6 and fields[6]:
            names.append(fields[6])
    kept: dict[str, Any] = {
        "gpu_sm_clock_p50_mhz": round(statistics.median(clocks)) if clocks else None,
        "gpu_temperature_max_c": round(max(temperatures)) if temperatures else None,
    }
    # The GPU's name, nvidia-smi's, where the samples carry it: campaign.sh's from 2026-10-25 on.
    if names:
        kept["gpu_name"] = statistics.mode(names)
    return kept


def _gpu_rows(path: Path, window: Mapping[str, float] | None) -> Iterator[list[str]]:
    """The fields of the point's GPU's samples in gpu.csv taken during the calls, as :func:`gpu`
    counts them."""
    if not path.exists() or not window:
        return
    text = path.read_text(errors="replace").splitlines()
    lent = next((line.split()[2] for line in text if line.startswith("# gpu ")), None)
    rows = [[field.strip() for field in line.split(",")] for line in text]
    uuids = {row[7] for row in rows if len(row) > 7}
    stamps = [row[0] for row in rows if len(row) > 1 and len(row) <= 7]
    if len(stamps) != len(set(stamps)):
        return  # several GPUs a second, unnamed
    for fields in rows:
        if len(uuids) > 1 and (len(fields) <= 7 or fields[7] != lent):
            continue
        try:
            at = datetime.strptime(fields[0], GPU_TIME).replace(tzinfo=UTC).timestamp()
        except ValueError:
            continue  # a line cut short when the sampling stopped
        if window["start"] <= at <= window["end"]:
            yield fields


def _number(fields: Sequence[str], index: int) -> float | None:
    """A field of an nvidia-smi line without its unit ("2805 MHz"); None for "[N/A]" and the like."""
    try:
        return float(fields[index].split()[0])
    except (IndexError, ValueError):
        return None


def measured_sum(*counts: int | None) -> int | None:
    """The sum of the counts that were measured; None when none was."""
    measured = [c for c in counts if c is not None]
    return sum(measured) if measured else None


# The compact record

RECORD = ("run.json", "arm.toml", "bots/summary.json", "bots/turns.jsonl")
"""What a point's compact record (``flatshare-bench e2 export``) keeps of its run directory's files, its
run.json as :func:`compact` gives it, beside its summary.json and a reduced engine.json. Left out:
the gateway's request log and the GPU's samples, whose numbers the record's summary keeps; AIPerf's
exports, the metric snapshots, the worker's journal and the tools' logs; and the callers' run.json,
which names LiveKit's API key."""

VLLM_ENGINE = {
    "model_config": ("model", "revision", "max_model_len", "dtype", "quantization"),
    "cache_config": (
        "block_size",
        "enable_prefix_caching",
        "cache_dtype",
        "kv_cache_memory_bytes",
        "num_gpu_blocks",
    ),
    "scheduler_config": (
        "policy",
        "max_num_batched_tokens",
        "long_prefill_token_threshold",
        "max_num_seqs",
        "async_scheduling",
        "watermark",
        "max_num_queued_tokens",
        "scheduler_cls",
    ),
    "kv_transfer_config": ("kv_connector", "kv_connector_extra_config"),
}
"""What a record keeps of a vLLM engine's configuration, beside all of its additional_config: the
settings flatshare-bench e2 run checks the arms' flags against, the scheduler class among them (a variant's
module, or None for vLLM's own), and what it takes to serve the same engine again (the model and its
revision, the context, dtype and quantization, the KV pool, the CPU tier)."""
VLLM_ENGINE_REPORTED = {"kernel_config": ("linear_backend",)}
"""What a record keeps of a vLLM engine's configuration where its engine.json has it, which those
before 2026-10-25 do not: the linear kernel the point asked for (``--linear-backend``; ``auto``
lets vLLM choose by the GPU, Marlin for this checkpoint on sm80 and sm89)."""
SGLANG_ENGINE = (
    "model_path",
    "revision",
    "context_length",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "page_size",
    "disable_radix_cache",
    "mem_fraction_static",
    "max_total_tokens",
    "max_total_num_tokens",
    "schedule_policy",
    "chunked_prefill_size",
    "max_running_requests",
    "enable_priority_scheduling",
    "schedule_low_priority_values_first",
    "priority_scheduling_preemption_threshold",
    "enable_hierarchical_cache",
    "hicache_ratio",
    "hicache_write_policy",
)
"""The same of an SGLang engine's server arguments."""
_SNAPSHOT = re.compile(r"models--(?P<name>[^/]+)/snapshots/(?P<revision>[0-9a-f]+)")
"""A model in the Hugging Face cache: models--ORG--NAME/snapshots/REVISION."""
_HOME = re.compile(r"/(?:home|Users)/[^/\s]+|/root(?=/)")


def compact(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The run.json and summary.json of the compact record of the point in ``run_dir``, a run
    directory or a record (whose own it returns); None for a point that did not complete. The
    run.json is the run's, its ``versions`` naming the GPU node's deployed revision as null where the
    run did not record it (``gpu_source``). The summary takes the first bytes of both classes exactly
    from the request log, the GPU's median SM clock and highest temperature from gpu.csv (``gpu``),
    the background goodput over the calls' first GOODPUT_WINDOW_S from AIPerf's records, and the CPU
    tier's loads and stores from the engine's metric snapshots: what a record keeps of the files it
    leaves out. The report reads every point through it, so a run directory and its record give the
    same tables."""
    run_file = run_dir / "run.json"
    run = orjson.loads(run_file.read_bytes()) if run_file.exists() else {}
    if run.get("status") != "complete":
        return None
    if "versions" in run and "gpu_source" not in run["versions"]:
        # Points measured before e2 run recorded it: beside cyan's revision, as e2 run writes it.
        versions = list(run["versions"].items())
        at = 1 + next((i for i, (key, _) in enumerate(versions) if key == "source"), -1)
        run["versions"] = dict([*versions[:at], ("gpu_source", None), *versions[at:]])
    summary = orjson.loads((run_dir / "summary.json").read_bytes())
    window = run.get("window")
    requests = gateway_requests(run_dir, window["start"], window["end"]) if window else None
    if requests is not None:
        for slo_class in ("realtime", "background"):
            summary["gateway"][f"{slo_class}_first_byte_ms"] = logged_first_byte(
                requests, slo_class
            )

    if (samples := run_dir / "gpu.csv").exists():
        columns = gpu(samples, window)
        summary["gpu"] = {column.removeprefix("gpu_"): value for column, value in columns.items()}
    # The goodput over the calls' first GOODPUT_WINDOW_S, from AIPerf's records, and the CPU tier's
    # loads and stores, from the engine's /metrics snapshots: points measured before the summary
    # kept them gain them, and the summary's other numbers stay as they are.
    records = run_dir / "aiperf" / "profile_export.jsonl"
    if summary.get("background") and window and records.exists():
        ended = profiling_requests(records, window["start"], window["end"])
        summary["background"]["fixed_window"] = fixed_window(ended, window["start"], window["end"])
        ok = [r for r in ended if not r.get("error") and not r["metadata"].get("was_cancelled")]
        summary["background"]["itl_ms"] = agent_itl(ok)
    metrics = run_dir / "metrics"
    if all((metrics / f"engine-{when}.prom").exists() for when in ("start", "end")):
        counts = engine_counts(*_snapshots(metrics, "engine"))
        summary["engine"] |= {key: counts[key] for key in CPU_TIER}
    return run, summary


def point_engine(run_dir: Path) -> dict[str, Any] | None:
    """The engine of a point as its record keeps it (:func:`engine_settings`), from a run directory's
    engine.json or a record's; None without one."""
    path = run_dir / "engine.json"
    return engine_settings(orjson.loads(path.read_bytes())) if path.exists() else None


def kv_token_bytes(engine: Mapping[str, Any] | None) -> float | None:
    """The bytes of KV a token takes in an engine's pool, which the CPU tier moves as they are: its KV
    cache's bytes over its tokens, from vLLM's cache_config (``kv_cache_memory_bytes`` over
    ``num_gpu_blocks`` times ``block_size``). The bytes left over by the whole blocks that fit, less
    than one block's, make it high by under one part in ``num_gpu_blocks`` (0.02% on the frozen box:
    147,492 bytes against Qwen3-4B's 147,456). None without them (SGLang)."""
    cache = (engine or {}).get("cache_config") or {}
    try:
        return cache["kv_cache_memory_bytes"] / (cache["num_gpu_blocks"] * cache["block_size"])
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def export_point(run_dir: Path, out: Path) -> Path | None:
    """Write the compact record of the point in ``run_dir`` into ``out``, at CAMPAIGN/POINT as the
    run directory sits in its campaign, and return its directory; None for a point that did not
    complete. Its run.json and summary.json are :func:`compact`'s, the run.json with every path under
    a home directory written from ``~``. Its engine.json keeps only what says which engine the point
    ran on (:func:`engine_settings`)."""
    if (point := compact(run_dir)) is None:
        return None
    run, summary = point
    record = out / run_dir.absolute().parent.name / run_dir.absolute().name
    record.mkdir(parents=True, exist_ok=True)
    for name in RECORD:
        if name != "run.json" and (run_dir / name).exists():
            (record / name).parent.mkdir(
                exist_ok=True
            )  # bots/, which a point without calls has none of
            shutil.copyfile(run_dir / name, record / name)
    # As e2 run writes them, so that a run.json the record keeps as it was stays the same bytes. The
    # gateway's configuration names files under the account's home directory (its cache-salt secret).
    (record / "run.json").write_bytes(orjson.dumps(_home(run), option=orjson.OPT_INDENT_2) + b"\n")
    (record / "summary.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2) + b"\n")
    if (engine := run_dir / "engine.json").exists():
        settings = engine_settings(orjson.loads(engine.read_bytes()))
        (record / "engine.json").write_bytes(
            orjson.dumps(settings, option=orjson.OPT_INDENT_2) + b"\n"
        )
    return record


def engine_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """An engine's configuration (engine.json, from its /server_info) reduced to VLLM_ENGINE or
    SGLANG_ENGINE, with the model named by its Hugging Face id and revision rather than its path in
    the cache, and every path under a home directory from ``~``."""
    if "model_config" not in config:  # SGLang's server arguments
        return _home({key: config.get(key) for key in SGLANG_ENGINE})
    settings: dict[str, Any] = {
        section: None if config.get(section) is None else {k: config[section].get(k) for k in keys}
        for section, keys in VLLM_ENGINE.items()
    }
    settings |= {
        section: {k: config[section].get(k) for k in keys}
        for section, keys in VLLM_ENGINE_REPORTED.items()
        if isinstance(config.get(section), Mapping)
    }
    settings["additional_config"] = config.get("additional_config")
    model = settings["model_config"]
    if model:
        model |= dict(zip(("model", "revision"), model_id(model), strict=True))
    return _home(settings)


def model_id(model_config: Mapping[str, Any]) -> tuple[Any, Any]:
    """A vLLM engine's model and revision as its configuration names them, by the Hugging Face id and
    revision rather than its path in the cache, where vLLM's offline mode put that path in place of
    the id (``models--ORG--NAME/snapshots/REVISION``)."""
    model, revision = model_config.get("model"), model_config.get("revision")
    if cached := _SNAPSHOT.search(str(model)):
        return cached["name"].replace("--", "/", 1), cached["revision"]
    return model, revision


def engine_model(engine: Mapping[str, Any] | None) -> str | None:
    """The model a point's engine served, from its record (:func:`engine_settings`): vLLM's or SGLang's."""
    if not engine:
        return None
    if "model_config" in engine:
        return (engine["model_config"] or {}).get("model")
    return engine.get("model_path")


def _home(value: Any) -> Any:
    """``value`` with every path under a home directory written from ``~``."""
    if isinstance(value, str):
        return _HOME.sub("~", value)
    if isinstance(value, Mapping):
        return {key: _home(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_home(item) for item in value]
    return value
