"""A soak's summary (:func:`summarize`), per hour of its window and over all of it, from its directory or from the
compact record :func:`export` writes of it, which give the same numbers (:func:`load` reads either).

Each period has:

- **calls**: the attempts due in it, ``attempted`` (accepted, refused and failed: what the service was offered),
  ``accepted``, ``refused`` and ``refused_by`` reason, ``failed`` and ``failed_by`` reason, ``skipped`` (the load
  generator's own bound) and ``abandoned``, and how the accepted calls ``ended``;
- **turns** that ended in it: the measured turns and those that failed, and their **replies**: each turn's reply and
  the reply to an interruption, a reply that never came censored at the callers' 30 s reply timeout, as E2 counts
  them: their count, p50, pooled p95 and p99, and, with an SLO (voice alone's p95 with the same calls, plus 250 ms;
  docs/SLO.md), how many and what share were over it; and the replies heard to their end
  (:func:`flatshare_stats.replies`): those ended by the agent's state, those with a pause over 500 ms;
- **the gateway's own counts**, from its counters' increases between scrapes (a counter that went down counts from zero
  again, as Prometheus' ``increase`` does): the realtime service level, ``good`` over ``valid``; the realtime and the
  background requests offered, admitted and ended by outcome, ``completed``, ``cancelled`` and ``failed`` (any other
  outcome); the background tokens a second over the time the counts span, less the guard's pauses; the requests in
  flight; each node's engine state in the scrapes and, with ``[[nodes]]``, its health and placements;
- **the engines**, per node: preemptions, KV use (mean and highest), running and waiting requests, the tokens a second
  and the prefix cache's hits;
- **errors and restarts**: the turns and calls that failed, the gateway's upstream errors, the user manager's lines
  about cyan's units (starts, stops, failures, scheduled restarts), counters that went down, scrapes that failed, the
  tools' runs that ended before the window did, and the guard's checks and pauses.

An attempt belongs to the period it was due in, a turn to the one it ended in, a counter's increase to the one its
later scrape is in; the last hour also has the calls that played their turns after the window's end, and the scrapes
up to the soak's last.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flatshare_bench.soak.scrape import is_failure, parse_key
from flatshare_stats import percentiles, replies, v2v_samples

REPLY_TIMEOUT_MS = 30_000.0
"""The callers' reply timeout (``flatshare-voice``'s BotsConfig): a reply that never came counts at it."""
SLO_MARGIN_MS = 250.0
"""What agent load may add to voice alone's p95 (docs/SLO.md)."""
HOUR_S = 3600.0
"""A period of the summary: an hour of the window, from its start."""
TURN_COLUMNS = (
    "run",
    "caller",
    "turn",
    "at",
    "warmup",
    "clip",
    "v2v_ms",
    "spoke_during_turn",
    "barge_in",
    "barge_in_clip",
    "interrupted_speech",
    "barge_in_to_silence_ms",
    "v2v_after_barge_in_ms",
    "end_by",
    "reply_audio_ms",
    "reply_pauses_ms",
    "interrupted_reply_pauses_ms",
    "received",
    "error",
    "extra",
)
"""turns.csv's columns, a turn line's keys (``flatshare-voice arrivals``' turns.jsonl); ``received`` and the pauses
are JSON, and ``extra`` holds any other key, so that the CSV gives the lines back as they were."""
_INTEGERS = frozenset({"run", "caller", "turn"})
_FLOATS = frozenset(
    {"at", "v2v_ms", "barge_in_to_silence_ms", "v2v_after_barge_in_ms", "reply_audio_ms"}
)
_BOOLS = frozenset({"warmup", "spoke_during_turn", "barge_in", "interrupted_speech"})
_JSON = frozenset({"reply_pauses_ms", "interrupted_reply_pauses_ms", "received", "extra"})
_HOME = re.compile(r"/(?:home|Users)/([^/\s\"']+)|/root(?=/)")
"""A path under a home directory, which a committed record writes from ``~`` (as e2 export does); its group is the
login the directory gives away."""
LOGIN_AS = "USER"
"""What a committed record writes in place of a login its home paths gave away, wherever else it appears."""
RECORD_FILES = (
    "soak.json",
    "summary.json",
    "summary.md",
    "attempts.jsonl",
    "turns.csv",
    "series.csv",
    "events.jsonl",
    "units.jsonl",
    "guard.csv",
)
"""A soak's compact record (:func:`export`): no raw scrape, AIPerf export, request log or credential."""


@dataclass
class SoakData:
    """What a soak recorded, from its directory or its compact record."""

    record: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    scrapes: list[dict[str, Any]] = field(default_factory=list)
    """Each scrape: ``at``, ``gateway`` (its series, None when it failed), ``engines`` (each node's, or None),
    ``worker`` and ``errors``."""
    events: list[dict[str, Any]] = field(default_factory=list)
    units: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """The JSON objects of a JSON-lines file, a line cut short by a soak that died left out."""
    if not path.exists():
        return []
    found = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            found.append(entry)
    return found


def load(path: Path) -> SoakData:
    """A soak's directory (``soak run``'s) or its compact record (``soak export``'s)."""
    record = json.loads((path / "soak.json").read_text())
    if (path / "series.csv").exists():
        return SoakData(
            record,
            _jsonl(path / "attempts.jsonl"),
            read_turns_csv((path / "turns.csv").read_text())
            if (path / "turns.csv").exists()
            else [],
            read_series_csv((path / "series.csv").read_text()),
            _jsonl(path / "events.jsonl"),
            _jsonl(path / "units.jsonl"),
            read_guard_csv((path / "guard.csv").read_text())
            if (path / "guard.csv").exists()
            else [],
        )
    attempts, turns = [], []
    for run in sorted((path / "calls").glob("run-*")):
        attempts += _jsonl(run / "attempts.jsonl")
        turns += _jsonl(run / "turns.jsonl")
    return SoakData(
        record,
        attempts,
        turns,
        _jsonl(path / "series.jsonl"),
        _jsonl(path / "events.jsonl"),
        _jsonl(path / "units.jsonl"),
        _jsonl(path / "guard.jsonl"),
    )


# The compact record's tables


def write_turns_csv(turns: Iterable[Mapping[str, Any]]) -> str:
    """turns.csv (TURN_COLUMNS), a row a turn line."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(TURN_COLUMNS)
    for turn in turns:
        extra = {k: v for k, v in turn.items() if k not in TURN_COLUMNS}
        cells = []
        for column in TURN_COLUMNS:
            if column == "extra":
                cells.append(json.dumps(extra) if extra else "")
            elif column not in turn:
                cells.append("")  # the line has no such key
            elif (value := turn[column]) is None:
                cells.append("null")
            elif column in _JSON:
                cells.append(json.dumps(value))
            elif column in _BOOLS:
                cells.append("true" if value else "false")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(_number(value))
            else:
                cells.append(str(value))
        writer.writerow(cells)
    return out.getvalue()


def read_turns_csv(text: str) -> list[dict[str, Any]]:
    """The turn lines of a turns.csv, as :func:`write_turns_csv` wrote them."""
    turns = []
    for row in csv.DictReader(io.StringIO(text)):
        turn: dict[str, Any] = {}
        for column in TURN_COLUMNS:
            cell = row.get(column, "")
            if cell == "":
                continue
            if column == "extra":
                turn |= json.loads(cell)
            elif cell == "null":
                turn[column] = None
            elif column in _JSON:
                turn[column] = json.loads(cell)
            elif column in _BOOLS:
                turn[column] = cell == "true"
            elif column in _INTEGERS:
                turn[column] = int(cell)
            elif column in _FLOATS:
                turn[column] = float(cell)
            else:
                turn[column] = cell
        turns.append(turn)
    return turns


def _number(value: float | int) -> str:
    """A number as the record writes it, read back exactly: a whole one without a point, any other as Python writes
    it (the shortest that reads back the same)."""
    if isinstance(value, int) or (value.is_integer() and abs(value) < 1e15):
        return str(int(value))
    return repr(value)


def _sources(scrape: Mapping[str, Any]) -> dict[str, Mapping[str, float] | None]:
    """A scrape's series by source: ``gateway``, ``engine/NODE`` and ``worker``."""
    found: dict[str, Mapping[str, float] | None] = {"gateway": scrape.get("gateway")}
    for node, series in (scrape.get("engines") or {}).items():
        found[f"engine/{node}"] = series
    found["worker"] = scrape.get("worker")
    return found


def write_series_csv(scrapes: Sequence[Mapping[str, Any]]) -> str:
    """series.csv: a row a scrape, its time, number, label and the errors of the sources that failed (JSON), then a
    column a series that any scrape had, ``SOURCE|SERIES``, empty where that scrape had none."""
    columns: dict[str, None] = {}
    nodes: dict[str, None] = {}
    for scrape in scrapes:
        for node in scrape.get("engines") or {}:
            nodes[node] = None
        for source, series in _sources(scrape).items():
            for key in series or {}:
                columns[f"{source}|{key}"] = None
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["at", "scrape", "label", "nodes", "errors", *columns])
    for scrape in scrapes:
        sources = _sources(scrape)
        cells = [
            _number(scrape["at"]),
            str(scrape.get("scrape", "")),
            scrape.get("label") or "",
            json.dumps(sorted(scrape.get("engines") or {})),
            json.dumps(scrape.get("errors") or {}) if scrape.get("errors") else "",
        ]
        for column in columns:
            source, _, key = column.partition("|")
            value = (sources.get(source) or {}).get(key)
            cells.append("" if value is None else _number(value))
        writer.writerow(cells)
    return out.getvalue()


def read_series_csv(text: str) -> list[dict[str, Any]]:
    """The scrapes of a series.csv, as :func:`write_series_csv` wrote them."""
    scrapes = []
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        return []
    columns = header[5:]
    for row in reader:
        errors = json.loads(row[4]) if row[4] else {}
        nodes = json.loads(row[3]) if row[3] else []
        scrape: dict[str, Any] = {"at": float(row[0])}
        if row[1]:
            scrape["scrape"] = int(row[1])
        if row[2]:
            scrape["label"] = row[2]
        found: dict[str, dict[str, float]] = defaultdict(dict)
        for column, cell in zip(columns, row[5:], strict=False):
            if cell != "":
                source, _, key = column.partition("|")
                found[source][key] = float(cell)
        scrape["gateway"] = None if "gateway" in errors else found.get("gateway", {})
        scrape["engines"] = {
            node: None if f"engine/{node}" in errors else found.get(f"engine/{node}", {})
            for node in nodes
        }
        scrape["worker"] = found.get("worker")
        scrape["errors"] = errors
        scrapes.append(scrape)
    return scrapes


def write_guard_csv(checks: Iterable[Mapping[str, Any]]) -> str:
    """guard.csv: a row a check of a node's GPU, its time, node, verdict and the other processes it found."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["at", "node", "verdict", "others"])
    for check in checks:
        others = sum(1 for p in check.get("processes") or [] if p.get("whose") == "other")
        writer.writerow([_number(check["at"]), check["node"], check["verdict"], others])
    return out.getvalue()


def read_guard_csv(text: str) -> list[dict[str, Any]]:
    return [
        {
            "at": float(row["at"]),
            "node": row["node"],
            "verdict": row["verdict"],
            "others": int(row["others"]),
        }
        for row in csv.DictReader(io.StringIO(text))
    ]


# The summary


def _labels(key: str) -> tuple[str, dict[str, str]]:
    return parse_key(key)


class _Counts:
    """The increases of every counter between consecutive successful scrapes of each source, each with the times of
    its two scrapes, and the gauges' values at every scrape."""

    def __init__(self, scrapes: Sequence[Mapping[str, Any]]) -> None:
        self.steps: dict[str, list[tuple[float, float, dict[str, float]]]] = defaultdict(list)
        self.values: dict[str, list[tuple[float, Mapping[str, float]]]] = defaultdict(list)
        last: dict[str, tuple[float, Mapping[str, float]]] = {}
        for scrape in sorted(scrapes, key=lambda s: s["at"]):
            at = scrape["at"]
            for source, series in _sources(scrape).items():
                if series is None:
                    continue
                self.values[source].append((at, series))
                if source in last:
                    then_at, then = last[source]
                    self.steps[source].append((then_at, at, _increase(then, series)))
                last[source] = (at, series)

    def increase(self, source: str, a: float, b: float) -> tuple[dict[str, float], float]:
        """Each counter's increase over the steps that end in (a, b], and the seconds those steps span."""
        total: dict[str, float] = defaultdict(float)
        span = 0.0
        for start, end, step in self.steps.get(source, []):
            if a < end <= b:
                span += end - start
                for key, value in step.items():
                    total[key] += value
        return dict(total), span

    def gauges(self, source: str, a: float, b: float) -> list[Mapping[str, float]]:
        return [series for at, series in self.values.get(source, []) if a < at <= b]


def _increase(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    """How much each counter (``*_total``) grew from one scrape to the next: by the difference, or, for one that went
    down (its process restarted), by its new value, as from zero; a series new to the later scrape grew from zero."""
    grown = {}
    for key, value in after.items():
        if "_total" not in key.partition("{")[0]:
            continue
        then = before.get(key, 0.0)
        grown[key] = value - then if value >= then else value
    return grown


def _sum(series: Mapping[str, float], metric: str, **match: str) -> float:
    """The sum of a metric's series whose labels hold ``match``."""
    total = 0.0
    for key, value in series.items():
        name, labels = _labels(key)
        if name == metric and all(labels.get(k) == v for k, v in match.items()):
            total += value
    return total


def _by(series: Mapping[str, float], metric: str, label: str, **match: str) -> dict[str, int]:
    """A metric's series that hold ``match``, added up by the value of ``label``, those that are zero left out."""
    found: dict[str, float] = defaultdict(float)
    for key, value in series.items():
        name, labels = _labels(key)
        if name == metric and all(labels.get(k) == v for k, v in match.items()):
            found[labels.get(label, "")] += value
    return {k: round(v) for k, v in sorted(found.items()) if round(v)}


def _gauge(
    rows: Sequence[Mapping[str, float]], metric: str, mean_of: bool = False, **match: str
) -> dict[str, Any]:
    """A gauge over a period's scrapes: its mean and highest, each scrape's value the sum (or ``mean_of``: the mean)
    of the metric's series that hold ``match``."""
    values = []
    for series in rows:
        found = [
            value
            for key, value in series.items()
            if (parsed := _labels(key))[0] == metric
            and all(parsed[1].get(k) == v for k, v in match.items())
        ]
        if found:
            values.append(statistics.fmean(found) if mean_of else sum(found))
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "max": round(max(values), 4),
    }


def _rate(count: float, seconds: float) -> float | None:
    return round(count / seconds, 1) if seconds > 0 else None


def _share(part: float, whole: float) -> float | None:
    return round(part / whole, 4) if whole else None


def _paused(pauses: Sequence[Mapping[str, Any]], a: float, b: float, until: float) -> float:
    """The seconds of the guard's pauses within [a, b); a pause that did not end lasts until ``until``."""
    total = 0.0
    for pause in pauses:
        start, end = pause["start"], pause.get("end") or until
        total += max(0.0, min(end, b) - max(start, a))
    return total


def _calls(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(a.get("outcome") for a in attempts)
    by = {
        outcome: dict(
            sorted(
                Counter(
                    str(a.get("reason")) for a in attempts if a.get("outcome") == outcome
                ).items()
            )
        )
        for outcome in ("refused", "failed")
    }
    accepted = [a for a in attempts if a.get("outcome") == "accepted"]
    return {
        "attempted": outcomes["accepted"] + outcomes["refused"] + outcomes["failed"],
        "accepted": outcomes["accepted"],
        "refused": outcomes["refused"],
        "refused_by": by["refused"],
        "failed": outcomes["failed"],
        "failed_by": by["failed"],
        "skipped": outcomes["skipped"],
        "abandoned": outcomes["abandoned"],
        "ended": {
            e: sum(1 for a in accepted if a.get("ended") == e)
            for e in ("completed", "dropped", "cut")
        },
        "refused_share": _share(
            outcomes["refused"], outcomes["accepted"] + outcomes["refused"] + outcomes["failed"]
        ),
    }


def _turns(turns: Sequence[Mapping[str, Any]], threshold: float | None) -> dict[str, Any]:
    measured = [t for t in turns if not t.get("warmup")]
    samples, censored = v2v_samples(measured, REPLY_TIMEOUT_MS)
    cell = percentiles(samples, censored)
    found: dict[str, Any] = {
        "measured": len(measured),
        "failed": sum(1 for t in measured if "error" in t),
        "warmup": len(turns) - len(measured),
        "replies": cell,
    }
    if threshold is not None:
        over = sum(1 for sample in samples if sample > threshold)
        cell |= {"over_slo": over, "over_slo_share": _share(over, len(samples))}
    heard = replies(measured)
    found["heard"] = {
        key: heard.get(key)
        for key in (
            "n",
            "ended_by_state",
            "with_a_pause",
            "with_a_pause_over_500_ms",
            "longest_pause_ms",
        )
    } | ({"concealed_share": heard["concealed_share"]} if "concealed_share" in heard else {})
    found["heard"]["pause_over_500_ms_share"] = _share(
        heard["with_a_pause_over_500_ms"], heard["n"]
    )
    return found


def _requests(counts: Mapping[str, float], slo_class: str) -> dict[str, Any]:
    outcomes = _by(
        counts, "flatshare_gateway_requests_terminal_total", "outcome", **{"class": slo_class}
    )
    ended = sum(outcomes.values())
    completed, cancelled = outcomes.get("completed", 0), outcomes.get("cancelled", 0)
    return {
        "offered": round(
            _sum(counts, "flatshare_gateway_requests_offered_total", **{"class": slo_class})
        ),
        "admitted": round(
            _sum(counts, "flatshare_gateway_requests_admitted_total", **{"class": slo_class})
        ),
        "ended": ended,
        "completed": completed,
        "cancelled": cancelled,
        "failed": ended - completed - cancelled,
        "outcomes": outcomes,
    }


def _gateway(
    counts: _Counts, a: float, b: float, paused_s: float, nodes: Sequence[str]
) -> dict[str, Any]:
    grown, span = counts.increase("gateway", a, b)
    rows = counts.gauges("gateway", a, b)
    active = max(0.0, span - paused_s)
    valid = _sum(grown, "flatshare_gateway_sli_valid_total", **{"class": "realtime"})
    good = _sum(grown, "flatshare_gateway_sli_good_total", **{"class": "realtime"})
    background = _requests(grown, "background")
    prompt = _sum(
        grown, "flatshare_gateway_usage_tokens_total", **{"class": "background", "kind": "prompt"}
    )
    completion = _sum(
        grown,
        "flatshare_gateway_usage_tokens_total",
        **{"class": "background", "kind": "completion"},
    )
    background |= {
        "prompt_tokens_per_s": _rate(prompt, active),
        "completion_tokens_per_s": _rate(completion, active),
        "tokens_per_s": _rate(prompt + completion, active),
    }
    found: dict[str, Any] = {
        "span_s": round(span, 1),
        "realtime_sli": {
            "valid": round(valid),
            "good": round(good),
            "good_share": _share(good, valid),
        },
        "realtime": _requests(grown, "realtime"),
        "background": background,
        "in_flight": {
            slo_class: _gauge(rows, "flatshare_gateway_in_flight", **{"class": slo_class})
            for slo_class in ("realtime", "background")
        },
        "upstream_errors": _by(grown, "flatshare_gateway_upstream_errors_total", "kind"),
        "error_events": _by(grown, "flatshare_gateway_error_events_total", "type"),
        "engine_state": {},
    }
    with_nodes = any(
        _labels(key)[1].get("node") for row in rows for key in row if "engine_state" in key
    )
    for node in nodes:
        match = {"node": node} if with_nodes else {}
        states: Counter[str] = Counter()
        for row in rows:
            for key, value in row.items():
                name, labels = _labels(key)
                if (
                    name == "flatshare_gateway_engine_state"
                    and value == 1
                    and all(labels.get(k) == v for k, v in match.items())
                ):
                    states[labels.get("state", "")] += 1
        found["engine_state"][node] = dict(sorted(states.items()))
    if with_nodes:
        found["nodes"] = {
            node: {
                "healthy_scrapes": sum(
                    1
                    for row in rows
                    if row.get(f'flatshare_gateway_node_healthy{{node="{node}"}}') == 1
                ),
                "covering_scrapes": sum(
                    1
                    for row in rows
                    if row.get(f'flatshare_gateway_node_covering{{node="{node}"}}') == 1
                ),
                "placements": _by(grown, "flatshare_gateway_placements_total", "reason", node=node),
                "retries": round(_sum(grown, "flatshare_gateway_node_retries_total", node=node)),
            }
            for node in nodes
        }
        found["placements_refused"] = _by(
            grown, "flatshare_gateway_placements_refused_total", "reason"
        )
    return found


def _engines(
    counts: _Counts, a: float, b: float, paused_s: float, nodes: Sequence[str]
) -> dict[str, Any]:
    found = {}
    for node in nodes:
        source = f"engine/{node}"
        grown, span = counts.increase(source, a, b)
        rows = counts.gauges(source, a, b)
        active = max(0.0, span - paused_s)
        queries = _sum(grown, "vllm:prefix_cache_queries_total")
        found[node] = {
            "scrapes": len(rows),
            "preemptions": round(_sum(grown, "vllm:num_preemptions_total")),
            "kv_cache_usage": _gauge(rows, "vllm:kv_cache_usage_perc", mean_of=True),
            "running": _gauge(rows, "vllm:num_requests_running"),
            "waiting": _gauge(rows, "vllm:num_requests_waiting"),
            "prompt_tokens_per_s": _rate(_sum(grown, "vllm:prompt_tokens_total"), active),
            "generation_tokens_per_s": _rate(_sum(grown, "vllm:generation_tokens_total"), active),
            "prefix_cache_hit_share": _share(_sum(grown, "vllm:prefix_cache_hits_total"), queries),
            "finished": _by(grown, "vllm:request_success_total", "finished_reason"),
        }
    return found


def _worker(counts: _Counts, a: float, b: float) -> dict[str, Any]:
    """The calls the voice worker carried, by its status at each scrape, and the job requests it refused, a count it
    keeps from its start: its increase, from zero again after a restart."""
    rows = counts.gauges("worker", a, b)
    refused = 0.0
    values = [(at, row.get("refused")) for at, row in counts.values.get("worker", [])]
    for (_, then), (at, now) in itertools.pairwise(values):
        if a < at <= b and then is not None and now is not None:
            refused += now - then if now >= then else now
    return {"active_calls": _gauge(rows, "active_calls"), "refused": round(refused)}


def _restarts(
    units: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_unit: dict[str, Counter[str]] = defaultdict(Counter)
    for line in units:
        message = str(line.get("message", ""))
        unit = str(line.get("unit"))
        if message.startswith("Started "):
            by_unit[unit]["started"] += 1
        elif message.startswith("Stopped "):
            by_unit[unit]["stopped"] += 1
        if "Failed with result" in message:
            by_unit[unit]["failed"] += 1
        elif is_failure(message):  # its main process exited with a status other than success
            by_unit[unit]["exited"] += 1
        if "Scheduled restart job" in message:
            by_unit[unit]["restart_scheduled"] += 1
        if message.startswith("Reloaded "):
            by_unit[unit]["reloaded"] += 1
    kinds = Counter(str(e.get("event")) for e in events)
    resets = Counter(str(e.get("source")) for e in events if e.get("event") == "counter_reset")
    failed_scrapes = Counter(
        str(e.get("source")) for e in events if e.get("event") == "scrape_failed"
    )
    return {
        "units": {unit: dict(sorted(c.items())) for unit, c in sorted(by_unit.items())},
        "unit_changes": kinds["unit_changed"],
        "counter_resets": dict(sorted(resets.items())),
        "scrapes_failed": dict(sorted(failed_scrapes.items())),
        "calls_runs_ended": kinds["calls_ended"],
        "agents_runs_ended": kinds["agents_ended"],
    }


def _guard(
    checks: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]], paused_s: float
) -> dict[str, Any]:
    verdicts = Counter(str(c.get("verdict")) for c in checks)
    kinds = Counter(str(e.get("event")) for e in events)
    return {
        "checks": dict(sorted(verdicts.items())),
        "taken": kinds["guard_taken"],
        "blind": kinds["guard_blind"],
        "pauses": kinds["load_paused"],
        "paused_s": round(paused_s, 1),
    }


def _period(
    data: SoakData,
    counts: _Counts,
    a: float,
    b: float,
    *,
    b_counts: float,
    threshold: float | None,
    until: float,
) -> dict[str, Any]:
    """One period, [a, b) for what happened in it, (a, b_counts] for the counters' increases."""
    record = data.record
    nodes = (record.get("serving") or {}).get("nodes") or sorted(
        {node for s in data.scrapes for node in (s.get("engines") or {})}
    )
    paused_s = _paused(record.get("pauses") or [], a, min(b, until), until)
    attempts = [x for x in data.attempts if a <= x.get("due", x.get("started", -math.inf)) < b]
    turns = [t for t in data.turns if a <= t.get("at", -math.inf) < b]
    events = [e for e in data.events if a <= e.get("at", -math.inf) < b]
    units = [u for u in data.units if a <= u.get("at", -math.inf) < b]
    checks = [c for c in data.checks if a <= c.get("at", -math.inf) < b]
    scrapes = [s for s in data.scrapes if a < s["at"] <= b_counts]
    turned = _turns(turns, threshold)
    calls = _calls(attempts)
    return {
        "start": round(a, 3),
        "seconds": round(max(0.0, min(b, until) - a), 1),
        "calls": calls,
        "turns": turned,
        "gateway": _gateway(counts, a, b_counts, paused_s, nodes),
        "engines": _engines(counts, a, b_counts, paused_s, nodes),
        "worker": _worker(counts, a, b_counts),
        "errors": {
            "turns_failed": turned["failed"],
            "calls_failed": calls["failed"],
            "calls_dropped": calls["ended"]["dropped"],
        },
        "restarts": _restarts(units, events),
        "guard": _guard(checks, events, paused_s),
        "scrapes": {
            "n": len(scrapes),
            "failed": dict(
                sorted(Counter(src for s in scrapes for src in (s.get("errors") or {})).items())
            ),
        },
    }


def summarize(
    data: SoakData, slo_reference_ms: float | None = None, slo_margin_ms: float = SLO_MARGIN_MS
) -> dict[str, Any]:
    """The summary of a soak (the module's docstring): ``overall`` and ``hours``; with ``slo_reference_ms`` (voice
    alone's V2V p95 with the same calls), the replies over it plus ``slo_margin_ms``."""
    record = data.record
    window = record.get("window")
    threshold = None if slo_reference_ms is None else slo_reference_ms + slo_margin_ms
    summary: dict[str, Any] = {
        "kind": "soak-summary",
        "status": record.get("status"),
        "ended_by": record.get("ended_by"),
        "slo": None
        if threshold is None
        else {
            "reference_p95_ms": slo_reference_ms,
            "margin_ms": slo_margin_ms,
            "threshold_ms": threshold,
        },
        "window": None,
        "overall": None,
        "hours": [],
    }
    if not window:
        return summary
    start, planned = window["start"], window["end"]
    last = max((s["at"] for s in data.scrapes), default=start)
    until = max(start, min(planned, _ended_at(data, planned)))
    hours = max(1, math.ceil((until - start) / HOUR_S - 1e-9))
    counts = _Counts(data.scrapes)
    summary["window"] = {
        "start": start,
        "end": round(until, 3),
        "planned_end": planned,
        "hours": round((until - start) / 3600, 3),
        "last_scrape": round(last, 3),
    }
    for hour in range(hours):
        a = start + HOUR_S * hour
        final = hour == hours - 1
        b = math.inf if final else a + HOUR_S
        summary["hours"].append(
            {"hour": hour}
            | _period(
                data,
                counts,
                a,
                b,
                b_counts=math.inf if final else b,
                threshold=threshold,
                until=until,
            )
        )
    summary["overall"] = _period(
        data, counts, start, math.inf, b_counts=math.inf, threshold=threshold, until=until
    )
    return summary


def _ended_at(data: SoakData, planned: float) -> float:
    """When the soak's window ended: when the soak said so (its ``window_ended`` event, earlier than planned where a
    signal ended it), or, for a soak that runs or died without saying, its last scrape or its end, if earlier."""
    for event in data.events:
        if event.get("event") == "window_ended":
            return min(planned, float(event["at"]))
    seen = [float(t) for t in (data.record.get("finished"),) if t is not None]
    seen += [max(s["at"] for s in data.scrapes)] if data.scrapes else []
    return min(planned, max(seen)) if seen else planned


# Rendering


def _ms(value: Any) -> str:
    return "-" if value is None else f"{value:,.0f}"


def render_hour(period: Mapping[str, Any]) -> str:
    """One line of a period's headline numbers."""
    calls, turns, gateway = period["calls"], period["turns"], period["gateway"]
    replies_ = turns["replies"]
    refused = ", ".join(f"{k} {v}" for k, v in calls["refused_by"].items())
    line = f"calls {calls['attempted']} (accepted {calls['accepted']}, refused {calls['refused']}"
    line += f"{': ' + refused if refused else ''}, failed {calls['failed']})"
    line += f"; turns {turns['measured']} ({turns['failed']} failed)"
    line += f"; replies p95 {_ms(replies_.get('p95'))} ms"
    if "over_slo_share" in replies_ and replies_["over_slo_share"] is not None:
        line += f", {replies_['over_slo_share']:.1%} over the SLO"
    line += f"; pauses over 500 ms {turns['heard']['with_a_pause_over_500_ms']}"
    sli = gateway["realtime_sli"]["good_share"]
    line += f"; realtime SLI {'-' if sli is None else f'{sli:.3f}'}"
    background = gateway["background"]
    rate = background["tokens_per_s"]
    line += f"; background {'-' if rate is None else f'{rate:,.0f}'} tok/s"
    line += f" ({background['completed']} completed, {background['failed']} failed)"
    if period["guard"]["paused_s"]:
        line += f"; paused {period['guard']['paused_s']:.0f} s"
    return line


def render_markdown(summary: Mapping[str, Any]) -> str:
    """summary.md: the soak's hours and its whole window, a row each."""
    lines = ["# Soak summary", ""]
    window = summary.get("window")
    if not window:
        return "\n".join([*lines, "The soak's window never started.", ""])
    slo = summary.get("slo")
    lines.append(
        f"{window['hours']:g} h from {_utc(window['start'])}, ended by {summary.get('ended_by')}; "
        + (
            f"the SLO {slo['threshold_ms']:,.0f} ms (voice alone's p95 {slo['reference_p95_ms']:,.0f} ms + "
            f"{slo['margin_ms']:g} ms)."
            if slo
            else "no SLO reference given."
        )
    )
    lines += [
        "",
        "| Hour | Calls | Accepted | Refused | Failed | Turns | Failed turns | p95 (ms) | Over the SLO | Pauses > 500 ms "
        "| Realtime SLI | Background tok/s | Background done / failed | Restarts | Paused (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, period in [
        *((str(h["hour"]), h) for h in summary["hours"]),
        ("all", summary["overall"]),
    ]:
        calls, turns, gateway = period["calls"], period["turns"], period["gateway"]
        replies_ = turns["replies"]
        over = replies_.get("over_slo_share")
        sli = gateway["realtime_sli"]["good_share"]
        background = gateway["background"]
        restarts = sum(
            c.get("failed", 0) + c.get("restart_scheduled", 0)
            for c in period["restarts"]["units"].values()
        )
        cells = [
            label,
            calls["attempted"],
            calls["accepted"],
            calls["refused"],
            calls["failed"],
            turns["measured"],
            turns["failed"],
            _ms(replies_.get("p95")),
            "-" if over is None else f"{over:.1%}",
            turns["heard"]["with_a_pause_over_500_ms"],
            "-" if sli is None else f"{sli:.3f}",
            "-" if background["tokens_per_s"] is None else f"{background['tokens_per_s']:,.0f}",
            f"{background['completed']} / {background['failed']}",
            restarts,
            f"{period['guard']['paused_s']:.0f}",
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join([*lines, ""])


def _utc(at: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(at))


def write_summary(out: Path, summary: Mapping[str, Any]) -> None:
    """summary.json and summary.md into ``out``, each replaced whole."""
    for name, text in (
        ("summary.json", json.dumps(summary, indent=1) + "\n"),
        ("summary.md", render_markdown(summary)),
    ):
        staged = out / f".{name}.new"
        staged.write_text(text)
        staged.replace(out / name)


# The compact record


def logins(*values: Any) -> set[str]:
    """The logins that the home directories named anywhere in ``values`` give away (``/home/LOGIN``)."""
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            found.update(match[1] for match in _HOME.finditer(value) if match[1])
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for value in values:
        visit(value)
    return found


def _login_words(names: Iterable[str]) -> list[re.Pattern[str]]:
    """Each login as a word of its own: not a part of a longer name, a host's or a file's."""
    return [
        re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-])")
        for name in sorted(names, key=len, reverse=True)
    ]


def scrub(value: Any, names: Iterable[str] = ()) -> Any:
    """``value`` with every path under a home directory written from ``~``, and each login of ``names`` (the ones
    its home paths gave away, :func:`logins`) as LOGIN_AS wherever else it appears."""
    words = _login_words(names)

    def scrubbed(item: Any) -> Any:
        if isinstance(item, str):
            text = _HOME.sub("~", item)
            for word in words:
                text = word.sub(LOGIN_AS, text)
            return text
        if isinstance(item, Mapping):
            return {key: scrubbed(inner) for key, inner in item.items()}
        if isinstance(item, list):
            return [scrubbed(inner) for inner in item]
        return item

    return scrubbed(value)


def export(
    soak: Path,
    out: Path,
    slo_reference_ms: float | None = None,
    slo_margin_ms: float = SLO_MARGIN_MS,
) -> Path:
    """Write the compact record of the soak in ``soak`` into ``out`` (RECORD_FILES): its record, the attempts, the
    turns as a CSV, the kept series as a CSV (a column a series), the events, the user manager's lines about cyan's
    units, the guard's checks, and the summary, which :func:`load` reads back to the same numbers. No raw scrape,
    AIPerf export, request log or credential, every path under a home directory from ``~``, and the login such a
    path names as LOGIN_AS wherever else it appears."""
    data = load(soak)
    out.mkdir(parents=True, exist_ok=True)
    # The logins the record's home paths give away, which it carries nowhere else either.
    names = logins(data.record, data.attempts, data.turns, data.events, data.units)
    record = scrub(data.record, names)
    files = {
        "soak.json": json.dumps(record, indent=1) + "\n",
        "attempts.jsonl": "".join(json.dumps(scrub(a, names)) + "\n" for a in data.attempts),
        "turns.csv": write_turns_csv(scrub(data.turns, names)),
        "series.csv": write_series_csv(data.scrapes),
        "events.jsonl": "".join(json.dumps(scrub(e, names)) + "\n" for e in data.events),
        "units.jsonl": "".join(json.dumps(scrub(u, names)) + "\n" for u in data.units),
        "guard.csv": write_guard_csv(data.checks),
    }
    for name, text in files.items():
        (out / name).write_text(text)
    reference = slo_reference_ms
    if reference is None:
        reference = (data.record.get("options") or {}).get("slo_reference_ms")
    write_summary(out, summarize(load(out), reference, slo_margin_ms))
    words = _login_words(names)
    for name in RECORD_FILES:
        text = (out / name).read_text()
        if _HOME.search(text) or any(word.search(text) for word in words):
            raise ValueError(
                f"{out / name} names a path under a home directory, or its login: the record is not one to commit"
            )
    return out
