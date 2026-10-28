"""The soak's scrapes (:class:`Scraper`): what the service says of itself, every minute, asked as a production
scraper asks it and nothing more.

- **The gateway**: its ``/metrics`` on its socket, which asks for no key: what it was offered, admitted and ended by
  outcome, its service level by class, the tokens it relayed, its requests in flight, its engines' states and, with
  ``[[nodes]]``, every node's health, sessions and placements (:data:`GATEWAY_KEPT`); and ``/nodes`` with nodes.
- **The engines**: each one's ``/metrics`` through the gateway (``/metrics/engine``, ``?node=NAME`` with nodes), as
  Prometheus scrapes them: preemptions, KV use, running and waiting requests, tokens, the prefix cache and the CPU
  tier (:data:`ENGINE_KEPT`). Never ``/server_info`` or any development route: a soak runs on a production box.
- **The voice worker**: its status file (the calls it carries, whether it admits, its refusals).
- **cyan's flatshare units**: their state (``systemctl --user show``), and the user manager's journal lines about them
  since the last scrape, their starts, stops, restarts and failures, a failure's with the unit's own last lines.

Every scrape is kept whole, gzipped by the soak's hour (``scrapes/HOUR.gz``: each source's text after a line
``# soak-scrape AT SOURCE STATUS``), and the series the summary reads go to ``series.jsonl``, a line a scrape, each
series by its name and labels; vLLM's ``model_name`` label is left out, since it holds the model's path under the home
directory. A counter that went down (its process restarted) is an event, and the summary counts such a series from
zero again, as Prometheus' ``increase`` does.
"""

from __future__ import annotations

import gzip
import json
import math
import subprocess
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from flatshare_bench.e2.arm import E2Error
from flatshare_bench.e2.cyan import Systemd
from flatshare_bench.transport import client
from flatshare_stats import metric_samples

GATEWAY_KEPT = (
    "flatshare_gateway_requests_total",
    "flatshare_gateway_requests_offered_total",
    "flatshare_gateway_requests_admitted_total",
    "flatshare_gateway_requests_terminal_total",
    "flatshare_gateway_request_outcomes_total",
    "flatshare_gateway_sli_valid_total",
    "flatshare_gateway_sli_good_total",
    "flatshare_gateway_usage_tokens_total",
    "flatshare_gateway_in_flight",
    "flatshare_gateway_streams_active",
    "flatshare_gateway_buffered_body_bytes",
    "flatshare_gateway_predecessor_in_flight",
    "flatshare_gateway_upstream_errors_total",
    "flatshare_gateway_error_events_total",
    "flatshare_gateway_first_byte_seconds_sum",
    "flatshare_gateway_first_byte_seconds_count",
    "flatshare_gateway_engine_state",
    "flatshare_gateway_node_ready",
    "flatshare_gateway_canary_probes_total",
    "flatshare_gateway_admission_admitted_total",
    "flatshare_gateway_admission_waiting",
    "flatshare_gateway_admission_refused_total",
    "flatshare_gateway_admission_resubmits_total",
    "flatshare_gateway_admission_deferred_total",
    "flatshare_gateway_admission_degraded_total",
    "flatshare_gateway_events_total",
    "flatshare_gateway_node_healthy",
    "flatshare_gateway_node_component_healthy",
    "flatshare_gateway_node_sessions",
    "flatshare_gateway_node_in_flight",
    "flatshare_gateway_node_covering",
    "flatshare_gateway_node_failover_waiting",
    "flatshare_gateway_placements_total",
    "flatshare_gateway_placements_refused_total",
    "flatshare_gateway_node_retries_total",
)
"""The gateway's series a soak keeps from every scrape (crates/flatshare-gateway/metrics.txt and metrics-nodes.txt
name them all; its histograms' buckets stay in the raw scrapes)."""
ENGINE_KEPT = (
    "vllm:num_preemptions_total",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_by_source_total",
    "vllm:request_success_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_store_bytes_total",
)
"""vLLM 0.32's series a soak keeps from every scrape of an engine (ops/metrics/vllm-0.32.0.txt)."""
DROPPED_LABELS = frozenset({"model_name"})
"""Labels a kept series leaves out: vLLM names its model by its path in the Hugging Face cache, under the home
directory."""
WORKER_KEPT = ("active_calls", "max_calls", "admitting", "refused")
"""What a soak keeps of the voice worker's status file at every scrape, as series ``worker|NAME``."""
SCRAPE_TIMEOUT_S = 10.0
"""The gateway relays an engine's /metrics within 10 s, and a scrape waits no longer for any source."""
FAILURE_LINES = 20
"""A failed unit's own last lines that the record keeps beside the manager's line of its failure."""


def series_key(name: str, labels: Mapping[str, str]) -> str:
    """A series as the record names it: its metric, then its labels by name (DROPPED_LABELS left out), each value
    escaped as the text format escapes it."""
    kept = sorted((k, v) for k, v in labels.items() if k not in DROPPED_LABELS)
    if not kept:
        return name
    escaped = (v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") for _, v in kept)
    return (
        name + "{" + ",".join(f'{k}="{v}"' for (k, _), v in zip(kept, escaped, strict=True)) + "}"
    )


def parse_key(key: str) -> tuple[str, dict[str, str]]:
    """A series key's metric and labels (:func:`series_key`'s)."""
    name, brace, rest = key.partition("{")
    if not brace:
        return name, {}
    labels = {}
    for _, found, _ in metric_samples(f"x{{{rest.removesuffix('}')}}} 0"):
        labels = found
    return name, labels


def kept_series(exposition: str, names: Collection[str]) -> dict[str, float]:
    """The samples of the metrics ``names`` in an exposition, by series key; two samples that differ only in a dropped
    label are added up."""
    found: dict[str, float] = {}
    for name, labels, value in metric_samples(exposition, tuple(names)):
        if math.isfinite(value):
            key = series_key(name, labels)
            found[key] = found.get(key, 0.0) + value
    return found


def worker_series(status: Mapping[str, Any] | None) -> dict[str, float] | None:
    """The worker's status file as series (WORKER_KEPT); None without it."""
    if status is None:
        return None
    found: dict[str, float] = {}
    for key in WORKER_KEPT:
        value = status.get(key)
        if isinstance(value, (bool, int, float)):  # admitting is a bool: 1 or 0
            found[key] = float(value)
    return found


class SoakSystemd(Systemd):
    """cyan's user units as a soak reads them, beside what e2 run reads (:class:`~flatshare_bench.e2.cyan.Systemd`):
    every flatshare unit's state, and the user manager's journal lines about them; the tests use a fake."""

    def units(self) -> dict[str, dict[str, str]]:
        """Each flatshare unit the user manager knows, with its state."""
        listing = self("list-units", "flatshare-*", "--all", "--plain", "--no-legend", "--no-pager")
        names = [line.split()[0] for line in listing.splitlines() if line.strip()]
        if not names:
            return {}
        shown = self("show", *names, "--property=Id,ActiveState,SubState,NRestarts,MainPID,Result")
        return parse_show(shown)

    def manager_journal(self, units: Sequence[str], since: float, until: float) -> str:
        """The user manager's journal lines about ``units`` from ``since`` to ``until``, as JSON (USER_UNIT is the
        manager's field for the unit a line is about; a unit's own output carries _SYSTEMD_USER_UNIT)."""
        if not units:
            return ""
        command = ["journalctl", "--user", "--since", f"@{since:.6f}", "--until", f"@{until:.6f}"]
        command += ["-o", "json", "--no-pager", *(f"USER_UNIT={unit}" for unit in units)]
        return subprocess.run(command, capture_output=True, text=True).stdout

    def unit_tail(self, unit: str, since: float, until: float, lines: int = FAILURE_LINES) -> str:
        """A unit's own last ``lines`` lines from ``since`` to ``until``."""
        command = ["journalctl", "--user", f"_SYSTEMD_USER_UNIT={unit}", "--since", f"@{since:.6f}"]
        command += ["--until", f"@{until:.6f}", "-o", "cat", "-n", str(lines * 5), "--no-pager"]
        return subprocess.run(command, capture_output=True, text=True).stdout


def parse_show(text: str) -> dict[str, dict[str, str]]:
    """``systemctl show UNIT...``'s blocks, one a unit and a blank line between, by unit."""
    found: dict[str, dict[str, str]] = {}
    for block in text.strip().split("\n\n"):
        fields = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if unit := fields.pop("Id", None):
            found[unit] = fields
    return found


def manager_lines(text: str, hidden: Sequence[str] = ()) -> list[dict[str, Any]]:
    """The user manager's journal lines (``journalctl -o json``) about flatshare units: when, which unit and what
    it said, with each of ``hidden`` written as ``<hidden>``."""
    found = []
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        unit = entry.get("USER_UNIT") or entry.get("UNIT")
        message = entry.get("MESSAGE")
        if (
            not isinstance(unit, str)
            or not unit.startswith("flatshare-")
            or not isinstance(message, str)
        ):
            continue
        try:
            at = int(entry["__REALTIME_TIMESTAMP"]) / 1e6
        except (KeyError, TypeError, ValueError):
            continue
        found.append({"at": round(at, 6), "unit": unit, "message": hide(message, hidden)})
    return found


def is_failure(message: str) -> bool:
    """Whether a manager's line says its unit failed: it failed with a result, or its main process exited with a
    status other than success."""
    if "Failed with result" in message:
        return True
    return "Main process exited" in message and "status=0/SUCCESS" not in message


def own_lines(text: str, hidden: Sequence[str] = (), keep: int = FAILURE_LINES) -> list[str]:
    """A unit's own last lines before its failure, without the gateway's request lines (a line a request)."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or (
            '"message":"request"' in stripped or '"message": "request"' in stripped
        ):
            continue
        lines.append(hide(stripped, hidden))
    return lines[-keep:]


def hide(text: str, hidden: Sequence[str]) -> str:
    for secret in hidden:
        if secret:
            text = text.replace(secret, "<hidden>")
    return text


@dataclass(frozen=True)
class Source:
    """One thing a scrape asks: its name in the record, the address and path it asks."""

    name: str
    address: str
    path: str
    params: Mapping[str, str] | None = None


class RawScrapes:
    """Every scrape whole, gzipped by the soak's hour from ``start``: ``HOUR.gz`` in ``directory``, each source's text
    after a line ``# soak-scrape AT SOURCE STATUS``, appended as a gzip member of its own, so that a soak that stops
    leaves every member before it whole."""

    HEADER = "# soak-scrape "

    def __init__(self, directory: Path, start: float) -> None:
        self.directory = directory
        self.start = start
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, at: float, texts: Mapping[str, tuple[str, str]]) -> None:
        """Append one scrape: each source's status (``ok``, or the error) and its text."""
        hour = max(0, math.floor((at - self.start) / 3600))
        with gzip.open(self.directory / f"{hour:03d}.gz", "at", encoding="utf-8") as out:
            for source, (status, text) in texts.items():
                out.write(
                    f"{self.HEADER}{at:.3f} {source} {status.split()[0] if status else 'ok'}\n"
                )
                out.write(text if text.endswith("\n") or not text else text + "\n")


def read_raw(directory: Path) -> Iterator[tuple[float, str, str, str]]:
    """The raw scrapes of a soak, in order: each source's time, name, status and text."""
    for path in sorted(directory.glob("*.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as raw:
            header: tuple[float, str, str] | None = None
            lines: list[str] = []
            for line in raw:
                if line.startswith(RawScrapes.HEADER):
                    if header is not None:
                        yield (*header, "".join(lines))
                    at, source, status = line.removeprefix(RawScrapes.HEADER).split()
                    header, lines = (float(at), source, status), []
                else:
                    lines.append(line)
            if header is not None:
                yield (*header, "".join(lines))


class Scraper:
    """A soak's scrapes into ``out``: of the gateway at ``gateway`` (its socket), the engines of ``nodes`` through it
    (``with_nodes``: the gateway has ``[[nodes]]``, and each engine is asked by name), the worker's status by
    ``read_worker``, and cyan's units through ``systemd``. ``hidden`` are strings no journal line the record keeps may
    carry (the tunnels' targets)."""

    def __init__(
        self,
        out: Path,
        gateway: str,
        nodes: Sequence[str],
        with_nodes: bool,
        read_worker: Any,
        systemd: SoakSystemd,
        start: float,
        hidden: Sequence[str] = (),
    ) -> None:
        self.out = out
        self.gateway = gateway
        self.nodes = list(nodes)
        self.with_nodes = with_nodes
        self.read_worker = read_worker
        self.systemd = systemd
        self.hidden = tuple(hidden)
        self.raw = RawScrapes(out / "scrapes", start)
        self.count = 0
        self.last: dict[str, dict[str, float]] = {}
        """Each source's last series that were read, against which a counter that went down is told."""
        self.units: dict[str, list[Any]] = {}
        """Each unit's state at the last scrape: ActiveState, SubState, NRestarts, MainPID."""
        self.journal_until = start
        self.errors: dict[str, str] = {}
        """Each source whose last scrape failed, with why."""

    def sources(self) -> list[Source]:
        found = [Source("gateway", self.gateway, "/metrics")]
        for node in self.nodes:
            params = {"node": node} if self.with_nodes else None
            found.append(Source(f"engine/{node}", self.gateway, "/metrics/engine", params))
        if self.with_nodes:
            found.append(Source("nodes", self.gateway, "/nodes"))
        return found

    def scrape(
        self, at: float, label: str | None = None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """One scrape at ``at``: its line of series.jsonl, written, and the events it found (a source that fails or
        answers again, a counter that went down, a unit whose state changed)."""
        texts: dict[str, tuple[str, str]] = {}
        line: dict[str, Any] = {"at": round(at, 3), "scrape": self.count}
        if label:
            line["label"] = label
        line |= {"gateway": None, "engines": {}, "nodes": None, "worker": None, "errors": {}}
        events: list[dict[str, Any]] = []
        for source in self.sources():
            status, text = self._get(source)
            texts[source.name] = (status, text)
            if status != "ok":
                line["errors"][source.name] = status
                if source.name not in self.errors:
                    events.append(
                        {"event": "scrape_failed", "source": source.name, "error": status}
                    )
                self.errors[source.name] = status
                if source.name.startswith("engine/"):
                    line["engines"][source.name.removeprefix("engine/")] = None
                continue
            if source.name in self.errors:
                events.append({"event": "scrape_answered", "source": source.name})
                del self.errors[source.name]
            if source.name == "nodes":
                line["nodes"] = _nodes_view(text)
                continue
            names = GATEWAY_KEPT if source.name == "gateway" else ENGINE_KEPT
            series = kept_series(text, names)
            events += self._resets(source.name, series)
            if source.name == "gateway":
                line["gateway"] = series
            else:
                line["engines"][source.name.removeprefix("engine/")] = series
        worker = self.read_worker()
        line["worker"] = worker_series(worker)
        texts["worker"] = ("ok", json.dumps(worker) if worker is not None else "")
        units, unit_events = self._units(at)
        line["units"] = units
        events += unit_events
        self.raw.write(at, texts)
        with (self.out / "series.jsonl").open("a") as series_file:
            series_file.write(json.dumps(line) + "\n")
        self.count += 1
        return line, events

    def _get(self, source: Source) -> tuple[str, str]:
        try:
            with client(source.address, timeout=SCRAPE_TIMEOUT_S) as http:
                response = http.get(source.path, params=source.params)
        except httpx.HTTPError as exc:
            return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__, ""
        if response.status_code != 200:
            return f"HTTP {response.status_code}: {response.text.strip()[:120]}", response.text
        return "ok", response.text

    def _resets(self, source: str, series: dict[str, float]) -> list[dict[str, Any]]:
        before = self.last.get(source, {})
        down = sorted(
            key for key, value in series.items()
            if "_total" in key.partition("{")[0] and value < before.get(key, -math.inf)
        )  # fmt: skip
        self.last[source] = series
        if not down:
            return []
        return [{"event": "counter_reset", "source": source, "series": len(down), "first": down[0]}]

    def _units(self, at: float) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
        """The units' states now, the events of those that changed, and the manager's journal lines about them since
        the last scrape, into units.jsonl."""
        events: list[dict[str, Any]] = []
        try:
            units = self.systemd.units()
        except E2Error as exc:
            events.append({"event": "units_unread", "error": str(exc)})
            return {}, events
        compact = {
            unit: [
                state.get("ActiveState"),
                state.get("SubState"),
                _int(state.get("NRestarts")),
                _int(state.get("MainPID")),
            ]
            for unit, state in sorted(units.items())
        }
        for unit, now in compact.items():
            then = self.units.get(unit)
            if then is not None and then != now:
                events.append({"event": "unit_changed", "unit": unit, "from": then, "to": now})
        self.units = compact
        since, self.journal_until = self.journal_until, at
        lines = manager_lines(self.systemd.manager_journal(sorted(units), since, at), self.hidden)
        if lines:
            with (self.out / "units.jsonl").open("a") as journal:
                for entry in lines:
                    if is_failure(entry["message"]):
                        tail = self.systemd.unit_tail(
                            entry["unit"], entry["at"] - 120, entry["at"] + 1
                        )
                        entry["own_lines"] = own_lines(tail, self.hidden)
                    journal.write(json.dumps(entry) + "\n")
        return compact, events


def _int(text: str | None) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _nodes_view(text: str) -> dict[str, Any] | None:
    """What a scrape keeps of the gateway's /nodes: each node's verdict, availability, whether it covers for a failed
    node, its calls and its sessions and requests by class."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    found: dict[str, Any] = {}
    for node in data.get("nodes", []) if isinstance(data, dict) else []:
        if isinstance(node, dict) and isinstance(node.get("name"), str):
            found[node["name"]] = {
                key: node.get(key)
                for key in ("verdict", "available", "covering", "sessions", "in_flight")
            }
    return found
