"""Failure drills (``lh-bench drill``): what the gateway's clients see while the engine dies or the
gateway restarts under them, and how long until a request succeeds again (``ops/drills``).

``lh-bench drill streams`` runs on cyan. It opens streaming chat completions through the gateway,
realtime ones on the voice plane's socket and background ones on its TCP port with the agents
tenant's key, prints ``streaming`` once every stream has its first byte, and reads each stream to its
end, while ``ops/drills/drill.sh`` kills the engine or restarts the gateway. Then it sends a one-token
completion every ``probe_interval_s`` until one succeeds. Every time is wall-clock (``time.time()``),
so that the drill script's times on the other hosts line up with it. ``lh-bench drill report``
summarizes a drill's record.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import orjson

from leaseholder_bench.e2.run import Host, read_env
from leaseholder_bench.kv_metrics import metric_samples
from leaseholder_bench.load import random_prompt, summarize
from leaseholder_bench.protocol import DEFAULT_MODEL, SloClass

PROMPT = "Count from 1 to 3000, one number per line, with nothing else."
"""A reply that keeps decoding until ``max_tokens``, whatever the sampling."""

DONE = "done"  # the stream's data: [DONE]
ERROR_EVENT = "error-event"  # a data: {"error": ...} chunk, then the stream's end
BROKEN = "broken"  # the connection or the chunked body ended without either
HTTP_ERROR = "http-error"  # an answer other than 200
TIMEOUT = "timeout"  # still open at the drill's deadline


@dataclass
class StreamRecord:
    """One stream as its client saw it."""

    stream: int
    slo_class: str
    sent: float
    status: int | None = None
    first_byte: float | None = None
    chunks: int = 0
    last_chunk: float | None = None
    end: str = ""
    ended: float | None = None
    error: str | None = None
    """The error event's or answer's type and message, or the exception's class and message."""


@dataclass
class Probe:
    """The recovery probe: each attempt's time and outcome (``ok``, ``http STATUS CODE`` or the
    exception's class)."""

    attempts: list[tuple[float, str]] = field(default_factory=list)
    recovered: float | None = None


@dataclass(frozen=True)
class DrillConfig:
    realtime: int = 2
    background: int = 4
    max_tokens: int = 2000
    deadline_s: float = 600.0
    """How long the streams and then the probe may take, each."""
    probe_interval_s: float = 1.0
    model: str = DEFAULT_MODEL


def _error_text(body: bytes) -> str:
    """An error answer's or event's ``type`` (or ``code``) and message, in OpenAI's error shape."""
    try:
        error = orjson.loads(body).get("error")
    except (orjson.JSONDecodeError, AttributeError):
        return body[:200].decode(errors="replace")
    if not isinstance(error, Mapping):
        return str(error)[:200]
    kind = error.get("type") or error.get("code") or "error"
    return f"{kind}: {str(error.get('message', ''))[:200]}"


async def read_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    record: StreamRecord,
) -> None:
    """Send one streaming completion and fill ``record`` as its chunks arrive, until the stream ends."""
    try:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            record.status = response.status_code
            if response.status_code != 200:
                record.end, record.ended = HTTP_ERROR, time.time()
                record.error = _error_text(await response.aread())
                return
            async for line in response.aiter_lines():
                now = time.time()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if record.first_byte is None:
                    record.first_byte = now
                if data == "[DONE]":
                    record.end, record.ended = DONE, now
                    return
                if data.startswith('{"error"'):
                    record.end, record.ended = ERROR_EVENT, now
                    record.error = _error_text(data.encode())
                    return
                record.chunks += 1
                record.last_chunk = now
            record.end, record.ended = BROKEN, time.time()
            record.error = "the stream ended without [DONE]"
    except httpx.HTTPError as exc:
        record.end, record.ended = BROKEN, time.time()
        record.error = f"{type(exc).__name__}: {exc}"[:300]


async def probe(
    client: httpx.AsyncClient, url: str, headers: Mapping[str, str], config: DrillConfig
) -> Probe:
    """Send a one-token completion every ``probe_interval_s`` until one succeeds or the deadline passes."""
    found = Probe()
    body = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": 1,
    }
    deadline = time.monotonic() + config.deadline_s
    while time.monotonic() < deadline:
        started = time.time()
        try:
            response = await client.post(url, headers=headers, json=body)
            outcome = (
                "ok"
                if response.status_code == 200
                else f"http {response.status_code} {_error_text(response.content).split(':')[0]}"
            )
        except httpx.HTTPError as exc:
            outcome = type(exc).__name__
        found.attempts.append((started, outcome))
        if outcome == "ok":
            found.recovered = started
            return found
        await asyncio.sleep(max(0.0, config.probe_interval_s - (time.time() - started)))
    return found


def clients(
    config: DrillConfig, gateway_url: str, realtime_uds: Path
) -> tuple[httpx.AsyncClient, httpx.AsyncClient]:
    """The voice plane's client (the gateway's socket, no key) and the agents' (its TCP port)."""
    timeout = httpx.Timeout(config.deadline_s, connect=10.0)
    voice = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=str(realtime_uds)),
        base_url="http://gateway",
        timeout=timeout,
    )
    return voice, httpx.AsyncClient(base_url=gateway_url, timeout=timeout)


async def run_streams(
    config: DrillConfig,
    voice: httpx.AsyncClient,
    agents: httpx.AsyncClient,
    agents_key: str,
    out: Path,
) -> dict[str, Any]:
    """The drill's client side: the streams, then the probe; writes and returns the record."""
    url_path = "/v1/chat/completions"
    agents_headers = {"Authorization": f"Bearer {agents_key}"}
    records = [
        StreamRecord(
            i, (SloClass.REALTIME if i < config.realtime else SloClass.BACKGROUND).value, 0.0
        )
        for i in range(config.realtime + config.background)
    ]
    body = {
        "model": config.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": config.max_tokens,
        "stream": True,
    }
    started = time.time()
    async with voice, agents:
        tasks = []
        for record in records:
            realtime = record.slo_class == SloClass.REALTIME
            record.sent = time.time()
            tasks.append(
                asyncio.create_task(
                    read_stream(
                        voice if realtime else agents,
                        url_path,
                        {} if realtime else agents_headers,
                        body,
                        record,
                    )
                )
            )
        # ops/drills/drill.sh waits for "streaming" (every stream has its first byte) or for "streams ended".
        announced = False
        deadline = time.monotonic() + config.deadline_s
        while True:
            if not announced and all(r.first_byte is not None for r in records):
                print(f"streaming {time.time():.3f}", flush=True)
                announced = True
            if all(task.done() for task in tasks) or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.05)
        for task, record in zip(tasks, records, strict=True):
            if not task.done():
                task.cancel()
                record.end, record.ended = TIMEOUT, time.time()
        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"streams ended {time.time():.3f}", flush=True)
        recovery = await probe(agents, url_path, agents_headers, config)
    print(f"probe done {time.time():.3f}", flush=True)
    result = {
        "kind": "drill",
        "started": started,
        "config": asdict(config),
        "streams": [asdict(r) for r in records],
        "probe": asdict(recovery),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2) + b"\n")
    return result


def agents_key(host: Host) -> str:
    """The agents tenant's key, from cyan's gateway-keys.env, as ``lh-bench e2 run`` reads it."""
    keys = host.config / "gateway-keys.env"
    key = read_env(keys).get("LH_GATEWAY_AGENTS_KEY") if keys.exists() else None
    if not key:
        raise SystemExit(f"{keys} has no LH_GATEWAY_AGENTS_KEY, the agents tenant's key")
    return key


def default_uds() -> Path:
    """The gateway's voice-plane socket on cyan."""
    return (
        Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "lh" / "gateway.sock"
    )


def report(record: Mapping[str, Any], event: float | None, label: str = "the event") -> str:
    """A drill's summary: how each class's streams ended and how long after ``event`` (the kill or the restart, a
    wall-clock time; None leaves those times out), and the probe's first failure and its recovery."""
    streams: Sequence[Mapping[str, Any]] = record["streams"]
    lines = []
    for slo_class in (SloClass.REALTIME.value, SloClass.BACKGROUND.value):
        mine = [s for s in streams if s["slo_class"] == slo_class]
        if not mine:
            continue
        ends: dict[str, int] = {}
        for s in mine:
            ends[s["end"]] = ends.get(s["end"], 0) + 1
        text = ", ".join(f"{n} {end}" for end, n in sorted(ends.items()))
        lines.append(f"{slo_class}: {len(mine)} streams: {text}")
        if event is not None:
            after = [
                s["ended"] - event for s in mine if s["ended"] is not None and s["ended"] >= event
            ]
            if after:
                cell = summarize([1000 * a for a in after])
                lines.append(
                    f"  ended {cell['p50']:,.0f} ms after {label} at the median, {max(after) * 1000:,.0f} ms at most"
                )
        for error in sorted({s["error"] for s in mine if s["error"]}):
            lines.append(f"  error: {error}")
    attempts = record["probe"]["attempts"]
    failures: dict[str, int] = {}
    for _, outcome in attempts:
        if outcome != "ok":
            failures[outcome] = failures.get(outcome, 0) + 1
    probe_line = f"probe: {len(attempts)} attempts"
    if failures:
        probe_line += " (" + ", ".join(f"{n} {o}" for o, n in sorted(failures.items())) + ")"
    recovered = record["probe"]["recovered"]
    if recovered is None:
        probe_line += "; no success before its deadline"
    elif event is not None:
        probe_line += f"; the first success {recovered - event:,.1f} s after {label}"
    lines.append(probe_line)
    return "\n".join(lines)


START_PHASES = (
    ("the unit starts", r"systemd\[\d+\]: Starting "),
    ("the API server is up", r"\(APIServer pid=\d+\) INFO .*\bversion \d+\.\d+"),
    ("the engine process is up", r"Initializing a V1 LLM engine"),
    ("the weights are loaded", r"Model loading took"),
    ("the KV cache is allocated", r"GPU KV cache size:"),
    ("the CUDA graphs are captured", r"Graph capturing finished in"),
    ("the engine is initialized", r"init engine \(profile, create kv cache, warmup model\) took"),
    (
        "the server listens",
        r"Starting vLLM server on|Uvicorn running on|Application startup complete",
    ),
    ("the unit is ready (its health check)", r"systemd\[\d+\]: Started "),
)
"""A server's start as its journal shows it (``journalctl -o short-unix``): vLLM's lines, uvicorn's, and
systemd's, whose "Started" comes once the unit's ExecStartPost health check passes."""


def start_phases(journal: str) -> list[tuple[str, float]]:
    """Each phase of :data:`START_PHASES` the journal reaches after its last "Starting" line, at the
    first line that marks it, in seconds since that line."""
    lines = [line.partition(" ") for line in journal.splitlines()]
    stamped = [(float(t), rest) for t, _, rest in lines if t.replace(".", "", 1).isdigit()]
    starts = [i for i, (_, rest) in enumerate(stamped) if re.search(START_PHASES[0][1], rest)]
    if not starts:
        return []
    begin = stamped[starts[-1]][0]
    phases = []
    for name, pattern in START_PHASES:
        at = next((t for t, rest in stamped[starts[-1] :] if re.search(pattern, rest)), None)
        if at is not None:
            phases.append((name, round(at - begin, 1)))
    return phases


STORES = "vllm:kv_offload_store_bytes_total"


async def fill_cpu_tier(
    client: httpx.AsyncClient, out: Path, *, factor: float = 2.0, tokens: int = 8192
) -> dict[str, Any]:
    """Write the engine's CPU tier past its capacity and check what it then holds: a prompt sent before
    ``factor`` times the tier's bytes were stored after it must be recomputed (the tier evicted it), and
    one sent after them restores from the tier once the GPU pool is flushed. ``client`` reaches the
    engine (on cyan, through the tunnel), which nothing else may use meanwhile; every request must
    succeed. The fills stop, failing the check, if the tier's store counter has not moved after three
    of them or once they number twice what its rate over those needs. Writes and returns the record."""
    from leaseholder_bench.tiers import TierProbe, TiersConfig, pool_tokens

    config = TiersConfig(url=str(client.base_url), model=DEFAULT_MODEL, lengths=(), priority=None)
    async with client:
        probe = TierProbe(client, config)
        info = (await client.get("/server_info", params={"config_format": "json"})).json()
        capacity = int(
            info["vllm_config"]["kv_transfer_config"]["kv_connector_extra_config"][
                "cpu_bytes_to_use"
            ]
        )

        def stored(exposition: str) -> float:
            return sum(v for name, _, v in metric_samples(exposition) if name == STORES)

        exposition = await probe.metrics()
        probe.pool = pool_tokens(exposition) or 0
        first = random_prompt(probe.rng, tokens)
        rows = [await probe.measure(0, "recompute", first)]
        start = stored(await probe.metrics())
        target = factor * capacity
        fills, fill_ms, limit, failure = 0, [], None, None
        while (now := stored(await probe.metrics())) - start < target:
            if limit is None and fills >= 3:
                if now <= start:
                    failure = f"the CPU tier's store counter did not move in {fills} fills"
                    break
                limit = 2 * math.ceil(target / ((now - start) / fills)) + 2
            if limit is not None and fills >= limit:
                failure = f"the CPU tier stored {now - start:,.0f} of {target:,.0f} bytes in {fills} fills"
                break
            began = time.perf_counter()
            await probe.complete(random_prompt(probe.rng, 16_384))
            fill_ms.append((time.perf_counter() - began) * 1000)
            fills += 1
        if failure is None:
            last = random_prompt(probe.rng, tokens)
            rows.append(await probe.measure(0, "recompute", last))
            await probe.flush()
            rows.append(await probe.measure(0, "cpu", last))  # the latest restores from the tier
            rows.append(await probe.measure(0, "recompute", first))  # the first was evicted from it
        healthy = (await client.get("/health")).status_code == 200
    record = {
        "kind": "drill-cpu-tier",
        "capacity_bytes": capacity,
        "stored_bytes": now - start,
        "fill_requests": fills,
        "fill_ms": summarize(fill_ms),
        "samples": [asdict(s) | {"computed": s.computed_tokens} for s in rows],
        "healthy": healthy,
        "failure": failure,
        "passed": healthy and failure is None and all(s.valid for s in rows),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2) + b"\n")
    return record
