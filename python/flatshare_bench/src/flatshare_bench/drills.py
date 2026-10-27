"""Failure drills (``flatshare-bench drill``): what the gateway's clients see while the engine dies or the
gateway restarts, reloads or dies under them, and how long until a request succeeds again
(``ops/drills``).

``flatshare-bench drill streams`` runs on cyan. It opens streaming chat completions through the gateway,
realtime ones on the voice plane's socket and background ones on its TCP port with the agents
tenant's key, prints ``streaming`` once every stream has its first byte, and reads each stream to its
end, while ``ops/drills/drill.sh`` kills the engine or restarts, reloads or kills the gateway. The
drill script writes the event's time to the event file in the same command as the event, and from
the moment the client sees it, it probes concurrently with the streams: a one-token completion on
each plane every ``probe_interval_s``, each on a new connection as a new caller's would be, until
every stream has ended and each plane's latest probe has succeeded. Every attempt is kept, with when
it was sent and answered. From the first byte of every stream on it also samples the engine's running
and waiting requests (through its tunnel's socket), each sample stamped. Every time is wall-clock
(``time.time()``), so that the drill script's times on the other hosts line up with it. ``flatshare-bench
drill report`` summarizes a drill's record.
"""

from __future__ import annotations

import asyncio
import contextlib
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

from flatshare_bench.e2.arm import E2Error
from flatshare_bench.e2.run import MEASURE, Host, read_env
from flatshare_bench.load import random_prompt
from flatshare_bench.protocol import DEFAULT_MODEL, SloClass
from flatshare_stats import metric_samples, summarize

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
class Attempt:
    """One probe request: the plane's class, when it was sent and answered, and its outcome (``ok``,
    ``http STATUS CODE``, such as ``http 503 upstream_not_ready``, or the exception's class)."""

    slo_class: str
    sent: float
    answered: float | None = None
    outcome: str = ""


@dataclass
class Probe:
    """The probe from the event on: every attempt, in the order they were sent."""

    attempts: list[Attempt] = field(default_factory=list)


@dataclass
class EngineSample:
    """The engine's running and waiting requests at ``at``, or why it did not answer."""

    at: float
    running: float | None = None
    waiting: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class DrillConfig:
    realtime: int = 2
    background: int = 4
    max_tokens: int = 2000
    deadline_s: float = 600.0
    """How long the streams may take, and the probe from the event on, each."""
    probe_interval_s: float = 1.0
    sample_interval_s: float = 0.5
    """How often the engine's running and waiting requests are sampled."""
    model: str = DEFAULT_MODEL


@dataclass
class Clients:
    """The streams' clients, the voice plane's (the gateway's socket, no key) and the agents' (its TCP
    port), and the probe's, which open a connection for each request."""

    voice: httpx.AsyncClient
    agents: httpx.AsyncClient
    voice_probe: httpx.AsyncClient
    agents_probe: httpx.AsyncClient


def _error_text(body: bytes) -> str:
    """An error answer's or event's name and message, in OpenAI's error shape. The gateway names its
    own errors by their ``code`` (``upstream_unavailable``, ``upstream_not_ready``) under the type
    ``server_error``; vLLM's error events, and the gateway's in a broken stream, by their ``type``, with
    the HTTP status as ``code``."""
    try:
        error = orjson.loads(body).get("error")
    except (orjson.JSONDecodeError, AttributeError):
        return body[:200].decode(errors="replace")
    if not isinstance(error, Mapping):
        return str(error)[:200]
    code = error.get("code")
    kind = code if isinstance(code, str) and code else error.get("type") or "error"
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


URL = "/v1/chat/completions"


async def attempt(
    client: httpx.AsyncClient,
    headers: Mapping[str, str],
    slo_class: str,
    config: DrillConfig,
    timeout: float,
) -> Attempt:
    """Send one one-token completion and record what came of it within ``timeout``."""
    found = Attempt(slo_class, time.time())
    body = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": 1,
    }
    try:
        response = await asyncio.wait_for(client.post(URL, headers=headers, json=body), timeout)
        found.outcome = (
            "ok"
            if response.status_code == 200
            else f"http {response.status_code} {_error_text(response.content).split(':')[0]}"
        )
    except (httpx.HTTPError, TimeoutError) as exc:
        found.outcome = type(exc).__name__
    found.answered = time.time()
    return found


async def probe(
    planes: Sequence[tuple[str, httpx.AsyncClient, Mapping[str, str]]],
    streams_ended: Any,
    config: DrillConfig,
) -> Probe:
    """From now on, an attempt on each plane (class, client, headers) every ``probe_interval_s``, while
    earlier ones may still wait for their answer, until ``streams_ended()`` and each plane's latest
    attempt has succeeded, or for ``deadline_s``; the attempts still out are then awaited within it."""
    deadline = time.monotonic() + config.deadline_s
    tasks: list[asyncio.Task[Attempt]] = []
    latest: dict[str, asyncio.Task[Attempt]] = {}

    def recovered() -> bool:
        return len(latest) == len(planes) and all(
            task.done() and task.result().outcome == "ok" for task in latest.values()
        )

    while time.monotonic() < deadline and not (streams_ended() and recovered()):
        tick = time.monotonic()
        for slo_class, client, headers in planes:
            task = asyncio.create_task(
                attempt(client, headers, slo_class, config, deadline - time.monotonic())
            )
            tasks.append(task)
            latest[slo_class] = task
        # Wait out the interval, or less once every stream has ended and the latest attempts succeeded.
        while True:
            await asyncio.sleep(min(0.02, config.probe_interval_s))
            if time.monotonic() - tick >= config.probe_interval_s or (
                streams_ended() and recovered()
            ):
                break
    if tasks:
        await asyncio.wait(tasks, timeout=max(0.0, deadline - time.monotonic()))
    found = Probe()
    for task in tasks:
        if task.done():
            found.attempts.append(task.result())
        else:
            task.cancel()
    found.attempts.sort(key=lambda a: a.sent)
    return found


async def sample_engine(
    engine: httpx.AsyncClient, stop: asyncio.Event, config: DrillConfig
) -> list[EngineSample]:
    """The engine's running and waiting requests every ``sample_interval_s`` until ``stop``."""
    samples = []
    while not stop.is_set():
        at = time.time()
        try:
            response = await engine.get("/metrics", timeout=5.0)
            if response.status_code != 200:
                samples.append(EngineSample(at, error=f"http {response.status_code}"))
            else:
                values = {"vllm:num_requests_running": 0.0, "vllm:num_requests_waiting": 0.0}
                for name, _, value in metric_samples(response.text, tuple(values)):
                    values[name] += value
                samples.append(
                    EngineSample(
                        at,
                        running=values["vllm:num_requests_running"],
                        waiting=values["vllm:num_requests_waiting"],
                    )
                )
        except httpx.HTTPError as exc:
            samples.append(EngineSample(at, error=type(exc).__name__))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                stop.wait(), max(0.0, config.sample_interval_s - (time.time() - at))
            )
    return samples


def read_event(path: Path) -> float | None:
    """The event's wall-clock time, once the drill script has written it to ``path``."""
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def clients(config: DrillConfig, gateway_url: str, realtime_uds: Path) -> Clients:
    """The streams' and the probe's clients, on the gateway's socket and its TCP port."""
    timeout = httpx.Timeout(config.deadline_s, connect=10.0)
    fresh = httpx.Limits(max_keepalive_connections=0)

    def voice(**options: Any) -> httpx.AsyncClient:
        transport = httpx.AsyncHTTPTransport(uds=str(realtime_uds), **options)
        return httpx.AsyncClient(transport=transport, base_url="http://gateway", timeout=timeout)

    return Clients(
        voice=voice(),
        agents=httpx.AsyncClient(base_url=gateway_url, timeout=timeout),
        voice_probe=voice(limits=fresh),
        agents_probe=httpx.AsyncClient(base_url=gateway_url, timeout=timeout, limits=fresh),
    )


async def run_streams(
    config: DrillConfig,
    clients: Clients,
    agents_key: str,
    out: Path,
    event_file: Path,
    engine: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """The drill's client side: the streams, the probe from the event (once ``event_file`` holds its
    time) and the engine's samples; writes and returns the record."""
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
    found, event, samples = Probe(), None, []
    async with contextlib.AsyncExitStack() as stack:
        for client in (clients.voice, clients.agents, clients.voice_probe, clients.agents_probe):
            await stack.enter_async_context(client)
        if engine is not None:
            await stack.enter_async_context(engine)
        tasks = []
        for record in records:
            realtime = record.slo_class == SloClass.REALTIME
            record.sent = time.time()
            tasks.append(
                asyncio.create_task(
                    read_stream(
                        clients.voice if realtime else clients.agents,
                        URL,
                        {} if realtime else agents_headers,
                        body,
                        record,
                    )
                )
            )
        stop_sampling = asyncio.Event()
        sampler: asyncio.Task[list[EngineSample]] | None = None

        async def announce() -> None:
            # ops/drills/drill.sh waits for "streaming" (every stream has its first byte) or "streams ended".
            nonlocal sampler
            while not all(r.first_byte is not None for r in records):
                if all(task.done() for task in tasks):
                    return
                await asyncio.sleep(0.02)
            print(f"streaming {time.time():.3f}", flush=True)
            if engine is not None:
                sampler = asyncio.create_task(sample_engine(engine, stop_sampling, config))

        announcer = asyncio.create_task(announce())
        deadline = time.monotonic() + config.deadline_s
        while (event := read_event(event_file)) is None and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        if event is not None:
            print(f"event {event:.3f}", flush=True)
            planes = [
                (SloClass.REALTIME.value, clients.voice_probe, {}),
                (SloClass.BACKGROUND.value, clients.agents_probe, agents_headers),
            ]
            found = await probe(planes, lambda: all(task.done() for task in tasks), config)
        remaining = max(0.0, deadline - time.monotonic())
        if tasks:
            await asyncio.wait(tasks, timeout=remaining)
        for task, record in zip(tasks, records, strict=True):
            if not task.done():
                task.cancel()
                record.end, record.ended = TIMEOUT, time.time()
        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"streams ended {time.time():.3f}", flush=True)
        await announcer
        stop_sampling.set()
        if sampler is not None:
            samples = await sampler
    print(f"probe done {time.time():.3f}", flush=True)
    result = {
        "kind": "drill",
        "started": started,
        "config": asdict(config),
        "event": event,
        "streams": [asdict(r) for r in records],
        "probe": asdict(found),
        "engine": [asdict(sample) for sample in samples],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2) + b"\n")
    return result


def agents_key(host: Host) -> str:
    """The agents tenant's key, from cyan's gateway-keys.env, as ``flatshare-bench e2 run`` reads it."""
    keys = host.config / "gateway-keys.env"
    key = read_env(keys).get("FLATSHARE_GATEWAY_AGENTS_KEY") if keys.exists() else None
    if not key:
        raise SystemExit(f"{keys} has no FLATSHARE_GATEWAY_AGENTS_KEY, the agents tenant's key")
    return key


def default_uds() -> Path:
    """The gateway's voice-plane socket on cyan."""
    return (
        Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        / "flatshare"
        / "gateway.sock"
    )


def report(record: Mapping[str, Any], event: float | None = None, label: str = "the event") -> str:
    """A drill's summary: how each class's streams ended and how long after ``event`` (the kill, the
    restart or the reload, a wall-clock time; the record's own by default, and None leaves those times
    out), what each plane's probe got from the event on, and the engine's running requests."""
    if event is None:
        event = record.get("event")
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
    lines += _probe_lines(record["probe"], event, label)
    lines += _engine_lines(record.get("engine", []), event, label)
    return "\n".join(lines)


def _counts(outcomes: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome] = counts.get(outcome, 0) + 1
    return ", ".join(f"{n} {o}" for o, n in sorted(counts.items()))


def _probe_lines(found: Mapping[str, Any], event: float | None, label: str) -> list[str]:
    attempts = found["attempts"]
    if attempts and not isinstance(attempts[0], Mapping):
        # A record from before the probe ran from the event: (sent, outcome) pairs after the streams.
        failed = [outcome for _, outcome in attempts if outcome != "ok"]
        line = f"probe: {len(attempts)} attempts"
        if failed:
            line += f" ({_counts(failed)})"
        recovered = found.get("recovered")
        if recovered is None:
            line += "; no success before its deadline"
        elif event is not None:
            line += f"; the first success {recovered - event:,.1f} s after {label}"
        return [line]
    if not attempts:
        return ["probe: none (the record has no event)"]
    lines = []
    for slo_class in sorted({a["slo_class"] for a in attempts}, key=_class_order):
        mine = [a for a in attempts if a["slo_class"] == slo_class]
        failed = [a for a in mine if a["outcome"] != "ok"]
        ok = [a for a in mine if a["outcome"] == "ok"]
        line = f"probe, {slo_class}: {len(mine)} attempts"
        line += f" ({_counts([a['outcome'] for a in failed])})" if failed else ", none failed"
        if not ok:
            line += "; no success before its deadline"
        elif event is not None:
            first = min(a["answered"] for a in ok)
            line += f"; the first success answered {first - event:,.1f} s after {label}"
            if failed:
                last = max(a["sent"] for a in failed)
                line += f", the last failure sent {last - event:,.1f} s after it"
        took = [a["answered"] - a["sent"] for a in mine if a["answered"] is not None]
        if took:
            line += f"; the slowest took {max(took):,.2f} s"
        lines.append(line)
    return lines


def _class_order(slo_class: str) -> int:
    order = [c.value for c in (SloClass.REALTIME, SloClass.INTERACTIVE, SloClass.BACKGROUND)]
    return order.index(slo_class) if slo_class in order else len(order)


def _engine_lines(
    samples: Sequence[Mapping[str, Any]], event: float | None, label: str
) -> list[str]:
    if not samples:
        return []
    if event is None:
        return [f"engine: {len(samples)} samples"]
    before = [s for s in samples if s["at"] < event and s["running"] is not None]
    after = [s for s in samples if s["at"] >= event]
    parts = []
    if before:
        parts.append(
            f"{before[-1]['running']:.0f} running and {before[-1]['waiting']:.0f} waiting before {label}"
        )
    unanswered = [s for s in after if s["error"]]
    if unanswered:
        parts.append(
            f"no answer from {unanswered[0]['at'] - event:,.1f} to {unanswered[-1]['at'] - event:,.1f} s "
            f"after it ({unanswered[0]['error']})"
        )
    idle = next((s for s in after if s["running"] == 0), None)
    if idle is not None:
        parts.append(f"none running from {idle['at'] - event:,.1f} s after it")
    elif after and after[-1]["running"] is not None:
        last = after[-1]
        parts.append(
            f"{last['running']:.0f} still running at the last sample, {last['at'] - event:,.1f} s after it"
        )
    return [f"engine: {'; '.join(parts)}"] if parts else []


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
AT_LAST = frozenset({"the CUDA graphs are captured"})
"""The phases a start can mark more than once, timed at their last line: vLLM that sizes its KV cache by
``gpu_memory_utilization`` (the STT) captures the CUDA graphs once to profile their memory, before it
allocates the KV cache, and then for good."""


def start_phases(journal: str) -> list[tuple[str, float]]:
    """Each phase of :data:`START_PHASES` the journal reaches after its last "Starting" line, at the
    first line that marks it (the last, for those of :data:`AT_LAST`), in seconds since that line."""
    lines = [line.partition(" ") for line in journal.splitlines()]
    stamped = [(float(t), rest) for t, _, rest in lines if t.replace(".", "", 1).isdigit()]
    starts = [i for i, (_, rest) in enumerate(stamped) if re.search(START_PHASES[0][1], rest)]
    if not starts:
        return []
    begin = stamped[starts[-1]][0]
    phases = []
    for name, pattern in START_PHASES:
        marked = [t for t, rest in stamped[starts[-1] :] if re.search(pattern, rest)]
        if marked:
            at = marked[-1] if name in AT_LAST else marked[0]
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
    from flatshare_bench.tiers import TierProbe, TiersConfig, pool_tokens

    config = TiersConfig(url=str(client.base_url), model=DEFAULT_MODEL, lengths=(), priority=None)
    async with client:
        probe = TierProbe(client, config)
        answer = await client.get("/server_info", params={"config_format": "json"})
        if answer.status_code == 404:
            raise E2Error(f"the engine has no /server_info. {MEASURE}")
        info = answer.json()
        capacity = int(
            info["vllm_config"]["kv_transfer_config"]["kv_connector_extra_config"][
                "cpu_bytes_to_use"
            ]
        )

        def stored(exposition: str) -> float:
            return sum(v for _, _, v in metric_samples(exposition, (STORES,)))

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
