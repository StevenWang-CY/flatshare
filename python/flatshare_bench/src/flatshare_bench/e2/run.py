"""``flatshare-bench e2 run`` measures one E2 point on cyan, the control host, from its arm file:

1. it records the engine behind the tunnel from the engine's own ``/server_info`` and checks the
   arm's ``[engine]`` against it and the GPU node's hardware profile (the frozen box's, unless the
   host's settings or ``--profile`` name another);
2. it makes cyan's units serve the point: the router (SMG, for arm 8b), the gateway's configuration
   (``gateway.toml`` with the arm's ``[gateway]`` table merged in) and the voice agent's preemptive
   generation and lifecycle events, restarting only the units whose configuration changes;
3. it empties the engine's prefix cache, so that every point starts from the same state, and, with
   agent sessions, starts AIPerf's ``agentic_replay`` (``configs/agentic_replay.yaml``) on the
   load-generator cores and waits for its profiling phase and a warm-up;
4. it snapshots the gateway's and the engine's ``/metrics``, runs ``flatshare-voice bots`` inside the
   voice-plane namespace (``--save-audio`` has them save each call's audio, for the demo video) and
   snapshots again: the calls are the measurement window, through which an agent point samples the
   engine's KV use and requests every half second (``--kv-sample-s``; 0 samples nothing, as W5's points
   ran: the sampler came after them), on one connection to the engine that it keeps for the window;
5. it stops AIPerf, keeps the gateway's request log of the calls and the voice worker's journal of
   them (its turn lines and its sessions' event counts), and writes ``summary.json``.

The parts it drives are modules of their own: the engine check (:mod:`.engine`), cyan's settings
and units (:mod:`.cyan`), and the tools' processes and the run's waits (:mod:`.tools`). The run
reaches the engine by its address (:mod:`flatshare_bench.transport`): on cyan, the tunnel's socket.
Its engine check and cache resets need vLLM's development routes, which only the measurement drop-in
serves (``deploy/phastform/variant.sh measure on``, which ``benchmarks/e2/campaign.sh`` runs).

The GPU node's engine is switched by ``benchmarks/e2/campaign.sh`` (``variant.sh use``), never here.
A failure ends the point: ``run.json`` keeps ``"status": "failed"`` and the error, and every process
the run started is stopped by its process group. ``flatshare-bench e2 report`` (:mod:`.report`) builds the
frontier from the run directories.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import orjson

from flatshare_bench.e2.arm import Arm, E2Error, flag_values, load_arm
from flatshare_bench.e2.cyan import (
    CONFIGS,
    GATEWAY,
    WORKER,
    Host,
    Systemd,
    agents_key,
    configure_cyan,
    exclusive,
    read_env,
    redacted,
)
from flatshare_bench.e2.engine import (
    FROZEN,
    check_engine,
    describe_engine,
    profile_flags,
    reset_prefix_cache,
)
from flatshare_bench.e2.record import (
    OUTCOMES,
    gateway_request_lines,
    kv_values,
    logged_requests,
    render,
    summarize_point,
    worker_lines,
)
from flatshare_bench.e2.tools import NEEDS_WAITID, Process, kill_left, pause, tool_scope, until
from flatshare_bench.kv_metrics import fetch
from flatshare_bench.transport import client
from flatshare_stats import metric_samples

PROFILING = "Credit phase start: profiling"
"""AIPerf's log line when its measured phase begins, after the dataset is built and warmed up."""
PROFILING_TIMEOUT_S = 1200.0
"""Building the full corpus's dataset takes about 4 minutes on cyan; configs/aiperf.env allows 900 s."""
AIPERF_DURATION_S = 10800
"""AIPerf's duration: the replay is stopped when the calls end, so this only has to outlast them."""
AIPERF_STOP_S = 120.0
"""How long AIPerf may take to cancel its requests and write its exports after SIGINT."""
CALL_TURN_S = 130.0
"""The longest a call's turn can take: the line, a reply, a barge-in and a reply, the quiet wait."""
CALLERS_EXIT_S = 120.0
"""The longest the callers may take to exit once their calls ended: with --save-audio they write each
call's audio then."""
SESSIONS_END_S = 30.0
"""How long the calls' sessions may take to log their end once the callers have left: a session ends
when its room is deleted, and its job's shutdown, which sends the last events, may take 10 s."""
GATEWAY_IDLE_S = 30.0
"""How long the gateway may take to end its requests once a point's tools have stopped."""
BUSY = ("flatshare_gateway_streams_active", "flatshare_gateway_admission_waiting")
"""The gateway's gauges of the responses it is relaying and of the requests waiting at its admission
point: it is idle when both are at zero (:func:`_ended`)."""
KV_SAMPLE_S = 0.5
"""How often an agent point samples the engine's KV use and its running and waiting requests from its
``/metrics`` while the calls run (record.KV_GAUGES), from which the summary takes the
measured running share; ``flatshare-bench e2 run --kv-sample-s`` sets another, 0 none (W5's points ran without
it)."""


def run_point(
    arm_file: Path,
    out: Path,
    host: Host | None = None,
    systemd: Systemd | None = None,
    *,
    save_audio: bool = False,
    gpu_source: Mapping[str, str | None] | None = None,
    agents_seed: int | None = None,
    kv_sample_s: float | None = KV_SAMPLE_S,
) -> dict[str, Any]:
    """Measure the point of ``arm_file`` into the directory ``out``; returns the summary. A
    directory with a complete point is never overwritten, and an earlier attempt that did not
    complete is kept beside it. ``save_audio``, a run option rather than the arm's, has the callers
    save each call's audio (``flatshare-voice bots --save-audio``). ``gpu_source`` is each GPU node's
    deployed revision (its ~/flatshare/src/DEPLOYED, None where it has none), which campaign.sh
    reads over its SSH at the point's start and run.json keeps beside cyan's; None when unknown.
    ``agents_seed``, a campaign's (``agents-seed`` in its file), replaces the arm file's ``[agents]
    seed`` for AIPerf; run.json keeps the seed the replay ran with (``agents_seed``). An agent point
    samples the engine's KV every ``kv_sample_s`` while the calls run; 0 or None samples nothing (W5's
    points ran without the sampler, and tests whose fake engines count their /metrics reads), and
    run.json keeps the interval the point sampled at (``kv_sample_s``, null when it did not)."""
    if not hasattr(os, "waitid"):
        raise E2Error(NEEDS_WAITID)
    if stale := _stale_install():
        raise E2Error(stale)
    if agents_seed is not None and agents_seed < 0:
        raise E2Error(f"the agents' seed cannot be negative: {agents_seed}")
    if kv_sample_s is not None and kv_sample_s < 0:
        raise E2Error(f"the KV sampler's interval cannot be negative: {kv_sample_s}")
    kv_sample_s = kv_sample_s or None  # 0 samples nothing
    arm = load_arm(arm_file)
    seed = arm.agents.seed if agents_seed is None else agents_seed
    host = host or Host.load()
    systemd = systemd or Systemd()
    with exclusive(host):
        _fresh(out)
        shutil.copyfile(arm_file, out / "arm.toml")
        record = _Record(out, arm, host, gpu_source)
        record["save_audio"] = save_audio
        # The seed AIPerf replays with: the campaign's when it set one, else the arm file's.
        record["agents_seed"] = seed if arm.agents.sessions else None
        # The KV sampler's interval, null where the point did not sample (voice alone, or 0).
        record["kv_sample_s"] = kv_sample_s if arm.agents.sessions else None
        try:
            # Under the lock no other run's tools are alive: a group named in its scope, or a
            # process marked in it, is left from a run that was killed before it could stop them.
            if leftovers := kill_left(tool_scope(host)):
                record["leftovers_killed"] = leftovers
                record.say(f"killed {len(leftovers)} processes left by an earlier run: {leftovers}")
            with contextlib.ExitStack() as running:
                summary = _measure(
                    arm, out, host, systemd, record, running, save_audio, seed, kv_sample_s
                )
        except BaseException as exc:
            record.fail(exc)
            raise
        record.finish()
        return summary


def _measure(
    arm: Arm,
    out: Path,
    host: Host,
    systemd: Systemd,
    record: _Record,
    running: contextlib.ExitStack,
    save_audio: bool,
    seed: int,
    kv_sample_s: float | None,
) -> dict[str, Any]:
    base = profile_flags(host.profile)
    record["profile"] = host.profile
    if host.preflight:
        record["preflight"] = preflight_summary(Path(host.preflight))
    server = _check_engine(arm, host.engine_url, out, record, base)
    record.say(f"configuring cyan for {arm.name}")
    record["cyan"] = configure_cyan(arm, host, systemd)
    until(
        lambda: _ready(host.gateway_url),
        60.0,
        f"{host.gateway_url}/ready did not answer within 60 s: the engine or the router is down",
    )
    reset_prefix_cache(host.engine_url, server)
    # Before the replay and the calls the gateway is idle: the base of the check that its journal kept a
    # line for every request it ended.
    idle = (time.time(), _ended(host))
    # The model the replay asks for: the engine's, as the node's profile and the arm's flags start it.
    model = (flag_values(base) | flag_values(arm.engine.flags))["--model"]
    aiperf = (
        _start_aiperf(arm, host, out, record, running, seed, model) if arm.agents.sessions else None
    )
    start = _snapshot(host, out, "start")
    sampler = None
    if aiperf is not None and kv_sample_s is not None:
        sampler = _KvSampler(host.engine_url, out / "metrics", kv_sample_s)
        running.callback(sampler.stop)
    end, bots = _call(arm, host, out, record, running, aiperf, save_audio)
    if sampler is not None:
        sampler.stop()
    _snapshot(host, out, "end")
    _callers_exit(bots)
    record["window"] = {"start": start, "end": end}
    if aiperf is not None:
        record.say("stopping AIPerf")
        record["aiperf"] = {"exit_status": aiperf.stop(AIPERF_STOP_S)}
        if aiperf.status:
            raise aiperf.failure("failed when stopped")
    _gateway_log(systemd, out, start, end)
    record["gateway_log"] = _check_gateway_log(host, systemd, idle)
    _worker_journal(systemd, out, start, arm.voice.callers)
    try:
        metrics = summarize_point(
            out,
            start,
            end,
            agents=bool(arm.agents.sessions),
            warmup_turns=arm.voice.warmup_turns,
        )
    except ValueError as exc:
        raise E2Error(str(exc)) from None
    summary = {"name": arm.name, "arm": arm.arm, "label": arm.label} | metrics
    _write_json(out / "summary.json", summary)
    record.say(render(summary))
    return summary


def _check_engine(
    arm: Arm, url: str, out: Path, record: _Record, base: Sequence[str] = FROZEN
) -> str:
    """Record the engine (``engine.json`` and run.json's ``engine``) and check it against the arm's;
    returns its server."""
    info = describe_engine(url)
    _write_json(out / "engine.json", redacted(info.config))
    engine = {"server": info.server, "version": info.version, "pool_tokens": info.pool_tokens}
    record["engine"] = engine  # recorded before the check, which fails on a wrong engine
    record["engine"] = engine | {"flags": check_engine(arm.engine, info, base=base)}
    return info.server


def _start_aiperf(
    arm: Arm,
    host: Host,
    out: Path,
    record: _Record,
    running: contextlib.ExitStack,
    seed: int,
    model: str,
) -> Process:
    """Start the replay with ``seed``, asking for ``model``, and return once it has been profiling for
    the warm-up."""
    agents = arm.agents
    record.say(f"AIPerf: {agents.sessions} session trees through the gateway, seed {seed}")
    env = _aiperf_env(arm, host, out, seed, model)
    # AIPerf 0.13.0 leaves its IPC directory and the run's dataset memory map in the temp directory, one of each a
    # point (cyan's /tmp held 234 maps, 264 MB). A temp directory of the point's own, in the same file system, holds
    # them until AIPerf has stopped, and goes with them.
    scratch = Path(tempfile.mkdtemp(prefix="flatshare-aiperf-"))
    running.callback(shutil.rmtree, scratch, ignore_errors=True)
    env["TMPDIR"] = str(scratch)
    aiperf = Process("AIPerf", host.aiperf(), out / "aiperf.log", env, tool_scope(host))
    running.callback(aiperf.stop, AIPERF_STOP_S)  # first: the stack unwinds the other way
    deadline = time.monotonic() + PROFILING_TIMEOUT_S
    while PROFILING not in aiperf.log.read_text(errors="replace"):
        if aiperf.exited():
            raise aiperf.failure("exited before its profiling phase")
        if time.monotonic() > deadline:
            raise aiperf.failure(f"did not start profiling within {PROFILING_TIMEOUT_S:.0f} s")
        pause()
    record.say(f"AIPerf is profiling; {agents.warmup:g} s of warm-up")
    deadline = time.monotonic() + agents.warmup
    while (left := deadline - time.monotonic()) > 0:
        if aiperf.exited():
            raise aiperf.failure("exited during the warm-up")
        pause(left)
    return aiperf


def _call(
    arm: Arm,
    host: Host,
    out: Path,
    record: _Record,
    running: contextlib.ExitStack,
    aiperf: Process | None,
    save_audio: bool,
) -> tuple[float, Process]:
    """Run the calls; returns when they ended, and the callers' process, which may still be writing
    the calls' audio (:func:`_callers_exit`). The replay must outlast the calls."""
    voice = arm.voice
    warmup = f" after {voice.warmup_turns} warm-up" if voice.warmup_turns else ""
    record.say(f"calls: {voice.callers} callers, {voice.turns} turns each{warmup}")
    options = {
        "--corpus": host.root / "data" / "callers",
        "--callers": voice.callers,
        "--turns": voice.turns,
        "--barge-in": voice.barge_in,
        "--seed": voice.seed,
        # Only when there are some, as the callers before the option took no such flag.
        **({"--warmup-turns": voice.warmup_turns} if voice.warmup_turns else {}),
        "--out": out / "bots",
    }
    command = [*host.bots(), *(str(word) for option in options.items() for word in option)]
    if save_audio:
        command.append("--save-audio")
    env = os.environ | read_env(host.config / "livekit.env")
    bots = Process("flatshare-voice bots", command, out / "bots.log", env, tool_scope(host))
    running.callback(bots.stop, 30.0)
    deadline = time.monotonic() + CALL_TURN_S * (voice.warmup_turns + voice.turns) + 120
    # The callers write their summary as their calls end, before any call's audio.
    ended = out / "bots" / "summary.json"
    while not bots.exited() and not ended.exists():
        if aiperf is not None and aiperf.exited():
            raise aiperf.failure("exited while the calls ran, so its load did not cover them")
        if time.monotonic() > deadline:
            raise bots.failure("did not finish in time")
        pause()
    end = time.time()
    if not ended.exists():  # they exited before their calls ended
        _callers_exit(bots)
    return end, bots


def _callers_exit(bots: Process) -> None:
    """Wait for the callers to exit once their calls ended, and check how: with --save-audio they
    write each call's audio first, which the point's window and its last metrics leave out."""
    deadline = time.monotonic() + CALLERS_EXIT_S
    while not bots.exited():
        if time.monotonic() > deadline:
            raise bots.failure("did not exit after the calls")
        pause()
    if bots.stop() not in (0, 3):  # 3: some turns failed, which the summary counts
        raise bots.failure("failed")


def _gateway_log(systemd: Systemd, out: Path, start: float, end: float) -> None:
    """Keep the gateway's request lines of the calls (gateway.jsonl), from which the summary takes the
    first bytes exactly; none when the gateway logs text."""
    if lines := gateway_request_lines(systemd.journal(GATEWAY, start, until=end)):
        (out / "gateway.jsonl").write_text("".join(f"{line}\n" for line in lines))


def _ended(host: Host) -> tuple[float, float]:
    """From the gateway's metrics: the requests it has ended (OUTCOMES, every class and outcome), and
    those it is relaying or holding at its admission point (BUSY)."""
    exposition = fetch(host.gateway_socket)
    ended = busy = 0.0
    for name, _, value in metric_samples(exposition, (OUTCOMES, *BUSY)):
        if name == OUTCOMES:
            ended += value
        else:
            busy += value
    return ended, busy


def _check_gateway_log(
    host: Host, systemd: Systemd, base: tuple[float, tuple[float, float]]
) -> dict[str, int]:
    """Check that the gateway's journal kept a request line for every request the gateway ended since
    ``base``, the time and counters of a moment it was idle: the first bytes come from those lines, and
    journald drops what comes faster than its rate limit, which for a user unit is journald.conf's. Waits
    until the gateway is idle again, so that both counts end at the same moment."""
    since, (before, _) = base
    deadline = time.monotonic() + GATEWAY_IDLE_S
    last = None
    while True:
        ended, busy = _ended(host)
        if busy == 0 and ended == last:
            break
        if time.monotonic() > deadline:
            raise E2Error(
                f"the gateway did not go idle within {GATEWAY_IDLE_S:.0f} s after the point's tools stopped"
            )
        last = ended
        pause()
    until = time.time()
    # Requests that went upstream with a class, as the counter counts them.
    lines = [
        record
        for at, record in logged_requests(systemd.journal(GATEWAY, since, until=until))
        if since < at <= until and "class" in record and "outcome" in record
    ]
    counted = round(ended - before)
    if len(lines) != counted:
        raise E2Error(
            f"the gateway's journal kept {len(lines)} request lines of the {counted} requests it ended "
            "during the point: journald dropped some, and the first bytes would come from the rest"
        )
    return {"request_lines": len(lines), "requests_ended": counted}


def _worker_journal(systemd: Systemd, out: Path, since: float, calls: int) -> None:
    """Keep the voice worker's journal from the start of the calls (voice-worker.log): a line per
    turn, and one per session as it ends, with its event counts. Waits until every call's session
    has ended, unless the worker logs neither kind of line."""
    deadline = time.monotonic() + SESSIONS_END_S
    while True:
        journal = systemd.journal(WORKER, since, until=time.time())
        turns, sessions = worker_lines(journal)
        if len(sessions) >= calls or not (turns or sessions) or time.monotonic() > deadline:
            break
        pause()
    (out / "voice-worker.log").write_text(journal)


def _ready(address: str, path: str = "/ready") -> bool:
    with contextlib.suppress(httpx.HTTPError), client(address, timeout=5) as http:
        return http.get(path).status_code == 200
    return False


class _KvSampler:
    """The engine's KV use and running and waiting requests (record.KV_GAUGES) from its ``/metrics``
    every ``interval`` seconds, from its start until :meth:`stop`, a line a sample (its time, then the
    three) into ``metrics``' engine-kv.csv. A sample the engine does not answer in time is left out;
    the summary counts the samples it has. The sampler has one client for the whole run, its
    connection kept alive between samples (the engine keeps an idle one 60 s): through cyan's tunnel a
    new connection is a new SSH session on the GPU node, about 12 ms of its CPU, which a client a
    sample would cost it twice a second while it measures. A request that fails drops its client, and
    the next sample opens another."""

    def __init__(self, address: str, metrics: Path, interval: float = KV_SAMPLE_S) -> None:
        self.interval = interval
        self._stopped = threading.Event()
        metrics.mkdir(exist_ok=True)
        self._thread = threading.Thread(
            target=self._sample, args=(address, metrics / "engine-kv.csv"), daemon=True
        )
        self._thread.start()

    def _sample(self, address: str, path: Path) -> None:
        http: httpx.Client | None = None
        with path.open("w") as samples:
            samples.write("time,kv_cache_usage,running,waiting\n")
            try:
                while not self._stopped.is_set():
                    at = time.time()
                    values = None
                    try:
                        http = http or client(address, timeout=2.0)
                        response = http.get("/metrics")
                        response.raise_for_status()
                        values = kv_values(response.text)
                    except httpx.HTTPError:
                        if http is not None:
                            http.close()
                        http = None
                    if values is not None:
                        samples.write(f"{at:.6f}," + ",".join(f"{v:g}" for v in values) + "\n")
                        samples.flush()
                    self._stopped.wait(max(0.0, at + self.interval - time.time()))
            finally:
                if http is not None:
                    http.close()

    def stop(self) -> None:
        """Stop sampling once the sample in flight is written; stopping twice is stopping."""
        self._stopped.set()
        self._thread.join()


def _snapshot(host: Host, out: Path, when: str) -> float:
    """Save the gateway's ``/metrics`` (from its socket, which asks for no key) and the engine's;
    returns the time it finished."""
    metrics = out / "metrics"
    metrics.mkdir(exist_ok=True)
    files = {f"gateway-{when}.prom": host.gateway_socket, f"engine-{when}.prom": host.engine_url}
    for name, address in files.items():
        try:
            (metrics / name).write_text(fetch(address))
        except httpx.HTTPError as exc:
            raise E2Error(f"GET /metrics from {address}: {exc}") from None
    return time.time()


def _aiperf_env(arm: Arm, host: Host, out: Path, seed: int, model: str) -> dict[str, str]:
    agents = arm.agents
    data = host.root / "data" / agents.data
    if not data.is_dir():
        raise E2Error(f"no trace corpus at {data} (flatshare-bench scale-traces writes one)")
    if not (CONFIGS / "agentic_replay.yaml").is_file():
        raise E2Error(
            f"no AIPerf configuration in {CONFIGS}: install the package from its source tree"
        )
    key = agents_key(host)
    env = os.environ | read_env(CONFIGS / "aiperf.env")
    env |= {
        "FLATSHARE_BENCH_URL": f"{host.gateway_url}/v1/chat/completions",
        "FLATSHARE_BENCH_API_KEY": key,
        "FLATSHARE_BENCH_DATA": str(data),
        "FLATSHARE_BENCH_SESSIONS": str(agents.sessions),
        "FLATSHARE_BENCH_DURATION": str(AIPERF_DURATION_S),
        "FLATSHARE_BENCH_GRACE": "30",
        "FLATSHARE_BENCH_THINK_CAP": f"{agents.think_cap:g}",
        "FLATSHARE_BENCH_SEED": str(seed),
        "FLATSHARE_BENCH_CLASS": "background",
        "FLATSHARE_BENCH_ARTIFACTS": str(out / "aiperf"),
        "FLATSHARE_BENCH_MODEL": model,
        "FLATSHARE_BENCH_TOKENIZER": model,
    }
    return env


def _fresh(out: Path) -> None:
    run = out / "run.json"
    if run.exists():
        if orjson.loads(run.read_bytes()).get("status") == "complete":
            raise E2Error(f"{out} holds a complete point; e2 run never overwrites one")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out.rename(out.with_name(f"{out.name}.incomplete-{stamp}"))
    elif out.exists() and any(out.iterdir()):
        raise E2Error(f"{out} is not empty and holds no point")
    out.mkdir(parents=True, exist_ok=True)


class _Record:
    """run.json, rewritten at each step so that a point that dies keeps its status, and run.log."""

    def __init__(
        self, out: Path, arm: Arm, host: Host, gpu_source: Mapping[str, str | None] | None
    ) -> None:
        self.path = out / "run.json"
        self.log = out / "run.log"
        self.data: dict[str, Any] = {
            "kind": "e2-point",
            "name": arm.name,
            "status": "running",
            "host": platform.node(),
            "started": time.time(),
            "arm": asdict(arm),
            "versions": _versions(host, gpu_source),
        }
        self._write()

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._write()

    def say(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {message}"
        with self.log.open("a") as log:
            log.write(line + "\n")
        # The point goes on when the ssh session that started it closes; run.log keeps every line.
        with contextlib.suppress(OSError):
            print(line, file=sys.stderr, flush=True)

    def fail(self, exc: BaseException) -> None:
        error = str(exc) if isinstance(exc, E2Error) else f"{type(exc).__name__}: {exc}"
        self.data |= {"status": "failed", "error": error, "finished": time.time()}
        self._write()
        self.say(f"failed: {error}")

    def finish(self) -> None:
        self.data |= {"status": "complete", "finished": time.time()}
        self._write()

    def _write(self) -> None:
        _write_json(self.path, self.data)


PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"
"""flatshare-bench's own pyproject.toml, where it runs from its source tree (bootstrap.sh's editable install)."""


def _stale_install(pyproject: Path = PYPROJECT) -> str | None:
    """Why flatshare-bench's install does not match the source it runs, or None. An editable install runs the
    source tree but keeps the metadata of its install, the version run.json records among it: until
    the venv is synced again, a version change in pyproject.toml would leave every record naming the
    old one. An installed wheel has no pyproject.toml beside it and its own metadata."""
    if not pyproject.exists():
        return None
    source = tomllib.loads(pyproject.read_text())["project"]["version"]
    if (installed := metadata.version("flatshare-bench")) != source:
        return (
            f"flatshare-bench's venv was installed at version {installed} and its source is {source}: sync "
            "the venv (deploy/cyan/bootstrap.sh) so that run.json names the version that runs"
        )
    return None


def preflight_summary(path: Path) -> dict[str, Any]:
    """What a point keeps of its host's preflight: when it ran and each check that was not ok, with
    its reason and what it changes in what the results mean (docs/HARDWARE.md, the preflight)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path),
        "time": data.get("time"),
        "counts": data.get("counts"),
        "not_ok": [
            {key: check.get(key) for key in ("check", "status", "value", "reason", "effect")}
            for check in data.get("checks", [])
            if check.get("status") != "ok"
        ],
    }


def _versions(host: Host, gpu_source: Mapping[str, str | None] | None) -> dict[str, Any]:
    deployed = host.root / "src" / "DEPLOYED"  # deploy/deploy.sh records the commit it synced
    gateway = host.root / "bin" / "flatshare-gateway"
    flatshare_gateway = None
    if gateway.exists():
        shown = subprocess.run([gateway, "--version"], capture_output=True, text=True)
        flatshare_gateway = shown.stdout.strip() or None
    return {
        "source": deployed.read_text().strip() if deployed.exists() else None,
        "gpu_source": None if gpu_source is None else dict(gpu_source),
        "flatshare_bench": metadata.version("flatshare-bench"),
        "aiperf": metadata.version("aiperf"),
        "flatshare_gateway": flatshare_gateway,
    }


NODE_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
"""A GPU node's name (its host's first label, as campaign.sh gives it), which unit, file and socket
names carry."""
_REVISION = re.compile(r"[A-Za-z0-9._+-]*")
"""A deployed revision as deploy/deploy.sh writes it (``git describe --always --dirty``)."""


def gpu_sources(values: Sequence[str]) -> dict[str, str | None]:
    """``NODE=REVISION`` values (``flatshare-bench e2 run --gpu-source``) by node, an empty revision None."""
    sources: dict[str, str | None] = {}
    for value in values:
        node, sep, revision = value.partition("=")
        if not sep or not NODE_NAME.fullmatch(node) or not _REVISION.fullmatch(revision):
            raise E2Error(f"--gpu-source {value!r} is not NODE=REVISION")
        if node in sources:
            raise E2Error(f"--gpu-source names {node} twice")
        sources[node] = revision or None
    return sources


def _write_json(path: Path, data: Any) -> None:
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2) + b"\n")
