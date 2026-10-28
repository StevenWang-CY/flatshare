"""``flatshare-bench soak run``: the production path under load for hours, on cyan (:mod:`flatshare_bench.soak`).

A soak changes nothing on the box. Before anything starts it checks that the box serves as it would for real callers,
and refuses to start otherwise (:meth:`Soak.preflight`): the voice worker in serve mode with its status fresh, no
drop-in of an E2 point left in cyan's units (``flatshare-bench e2 reset`` removes them), the gateway ready, the model
AIPerf asks for served, the callers' and the agents' corpora in place, and every GPU node's guard able to read its GPU
and finding no other process there. Then:

1. **Agents.** With agent sessions, AIPerf's agentic replay starts as E2 runs it (``configs/agentic_replay.yaml`` with
   E2's variables: the corpus, the seed, the think-time cap, the sessions), through the gateway's TCP listener as the
   agents tenant, on the load generators' cores, and the soak waits for its profiling phase and a warm-up.
2. **The window**, ``hours`` long. The calls (``flatshare-voice arrivals``, inside the voice-plane namespace on the
   callers' cores) arrive over it, the scraper (:mod:`.scrape`) reads the service every ``scrape_s``, and the guard
   (:mod:`.guard`) the GPUs every ``guard_s``. A tool that ends before the window does is recorded and started again,
   after a backoff when it keeps ending soon after its start; a run of AIPerf that ends at its ``agents_run_hours`` is
   started again at once. While the guard holds the load back, both tools are stopped, and they start again once
   every GPU has been clear for its hold-off.
3. **The end.** At the window's end no call arrives any more and the calls in progress play their turns to the end
   (for ``drain_s`` at most), then AIPerf stops; a last scrape follows, and the summary (:mod:`.summary`). SIGINT or
   SIGTERM ends a soak early the same way, but each call ends with its current turn; a second signal hangs the calls up
   and gives AIPerf a short grace. Either way the record is complete and says what ended it. Each hour's summary is
   written as the hour ends, so that a soak that dies keeps them.

The soak directory holds ``soak.json`` (the record), ``soak.log``, ``events.jsonl``, ``series.jsonl``,
``units.jsonl``, ``guard.jsonl``, ``scrapes/`` (every scrape whole), ``calls/run-NNN/`` (each run of the callers:
``attempts.jsonl``, ``turns.jsonl``, ``summary.json``), ``agents/run-NNN/`` (each run of AIPerf: its log and exports),
``summary.json`` and ``summary.md``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx

from flatshare_bench.e2.arm import E2Error, flag_values
from flatshare_bench.e2.cyan import (
    CONFIGS,
    GATEWAY,
    WORKER,
    Host,
    agents_key,
    exclusive,
    read_env,
    redacted,
)
from flatshare_bench.e2.engine import profile_flags
from flatshare_bench.e2.run import AIPERF_STOP_S, PROFILING, PROFILING_TIMEOUT_S
from flatshare_bench.e2.tools import NEEDS_WAITID, Process, kill_left, tool_scope
from flatshare_bench.soak import summary as summaries
from flatshare_bench.soak.guard import BLIND_AFTER, CLEAR_S, Guard, node_guard
from flatshare_bench.soak.scrape import Scraper, SoakSystemd
from flatshare_bench.soak.summary import load, render_hour, summarize, write_summary
from flatshare_bench.transport import UNIX, client, socket_path

POLL_S = 1.0
"""How often the soak looks at its tools, its clocks and its signals."""
STALE_S = 5.0
"""A worker status older than this was not written by a running worker, which writes it every second."""
CALLS_STOP_S = 150.0
"""How long the callers may take to end each call with its current turn after SIGINT (a turn's longest, E2's
CALL_TURN_S, and a margin)."""
CALLS_HANG_UP_S = 30.0
"""How long the callers may take to hang up at a second SIGINT and write their summary."""
HURRIED_S = 10.0
"""A tool's grace once a second signal hurries the soak."""
RESTART_BACKOFF_S = (10.0, 30.0, 120.0, 600.0)
"""How long a tool that ended waits to start again, by how many of its runs in a row ended within QUICK_S of their
start."""
QUICK_S = 300.0
"""A run that ends sooner than this after its start counts toward the backoff."""
LEFT_S = 60.0
"""A tool is not started again with less of the window than this left."""
AGENTS_OUTLAST_S = 300
"""How much longer than the window AIPerf is told to run: the soak stops it once the calls have ended."""
E2_DROP_INS = {
    GATEWAY: ("e2-gateway",),
    WORKER: ("no-preemptive-generation", "no-events", "e2-node"),
}
"""Drop-ins of an E2 point (``variant-NAME.conf``) that ``flatshare-bench e2 reset`` removes; a soak refuses to start
beside them, since the box would not serve as deployed."""
WORKER_SETTINGS = (
    "FLATSHARE_VOICE_MODE",
    "FLATSHARE_AUDIO",
    "FLATSHARE_TURN_DETECTION",
    "FLATSHARE_PREEMPTIVE_GENERATION",
    "FLATSHARE_EVENTS",
    "FLATSHARE_NODE_BINDING",
    "FLATSHARE_LLM_UDS",
    "FLATSHARE_SERVING_PROFILE",
    "FLATSHARE_VOICE_STATUS",
    "FLATSHARE_IDLE_PROCESSES",
)
"""The voice worker's settings a soak's record keeps, from its unit's environment."""


@dataclass(frozen=True)
class SoakOptions:
    """What a soak runs (``flatshare-bench soak run``'s options)."""

    hours: float
    slo_reference_ms: float | None = None
    """Voice alone's V2V p95, with the same calls on the same box: the SLO is it plus ``slo_margin_ms``."""
    slo_margin_ms: float = 250.0
    calls_per_hour: float = 20.0
    turns: int = 10
    warmup_turns: int = 1
    barge_in: float = 0.25
    caller_seed: int = 0
    dispatch_timeout_s: float = 15.0
    max_open: int = 8
    agents_sessions: int = 16
    agents_data: str = "weka-x32"
    agents_seed: int = 11
    think_cap_s: float = 120.0
    agents_warmup_s: float = 60.0
    agents_run_hours: float = 0.0
    """A run of AIPerf's length, after which it starts again; 0: one run for the whole window."""
    scrape_s: float = 60.0
    guard_s: float = 60.0
    guards: Mapping[str, str] = field(default_factory=dict)
    """Each node's guard method (:func:`~flatshare_bench.soak.guard.parse_guards`); a node not named has ``ssh``, or
    ``local`` on a single host."""
    guard_clear_s: float = CLEAR_S
    guard_blind_after: int = BLIND_AFTER
    drain_s: float = 600.0
    model: str | None = None
    gpu_source: Mapping[str, str | None] | None = None

    def __post_init__(self) -> None:
        checks = (
            (
                0 < self.hours <= 24 * 14,
                f"hours must be above 0 and at most two weeks, got {self.hours}",
            ),
            (self.calls_per_hour > 0, "calls_per_hour must be above 0"),
            (self.turns >= 1 and self.warmup_turns >= 0, "a call needs a measured turn"),
            (0 <= self.barge_in <= 1, "barge_in is a share of the turns, 0 to 1"),
            (self.agents_sessions >= 0, "agents_sessions cannot be negative"),
            (
                self.think_cap_s > 0 and self.agents_warmup_s >= 0,
                "think_cap_s and agents_warmup_s are seconds",
            ),
            (self.agents_run_hours >= 0, "agents_run_hours cannot be negative"),
            (self.scrape_s >= 1 and self.guard_s >= 1, "scrape_s and guard_s are 1 s or more"),
            (
                self.guard_clear_s >= 0 and self.guard_blind_after >= 1,
                "the guard's hold-off and blind count",
            ),
            (self.max_open >= 1 and self.dispatch_timeout_s > 0, "max_open and dispatch_timeout_s"),
            (self.drain_s >= 0, "drain_s cannot be negative"),
            (
                self.slo_reference_ms is None or self.slo_reference_ms > 0,
                "slo_reference_ms is voice alone's V2V p95 in ms",
            ),
        )
        for ok, what in checks:
            if not ok:
                raise E2Error(f"soak: {what}")


class Tools:
    """How a soak runs its tools on ``host``: AIPerf from this environment on the load generators' cores, and the
    callers from the voice venv inside the voice-plane namespace on theirs, as e2 run runs ``bots``. The tests swap
    them for fakes."""

    def __init__(self, host: Host) -> None:
        self.host = host

    def aiperf(self) -> list[str]:
        return self.host.aiperf()

    def calls(self) -> list[str]:
        voicenet = self.host.root / "src" / "deploy" / "bin" / "in-voicenet"
        flatshare_voice = self.host.root / "venvs" / "voice" / "bin" / "flatshare-voice"
        return [
            "taskset",
            "-c",
            self.host.caller_cores,
            str(voicenet),
            str(flatshare_voice),
            "arrivals",
        ]


@dataclass
class ToolRun:
    """One run of a tool: its record (``info``, in soak.json's ``calls_runs`` or ``agents_runs``) and process."""

    kind: str
    process: Process
    info: dict[str, Any]
    directory: Path
    scratch: Path | None = None
    planned_end: float | None = None


def read_worker_status(path: Path, now: float | None = None) -> dict[str, Any] | None:
    """The voice worker's status file (flatshare_voice.admission.Monitor writes it every second), with its age in
    seconds (``age_s``); None without one, or with one that is not its JSON."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    updated = data.get("updated")
    age = (
        (time.time() if now is None else now) - updated
        if isinstance(updated, (int, float))
        else None
    )
    keys = (
        "mode",
        "max_calls",
        "active_calls",
        "admitting",
        "reason",
        "refused",
        "audio",
        "turn_detection",
        "endpointing_min_delay",
        "vad_min_silence",
        "node_binding",
        "profile",
    )
    return {key: data.get(key) for key in keys} | {"age_s": None if age is None else round(age, 1)}


def gateway_nodes(config: Mapping[str, Any]) -> tuple[list[str], bool]:
    """The GPU nodes a gateway's configuration fronts, and whether it has ``[[nodes]]``: the nodes' names, or the one
    its ``upstream`` reaches through a tunnel's socket (``NAME-engine.sock``), or ``localhost`` on a single host."""
    if "nodes" in config:
        return [str(node["name"]) for node in config["nodes"]], True
    upstream = str(config.get("upstream", ""))
    path = socket_path(upstream) if upstream.startswith(UNIX) else None
    name = Path(path).name if path else ""
    return (
        [name.removesuffix("-engine.sock")] if name.endswith("-engine.sock") else ["localhost"]
    ), False


def aiperf_env(
    host: Host, out: Path, options: SoakOptions, seed: int, model: str, duration: int
) -> dict[str, str]:
    """AIPerf's environment for a run of the soak's agents into ``out``: E2's (``e2/run.py``'s ``_aiperf_env``, the
    same variables from the same files, which a test holds them to), with the run's duration."""
    data = host.root / "data" / options.agents_data
    env = os.environ | read_env(CONFIGS / "aiperf.env")
    env |= {
        "FLATSHARE_BENCH_URL": f"{host.gateway_url}/v1/chat/completions",
        "FLATSHARE_BENCH_API_KEY": agents_key(host),
        "FLATSHARE_BENCH_DATA": str(data),
        "FLATSHARE_BENCH_SESSIONS": str(options.agents_sessions),
        "FLATSHARE_BENCH_DURATION": str(duration),
        "FLATSHARE_BENCH_GRACE": "30",
        "FLATSHARE_BENCH_THINK_CAP": f"{options.think_cap_s:g}",
        "FLATSHARE_BENCH_SEED": str(seed),
        "FLATSHARE_BENCH_CLASS": "background",
        "FLATSHARE_BENCH_ARTIFACTS": str(out / "aiperf"),
        "FLATSHARE_BENCH_MODEL": model,
        "FLATSHARE_BENCH_TOKENIZER": model,
    }
    return env


class SoakRecord:
    """soak.json, rewritten at each change so that a soak that dies keeps its state; soak.log and events.jsonl."""

    def __init__(self, out: Path, data: dict[str, Any]) -> None:
        self.path = out / "soak.json"
        self.log = out / "soak.log"
        self.events = out / "events.jsonl"
        self.data = data
        self.write()

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.write()

    def write(self) -> None:
        staged = self.path.with_name(f".{self.path.name}.{os.getpid()}")
        staged.write_text(json.dumps(self.data, indent=1) + "\n")
        staged.replace(self.path)

    def say(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}"
        with self.log.open("a") as log:
            log.write(line + "\n")
        with contextlib.suppress(
            OSError
        ):  # the soak goes on when the ssh session that started it closes
            print(line, file=sys.stderr, flush=True)

    def event(self, event: Mapping[str, Any], at: float | None = None) -> None:
        entry = {"at": round(time.time() if at is None else at, 3), **event}
        with self.events.open("a") as events:
            events.write(json.dumps(entry) + "\n")

    def fail(self, exc: BaseException) -> None:
        error = str(exc) if isinstance(exc, E2Error) else f"{type(exc).__name__}: {exc}"
        self.data |= {"status": "failed", "error": error, "finished": time.time()}
        self.write()
        self.say(f"failed: {error}")

    def finish(self, ended_by: str) -> None:
        self.data |= {"status": "complete", "ended_by": ended_by, "finished": time.time()}
        self.write()


class Soak:
    """One soak into ``out`` (the module's docstring)."""

    def __init__(
        self,
        options: SoakOptions,
        out: Path,
        host: Host,
        systemd: SoakSystemd,
        tools: Tools,
        guard_runner: Any = subprocess.run,
    ) -> None:
        self.options = options
        self.out = out
        self.host = host
        self.systemd = systemd
        self.tools = tools
        self.guard_runner = guard_runner
        self.stopping: str | None = None
        """The signal that ends the soak early, once one came."""
        self.hurry = False
        self.ended_by: str | None = None
        self.window: tuple[float, float] = (0.0, 0.0)
        self.calls: ToolRun | None = None
        self.agents: ToolRun | None = None
        self.restart_at: dict[str, float] = {}
        self.quick: dict[str, int] = {"calls": 0, "agents": 0}
        self.paused = False
        self.signals = 0
        self.node_lines: dict[tuple[str, float, str, str], float] = {}
        """The GPU nodes' journal lines already kept, by node, time, unit and message, with when: a check asks a
        little before the last one, against the hosts' clocks differing (OVERLAP_S)."""

    # Signals

    def _signal(self, signum: int, _frame: object) -> None:
        """SIGINT or SIGTERM: the first ends the soak, the second hurries its end, the third aborts it. The handler
        only sets the flags; the loop and the waits say what they do of them."""
        self.signals += 1
        if self.signals == 1:
            self.stopping = signal.Signals(signum).name
        elif self.signals == 2:
            self.hurry = True
        else:
            raise KeyboardInterrupt(signal.Signals(signum).name)

    # The soak

    def run(self) -> dict[str, Any]:
        options, host = self.options, self.host
        self.record = SoakRecord(
            self.out,
            {
                "kind": "soak",
                "status": "running",
                "host": platform.node().split(".")[0],
                "started": time.time(),
                "options": _options_record(options),
                "versions": versions(host, options.gpu_source),
                "calls_runs": [],
                "agents_runs": [],
                "pauses": [],
            },
        )
        previous = {s: signal.signal(s, self._signal) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            if leftovers := kill_left(tool_scope(host)):
                self.record["leftovers_killed"] = leftovers
                self.record.say(
                    f"killed {len(leftovers)} processes left by an earlier run: {leftovers}"
                )
            self.record["serving"] = self.preflight()
            with contextlib.ExitStack() as running:
                running.callback(self._stop_tools)  # whatever ends the soak, its tools stop
                self._measure()
            summary = self._summarize()
            self.record.finish(self.ended_by or "hours")
            self.record.say(f"soak complete ({self.ended_by}): {self.out}")
            return summary
        except BaseException as exc:
            self.record.fail(exc)
            with contextlib.suppress(Exception):
                self._summarize()
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def _measure(self) -> None:
        options = self.options
        start = time.time()
        self.scraper = Scraper(
            self.out,
            self.host.gateway_socket,
            self.nodes,
            self.with_nodes,
            lambda: read_worker_status(self.status_file),
            self.systemd,
            start,
            self.hidden,
        )
        self._scrape(start, "before")
        if options.agents_sessions:
            self._start_agents(first=True)
        if self.stopping:
            self.ended_by = self.stopping
            return
        start = time.time()
        self.window = (start, start + options.hours * 3600)
        self.record["window"] = {"start": start, "end": self.window[1], "hours": options.hours}
        self.record.event({"event": "window_started", "hours": options.hours})
        self.record.say(f"the window starts: {options.hours:g} h of calls and agents")
        self._scrape(start, "window")
        self._start_calls()
        self._loop()
        self.record.event({"event": "window_ended", "by": self.ended_by})
        self._wind_down()
        self._scrape(time.time(), "after")

    def _loop(self) -> None:
        options = self.options
        start, end = self.window
        scrape_at, guard_at, hour_at = (
            start + options.scrape_s,
            start + options.guard_s,
            start + summaries.HOUR_S,
        )
        while True:
            now = time.time()
            if self.stopping:
                self.ended_by = self.stopping
                self.record.say(
                    f"{self.stopping}: the soak ends; each call ends with its current turn"
                )
                break
            if now >= end:
                self.ended_by = "hours"
                break
            if now >= scrape_at:
                self._scrape(now)
                scrape_at += options.scrape_s * max(
                    1, math.ceil((now - scrape_at) / options.scrape_s + 1e-9)
                )
            if now >= guard_at:
                self._guard(now)
                guard_at = now + options.guard_s
            if now >= hour_at:
                self._hour(hour_at)
                hour_at += summaries.HOUR_S
            self._tend(now)
            time.sleep(max(0.0, min(POLL_S, min(scrape_at, guard_at, end) - time.time())))

    # Preflight

    def preflight(self) -> dict[str, Any]:
        """Check that the box serves as it would for real callers, and that the soak can load and read it; returns
        what the record keeps of how it serves."""
        options, host, systemd = self.options, self.host, self.systemd
        env = systemd.environment(WORKER)
        mode = env.get("FLATSHARE_VOICE_MODE", "bench")
        if mode != "serve":
            raise E2Error(
                f"{WORKER} runs {mode} mode, and a soak measures the serving path, whose admission refuses calls "
                "past the serving profile's: link deploy/cyan/variants/serve (ops/RUNBOOK.md, Soaks)"
            )
        left = [
            f"{unit}.service.d/variant-{name}.conf"
            for unit, names in E2_DROP_INS.items()
            for name in names
            if (link := host.units / f"{unit}.service.d" / f"variant-{name}.conf").exists()
            or link.is_symlink()
        ]
        if left:
            raise E2Error(
                f"cyan's units carry an E2 point's drop-ins ({', '.join(left)}), so the box does not serve as "
                "deployed: flatshare-bench e2 reset removes them"
            )
        self.status_file = Path(
            env.get("FLATSHARE_VOICE_STATUS")
            or f"{os.environ.get('XDG_RUNTIME_DIR') or f'/run/user/{os.getuid()}'}/flatshare/voice-worker.json"
        )
        status = read_worker_status(self.status_file)
        if status is None or status["age_s"] is None or status["age_s"] > STALE_S:
            why = "has none" if status is None else f"its last is {status['age_s']} s old"
            raise E2Error(
                f"in serve mode the voice worker writes its status to {self.status_file} every second, and it {why}: "
                f"is {WORKER} running?"
            )
        try:
            config = tomllib.loads((host.config / "gateway.toml").read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise E2Error(f"{host.config / 'gateway.toml'}: {exc}") from None
        self.nodes, self.with_nodes = gateway_nodes(config)
        ready = self._get("/ready")
        if ready.status_code != 200:
            raise E2Error(
                f"the gateway's /ready answered {ready.status_code} ({ready.text.strip()[:120]}): a soak starts on a "
                "box that serves"
            )
        profile = self._serving_profile(env)
        self.model = options.model or (profile or {}).get("model") or _profile_model(host)
        models = self._get("/v1/models")
        served: list[Any] = []
        with contextlib.suppress(ValueError, AttributeError, TypeError):
            served = (
                [m.get("id") for m in models.json().get("data", [])]
                if models.status_code == 200
                else []
            )
        if self.model not in served:
            raise E2Error(
                f"the gateway serves {', '.join(map(str, served)) or 'no model'} (/v1/models: HTTP "
                f"{models.status_code}), and the soak's agents ask for {self.model}: pass --model"
            )
        corpus = host.root / "data" / "callers"
        if not (corpus / "manifest.json").is_file():
            raise E2Error(
                f"no callers' corpus at {corpus} (flatshare-voice render-callers writes one)"
            )
        if options.agents_sessions:
            data = host.root / "data" / options.agents_data
            if not data.is_dir():
                raise E2Error(
                    f"no trace corpus at {data} (flatshare-bench scale-traces writes one)"
                )
            if not (CONFIGS / "agentic_replay.yaml").is_file():
                raise E2Error(
                    f"no AIPerf configuration in {CONFIGS}: install the package from its source tree"
                )
            agents_key(host)  # refuses a host without the agents tenant's key
        unknown = sorted(set(options.guards) - set(self.nodes))
        if unknown:
            raise E2Error(
                f"--guard names {', '.join(unknown)}, and the gateway fronts {', '.join(self.nodes)}"
            )
        methods = {
            node: options.guards.get(node, "local" if node == "localhost" else "ssh")
            for node in self.nodes
        }
        guards = {
            node: node_guard(node, method, host, self.guard_runner)
            for node, method in methods.items()
        }
        started = self.record["started"]
        for guard in guards.values():
            guard.since = started  # the nodes' journals from the soak's start
        self.guard = Guard(guards, options.guard_clear_s, options.guard_blind_after)
        self.hidden = _targets(host)
        checks = self.guard.check()
        for check in checks:
            self._log_check(check)
            if check.verdict == "taken":
                raise E2Error(
                    f"{check.node}'s GPU is in use by {check.describe()}, outside the box's units: a soak does not "
                    "start beside another's work"
                )
            if check.verdict == "blind":
                raise E2Error(
                    f"the guard cannot read {check.node}'s GPU ({check.error}): authorize the guard key from a "
                    f"workstation (deploy/authorize-guard.sh cyan {check.node}), or pass --guard {check.node}=off"
                )
        unguarded = [node for node, method in methods.items() if method == "off"]
        if unguarded:
            self.record.say(
                f"no guard on {', '.join(unguarded)} (--guard NODE=off): its GPU is not checked"
            )
        audio = env.get("FLATSHARE_AUDIO") or status.get("audio")
        if audio == "microphone":
            self.record.say(
                "the worker runs the microphone configuration (the end-of-turn model, LiveKit's AEC warm-up): the "
                "synthetic callers' V2V holds its endpointing, and their SLO's reference must be a voice-alone soak "
                "of the same configuration"
            )
        return {
            "worker": {key: env[key] for key in WORKER_SETTINGS if key in env},
            "worker_status": status,
            "serving_profile": profile,
            "gateway": redacted(config),
            "nodes": self.nodes,
            "with_nodes": self.with_nodes,
            "model": self.model,
            "guards": [guard.record() for guard in guards.values()],
            "first_checks": [check.record() for check in checks],
        }

    def _get(self, path: str) -> httpx.Response:
        try:
            with client(self.host.gateway_socket, timeout=10.0) as http:
                return http.get(path)
        except httpx.HTTPError as exc:
            raise E2Error(
                f"GET {path} on the gateway's socket {self.host.gateway_socket}: {exc}"
            ) from None

    def _serving_profile(self, env: Mapping[str, str]) -> dict[str, Any] | None:
        """What the record keeps of the serving profile the worker runs (its serving.json): its name, the node and
        variant, the model, make-room's settings and the voice table."""
        path = env.get("FLATSHARE_SERVING_PROFILE")
        if not path:
            return None
        try:
            profile = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            raise E2Error(f"the worker's serving profile {path}: {exc}") from None
        engine = profile.get("engine") or {}
        return {
            "profile": profile.get("profile"),
            "node": profile.get("node"),
            "variant": profile.get("variant"),
            "model": (engine.get("model") or {}).get("model"),
            "scheduler": engine.get("scheduler"),
            "make_room": engine.get("make_room"),
            "voice": profile.get("voice"),
        }

    # The scrapes, the guard and the hours

    def _scrape(self, at: float, label: str | None = None) -> None:
        _, events = self.scraper.scrape(at, label)
        for event in events:
            self.record.event(event, at)
            if event["event"] in ("scrape_failed", "counter_reset", "unit_changed", "units_unread"):
                self.record.say(
                    f"{event['event']}: "
                    + ", ".join(f"{k} {v}" for k, v in event.items() if k != "event")
                )

    def _log_check(self, check: Any) -> None:
        """Keep a check in guard.jsonl, and its node's journal lines in units.jsonl beside cyan's, each unit named
        after its node (``phastform:flatshare-engine.service``), each line once."""
        with (self.out / "guard.jsonl").open("a") as checks:
            checks.write(json.dumps(check.record()) + "\n")
        fresh = []
        for line in check.journal:
            key = (check.node, float(line["at"]), str(line["unit"]), str(line["message"]))
            if key not in self.node_lines:
                self.node_lines[key] = check.at
                fresh.append(line | {"unit": f"{check.node}:{line['unit']}"})
        # Lines older than the overlap cannot come again.
        for key in [k for k, seen in self.node_lines.items() if seen < check.at - 600]:
            del self.node_lines[key]
        if fresh:
            with (self.out / "units.jsonl").open("a") as journal:
                for line in fresh:
                    journal.write(json.dumps(line) + "\n")

    def _guard(self, now: float) -> None:
        checks = self.guard.check(now)
        for check in checks:
            self._log_check(check)
        for event in self.guard.observe(checks):
            self.record.event(event, now)
            self.record.say(
                f"{event['event']}: {event['node']}"
                + (f" ({event.get('error')})" if event.get("error") else "")
            )
        if self.guard.paused and not self.paused:
            self._pause(self.guard.why())
        elif not self.guard.paused and self.paused:
            self._resume()

    def _pause(self, why: str) -> None:
        self.record.say(f"the load pauses: {why}")
        self._stop_calls(drain=False, hang_up=True)
        self._stop_agents()
        self.paused = True
        self.record["pauses"] = [
            *self.record["pauses"],
            {"start": time.time(), "end": None, "why": why},
        ]
        self.record.event({"event": "load_paused", "why": why})

    def _resume(self) -> None:
        pauses = self.record["pauses"]
        pauses[-1]["end"] = time.time()
        self.record["pauses"] = pauses
        self.paused = False
        self.record.event({"event": "load_resumed"})
        self.record.say("every GPU is clear again: the load resumes")
        self.restart_at = {}
        self.quick = {"calls": 0, "agents": 0}

    def _hour(self, at: float) -> None:
        hour = round((at - self.window[0]) / summaries.HOUR_S) - 1
        summary = self._summarize()
        if 0 <= hour < len(summary["hours"]):
            self.record.say(f"hour {hour}: {render_hour(summary['hours'][hour])}")

    def _summarize(self) -> dict[str, Any]:
        summary = summarize(
            load(self.out), self.options.slo_reference_ms, self.options.slo_margin_ms
        )
        write_summary(self.out, summary)
        return summary

    # The tools

    def _tend(self, now: float) -> None:
        """Start again a tool whose run ended before the window does, once its backoff has passed."""
        if self.paused:
            return
        end = self.window[1]
        if self.calls is not None and self.calls.process.exited():
            self._ended("calls")
        if self.agents is not None and self.agents.process.exited():
            self._ended("agents")
        left = end - now
        if self.calls is None and left > LEFT_S and now >= self.restart_at.get("calls", 0.0):
            self._start_calls()
        if (
            self.options.agents_sessions
            and self.agents is None
            and left > LEFT_S
            and now >= self.restart_at.get("agents", 0.0)
        ):
            self._start_agents(first=False)
        if self.agents is not None and "profiling" not in self.agents.info:
            self._profiling(self.agents)

    def _ended(self, kind: str) -> None:
        """A tool's run that ended by itself: recorded, and its next run's backoff set."""
        run = self.calls if kind == "calls" else self.agents
        assert run is not None
        status = run.process.stop()
        now = time.time()
        planned = run.planned_end is not None and now >= run.planned_end - 5 and status == 0
        self._end_run(run, status, "planned" if planned else "ended")
        if kind == "calls":
            self.calls = None
        else:
            self.agents = None
        if planned:
            self.quick[kind] = 0
            self.restart_at[kind] = now
            return
        quick = now - run.info["started"] < QUICK_S
        self.quick[kind] = self.quick[kind] + 1 if quick else 0
        wait = (
            RESTART_BACKOFF_S[min(self.quick[kind], len(RESTART_BACKOFF_S)) - 1]
            if quick
            else RESTART_BACKOFF_S[0]
        )
        self.restart_at[kind] = now + wait
        self.record.event(
            {
                "event": f"{kind}_ended",
                "run": run.info["run"],
                "exit_status": status,
                "restart_in_s": wait,
            }
        )
        self.record.say(
            f"{run.process.name} run {run.info['run']} ended (exit status {status}); again in {wait:g} s"
        )

    def _end_run(self, run: ToolRun, status: int, how: str) -> None:
        info = run.info | {"ended": time.time(), "exit_status": status, "how": how}
        if run.kind == "calls":
            summary = run.directory / "summary.json"
            if summary.exists():
                with contextlib.suppress(ValueError):
                    counts = json.loads(summary.read_text())
                    info["summary"] = {
                        k: counts.get(k) for k in ("attempts", "outcomes", "turns", "errors")
                    }
        else:
            info["export"] = aiperf_export(run.directory / "aiperf")
            if run.scratch is not None:
                shutil.rmtree(run.scratch, ignore_errors=True)
        if status not in (0, None):
            info["log_tail"] = _tail(run.process.log)
        key = "calls_runs" if run.kind == "calls" else "agents_runs"
        runs = self.record[key]
        runs[info["run"]] = info
        self.record[key] = runs

    def _next_call(self) -> int:
        """The number of the next run's first call: one past every call the soak's runs have numbered."""
        found = -1
        for path in self.out.glob("calls/run-*/*.jsonl"):
            field = "call" if path.name == "attempts.jsonl" else "caller"
            for line in path.read_text(errors="replace").splitlines():
                with contextlib.suppress(ValueError, AttributeError, TypeError):
                    found = max(found, int(json.loads(line).get(field)))
        return found + 1

    def _start_calls(self) -> None:
        options, host = self.options, self.host
        number = len(self.record["calls_runs"])
        directory = self.out / "calls" / f"run-{number:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        duration = max(0.0, self.window[1] - time.time())
        first_call = self._next_call()
        settings = {
            "--corpus": host.root / "data" / "callers",
            "--calls-per-hour": f"{options.calls_per_hour:g}",
            "--duration": f"{duration:.0f}",
            "--turns": options.turns,
            "--warmup-turns": options.warmup_turns,
            "--barge-in": f"{options.barge_in:g}",
            "--seed": options.caller_seed,
            "--run": number,
            "--first-call": first_call,
            "--dispatch-timeout": f"{options.dispatch_timeout_s:g}",
            "--max-open": options.max_open,
            "--worker-status": self.status_file,
            "--out": directory,
        }
        command = [*self.tools.calls(), *(str(word) for item in settings.items() for word in item)]
        env = os.environ | read_env(host.config / "livekit.env")
        process = Process(
            "flatshare-voice arrivals", command, directory / "arrivals.log", env, tool_scope(host)
        )
        info = {
            "run": number,
            "started": time.time(),
            "first_call": first_call,
            "duration_s": round(duration),
        }
        # No call arrives after the window's end, and the run ends once the calls then in progress have ended.
        self.calls = ToolRun("calls", process, info, directory, planned_end=self.window[1])
        self.record["calls_runs"] = [*self.record["calls_runs"], info]
        self.record.event({"event": "calls_started", "run": number, "first_call": first_call})
        self.record.say(
            f"calls run {number}: {options.calls_per_hour:g} an hour for {duration / 3600:.2f} h, from call {first_call}"
        )

    def _start_agents(self, first: bool) -> None:
        options = self.options
        number = len(self.record["agents_runs"])
        directory = self.out / "agents" / f"run-{number:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        now = time.time()
        left = (
            (self.window[1] - now) if self.window[1] else options.hours * 3600 + PROFILING_TIMEOUT_S
        )
        duration = math.ceil(left) + AGENTS_OUTLAST_S
        planned_end = None
        if options.agents_run_hours:
            run_s = math.ceil(options.agents_run_hours * 3600)
            if run_s < duration:
                duration, planned_end = run_s, now + run_s
        env = aiperf_env(self.host, directory, options, options.agents_seed, self.model, duration)
        # AIPerf 0.13.0 leaves its IPC directory and the dataset's memory map in the temp directory: a run's own,
        # removed once it stops (as e2 run's points do).
        scratch = Path(tempfile.mkdtemp(prefix="flatshare-aiperf-"))
        env["TMPDIR"] = str(scratch)
        process = Process(
            "AIPerf", self.tools.aiperf(), directory / "aiperf.log", env, tool_scope(self.host)
        )
        info = {
            "run": number,
            "started": now,
            "seed": options.agents_seed,
            "sessions": options.agents_sessions,
            "duration_s": duration,
        }
        self.agents = ToolRun("agents", process, info, directory, scratch, None)
        self.record["agents_runs"] = [*self.record["agents_runs"], info]
        self.record.event({"event": "agents_started", "run": number, "seed": options.agents_seed})
        self.record.say(
            f"AIPerf run {number}: {options.agents_sessions} session trees, seed {options.agents_seed}"
        )
        if not first:
            if planned_end is not None:
                self.agents.planned_end = planned_end + 30  # AIPerf's grace after its duration
            return
        deadline = time.monotonic() + PROFILING_TIMEOUT_S
        while not self._profiling(self.agents):
            if process.exited():
                raise process.failure("exited before its profiling phase")
            if time.monotonic() > deadline:
                raise process.failure(f"did not start profiling within {PROFILING_TIMEOUT_S:.0f} s")
            if self.stopping:
                return
            time.sleep(POLL_S)
        self.record.say(f"AIPerf is profiling; {options.agents_warmup_s:g} s of warm-up")
        warm = time.monotonic() + options.agents_warmup_s
        while time.monotonic() < warm and not self.stopping:
            if process.exited():
                raise process.failure("exited during the warm-up")
            time.sleep(min(POLL_S, max(0.0, warm - time.monotonic())))
        if planned_end is not None:
            self.agents.planned_end = (
                self.agents.info["profiling"] + options.agents_run_hours * 3600 + 30
            )

    def _profiling(self, run: ToolRun) -> bool:
        """Whether AIPerf's run is in its profiling phase, stamped in its record the first time."""
        if "profiling" in run.info:
            return True
        with contextlib.suppress(OSError):
            if PROFILING in run.process.log.read_text(errors="replace"):
                run.info["profiling"] = time.time()
                self.record.write()
                return True
        return False

    def _wait(self, process: Process, seconds: float) -> None:
        """Wait for ``process`` to exit, ``seconds`` at most, and no longer than HURRIED_S once the soak hurries."""
        deadline = time.monotonic() + seconds
        hurried = None
        while not process.exited() and time.monotonic() < deadline:
            if self.hurry:
                hurried = hurried or time.monotonic() + HURRIED_S
                if time.monotonic() > hurried:
                    return
            time.sleep(0.2)

    def _stop_calls(self, *, drain: bool, hang_up: bool = False) -> None:
        """Stop the callers' run: at the window's end (``drain``) after the calls in progress have played their turns,
        for ``drain_s`` at most; then SIGINT, after which each call ends with its current turn, and a second SIGINT,
        which hangs them up; ``hang_up`` (a guard's pause) sends the second soon after the first."""
        run = self.calls
        if run is None:
            return
        process = run.process
        if drain and not self.stopping:
            self.record.say(
                "the window ends: no call arrives, and the calls in progress play to their end"
            )
            self._wait(process, self.options.drain_s)
        # Ended by itself at the window's end, its calls played to their end: as planned. Any other end is the soak's.
        how = "planned" if drain and not self.stopping and process.exited() else "stopped"
        graces = (HURRIED_S if hang_up else CALLS_STOP_S, CALLS_HANG_UP_S)
        for grace in graces:
            if process.exited():
                break
            with contextlib.suppress(ProcessLookupError):
                os.kill(process.popen.pid, signal.SIGINT)
            self._wait(process, grace if not self.hurry else HURRIED_S)
        self._end_run(run, process.stop(5.0), how)
        self.calls = None

    def _stop_agents(self) -> None:
        run = self.agents
        if run is None:
            return
        self.record.say("stopping AIPerf")
        status = run.process.stop(HURRIED_S if self.hurry else AIPERF_STOP_S)
        self._end_run(run, status, "stopped")
        self.agents = None

    def _wind_down(self) -> None:
        self._stop_calls(drain=True)
        self._stop_agents()
        if self.paused:
            self._resume_record()

    def _resume_record(self) -> None:
        pauses = self.record["pauses"]
        if pauses and pauses[-1]["end"] is None:
            pauses[-1]["end"] = time.time()
            self.record["pauses"] = pauses

    def _stop_tools(self) -> None:
        """Stop whatever tool still runs: the soak's last step whatever ended it."""
        self.hurry = self.hurry or self.ended_by is None
        with contextlib.suppress(Exception):
            self._stop_calls(drain=False)
        with contextlib.suppress(Exception):
            self._stop_agents()
        self._resume_record()


def aiperf_export(artifacts: Path) -> dict[str, Any] | None:
    """What a soak's record keeps of a run of AIPerf's own summary (profile_export_aiperf.json): its requests, errors,
    tokens and duration, and its first token's p50 and p99; None without one."""
    path = artifacts / "profile_export_aiperf.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None

    def avg(key: str) -> Any:
        return (data.get(key) or {}).get("avg")

    ttft = data.get("time_to_first_token") or {}
    return {
        "request_count": avg("request_count"),
        "completed_request_count": avg("completed_request_count"),
        "request_error_rate": avg("request_error_rate"),
        "total_usage_prompt_tokens": avg("total_usage_prompt_tokens"),
        "total_usage_completion_tokens": avg("total_usage_completion_tokens"),
        "benchmark_duration_s": avg("benchmark_duration"),
        "ttft_ms": {key: ttft.get(key) for key in ("p50", "p99")} if ttft else None,
        "errors": [
            {"type": (e.get("error_details") or {}).get("type"), "count": e.get("count")}
            for e in data.get("error_summary") or []
            if isinstance(e, dict)
        ],
        "was_cancelled": data.get("was_cancelled"),
    }


def versions(host: Host, gpu_source: Mapping[str, str | None] | None) -> dict[str, Any]:
    """What ran: cyan's deployed commit and release, the GPU nodes' revisions where given (``--gpu-source``), and the
    tools' versions."""
    deployed = host.root / "src" / "DEPLOYED"
    current = host.root / "current"
    gateway = host.root / "bin" / "flatshare-gateway"
    flatshare_gateway = None
    if gateway.exists():
        with contextlib.suppress(OSError):
            shown = subprocess.run([gateway, "--version"], capture_output=True, text=True)
            flatshare_gateway = shown.stdout.strip() or None
    aiperf = None
    with contextlib.suppress(metadata.PackageNotFoundError):
        aiperf = metadata.version("aiperf")
    return {
        "source": deployed.read_text().strip() if deployed.exists() else None,
        "release": current.resolve().name if current.is_symlink() else None,
        "gpu_source": None if gpu_source is None else dict(gpu_source),
        "flatshare_bench": metadata.version("flatshare-bench"),
        "aiperf": aiperf,
        "flatshare_gateway": flatshare_gateway,
    }


def _profile_model(host: Host) -> str:
    """The model of the GPU node's hardware profile, as its engine.yaml serves it."""
    return flag_values(profile_flags(host.profile))["--model"]


def _targets(host: Host) -> list[str]:
    """The tunnels' SSH targets and their hosts, which no journal line a soak keeps may carry."""
    found: set[str] = set()
    for path in host.config.glob("tunnel-*.env"):
        with contextlib.suppress(OSError):
            if target := read_env(path).get("FLATSHARE_TUNNEL_TARGET"):
                found |= {target, target.rpartition("@")[2]}
    return sorted(found, key=len, reverse=True)


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _options_record(options: SoakOptions) -> dict[str, Any]:
    record = asdict(options)
    record["guards"] = dict(options.guards)
    return record


def fresh(out: Path) -> None:
    """Make ``out`` the soak's directory: a complete soak is never written over, an incomplete one is moved aside to
    ``out.incomplete-TIME``, and a directory that holds anything else is refused."""
    record = out / "soak.json"
    if record.exists():
        with contextlib.suppress(ValueError):
            if json.loads(record.read_text()).get("status") == "complete":
                raise E2Error(f"{out} holds a complete soak; soak run never writes over one")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out.rename(out.with_name(f"{out.name}.incomplete-{stamp}"))
    elif out.exists() and any(out.iterdir()):
        raise E2Error(f"{out} is not empty and holds no soak")
    out.mkdir(parents=True, exist_ok=True)


def run_soak(
    options: SoakOptions,
    out: Path,
    host: Host | None = None,
    systemd: SoakSystemd | None = None,
    tools: Tools | None = None,
    guard_runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Run a soak into ``out`` (the module's docstring) under cyan's lock, which no E2 point, reset or drill holds
    meanwhile; returns its summary."""
    if not hasattr(os, "waitid"):
        raise E2Error(NEEDS_WAITID.replace("e2 run", "soak run"))
    host = host or Host.load()
    systemd = systemd or SoakSystemd()
    with exclusive(host):
        fresh(out)
        soak = Soak(options, out, host, systemd, tools or Tools(host), guard_runner)
        return soak.run()
