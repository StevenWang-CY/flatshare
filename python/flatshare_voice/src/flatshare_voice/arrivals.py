"""Calls that arrive on a schedule for hours, every attempt recorded (``flatshare-voice arrivals``): a soak's callers.

``flatshare-voice bots`` runs a fixed number of callers for a fixed number of turns, all at once: E2's points. A soak
(``flatshare-bench soak``) needs calls as a phone line gets them: arriving one at a time at random moments, a Poisson
process of ``calls_per_hour``, each of ``turns`` turns with the callers' think times, so that some arrive while the
voice worker is full. Each call is one :class:`~flatshare_voice.bots.Caller`, whose turns are measured exactly as E2's
are (:meth:`~flatshare_voice.bots.Caller.turn`); what this module adds is when the calls come and how each attempt
ended (``outcome``):

- ``failed``: the caller could not join its room and publish its microphone within ``connect_timeout`` (LiveKit
  unreachable, the join refused): the service never saw the call;
- ``refused``: it joined, and no agent came within ``dispatch_timeout``. A worker in serve mode takes no call past its
  serving profile's ``max_calls`` (LiveKit sends a full worker no job, and the worker refuses one past it), none while
  the gateway says the engine does not serve and none while a release drains it, and LiveKit then dispatches the room
  to no one. Why is read from the worker's status file (``FLATSHARE_VOICE_STATUS``, which it writes every second) as
  the attempt was made and again as it was given up (:func:`refusal`);
- ``accepted``: an agent joined. The call then ``completed`` its turns, was ``dropped`` (the agent never listened, or
  left, or the room went away, before its last turn), or was ``cut`` by the run's end (``ended``);
- ``skipped``: the call was due while ``max_open`` attempts were still open, a bound of the load generator's own, never
  of the service's: it was not made;
- ``abandoned``: a second signal hung the attempt up before its outcome was known.

A run ends when ``duration_s`` has passed since its start: no call arrives after that, and the calls in progress play
their turns to the end. SIGINT or SIGTERM ends it early: no call arrives after it, and each call in progress ends with
its current turn (``cut``); a second signal hangs every call up at once. Either way the run writes its summary and
exits 0.

The run directory gets ``run.json`` (the configuration without the LiveKit secret, and when the run started and
ended), ``attempts.jsonl`` (a line per attempt as it ends), ``turns.jsonl`` (each turn's line as ``bots`` writes it,
with the run's number, ``run``, and when the turn ended on the wall clock, ``at``) and ``summary.json`` (the attempts
by outcome and reason, the turns, and the measured replies' voice-to-voice percentiles).

Calls are numbered from ``first_call`` up, through every run of a soak, and each call's lines, interruptions and
think times are drawn from the callers' seed and its number, as a caller's are in ``bots``; the arrival times are
drawn from the seed and the run's number (:func:`arrival_offsets`), so that a run started again after a pause does not
replay the times of the first.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import io
import itertools
import json
import math
import os
import platform
import random
import signal
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

from flatshare_stats import percentiles, replies, v2v_samples

from .audio import agents, until
from .bots import JOIN_STAGGER_S, BotsConfig, Caller, Corpus

OUTCOMES = ("accepted", "refused", "failed", "skipped", "abandoned")
"""How an attempt ended (the module's docstring)."""
ENDED = ("completed", "dropped", "cut")
"""How an accepted call ended."""
STALE_S = 5.0
"""A worker status older than this was not written by a running worker, which writes it every second."""
CONNECT_TIMEOUT_S = 30.0
"""How long joining a room may take: LiveKit's SDK tries each signal connection for 5 s, more than once."""
DISPATCH_TIMEOUT_S = 15.0
"""How long a caller waits for the agent to join its room: a worker with idle job processes joins in about a second,
and LiveKit dispatches a room once, so an agent that has not come by then will not."""


@dataclass(frozen=True)
class ArrivalsConfig:
    """A run of arriving calls; ``bots`` holds the callers' settings (its ``callers`` is not read: one per call)."""

    bots: BotsConfig
    calls_per_hour: float
    duration_s: float
    """Calls arrive for this long from the run's start."""
    run: int = 0
    """The run's number in its soak, from 0, which the arrival times are drawn with."""
    first_call: int = 0
    """The number of the run's first call: the soak's calls so far."""
    dispatch_timeout: float = DISPATCH_TIMEOUT_S
    connect_timeout: float = CONNECT_TIMEOUT_S
    max_open: int = 8
    """Attempts open at once at most; a call due beyond it is skipped."""
    worker_status: Path | None = None
    """The voice worker's status file (FLATSHARE_VOICE_STATUS); None reads no reason for a refusal."""

    def __post_init__(self) -> None:
        if not 0 < self.calls_per_hour < math.inf:
            raise ValueError(f"calls_per_hour must be above 0, got {self.calls_per_hour}")
        if not 0 <= self.duration_s < math.inf:
            raise ValueError(f"duration_s must be 0 or more seconds, got {self.duration_s}")
        if self.run < 0 or self.first_call < 0:
            raise ValueError("run and first_call cannot be negative")
        if self.max_open < 1:
            raise ValueError(f"max_open must be 1 or more, got {self.max_open}")
        if not (self.dispatch_timeout > 0 and self.connect_timeout > 0):
            raise ValueError("dispatch_timeout and connect_timeout are seconds above 0")


def arrival_offsets(
    calls_per_hour: float, duration_s: float, seed: int, run: int = 0
) -> Iterator[float]:
    """The arrival times of a Poisson process of ``calls_per_hour``, in seconds from a run's start, up to
    ``duration_s``: exponential gaps of mean 3,600 / ``calls_per_hour`` s, drawn from a generator of the seed and the
    run's number alone, so that the same run arrives at the same times whatever its calls meet."""
    rng = random.Random(f"arrivals-{seed}-{run}")
    rate = calls_per_hour / 3600.0
    t = 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= duration_s:
            return
        yield t


def read_status(path: Path | None, now: float | None = None) -> dict[str, Any] | None:
    """What the voice worker last wrote to its status file (admission.Monitor), with its age in seconds (``age_s``);
    None without the file, or with one that is not the worker's JSON."""
    if path is None:
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    keys = ("mode", "max_calls", "active_calls", "admitting", "reason", "refused", "pid")
    found = {key: data.get(key) for key in keys}
    updated = data.get("updated")
    found["age_s"] = (
        round((time.time() if now is None else now) - updated, 1)
        if isinstance(updated, (int, float))
        else None
    )
    return found


def refusal(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> tuple[str, str]:
    """Why no agent came to a room, from the worker's status as the attempt was made (``before``) and as it was given
    up (``after``): its reason and what the status said.

    - ``draining``: a release drains the worker; ``engine``: it takes no call while the gateway says the engine does
      not serve (either status says so);
    - ``capacity``: it was full (either status says "N of M calls"), or it refused a job request meanwhile (its count of
      refusals rose: a request it decided on between two writes of the file);
    - ``no_status``: no status file (a worker in bench mode writes none, and takes every call), or ``worker_down``: one
      too old to be a running worker's;
    - ``no_agent``: the worker admitted calls throughout and refused none: LiveKit gave the room no agent (the
      dispatch was lost).
    """
    seen = [status for status in (before, after) if status is not None]
    if not seen:
        return "no_status", "the worker wrote no status file: it does not run serve mode"
    fresh = [s for s in seen if isinstance(s.get("age_s"), (int, float)) and s["age_s"] <= STALE_S]
    if not fresh:
        return (
            "worker_down",
            f"the worker's status is {seen[-1].get('age_s')} s old: no worker writes it",
        )
    closed = [str(s.get("reason") or "") for s in fresh if not s.get("admitting")]
    for reason in closed:
        if reason.startswith("draining"):
            return "draining", reason
        if "does not serve" in reason:
            return "engine", reason
    if closed:
        return "capacity", closed[0]
    first, last = ((status or {}).get("refused") for status in (before, after))
    if isinstance(first, int) and isinstance(last, int) and last > first:
        return "capacity", f"the worker refused {last - first} job request(s) meanwhile"
    return "no_agent", f"the worker admitted calls ({fresh[-1].get('reason')}) and refused none"


class _TurnLog(io.TextIOBase):
    """turns.jsonl, to which a caller writes its turns as ``bots`` does (:meth:`Caller.play`): each line gains the run's
    number first and, last, when the turn ended on the wall clock."""

    def __init__(self, out: TextIO, run: int) -> None:
        self.out = out
        self.run = run

    def write(self, text: str) -> int:
        for line in text.splitlines():
            if line.strip():
                row = {"run": self.run, **json.loads(line), "at": round(time.time(), 3)}
                self.out.write(json.dumps(row) + "\n")
        return len(text)

    def flush(self) -> None:
        self.out.flush()


class CallerLike(Protocol):
    """What a run needs of a call's caller (:class:`~flatshare_voice.bots.Caller`; the tests' fakes)."""

    room_name: str
    max_lateness_s: float

    @property
    def room(self) -> Any: ...

    async def join(self, tasks: set[asyncio.Task[None]]) -> None: ...

    async def listening(self) -> None: ...

    async def play(
        self, rows: list[dict[str, Any]], log: TextIO, go_on: Callable[[], bool] | None = None
    ) -> None: ...

    async def hang_up(self, tasks: set[asyncio.Task[None]]) -> None: ...


class Arrivals:
    """One run of arriving calls into ``out``; ``caller`` makes each call's caller (``Caller(config, corpus, number,
    run_id)``)."""

    def __init__(
        self,
        config: ArrivalsConfig,
        out: Path,
        corpus: Any | None = None,
        caller: Callable[..., CallerLike] = Caller,
    ) -> None:
        self.config = config
        self.out = out
        self.corpus = corpus
        self.make_caller = caller
        self.run_id = uuid.uuid4().hex[:8]
        self.stopping: str | None = None
        """The signal that ends the run early, once one came."""
        self._stop = asyncio.Event()
        self._open: set[asyncio.Task[dict[str, Any]]] = set()
        self.records: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []

    def _signal(self, signum: int) -> None:
        name = signal.Signals(signum).name
        if self.stopping is None:
            self.stopping = name
            self._stop.set()
            print(
                f"{name}: no call arrives from now; each call ends with its current turn",
                flush=True,
            )
            return
        print(f"{name} again: every call hangs up now", flush=True)
        for task in self._open:
            task.cancel()

    async def _until(self, at: float) -> bool:
        """Wait until monotonic time ``at``; False once the run is to stop."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), max(0.0, at - time.monotonic()))
        return not self._stop.is_set()

    async def run(self) -> dict[str, Any]:
        config = self.config
        self.out.mkdir(parents=True, exist_ok=True)
        if self.corpus is None:
            self.corpus = Corpus(config.bots.corpus)
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, self._signal, signum)
        started, start = time.time(), time.monotonic()
        run = {
            "kind": "voice-arrivals",
            "config": _config_record(config),
            "host": platform.node(),
            "run_id": self.run_id,
            "started": started,
        }
        _write_json(self.out / "run.json", run)
        try:
            with (
                (self.out / "attempts.jsonl").open("a") as attempts,
                (self.out / "turns.jsonl").open("a") as turns,
            ):
                await self._arrive(start, attempts, _TurnLog(turns, config.run))
        finally:
            for signum in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(signum)
        summary = summarize_run(self.records, self.rows, config.bots.reply_timeout)
        run |= {"finished": time.time(), "stopped_by": self.stopping}
        _write_json(self.out / "run.json", run)
        _write_json(self.out / "summary.json", summary)
        return summary

    async def _arrive(self, start: float, attempts: TextIO, turns: _TurnLog) -> None:
        config = self.config
        offsets = arrival_offsets(
            config.calls_per_hour, config.duration_s, config.bots.seed, config.run
        )
        last_join = -math.inf
        for number, offset in zip(itertools.count(config.first_call), offsets, strict=False):
            due = start + offset
            # LiveKit's dispatcher drops the job of a room created within milliseconds of another's (bots'
            # JOIN_STAGGER_S): a call due that soon after the last one joins a stagger later, which its lateness says.
            if not await self._until(max(due, last_join + JOIN_STAGGER_S)):
                break
            wall_due = time.time() - (time.monotonic() - due)
            if len(self._open) >= config.max_open:
                self._done(attempts, _skipped(config.run, number, wall_due, len(self._open)))
                continue
            last_join = time.monotonic()
            task = asyncio.create_task(self._attempt(number, wall_due, turns))
            self._open.add(task)
            task.add_done_callback(functools.partial(self._finished, attempts, number, wall_due))
        while self._open:
            await asyncio.wait(set(self._open))

    def _finished(
        self, attempts: TextIO, number: int, due: float, task: asyncio.Task[dict[str, Any]]
    ) -> None:
        self._open.discard(task)
        if task.cancelled():  # hung up before it began
            record = {"run": self.config.run, "call": number, "due": round(due, 3)}
            self._done(attempts, record | {"outcome": "abandoned", "reason": "hung_up"})
        else:
            self._done(attempts, task.result())

    def _done(self, attempts: TextIO, record: dict[str, Any]) -> None:
        self.records.append(record)
        attempts.write(json.dumps(record) + "\n")
        attempts.flush()
        say = record["outcome"] + (
            f" ({record['reason']}: {record.get('detail')})" if "reason" in record else ""
        )
        if record["outcome"] == "accepted":
            say += f", {record.get('ended')} after {record.get('turns', 0)} turns"
        print(f"call {record['call']}: {say}", flush=True)

    def _go_on(self, caller: CallerLike) -> bool:
        """Whether a call plays its next turn: not once the run is to stop, nor once the agent or the room is gone."""
        room = caller.room
        return self.stopping is None and room.isconnected() and bool(agents(room))

    async def _attempt(self, number: int, due: float, turns: _TurnLog) -> dict[str, Any]:
        config = self.config
        record: dict[str, Any] = {
            "run": config.run,
            "call": number,
            "due": round(due, 3),
            "started": round(time.time(), 3),
        }
        record["late_ms"] = round((record["started"] - due) * 1000, 1)
        status = config.worker_status
        before = read_status(status)
        record["worker"] = before
        caller = self.make_caller(config.bots, self.corpus, number, self.run_id)
        record["room"] = caller.room_name
        tasks: set[asyncio.Task[None]] = set()
        rows: list[dict[str, Any]] = []
        try:
            try:
                await asyncio.wait_for(caller.join(tasks), config.connect_timeout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # LiveKit's ConnectError, a timeout, a refused publication
                record |= {"outcome": "failed", "reason": "connect", "detail": _describe(exc)}
                return record
            joined = time.monotonic()
            try:
                await until(lambda: bool(agents(caller.room)), config.dispatch_timeout, "no agent")
            except TimeoutError:
                after = read_status(status)
                reason, detail = refusal(before, after)
                record |= {"outcome": "refused", "reason": reason, "detail": detail}
                record["worker_after"] = after
                return record
            record |= {
                "outcome": "accepted",
                "dispatch_ms": round((time.monotonic() - joined) * 1000, 1),
            }
            try:
                await caller.listening()
            except TimeoutError as exc:
                record |= {"ended": "dropped", "reason": "never_listened", "detail": str(exc)}
                return record
            record["listening_ms"] = round((time.monotonic() - joined) * 1000, 1)
            await caller.play(rows, turns, go_on=lambda: self._go_on(caller))
            record |= _ended(rows, config.bots, self.stopping, caller.room)
        except asyncio.CancelledError:
            # A second signal: the run hangs up now. The attempt keeps what it measured.
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            if record.get("outcome") == "accepted":
                record |= {"ended": "cut", "reason": "hung_up"}
            else:
                record |= {"outcome": "abandoned", "reason": "hung_up"}
        except Exception as exc:
            # A fault of the caller's own: the attempt says so, and the run goes on.
            record.setdefault("outcome", "failed")
            record |= {"reason": "error", "detail": _describe(exc)}
            if record["outcome"] == "accepted":
                record["ended"] = "dropped"
        finally:
            with contextlib.suppress(Exception):
                await caller.hang_up(tasks)
            record["caller_max_lateness_ms"] = round(caller.max_lateness_s * 1000, 1)
            record["ended_at"] = round(time.time(), 3)
            record.setdefault("turns", len(rows))
            record.setdefault("errors", sum(1 for row in rows if "error" in row))
            self.rows += rows
        return record


def _ended(
    rows: list[dict[str, Any]], bots: BotsConfig, stopping: str | None, room: Any
) -> dict[str, Any]:
    """How an accepted call that played its turns, or fewer, ended."""
    played = {"turns": len(rows), "errors": sum(1 for row in rows if "error" in row)}
    if len(rows) >= bots.warmup_turns + bots.turns:
        return played | {"ended": "completed"}
    if stopping is not None:
        return played | {"ended": "cut", "reason": stopping}
    why = "the room disconnected" if not room.isconnected() else "the agent left the room"
    return played | {"ended": "dropped", "reason": "agent_left", "detail": why}


def _skipped(run: int, number: int, due: float, open_: int) -> dict[str, Any]:
    return {
        "run": run,
        "call": number,
        "due": round(due, 3),
        "outcome": "skipped",
        "reason": "max_open",
        "detail": f"{open_} attempts were open",
    }


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def summarize_run(
    records: list[dict[str, Any]], rows: list[dict[str, Any]], timeout_s: float
) -> dict[str, Any]:
    """A run's attempts by outcome, its refusals and failures by reason, how its accepted calls ended, and its turns:
    the measured ones (warm-up turns left out), their failures, and the replies' voice-to-voice percentiles, a reply
    that never came censored at the reply timeout, as ``bots`` gives them."""
    outcomes = Counter(record["outcome"] for record in records)
    by_reason = {
        outcome: dict(
            sorted(Counter(r.get("reason") for r in records if r["outcome"] == outcome).items())
        )
        for outcome in ("refused", "failed")
    }
    measured = [row for row in rows if not row.get("warmup")]
    v2v, censored = v2v_samples(measured, timeout_s * 1000)
    return {
        "attempts": sum(outcomes[o] for o in ("accepted", "refused", "failed")),
        "outcomes": {outcome: outcomes[outcome] for outcome in OUTCOMES},
        "refused_by": by_reason["refused"],
        "failed_by": by_reason["failed"],
        "ended": {e: sum(1 for r in records if r.get("ended") == e) for e in ENDED},
        "turns": len(measured),
        "errors": sum(1 for row in measured if "error" in row),
        "v2v_ms": percentiles(v2v, censored),
        "replies": replies(measured),
        "caller_max_lateness_ms": max(
            (r.get("caller_max_lateness_ms", 0.0) for r in records), default=0.0
        ),
    }


def _config_record(config: ArrivalsConfig) -> dict[str, Any]:
    record = asdict(config)
    record["bots"] |= {"api_secret": "<redacted>", "corpus": str(config.bots.corpus)}
    record["worker_status"] = None if config.worker_status is None else str(config.worker_status)
    return record


def _write_json(path: Path, data: Any) -> None:
    staged = path.with_name(f".{path.name}.{os.getpid()}")
    staged.write_text(json.dumps(data, indent=1) + "\n")
    staged.replace(path)


async def run_arrivals(config: ArrivalsConfig, out: Path) -> dict[str, Any]:
    """Run the calls of ``config`` into ``out`` (the module's docstring); returns the run's summary."""
    return await Arrivals(config, out).run()


def default_status_file(env: Mapping[str, str] = os.environ) -> Path:
    """The voice worker's status file as its unit names it (FLATSHARE_VOICE_STATUS), or where the unit puts it."""
    if path := env.get("FLATSHARE_VOICE_STATUS"):
        return Path(path)
    runtime = env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(runtime) / "flatshare" / "voice-worker.json"
