"""``flatshare-voice arrivals``: the calls' Poisson arrivals, the accounting of every attempt (accepted, refused and
why, failed, skipped), the turns' lines, and the end of a run on SIGINT, against fake callers that meet what a test
plans for each call."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import signal
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from livekit import rtc

from flatshare_voice import arrivals, audio, bots, cli
from flatshare_voice.arrivals import (
    Arrivals,
    ArrivalsConfig,
    arrival_offsets,
    read_status,
    refusal,
    summarize_run,
)
from flatshare_voice.audio import AGENT_STATE
from flatshare_voice.bots import BotsConfig

from .test_callers_and_bots import Line, line_clip

AGENT = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT


# The arrival schedule


def test_the_calls_arrive_as_a_poisson_process_of_the_rate() -> None:
    # 60 calls an hour over 2,000 hours: a gap's mean is 60 s, its spread as large as its mean (exponential gaps),
    # and as many calls arrive in each half of the time.
    times = list(arrival_offsets(60.0, 2000 * 3600.0, seed=0))
    gaps = [b - a for a, b in zip([0.0, *times], times, strict=False)]
    assert len(times) == pytest.approx(120_000, rel=0.02)
    assert statistics.fmean(gaps) == pytest.approx(60.0, rel=0.02)
    assert statistics.stdev(gaps) == pytest.approx(60.0, rel=0.03)
    # Memoryless: as many gaps over a minute after the first minute as over a minute from 0, in proportion.
    over_60 = [g - 60 for g in gaps if g > 60]
    assert len(over_60) / len(gaps) == pytest.approx(1 / 2.71828, rel=0.03)
    assert statistics.fmean(over_60) == pytest.approx(60.0, rel=0.03)
    assert all(0 < t < 2000 * 3600 for t in times) and times == sorted(times)


def test_a_runs_arrival_times_are_its_own() -> None:
    first = list(arrival_offsets(30.0, 3 * 3600.0, seed=4, run=0))
    assert first == list(
        arrival_offsets(30.0, 3 * 3600.0, seed=4, run=0)
    )  # the same run, the same times
    assert first != list(
        arrival_offsets(30.0, 3 * 3600.0, seed=4, run=1)
    )  # a run started again: others
    assert first != list(arrival_offsets(30.0, 3 * 3600.0, seed=5, run=0))
    # A shorter run arrives at the longer one's times, up to its end.
    assert list(arrival_offsets(30.0, 3600.0, seed=4)) == [t for t in first if t < 3600.0]
    assert list(arrival_offsets(30.0, 0.0, seed=4)) == []


def test_a_run_refuses_settings_no_call_can_arrive_with(tmp_path: Path) -> None:
    bots = BotsConfig("ws://fake", "k", "s", tmp_path)
    for wrong, says in [
        ({"calls_per_hour": 0.0}, "calls_per_hour must be above 0"),
        ({"duration_s": -1.0}, "duration_s must be 0 or more"),
        ({"max_open": 0}, "max_open must be 1 or more"),
        ({"run": -1}, "run and first_call cannot be negative"),
        ({"dispatch_timeout": 0.0}, "are seconds above 0"),
    ]:
        with pytest.raises(ValueError, match=says):
            ArrivalsConfig(bots, **({"calls_per_hour": 10.0, "duration_s": 60.0} | wrong))


# The worker's status and the reason of a refusal


def status(**fields: Any) -> dict[str, Any]:
    """The worker's status as read_status gives it: serve mode with two calls admitted, written a moment ago."""
    return {
        "mode": "serve",
        "max_calls": 2,
        "active_calls": 1,
        "admitting": True,
        "reason": "1 of 2 calls",
        "refused": 0,
        "pid": 7,
        "age_s": 0.4,
    } | fields


@pytest.mark.parametrize(
    ("before", "after", "reason", "detail"),
    [
        # Full as the call was made, or by the time it gave up, or a job request refused between two writes.
        (
            status(admitting=False, active_calls=2, reason="2 of 2 calls"),
            status(),
            "capacity",
            "2 of 2 calls",
        ),
        (status(), status(admitting=False, reason="2 of 2 calls"), "capacity", "2 of 2 calls"),
        (status(refused=3), status(refused=4), "capacity", "refused 1 job request(s) meanwhile"),
        # The gate: an engine the gateway says does not serve, a release's drain; before full calls.
        (
            status(admitting=False, reason="the engine does not serve: /ready answered 503"),
            status(admitting=False, reason="2 of 2 calls"),
            "engine",
            "the engine does not serve: /ready answered 503",
        ),
        (
            status(admitting=False, reason="draining: /run/x.drain exists"),
            None,
            "draining",
            "draining",
        ),
        # No worker writes it: none in bench mode, or a stale one.
        (None, None, "no_status", "does not run serve mode"),
        (status(age_s=40.0), status(age_s=55.0), "worker_down", "55.0 s old"),
        (status(age_s=40.0), status(admitting=False, reason="2 of 2 calls"), "capacity", "2 of 2"),
        # The worker admitted throughout and refused nothing: LiveKit lost the dispatch.
        (status(), status(), "no_agent", "admitted calls (1 of 2 calls) and refused none"),
    ],
)
def test_a_refusal_says_why_from_the_workers_status(before, after, reason, detail) -> None:
    found, said = refusal(before, after)
    assert found == reason and detail in said


def test_the_workers_status_is_read_as_it_wrote_it(tmp_path: Path) -> None:
    path = tmp_path / "voice-worker.json"
    assert read_status(path) is None  # no file: bench mode, or no worker
    path.write_text("{not json")
    assert read_status(path) is None
    path.write_text(json.dumps({"schema": 1, "updated": 100.0, "mode": "serve", "max_calls": 2,
                                "active_calls": 2, "admitting": False, "reason": "2 of 2 calls",
                                "refused": 5, "pid": 9, "profile": "rtx-5060-ti-serve"}))  # fmt: skip
    assert read_status(path, now=102.5) == {
        "mode": "serve", "max_calls": 2, "active_calls": 2, "admitting": False, "reason": "2 of 2 calls",
        "refused": 5, "pid": 9, "age_s": 2.5,
    }  # fmt: skip
    assert read_status(None) is None


# The attempts, against fake callers


@dataclass(frozen=True)
class Plan:
    """What a call meets: a join that fails (``connect``), an agent that comes ``agent_after`` seconds after the join
    (None: never), listens or not, and leaves after ``leaves_after`` turns (None: stays)."""

    connect: str | None = None
    agent_after: float | None = 0.0
    listens: bool = True
    leaves_after: int | None = None
    turn_s: float = 0.01


class FakeRoom:
    def __init__(self) -> None:
        self.connected = False
        self.remote_participants: dict[str, Any] = {}

    def isconnected(self) -> bool:
        return self.connected


class FakeCaller:
    """A call's caller as Arrivals drives it (bots.Caller's join, listening, play and hang_up), meeting its plan."""

    def __init__(self, config: BotsConfig, index: int, run_id: str, plan: Plan) -> None:
        self.config, self.index, self.plan = config, index, plan
        self.room_name = f"flatshare-bot-{run_id}-{index}"
        self.room = FakeRoom()
        self.max_lateness_s = 0.002
        self.hung_up = False

    async def join(self, tasks: set[asyncio.Task[None]]) -> None:
        if self.plan.connect is not None:
            raise ConnectionError(self.plan.connect)
        self.room.connected = True
        if self.plan.agent_after is not None:
            agent = SimpleNamespace(kind=AGENT, attributes={AGENT_STATE: "listening"})
            asyncio.get_running_loop().call_later(
                self.plan.agent_after, self.room.remote_participants.__setitem__, "agent", agent
            )

    async def listening(self) -> None:
        if not self.plan.listens:
            raise TimeoutError("the agent never listened")

    async def play(
        self, rows: list[dict[str, Any]], log: Any, go_on: Callable[[], bool] | None = None
    ) -> None:
        config = self.config
        for number in range(config.warmup_turns + config.turns):
            if go_on is not None and not go_on():
                return
            await asyncio.sleep(self.plan.turn_s)
            row: dict[str, Any] = {"caller": self.index, "turn": number, "barge_in": False}
            row["v2v_ms"] = 1500.0 + 100 * self.index + number
            if number < config.warmup_turns:
                row["warmup"] = True
            if self.plan.leaves_after is not None and number + 1 >= self.plan.leaves_after:
                self.room.remote_participants.clear()
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()

    async def hang_up(self, tasks: set[asyncio.Task[None]]) -> None:
        self.hung_up = True
        self.room.connected = False


def run_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plans: dict[int, Plan],
    offsets: list[float],
    *,
    worker: dict[str, Any] | None = None,
    signals: tuple[float, ...] = (),
    turns: int = 3,
    **config: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Arrivals]:
    """A run of calls due at ``offsets`` (seconds from its start), call k meeting ``plans[k]`` (a plain call by
    default), with the worker's status ``worker`` (written as fresh), SIGINT sent at each of ``signals``. Returns the
    summary and the attempts' and turns' lines."""
    monkeypatch.setattr(arrivals, "arrival_offsets", lambda *_: iter(offsets))
    monkeypatch.setattr(arrivals, "JOIN_STAGGER_S", 0.0)
    out = tmp_path / "run"
    path = tmp_path / "voice-worker.json"
    if worker is not None:
        path.write_text(json.dumps(worker | {"updated": time.time() + 3600}))
    bots = BotsConfig("ws://fake", "key", "secret", tmp_path, turns=turns, warmup_turns=1)
    settings = {"calls_per_hour": 60.0, "duration_s": 3600.0, "dispatch_timeout": 0.2} | config
    made: list[FakeCaller] = []

    def caller(config: BotsConfig, corpus: Any, index: int, run_id: str) -> FakeCaller:
        made.append(FakeCaller(config, index, run_id, plans.get(index, Plan())))
        return made[-1]

    run = Arrivals(ArrivalsConfig(bots, worker_status=path, **settings), out, object(), caller)

    async def main() -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        for at in signals:
            loop.call_later(at, os.kill, os.getpid(), signal.SIGINT)
        return await run.run()

    summary = asyncio.run(main())
    lines = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines()]
    turned = [json.loads(line) for line in (out / "turns.jsonl").read_text().splitlines()]
    assert all(c.hung_up for c in made)  # every caller left its room, however its call went
    return summary, lines, turned, run


def test_every_attempt_is_recorded_with_its_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = {
        1: Plan(connect="could not connect: 127.0.0.1:17880 refused"),  # the service never saw it
        2: Plan(agent_after=None),  # no agent came
        3: Plan(listens=False),  # an agent came and never listened
        4: Plan(leaves_after=2),  # it left after two turns
    }
    worker = status(active_calls=1)
    summary, lines, turned, _ = run_calls(
        tmp_path, monkeypatch, plans, [0.0, 0.02, 0.04, 0.06, 0.08], worker=worker
    )
    by_call = {line["call"]: line for line in lines}
    assert sorted(by_call) == [0, 1, 2, 3, 4]
    assert by_call[0]["outcome"] == "accepted" and by_call[0]["ended"] == "completed"
    assert (by_call[0]["turns"], by_call[0]["errors"]) == (
        4,
        0,
    )  # a warm-up turn and three measured
    assert (by_call[1]["outcome"], by_call[1]["reason"]) == ("failed", "connect")
    assert "127.0.0.1:17880 refused" in by_call[1]["detail"]
    # The worker admitted calls (one of two) throughout and refused none: the dispatch was lost.
    assert (by_call[2]["outcome"], by_call[2]["reason"]) == ("refused", "no_agent")
    assert by_call[2]["worker"]["active_calls"] == 1 and by_call[2]["worker_after"] is not None
    assert (by_call[3]["outcome"], by_call[3]["ended"], by_call[3]["reason"]) == (
        "accepted",
        "dropped",
        "never_listened",
    )
    assert (by_call[4]["ended"], by_call[4]["reason"], by_call[4]["turns"]) == (
        "dropped",
        "agent_left",
        2,
    )
    for line in lines:
        assert (
            line["run"] == 0 and line["late_ms"] >= 0 and line["room"].endswith(f"-{line['call']}")
        )
        assert line["caller_max_lateness_ms"] == 2.0 and line["ended_at"] >= line["started"]
    accepted = [line for line in lines if line["outcome"] == "accepted"]
    assert all(line["dispatch_ms"] >= 0 for line in accepted)
    # Every turn as the caller wrote it, with the run's number and when it ended.
    assert [(t["caller"], t["turn"]) for t in turned if t["caller"] == 0] == [
        (0, n) for n in range(4)
    ]
    assert all(t["run"] == 0 and t["at"] > 0 for t in turned)
    assert summary["outcomes"] == {
        "accepted": 3,
        "refused": 1,
        "failed": 1,
        "skipped": 0,
        "abandoned": 0,
    }
    assert (summary["attempts"], summary["refused_by"], summary["failed_by"]) == (
        5,
        {"no_agent": 1},
        {"connect": 1},
    )
    assert summary["ended"] == {"completed": 1, "dropped": 2, "cut": 0}
    # Measured turns: call 0's three and call 4's one after its warm-up turn.
    assert summary["turns"] == 4 and summary["v2v_ms"]["n"] == 4


def test_a_full_worker_refuses_by_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = status(active_calls=2, admitting=False, reason="2 of 2 calls", refused=7)
    summary, lines, _, _ = run_calls(
        tmp_path, monkeypatch, {0: Plan(agent_after=None)}, [0.0], worker=worker
    )
    [line] = lines
    assert (line["outcome"], line["reason"], line["detail"]) == (
        "refused",
        "capacity",
        "2 of 2 calls",
    )
    assert line["worker"]["admitting"] is False and summary["refused_by"] == {"capacity": 1}


def test_a_call_due_while_too_many_are_open_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One attempt open at most: the second call is due while the first plays its turns.
    plans = {0: Plan(turn_s=0.1)}
    summary, lines, _, _ = run_calls(
        tmp_path, monkeypatch, plans, [0.0, 0.05, 0.6], worker=status(), max_open=1
    )
    by_call = {line["call"]: line for line in lines}
    assert by_call[1] | {"due": 0} == {
        "run": 0, "call": 1, "due": 0, "outcome": "skipped", "reason": "max_open", "detail": "1 attempts were open",
    }  # fmt: skip
    assert by_call[0]["ended"] == by_call[2]["ended"] == "completed"
    assert (
        summary["outcomes"]["skipped"] == 1 and summary["attempts"] == 2
    )  # a skipped call was never made


def test_calls_are_numbered_through_a_soaks_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lines, turned, _ = run_calls(
        tmp_path, monkeypatch, {}, [0.0, 0.02], worker=status(), run=2, first_call=40
    )
    assert [(line["run"], line["call"]) for line in lines] == [(2, 40), (2, 41)]
    assert {(t["run"], t["caller"]) for t in turned} == {(2, 40), (2, 41)}


def test_sigint_ends_each_call_with_its_current_turn_and_writes_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A call of 40 turns of 0.05 s, and calls due long after: SIGINT 0.3 s in stops the arrivals, and the call ends
    # with the turn it plays.
    plans = {0: Plan(turn_s=0.05)}
    summary, lines, turned, run = run_calls(
        tmp_path, monkeypatch, plans, [0.0, 30.0, 60.0], worker=status(), signals=(0.3,), turns=40
    )
    [line] = lines  # the calls due later never came
    assert (line["outcome"], line["ended"], line["reason"]) == ("accepted", "cut", "SIGINT")
    assert 2 <= line["turns"] < 41 and len(turned) == line["turns"]
    assert summary["ended"]["cut"] == 1 and run.stopping == "SIGINT"
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert record["stopped_by"] == "SIGINT" and record["finished"] >= record["started"]
    assert json.loads((tmp_path / "run" / "summary.json").read_text()) == summary
    assert record["config"]["bots"]["api_secret"] == "<redacted>"
    # The loop's handlers are gone with the run: SIGINT is Python's own again.
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_a_second_sigint_hangs_every_call_up_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Turns of 5 s: the first signal waits for the turn's end, the second does not.
    plans = {0: Plan(turn_s=5.0), 1: Plan(agent_after=None)}
    started = time.monotonic()
    summary, lines, _, _ = run_calls(
        tmp_path,
        monkeypatch,
        plans,
        [0.0, 0.01],
        worker=status(),
        signals=(0.1, 0.2),
        dispatch_timeout=5.0,
    )
    assert time.monotonic() - started < 3.0
    by_call = {line["call"]: line for line in lines}
    assert (by_call[0]["outcome"], by_call[0]["ended"], by_call[0]["reason"]) == (
        "accepted",
        "cut",
        "hung_up",
    )
    assert (by_call[1]["outcome"], by_call[1]["reason"]) == (
        "abandoned",
        "hung_up",
    )  # no outcome yet
    assert summary["outcomes"]["abandoned"] == 1 and summary["attempts"] == 1


def test_a_run_ends_when_its_calls_end_after_the_last_arrival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No call arrives after the run's time, and the calls in progress play their turns to the end.
    summary, lines, _, run = run_calls(tmp_path, monkeypatch, {0: Plan(turn_s=0.05)}, [0.0])
    assert [line["ended"] for line in lines] == ["completed"] and run.stopping is None
    assert summary["outcomes"]["accepted"] == 1


def test_the_runs_summary_counts_its_attempts_and_measured_turns() -> None:
    records = [
        {"outcome": "accepted", "ended": "completed", "caller_max_lateness_ms": 4.0},
        {"outcome": "accepted", "ended": "cut", "reason": "SIGINT"},
        {"outcome": "refused", "reason": "capacity"},
        {"outcome": "refused", "reason": "capacity"},
        {"outcome": "refused", "reason": "engine"},
        {"outcome": "failed", "reason": "connect"},
        {"outcome": "skipped", "reason": "max_open"},
    ]
    rows = [
        {"caller": 0, "turn": 0, "warmup": True, "v2v_ms": 9000.0},  # left out: a warm-up turn
        {"caller": 0, "turn": 1, "v2v_ms": 1500.0, "barge_in": False},
        {"caller": 0, "turn": 2, "error": "the agent did not start speaking"},
    ]
    summary = summarize_run(records, rows, 30.0)
    assert summary["attempts"] == 6
    assert summary["outcomes"] == {
        "accepted": 2,
        "refused": 3,
        "failed": 1,
        "skipped": 1,
        "abandoned": 0,
    }
    assert summary["refused_by"] == {"capacity": 2, "engine": 1}
    assert summary["ended"] == {"completed": 1, "dropped": 0, "cut": 1}
    assert (summary["turns"], summary["errors"]) == (2, 1)
    assert summary["v2v_ms"] == {
        "n": 2,
        "censored": 1,
        "p50": 15_750.0,
        "p95": 28_575.0,
        "p99": 29_715.0,
    }
    assert summary["caller_max_lateness_ms"] == 4.0


# The real callers, in a fake room


class FakeLiveKitRoom:
    """rtc.Room as bots.Caller drives it: ``connect`` fails with ``refuse`` (LiveKit's ConnectError), or an agent
    joins ``agent_after`` seconds after it (None: none comes), listening, its audio on ``line``'s fake track."""

    refuse: str | None = None
    agent_after: float | None = 0.05
    line: Line | None = None
    rooms: ClassVar[list[FakeLiveKitRoom]] = []

    def __init__(self) -> None:
        self.handlers: dict[str, list[Callable[..., Any]]] = {}
        self.remote_participants: dict[str, Any] = {}
        self.connected = False
        self.disconnected = False
        self.local_participant = SimpleNamespace(publish_track=self._publish)
        FakeLiveKitRoom.rooms.append(self)

    def on(self, event: str, callback: Callable[..., Any] | None = None) -> Any:
        def register(handler: Callable[..., Any]) -> Callable[..., Any]:
            self.handlers.setdefault(event, []).append(handler)
            return handler

        return register if callback is None else register(callback)

    async def connect(self, url: str, token: str) -> None:
        if self.refuse is not None:
            raise rtc.ConnectError(self.refuse)
        assert url == "ws://fake" and token  # a join token signed with the key and secret
        self.connected = True
        if self.agent_after is not None:
            asyncio.get_running_loop().call_later(self.agent_after, self._agent_joins)

    def _agent_joins(self) -> None:
        agent = SimpleNamespace(kind=AGENT, attributes={AGENT_STATE: "listening"})
        self.remote_participants["agent"] = agent
        track = SimpleNamespace(kind=rtc.TrackKind.KIND_AUDIO)
        for handler in self.handlers.get("track_subscribed", []):
            handler(track, None, agent)

    async def _publish(self, track: Any, options: Any) -> None:
        return None

    def isconnected(self) -> bool:
        return self.connected

    async def disconnect(self) -> None:
        if self.connected:  # as rtc.Room's: a room never joined has nothing to leave
            self.connected, self.disconnected = False, True


def test_the_real_callers_record_each_outcome_through_a_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bots.Caller's own join, listening, turns and hang-up, in a fake room whose agent answers each line: a call
    # accepted plays its turns as E2's do; one whose room no agent joins is refused; one LiveKit refuses failed.
    line = Line()
    monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
    monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)
    monkeypatch.setattr(bots.rtc, "Room", FakeLiveKitRoom)
    monkeypatch.setattr(
        bots.rtc.LocalAudioTrack, "create_audio_track", staticmethod(lambda *_: object())
    )
    monkeypatch.setattr(bots, "QUIET_S", 0.2)
    monkeypatch.setattr(arrivals, "JOIN_STAGGER_S", 0.0)
    FakeLiveKitRoom.rooms = []
    corpus = SimpleNamespace(
        turns=[line_clip(f"turn-{n:02}", 0.2) for n in range(4)], barge_ins=[line_clip("b", 0.2)]
    )
    secret = "s" * 32  # a key long enough for the join token's HMAC
    bots_config = BotsConfig(
        "ws://fake", "key", secret, tmp_path, turns=2, barge_in=0.0, warmup_turns=1
    )
    # The fake agent publishes no speaking state: each reply's end waits the reply timeout twice, then quiet.
    bots_config = dataclasses.replace(bots_config, think=(0.0, 0.0), reply_timeout=0.6)
    path = tmp_path / "voice-worker.json"
    path.write_text(
        json.dumps(
            status(admitting=False, active_calls=2, reason="2 of 2 calls") | {"updated": 1e12}
        )
    )

    def one(**room: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        for key, value in ({"refuse": None, "agent_after": 0.05} | room).items():
            monkeypatch.setattr(FakeLiveKitRoom, key, value)
        out = tmp_path / f"run-{len(FakeLiveKitRoom.rooms)}"
        monkeypatch.setattr(arrivals, "arrival_offsets", lambda *_: iter([0.0]))
        config = ArrivalsConfig(bots_config, 60.0, 60.0, dispatch_timeout=0.5, worker_status=path)
        asyncio.run(Arrivals(config, out, corpus).run())
        [attempt] = [json.loads(x) for x in (out / "attempts.jsonl").read_text().splitlines()]
        turns = [json.loads(x) for x in (out / "turns.jsonl").read_text().splitlines()]
        return attempt, turns

    accepted, turns = one()
    assert (accepted["outcome"], accepted["ended"], accepted["turns"], accepted["errors"]) == (
        "accepted",
        "completed",
        3,
        0,
    )
    assert [(t["turn"], t.get("warmup", False)) for t in turns] == [
        (0, True),
        (1, False),
        (2, False),
    ]
    for t in turns:
        assert REPLY_MS - 30 <= t["v2v_ms"] <= REPLY_MS + 60 + accepted["caller_max_lateness_ms"], t
        assert t["end_by"] == "silence" and "at" in t and t["run"] == 0
    refused, none = one(agent_after=None)
    assert (refused["outcome"], refused["reason"], refused["detail"]) == (
        "refused",
        "capacity",
        "2 of 2 calls",
    )
    assert none == []
    failed, _ = one(refuse="failed to connect to ws://fake: connection refused")
    assert (failed["outcome"], failed["reason"]) == ("failed", "connect")
    assert failed["detail"] == "ConnectError: failed to connect to ws://fake: connection refused"
    # Every room was left, the refused one too; the one never joined had nothing to leave.
    assert [room.disconnected for room in FakeLiveKitRoom.rooms] == [True, True, False]


REPLY_MS = 200.0
"""The fake agent's reply, REPLY_S of test_callers_and_bots after each line's end."""


# The command line


ARRIVALS = [
    "arrivals",
    "--corpus",
    "corpus",
    "--calls-per-hour",
    "20",
    "--duration",
    "86400",
    "--out",
    "run",
]


def test_arrivals_takes_its_settings_from_the_command_line() -> None:
    args = cli.build_parser().parse_args(
        [*ARRIVALS, "--turns", "8", "--run", "2", "--first-call", "77", "--max-open", "4"]
    )
    assert (args.calls_per_hour, args.duration, args.turns, args.warmup_turns) == (
        20.0,
        86400.0,
        8,
        1,
    )
    assert (args.run, args.first_call, args.max_open, args.dispatch_timeout) == (2, 77, 4, 15.0)
    assert args.worker_status is None  # the unit's status file, found when the run starts


@pytest.mark.parametrize(
    ("option", "value", "says"),
    [
        ("--calls-per-hour", "0", "must be above 0, got 0.0"),
        ("--duration", "-5", "must be 0 or more seconds, got -5.0"),
        ("--dispatch-timeout", "0", "must be above 0, got 0.0"),
        ("--max-open", "0", "must be 1 or more, got 0"),
    ],
)
def test_arrivals_refuses_settings_no_call_can_arrive_with(option, value, says, capsys) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli.build_parser().parse_args([*ARRIVALS, option, value])
    assert exit_.value.code == 2
    assert f"argument {option}: {says}" in capsys.readouterr().err


def test_arrivals_names_the_livekit_credentials_it_lacks(monkeypatch, capsys) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    with pytest.raises(SystemExit):
        cli.main(ARRIVALS)
    assert "arrivals needs LIVEKIT_API_KEY and LIVEKIT_API_SECRET" in capsys.readouterr().err


def test_arrivals_takes_the_options_a_soak_passes() -> None:
    # flatshare-bench soak runs the callers with these options (soak/run.py, Soak._start_calls); its tests' stand-in
    # for this command requires the same ones.
    args = cli.build_parser().parse_args(
        ["arrivals", "--corpus", "/c", "--calls-per-hour", "20", "--duration", "86285", "--turns", "10",
         "--warmup-turns", "1", "--barge-in", "0.25", "--seed", "0", "--run", "1", "--first-call", "37",
         "--dispatch-timeout", "15", "--max-open", "8", "--worker-status", "/run/user/1/flatshare/voice-worker.json",
         "--out", "/soak/calls/run-001"]
    )  # fmt: skip
    assert (args.duration, args.run, args.first_call, args.worker_status) == (
        86285.0,
        1,
        37,
        Path("/run/user/1/flatshare/voice-worker.json"),
    )


def test_the_status_file_is_the_units(monkeypatch) -> None:
    env = {"FLATSHARE_VOICE_STATUS": "/run/user/7/flatshare/voice-worker.json"}
    assert arrivals.default_status_file(env) == Path(env["FLATSHARE_VOICE_STATUS"])
    assert arrivals.default_status_file({"XDG_RUNTIME_DIR": "/run/user/8"}) == Path(
        "/run/user/8/flatshare/voice-worker.json"
    )
