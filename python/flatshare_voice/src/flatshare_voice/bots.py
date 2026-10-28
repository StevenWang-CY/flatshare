"""Measurement callers for the voice agent (``flatshare-voice bots``): latency read from waveforms.

Each caller joins its own room with a token that dispatches the agent, publishes a microphone
track paced in real time and records the agent's audio. A call is a sequence of turns. The caller
plays a line from the corpus and waits for the reply:

- **voice-to-voice** (V2V) runs from the end of the caller's speech to the first agent audio above
  the speech threshold;
- on a share of turns (``--barge-in``) the caller interrupts the reply after a random delay;
  **barge-in to silence** runs from the interruption's speech onset until the agent's audio
  stops, and the reply to the interruption gives a second V2V sample.

The caller then waits for the reply to end and pauses for a think time before the next turn. The
agent says when it has finished: its state (``lk.agent.state``) leaves ``speaking`` once its last
frame has gone out, and the reply ended when the caller has heard that frame. Within each reply the
caller times its pauses, the stretches of quiet while the agent has not finished, and reads its
WebRTC receiver's count of samples played without a packet (concealed); an agent that publishes no
state has finished once it has been quiet for a second, as every caller before the state was read
had it, which a pause of a second or more mistakes for the end. Either way the next turn starts no
sooner than a second after the reply's last audio, so the turns keep the pace they always had. Every time comes from one clock in one process, so no clock offset enters
the numbers. Each run directory gets ``run.json``, ``turns.jsonl`` and ``summary.json``.

A call may open with warm-up turns (``--warmup-turns``): a line and its reply, without an
interruption, played but not measured. Their lines in turns.jsonl say ``"warmup": true`` and the
summary leaves them out. They draw their lines and pauses from a generator of their own, so the
measured turns after them are those of the same call without them.

A turn fails when the agent does not answer within ``reply_timeout``, or does not stop speaking: its
line in turns.jsonl has an ``error`` and keeps what the turn measured before, and the turns after it
play what they would have. In the summary a sample that the timeout cut short counts as censored at
the timeout, never as missing, so that a failure raises a percentile rather than leaving it out.

With ``--save-audio`` each caller also keeps the frames it already handles, what it sent and the
agent's audio as received, and once every call has ended writes them as ``call-<i>.wav`` with the
times of each turn's events in ``call-<i>.json``, for the demo video (benchmarks/e2/demo.py).
"""

from __future__ import annotations

import asyncio
import json
import math
import platform
import random
import time
import uuid
import wave
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from livekit import api, rtc

from flatshare_stats import percentiles, v2v_samples
from flatshare_stats import replies as reply_summary

from .audio import (
    AGENT_RATE,
    AGENT_STATE,
    SAMPLE_RATE,
    AgentAudio,
    AgentState,
    Clip,
    Microphone,
    Recorder,
    agents,
    until,
    upsample_track,
)
from .protocol import AGENT_NAME

QUIET_S = 1.0
"""Agent silence that ends its reply when the agent publishes no state, and the least time from a reply's last audio
to the next turn."""

TAIL_S = 0.3
"""Quiet after the agent's state leaves speaking by which the reply's last frames have arrived: they travel through
LiveKit's server and the receiver's jitter buffer, the state change through its signalling."""

PAUSE_S = 0.3
"""The shortest quiet stretch inside a reply that counts as a pause (a gap the caller hears); a speech span already
closes over quiet shorter than AgentAudio.GAP."""

JOIN_STAGGER_S = 0.25
"""How far apart the callers join their rooms. LiveKit's dispatcher can drop the job of a room created within
milliseconds of another's ("no workers with sufficient capacity", with six callers in W5's calls per GPU) and does
not retry, so that room's caller waits for an agent that never comes; a quarter second apart, each room's job is
assigned before the next room exists. The stagger only shifts when each call starts, by at most 1.25 s with six
callers: every turn is timed within its own call, and the callers' think times set the calls' turns apart anyway."""


@dataclass(frozen=True)
class BotsConfig:
    url: str
    """LiveKit signalling URL, e.g. ``ws://127.0.0.1:17880``."""
    api_key: str
    api_secret: str
    corpus: Path
    callers: int = 2
    turns: int = 20
    barge_in: float = 0.25
    """Share of turns on which the caller interrupts the reply."""
    barge_in_delay: tuple[float, float] = (0.6, 1.5)
    """Seconds after the reply's first audio at which an interruption starts."""
    think: tuple[float, float] = (1.0, 3.0)
    """Seconds between the end of a reply and the next turn."""
    reply_timeout: float = 30.0
    seed: int = 0
    warmup_turns: int = 0
    """Turns that open each call unmeasured: a line and its reply, no interruption."""
    in_order: bool = False
    """Play the corpus's turn lines in order rather than drawing each: caller i's measured turns from line
    i x ``turns`` on, so that one caller with as many turns as lines plays every line once (the endpointing
    sweep's check for replies that start inside a line)."""
    save_audio: bool = False
    """Write each call's audio and its turns' times after the calls: call-<i>.wav, call-<i>.json."""

    def __post_init__(self) -> None:
        if self.callers < 1 or self.turns < 1:
            raise ValueError("callers and turns must be 1 or more")
        if not 0.0 <= self.barge_in <= 1.0:
            raise ValueError("barge_in is a share of the turns, from 0 to 1")
        if self.warmup_turns < 0:
            raise ValueError("warmup_turns cannot be negative")


class Corpus:
    def __init__(self, directory: Path) -> None:
        manifest = json.loads((directory / "manifest.json").read_text())
        self.turns = [
            Clip.load(name, str(directory / f"{name}.wav"))
            for name, entry in sorted(manifest.items())
            if entry["kind"] == "turn"
        ]
        self.barge_ins = [
            Clip.load(name, str(directory / f"{name}.wav"))
            for name, entry in sorted(manifest.items())
            if entry["kind"] == "barge_in"
        ]
        if not self.turns or not self.barge_ins:
            raise ValueError(f"{directory}: the corpus needs turn and barge_in clips")


def token(config: BotsConfig, room: str, identity: str) -> str:
    """A join token for ``room`` that also dispatches the agent into it."""
    dispatch = api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)])
    return (
        api.AccessToken(config.api_key, config.api_secret)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .with_room_config(dispatch)
        .to_jwt()
    )


class Caller:
    def __init__(self, config: BotsConfig, corpus: Corpus, index: int, run_id: str) -> None:
        self.config = config
        self.corpus = corpus
        self.index = index
        self.room_name = f"flatshare-bot-{run_id}-{index}"
        self.rng = random.Random(config.seed * 1000 + index)
        self.room = rtc.Room()
        self.sent = Recorder(SAMPLE_RATE) if config.save_audio else None
        self.received = Recorder(AGENT_RATE) if config.save_audio else None
        self.agent_audio = AgentAudio(self.received)
        self.agent_state = AgentState()
        self.agent_track: rtc.Track | None = None
        self.microphone = Microphone(recorder=self.sent)
        self.max_lateness_s = 0.0
        self.moments: dict[int, dict[str, float]] = {}
        """When each turn's events happened (its line's start and speech end, the reply, ...), as far
        as the turn got; call-<i>.json gives them."""

    async def turn(self, number: int, warmup: random.Random | None = None) -> dict[str, Any]:
        """One turn of the call; a warm-up turn draws its line from ``warmup`` and is never
        interrupted. The turn draws its interruption with its line, before either is played, so
        that a turn that fails takes as many draws as one answered: the turns after it play the
        lines and interruptions of the same call without the failure."""
        config, audio = self.config, self.agent_audio
        if config.in_order and warmup is None:
            measured = number - config.warmup_turns + self.index * config.turns
            clip = self.corpus.turns[measured % len(self.corpus.turns)]
        else:
            clip = (warmup or self.rng).choice(self.corpus.turns)
        interruption, delay = None, 0.0
        if warmup is None and self.rng.random() < config.barge_in:
            delay = self.rng.uniform(*config.barge_in_delay)
            interruption = self.rng.choice(self.corpus.barge_ins)
        at = self.moments[number] = {}
        played = await self.microphone.play(clip)
        at |= {"start": played["start"], "speech_end": played["speech_end"]}
        reply = await audio.first_onset_after(played["speech_end"], config.reply_timeout)
        at["reply"] = reply
        before = asyncio.create_task(self.receiver_counts())
        row: dict[str, Any] = {
            "caller": self.index,
            "turn": number,
            "clip": clip.name,
            "v2v_ms": round((reply - played["speech_end"]) * 1000, 1),
            # the agent answered part of the line before the caller finished (a VAD split)
            "spoke_during_turn": audio.started_between(played["onset"], played["speech_end"]),
            "barge_in": False,
        }
        try:
            if interruption is not None:
                await asyncio.sleep(max(0.0, reply + delay - time.monotonic()))
                barged = await self.microphone.play(interruption)
                onset = barged["onset"]
                at |= {"barge_in": onset, "barge_in_speech_end": barged["speech_end"]}
                row.update(barge_in=True, barge_in_clip=interruption.name)
                row["interrupted_speech"] = audio.speaking_at(onset)
                if row["interrupted_speech"]:
                    stopped = await audio.end_of_span_at(onset, config.reply_timeout)
                    at["silence"] = stopped
                    row["barge_in_to_silence_ms"] = round(max(0.0, stopped - onset) * 1000, 1)
                second = await audio.first_onset_after(barged["speech_end"], config.reply_timeout)
                at["barge_in_reply"] = second
                row["v2v_after_barge_in_ms"] = round((second - barged["speech_end"]) * 1000, 1)
                row["interrupted_reply_pauses_ms"] = self.pauses(reply, onset)
                await before
                reply, before = second, asyncio.create_task(self.receiver_counts())
            end, row["end_by"] = await self.reply_end(reply)
            at["end"] = end
            row["reply_audio_ms"] = round((end - reply) * 1000, 1)
            row["reply_pauses_ms"] = self.pauses(reply, end)
            counted = (await before, await self.receiver_counts())
            if None not in counted:
                row["received"] = {key: counted[1][key] - counted[0][key] for key in counted[1]}
        except TimeoutError as exc:  # the turn fails, keeping the reply it measured
            row["error"] = str(exc)
        finally:
            before.cancel()
        return row

    async def reply_end(self, reply: float) -> tuple[float, str]:
        """When the reply that began at ``reply`` ended, its last audio, and how the caller knew: ``state``, once the
        agent's state has left speaking and TAIL_S of quiet followed, or ``silence``, after QUIET_S of quiet, from an
        agent that publishes no state. It returns no sooner than QUIET_S after the reply's last audio."""
        config, audio = self.config, self.agent_audio
        left = None
        if self.agent_state.changes:
            left = await self.agent_state.left_speaking_after(reply, config.reply_timeout * 2)
        if left is None:
            end = await audio.silent_after(time.monotonic(), QUIET_S, config.reply_timeout * 2)
            return end, "silence"
        await audio.silent_after(left, TAIL_S, config.reply_timeout)
        end = max((e for s, e in audio.spans if reply <= s <= time.monotonic()), default=reply)
        await asyncio.sleep(max(0.0, end + QUIET_S - time.monotonic()))
        return end, "state"

    def pauses(self, a: float, b: float) -> list[float]:
        """The reply's pauses between ``a`` and ``b``, each in ms."""
        return [round(g * 1000, 1) for g in self.agent_audio.gaps_between(a, b, PAUSE_S)]

    async def receiver_counts(self) -> dict[str, int] | None:
        """The agent's audio as the caller's WebRTC receiver has counted it so far: the samples it played, those of
        them it concealed (played without a packet) and those it concealed as silence; None without the agent's track
        or its statistics."""
        track = self.agent_track
        if track is None:
            return None
        try:
            stats = await asyncio.wait_for(track.get_stats(), 1.0)
        except Exception:  # a missing count drops the counts, never the turn
            return None
        for stat in stats:
            if stat.HasField("inbound_rtp"):
                inbound = stat.inbound_rtp.inbound
                return {
                    "samples": inbound.total_samples_received,
                    "concealed": inbound.concealed_samples,
                    "silent_concealed": inbound.silent_concealed_samples,
                }
        return None

    async def call(self, rows: list[dict[str, Any]], log: TextIO) -> None:
        tasks: set[asyncio.Task[None]] = set()
        await self.join(tasks)
        try:
            await self.listening()
            await self.play(rows, log)
        finally:
            await self.hang_up(tasks)

    async def join(self, tasks: set[asyncio.Task[None]]) -> None:
        """Join the call's room and publish the microphone, following the agent there: its audio once its track is
        subscribed and its state as it changes. The microphone's loop and the agent's audio run as tasks in ``tasks``,
        which :meth:`hang_up` ends."""
        config = self.config

        @self.room.on("track_subscribed")
        def _on_track(track: rtc.Track, _: Any, participant: rtc.RemoteParticipant) -> None:
            is_agent = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
            if is_agent and track.kind == rtc.TrackKind.KIND_AUDIO:
                self.agent_track = track
                tasks.add(asyncio.create_task(self.agent_audio.consume(track)))

        @self.room.on("participant_attributes_changed")
        def _on_attributes(changed: dict[str, str], participant: rtc.Participant) -> None:
            is_agent = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
            if is_agent and AGENT_STATE in changed:
                self.agent_state.record(changed[AGENT_STATE])

        identity = f"caller-{self.index}"
        await self.room.connect(config.url, token(config, self.room_name, identity))
        track = rtc.LocalAudioTrack.create_audio_track("microphone", self.microphone.source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, options)
        tasks.add(asyncio.create_task(self.microphone.run()))

    async def listening(self) -> None:
        """Wait for the agent in the room to listen, within the reply timeout, then let its audio input run on
        silence for half a second before the first line."""
        await until(
            lambda: any(p.attributes.get(AGENT_STATE) == "listening" for p in agents(self.room)),
            self.config.reply_timeout,
            "the agent never listened",
        )
        self.agent_state.record("listening")
        await asyncio.sleep(0.5)  # let the agent's audio input run on silence first

    async def hang_up(self, tasks: set[asyncio.Task[None]]) -> None:
        """Leave the room: the tasks :meth:`join` started end, and the microphone closes."""
        self.max_lateness_s = self.microphone.max_lateness_s
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.microphone.source.aclose()
        await self.room.disconnect()

    async def play(
        self, rows: list[dict[str, Any]], log: TextIO, go_on: Callable[[], bool] | None = None
    ) -> None:
        """The call's turns, each line to ``rows`` and ``log`` (turns.jsonl) as it ends, numbered in
        the order they are played: first the warm-up turns, marked ``"warmup": true``, whose lines
        and pauses come from a generator of their own, then the measured turns. With ``go_on``, the
        call ends before any turn it answers false for (``flatshare-voice arrivals``: its run ends,
        or the agent left); without it, as E2's calls do, it plays every turn."""
        config = self.config
        warmup = random.Random(f"warmup-{config.seed}-{self.index}")
        for number in range(config.warmup_turns + config.turns):
            if go_on is not None and not go_on():
                return
            warm = number < config.warmup_turns
            try:
                row = await self.turn(number, warmup if warm else None)
            except TimeoutError as exc:
                row = {"caller": self.index, "turn": number, "error": str(exc)}
            if warm:
                row["warmup"] = True
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            await asyncio.sleep((warmup if warm else self.rng).uniform(*config.think))

    def save(self, out_dir: Path, rows: Iterable[dict[str, Any]]) -> None:
        """Write the call as the caller had it (``--save-audio``), from the frames it kept.

        ``call-<i>.wav`` is stereo 16-bit at the agent's rate: on the left what the caller sent,
        upsampled from 16 kHz, on the right the agent's audio as received, both on the caller's
        clock from the first frame of either. ``call-<i>.json`` gives when the WAV starts, on that
        clock and on the wall clock, and each of the caller's turns as turns.jsonl has it, with
        when its events happened in seconds into the WAV (``at``): its line's start and end of
        speech, the reply, an interruption's onset and end of speech, the agent's silence after it
        and its reply to it, and the end of the turn.
        """
        sent, received = self.sent, self.received
        if sent is None or received is None:
            raise ValueError("the caller kept no audio: save_audio is off")
        tracks = [track for track in (sent, received) if track.frames]
        start = min(track.frames[0][0] for track in tracks)
        end = max(track.frames[-1][0] + len(track.frames[-1][1]) / track.rate for track in tracks)
        length = math.ceil((end - start) * AGENT_RATE)
        caller = upsample_track(sent.signal(start, math.ceil((end - start) * SAMPLE_RATE)))
        stereo = np.zeros((length, 2), dtype=np.int16)
        stereo[: len(caller), 0] = caller[:length]
        stereo[:, 1] = received.signal(start, length)
        with wave.open(str(out_dir / f"call-{self.index}.wav"), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(AGENT_RATE)
            wav.writeframes(stereo.tobytes())
        turns = [
            {key: value for key, value in row.items() if key != "caller"}
            | {"at": {event: round(t - start, 4) for event, t in self.moments[row["turn"]].items()}}
            for row in rows
        ]
        call = {
            "caller": self.index,
            "room": self.room_name,
            "rate": AGENT_RATE,
            "channels": ["caller", "agent"],
            "start": {"monotonic": start, "wall": start + time.time() - time.monotonic()},
            "seconds": round(length / AGENT_RATE, 3),
            "turns": turns,
        }
        (out_dir / f"call-{self.index}.json").write_text(json.dumps(call, indent=1) + "\n")


def summarize(
    rows: list[dict[str, Any]], lateness: list[float], timeout_s: float = 30.0
) -> dict[str, Any]:
    """The calls' summary of their measured turns, the warm-up turns left out, in flatshare-stats'
    percentiles. A sample a timeout cut short counts at the timeout (``timeout_s``): the reply to a
    turn or to an interruption that never came, the agent's audio that never stopped."""
    rows = [row for row in rows if not row.get("warmup")]
    timeout_ms = timeout_s * 1000
    v2v, censored = v2v_samples(rows, timeout_ms)
    silence = [r["barge_in_to_silence_ms"] for r in rows if "barge_in_to_silence_ms" in r]
    cut = sum(1 for r in rows if r.get("interrupted_speech") and "barge_in_to_silence_ms" not in r)
    barge_ins = [r for r in rows if r.get("barge_in")]
    summary = {
        "turns": len(rows),
        "errors": sum(1 for r in rows if "error" in r),
        "v2v_ms": percentiles(v2v, censored),
        "barge_in_to_silence_ms": percentiles([*silence, *[timeout_ms] * cut], cut),
        "barge_ins": len(barge_ins),
        "barge_ins_after_the_reply": sum(1 for r in barge_ins if not r["interrupted_speech"]),
        "caller_max_lateness_ms": round(max(lateness, default=0.0) * 1000, 1),
    }
    heard = [r for r in rows if "reply_pauses_ms" in r]
    return summary | ({"replies": replies(heard)} if heard else {})


def replies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The replies the calls heard to their end (on a turn with an interruption, the reply to it): how the caller knew
    each had ended, how long its audio ran, its pauses of PAUSE_S or more, and, where the receiver counted them, the
    share of the samples it played over them that it had to conceal, silence left out (flatshare_stats.replies, which
    the soak's summary shares, with a reply ended by QUIET_S of quiet)."""
    return reply_summary(rows, quiet_ms=QUIET_S * 1000)


async def run_bots(config: BotsConfig, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(config.corpus)
    run_id = uuid.uuid4().hex[:8]
    callers = [Caller(config, corpus, i, run_id) for i in range(config.callers)]
    rows: list[dict[str, Any]] = []
    started = time.time()
    with (out_dir / "turns.jsonl").open("w") as log:

        async def join(caller: Caller) -> None:
            await asyncio.sleep(caller.index * JOIN_STAGGER_S)
            await caller.call(rows, log)

        await asyncio.gather(*(join(caller) for caller in callers))
    summary = summarize(rows, [c.max_lateness_s for c in callers], config.reply_timeout)
    run = {
        "kind": "voice-bots",
        "config": asdict(config) | {"api_secret": "<redacted>", "corpus": str(config.corpus)},
        "host": platform.node(),
        "run_id": run_id,
        "started": started,
        "finished": time.time(),
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if config.save_audio:  # once every call has ended, so that writing them delays no measurement
        for caller in callers:
            caller.save(out_dir, [row for row in rows if row["caller"] == caller.index])
    return summary
