"""Lifecycle events from the voice agent to the control plane, and the stage times of each turn.

The control plane can hold KV and prefill headroom for a voice turn only if it knows when the caller
started speaking, when the end of the turn was decided, and when the agent's reply started, ended or
was cut off (docs/PROTOCOL.md, lifecycle events). ``Lifecycle`` maps one ``AgentSession``'s signals in
LiveKit Agents 1.8.8 to those events, and ``EventSender`` posts them to the control plane's
``/v1/events`` over the socket the LLM uses. Each event comes from one hook, which fires this early:

- ``user_speech_started``: ``user_state_changed`` to ``speaking``, at Silero VAD's start of speech,
  once it has heard 0.05 s of it: 0.10-0.16 s after the onset on cyan (W0).
- ``user_turn_ended``: ``conversation_item_added`` with the caller's message, once the end of the
  turn is decided and its reply scheduled, together with LiveKit's end-of-utterance metrics and the
  agent turning ``thinking``. The reply's LLM request starts then, or at the final transcript under
  preemptive generation.
- ``playback_started``: the audio output's ``playback_started``, as the reply's first audio frame
  goes to the published track.
- ``playback_ended`` and ``interrupted``: the audio output's ``playback_finished``, which says whether
  the reply was interrupted and how much of it played (``played_ms``, what the caller heard). It fires
  once the reply has played out, or when a barge-in interrupts it: on the barge-in's final transcript
  with LiveKit's default handling, which first pauses the audio after 0.5 s of speech.
- ``session_closed``: ``close``, when the session closes after the caller has left.

``metrics_collected`` reports the end of the turn too, but 1.8.8 deprecates it, with a warning on
every subscription, for each message's ``metrics``. Those also hold the stage times that ``Lifecycle``
logs once per turn (logger ``flatshare_voice.turns``), with the reply's playout: the audio it had
ready to play and the times its output ran out (``Playout``).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import itertools
import logging
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2 as httpx
from livekit import rtc
from livekit.agents import (
    AgentSession,
    CloseEvent,
    ConversationItemAddedEvent,
    JobContext,
    UserStateChangedEvent,
    llm,
)
from livekit.agents.voice import io

EVENTS_URL = "http://flatshare.invalid/v1/events"
"""The control plane's route; requests go over its Unix socket, so the host never resolves."""

SOURCE = "voice-worker"

QUEUE_LIMIT = 64
"""Events waiting to be sent, about ten turns; when it is full the oldest is dropped."""
POST_TIMEOUT_S = 1.0
"""The longest a POST may take. The control plane answers in a millisecond; an event any later than
this is stale."""
CLOSE_TIMEOUT_S = 2.0
"""How long the end of a session waits for its last events to be sent."""

STAGES = (
    "transcript_ms",
    "end_of_turn_ms",
    "llm_ttft_ms",
    "llm_first_sentence_ms",
    "turn_to_first_sentence_ms",
    "tts_first_audio_ms",
    "e2e_ms",
)
"""The stage times of a turn line, in the order of a turn (``Lifecycle``; flatshare-bench reads them)."""
STALL_MIN_S = 0.010
"""The shortest time without audio that counts as a stall: one 10 ms frame, the unit WebRTC sends
audio in."""

logger = logging.getLogger(__name__)
turn_logger = logging.getLogger("flatshare_voice.turns")


class EventSender:
    """Posts one session's events to the control plane from a background task.

    ``emit`` only queues, so it never waits on the control plane, and nothing here raises into the
    session: a full queue drops its oldest event, and a POST that fails or outlasts its timeout loses
    its batch. Both are counted and logged. A lost batch is not retried: the events are timing
    signals, and one that arrives late would mislead.
    """

    def __init__(
        self,
        session_id: str,
        url: str = EVENTS_URL,
        *,
        uds: str | None = None,
        limit: int = QUEUE_LIMIT,
        timeout: float = POST_TIMEOUT_S,
    ) -> None:
        self.session_id = session_id
        self.sent = 0
        """Events the control plane accepted."""
        self.dropped = 0
        """Events never sent: pushed out of a full queue, or still queued at the session's end."""
        self.failed = 0
        """Events sent but not accepted: an error, an answer other than 202, or none in time."""
        self._url = url
        self._limit = limit
        self._client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=uds), timeout=timeout
        )
        self._queue: collections.deque[dict[str, Any]] = collections.deque()
        self._batch: list[dict[str, Any]] = []
        self._seq = 0
        # The control plane remembers ids for 10 minutes: a sender made anew for the same session, as when a
        # worker rejoins its room, must not repeat the ids of the one before.
        self._nonce = uuid.uuid4().hex[:8]
        self._failing = False
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task = asyncio.create_task(self._run(), name="lifecycle-events")

    def emit(self, event_type: str, **fields: int) -> None:
        """Queue one event, stamped now; ``event_id`` is the session, the sender's nonce and a sequence
        number."""
        self._seq += 1
        if len(self._queue) >= self._limit:
            self._queue.popleft()
            self.dropped += 1
        self._queue.append(
            {
                "event_id": f"{self.session_id}-{self._nonce}-{self._seq}",
                "session_id": self.session_id,
                "at_ms": time.time_ns() // 1_000_000,
                "source": SOURCE,
                "type": event_type,
                **fields,
            }
        )
        self._idle.clear()
        self._wake.set()

    async def aclose(self) -> None:
        """Send what is queued, waiting at most ``CLOSE_TIMEOUT_S``, and stop."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._idle.wait(), CLOSE_TIMEOUT_S)
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self.failed += len(self._batch)
        self.dropped += len(self._queue)
        self._queue.clear()
        await self._client.aclose()

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            # What was queued during a POST goes out together in the next one.
            while self._queue:
                self._batch = list(self._queue)
                self._queue.clear()
                await self._post(self._batch)
                self._batch = []
            self._idle.set()

    async def _post(self, batch: list[dict[str, Any]]) -> None:
        try:
            response = await self._client.post(self._url, json=batch)
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"  # a timeout's own message can be empty
        else:
            if response.status_code == 202:
                self.sent += len(batch)
                if self._failing:
                    self._failing = False
                    logger.info("lifecycle events reach the control plane again")
                return
            error = f"HTTP {response.status_code}: {response.text[:200]}"
        self.failed += len(batch)
        if not self._failing:  # once per outage: the counts are logged when the session ends
            self._failing = True
            logger.warning(
                "lifecycle events do not reach the control plane: %s",
                error,
                extra={"session_id": self.session_id},
            )


@dataclass
class _Segment:
    """One flushed piece of a reply's audio, from its first frame until it played out or was cut off."""

    frames: list[tuple[float, float]]
    """When each frame reached the output (``time.time()``), and its length in seconds."""
    runs: list[tuple[float, float, float]]
    """The stretches that played: when each began, where it starts in the segment's audio, and how
    long it played, all in seconds."""
    played: float
    """How much of the segment played, as the output reported at its end."""


class Playout(io.AudioOutput):
    """The playable audio of each reply: a pass-through at the top of the session's audio output.

    It notes when each frame of a reply reaches the output, ready to play; the stretches the room's
    output reports as played (``playback_progressed``, which LiveKit Agents 1.8.8 reports once a
    stretch can no longer be discarded); and when the session pauses the output, as it does while it
    waits to see whether the caller's sound is an interruption. Over the segments that ended since it
    was last called, :meth:`take` gives:

    - ``audio_ms``, the audio the reply handed to the output, and ``played_ms``, how much of it played;
    - ``stalls_ms``: each time the output ran out of audio while the reply had more to come, from
      STALL_MIN_S on: the gaps between stretches of playback that no pause explains, the gap between
      two segments of the reply among them;
    - ``inventory_low_ms``: the least audio ready ahead of the playhead once synthesis had fallen
      behind it, taken just before each frame that arrived later than the frame before it lasted. A
      reply whose synthesis stayed ahead of its playback has none: its audio never ran low. The start
      of a reply is no such moment: its first frame plays as soon as it arrives.

    A sink that reports no stretch of playback gives no stalls and no inventory, which are then
    unmeasured, not zero.
    """

    def __init__(self, sink: io.AudioOutput, *, clock: Callable[[], float] = time.time) -> None:
        super().__init__(
            label="FlatsharePlayout",
            next_in_chain=sink,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._clock = clock
        """The wall clock (``time.time``), which the stretches of playback are reported on."""
        self._frames: list[tuple[float, float]] = []
        self._runs: list[tuple[float, float, float]] = []
        self._pauses: list[list[float]] = []
        """When the session paused the output and resumed it (``math.inf`` while it is paused)."""
        self._segments: list[_Segment] = []

    @property
    def sample_rate(self) -> int | None:
        if self._sample_rate is not None:
            return self._sample_rate
        return self.next_in_chain.sample_rate if self.next_in_chain else None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self._frames.append((self._clock(), frame.duration))
        if self.next_in_chain:
            await self.next_in_chain.capture_frame(frame)
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        if self.next_in_chain:
            self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.clear_buffer()

    def pause(self) -> None:
        self._pauses.append([self._clock(), math.inf])
        super().pause()

    def resume(self) -> None:
        if self._pauses and self._pauses[-1][1] == math.inf:
            self._pauses[-1][1] = self._clock()
        super().resume()

    def on_playback_progressed(self, *, started_at: float, offset: float, duration: float) -> None:
        self._runs.append((started_at, offset, duration))
        super().on_playback_progressed(started_at=started_at, offset=offset, duration=duration)

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        if self._pending_playback_count > 0:  # else LiveKit ignores it, a segment never captured
            self._segments.append(_Segment(self._frames, self._runs, playback_position))
            self._frames, self._runs = [], []
        super().on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )

    def take(self) -> dict[str, Any]:
        """The playout of the segments that ended since the last call (the class's account), in ms;
        nothing without one. Pauses that have ended are forgotten with them."""
        segments, self._segments = self._segments, []
        pauses = self._pauses
        self._pauses = [p for p in pauses if p[1] == math.inf]
        if not segments:
            return {}
        found: dict[str, Any] = {
            "audio_ms": _ms(sum(length for s in segments for _, length in s.frames)),
            "played_ms": _ms(sum(s.played for s in segments)),
        }
        if not any(s.runs for s in segments):
            return found
        # The segments play one after another, so their stretches in order are the reply's; a gap
        # runs from the end of one stretch to the start of the next.
        stretches = list(itertools.chain.from_iterable(sorted(s.runs) for s in segments))
        gaps = [(a[0] + a[2], b[0]) for a, b in itertools.pairwise(stretches)]
        found["stalls_ms"] = [
            _ms(b - a)
            for a, b in gaps
            if b - a >= STALL_MIN_S and not any(p0 < b and p1 > a for p0, p1 in pauses)
        ]
        lows = [low for s in segments if (low := _inventory_low(s)) is not None]
        if lows:
            found["inventory_low_ms"] = _ms(min(lows))
        return found


def _inventory_low(segment: _Segment) -> float | None:
    """The least audio ready ahead of a segment's playhead just before a frame that arrived later than
    the frame before it lasted, once it had begun to play; None without such a frame, or without a
    stretch of playback to place the playhead by."""
    runs = sorted(segment.runs)
    if not runs:
        return None

    def playhead(at: float) -> float:
        position = 0.0
        for started, offset, length in runs:
            if at < started:
                break
            position = offset + min(at - started, length)
        return position

    lows, ready = [], 0.0
    for (before, length), (at, _) in itertools.pairwise(segment.frames):
        ready += length  # the audio that arrived before this frame
        if at >= runs[0][0] and at - before > length:
            lows.append(max(0.0, ready - playhead(at)))
    return min(lows) if lows else None


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


class Lifecycle:
    """One session's lifecycle events, and a line per turn with its stage times.

    Without a sender (``FLATSHARE_EVENTS=off``, or an LLM that bypasses the control plane) only the turn
    lines are logged.
    """

    def __init__(self, session_id: str, sender: EventSender | None) -> None:
        self.session_id = session_id
        self.turns = 0
        self._sender = sender
        self._user: llm.MetricsReport = {}
        """The metrics of the caller's last message, until the reply to it is added."""
        self._playout: Playout | None = None

    def attach(self, session: AgentSession) -> None:
        """Watch a started session: its audio output exists only from then on. The replies' audio
        then passes through a ``Playout`` on its way to it, before the first reply: the agent never
        speaks first."""
        session.on("user_state_changed", _safely(self._user_state_changed))
        session.on("conversation_item_added", _safely(self._item_added))
        session.on("close", _safely(self._closed))
        if (audio := session.output.audio) is not None:
            self._playout = Playout(audio)
            session.output.audio = self._playout
            self._playout.on("playback_started", _safely(self._playback_started))
            self._playout.on("playback_finished", _safely(self._playback_finished))

    async def aclose(self) -> None:
        """Send the last events and log the session's end, with its turns and event counts."""
        counts: dict[str, int] = {}
        if self._sender is not None:
            await self._sender.aclose()
            sender = self._sender
            counts = {
                "events_sent": sender.sent,
                "events_dropped": sender.dropped,
                "events_failed": sender.failed,
            }
        logger.info(
            "session ended", extra={"session_id": self.session_id, "turns": self.turns, **counts}
        )

    def _emit(self, event_type: str, **fields: int) -> None:
        if self._sender is not None:
            self._sender.emit(event_type, **fields)

    def _user_state_changed(self, ev: UserStateChangedEvent) -> None:
        if ev.new_state == "speaking":
            self._emit("user_speech_started")

    def _item_added(self, ev: ConversationItemAddedEvent) -> None:
        message = ev.item
        if not isinstance(message, llm.ChatMessage):  # an agent handoff
            return
        if message.role == "user":
            self._emit("user_turn_ended")
            self._user = message.metrics
            if self._playout is not None:
                # Audio no reply has claimed, which no later reply's line may take for its own.
                self._playout.take()
        elif message.role == "assistant":
            self._log_turn(message)

    def _playback_started(self, _: io.PlaybackStartedEvent) -> None:
        self._emit("playback_started")

    def _playback_finished(self, ev: io.PlaybackFinishedEvent) -> None:
        played_ms = round(ev.playback_position * 1000)
        self._emit("interrupted" if ev.interrupted else "playback_ended", played_ms=played_ms)

    def _closed(self, _: CloseEvent) -> None:
        self._emit("session_closed")

    def _log_turn(self, reply: llm.ChatMessage) -> None:
        """One line per reply, from LiveKit's metrics of the caller's message and of the reply:

        - ``transcript_ms``: from the end of speech (VAD) to the final transcript;
        - ``end_of_turn_ms``: from the end of speech to the end-of-turn decision;
        - ``llm_ttft_ms`` and ``llm_first_sentence_ms``: from the LLM request to its first token, and
          to the first sentence handed to TTS;
        - ``turn_to_first_sentence_ms``: from the end-of-turn decision to that sentence;
        - ``tts_first_audio_ms``: from that sentence to its first audio;
        - ``e2e_ms``: from the end of speech to the reply's first audio frame on the track, the agent's
          side of voice-to-voice.

        A stage LiveKit did not measure is left out. The line ends with the reply's playout
        (``Playout``): it plays out before LiveKit adds the reply.
        """
        user, metrics = self._user, reply.metrics
        self._user = {}
        stages = {
            "transcript_ms": user.get("transcription_delay"),
            "end_of_turn_ms": user.get("end_of_turn_delay"),
            "llm_ttft_ms": metrics.get("llm_node_ttft"),
            "llm_first_sentence_ms": metrics.get("llm_node_ttfs"),
            "turn_to_first_sentence_ms": None,
            "tts_first_audio_ms": metrics.get("tts_node_ttfb"),
            "e2e_ms": metrics.get("e2e_latency"),
        }
        # The reply's first frame went out on the track e2e_latency after the end of speech, its
        # playback_latency after it was forwarded. It was forwarded as soon as TTS made it, its
        # tts_node_ttfb after the first sentence went in, because TTS starts only once the reply is
        # scheduled (preemptive generation runs the LLM alone).
        parts = (
            metrics.get("e2e_latency"),
            metrics.get("playback_latency"),
            metrics.get("tts_node_ttfb"),
            user.get("end_of_turn_delay"),
        )
        if all(part is not None for part in parts):
            e2e, playback, tts, end_of_turn = parts
            stages["turn_to_first_sentence_ms"] = e2e - playback - tts - end_of_turn
        times = {stage: _ms(s) for stage, s in stages.items() if s is not None}
        playout = self._playout.take() if self._playout is not None else {}
        turn_logger.info(
            "turn",
            extra={
                "session_id": self.session_id,
                "turn": self.turns,
                "interrupted": reply.interrupted,
                **times,
                **playout,
            },
        )
        self.turns += 1


def observe(
    ctx: JobContext, session: AgentSession, session_id: str, *, uds: str | None
) -> Lifecycle:
    """Report a started session's lifecycle: its events to the control plane through ``uds`` (none
    without one) and a line per turn. The job's shutdown sends the last events."""
    sender = EventSender(session_id, uds=uds) if uds is not None else None
    lifecycle = Lifecycle(session_id, sender)
    lifecycle.attach(session)
    ctx.add_shutdown_callback(lifecycle.aclose)
    return lifecycle


def _safely(handler: Callable[[Any], None]) -> Callable[[Any], None]:
    """``handler``, with its errors logged rather than raised: LiveKit's event emitters pass a
    handler's TypeError on to the code that emitted the event, the audio path among it."""

    def guarded(event: Any) -> None:
        try:
            handler(event)
        except Exception:
            logger.exception("a lifecycle hook failed")

    return guarded
