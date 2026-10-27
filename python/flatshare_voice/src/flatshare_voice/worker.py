"""The Flatshare voice agent: a stock LiveKit Agents worker for realtime voice sessions.

``flatshare-voice-worker start`` registers the agent ``flatshare-voice`` with the LiveKit server and
serves every dispatched room with one ``AgentSession``: Silero VAD and LiveKit's audio
end-of-turn model (``turn-detector-v1-mini``, run locally) on the CPU, then STT, the LLM and TTS
on the GPU node. Turn handling (endpointing, interruptions, preemptive generation) keeps LiveKit's
defaults, with two settings pinned for this deployment, whose only callers are the measurement
bots inside the namespace:

- interruptions by VAD, which production mode picks anyway: adaptive detection needs LiveKit
  Inference, and the namespace has no route out;
- no AEC warm-up: the callers play files, so there is no echo to wait out, and the 3 s warm-up
  would only discard a barge-in on the first reply of each call.

Experiments use ``FLATSHARE_TURN_DETECTION=vad`` instead: the end of a turn is then a fixed delay after
the end of speech, the same for every arm, so voice-to-voice latency moves only with serving. The
end-of-turn model waits up to the endpointing maximum (3 s) whenever it judges a line unfinished,
which on synthesized callers happened on about one turn in six (W1) and would hide serving effects
in the tail. ``FLATSHARE_PREEMPTIVE_GENERATION=false`` turns off LiveKit's preemptive generation, which
starts the reply from the final transcript before the end of the turn is confirmed: one of E2's
factors.

Each session posts its lifecycle events (speech onset, end of turn, playback, barge-in) to the
control plane over the LLM's socket, and logs a line per turn with its stage times (:mod:`.events`).
``FLATSHARE_EVENTS=off`` stops the events, so that E2 can measure arm 9, the lifecycle policy, with
and without them; they stop as well when the LLM bypasses the control plane, whose socket alone
takes them.

The worker runs inside the voice-plane network namespace, where the only route to the servers is
a Unix socket each. The environment, set by ``flatshare-voice-worker.service``, configures it:

| Variable | Meaning |
|---|---|
| ``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, ``LIVEKIT_API_SECRET`` | The LiveKit server |
| ``FLATSHARE_LLM_UDS`` | The control plane's socket (the LLM path), or the engine's tunnel |
| ``FLATSHARE_STT_UDS``, ``FLATSHARE_TTS_UDS`` | The tunnels to the STT and TTS servers |
| ``FLATSHARE_IDLE_PROCESSES`` | Pre-started job processes (default 2) |
| ``FLATSHARE_LLM_TIMEOUT`` | Seconds before an LLM request fails (default 30: a cold prefill exceeds 10) |
| ``FLATSHARE_TURN_DETECTION`` | ``model`` (default: the end-of-turn model) or ``vad`` (fixed endpointing, for experiments) |
| ``FLATSHARE_PREEMPTIVE_GENERATION`` | ``true`` (default, LiveKit's) or ``false`` |
| ``FLATSHARE_EVENTS`` | ``on`` (default: lifecycle events to the control plane) or ``off`` |
| ``FLATSHARE_LLM_MODEL`` | The model the agent's requests name (default the box's, ``plugins.LLM_MODEL``) |

The worker reads them whenever it is to serve (``start``, which the unit runs, or ``dev``, ``console``
or ``connect``), before it registers with LiveKit (:class:`Settings`): a socket that is not set, or a
value its setting does not take, stops it with a line that names the setting, so that no call reaches
a worker that would fail it.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import silero

from . import events, plugins
from .protocol import AGENT_NAME, room_session_id

BASE_URL = "http://flatshare.invalid/v1"
"""Requests go over the Unix sockets; the host name never resolves."""

ENGINE_TUNNEL = "-engine.sock"
"""How the tunnel names its forward to an engine (``phastform-engine.sock``), the bypass's LLM path."""

INSTRUCTIONS = (
    "You are Ava, the phone assistant of a small neighborhood restaurant. You take reservations, "
    "answer questions about the menu, opening hours and directions, and keep every answer to one "
    "or two short, natural sentences. You are speaking, so never use lists, markdown or emoji."
)


SERVES = frozenset({"start", "dev", "console", "connect"})
"""LiveKit's worker commands that run the agent's sessions, for which :func:`main` reads the settings."""


@dataclass(frozen=True)
class Settings:
    """The worker's settings, from its environment (the table above), each checked as it is read."""

    llm_uds: str
    stt_uds: str
    tts_uds: str
    idle_processes: int = 2
    llm_timeout: float = 30.0
    turn_detection: str = "model"
    preemptive_generation: bool = True
    events: bool = True
    llm_model: str = plugins.LLM_MODEL

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        """The settings ``env`` gives; a socket it does not set, or a value that is not one of its
        setting's, is a ValueError that names the setting."""
        for name in ("FLATSHARE_LLM_UDS", "FLATSHARE_STT_UDS", "FLATSHARE_TTS_UDS"):
            if not env.get(name):
                raise ValueError(f"{name} is not set: the worker reaches its servers by sockets")
        preemptive = _choice(env, "FLATSHARE_PREEMPTIVE_GENERATION", ("true", "false"))
        return cls(
            llm_uds=env["FLATSHARE_LLM_UDS"],
            stt_uds=env["FLATSHARE_STT_UDS"],
            tts_uds=env["FLATSHARE_TTS_UDS"],
            idle_processes=_count(env, "FLATSHARE_IDLE_PROCESSES", 2),
            llm_timeout=_seconds(env, "FLATSHARE_LLM_TIMEOUT", 30.0),
            turn_detection=_choice(env, "FLATSHARE_TURN_DETECTION", ("model", "vad")),
            preemptive_generation=preemptive == "true",
            events=_choice(env, "FLATSHARE_EVENTS", ("on", "off")) == "on",
            llm_model=env.get("FLATSHARE_LLM_MODEL") or plugins.LLM_MODEL,
        )

    def turn_handling(self) -> dict[str, Any]:
        """LiveKit's turn handling as the settings ask; interruptions are always detected by VAD."""
        # Fixed endpointing, or the local end-of-turn model: the cloud one needs LiveKit Cloud, and
        # the namespace has no route out.
        detection = (
            "vad" if self.turn_detection == "vad" else inference.TurnDetector(version="v1-mini")
        )
        return {
            "turn_detection": detection,
            "interruption": {"mode": "vad"},
            "preemptive_generation": {"enabled": self.preemptive_generation},
        }

    @property
    def events_socket(self) -> str | None:
        """The socket lifecycle events go to: the LLM's, unless they are off or the LLM bypasses the
        control plane through the engine's tunnel, which has no ``/v1/events``."""
        return self.llm_uds if self.events and not self.llm_uds.endswith(ENGINE_TUNNEL) else None


def _choice(env: Mapping[str, str], name: str, choices: tuple[str, ...]) -> str:
    """Setting ``name``, one of ``choices``, the first of them by default."""
    value = env.get(name, choices[0])
    if value not in choices:
        raise ValueError(f"{name} must be {' or '.join(choices)}, got {value!r}")
    return value


def _count(env: Mapping[str, str], name: str, default: int) -> int:
    """Setting ``name``, a count of 0 or more."""
    text = env.get(name, str(default))
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{name} must be a count of 0 or more, got {text!r}")
    return value


def _seconds(env: Mapping[str, str], name: str, default: float) -> float:
    """Setting ``name``, a number of seconds above 0."""
    text = env.get(name, str(default))
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not 0 < value < math.inf:
        raise ValueError(f"{name} must be a number of seconds above 0, got {text!r}")
    return value


def prewarm(proc: JobProcess) -> None:
    """Load the VAD once per job process, before a room is assigned."""
    proc.userdata["vad"] = silero.VAD.load()


# The idle processes' count is the settings', which main() reads before the server starts.
server = AgentServer(
    host="127.0.0.1",  # the default "" puts the health server on every interface
    port=0,
    load_fnc=lambda: 0.0,  # other tenants' CPU load on a shared host must not drain the worker
    setup_fnc=prewarm,
)


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    settings = Settings.from_env()  # as main() read them, in this job's process
    session_id = room_session_id(ctx.job.room.sid)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=plugins.make_stt(BASE_URL, uds=settings.stt_uds),
        llm=plugins.make_llm(BASE_URL, session_id, uds=settings.llm_uds, model=settings.llm_model),
        tts=plugins.make_tts(BASE_URL, uds=settings.tts_uds),
        turn_handling=settings.turn_handling(),
        aec_warmup_duration=None,
        conn_options=SessionConnectOptions(
            # One retry, not LiveKit's three: a dropped connection still gets a second try, and a
            # stalled turn gives up after two timeouts rather than four.
            llm_conn_options=APIConnectOptions(max_retry=1, timeout=settings.llm_timeout)
        ),
    )
    # One caller per room: when the caller hangs up, the session closes and the room is deleted.
    room_options = room_io.RoomOptions(delete_room_on_close=True)
    await session.start(
        agent=Agent(instructions=INSTRUCTIONS),
        room=ctx.room,
        room_options=room_options,
        record=False,
    )
    # Nothing it reports can happen earlier: the caller's audio flows only once start() returns.
    events.observe(ctx, session, session_id, uds=settings.events_socket)


def main() -> None:
    """LiveKit's worker command line. A command that serves reads the settings first, so that a
    setting it cannot take stops the worker before it registers."""
    if sys.argv[1:2] and sys.argv[1] in SERVES:
        try:
            settings = Settings.from_env()
        except ValueError as error:
            sys.exit(f"flatshare-voice-worker: {error}")
        server.update_options(num_idle_processes=settings.idle_processes)
    cli.run_app(server)


if __name__ == "__main__":
    main()
