"""The Flatshare voice agent: a stock LiveKit Agents worker for realtime voice sessions.

``flatshare-voice-worker start`` registers the agent ``flatshare-voice`` with the LiveKit server and
serves every dispatched room with one ``AgentSession``: Silero VAD and LiveKit's audio
end-of-turn model (``turn-detector-v1-mini``, run locally) on the CPU, then STT, the LLM and TTS
on the GPU node.

It runs in one of two modes (``FLATSHARE_VOICE_MODE``):

- ``bench``, the default and what every measurement ran: the worker takes every call LiveKit
  dispatches, its load always 0, so that other tenants' CPU load on a shared host cannot drain it.
- ``serve``: call-capacity admission (:mod:`.admission`). The serving profile
  (``FLATSHARE_SERVING_PROFILE``, a serving profile's serving.json) names the calls the measured
  envelope admits: the worker's load is its calls over that number, with LiveKit's load threshold
  at 1, so that a full worker is marked full and the server sends it no job, and a job request
  beyond it is refused. It takes no call while the gateway says the engine does not serve
  (``/ready`` on the LLM's socket, or the engine's ``/health`` on the bypass), or while the file
  ``FLATSHARE_VOICE_DRAIN`` names exists (a release's drain), and writes its state to the file
  ``FLATSHARE_VOICE_STATUS`` names every second.

Two audio configurations (``FLATSHARE_AUDIO``), :data:`AUDIO`. ``synthetic``, bench mode's default,
is what the measurements ran: the measurement bots inside the namespace play files, so turn handling
keeps LiveKit's defaults with two settings pinned:

- interruptions by VAD, which production mode picks anyway: adaptive detection needs LiveKit
  Inference, and the namespace has no route out;
- no AEC warm-up: the callers play files, so there is no echo to wait out, and the 3 s warm-up
  would only discard a barge-in on the first reply of each call.

``microphone``, serve mode's default (the serving profile names it), is for real callers: the
end-of-turn model, LiveKit's AEC warm-up for the echo of a loudspeaker, and a VAD and interruptions
that a noisy room trips less. No benchmark measured it: the callers E2 measures are synthetic.

Experiments use ``FLATSHARE_TURN_DETECTION=vad`` instead: the end of a turn is then a fixed delay after
the end of speech, the same for every arm, so voice-to-voice latency moves only with serving. The
end-of-turn model waits up to the endpointing maximum (3 s) whenever it judges a line unfinished,
which on synthesized callers happened on about one turn in six (W1) and would hide serving effects
in the tail. ``FLATSHARE_PREEMPTIVE_GENERATION=false`` turns off LiveKit's preemptive generation, which
starts the reply from the final transcript before the end of the turn is confirmed: one of E2's
factors.

With node binding (``FLATSHARE_NODE_BINDING``, serve mode's default), a call starts by asking the
gateway which GPU node it is on, binds its STT and TTS to that node's sockets and its LLM to the
gateway with the session id, and when its node fails it ends the reply under way and asks again at
the caller's next speech, the conversation's transcript kept (:mod:`.nodes`); a gateway without
nodes answers that it places nothing, and the call is served on the configured sockets as before.

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
| ``FLATSHARE_VOICE_MODE`` | ``bench`` (default: every call taken) or ``serve`` (call-capacity admission) |
| ``FLATSHARE_SERVING_PROFILE`` | Serve mode: the serving profile's serving.json, whose ``voice`` table it runs |
| ``FLATSHARE_AUDIO`` | ``synthetic`` (bench mode's default) or ``microphone`` (serve mode's: the profile's) |
| ``FLATSHARE_ENDPOINTING_MIN_DELAY`` | Seconds after the end of speech before a turn may end, from 0 to 3, over the audio configuration's (LiveKit's 0.5 by default) |
| ``FLATSHARE_VAD_MIN_SILENCE`` | Seconds of silence that end speech for Silero VAD, above 0 and at most 3, over the audio configuration's (Silero's 0.55 by default) |
| ``FLATSHARE_VOICE_STATUS`` | Serve mode: the file its state is written to every second |
| ``FLATSHARE_VOICE_DRAIN`` | Serve mode: while this file exists, the worker takes no new call |
| ``FLATSHARE_NODE_BINDING`` | ``on`` (serve mode's default): each call asks the gateway for its GPU node and binds its STT and TTS to that node's sockets (:mod:`.nodes`); ``off`` (bench mode's default) |

The worker reads them whenever it is to serve (``start``, which the unit runs, or ``dev``, ``console``
or ``connect``), before it registers with LiveKit (:class:`Settings`): a socket that is not set, or a
value its setting does not take, stops it with a line that names the setting, so that no call reaches
a worker that would fail it.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    JobContext,
    JobProcess,
    JobRequest,
    cli,
    inference,
    room_io,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import silero

from . import events, nodes, plugins
from .admission import Admission, EngineProbe, Monitor
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

MAX_DELAY_S = 3.0
"""LiveKit's longest endpointing delay, its default ``max_delay``: the most a turn's end may wait, and so the most
either of the endpointing settings may set."""

AUDIO: dict[str, dict[str, Any]] = {
    # The measurement bots' audio, as every E2 point ran: files played into the room, so no echo and no room noise.
    "synthetic": {
        "vad": {},  # Silero's defaults: activation 0.5, 0.05 s of speech, 0.55 s of silence
        "endpointing": None,  # LiveKit's: 0.5 s at least after speech, 3 s at most
        "interruption": {"mode": "vad"},
        "aec_warmup_s": None,
    },
    # Real callers' microphones, not measured by any benchmark. The end-of-turn model decides the end of a turn, as
    # people pause inside a sentence (FLATSHARE_TURN_DETECTION=vad is refused with it). The VAD needs a clearer
    # and longer onset to count speech (0.6 against 0.5, 0.1 s against 0.05), so that a click or a door is no
    # barge-in, and so does an interruption (0.6 s against 0.5); one that no transcript follows within 2 s is taken
    # for a false one, and the agent resumes (LiveKit's default). LiveKit's 3 s AEC warm-up stays on: a
    # loudspeaker's echo of the agent would otherwise cut its first reply. Noise suppression is the caller's
    # client's (a browser's WebRTC applies its own), since LiveKit's noise cancellation needs LiveKit Cloud, which
    # the namespace cannot reach; the room input's gain control is LiveKit's default.
    "microphone": {
        "vad": {"activation_threshold": 0.6, "min_speech_duration": 0.1},
        "endpointing": {"mode": "fixed", "min_delay": 0.5, "max_delay": 3.0},
        "interruption": {
            "mode": "vad",
            "min_duration": 0.6,
            "resume_false_interruption": True,
            "false_interruption_timeout": 2.0,
        },
        "aec_warmup_s": 3.0,
    },
}
"""The audio configurations (``FLATSHARE_AUDIO``): Silero VAD's options, LiveKit's endpointing and interruption
handling, and the AEC warm-up (None: none)."""


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
    mode: str = "bench"
    audio: str = "synthetic"
    max_calls: int | None = None
    serving_profile: str | None = None
    status_file: str | None = None
    drain_file: str | None = None
    serving: Mapping[str, Any] = field(default_factory=dict)
    node_binding: bool = False
    endpointing_min_delay: float | None = None
    """Over the audio configuration's, as the endpointing sweep sets it; None keeps the configuration's."""
    vad_min_silence: float | None = None
    """Over the audio configuration's VAD, as the endpointing sweep sets it; None keeps the configuration's."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        """The settings ``env`` gives; a socket it does not set, or a value that is not one of its
        setting's, is a ValueError that names the setting."""
        for name in ("FLATSHARE_LLM_UDS", "FLATSHARE_STT_UDS", "FLATSHARE_TTS_UDS"):
            if not env.get(name):
                raise ValueError(f"{name} is not set: the worker reaches its servers by sockets")
        preemptive = _choice(env, "FLATSHARE_PREEMPTIVE_GENERATION", ("true", "false"))
        mode = _choice(env, "FLATSHARE_VOICE_MODE", ("bench", "serve"))
        serving = _serving(env) if mode == "serve" else {}
        audio = env.get("FLATSHARE_AUDIO") or serving.get("audio", "synthetic")
        if audio not in AUDIO:
            raise ValueError(f"FLATSHARE_AUDIO must be {' or '.join(AUDIO)}, got {audio!r}")
        turn_detection = _choice(env, "FLATSHARE_TURN_DETECTION", ("model", "vad"))
        if audio == "microphone" and turn_detection == "vad":
            raise ValueError(
                "FLATSHARE_TURN_DETECTION=vad is the synthetic callers' fixed endpointing; the "
                "microphone configuration ends a turn by the end-of-turn model (unlink vad-turns)"
            )
        # Serve mode asks the gateway for each call's node, bench mode only when told to.
        binding = _choice(
            env, "FLATSHARE_NODE_BINDING", ("on", "off") if mode == "serve" else ("off", "on")
        )
        if binding == "on" and env["FLATSHARE_LLM_UDS"].endswith(ENGINE_TUNNEL):
            raise ValueError(
                "FLATSHARE_NODE_BINDING=on asks the gateway for each call's node, and FLATSHARE_LLM_UDS "
                "bypasses it for an engine's tunnel: set FLATSHARE_NODE_BINDING=off"
            )
        return cls(
            llm_uds=env["FLATSHARE_LLM_UDS"],
            stt_uds=env["FLATSHARE_STT_UDS"],
            tts_uds=env["FLATSHARE_TTS_UDS"],
            idle_processes=_count(env, "FLATSHARE_IDLE_PROCESSES", 2),
            llm_timeout=_seconds(env, "FLATSHARE_LLM_TIMEOUT", 30.0),
            turn_detection=turn_detection,
            preemptive_generation=preemptive == "true",
            events=_choice(env, "FLATSHARE_EVENTS", ("on", "off")) == "on",
            llm_model=env.get("FLATSHARE_LLM_MODEL") or plugins.LLM_MODEL,
            mode=mode,
            audio=audio,
            max_calls=serving.get("max_calls"),
            serving_profile=env.get("FLATSHARE_SERVING_PROFILE") if mode == "serve" else None,
            status_file=env.get("FLATSHARE_VOICE_STATUS") or None,
            drain_file=env.get("FLATSHARE_VOICE_DRAIN") or None,
            serving=serving,
            node_binding=binding == "on",
            endpointing_min_delay=_bounded(
                env, "FLATSHARE_ENDPOINTING_MIN_DELAY", 0.0, MAX_DELAY_S
            ),
            vad_min_silence=_bounded(
                env, "FLATSHARE_VAD_MIN_SILENCE", 0.0, MAX_DELAY_S, low_ok=False
            ),
        )

    def turn_handling(self) -> dict[str, Any]:
        """LiveKit's turn handling as the settings ask; interruptions are always detected by VAD."""
        # Fixed endpointing, or the local end-of-turn model: the cloud one needs LiveKit Cloud, and
        # the namespace has no route out.
        detection = (
            "vad" if self.turn_detection == "vad" else inference.TurnDetector(version="v1-mini")
        )
        audio = AUDIO[self.audio]
        handling: dict[str, Any] = {
            "turn_detection": detection,
            "interruption": dict(audio["interruption"]),
            "preemptive_generation": {"enabled": self.preemptive_generation},
        }
        endpointing = dict(audio["endpointing"] or {})
        if self.endpointing_min_delay is not None:
            # LiveKit keeps its defaults for the keys left out: the fixed mode and 3 s at most.
            endpointing["min_delay"] = self.endpointing_min_delay
        if endpointing:
            handling["endpointing"] = endpointing
        return handling

    def vad_options(self) -> dict[str, float]:
        """Silero VAD's options for the audio configuration (its defaults for synthetic callers), with the silence that
        ends speech set over them when the settings name one."""
        options = dict(AUDIO[self.audio]["vad"])
        if self.vad_min_silence is not None:
            options["min_silence_duration"] = self.vad_min_silence
        return options

    @property
    def aec_warmup(self) -> float | None:
        """Seconds of the agent's first speech in which the caller's audio cannot interrupt it."""
        return AUDIO[self.audio]["aec_warmup_s"]

    @property
    def probe_path(self) -> str:
        """Where the LLM's socket says whether the engine serves: the gateway's /ready, or the
        engine's /health on the bypass."""
        return "/health" if self.llm_uds.endswith(ENGINE_TUNNEL) else "/ready"

    def describe(self) -> dict[str, Any]:
        """What the worker runs, for its status file: what deploy/bin/serving-check holds to the
        serving profile's ``voice`` table, and the turn handling."""
        return {
            "mode": self.mode,
            "max_calls": self.max_calls,
            "audio": self.audio,
            "turn_detection": self.turn_detection,
            "serving_profile": self.serving_profile,
            "profile": self.serving.get("profile"),
            "node_binding": self.node_binding,
            "endpointing_min_delay": self.endpointing_min_delay,
            "vad_min_silence": self.vad_min_silence,
        }

    @property
    def events_socket(self) -> str | None:
        """The socket lifecycle events go to: the LLM's, unless they are off or the LLM bypasses the
        control plane through the engine's tunnel, which has no ``/v1/events``."""
        return self.llm_uds if self.events and not self.llm_uds.endswith(ENGINE_TUNNEL) else None


def _serving(env: Mapping[str, str]) -> dict[str, Any]:
    """Serve mode's ``voice`` table of the serving profile FLATSHARE_SERVING_PROFILE names, with
    the profile's name."""
    path = env.get("FLATSHARE_SERVING_PROFILE")
    if not path:
        raise ValueError(
            "FLATSHARE_SERVING_PROFILE is not set: serve mode admits the calls its serving.json names"
        )
    try:
        profile = json.loads(open(path, encoding="utf-8").read())  # noqa: SIM115
        voice = profile["voice"]
        max_calls = voice["max_calls"]
        audio = voice["audio"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"FLATSHARE_SERVING_PROFILE {path} is no serving profile's serving.json: {error!r}"
        ) from None
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
        raise ValueError(f"FLATSHARE_SERVING_PROFILE {path}: voice.max_calls must be 1 or more")
    return {"max_calls": max_calls, "audio": audio, "profile": profile.get("profile")}


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


def _bounded(
    env: Mapping[str, str], name: str, low: float, high: float, *, low_ok: bool = True
) -> float | None:
    """Setting ``name``, a number of seconds from ``low`` (or above it) to ``high``; None when it is not set."""
    text = env.get(name)
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not ((low <= value if low_ok else low < value) and value <= high):
        bound = f"from {low:g} to {high:g}" if low_ok else f"above {low:g} and at most {high:g}"
        raise ValueError(f"{name} must be a number of seconds {bound}, got {text!r}")
    return value


def prewarm(proc: JobProcess) -> None:
    """Load the VAD once per job process, before a room is assigned, with the audio configuration's
    options."""
    proc.userdata["vad"] = silero.VAD.load(**Settings.from_env().vad_options())


ADMISSION: Admission | None = None
"""Serve mode's admission, which main() sets up; None in bench mode."""


def load(worker: AgentServer) -> float:
    """The worker's load as LiveKit reads it. Bench mode: 0, as every measurement ran, so that other
    tenants' CPU load on a shared host cannot drain the worker. Serve mode: its calls over the
    calls the profile admits, and 1 while it takes none."""
    return 0.0 if ADMISSION is None else ADMISSION.load()


async def request(job: JobRequest) -> None:
    """A job request: bench mode accepts it, as LiveKit's default does; serve mode's admission
    decides."""
    if ADMISSION is None:
        await job.accept()
    else:
        await ADMISSION.request(job)


# The idle processes' count is the settings', which main() reads before the server starts.
server = AgentServer(
    host="127.0.0.1",  # the default "" puts the health server on every interface
    port=0,
    load_fnc=load,
    setup_fnc=prewarm,
)


async def bind_call(settings: Settings, session_id: str) -> nodes.CallBinding | None:
    """The call's node binding (:mod:`.nodes`), placed: None with node binding off, or where the gateway
    has no nodes (it answers that it places nothing), and the call is served on the configured sockets as
    it always was. A placement that did not answer leaves the binding to ask again at the call's first
    request."""
    if not settings.node_binding:
        return None
    binding = nodes.CallBinding(session_id, settings.llm_uds, settings.stt_uds, settings.tts_uds)
    await binding.place("start")
    if binding.nodes is False:
        await binding.aclose()
        return None
    return binding


def call_plugins(
    settings: Settings, session_id: str, binding: nodes.CallBinding | None
) -> tuple[Any, Any, Any]:
    """The call's STT, LLM and TTS: on the configured sockets, or through ``binding``'s transports, which
    send each request to the call's node."""

    def transport(server: str) -> Any:
        return None if binding is None else binding.transport(server, plugins.LIMITS)

    return (
        plugins.make_stt(BASE_URL, uds=settings.stt_uds, transport=transport("stt")),
        plugins.make_llm(
            BASE_URL,
            session_id,
            uds=settings.llm_uds,
            model=settings.llm_model,
            transport=transport("llm"),
        ),
        plugins.make_tts(BASE_URL, uds=settings.tts_uds, transport=transport("tts")),
    )


@server.rtc_session(agent_name=AGENT_NAME, on_request=request)
async def entrypoint(ctx: JobContext) -> None:
    settings = Settings.from_env()  # as main() read them, in this job's process
    session_id = room_session_id(ctx.job.room.sid)
    # The call's GPU node, before anything of it runs, where the gateway fronts several.
    binding = await bind_call(settings, session_id)
    stt, llm, tts = call_plugins(settings, session_id, binding)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt,
        llm=llm,
        tts=tts,
        turn_handling=settings.turn_handling(),
        aec_warmup_duration=settings.aec_warmup,
        conn_options=SessionConnectOptions(
            # One retry, not LiveKit's three: a dropped connection still gets a second try, and a
            # stalled turn gives up after two timeouts rather than four.
            llm_conn_options=APIConnectOptions(max_retry=1, timeout=settings.llm_timeout)
        ),
    )
    if binding is not None:
        nodes.watch(
            session, binding
        )  # a node failure ends the reply; the caller's next turn rebinds
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


def active_calls() -> int:
    """The calls in progress: LiveKit's running jobs, none before the server runs (its process pool
    is made then, after main() has started serve mode's monitor)."""
    try:
        return len(server.active_jobs)
    except AttributeError:
        return 0


def serve(settings: Settings) -> Admission:
    """Serve mode's admission over ``server``'s running jobs, its probe of the engine and its
    status file, started; LiveKit's load threshold at 1 (a plain float, so that dev mode holds it
    too), so that the worker is full at max_calls."""
    global ADMISSION
    assert settings.max_calls is not None

    async def place(job: Any) -> tuple[bool, str]:
        return await nodes.admit(settings.llm_uds, room_session_id(job.room.sid))

    ADMISSION = Admission(
        settings.max_calls,
        active=active_calls,
        probe=EngineProbe(settings.llm_uds, settings.probe_path),
        drain=settings.drain_file,
        place=place if settings.node_binding else None,
    )
    server.update_options(load_threshold=1.0)
    Monitor(ADMISSION, settings.status_file, settings.describe()).start()
    return ADMISSION


def main() -> None:
    """LiveKit's worker command line. A command that serves reads the settings first, so that a
    setting it cannot take stops the worker before it registers."""
    if sys.argv[1:2] and sys.argv[1] in SERVES:
        try:
            settings = Settings.from_env()
        except ValueError as error:
            sys.exit(f"flatshare-voice-worker: {error}")
        server.update_options(num_idle_processes=settings.idle_processes)
        if settings.mode == "serve":
            serve(settings)
    cli.run_app(server)


if __name__ == "__main__":
    main()
