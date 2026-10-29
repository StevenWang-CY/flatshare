"""A call's GPU node (flatshare_voice.nodes), against a fake gateway that places sessions (POST /v1/sessions, the
contract in the README's "Nodes") and two nodes' mock STT and TTS servers on Unix sockets: the call's speech follows
the node the gateway names; a gateway without nodes changes nothing; a node that fails ends the reply under way, and
nothing of it is generated or played again; the caller's next speech places the call anew, and its STT, TTS and LLM
go to the new node with the conversation intact."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from aiohttp import web
from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, llm

from flatshare_voice import nodes, plugins, worker
from flatshare_voice.worker import Settings

from .mocks import FIRST_SENTENCE, TRANSCRIPT, OpenAIMocks, _tone, gateway_answer, gateway_event

BASE_URL = "http://flatshare.invalid/v1"
SESSION = "voice-RM_a1b2c3"
HALF_SECOND = rtc.AudioFrame(np.zeros(8000, np.int16).tobytes(), 16000, 1, 8000)


class Gateway:
    """The gateway's socket as a call sees it: ``POST /v1/sessions`` answers ``status`` and the placement on
    ``node``, its speech sockets as the gateway's ``[[nodes]]`` name them (``unix:`` addresses in ``sockets``, the
    tunnels' directory), the gateway's own answer for ``refusal`` otherwise (its 404 for a ``status`` of 404), or
    nothing while ``hang``; the chat route streams a reply, breaks it off with the gateway's error event for ``error``
    (``stream = "broken"``), or answers with the gateway's own answer for ``error``, whose status is ``chat_status``.
    The errors are the gateway's, from its fixture of them. Every request is kept, with its headers, and each session
    ``DELETE /v1/sessions/ID`` ends."""

    def __init__(self, sockets: Path) -> None:
        self.sockets = sockets
        self.node = "gray"
        self.status = 200
        self.refusal = "no_call_slot"
        self.hang = False
        self.chat = "ok"
        self.chat_status = 200
        self.error = "upstream_unavailable"
        self.placements: list[dict[str, Any]] = []
        self.chats: list[dict[str, Any]] = []
        self.ended: list[str] = []
        self.released = asyncio.Event()

    async def sessions(self, request: web.Request) -> web.Response:
        headers = {name.lower(): value for name, value in request.headers.items()}
        self.placements.append({"headers": headers, "body": await request.json()})
        if self.hang:
            await self.released.wait()
        if self.status != 200:
            status, body = gateway_answer("not_found" if self.status == 404 else self.refusal)
            return web.json_response(body, status=status)
        answer = {"session_id": SESSION, "class": "realtime", "node": self.node}
        for server in ("stt", "tts"):
            answer[server] = f"unix:{self.sockets / f'{self.node}-{server}.sock'}"
        return web.json_response(answer)

    async def end(self, request: web.Request) -> web.Response:
        self.ended.append(request.match_info["id"])
        return web.json_response({"session_id": request.match_info["id"], "ended": True})

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.chats.append(
            {"headers": {k.lower(): v for k, v in request.headers.items()}, "body": body}
        )
        if self.chat_status != 200:
            status, error = gateway_answer(self.error)
            assert status == self.chat_status, (self.error, status)
            return web.json_response(error, status=status)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        chunk = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": FIRST_SENTENCE}, "finish_reason": None}],
        }
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if self.chat == "broken":
            # As the gateway breaks a stream off when its upstream does: its error event, then the end.
            await response.write(gateway_event(self.error))
            return response
        await response.write(b"data: [DONE]\n\n")
        return response


class BreakingTTS:
    """A node's TTS whose answers stop halfway: half the audio, then the connection closes, as a node that dies
    mid-reply leaves it. Its requests are counted."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.pcm = _tone(1.0)

    async def speech(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(self.pcm[: len(self.pcm) // 2])
        await asyncio.sleep(0.05)
        assert request.transport is not None
        request.transport.close()
        return response


@pytest.fixture
def sockets() -> Iterator[Path]:
    # a short directory: socket paths are limited to 104 bytes on macOS, 108 on Linux
    with tempfile.TemporaryDirectory(prefix="fs-nodes-") as directory:
        yield Path(directory)


@asynccontextmanager
async def serving(path: Path, routes: list[web.RouteDef]) -> AsyncIterator[None]:
    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1, handler_cancellation=True)
    await runner.setup()
    await web.UnixSite(runner, str(path)).start()
    try:
        yield
    finally:
        await runner.cleanup()


@asynccontextmanager
async def cluster(sockets: Path) -> AsyncIterator[tuple[Gateway, BreakingTTS]]:
    """The gateway on gateway.sock; Phastform's speech servers as the tunnels' phastform-*.sock (mocks in their own
    directory, linked), and Gray's STT as gray-stt.sock beside them and its TTS, which breaks, as gray-tts.sock."""
    gateway, breaking = Gateway(sockets), BreakingTTS()
    routes = [
        web.post("/v1/sessions", gateway.sessions),
        web.delete("/v1/sessions/{id}", gateway.end),
        web.post("/v1/chat/completions", gateway.chat_completions),
    ]
    async with (
        serving(sockets / "gateway.sock", routes),
        serving(sockets / "gray-tts.sock", [web.post("/v1/audio/speech", breaking.speech)]),
    ):
        yield gateway, breaking


@pytest.fixture
def mocks(sockets: Path) -> Iterator[dict[str, OpenAIMocks]]:
    """Each node's mock servers in a directory of its own, their STT and TTS linked as the tunnels name them."""
    found = {}
    with (
        OpenAIMocks(_dir(sockets, "p"), speech_seconds=0.2) as phastform,
        OpenAIMocks(_dir(sockets, "g"), speech_seconds=0.2) as gray,
    ):
        for node, server in (("phastform", phastform), ("gray", gray)):
            (sockets / f"{node}-stt.sock").symlink_to(server.stt_uds)
            found[node] = server
        (sockets / "phastform-tts.sock").symlink_to(phastform.tts_uds)
        yield found


def _dir(sockets: Path, name: str) -> Path:
    (sockets / name).mkdir()
    return sockets / name


def binding(sockets: Path, **options: Any) -> nodes.CallBinding:
    return nodes.CallBinding(
        SESSION,
        str(sockets / "gateway.sock"),
        str(sockets / "phastform-stt.sock"),
        str(sockets / "phastform-tts.sock"),
        **options,
    )


def speech(call: nodes.CallBinding) -> tuple[Any, Any]:
    stt = plugins.make_stt(BASE_URL, transport=call.transport("stt", plugins.LIMITS))
    tts = plugins.make_tts(BASE_URL, transport=call.transport("tts", plugins.LIMITS))
    return stt, tts


async def synthesize(tts: Any, text: str = "Hello there.") -> list[rtc.AudioFrame]:
    async with tts.synthesize(text) as stream:
        return [audio.frame async for audio in stream]


def test_a_calls_speech_goes_to_the_node_the_gateway_places_it_on(
    sockets: Path, mocks: dict[str, OpenAIMocks]
) -> None:
    async def scenario() -> tuple[Gateway, nodes.CallBinding, str]:
        async with cluster(sockets) as (gateway, _):
            gateway.node = "phastform"
            call = binding(sockets)
            assert await call.place("start") == nodes.Placement(
                "phastform",
                str(sockets / "phastform-stt.sock"),
                str(sockets / "phastform-tts.sock"),
            )
            gateway.node = "gray"  # a call placed elsewhere
            other = binding(sockets)
            await other.place("start")
            stt, _ = speech(other)
            event = await stt.recognize([HALF_SECOND])
            await other.aclose()
            await call.aclose()
            return gateway, other, event.alternatives[0].text

    gateway, call, text = asyncio.run(scenario())
    assert text == TRANSCRIPT
    assert (len(mocks["gray"].transcriptions()), len(mocks["phastform"].transcriptions())) == (1, 0)
    assert call.placement is not None and call.placement.node == "gray" and call.nodes is True
    first = gateway.placements[0]
    assert first["body"] == {"session_id": SESSION, "class": "realtime", "reason": "start"}
    assert first["headers"]["x-session-id"] == SESSION
    assert first["headers"]["x-flatshare-class"] == "realtime"


def test_without_nodes_a_call_keeps_its_configured_sockets_and_nothing_changes(
    sockets: Path, mocks: dict[str, OpenAIMocks]
) -> None:
    # Today's single upstream: the gateway has no placement route and answers 404, as for any route it lacks.
    env = {
        "FLATSHARE_LLM_UDS": str(sockets / "gateway.sock"),
        "FLATSHARE_STT_UDS": str(sockets / "phastform-stt.sock"),
        "FLATSHARE_TTS_UDS": str(sockets / "phastform-tts.sock"),
        "FLATSHARE_NODE_BINDING": "on",
    }

    async def scenario() -> tuple[nodes.CallBinding | None, Gateway]:
        async with cluster(sockets) as (gateway, _):
            gateway.status = 404
            settings = Settings.from_env(env)
            callbacks: list[Any] = []
            ctx = SimpleNamespace(add_shutdown_callback=callbacks.append)
            bound = await worker.bind_call(ctx, settings, SESSION)  # type: ignore[arg-type]
            stt, _, tts = worker.call_plugins(settings, SESSION, bound)
            await stt.recognize([HALF_SECOND])
            await synthesize(tts)
            for callback in callbacks:  # the job's end, which ends no placement: there is none
                await callback()
            return bound, gateway

    bound, gateway = asyncio.run(scenario())
    assert bound is None and len(gateway.placements) == 1 and gateway.ended == []
    assert (
        len(mocks["phastform"].transcriptions()) == len(mocks["phastform"].speech_requests()) == 1
    )
    # Node binding off, as bench mode has it: no placement is asked for at all, and none ended.
    off = Settings.from_env(env | {"FLATSHARE_NODE_BINDING": "off"})
    callbacks: list[Any] = []
    ctx = SimpleNamespace(add_shutdown_callback=callbacks.append)
    assert asyncio.run(worker.bind_call(ctx, off, SESSION)) is None  # type: ignore[arg-type]
    assert callbacks == []


def test_a_node_failure_ends_the_reply_and_the_callers_next_speech_rebinds_the_call(
    sockets: Path, mocks: dict[str, OpenAIMocks]
) -> None:
    # Gray dies while the reply plays: its TTS breaks off halfway. The reply ends there: LiveKit's own retry of the
    # sentence is refused before it leaves the worker, so no node synthesizes it again and no audio follows the half
    # already played, and the call takes no request until the caller speaks; then it is placed anew, on Phastform,
    # where its STT and TTS go from that turn on.
    async def scenario() -> dict[str, Any]:
        async with cluster(sockets) as (gateway, breaking):
            failures: list[str] = []
            call = binding(sockets, on_failure=lambda: failures.append("cut"))
            await call.place("start")
            # The requests the fence refuses, which LiveKit's retries of the cut sentence are.
            refused: list[str] = []
            socket = call.socket

            async def counted(server: str) -> str:
                try:
                    return await socket(server)
                except nodes.NodeFailed:
                    refused.append(server)
                    raise

            call.socket = counted  # type: ignore[method-assign]
            stt, tts = speech(call)
            options = APIConnectOptions(max_retry=3, retry_interval=0.05, timeout=5.0)
            frames: list[rtc.AudioFrame] = []
            with pytest.raises(APIError):
                async with tts.synthesize("Hello there.", conn_options=options) as stream:
                    async for audio in stream:
                        frames.append(audio.frame)
            fenced = call.fenced
            with pytest.raises(APIError):
                await stt.recognize([HALF_SECOND])
            gateway.node = "phastform"  # the gateway places the session anew
            call.turn_started()
            event = await stt.recognize([HALF_SECOND])
            after = await synthesize(tts)
            await call.aclose()
            return {
                "failures": failures,
                "fenced": fenced,
                "breaking": len(breaking.requests),
                "frames": frames,
                "pcm": breaking.pcm,
                "text": event.alternatives[0].text,
                "after": after,
                "placements": gateway.placements,
                "history": call.history,
                "refused": refused,
            }

    seen = asyncio.run(scenario())
    assert seen["failures"] == ["cut"] and seen["fenced"]
    assert seen["breaking"] == 1  # the cut sentence was synthesized once, by Gray, and never again
    # LiveKit retried the sentence three times, and the fence refused each before it left the worker; then the STT's
    # four attempts (LiveKit's default, a try and three retries) of the fenced call.
    assert seen["refused"] == ["tts"] * 3 + ["stt"] * 4, seen["refused"]
    played = b"".join(bytes(frame.data) for frame in seen["frames"])
    assert len(played) <= len(seen["pcm"]) // 2 and seen["pcm"].startswith(played.rstrip(b"\0"))
    # The fenced call's STT reached no node; the next turn's went to Phastform, as did its reply's audio.
    assert len(mocks["gray"].transcriptions()) == 0
    assert seen["text"] == TRANSCRIPT and len(mocks["phastform"].transcriptions()) == 1
    assert len(mocks["phastform"].speech_requests()) == 1 and seen["after"]
    assert [p["body"] for p in seen["placements"]] == [
        {"session_id": SESSION, "class": "realtime", "reason": "start"},
        {
            "session_id": SESSION,
            "class": "realtime",
            "reason": "node_failed",
            "failed_node": "gray",
        },
    ]
    assert [step.get("placed") or step.get("failed") for step in seen["history"]] == [
        "gray",
        "gray",
        "phastform",
    ]
    assert seen["history"][1]["origin"] == "tts"
    assert seen["history"][2]["moved"] is True


@pytest.mark.parametrize(
    ("chat", "status", "error", "fenced"),
    [
        ("ok", 200, "", False),
        (
            "broken",
            200,
            "upstream_unavailable",
            True,
        ),  # the stream broken off by the node's upstream
        ("broken", 200, "upstream_timeout", True),
        ("ok", 502, "upstream_unavailable", True),
        ("ok", 503, "no_healthy_node", True),  # no node of the session's role available
        ("ok", 503, "in_flight_limit", False),  # the gateway's capacity, not the node
        ("ok", 400, "invalid_json", False),
    ],
)
def test_the_calls_llm_goes_to_the_gateway_with_its_session_and_a_node_failure_fences_it(
    sockets: Path,
    mocks: dict[str, OpenAIMocks],
    chat: str,
    status: int,
    error: str,
    fenced: bool,
) -> None:
    async def scenario() -> tuple[Gateway, nodes.CallBinding, bool]:
        async with cluster(sockets) as (gateway, _):
            gateway.chat, gateway.chat_status, gateway.error = chat, status, error
            call = binding(sockets)
            await call.place("start")
            model = plugins.make_llm(
                BASE_URL, SESSION, transport=call.transport("llm", plugins.LIMITS)
            )
            ctx = llm.ChatContext.empty()
            ctx.add_message(role="user", content=TRANSCRIPT)
            failed = False
            try:
                async with model.chat(
                    chat_ctx=ctx, conn_options=APIConnectOptions(max_retry=0)
                ) as stream:
                    async for _ in stream:
                        pass
            except APIError:
                failed = True
            # A fenced call's next LLM request never reaches the gateway.
            before = len(gateway.chats)
            if call.fenced:
                with pytest.raises(APIError):
                    async with model.chat(
                        chat_ctx=ctx, conn_options=APIConnectOptions(max_retry=0)
                    ) as stream:
                        async for _ in stream:
                            pass
                assert len(gateway.chats) == before
            await call.aclose()
            return gateway, call, failed

    gateway, call, failed = asyncio.run(scenario())
    assert call.fenced is fenced
    assert failed is (chat == "broken" or status != 200)
    headers = gateway.chats[0]["headers"]
    assert headers["x-session-id"] == SESSION and headers["x-flatshare-class"] == "realtime"


def test_the_conversation_survives_the_rebinding(
    sockets: Path, mocks: dict[str, OpenAIMocks]
) -> None:
    # The transcript is the session's chat context, kept in the worker: after a node fails and the call is placed
    # anew, the same LLM, with the same context, sends the whole conversation to the gateway, which routes it to the
    # new node.
    async def scenario() -> Gateway:
        async with cluster(sockets) as (gateway, _):
            call = binding(sockets)
            await call.place("start")
            model = plugins.make_llm(
                BASE_URL, SESSION, transport=call.transport("llm", plugins.LIMITS)
            )
            ctx = llm.ChatContext.empty()
            for n in range(3):
                ctx.add_message(role="user", content=f"turn {n}")
                ctx.add_message(role="assistant", content=f"reply {n}")
            call.node_failed("tts", "gray's TTS broke off")
            gateway.node = "phastform"
            call.turn_started()
            await call.place(
                "node_failed"
            )  # the turn's placement, which the next request waits for
            ctx.add_message(role="user", content=TRANSCRIPT)
            async with model.chat(chat_ctx=ctx) as stream:
                async for _ in stream:
                    pass
            await call.aclose()
            return gateway

    gateway = asyncio.run(scenario())
    [request] = gateway.chats
    contents = [message["content"] for message in request["body"]["messages"]]
    assert contents == [
        "turn 0", "reply 0", "turn 1", "reply 1", "turn 2", "reply 2", TRANSCRIPT
    ]  # fmt: skip


@pytest.mark.parametrize(
    "answer",
    [
        {"session_id": SESSION, "node": "gray/..", "stt": "gray-stt.sock", "tts": "gray-tts.sock"},
        {"session_id": SESSION, "node": "gray", "stt": "../gray-stt.sock", "tts": "gray-tts.sock"},
        {"session_id": SESSION, "node": "gray", "stt": "gray-stt", "tts": "gray-tts.sock"},
        # a unix: address or a path outside the configured sockets' directory
        {
            "session_id": SESSION,
            "node": "gray",
            "stt": "unix:/elsewhere/gray-stt.sock",
            "tts": "gray-tts.sock",
        },
        {
            "session_id": SESSION,
            "node": "gray",
            "stt": "/elsewhere/gray-stt.sock",
            "tts": "gray-tts.sock",
        },
        {"session_id": "voice-RM_other", "node": "gray", "stt": "a.sock", "tts": "b.sock"},
        ["gray"],
    ],
)
def test_a_placement_names_a_node_and_its_sockets_beside_the_calls(
    sockets: Path, answer: Any
) -> None:
    # A placement names its speech sockets by their file names, which the worker finds beside its own: anything
    # else, or another session's answer, places the call nowhere, and its first request asks again.
    async def scenario() -> nodes.CallBinding:
        async def sessions(request: web.Request) -> web.Response:
            return web.json_response(answer)

        async with serving(sockets / "gateway.sock", [web.post("/v1/sessions", sessions)]):
            call = binding(sockets)
            assert await call.place("start") is None
            with pytest.raises(nodes.NodeFailed, match="placed the call on no node"):
                await call.socket("stt")
            await call.aclose()
            return call

    call = asyncio.run(scenario())
    assert call.placement is None and call.nodes is None and not call.fenced


def test_a_placement_that_does_not_answer_leaves_the_call_to_ask_at_its_first_request(
    sockets: Path, mocks: dict[str, OpenAIMocks]
) -> None:
    async def scenario() -> tuple[int, str]:
        async with cluster(sockets) as (gateway, _):
            gateway.hang = True
            call = binding(sockets, timeout=0.2)
            assert await call.place("start") is None
            gateway.hang = False
            gateway.released.set()
            path = await call.socket("stt")  # asks again, and is placed
            await call.aclose()
            return len(gateway.placements), path

    asked, path = asyncio.run(scenario())
    assert asked == 2 and path == str(sockets / "gray-stt.sock")


def test_watch_cuts_the_reply_on_a_node_failure_and_rebinds_at_the_callers_next_speech(
    sockets: Path, mocks: dict[str, OpenAIMocks]
) -> None:
    class Session:
        """The AgentSession's surface the binding uses: its events and its interrupt."""

        def __init__(self) -> None:
            self.handlers: dict[str, list[Any]] = {}
            self.interrupted: list[bool] = []

        def on(self, event: str, handler: Any) -> None:
            self.handlers.setdefault(event, []).append(handler)

        def emit(self, event: str, payload: Any) -> None:
            for handler in self.handlers.get(event, []):
                handler(payload)

        def interrupt(self, *, force: bool = False) -> None:
            self.interrupted.append(force)

    class State:
        def __init__(self, state: str) -> None:
            self.new_state = state

    async def scenario() -> tuple[Session, nodes.CallBinding, Gateway]:
        async with cluster(sockets) as (gateway, _):
            session, call = Session(), binding(sockets)
            await call.place("start")
            nodes.watch(session, call)
            session.emit("user_state_changed", State("speaking"))  # placed and well: nothing asked
            call.node_failed("stt", "gray-stt.sock refused the connection")
            call.node_failed("tts", "and so did its TTS")  # one failure, one cut
            await asyncio.sleep(0.05)
            gateway.node = "phastform"
            session.emit("user_state_changed", State("listening"))
            session.emit("user_state_changed", State("speaking"))
            assert await call.socket("tts") == str(sockets / "phastform-tts.sock")
            session.emit("close", None)
            await asyncio.sleep(0.05)
            return session, call, gateway

    session, call, gateway = asyncio.run(scenario())
    assert session.interrupted == [True]
    assert [p["body"]["reason"] for p in gateway.placements] == ["start", "node_failed"]
    assert call.placement is not None and call.placement.node == "phastform"


@pytest.mark.parametrize(
    ("mode", "setting", "bound"),
    [("bench", None, False), ("bench", "on", True), ("serve", None, True), ("serve", "off", False)],
)
def test_serve_mode_binds_each_call_to_its_node_and_bench_mode_when_told(
    mode: str, setting: str | None, bound: bool
) -> None:
    env = {
        "FLATSHARE_LLM_UDS": "/run/user/1007/flatshare/gateway.sock",
        "FLATSHARE_STT_UDS": "/run/user/1007/flatshare/phastform-stt.sock",
        "FLATSHARE_TTS_UDS": "/run/user/1007/flatshare/phastform-tts.sock",
        "FLATSHARE_VOICE_MODE": mode,
        "FLATSHARE_SERVING_PROFILE": str(
            Path(__file__).resolve().parents[3] / "deploy" / "phastform" / "serving.json"
        ),
    }
    if setting is not None:
        env["FLATSHARE_NODE_BINDING"] = setting
    assert Settings.from_env(env).node_binding is bound


def test_node_binding_needs_the_gateway() -> None:
    env = {
        "FLATSHARE_LLM_UDS": "/run/user/1007/flatshare/phastform-engine.sock",  # the bypass
        "FLATSHARE_STT_UDS": "/run/user/1007/flatshare/phastform-stt.sock",
        "FLATSHARE_TTS_UDS": "/run/user/1007/flatshare/phastform-tts.sock",
    }
    assert Settings.from_env(env).node_binding is False
    with pytest.raises(ValueError, match="FLATSHARE_NODE_BINDING=on asks the gateway"):
        Settings.from_env(env | {"FLATSHARE_NODE_BINDING": "on"})
    with pytest.raises(ValueError, match="FLATSHARE_NODE_BINDING must be off or on"):
        Settings.from_env(env | {"FLATSHARE_NODE_BINDING": "yes"})


def test_the_node_binding_variant_has_bench_modes_calls_ask_for_their_node() -> None:
    # deploy/cyan/variants/node-binding, for points and drills on two nodes: bench mode, which asks nothing by default,
    # binds each call as serve mode does.
    conf = (
        Path(__file__).resolve().parents[3]
        / "deploy"
        / "cyan"
        / "variants"
        / "node-binding"
        / "flatshare-voice-worker.conf"
    )
    set_by = dict(
        line.removeprefix("Environment=").split("=", 1)
        for line in conf.read_text().splitlines()
        if line.startswith("Environment=")
    )
    env = {
        "FLATSHARE_LLM_UDS": "/run/user/1007/flatshare/gateway.sock",
        "FLATSHARE_STT_UDS": "/run/user/1007/flatshare/phastform-stt.sock",
        "FLATSHARE_TTS_UDS": "/run/user/1007/flatshare/phastform-tts.sock",
    }
    assert set_by == {"FLATSHARE_NODE_BINDING": "on"}
    assert Settings.from_env(env).node_binding is False
    assert Settings.from_env(env | set_by).node_binding is True
