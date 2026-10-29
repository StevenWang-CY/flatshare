"""The plugin factory against the mock servers, over Unix sockets (no network).

The base URL's host is in the reserved ``.invalid`` domain, which never resolves, so
every request that succeeds went through the socket the factory was given.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import httpx2 as httpx
import numpy as np
import pytest
from livekit import rtc
from livekit.agents import llm
from livekit.plugins import openai as lk_openai

from flatshare_voice import plugins

from .mocks import FIRST_SENTENCE, TRANSCRIPT, OpenAIMocks

BASE_URL = "http://flatshare.invalid/v1"


def test_the_worker_lets_idle_connections_go_before_the_servers_do() -> None:
    assert plugins._LIMITS.keepalive_expiry < plugins.SERVER_KEEP_ALIVE_S


@pytest.fixture(scope="module")
def mocks() -> Iterator[OpenAIMocks]:
    # a short directory: socket paths are limited to 104 bytes on macOS, 108 on Linux
    with (
        tempfile.TemporaryDirectory(prefix="flatshare-voice-") as sockets,
        OpenAIMocks(
            Path(sockets), token_interval=0.01, stream_seconds=0.1, speech_seconds=0.5
        ) as m,
    ):
        yield m


def test_llm_streams_with_session_headers_over_uds(mocks: OpenAIMocks) -> None:
    model = plugins.make_llm(BASE_URL, "voice-RM_test", uds=mocks.llm_uds)
    assert isinstance(model, lk_openai.LLM)

    async def chat() -> str:
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="user", content=TRANSCRIPT)
        text = []
        async with model.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text.append(chunk.delta.content)
        await model.aclose()
        return "".join(text)

    assert asyncio.run(chat()).startswith(FIRST_SENTENCE)
    [stream] = mocks.chat_streams()
    assert stream.headers["x-session-id"] == "voice-RM_test"
    assert stream.headers["x-flatshare-class"] == "realtime"
    assert stream.body["model"] == plugins.LLM_MODEL
    assert stream.body["stream"] is True
    assert stream.completed_at is not None
    assert model._client.is_closed(), "aclose() must close the client the factory created"


def test_stt_transcribes_over_uds(mocks: OpenAIMocks) -> None:
    stt = plugins.make_stt(BASE_URL, uds=mocks.stt_uds)
    assert isinstance(stt, lk_openai.STT)
    half_second = rtc.AudioFrame(np.zeros(8000, np.int16).tobytes(), 16000, 1, 8000)

    event = asyncio.run(stt.recognize([half_second]))

    assert event.alternatives[0].text == TRANSCRIPT
    [request] = mocks.transcriptions()
    assert request.fields["model"] == plugins.STT_MODEL
    assert request.fields["language"] == "en"
    assert request.audio_bytes == 44 + 16000  # WAV header + 0.5 s of 16-bit mono


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_the_stt_fake_reads_an_upload_whole_however_its_bytes_arrive() -> None:
    # The fake parses the multipart body itself (mocks._form). aiohttp's own reader calls its deprecated
    # unread_data() when a short read is followed by a long one, as here: the file's bytes arrive 8, then 4, then the
    # rest. A field outside ASCII, and file bytes that look like line ends and a boundary's dashes, arrive as sent.
    audio = bytes(range(256)) * 64 + b"\r\n--\r\n" * 8
    fields = {"model": plugins.STT_MODEL, "language": "en", "prompt": "Grüße aus Köln"}
    boundary = "flatshare-test-boundary"
    head = "".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        for name, value in fields.items()
    )
    head += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="turn.wav"\r\n'
    head += "Content-Type: audio/wav\r\n\r\n"
    tail = f"\r\n--{boundary}--\r\n".encode()
    pieces = [head.encode() + audio[:8], audio[8:12], audio[12:] + tail]

    def upload() -> Iterator[bytes]:
        for i, piece in enumerate(pieces):
            if i:
                time.sleep(0.05)
            yield piece

    with (
        tempfile.TemporaryDirectory(prefix="flatshare-voice-") as sockets,
        OpenAIMocks(Path(sockets)) as mocks,
        httpx.Client(transport=httpx.HTTPTransport(uds=mocks.stt_uds)) as client,
    ):
        response = client.post(
            f"{BASE_URL}/audio/transcriptions",
            content=upload(),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(sum(map(len, pieces))),
            },
        )
        seen = mocks.transcriptions()

    assert response.status_code == 200, response.text
    assert response.json() == {"text": TRANSCRIPT}
    [request] = seen
    assert request.fields == fields
    assert request.audio_bytes == len(audio)


def test_tts_returns_raw_pcm_over_uds(mocks: OpenAIMocks) -> None:
    tts = plugins.make_tts(BASE_URL, uds=mocks.tts_uds)
    assert isinstance(tts, lk_openai.TTS)

    async def synthesize() -> list[rtc.AudioFrame]:
        async with tts.synthesize("Hello there.") as stream:
            frames = [audio.frame async for audio in stream]
        await tts.aclose()
        return frames

    frames = asyncio.run(synthesize())

    assert {(frame.sample_rate, frame.num_channels) for frame in frames} == {(24000, 1)}
    pcm = b"".join(bytes(frame.data) for frame in frames)
    # the PCM passes through untouched; the emitter ends with at most 10 ms of silence
    assert pcm[: len(mocks.speech_pcm)] == mocks.speech_pcm
    assert pcm[len(mocks.speech_pcm) :] == bytes(len(pcm) - len(mocks.speech_pcm))
    assert len(pcm) - len(mocks.speech_pcm) <= 24000 * 2 // 100
    [request] = mocks.speech_requests()
    assert request.body["response_format"] == "pcm"
    assert request.body["model"] == plugins.TTS_MODEL
    assert request.body["voice"] == plugins.TTS_VOICE
    assert request.body["input"] == "Hello there."
    assert tts._client.is_closed(), "aclose() must close the client the factory created"
