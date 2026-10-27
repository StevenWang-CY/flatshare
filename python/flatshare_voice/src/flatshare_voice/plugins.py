"""OpenAI-compatible STT, LLM and TTS plugins for the voice worker.

Each factory returns a stock LiveKit Agents 1.8.8 OpenAI plugin pointed at an
OpenAI-compatible server by ``base_url``. With ``uds``, the HTTP connection goes over
that Unix socket instead of TCP: inside the voice-plane network namespace, the engine
(through the control plane), the STT and the TTS are reachable only as socket files.
"""

from __future__ import annotations

import httpx2 as httpx
import openai
from livekit.plugins import openai as lk_openai

from .protocol import realtime_headers

LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
STT_MODEL = "openai/whisper-large-v3-turbo"
TTS_MODEL = "kokoro"
TTS_VOICE = "af_heart"

API_KEY = "local"
"""Placeholder key: the OpenAI client requires one and the local servers ignore it."""

# The plugins' own client defaults. The plugins set each request's timeout themselves
# (the LLM uses the session's ``llm_conn_options.timeout``, 10 s by default), so these
# only apply to calls without one, such as the LLM prewarm. Idle connections go after 55 s,
# 5 s before the GPU node's servers would close them (60 s), so that a turn never sends on a
# connection the server is closing; through the tunnel a new one is a new SSH session.
_TIMEOUT = httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0)
_LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=50, keepalive_expiry=55)
LIMITS = _LIMITS
"""The connection limits of every client's transport, a node binding's too."""
SERVER_KEEP_ALIVE_S = 60
"""How long the engine, STT and TTS keep an idle connection (deploy/phastform/systemd)."""


def openai_client(
    base_url: str,
    *,
    uds: str | None = None,
    api_key: str = API_KEY,
    transport: httpx.AsyncBaseTransport | None = None,
) -> openai.AsyncClient:
    """Return an OpenAI client for ``base_url``, connected through ``uds`` when given, or through
    ``transport`` (a call's node binding, :mod:`.nodes`, which picks the socket per request).

    The transport is explicit, so proxy environment variables never apply. Retries are
    left to the plugins (``APIConnectOptions.max_retry``).
    """
    transport = transport or httpx.AsyncHTTPTransport(uds=uds, limits=_LIMITS)
    return openai.AsyncClient(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport, timeout=_TIMEOUT, follow_redirects=True),
    )


def make_stt(
    base_url: str,
    *,
    uds: str | None = None,
    model: str = STT_MODEL,
    language: str = "en",
    api_key: str = API_KEY,
    transport: httpx.AsyncBaseTransport | None = None,
) -> lk_openai.STT:
    """Speech-to-text over ``/audio/transcriptions`` on VAD-cut segments (not streaming)."""
    client = openai_client(base_url, uds=uds, api_key=api_key, transport=transport)
    return lk_openai.STT(model=model, language=language, client=client)


def make_llm(
    base_url: str,
    session_id: str,
    *,
    uds: str | None = None,
    model: str = LLM_MODEL,
    api_key: str = API_KEY,
    transport: httpx.AsyncBaseTransport | None = None,
) -> lk_openai.LLM:
    """Streaming chat completions tagged with the session id and the realtime class.

    The plugin's ``extra_headers`` option puts the headers on every chat request of
    this instance, so the worker creates one instance per session.
    """
    client = openai_client(base_url, uds=uds, api_key=api_key, transport=transport)
    llm = lk_openai.LLM(model=model, client=client, extra_headers=realtime_headers(session_id))
    return _owning_client(llm)


def make_tts(
    base_url: str,
    *,
    uds: str | None = None,
    model: str = TTS_MODEL,
    voice: str = TTS_VOICE,
    api_key: str = API_KEY,
    transport: httpx.AsyncBaseTransport | None = None,
) -> lk_openai.TTS:
    """Text-to-speech over ``/audio/speech`` as raw 24 kHz PCM, with no MP3 round trip."""
    client = openai_client(base_url, uds=uds, api_key=api_key, transport=transport)
    tts = lk_openai.TTS(model=model, voice=voice, response_format="pcm", client=client)
    return _owning_client(tts)


def _owning_client[P: (lk_openai.LLM, lk_openai.TTS)](plugin: P) -> P:
    """Make ``aclose()`` close the client we passed in, as the plugin's own
    ``with_azure`` constructors do; by default the plugins close only clients they
    created. (The STT never closes its OpenAI client, ours or its own.)"""
    plugin._owns_client = True
    return plugin
