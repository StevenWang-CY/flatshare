"""A call's turn announced to the engine before its request leaves (``FLATSHARE_TURN_HINTS``).

LiveKit Agents sends a turn's LLM request at Whisper's final transcript; the user stopped speaking a speech
recognition round trip before that, when Silero's VAD ended the speech and the session's user state went from
``speaking`` to ``listening``. Then the worker sends the engine a hint: the conversation so far, as the LLM plugin will
send it (the agent's chat context in the OpenAI format), for one token, with the call's realtime headers, so that the
control plane gives it the realtime priority. The engine's make-room scheduler takes a realtime request for one token
as a turn announced: its budget cap holds lower-priority prefill back for a while (``make_room.hint_hold_ms``), so that
the turn arrives into decode-only steps rather than behind an agent's long-context prefill chunk, and the hint brings
the conversation's prefix to the GPU if it was not there. Nothing waits for the hint's answer, and a hint that fails
only loses itself.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from livekit.agents import AgentSession, UserStateChangedEvent

from .protocol import realtime_headers

logger = logging.getLogger("flatshare.voice.hints")

TIMEOUT_S = 2.0
"""How long a hint may take: past it the turn's own request has left anyway."""


def attach(session: AgentSession, client: Any, *, model: str, session_id: str) -> set[asyncio.Task]:
    """Announce each of ``session``'s turns through ``client`` (an ``openai.AsyncClient`` on the call's LLM path) when
    its user stops speaking. Returns the hints in flight, which end on their own."""
    pending: set[asyncio.Task] = set()

    def state_changed(ev: UserStateChangedEvent) -> None:
        if ev.old_state != "speaking" or ev.new_state != "listening":
            return
        try:
            messages, _ = session.current_agent.chat_ctx.to_provider_format(format="openai")
        except Exception:  # an agent between activities has no chat context to announce
            logger.debug("no turn hint: the session has no current agent", exc_info=True)
            return
        task = asyncio.create_task(
            announce(client, model=model, session_id=session_id, messages=messages)
        )
        pending.add(task)
        task.add_done_callback(pending.discard)

    session.on("user_state_changed", state_changed)
    return pending


async def announce(client: Any, *, model: str, session_id: str, messages: list[Any]) -> None:
    """One hint: ``messages`` for one token at the realtime class, its answer read and dropped."""
    try:
        await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            extra_headers=realtime_headers(session_id),
            timeout=TIMEOUT_S,
        )
    except Exception:  # a hint is best effort: the turn's own request carries the call
        logger.debug("turn hint failed", exc_info=True)
