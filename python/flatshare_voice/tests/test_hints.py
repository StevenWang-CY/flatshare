"""A call's turn announced to the engine when its user stops speaking (src/flatshare_voice/hints.py), on a stand-in
session and client: when it is sent, what it carries, and that a failing one is only lost."""

import asyncio
from types import SimpleNamespace

from livekit import rtc
from livekit.agents import UserStateChangedEvent, llm

from flatshare_voice import hints
from flatshare_voice.protocol import realtime_headers

SESSION_ID = "room-hints"


class Completions:
    """The client's chat completions: each request's arguments recorded, failing when told to."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("engine down")


class Session(rtc.EventEmitter[str]):
    def __init__(self) -> None:
        super().__init__()
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="You are Ava.")
        chat_ctx.add_message(role="user", content="a table for four")
        chat_ctx.add_message(role="assistant", content="For what time?")
        self.current_agent = SimpleNamespace(chat_ctx=chat_ctx)

    def user(self, old: str, new: str) -> None:
        self.emit("user_state_changed", UserStateChangedEvent(old_state=old, new_state=new))


def announced(
    transitions: list[tuple[str, str]], fail: bool = False
) -> tuple[Session, Completions]:
    async def run() -> tuple[Session, Completions]:
        session, completions = Session(), Completions(fail)
        pending = hints.attach(
            session,
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            model="m",
            session_id=SESSION_ID,
        )
        for old, new in transitions:
            session.user(old, new)
        await asyncio.gather(*pending)
        return session, completions

    return asyncio.run(run())


def test_the_end_of_speech_announces_the_conversation_so_far_for_one_token():
    session, completions = announced([("listening", "speaking"), ("speaking", "listening")])
    [call] = completions.calls
    messages, _ = session.current_agent.chat_ctx.to_provider_format(format="openai")
    assert call == {
        "model": "m",
        "messages": messages,
        "max_tokens": 1,
        "extra_headers": realtime_headers(SESSION_ID),
        "timeout": hints.TIMEOUT_S,
    }
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


def test_only_the_end_of_speech_announces_a_turn():
    # The user starting to speak, or the agent's own state changes the session reports as the user's, send nothing.
    _, completions = announced(
        [("listening", "speaking"), ("away", "listening"), ("listening", "away")]
    )
    assert completions.calls == []


def test_a_hint_that_fails_is_only_lost():
    _, completions = announced([("speaking", "listening")], fail=True)
    assert len(completions.calls) == 1  # and nothing was raised
