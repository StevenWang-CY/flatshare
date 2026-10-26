"""The voice agent's turn handling and lifecycle events, as its environment sets them."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from livekit.agents import inference

from flatshare_voice import plugins
from flatshare_voice.worker import events_socket, llm_model, turn_handling

GATEWAY = "/run/user/1007/flatshare/gateway.sock"


@pytest.fixture(autouse=True)
def _defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FLATSHARE_TURN_DETECTION",
        "FLATSHARE_PREEMPTIVE_GENERATION",
        "FLATSHARE_EVENTS",
        "FLATSHARE_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FLATSHARE_LLM_UDS", GATEWAY)


def test_the_worker_keeps_livekits_turn_handling_by_default() -> None:
    handling = turn_handling()
    assert isinstance(handling["turn_detection"], inference.TurnDetector)
    assert handling["interruption"] == {"mode": "vad"}
    assert handling["preemptive_generation"] == {"enabled": True}


def test_experiments_fix_the_endpointing_and_can_turn_preemptive_generation_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLATSHARE_TURN_DETECTION", "vad")
    monkeypatch.setenv("FLATSHARE_PREEMPTIVE_GENERATION", "false")
    handling = turn_handling()
    assert handling["turn_detection"] == "vad"
    assert handling["preemptive_generation"] == {"enabled": False}


def test_events_go_to_the_control_planes_socket_unless_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert events_socket() == GATEWAY
    monkeypatch.setenv("FLATSHARE_EVENTS", "off")
    assert events_socket() is None
    # The bypass: the LLM goes straight to the engine, which takes no events.
    monkeypatch.setenv("FLATSHARE_EVENTS", "on")
    monkeypatch.setenv("FLATSHARE_LLM_UDS", "/run/user/1007/flatshare/phastform-engine.sock")
    assert events_socket() is None


def test_the_agent_names_the_boxs_model_unless_told_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert llm_model() == plugins.LLM_MODEL == "Qwen/Qwen3-4B-Instruct-2507-FP8"
    monkeypatch.setenv("FLATSHARE_LLM_MODEL", "Qwen/Qwen3-14B-FP8")
    assert llm_model() == "Qwen/Qwen3-14B-FP8"


@pytest.mark.parametrize(
    ("name", "value", "setting"),
    [
        ("FLATSHARE_TURN_DETECTION", "eou", turn_handling),
        ("FLATSHARE_PREEMPTIVE_GENERATION", "off", turn_handling),
        ("FLATSHARE_EVENTS", "false", events_socket),
    ],
)
def test_a_misspelled_setting_stops_the_worker(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, setting: Callable[[], object]
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        setting()
