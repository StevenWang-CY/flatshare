"""The voice agent's settings, read from its environment, and the turn handling and lifecycle events they set."""

from __future__ import annotations

import os
import sys

import pytest
from livekit.agents import inference

from flatshare_voice import plugins, worker
from flatshare_voice.worker import Settings

GATEWAY = "/run/user/1007/flatshare/gateway.sock"
SOCKETS = {
    "FLATSHARE_LLM_UDS": GATEWAY,
    "FLATSHARE_STT_UDS": "/run/user/1007/flatshare/phastform-stt.sock",
    "FLATSHARE_TTS_UDS": "/run/user/1007/flatshare/phastform-tts.sock",
}
"""What flatshare-voice-worker.service sets on cyan."""


def settings(**env: str) -> Settings:
    return Settings.from_env(SOCKETS | env)


def test_the_worker_keeps_livekits_turn_handling_by_default() -> None:
    handling = settings().turn_handling()
    assert isinstance(handling["turn_detection"], inference.TurnDetector)
    assert handling["interruption"] == {"mode": "vad"}
    assert handling["preemptive_generation"] == {"enabled": True}
    assert (settings().idle_processes, settings().llm_timeout) == (2, 30.0)
    assert settings(FLATSHARE_IDLE_PROCESSES="4", FLATSHARE_LLM_TIMEOUT="45").idle_processes == 4


def test_experiments_fix_the_endpointing_and_can_turn_preemptive_generation_off() -> None:
    handling = settings(
        FLATSHARE_TURN_DETECTION="vad", FLATSHARE_PREEMPTIVE_GENERATION="false"
    ).turn_handling()
    assert handling["turn_detection"] == "vad"
    assert handling["preemptive_generation"] == {"enabled": False}


def test_events_go_to_the_control_planes_socket_unless_switched_off() -> None:
    assert settings().events_socket == GATEWAY
    assert settings(FLATSHARE_EVENTS="off").events_socket is None
    # The bypass: the LLM goes straight to the engine, which takes no events.
    bypass = "/run/user/1007/flatshare/phastform-engine.sock"
    assert settings(FLATSHARE_EVENTS="on", FLATSHARE_LLM_UDS=bypass).events_socket is None


def test_the_agent_names_the_boxs_model_unless_told_another() -> None:
    assert settings().llm_model == plugins.LLM_MODEL == "Qwen/Qwen3-4B-Instruct-2507-FP8"
    assert settings(FLATSHARE_LLM_MODEL="Qwen/Qwen3-14B-FP8").llm_model == "Qwen/Qwen3-14B-FP8"


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """What LiveKit's command line was given to run, with the options the worker set on its server
    first, in place of running it; the worker is told to serve, with the unit's sockets."""
    runs: list[object] = []
    monkeypatch.setattr(worker.cli, "run_app", runs.append)
    monkeypatch.setattr(worker.server, "update_options", lambda **options: runs.append(options))
    monkeypatch.setattr(sys, "argv", ["flatshare-voice-worker", "start"])
    for name in [name for name in os.environ if name.startswith("FLATSHARE_")]:
        monkeypatch.delenv(name)
    for name, value in SOCKETS.items():
        monkeypatch.setenv(name, value)
    return runs


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FLATSHARE_TURN_DETECTION", "eou"),
        ("FLATSHARE_PREEMPTIVE_GENERATION", "off"),
        ("FLATSHARE_EVENTS", "false"),
        ("FLATSHARE_LLM_TIMEOUT", "30s"),
        ("FLATSHARE_LLM_TIMEOUT", "0"),
        ("FLATSHARE_IDLE_PROCESSES", "four"),
        ("FLATSHARE_IDLE_PROCESSES", "-1"),
        ("FLATSHARE_STT_UDS", ""),
    ],
)
def test_a_misspelled_setting_stops_the_worker_before_it_registers(
    started: list[object], monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    # Each was read only as a session started, after the worker had registered with LiveKit: every
    # call then failed on it, where the worker should not have served at all.
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit, match=name):
        worker.main()
    assert started == []


def test_the_worker_serves_with_its_settings_and_other_commands_need_none(
    started: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLATSHARE_IDLE_PROCESSES", "4")
    worker.main()
    assert started == [{"num_idle_processes": 4}, worker.server]
    monkeypatch.setattr(sys, "argv", ["flatshare-voice-worker", "download-files"])
    for name in SOCKETS:
        monkeypatch.delenv(name)
    worker.main()
    assert started[2:] == [worker.server]
