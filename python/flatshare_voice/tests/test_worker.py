"""The voice agent's settings, read from its environment, and the turn handling and lifecycle events they set; its
two modes, bench (every call taken) and serve (call-capacity admission from the serving profile), and its two audio
configurations, the synthetic callers' and real microphones'."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from livekit.agents import inference

from flatshare_voice import admission, plugins, worker
from flatshare_voice.worker import Settings

REPO = Path(__file__).resolve().parents[3]
SERVING = REPO / "deploy" / "phastform" / "serving.json"
"""The box's serving profile, which serve mode reads (deploy/profiles/rtx-5060-ti-serve.toml)."""

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


def test_the_endpointing_sweep_sets_the_delay_and_the_vads_silence_over_the_configurations() -> (
    None
):
    # Unset, the configuration's own: LiveKit's endpointing and Silero's defaults for the synthetic callers.
    plain = settings(FLATSHARE_TURN_DETECTION="vad")
    assert "endpointing" not in plain.turn_handling() and plain.vad_options() == {}
    assert (plain.describe()["endpointing_min_delay"], plain.describe()["vad_min_silence"]) == (
        None,
        None,
    )
    swept = settings(
        FLATSHARE_TURN_DETECTION="vad",
        FLATSHARE_ENDPOINTING_MIN_DELAY="0.2",
        FLATSHARE_VAD_MIN_SILENCE="0.3",
    )
    # Only the delay is given: LiveKit keeps the fixed mode and its 3 s at most.
    assert swept.turn_handling()["endpointing"] == {"min_delay": 0.2}
    assert swept.vad_options() == {"min_silence_duration": 0.3}
    assert (swept.describe()["endpointing_min_delay"], swept.describe()["vad_min_silence"]) == (
        0.2,
        0.3,
    )
    # Over the microphone configuration, its other endpointing keys and VAD options stay.
    microphone = settings(
        FLATSHARE_AUDIO="microphone",
        FLATSHARE_ENDPOINTING_MIN_DELAY="0",
        FLATSHARE_VAD_MIN_SILENCE="0.4",
    )
    assert microphone.turn_handling()["endpointing"] == {
        "mode": "fixed",
        "min_delay": 0.0,
        "max_delay": 3.0,
    }
    assert microphone.vad_options() == {
        "activation_threshold": 0.6,
        "min_speech_duration": 0.1,
        "min_silence_duration": 0.4,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FLATSHARE_ENDPOINTING_MIN_DELAY", "-0.1"),
        ("FLATSHARE_ENDPOINTING_MIN_DELAY", "3.5"),
        ("FLATSHARE_ENDPOINTING_MIN_DELAY", "soon"),
        ("FLATSHARE_VAD_MIN_SILENCE", "0"),
        ("FLATSHARE_VAD_MIN_SILENCE", "4"),
    ],
)
def test_an_endpointing_setting_out_of_its_range_is_refused_by_name(name: str, value: str) -> None:
    with pytest.raises(ValueError, match=name):
        settings(**{name: value})


def test_events_go_to_the_control_planes_socket_unless_switched_off() -> None:
    assert settings().events_socket == GATEWAY
    assert settings(FLATSHARE_EVENTS="off").events_socket is None
    # The bypass: the LLM goes straight to the engine, which takes no events.
    bypass = "/run/user/1007/flatshare/phastform-engine.sock"
    assert settings(FLATSHARE_EVENTS="on", FLATSHARE_LLM_UDS=bypass).events_socket is None
    # Every event in bench mode, for E2's lifecycle policy; session_closed alone in serve mode, whose gateway acts on
    # no other (test_events.py).
    assert settings().turn_events is True
    serve = settings(FLATSHARE_VOICE_MODE="serve", FLATSHARE_SERVING_PROFILE=str(SERVING))
    assert (serve.events_socket, serve.turn_events) == (GATEWAY, False)


def test_the_agent_names_the_boxs_model_unless_told_another() -> None:
    assert settings().llm_model == plugins.LLM_MODEL == "Qwen/Qwen3-4B-Instruct-2507-FP8"
    assert settings(FLATSHARE_LLM_MODEL="Qwen/Qwen3-14B-FP8").llm_model == "Qwen/Qwen3-14B-FP8"


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """What LiveKit's command line was given to run, with the options the worker set on its server
    first, in place of running it; the worker is told to serve, with the unit's sockets. Serve
    mode's admission is the test's alone, and its monitor runs one round, without its thread."""
    runs: list[object] = []
    monkeypatch.setattr(worker.cli, "run_app", runs.append)
    monkeypatch.setattr(worker.server, "update_options", lambda **options: runs.append(options))
    monkeypatch.setattr(sys, "argv", ["flatshare-voice-worker", "start"])
    monkeypatch.setattr(worker, "ADMISSION", None)
    monkeypatch.setattr(admission.Monitor, "start", admission.Monitor.tick)
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
        ("FLATSHARE_VOICE_MODE", "production"),
        ("FLATSHARE_AUDIO", "phone"),
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


def test_bench_mode_takes_every_call_with_no_load_as_every_measurement_ran() -> None:
    # The default, and what the unit sets on cyan: LiveKit's load is 0 whatever runs, and every job request is accepted,
    # as LiveKit's default request handler does.
    bench = settings()
    assert (bench.mode, bench.max_calls, bench.audio) == ("bench", None, "synthetic")
    assert worker.ADMISSION is None and worker.load(worker.server) == 0.0

    class Job:
        answer = None

        async def accept(self) -> None:
            self.answer = "accepted"

    job = Job()
    asyncio.run(worker.request(job))
    assert job.answer == "accepted"


def test_serve_mode_admits_the_calls_its_serving_profile_names(tmp_path: Path) -> None:
    serve = settings(FLATSHARE_VOICE_MODE="serve", FLATSHARE_SERVING_PROFILE=str(SERVING))
    profile = json.loads(SERVING.read_text())
    assert (serve.mode, serve.max_calls, serve.audio) == ("serve", 2, "microphone")
    assert serve.max_calls == profile["voice"]["max_calls"]
    assert serve.describe() == {
        "mode": "serve",
        "max_calls": 2,
        "audio": "microphone",
        "turn_detection": "model",
        "serving_profile": str(SERVING),
        "profile": "rtx-5060-ti-serve",
        "node_binding": True,  # serve mode asks the gateway for each call's node
        "endpointing_min_delay": None,  # the configuration's own
        "vad_min_silence": None,
    }
    # The synthetic callers in serve mode, for a soak of the serving configuration.
    soak = settings(
        FLATSHARE_VOICE_MODE="serve",
        FLATSHARE_SERVING_PROFILE=str(SERVING),
        FLATSHARE_AUDIO="synthetic",
        FLATSHARE_TURN_DETECTION="vad",
    )
    assert (soak.audio, soak.turn_detection, soak.max_calls) == ("synthetic", "vad", 2)
    # Bench mode reads no profile.
    assert settings(FLATSHARE_SERVING_PROFILE=str(tmp_path / "none.json")).max_calls is None


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (None, "FLATSHARE_SERVING_PROFILE is not set"),
        ("none.json", "is no serving profile's serving.json"),
        ({"voice": {"audio": "microphone"}}, "is no serving profile's serving.json"),
        ({"voice": {"max_calls": 0, "audio": "microphone"}}, "voice.max_calls must be 1 or more"),
        (
            {"voice": {"max_calls": True, "audio": "microphone"}},
            "voice.max_calls must be 1 or more",
        ),
        ({"voice": {"max_calls": 2, "audio": "phone"}}, "FLATSHARE_AUDIO must be"),
    ],
)
def test_serve_mode_refuses_to_start_without_a_serving_profile_it_can_read(
    tmp_path: Path, profile: object, message: str
) -> None:
    env = {"FLATSHARE_VOICE_MODE": "serve"}
    if isinstance(profile, str):
        env["FLATSHARE_SERVING_PROFILE"] = str(tmp_path / profile)
    elif profile is not None:
        (tmp_path / "serving.json").write_text(json.dumps(profile))
        env["FLATSHARE_SERVING_PROFILE"] = str(tmp_path / "serving.json")
    with pytest.raises(ValueError, match=message):
        settings(**env)


def test_the_microphone_configuration_is_not_the_synthetic_callers() -> None:
    synthetic, microphone = settings(), settings(FLATSHARE_AUDIO="microphone")
    # The measured one, as every E2 point ran: Silero's and LiveKit's defaults, interruptions by VAD, no AEC warm-up.
    assert synthetic.vad_options() == {} and synthetic.aec_warmup is None
    handling = synthetic.turn_handling()
    assert "endpointing" not in handling and handling["interruption"] == {"mode": "vad"}
    # Real callers: the end-of-turn model, a VAD and interruptions that noise trips less, and the AEC warm-up.
    assert microphone.vad_options() == {"activation_threshold": 0.6, "min_speech_duration": 0.1}
    assert microphone.aec_warmup == 3.0
    handling = microphone.turn_handling()
    assert isinstance(handling["turn_detection"], inference.TurnDetector)
    assert handling["endpointing"] == {"mode": "fixed", "min_delay": 0.5, "max_delay": 3.0}
    assert handling["interruption"] == {
        "mode": "vad",
        "min_duration": 0.6,
        "resume_false_interruption": True,
        "false_interruption_timeout": 2.0,
    }
    # Fixed endpointing is the synthetic callers': with a microphone it is refused, naming the drop-in to unlink.
    with pytest.raises(ValueError, match="unlink vad-turns"):
        settings(FLATSHARE_AUDIO="microphone", FLATSHARE_TURN_DETECTION="vad")


def test_the_engine_is_asked_where_the_llms_socket_says_it_serves() -> None:
    assert settings().probe_path == "/ready"  # the gateway's: 200 only while the engine serves
    bypass = "/run/user/1007/flatshare/phastform-engine.sock"
    assert settings(FLATSHARE_LLM_UDS=bypass).probe_path == "/health"  # the engine's own


def test_serve_mode_starts_admission_with_livekits_threshold_at_one(
    started: list[object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status = tmp_path / "voice-worker.json"
    monkeypatch.setenv("FLATSHARE_VOICE_MODE", "serve")
    monkeypatch.setenv("FLATSHARE_SERVING_PROFILE", str(SERVING))
    monkeypatch.setenv("FLATSHARE_VOICE_STATUS", str(status))
    monkeypatch.setenv("FLATSHARE_VOICE_DRAIN", str(tmp_path / "voice-worker.drain"))
    worker.main()
    assert started == [{"num_idle_processes": 2}, {"load_threshold": 1.0}, worker.server]
    assert worker.ADMISSION is not None and worker.ADMISSION.max_calls == 2
    # Its first round has asked the gateway, which this test has none of: no call is taken, and the file says why.
    written = json.loads(status.read_text())
    assert (written["mode"], written["max_calls"], written["admitting"]) == ("serve", 2, False)
    assert written["reason"].startswith("the engine does not serve: /ready on ")
    assert worker.load(worker.server) == 1.0
