from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import threading
import time
import wave
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from flatshare_voice import audio, bots
from flatshare_voice.audio import FRAME, Clip, Recorder, resample, upsample_track
from flatshare_voice.bots import BotsConfig, Caller, Corpus, summarize
from flatshare_voice.callers import (
    BARGE_IN_LINES,
    TURN_LINES,
    VOICES,
    render,
)


def tone(hz: float, seconds: float = 1.0, rate: int = 24_000) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * hz * t) * 10_000).astype(np.int16)


def test_resampling_keeps_speech_band_and_rejects_aliases():
    speech = resample(tone(440), 2, 3)  # 24 kHz to the corpus's 16 kHz
    assert len(speech) == 16_000
    assert abs(int(np.abs(speech[2000:14000]).max()) - 10_000) < 100
    alias = resample(tone(11_000), 2, 3)  # above the new 8 kHz Nyquist
    assert int(np.abs(alias[2000:14000]).max()) < 30
    back = resample(speech, 3, 2)  # and up again, as --save-audio records the caller
    assert len(back) == 24_000
    assert abs(int(np.abs(back[3000:21000]).max()) - 10_000) < 150
    assert np.argmax(np.abs(np.fft.rfft(back))) == 440  # 1 Hz bins over the second


def test_a_track_is_upsampled_a_stretch_of_sound_at_a_time_exactly():
    silence = np.zeros(4001, dtype=np.int16)
    line = tone(300, 0.5, rate=16_000)
    track = np.concatenate([silence, line, silence[:100], line[:999], silence])  # odd offsets
    assert np.array_equal(upsample_track(track), resample(track, 3, 2))
    assert np.array_equal(upsample_track(silence), resample(silence, 3, 2))


class FakeTTS(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeTTS.requests.append(body)
        pcm = tone(300, 0.5).tobytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(pcm)))
        self.end_headers()
        self.wfile.write(pcm)

    def log_message(self, *args: object) -> None:
        pass


def test_render_writes_16k_clips_and_a_manifest(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTTS)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        manifest = render(f"http://127.0.0.1:{server.server_port}/v1", tmp_path)
    finally:
        server.shutdown()
    assert len(manifest) == len(TURN_LINES) + len(BARGE_IN_LINES)
    assert {r["voice"] for r in FakeTTS.requests} == set(VOICES)
    assert all(r["response_format"] == "pcm" and r["model"] == "kokoro" for r in FakeTTS.requests)
    first = manifest["turn-00"]
    assert first["kind"] == "turn" and first["text"] == TURN_LINES[0]
    assert first["duration_s"] == 0.9  # 0.2 s of padding on each side of 0.5 s of speech
    with wave.open(str(tmp_path / "turn-00.wav")) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16_000, 1, 2)
    clip = Clip.load("turn-00", str(tmp_path / "turn-00.wav"))
    assert (clip.onset_frame, clip.offset_frame) == (20, 70)  # speech between the paddings
    corpus = Corpus(tmp_path)
    assert len(corpus.turns) == len(TURN_LINES) and len(corpus.barge_ins) == len(BARGE_IN_LINES)


def test_summary_pools_v2v_samples_and_censors_the_replies_that_never_came():
    rows = [
        {"caller": 0, "turn": 0, "v2v_ms": 800.0, "barge_in": False},
        {"caller": 0, "turn": 1, "v2v_ms": 900.0, "barge_in": True, "interrupted_speech": True,
         "barge_in_to_silence_ms": 620.0, "v2v_after_barge_in_ms": 1000.0},
        {"caller": 1, "turn": 0, "v2v_ms": 1100.0, "barge_in": True, "interrupted_speech": False,
         "v2v_after_barge_in_ms": 950.0},
        {"caller": 1, "turn": 1, "error": "the agent did not start speaking"},
        # The reply came; after the interruption the agent never stopped speaking.
        {"caller": 1, "turn": 2, "v2v_ms": 700.0, "barge_in": True, "interrupted_speech": True,
         "error": "the agent did not stop speaking"},
    ]  # fmt: skip
    summary = summarize(rows, [0.004, 0.012], timeout_s=30.0)
    assert summary["turns"] == 5 and summary["errors"] == 2
    # The failed turn's reply and the unanswered interruption count at the 30 s timeout.
    assert summary["v2v_ms"] == {
        "n": 8, "censored": 2, "p50": 975.0, "p95": 30_000.0, "p99": 30_000.0
    }  # fmt: skip
    silence = summary["barge_in_to_silence_ms"]
    assert (silence["n"], silence["censored"], silence["p50"]) == (2, 1, 15_310.0)
    assert summary["barge_ins"] == 3 and summary["barge_ins_after_the_reply"] == 1
    assert summary["caller_max_lateness_ms"] == 12.0
    # Without a failure the summary is as it always was: no censored count.
    answered = summarize(rows[:3], [0.0])
    assert answered["v2v_ms"] == {"n": 5, "p50": 950.0, "p95": 1080.0, "p99": 1096.0}


def test_the_summary_leaves_the_warm_up_turns_out():
    rows = [
        {"caller": 0, "turn": 0, "v2v_ms": 2500.0, "barge_in": False, "warmup": True},
        {"caller": 0, "turn": 1, "v2v_ms": 800.0, "barge_in": False},
        {"caller": 1, "turn": 0, "error": "the agent did not start speaking", "warmup": True},
        {"caller": 1, "turn": 1, "v2v_ms": 900.0, "barge_in": False},
    ]
    summary = summarize(rows, [0.0])
    assert (summary["turns"], summary["errors"]) == (2, 0)
    assert summary["v2v_ms"] == {"n": 2, "p50": 850.0, "p95": 895.0, "p99": 899.0}
    assert summary == summarize([rows[1], rows[3]], [0.0])
    with pytest.raises(ValueError, match="warmup_turns cannot be negative"):
        BotsConfig("ws://fake", "key", "secret", Path("."), warmup_turns=-1)


def test_a_recorded_track_keeps_its_frames_together_and_its_pauses():
    recorder = Recorder(1000)  # 10-sample frames of 10 ms
    # Frames 0-2 come in one burst at 25 ms, 3 on time, 4 a millisecond late, 5 after a pause.
    for k, at in enumerate((0.025, 0.0251, 0.0252, 0.030, 0.041, 0.080)):
        recorder.add(at, np.full(10, k + 1, dtype=np.int16))
    signal = recorder.signal(0.0, 100)
    assert list(signal[:50]) == [n for n in range(1, 6) for _ in range(10)]  # from 0 ms, no gap
    assert not signal[50:80].any() and list(signal[80:90]) == [6] * 10


REPLY_S, SPOKEN_S = 0.2, 0.4
"""The fake agent answers each line REPLY_S after its end of speech, with SPOKEN_S of tone."""
FRAME_MS = 10.0
"""A frame of either side: the caller's microphone and the agent's track."""


def replied(v2v_ms: float, late_s: float) -> bool:
    """Whether a V2V the caller measured is the fake agent's REPLY_S, as far as the event loop that
    runs both sides lets it be: up to a frame short, since the caller stamps a line's end after its
    last frame has gone out, and up to a frame of each side long; and either way by as much as the
    caller's pacing loop ran late (its worst lateness, ``max_lateness_s``), which the agent's side
    ran late by too."""
    late = late_s * 1000
    return REPLY_S * 1000 - FRAME_MS - late <= v2v_ms <= REPLY_S * 1000 + 2 * FRAME_MS + late


class Line:
    """The fake room: the caller's microphone reaches a fake agent, which answers each line on the
    track the caller's AgentAudio reads, as LiveKit's AudioSource and AudioStream would carry it."""

    def __init__(self) -> None:
        self.replies: list[float] = []
        self._speaking = False

    def source(self, *_: Any, **__: Any) -> Line:  # rtc.AudioSource(rate, channels, ...)
        return self

    async def capture_frame(self, frame: Any) -> None:
        loud = int(np.abs(np.frombuffer(frame.data, dtype=np.int16)).max()) > 1000
        if self._speaking and not loud:
            self.replies.append(time.monotonic() + REPLY_S)
        self._speaking = loud
        await asyncio.sleep(0)

    async def aclose(self) -> None:
        pass

    def stream(self, *_: Any, **__: Any) -> Line:  # rtc.AudioStream(track, ...)
        return self

    async def __aiter__(self) -> AsyncIterator[Any]:
        """A frame of 10 ms every 10 ms, of the reply's tone or of silence."""
        speech, silence = tone(440, 0.01)[:240], np.zeros(240, dtype=np.int16)
        deadline = time.monotonic()
        while True:
            deadline += 0.01
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
            now = time.monotonic()
            replying = any(start <= now < start + SPOKEN_S for start in self.replies)
            frame = SimpleNamespace(data=memoryview(speech if replying else silence))
            yield SimpleNamespace(frame=frame)


def line_clip(name: str, seconds: float) -> Clip:
    """A 300 Hz line with 0.1 s of silence on each side, as the corpus pads its lines."""
    pad = np.zeros(10 * FRAME, dtype=np.int16)
    samples = np.concatenate([pad, tone(300, seconds, rate=16_000), pad])
    return Clip(name, samples, 10, 10 + round(seconds * 100))


def runs(channel: np.ndarray) -> list[tuple[int, int]]:
    """The first and last loud sample of each stretch of a channel without 10 ms of quiet."""
    loud = np.flatnonzero(np.abs(channel.astype(int)) > 3000)
    breaks = np.flatnonzero(np.diff(loud) > 240)  # longer than a tone's zero crossings
    return list(zip(loud[np.r_[0, breaks + 1]], loud[np.r_[breaks, -1]], strict=True))


def test_a_saved_call_lines_up_what_the_caller_sent_with_what_it_heard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    line = Line()
    monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
    monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)
    monkeypatch.setattr(bots, "QUIET_S", 0.2)
    corpus = SimpleNamespace(turns=[line_clip("turn-00", 0.3)], barge_ins=[line_clip("barge", 0.2)])
    config = BotsConfig("ws://fake", "key", "secret", tmp_path, barge_in=0.0, save_audio=True)
    config = dataclasses.replace(config, barge_in_delay=(0.05, 0.05))

    async def call() -> tuple[Caller, list[dict[str, Any]]]:
        caller = Caller(config, corpus, 0, "test")
        tasks = [
            asyncio.create_task(caller.microphone.run()),
            asyncio.create_task(caller.agent_audio.consume(line)),
        ]
        await asyncio.sleep(0.2)
        rows = [await caller.turn(0)]
        caller.config = dataclasses.replace(config, barge_in=1.0)  # the next one barges in
        rows.append(await caller.turn(1))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return caller, [{"caller": 0} | row for row in rows]

    caller, rows = asyncio.run(call())
    caller.save(tmp_path, rows)
    with wave.open(str(tmp_path / "call-0.wav")) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (2, 2, 24_000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).reshape(-1, 2)
    saved = json.loads((tmp_path / "call-0.json").read_text())
    assert (saved["caller"], saved["rate"], saved["channels"]) == (0, 24_000, ["caller", "agent"])
    assert saved["seconds"] == pytest.approx(len(pcm) / 24_000, abs=1e-3)
    wall = time.time() - time.monotonic()  # the wall clock's lead on the monotonic one
    assert saved["start"]["wall"] - saved["start"]["monotonic"] == pytest.approx(wall, abs=0.1)
    first, second = saved["turns"]
    assert {k: v for k, v in first.items() if k != "at"} == {
        k: v for k, v in rows[0].items() if k != "caller"
    }
    assert first["at"].keys() == {"start", "speech_end", "reply", "end"}
    assert second["barge_in"] and second["at"].keys() == {
        "start", "speech_end", "reply", "barge_in", "barge_in_speech_end", "silence",
        "barge_in_reply", "end",
    }  # fmt: skip
    # The caller's lines on the left, the agent's replies on the right, where the times say, as
    # closely as the caller's pacing loop kept time.
    late = caller.microphone.max_lateness_s
    sent, heard = runs(pcm[:, 0]), runs(pcm[:, 1])
    assert len(sent) == 3 and len(heard) == 3
    turns = [(first, ""), (second, ""), (second, "barge_in_")]
    ends = [turn["at"][f"{event}speech_end"] for turn, event in turns]
    replies = [turn["at"][f"{event}reply"] for turn, event in turns]
    assert [last / 24_000 for _, last in sent] == pytest.approx(ends, abs=0.015 + late)
    assert [onset / 24_000 for onset, _ in heard] == pytest.approx(replies, abs=0.015 + late)
    v2v = [first["v2v_ms"], second["v2v_ms"], second["v2v_after_barge_in_ms"]]
    gaps = [(onset - last) / 24 for (_, last), (onset, _) in zip(sent, heard, strict=True)]
    assert gaps == pytest.approx(v2v, abs=25 + 1000 * late)  # the V2V the callers measured
    assert all(replied(ms, late) for ms in v2v), v2v


def test_a_call_opens_with_its_warm_up_turns_and_measures_the_same_turns_after_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bots, "QUIET_S", 0.2)
    lines = [line_clip(f"turn-{n:02}", 0.2) for n in range(8)]
    corpus = SimpleNamespace(turns=lines, barge_ins=[line_clip("barge", 0.2)])
    config = BotsConfig("ws://fake", "key", "secret", tmp_path, turns=2, barge_in=1.0, seed=3)
    config = dataclasses.replace(config, barge_in_delay=(0.05, 0.05), think=(0.0, 0.0))

    def call(warmup_turns: int) -> tuple[list[dict[str, Any]], float]:
        line = Line()
        monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
        monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)

        async def play() -> tuple[list[dict[str, Any]], float]:
            caller = Caller(dataclasses.replace(config, warmup_turns=warmup_turns), corpus, 0, "t")
            tasks = [
                asyncio.create_task(caller.microphone.run()),
                asyncio.create_task(caller.agent_audio.consume(line)),
            ]
            await asyncio.sleep(0.2)
            rows: list[dict[str, Any]] = []
            with (tmp_path / f"turns-{warmup_turns}.jsonl").open("w") as log:
                await caller.play(rows, log)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return rows, caller.microphone.max_lateness_s

        return asyncio.run(play())

    (plain, _), (warmed, late) = call(0), call(1)
    # One warm-up turn first, a line and its reply without an interruption, marked as such.
    assert [(row["turn"], row.get("warmup", False), row["barge_in"]) for row in warmed] == [
        (0, True, False),
        (1, False, True),
        (2, False, True),
    ]
    assert replied(warmed[0]["v2v_ms"], late), warmed[0]["v2v_ms"]
    # The measured turns play the lines and interruptions of the same call without it.
    script = [(row["clip"], row["barge_in_clip"]) for row in plain]
    assert [(row["clip"], row["barge_in_clip"]) for row in warmed[1:]] == script
    logged = (tmp_path / "turns-1.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in logged] == warmed


def test_a_caller_in_order_plays_the_corpus_lines_one_after_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bots, "QUIET_S", 0.2)
    lines = [line_clip(f"turn-{n:02}", 0.2) for n in range(8)]
    corpus = SimpleNamespace(turns=lines, barge_ins=[line_clip("barge", 0.2)])
    config = BotsConfig(
        "ws://fake", "key", "secret", tmp_path, turns=3, barge_in=0.0, seed=3, warmup_turns=1
    )
    config = dataclasses.replace(config, think=(0.0, 0.0), in_order=True)

    def call(index: int) -> list[dict[str, Any]]:
        line = Line()
        monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
        monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)

        async def play() -> list[dict[str, Any]]:
            caller = Caller(config, corpus, index, "t")
            tasks = [
                asyncio.create_task(caller.microphone.run()),
                asyncio.create_task(caller.agent_audio.consume(line)),
            ]
            await asyncio.sleep(0.2)
            rows: list[dict[str, Any]] = []
            with (tmp_path / f"turns-{index}.jsonl").open("w") as log:
                await caller.play(rows, log)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return rows

        return asyncio.run(play())

    first, second = call(0), call(1)
    # The warm-up turn draws its line as before; the measured turns take the lines in order, caller i
    # from line i x turns.
    assert [row["clip"] for row in first[1:]] == ["turn-00", "turn-01", "turn-02"]
    assert [row["clip"] for row in second[1:]] == ["turn-03", "turn-04", "turn-05"]
    assert first[0].get("warmup") and second[0].get("warmup")


def test_a_call_told_to_stop_ends_before_its_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The arrivals' calls end when their run does, or when the agent has left (go_on); E2's calls, without go_on,
    # play every turn. The lines up to the stop are those of the same call played through.
    monkeypatch.setattr(bots, "QUIET_S", 0.2)
    lines = [line_clip(f"turn-{n:02}", 0.2) for n in range(8)]
    corpus = SimpleNamespace(turns=lines, barge_ins=[line_clip("barge", 0.2)])
    config = BotsConfig("ws://fake", "key", "secret", tmp_path, turns=4, barge_in=0.0, seed=5)
    config = dataclasses.replace(config, think=(0.0, 0.0))

    def call(go_on: Any) -> list[dict[str, Any]]:
        line = Line()
        monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
        monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)

        async def play() -> list[dict[str, Any]]:
            caller = Caller(config, corpus, 0, "t")
            tasks = [
                asyncio.create_task(caller.microphone.run()),
                asyncio.create_task(caller.agent_audio.consume(line)),
            ]
            await asyncio.sleep(0.2)
            rows: list[dict[str, Any]] = []
            with (tmp_path / "turns.jsonl").open("w") as log:
                if go_on is None:
                    await caller.play(rows, log)
                else:
                    await caller.play(rows, log, go_on=lambda: go_on(rows))
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return rows

        return asyncio.run(play())

    every = call(None)
    stopped = call(lambda rows: len(rows) < 2)
    assert [row["turn"] for row in every] == [0, 1, 2, 3]
    assert [row["clip"] for row in stopped] == [row["clip"] for row in every[:2]]
    assert call(lambda rows: False) == []  # told before its first turn: none


def test_a_failed_turn_leaves_the_turns_after_it_as_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A turn whose reply never came drew no interruption, so the turns after it played other lines
    # and interruptions than the same call answered: a failing point's calls were not the others'.
    # The caller's audio is stood in for, each line answered as it ends but the failing one.
    monkeypatch.setattr(audio.rtc, "AudioSource", Line().source)
    lines = [line_clip(f"turn-{n:02}", 0.2) for n in range(8)]
    barges = [line_clip(f"barge-{n}", 0.2) for n in range(4)]
    corpus = SimpleNamespace(turns=lines, barge_ins=barges)

    async def call(seed: int, unanswered: int | None) -> list[tuple[str | None, str | None]]:
        config = BotsConfig(
            "ws://fake", "key", "secret", tmp_path, turns=8, barge_in=0.5, seed=seed
        )
        config = dataclasses.replace(config, barge_in_delay=(0.0, 0.0), think=(0.0, 0.0))
        caller = Caller(config, corpus, 0, "t")

        async def play(clip: Clip) -> dict[str, float]:
            return dict.fromkeys(("start", "onset", "speech_end", "end"), time.monotonic())

        async def first_onset_after(t: float, timeout: float) -> float:
            if max(caller.moments) == unanswered:
                raise TimeoutError("the agent did not start speaking")
            return t

        async def silent_after(t: float, quiet: float, timeout: float) -> float:
            return t

        monkeypatch.setattr(caller.microphone, "play", play)
        monkeypatch.setattr(caller.agent_audio, "first_onset_after", first_onset_after)
        monkeypatch.setattr(caller.agent_audio, "silent_after", silent_after)
        rows: list[dict[str, Any]] = []
        with (tmp_path / "turns.jsonl").open("w") as log:
            await caller.play(rows, log)
        assert ("error" in rows[2]) == (unanswered == 2)
        return [(row.get("clip"), row.get("barge_in_clip")) for row in rows]

    for seed in range(10):
        answered, failed = (asyncio.run(call(seed, turn)) for turn in (None, 2))
        assert failed[3:] == answered[3:], seed  # the same lines and interruptions after it
        assert any(barge for _, barge in answered) and not all(barge for _, barge in answered)


class FirstReplyOnly(Line):
    """The fake agent answers the first line it hears and never the interruption after it."""

    async def capture_frame(self, frame: Any) -> None:
        await super().capture_frame(frame)
        del self.replies[1:]


def test_an_unanswered_interruption_fails_the_turn_and_keeps_its_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    line = FirstReplyOnly()
    monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
    monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)
    monkeypatch.setattr(bots, "QUIET_S", 0.2)
    corpus = SimpleNamespace(turns=[line_clip("turn-00", 0.3)], barge_ins=[line_clip("barge", 0.2)])
    config = BotsConfig("ws://fake", "key", "secret", tmp_path, barge_in=1.0, reply_timeout=0.5)
    config = dataclasses.replace(config, barge_in_delay=(0.05, 0.05))

    async def call() -> tuple[dict[str, Any], float]:
        caller = Caller(config, corpus, 0, "test")
        tasks = [
            asyncio.create_task(caller.microphone.run()),
            asyncio.create_task(caller.agent_audio.consume(line)),
        ]
        await asyncio.sleep(0.2)
        row = await caller.turn(0)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return row, caller.microphone.max_lateness_s

    row, late = asyncio.run(call())
    assert row["error"] == "the agent did not start speaking"
    assert row["barge_in"] and "v2v_after_barge_in_ms" not in row
    assert replied(row["v2v_ms"], late), row["v2v_ms"]  # the reply it measured stays
    summary = summarize([row], [0.0], timeout_s=0.5)
    assert summary["errors"] == 1
    assert (summary["v2v_ms"]["n"], summary["v2v_ms"]["censored"]) == (2, 1)
    assert summary["v2v_ms"]["p99"] == pytest.approx(500.0, abs=5)


def test_the_callers_join_their_rooms_a_stagger_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # LiveKit's dispatcher dropped the job of a room created within milliseconds of the others' (W5's calls per GPU,
    # six callers: "no workers with sufficient capacity", and it does not retry), so caller-1 waited for an agent
    # that never came. The callers join one stagger apart.
    joined: list[tuple[int, float]] = []

    async def call(self: Caller, rows: list[dict[str, Any]], log: Any) -> None:
        joined.append((self.index, time.monotonic()))

    monkeypatch.setattr(bots.Caller, "call", call)
    monkeypatch.setattr(bots, "Corpus", lambda _: SimpleNamespace(turns=[], barge_ins=[]))
    monkeypatch.setattr(bots, "summarize", lambda *_: {})
    monkeypatch.setattr(bots, "JOIN_STAGGER_S", 0.05, raising=False)
    config = dataclasses.replace(BotsConfig("ws://fake", "key", "secret", tmp_path), callers=6)
    asyncio.run(bots.run_bots(config, tmp_path / "out"))
    order = [index for index, _ in sorted(joined, key=lambda j: j[1])]
    assert order == list(range(6))
    starts = [at for _, at in sorted(joined, key=lambda j: j[1])]
    assert all(b - a >= 0.045 for a, b in itertools.pairwise(starts)), starts


def test_the_agents_state_says_when_it_stops_speaking() -> None:
    state = audio.AgentState()
    state.record("listening", at=1.0)
    state.record("listening", at=1.5)  # no change, no entry
    state.record("speaking", at=2.0)
    state.record("listening", at=4.0)
    assert state.changes == [(1.0, "listening"), (2.0, "speaking"), (4.0, "listening")]
    assert [state.at(t) for t in (0.5, 1.2, 2.0, 3.9, 9.0)] == [
        None,
        "listening",
        "speaking",
        "speaking",
        "listening",
    ]
    # Speaking at 2.5 or from a change after 1.5 (the state can arrive before the caller hears the audio): 4.0.
    assert asyncio.run(state.left_speaking_after(2.5, 0.0)) == 4.0
    assert asyncio.run(state.left_speaking_after(1.5, 0.0)) == 4.0
    # After the reply's end, no change says it ended: None once the timeout passes.
    assert asyncio.run(state.left_speaking_after(4.5, 0.05)) is None


def test_the_agents_state_is_read_on_a_long_running_hosts_clock() -> None:
    # The callers' clock is the process's monotonic one, which on a host up for a year reads in the tens of millions
    # of seconds, where a nanosecond is below the clock's resolution: the times rg-r1's first call saw on cyan.
    state = audio.AgentState()
    for at, name in (
        (38207125.226, "listening"),
        (38207128.66, "thinking"),
        (38207129.288, "speaking"),
    ):
        state.record(name, at=at)
    state.record("listening", at=38207135.704)
    assert asyncio.run(state.left_speaking_after(38207129.36, 0.0)) == 38207135.704


def test_a_replys_pauses_are_its_quiet_stretches_between_spans() -> None:
    heard = audio.AgentAudio()
    heard.spans = [[1.0, 1.8], [2.0, 2.5], [3.2, 3.6], [9.0, 9.5]]
    assert [round(g, 3) for g in heard.gaps_between(1.0, 4.0, 0.3)] == [0.7]
    assert [round(g, 3) for g in heard.gaps_between(1.0, 4.0, 0.1)] == [0.2, 0.7]
    assert heard.gaps_between(9.0, 9.5, 0.1) == []


PAUSE = 0.6
"""The fake agent's pause inside its reply: SPOKEN_S of speech, PAUSE of quiet, SPOKEN_S again."""


class PausedLine(Line):
    """A fake agent whose replies pause halfway, and which publishes its state as LiveKit Agents does: speaking from
    the reply's first frame, listening once its last has gone out."""

    def __init__(self, state: audio.AgentState | None) -> None:
        super().__init__()
        self.state = state

    async def capture_frame(self, frame: Any) -> None:
        before = len(self.replies)
        await super().capture_frame(frame)
        if self.state is not None and len(self.replies) > before:
            start = self.replies[-1]
            asyncio.get_running_loop().call_at(start, self.state.record, "speaking")
            end = start + 2 * SPOKEN_S + PAUSE
            asyncio.get_running_loop().call_at(end, self.state.record, "listening")

    async def __aiter__(self) -> AsyncIterator[Any]:
        speech, silence = tone(440, 0.01)[:240], np.zeros(240, dtype=np.int16)
        deadline = time.monotonic()
        while True:
            deadline += 0.01
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
            now = time.monotonic()
            replying = any(
                start <= now < start + SPOKEN_S
                or start + SPOKEN_S + PAUSE <= now < start + 2 * SPOKEN_S + PAUSE
                for start in self.replies
            )
            frame = SimpleNamespace(data=memoryview(speech if replying else silence))
            yield SimpleNamespace(frame=frame)


def test_the_agents_state_ends_a_reply_that_pauses_where_silence_would_cut_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A reply that pauses for 0.6 s halfway: with the agent's state the caller waits for its second half and counts
    # the pause; by silence alone, with the quiet that ends a reply shorter than the pause, it takes the pause for the
    # end and its next line would talk over the rest of the reply.
    monkeypatch.setattr(bots, "QUIET_S", 0.5)
    monkeypatch.setattr(bots, "TAIL_S", 0.1)
    lines = [line_clip(f"turn-{n:02}", 0.2) for n in range(4)]
    corpus = SimpleNamespace(turns=lines, barge_ins=[line_clip("barge", 0.2)])
    config = BotsConfig("ws://fake", "key", "secret", tmp_path, turns=2, barge_in=0.0, seed=1)
    config = dataclasses.replace(config, think=(0.0, 0.0))

    def call(publishes_state: bool) -> list[dict[str, Any]]:
        async def play() -> list[dict[str, Any]]:
            state = audio.AgentState()
            line = PausedLine(state if publishes_state else None)
            monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
            monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)
            caller = Caller(config, corpus, 0, "t")
            caller.agent_state = state
            if publishes_state:
                state.record("listening")
            tasks = [
                asyncio.create_task(caller.microphone.run()),
                asyncio.create_task(caller.agent_audio.consume(line)),
            ]
            await asyncio.sleep(0.2)
            rows: list[dict[str, Any]] = []
            with (tmp_path / f"turns-{publishes_state}.jsonl").open("w") as log:
                await caller.play(rows, log)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return rows

        return asyncio.run(play())

    told = call(True)
    assert [row["end_by"] for row in told] == ["state", "state"]
    for row in told:
        [pause] = row["reply_pauses_ms"]
        assert PAUSE * 1000 - 30 <= pause <= PAUSE * 1000 + 30, row
        assert row["reply_audio_ms"] >= (2 * SPOKEN_S + PAUSE) * 1000 - 30, row
    guessed = call(False)
    assert [row["end_by"] for row in guessed] == ["silence", "silence"]
    # Silence took the pause for the end: the reply it measured is its first half, with no pause in it.
    for row in guessed:
        assert row["reply_pauses_ms"] == [] and row["reply_audio_ms"] < (SPOKEN_S + 0.1) * 1000, row
    summary = summarize(told, [0.0])["replies"]
    assert (summary["n"], summary["ended_by_state"], summary["with_a_pause_over_500_ms"]) == (
        2,
        2,
        2,
    )
    assert (
        summary["with_a_pause_of_quiet_s"] == 2
    )  # a pause as long as the quiet that would end a reply
