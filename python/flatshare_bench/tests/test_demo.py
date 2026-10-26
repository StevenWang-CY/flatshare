"""The demo video's renderer (benchmarks/e2/demo.py) on a synthetic pair of saved calls: tones for
the caller's lines and the agent's replies, in the files flatshare-voice bots --save-audio writes, and a
report's frontier.csv for the SLO and the closing card.

The turns, the soundtrack, the motion's curves and the frames run here, headless with this venv's
numpy, matplotlib and Pillow, at a low resolution: their size, their count, and that no text leaves
its room or overlaps another at any moment the calls change. Encoding needs the ffmpeg that the
script's own dependencies bring: that test runs the script with uv as the docs do, and is marked
integration, so CI leaves it out."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parents[3] / "benchmarks" / "e2" / "demo.py"
_spec = importlib.util.spec_from_file_location("demo", DEMO)
assert _spec is not None and _spec.loader is not None
demo = importlib.util.module_from_spec(_spec)
sys.modules["demo"] = demo
_spec.loader.exec_module(demo)

RATE = 24_000
WALL = 1_800_000_000.0  # 2027-01-15 UTC
LINE_S, REPLY_S, PAUSE_S = 1.6, 2.0, 2.5
"""Each line's speech, each reply, and the quiet then the think time before the next line."""


def tone(hz: float, seconds: float) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * hz * t) * 12_000).astype(np.int16)


def saved_call(run_dir: Path, v2v_s: list[float], *, label: str, point: str) -> Path:
    """A point's run directory with one saved call: a warm-up turn, then a turn per V2V, each a line,
    the reply V2V later and a pause; the second measured turn is interrupted and answered again."""
    pcm = np.zeros((round(20 * (len(v2v_s) + 1) * RATE), 2), dtype=np.int16)
    turns, t = [], 1.0
    for number, v2v in enumerate([1.0, *v2v_s]):
        start, speech_end = t, t + 0.2 + LINE_S
        pcm[round((start + 0.2) * RATE) :][: round(LINE_S * RATE), 0] = tone(300, LINE_S)
        reply = speech_end + v2v
        pcm[round(reply * RATE) :][: round(REPLY_S * RATE), 1] = tone(440, REPLY_S)
        row: dict[str, Any] = {"turn": number, "clip": f"turn-{number:02d}", "v2v_ms": v2v * 1000}
        at = {"start": start, "speech_end": speech_end, "reply": reply}
        end = reply + REPLY_S
        if number == 0:
            row["warmup"] = True
        if number == 2:
            onset = reply + 0.8
            pcm[round(onset * RATE) :][: round(0.6 * RATE), 0] = tone(300, 0.6)
            pcm[round((onset + 0.3) * RATE) : round(end * RATE), 1] = 0  # the agent stops
            second = onset + 0.6 + v2v
            pcm[round(second * RATE) :][: round(REPLY_S * RATE), 1] = tone(440, REPLY_S)
            row |= {"barge_in": True, "barge_in_clip": "barge_in-40", "interrupted_speech": True}
            row |= {"barge_in_to_silence_ms": 300.0, "v2v_after_barge_in_ms": v2v * 1000}
            at |= {"barge_in": onset, "barge_in_speech_end": onset + 0.6, "silence": onset + 0.3}
            at |= {"barge_in_reply": second}
            end = second + REPLY_S
        row.setdefault("barge_in", False)
        turns.append(row | {"at": at | {"end": end}})
        t = end + PAUSE_S
    pcm = pcm[: round((t + 1) * RATE)]
    (run_dir / "bots").mkdir(parents=True)
    with wave.open(str(run_dir / "bots" / "call-0.wav"), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())
    call = {"caller": 0, "rate": RATE, "channels": ["caller", "agent"], "turns": turns}
    call["start"] = {"monotonic": 5000.0, "wall": WALL}
    (run_dir / "bots" / "call-0.json").write_text(json.dumps(call))
    arm = {"label": label, "voice": {"callers": 2}, "agents": {"sessions": 24}}
    (run_dir / "run.json").write_text(json.dumps({"name": point, "arm": arm}))
    return run_dir


def frontier(path: Path, on: str = "", scale: float = 1.0) -> Path:
    """A report's frontier.csv: voice alone with two callers (V2V p95 1,500 and 1,600 ms, so an
    SLO of 1,800 ms), and the two points' runs in another corpus with as many callers; ``on``
    names the GPU in the workloads, as off the box, and ``scale`` multiplies the agents' tokens/s."""
    rows = [("1", f"weka-x32, 2 callers{on}", "alone", 1500.0, "")]
    rows += [("1", f"weka-x32, 2 callers{on}", "alone", 1600.0, "")]
    long = f"weka-x32-long, 2 callers{on}"
    rows += [("4", long, "frozen", p95, 900.0 * scale) for p95 in (3000.0, 3500.0)]
    rows += [("9", long, "a9", p95, 880.0 * scale) for p95 in (1100.0, 1300.0, 1200.0)]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "workload", "point", "v2v_p95_ms", "fixed_window_tokens_per_s"])
        writer.writerows(rows)
    return path


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    slow = saved_call(tmp_path / "frozen", [3.0, 2.5, 4.0], label="Frozen box", point="frozen")
    fast = saved_call(tmp_path / "a9", [1.0, 0.8, 1.2], label="Arm 9", point="a9")
    return slow, fast


@pytest.fixture
def timeline(pair: tuple[Path, Path]) -> Any:
    return demo.Timeline.choose((demo.Call.load(pair[0]), demo.Call.load(pair[1])))


@pytest.fixture
def summary(pair: tuple[Path, Path], tmp_path: Path) -> Any:
    return demo.Summary.read(frontier(tmp_path / "frontier.csv"), ["frozen", "a9"])


def test_both_calls_take_their_measured_turns_together(timeline: Any) -> None:
    calls = timeline.calls
    assert [call.label for call in calls] == ["Frozen box", "Arm 9"]
    assert [call.gpu for call in calls] == ["RTX 5060 Ti"] * 2  # no samples: the box's
    # The warm-up turn is left out; each turn lasts as long as the slower call's.
    assert timeline.first == 1 and timeline.turns == 3
    lengths = np.diff(timeline.offsets)
    assert lengths == pytest.approx([9.3, 12.7, 11.8])  # the last to the end of the call
    assert all(
        b - a <= slow + 1e-9 for (a, b), slow in zip(timeline.spans[1], lengths, strict=True)
    )
    for side in (0, 1):  # each turn's line starts at the same moment on both sides
        speaking = demo.envelope(timeline.track(side)[:, 0], RATE) > 0.1
        starts = (np.flatnonzero(np.diff(speaking.astype(int)) == 1) + 1) / 100
        for offset in timeline.offsets[:-1]:
            assert np.min(np.abs(starts - (offset + 0.7))) <= 0.015
    # The faster call waits in silence for the slower one's turn to end.
    quiet = timeline.track(1)[round((timeline.offsets[1] - 0.2) * RATE) :][: round(0.1 * RATE)]
    assert not quiet.any()
    assert demo.Timeline.choose(calls, first=2, turns=2).offsets[0] == 0.0
    with pytest.raises(ValueError, match="both calls have 2 turns from turn 2, not 3"):
        demo.Timeline.choose(calls, first=2, turns=3)


def test_the_waits_and_the_soundtrack(timeline: Any) -> None:
    slow, fast = timeline.waits(0), timeline.waits(1)
    assert [w.ms for w in slow] == [3000.0, 2500.0, 2500.0, 4000.0]  # turn 2 is answered again
    assert [w.again for w in slow] == [False, False, True, False]
    assert [round(w.until - w.ended, 3) for w in fast] == [1.0, 0.8, 0.8, 1.2]
    assert fast[0].ended == pytest.approx(slow[0].ended)  # the same line ends together
    assert [k for k, _ in timeline.interruptions(0)] == [1]
    # One call at a time: the baseline's turns until the start nearest the middle, then the other's.
    assert timeline.switch == 2  # 22.0 s, the start nearest 16.9 s
    sound, tracks = demo.soundtrack(timeline), [timeline.track(side) for side in (0, 1)]
    mixes = [np.clip(track.astype(float).sum(axis=1) * 0.7, -32768, 32767) for track in tracks]
    at = round(timeline.offsets[2] * RATE)
    assert np.array_equal(sound[: at - 2000], mixes[0][: at - 2000].astype(np.int16))
    assert np.array_equal(sound[at + 2000 :], mixes[1][at + 2000 :].astype(np.int16))


def test_the_summary_is_the_reports(summary: Any, tmp_path: Path) -> None:
    assert summary.slo_ms == pytest.approx(1800.0)  # voice alone's median p95 plus 250 ms
    assert summary.gpu is None  # the box's workloads name no GPU
    assert summary.p95_ms == (3250.0, 1200.0) and summary.runs == (2, 3)
    assert summary.tokens == (900.0, 880.0)
    with pytest.raises(ValueError, match="0 runs of absent"):
        demo.Summary.read(tmp_path / "frontier.csv", ["frozen", "absent"])


def test_motion_follows_its_curves() -> None:
    move, enter = demo.EASE_MOVE, demo.EASE_ENTER
    assert (move(0.0), move(1.0), enter(0.0), enter(1.0)) == (0.0, 1.0, 0.0, 1.0)
    xs = np.linspace(0, 1, 101)
    assert np.all(np.diff([move(x) for x in xs]) >= 0) and np.all(
        np.diff([enter(x) for x in xs]) >= 0
    )
    # Each is CSS's curve: (x, y) of the Bezier through its control points, at the same parameter.
    for curve, (x1, y1, x2, y2) in ((move, (0.77, 0, 0.175, 1)), (enter, (0.23, 1, 0.32, 1))):
        u = np.linspace(0, 1, 200_001)
        bx = 3 * x1 * (1 - u) ** 2 * u + 3 * x2 * (1 - u) * u**2 + u**3
        by = 3 * y1 * (1 - u) ** 2 * u + 3 * y2 * (1 - u) * u**2 + u**3
        for x in (0.1, 0.3, 0.5, 0.7, 0.9):
            assert curve(x) == pytest.approx(np.interp(x, bx, by), abs=1e-4)
    assert move(0.2) < 0.2 and move(0.8) > 0.8 and enter(0.2) > 0.5  # in and out; quick to arrive
    # An arrival starts at 0.96 of its size and nothing, a leave is faster than an arrival.
    assert demo.arriving(1.0, 1.0) == (0.0, 0.96) and demo.arriving(9.0, 1.0) == (1.0, 1.0)
    assert demo.EXIT_S < demo.ENTER_S and demo.CARD_EXIT_S < demo.CARD_ENTER_S
    assert demo.leaving(0.0, 1.0) == 1.0 and demo.leaving(1.0, 1.0) == 0.0


def test_a_frame_shows_the_live_wait_and_each_reply(timeline: Any, summary: Any) -> None:
    frames = demo.Frames(timeline, summary, width=320)
    first = timeline.waits(0)[0]
    image = frames.draw(first.ended + 2.0)  # the baseline still waits; the other replied after 1 s
    assert image.shape == (180, 320, 3) and image.dtype == np.uint8
    slow, fast = frames.rows
    assert (slow.status.get_text(), slow.clock.get_text()) == ("waiting for the reply", "2.0 s")
    assert (fast.status.get_text(), fast.clock.get_text()) == ("answered after", "1.0 s")
    assert (slow.heard.get_text(), fast.heard.get_text()) == ("sound on", "")
    (_, bar, value, _), (_, _, fast_value, _) = frames.bars[0][0], frames.bars[1][0]
    assert bar.get_visible() and value.get_alpha() == 0.0  # still growing, no value yet
    height = np.ptp(bar.get_xy()[:, 1])
    assert height == pytest.approx(144 * 2.0 / frames.bar_top, abs=0.1)  # two seconds so far
    assert fast_value.get_text() == "1.0" and fast_value.get_alpha() == 1.0
    assert not frames.bars[0][1][1].get_visible()  # the next turn's bar is not yet drawn
    # The SLO rule stands 1.8 s after the line on the wait's lane.
    rule = frames.rows[0].rules[0]
    assert rule.get_xdata()[0] == pytest.approx(frames.x_of(first.ended - 0 + 1.8))
    frames.draw(timeline.seconds - 0.01)
    assert slow.clock.get_text() == "4.0 s"
    assert (slow.heard.get_text(), fast.heard.get_text()) == ("", "sound on")
    assert [v.get_text() for _, _, v, _ in frames.bars[0]] == ["3.0", "2.5", "2.5", "4.0"]


def test_each_v2v_holds_once_it_is_given(timeline: Any, summary: Any) -> None:
    frames = demo.Frames(timeline, summary, width=320)
    for side, row in enumerate(frames.rows):
        for wait in timeline.waits(side):  # an interruption's among them, 1.4 s after its reply
            for t in (wait.until + 0.01, wait.until + demo.HOLD_S):
                frames.update(t)
                assert row.clock.get_text() == f"{wait.seconds:.1f} s"
                assert row.clock.get_alpha() == 1.0


def test_no_text_leaves_its_room_or_overlaps_another(
    timeline: Any, summary: Any, tmp_path: Path
) -> None:
    forms = [(1280, False, False), (1280, True, False), (680, True, True)]
    for theme in ("light", "dark"):
        for width, excerpt, narrow in forms:  # the video's, the README's and its phone layout's
            frames = demo.Frames(
                timeline,
                summary,
                width=width,
                theme=theme,
                sound=not excerpt,
                excerpt=excerpt,
                narrow=narrow,
            )
            moments = frames.moments()
            assert len(moments) > 20
            for t in moments:
                frames.check(t)
    demo.title_card(timeline, "A voice agent and coding agents", width=1280)
    demo.closing_card(summary, timeline, width=1280, theme="dark")
    # Another GPU's report of the same two points: a group of its own, against its own SLO.
    on = ", NVIDIA A100-SXM4-80GB, 1 warm-up turn"
    a100 = demo.Summary.read(frontier(tmp_path / "a100.csv", on, 3.0), ["frozen", "a9"])
    assert a100.gpu == "A100 SXM4 80 GB" and a100.tokens == (2700.0, 2640.0)
    assert demo.shown("NVIDIA GeForce RTX 5060 Ti") == "RTX 5060 Ti"
    assert demo.shown("NVIDIA H100 80GB HBM3") == "H100 80 GB HBM3"
    for theme in ("light", "dark"):
        demo.closing_card(summary, timeline, also=[a100], width=1280, theme=theme)
    # The check refuses a text out of its room.
    frames = demo.Frames(timeline, summary, width=640)
    artist, room = frames.board.texts[0]
    frames.board.texts[0] = (artist, (room[0] + 400, *room[1:]))
    with pytest.raises(ValueError, match="leaves its room"):
        frames.check(0.0)


def test_the_excerpt_loops_without_the_sound_and_the_bars(timeline: Any, summary: Any) -> None:
    narrow = demo.Frames(timeline, summary, width=340, sound=False, excerpt=True, narrow=True)
    assert narrow.board.across == 420 and narrow.draw(1.0).shape[1] == 340
    assert len(narrow.playheads) == 2  # one a lane: each call's name stands between them
    with pytest.raises(ValueError, match="an excerpt's"):
        demo.Frames(timeline, summary, width=340, narrow=True)
    frames = demo.Frames(timeline, summary, width=320, sound=False, excerpt=True)
    assert frames.draw(1.0).shape == (112, 320, 3)  # 1120 by 392 units
    texts = [artist.get_text() for artist, _ in frames.board.texts]
    assert "Each turn's wait for the reply" not in texts and not hasattr(frames, "bars")
    frames.update(0.0)
    assert all(slot.artist.get_alpha() == 0.0 for slot in frames.slots if slot.loop_end)
    frames.update(timeline.seconds - 1e-3)  # the last turn leaves before the loop restarts
    assert all(playhead.get_alpha() < 0.05 for playhead in frames.playheads)
    assert all(row.heard is None for row in frames.rows)  # silent: no call is marked as heard


def test_the_video_is_its_planned_frames_each_its_size(timeline: Any, summary: Any) -> None:
    steps, generate = demo.video_frames(timeline, summary, fps=4, width=160, heading="A title")
    assert (steps.title, steps.arrival, steps.tail) == (24, 2, 37)
    assert steps.calls == round(timeline.seconds * 4) and steps.lead == 26
    frames = list(generate())
    assert len(frames) == steps.total
    assert {frame.shape for frame in frames} == {(90, 160, 3)}
    page = frames[0][0, 0]
    assert (frames[0] == page).all()  # the opening card arrives from the page
    assert not (frames[12] == page).all() and not (frames[-1] == page).all()
    # The same records draw the same frames.
    again = list(demo.video_frames(timeline, summary, fps=4, width=160, heading="A title")[1]())
    assert all(np.array_equal(a, b) for a, b in zip(frames, again, strict=True))
    without = demo.plan(timeline, None, None, 4)
    assert (without.title, without.tail, without.total) == (0, 0, without.arrival + without.calls)


def test_another_gpus_report_needs_the_calls_own(
    pair: tuple[Path, Path], tmp_path: Path, capsys
) -> None:
    also = frontier(tmp_path / "a100.csv", ", NVIDIA A100-SXM4-80GB")
    with pytest.raises(SystemExit):
        demo.main([*map(str, pair), "--out", str(tmp_path / "demo.mp4"), "--also", str(also)])
    assert "--also adds to the closing card that --summary makes" in capsys.readouterr().err


def test_a_text_needs_its_glyphs() -> None:
    board = demo.Board(160, demo.FULL_H)
    with pytest.raises(ValueError, match="no glyph"):
        board.text(10, 10, "\N{SNOWMAN}")
    with pytest.raises(ValueError, match="an even number of pixels"):
        demo.Board(171, demo.FULL_H)


@pytest.mark.integration
def test_the_script_renders_a_video_and_an_excerpt(pair: tuple[Path, Path], tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv, which brings the script's dependencies")
    out = tmp_path / "demo.mp4"
    command = [uv, "run", "-q", str(DEMO), *map(str, pair), "--out", str(out)]
    command += ["--summary", str(frontier(tmp_path / "frontier.csv")), "--heading", "A title"]
    command += ["--also", str(frontier(tmp_path / "a100.csv", ", NVIDIA A100-SXM4-80GB", 3.0))]
    command += ["--fps", "10", "--width", "320", "--turns", "2"]
    command += [
        "--excerpt",
        str(tmp_path / "demo"),
        "--excerpt-width",
        "320",
        "--excerpt-fps",
        "10",
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    frames = re.search(r"demo\.mp4: (\d+) frames at 10 fps, ([\d.]+) s of calls", done.stdout)
    assert frames is not None, done.stdout
    assert "an opening card of 6 s, a closing card of 9 s, turns 1-2" in done.stdout
    assert int(frames[1]) == 60 + 6 + round(float(frames[2]) * 10) + 94
    assert out.read_bytes()[4:8] == b"ftyp"
    from PIL import Image

    sizes = {}
    for name in ("demo", "demo-dark", "demo-narrow", "demo-narrow-dark"):
        with Image.open(tmp_path / f"{name}.webp") as image:
            assert image.format == "WEBP" and image.is_animated, name
            assert image.n_frames == round(9.3 * 10), name  # turn 1, the slower call's
            sizes[name] = image.size
    assert sizes["demo"] == sizes["demo-dark"] == (320, 112)
    assert sizes["demo-narrow"] == sizes["demo-narrow-dark"] and sizes["demo-narrow"][0] == 130
