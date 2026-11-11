# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.5.4",
#     "matplotlib==3.11.2",
#     "pillow==13.0.0",
#     "fonttools==4.67.0",
#     "imageio-ffmpeg==0.6.0",
# ]
# ///
"""E2's demo video: the same caller on two engines, from two calls that ``flatshare-bench e2 run
--save-audio`` saved.

    uv run benchmarks/e2/demo.py BASELINE FEATURED --summary FRONTIER_CSV --out demo.mp4 \\
      --excerpt docs/figures/e2-demo

BASELINE and FEATURED are the run directories of two points: the engine a viewer compares against,
in the upper row, and the configuration the video is about, in the lower. Each side plays one of its
point's calls (``--caller``, by default the first), and both play the same turns: ``--turns`` of
them from ``--first-turn``, by default from the first turn the callers measured (after the warm-up
turns that open each call) and as many as fit in ``--seconds``. Turn k of both starts at the same
moment, so the same line is heard on both at once, and the call that is ahead waits in silence for
the other: only a caller's pause between turns is stretched, never a measured wait.

A frame is the README's chart of the two configurations with a turn of the call in place of its
first panel: a row a call, the baseline above, on that chart's columns. Beside a row's name: what
the call is doing, the wait's clock, which counts from the end of the caller's line to the reply's
first audio and then gives the V2V the caller measured, and that wait against the SLO. The row's
lane is the turn under way, read from left to right on one time axis for both rows: the caller's
line as a waveform, the wait as a flat bar in the row's fill, the voice agent's reply as a waveform,
each drawn as it plays, under one playhead. At the right, the agents' tokens/s over the calls' first
240 s, the README's measure: a bar a row on one scale, each with its share of the baseline's. Under
both, in the video, each reply's V2V: a pair of bars a turn, which grow while the caller waits.
``--summary`` is a report's frontier.csv (``flatshare-bench e2 report``): the voice SLO is the
report's, 250 ms above voice alone's median V2V p95 with as many callers, drawn as a red dashed rule
where it falls after the caller's line and across the replies' bars; and a closing card gives both
configurations (by their points' names) over all the report's runs, the median of their V2V p95 and
of the agents' tokens/s over the calls' first 240 s, as the README's chart draws them. ``--also``
adds another GPU's report of the same two points (its own frontier.csv): the closing card then gives
each GPU's as a group under its name, against its own SLO. An opening card names the calls and the
engines (``--heading`` is its heading).

The soundtrack is one call at a time, the caller and the agent mixed: the baseline's for the first
half of the turns and the featured configuration's for the rest, and a speaker before the words of
the row heard marks it. The excerpt (``--excerpt PREFIX``, a README's animation) is the same two
rows, silent and looping: ``--excerpt-turns`` turns from the first, without the cards, without the
frame's title and subtitle, which the README's caption gives, and without the replies' bars, as
animated lossless WebPs in each theme, each row beside its lane and, for a phone, each row over its
lane 420 units across (PREFIX.webp, PREFIX-dark.webp, PREFIX-narrow.webp, PREFIX-narrow-dark.webp).

Frames follow the docs' figures (``figure_style.py`` and ``fonts/`` in
python/flatshare_bench/src/flatshare_bench): STIX Two at the figures' four sizes, and the clock at
HERO, on a layout 1120 units wide; the palette of ``--theme``; flat marks on the page, with no frame
around a part and no fill behind one; the configuration to deploy in the saturated blue and the
baseline in gray, the voice agent's reply in green, the SLO in red. Each frame is drawn at twice its
size and halved by a box filter, so type and hairlines stay crisp. Motion follows the calls: the
playhead, the waveforms, the clock and the growing bars move linearly with their time. A text that
changes leaves faster than its successor arrives (ease-in-out, then ease-out from 0.96 of its size),
a V2V once given holds at least HOLD_S, a turn's drawing leaves before the next turn starts, and
nothing a viewer reads moves otherwise. The video is 1920 by 1080, H.264 (yuv420p, CRF 18, BT.709)
at 60 frames a second with AAC sound, encoded by the ffmpeg that imageio-ffmpeg bundles.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "flatshare_bench" / "src" / "flatshare_bench"
_spec = importlib.util.spec_from_file_location("demo_figure_style", PACKAGE / "figure_style.py")
assert _spec is not None and _spec.loader is not None
style = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = style  # its dataclasses look their module up
_spec.loader.exec_module(style)
_spec = importlib.util.spec_from_file_location("demo_labels", PACKAGE / "labels.py")
assert _spec is not None and _spec.loader is not None
labels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labels)  # the figures' names, and how a name reads inside a sentence

W = 1120.0
"""The layout's width in units, a README figure's; 1920 pixels make 1.71 of them a unit."""
FULL_H = 630.0
"""The video's layout height: 16 by 9 on its 1120 units."""
DPI = 100
LEAD_S = 0.5
"""How long before its line a turn starts on the video."""
FADE_S = 0.05
"""The soundtrack's crossfade from one call to the other, at a turn's start."""
ENTER_S, EXIT_S = 0.40, 0.24
"""How long a text takes to arrive and to leave: a change leaves faster than its successor arrives."""
TURN_EXIT_S = 0.32
"""A turn's drawing leaves over its last moments, before the next turn starts."""
CARD_ENTER_S, CARD_EXIT_S = 0.60, 0.36
"""How long a card, or the calls, take to arrive and to leave."""
TITLE_S, CLOSING_S = 6.0, 9.0
"""How long the opening card shows, its exit included, and the closing card, its arrival included."""
HOLD_S = 1.2
"""The least time a number stays on screen once it is final."""
REFERENCE = 0.4
"""The share of full scale at which a waveform fills its lane: one scale for both calls."""
SLO_MARGIN_MS = 250.0
"""The voice SLO: voice alone's V2V p95 plus this (docs/SLO.md)."""
BOX_GPU = "RTX 5060 Ti"
BOX_PROFILE = "rtx-5060-ti"
"""The frozen box's GPU and hardware profile: a record whose samples name no GPU ran on the box, as
every point did before the samples named it (e2/report.py's BOX_GPU)."""
NUMBERS = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")
BARS = ("bar-lo", "bar-hi")
"""Each row's fill: the baseline the chart's gray, the featured configuration the saturated blue,
as the README's chart has them."""


def cubic_bezier(x1: float, y1: float, x2: float, y2: float) -> Callable[[float], float]:
    """CSS's cubic-bezier(x1, y1, x2, y2): progress in [0, 1] to eased progress."""

    def coordinate(a: float, b: float, s: float) -> float:
        return 3 * a * (1 - s) ** 2 * s + 3 * b * (1 - s) * s**2 + s**3

    def ease(x: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        low, high = 0.0, 1.0
        for _ in range(40):  # x(s) increases with s, so bisection finds the s of x
            mid = (low + high) / 2
            low, high = (mid, high) if coordinate(x1, x2, mid) < x else (low, mid)
        return coordinate(y1, y2, (low + high) / 2)

    return ease


EASE_MOVE = cubic_bezier(0.77, 0.0, 0.175, 1.0)
"""Ease-in-out: a move, and a leave."""
EASE_ENTER = cubic_bezier(0.23, 1.0, 0.32, 1.0)
"""Ease-out: an arrival."""


def arriving(t: float, at: float, seconds: float = ENTER_S) -> tuple[float, float]:
    """The opacity and the size's scale of what arrives at ``at``: from nothing at 0.96 of its
    size, eased out over ``seconds``."""
    eased = EASE_ENTER(min(max((t - at) / seconds, 0.0), 1.0))
    return eased, 0.96 + 0.04 * eased


def leaving(t: float, by: float, seconds: float = EXIT_S) -> float:
    """The opacity of what has left by ``by``, eased in and out over the ``seconds`` before it."""
    return 1.0 - EASE_MOVE(min(max((t - (by - seconds)) / seconds, 0.0), 1.0))


# The records ------------------------------------------------------------------------------------


@dataclass
class Call:
    """A call that ``flatshare-voice bots --save-audio`` saved, and the point it was measured at."""

    label: str
    point: str
    """The point's name, by which the summary finds its configuration's runs."""
    pcm: np.ndarray
    """(samples, 2) of int16: what the caller sent, and the agent's audio as received."""
    rate: int
    wall: float
    """The WAV's start on the wall clock."""
    turns: list[dict[str, Any]]
    """The caller's turns as turns.jsonl has them, with ``at``: when their events happened, in
    seconds into the WAV."""
    callers: int
    sessions: int
    """The coding-agent sessions beside the calls."""
    gpu: str | None
    """The GPU the point ran on, as its samples name it in the docs' spelling (``shown``), or the
    box's where they name none on the box's profile; None off it."""
    tokens: float | None = None
    """The point's agents' tokens/s over the calls' first 240 s, as its summary.json gives them
    (the README's measure); None without them."""

    @classmethod
    def load(cls, run_dir: Path, caller: int = 0, label: str | None = None) -> Call:
        saved = json.loads((run_dir / "bots" / f"call-{caller}.json").read_text())
        with wave.open(str(run_dir / "bots" / f"call-{caller}.wav")) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).reshape(-1, 2)
        run = json.loads((run_dir / "run.json").read_text())
        arm = run["arm"]
        summary = run_dir / "summary.json"
        background = json.loads(summary.read_text()).get("background") if summary.exists() else None
        window = (background or {}).get("fixed_window") or {}
        return cls(
            label if label is not None else arm["label"],
            run.get("name", run_dir.name),
            pcm,
            saved["rate"],
            saved["start"]["wall"],
            saved["turns"],
            arm.get("voice", {}).get("callers", 1),
            arm.get("agents", {}).get("sessions", 0),
            gpu_name(run_dir / "gpu.csv")
            or (BOX_GPU if run.get("profile", BOX_PROFILE) == BOX_PROFILE else None),
            window.get("tokens_per_s"),
        )

    @property
    def warmups(self) -> int:
        """The warm-up turns that open the call, played but not measured."""
        return next((k for k, turn in enumerate(self.turns) if not turn.get("warmup")), 0)


def shown(gpu: str) -> str:
    """A GPU's name as nvidia-smi gives it, in the docs' spelling: without the vendor's prefixes,
    spaced, its memory in GB apart (NVIDIA A100-SXM4-80GB as A100 SXM4 80 GB)."""
    name = gpu.removeprefix("NVIDIA ").removeprefix("GeForce ").replace("-", " ")
    return re.sub(r"(\d)GB\b", r"\1 GB", name)


def gpu_name(samples: Path) -> str | None:
    """The GPU campaign.sh's samples name in their last field (from 2026-10-25 on), ``shown``;
    None without the samples or where they name none."""
    if not samples.exists():
        return None
    for line in samples.read_text().splitlines():
        last = line.rsplit(",", 1)[-1].strip()
        if last.startswith("NVIDIA"):
            return shown(last)
    return None


@dataclass(frozen=True)
class Wait:
    """A wait of a shown turn, on the video's clock: from the end of the caller's speech to the
    agent's reply."""

    turn: int
    """The shown turn's index."""
    ended: float
    until: float
    """When the reply came; for a reply that never came, the turn's end."""
    ms: float | None
    """The V2V the caller measured; None for a reply that never came."""
    again: bool = False
    """The reply to an interruption, after the turn's first reply."""

    @property
    def seconds(self) -> float:
        return self.ms / 1000 if self.ms is not None else self.until - self.ended


@dataclass
class Timeline:
    """Where the turns both calls play are on the video: turn k of both starts at ``offsets[k]``
    and lasts as long as the longer of the two; the call that is ahead waits in silence."""

    calls: tuple[Call, Call]
    first: int
    """The first turn shown, by its number in the calls."""
    spans: tuple[list[tuple[float, float]], list[tuple[float, float]]]
    """Each shown turn's stretch of each call, in seconds into its WAV: from LEAD_S before its
    line to LEAD_S before the next one's, or to the end of the call."""
    offsets: list[float]
    """When each shown turn starts on the video, then when the calls end."""

    @classmethod
    def choose(
        cls,
        calls: tuple[Call, Call],
        first: int | None = None,
        turns: int | None = None,
        seconds: float = 180,
    ) -> Timeline:
        """``turns`` turns of both calls from turn ``first`` (by default the first measured one),
        by default as many as fit in ``seconds``."""
        if calls[0].rate != calls[1].rate:
            raise ValueError("the two calls are saved at different rates")
        if first is None:
            first = max(call.warmups for call in calls)
        spans = (_spans(calls[0], first), _spans(calls[1], first))
        lengths = [max(b - a for a, b in pair) for pair in zip(*spans, strict=False)]
        offsets = [0.0, *np.cumsum(lengths).tolist()]
        if turns is None:
            turns = max(1, bisect.bisect_right(offsets, seconds) - 1)
        if not 1 <= turns <= len(lengths):
            raise ValueError(f"both calls have {len(lengths)} turns from turn {first}, not {turns}")
        return cls(calls, first, (spans[0][:turns], spans[1][:turns]), offsets[: turns + 1])

    @property
    def seconds(self) -> float:
        return self.offsets[-1]

    @property
    def turns(self) -> int:
        return len(self.offsets) - 1

    def turn_at(self, t: float) -> int:
        """The shown turn under way at video time t."""
        return min(max(bisect.bisect_right(self.offsets, t) - 1, 0), self.turns - 1)

    def video_time(self, side: int, k: int, t: float) -> float:
        """A moment of a side's shown turn k, given in seconds into its WAV, on the video."""
        return self.offsets[k] + t - self.spans[side][k][0]

    def track(self, side: int) -> np.ndarray:
        """A side's audio on the video, both channels, silent where it waits."""
        call = self.calls[side]
        out = np.zeros((round(self.seconds * call.rate), 2), dtype=np.int16)
        for (start, end), offset in zip(self.spans[side], self.offsets, strict=False):
            a, b, at = (round(x * call.rate) for x in (start, end, offset))
            out[at : at + b - a] = call.pcm[a:b][: len(out) - at]
        return out

    def waits(self, side: int) -> list[Wait]:
        """Each wait of a side's shown turns, in order, as the caller measured it: the line's, and
        an interruption's after it."""
        found = []
        shown = self.calls[side].turns[self.first : self.first + self.turns]
        for k, turn in enumerate(shown):
            at = turn["at"]
            pairs = [("speech_end", "reply", "v2v_ms", False)]
            if turn.get("barge_in"):
                pairs.append(
                    ("barge_in_speech_end", "barge_in_reply", "v2v_after_barge_in_ms", True)
                )
            for ended, reply, measured, again in pairs:
                if ended not in at:
                    continue
                until = at.get(reply, self.spans[side][k][1])
                ms = turn.get(measured) if reply in at else None
                found.append(
                    Wait(
                        k,
                        self.video_time(side, k, at[ended]),
                        self.video_time(side, k, until),
                        ms,
                        again,
                    )
                )
        return found

    def interruptions(self, side: int) -> list[tuple[int, float]]:
        """When on the video the caller interrupts a reply: each shown turn's barge-in."""
        shown = self.calls[side].turns[self.first : self.first + self.turns]
        return [
            (k, self.video_time(side, k, turn["at"]["barge_in"]))
            for k, turn in enumerate(shown)
            if turn.get("barge_in") and "barge_in" in turn["at"]
        ]

    @property
    def switch(self) -> int:
        """The shown turn at whose start the soundtrack turns from the baseline to the featured
        call: the start nearest the middle (the calls' end, with a single turn)."""
        if self.turns == 1:
            return 1
        middle = self.seconds / 2
        return min(range(1, self.turns), key=lambda k: abs(self.offsets[k] - middle))


def _spans(call: Call, first: int) -> list[tuple[float, float]]:
    starts = [max(0.0, turn["at"]["start"] - LEAD_S) for turn in call.turns[first:]]
    return list(zip(starts, [*starts[1:], len(call.pcm) / call.rate], strict=True))


def soundtrack(timeline: Timeline) -> np.ndarray:
    """One call at a time, the caller and the agent mixed: the baseline's turns before the
    switch (Timeline.switch), the featured call's from it, crossfaded in FADE_S around its start,
    where both are quiet."""
    mixes = [timeline.track(side).astype(np.float64).sum(axis=1) * 0.7 for side in (0, 1)]
    rate = timeline.calls[0].rate
    at = timeline.offsets[timeline.switch] * rate
    fade = np.clip((np.arange(len(mixes[0])) - at) / (FADE_S * rate) + 0.5, 0.0, 1.0)
    return np.clip(mixes[0] * (1 - fade) + mixes[1] * fade, -32768, 32767).astype(np.int16)


def envelope(signal: np.ndarray, rate: int) -> np.ndarray:
    """A channel's peak in each 10 ms, as a share of full scale."""
    bins = rate // 100
    peaks = np.abs(signal[: len(signal) // bins * bins].astype(np.int32)).reshape(-1, bins)
    return peaks.max(axis=1) / 32768


@dataclass
class Summary:
    """What a report's frontier.csv says of the two sides' configurations, over all its runs."""

    slo_ms: float | None
    """250 ms above voice alone's V2V p95 with as many callers, as the report reads its SLO."""
    p95_ms: tuple[float, float]
    """Each side's configuration's median V2V p95 over its runs."""
    runs: tuple[int, int]
    tokens: tuple[float | None, float | None]
    """Each one's median of the agents' tokens/s over the calls' first 240 s (None unmeasured)."""
    gpu: str | None = None
    """The GPU the workload names, ``shown``; None on the box, whose workloads name none."""

    @classmethod
    def read(cls, path: Path, points: Sequence[str]) -> Summary:
        """The configurations of ``points`` (each a point's name) and their workload's SLO. A
        configuration the file lacks, or measured in two workloads, is a ValueError."""
        with path.open() as f:
            rows = list(csv.DictReader(f))
        found = []
        for point in points:
            runs = [r for r in rows if r["point"] == point]
            if not runs or len({r["workload"] for r in runs}) > 1:
                raise ValueError(f"{path}: {len(runs)} runs of {point}, not one workload's")
            found.append(runs)

        def median(runs: list[dict[str, str]], column: str) -> float | None:
            values = [float(r[column]) for r in runs if r[column]]
            return statistics.median(values) if values else None

        p95 = [median(runs, "v2v_p95_ms") for runs in found]
        if p95[0] is None or p95[1] is None:
            raise ValueError(f"{path}: a configuration of {list(points)} without a V2V p95")
        tokens = [median(runs, "fixed_window_tokens_per_s") for runs in found]
        # The workload: the corpus, the callers, then the GPU and the model where they are not the
        # box's (a model's name has its organization's slash), then any warm-up turns.
        named = [
            part
            for part in found[0][0]["workload"].split(", ")[2:]
            if "/" not in part and "warm-up turn" not in part
        ]
        return cls(
            _slo(rows, found[0][0]),
            (p95[0], p95[1]),
            (len(found[0]), len(found[1])),
            (tokens[0], tokens[1]),
            shown(named[0]) if named else None,
        )


def _slo(rows: list[dict[str, str]], row: dict[str, str]) -> float | None:
    """The SLO of ``row``'s workload, as the report reads it: voice alone (arm 1) in the same
    workload but for its corpus, which voice alone replays none of; the median over its
    configurations of their median V2V p95 over their runs, plus SLO_MARGIN_MS."""
    rest = row["workload"].split(", ", 1)[1]
    alone: dict[str, list[float]] = {}
    for r in rows:
        if r["arm"] == "1" and r["workload"].split(", ", 1)[1] == rest and r["v2v_p95_ms"]:
            alone.setdefault(r["point"], []).append(float(r["v2v_p95_ms"]))
    if not alone:
        return None
    return statistics.median(statistics.median(v) for v in alone.values()) + SLO_MARGIN_MS


def scale(top: float, most: int = 4) -> list[float]:
    """Round marks from zero that reach ``top``: the finest step of 1, 2, 4 or 5 of a power of
    ten that needs at most ``most`` of them past zero."""
    magnitude = 10 ** math.floor(math.log10(max(top, 1e-9)))
    for factor in (0.1, 0.2, 0.4, 0.5, 1, 2, 4, 5, 10):
        step = factor * magnitude
        count = math.ceil(top / step - 1e-9)
        if count <= most:
            return [round(step * k, 6) for k in range(count + 1)]
    return [0.0, 10 * magnitude]


def ticks(top: float, length: float) -> list[float]:
    """Round ticks from zero that reach ``top``, at most seven, on a horizontal axis ``length``
    long: the finest step whose labels keep a clear gap between them, as the figures choose them
    (e2/figures.py's ticks)."""
    magnitude = 10 ** math.floor(math.log10(max(top, 1e-9)))
    for factor in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20):
        step = factor * magnitude
        count = math.ceil(top / step - 1e-9)
        widest = max(style.measure(mark_text(step * k, step), style.NOTE) for k in range(count + 1))
        if count <= 6 and length / count >= widest + 16:
            return [step * k for k in range(count + 1)]
    return [0.0, 20 * magnitude * math.ceil(top / (20 * magnitude))]


def mark_text(value: float, step: float) -> str:
    """A mark's value with as many decimals as its step needs, the same for every mark."""
    digits = next(d for d in range(4) if abs(step * 10**d - round(step * 10**d)) < 1e-9)
    return f"{value:,.{digits}f}"


# Drawing ---------------------------------------------------------------------------------------

Box = tuple[float, float, float, float]
"""A text's room, (x0, y0, x1, y1) in units: what it must stay inside."""

HERO = 44.0
"""The size of a row's clock, the number each row builds to: about twice a label's."""
BAR = 24.0
"""A bar's thickness: the figures' (report/figures.py's BAR)."""
LABELS_END, PLOT_START, PLOT_END = 264.0, 280.0, 728.0
"""Where the rows' names end, and where the rows' first panel starts and ends: the columns of the
README's chart, which the animation stands above (report/figures.py's LABELS_END and PLOT_START,
and its headline's first scale)."""
AGENTS_START, AGENTS_END = 818.0, 998.0
"""The scale of the agents' tokens/s: that chart's second panel's."""
VALUE_GAP, SHARE_GAP = 9.0, 8.0
"""From a bar's end to its value, and from the value to its share: the figures'."""
ROW_PITCH = 56.0
"""From a row to the next on the closing card: that chart's (HEADLINE_PITCH)."""
SLO_DASH = (5.0, 4.0)
"""The dashes of the SLO's rule, on and off: the figures' (figure_style's .slo)."""
SLO_RISE = 8.0
"""How far the SLO's rule rises over its plot, and its name's baseline over the rule's top: the
figures' (report/figures.py's _v_threshold)."""
HALO = 6.0
"""A halo's width, in the page's color behind a text that a rule may cross: the figures'."""
KEY_MARK = 22.0
"""A key's mark's length, before its name."""
KEY_WAVE = np.array([1.5, 4.0, 2.5, 6.5, 3.5, 7.0, 4.0, 5.5, 2.0, 6.0, 7.0, 3.0, 5.0, 2.5, 1.5])
"""The key's waveform: its heights about its middle, in units."""
KEY_SWATCH = ((KEY_MARK - style.SEAM) / 2, 2 * float(KEY_WAVE.max()))
"""A key's swatch of a row's fill, across and down: two with the seam between them make a key's
mark, as tall as its waveform."""
WAVES = ("bar-mid", "green-m")
"""A lane's waveforms' fills: the caller's line the chart's second gray, which stands apart from
the baseline's wait after it, and the reply the green of a voice turn's marks."""
SPEAKER_W, SPEAKER_GAP = 18.0, 7.0
"""The width of the speaker that marks the call heard, and its distance before the row's words."""
VALUE_LIFT = 6.0
"""A value's baseline over the top of its bar in the video's plot, or over the SLO's rule there."""
PAIR_PITCH = 3 * BAR
"""The most a turn's pair of bars takes of the video's plot, where the turns are few."""


class Board:
    """A matplotlib canvas in a layout's units (``across`` by ``down``, y down), ``width`` pixels
    wide once halved and drawn at twice that, in a theme's palette, its text in STIX Two Text
    (Regular, or SemiBold for ``bold``). It keeps each text it draws with the room the text must
    stay inside, for :meth:`check`. The text is drawn unhinted: a text is then as wide as the
    fonts' metrics say at every size, and a layout, which places by those metrics, holds at a
    low resolution as it does at the video's."""

    def __init__(self, width: int, down: float, theme: str = "light", *, across: float = W) -> None:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.font_manager import FontProperties

        if width % 2:
            raise ValueError(f"a frame's width is an even number of pixels, not {width}")
        matplotlib.rcParams["text.hinting"] = "no_hinting"
        self.width, self.across = width, across
        self.height = round(width * down / across / 2) * 2
        self.scale = 2 * width / across
        """Drawn pixels per unit."""
        self.points = self.scale * 72 / DPI
        """Points per unit, at the drawn size."""
        self.units = 2 * self.height / self.scale
        self.colors = style.LIGHT if theme == "light" else style.DARK
        self.figure = Figure(
            figsize=(2 * width / DPI, 2 * self.height / DPI), dpi=DPI, facecolor=self.colors["page"]
        )
        self.canvas = FigureCanvasAgg(self.figure)
        self.ax = self.figure.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, across)
        self.ax.set_ylim(self.units, 0)
        self.ax.set_axis_off()
        faces = ((False, "Regular", "regular"), (True, "SemiBold", "bold"))
        self.fonts = {
            bold: FontProperties(fname=str(style.FONT_DIR / f"STIXTwoText-{file}.ttf"))
            for bold, file, _ in faces
        }
        self.glyphs = {bold: style._advances(face)[1] for bold, _, face in faces}
        self.bold: set[int] = set()
        """The texts set in the SemiBold face, by their ids."""
        self.texts: list[tuple[Any, Box]] = []

    def color(self, name: str) -> str:
        return self.colors[name]

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = style.LABEL,
        ink: str = "ink",
        *,
        ha: str = "left",
        room: Box | None = None,
        halo: bool = False,
        bold: bool = False,
    ) -> Any:
        """A text at baseline ``y``, which must stay inside ``room`` (the canvas by default);
        ``bold`` in the SemiBold face, as the figures set a panel's title and the featured row's
        name and value; with a ``halo`` of the page's color behind it, as they set a value that a
        rule may cross (HALO wide, half of it each side of a stroke)."""
        artist = self.ax.text(
            x,
            y,
            "",
            fontproperties=self.fonts[bold],
            fontsize=size * self.points,
            color=self.colors[ink],
            ha=ha,
            va="baseline",
            zorder=3,
        )
        if halo:
            from matplotlib import patheffects

            stroke = patheffects.withStroke(linewidth=self.lw(HALO), foreground=self.colors["page"])
            artist.set_path_effects([stroke])
        if bold:
            self.bold.add(id(artist))
        self.set_text(artist, text)
        self.texts.append((artist, room or (0.0, 0.0, self.across, self.units)))
        return artist

    def set_text(self, artist: Any, text: str) -> None:
        """Set a text, refusing a character its face has no glyph for."""
        missing = {c for c in text if ord(c) not in self.glyphs[id(artist) in self.bold]}
        if missing:
            raise ValueError(f"no glyph in STIX Two Text for {sorted(missing)} in {text!r}")
        artist.set_text(text)

    def width_of(self, text: str, size: float, *, bold: bool = False) -> float:
        return style.measure(text, size, bold=bold)

    def lw(self, units: float) -> float:
        return units * self.points

    def line(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        color: str,
        width: float = style.THIN,
        *,
        dashes: tuple[float, float] | None = None,
    ) -> Any:
        """A line ``width`` units wide; ``dashes`` (on, off) in units. Gridlines and hairlines lie
        behind bars, every other line over them."""
        behind = color in ("grid", "rule")
        (artist,) = self.ax.plot(
            xs, ys, color=self.colors[color], linewidth=self.lw(width), zorder=0.5 if behind else 2
        )
        artist.set_solid_capstyle("butt")
        artist.set_dash_capstyle("butt")
        if dashes:
            artist.set_dashes((dashes[0] / width, dashes[1] / width))
        return artist

    def patch(self, vertices: np.ndarray, fill: str) -> Any:
        """A flat polygon in ``fill``, without an outline, as the figures draw a quantity."""
        from matplotlib.patches import Polygon

        artist = Polygon(vertices, closed=True, facecolor=self.colors[fill], edgecolor="none")
        artist.set_linewidth(0)
        self.ax.add_patch(artist)
        return artist

    def bar(self, fill: str) -> Any:
        """A flat bar in ``fill``, its shape set later by :func:`bar_shape`."""
        return self.patch(np.zeros((4, 2)), fill)

    def wave(self, fill: str) -> Any:
        """A waveform, a flat shape in ``fill`` without an outline as every mark here is; its
        shape set later by :func:`wave_path`."""
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as Outline

        artist = PathPatch(Outline(np.zeros((2, 2))), facecolor=self.colors[fill], edgecolor="none")
        artist.set_linewidth(0)
        self.ax.add_patch(artist)
        return artist

    def halve(self, rgba: np.ndarray) -> np.ndarray:
        """The drawn pixels at half their width and height, each the mean of its four: RGB."""
        from PIL import Image

        size = (2 * self.width, 2 * self.height)
        image = Image.frombuffer("RGBA", size, rgba, "raw", "RGBA", 0, 1)
        return np.asarray(image.convert("RGB").reduce(2))

    def pixels(self) -> np.ndarray:
        """The canvas drawn in full and halved."""
        self.canvas.draw()
        return self.halve(np.asarray(self.canvas.buffer_rgba()))

    def check(self) -> None:
        """Refuse a shown text that leaves its room or overlaps another shown text, as the
        figures' checks do, at the canvas's last full drawing."""
        renderer = self.canvas.get_renderer()
        shown: list[tuple[str, Box]] = []
        for artist, (x0, y0, x1, y1) in self.texts:
            alpha = artist.get_alpha()
            hidden = alpha is not None and alpha <= 0.02
            if not artist.get_visible() or not artist.get_text() or hidden:
                continue
            extent = artist.get_window_extent(renderer)
            box = (
                extent.x0 / self.scale,
                self.units - extent.y1 / self.scale,
                extent.x1 / self.scale,
                self.units - extent.y0 / self.scale,
            )
            text = artist.get_text()
            if box[0] < x0 - 0.5 or box[1] < y0 - 0.5 or box[2] > x1 + 0.5 or box[3] > y1 + 0.5:
                rounded = tuple(round(v, 1) for v in box)
                raise ValueError(f"{text!r} leaves its room: {rounded} not in {(x0, y0, x1, y1)}")
            for other, seen in shown:
                apart = (
                    box[2] + 1 <= seen[0]
                    or seen[2] + 1 <= box[0]
                    or box[3] + 1 <= seen[1]
                    or seen[3] + 1 <= box[1]
                )
                if not apart:
                    raise ValueError(f"{text!r} overlaps {other!r}")
            shown.append((text, box))


def bar_shape(
    x: float, y: float, w: float, h: float, end: str, r: float = style.BAR_CORNER
) -> np.ndarray:
    """The outline of a flat bar at (x, y), ``w`` by ``h``, the corners of its free ``end`` ("top"
    or "right") rounded as the figures round them (quadratic corners of radius ``r``), as a
    polygon."""
    r = max(0.0, min(r, w / 2, h / 2))
    s = np.linspace(0.0, 1.0, 7)[:, None]

    def corner(
        a: tuple[float, float], c: tuple[float, float], b: tuple[float, float]
    ) -> np.ndarray:
        pa, pc, pb = (np.asarray(p, dtype=float) for p in (a, c, b))
        return (1 - s) ** 2 * pa + 2 * (1 - s) * s * pc + s**2 * pb

    x1, y1 = x + w, y + h
    if end == "top":
        return np.vstack(
            [
                [(x, y1)],
                corner((x, y + r), (x, y), (x + r, y)),
                corner((x1 - r, y), (x1, y), (x1, y + r)),
                [(x1, y1)],
            ]
        )
    return np.vstack(
        [
            [(x, y)],
            corner((x1 - r, y), (x1, y), (x1, y + r)),
            corner((x1, y1 - r), (x1, y1), (x1 - r, y1)),
            [(x, y1)],
        ]
    )


def wave_path(xs: np.ndarray, heights: np.ndarray, mid: float, least: float = 0.75) -> Any:
    """A waveform's outline, symmetric about the line ``mid``: ``heights`` above it and below it
    at ``xs``, a closed shape for each stretch higher than ``least`` units, so that silence draws
    nothing over what the lane holds beside it. None where all of it is silent."""
    from matplotlib.path import Path as Outline

    loud = heights >= least
    if not loud.any():
        return None
    edges = np.flatnonzero(np.diff(np.concatenate([[False], loud, [False]]).astype(np.int8)))
    vertices, codes = [], []
    for a, b in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
        x, h = xs[a:b], heights[a:b]
        # A stretch starts and ends on the middle line, half a step out where the wave goes on.
        left = (xs[a - 1] + x[0]) / 2 if a else x[0]
        right = (x[-1] + xs[b]) / 2 if b < len(xs) else x[-1]
        ring = np.vstack(
            [
                [(left, mid)],
                np.column_stack([x, mid - h]),
                [(right, mid)],
                np.column_stack([x[::-1], mid + h[::-1]]),
                [(left, mid)],
            ]
        )
        vertices.append(ring)
        codes += [Outline.MOVETO, *[Outline.LINETO] * (len(ring) - 2), Outline.CLOSEPOLY]
    return Outline(np.vstack(vertices), codes)


@dataclass
class Segment:
    """A text's content from ``start`` until the next segment's: fixed, or a function of the
    video's time (the clock while it counts). ``enter`` is whether it arrives as a change, or
    continues the one before it, as the clock's final value continues its count."""

    start: float
    text: str | Callable[[float], str]
    enter: bool = True


class Slot:
    """A text whose content changes at moments of the video: each segment arrives at its start
    and leaves, faster, before the next one's (ENTER_S, EXIT_S). In a loop, the last segment also
    leaves before the end, and the first arrives at the start, unless the two read the same."""

    def __init__(
        self, board: Board, artist: Any, size: float, segments: list[Segment], end: float | None
    ) -> None:
        self.board, self.artist, self.size = board, artist, size
        self.segments = sorted(segments, key=lambda s: s.start)
        self.starts = [s.start for s in self.segments]
        first, last = self.segments[0], self.segments[-1]
        self.loop_end = end if end is not None and first.text != last.text else None

    def show(self, t: float) -> None:
        k = bisect.bisect_right(self.starts, t) - 1
        if k < 0:
            self.artist.set_alpha(0.0)
            return
        segment = self.segments[k]
        text = segment.text(t) if callable(segment.text) else segment.text
        alpha, size = 1.0, 1.0
        if segment.enter and (segment.start > 0 or self.loop_end is not None):
            alpha, size = arriving(t, segment.start)
        if k + 1 < len(self.segments):
            if self.segments[k + 1].enter:
                alpha *= leaving(t, self.segments[k + 1].start)
        elif self.loop_end is not None:
            alpha *= leaving(t, self.loop_end)
        self.board.set_text(self.artist, text)
        self.artist.set_alpha(alpha)
        self.artist.set_fontsize(self.size * self.board.points * size)


def headings(timeline: Timeline) -> tuple[str, str]:
    """The video's title and subtitle, from the records."""
    calls = timeline.calls
    gpu = calls[0].gpu or calls[1].gpu
    callers = calls[0].callers
    title = f"The same caller on two engines, beside {calls[0].sessions} coding-agent sessions"
    when = datetime.fromtimestamp(calls[0].wall, UTC).strftime("%Y-%m-%d")
    where = f" on one {gpu}" if gpu else ""
    who = f"One of {_counted(callers).lower()} callers" if callers > 1 else "The only caller"
    subtitle = f"{who}{where}, recorded {when} \N{BULLET} the same line on both engines at once"
    return title, subtitle


def _counting(ended: float) -> Callable[[float], str]:
    """The clock while the caller waits: the seconds since the speech ended, to a tenth."""

    def text(t: float) -> str:
        return f"{max(t - ended, 0.0):.1f} s"

    return text


def _time_step(seconds: float) -> int:
    """The time axis's step: the least of 1, 2, 5 and 10 s that marks it at most six times."""
    return next((s for s in (1, 2, 5, 10, 20, 30, 60) if seconds / s <= 6), 60)


# The board --------------------------------------------------------------------------------------


@dataclass
class Row:
    """Where a configuration's row goes, in units."""

    name: tuple[float, float]
    status: tuple[float, float]
    clock: tuple[float, float]
    verdict: tuple[float, float]
    """Where the row's name and its readout start or end (by ``ha``), each with its baseline: what
    the call is doing, the wait's clock, and the wait against the SLO."""
    ha: str
    room: Box
    """The readout's room."""
    lane: tuple[float, float]
    """The lane's top and bottom: its parts lie about its middle line."""
    agents: float
    """The middle line of the row's bar of the agents' tokens/s."""
    agents_name: float | None = None
    """The baseline of the row's name over that bar, where the bar is not beside the lane."""
    sound: bool = False
    """Whether the row marks, before its words, that the soundtrack is its call's: the video's
    rows do, not the README's."""


@dataclass
class Layout:
    """Where the board's parts go, in units: the two configurations as rows on the columns of the
    README's chart, each row's lane beside its name (the wide form and the video's, 1120 across),
    or one row above the other and the agents' tokens/s under both (the phone form, 420 across)."""

    across: float
    down: float
    title: list[tuple[float, str]]
    """The title's lines, centred, at their baselines; then the subtitle's."""
    subtitle: list[tuple[float, str]]
    rows: tuple[Row, Row]
    span: tuple[float, float]
    """Where the turn's time axis starts and ends: one axis for both lanes."""
    heads: tuple[float, float]
    """Baselines: the first panel's title, and the key under it."""
    slo: float
    """The baseline of the SLO's name, over the upper lane."""
    joined: bool
    """Whether a gridline, the SLO's rule and the playhead each run through both lanes as one
    line: they do where nothing stands between the lanes."""
    ticks: float
    axis_name: float
    """Baselines: the time axis's marks and its name."""
    agents: tuple[float, float]
    """The scale of the agents' tokens/s: where it starts and ends."""
    agents_heads: tuple[float, float]
    """Baselines: that panel's title, and the note under it."""
    history: Box | None = None
    """The plot of each reply's V2V, a pair of bars a turn: its left, top, right and base. The
    video's."""
    history_heads: tuple[float, float] = (0.0, 0.0)
    """Baselines: that plot's title, and its turns' numbers."""


def _beside(
    lane: tuple[float, float], lines: tuple[float, float, float, float], sound: bool = False
) -> Row:
    """A row whose name and readout stand left of its lane and end where the chart's names end:
    ``lines`` are their baselines from the lane's top."""
    top, bottom = lane
    name, status, clock, verdict = (top + line for line in lines)
    return Row(
        name=(LABELS_END, name),
        status=(LABELS_END, status),
        clock=(LABELS_END, clock),
        verdict=(LABELS_END, verdict),
        ha="right",
        room=(8.0, name + 6, LABELS_END + 8, bottom + 6),
        lane=lane,
        agents=(top + bottom) / 2,
        sound=sound,
    )


def wide_layout(timeline: Timeline) -> Layout:
    """The README's animation, 1120 by 412 units: the two rows on the columns of the chart it
    stands above. It draws no title or subtitle, which the README's caption gives, as no figure in
    the docs does."""
    lines = (15.0, 38.0, 82.0, 109.0)
    return Layout(
        across=W,
        down=412.0,
        title=[],
        subtitle=[],
        rows=(_beside((104.0, 216.0), lines), _beside((236.0, 348.0), lines)),
        span=(PLOT_START, PLOT_END),
        heads=(30.0, 55.0),
        slo=88.0,
        joined=True,
        ticks=371.0,
        axis_name=398.0,
        agents=(AGENTS_START, AGENTS_END),
        agents_heads=(30.0, 55.0),
    )


def video_layout(timeline: Timeline) -> Layout:
    """The video's frame, 1120 by 630 units (16 by 9): the README's rows under a title, each
    marking when the soundtrack is its call's, and under both panels each reply's V2V, a pair of
    bars a turn, from the lanes' start to the agents' scale's end."""
    lines = (13.0, 35.0, 76.0, 102.0)
    title, subtitle = headings(timeline)
    return Layout(
        across=W,
        down=FULL_H,
        title=[(44.0, title)],
        subtitle=[(70.0, subtitle)],
        rows=(_beside((174.0, 278.0), lines, True), _beside((292.0, 396.0), lines, True)),
        span=(PLOT_START, PLOT_END),
        heads=(106.0, 130.0),
        slo=158.0,
        joined=True,
        ticks=419.0,
        axis_name=441.0,
        agents=(AGENTS_START, AGENTS_END),
        agents_heads=(106.0, 130.0),
        history=(PLOT_START, 500.0, AGENTS_END, 582.0),
        history_heads=(472.0, 608.0),
    )


def narrow_layout(timeline: Timeline) -> Layout:
    """The README's phone layout, 420 units across like the figures' phone layouts: each row is
    its name, its readout (the clock, and the words beside it) and its lane across the width; one
    time axis under the lower lane; then the agents' tokens/s, a bar under each row's name. No
    title or subtitle, as the wide form."""
    x0, x1 = 18.0, 402.0
    # The words stand right of the widest clock either row shows, so that neither row's moves.
    clocks = (
        style.measure(f"{wait.seconds:.1f} s", HERO, bold=side == 1)
        for side in (0, 1)
        for wait in timeline.waits(side)
    )
    words = x0 + max(clocks, default=0.0) + 14

    def row(name: float, lane: float, agents: float) -> Row:
        return Row(
            name=(x0, name),
            status=(words, name + 28),
            clock=(x0, name + 48),
            verdict=(words, name + 50),
            ha="left",
            room=(x0 - 6, name + 6, x1 + 6, lane - 2),
            lane=(lane, lane + 96),
            agents=agents + 26,
            agents_name=agents,
        )

    return Layout(
        across=420.0,
        down=770.0,
        title=[],
        subtitle=[],
        rows=(row(98.0, 192.0, 646.0), row(324.0, 394.0, 716.0)),
        span=(x0, x1),
        heads=(30.0, 56.0),
        slo=176.0,
        joined=False,
        ticks=513.0,
        axis_name=538.0,
        agents=(x0, 300.0),
        agents_heads=(586.0, 611.0),
    )


FORMS = {"wide": wide_layout, "narrow": narrow_layout, "video": video_layout}
"""The board's layouts: the README's animation, its phone layout, and the video's frame."""


@dataclass
class Side:
    """A configuration's row: what changes in it."""

    status: Any
    """What the call is doing: the caller speaks, waits, or was answered."""
    clock: Any
    """The wait so far, then the V2V the caller measured."""
    waves: list[Any]
    """The caller's line and the voice agent's reply, each about the lane's middle line."""
    bands: list[Any]
    """The waits of the turn, one bar each in the lane: the line's, and an interruption's."""
    bars: list[tuple[Wait, Any, Any, float]] = field(default_factory=list)
    """Each wait in the video's plot: its bar, its value, and the bar's left."""
    speaker: list[tuple[Any, np.ndarray, np.ndarray]] = field(default_factory=list)
    """The video's mark that the soundtrack is this call's: a speaker's parts, each with its
    outline from the speaker's left and middle."""
    verdicts: tuple[Any, Any] = (None, None)
    """The wait against the SLO: over it, as soon as the clock passes it; within it, at the
    reply."""
    agents: list[Any] = field(default_factory=list)
    """The agents' tokens/s: the bar, its value, and its share of the baseline's."""


class Dashboard:
    """The two calls as two rows on one time axis, ``width`` pixels wide, as RGB pixels: the
    README's chart with a turn of the call in place of its first panel. A row is a
    configuration: its name, what its call is doing, the wait's clock and the wait against the
    SLO; then its lane, the turn under way read from left to right (the caller's line as a
    waveform, the wait as a bar in the row's fill, the voice agent's reply as a waveform), a
    playhead through both lanes and the SLO as a rule where it falls after the line's end; then
    the agents' tokens/s over the calls' first 240 s as a bar on one scale. Each turn's drawing
    leaves before the next turn starts. ``form`` is one of FORMS: the README's animations loop,
    their last turn's drawing leaving before the end; the video's frame holds its last turn,
    marks the call heard, and keeps each reply's V2V under the lanes, a pair of bars a turn that
    grow while the caller waits."""

    def __init__(
        self,
        timeline: Timeline,
        summary: Summary | None,
        *,
        width: int = 2 * 840,
        theme: str = "light",
        form: str = "wide",
    ) -> None:
        self.timeline, self.summary = timeline, summary
        self.slo = summary.slo_ms / 1000 if summary and summary.slo_ms else None
        self.layout = FORMS[form](timeline)
        self.loop = form != "video"
        lay = self.layout
        self.board = Board(width, lay.down, theme, across=lay.across)
        self.waits = [timeline.waits(side) for side in (0, 1)]
        rate = timeline.calls[0].rate
        self.envelopes = [
            [envelope(timeline.track(side)[:, channel], rate) for channel in (0, 1)]
            for side in (0, 1)
        ]
        lengths = [b - a for a, b in zip(timeline.offsets, timeline.offsets[1:], strict=False)]
        step = _time_step(max(lengths))
        self.axis_s = float(step * math.ceil(max(lengths) / step))
        """Where the time axis ends: the first of its marks that holds the longest turn."""
        # A wait starts where the caller's line ended by the caller's own clock: what the channel
        # still carries after that is not drawn, so that the wait's bar starts on a clean seam.
        for side, waits in enumerate(self.waits):
            for wait in waits:
                self.envelopes[side][0][round(wait.ended * 100) : round(wait.until * 100)] = 0.0
        self.stride = max(1, round(100 * self.axis_s / (lay.span[1] - lay.span[0])))
        """How many of an envelope's 10 ms make a waveform's point: about a unit of the axis."""
        values = [w.seconds for side in self.waits for w in side]
        self.marks = scale(max([*values, self.slo or 0.0, 1.0]), most=3)
        self.shapes: list[Any] = []
        self.words: list[Any] = []
        """What changes, in the order it is drawn: the marks, then the texts over them."""
        self.slots: list[Slot] = []
        self._frame()
        self.sides = [self._row(side) for side in (0, 1)]
        if lay.history is not None:
            self._history(lay.history)
        self._limits()
        self.playheads = [
            self.board.line([lay.span[0]] * 2, list(span), "ink", style.THIN)
            for span in self._spans()
        ]
        self.shapes += self.playheads
        self.dynamic = [*self.shapes, *self.words]
        for artist in self.dynamic:
            artist.set_animated(True)
        self.board.canvas.draw()
        self.background = self.board.canvas.copy_from_bbox(self.board.figure.bbox)

    # Layout ---------------------------------------------------------------------------------

    def x_of(self, seconds: float) -> float:
        """A moment of the turn, in seconds from its start, on the time axis."""
        a, b = self.layout.span
        return a + (b - a) * min(max(seconds, 0.0), self.axis_s) / self.axis_s

    def _spans(self) -> list[tuple[float, float]]:
        """What a line through the lanes covers: both as one, or each lane."""
        upper, lower = self.layout.rows
        if self.layout.joined:
            return [(upper.lane[0], lower.lane[1])]
        return [upper.lane, lower.lane]

    def _slot(self, artist: Any, size: float, segments: list[Segment]) -> None:
        end = self.timeline.seconds if self.loop else None
        self.slots.append(Slot(self.board, artist, size, segments, end))
        self.words.append(artist)

    def _frame(self) -> None:
        """What does not change: the titles, the key, and the time axis with its marks."""
        board, lay, timeline = self.board, self.layout, self.timeline
        middle = lay.across / 2
        for y, line in lay.title:
            board.text(middle, y, line, style.TITLE, ha="center")
        for y, line in lay.subtitle:
            board.text(middle, y, line, style.NOTE, "soft", ha="center")
        a, b = lay.span
        one = timeline.turns == 1
        heading = (
            "One turn of the same call, as it plays" if one else "The turn under way, as it plays"
        )
        board.text(a, lay.heads[0], heading, style.LABEL, bold=True)
        self._key(a, lay.heads[1])
        for mark in range(0, int(self.axis_s) + 1, _time_step(self.axis_s)):
            x = self.x_of(mark)
            for top, bottom in self._spans():
                if mark:
                    board.line([x, x], [top, bottom], "grid", style.HAIR)
                else:
                    board.line([x, x], [top, bottom], "axis", style.THIN)
            board.text(x, lay.ticks, f"{mark}", style.NOTE, "mute", ha="center")
        for row in lay.rows:
            mid = sum(row.lane) / 2
            board.line([a, b], [mid, mid], "rule", style.HAIR)
        base = lay.rows[1].lane[1]
        board.line([a, b], [base, base], "axis", style.THIN)
        name = board.text(
            middle if not lay.joined else (a + b) / 2,
            lay.axis_name,
            "",
            style.NOTE,
            "soft",
            ha="center",
        )
        if one:
            board.set_text(name, "seconds into the turn")
        else:
            named = [
                Segment(timeline.offsets[k], f"seconds into turn {timeline.first + k}")
                for k in range(timeline.turns)
            ]
            self._slot(name, style.NOTE, named)
        x = lay.agents[0]
        board.text(x, lay.agents_heads[0], "Agents' tokens/s", style.LABEL, bold=True)
        board.text(x, lay.agents_heads[1], "over the calls' first 240 s", style.NOTE, "soft")

    def _key(self, x: float, baseline: float) -> None:
        """The lanes' three parts, each a small mark before its name, on one line: the caller's
        line and the reply as waveforms in their fills, the wait as a bar in both rows' fills."""
        board = self.board
        mid = baseline - 5.0
        xs = np.linspace(0.0, KEY_MARK, len(KEY_WAVE))
        across, down = KEY_SWATCH
        for name, hue in (
            ("The caller's line", WAVES[0]),
            ("The wait", None),
            ("The reply", WAVES[1]),
        ):
            if hue is None:
                # The baseline's half ends square at the seam; the featured half is the bar's
                # free end.
                top = mid - down / 2
                board.patch(bar_shape(x, top, across, down, "right", 0.0), BARS[0])
                after = x + across + style.SEAM
                board.patch(bar_shape(after, top, across, down, "right"), BARS[1])
            else:
                board.wave(hue).set_path(wave_path(x + xs, KEY_WAVE, mid))
            board.text(x + KEY_MARK + 6, baseline, name, style.NOTE, "soft")
            x += KEY_MARK + 6 + board.width_of(name, style.NOTE) + 18

    def _row(self, side: int) -> Side:
        """A configuration's row: its name and readout, its lane's parts, and its agents' bar."""
        board, timeline = self.board, self.timeline
        row = self.layout.rows[side]
        featured = side == 1  # its name, its clock and its value in the SemiBold face
        board.text(*row.name, timeline.calls[side].label, style.VALUE, ha=row.ha, bold=featured)
        status = board.text(*row.status, "", style.NOTE, "soft", ha=row.ha, room=row.room)
        clock = board.text(*row.clock, "", HERO, ha=row.ha, room=row.room, bold=featured)
        over = board.text(*row.verdict, "", style.NOTE, "red", ha=row.ha, room=row.room)
        within = board.text(*row.verdict, "", style.NOTE, "soft", ha=row.ha, room=row.room)
        self._hero(side, status, clock, over, within)
        speaker = self._sound() if row.sound else []
        waves = [board.wave(fill) for fill in WAVES]
        most = max(sum(1 for w in self.waits[side] if w.turn == k) for k in range(timeline.turns))
        bands = [board.bar(BARS[side]) for _ in range(max(most, 1))]
        self.shapes += [*waves, *bands, *(part for part, _, _ in speaker)]
        return Side(
            status,
            clock,
            waves,
            bands,
            speaker=speaker,
            verdicts=(over, within),
            agents=self._agents(side),
        )

    def _hero(self, side: int, status: Any, clock: Any, over: Any, within: Any) -> None:
        """The row's texts that follow the call: what it is doing, the clock, and the verdict."""
        timeline = self.timeline
        starts = [timeline.offsets[k] for k in range(timeline.turns)]
        statuses = [Segment(at, "the caller speaks") for at in starts]
        clocks = [Segment(at, "") for at in starts]
        overs = [Segment(at, "") for at in starts]
        withins = [Segment(at, "") for at in starts]
        for _, at in timeline.interruptions(side):
            statuses.append(Segment(at, "the caller interrupts"))
        shown = -math.inf  # when the last V2V was given
        for wait in self.waits[side]:
            statuses.append(Segment(wait.ended, "waiting for the reply"))
            counts = max(wait.ended, shown + HOLD_S + EXIT_S) if wait.again else wait.ended
            if counts < wait.until:
                clocks.append(Segment(counts, _counting(wait.ended)))
            answered = wait.ms is not None
            statuses.append(Segment(wait.until, "answered after" if answered else "no reply"))
            clocks.append(
                Segment(wait.until, f"{wait.seconds:.1f} s" if answered else "", enter=not answered)
            )
            if self.slo is not None:
                overs.append(Segment(wait.ended, ""))
                withins.append(Segment(wait.ended, ""))
                if wait.seconds > self.slo:
                    overs.append(Segment(wait.ended + self.slo, f"over the {self.slo:.2f} s SLO"))
                else:
                    withins.append(Segment(wait.until, f"within the {self.slo:.2f} s SLO"))
            shown = wait.until
        self._slot(status, style.NOTE, statuses)
        self._slot(clock, HERO, clocks)
        if self.slo is not None:
            self._slot(over, style.NOTE, overs)
            self._slot(within, style.NOTE, withins)

    def _sound(self) -> list[tuple[Any, np.ndarray, np.ndarray]]:
        """A speaker, the mark that the soundtrack is the row's call's: its body and two arcs in
        the words' ink, each with its outline from the speaker's left and middle. It stands
        before the row's words, wherever they start (:meth:`_heard`)."""
        board = self.board
        body = np.array([(0, -2.5), (3.5, -2.5), (8, -6.5), (8, 6.5), (3.5, 2.5), (0, 2.5)])
        parts = [(board.patch(body, "soft"), body[:, 0], body[:, 1])]
        angles = np.linspace(-math.pi / 4, math.pi / 4, 9)
        for reach in (4.5, SPEAKER_W - 9.5):
            xs, ys = 9.5 + reach * np.cos(angles), reach * np.sin(angles)
            parts.append((board.line(xs, ys, "soft", style.THIN), xs, ys))
        return parts

    def heard(self, side: int, t: float) -> float:
        """How far the soundtrack is a side's call's at t: the baseline's until the turn the
        sound switches at, then the featured call's, the one's mark leaving before the other's
        arrives."""
        switch = self.timeline.offsets[self.timeline.switch]
        return leaving(t, switch) if side == 0 else arriving(t, switch)[0]

    def _heard(self, side: int, row: Side, t: float) -> None:
        """Set the row's speaker for time t: before the words that say what the call is doing,
        and as present as they are, while the soundtrack is this call's."""
        x, baseline = self.layout.rows[side].status
        size = row.status.get_fontsize() / self.board.points
        left = x - self.board.width_of(row.status.get_text(), size) - SPEAKER_GAP - SPEAKER_W
        alpha = self.heard(side, t) * (row.status.get_alpha() or 0.0)
        for part, xs, ys in row.speaker:
            if hasattr(part, "set_xy"):
                part.set_xy(np.column_stack([left + xs, baseline - 6 + ys]))
            else:
                part.set_data(left + xs, baseline - 6 + ys)
            part.set_alpha(alpha)

    def _agents(self, side: int) -> list[Any]:
        """The agents' tokens/s over the calls' first 240 s: the row's bar, both rows' on one
        scale, its value, and its share of the baseline's as the chart's second panel gives it."""
        board, lay, calls = self.board, self.layout, self.timeline.calls
        row = lay.rows[side]
        x0, x1 = lay.agents
        featured = side == 1
        if row.agents_name is not None:
            board.text(x0, row.agents_name, calls[side].label, style.VALUE, bold=featured)
            reach = BAR / 2 + 4
            board.line([x0, x0], [row.agents - reach, row.agents + reach], "axis", style.THIN)
        elif side == 0:
            # Beside the lanes, one line where both bars start, as tall as the lanes' own.
            ends = [lay.rows[0].lane[0], lay.rows[1].lane[1]]
            board.line([x0, x0], ends, "axis", style.THIN)
        value, baseline = calls[side].tokens, calls[0].tokens
        if value is None:
            return [board.text(x0 + VALUE_GAP, row.agents + 7, "\N{EN DASH}", style.VALUE, "soft")]
        most = scale(max(c.tokens or 0.0 for c in calls))[-1]
        length = (x1 - x0) * value / most
        shape = bar_shape(x0, row.agents - BAR / 2, max(length, 0.01), BAR, "right")
        text = f"{value:,.0f}"
        after = x0 + length + VALUE_GAP
        drawn = [
            board.patch(shape, BARS[side]),
            board.text(after, row.agents + 7, text, style.VALUE, bold=featured),
        ]
        if baseline:
            after += board.width_of(text, style.VALUE, bold=featured) + SHARE_GAP
            share = f"{value / baseline:.0%}"
            drawn.append(board.text(after, row.agents + 7, share, style.NOTE, "soft"))
        return drawn

    def _history(self, plot: Box) -> None:
        """The video's plot: each reply's V2V as a bar, a pair a turn (the baseline's, a seam, the
        featured one's) on one scale, their turns numbered under them where there are several,
        and the SLO as a rule across them. A baseline's value stays over its bar where the
        columns leave it room; the featured call's bars end about the rule, so only the latest
        of their values stands, clear of the rule and of the bar beside it."""
        board, lay, timeline = self.board, self.layout, self.timeline
        x0, top, x1, base = plot
        title, numbers = lay.history_heads
        board.text(x0, title, "Each reply's voice-to-voice time (s)", style.LABEL, bold=True)
        height = base - top
        step = self.marks[1] - self.marks[0]
        for mark in self.marks:
            y = base - height * mark / self.marks[-1]
            if mark:
                board.line([x0, x1], [y, y], "grid", style.HAIR)
            board.text(x0 - 8, y + 6, mark_text(mark, step), style.NOTE, "mute", ha="right")
        board.line([x0, x0], [top, base], "axis", style.THIN)
        board.line([x0, x1], [base, base], "axis", style.THIN)
        # A turn takes a column, and one more for each interruption answered again.
        turns = range(timeline.turns)
        counts = [max(sum(1 for w in side if w.turn == k) for side in self.waits) for k in turns]
        pitch = min((x1 - x0) / max(sum(counts), 1), PAIR_PITCH)
        self.bar_width = min(BAR, float(math.floor((0.72 * pitch - style.SEAM) / 2)))
        firsts = [sum(counts[:k]) for k in turns]
        if timeline.turns > 1:
            board.text(x0 - 8, numbers, "turn", style.NOTE, "mute", ha="right")
            for k in turns:
                x = x0 + pitch * (firsts[k] + 0.5)
                board.text(x, numbers, f"{timeline.first + k}", style.NOTE, "mute", ha="center")
        room = (x0 - 40, title + 6, x1 + 40, base)
        values = []
        for side, waits in enumerate(self.waits):
            seen = [0] * timeline.turns
            for wait in waits:
                cx = x0 + pitch * (firsts[wait.turn] + seen[wait.turn] + 0.5)
                seen[wait.turn] += 1
                # The pair's values stand over their own bars and turn away from each other, so
                # that two replies of one length do not print over each other; the featured one
                # starts clear of the baseline's bar.
                bx = cx + style.SEAM / 2 if side else cx - style.SEAM / 2 - self.bar_width
                at, ha = (bx + 2 * style.SEAM, "left") if side else (bx + self.bar_width, "right")
                bar = board.bar(BARS[side])
                value = board.text(
                    at, base, "", style.NOTE, ha=ha, room=room, halo=True, bold=side == 1
                )
                self.sides[side].bars.append((wait, bar, value, bx))
                self.shapes.append(bar)
                values.append(value)
        self.words += values
        widest = max(
            (board.width_of(f"{w.seconds:.1f}", style.NOTE) for w in self.waits[0]), default=0.0
        )
        self.keeps = widest + VALUE_LIFT <= pitch
        """Whether each of the baseline's values stays over its bar, or its latest alone."""
        if self.slo is not None:
            y = base - height * self.slo / self.marks[-1]
            # Over the bars, as the chart's rule is: it is drawn after them.
            self.shapes.append(board.line([x0, x1], [y, y], "red", style.THIN, dashes=SLO_DASH))
            board.text(x1 + 8, y + 6, f"SLO {self.slo:.2f} s", style.NOTE, "red")

    def _limits(self) -> None:
        """The SLO against each wait of each turn: a rule where the SLO falls after the line's
        end, through both lanes where both calls' lines end within a unit of each other and the
        lanes are joined, a rule a lane otherwise; and its name over the turn's first rule."""
        board, lay, timeline = self.board, self.layout, self.timeline
        self.limits: list[list[tuple[float, float, float, float]]] = []
        """Each turn's rules: where it stands, its top and bottom, and when it arrives."""
        self.rules: list[Any] = []
        if self.slo is None:
            return
        upper, lower = lay.rows
        a, b = lay.span
        name = f"SLO: {self.slo:.2f} s after the line"
        half = board.width_of(name, style.NOTE) / 2
        for k in range(timeline.turns):
            start = timeline.offsets[k]
            waits = [[w for w in side if w.turn == k] for side in self.waits]
            found = []
            for j in range(max(len(side) for side in waits)):
                pair = [(lay.rows[side], w[j]) for side, w in enumerate(waits) if j < len(w)]
                xs = [self.x_of(w.ended - start + self.slo) for _, w in pair]
                if lay.joined and len(pair) == 2 and abs(xs[0] - xs[1]) <= 1.0:
                    at = min(w.ended for _, w in pair)
                    found.append((xs[0], upper.lane[0] - SLO_RISE, lower.lane[1], at))
                    continue
                for (row, w), x in zip(pair, xs, strict=True):
                    found.append((x, row.lane[0] - SLO_RISE, row.lane[1], w.ended))
            self.limits.append(found)
            if not found:
                continue
            x, _, _, at = found[0]
            x = min(max(x, a + half), b - half)  # its name stays over the axis
            named = board.text(x, lay.slo, "", style.NOTE, "red", ha="center", halo=True)
            segments = [Segment(start, ""), Segment(at, name)]
            if k + 1 < timeline.turns:
                segments.append(Segment(timeline.offsets[k + 1], ""))
            self._slot(named, style.NOTE, segments)
        most = max((len(found) for found in self.limits), default=0)
        self.rules = [
            board.line([a, a], [0.0, 0.0], "red", style.THIN, dashes=SLO_DASH) for _ in range(most)
        ]
        self.shapes += self.rules

    # Drawing --------------------------------------------------------------------------------

    def draw(self, t: float) -> np.ndarray:
        """The frame at time t, halved: RGB."""
        board = self.board
        board.canvas.restore_region(self.background)
        self.update(t)
        for artist in self.dynamic:
            board.ax.draw_artist(artist)
        return board.halve(np.asarray(board.canvas.buffer_rgba()))

    def update(self, t: float) -> None:
        """Set every changing artist for time t."""
        timeline = self.timeline
        for slot in self.slots:
            slot.show(t)
        k = timeline.turn_at(t)
        start = timeline.offsets[k]
        last = k == timeline.turns - 1
        gone = 1.0 if last and not self.loop else leaving(t, timeline.offsets[k + 1], TURN_EXIT_S)
        # A loop's end: every drawing leaves before the first turn comes back.
        end = leaving(t, timeline.seconds, TURN_EXIT_S) if self.loop else 1.0
        for side, row in enumerate(self.sides):
            self._lane(side, row, t, k, start, gone)
            self._grow(side, row, t, end)
            self._heard(side, row, t)
        found = self.limits[k] if self.limits else []
        for index, rule in enumerate(self.rules):
            if index >= len(found) or t < found[index][3]:
                rule.set_visible(False)
                continue
            x, top, bottom, at = found[index]
            rule.set_data([x, x], [top, bottom])
            rule.set_visible(True)
            rule.set_alpha(arriving(t, at)[0] * gone)
        x = self.x_of(t - start)
        for playhead in self.playheads:
            playhead.set_xdata([x, x])
            playhead.set_alpha(gone)

    def _lane(self, side: int, row: Side, t: float, k: int, start: float, gone: float) -> None:
        """A row's lane at t: the turn's waveforms as far as the playhead, and its waits' bars."""
        top, bottom = self.layout.rows[side].lane
        mid = (top + bottom) / 2
        reach = (bottom - top) / 2 - 2  # a full waveform's height about the middle line
        first, now = round(start * 100), round(t * 100)
        for patch, env in zip(row.waves, self.envelopes[side], strict=True):
            outline = self._wave(env[first:now], mid, reach)
            patch.set_visible(outline is not None)
            if outline is not None:
                patch.set_path(outline)
                patch.set_alpha(gone)
        waits = [w for w in self.waits[side] if w.turn == k]
        for index, band in enumerate(row.bands):
            if index >= len(waits) or t < waits[index].ended:
                band.set_visible(False)
                continue
            wait = waits[index]
            # A seam after the caller's line, and one before the reply.
            left = self.x_of(wait.ended - start) + style.SEAM
            right = min(self.x_of(t - start), self.x_of(wait.until - start) - style.SEAM)
            band.set_visible(right > left)
            if right > left:
                band.set_xy(bar_shape(left, mid - BAR / 2, right - left, BAR, "right"))
                band.set_alpha(gone)

    def _wave(self, peaks: np.ndarray, mid: float, reach: float) -> Any:
        """A waveform's outline from a turn's start: a point for each ``stride`` of the envelope's
        10 ms (the loudest of them), and one at the playhead for those under way."""
        stride = self.stride
        whole = len(peaks) // stride
        points = peaks[: whole * stride].reshape(whole, stride).max(axis=1)
        seconds = (np.arange(whole) + 0.5) * stride / 100
        if len(peaks) > whole * stride:
            points = np.append(points, peaks[whole * stride :].max())
            seconds = np.append(seconds, len(peaks) / 100)
        if len(points) < 2:
            return None
        a, b = self.layout.span
        xs = a + (b - a) * seconds / self.axis_s
        return wave_path(xs, np.minimum(points / REFERENCE, 1.0) * reach, mid)

    def _grow(self, side: int, row: Side, t: float, end: float) -> None:
        """Each wait's bar in the video's plot: growing while the caller waits, then its V2V over
        it, or over the SLO's rule where the bar ends under the rule."""
        if self.layout.history is None:
            return
        _, top, _, base = self.layout.history
        height = base - top
        rule = base if self.slo is None else base - height * self.slo / self.marks[-1]
        stays = side == 0 and self.keeps
        starts = [*(w.ended for w, _, _, _ in row.bars[1:]), math.inf]
        for (wait, bar, value, bx), following in zip(row.bars, starts, strict=True):
            if t < wait.ended:
                bar.set_visible(False)
                value.set_alpha(0.0)
                continue
            done = t >= wait.until
            seconds = wait.seconds if done else t - wait.ended
            h = height * min(seconds, self.marks[-1]) / self.marks[-1]
            bar.set_xy(bar_shape(bx, base - h, self.bar_width, max(h, 0.01), "top"))
            bar.set_visible(True)
            bar.set_alpha(end)
            if not done:
                value.set_alpha(0.0)
                continue
            alpha, size = arriving(t, wait.until)
            text = f"{wait.seconds:.1f}" if wait.ms is not None else "\N{EN DASH}"
            self.board.set_text(value, text)
            value.set_y(min(base - h, rule) - VALUE_LIFT)
            value.set_alpha(alpha * end * (1.0 if stays else leaving(t, following)))
            value.set_fontsize(style.NOTE * self.board.points * size)

    def check(self, t: float) -> None:
        """Draw the frame at t in full and refuse a text that leaves its room or overlaps
        another."""
        self.update(t)
        for artist in self.dynamic:
            artist.set_animated(False)
        try:
            self.board.canvas.draw()
            self.board.check()
        finally:
            for artist in self.dynamic:
                artist.set_animated(True)
            self.board.canvas.draw()
            self.background = self.board.canvas.copy_from_bbox(self.board.figure.bbox)

    def moments(self) -> list[float]:
        """The moments a check draws: each turn's start and end, each wait's end of speech, its
        passing the SLO, its reply and a hold after it."""
        found = {0.0, self.timeline.seconds - 1e-3}
        for k, offset in enumerate(self.timeline.offsets[:-1]):
            found |= {offset + ENTER_S, self.timeline.offsets[k + 1] - EXIT_S / 2}
        for side in self.waits:
            for w in side:
                found |= {w.ended + 0.05, w.ended + ENTER_S, w.until + ENTER_S, w.until + HOLD_S}
                if self.slo is not None:
                    found.add(w.ended + self.slo + ENTER_S)
        return sorted(t for t in found if 0 <= t < self.timeline.seconds)


# Cards ------------------------------------------------------------------------------------------


def _counted(callers: int) -> str:
    return NUMBERS[callers] if callers < len(NUMBERS) else str(callers)


def _turns(first: int, last: int) -> str:
    """Turns ``first`` to ``last`` in words: "turn 3", "turns 1\N{EN DASH}5"."""
    return f"turn {first}" if first == last else f"turns {first}\N{EN DASH}{last}"


def title_card(
    timeline: Timeline, heading: str, *, width: int = 1920, theme: str = "light"
) -> np.ndarray:
    """The opening card: ``heading``, the calls and the engines as the records name them, each
    after the key's swatch of its row's fill, and how the video plays them."""
    board = Board(width, FULL_H, theme)
    calls = timeline.calls
    gpu = calls[0].gpu or calls[1].gpu
    callers = calls[0].callers
    when = datetime.fromtimestamp(calls[0].wall, UTC).strftime("%Y-%m-%d")
    where = f" on one {gpu}" if gpu else ""
    first, last, switch = timeline.first, timeline.first + timeline.turns - 1, timeline.switch
    above, below = (f"the call on {labels.in_prose(call.label)}" for call in calls)
    if switch < timeline.turns:
        early, late = _turns(first, first + switch - 1), _turns(first + switch, last)
        heard = f"Sound: {early} from {above}, then {late} from {below}"
    else:
        heard = f"Sound: {above}"
    top = FULL_H / 2 - 104
    lines = [
        (0.0, heading, style.TITLE, "ink"),
        (
            30.0,
            f"{_counted(callers)} caller{'s' if callers != 1 else ''} beside "
            f"{calls[0].sessions} coding-agent sessions{where}, recorded {when}",
            style.NOTE,
            "soft",
        ),
        (
            160.0,
            "Each wait is as the caller measured it; only the pauses between turns are stretched",
            style.NOTE,
            "soft",
        ),
        (186.0, heard, style.NOTE, "soft"),
    ]
    for dy, text, size, ink in lines:
        board.text(W / 2, top + dy, text, size, ink, ha="center")
    # The two rows the video draws: the row's fill as the lanes' key shows it, then its name.
    across, down = KEY_SWATCH
    widest = max(board.width_of(c.label, style.VALUE, bold=k == 1) for k, c in enumerate(calls))
    column = W / 2 - (across + 9 + widest) / 2
    for side, (call, dy) in enumerate(zip(calls, (84.0, 114.0), strict=True)):
        mark = bar_shape(column, top + dy - 6 - down / 2, across, down, "right")
        board.patch(mark, BARS[side])
        board.text(column + across + 9, top + dy, call.label, style.VALUE, bold=side == 1)
    image = board.pixels()
    board.check()
    return image


def closing_card(
    summary: Summary,
    timeline: Timeline,
    *,
    also: Sequence[Summary] = (),
    width: int = 1920,
    theme: str = "light",
) -> np.ndarray:
    """The closing card: both configurations over all the report's runs, as the README's chart
    draws them: the median of their V2V p95 against the SLO, and of the agents' tokens/s over the
    calls' first 240 s with each one's share of the baseline's. ``also`` is other GPUs' reports of
    the same two points: each is a group under its GPU's name after the calls' own, against its own
    SLO and its own baseline's tokens/s, on one scale."""
    board = Board(width, FULL_H, theme)
    calls = timeline.calls
    gpu = calls[0].gpu or calls[1].gpu
    groups = [(summary.gpu or gpu, summary), *((other.gpu, other) for other in also)]
    several = len(groups) > 1
    top = 100.0 if several else 176.0
    board.text(W / 2, top, "Both configurations over all their runs", style.TITLE, ha="center")
    where = " on each GPU" if several else f" on one {gpu}" if gpu else ""
    sub = (
        f"{_counted(calls[0].callers)} voice calls beside {calls[0].sessions} coding-agent "
        f"sessions{where}"
    )
    board.text(W / 2, top + 27, sub, style.NOTE, "soft", ha="center")
    # The chart's columns: the names end where its names end, unless the longer name needs more,
    # and then the first panel gives up the difference; it ends where it always does.
    widest = max(board.width_of(c.label, style.VALUE, bold=k == 1) for k, c in enumerate(calls))
    names = max(LABELS_END, 14.0 + widest)
    margin = names - widest
    ax0, bx0 = names + PLOT_START - LABELS_END, AGENTS_START
    alen, blen = PLOT_END - ax0, AGENTS_END - AGENTS_START
    heads = top + 84
    reference = labels.in_prose(calls[0].label)  # the shares' reference, named as its rows name it
    panels = (
        (ax0, "Voice-to-voice p95", "seconds, median of the runs", "Lower"),
        (bx0, "Agents' tokens/s", f"% of {reference}'s", "Higher"),
    )
    for x, head, unit, better in panels:
        board.text(x, heads, head, style.LABEL, bold=True)
        board.text(x, heads + 25, f"{unit} \N{BULLET} {better} is better", style.NOTE, "soft")
    p95_top = max(max(*g.p95_ms, g.slo_ms or 0.0) for _, g in groups) / 1000
    marks_a = ticks(p95_top, alen)
    marks_b = ticks(max([v or 0.0 for _, g in groups for v in g.tokens] + [1.0]), blen)
    reach = ROW_PITCH / 2  # from a row's middle line to its edge
    rows = []  # each row's middle line, its group and its side
    slos = []  # each group's SLO, and its rule's top and bottom
    y = heads + 25 + 43  # the first group's line: its GPU's name, and its SLO's
    plot = y + SLO_RISE
    for name, group in groups:
        if several:
            board.text(margin, y, name or "", style.NOTE, "soft")
        first = y + SLO_RISE + reach
        rows += [(first + ROW_PITCH * side, group, side) for side in (0, 1)]
        bottom = first + ROW_PITCH + reach
        if group.slo_ms is not None:
            slos.append((group.slo_ms / 1000, y + SLO_RISE, bottom))
        y = bottom + 34
    bottom = rows[-1][0] + reach
    for x0, length, marks in ((ax0, alen, marks_a), (bx0, blen, marks_b)):
        step = marks[1] - marks[0]
        for mark in marks:
            x = x0 + length * mark / marks[-1]
            if mark:
                board.line([x, x], [plot, bottom], "grid", style.HAIR)
            else:
                board.line([x, x], [plot, bottom], "axis", style.THIN)
            board.text(x, bottom + 23, mark_text(mark, step), style.NOTE, "mute", ha="center")
        board.line([x0, x0 + length], [bottom, bottom], "axis", style.THIN)
    values = []  # over the SLO's rules, in the page's halo
    for ry, group, side in rows:
        featured = side == 1
        board.text(names, ry - 3, calls[side].label, style.VALUE, ha="right", bold=featured)
        runs = group.runs[side]
        runs_text = f"{runs} run{'s' if runs != 1 else ''}"
        board.text(names, ry + 19, runs_text, style.NOTE, "soft", ha="right")
        p95 = group.p95_ms[side] / 1000
        length = alen * p95 / marks_a[-1]
        board.patch(bar_shape(ax0, ry - BAR / 2, length, BAR, "right"), BARS[side])
        values.append((ax0 + length + VALUE_GAP, ry, f"{p95:.2f}", style.VALUE, "ink", featured))
        value, baseline = group.tokens[side], group.tokens[0]
        if value is None:
            values.append((bx0 + VALUE_GAP, ry, "\N{EN DASH}", style.VALUE, "soft", False))
            continue
        length = blen * value / marks_b[-1]
        board.patch(bar_shape(bx0, ry - BAR / 2, length, BAR, "right"), BARS[side])
        text = f"{value:,.0f}"
        after = bx0 + length + VALUE_GAP
        values.append((after, ry, text, style.VALUE, "ink", featured))
        if baseline:
            after += board.width_of(text, style.VALUE, bold=featured) + SHARE_GAP
            values.append((after, ry, f"{value / baseline:.0%}", style.NOTE, "soft", False))
    for slo, rise, end in slos:
        x = ax0 + alen * slo / marks_a[-1]
        board.line([x, x], [rise, end if several else bottom], "red", style.THIN, dashes=SLO_DASH)
        name = f"SLO {slo:.2f} s"
        board.text(x, rise - SLO_RISE, name, style.NOTE, "red", ha="center", halo=True)
    for x, ry, text, size, ink, featured in values:
        board.text(x, ry + 7, text, size, ink, halo=True, bold=featured)
    image = board.pixels()
    board.check()
    return image


def arrive(image: np.ndarray, page: np.ndarray, t: float, at: float) -> np.ndarray:
    """``image`` arriving at ``at`` over the page: from nothing at 0.96 of its size, eased out
    over CARD_ENTER_S."""
    from PIL import Image

    alpha, size = arriving(t, at, CARD_ENTER_S)
    if alpha >= 1:
        return image
    h, w = image.shape[:2]
    shrunk = (round(w * size), round(h * size))
    small = Image.fromarray(image).resize(shrunk, Image.Resampling.LANCZOS)
    canvas = page.copy()
    top, left = (h - small.height) // 2, (w - small.width) // 2
    canvas[top : top + small.height, left : left + small.width] = np.asarray(small)
    return blend(page, canvas, alpha)


def blend(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """``weight`` of b over a."""
    if weight >= 1:
        return b
    if weight <= 0:
        return a
    mixed = a.astype(np.float32) * (1 - weight) + b.astype(np.float32) * weight + 0.5
    return mixed.astype(np.uint8)


@dataclass
class Plan:
    """The video's frames in order: the opening card, the calls arriving, the calls, the calls
    leaving with the closing card after them."""

    fps: float
    title: int
    """The opening card's frames, its arrival and its exit included; none without a heading."""
    arrival: int
    calls: int
    tail: int
    """The calls' exit and the closing card; none without a summary."""

    @property
    def lead(self) -> int:
        """The frames before the calls' first moment, which the soundtrack's silence matches."""
        return self.title + self.arrival

    @property
    def total(self) -> int:
        return self.lead + self.calls + self.tail


def plan(timeline: Timeline, summary: Summary | None, heading: str | None, fps: float) -> Plan:
    title = round(TITLE_S * fps) if heading else 0
    tail = round((CARD_EXIT_S + CLOSING_S) * fps) if summary else 0
    return Plan(fps, title, round(CARD_ENTER_S * fps), round(timeline.seconds * fps), tail)


def video_frames(
    timeline: Timeline,
    summary: Summary | None,
    *,
    fps: float = 60,
    width: int = 1920,
    theme: str = "light",
    heading: str | None = None,
    also: Sequence[Summary] = (),
) -> tuple[Plan, Callable[[], Iterator[np.ndarray]]]:
    """The video's plan, and a generator of its frames; ``also`` as :func:`closing_card` takes it."""
    steps = plan(timeline, summary, heading, fps)

    def generate() -> Iterator[np.ndarray]:
        frames = Dashboard(timeline, summary, width=width, theme=theme, form="video")
        page = np.empty((frames.board.height, width, 3), dtype=np.uint8)
        page[:] = [int(frames.board.color("page")[i : i + 2], 16) for i in (1, 3, 5)]
        first = frames.draw(0.0).copy()
        if steps.title:
            card = title_card(timeline, heading or "", width=width, theme=theme)
            for index in range(steps.title):
                t = index / fps
                yield blend(page, arrive(card, page, t, 0.0), leaving(t, TITLE_S, CARD_EXIT_S))
        for index in range(steps.arrival):
            yield arrive(first, page, index / fps, 0.0)
        last = first
        for index in range(steps.calls):
            last = frames.draw(index / fps)
            yield last
        if steps.tail:
            assert summary is not None
            card = closing_card(summary, timeline, also=also, width=width, theme=theme)
            held = last.copy()
            for index in range(steps.tail):
                t = index / fps
                if t < CARD_EXIT_S:
                    yield blend(page, held, leaving(t, CARD_EXIT_S, CARD_EXIT_S))
                else:
                    yield arrive(card, page, t, CARD_EXIT_S)

    return steps, generate


def render(
    timeline: Timeline,
    out: Path,
    summary: Summary | None = None,
    *,
    fps: float = 60,
    width: int = 1920,
    theme: str = "light",
    heading: str | None = None,
    also: Sequence[Summary] = (),
) -> Plan:
    """Write the video with its soundtrack: H.264 at CRF 18 in yuv420p, BT.709, AAC."""
    import imageio_ffmpeg

    steps, generate = video_frames(
        timeline, summary, fps=fps, width=width, theme=theme, heading=heading, also=also
    )
    rate = timeline.calls[0].rate
    height = round(width * FULL_H / W / 2) * 2
    with tempfile.TemporaryDirectory() as scratch:
        sound = Path(scratch) / "soundtrack.wav"
        with wave.open(str(sound), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(np.zeros(round(steps.lead / fps * rate), dtype=np.int16).tobytes())
            wav.writeframes(soundtrack(timeline)[: round(steps.calls / fps * rate)].tobytes())
            wav.writeframes(np.zeros(round(steps.tail / fps * rate), dtype=np.int16).tobytes())
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
            "-i", "-", "-i", str(sound), "-map", "0:v", "-map", "1:a",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,"
            "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
            "-c:v", "libx264", "-preset", "slow", "-tune", "animation", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out),
        ]  # fmt: skip
        _pipe(command, generate())
    return steps


EXCERPTS = (("", "light", False), ("-dark", "dark", False), ("-narrow", "light", True))
EXCERPTS += (("-narrow-dark", "dark", True),)
"""The README's animations, by their names' suffix: each theme, wide and in the phone layout."""
README_PX, PHONE_PX = 840, 340
"""The widths the README shows a figure at, and at which a phone shows its narrow layout."""


def excerpt_paths(prefix: Path) -> list[Path]:
    """The animations ``--excerpt PREFIX`` writes: PREFIX.webp, its dark theme PREFIX-dark.webp,
    and their phone layouts PREFIX-narrow.webp and PREFIX-narrow-dark.webp."""
    return [prefix.with_name(f"{prefix.name}{suffix}.webp") for suffix, _, _ in EXCERPTS]


def write_excerpt(
    timeline: Timeline,
    summary: Summary | None,
    path: Path,
    *,
    width: int,
    fps: float,
    theme: str,
    narrow: bool = False,
) -> int:
    """The calls' turns as the board's two rows, a looping animated WebP (lossless), silent: each
    frame stored as what changed from the one before. The loop starts on the first turn's result
    held (:func:`held_result`), so that a viewer that shows only the first frame, or a poster of
    it, shows both replies and their V2V rather than two empty lanes. Returns its frames."""
    import imageio_ffmpeg

    if path.suffix != ".webp":
        raise ValueError(f"an excerpt is an animated .webp, not {path.suffix}")
    frames = Dashboard(
        timeline, summary, width=width, theme=theme, form="narrow" if narrow else "wide"
    )
    count = round(timeline.seconds * fps)
    start = min(math.ceil(held_result(timeline) * fps), count - 1)
    size = f"{width}x{frames.board.height}"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libwebp_anim", "-lossless", "1", "-compression_level", "6", "-loop", "0",
        str(path),
    ]  # fmt: skip
    order = [*range(start, count), *range(start)]  # the same loop, from the held result
    _pipe(command, (frames.draw(index / fps) for index in order))
    return count


def held_result(timeline: Timeline) -> float:
    """When the first shown turn's result is held, on the video's clock: every reply of the turn
    on both calls given and its V2V arrived (ENTER_S after the last), before the turn's drawing
    leaves. A frame from then until the turn leaves shows both replies and both V2Vs."""
    last = max(w.until for side in (0, 1) for w in timeline.waits(side) if w.turn == 0)
    held = last + ENTER_S
    if held >= timeline.offsets[1] - TURN_EXIT_S:
        raise ValueError(
            f"the first turn leaves at {timeline.offsets[1]:.2f} s, before its result holds"
        )
    return held


def _pipe(command: list[str], frames: Iterator[np.ndarray]) -> None:
    """Feed raw frames to ffmpeg, and fail with its own words if it fails."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
        pass
    finally:
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait():
            raise RuntimeError(f"ffmpeg failed: {error.strip()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "baseline", type=Path, help="run directory of the baseline point (the upper row)"
    )
    parser.add_argument(
        "featured", type=Path, help="run directory of the featured point (the lower row)"
    )
    parser.add_argument("--out", type=Path, help="the MP4 to write")
    parser.add_argument(
        "--summary", type=Path, help="a report's frontier.csv: the SLO and the closing card"
    )
    parser.add_argument(
        "--also",
        type=Path,
        action="append",
        default=[],
        metavar="FRONTIER_CSV",
        help="another GPU's report of the same two points: a group of the closing card, against "
        "its own SLO (repeatable)",
    )
    parser.add_argument("--heading", help="the opening card's heading (default: no opening card)")
    parser.add_argument(
        "--excerpt",
        type=Path,
        metavar="PREFIX",
        help="write the README's animations, PREFIX.webp, PREFIX-dark.webp, PREFIX-narrow.webp "
        "and PREFIX-narrow-dark.webp",
    )
    parser.add_argument("--excerpt-turns", type=int, default=1, help="their turns (default 1)")
    parser.add_argument(
        "--excerpt-width",
        type=int,
        default=2 * README_PX,
        help=f"the wide ones' width (default {2 * README_PX}, the narrow ones' in proportion)",
    )
    parser.add_argument("--excerpt-fps", type=float, default=60.0, help="their rate (default 60)")
    parser.add_argument("--caller", type=int, default=0, help="each point's call (default 0)")
    parser.add_argument(
        "--labels", nargs=2, metavar=("BASELINE", "FEATURED"), help="instead of the arms' labels"
    )
    parser.add_argument("--first-turn", type=int, help="the first turn shown (default: measured)")
    parser.add_argument("--turns", type=int, help="the turns shown (default: as many as fit)")
    parser.add_argument(
        "--seconds", type=float, default=180.0, help="the calls' most (default 180)"
    )
    parser.add_argument("--fps", type=float, default=60.0, help="frames a second (default 60)")
    parser.add_argument("--width", type=int, default=1920, help="the video's width (default 1920)")
    parser.add_argument("--theme", choices=("light", "dark"), default="light", help="the video's")
    args = parser.parse_args(argv)
    if args.out is None and args.excerpt is None:
        parser.error("nothing to write: give --out, --excerpt or both")
    if args.also and args.summary is None:
        parser.error("--also adds to the closing card that --summary makes: give --summary")
    labels = args.labels or (None, None)
    try:
        baseline, featured = (
            Call.load(run_dir, args.caller, label)
            for run_dir, label in zip((args.baseline, args.featured), labels, strict=True)
        )
        calls = (baseline, featured)
        timeline = Timeline.choose(calls, args.first_turn, args.turns, args.seconds)
        points = [c.point for c in calls]
        summary = Summary.read(args.summary, points) if args.summary else None
        also = [Summary.read(path, points) for path in args.also]
        short = Timeline.choose(calls, timeline.first, args.excerpt_turns)
    except (OSError, KeyError, ValueError) as exc:
        print(f"demo.py: {exc}", file=sys.stderr)
        return 1
    if args.out is not None:
        steps = render(
            timeline,
            args.out,
            summary,
            fps=args.fps,
            width=args.width,
            theme=args.theme,
            heading=args.heading,
            also=also,
        )
        cards = (f", an opening card of {TITLE_S:g} s" if steps.title else "") + (
            f", a closing card of {CLOSING_S:g} s" if steps.tail else ""
        )
        last = timeline.first + timeline.turns - 1
        print(
            f"{args.out}: {steps.total} frames at {args.fps:g} fps, {timeline.seconds:.1f} s of "
            f"calls{cards}, turns {timeline.first}-{last}"
        )
    if args.excerpt is None:
        return 0
    phone = round(args.excerpt_width * PHONE_PX / README_PX / 2) * 2
    ends = short.first + short.turns - 1
    turns = f"turn {short.first}" if short.turns == 1 else f"turns {short.first}-{ends}"
    for path, (_, theme, narrow) in zip(excerpt_paths(args.excerpt), EXCERPTS, strict=True):
        width = phone if narrow else args.excerpt_width
        count = write_excerpt(
            short, summary, path, width=width, fps=args.excerpt_fps, theme=theme, narrow=narrow
        )
        form = "the phone layout" if narrow else "wide"
        print(f"{path}: {count} frames at {args.excerpt_fps:g} fps, {theme}, {form}, {turns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
