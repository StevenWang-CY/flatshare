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
in the left panel, and the configuration the video is about, in the right. Each side plays one of
its point's calls (``--caller``, by default the first), and both play the same turns: ``--turns`` of
them from ``--first-turn``, by default from the first turn the callers measured (after the warm-up
turns that open each call) and as many as fit in ``--seconds``. Turn k of both starts at the same
moment, so the same line is heard on both at once, and the call that is ahead waits in silence for
the other: only a caller's pause between turns is stretched, never a measured wait.

A frame is the dashboard the v1.0 demo drew, a panel a call side by side. In each panel: what the
call is doing, the wait's clock, which counts from the end of the caller's line to the reply's
first audio and then gives the V2V the caller measured, and that wait against the SLO; each reply's
V2V as a bar that grows while the caller waits, on one scale for both panels; the turn under way on
its time axis, the caller's line above it and the agent's reply below as their waveforms, drawn as
they play, with the wait between them as a bar; and the agents' tokens/s over the calls' first
240 s, the README's measure, on one scale for both. ``--summary`` is a report's frontier.csv
(``flatshare-bench e2 report``): the voice SLO is the report's, 250 ms above voice alone's median V2V
p95 with as many callers, drawn as a dashed rule against each wait and each bar; and a closing card
gives both configurations (by their points' names) over all the report's runs, the median of their
V2V p95 and of the agents' tokens/s over the calls' first 240 s, as the README's chart draws them.
``--also`` adds another GPU's report of the same two points (its own frontier.csv): the closing card
then gives each GPU's as a group under its name, against its own SLO. An opening card names the
calls and the engines (``--heading`` is its heading).

The soundtrack is one call at a time, the caller and the agent mixed: the baseline's for the first
half of the turns and the featured configuration's for the rest, and the panel heard is marked. The
excerpt (``--excerpt PREFIX``, a README's animation) is the same dashboard, silent and looping:
``--excerpt-turns`` turns from the first, without the cards, as animated lossless WebPs in each
theme, the panels side by side and, for a phone, one above the other 420 units across
(PREFIX.webp, PREFIX-dark.webp, PREFIX-narrow.webp, PREFIX-narrow-dark.webp).

Frames follow the docs' figures (``figure_style.py`` and ``fonts/`` in
python/flatshare_bench/src/flatshare_bench): STIX Two at the figures' four sizes, and the clock at
HERO, on a layout 1120 units wide; the palette of ``--theme``; the panels on the figures' gray ground
in dashed borders; the configuration to deploy in the saturated blue and the baseline in gray. Each
frame is drawn at twice its size and halved by a box filter, so type and hairlines stay crisp.
Motion follows the calls: the playhead, the waveforms, the clock and the growing bars move linearly
with their time. A text that changes leaves faster than its successor arrives (ease-in-out, then
ease-out from 0.96 of its size), a V2V once given holds at least HOLD_S, a turn's drawing leaves
before the next turn starts, and nothing a viewer reads moves otherwise.
The video is 1920 by 1080, H.264 (yuv420p, CRF 18, BT.709) at 60 frames a second with AAC sound,
encoded by the ffmpeg that imageio-ffmpeg bundles.
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
BARS = ("bar-mid", "bar-hi")
"""Each side's fill: the baseline a comparison's mid gray, the featured configuration the
saturated blue, as the README's chart has them."""


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


class Board:
    """A matplotlib canvas in a layout's units (``across`` by ``down``, y down), ``width`` pixels
    wide once halved and drawn at twice that, in a theme's palette, its text in STIX Two Text. It
    keeps each text it draws with the room the text must stay inside, for :meth:`check`."""

    def __init__(self, width: int, down: float, theme: str = "light", *, across: float = W) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.font_manager import FontProperties

        if width % 2:
            raise ValueError(f"a frame's width is an even number of pixels, not {width}")
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
        self.font = FontProperties(fname=str(style.FONT_DIR / "STIXTwoText-Regular.ttf"))
        self.glyphs = style._advances("regular")[1]
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
        halo: float = 0.0,
    ) -> Any:
        """A text at baseline ``y``, which must stay inside ``room`` (the canvas by default); with a
        ``halo`` of the page's color that many units wide, as the figures set a value that a rule may
        cross (6, half of it each side of a stroke)."""
        artist = self.ax.text(
            x,
            y,
            "",
            fontproperties=self.font,
            fontsize=size * self.points,
            color=self.colors[ink],
            ha=ha,
            va="baseline",
            zorder=3,
        )
        if halo:
            from matplotlib import patheffects

            stroke = patheffects.withStroke(linewidth=self.lw(halo), foreground=self.colors["page"])
            artist.set_path_effects([stroke])
        self.set_text(artist, text)
        self.texts.append((artist, room or (0.0, 0.0, self.across, self.units)))
        return artist

    def set_text(self, artist: Any, text: str) -> None:
        """Set a text, refusing a character the face has no glyph for."""
        missing = {c for c in text if ord(c) not in self.glyphs}
        if missing:
            raise ValueError(f"no glyph in STIX Two Text for {sorted(missing)} in {text!r}")
        artist.set_text(text)

    def width_of(self, text: str, size: float) -> float:
        return style.measure(text, size)

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

    def patch(
        self,
        vertices: np.ndarray,
        fill: str,
        edge: str | None,
        width: float,
        *,
        dashes: tuple[float, float] | None = None,
    ) -> Any:
        """A filled polygon, its outline ``width`` units wide; ``dashes`` (on, off) in units."""
        from matplotlib.patches import Polygon

        artist = Polygon(
            vertices,
            closed=True,
            facecolor=self.colors[fill],
            edgecolor=self.colors[edge] if edge else "none",
            linewidth=self.lw(width) if edge else 0,
            joinstyle="round",
        )
        if dashes and edge:
            artist.set_linestyle((0, (dashes[0] / width, dashes[1] / width)))
        self.ax.add_patch(artist)
        return artist

    def bar(self, fill: str) -> Any:
        """A flat bar in ``fill`` with the bars' outline, its shape set by :func:`bar_shape`."""
        return self.patch(np.zeros((4, 2)), fill, "bar-line", style.BAR_LINE)

    def swatch(self, x: float, baseline: float, fill: str, size: float = style.NOTE) -> float:
        """A legend's swatch before a name at ``baseline``, as the figures draw it; returns where
        the name starts."""
        shape = bar_shape(x, baseline - 0.62 * size, 22, 0.66 * size, "right", 2.0)
        self.patch(shape, fill, "bar-line", style.BAR_LINE)
        return x + 30

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


def bar_shape(x: float, y: float, w: float, h: float, end: str, r: float = 3.0) -> np.ndarray:
    """The outline of a flat bar at (x, y), ``w`` by ``h``, its ``end`` corners ("top" or "right")
    rounded as the figures round them (quadratic corners of radius ``r``), as a polygon."""
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


def headings(timeline: Timeline) -> tuple[str, str, str]:
    """The frame's title and subtitle, and the turn's note, from the records."""
    calls = timeline.calls
    gpu = calls[0].gpu or calls[1].gpu
    callers = calls[0].callers
    title = f"The same caller on two engines, beside {calls[0].sessions} coding-agent sessions"
    when = datetime.fromtimestamp(calls[0].wall, UTC).strftime("%Y-%m-%d")
    where = f" on one {gpu}" if gpu else ""
    who = f"One of {_counted(callers).lower()} callers" if callers > 1 else "The only caller"
    note = "The same line on both calls at once; only the pauses between turns are stretched"
    return title, f"{who}{where}, recorded {when}", note


def _counting(ended: float) -> Callable[[float], str]:
    """The clock while the caller waits: the seconds since the speech ended, to a tenth."""

    def text(t: float) -> str:
        return f"{max(t - ended, 0.0):.1f} s"

    return text


def _time_step(seconds: float) -> int:
    """The time axis's step: the least of 1, 2, 5 and 10 s that marks it at most six times."""
    return next((s for s in (1, 2, 5, 10, 20, 30, 60) if seconds / s <= 6), 60)


# The dashboard ----------------------------------------------------------------------------------

HERO = 44.0
"""The size of a panel's clock, the number each panel builds to: about twice a label's."""


@dataclass
class Panels:
    """Where the dashboard's parts go, in units: the calls in two panels side by side (the wide form,
    1120 across) or one above the other (the phone form, 420 across), each panel laid out the same
    from its own top left (:func:`wide_panels`, :func:`narrow_panels`)."""

    across: float
    down: float
    title: list[tuple[float, str]]
    """The title's lines, centred, at their baselines; then the subtitle's."""
    subtitle: list[tuple[float, str]]
    boxes: tuple[Box, Box]
    """Each panel's left, top, right and bottom."""
    pad: float
    """A panel's inner margin."""
    name: float
    status: float
    clock: float
    verdict: float
    """Baselines from a panel's top: its call's name, what the call is doing, the wait's clock, and
    the wait against the SLO."""
    plot: tuple[float, float, float, float]
    """The waits' plot, from the panel's left and top: its left, top, right and base."""
    lanes: tuple[float, float, float, float]
    """The turn's time axis, from the panel's left and top: where it starts and ends (the lanes'
    names stand left of it), and its top and bottom; the caller's line is drawn above its middle,
    the agent's reply below."""
    ticks: float
    axis_name: float
    """Baselines from a panel's top: the time axis's marks and its name."""
    agents: tuple[float, float, float]
    """The agents' tokens/s: the label's baseline, and the bar's top and bottom, from the panel's
    top."""
    sound: bool = False
    """Whether a panel marks when the soundtrack is its call's: the video's, not the README's."""


def _subtitle(timeline: Timeline) -> str:
    return f"{headings(timeline)[1]} \N{BULLET} the same line on both engines at once"


def wide_panels(timeline: Timeline) -> Panels:
    """The README's animation, 1120 by 612 units: the calls' panels side by side under the title."""
    top, height = 100.0, 488.0
    return Panels(
        across=W,
        down=top + height + 24,
        title=[(46.0, headings(timeline)[0])],
        subtitle=[(72.0, _subtitle(timeline))],
        boxes=((24.0, top, 548.0, top + height), (572.0, top, 1096.0, top + height)),
        pad=24.0,
        name=40.0,
        status=82.0,
        clock=134.0,
        verdict=164.0,
        plot=(330.0, 72.0, 500.0, 164.0),
        lanes=(92.0, 500.0, 200.0, 356.0),
        ticks=380.0,
        axis_name=404.0,
        agents=(440.0, 452.0, 470.0),
    )


def video_panels(timeline: Timeline) -> Panels:
    """The video's frame, 1120 by 630 units (16 by 9): the README's panels with room for a bar a
    turn, and each marks when the soundtrack is its call's."""
    top, height = 96.0, 510.0
    return Panels(
        across=W,
        down=FULL_H,
        title=[(44.0, headings(timeline)[0])],
        subtitle=[(70.0, _subtitle(timeline))],
        boxes=((24.0, top, 548.0, top + height), (572.0, top, 1096.0, top + height)),
        pad=24.0,
        name=40.0,
        status=82.0,
        clock=134.0,
        verdict=164.0,
        plot=(262.0, 72.0, 500.0, 182.0),
        lanes=(92.0, 500.0, 222.0, 378.0),
        ticks=402.0,
        axis_name=426.0,
        agents=(462.0, 474.0, 492.0),
        sound=True,
    )


def narrow_panels(timeline: Timeline) -> Panels:
    """The README's phone layout, 420 units across like the figures' phone layouts: the title and
    subtitle broken evenly, then the calls' panels one above the other."""
    title, _, _ = headings(timeline)
    subtitle = _subtitle(timeline)
    left, right = 12.0, 408.0
    width = right - left - 8
    y = 32.0
    titles = []
    for line in style.wrap(title, width, style.TITLE):
        titles.append((y, line))
        y += 29
    y += 2
    subtitles = []
    head, _, note = subtitle.partition(" \N{BULLET} ")
    where, _, when = head.rpartition(", ")
    lines = [head] if style.measure(head, style.NOTE) <= width else [f"{where},", when]
    for line in [*lines, note]:  # broken before the date, never inside the GPU's name
        subtitles.append((y, line))
        y += 23
    top, height, gap = y + 4, 424.0, 16.0
    boxes = (
        (left, top, right, top + height),
        (left, top + height + gap, right, top + 2 * height + gap),
    )
    return Panels(
        across=420.0,
        down=boxes[1][3] + 12,
        title=titles,
        subtitle=subtitles,
        boxes=boxes,
        pad=18.0,
        name=36.0,
        status=72.0,
        clock=120.0,
        verdict=148.0,
        plot=(240.0, 68.0, 378.0, 148.0),
        lanes=(76.0, 378.0, 172.0, 304.0),
        ticks=326.0,
        axis_name=350.0,
        agents=(382.0, 392.0, 408.0),
    )


@dataclass
class Side:
    """A call's panel: what changes in it."""

    status: Any
    """What the call is doing: the caller speaks, waits, or was answered."""
    clock: Any
    """The wait so far, then the V2V the caller measured."""
    waves: list[Any]
    """The caller's line, drawn above the time axis, and the agent's reply, below it."""
    bands: list[Any]
    """The waits of the turn, one bar each on the time axis: the line's, and an interruption's."""
    rules: list[Any]
    """The SLO against each wait."""
    playhead: Any
    bars: list[tuple[Wait, Any, Any, float]] = field(default_factory=list)
    """Each wait in the plot: its bar, its value, and the bar's left."""
    heard: Any = None
    """Whether the soundtrack is this call's: the video's mark."""
    verdicts: tuple[Any, Any] = (None, None)
    """The wait against the SLO: over it, as soon as the clock passes it; within it, at the
    reply."""
    agents: list[Any] = field(default_factory=list)
    """The agents' tokens/s: the bar, then its value."""


FORMS = {"wide": wide_panels, "narrow": narrow_panels, "video": video_panels}
"""The dashboard's layouts: the README's animation, its phone layout, and the video's frame."""


class Dashboard:
    """The calls in two panels, ``width`` pixels wide, as RGB pixels, as the v1.0 demo set them
    side by side: in each panel what the call is doing and the wait's clock, the wait against the
    SLO, each reply's V2V as a bar that grows while the caller waits, the turn under way on its time
    axis (the caller's line above it, the agent's reply below, the wait between them as a bar) and
    the agents' tokens/s over the calls' first 240 s. Each turn's drawing leaves before the next
    turn starts. ``form`` is one of FORMS: the README's animations loop, their last turn's drawing
    leaving before the end; the video's frame holds its last turn and marks the call heard."""

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
        self.axis_s = float(math.ceil(max(lengths)))
        values = [w.seconds for side in self.waits for w in side]
        self.marks = scale(max([*values, self.slo or 0.0, 1.0]), most=2)
        self.dynamic: list[Any] = []
        self.slots: list[Slot] = []
        board = self.board
        middle = lay.across / 2
        for y, line in lay.title:
            board.text(middle, y, line, style.TITLE, ha="center")
        for y, line in lay.subtitle:
            board.text(middle, y, line, style.NOTE, "soft", ha="center")
        self.sides = [self._panel(side) for side in (0, 1)]
        for artist in self.dynamic:
            artist.set_animated(True)
        board.canvas.draw()
        self.background = board.canvas.copy_from_bbox(board.figure.bbox)

    # Layout ---------------------------------------------------------------------------------

    def x_of(self, side: int, seconds: float) -> float:
        """A moment of the turn, in seconds from its start, on a panel's time axis."""
        lay = self.layout
        x0 = lay.boxes[side][0]
        a, b = x0 + lay.lanes[0], x0 + lay.lanes[1]
        return a + (b - a) * min(max(seconds, 0.0), self.axis_s) / self.axis_s

    def _slot(self, artist: Any, size: float, segments: list[Segment]) -> None:
        end = self.timeline.seconds if self.loop else None
        self.slots.append(Slot(self.board, artist, size, segments, end))
        self.dynamic.append(artist)

    def _panel(self, side: int) -> Side:
        board, lay, timeline = self.board, self.layout, self.timeline
        call = timeline.calls[side]
        x0, y0, x1, y1 = lay.boxes[side]
        pad = lay.pad
        room = (x0 + 4, y0 + 4, x1 - 4, y1 - 4)
        # The panel's ground, square as the figures' panels are.
        ground = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=float)
        board.patch(ground, "ground-b", "border", style.PANEL_LINE, dashes=(6.0, 4.0))
        board.text(x0 + pad, y0 + lay.name, call.label, style.LABEL, room=room)
        heard = None
        if lay.sound:
            after = x0 + pad + style.measure(call.label, style.LABEL) + 16
            heard = board.text(after, y0 + lay.name, "", style.NOTE, "mute", room=room)
            switch = timeline.offsets[timeline.switch]
            marks = [
                Segment(0.0, "sound on" if side == 0 else ""),
                Segment(switch, "" if side == 0 else "sound on"),
            ]
            self._slot(heard, style.NOTE, marks)
        # What the call is doing, the wait's clock and the wait against the SLO, left of the plot.
        plot_x0, plot_top, plot_x1, plot_base = (
            x0 + lay.plot[0],
            y0 + lay.plot[1],
            x0 + lay.plot[2],
            y0 + lay.plot[3],
        )
        hero = (x0 + 4, y0 + lay.name + 8, plot_x0 - 44, y0 + lay.lanes[2] - 8)
        status = board.text(x0 + pad, y0 + lay.status, "", style.NOTE, "soft", room=hero)
        clock = board.text(x0 + pad, y0 + lay.clock, "", HERO, room=hero)
        over = board.text(x0 + pad, y0 + lay.verdict, "", style.NOTE, "orange", room=hero)
        within = board.text(x0 + pad, y0 + lay.verdict, "", style.NOTE, "soft", room=hero)
        self._hero(side, status, clock, over, within)
        # The waits' plot: each reply's V2V as a bar, on one scale for both panels. Its note shares the names' line,
        # so where the longer name leaves it no room, both panels give it in the glossary's short form.
        note = "voice-to-voice, s"
        taken = pad + max(style.measure(c.label, style.LABEL) for c in timeline.calls) + 16
        if lay.sound:
            taken += style.measure("sound on", style.NOTE) + 16
        if lay.plot[2] - style.measure(note, style.NOTE) < taken:
            note = "V2V, s"
        board.text(plot_x1, y0 + lay.name, note, style.NOTE, "mute", ha="right", room=room)
        height = plot_base - plot_top
        step = self.marks[1] - self.marks[0]
        for mark in self.marks:
            y = plot_base - height * mark / self.marks[-1]
            if mark:
                board.line([plot_x0, plot_x1], [y, y], "grid", style.THIN)
            text = mark_text(mark, step)
            board.text(plot_x0 - 8, y + 6, text, style.NOTE, "mute", ha="right", room=room)
        board.line([plot_x0, plot_x0], [plot_top - 6, plot_base], "axis", style.THIN)
        board.line([plot_x0, plot_x1], [plot_base, plot_base], "axis", style.THIN)
        if self.slo is not None:
            y = plot_base - height * self.slo / self.marks[-1]
            board.line([plot_x0, plot_x1], [y, y], "orange", style.THIN, dashes=(5, 4))
            board.text(plot_x1, y - 6, "SLO", style.NOTE, "orange", ha="right", room=room, halo=6.0)
        # The bars keep clear of the SLO's name at the plot's right end.
        clear = plot_x1 - (style.measure("SLO", style.NOTE) + 14 if self.slo is not None else 0)
        bars = self._bars(side, plot_x0, clear, plot_base, room)
        # The turn's time axis: the caller's line above its middle, the agent's reply below.
        a, b = x0 + lay.lanes[0], x0 + lay.lanes[1]
        top, bottom = y0 + lay.lanes[2], y0 + lay.lanes[3]
        mid = (top + bottom) / 2
        names = (("caller", (top + mid) / 2 + 6), ("agent", (mid + bottom) / 2 + 6))
        for name, baseline in names:
            board.text(x0 + pad, baseline, name, style.NOTE, "mute", room=room)
        marks = [float(s) for s in range(0, int(self.axis_s) + 1, _time_step(self.axis_s))]
        for mark in marks:
            x = self.x_of(side, mark)
            board.line([x, x], [top, bottom], "axis" if mark == 0 else "grid", style.THIN)
            board.text(x, y0 + lay.ticks, f"{mark:.0f}", style.NOTE, "mute", ha="center", room=room)
        board.line([a, b], [mid, mid], "axis", style.THIN)
        name = board.text(b, y0 + lay.axis_name, "", style.NOTE, "mute", ha="right", room=room)
        named = [
            Segment(timeline.offsets[k], f"seconds into turn {timeline.first + k}")
            for k in range(timeline.turns)
        ]
        self._slot(name, style.NOTE, named)
        waves = [board.patch(np.zeros((3, 2)), f"{hue}-w", hue, 0.8) for hue in ("gray", "green")]
        most = max(sum(1 for w in self.waits[side] if w.turn == k) for k in range(timeline.turns))
        bands, rules = [], []
        for _ in range(max(most, 1)):
            bands.append(board.bar(BARS[side]))
            if self.slo is not None:
                rules.append(
                    board.line([0, 0], [top + 4, bottom - 4], "orange", style.THIN, dashes=(4, 3))
                )
        playhead = board.line([a, a], [top, bottom], "axis", style.THIN)
        self.dynamic += [*waves, *bands, *rules, playhead]
        # The agents' tokens/s over the calls' first 240 s, each panel's on one scale.
        agents = self._agents(side, x0 + pad, y0, x1 - pad, room)
        return Side(
            status, clock, waves, bands, rules, playhead, bars, heard, (over, within), agents
        )

    def _hero(self, side: int, status: Any, clock: Any, over: Any, within: Any) -> None:
        """The panel's texts that follow the call: what it is doing, the clock, and the verdict."""
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

    def _bars(
        self, side: int, x0: float, x1: float, base: float, room: Box
    ) -> list[tuple[Wait, Any, Any, float]]:
        """A bar for each wait of the panel's call, in the plot between x0 and x1; their turns named
        under them where there are several."""
        board, timeline = self.board, self.timeline
        waits = self.waits[side]
        pitch = (x1 - x0) / max(len(waits), 1)
        self.bar_width = min(28.0, pitch * 0.5)
        found = []
        for j, wait in enumerate(waits):
            cx = x0 + pitch * (j + 0.5)
            if len(waits) > 1 and not wait.again:  # an interruption's bar follows its turn's
                turn = str(timeline.first + wait.turn)
                board.text(cx, base + 22, turn, style.NOTE, "mute", ha="center", room=room)
            bx = cx - self.bar_width / 2
            bar = board.bar(BARS[side])
            value = board.text(cx, base, "", style.NOTE, ha="center", room=room, halo=6.0)
            found.append((wait, bar, value, bx))
            self.dynamic += [bar, value]
        return found

    def _agents(self, side: int, left: float, top: float, right: float, room: Box) -> list[Any]:
        """The agents' tokens/s over the calls' first 240 s: a bar on one scale for both panels,
        and the featured call's share of the baseline's."""
        board, lay, calls = self.board, self.layout, self.timeline.calls
        label_y, bar_top, bar_bottom = (top + v for v in lay.agents)
        label = "the agents' tokens/s over the calls' first 240 s"
        board.text(left, label_y, label[0].upper() + label[1:], style.NOTE, "soft", room=room)
        value, baseline = calls[side].tokens, calls[0].tokens
        if value is None:
            return [board.text(left, bar_bottom, "\N{EN DASH}", style.NOTE, "soft", room=room)]
        text = f"{value:,.0f}"
        share = f"{value / baseline:.0%} of the baseline's" if side and baseline else ""
        # One scale for both panels: the longest bar leaves room for the widest value and share.
        widest = max(
            style.measure(f"{c.tokens:,.0f}", style.NOTE)
            + (
                16 + style.measure(f"{c.tokens / baseline:.0%} of the baseline's", style.NOTE)
                if k and baseline
                else 0
            )
            for k, c in enumerate(calls)
            if c.tokens is not None
        )
        most = max(c.tokens or 0.0 for c in calls)
        length = (right - left - widest - 12) * value / most
        shape = bar_shape(left, bar_top, max(length, 0.01), bar_bottom - bar_top, "right")
        bar = board.patch(shape, BARS[side], "bar-line", style.BAR_LINE)
        baseline_y = (bar_top + bar_bottom) / 2 + 6
        drawn = [bar, board.text(left + length + 8, baseline_y, text, style.NOTE, room=room)]
        if share:
            after = left + length + 16 + style.measure(text, style.NOTE)
            drawn.append(board.text(after, baseline_y, share, style.NOTE, "soft", room=room))
        return drawn

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
        for side, panel in enumerate(self.sides):
            self._lanes(side, panel, t, k, start, gone)
            x = self.x_of(side, t - start)
            panel.playhead.set_xdata([x, x])
            panel.playhead.set_alpha(gone)
            self._grow(side, panel, t, end)

    def _lanes(self, side: int, panel: Side, t: float, k: int, start: float, gone: float) -> None:
        lay = self.layout
        x0, y0 = lay.boxes[side][:2]
        a, b = x0 + lay.lanes[0], x0 + lay.lanes[1]
        top, bottom = y0 + lay.lanes[2], y0 + lay.lanes[3]
        mid = (top + bottom) / 2
        reach = mid - top - 6  # a full waveform's height
        first, now = round(start * 100), round(t * 100)
        for wave_patch, env, sign in zip(panel.waves, self.envelopes[side], (-1, 1), strict=True):
            values = env[first:now]
            if len(values) < 2:
                wave_patch.set_visible(False)
                continue
            xs = a + (b - a) * (np.arange(len(values)) / 100) / self.axis_s
            ys = mid + sign * np.minimum(values / REFERENCE, 1.0) * reach
            outline = np.concatenate([np.column_stack([xs, ys]), [[xs[-1], mid], [xs[0], mid]]])
            wave_patch.set_xy(outline)
            wave_patch.set_visible(True)
            wave_patch.set_alpha(gone)
        waits = [w for w in self.waits[side] if w.turn == k]
        for index, band in enumerate(panel.bands):
            rule = panel.rules[index] if panel.rules else None
            if index >= len(waits) or t < waits[index].ended:
                band.set_visible(False)
                if rule is not None:
                    rule.set_visible(False)
                continue
            wait = waits[index]
            left = self.x_of(side, wait.ended - start)
            right = self.x_of(side, min(t, wait.until) - start)
            band.set_xy(bar_shape(left, mid - 8, max(right - left, 0.01), 16, "right"))
            band.set_visible(True)
            band.set_alpha(gone)
            if rule is not None and self.slo is not None:
                x = self.x_of(side, wait.ended - start + self.slo)
                rule.set_xdata([x, x])
                rule.set_visible(True)
                rule.set_alpha(arriving(t, wait.ended)[0] * gone)

    def _grow(self, side: int, panel: Side, t: float, end: float) -> None:
        """Each wait's bar in the plot: growing while the caller waits, then its V2V over it."""
        lay = self.layout
        y0 = lay.boxes[side][1]
        top, base = y0 + lay.plot[1], y0 + lay.plot[3]
        height = base - top
        starts = [*(w.ended for w, _, _, _ in panel.bars[1:]), math.inf]
        for (wait, bar, value, bx), following in zip(panel.bars, starts, strict=True):
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
            value.set_y(base - h - 8)
            value.set_alpha(alpha * end * leaving(t, following))  # the latest V2V alone
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
    """The opening card: ``heading``, the calls and the engines as the records name them, and how
    the video plays them."""
    board = Board(width, FULL_H, theme)
    calls = timeline.calls
    gpu = calls[0].gpu or calls[1].gpu
    callers = calls[0].callers
    when = datetime.fromtimestamp(calls[0].wall, UTC).strftime("%Y-%m-%d")
    where = f" on one {gpu}" if gpu else ""
    first, last, switch = timeline.first, timeline.first + timeline.turns - 1, timeline.switch
    if switch < timeline.turns:
        above, below = _turns(first, first + switch - 1), _turns(first + switch, last)
        heard = f"Sound: {above} from the call on the left, then {below} from the call on the right"
    else:
        heard = "Sound: the call on the left"
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
    places = (("Left", 84.0), ("Right", 114.0))
    left = max(board.width_of(place, style.NOTE) for place, _ in places)
    right = max(board.width_of(call.label, style.VALUE) for call in calls)
    column = W / 2 - (left + 14 + right) / 2 + left
    for (place, dy), call in zip(places, calls, strict=True):
        board.text(column, top + dy, place, style.NOTE, "mute", ha="right")
        board.text(column + 14, top + dy, call.label, style.VALUE)
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
    top = 84.0 if several else 184.0
    board.text(W / 2, top, "Both configurations over all their runs", style.TITLE, ha="center")
    where = " on each GPU" if several else f" on one {gpu}" if gpu else ""
    sub = (
        f"{_counted(calls[0].callers)} voice calls beside {calls[0].sessions} coding-agent "
        f"sessions{where}"
    )
    board.text(W / 2, top + 27, sub, style.NOTE, "soft", ha="center")
    # The names' column is as wide as the longer name needs, and the first panel gives up the rest; it ends where it
    # always does.
    right = max(238.0, 24.0 + max(board.width_of(call.label, style.VALUE) for call in calls))
    ax0, bx0, blen = right + 24.0, 764.0, 200.0
    alen = 682.0 - ax0
    heads = top + 84
    panels = (
        (ax0 + alen / 2, "Voice-to-voice p95", "seconds, median of the runs"),
        (bx0 + blen / 2 + 24, "Agents' throughput", "tokens/s, % of the baseline"),
    )
    for (cx, head, sub_text), better in zip(panels, ("Lower", "Higher"), strict=True):
        board.text(cx, heads, head, style.LABEL, ha="center")
        line = f"{sub_text} \N{BULLET} {better} is better"
        board.text(cx, heads + 25, line, style.NOTE, "soft", ha="center")
    plot = heads + 68  # the first SLO's label; the plot starts under it
    pitch, thick = 54.0, 26.0
    p95_top = max(max(*g.p95_ms, g.slo_ms or 0.0) for _, g in groups) / 1000
    marks_a = ticks(p95_top, alen)
    marks_b = ticks(max([v or 0.0 for _, g in groups for v in g.tokens] + [1.0]), blen)
    rows = []  # each row's top, its group and its side
    slos = []  # each group's SLO: its label's baseline and its rows' top and end
    y = plot + 16
    for k, (name, group) in enumerate(groups):
        if several:
            board.text(24, y + 22, name or "", style.NOTE, "mute")
            y += 36
        rows += [(y + pitch * side, group, side) for side in (0, 1)]
        if group.slo_ms is not None:
            slos.append((group.slo_ms / 1000, plot if k == 0 else y - 8, y + pitch + thick + 6))
        y += 2 * pitch + 10
    bottom = rows[-1][0] + thick + 8
    for x0, length, marks in ((ax0, alen, marks_a), (bx0, blen, marks_b)):
        step = marks[1] - marks[0]
        for mark in marks:
            x = x0 + length * mark / marks[-1]
            board.line([x, x], [plot + 8, bottom], "axis" if mark == 0 else "grid", style.THIN)
            board.text(x, bottom + 23, mark_text(mark, step), style.NOTE, "mute", ha="center")
        board.line([x0, x0 + length], [bottom, bottom], "axis", style.HAIR)
    values = []
    for ry, group, side in rows:
        board.text(right, ry + 15, calls[side].label, style.VALUE, ha="right")
        runs = group.runs[side]
        runs_text = f"{runs} run{'s' if runs != 1 else ''}"
        board.text(right, ry + 37, runs_text, style.NOTE, "soft", ha="right")
        p95 = group.p95_ms[side] / 1000
        length = alen * p95 / marks_a[-1]
        shape = bar_shape(ax0, ry, length, thick, "right")
        board.patch(shape, BARS[side], "bar-line", style.BAR_LINE)
        values.append((ax0 + length + 8, ry + 19, f"{p95:.2f}", "ink"))
        value, baseline = group.tokens[side], group.tokens[0]
        if value is None:
            values.append((bx0 + 8, ry + 19, "\N{EN DASH}", "soft"))
            continue
        length = blen * value / marks_b[-1]
        shape = bar_shape(bx0, ry, length, thick, "right")
        board.patch(shape, BARS[side], "bar-line", style.BAR_LINE)
        values.append((bx0 + length + 8, ry + 19, f"{value:,.0f}", "ink"))
        if baseline:
            after = bx0 + length + 16 + board.width_of(f"{value:,.0f}", style.NOTE)
            values.append((after, ry + 19, f"{value / baseline:.0%}", "soft"))
    for slo, label, end in slos:
        x = ax0 + alen * slo / marks_a[-1]
        board.line(
            [x, x], [label + 8, end if several else bottom], "orange", style.THIN, dashes=(5, 4)
        )
        board.text(x, label, f"SLO {slo:.2f} s", style.NOTE, "orange", ha="center", halo=6.0)
    for x, baseline, text, ink in values:
        board.text(x, baseline, text, style.NOTE, ink, halo=6.0)
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
    """The calls' turns as the dashboard, a looping animated WebP (lossless), silent: each frame
    stored as what changed from the one before. The loop starts on the first turn's result held
    (:func:`held_result`), so that a viewer that shows only the first frame, or a poster of it,
    shows both replies and their V2V rather than two empty panels. Returns its frames."""
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
    parser.add_argument("baseline", type=Path, help="run directory of the baseline point (left)")
    parser.add_argument("featured", type=Path, help="run directory of the featured point (right)")
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
