"""The docs' diagrams, each with its phone layout, in the figures' style (:mod:`flatshare_bench.figure_style`):

    python -m flatshare_bench.diagrams OUT_DIR

The method figure, docs/figures/method.svg and method-narrow.svg, the README's and the report's: a voice agent and
coding agents on one GPU, and the two scheduler rules of vLLM #63350 and #63351, in three panels one under another.
(a) is the serving path: a caller's turn through speech to text, vLLM's scheduler and text to speech, on green
flows, beside the coding agents' requests, and the KV pool the scheduler allocates from, a tray of blocks. (b) and
(c) each compare vLLM 0.32 with one rule, named as the report names it and by its number, a row each under one
column of names: (b), the KV preemption (#63350), in the pool's blocks, and (c), the budget cap (#63351), in the
engine's steps along time.

The blocks and steps are illustrative, and obey the rules they show (docs/REPORT.md §3). Make-room
preempts lower-priority running requests in vLLM's own order, the latest arrival first, as few as make the
waiting request fit; (b) works out its victims with :func:`make_room` and draws what it returns. The
budget cap keeps lower-priority prefill out of the steps from the one where the turn fits until it has its first
tokens, and never cuts a decode. A step's length is drawn in the ratio of the recorded gaps between a reply's tokens
beside agent prefill and without it (the README's "What it shows": 60-65 ms and 16-20 ms), at the middle of each.

The system diagrams: docs/ARCHITECTURE.md's figures, overview.svg (the system at a glance: the engine that carries the change
on the serving path, and what measures it and what runs it as two lists beside it), then the serving path's,
architecture.svg (the hosts, their units and the sockets between them), request-path.svg (a request's steps through
the gateway and each one's refusals), two-cards.svg (the two-card arrangement's placement and its measured failover)
and release.svg (a release's states and exits), and ops/RUNBOOK.md's triage.svg (five checks from a failing call to
its runbook row). Each draws what the files it depicts say, which tests/test_diagrams.py holds it to.

One grammar for all of them (the design system's): a component is a box, pale inside a thin outline at corner 5,
and the one that carries the change is the only blue one; a host or a namespace is a hairline frame at corner 8
with its name inside its top left; KV blocks, steps and slots are flat cells at corner 2; the voice turn's path and
blocks are green, everything else neutral; dashes mean a part that is there only with the second card.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

from flatshare_bench.figure_style import (
    CORNER,
    LABEL,
    NARROW,
    NOTE,
    WIDTH,
    Figure,
    measure,
    wrap,
)

VOICE, AGENT = "green", "gray"
"""The voice turn's hue and the agents' (blue is the change's own: the component that carries it)."""
POOL = 12
"""KV blocks in (b)'s pool."""
HOLDERS = (("agent A", 5), ("agent B", 5))
"""The running agent requests, in the order they arrived, and the blocks each holds."""
TURN = 4
"""The blocks the waiting voice turn needs for its whole sequence."""
WINDOW = 48
"""The reply's first tokens during which #63351 keeps agent prefill out: the configuration to deploy's."""
DECODE_MS, SHARED_MS = 18.0, 62.0
"""A step that carries only the reply's decode, and one that carries an agent's prefill chunk beside it: the
middles of the recorded 16-20 ms and 60-65 ms gaps between the reply's tokens."""
RESUMED = 2
"""Steps drawn after the window, where agent prefill shares the steps again."""
STEPS_AXIS = "Time from the reply's first token"
"""(c)'s time axis: one green block a step."""
EDGE = 24.0
"""A wide figure's margin; a phone layout's is 16."""
NAMES = 264.0
"""Where (b)'s pool and (c)'s steps begin in the wide figure: right of the one column of their rows' names."""
BLOCK = 32.0
"""The height of a KV block in (b) and of a step in (c): one mark at one weight."""
TRAY = 6.0
"""The pool's tray around its blocks."""
SPAN = 8.0
"""From a row of blocks, or its tray, down to the bracket that names its spans."""


def make_room(free: int, holders: tuple[tuple[str, int], ...], need: int) -> list[str]:
    """The victims make-room takes for a request that needs ``need`` blocks while ``free`` are free: lower
    priority first (every holder here is an agent, of one priority), then the latest arrival, as few as make
    it fit; none when all of them would not suffice."""
    victims, room = [], free
    for name, blocks in reversed(holders):
        if room >= need:
            break
        victims.append(name)
        room += blocks
    return victims if room >= need else []


Spans = list[tuple[str, int, int]]
"""The named runs of a row of blocks: a name, its first block and the one past its last."""


def pool_states() -> tuple[list[str | None], Spans, list[str | None], Spans, str, int]:
    """(b)'s two rows. On vLLM 0.32: the pool's blocks, the agents' and the free ones, and their spans, while the
    turn waits. With the KV preemption: the pool after make-room's victim left it and the turn took its blocks.
    Then the victim's name and how many blocks it held."""
    held = sum(blocks for _, blocks in HOLDERS)
    free = POOL - held
    victims = make_room(free, HOLDERS, TURN)
    # What the panel shows: the turn does not fit, and one request, the latest, makes room for it.
    assert free < TURN and victims == [HOLDERS[-1][0]] == ["agent B"], (free, victims)
    before: list[str | None] = [AGENT] * held + [None] * free
    spans: Spans = []
    start = 0
    for name, blocks in HOLDERS:
        spans.append((name, start, start + blocks))
        start += blocks
    (victim,) = victims
    _, first, last = spans[-1]
    after = [*before[:first], *[VOICE] * TURN, *[None] * (POOL - first - TURN)]
    return (
        before,
        [*spans, ("free", held, POOL)],
        after,
        [*spans[:-1], ("voice turn", first, first + TURN), ("free", first + TURN, POOL)],
        victim,
        last - first,
    )


def _fit(text: str, width: float, size: float, *, bold: bool = False) -> list[str]:
    """``text`` in lines no wider than ``width``, broken at its spaces."""
    lines, line = [], ""
    for word in text.split(" "):
        trial = f"{line} {word}".strip()
        if line and measure(trial, size, bold=bold) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def _title(s: Figure, x: float, y: float, letter: str, title: str) -> float:
    """A panel's title with its letter, in the semibold from ``x``, its capitals' top at ``y``, on as many lines as
    the figure's width asks. A panel of the method figure has no frame: its title and the space over it part it
    from the one above. Returns the last line's baseline."""
    baseline = y + 0.7 * LABEL
    lines = _fit(f"({letter}) {title}", s.width - 2 * x, LABEL, bold=True)
    for k, line in enumerate(lines):
        s.text(x, baseline + 26 * k, line, bold=True)
    return baseline + 26 * (len(lines) - 1)


def _row_name(
    s: Figure, x: float, y: float, name: str, note: str, *, ours: bool = False, right: float = 0
) -> float:
    """A row's name and under it, softer, what happens in the row; the change's own row (``ours``) in the semibold.
    Beside the row (the wide figure), both end at ``x`` and stand about the row's middle, ``y``. Over the row (a
    phone's, ``right`` its right edge), the name starts at ``x`` on the baseline ``y`` and the note ends at
    ``right`` on the same line, or stands on a line of its own when the two do not fit. Returns the last baseline."""
    if not right:
        s.text(x, y - 4, name, anchor="end", bold=ours)
        s.text(x, y + 18, note, size=NOTE, anchor="end", cls="soft")
        return y + 18
    box = s.text(x, y, name, bold=ours)
    if box.x1 + 14 + measure(note, NOTE) <= right:
        s.text(right, y, note, size=NOTE, anchor="end", cls="soft")
        return y
    s.text(x, y + 22, note, size=NOTE, cls="soft")
    return y + 22


def _blocks(
    s: Figure,
    x: float,
    y: float,
    pitch: float,
    blocks: list[str | None],
    spans: Spans,
    *,
    tray: bool = True,
    taken: bool = False,
) -> float:
    """A row of KV blocks from ``x``, ``pitch`` apart, in the pool's ``tray`` or, waiting, outside it; ``taken``
    hatches them, a preempted request's. A bracket under the row names each of its ``spans``. Returns the lowest
    edge of the names."""
    if tray:
        s.memory(x - TRAY, y - TRAY, pitch * len(blocks) + 2 * TRAY, BLOCK + 2 * TRAY)
    s.cells(
        x,
        y,
        blocks,
        cell=pitch,
        height=BLOCK,
        strong=(VOICE,),
        hatched=frozenset(range(len(blocks))) if taken else frozenset(),
    )
    under = y + BLOCK + TRAY + SPAN
    for name, a, b in spans:
        s.bracket(x + pitch * a + 3, x + pitch * b - 3, under, name)
    return under + 0.72 * NOTE + 6 + 0.24 * NOTE


def make_room_panel(s: Figure, x: float, y: float, w: float, *, stacked: bool = False) -> float:
    """(b): the pool's blocks on vLLM 0.32, full of two agent requests' while the voice turn waits outside it, and
    with the KV preemption, where the latest agent request waits instead, preempted, and the turn has its blocks.
    The rows' names stand in a column left of them, or, ``stacked`` (a phone's layout), over them, with what waits
    under each pool. Returns the panel's lowest edge."""
    before, before_spans, after, after_spans, victim, lost = pool_states()
    base = _title(s, x, y, "b", "The KV preemption, #63350")
    pool = f"KV pool, {POOL} blocks"
    rows = (
        ("vLLM 0.32", "the voice turn waits", False, before, before_spans),
        ("The KV preemption", "the latest agent waits", True, after, after_spans),
    )
    waits = (
        ([VOICE] * TURN, [("voice turn", 0, TURN)], False),
        ([AGENT] * lost, [(f"{victim}, preempted", 0, lost)], True),
    )
    if not stacked:
        pitch = 44.0
        out = NAMES + pitch * POOL + TRAY + 64
        s.text(NAMES, base + 34, pool, size=NOTE, cls="soft")
        s.text(out, base + 34, "waiting", size=NOTE, cls="soft")
        top = base + 34 + 18
        low = top
        for (name, note, ours, blocks, spans), (waiting, who, taken) in zip(
            rows, waits, strict=True
        ):
            _row_name(s, NAMES - TRAY - 18, top + BLOCK / 2, name, note, ours=ours)
            _blocks(s, NAMES, top, pitch, blocks, spans)
            low = _blocks(s, out, top, pitch, waiting, who, tray=False, taken=taken)
            if taken:  # the victim left the pool
                mid = top + BLOCK / 2
                s.flow([(NAMES + pitch * POOL + TRAY + 8, mid), (out - 4, mid)], hue=AGENT)
            top = low + 22
        return low
    pitch = math.floor((w - 2 * TRAY) / POOL)
    left = x + TRAY
    top = base + 34
    low = top
    for k, ((name, note, ours, blocks, spans), (waiting, who, taken)) in enumerate(
        zip(rows, waits, strict=True)
    ):
        top = _row_name(s, x, top, name, note, ours=ours, right=x + w) + 18
        if not k:  # the tray's name, once, over the first
            s.text(x, top + 10, pool, size=NOTE, cls="soft")
            top += 28
        low = _blocks(s, left, top, pitch, blocks, spans)
        top = low + 14
        low = _blocks(s, left, top, pitch, waiting, who, tray=False, taken=taken)
        s.text(
            left + pitch * len(waiting) + 12,
            top + BLOCK / 2 + 0.33 * NOTE,
            "waiting",
            size=NOTE,
            cls="soft",
        )
        top = low + 34
    return low


def _steps(s: Figure, x: float, y: float, unit: float, alone: int, shared: int) -> float:
    """A row of the engine's steps from ``x``, ``unit`` a millisecond: ``alone`` steps that carry the reply's token
    only, then ``shared`` ones that carry an agent's prefill chunk beside it. Returns the row's right edge."""
    token, chunk = DECODE_MS * unit, (SHARED_MS - DECODE_MS) * unit
    s.cells(x, y, [VOICE] * alone, cell=token, height=BLOCK, strong=(VOICE,))
    at = x + token * alone
    for _ in range(shared):
        s.cells(at, y, [VOICE], cell=token, height=BLOCK, strong=(VOICE,))
        s.cells(at + token, y, [AGENT], cell=chunk, height=BLOCK)
        at += token + chunk
    return at


def hold_prefill_panel(s: Figure, x: float, y: float, w: float, *, stacked: bool = False) -> float:
    """(c): the engine's steps on one time axis from the reply's first. On vLLM 0.32 (or with the KV preemption
    alone, whose gaps the steps are drawn from) each carries an agent's prefill chunk beside the reply's token;
    with the budget cap the window's steps carry the reply alone, at decode speed, and agent prefill resumes
    after it. The rows' names stand in (b)'s column, or, ``stacked``, over the rows. Returns the panel's lowest
    edge."""
    base = _title(s, x, y, "c", "The budget cap, #63351")
    window_ms = WINDOW * DECODE_MS
    capped_ms = window_ms + RESUMED * SHARED_MS
    stock = round(capped_ms / SHARED_MS)  # the steps vLLM 0.32 takes in the time drawn
    in_window = round(window_ms / SHARED_MS)  # and how many of them in the window's time
    left = x if stacked else NAMES
    # A phone's row ends short of the edge, so that the last bracket's name, wider than its steps, stays inside.
    length = x + w - left - (22 if stacked else 0)
    unit = length / max(capped_ms, stock * SHARED_MS)
    # The key: what a step holds, on one line where it fits.
    key = base + 34
    kx = left
    for hue, width, name in (
        (VOICE, 16.0, "The reply's token"),
        (AGENT, 34.0, "An agent's prefill chunk"),
    ):
        if kx > left and kx + width + 8 + measure(name, NOTE) > x + w:
            kx, key = left, key + 26
        s.cells(kx, key - 14, [hue], cell=width, height=16, strong=(VOICE,))
        kx = s.text(kx + width + 8, key, name, size=NOTE).x1 + 28
    rows = (
        ("vLLM 0.32", "the reply shares its steps", False, 0, stock),
        ("The budget cap", "agent prefill waits", True, WINDOW, RESUMED),
    )
    top = key + 18
    low = top
    for name, note, ours, alone, shared in rows:
        if stacked:
            top = _row_name(s, x, top + 22, name, note, ours=ours, right=x + w) + 14
        else:
            _row_name(s, NAMES - TRAY - 18, top + BLOCK / 2, name, note, ours=ours)
        _steps(s, left, top, unit, alone, shared)
        under = top + BLOCK + SPAN
        if alone:
            s.bracket(
                left + 1, left + window_ms * unit - 1, under, f"the reply's first {WINDOW} tokens"
            )
            s.bracket(left + window_ms * unit + 1, left + capped_ms * unit - 1, under, "resumed")
        else:
            s.bracket(
                left + 1, left + in_window * SHARED_MS * unit - 1, under, f"{in_window} tokens"
            )
        low = under + 0.72 * NOTE + 6 + 0.24 * NOTE
        top = low + (10 if stacked else 22)
    # The time axis, from the reply's first token: a green block a step.
    axis = low + 16
    s.time_axis(left, left + length, axis)
    for k, line in enumerate(wrap(STEPS_AXIS, length, NOTE)):
        low = axis + 24 + 22 * k
        s.text(left, low, line, size=NOTE, cls="mute")
    return low + 0.24 * NOTE


SCHEDULER = ["vLLM 0.32 scheduler", "with #63350, #63351"]
"""(a)'s one blue component: the scheduler, which carries both rules."""
STRIP_TITLE = "One GPU: a voice agent beside coding agents"


def strip(s: Figure, y: float) -> float:
    """(a), wide: the voice turn's path left to right on green flows, the agents' requests under it into the same
    scheduler, and right of the scheduler the KV pool it allocates from, (b)'s first row in small. Returns the
    panel's lowest edge."""
    base = _title(s, EDGE, y, "a", STRIP_TITLE)
    h, pad = 46.0, 28.0
    lane1 = base + 24 + h / 2
    lane2 = lane1 + h + 18

    def width(text: str) -> float:
        return max(104.0, measure(text, LABEL) + pad)

    caller = (EDGE, width("Caller"))
    stt = (caller[0] + caller[1] + 48, width("Speech to text"))
    sched_w = max(measure(line, LABEL) for line in SCHEDULER) + 2 * pad
    sched_x = stt[0] + stt[1] + 108
    back = (WIDTH - EDGE - width("Caller"), width("Caller"))
    tts = (back[0] - 48 - width("Text to speech"), width("Text to speech"))
    agents = (EDGE, stt[0] + stt[1] - EDGE)
    assert tts[0] - (sched_x + sched_w) >= 100, (sched_x, sched_w, tts)  # room for the reply's flow
    s.box(caller[0], lane1 - h / 2, caller[1], h, "Caller")
    s.box(stt[0], lane1 - h / 2, stt[1], h, "Speech to text")
    s.box(agents[0], lane2 - h / 2, agents[1], h, "24 coding-agent sessions")
    s.box(tts[0], lane1 - h / 2, tts[1], h, "Text to speech")
    s.box(back[0], lane1 - h / 2, back[1], h, "Caller")
    top, bottom = lane1 - h / 2, lane2 + h / 2
    s.box(sched_x, top, sched_w, bottom - top, SCHEDULER, kind="ours")
    # The KV pool, the scheduler's memory, beside it: a tray of blocks, as (b) draws it.
    pool_x = sched_x + sched_w + 48
    pool_w = WIDTH - EDGE - pool_x
    s.memory(pool_x, lane2 - h / 2, pool_w, h)
    name = s.text(pool_x + 14, lane2 + 0.33 * LABEL, "KV pool")
    pitch = math.floor((pool_x + pool_w - 11 - name.x1 - 14) / POOL)
    s.cells(
        pool_x + pool_w - 11 - pitch * POOL, lane2 - 12, pool_states()[0], cell=pitch, height=24
    )
    s.flow([(sched_x + sched_w, lane2), (pool_x, lane2)], arrow=False)
    # Dataflow: the turn and the reply, green, and the agents' requests.
    s.flow([(caller[0] + caller[1], lane1), (stt[0], lane1)], hue=VOICE)
    s.flow([(stt[0] + stt[1], lane1), (sched_x, lane1)], hue=VOICE)
    s.text(stt[0] + stt[1] + 12, lane1 - 10, "priority 0", size=NOTE, cls="soft")
    s.flow([(agents[0] + agents[1], lane2), (sched_x, lane2)])
    s.text(agents[0] + agents[1] + 12, lane2 - 10, "priority 2", size=NOTE, cls="soft")
    s.flow([(sched_x + sched_w, lane1), (tts[0], lane1)], hue=VOICE)
    s.text(sched_x + sched_w + 12, lane1 - 10, "the reply", size=NOTE, cls="soft")
    s.flow([(tts[0] + tts[1], lane1), (back[0], lane1)], hue=VOICE)
    return bottom


def strip_narrow(s: Figure, y: float) -> float:
    """(a) on a phone: the voice turn's path down a column on green flows, the agents' requests beside it into the
    same scheduler, and the KV pool under the scheduler's right. Returns the panel's lowest edge."""
    x0, w = 16.0, NARROW - 32
    base = _title(s, x0, y, "a", STRIP_TITLE)
    h = 46.0
    col, box_w = 104.0, 176.0  # the voice's column
    side = (col + box_w / 2 + 34, NARROW - 16.0)  # the agents' and the pool's, beside it
    mid = (side[0] + side[1]) / 2
    caller = base + 24
    stt = caller + h + 40
    sched = stt + h + 62
    sched_h = 72.0
    tts = sched + sched_h + 62
    back = tts + h + 40
    s.box(col - box_w / 2, caller, box_w, h, "Caller")
    s.box(col - box_w / 2, stt, box_w, h, "Speech to text")
    s.box(side[0], stt - 6, side[1] - side[0], h + 12, ["24 coding-agent", "sessions"])
    s.box(x0, sched, w, sched_h, SCHEDULER, kind="ours")
    s.box(col - box_w / 2, tts, box_w, h, "Text to speech")
    s.box(col - box_w / 2, back, box_w, h, "Caller")
    pitch = math.floor((side[1] - side[0] - 12) / POOL)
    tray_w = pitch * POOL + 12
    tray = mid - tray_w / 2
    s.memory(tray, tts, tray_w, h)
    s.cells(tray + 6, tts + h / 2 - 12, pool_states()[0], cell=pitch, height=24)
    s.text(mid, tts + h + 24, "KV pool", anchor="middle")
    s.flow([(col, caller + h), (col, stt)], hue=VOICE)
    s.flow([(col, stt + h), (col, sched)], hue=VOICE)
    s.text(col + 10, stt + h + 36, "priority 0", size=NOTE, cls="soft")
    s.flow([(mid, stt + h + 6), (mid, sched)])
    s.text(mid + 10, stt + h + 36, "priority 2", size=NOTE, cls="soft")
    s.flow([(col, sched + sched_h), (col, tts)], hue=VOICE)
    s.text(col + 10, sched + sched_h + 36, "the reply", size=NOTE, cls="soft")
    s.flow([(mid, sched + sched_h), (mid, tts)], arrow=False)
    s.flow([(col, tts + h), (col, back)], hue=VOICE)
    return back + h


# The system diagrams ----------------------------------------------------------------------------------------------
#
# docs/ARCHITECTURE.md's figures of the serving path and ops/RUNBOOK.md's triage, each with a phone layout. They keep
# the method figure's grammar: a component is a box at the style's corner, and the engine, which carries the change,
# is the only blue one; dataflow runs on the neutral flow, and the voice turn's own path on the green one; a host or
# a namespace is a hairline frame with its name inside its top left; facts that are not on the serving path are text
# under a heading, not boxes. In a process diagram the main path runs through its steps on dataflow, down a column
# or, for the release, along a row, and each step's way out leaves it on a gray arrow to what it answers.

STEP_H = 40.0
"""A step's box in a process diagram."""
LINE = 22.0
"""Between the lines of what a step answers."""
EXIT = "gray"
"""The hue of a way out: a comparison's gray, beside the main path's flow."""
SECOND = "second"
"""The flows to the second card: dataflow's color, dashed, as the paths the arrangement has only with it
(DIAGRAM_CSS)."""
UNIT_H = 56.0
"""A unit's box: its name, and under it a note, its socket or its directory."""
HEAD = 36.0
"""From a heading's top to what stands under it."""


def _unit(
    s: Figure,
    x: float,
    y: float,
    w: float,
    name: str | list[str],
    note: str,
    kind: str = "box",
    *,
    dashed: bool = False,
    h: float | None = None,
) -> float:
    """A unit: a component's box, plain or the change's own (``kind`` "ours"), its outline ``dashed`` where the
    arrangement has it only with a second card, its name on one line or several and under it its ``note``, smaller
    and soft, the lines in the middle of a box taller than they need (``h``). Returns the box's height."""
    lines = [name] if isinstance(name, str) else name
    own = UNIT_H + 24 * (len(lines) - 1)
    h = own if h is None else h
    pad = (h - own) / 2
    if dashed:
        s.rect(x, y, w, h, f"{kind} dashed", rx=CORNER)
    else:
        s.box(x, y, w, h, [], kind=kind)
    texts = [(line, y + pad + 24 + 24 * k, LABEL, "") for k, line in enumerate(lines)]
    for text, baseline, size, cls in [*texts, (note, y + pad + own - 11, NOTE, "soft")]:
        box = s.text(x + w / 2, baseline, text, size=size, anchor="middle", cls=cls)
        if box.x0 < x + 6 or box.x1 > x + w - 6:
            raise ValueError(f"{text!r} does not fit inside its box")
    return h


@dataclass(frozen=True)
class Step:
    """A step of a process diagram: its name, the label of its way out (``when``, which may be empty), and what the
    way out answers, a line each as (``lead``, ``text``): a status or an exit code in a column of its own, then the
    rest. A step with no answers has no way out, and ``when`` stands beside it as a note."""

    name: str
    when: str = ""
    answers: tuple[tuple[str, str], ...] = ()
    go: str = ""
    """The label of the main path out of the step, beside its arrow: a decision's other answer."""
    kind: str = "box"
    """The box's kind: "ours" for the one component that carries the change."""


def _answers(
    s: Figure,
    x: float,
    baseline: float,
    answers: tuple[tuple[str, str], ...],
    lead_w: float,
    columns: int = 1,
    column_w: float = 0.0,
) -> float:
    """A way out's answers from ``x``, in ``columns`` columns ``column_w`` apart filled row by row, each lead
    right-aligned in a column ``lead_w`` wide and its text after it. Returns the last baseline."""
    for k, (lead, text) in enumerate(answers):
        ax, y = x + column_w * (k % columns), baseline + LINE * (k // columns)
        if lead:
            s.text(ax + lead_w, y, lead, size=NOTE, anchor="end")
        s.text(ax + lead_w + (8 if lead_w else 0), y, text, size=NOTE)
    return baseline + LINE * ((len(answers) - 1) // columns)


def _lines(text: str, width: float) -> list[str]:
    """``text`` in lines that fit ``width`` at a note's size, broken at its commas first, then at spaces."""
    if measure(text, NOTE) <= width or ", " not in text:
        return wrap(text, width, NOTE)
    parts = text.split(", ")
    clauses = [part + ("," if k + 1 < len(parts) else "") for k, part in enumerate(parts)]
    return [line for clause in clauses for line in wrap(clause, width, NOTE)]


DROP = 32.0
"""A phone layout's way out: an arrow down from the step's box to its answers, its label beside it."""


def _row_height(step: Step, *, below: bool, label_w: float, columns: int) -> float:
    """How far a step's row reaches below its box's top: its box, or its answers' last line."""
    if not step.answers:
        return STEP_H
    if below:
        labels = len(_lines(step.when, label_w)) if step.when else 1
        return STEP_H + DROP + LINE * max(0, labels - 2) + 18 + LINE * (len(step.answers) - 1) + 8
    return max(STEP_H, STEP_H / 2 + 6 + LINE * ((len(step.answers) - 1) // columns) + 8)


def _ladder(
    s: Figure,
    steps: tuple[Step, ...],
    *,
    x: float,
    top: float,
    width: float,
    exit_to: float = 0.0,
    lead_w: float,
    below: bool = False,
    gap: float = 30.0,
    columns: int = 1,
    column_w: float = 0.0,
) -> tuple[list[float], float]:
    """A column of steps from ``top``, each ``width`` wide at ``x``, joined by dataflow. Each way out is an arrow from
    its box's right edge to ``exit_to``, where its answers begin, its label ``when`` above it, the rows one pitch
    apart; or, ``below`` (a phone's layout), an arrow down from the box's left part to its answers, its label beside
    it, while the main path runs down beside them, each row as tall as its answers. Beside the column the answers
    may run in ``columns`` columns ``column_w`` apart. A step with no way out has its note beside its box, or under
    it where the figure is too narrow. Returns each box's top, and the lowest baseline or box edge drawn."""
    main = x + (24 if below else width / 2)
    drop = x + 58
    label_w = s.width - 16 - drop - 12
    heights = [_row_height(step, below=below, label_w=label_w, columns=columns) for step in steps]
    for k, step in enumerate(steps):  # a note with no room beside its box stands under it
        if (
            step.when
            and not step.answers
            and x + width + 14 + measure(step.when, NOTE) > s.width - 16
        ):
            heights[k] = STEP_H + 24 + LINE * (len(_lines(step.when, s.width - 16 - drop)) - 1) + 10
    pitch = max(max(heights) + 14, STEP_H + gap)
    y = top
    lowest = top
    tops = []
    for k, step in enumerate(steps):
        tops.append(y)
        s.box(x, y, width, STEP_H, step.name, kind=step.kind)
        mid = y + STEP_H / 2
        lowest = max(lowest, y + STEP_H)
        if step.answers and below:
            s.flow([(drop, y + STEP_H), (drop, y + STEP_H + DROP)], hue=EXIT)
            labels = _lines(step.when, label_w) if step.when else []
            for j, line in enumerate(labels):
                s.text(drop + 12, y + STEP_H + DROP / 2 + 6 + LINE * j, line, size=NOTE, cls="soft")
            baseline = y + STEP_H + DROP + LINE * max(0, len(labels) - 2) + 18
            lowest = max(lowest, _answers(s, drop - 6, baseline, step.answers, lead_w))
        elif step.answers:
            s.flow([(x + width, mid), (exit_to, mid)], hue=EXIT)
            if step.when:
                s.text(x + width + 14, mid - 9, step.when, size=NOTE, cls="soft")
            last = _answers(s, exit_to + 10, mid + 6, step.answers, lead_w, columns, column_w)
            lowest = max(lowest, last)
        elif step.when and x + width + 14 + measure(step.when, NOTE) <= s.width - 16:
            s.text(x + width + 14, mid + 6, step.when, size=NOTE, cls="mute")
        elif step.when:  # no room beside the box: under it
            for j, line in enumerate(_lines(step.when, s.width - 16 - drop)):
                lowest = y + STEP_H + 24 + LINE * j
                s.text(drop - 6, lowest, line, size=NOTE, cls="mute")
        if k + 1 < len(steps):
            step_to = y + (max(heights[k] + 14, STEP_H + gap) if below else pitch)
            s.flow([(main, y + STEP_H), (main, step_to)])
            if step.go and not below:
                s.text(main + 10, (y + STEP_H + step_to) / 2 + 6, step.go, size=NOTE, cls="mute")
            y = step_to
    return tops, lowest


# The serving architecture -----------------------------------------------------------------------------------------

SERVERS = (
    ("vLLM 0.32 + make-room", "engine.sock", "ours"),
    ("Whisper STT", "stt.sock", "box"),
    ("Kokoro TTS", "tts.sock", "box"),
)
"""A GPU node's servers (deploy/phastform/systemd), each on its Unix socket in $XDG_RUNTIME_DIR/flatshare: the engine,
which runs the change (variant make-room-serve) and is the blue one, the STT and the TTS."""
NODES = (
    ("Phastform", "RTX 5060 Ti", "each server in its own namespace", False),
    ("Gray", "RTX 5060 Ti", "servers in the host's namespace", True),
)
"""The GPU nodes: the name, the card, where its servers' network namespaces are (Phastform's each the server's own,
deploy/bin/in-private-net; Gray's the host's, deploy/gray/private-net-host.conf), and whether it is drawn as the
second card of the two-card arrangement (ADR-0008), dashed, which the gateway reaches only with [[nodes]]: its dashes,
and the caption, say that it is there only when present."""
KEY = "TCP, a tenant's key"
"""How the coding agents reach the gateway: its one TCP listener, where every request needs a tenant's key, since other
users of cyan reach loopback (deploy/cyan/gateway.example.toml; the agents' tenant is [tenants.agents])."""
DIAGRAM_CSS = (
    ".dashed{stroke-dasharray:6 4}.second{--c:var(--flow)}.second.line{stroke-dasharray:6 4}"
)
"""The system diagrams' additions to the figures' style: the dashed outline of a host, a unit or a tunnel the
arrangement has only with a second card, and its flows' dataflow color, dashed."""
FRAME = 8.0
"""A frame's corner, the style's (Figure.group's)."""
ARCHITECTURE_DESCRIPTION = (
    "Hosts as hairline frames, each unit a box on its Unix socket: on cyan, the control host, the callers, LiveKit "
    "and the voice worker run in the voice plane's network namespace, which has no route out; the worker reaches "
    "the gateway through gateway.sock, and the coding agents reach it through 127.0.0.1:18080 with a tenant's key, "
    "the one TCP listener; a tunnel per GPU node carries the gateway's engine requests and the worker's speech "
    "over SSH, a key per socket, to the node's three servers: vLLM 0.32 with make-room on engine.sock, Whisper on "
    "stt.sock and Kokoro on tts.sock, on Phastform each in a network namespace of its own; Gray, drawn dashed, is a "
    "second card when present, its servers in the host's namespace."
)


class Diagram(Figure):
    """A figure_style Figure whose style also holds DIAGRAM_CSS."""

    def render(self) -> str:
        return super().render().replace("</style>", DIAGRAM_CSS + "</style>", 1)


def _box(
    s: Figure, x: float, y: float, w: float, h: float, lines: list[str], *, dashed: bool = False
) -> None:
    """A component's box with its ``lines`` centred inside, its outline ``dashed`` where the arrangement has it only
    with a second card."""
    if not dashed:
        s.box(x, y, w, h, lines)
        return
    s.rect(x, y, w, h, "box dashed", rx=CORNER)
    pitch = LABEL * 1.22
    first = y + h / 2 - pitch * (len(lines) - 1) / 2 + 0.33 * LABEL
    for k, line in enumerate(lines):
        box = s.text(x + w / 2, first + pitch * k, line, anchor="middle")
        if box.x0 < x + 6 or box.x1 > x + w - 6 or box.y0 < y + 3 or box.y1 > y + h - 3:
            raise ValueError(f"{line!r} does not fit inside its box")


def _host(
    s: Figure, x: float, y: float, w: float, h: float, name: str, note: str, *, second: bool = False
) -> float:
    """A host: a hairline frame, dashed for the second card, with its name in the semibold inside its top left and
    ``note``, what it is, at its top right. Returns the name's baseline."""
    if second:
        s.rect(x, y, w, h, "group dashed", rx=FRAME)
        s.borders.append((x, y, w, h))
        baseline = y + 16 + 0.7 * LABEL
        s.text(x + 20, baseline, name, bold=True)
    else:
        baseline = s.panel(x, y, w, h, name)
    if note:
        s.text(x + w - 20, baseline, note, size=NOTE, anchor="end", cls="soft")
    return baseline


NODE_GAP = 14.0
"""Between a GPU node's servers."""


def _node_units(
    s: Figure, x: float, top: float, w: float, names: list[list[str]], second: bool
) -> list[float]:
    """A GPU node's servers in a column from ``top``, dashed on the second card. Returns each one's middle."""
    middles = []
    y = top
    for lines, (_, socket, kind) in zip(names, SERVERS, strict=True):
        h = _unit(s, x, y, w, lines, socket, kind, dashed=second)
        middles.append(y + h / 2)
        y += h + NODE_GAP
    return middles


def architecture(out: Path) -> list[Path]:
    """architecture.svg and its phone layout: the hosts, their units and the sockets between them."""
    title = "The serving architecture"
    return [
        _architecture_wide(out / "architecture.svg", title),
        _architecture_narrow(out / "architecture-narrow.svg", title),
    ]


def _architecture_wide(path: Path, title: str) -> Path:
    """cyan left, the GPU nodes right, one above the other, each reached through its tunnel. The voice plane's own
    flows are green; the gateway's, which carry both classes, and the tunnels' are neutral."""
    s = Diagram(WIDTH, 800, title, ARCHITECTURE_DESCRIPTION)
    nw = 296.0
    nx = WIDTH - EDGE - nw
    names = [[unit] for unit, _, _ in SERVERS]
    head = 78.0
    node_h = head + len(SERVERS) * (UNIT_H + NODE_GAP) - NODE_GAP + 20
    middles = []
    y = 16.0
    for name, card, spaces, second in NODES:
        _host(s, nx, y, nw, node_h, name, card, second=second)
        s.text(nx + 20, y + 60, spaces, size=NOTE, cls="soft")
        middles.append(_node_units(s, nx + 20, y + head, nw - 40, names, second))
        y += node_h + 24
    bottom = y - 24
    # cyan, left: the voice plane, the gateway, the agents and a tunnel per node.
    cyan_right = nx - 52
    _host(s, EDGE, 16, cyan_right - EDGE, bottom - 16, "cyan", "control host, CPU only")
    tw = 112.0
    tx = cyan_right - 16 - tw
    tunnels = []
    for rows, (name, _, _, second) in zip(middles, NODES, strict=True):
        top, low = rows[0] - UNIT_H / 2, rows[-1] + UNIT_H / 2
        tunnels.append((top, low))
        _box(s, tx, top, tw, low - top, ["Tunnel to", name], dashed=second)
        for mid in rows:
            s.flow([(tx + tw, mid), (nx + 20, mid)], hue=SECOND if second else None)
        s.text((cyan_right + nx) / 2, rows[0] - 9, "SSH", size=NOTE, anchor="middle", cls="soft")
    # The voice plane, in its namespace: the callers, LiveKit and the worker in a column. The worker is level with
    # the gateway, between the tunnels, so that its LLM requests run straight to the gateway's socket; its speech
    # leaves its top and turns, under LiveKit, into the first node's tunnel.
    gy = (tunnels[0][1] + tunnels[1][0]) / 2 - 22
    vx, vw = EDGE + 32, 172.0
    speech = tunnels[0][1] - 16
    rows = [speech - 14 - 44 - 34 - 44, speech - 14 - 44, gy]
    space = (EDGE + 16, rows[0] - 62, vw + 32, rows[-1] + 44 + 16 - (rows[0] - 62))
    s.group(*space)
    s.text(space[0] + 14, space[1] + 26, "voice-plane namespace", size=NOTE, cls="soft")
    s.text(space[0] + 14, space[1] + 48, "no route out", size=NOTE, cls="mute")
    for row, name in zip(rows, ("Callers", "LiveKit server", "Voice worker"), strict=True):
        s.box(vx, row, vw, 44, name)
    chain = vx + 56
    s.flow([(chain, rows[0] + 44), (chain, rows[1])], hue=VOICE)
    s.text(chain + 12, rows[0] + 44 + 17 + 0.33 * NOTE, "WebRTC", size=NOTE, cls="soft")
    s.flow([(chain, rows[1] + 44), (chain, rows[2])], hue=VOICE)
    label_x = space[0] + space[2] + 12
    gx = label_x + measure("stt.sock, tts.sock", NOTE) + 14
    gw = 120.0
    s.box(gx, gy, gw, 44, "Gateway")
    s.flow([(vx + vw, gy + 22), (gx, gy + 22)], hue=VOICE)
    s.text(label_x, gy + 13, "gateway.sock", size=NOTE, cls="soft")
    s.flow([(vx + vw - 28, gy), (vx + vw - 28, speech), (tx, speech)], hue=VOICE)
    s.text(label_x, speech - 9, "stt.sock, tts.sock", size=NOTE, cls="soft")
    # The gateway's engine requests, to each node's tunnel.
    split = tx + tw / 2
    s.flow([(gx + gw, gy + 22), (split, gy + 22), (split, tunnels[0][1])])
    s.flow([(split, gy + 22), (split, tunnels[1][0])], hue=SECOND, radius=0)
    s.text(gx + gw + 12, gy + 13, "engine.sock", size=NOTE, cls="soft")
    # The coding agents, in cyan's host network, through the one TCP listener.
    ay = middles[1][1] - 22
    s.box(vx, ay, vw, 44, "Coding agents")
    s.flow([(vx + vw, ay + 22), (gx + gw / 2, ay + 22), (gx + gw / 2, gy + 44)])
    s.text(label_x, ay + 13, "127.0.0.1:18080", size=NOTE, cls="soft")
    s.text(label_x, ay + 22 + 23, KEY, size=NOTE, cls="mute")
    s.height = bottom + 12
    return s.save(path)


def _architecture_narrow(path: Path, title: str) -> Path:
    """The phone layout, read down: cyan above, the voice plane's column over the gateway, which the coding agents
    reach from its side, and a tunnel per node under it; then the GPU nodes, side by side, each under its tunnel."""
    s = Diagram(NARROW, 1600, title, ARCHITECTURE_DESCRIPTION)
    x0, x1 = 16.0, NARROW - 16.0
    mid = NARROW / 2
    nw = (x1 - x0 - 16) / 2
    drops = [x0 + nw / 2, x1 - nw / 2]
    # The voice plane, its units in a column.
    space_top, vw = 58.0, 168.0
    vx = mid - vw / 2
    rows = [space_top + 62 + 76 * k for k in range(3)]
    space_bottom = rows[-1] + 44 + 18
    s.group(x0 + 16, space_top, x1 - x0 - 32, space_bottom - space_top)
    s.text(x0 + 30, space_top + 26, "voice-plane namespace", size=NOTE, cls="soft")
    s.text(x0 + 30, space_top + 48, "no route out", size=NOTE, cls="mute")
    for row, name in zip(rows, ("Callers", "LiveKit server", "Voice worker"), strict=True):
        s.box(vx, row, vw, 44, name)
    s.flow([(mid, rows[0] + 44), (mid, rows[1])], hue=VOICE)
    s.text(mid + 12, rows[0] + 44 + 16 + 0.33 * NOTE, "WebRTC", size=NOTE, cls="soft")
    s.flow([(mid, rows[1] + 44), (mid, rows[2])], hue=VOICE)
    # Under it the gateway, which the worker reaches from above and the coding agents, through the one TCP
    # listener, from its side; and under the gateway a tunnel per node, the worker's speech into the first node's.
    gw, aw, tw, th = 92.0, 88.0, 150.0, 64.0
    gy = space_bottom + 106
    fork = gy + 44 + 40
    ty = fork + 34
    s.box(mid - gw / 2, gy, gw, 44, "Gateway")
    s.flow([(mid, rows[2] + 44), (mid, gy)], hue=VOICE)
    s.text(mid + 12, space_bottom + 24, "gateway.sock", size=NOTE, cls="soft")
    s.box(x1 - 16 - aw, gy - 10, aw, 64, ["Coding", "agents"])
    s.flow([(x1 - 16 - aw, gy + 22), (mid + gw / 2, gy + 22)])
    s.text(x1 - 16, gy - 44, "127.0.0.1:18080", size=NOTE, anchor="end", cls="soft")
    s.text(x1 - 16, gy - 22, KEY, size=NOTE, anchor="end", cls="mute")
    for at, (name, _, _, second) in zip(drops, NODES, strict=True):
        _box(s, at - tw / 2, ty, tw, th, ["Tunnel to", name], dashed=second)
    s.flow([(mid, gy + 44), (mid, fork), (drops[0], fork), (drops[0], ty)])
    s.flow([(mid, fork), (drops[1], fork), (drops[1], ty)], hue=SECOND)
    s.text(mid + 12, gy + 44 + 24, "engine.sock", size=NOTE, cls="soft")
    speech = drops[0] - tw / 2 + 24
    s.flow([(vx, rows[2] + 22), (speech, rows[2] + 22), (speech, ty)], hue=VOICE)
    s.text(speech + 12, space_bottom + 24, "stt.sock,", size=NOTE, cls="soft")
    s.text(speech + 12, space_bottom + 46, "tts.sock", size=NOTE, cls="soft")
    cyan_bottom = ty + th + 18
    _host(s, x0, 12, x1 - x0, cyan_bottom - 12, "cyan", "control host, CPU only")
    # The GPU nodes, side by side, each under its tunnel: its name, its card, where its servers' namespaces are, and
    # its servers, the engine's name on two lines.
    ny = cyan_bottom + 44
    names = [
        ["vLLM 0.32", "+ make-room"] if kind == "ours" else [unit] for unit, _, kind in SERVERS
    ]
    units_h = sum(UNIT_H + 24 * (len(lines) - 1) + NODE_GAP for lines in names) - NODE_GAP
    notes = [[*_lines(card, nw - 36), *_lines(spaces, nw - 36)] for _, card, spaces, _ in NODES]
    top = ny + 70 + LINE * max(len(lines) for lines in notes)
    bottom = top + units_h + 18
    for k, ((name, card, _, second), at) in enumerate(zip(NODES, drops, strict=True)):
        nx = x0 + k * (nw + 16)
        _host(s, nx, ny, nw, bottom - ny, name, "", second=second)
        cards = len(_lines(card, nw - 36))
        for j, note in enumerate(notes[k]):
            s.text(
                nx + 18, ny + 62 + LINE * j, note, size=NOTE, cls="soft" if j < cards else "mute"
            )
        _node_units(s, nx + 18, top, nw - 36, names, second)
        s.flow([(at, ty + th), (at, ny)], hue=SECOND if second else None)
        s.text(at + 12, cyan_bottom + 28, "SSH", size=NOTE, cls="soft")
    s.height = bottom + 12
    return s.save(path)


# The system at a glance --------------------------------------------------------------------------------------------

SERVING_PARTS = (
    (["Voice worker"], "python/flatshare_voice"),
    (["Gateway"], "crates/flatshare-gateway"),
    (["vLLM 0.32 + #63350, #63351", "the KV preemption, the budget cap"], "python/flatshare_vllm"),
)
"""The serving plane's own parts, each with the directory it lives in, drawn by its last name: the voice worker, the
gateway, and the engine with the change, its two rules named as the report names them (docs/REPORT.md §3)."""
OVERVIEW_DESCRIPTION = (
    "The serving path, each flow an arrow: callers reach the voice worker, and the voice worker and the coding "
    "agents reach the gateway, which sends the voice turns at priority 0 and the agents' requests at priority 2 to "
    "vLLM 0.32 with #63350 and #63351, the KV preemption and the budget cap, over the GPU's KV pool and the CPU "
    "tier; under each part's name is its directory in the repository."
)
MEMORY = "GPU KV pool · CPU tier"
"""The engine's memory, under it (docs/REPORT.md §5.1)."""
PRIORITIES = ("priority 0", "priority 2")
"""What the gateway gives the voice turns and the agents' requests (docs/ARCHITECTURE.md, At a glance)."""
RISE = 32.0
"""Between the serving path's boxes on a phone: a flow's length, and room for its label beside it."""
HOP = 52.0
"""Between the serving path's boxes in the wide layout: a flow's length."""
TRAY_H = 44.0
"""The engine's memory, a tray under it."""


def overview(out: Path) -> list[Path]:
    """overview.svg and its phone layout: the serving path at a glance."""
    title = "The serving path at a glance"
    return [
        _overview_wide(out / "overview.svg", title),
        _overview_narrow(out / "overview-narrow.svg", title),
    ]


def _serving(s: Figure, x: float, y: float, w: float) -> float:
    """The serving path from ``y``, each flow downward: the callers and under them the voice worker, on the voice
    turn's green; the coding agents beside them; the gateway, which sends both on at their priorities; the engine
    with the change, and its memory under it. Returns the path's lowest edge."""
    half = (w - 16) / 2
    left, right = x + half / 2, x + w - half / 2
    (worker, worker_dir), (gateway, gateway_dir), (engine, engine_dir) = SERVING_PARTS
    s.box(x, y, half, 44, "Callers")
    s.box(x + w - half, y, half, 44, "Coding agents")
    worker_y = y + 44 + RISE
    gateway_y = worker_y + _unit(s, x, worker_y, half, worker, Path(worker_dir).name) + RISE
    engine_y = gateway_y + _unit(s, x, gateway_y, w, gateway, Path(gateway_dir).name) + RISE
    memory_y = engine_y + _unit(s, x, engine_y, w, engine, Path(engine_dir).name, "ours") + 16
    s.memory(x, memory_y, w, TRAY_H)
    s.text(x + w / 2, memory_y + TRAY_H / 2 + 0.33 * LABEL, MEMORY, anchor="middle")
    s.flow([(left, y + 44), (left, worker_y)], hue=VOICE)
    s.flow([(left, gateway_y - RISE), (left, gateway_y)], hue=VOICE)
    s.flow([(right, y + 44), (right, gateway_y)])
    for at, priority, hue in zip((left, right), PRIORITIES, (VOICE, None), strict=True):
        s.flow([(at, engine_y - RISE), (at, engine_y)], hue=hue)
        s.text(at + 12, engine_y - RISE / 2 + 0.33 * NOTE, priority, size=NOTE, cls="soft")
    s.flow([(x + w / 2, memory_y - 16), (x + w / 2, memory_y)], arrow=False)
    return memory_y + TRAY_H


def _overview_wide(path: Path, title: str) -> Path:
    """The serving path left to right across the page: the callers and the voice worker on the upper lane, on the
    voice turn's green, and the coding agents on the lower; the gateway, as tall as both, which sends each on at its
    priority; and the engine with the change, its memory under it, the two as tall as the gateway."""
    s = Diagram(WIDTH, 400, title, OVERVIEW_DESCRIPTION)
    (worker, worker_dir), (gateway, gateway_dir), (engine, engine_dir) = SERVING_PARTS
    top = 16.0
    engine_h = UNIT_H + 24 * (len(engine) - 1)
    low = top + engine_h + 8 + TRAY_H
    upper, lower = top + UNIT_H / 2, low - UNIT_H / 2
    source_w = measure("Coding agents", LABEL) + 38
    worker_w = max(measure(worker[0], LABEL), measure(Path(worker_dir).name, NOTE)) + 44
    gateway_w = measure(Path(gateway_dir).name, NOTE) + 38
    engine_w = max(measure(line, LABEL) for line in engine) + 48
    engine_x = WIDTH - EDGE - engine_w
    send = engine_x - (EDGE + source_w + HOP + worker_w + HOP + gateway_w)
    worker_x = EDGE + source_w + HOP
    gateway_x = worker_x + worker_w + HOP
    assert send >= measure(PRIORITIES[0], NOTE) + 36, send  # room for a priority over its flow
    s.box(EDGE, top, source_w, UNIT_H, "Callers")
    s.box(EDGE, low - UNIT_H, source_w, UNIT_H, "Coding agents")
    _unit(s, worker_x, top, worker_w, worker, Path(worker_dir).name)
    _unit(s, gateway_x, top, gateway_w, gateway, Path(gateway_dir).name, h=low - top)
    _unit(s, engine_x, top, engine_w, engine, Path(engine_dir).name, "ours")
    s.memory(engine_x, low - TRAY_H, engine_w, TRAY_H)
    s.text(engine_x + engine_w / 2, low - TRAY_H / 2 + 0.33 * LABEL, MEMORY, anchor="middle")
    s.flow([(EDGE + source_w, upper), (worker_x, upper)], hue=VOICE)
    s.flow([(worker_x + worker_w, upper), (gateway_x, upper)], hue=VOICE)
    s.flow([(EDGE + source_w, lower), (gateway_x, lower)])
    # The gateway sends both on into the engine: the voice turn level with its lane, its priority over its flow, and
    # the agents' requests under it, theirs under their flow.
    sent = ((upper, -10.0), (top + engine_h - 24, 10 + 0.7 * NOTE))
    for (at, label), priority, hue in zip(sent, PRIORITIES, (VOICE, None), strict=True):
        s.flow([(gateway_x + gateway_w, at), (engine_x, at)], hue=hue)
        s.text(gateway_x + gateway_w + 12, at + label, priority, size=NOTE, cls="soft")
    s.height = math.ceil(low + 16)
    return s.save(path)


def _overview_narrow(path: Path, title: str) -> Path:
    """The phone layout: the serving path down the page."""
    s = Diagram(NARROW, 700, title, OVERVIEW_DESCRIPTION)
    low = _serving(s, 16.0, 16.0, NARROW - 32.0)
    s.height = math.ceil(low + 16)
    return s.save(path)


# A request's path through the gateway ------------------------------------------------------------------------------

REQUEST_STEPS = (
    Step(
        "Authenticate",
        "no known key, an unknown class",
        (("401", "invalid_api_key"), ("400", "invalid_class")),
    ),
    Step(
        "Bound",
        "a bound full, a body too large",
        (("503", "in_flight_limit"), ("429", "tenant_in_flight_limit"), ("413", "body_too_large")),
    ),
    Step(
        "Sanitize",
        "no JSON object, too many choices",
        (("400", "invalid_json"), ("400", "too_many_choices")),
    ),
    Step(
        "Place",
        "no engine ready, no free slot",
        (
            ("503", "upstream_not_ready"),
            ("503", "no_healthy_node"),
            ("503", "no_call_slot"),
            ("503", "no_agent_slot"),
        ),
    ),
    Step(
        "Forward",
        "no answer, no headers in time",
        (("502", "upstream_unavailable"), ("504", "upstream_timeout")),
    ),
    Step(
        "Relay",
        "a stream cut short: an error event",
        (("", "upstream_unavailable"), ("", "upstream_timeout"), ("", "draining")),
    ),
    Step("Account", "one outcome, every request"),
)
"""The gateway's path for an inference request, in the order crates/flatshare-gateway/src/proxy.rs takes it, and
each step's refusals as the gateway README's error table names them: the status and the code, or the error event's
type once a stream has begun."""
ENGINE = Step("Engine", "sees no request the first four steps refuse", kind="ours")
"""Where the engine stands on the path, between the step that forwards to it and the step that relays its stream: the
one blue box, and beside it what the four steps before the forwarding spare it."""
FORWARDED = 5
"""How many of REQUEST_STEPS come before the engine: all up to the forwarding."""
REQUEST_DESCRIPTION = (
    "A column of seven steps a request takes through the gateway, each with its main refusals, which the "
    "gateway's error table gives in full: authenticate "
    "(401 invalid_api_key, 400 invalid_class), bound (503 in_flight_limit, 429 tenant_in_flight_limit, 413 "
    "body_too_large), sanitize (400 invalid_json, 400 too_many_choices), place (503 upstream_not_ready, "
    "no_healthy_node, no_call_slot or no_agent_slot), forward (502 upstream_unavailable, 504 upstream_timeout), "
    "then the engine, the one blue box, which sees no request that the first four steps refuse, relay (an error "
    "event of type upstream_unavailable, upstream_timeout or draining once a stream has begun), and account, which "
    "records one outcome for every request."
)


def request_path(out: Path) -> list[Path]:
    """request-path.svg and its phone layout: a request's path through the gateway, each step's refusals beside it,
    and the engine where the path reaches it."""
    title = "A request's path through the gateway"
    steps = (*REQUEST_STEPS[:FORWARDED], ENGINE, *REQUEST_STEPS[FORWARDED:])
    wide = Diagram(WIDTH, 800, title, REQUEST_DESCRIPTION)
    wide.text(24 + 100, 30, "a request", size=NOTE, anchor="middle", cls="soft")
    wide.flow([(124, 40), (124, 64)])
    _, lowest = _ladder(
        wide,
        steps,
        x=24,
        top=64,
        width=200,
        exit_to=560,
        lead_w=36,
        columns=2,
        column_w=270,
    )
    wide.height = lowest + 24
    narrow = Diagram(NARROW, 1200, title, REQUEST_DESCRIPTION)
    narrow.text(16, 30, "a request", size=NOTE, cls="soft")
    narrow.flow([(40, 40), (40, 64)])
    narrow.height = _ladder(narrow, steps, x=16, top=64, width=200, lead_w=34, below=True)[1] + 24
    return [wide.save(out / "request-path.svg"), narrow.save(out / "request-path-narrow.svg")]


# Two cards -------------------------------------------------------------------------------------------------------

CARDS = (("Phastform", 2, 4), ("Gray", 2, 12))
"""The two cards as deploy/cyan/gateway.example.toml's [[nodes]] give them, in placement order: the name, max_calls
and max_agents. With calls = "fill" a call takes the first card with a free call slot."""
FILL = ("calls fill in this order",)
"""What (a)'s arrow from the first card to the second says: with calls = "fill", the file's order."""
MOVED = ("next turn: 0.7 s after a kill,", "1.9 s after a stop")
"""When both calls reached the spare in the failover drills (benchmarks/results/2026-10-28-node-failover): what (b)'s
arrow, the calls' own, says."""
HELD = "held until the calls end"
"""What the spare's hatched agents' slots mean in (b): a node that covers holds its agents' requests to
failover_background_max_in_flight, 0 in deploy/cyan/gateway.example.toml, until the moved calls end
(crates/flatshare-gateway/README.md, Nodes)."""
CARD_STATES = (
    ("a", "In service", ((2, "run", "the calls"), (0, "run", "warm spare")), FILL, False),
    (
        "b",
        "The first card failed",
        ((0, None, "engine down"), (2, "held", "covering")),
        MOVED,
        True,
    ),
)
"""The two panels: the letter, the title, each card's state (its calls, its agents' slots running, held or free,
and its note), what the arrow between the cards says, and whether the arrow is the calls moving."""
ROW_NAME = 66.0
"""A card's column of row names, before its slots."""
CARDS_DESCRIPTION = (
    "Two panels of the two cards' slots as the gateway bounds them, a tray a card, two call slots each and up to 4 "
    "agents' sessions on Phastform and 12 on Gray: in panel a, in service, both calls fill Phastform, the first "
    "card in the file's order, and Gray, the warm spare, runs agents only; in panel b, Phastform's engine failed, "
    "both calls take Gray's call slots at their next turn, 0.7 s after a kill and 1.9 s after a stop in the "
    "drills, and Gray holds its agents' requests until the moved calls end."
)


def _card_size(card: tuple[str, int, int], pitch: float, slot: float) -> tuple[float, float]:
    """A card's tray, across and down: its row names and its agents' slots at ``pitch``; its name, then two rows of
    slots ``slot`` high."""
    return 32 + ROW_NAME + card[2] * pitch - 2, 44 + 2 * slot + 10 + 16


def _card(
    s: Figure,
    x: float,
    y: float,
    w: float,
    card: tuple[str, int, int],
    state: tuple[int, str | None, str],
    *,
    pitch: float,
    slot: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """A card: a tray with its name and its state's note, a row of its call slots, as many taken as the state says,
    and a row of its agents' slots, running ("run"), held back ("held", hatched) or free (None). Returns the middle
    of its call slots, across and down, and its agents' row's left and right edges."""
    name, max_calls, max_agents = card
    calls, agents, note = state
    s.memory(x, y, w, _card_size(card, pitch, slot)[1])
    named = s.text(x + 16, y + 30, name)
    s.text(named.x1 + 12, y + 30, note, size=NOTE, cls="soft")
    left = x + 16 + ROW_NAME
    rows = (
        ("calls", max_calls, calls, VOICE),
        ("agents", max_agents, max_agents if agents else 0, AGENT),
    )
    for k, (label, count, taken, hue) in enumerate(rows):
        top = y + 44 + (slot + 10) * k
        s.text(x + 16, top + slot / 2 + 0.33 * NOTE, label, size=NOTE, cls="soft")
        s.cells(
            left,
            top,
            [hue] * taken + [None] * (count - taken),
            cell=pitch - 2,
            height=slot,
            gap=2,
            hatched=frozenset(range(count)) if agents == "held" and hue == AGENT else frozenset(),
            strong=(VOICE,),
        )
    return (left + max_calls * pitch / 2, y + 44 + slot / 2), (
        left + 1,
        left + max_agents * pitch - 3,
    )


def two_cards(out: Path) -> list[Path]:
    """two-cards.svg and its phone layout: the two cards, (a) in service and (b) after the first failed, its calls on
    the spare."""
    title = "Two cards: placement and failover"
    return [
        _cards_wide(out / "two-cards.svg", title),
        _cards_narrow(out / "two-cards-narrow.svg", title),
    ]


def _cards_wide(path: Path, title: str) -> Path:
    """The panels one under another, each the two cards side by side in the file's order, a tray as wide as its
    slots, and between them the arrow at the call slots' row with what it says under it."""
    s = Diagram(WIDTH, 800, title, CARDS_DESCRIPTION)
    pitch, slot = 38.0, 30.0
    sizes = [_card_size(card, pitch, slot) for card in CARDS]
    xs = (EDGE, WIDTH - EDGE - sizes[1][0])
    y = 16.0
    for letter, heading, states, lines, moved in CARD_STATES:
        _title(s, EDGE, y, letter, heading)
        top = y + HEAD
        marks = [
            _card(s, x, top, w, card, state, pitch=pitch, slot=slot)
            for x, (w, _), card, state in zip(xs, sizes, CARDS, states, strict=True)
        ]
        (_, row), _ = marks[0]
        a, b = xs[0] + sizes[0][0], xs[1]
        s.flow([(a, row), (b, row)], hue=VOICE if moved else None)
        for j, line in enumerate(lines):
            s.text((a + b) / 2, row + 26 + LINE * j, line, size=NOTE, anchor="middle", cls="soft")
        y = top + sizes[0][1]
        if moved:
            y = s.bracket(*marks[1][1], y + 10, HELD).y1
        y += BETWEEN
    s.height = math.ceil(y - BETWEEN + 12)
    return s.save(path)


def _cards_narrow(path: Path, title: str) -> Path:
    """The phone layout: the panels one under another, each the two cards one under the other at the column's width,
    and between them the arrow from the call slots with what it says beside it."""
    s = Diagram(NARROW, 1400, title, CARDS_DESCRIPTION)
    x, w, slot = 16.0, NARROW - 32.0, 26.0
    pitch = float(math.floor((w - 32 - ROW_NAME) / max(card[2] for card in CARDS)))
    h = _card_size(CARDS[0], pitch, slot)[1]
    y = 16.0
    for letter, heading, states, lines, moved in CARD_STATES:
        _title(s, x, y, letter, heading)
        gap = 28 + LINE * len(lines)
        tops = (y + HEAD, y + HEAD + h + gap)
        marks = [
            _card(s, x, top, w, card, state, pitch=pitch, slot=slot)
            for top, card, state in zip(tops, CARDS, states, strict=True)
        ]
        (at, _), _ = marks[0]
        s.flow([(at, tops[0] + h), (at, tops[1])], hue=VOICE if moved else None)
        first = tops[0] + h + gap / 2 - LINE * (len(lines) - 1) / 2 + 0.33 * NOTE
        for j, line in enumerate(lines):
            s.text(at + 14, first + LINE * j, line, size=NOTE, cls="soft")
        y = tops[1] + h
        if moved:
            y = s.bracket(*marks[1][1], y + 10, HELD).y1
        y += BETWEEN
    s.height = math.ceil(y - BETWEEN + 12)
    return s.save(path)


# The release -----------------------------------------------------------------------------------------------------

RELEASE_STEPS = (
    Step("Prepare", "the host locked, the upload changed", (("exit 2", "nothing switched"),)),
    Step("Verify", "a check fails", (("exit 1", "nothing switched"),)),
    Step("Drain", "calls do not end, an E2 point runs", (("exit 2", "nothing switched"),)),
    Step("Switch"),
    Step("Restart"),
    Step("Check"),
)
"""deploy/deploy.sh HOST's release, as deploy/bin/flatshare-release deploy takes it (ops/RUNBOOK.md's Deploy, steps 1
to 7), and its exits before the switch: 2 for a host or an upload it cannot work with and a drain that does not end
(NotReady), 1 for a check that fails (ReleaseError). From the switch on, any failure switches back."""
SWITCH_BACK = (
    ("exit 1", "the previous release runs again"),
    ("exit 3", "switching back failed too"),
)
"""What a switch back ends with: the previous release up and checked, or not."""
ENTRIES = ("deploy.sh HOST", "--rollback")
"""How a release is entered: a deploy at its first step, a rollback at the second, with the previous release."""
SERVED = ("Serving", "exit 0")
"""A release that passes its checks: its state and its exit."""
FAILURE = "a failure"
"""What takes a release from any of the three states after the switch to the switch back."""
RELEASE_DESCRIPTION = (
    "The release's states in order, prepare, verify, drain, switch, restart, check and serving: a host that is "
    "locked or an upload that changed exits 2, a failed verification exits 1, and a drain whose calls do not end or "
    "that meets a running E2 point exits 2, each with nothing switched; any failure from the switch through the check "
    "switches back, which exits 1 with the previous release running again or 3 when switching back fails too; a "
    "release that passes its checks serves "
    "and exits 0, and a rollback enters at the verification with the previous release."
)


def release(out: Path) -> list[Path]:
    """release.svg and its phone layout: a release's states and its exits, the switch back by itself."""
    title = "A release and its exits"
    return [
        _release_wide(out / "release.svg", title),
        _release_narrow(out / "release-narrow.svg", title),
    ]


def _release_wide(path: Path, title: str) -> Path:
    """The states in a row, the main path left to right. Under the three before the switch, each one's way out, and
    one bracket under all three for what they share; the three from the switch on join into the switch back."""
    s = Diagram(WIDTH, 600, title, RELEASE_DESCRIPTION)
    names = [step.name for step in RELEASE_STEPS] + [SERVED[0]]
    gap = 30.0
    before = [(k, step) for k, step in enumerate(RELEASE_STEPS) if step.answers]
    reasons = [_lines(step.when, 160) for _, step in before]
    # The margin that keeps the widest reason, centred under the first state, inside the figure's edge.
    reach = max(measure(line, NOTE) for lines in reasons for line in lines)
    room = WIDTH - gap * (len(names) - 1)
    margin = max(EDGE, (EDGE + reach / 2 - room / (2 * len(names))) * len(names) / (len(names) - 1))
    w = (room - 2 * margin) / len(names)
    xs = [margin + (w + gap) * k for k in range(len(names))]
    top = 64.0
    mid, bottom = top + STEP_H / 2, top + STEP_H
    for k, (x, name) in enumerate(zip(xs, names, strict=True)):
        s.box(x, top, w, STEP_H, name)
        if k:
            s.flow([(xs[k - 1] + w, mid), (x, mid)])
    for x, entry in zip(xs, ENTRIES, strict=False):
        s.text(x + w / 2, 30, entry, size=NOTE, anchor="middle", cls="soft")
        s.flow([(x + w / 2, 40), (x + w / 2, top)])
    # Before the switch: each state's way out, and what all three leave behind.
    code = bottom + DROP + 20 + LINE * max(len(lines) for lines in reasons) + 2
    for (k, step), lines in zip(before, reasons, strict=True):
        x = xs[k]
        s.flow([(x + w / 2, bottom), (x + w / 2, bottom + DROP)], hue=EXIT)
        for j, line in enumerate(lines):
            s.text(
                x + w / 2,
                bottom + DROP + 20 + LINE * j,
                line,
                size=NOTE,
                anchor="middle",
                cls="soft",
            )
        s.text(x + w / 2, code, step.answers[0][0], size=NOTE, anchor="middle")
    kept = s.bracket(xs[before[0][0]], xs[before[-1][0]] + w, code + 12, before[0][1].answers[0][1])
    # From the switch on: any failure switches back.
    switch, restart, check = (xs[k] + w / 2 for k in (3, 4, 5))
    bus = bottom + 22
    back = bus + 48
    s.flow([(check, bottom), (check, bus), (switch, bus)], arrow=False, hue=EXIT)
    s.flow([(restart, bottom), (restart, bus)], arrow=False, hue=EXIT)
    s.flow([(switch, bottom), (switch, back)], hue=EXIT)
    s.text(switch + 12, bus + 30, FAILURE, size=NOTE, cls="soft")
    s.box(xs[3], back, w, STEP_H, "Switch back")
    s.flow([(xs[3] + w, back + STEP_H / 2), (xs[4], back + STEP_H / 2)], hue=EXIT)
    last = _answers(s, xs[4] + 8, back + STEP_H / 2 + 6, SWITCH_BACK, 46)
    s.text(xs[6] + w / 2, bottom + 26, SERVED[1], size=NOTE, anchor="middle")
    s.height = max(kept.y1, last + 6) + 22
    return s.save(path)


def _release_narrow(path: Path, title: str) -> Path:
    """The states down a column, each way out under its state; the three from the switch on join at their right
    into the switch back, and the main path runs on down the left to the last state."""
    s = Diagram(NARROW, 1200, title, RELEASE_DESCRIPTION)
    x, w = 16.0, 200.0
    main = x + 24
    s.text(x, 30, ENTRIES[0], size=NOTE, cls="soft")
    s.flow([(main, 40), (main, 64)])
    tops, lowest = _ladder(s, RELEASE_STEPS, x=x, top=64, width=w, lead_w=46, below=True)
    verify = tops[1] + STEP_H / 2
    tail = x + w + 48
    s.flow([(tail, verify), (x + w, verify)])
    s.text(tail + 8, verify + 6, ENTRIES[1], size=NOTE, cls="soft")
    switch, restart, check = (tops[k] + STEP_H / 2 for k in (3, 4, 5))
    spine = x + w + 22
    back_w = 150.0
    back = check + STEP_H / 2 + 58
    s.flow([(x + w, switch), (spine, switch), (spine, back)], hue=EXIT)
    s.flow([(x + w, restart), (spine, restart)], arrow=False, hue=EXIT)
    s.flow([(x + w, check), (spine, check)], arrow=False, hue=EXIT)
    s.text(spine + 12, check + STEP_H / 2 + 26, FAILURE, size=NOTE, cls="soft")
    s.box(spine - back_w / 2, back, back_w, STEP_H, "Switch back")
    drop = spine - back_w / 2 + 24
    s.flow([(drop, back + STEP_H), (drop, back + STEP_H + 26)], hue=EXIT)
    last = _answers(s, x + 58, back + STEP_H + 26 + 22, SWITCH_BACK, 46)
    serving = last + 26
    s.flow([(main, tops[5] + STEP_H), (main, serving)])
    s.box(x, serving, w, STEP_H, SERVED[0])
    s.text(x + w + 14, serving + STEP_H / 2 + 6, SERVED[1], size=NOTE)
    s.height = max(lowest, serving + STEP_H) + 24
    return s.save(path)


# Triage ----------------------------------------------------------------------------------------------------------

TRIAGE_STEPS = (
    Step("Gateway up?", "no", (("", "FlatshareGatewayDown"),), go="yes"),
    Step(
        "/ready 200?",
        "no: not ready, degraded or warming",
        (
            ("", "FlatshareEngineNotReady"),
            ("", "FlatshareEngineDegraded"),
            ("", "FlatshareEngineWarming"),
        ),
        go="yes",
    ),
    Step(
        "Nodes healthy?",
        "no: a node unhealthy or covering",
        (("", "FlatshareNodeUnhealthy"), ("", "FlatshareNodeCovering")),
        go="yes",
    ),
    Step(
        "Requests refused?",
        "yes: a bound full, no slot",
        (
            ("", "FlatshareRequestsRefusedForCapacity"),
            ("", "FlatshareCallsRefused"),
            ("", "503 no_agent_slot"),
        ),
        go="no",
    ),
    Step(
        "KV pool full?",
        "yes, with requests waiting",
        (("", "FlatshareKvPoolFullWithWaiting"),),
        go="no",
    ),
    Step("Voice plane", "the worker's journal, LiveKit's"),
)
"""From a call that fails or waits to its row of ops/RUNBOOK.md's Failures, in five checks: whether Prometheus reads
the gateway, the gateway's /ready, its /nodes, its refusals by outcome (flatshare:requests_terminal:rate5m) and the
engines' KV pools with requests waiting; each check that finds its fault names the rows, by the alerts that open
them (ops/prometheus/rules.yml), and what is left is the voice plane's."""
TRIAGE_DESCRIPTION = (
    "A column of five checks from a call that fails or waits, each leading to the runbook rows it "
    "names: the gateway down (FlatshareGatewayDown); /ready not 200, the engine not ready, degraded or warming "
    "(FlatshareEngineNotReady, FlatshareEngineDegraded, FlatshareEngineWarming); a node unhealthy or covering "
    "(FlatshareNodeUnhealthy, FlatshareNodeCovering); refusals for a full bound or no slot "
    "(FlatshareRequestsRefusedForCapacity, FlatshareCallsRefused, 503 no_agent_slot); a full KV pool with "
    "requests waiting (FlatshareKvPoolFullWithWaiting); and, when all five pass, the voice plane: the worker's "
    "journal and LiveKit's."
)


def triage(out: Path) -> list[Path]:
    """triage.svg and its phone layout: five checks from a call that fails or waits to its runbook row."""
    title = "From a failing call to its runbook row"
    wide = Diagram(WIDTH, 900, title, TRIAGE_DESCRIPTION)
    wide.text(24 + 110, 30, "a call fails or waits", size=NOTE, anchor="middle", cls="soft")
    wide.flow([(134, 40), (134, 64)])
    wide.height = (
        _ladder(wide, TRIAGE_STEPS, x=24, top=64, width=220, exit_to=720, lead_w=0)[1] + 24
    )
    narrow = Diagram(NARROW, 1400, title, TRIAGE_DESCRIPTION)
    narrow.text(16, 30, "a call fails or waits", size=NOTE, cls="soft")
    narrow.flow([(40, 40), (40, 64)])
    narrow.height = (
        _ladder(narrow, TRIAGE_STEPS, x=16, top=64, width=200, lead_w=0, below=True)[1] + 24
    )
    return [wide.save(out / "triage.svg"), narrow.save(out / "triage-narrow.svg")]


def render(out_dir: Path) -> list[Path]:
    """Every diagram in ``out_dir``, each with its phone layout: the method figure, then the system diagrams."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _wide(out_dir / "method.svg"),
        _narrow(out_dir / "method-narrow.svg"),
        *(path for diagram in SYSTEM for path in diagram(out_dir)),
    ]


METHOD_TITLE = "How #63350 and #63351 make room for a voice turn"
DESCRIPTION = (
    "Panel a: a caller's speech goes to speech to text, then to vLLM 0.32's scheduler, which carries #63350 "
    "and #63351, then to text to speech and back to the caller; 24 coding-agent sessions send their requests "
    "to the same scheduler at priority 2, the voice turn's at priority 0; the scheduler allocates from the KV "
    f"pool beside it. Panel b, the KV preemption, #63350: in a pool of {POOL} KV blocks, two agent requests "
    f"hold 5 each and 2 are free, fewer than the {TURN} a voice turn needs; on vLLM 0.32 the turn waits outside "
    "the pool, while the KV preemption preempts the latest agent request, agent B, which waits instead, and the "
    f"turn takes {TURN} of its blocks. Panel c, the budget cap, #63351: the engine's steps in time, one green "
    "block a step, from the reply's first token: on vLLM 0.32 each step of the reply also carries an agent's "
    f"prefill chunk, so in the time the budget cap gives the reply its first {WINDOW} tokens the reply gets 14 "
    "without it; after the window, agent prefill resumes. Blocks and steps are illustrative."
)
BETWEEN = 34.0
"""Between the method figure's panels: from one's lowest edge to the next one's title."""


def _wide(path: Path) -> Path:
    """(a), (b) and (c) one under another, (b)'s and (c)'s rows named in one column."""
    s = Figure(WIDTH, 900, METHOD_TITLE, DESCRIPTION)
    low = strip(s, 16)
    low = make_room_panel(s, EDGE, low + BETWEEN, WIDTH - 2 * EDGE)
    low = hold_prefill_panel(s, EDGE, low + BETWEEN, WIDTH - 2 * EDGE)
    s.height = math.ceil(low + 12)
    return s.save(path)


def _narrow(path: Path) -> Path:
    """The phone layout: the same three panels, each row's name over it."""
    s = Figure(NARROW, 1600, METHOD_TITLE, DESCRIPTION)
    x0, w = 16.0, NARROW - 32
    low = strip_narrow(s, 16)
    low = make_room_panel(s, x0, low + BETWEEN, w, stacked=True)
    low = hold_prefill_panel(s, x0, low + BETWEEN, w, stacked=True)
    s.height = math.ceil(low + 12)
    return s.save(path)


SYSTEM = (overview, architecture, request_path, two_cards, release, triage)
"""The system diagrams, in docs/ARCHITECTURE.md's order, then the runbook's."""


if __name__ == "__main__":
    for written in render(Path(sys.argv[1])):
        print(written)
