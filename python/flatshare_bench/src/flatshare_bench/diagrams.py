"""The docs' diagrams, each with its phone layout, in the figures' style (:mod:`flatshare_bench.figure_style`):

    python -m flatshare_bench.diagrams OUT_DIR

The method figure, docs/figures/method.svg and method-narrow.svg, the README's and the report's: a voice agent and
coding agents on one GPU, and the two scheduler rules of vLLM #63350 and #63351. (a) is a stage strip: a caller's turn
through speech to text, vLLM's scheduler and text to speech, beside the coding agents' requests, the KV pool the
scheduler allocates from below it. Its dashed outlines zoom into the panels below: the pool into (b), #63350's
make-room, and the scheduler into (c), #63351's hold on agent prefill.

The blocks and steps are illustrative, and obey the rules they show (docs/REPORT.md §3). Make-room
preempts lower-priority running requests in vLLM's own order, the latest arrival first, as few as make the
waiting request fit; render() works out its victims with :func:`make_room` and draws what it returns. The
budget cap holds lower-priority prefill from the step the turn fits until it has its first tokens, and never
cuts a decode. A step's length is drawn in the ratio of the recorded gaps between a reply's tokens beside
agent prefill and with it held (the README's "What it shows": 60-65 ms and 16-20 ms), at the middle of each.

The system diagrams, square-cornered: docs/ARCHITECTURE.md's figures of the serving path, architecture.svg (the hosts,
their units and the sockets between them), request-path.svg (a request's steps through the gateway and each one's
refusals), two-cards.svg (the two-card arrangement's placement and its measured failover) and release.svg (a release's
states and exits), and ops/RUNBOOK.md's triage.svg (five checks from a failing call to its runbook row). Each draws
what the files it depicts say, which tests/test_diagrams.py holds it to.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from flatshare_bench.figure_style import (
    LABEL,
    NARROW,
    NOTE,
    VALUE,
    WIDTH,
    Figure,
    fmt,
    measure,
    wrap,
)

VOICE, AGENT = "green", "gray"
"""The voice turn's hue and the agents' (blue is the change's own: its components)."""
POOL = 12
"""KV blocks in (b)'s pool."""
HOLDERS = (("agent A", 5), ("agent B", 5))
"""The running agent requests, in the order they arrived, and the blocks each holds."""
TURN = 4
"""The blocks the waiting voice turn needs for its whole sequence."""
WINDOW = 48
"""The reply's first tokens during which #63351 holds agent prefill: the configuration to deploy's."""
DECODE_MS, SHARED_MS = 18.0, 62.0
"""A step that carries only the reply's decode, and one that carries an agent's prefill chunk beside it: the
middles of the recorded 16-20 ms and 60-65 ms gaps between the reply's tokens."""
RESUMED = 2
"""Steps drawn after the window, where agent prefill shares the steps again."""


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


def _cell(s: Figure, x: float, y: float, width: float, height: float, kind: str | None) -> None:
    """One KV block: an agent's (gray), the voice turn's (green), preempted (an agent's, hatched) or free."""
    if kind is None:
        s.rect(x, y, width, height, "cell-empty")
    elif kind == "preempted":
        s.cells(x, y, [AGENT], cell=width, height=height, hatched=frozenset({0}))
    else:
        s.rect(x, y, width, height, f"{kind} cell")


def _pool(
    s: Figure, x: float, y: float, kinds: list[str | None], pitch: float, height: float
) -> float:
    """The KV pool's blocks inside its dashed outline, from ``x``; returns the outline's right edge."""
    s.memory(x - 6, y - 6, pitch * len(kinds) + 8, height + 12)
    for k, kind in enumerate(kinds):
        _cell(s, x + pitch * k, y, pitch - 4, height, kind)
    return x - 6 + pitch * len(kinds) + 8


def make_room_panel(
    s: Figure, x: float, y: float, w: float, h: float, *, ground: str = "a"
) -> None:
    """(b): the pool full of two agent requests' blocks while a voice turn waits beside it, then make-room
    preempting the latest of them, and the turn in its blocks."""
    held = sum(blocks for _, blocks in HOLDERS)
    free = POOL - held
    victims = make_room(free, HOLDERS, TURN)
    assert free < TURN and victims == ["agent B"], (free, victims)  # what the panel shows
    spans, start = {}, 0
    for name, blocks in HOLDERS:
        spans[name] = (start, start + blocks)
        start += blocks
    before: list[str | None] = [AGENT] * held + [None] * free
    preempted: list[str | None] = [
        "preempted" if any(a <= k < b for v in victims for a, b in [spans[v]]) else kind
        for k, kind in enumerate(before)
    ]
    taken = [k for k, kind in enumerate(preempted) if kind == "preempted"][:TURN]
    after: list[str | None] = [
        VOICE if k in taken else (None if kind == "preempted" else kind)
        for k, kind in enumerate(preempted)
    ]

    s.panel(x, y, w, h, "#63350: make room for KV", "b", ground=ground)
    x0 = x + 26
    pitch = min(26.0, (w - 52 - 30) / (POOL + TURN))
    height = 24
    col = [x0 + pitch * k for k in range(POOL + 1)]
    turn_x = col[POOL] + 30  # the waiting turn's blocks, beside the pool

    def span(a: int, b: int) -> tuple[float, float]:
        return col[a], col[b] - 4

    # Stock vLLM: the pool is full; the turn waits beside it.
    y1 = y + 84
    s.text(x0, y1 - 14, "vLLM 0.32", size=VALUE)
    _pool(s, x0, y1, before, pitch, height)
    for k in range(TURN):
        _cell(s, turn_x + pitch * k, y1, pitch - 4, height, VOICE)
    for name, (a, b) in spans.items():
        s.bracket(*span(a, b), y1 + height + 12, name)
    s.bracket(*span(held, POOL), y1 + height + 12, "free")
    s.bracket(turn_x, turn_x + pitch * TURN - 4, y1 + height + 12, "voice turn")
    s.text(
        x0,
        y1 + height + 62,
        "The turn waits until agent blocks are freed.",
        size=NOTE,
        cls="soft",
    )
    # #63350: the latest agent request is preempted, then the turn takes its blocks.
    y2 = y1 + 160
    s.text(x0, y2 - 14, "#63350", size=VALUE)
    _pool(s, x0, y2, preempted, pitch, height)
    for k in range(TURN):
        _cell(s, turn_x + pitch * k, y2, pitch - 4, height, VOICE)
    (a, b), victim = spans[victims[0]], victims[0]
    s.bracket(*span(a, b), y2 - 12, f"{victim}, preempted", below=False)
    y3 = y2 + 88
    _pool(s, x0, y3, after, pitch, height)
    s.flow(
        [
            (turn_x + (pitch * TURN - 4) / 2, y2 + height + 3),
            (turn_x + (pitch * TURN - 4) / 2, y2 + height + 28),
            ((col[taken[0]] + col[taken[-1] + 1] - 4) / 2, y2 + height + 28),
            ((col[taken[0]] + col[taken[-1] + 1] - 4) / 2, y3 - 6),
        ],
        hue=VOICE,
        radius=6,
    )
    s.bracket(*span(taken[0], taken[-1] + 1), y3 + height + 12, "voice turn")
    s.bracket(*span(taken[-1] + 1, POOL), y3 + height + 12, "free")
    # Where the victim went (docs/ARCHITECTURE.md §2): its KV is in the CPU tier when it is preempted.
    note = f"{victims[0][0].upper()}{victims[0][1:]} waits; its KV is in the CPU tier."
    for k, line in enumerate(wrap(note, w - 52, NOTE)):
        s.text(x0, y3 + height + 64 + 22 * k, line, size=NOTE, cls="mute")


def hold_prefill_panel(
    s: Figure, x: float, y: float, w: float, h: float, *, ground: str = "b"
) -> None:
    """(c): the engine's steps on a time axis from the reply's first: without #63351 (on stock vLLM, or with
    #63350 alone, whose gaps the steps are drawn from) each carries an agent's prefill chunk beside the
    reply's token; with #63351 the window's steps carry the reply alone, at decode speed, and agent prefill
    resumes after it."""
    s.panel(x, y, w, h, "#63351: hold agent prefill", "c", ground=ground)
    x0 = x + 26
    length = w - 52
    span_ms = WINDOW * DECODE_MS + RESUMED * SHARED_MS
    unit = length / span_ms
    height = 24
    # The panel's key: what a step holds, on one line where it fits.
    ky = y + 72
    s.rect(x0, ky - 13, 14, 16, f"{VOICE} cell")
    first = s.text(x0 + 22, ky, "the reply's token", size=NOTE)
    second = "an agent's prefill chunk"
    kx = first.x1 + 22
    if kx + 38 + measure(second, NOTE) > x + w - 18:
        kx, ky = x0, ky + 26
    s.rect(kx, ky - 13, 30, 16, f"{AGENT} cell")
    s.text(kx + 38, ky, second, size=NOTE)
    shift = ky - (y + 72)  # the rows start under the key

    def step(at: float, row_y: float, shared: bool) -> float:
        """One step from ``at`` ms: the reply's token, and beside it an agent's prefill chunk when
        ``shared``; returns where the next step starts."""
        ms = SHARED_MS if shared else DECODE_MS
        left, width = x0 + at * unit, ms * unit - 1.2
        decode = DECODE_MS * unit - 1.2
        s.rect(left, row_y, decode, height, f"{VOICE} cell")
        if shared:
            s.rect(left + decode, row_y, width - decode, height, f"{AGENT} cell")
        return at + ms

    # Without #63351 (stock vLLM, or #63350 alone, whose gaps the steps are drawn from): every step shares
    # the reply's token with an agent's prefill chunk.
    y1 = y + 122 + shift
    s.text(x0, y1 - 12, "without #63351", size=VALUE)
    at, steps = 0.0, 0
    while at + SHARED_MS <= span_ms + 1e-9:
        at = step(at, y1, True)
        steps += 1
    window_ms = WINDOW * DECODE_MS
    in_window = round(window_ms / SHARED_MS)
    s.bracket(x0, x0 + in_window * SHARED_MS * unit - 1.2, y1 + height + 12, f"{in_window} tokens")
    # With #63351: the window's steps carry the reply alone; agent prefill resumes after it.
    y2 = y1 + 100
    s.text(x0, y2 - 12, "with #63351", size=VALUE)
    at = 0.0
    for _ in range(WINDOW):
        at = step(at, y2, False)
    for _ in range(RESUMED):
        at = step(at, y2, True)
    s.bracket(
        x0, x0 + window_ms * unit - 1.2, y2 + height + 12, f"the reply's first {WINDOW} tokens"
    )
    s.bracket(x0 + window_ms * unit, x0 + span_ms * unit - 1.2, y2 + height + 12, "resumed")
    # The time axis, from the reply's first token.
    ay = y2 + height + 62
    s.time_axis(x0, x0 + length, ay)
    caption = "time, one box per engine step, from the reply's first token"
    for k, line in enumerate(wrap(caption, length, NOTE)):
        s.text(x0, ay + 24 + 22 * k, line, size=NOTE, cls="mute")


def strip(s: Figure, top: float) -> dict[str, tuple[float, float, float, float]]:
    """(a), wide: returns the outlines that zoom into (b) and (c), as (left, top, right, bottom)."""
    s.text(24, top + 26, "One GPU: a voice agent beside coding agents")
    s.text(WIDTH - 24, top + 26, "(a)", anchor="end")
    lane1, lane2, h = top + 96, top + 160, 46
    pad = 28

    def width(text: str) -> float:
        return max(104.0, measure(text, LABEL) + pad)

    caller = (24.0, width("Caller"))
    stt = (caller[0] + caller[1] + 48, width("Speech to text"))
    sched_lines = ["vLLM 0.32 scheduler", "with #63350, #63351"]
    sched_w = max(measure(line, VALUE) for line in sched_lines) + 2 * pad
    sched_x = stt[0] + stt[1] + 108
    tts = (sched_x + sched_w + 8 + 104, width("Text to speech"))
    back = (WIDTH - 24 - width("Caller"), width("Caller"))
    agents = (24.0, stt[0] + stt[1] - 24)
    s.box(caller[0], lane1 - h / 2, caller[1], h, "Caller")
    s.box(stt[0], lane1 - h / 2, stt[1], h, "Speech to text")
    s.box(agents[0], lane2 - h / 2, agents[1], h, "24 coding-agent sessions")
    s.box(tts[0], lane1 - h / 2, tts[1], h, "Text to speech")
    s.box(back[0], lane1 - h / 2, back[1], h, "Caller")
    sched_top, sched_bottom = lane1 - h / 2, lane2 + h / 2
    s.box(
        sched_x, sched_top, sched_w, sched_bottom - sched_top, sched_lines, kind="ours", size=VALUE
    )
    zoom_c = (sched_x - 8, sched_top - 8, sched_x + sched_w + 8, sched_bottom + 8)
    s.group(zoom_c[0], zoom_c[1], zoom_c[2] - zoom_c[0], zoom_c[3] - zoom_c[1])
    # The KV pool, the scheduler's memory, under it.
    pool_x, pool_w = sched_x + 16, sched_w - 64
    pool_top = zoom_c[3] + 30
    s.memory(pool_x, pool_top, pool_w, 40)
    s.text(pool_x + pool_w / 2, pool_top + 26, "KV pool", size=VALUE, anchor="middle")
    link_x = pool_x + pool_w / 2
    s.flow([(link_x, sched_bottom), (link_x, pool_top)], arrow=False)
    # Dataflow: the turn, the agents' requests, the reply.
    s.flow([(caller[0] + caller[1], lane1), (stt[0], lane1)])
    s.flow([(stt[0] + stt[1], lane1), (zoom_c[0], lane1)])
    s.text(stt[0] + stt[1] + 12, lane1 - 10, "priority 0", size=NOTE, cls="mute")
    s.flow([(agents[0] + agents[1], lane2), (zoom_c[0], lane2)])
    s.text(agents[0] + agents[1] + 12, lane2 - 10, "priority 2", size=NOTE, cls="mute")
    s.flow([(zoom_c[2], lane1), (tts[0], lane1)])
    s.text(zoom_c[2] + 12, lane1 - 10, "the reply", size=NOTE, cls="mute")
    s.flow([(tts[0] + tts[1], lane1), (back[0], lane1)])
    return {"b": (pool_x, pool_top, pool_x + pool_w, pool_top + 40), "c": zoom_c}


# The system diagrams ----------------------------------------------------------------------------------------------
#
# docs/ARCHITECTURE.md's figures of the serving path and ops/RUNBOOK.md's triage, each with a phone layout. They keep
# the method figure's grammar, with square corners: a component is a box, the change's own pale blue; dataflow runs on
# steel-blue arrows; a host or a namespace is a dashed group. In a process diagram the main path runs down a column of
# steps, and each step's way out leaves to its right on a thin gray arrow to what it answers.

STEP_H = 40.0
"""A step's box in a process diagram."""
LINE = 22.0
"""Between the lines of what a step answers."""
EXIT = "gray"
"""The hue of a way out: thinner than dataflow, in a comparison's gray."""
SECOND = "second"
"""The flows to the second card: dataflow's color, thinner and dashed, as the paths the arrangement has only with
it (DIAGRAM_CSS)."""


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
    may run in ``columns`` columns ``column_w`` apart. Returns each box's top, and the lowest baseline or box edge
    drawn."""
    main = x + (24 if below else width / 2)
    drop = x + 58
    label_w = s.width - 16 - drop - 12
    heights = [_row_height(step, below=below, label_w=label_w, columns=columns) for step in steps]
    pitch = max(max(heights) + 14, STEP_H + gap)
    y = top
    lowest = top
    tops = []
    for k, step in enumerate(steps):
        tops.append(y)
        s.box(x, y, width, STEP_H, step.name)
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

UNIT_H = 52.0
"""A unit's box on a GPU node: its name, and under it its socket."""
SERVERS = (
    ("vLLM 0.32 + make-room", "engine.sock", "ours"),
    ("Whisper STT", "stt.sock", "box"),
    ("Kokoro TTS", "tts.sock", "box"),
)
"""A GPU node's servers (deploy/phastform/systemd), each on its Unix socket in $XDG_RUNTIME_DIR/flatshare: the engine,
which runs the change (variant make-room-serve), the STT and the TTS."""
NODES = (
    ("Phastform", "RTX 5060 Ti", "each server in its own namespace", False),
    ("Gray", "a second card, when present", "servers in the host's namespace", True),
)
"""The GPU nodes: the name, the card, where its servers' network namespaces are (Phastform's each the server's own,
deploy/bin/in-private-net; Gray's the host's, deploy/gray/private-net-host.conf), and whether it is drawn as the
second card of the two-card arrangement (ADR-0008), dashed, which the gateway reaches only with [[nodes]]."""
KEY = "TCP, a tenant's key"
"""How the coding agents reach the gateway: its one TCP listener, where every request needs a tenant's key, since other
users of cyan reach loopback (deploy/cyan/gateway.example.toml; the agents' tenant is [tenants.agents])."""
DIAGRAM_CSS = (
    ".dashed{stroke-dasharray:6 4}.second{--c:var(--flow)}.second.line{stroke-dasharray:6 4}"
)
"""The system diagrams' additions to the figures' style: the dashed outline of a unit the arrangement has only with a
second card, and its flows' dataflow color, dashed."""
ARCHITECTURE_DESCRIPTION = (
    "Hosts as dashed groups, each unit a box on its Unix socket: on cyan, the control host, the callers, LiveKit "
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


def _unit(
    s: Figure,
    x: float,
    y: float,
    w: float,
    name: str | list[str],
    socket: str,
    kind: str = "box",
) -> float:
    """A unit: a square box, white or the change's own (``kind`` "ours"), dashed where ``kind`` says so (e.g. "ours
    dashed"), its name on one line or several and under it its socket. Returns the box's height."""
    lines = [name] if isinstance(name, str) else name
    h = UNIT_H + 22 * (len(lines) - 1)
    s.rect(x, y, w, h, kind)
    texts = [(line, y + 22 + 22 * k, VALUE, "") for k, line in enumerate(lines)]
    for text, baseline, size, cls in [*texts, (socket, y + h - 9, NOTE, "soft")]:
        box = s.text(x + w / 2, baseline, text, size=size, anchor="middle", cls=cls)
        if box.x0 < x + 6 or box.x1 > x + w - 6:
            raise ValueError(f"{text!r} does not fit inside its box")
    return h


def _box(s: Figure, x: float, y: float, w: float, h: float, lines: list[str], cls: str) -> None:
    """A square box of class ``cls`` (a unit, dashed where the arrangement has it only with a second card), its
    ``lines`` centred inside at a category label's size."""
    s.rect(x, y, w, h, cls)
    first = y + h / 2 - 12 * (len(lines) - 1) + 0.33 * VALUE
    for k, line in enumerate(lines):
        box = s.text(x + w / 2, first + 24 * k, line, size=VALUE, anchor="middle")
        if box.x0 < x + 6 or box.x1 > x + w - 6 or box.y0 < y + 3 or box.y1 > y + h - 3:
            raise ValueError(f"{line!r} does not fit inside its box")


def _host(s: Figure, x: float, y: float, w: float, h: float, name: str, *, second: bool) -> float:
    """A host: a dashed group on the gray ground with its name inside at the top left, or, for the second card, the
    dashed outline alone. Returns the name's baseline."""
    if second:
        s.group(x, y, w, h)
        baseline = y + 14 + 0.7 * LABEL
        s.text(x + 18, baseline, name)
        return baseline
    return s.panel(x, y, w, h, name, ground="b")


def _node_units(
    s: Figure, x: float, top: float, w: float, names: list[list[str]], own: bool, second: bool
) -> list[float]:
    """A GPU node's servers in a column from ``top``, each in a dashed namespace of its own where ``own``. Returns
    each one's middle."""
    middles = []
    y = top
    for lines, (_, socket, kind) in zip(names, SERVERS, strict=True):
        h = UNIT_H + 22 * (len(lines) - 1)
        if own:
            s.group(x - 6, y - 6, w + 12, h + 12)
        _unit(s, x, y, w, lines, socket, f"{kind} dashed" if second else kind)
        middles.append(y + h / 2)
        y += h + 22
    return middles


def architecture(out: Path) -> list[Path]:
    """architecture.svg and its phone layout: the hosts, their units and the sockets between them."""
    title = "The serving architecture"
    return [
        _architecture_wide(out / "architecture.svg", title),
        _architecture_narrow(out / "architecture-narrow.svg", title),
    ]


def _architecture_wide(path: Path, title: str) -> Path:
    """cyan left, the GPU nodes right, one above the other, each reached through its tunnel."""
    s = Diagram(WIDTH, 700, title, ARCHITECTURE_DESCRIPTION)
    nx, nw = 752.0, WIDTH - 24 - 752.0
    names = [[unit] for unit, _, _ in SERVERS]
    node_h = 76 + len(SERVERS) * (UNIT_H + 22)
    middles = []
    y = 12.0
    for name, card, spaces, second in NODES:
        baseline = _host(s, nx, y, nw, node_h, name, second=second)
        s.text(nx + nw - 18, baseline, card, size=NOTE, anchor="end", cls="soft")
        s.text(nx + 18, y + 62, spaces, size=NOTE, cls="mute")
        middles.append(_node_units(s, nx + 36, y + 76, nw - 72, names, not second, second))
        y += node_h + 24
    bottom = y - 24
    # cyan, left: the voice plane, the gateway, the agents and a tunnel per node.
    cyan_right = nx - 60
    _host(s, 24, 12, cyan_right - 24, bottom - 12, "cyan", second=False)
    s.text(
        cyan_right - 18,
        12 + 14 + 0.7 * LABEL,
        "control host, CPU only",
        size=NOTE,
        anchor="end",
        cls="soft",
    )
    tx, tw = 560.0, 116.0
    tunnels = []
    for rows, (name, _, _, second) in zip(middles, NODES, strict=True):
        top, low = rows[0] - 26, rows[-1] + 26
        tunnels.append((top, low))
        _box(s, tx, top, tw, low - top, ["Tunnel to", name], "box dashed" if second else "box")
        for mid in rows:
            s.flow([(tx + tw, mid), (nx + 36, mid)], hue=SECOND if second else None)
        s.text((cyan_right + nx) / 2, rows[0] - 9, "SSH", size=NOTE, anchor="middle", cls="soft")
    # The gateway, under the first node's tunnel, so that the worker's speech beside it reaches that tunnel's side;
    # its engine requests go to each node's tunnel.
    gx, gw, gy = 364.0, 120.0, tunnels[0][1] + 16
    s.box(gx, gy, gw, 44, "Gateway")
    split = tx + tw / 2
    s.flow([(gx + gw, gy + 22), (split, gy + 22), (split, tunnels[0][1])], radius=0)
    s.flow([(split, gy + 22), (split, tunnels[1][0])], hue=SECOND, radius=0)
    s.text(gx + gw + 10, gy + 13, "engine.sock", size=NOTE, cls="soft")
    # The voice plane, in its namespace: the callers, LiveKit and the worker, whose LLM requests go to the gateway's
    # socket and whose speech to the node's tunnel, each flow labelled right of the namespace's border.
    vx, vw = 56.0, 160.0
    space = (40.0, gy - 228, 200.0, 228 + 44 + 22)
    s.group(*space)
    label_x = space[0] + space[2] + 10
    s.text(space[0] + 12, space[1] + 26, "voice-plane namespace", size=NOTE, cls="soft")
    s.text(space[0] + 12, space[1] + 48, "no route out", size=NOTE, cls="mute")
    s.box(vx, gy - 164, vw, 44, "Callers")
    s.box(vx, gy - 82, vw, 44, "LiveKit server")
    s.box(vx, gy, vw, 44, "Voice worker")
    s.flow([(vx + 50, gy - 120), (vx + 50, gy - 82)])
    s.text(vx + 62, gy - 95, "WebRTC", size=NOTE, cls="soft")
    s.flow([(vx + 50, gy - 38), (vx + 50, gy)])
    s.flow([(vx + vw, gy + 22), (gx, gy + 22)])
    s.text(label_x, gy + 13, "gateway.sock", size=NOTE, cls="soft")
    speech = gy - 30
    s.flow([(vx + vw - 16, gy), (vx + vw - 16, speech), (tx, speech)], radius=0)
    s.text(label_x, speech - 9, "stt.sock, tts.sock", size=NOTE, cls="soft")
    # The coding agents, in cyan's host network, through the one TCP listener.
    ay = space[1] + space[3] + 70
    s.box(vx, ay, vw, 44, "Coding agents")
    s.flow([(vx + vw, ay + 22), (gx + gw / 2, ay + 22), (gx + gw / 2, gy + 44)], radius=0)
    s.text(label_x, ay + 13, "127.0.0.1:18080", size=NOTE)
    s.text(label_x, ay + 45, KEY, size=NOTE, cls="mute")
    s.height = bottom + 12
    return s.save(path)


def _architecture_narrow(path: Path, title: str) -> Path:
    """The phone layout: cyan above, its units in a column and a tunnel either side of the gateway; the GPU nodes
    under it, side by side, each reached through its tunnel."""
    s = Diagram(NARROW, 1600, title, ARCHITECTURE_DESCRIPTION)
    x0, x1 = 16.0, NARROW - 16.0
    # Where cyan's units go, which sets its panel's height: the voice plane's column, the gateway's row with a tunnel
    # either side, and the coding agents under it.
    space_top, vx, vw = 64.0, 112.0, 196.0
    rows = [space_top + 64 + 84 * k for k in range(3)]
    gy = rows[-1] + 44 + 96
    tw, th, gw = 104.0, 64.0, 100.0
    gx, left, right = (NARROW - gw) / 2, x0 + 16, x1 - 16 - tw
    ay, line = gy + 44 + 120, gx + 16
    cyan_bottom = ay + 44 + 24
    _host(s, x0, 12, x1 - x0, cyan_bottom - 12, "cyan", second=False)
    s.text(
        x1 - 18,
        12 + 14 + 0.7 * LABEL,
        "control host, CPU only",
        size=NOTE,
        anchor="end",
        cls="soft",
    )
    # The voice plane, its units in a column, each flow labelled on its right.
    s.group(x0 + 16, space_top, x1 - x0 - 32, rows[-1] + 44 + 22 - space_top)
    s.text(x0 + 28, space_top + 26, "voice-plane namespace", size=NOTE, cls="soft")
    s.text(x1 - 28, space_top + 26, "no route out", size=NOTE, anchor="end", cls="mute")
    for row, name in zip(rows, ("Callers", "LiveKit server", "Voice worker"), strict=True):
        s.box(vx, row, vw, 44, name)
    mid = vx + vw / 2
    s.flow([(mid, rows[0] + 44), (mid, rows[1])])
    s.text(mid + 10, rows[0] + 44 + 26, "WebRTC", size=NOTE, cls="soft")
    s.flow([(mid, rows[1] + 44), (mid, rows[2])])
    s.flow([(mid, rows[2] + 44), (mid, gy)])
    s.text(mid + 10, rows[2] + 44 + 50, "gateway.sock", size=NOTE, cls="soft")
    # The gateway between the tunnels, each node's beside it; the worker's speech to the first node's.
    s.box(gx, gy, gw, 44, "Gateway")
    for tx, (name, _, _, second) in zip((left, right), NODES, strict=True):
        _box(s, tx, gy - 10, tw, th, ["Tunnel to", name], "box dashed" if second else "box")
    s.flow([(gx, gy + 22), (left + tw, gy + 22)])
    s.flow([(gx + gw, gy + 22), (right, gy + 22)], hue=SECOND)
    s.text(gx + 54, gy + 44 + 26, "engine.sock", size=NOTE, cls="soft")
    speech = left + tw / 2
    s.flow([(vx, rows[2] + 22), (speech, rows[2] + 22), (speech, gy - 10)], radius=0)
    s.text(speech + 10, rows[2] + 44 + 44, "stt.sock,", size=NOTE, cls="soft")
    s.text(speech + 10, rows[2] + 44 + 66, "tts.sock", size=NOTE, cls="soft")
    # The coding agents, under the gateway, through the one TCP listener.
    s.box(gx - 20, ay, gw + 40, 44, "Coding agents")
    s.flow([(line, ay), (line, gy + 44)])
    s.text(line + 10, ay - 38, "127.0.0.1:18080", size=NOTE)
    s.text(line + 10, ay - 16, KEY, size=NOTE, cls="mute")
    # The GPU nodes, side by side, each reached through its tunnel: its name, its card, where its servers'
    # namespaces are, and its servers, the engine's name on two lines.
    nw, ny = (x1 - x0 - 16) / 2, cyan_bottom + 56
    names = [
        ["vLLM 0.32", "+ make-room"] if kind == "ours" else [unit] for unit, _, kind in SERVERS
    ]
    units_h = sum(UNIT_H + 22 * (len(lines) - 1) + 22 for lines in names)
    notes = [[*_lines(card, nw - 36), *_lines(spaces, nw - 36)] for _, card, spaces, _ in NODES]
    top = ny + 62 + LINE * max(len(lines) for lines in notes)
    bottom = top + units_h - 2
    for k, ((name, card, _, second), tx) in enumerate(zip(NODES, (left, right), strict=True)):
        nx = x0 + k * (nw + 16)
        _host(s, nx, ny, nw, bottom - ny, name, second=second)
        cards = len(_lines(card, nw - 36))
        for j, note in enumerate(notes[k]):
            s.text(
                nx + 18, ny + 62 + LINE * j, note, size=NOTE, cls="soft" if j < cards else "mute"
            )
        _node_units(s, nx + 18, top, nw - 36, names, not second, second)
        s.flow(
            [(tx + tw / 2, gy - 10 + th), (tx + tw / 2, ny)],
            hue=SECOND if second else None,
            radius=0,
        )
        s.text(tx + tw / 2 + 10, cyan_bottom + 34, "SSH", size=NOTE, cls="soft")
    s.height = bottom + 12
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
REQUEST_DESCRIPTION = (
    "A column of seven steps a request takes through the gateway, each step's main refusals leaving to its right, "
    "which the gateway's error table gives in full: authenticate "
    "(401 invalid_api_key, 400 invalid_class), bound (503 in_flight_limit, 429 tenant_in_flight_limit, 413 "
    "body_too_large), sanitize (400 invalid_json, 400 too_many_choices), place (503 upstream_not_ready, "
    "no_healthy_node, no_call_slot or no_agent_slot), forward (502 upstream_unavailable, 504 upstream_timeout), "
    "relay (an error event of type "
    "upstream_unavailable, upstream_timeout or draining once a stream has begun), and account, which records one "
    "outcome for every request."
)


def request_path(out: Path) -> list[Path]:
    """request-path.svg and its phone layout: a request's path through the gateway, each step's refusals beside it."""
    title = "A request's path through the gateway"
    wide = Diagram(WIDTH, 800, title, REQUEST_DESCRIPTION)
    wide.text(24 + 100, 30, "a request", size=NOTE, anchor="middle", cls="soft")
    wide.flow([(124, 40), (124, 64)])
    _, lowest = _ladder(
        wide,
        REQUEST_STEPS,
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
    narrow.height = (
        _ladder(narrow, REQUEST_STEPS, x=16, top=64, width=200, lead_w=34, below=True)[1] + 24
    )
    return [wide.save(out / "request-path.svg"), narrow.save(out / "request-path-narrow.svg")]


# Two cards -------------------------------------------------------------------------------------------------------

CARDS = (("Phastform", 2, 4), ("Gray", 2, 12))
"""The serving arrangement's cards as deploy/cyan/gateway.example.toml's [[nodes]] give them, in placement order: the
name, max_calls and max_agents. With calls = "fill" a call takes the first card with a free call slot."""
MOVED = ("next turn: 0.7 s after a kill,", "1.9 s after a stop")
"""When both calls reached the spare in the failover drills (benchmarks/results/2026-10-28-node-failover)."""
CARD_H = 124.0
"""A card's box: its name, then a row of call slots and a row of agents' slots."""
HELD = "held until the calls end"
"""What the spare's hatched agents' slots mean in (b): a node that covers holds its agents' requests to
failover_background_max_in_flight, 0 in deploy/cyan/gateway.example.toml, until the moved calls end
(crates/flatshare-gateway/README.md, Failover)."""
HELD_H = 28.0
"""The bracket under the spare's agents in (b) and its name, below the card."""
CARDS_DESCRIPTION = (
    "Two panels of the two cards' slots as the gateway bounds them, two call slots each and up to 4 agents' "
    "sessions on Phastform and 12 on Gray: in panel a, in service, both calls fill Phastform and Gray, the warm "
    "spare, runs agents only; in panel b, Phastform's engine failed, both calls take Gray's call slots at their "
    "next turn, 0.7 s after a kill and 1.9 s after a stop in the drills, and Gray holds its agents' requests until "
    "the moved calls end."
)


def _card(
    s: Figure,
    x: float,
    y: float,
    w: float,
    card: tuple[str, int, int],
    *,
    calls: int,
    agents: str | None,
    note: str,
) -> tuple[list[float], tuple[float, float]]:
    """A card: its name and ``note``, its call slots, ``calls`` of them taken, and its agents' slots, running
    (``agents`` "run"), held back ("held", hatched) or none (None). The slots share one pitch, the most at which
    the most slots fit the card. Returns its call slots' centres and its agents' row's left and right edges."""
    name, max_calls, max_agents = card
    s.rect(x, y, w, CARD_H, "box")
    s.text(x + 16, y + 30, name)
    s.text(x + w - 16, y + 30, note, size=NOTE, anchor="end", cls="soft")
    left = x + 86
    pitch = min(26.0, (x + w - 16 - left) / max(c[2] for c in CARDS))
    slot = round(pitch * 0.77)
    centres = []
    for row, (label, count, taken, hue) in enumerate(
        (
            ("calls", max_calls, calls, VOICE),
            ("agents", max_agents, max_agents if agents else 0, AGENT),
        )
    ):
        top = y + 52 + 38 * row
        s.text(x + 16, top + 15, label, size=NOTE, cls="mute")
        cells: list[str | None] = [hue] * taken + [None] * (count - taken)
        s.cells(
            left,
            top + (20 - slot) / 2,
            cells,
            cell=slot,
            height=slot,
            gap=pitch - slot,
            hatched=frozenset(range(count)) if agents == "held" and row == 1 else frozenset(),
        )
        if row == 0:
            centres = [left + k * pitch + slot / 2 for k in range(count)]
    return centres, (left, left + (max_agents - 1) * pitch + slot)


def two_cards(out: Path) -> list[Path]:
    """two-cards.svg and its phone layout: the serving arrangement's two cards, (a) in service and (b) after the first
    failed, its calls on the spare."""
    title = "Two cards: placement and failover"
    paths = []
    for path, width in ((out / "two-cards.svg", WIDTH), (out / "two-cards-narrow.svg", NARROW)):
        s = Diagram(width, 1200, title, CARDS_DESCRIPTION)
        wide = width == WIDTH
        panel_w = (WIDTH - 48 - 24) / 2 if wide else NARROW - 32
        card_w = panel_w - 40
        bottom = 12.0
        for k, (letter, heading) in enumerate(
            (("a", "In service"), ("b", "The first card failed"))
        ):
            px = 24 + k * (panel_w + 24) if wide else 16.0
            py = 12.0 if wide else bottom + (20 if k else 0)
            first, second = py + 56, py + 56 + CARD_H + 84
            # Side by side, the panels share a height; stacked, only (b) holds the bracket under its spare.
            ph = second + CARD_H + 20 + (HELD_H if wide or letter == "b" else 0) - py
            s.panel(px, py, panel_w, ph, heading, letter, ground="b")
            cx = px + 20
            between = first + CARD_H + 48
            if letter == "a":
                _card(s, cx, first, card_w, CARDS[0], calls=2, agents="run", note="the calls")
                _card(s, cx, second, card_w, CARDS[1], calls=0, agents="run", note="warm spare")
                s.text(cx, between, 'calls = "fill"', size=NOTE, cls="soft")
            else:
                taken, _ = _card(
                    s, cx, first, card_w, CARDS[0], calls=0, agents=None, note="engine down"
                )
                spare, (held_x0, held_x1) = _card(
                    s, cx, second, card_w, CARDS[1], calls=2, agents="held", note="covering"
                )
                s.bracket(held_x0, held_x1, second + CARD_H + 9, HELD)
                for a, b in zip(
                    taken, spare, strict=True
                ):  # from the card, clear of its agents' slots
                    s.flow([(a, first + CARD_H), (b, second + 52)], hue=VOICE, radius=0)
                for j, line in enumerate(MOVED):
                    s.text(taken[-1] + 24, between - 12 + LINE * j, line, size=NOTE, cls="soft")
            bottom = py + ph
        s.height = bottom + 12
        paths.append(s.save(path))
    return paths


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
SWITCH_BACK = (("exit 1", "the release before runs again"), ("exit 3", "switching back failed too"))
"""What a switch back ends with: the release before up and checked, or not."""
RELEASE_DESCRIPTION = (
    "A column of the release's states, prepare, verify, drain, switch, restart, check and serving: a host that is "
    "locked or an upload that changed exits 2, a failed verification exits 1 and a drain whose calls do not end exits "
    "2, each with nothing switched; any failure from the switch through the check switches back, which exits 1 with "
    "the release before running again or 3 when switching back fails too; a release that passes its checks serves "
    "and exits 0, and a rollback enters at the verification with the previous release."
)


def release(out: Path) -> list[Path]:
    """release.svg and its phone layout: a release's states and its exits, the switch back by itself."""
    title = "A release and its exits"
    paths = []
    for path, width, below in (
        (out / "release.svg", WIDTH, False),
        (out / "release-narrow.svg", NARROW, True),
    ):
        s = Diagram(width, 1200, title, RELEASE_DESCRIPTION)
        x, box_w = (200.0, 180.0) if not below else (16.0, 200.0)
        main = x + (24 if below else box_w / 2)
        s.text(
            main if not below else x,
            30,
            "deploy.sh HOST",
            size=NOTE,
            anchor="start" if below else "middle",
            cls="soft",
        )
        s.flow([(main, 40), (main, 64)])
        tops, lowest = _ladder(
            s, RELEASE_STEPS, x=x, top=64, width=box_w, exit_to=700, lead_w=46, below=below
        )
        switch, restart, check = tops[3:6]
        # A rollback is the same path from the verification, to the previous release.
        verify = tops[1] + STEP_H / 2
        if below:
            s.text(x + box_w + 30, verify - 9, "--rollback", size=NOTE, cls="soft")
            s.flow([(NARROW - 16, verify), (x + box_w, verify)])
        else:
            s.text(24, verify - 9, "--rollback", size=NOTE, cls="soft")
            s.flow([(24, verify), (x, verify)])
        # Any failure from the switch through the check switches back.
        edge = x + box_w
        spine = edge + 16
        s.path(
            f"M{fmt(edge + 4)} {fmt(switch + STEP_H / 2)}H{fmt(spine)}V{fmt(check + STEP_H / 2)}"
            f"H{fmt(edge + 4)}M{fmt(edge + 4)} {fmt(restart + STEP_H / 2)}H{fmt(spine)}",
            "bracket",
        )
        s.wires.append(([(spine, switch + STEP_H / 2), (spine, check + STEP_H / 2)], ()))
        back_w = 150.0
        if below:
            back_x, back_y = x + 104, check + STEP_H + 52
            s.flow(
                [
                    (spine, check + STEP_H / 2),
                    (spine, back_y - 22),
                    (back_x + back_w / 2, back_y - 22),
                    (back_x + back_w / 2, back_y),
                ],
                hue=EXIT,
                radius=0,
            )
            s.text(spine + 10, check + STEP_H / 2 + 22, "a failure", size=NOTE, cls="soft")
            answers_x, baseline = x + 58, back_y + STEP_H + 44
        else:
            back_x, back_y = 520.0, restart
            s.flow([(spine, restart + STEP_H / 2), (back_x, restart + STEP_H / 2)], hue=EXIT)
            s.text(spine + 12, restart + STEP_H / 2 - 9, "a failure", size=NOTE, cls="soft")
            s.flow([(back_x + back_w, restart + STEP_H / 2), (700, restart + STEP_H / 2)], hue=EXIT)
            answers_x, baseline = 710.0, restart + STEP_H / 2 + 6
        s.box(back_x, back_y, back_w, STEP_H, "Switch back")
        if below:
            s.flow([(back_x + 20, back_y + STEP_H), (back_x + 20, back_y + STEP_H + 26)], hue=EXIT)
        last = _answers(s, answers_x, baseline, SWITCH_BACK, 46)
        # Through the check, it serves.
        serving = max(check + STEP_H + 30, (last + 24) if below else 0)
        s.flow([(main, check + STEP_H), (main, serving)])
        s.box(x, serving, box_w, STEP_H, "Serving")
        s.text(x + box_w + 14, serving + STEP_H / 2 + 6, "exit 0", size=NOTE, cls="mute")
        s.height = max(lowest, last, serving + STEP_H) + 24
        paths.append(s.save(path))
    return paths


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
    "A column of five checks from a call that fails or waits, each leaving to its right for the runbook rows it "
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
        _ladder(wide, TRIAGE_STEPS, x=24, top=64, width=220, exit_to=600, lead_w=0)[1] + 24
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


DESCRIPTION = (
    "Panel a: a caller's speech goes to speech to text, then to vLLM 0.32's scheduler, which carries the "
    "two changes, then to text to speech and back to the caller; 24 coding-agent sessions send their "
    "requests to the same scheduler at priority 2, the voice turn's at priority 0; the scheduler allocates "
    "from the KV pool beneath it. Panel b, #63350, a pool of 12 KV blocks: two agent requests hold 5 each and "
    f"2 are free, while a voice turn that needs {TURN} waits beside the pool; on vLLM 0.32 it waits until an "
    "agent request frees its blocks, while #63350 preempts the latest agent request, agent B, and the turn takes "
    "4 of its blocks; agent B waits, its KV in the CPU tier. Panel c, #63351, the "
    "engine's steps in time, one box a step, from the reply's first token: without #63351 each step of the reply also "
    f"carries an agent's prefill chunk, so in the time #63351 gives the reply its first {WINDOW} tokens the "
    "reply gets 14 without it; after the window, agent prefill resumes. Blocks and steps are illustrative."
)


def _wide(path: Path) -> Path:
    s = Figure(WIDTH, 760, "How #63350 and #63351 make room for a voice turn", DESCRIPTION)
    zooms = strip(s, 8)
    py, ph = 330, 440
    bx, bw = 24.0, 532.0
    cx = bx + bw + 32
    cw = WIDTH - 24 - cx
    make_room_panel(s, bx, py, bw, ph)
    hold_prefill_panel(s, cx, py, cw, ph)
    # The zooms' leaders: the pool into (b), the scheduler into (c).
    pool = zooms["b"]
    s.leader([(pool[0] + 22, pool[3]), (pool[0] + 22, py)])
    sched = zooms["c"]
    x = (pool[2] + sched[2]) / 2  # between the pool's right edge and the scheduler's, over (c)
    assert cx < x < cx + cw, (cx, x)
    s.leader([(x, sched[3]), (x, py)])
    s.height = py + ph + 12
    return s.save(path)


def _narrow(path: Path) -> Path:
    s = Figure(NARROW, 1400, "How #63350 and #63351 make room for a voice turn", DESCRIPTION)
    x0, w = 16.0, NARROW - 32
    s.text(x0, 30, "One GPU: a voice agent beside")
    s.text(x0, 56, "coding agents")
    s.text(NARROW - 16, 30, "(a)", anchor="end")
    h = 44
    col = 104.0  # the voice's column
    rows = [92.0, 182.0, 0.0, 0.0, 0.0]
    s.box(col - 82, rows[0] - h / 2, 164, h, "Caller")
    s.box(col - 82, rows[1] - h / 2, 164, h, "Speech to text")
    s.box(
        230,
        rows[1] - 28,
        NARROW - 16 - 230,
        56,
        ["24 coding-agent", "sessions"],
        size=VALUE,
        pitch=22,
    )
    sched_top, sched_h = rows[1] + 100, 70
    s.box(
        x0,
        sched_top,
        w,
        sched_h,
        ["vLLM 0.32 scheduler", "with #63350, #63351"],
        kind="ours",
        size=VALUE,
    )
    tts_y = sched_top + sched_h + 96
    s.box(col - 82, tts_y - h / 2, 164, h, "Text to speech")
    s.memory(230, tts_y - 22, NARROW - 16 - 230, 44)
    s.text((230 + NARROW - 16) / 2, tts_y + 7, "KV pool", size=VALUE, anchor="middle")
    back_y = tts_y + 96
    s.box(col - 82, back_y - h / 2, 164, h, "Caller")
    s.flow([(col, rows[0] + h / 2), (col, rows[1] - h / 2)])
    s.flow([(col, rows[1] + h / 2), (col, sched_top)])
    s.text(col + 10, rows[1] + h / 2 + 32, "priority 0", size=NOTE, cls="mute")
    agents_x = (230 + NARROW - 16) / 2
    s.flow([(agents_x, rows[1] + 28), (agents_x, sched_top)])
    s.text(agents_x + 10, rows[1] + h / 2 + 32, "priority 2", size=NOTE, cls="mute")
    s.flow([(col, sched_top + sched_h), (col, tts_y - h / 2)])
    s.text(col + 10, sched_top + sched_h + 32, "the reply", size=NOTE, cls="mute")
    s.flow([(agents_x, sched_top + sched_h), (agents_x, tts_y - 22)], arrow=False)
    s.flow([(col, tts_y + h / 2), (col, back_y - h / 2)])
    py = back_y + h / 2 + 40
    make_room_panel(s, x0, py, w, 464)
    py += 464 + 24
    hold_prefill_panel(s, x0, py, w, 440)
    s.height = py + 440 + 12
    return s.save(path)


SYSTEM = (architecture, request_path, two_cards, release, triage)
"""The system diagrams, in docs/ARCHITECTURE.md's order, then the runbook's."""


if __name__ == "__main__":
    for written in render(Path(sys.argv[1])):
        print(written)
