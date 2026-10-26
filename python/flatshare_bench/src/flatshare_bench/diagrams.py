"""The README's method figure, docs/figures/method.svg and its phone layout method-narrow.svg: a voice agent
and coding agents on one GPU, and the two scheduler rules of vLLM #63350 and #63351.

    python -m flatshare_bench.diagrams OUT_DIR

(a) is a stage strip: a caller's turn through speech to text, vLLM's scheduler and text to speech, beside the
coding agents' requests, the KV pool the scheduler allocates from below it. Its dashed outlines zoom into the
panels below: the pool into (b), #63350's make-room, and the scheduler into (c), #63351's hold on agent prefill.

The blocks and steps are illustrative, and obey the rules they show (docs/ARCHITECTURE.md §2). Make-room
preempts lower-priority running requests in vLLM's own order, the latest arrival first, as few as make the
waiting request fit; render() works out its victims with :func:`make_room` and draws what it returns. The
budget cap holds lower-priority prefill from the step the turn fits until it has its first tokens, and never
cuts a decode. A step's length is drawn in the ratio of the recorded gaps between a reply's tokens beside
agent prefill and with it held (the README's "What it shows": 60-65 ms and 16-20 ms), at the middle of each.
"""

from __future__ import annotations

import sys
from pathlib import Path

from flatshare_bench.figure_style import LABEL, NARROW, NOTE, VALUE, WIDTH, Figure, measure, wrap

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
        s.rect(x, y, width, height, "cell-empty", rx=2)
    elif kind == "preempted":
        s.cells(x, y, [AGENT], cell=width, height=height, hatched=frozenset({0}))
    else:
        s.rect(x, y, width, height, f"{kind} cell", rx=2)


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
    s.text(x0, y1 - 14, "stock vLLM", size=VALUE)
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
        "The turn waits for an agent request to finish.",
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
    # Where the victim went (docs/ARCHITECTURE.md §2): its KV stays in the CPU tier for its resume.
    note = f"{victims[0][0].upper()}{victims[0][1:]} waits; its KV stays in the CPU tier for its resume."
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
    s.rect(x0, ky - 13, 14, 16, f"{VOICE} cell", rx=2)
    first = s.text(x0 + 22, ky, "the reply's token", size=NOTE)
    second = "an agent's prefill chunk"
    kx = first.x1 + 22
    if kx + 38 + measure(second, NOTE) > x + w - 18:
        kx, ky = x0, ky + 26
    s.rect(kx, ky - 13, 30, 16, f"{AGENT} cell", rx=2)
    s.text(kx + 38, ky, second, size=NOTE)
    shift = ky - (y + 72)  # the rows start under the key

    def step(at: float, row_y: float, shared: bool) -> float:
        """One step from ``at`` ms: the reply's token, and beside it an agent's prefill chunk when
        ``shared``; returns where the next step starts."""
        ms = SHARED_MS if shared else DECODE_MS
        left, width = x0 + at * unit, ms * unit - 1.2
        decode = DECODE_MS * unit - 1.2
        s.rect(left, row_y, decode, height, f"{VOICE} cell", rx=1.5)
        if shared:
            s.rect(left + decode, row_y, width - decode, height, f"{AGENT} cell", rx=1.5)
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


def render(out_dir: Path) -> list[Path]:
    """method.svg and method-narrow.svg in ``out_dir``."""
    return [_wide(out_dir / "method.svg"), _narrow(out_dir / "method-narrow.svg")]


DESCRIPTION = (
    "Panel a: a caller's speech goes to speech to text, then to vLLM 0.32's scheduler, which carries the "
    "two changes, then to text to speech and back to the caller; 24 coding-agent sessions send their "
    "requests to the same scheduler at priority 2, the voice turn's at priority 0; the scheduler allocates "
    "from the KV pool beneath it. Panel b, #63350, a pool of 12 KV blocks: two agent requests hold 5 each and "
    f"2 are free, while a voice turn that needs {TURN} waits beside the pool; on stock vLLM it waits for an "
    "agent request to finish, while #63350 preempts the latest agent request, agent B, and the turn takes "
    "4 of its blocks; agent B waits, its KV kept in the CPU tier for its resume. Panel c, #63351, the "
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
    s.leader(
        [
            (sched[2] - 14, sched[3]),
            (sched[2] - 14, sched[3] + 12),
            (cx + 40, py - 18),
            (cx + 40, py),
        ]
    )
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


if __name__ == "__main__":
    for written in render(Path(sys.argv[1])):
        print(written)
