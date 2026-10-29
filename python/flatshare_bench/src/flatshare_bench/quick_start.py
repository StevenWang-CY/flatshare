"""The README's quick-start result: the probe's first-token times on stock vLLM and with make-room, from the logs of
the runs that measured them, as the block between README.md's quick-start probe markers.

    python -m flatshare_bench.quick_start STOCK_LOG MAKE_ROOM_LOG [--write README.md]

Each log is probe.py's output in the KV case (python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py), one line
a run, appended: ``kv: priority-0 TTFT 4900 ms, 16 tokens 5185 ms, ...``. The block gives each engine's range of those
times, the count of runs, which both logs must share, and both logs linked from the README. The command prints the
block, or with --write replaces the one between the markers, so that a record of new runs changes the README by this
command alone. test_serving holds the README's block to the logs it links.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path

README = Path(__file__).resolve().parents[4] / "README.md"
"""The repository's README, whose quick start shows the block."""
START, END = "<!-- quick-start probe -->", "<!-- /quick-start probe -->"
"""The README's lines around the block, each a line of its own."""
BLOCK = re.compile(rf"^{re.escape(START)}\n(.*?)^{re.escape(END)}$", re.M | re.S)
WIDTH = 120
"""The README's line width."""
RUN = re.compile(r"kv: priority-0 TTFT (\d+) ms, ")
"""A run's line in the KV case, as probe.py prints it, with the voice turn's first-token time in ms."""
DASH = "\u2013"
"""The en dash between a range's ends."""
WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
"""The counts the docs spell out; from 10 on they write digits."""


def ttfts(log: Path) -> list[int]:
    """The first-token time of each run in a probe log, in ms: each line's ``priority-0 TTFT N ms``. A line that is no
    run of the KV case is refused, so that a failed or misplaced run cannot change the count unseen."""
    times = []
    for number, line in enumerate(Path(log).read_text().splitlines(), 1):
        if not line.strip():
            continue
        run = RUN.match(line)
        if run is None:
            raise ValueError(f"{log}:{number}: not a run of the KV case: {line!r}")
        times.append(int(run.group(1)))
    if not times:
        raise ValueError(f"{log}: no run")
    return times


def span(times: list[int]) -> str:
    """The range of ``times`` as the docs print it, with thousands separators and an en dash, or the one value all
    the runs gave."""
    low, high = min(times), max(times)
    return f"{low:,}" if low == high else f"{low:,}{DASH}{high:,}"


def runs(count: int) -> str:
    """A count of runs as the docs write it: in words below 10, in digits from 10 on."""
    return (
        ("one run" if count == 1 else f"{WORDS[count - 1]} runs")
        if count < 10
        else f"{count:,} runs"
    )


def block(stock_log: Path, make_room_log: Path, readme: Path = README) -> str:
    """The block's text: each engine's range of first-token times, the count of runs both logs share, and the two
    logs linked from ``readme``, wrapped at the README's width."""
    stock, make_room = ttfts(stock_log), ttfts(make_room_log)
    if len(stock) != len(make_room):
        raise ValueError(
            f"{stock_log} has {len(stock)} runs and {make_room_log} {len(make_room)}: "
            "the block gives one count for both"
        )
    base = Path(readme).resolve().parent

    def link(log: Path) -> str:
        return Path(os.path.relpath(Path(log).resolve(), base)).as_posix()

    text = (
        "The probe prints the voice turn's time to its first token as `priority-0 TTFT`. On the box it was "
        f"{span(stock)} ms on stock vLLM and {span(make_room)} ms with make-room, in {runs(len(stock))} each "
        f"([stock]({link(stock_log)}), [make-room]({link(make_room_log)}))."
    )
    return textwrap.fill(text, WIDTH, break_long_words=False, break_on_hyphens=False) + "\n"


def current(text: str) -> str:
    """The block in a README's text: what lies between its two markers, which it must hold once each, in order."""
    found = BLOCK.findall(text)
    if text.count(START) != 1 or text.count(END) != 1 or len(found) != 1:
        raise ValueError(f"the README needs one block, between a line {START} and a line {END}")
    return found[0]


def write(readme: Path, new: str) -> None:
    """Replaces the block in ``readme`` with ``new``, and nothing else of the file."""
    text = Path(readme).read_text()
    current(text)
    [found] = BLOCK.finditer(text)
    Path(readme).write_text(text[: found.start(1)] + new + text[found.end(1) :])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m flatshare_bench.quick_start",
        description="The README's quick-start result, from the probe's logs on stock vLLM and with make-room.",
    )
    parser.add_argument(
        "stock_log", type=Path, help="probe.py's KV-case lines on stock vLLM, one a run"
    )
    parser.add_argument(
        "make_room_log", type=Path, help="probe.py's KV-case lines with make-room, one a run"
    )
    parser.add_argument(
        "--write",
        type=Path,
        metavar="README",
        help="replace the block between this README's markers instead of printing it",
    )
    args = parser.parse_args(argv)
    try:
        text = block(args.stock_log, args.make_room_log, args.write or README)
        if args.write is None:
            sys.stdout.write(text)
        else:
            write(args.write, text)
    except (OSError, ValueError) as e:
        print(f"quick_start: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
