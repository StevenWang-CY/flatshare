"""The README's table, from the probe's two logs: `python3 table.py` prints it, and the record's test holds the README
to it. Each log is probe.py's output in the KV case, one line a run: the voice turn's first-token time and the load it
was sent into."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINES = {"stock": "Stock vLLM", "make-room": "make-room"}
WIDTH = 120
"""The README's line width."""
DASH = "\u2013"
"""The en dash between a range's ends."""
RUN = re.compile(
    r"kv: priority-0 TTFT (?P<ttft>\d+) ms, 16 tokens (?:\d+ ms|—), for (?P<prompt>\d+) prompt tokens, sent with"
    r" (?P<running>\d+) priority-2 requests of (?P<tokens>\d+) tokens running and (?P<free>\d+) of (?P<blocks>\d+)"
    r" blocks free; after it, (?P<after_running>\d+) running and (?P<after_waiting>\d+) waiting"
)
"""A run's line, as probe.py prints it in the KV case. Its time to the 16th token is "—" for a reply that ended before
it; the README does not use it."""


def runs(engine: str) -> list[dict[str, int]]:
    """Each run in ``engine``'s log, its numbers by name. A line that is no run of the KV case is refused."""
    found = []
    for number, line in enumerate((HERE / f"{engine}-kv.log").read_text().splitlines(), 1):
        run = RUN.fullmatch(line)
        if run is None:
            raise ValueError(f"{engine}-kv.log:{number}: not a run of the KV case: {line!r}")
        found.append({key: int(value) for key, value in run.groupdict().items()})
    return found


def span(values: list[int], unit: str = "") -> str:
    """The range of ``values`` as the docs print it, with an en dash, or the one value they all share."""
    low, high = min(values), max(values)
    return (f"{low:,}" if low == high else f"{low:,}{DASH}{high:,}") + unit


def table() -> str:
    """The first-token table, one row a run, and the load every run was sent into, which must be the same on both
    engines but for the voice turn's few tokens of difference."""
    by_engine = {engine: runs(engine) for engine in ENGINES}
    counts = {len(found) for found in by_engine.values()}
    if len(counts) != 1:
        raise ValueError(f"the logs have different counts of runs: {sorted(counts)}")
    every = [run for found in by_engine.values() for run in found]
    load = {key: {run[key] for run in every} for key in ("running", "tokens", "free", "blocks")}
    if any(len(values) != 1 for values in load.values()):
        raise ValueError(f"the runs were sent into different loads: {load}")
    running, tokens, free, blocks = (values.pop() for values in load.values())
    rows = [
        "| Run | " + " | ".join(ENGINES.values()) + " |",
        "|---:|" + "---:|" * len(ENGINES),
        *(
            f"| {number} | " + " | ".join(f"{run['ttft']:,} ms" for run in row) + " |"
            for number, row in enumerate(zip(*by_engine.values(), strict=True), 1)
        ),
    ]
    caption = (
        f"The voice turn's time to its first token. Every run sent the turn, {span([r['prompt'] for r in every])}"
        f" tokens, with {running} agent requests of {tokens:,} tokens running and {free} of the pool's {blocks:,} blocks"
        " free."
    )
    return "\n".join(rows) + "\n\n" + textwrap.fill(caption, WIDTH, break_on_hyphens=False)


if __name__ == "__main__":
    print(table())
