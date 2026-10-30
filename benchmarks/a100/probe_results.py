"""The A100 run's probe results (benchmarks/a100/probes.sh): what probe.py and tail.py print, read back.

    python3 probe_results.py table [RUNS]           every probe run as a Markdown table
    python3 probe_results.py pick-stock [RUNS] [--cap TOKENS]

RUNS is the probe blocks' directory (default ~/flatshare/runs/a100: block1, block2, block3 with one NAME.txt a run).
`pick-stock` picks C''s stock arm by P-1's rule (docs/BENCHMARKS.md, the optional arms): of the stock step-cap settings
blocks 1 and 2 ran, 512/256 (frozen), 2,048/0 and the time-equivalent cap, the one whose tail runs gave the calls the
lowest first-token p95, the median of its runs; it prints the setting's name and its engine flags on one line, and the
table on stderr. A percentile set by a turn that failed (printed with ≥) counts at its value, as E2's report censors
one.

Standard library only.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROBE = re.compile(
    r"^(?P<case>kv|prefill): priority-0 TTFT (?P<ttft>\d+) ms, (?P<marks>.*?), for (?P<voice>\d+) prompt tokens, sent"
    r" with (?P<running>\d+) priority-2 requests of (?P<prompt>\d+) tokens running and (?P<free>\d+) of"
    r" (?P<blocks>\d+) blocks free; after it, (?P<after_running>\d+) running and (?P<after_waiting>\d+) waiting$"
)
MARK = re.compile(r"(?P<tokens>\d+) tokens (?:(?P<ms>\d+) ms|—)")
SPREAD = re.compile(
    r"p50 (?P<p50>≥?\d+) p90 (?P<p90>≥?\d+) p95 (?P<p95>≥?\d+) max (?P<max>≥?\d+) ms"
)
CALLS = re.compile(
    r"^calls: (?P<turns>\d+) turns, (?P<failed>\d+) failed(?: \(counted at [\d,]+ ms\), TTFT (?P<spread>.*? ms))?;"
    r" agents: (?P<stopped>\d+) of (?P<agents>\d+) stopped early$"
)
REPLIES = re.compile(r"^replies: (?P<turns>\d+) turns, (?P<marks>.*)$")
THROUGHPUT = re.compile(
    r"^agents: (?P<requests>\d+) requests in (?P<seconds>\d+) s, (?P<rate>[\d.]+) a second, with (?P<prompt>[\d,]+)"
    r" prompt and (?P<output>[\d,]+) output tokens a second$"
)
STOCK = (
    ("frozen", "block1", "frozen", ()),
    (
        "2048",
        "block2",
        "stock2048",
        ("--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "0"),
    ),
)
"""The stock settings blocks 1 and 2 ran before the time-equivalent cap: name, block, run prefix, engine flags."""


def spread(text: str) -> dict[str, tuple[int, bool]]:
    """A tail line's percentiles in ms, each with whether a failed turn set it (≥)."""
    m = SPREAD.search(text)
    if not m:
        raise ValueError(f"no percentiles in {text!r}")
    return {k: (int(v.lstrip("≥")), v.startswith("≥")) for k, v in m.groupdict().items()}


@dataclass
class Probe:
    """One probe.py run: the voice turn's first token and marks in ms (None where the stream ended first)."""

    case: str
    ttft_ms: int
    marks: dict[int, int | None]
    voice_tokens: int
    running: int
    prompt_tokens: int
    free: int
    blocks: int
    after_running: int
    after_waiting: int


@dataclass
class Tail:
    """One tail.py run: the calls' turns and first-token percentiles, the whole replies' marks, the agents'."""

    turns: int
    failed: int
    ttft: dict[str, tuple[int, bool]] | None
    stopped: int
    agents: int
    replies: dict[int, dict[str, tuple[int, bool]]] = field(default_factory=dict)
    requests_per_s: float | None = None


def parse_probe(text: str) -> Probe:
    for line in text.splitlines():
        if m := PROBE.match(line.strip()):
            marks = {
                int(mark["tokens"]): (int(mark["ms"]) if mark["ms"] else None)
                for mark in MARK.finditer(m["marks"])
            }
            return Probe(
                m["case"],
                int(m["ttft"]),
                marks,
                *(
                    int(m[k])
                    for k in (
                        "voice",
                        "running",
                        "prompt",
                        "free",
                        "blocks",
                        "after_running",
                        "after_waiting",
                    )
                ),
            )
    raise ValueError("no probe result line")


def parse_tail(text: str) -> Tail:
    result = None
    replies: dict[int, dict[str, tuple[int, bool]]] = {}
    rate = None
    for line in text.splitlines():
        line = line.strip()
        if m := CALLS.match(line):
            ttft = spread(m["spread"]) if m["spread"] else None
            result = Tail(
                int(m["turns"]), int(m["failed"]), ttft, int(m["stopped"]), int(m["agents"])
            )
        elif m := REPLIES.match(line):
            for part in m["marks"].split(", "):
                tokens, _, rest = part.partition(" tokens ")
                replies[int(tokens)] = spread(rest)
        elif m := THROUGHPUT.match(line):
            rate = float(m["rate"])
    if result is None:
        raise ValueError("no calls line")
    result.replies, result.requests_per_s = replies, rate
    return result


def runs(root: Path, block: str, prefix: str) -> dict[str, Probe | Tail]:
    """The block's completed runs of a prefix (NAME.txt; a .partial is a run that did not end), by name."""
    found: dict[str, Probe | Tail] = {}
    for path in sorted((root / block).glob(f"{prefix}-*.txt")):
        text = path.read_text()
        found[path.stem] = parse_tail(text) if "-tail-" in path.stem else parse_probe(text)
    return found


def pick_stock(
    root: Path, cap: int
) -> tuple[str, tuple[str, ...], list[tuple[str, list[int], float]]]:
    """The stock setting with the lowest median p95 of the calls' first tokens over its tail runs."""
    settings = [
        *STOCK,
        (
            str(cap),
            "block2",
            f"stock{cap}-{cap // 2}",  # by cap and threshold: 2,048/1,024 is not 2,048/0's stock2048
            ("--max-num-batched-tokens", str(cap), "--long-prefill-token-threshold", str(cap // 2)),
        ),
    ]
    table = []
    best = None
    for name, block, prefix, flags in settings:
        tails = [
            run for run in runs(root, block, f"{prefix}-tail").values() if isinstance(run, Tail)
        ]
        p95s = [run.ttft["p95"][0] for run in tails if run.ttft]
        if not p95s:
            raise ValueError(f"no tail run of {name} in {root / block} ({prefix}-tail-N.txt)")
        median = statistics.median(p95s)
        table.append((name, p95s, median))
        if best is None or median < best[2]:
            best = (name, flags, median)
    assert best is not None
    return best[0], best[1], table


def markdown(root: Path) -> str:
    """Every run: a probe's voice turn (its first token, 16th and 40th), a tail run's calls (p50 of each, and the
    first token's p95), in ms; ≥ marks a value a failed turn set."""

    def ms(value: tuple[int, bool] | int | None) -> str:
        if value is None:
            return ""
        if isinstance(value, tuple):
            return ("≥" if value[1] else "") + str(value[0])
        return str(value)

    lines = [
        "| Run | Case | First token | 16th token | 40th token | First token p95 | Notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for block in ("block1", "block2", "block3"):
        for name, run in runs(root, block, "*").items():
            if isinstance(run, Probe):
                cells = [
                    run.case,
                    ms(run.ttft_ms),
                    ms(run.marks.get(16)),
                    ms(run.marks.get(40)),
                    "",
                ]
                note = f"{run.running} x {run.prompt_tokens:,} tokens, {run.free} of {run.blocks} blocks free"
            else:
                ttft = run.ttft or {}
                cells = [
                    "tail (p50)",
                    ms(ttft.get("p50")),
                    ms(run.replies.get(16, {}).get("p50")),
                    ms(run.replies.get(40, {}).get("p50")),
                    ms(ttft.get("p95")),
                ]
                note = f"{run.turns} turns, {run.failed} failed, {run.stopped} of {run.agents} agents stopped early"
            lines.append(f"| {block}/{name} | " + " | ".join(cells) + f" | {note} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_results.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    default = Path.home() / "flatshare" / "runs" / "a100"
    p = commands.add_parser("table", help="every probe run as a Markdown table")
    p.add_argument("runs", type=Path, nargs="?", default=default)
    p = commands.add_parser("pick-stock", help="C''s stock arm by P-1's rule")
    p.add_argument("runs", type=Path, nargs="?", default=default)
    p.add_argument(
        "--cap", type=int, default=1536, help="block 2's time-equivalent cap (default %(default)s)"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "table":
            sys.stdout.write(markdown(args.runs))
        else:
            name, flags, table = pick_stock(args.runs, args.cap)
            for setting, p95s, median in table:
                print(f"{setting}: first-token p95 {p95s} ms, median {median:g}", file=sys.stderr)
            print(" ".join((name, *flags)))
    except ValueError as e:
        print(f"probe_results: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
