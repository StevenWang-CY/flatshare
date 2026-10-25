"""table.py [DIR]: the budget cap's window sweep (knee.sh) as a table, from its runs' outputs in DIR (wWINDOW-RUN.out):
for each window, the two runs' calls' first token, 16th token and 40th (last) token, and the agents' prompt tokens a
second, each as both runs' values; then each window's medians over its runs. Prints Markdown; medians(DIR) gives the
second table's values, as readme/fill.py reads them."""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLIES = re.compile(
    r"^replies: (\d+) turns, 16 tokens p50 (\d+) p90 (\d+) p95 (\d+) max (\d+) ms, "
    r"40 tokens p50 (\d+) p90 (\d+) p95 (\d+) max (\d+) ms$"
)
AGENTS = re.compile(
    r"^agents: (\d+) requests in (\d+) s, [\d.]+ a second, with ([\d,]+) prompt and ([\d,]+) output"
)
CALLS = re.compile(
    r"^calls: (\d+) turns, (\d+) failed .*TTFT p50 (\d+) p90 (\d+) p95 (\d+) max (\d+) ms"
)
KEYS = ("ttft_p95", "t16_p95", "t40_p95", "prompt_tps")


def run(path: Path) -> dict:
    """One run's summary lines: its calls' first token, its replies' 16th and 40th tokens, the agents' throughput."""
    out = {}
    for line in path.read_text().splitlines():
        if m := REPLIES.match(line):
            out |= {"t16_p95": int(m[4]), "t40_p50": int(m[6]), "t40_p95": int(m[8])}
        elif m := AGENTS.match(line):
            out |= {"prompt_tps": int(m[3].replace(",", ""))}
        elif m := CALLS.match(line):
            out |= {
                "turns": int(m[1]),
                "failed": int(m[2]),
                "ttft_p50": int(m[3]),
                "ttft_p95": int(m[5]),
            }
    missing = set(KEYS) - out.keys()
    if missing:
        raise SystemExit(f"{path}: no {', '.join(sorted(missing))} line")
    return out


def runs(directory: Path) -> dict[int, list[dict]]:
    """Each window's runs, in window order."""
    windows = sorted({int(p.name[1:].split("-")[0]) for p in directory.glob("w*-[0-9].out")})
    return {w: [run(p) for p in sorted(directory.glob(f"w{w}-[0-9].out"))] for w in windows}


def medians(directory: Path = HERE) -> dict[int, dict[str, float]]:
    """Each window's medians over its runs, of the first token's, the 16th's and the 40th's p95 and the agents'
    prompt tokens a second."""
    return {
        w: {k: statistics.median(r[k] for r in rs) for k in KEYS}
        for w, rs in runs(directory).items()
    }


def both(rs: list[dict], key: str) -> str:
    """A window's runs' values of ``key``, one after the other."""
    return " / ".join(f"{r[key]:,}" for r in rs)


if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    rows = runs(directory)
    print(
        "| Window (tokens) | First token p95 | 16th token p95 | 40th token p50 / p95 | Agents' prompt tokens/s | Turns, failed |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for w, rs in rows.items():
        t40 = " · ".join(f"{r['t40_p50']:,} / {r['t40_p95']:,}" for r in rs)
        turns = " · ".join(f"{r['turns']}, {r['failed']}" for r in rs)
        print(
            f"| {w} | {both(rs, 'ttft_p95')} | {both(rs, 't16_p95')} | {t40} | {both(rs, 'prompt_tps')} | {turns} |"
        )
    print()
    print(
        "| Window (tokens) | First token p95 | 16th token p95 | 40th token p95 | Agents' prompt tokens/s |"
    )
    print("|---:|---:|---:|---:|---:|")
    for w, m in medians(directory).items():
        print(
            f"| {w} | {m['ttft_p95']:,.0f} | {m['t16_p95']:,.0f} | {m['t40_p95']:,.0f} | {m['prompt_tps']:,.0f} |"
        )
