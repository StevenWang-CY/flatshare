"""long_traces.py CORPUS OUT N: link the N traces of a scaled Weka corpus whose requests carry the largest contexts
(the median `in` of the main agent's requests; subagent entries nest their own) into OUT. E2's discriminating scenario replays them:

    python benchmarks/e2/long_traces.py ~/flatshare/data/weka-x32 ~/flatshare/data/weka-x32-long 48

OUT's links are then those N, each to its trace in CORPUS: a link of an earlier run to a trace not among them goes,
and one to another file is pointed at the trace; a file that is not a link stays. A corpus with no trace is an error.
"""

import json
import statistics
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3 or not argv[2].isdigit() or int(argv[2]) < 1:
        print("usage: long_traces.py CORPUS OUT N (N: 1 or more traces)", file=sys.stderr)
        return 2
    corpus, out, n = Path(argv[0]), Path(argv[1]), int(argv[2])
    medians = []
    for path in sorted(corpus.glob("*.json")):
        inputs = [r["in"] for r in json.loads(path.read_text())["requests"] if "in" in r]
        if inputs:
            medians.append((statistics.median(inputs), len(inputs), path))
    if not medians:
        print(f"long_traces: no trace with requests in {corpus} (*.json)", file=sys.stderr)
        return 1
    medians.sort(key=lambda row: row[0], reverse=True)
    chosen = medians[:n]
    targets = {path.name: path.resolve() for _, _, path in chosen}
    out.mkdir(parents=True, exist_ok=True)
    for link in out.glob("*.json"):
        if link.is_symlink() and link.readlink() != targets.get(link.name):
            link.unlink()
    for name, target in targets.items():
        link = out / name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(target)
    print(
        f"{len(medians)} traces, median input {statistics.median(m for m, _, _ in medians):,.0f}; "
        f"the {len(chosen)} largest: median input {statistics.median(m for m, _, _ in chosen):,.0f}, "
        f"{sum(c for _, c, _ in chosen):,} requests"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
