"""The search's rows from its summaries and probes: each code's workloads, stalls the options cause, other stalls and stuck
workloads. Checks that no head has a stall the options cause, that the code before the fix has some, and that each probed
stall repeated. Exits 1 if a check fails.

    python3 check.py [DIR]
"""

import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
summary = json.loads((d / "summary.json").read_text())
module = json.loads((d / "module-summary.json").read_text())
fails = []


def row(name, keys, table):
    rows = [table[k] for k in keys]
    seeds = sum(r["seeds"] for r in rows)
    caused = sum(r["caused"] for r in rows)
    other = sum(r["only_on_stuck"] for r in rows) - caused
    off, on = sum(r["off_stuck"] for r in rows), sum(r["on_stuck"] for r in rows)
    print(f"{name}: {seeds:,} workloads, {caused:,} stalls the options cause, {other:,} other stalls, stuck off/on {off:,}/{on:,}")
    if any(r["errors"] for r in rows):
        fails.append(f"{name}: errors")
    if any(r["repeated"] != r["probed"] for r in rows):
        fails.append(f"{name}: a stall did not repeat")
    return caused


before = row("#65283 before the fix", ["base #65283 before the fix", "ext #65283 before the fix"], summary)
head = row("#65283", ["base #65283", "ext #65283"], summary)
alone = row("#65282 alone", ["base #65282", "ext #65282"], summary)
old = row("the make-room module of v1.22", ["old base", "old ext"], module)
new = row("the make-room module of v1.23", ["new base", "new ext"], module)
if head or alone or new:
    fails.append("a head has a stall the options cause")
if not before or not old:
    fails.append("the code before the fix has no stall the options cause")
for label in ("fix", "ctl", "a", "xfix", "xctl", "xa"):
    for line in (d / f"probe-{label}.jsonl").read_text().splitlines():
        if json.loads(line)["on"]:
            fails.append(f"probe-{label}: a stall did not repeat")
features = json.loads((d / "features-summary.json").read_text())
print(f"the features search: {features['fix']['seeds']:,} workloads, {features['fix']['caused']:,} stalls the options cause on the head, {features['ctl']['caused']:,} on the revision before")
if features["fix"]["caused"] or features["fix"]["errors"] or not features["ctl"]["caused"]:
    fails.append("the features search")
replay = (d / "probe-replay.out").read_text()
if not replay.strip().endswith("ok") or replay.count("same on both modules: True") != 8:
    fails.append("the probes' replay")
print("\n".join(fails) or "all checks pass")
sys.exit(1 if fails else 0)
