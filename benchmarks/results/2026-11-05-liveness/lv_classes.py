"""Classify liveness rows: the 2x2 of baseline against candidate, and each stall's final state.
python3 lv_classes.py A.jsonl [B.jsonl ...]"""
import collections
import json
import sys

rows = [json.loads(line) for p in sys.argv[1:] for line in open(p)]
good = [r for r in rows if "error" not in r]
print("off finished/on finished", sum(r["off"]["finished"] and r["on"]["finished"] for r in good))
print("off finished/on STUCK   ", sum(r["off"]["finished"] and not r["on"]["finished"] for r in good))
print("off STUCK/on finished   ", sum(not r["off"]["finished"] and r["on"]["finished"] for r in good))
print("off STUCK/on STUCK      ", sum(not r["off"]["finished"] and not r["on"]["finished"] for r in good))


def shape(state):
    names = [v.split("(")[0] for v in state.values()]
    return ("running" if "RUNNING" in names else "no-running",
            "loads" if "WAITING_FOR_REMOTE_KVS" in names else "no-loads",
            "held-waiting" if any(v.startswith("WAITING(") and not v.endswith("(0)") for v in state.values()) else "no-held")


for label, sel, side in (
    ("baseline stuck, final state with options off", [r for r in good if not r["off"]["finished"]], "off"),
    ("candidate stuck, baseline finished, final state with options on",
     [r for r in good if r["off"]["finished"] and not r["on"]["finished"]], "on"),
):
    print(label, len(sel))
    for k, v in collections.Counter(shape(r[side]["state"]) for r in sel).most_common():
        print("   ", k, v)
    print("    session present:", sum("session" in r[side]["state"] for r in sel))
    print("    by kv:", dict(collections.Counter(r["cfg"]["kv"] for r in sel)), "by mode:", dict(collections.Counter(r["cfg"]["mode"] for r in sel)),
          "by blocks:", dict(collections.Counter(r["cfg"]["num_blocks"] for r in sel)), "by prefix:", dict(collections.Counter(r["cfg"]["prefix"] for r in sel)))

over = [r for r in good if r["off"]["finished"] and r["on"]["finished"] and r["on"]["longest_idle"] > r["cfg"]["window"]]
print("idle over the window:", len(over), "of which the baseline idles as long:", sum(r["off"]["longest_idle"] >= r["on"]["longest_idle"] for r in over))
c = collections.Counter()
for r in over:
    d = r["on"]["longest_idle"] - r["off"]["longest_idle"]
    c["<=0" if d <= 0 else "1-4" if d <= 4 else "5-16" if d <= 16 else "17-64" if d <= 64 else ">64"] += 1
print("idle on - idle off:", dict(c))
worst = sorted(over, key=lambda r: r["off"]["longest_idle"] - r["on"]["longest_idle"])[:5]
for r in worst:
    print("  seed", r["seed"], "window", r["cfg"]["window"], "off idle", r["off"]["longest_idle"], "on idle", r["on"]["longest_idle"], "kv", r["cfg"]["kv"], "mode", r["cfg"]["mode"])
