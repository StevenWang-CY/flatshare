"""Summarize liveness.py's rows: seeds where the baseline finishes and the candidate does not (stalls), errors, the
longest idle run against the window. python3 liveness_sum.py A.jsonl [B.jsonl ...]"""
import collections
import json
import sys

rows = [json.loads(line) for p in sys.argv[1:] for line in open(p)]
errors = [r for r in rows if "error" in r]
good = [r for r in rows if "error" not in r]
base_ok = [r for r in good if r["off"]["finished"]]
stalls = [r for r in base_ok if not r["on"]["finished"]]
print(f"{len(rows)} seeds; errors {len(errors)}; baseline finished {len(base_ok)}; baseline stuck {len(good) - len(base_ok)}; "
      f"STALLS with the options on: {len(stalls)}")
by = collections.Counter((r["cfg"]["kv"], r["cfg"]["mode"], r["cfg"]["window"]) for r in stalls)
if by:
    print("stalls by (kv, mode, window):", dict(by))
for r in stalls[:4]:
    print("  seed", r["seed"], {k: r["cfg"][k] for k in ("num_blocks", "budget", "seqs", "window", "kv", "inflight", "mode")},
          r["on"].get("state"))
for r in errors[:4]:
    print("  error seed", r["seed"], r["error"][:200])
over = [r for r in base_ok if r["on"]["finished"] and r["on"]["longest_idle"] > r["cfg"]["window"]]
print("finished with an idle run longer than the window:", len(over), "; longest idle overall:",
      max((r["on"]["longest_idle"] for r in base_ok), default=0))
slow = sorted(((r["on"]["steps"] - r["off"]["steps"], r["seed"]) for r in base_ok if r["on"]["finished"]), reverse=True)[:3]
print("largest extra steps (on - off):", slow)
