"""The features search's rows by label: totals and stalls the options cause. python3 features_sum.py"""
import collections
import glob
import json

out = {}
for label in ("fix", "ctl"):
    totals = collections.Counter()
    caused = []
    not_caused = 0
    not_caused_running = 0
    errors = []
    long_idle = 0
    for p in glob.glob(f"features-{label}-*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            if "totals" in r:
                totals.update(r["totals"])
            elif "error" in r:
                errors.append(r)
            elif r.get("kind") == "long_idle":
                long_idle += 1
            elif r.get("caused"):
                caused.append(r)
            elif "caused" in r:
                not_caused += 1
                not_caused_running += any(v.startswith("RUNNING") for v in r["flip"]["stall_state"].values())
    print(label, dict(totals))
    print("  caused", len(caused), "not caused", not_caused, "errors", len(errors), "long idle", long_idle)
    assert totals["off_finished"] + totals["off_stuck"] + totals["off_error"] > 0, label
    out[label] = dict(seeds=totals["off_finished"] + totals["off_stuck"] + totals["off_error"] + totals["on_error"], caused=len(caused),
                      not_caused=not_caused, not_caused_running=not_caused_running, errors=len(errors), long_idle=long_idle, off_stuck=totals["off_stuck"], on_stuck=totals["on_stuck"])
    for e in errors[:6]:
        print("   error", e["seed"], e["side"], e["error"][:150], e.get("at", "")[:90])
    for r in caused[:6]:
        c = r["cfg"]
        print("   caused", r["seed"], {k: c[k] for k in ("kv", "cap", "window", "mode", "disable_chunked", "threshold", "budget")}, r["flip"].get("stall_state"))
with open("features-summary.json", "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
