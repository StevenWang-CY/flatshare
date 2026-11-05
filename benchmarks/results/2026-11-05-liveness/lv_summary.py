"""The search's numbers, read off the rows and probes in this directory, as JSON on the last line."""
import collections
import glob
import json

out = {}
for prefix, profile in (("", "base"), ("x", "ext")):
    for label, name in (("fix", "#65283"), ("ctl", "#65283 before the fix"), ("a", "#65282")):
        rows = [json.loads(line) for p in glob.glob(f"lv-{prefix}{label}-*.jsonl") for line in open(p)]
        probes = [json.loads(line) for line in open(f"probe-{prefix}{label}.jsonl")]
        errors = sum("error" in r for r in rows)
        good = [r for r in rows if "error" not in r]
        c = collections.Counter()
        for r in probes:
            names = [v.split("(")[0] for v in (r["stall_state"] or {}).values()]
            c["running" if "RUNNING" in names else "nothing running", "ends with the options off" if r["resumed"] or r["after_flip_finished"] else "stays stuck"] += 1
        assert not errors, f"{profile} {name}: {errors} rows are errors, e.g. {next(r['error'] for r in rows if 'error' in r)[:200]}"
        row = dict(
            seeds=len(rows), errors=errors,
            off_stuck=sum(not r["off"]["finished"] for r in good), on_stuck=sum(not r["on"]["finished"] for r in good),
            only_on_stuck=sum(r["off"]["finished"] and not r["on"]["finished"] for r in good),
            only_off_stuck=sum(r["on"]["finished"] and not r["off"]["finished"] for r in good),
            caused=sum(r["resumed"] or r["after_flip_finished"] for r in probes),
            repeated=sum(not r["on"] for r in probes), probed=len(probes),
            with_session=sum(r["session"] for r in probes),
            over_window=sum(r["off"]["finished"] and r["on"]["finished"] and r["on"]["longest_idle"] - r["off"]["longest_idle"] > r["cfg"]["window"] + 1 for r in good),
            shapes={" / ".join(k): v for k, v in c.most_common()})
        out[f"{profile} {name}"] = row
        print(profile, name, row)
print(json.dumps(out))
with open("summary.json", "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
