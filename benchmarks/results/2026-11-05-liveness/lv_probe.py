"""For every seed in ROWS where the options stall a workload the baseline finishes: attribute it (which option alone
stalls it) and probe it (turn both options off at the stall and go on: a stall the options cause then resumes; a state
the scheduler cannot leave stays stuck).  python lv_probe.py ROWS.jsonl... > OUT  (from a vLLM tree)"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import liveness as lv  # noqa: E402

rows = [json.loads(line) for p in sys.argv[1:] for line in open(p)]
sel = [r for r in rows if "error" not in r and r["off"]["finished"] and not r["on"]["finished"]]
for r in sel:
    cfg, reqs = lv.config(r["seed"])
    res = dict(seed=r["seed"], window=cfg["window"], kv=cfg["kv"], mode=cfg["mode"], session=any(q.get("session") for q in reqs))
    res["on"] = lv.run(cfg, reqs, True)["finished"]
    res["kv_only"] = lv.run(cfg, reqs, True, cap=False)["finished"]
    kv0 = dict(cfg, kv=0)
    res["cap_only"] = lv.run(kv0, reqs, True, cap=True)["finished"]
    flip = lv.run(cfg, reqs, True, flip=True)
    res["after_flip_finished"] = flip["finished"]
    res["resumed"] = flip.get("resumed", False)  # The next step schedules something, whatever happens later.
    res["stall_state"] = flip.get("stall_state")
    res["free"] = flip.get("free")
    print(json.dumps(res), flush=True)
