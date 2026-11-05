"""The baseline on main itself: the search's workloads on a tree of main (with the PRs' test file beside it, for its helpers
only), against the off side the rows recorded on #65283's head. All the workloads stuck there, and the first FIRST of each
profile. python lv_main.py PROFILE ROWS_DIR LABEL FIRST SHARD SHARDS OUT  (from the main tree)"""
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import liveness as lv  # noqa: E402

profile_label, first, shard, shards, out = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
rows_dir = sys.argv[2]
SIDE = os.environ.get("LV_COMPARE", "off")  # "on": the negative control, main against the options' schedule
helpers = lv.t


def drop_unknown(fn):
    def create(**kw):
        for name in ("max_priority_preemptions_per_step", "priority_prefill_budget_cap", "priority_prefill_budget_cap_tokens"):
            assert not kw.pop(name, None), name
        inflight = kw.pop("max_num_inflight_kv_loads", None)
        s = fn(**kw)
        s.scheduler_config.max_num_inflight_kv_loads = inflight
        return s
    return create


# _create_deferring_scheduler calls _create_scheduler through the module's global, so wrap that one only.
helpers.create_scheduler = (lambda orig: lambda **kw: drop_unknown(orig)(**kw))(helpers.create_scheduler)
assert not hasattr(helpers.Scheduler, "_make_room"), "this is not main"

rows = [json.loads(line) for p in glob.glob(f"{rows_dir}/lv-{profile_label}-*.jsonl") for line in open(p)]
sel = [r for r in rows if not r["off"]["finished"] or r["seed"] < first]
sel = [r for i, r in enumerate(sorted(sel, key=lambda r: r["seed"])) if i % shards == shard]
same = differ = 0
with open(out, "w") as f:
    for r in sel:
        cfg, reqs = lv.config(r["seed"])
        got = lv.run(cfg, reqs, False)
        keys = ("finished", "steps", "longest_idle")
        if all(got[k] == r[SIDE][k] for k in keys):
            same += 1
        else:
            differ += 1
            f.write(json.dumps(dict(seed=r["seed"], main={k: got[k] for k in keys}, head={k: r[SIDE][k] for k in keys})) + "\n")
print(f"{profile_label} shard {shard}: {len(sel)} workloads, stuck on the head's off side {sum(not r['off']['finished'] for r in sel)}, same on main {same}, different {differ}")
