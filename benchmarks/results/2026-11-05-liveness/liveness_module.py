"""liveness.py on the make-room module built for vLLM 0.32.0, with the serving profile's switches: two preemptions a step
or none as drawn, the budget cap with T as drawn (1, 4, 16 or 48, the profile's), take_reservations, decode_hold_tokens =
T, and retention of two blocks on half the workloads, on a clock that a step moves by 50 ms, so that it expires. The
baseline runs the module with no switches, which schedules as stock vLLM. 0.32.0 has no prefill cadence and this
harness's helper no encoder input at an offset, so the ext profile here adds failed loads, aborts, a single slot, small
budgets and longer outputs. A turn's announced hold is left out: it ends by the clock.

  LV_MODULE=<dir with flatshare_make_room_scheduler.py> LV_FLATSHARE=<checkout> PYTHONPATH=<vLLM 0.32.0> \
    python liveness_module.py OUT.jsonl FIRST LAST
"""
import importlib.util
import json
import os
import pathlib
import random
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.environ["LV_MODULE"])
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import flatshare_make_room_scheduler as module  # noqa: E402
import liveness as lv  # noqa: E402

tests = pathlib.Path(os.environ["LV_FLATSHARE"]) / "python/flatshare_vllm/tests"
spec = importlib.util.spec_from_file_location("flatshare_test_make_room", tests / "test_make_room.py")
tm = importlib.util.module_from_spec(spec)
sys.modules["flatshare_test_make_room"] = tm
spec.loader.exec_module(tm)
spec = importlib.util.spec_from_file_location("flatshare_conftest", tests / "conftest.py")
conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conftest)
tm.init_none_hash(tm.sha256)

model_dir = pathlib.Path(os.environ["LV_MODEL_DIR"])
model_dir.mkdir(parents=True, exist_ok=True)
conftest.write_model(model_dir, 65536)

clock = SimpleNamespace(now=1000.0)
module.time = SimpleNamespace(monotonic=lambda: clock.now, time=time.time)
current = {}
lv.FLIP.update(take_reservations=False, decode_hold_tokens=0, retain_blocks=0)


def create(cls, **o):
    cap, kv, window = o["priority_prefill_budget_cap"], o["max_priority_preemptions_per_step"], o["priority_prefill_budget_cap_tokens"]
    make_room = {}
    if kv:
        make_room["max_preemptions_per_step"] = kv
    if cap:
        make_room.update(budget_cap=True, budget_cap_tokens=window, take_reservations=True, decode_hold_tokens=window)
    if make_room and current["cfg"]["retain"]:
        make_room.update(retain_tokens=current["cfg"]["retain"] * 16, retain_s=1)
    num_blocks = o["num_blocks"]
    s = tm.scheduler(
        cls, str(model_dir), num_blocks, make_room, max_num_batched_tokens=o["max_num_batched_tokens"],
        long_prefill_token_threshold=o["long_prefill_token_threshold"], max_model_len=(num_blocks - 3) * 16,
        watermark=o["watermark"], prefix_caching=o["enable_prefix_caching"], max_num_seqs=o["max_num_seqs"],
        connector=o["use_kv_connector"])
    schedule = s.schedule

    def step(_throttle=False):
        clock.now += 0.05
        return schedule()

    s.schedule = step
    return s


def deferring(**o):
    s = create(module.AsyncScheduler, **o)
    s.defer_block_free = True
    return s


lv.t = SimpleNamespace(
    RequestStatus=tm.RequestStatus,
    mock_kv=lambda matched_tokens, is_async: tm.Connector(),
    _create_scheduler=lambda **o: create(module.Scheduler, **o),
    _create_deferring_scheduler=deferring,
    _session_request=lambda n, max_tokens, prio: tm.request("session", n, prio, 0, max_tokens=max_tokens, resumable=True),
    _request=lambda rid, n, prio, arrival_time, max_tokens, prompt_token: tm.request(
        rid, n, prio, arrival_time, max_tokens=max_tokens, token=prompt_token),
    _model_runner_output=tm.output,
)
base_config = lv.config


def config(seed):
    cfg, reqs = base_config(seed)
    rng = random.Random(seed * 40503 % 2**32)
    cfg.update(window=rng.choice([1, 4, 16, 48]), inflight=None, retain=rng.choice([0, 2]), encoder_cache=0, drafts=0, interval=1)
    for r in reqs:
        r.pop("encoder", None)
        # A prompt longer than the module's max_model_len (the pool less three blocks) would never be admitted.
        limit = (cfg["num_blocks"] - 3) * 16 - 32
        r["n"] = min(r["n"], limit)
        r["max_tokens"] = min(r["max_tokens"], 16)
        r["matched"] = min(r["matched"], (r["n"] - 1) // 16 * 16)
    return cfg, reqs


base_run = lv.run


def run(cfg, reqs, on, cap=None, flip=False, flip_idle=600):
    current["cfg"] = cfg
    return base_run(cfg, reqs, on, cap=cap, flip=flip, flip_idle=flip_idle)


lv.config = config
lv.run = run


def main():
    out, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with open(out, "w") as f:
        for seed in range(first, last):
            cfg, reqs = lv.config(seed)
            row = dict(seed=seed, cfg=cfg)
            try:
                row["off"] = lv.run(cfg, reqs, False)
                row["on"] = lv.run(cfg, reqs, True)
                if row["off"]["finished"] and not row["on"]["finished"]:
                    row["on_flip"] = lv.run(cfg, reqs, True, flip=True)
            except Exception as e:  # recorded, not hidden: a crash on one side is a finding
                tb = traceback.extract_tb(e.__traceback__)[-1]
                row["error"] = f"{type(e).__name__}: {e} at {tb.filename.rsplit('/', 1)[-1]}:{tb.lineno} {tb.line}"[:300]
                row["side"] = "on" if "off" in row else "off"
            f.write(json.dumps(row) + "\n")
            f.flush()


main()
