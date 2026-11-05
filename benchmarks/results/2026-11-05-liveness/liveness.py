"""A liveness search over the real scheduler: random workloads on small KV pools, with KV connector loads, shared
prefixes, preemption and async scheduling. A workload that finishes with both options off must finish with them on, and
the longest run of steps that schedule nothing (with no load in flight to wait for) is reported.

    python liveness.py OUT.jsonl FIRST_SEED LAST_SEED     (from a vLLM tree with the PRs' test helpers)

One JSON line per seed: the config, the baseline's and the candidate's steps, whether each finished, the longest idle run.
"""
import collections
import importlib
import inspect
import json
import os
import random
import sys
import traceback

if os.environ.get("LV_MODULE"):  # liveness_module.py sets t and HAS_CAP to the make-room module's.
    t = None
    HAS_CAP = True
else:
    t = importlib.import_module("tests.v1.core.test_priority_make_room")
    from tests.v1.core.utils import create_scheduler

    HAS_CAP = "priority_prefill_budget_cap" in inspect.signature(create_scheduler).parameters
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager  # noqa: E402

# What `flip` turns off.
FLIP = {"max_priority_preemptions_per_step": 0, "priority_prefill_budget_cap": False}
LIMIT = 4000
B = 16
# "ext" adds failed loads, aborts, encoder inputs on a small encoder cache, prefill throttling, draft tokens, a single
# running slot, budgets below the block size and longer outputs, on a second generator so "base" keeps its seeds' meaning.
PROFILE = os.environ.get("LV_PROFILE", "base")


def workload(rng, num_blocks):
    pool = (num_blocks - 1) * B
    reqs = []
    for i in range(rng.randint(3, 10)):
        n = rng.randint(4, max(5, int(pool * 0.6)))
        shared = rng.random() < 0.3
        matched = 0
        if rng.random() < 0.5 and n > B:
            matched = rng.randint(1, max(1, (n - 1) // B)) * B
        reqs.append(dict(
            id=f"r{i}", n=n, prio=rng.randint(0, 2), arrive=rng.randint(0, 25), max_tokens=rng.randint(1, 6),
            token=1 if shared else 100 + i, matched=matched, delay=rng.randint(1, 14)))
    if rng.random() < 0.3:  # A streaming-input session: a chunk, a pause holding its KV blocks, a next chunk, its end.
        reqs.append(dict(
            id="session", session=True, n=rng.randint(4, max(5, int(pool * 0.3))), next=rng.randint(4, max(5, int(pool * 0.3))),
            prio=rng.randint(0, 2), arrive=rng.randint(0, 25), max_tokens=rng.randint(1, 3), gap=rng.randint(0, 12),
            token=0, matched=0, delay=1))
    return reqs


def config(seed):
    rng = random.Random(seed)
    num_blocks = rng.choice([12, 15, 20, 32, 48])
    cfg = dict(
        num_blocks=num_blocks, budget=rng.choice([16, 32, 64, 128]), seqs=rng.choice([2, 3, 4, 8]),
        window=rng.choice([1, 4, 16]), kv=rng.choice([0, 2]), threshold=rng.choice([0, 16, 64]),
        watermark=rng.choice([0.0, 0.07]), prefix=rng.random() < 0.5, inflight=rng.choice([None, 2]),
        mode=rng.choice(["sync", "sync", "async"]), cap=HAS_CAP)
    reqs = workload(rng, num_blocks)
    if PROFILE == "ext":
        extend(random.Random(seed * 2654435761 % 2**32), cfg, reqs)
    return cfg, reqs


def extend(rng, cfg, reqs):
    cfg.update(
        encoder_cache=rng.choice([0, 0, 16, 32, 64]), drafts=rng.choice([0, 0, 0, 2]), interval=rng.choice([1, 1, 1, 2, 3]),
        recompute=rng.random() < 0.5, seqs=rng.choice([1, 2, 3, 4, 8]), budget=rng.choice([8, 16, 32, 64, 128]))
    for r in reqs:
        if r.get("session"):
            continue
        r["fail"] = bool(r["matched"]) and rng.random() < 0.2
        r["abort"] = r["arrive"] + rng.randint(0, 60) if rng.random() < 0.15 else None
        r["max_tokens"] = rng.choice([r["max_tokens"], rng.randint(1, 24)])
        if cfg["encoder_cache"] and r["n"] > 8 and rng.random() < 0.4:
            length = rng.randint(4, min(cfg["encoder_cache"], r["n"] - 1))
            r["encoder"] = (length, rng.randint(0, r["n"] - length))


def _settled(s, cfg, in_flight):
    """After the last request finished and an idle step: the blocks not back in the pool and what the scheduler still keeps."""
    while in_flight:
        s.update_from_output(*in_flight.popleft())
    out = s.schedule()
    s.update_from_output(out, t._model_runner_output(s, out))
    left = {name: len(getattr(s, name)) for name in (
        "running", "requests", "finished_recving_kv_req_ids", "failed_recving_kv_req_ids", "_inflight_prefills",
        "_num_wait_hold_steps", "_load_hold_start", "_make_room_untried") if len(getattr(s, name, ()))}
    return dict(blocks_out=cfg["num_blocks"] - 1 - s.kv_cache_manager.block_pool.get_num_free_blocks(), left=left)


def _state(objs):
    return {k: f"{q.status.name}({q.num_computed_tokens})" for k, q in objs.items() if not q.is_finished()}


def run(cfg, reqs, on: bool, cap=None, flip=False, flip_idle=600):
    """`cap` overrides whether the budget cap is on (attribution); `flip` turns both options off after `flip_idle` idle
    steps and goes on, which tells a stall the options cause (it then finishes) from a state the scheduler cannot leave
    without them."""
    cap = (on and cfg["cap"]) if cap is None else cap
    options = dict(
        max_num_batched_tokens=cfg["budget"], num_blocks=cfg["num_blocks"], max_num_seqs=cfg["seqs"],
        long_prefill_token_threshold=cfg["threshold"], watermark=cfg["watermark"], enable_prefix_caching=cfg["prefix"],
        max_num_inflight_kv_loads=cfg["inflight"], max_model_len=(cfg["num_blocks"] + 8) * B,
        use_kv_connector=t.mock_kv(matched_tokens=0, is_async=False),
        max_priority_preemptions_per_step=cfg["kv"] if on else 0)
    if cfg.get("drafts"):
        options.update(num_speculative_tokens=cfg["drafts"], speculative_method="ngram_gpu")
    if HAS_CAP:
        options.update(priority_prefill_budget_cap=cap, priority_prefill_budget_cap_tokens=cfg["window"])
    if cfg["mode"] == "async":
        s = t._create_deferring_scheduler(**options)
    else:
        s = t._create_scheduler(**options)
    if cfg.get("encoder_cache"):
        s.max_num_encoder_input_tokens = cfg["encoder_cache"]
        s.encoder_cache_manager = EncoderCacheManager(cache_size=cfg["encoder_cache"])
    if PROFILE == "ext":
        s.recompute_kv_load_failures = cfg["recompute"]
    by_id = {r["id"]: r for r in reqs}
    s.connector.may_load_async = lambda request: bool(by_id[request.request_id]["matched"])
    s.connector.get_num_new_matched_tokens = lambda request, num_computed: (
        (by_id[request.request_id]["matched"] - num_computed, True)
        if by_id[request.request_id]["matched"] > num_computed else (0, False))
    objs = {}
    loading = {}
    reported = {}
    session = None
    failed_once = set()
    in_flight = collections.deque()
    idle = longest = 0
    stalled = {}
    pending = sorted(reqs, key=lambda r: r["arrive"])
    step = -1
    while step < LIMIT:
        step += 1
        while pending and pending[0]["arrive"] <= step:
            r = pending.pop(0)
            if r.get("session"):
                session = dict(r=r, stage=0, since=None)
                objs[r["id"]] = t._session_request(r["n"], r["max_tokens"], r["prio"])
            else:
                encoder = {}
                if r.get("encoder"):  # #65282's helper places an encoder input at the prompt's start only.
                    encoder["num_encoder_tokens"] = r["encoder"][0]
                    if "encoder_offset" in inspect.signature(t._request).parameters:
                        encoder["encoder_offset"] = r["encoder"][1]
                objs[r["id"]] = t._request(r["id"], r["n"], r["prio"], arrival_time=r["arrive"], max_tokens=r["max_tokens"],
                                           prompt_token=r["token"], **encoder)
            s.add_request(objs[r["id"]])
        for rid, q in objs.items():
            if by_id[rid].get("abort") is not None and step >= by_id[rid]["abort"] and not q.is_finished():
                s.finish_requests(rid, t.RequestStatus.FINISHED_ABORTED)
        if session and session["stage"] < 2 and objs["session"].status.name == "WAITING_FOR_STREAMING_REQ":
            session["since"] = step if session["since"] is None else session["since"]
            if session["stage"] == 1:
                s.finish_requests("session", t.RequestStatus.FINISHED_ABORTED)
                session["stage"] = 2
            elif step - session["since"] >= session["r"]["gap"]:
                s.add_request(t._session_request(session["r"]["next"], session["r"]["max_tokens"], session["r"]["prio"]))
                session.update(stage=1, since=None)
        out = s.schedule(cfg.get("interval", 1) > 1 and step % cfg["interval"] != 0)
        done = set()
        for rid, q in objs.items():
            if q.status.name != "WAITING_FOR_REMOTE_KVS" or (
                rid in reported and step - reported[rid] >= 2 and rid not in s.finished_recving_kv_req_ids
            ):  # The second: a failed load started over within one schedule() call.
                reported.pop(rid, None)
            if q.status.name == "WAITING_FOR_REMOTE_KVS" and rid not in loading and rid not in reported:
                loading[rid] = step + by_id[rid]["delay"]
        for rid, due in list(loading.items()):
            if step >= due:
                done.add(rid)
                reported[rid] = step  # Once: under async scheduling the report is still in flight next step.
                del loading[rid]
        output = t._model_runner_output(s, out, done or None)
        failed = {rid for rid in done if by_id[rid].get("fail") and rid not in failed_once}
        if failed:
            failed_once |= failed
            output.kv_connector_output.failed_recving = failed
        if cfg["mode"] == "async":
            in_flight.append((out, output))
            if len(in_flight) > 1:
                s.update_from_output(*in_flight.popleft())
        else:
            s.update_from_output(out, output)
        busy = out.num_scheduled_tokens or loading or in_flight and any(o[0].num_scheduled_tokens for o in in_flight)
        if busy or pending or (session and session["stage"] == 0 and session["since"] is not None):
            idle = 0
        else:
            idle += 1
            longest = max(longest, idle)
        if not pending and len(objs) == len(reqs) and all(q.is_finished() for q in objs.values()):
            return dict(finished=True, steps=step + 1, longest_idle=longest, **stalled, **_settled(s, cfg, in_flight))
        if stalled.get("stall_step") == step - 1:
            stalled["resumed"] = bool(out.num_scheduled_tokens)
        if idle > (flip_idle if flip else 600):
            if not flip:
                break
            stalled = dict(stall_step=step, stall_state=_state(objs), free=s.kv_cache_manager.block_pool.get_num_free_blocks())
            for name, value in FLIP.items():
                if hasattr(s, name):
                    setattr(s, name, value)
            flip, idle = False, 0
    return dict(finished=False, steps=step + 1, longest_idle=longest, state=_state(objs), **stalled)


def main():
    out, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with open(out, "w") as f:
        for seed in range(first, last):
            cfg, reqs = config(seed)
            row = dict(seed=seed, cfg=cfg)
            try:
                row["off"] = run(cfg, reqs, False)
                row["on"] = run(cfg, reqs, True)
            except Exception as e:  # recorded, not hidden: a crash on one side is a finding
                tb = traceback.extract_tb(e.__traceback__)[-1]
                row["error"] = f"{type(e).__name__}: {e} at {tb.filename.rsplit('/', 1)[-1]}:{tb.lineno} {tb.line}"[:300]
                row["side"] = "on" if "off" in row else "off"
            f.write(json.dumps(row) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
