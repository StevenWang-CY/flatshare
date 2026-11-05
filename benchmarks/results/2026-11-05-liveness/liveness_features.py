"""A search over what liveness.py does not model: LoRA limits, connector lookups answered later, grammars (pending and
failed), pause windows, throttled and cadenced prefill, sliding-window groups, scheduler_reserve_full_isl off, deferred
frees, disable_chunked_mm_input and combinations. Prints a line for each stall or crash the options cause: off finishes, on
does not; `flip` turns the options off at the stall and goes on.
    SEARCH_CONTROL=1 python liveness_features.py FIRST LAST > OUT.jsonl   (from a vLLM tree)"""
import collections
import inspect
import json
import os
import random
import sys
import traceback
import types

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *  # noqa
import torch  # noqa: E402
from vllm.config import LoRAConfig  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402
from vllm.v1.core.sched.interface import PauseState  # noqa: E402
from vllm.v1.kv_cache_interface import SlidingWindowSpec  # noqa: E402

LIMIT = 1500
IDLE = 300
EXCLUDE_DISABLE_CHUNKED_THRESHOLD = True


def config(seed):
    r = random.Random(seed)
    num_blocks = r.choice([12, 15, 20, 32, 48])
    cfg = dict(
        num_blocks=num_blocks, budget=r.choice([8, 16, 32, 64, 128]), seqs=r.choice([1, 2, 3, 4, 8]), window=r.choice([1, 2, 4, 16]),
        kv=r.choice([0, 1, 2, 4]), cap=r.random() < 0.7, threshold=r.choice([0, 16, 64]), watermark=r.choice([0.0, 0.07]),
        prefix=r.random() < 0.5, inflight=r.choice([None, 2]), mode=r.choice(["sync", "sync", "async", "defer"]),
        reserve_full=r.random() < 0.8, max_loras=r.choice([None, None, 1, 2]), encoder_cache=r.choice([0, 0, 16, 32, 64]),
        disable_chunked=r.random() < 0.2, interval=r.choice([1, 1, 1, 2, 3]), cadence=r.choice([1, 1, 2]),
        window_spec=r.random() < 0.2, recompute=r.random() < 0.5,
        pauses=[(a, a + r.randint(1, 12), r.choice(["new", "new", "all"])) for a in sorted(r.sample(range(40), r.choice([0, 0, 1, 2])))])
    if not cfg["kv"] and not cfg["cap"]:
        cfg["cap"] = True
    if cfg["disable_chunked"] and not os.environ.get("SEARCH_CONTROL"):
        cfg["threshold"] = 0  # SEARCH_CONTROL=1 keeps it: a prefill may then wait at an image for the threshold to lift.
    pool = (num_blocks - 1) * BLOCK
    reqs = []
    for i in range(r.randint(3, 9)):
        n = r.randint(4, max(5, int(pool * 0.6)))
        matched = r.randint(1, max(1, (n - 1) // BLOCK)) * BLOCK if r.random() < 0.3 and n > BLOCK else 0
        q = dict(id=f"r{i}", n=n, prio=r.randint(0, 2), arrive=r.randint(0, 25), max_tokens=r.choice([1, 2, 3, 6, 24]),
                 token=1 if r.random() < 0.3 else 100 + i, matched=matched, delay=r.randint(1, 14), fail=bool(matched) and r.random() < 0.2,
                 none=r.choice([0, 0, 0, 2, 9]), abort=r.randint(0, 60) if r.random() < 0.1 else None, grammar=None, lora=None, encoder=None)
        if r.random() < 0.15:
            q["grammar"] = (r.randint(1, 12), r.random() < 0.2)
        if cfg["max_loras"] and r.random() < 0.5:
            q["lora"] = r.randint(1, 3)
        if cfg["encoder_cache"] and n > 8 and r.random() < 0.4:
            length = r.randint(4, min(cfg["encoder_cache"], cfg["budget"], n - 1))
            q["encoder"] = (length, r.randint(0, n - length))
        reqs.append(q)
    if r.random() < 0.25:
        reqs.append(dict(id="session", session=True, n=r.randint(4, max(5, int(pool * 0.3))), next=r.randint(4, max(5, int(pool * 0.3))),
                         prio=r.randint(0, 2), arrive=r.randint(0, 25), max_tokens=r.randint(1, 3), gap=r.randint(0, 12), token=0, matched=0,
                         delay=1, fail=False, none=0, abort=None))
    return cfg, reqs


BOUND = os.environ.get("SEARCH_BOUND")
violations = []


def watch(s, cfg):
    """SEARCH_BOUND=1: count the steps a running prefill with no encoder input gets nothing though budget is left and no
    higher-priority request is running with first tokens due, so only requests not admitted can be holding it back."""
    real = s.schedule
    counts = {}

    def schedule(throttle=False):
        out = real(throttle)
        left = s.max_num_scheduled_tokens - sum(out.num_scheduled_tokens.values())
        for q in list(s.running):
            if not q.is_prefill_chunk or q.request_id in out.num_scheduled_tokens or q.mm_features or left <= 0 or s.pause_state != PauseState.UNPAUSED:
                counts.pop(q, None)
                continue
            if any(r.priority < q.priority and r.num_output_tokens + r.num_output_placeholders < cfg["window"] for r in s.running):
                counts.pop(q, None)
                continue
            counts[q] = counts.get(q, 0) + 1
            above = sum(1 for r in s.requests.values() if r.priority < q.priority and not r.is_finished())
            if counts[q] > 2 * cfg["window"] * max(above, 1) + 3:
                violations.append(dict(step=s.current_step, req=q.request_id, prio=q.priority, count=counts[q], above=above))
        return out
    s.schedule = schedule


def make(cfg, on):
    if BOUND:
        cfg = dict(cfg, interval=1, cadence=1, encoder_cache=0)
    options = dict(
        max_num_batched_tokens=cfg["budget"], num_blocks=cfg["num_blocks"], max_num_seqs=cfg["seqs"], long_prefill_token_threshold=cfg["threshold"],
        watermark=cfg["watermark"], enable_prefix_caching=cfg["prefix"], max_num_inflight_kv_loads=cfg["inflight"],
        max_model_len=(cfg["num_blocks"] + 8) * BLOCK, use_kv_connector=t.mock_kv(matched_tokens=0, is_async=False),
        max_priority_preemptions_per_step=cfg["kv"] if on else 0, priority_prefill_budget_cap=on and cfg["cap"],
        priority_prefill_budget_cap_tokens=cfg["window"], disable_chunked_mm_input=cfg["disable_chunked"], prefill_schedule_interval=cfg["cadence"])
    if cfg["max_loras"]:
        options["lora_config"] = LoRAConfig(max_loras=cfg["max_loras"])
    if cfg["window_spec"]:
        options["kv_cache_spec"] = SlidingWindowSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32, sliding_window=4 * BLOCK)
        options["enable_prefix_caching"] = False
    if cfg["mode"] == "defer":
        s = t._create_deferring_scheduler(**options)
    elif cfg["mode"] == "async":
        s = t._create_scheduler(async_scheduling=True, **options)
    else:
        s = t._create_scheduler(**options)
    if cfg["encoder_cache"]:
        s.max_num_encoder_input_tokens = cfg["encoder_cache"]
        s.encoder_cache_manager = EncoderCacheManager(cache_size=cfg["encoder_cache"])
    s.recompute_kv_load_failures = cfg["recompute"]
    s.scheduler_reserve_full_isl = cfg["reserve_full"]
    s.structured_output_manager.accept_tokens = lambda request, tokens: True
    if BOUND and on:
        watch(s, cfg)
    return s


def run(cfg, reqs, on, flip=False):
    s = make(cfg, on)
    by_id = {r["id"]: r for r in reqs}
    nones = {r["id"]: r["none"] for r in reqs}

    def matched(request, num_computed):
        q = by_id[request.request_id]
        if nones[q["id"]] > 0:
            nones[q["id"]] -= 1
            return None, False
        return (q["matched"] - num_computed, True) if q["matched"] > num_computed else (0, False)
    s.connector.may_load_async = lambda request: bool(by_id[request.request_id]["matched"])
    s.connector.get_num_new_matched_tokens = matched
    objs, loading, reported, failed_once = {}, {}, {}, set()
    grammar_at = {}
    session = None
    in_flight = collections.deque()
    idle = longest = 0
    pending = sorted(reqs, key=lambda r: r["arrive"])
    stalled = {}
    for step in range(LIMIT):
        for a, b, kind in cfg["pauses"]:
            if step == a:
                s.set_pause_state(PauseState.PAUSED_NEW if kind == "new" else PauseState.PAUSED_ALL)
            if step == b:
                s.set_pause_state(PauseState.UNPAUSED)
        paused = s.pause_state != PauseState.UNPAUSED
        while pending and pending[0]["arrive"] <= step:
            r = pending.pop(0)
            if r.get("session"):
                session = dict(r=r, stage=0, since=None)
                objs[r["id"]] = t._session_request(r["n"], r["max_tokens"], r["prio"])
            else:
                encoder = dict(num_encoder_tokens=r["encoder"][0], encoder_offset=r["encoder"][1]) if r["encoder"] else {}
                q = t._request(r["id"], r["n"], r["prio"], arrival_time=r["arrive"], max_tokens=r["max_tokens"], prompt_token=r["token"], **encoder)
                if r["lora"]:
                    q.lora_request = LoRARequest(f"lora-{r['lora']}", r["lora"], f"/path/{r['lora']}")
                if r["grammar"]:
                    q.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
                    q.structured_output_request = types.SimpleNamespace(grammar=None)
                    grammar_at[r["id"]] = step + r["grammar"][0]
                objs[r["id"]] = q
            s.add_request(objs[r["id"]])
        for rid, at in list(grammar_at.items()):
            if step >= at:
                objs[rid].structured_output_request.grammar = ValueError("bad") if by_id[rid]["grammar"][1] else object()
                del grammar_at[rid]
        for rid, q in objs.items():
            if by_id[rid].get("abort") is not None and step >= by_id[rid]["abort"] and not q.is_finished():
                s.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
        if session and session["stage"] < 2 and objs["session"].status.name == "WAITING_FOR_STREAMING_REQ":
            session["since"] = step if session["since"] is None else session["since"]
            if session["stage"] == 1:
                s.finish_requests("session", RequestStatus.FINISHED_ABORTED)
                session["stage"] = 2
            elif step - session["since"] >= session["r"]["gap"]:
                s.add_request(t._session_request(session["r"]["next"], session["r"]["max_tokens"], session["r"]["prio"]))
                session.update(stage=1, since=None)
        out = s.schedule(cfg["interval"] > 1 and step % cfg["interval"] != 0)
        done = set()
        for rid, q in objs.items():
            if q.status.name != "WAITING_FOR_REMOTE_KVS" or (rid in reported and step - reported[rid] >= 2 and rid not in s.finished_recving_kv_req_ids):
                reported.pop(rid, None)
            if q.status.name == "WAITING_FOR_REMOTE_KVS" and rid not in loading and rid not in reported:
                loading[rid] = step + by_id[rid]["delay"]
        for rid, due in list(loading.items()):
            if step >= due:
                done.add(rid)
                reported[rid] = step
                del loading[rid]
        output = t._model_runner_output(s, out, done or None)
        failed = {rid for rid in done if by_id[rid].get("fail") and rid not in failed_once}
        if failed:
            failed_once |= failed
            output.kv_connector_output.failed_recving = failed
        if cfg["mode"] == "sync":
            s.update_from_output(out, output)
        else:
            in_flight.append((out, output))
            if len(in_flight) > 1:
                s.update_from_output(*in_flight.popleft())
        busy = (out.num_scheduled_tokens or loading or paused or grammar_at or any(o[0].num_scheduled_tokens for o in in_flight)
                or (session and session["stage"] == 0 and session["since"] is not None))
        if busy or pending:
            idle = 0
        else:
            idle += 1
            longest = max(longest, idle)
        if not pending and len(objs) == len(reqs) and all(q.is_finished() for q in objs.values()):
            return dict(finished=True, steps=step + 1, longest_idle=longest, **stalled)
        if stalled.get("stall_step") == step - 1:
            stalled["resumed"] = bool(out.num_scheduled_tokens)
        if idle > IDLE:
            state = {k: f"{q.status.name}({q.num_computed_tokens},{q.num_output_tokens})" for k, q in objs.items() if not q.is_finished()}
            if not flip:
                return dict(finished=False, steps=step + 1, longest_idle=longest, state=state, free_now=s.kv_cache_manager.block_pool.get_num_free_blocks(), **stalled)
            stalled = dict(stall_step=step, stall_state=state, free=s.kv_cache_manager.block_pool.get_num_free_blocks())
            s.max_priority_preemptions_per_step = 0
            s.priority_prefill_budget_cap = False
            flip, idle = False, 0
    state = {k: f"{q.status.name}({q.num_computed_tokens},{q.num_output_tokens})" for k, q in objs.items() if not q.is_finished()}
    return dict(finished=False, steps=LIMIT, longest_idle=longest, state=state, limit=True, **stalled)


def main():
    first, last = int(sys.argv[1]), int(sys.argv[2])
    totals = collections.Counter()
    for seed in range(first, last):
        cfg, reqs = config(seed)
        if BOUND:
            cfg.update(interval=1, cadence=1, encoder_cache=0)
            for q in reqs:
                q["encoder"] = None
        violations.clear()
        row = dict(seed=seed)
        try:
            row["off"] = run(cfg, reqs, False)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            totals["off_error"] += 1
            print(json.dumps(dict(seed=seed, side="off", error=f"{type(e).__name__}: {e}"[:200], at=f"{tb.filename.rsplit('/', 1)[-1]}:{tb.lineno} {tb.line}"[:160])), flush=True)
            continue
        try:
            row["on"] = run(cfg, reqs, True)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            totals["on_error"] += 1
            print(json.dumps(dict(seed=seed, side="on", cfg=cfg, reqs=reqs, error=f"{type(e).__name__}: {e}"[:200], at=f"{tb.filename.rsplit('/', 1)[-1]}:{tb.lineno} {tb.line}"[:160])), flush=True)
            continue
        if violations:
            totals["bound_violation"] += 1
            print(json.dumps(dict(seed=seed, kind="bound", first=violations[0], num=len(violations), cfg=cfg, reqs=reqs)), flush=True)
        totals["off_finished" if row["off"]["finished"] else "off_stuck"] += 1
        totals["on_finished" if row["on"]["finished"] else "on_stuck"] += 1
        if row["off"]["finished"] and not row["on"]["finished"]:
            row["flip"] = run(cfg, reqs, True, flip=True)
            row["caused"] = row["flip"]["finished"] or row["flip"].get("resumed", False)
            totals["caused" if row["caused"] else "stuck_after_flip"] += 1
            row.update(cfg=cfg, reqs=reqs)
            print(json.dumps(row), flush=True)
        elif row["on"]["finished"] and row["on"]["longest_idle"] > row["off"].get("longest_idle", 0) + 2 * cfg["window"] + 4:
            totals["long_idle"] += 1
            print(json.dumps(dict(seed=seed, kind="long_idle", on_idle=row["on"]["longest_idle"], off_idle=row["off"]["longest_idle"], window=cfg["window"], cfg=cfg, reqs=reqs)), flush=True)
    print(json.dumps(dict(totals=totals)), flush=True)


if __name__ == "__main__":
    main()
