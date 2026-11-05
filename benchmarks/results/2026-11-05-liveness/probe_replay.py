"""The two GPU probes' cases (make-room/probe.py) on the real scheduler, on CPU, step by step, on two make-room modules: whether
the revision can change what the GPU tables measured. The frozen box's pool of 2,275 blocks of 16 tokens, its step cap of 512
tokens and long-prefill threshold of 256, asynchronous scheduling with one step in flight, two preemptions a step and a
16-token window, the turn's prompt 750 tokens: as new, and with its prefix cached in the CPU tier (a KV load of 2, 8 or 32
steps). Prints whether each case schedules the same on both modules. A module with no switches is the control: it differs.

  PYTHONPATH=<vLLM 0.32.0> python probe_replay.py FLATSHARE_CHECKOUT MODEL_DIR MODULE_DIR_A MODULE_DIR_B
"""
import importlib.util
import pathlib
import sys

flatshare, model_dir, dir_a, dir_b = sys.argv[1:5]
tests = pathlib.Path(flatshare) / "python/flatshare_vllm/tests"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tm = load("flatshare_test_make_room", tests / "test_make_room.py")
tm.init_none_hash(tm.sha256)
a = load("module_a", pathlib.Path(dir_a) / "flatshare_make_room_scheduler.py")
b = load("module_b", pathlib.Path(dir_b) / "flatshare_make_room_scheduler.py")
BLOCK, POOL = 16, 2275
SWITCHES = {"max_preemptions_per_step": 2, "budget_cap": True, "budget_cap_tokens": 16, "reserve_tokens": 0}


def run(module, make_room, case, load_steps):
    matched = {"voice": 720} if load_steps else {}
    connector = tm.Connector(matched)
    sched = tm.scheduler(module.AsyncScheduler, model_dir, POOL, make_room, max_num_batched_tokens=512,
                         long_prefill_token_threshold=256, max_model_len=32768, max_num_seqs=256, connector=connector)
    engine = tm.Pipelined(sched)
    if case == "kv":  # two of 17,999 tokens, 24 blocks free, decoding
        requests = [tm.request(f"bg{i}", 17999, priority=2, arrival=i, max_tokens=4096, token=10 + i) for i in range(2)]
    else:  # three of 10,000, prefilling
        requests = [tm.request(f"bg{i}", 10000, priority=2, arrival=i, max_tokens=4096, token=10 + i) for i in range(3)]
    for r in requests:
        sched.add_request(r)
    trace = []
    if case == "kv":
        while not all(r.status == tm.RequestStatus.RUNNING and not r.is_prefill_chunk for r in requests):
            trace.append(dict(engine.step().num_scheduled_tokens))
            assert len(trace) < 400
    else:
        while not all(r.status == tm.RequestStatus.RUNNING for r in requests):
            trace.append(dict(engine.step().num_scheduled_tokens))
            assert len(trace) < 400
    free_at_arrival = tm.free_blocks(sched)
    voice = tm.request("voice", 750, priority=0, arrival=9, max_tokens=64, token=1)
    sched.add_request(voice)
    started = reported = None
    for i in range(160):
        out = engine.schedule()
        trace.append((dict(out.num_scheduled_tokens), sorted(out.preempted_req_ids or ()), tm.free_blocks(sched)))
        if started is None and voice.status == tm.RequestStatus.WAITING_FOR_REMOTE_KVS:
            started = i
        if load_steps and started is not None and not reported and i >= started + load_steps:
            engine.in_flight[-1][1].kv_connector_output = tm.KVConnectorOutput(finished_recving={"voice"})
            reported = True
        if len(engine.in_flight) > 1:
            engine.process()
    return trace, voice, started, reported, free_at_arrival


fails = 0
for case in ("kv", "prefill"):
    for load_steps in (0, 2, 8, 32):
        ta, *_ = run(a, SWITCHES, case, load_steps)
        tb, vb, started, reported, free = run(b, SWITCHES, case, load_steps)
        tc, *_ = run(b, {}, case, load_steps)
        same = ta == tb
        first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        control = ta != tc
        loaded = bool(started is not None and reported) == bool(load_steps)
        print(f"{case} load {load_steps}: same on both modules: {same}; steps {len(ta)}; {free} blocks free at the turn's arrival; load from step {started}, completed: {bool(reported)}; "
              f"voice computed {vb.num_computed_tokens}, output {vb.num_output_tokens}; "
              f"control (no switches) differs: {control}" + ("" if same else f"; first difference at {first}: {ta[first]} vs {tb[first]}"))
        fails += (not same) + (not control) + (not loaded)
print("ok" if not fails else f"{fails} checks failed")
sys.exit(1 if fails else 0)
