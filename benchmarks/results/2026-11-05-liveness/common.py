"""Shared helpers for the deterministic reproducers: real Scheduler from the PR's test module, a stepping loop that reports
when each request finishes, and the longest run of steps that schedule nothing. Run from a vLLM tree (PYTHONPATH=$PWD)."""
import importlib
import logging

logging.disable(logging.WARNING)
t = importlib.import_module("tests.v1.core.test_priority_make_room")
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

BLOCK = 16
# #65282's tree (r5a) has no budget cap: its scripts skip those lines there.
import inspect  # noqa: E402
from tests.v1.core.utils import create_scheduler as _cs  # noqa: E402
HAS_CAP = "priority_prefill_budget_cap" in inspect.signature(_cs).parameters


class MP:
    def setattr(self, obj, name, value):
        setattr(obj, name, value)


def req(request_id, num_tokens, priority, arrival_time, max_tokens=64, prompt_token=0, num_encoder_tokens=0, encoder_offset=0):
    """As the r5b test file's _request; #65282's has no encoder offset."""
    from tests.v1.core.test_scheduler import create_requests_with_priority
    from vllm.multimodal.inputs import PlaceholderRange
    encoder_inputs = {}
    if num_encoder_tokens:
        encoder_inputs = dict(mm_hashes_list=[[f"encoder-{request_id}"]],
                              mm_positions=[[PlaceholderRange(offset=encoder_offset, length=num_encoder_tokens)]])
    return create_requests_with_priority(num_requests=1, priorities=[priority], arrival_times=[arrival_time], num_tokens=num_tokens,
                                         max_tokens=max_tokens, starting_idx=prompt_token, req_ids=[request_id], **encoder_inputs)[0]


def match_tokens(s, matched):
    """The KV connector loads matched[request_id] tokens asynchronously, none of the others'."""
    s.connector.get_num_new_matched_tokens = lambda request, num_computed: (
        (matched[request.request_id], True) if request.request_id in matched else (0, False))


def drive(s, reqs, steps=400, on_step=None, done_for=None, arrivals=None, quiet=True):
    """Steps the scheduler synchronously. `reqs`: already-added requests. `arrivals`: {step: [request, ...]}.
    `done_for(step, out) -> set|None`: async loads reported done in this step's output."""
    finished, trace = {}, []
    idle = longest = 0
    objs = list(reqs)
    arrivals = dict(arrivals or {})
    step = -1
    for step in range(steps):
        for r in arrivals.pop(step, ()):
            s.add_request(r)
            objs.append(r)
        if on_step:
            on_step(step)
        out = s.schedule()
        done = done_for(step, out) if done_for else None
        s.update_from_output(out, t._model_runner_output(s, out, done))
        trace.append(dict(out.num_scheduled_tokens))
        for r in objs:
            if r.request_id not in finished and r.is_finished():
                finished[r.request_id] = step
        idle = 0 if out.num_scheduled_tokens else idle + 1
        longest = max(longest, idle)
        if not arrivals and len(finished) == len(objs):
            break
    state = {r.request_id: f"{r.status.name}(computed={r.num_computed_tokens},out={r.num_output_tokens})" for r in objs if not r.is_finished()}
    return dict(finished=finished, steps=step + 1, longest_idle=longest, unfinished=state, trace=trace)


def show(label, r, trace_tail=0):
    print(f"{label}: steps={r['steps']} finished={r['finished']} longest_idle={r['longest_idle']} unfinished={r['unfinished']}")
    if trace_tail:
        print("    last schedules:", r["trace"][-trace_tail:])
