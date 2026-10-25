"""Write build-loadhold.py: Lane A's final build.py (e277e85) with two experimental changes, for a GPU A/B only.
1. A KV load in flight holds lower-priority prefill again, as e19fba5's rule did, but for at most budget_cap_tokens
   steps from the step make-room first sees it loading, so a stuck load holds nothing after that.
2. The generated module keeps vLLM's scheduler logger: the appended async_scheduler.py rebinds `logger` to the module's
   own name, which vLLM's logging config does not cover, and make-room's INFO counter lines are lost.
Usage: make-loadhold.py FINAL_BUILD_PY OUT_PY"""

import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
s = src.read_text()


def sub(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub(
    "        self._make_room_untried: set[Request] = set()\n",
    "        self._make_room_untried: set[Request] = set()\n"
    "        # EXPERIMENT: the step in which make-room first saw each load in flight.\n"
    "        self._make_room_load_start: dict[str, int] = {}\n",
)
sub(
    """        if self._make_room_slot_free():
            priorities.extend(
                self.requests[req_id].priority
                for req_id in self.finished_recving_kv_req_ids
                if req_id in self.requests
            )
        return min(priorities, default=None)
""",
    """        if self._make_room_slot_free():
            priorities.extend(
                self.requests[req_id].priority
                for req_id in self.finished_recving_kv_req_ids
                if req_id in self.requests
            )
            # EXPERIMENT: a load in flight holds too, for at most
            # budget_cap_tokens steps.
            loading: dict[str, int] = {}
            for queue in (self.kv_holding_waiting, self.waiting):
                for req in queue:
                    if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        start = self._make_room_load_start.get(
                            req.request_id, self.current_step
                        )
                        loading[req.request_id] = start
                        if self.current_step - start < self.prefill_budget_cap_tokens:
                            priorities.append(req.priority)
            self._make_room_load_start = loading
        return min(priorities, default=None)
""",
)
sub(
    'ASYNC_EDITS = [("from vllm.v1.core.sched.scheduler import Scheduler\\n", "")]\n',
    'ASYNC_EDITS = [\n    ("from vllm.v1.core.sched.scheduler import Scheduler\\n", ""),\n'
    '    ("logger = init_logger(__name__)\\n", ""),\n]\n',
)
out.write_text(s)
compile(s, str(out), "exec")
print("wrote", out)
