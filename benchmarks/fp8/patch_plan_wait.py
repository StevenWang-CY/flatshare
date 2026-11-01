"""Apply vLLM #64165 (wait for the previous FlashInfer plan's page-locked copy before planning again), with counts, to
a scratch venv's vllm/v1/attention/backends/flashinfer.py (vLLM 0.32.0).

FP8INV_PLAN_WAIT=block (default) waits as #64165 does; FP8INV_PLAN_WAIT=count only counts, the negative control. Both
log the plans that found the previous plan's copy still pending. The file is written anew and renamed over the old one,
so a hard link it shares with uv's cache or another venv keeps the stock file.

usage: python patch_plan_wait.py VENV
"""

import os
import sys
from pathlib import Path

venv = Path(sys.argv[1])
target = next(venv.glob("lib/python3*/site-packages/vllm/v1/attention/backends/flashinfer.py"))
source = target.read_text()

EDITS = [
    (
        "import numpy as np\nimport torch\n",
        "import os\nimport time\n\nimport numpy as np\nimport torch\n",
    ),
    (
        "        self._decode_wrapper = None  # Wrapper for decode (general shape)\n",
        "        self._decode_wrapper = None  # Wrapper for decode (general shape)\n"
        "        # FP8 investigation, vLLM #64165 with counts: plan() fills the wrapper's page-locked buffer and copies\n"
        "        # it to the GPU asynchronously, so it must not run again before that copy has. FP8INV_PLAN_WAIT=count\n"
        "        # only counts the plans that find it pending.\n"
        "        self._plan_copied = torch.cuda.Event(blocking=True)\n"
        "        self._plan_wait = os.environ.get('FP8INV_PLAN_WAIT', 'block')\n"
        "        self._plans = self._plans_pending = self._plan_waits_blocked = 0\n",
    ),
    (
        "                num_reqs,\n"
        "                page_size,\n"
        "            )\n"
        "        else:\n"
        "            paged_kv_indices = None\n",
        "                num_reqs,\n"
        "                page_size,\n"
        "            )\n"
        "            self._plans += 1\n"
        "            if not self._plan_copied.query():\n"
        "                self._plans_pending += 1\n"
        "                if self._plan_wait == 'block':\n"
        "                    start = time.perf_counter()\n"
        "                    self._plan_copied.synchronize()\n"
        "                    if time.perf_counter() - start > 50e-6:\n"
        "                        self._plan_waits_blocked += 1\n"
        "                if self._plans_pending <= 20 or self._plans_pending % 100 == 0:\n"
        "                    logger.info(\n"
        "                        'fp8inv: plan %d found the previous plan copy pending: %d of %d plans, '\n"
        "                        '%d waits blocked over 50 us, mode %s',\n"
        "                        self._plans, self._plans_pending, self._plans,\n"
        "                        self._plan_waits_blocked, self._plan_wait,\n"
        "                    )\n"
        "            if self._plans % 5000 == 0:\n"
        "                logger.info(\n"
        "                    'fp8inv: %d plans, %d found the previous copy pending, %d waits blocked, mode %s',\n"
        "                    self._plans, self._plans_pending, self._plan_waits_blocked, self._plan_wait,\n"
        "                )\n"
        "        else:\n"
        "            paged_kv_indices = None\n",
    ),
    (
        "                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)\n"
        "        return attn_metadata\n",
        "                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)\n"
        "        if needs_paged_kv_indices:\n"
        "            self._plan_copied.record()\n"
        "        return attn_metadata\n",
    ),
]

for old, new in EDITS:
    count = source.count(old)
    if count != 1:
        sys.exit(f"anchor found {count} times, not once: {old[:70]!r}")
    source = source.replace(old, new)

before = os.stat(target).st_ino
fresh = target.with_name(target.name + ".fp8inv")
fresh.write_text(source)
os.replace(fresh, target)
after = os.stat(target).st_ino
print(f"patched {target}: inode {before} -> {after}")
if before == after:
    sys.exit("the file kept its inode")
