"""The round-3 probes' rows from their logs: for each block (stock: vLLM 0.32.0, the frozen box; a: the KV preemption
alone; ab: both pull requests at the budget cap's 16-token window) and case (kv, prefill), the four runs' priority-0 TTFT and 16th token, and their medians
and ranges, as the PR descriptions give them. Checks that each block's engine came up with its switches, that each log
has four runs, and that no block logged an error. Exits 1 if a check fails.

    python3 check.py [DIR]
"""

import re
import statistics
import sys
from pathlib import Path

here = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
run = (here / "run.log").read_text()
LINE = re.compile(r"^(kv|prefill): priority-0 TTFT (\d+) ms, 16 tokens (\d+|—)", re.M)
fails = 0
for block, switches in (("stock", None), ("a", "'budget_cap': False, 'reserve_tokens': 0"), ("ab", "'reserve_tokens': 0")):
    ready = re.search(rf"^\S+ {block}: engine ready; scheduler_cls (\S+) additional_config (.*?); GPU", run, re.M)
    if switches is None:  # the frozen box: stock vLLM's scheduler, no make_room table
        if not ready or ready.group(1) != "None" or ready.group(2) != "{}":
            print(f"FAIL {block}: not the frozen box")
            fails += 1
    elif not ready or "make_room" not in ready.group(2) or switches not in ready.group(2):
        print(f"FAIL {block}: engine not up with {switches}")
        fails += 1
    done = re.search(rf"^\S+ {block}: done; engine make-room lines: (\d+), errors: (\d+)", run, re.M)
    if not done or done.group(2) != "0":
        print(f"FAIL {block}: no done line, or errors")
        fails += 1
    for case in ("kv", "prefill"):
        runs = LINE.findall((here / f"{block}-{case}.log").read_text())
        if len(runs) != 4 or any(c != case for c, _, _ in runs):
            print(f"FAIL {block} {case}: {len(runs)} runs")
            fails += 1
            continue
        ttft = [int(t) for _, t, _ in runs]
        tok16 = [int(t) for _, _, t in runs if t != "—"]
        print(f"{block} {case}: TTFT {ttft} median {round(statistics.median(ttft))} ({min(ttft)}–{max(ttft)}); "
              f"16th token {tok16} median {round(statistics.median(tok16))} ({min(tok16)}–{max(tok16)})"
              + (f"; {4 - len(tok16)} replies ended before 16 tokens" if len(tok16) < 4 else ""))
print("ok" if not fails else f"{fails} checks failed")
sys.exit(1 if fails else 0)
