"""The FP8 runs as one Markdown table, a row a run, from what each run wrote: result.txt (the build, the flags, run.sh's
verdict), summary.json and requests.jsonl (load.py's), and server.log (the attention backend vLLM chose, and the plan
counters of patch_plan_wait.py).

A run from before load.py could detect a stall (its summary has no stalled_at) is held to load.py's rule after the
fact: requests failed and none was answered in its last 180 s means it stalled, whatever run.sh's verdict said. A
survivor that ended in more than 30 s without an answer, too short a silence for the rule, says how long.

usage: python summarize.py RUNS_DIR
"""

import json
import re
import sys
from pathlib import Path

STALL_S = 180.0  # load.py's --stall default
QUIET_S = 30.0  # a survivor's silence before its end worth reporting; the clean runs' last answer is within 5 s
BACKEND = re.compile(r"Using (?:AttentionBackendEnum\.)?(\w+) (?:attention )?backend")
COUNT_LINE = re.compile(r"fp8inv: (\d+) plans, (\d+) found the previous copy pending")
EVENT_LINE = re.compile(
    r"fp8inv: plan \d+ found the previous plan copy pending: (\d+) of (\d+) plans"
)


def fields(path):
    """result.txt's `key: value` lines (the first of each key)."""
    out = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(": ")
        if sep and key not in out:
            out[key] = value
    return out


def build(result):
    venv, env, flags = result["venv"], result.get("env", ""), result["flags"]
    # The patched environments (venv-fix, venv-rc3-fix) are the stock ones with patch_plan_wait.py applied.
    name = "+ #64165" if Path(venv).name.endswith("-fix") else "stock"
    if "FP8INV_PLAN_WAIT=count" in env:
        name += ", counting only"
    extras = [
        label
        for flag, label in (
            ("--no-async-scheduling", "async scheduling off"),
            ("VLLM_HAS_FLASHINFER_CUBIN=1", "XQA decode"),
            ("CUDA_LAUNCH_BLOCKING=1", "blocking launches"),
            ("CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1", "core dump"),
        )
        if flag in flags or flag in env
    ]
    return ", ".join([name, *extras])


def version(result):
    """The vLLM a run served: a release by its name (0.33.0rc3, not its build's dev suffix), main by its commit."""
    full = result["versions"].split()[0]
    if Path(result["venv"]).name == "venv-main":
        return f"main ({full.rsplit('+g', 1)[-1]})"
    return full.split(".dev")[0]


def plans(server_log):
    """The last plan count the server logged: (pending, plans), or None for a stock build."""
    last = None
    for line in server_log.read_text(errors="replace").splitlines():
        if m := COUNT_LINE.search(line):
            last = (int(m.group(2)), int(m.group(1)))
        elif m := EVENT_LINE.search(line):
            last = (int(m.group(1)), int(m.group(2)))
    return last


def row(run):
    result = fields(run / "result.txt")
    summary = json.loads((run / "summary.json").read_text())
    requests = [
        json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines() if line
    ]
    answered = [r for r in requests if r["status"] == 200]
    failed = len(requests) - len(answered)
    last = max((r["t"] for r in answered), default=0.0)
    log = (run / "server.log").read_text(errors="replace")
    backends = BACKEND.findall(log)
    verdict = result.get("verdict", "?")
    if "stalled_at" not in summary and failed and summary["seconds"] - last > STALL_S:
        verdict = f"STALLED (by its requests; run.sh said {verdict})"
    elif verdict == "SURVIVED" and summary["seconds"] - last > QUIET_S:
        # A run shorter than the stall rule can end in silence the rule never saw: say so, without calling it a stall.
        verdict += f" (no answer in its last {summary['seconds'] - last:.0f} s)"
    end = summary.get("server_died_at") or summary.get("stalled_at")
    if end:
        verdict += f" at {end:.0f} s"
    counted = plans(run / "server.log")
    return [
        run.name,
        version(result),
        build(result),
        "FP8" if "--kv-cache-dtype fp8" in result["flags"] else "BF16",
        backends[-1] if backends else "?",
        "yes" if result["config"] == "offload.yaml" else "no",
        verdict,
        f"{summary['args']['duration']:.0f}",
        f"{len(answered)} / {failed}",
        f"{last:.0f}",
        f"{counted[0]:,} of {counted[1]:,}" if counted else "",
    ]


def main():
    runs = sorted(
        (
            p
            for p in Path(sys.argv[1]).iterdir()
            if (p / "result.txt").is_file() and (p / "summary.json").is_file()
        ),
        key=lambda p: fields(p / "result.txt")["started"],
    )
    head = [
        "Run",
        "vLLM",
        "Build",
        "KV cache",
        "Attention",
        "CPU tier",
        "Verdict",
        "Load planned, s",
        "Answered / failed",
        "Last answer, s",
        "Plans finding the last copy pending",
    ]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for run in runs:
        print("| " + " | ".join(row(run)) + " |")


if __name__ == "__main__":
    main()
