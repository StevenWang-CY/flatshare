"""The engine points' tables (README.md), from this record's files alone: each point's tail.txt (tail.py's lines),
stages.json (what the program's analysis derived from the point's scheduler trace, which the record does not keep),
spec.json (speculative decoding with the calls alone) and agree.json (the CPU tier's restore).

An arm is a point's name less its node letter and its replicate number. For each node and arm: the calls' first
token as tail.py measured it, p50 and p95 (the median over the replicates, with their range); their 16th token, about
a first sentence (p50 and p95); the calls' later tokens (ms a token from the 16th to the 48th, p50 over every turn); the
agents' computed tokens a second (the median over the replicates); and in the stages table the engine's first token
by stage (means over every turn of the replicates). `python3 table.py NAME` prints one table, NAME one of TABLES.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NODES = ("phastform", "gray")
STAGES = ("input", "load", "queue", "prep", "gpu", "own")
CALLS = re.compile(r"TTFT p50 (\d+) p90 (\d+) p95 (\d+) max (\d+)")
SENTENCE = re.compile(r"16 tokens p50 (\d+) p90 (\d+) p95 (\d+)")
REPLY = re.compile(r"48 tokens p50 (\d+) p90 (\d+) p95 (\d+)")
ARMS = {
    # What was adopted, in the order the serving profile gained it.
    "a0-grow": "v1.14's serving profile (asynchronous scheduling)",
    "r0-grow": "+ the calls' prefixes kept on the GPU (retention)",
    "r0j-grow": "+ the next batch just in time (30 ms margin)",
    "r0h-grow": "+ each turn announced (250 ms ahead, held 400 ms)",
    "r0hj-grow": "+ the turn announced and just in time",
    "r0hjt-grow": "+ the calls taking the agents' reservations",
    "r0hjtd-grow": "+ the agents' decodes held from a call's first 16 tokens",
    # How long the decode hold lasts.
    "r0hjtd32-grow": "the same, the decodes held from a call's first 32 tokens",
    "r0hjtd48-grow": "the same, the decodes held from a call's first 48 tokens",
    # The alternatives.
    "a1-grow": "synchronous scheduling",
    "r1-grow": "synchronous scheduling, retention",
    "t0-grow": "Triton attention, full CUDA graphs",
    "t1-grow": "Triton, full graphs, synchronous",
    "t0rj-grow": "Triton, full graphs, retention, just in time (6 ms)",
    "tf0-grow": "Triton, full graphs, FP8 KV",
    "f0-grow": "FlashInfer, FP8 KV (#64165's wait)",
    "r0a2-grow": "retention, two API servers",
    "s3r-grow": "retention, EAGLE-3 with 3 drafts",
    "s2r-grow": "retention, EAGLE-3 with 2 drafts",
    "s3rr-grow": "retention, EAGLE-3 with 3 drafts, short loads recomputed",
    "s2rr-grow": "retention, EAGLE-3 with 2 drafts, short loads recomputed",
    "s2rrhj-grow": "the turn announced and just in time, EAGLE-3 with 2 drafts, short loads recomputed",
    "s2rrhjt-grow": "the turn announced, just in time, the reservations taken, EAGLE-3 with 2 drafts, short loads recomputed",
    "s2rrhjtd-grow": "the same, the agents' decodes held from a call's first 16 tokens",
    # The controls.
    "r0hw-grow": "retention, the turns announced but nothing held",
    "r0hn-grow": "retention, the hint scheduler with no turn announced",
    "r0rr-grow": "retention, short loads recomputed, no speculation",
    "r0t-grow": "retention, the agents' reservations taken, no turn announced",
    # Fresh agent prompts.
    "a0-fresh": "v1.14's serving profile",
    "r0-fresh": "retention",
    "r0j-fresh": "retention, just in time",
    "r0h-fresh": "retention, each turn announced",
    "a1-fresh": "synchronous scheduling",
    "r1-fresh": "synchronous scheduling, retention",
    "s3r-fresh": "retention, EAGLE-3 with 3 drafts",
}
TABLES = {
    "adopted": ("a0-grow", "r0-grow", "r0j-grow", "r0h-grow", "r0hj-grow", "r0hjt-grow", "r0hjtd-grow", "r0hjtd48-grow"),
    "alternatives": (
        "a1-grow", "r1-grow", "t0-grow", "t1-grow", "t0rj-grow", "tf0-grow", "f0-grow", "r0a2-grow",
        "s3r-grow", "s2r-grow", "s3rr-grow", "s2rr-grow", "s2rrhj-grow", "s2rrhjt-grow", "s2rrhjtd-grow",
    ),
    "hold": ("r0hjt-grow", "r0hjtd-grow", "r0hjtd32-grow", "r0hjtd48-grow"),
    "controls": ("r0hw-grow", "r0hn-grow", "r0rr-grow", "r0t-grow"),
    "fresh": ("a0-fresh", "r0-fresh", "r0j-fresh", "r0h-fresh", "a1-fresh", "r1-fresh", "s3r-fresh"),
    "stages": ("a0-grow", "r0-grow", "r0j-grow", "r0h-grow", "a1-grow", "t0-grow", "f0-grow", "s2r-grow", "s2rr-grow"),
    "speculation": (),
    "restore": (),
}  # fmt: skip


def points() -> dict[tuple[str, str], list[Path]]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for node in NODES:
        for run in sorted((HERE / node).glob("*")):
            if (run / "tail.txt").exists() and (run / "stages.json").exists():
                groups[(node, re.sub(r"-\d+$", "", run.name)[2:])].append(run)
    return groups


def client(run: Path) -> tuple[float, float, float, float]:
    """The calls' first token p50 and p95 and their 16th token p50 and p95, as tail.py printed them."""
    text = (run / "tail.txt").read_text()
    calls = CALLS.search(text.split("\ncalls: ", 1)[1])
    sentence = SENTENCE.search(text.split("\nreplies: ", 1)[1])
    return float(calls[1]), float(calls[3]), float(sentence[1]), float(sentence[3])


def later_tokens(run: Path) -> list[float]:
    """Each call turn's ms a token from its 16th to its 48th token."""
    out = []
    for line in (run / "tail.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) > 12 and parts[1] == "call" and parts[3] == "ttft" and parts[10] == "48":
            out.append((float(parts[12]) - float(parts[8])) / 32)
    return out


def spread(values: list[float]) -> str:
    mid = statistics.median(values)
    return f"{mid:.0f}" if len(values) == 1 else f"{mid:.0f} ({min(values):.0f}-{max(values):.0f})"


def outcomes(name: str) -> str:
    rows = [
        "| Node | Arm | Runs | First token p50, ms | p95 | 16th token p50 | p95 | Later tokens, ms each | Agents' tokens/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = points()
    for node in NODES:
        for arm in TABLES[name]:
            runs = groups.get((node, arm))
            if not runs:
                continue
            seen = [client(run) for run in runs]
            agents = [json.loads((run / "stages.json").read_text())["agent_tokens_per_s"] for run in runs]
            later = [ms for run in runs for ms in later_tokens(run)]
            rows.append(
                f"| {node} | {ARMS[arm]} | {len(runs)} | {spread([s[0] for s in seen])} | "
                f"{spread([s[1] for s in seen])} | {spread([s[2] for s in seen])} | {spread([s[3] for s in seen])} | "
                f"{statistics.median(later):.1f} | {statistics.median(agents):,.0f} |"
            )
    return "\n".join(rows)


def hold() -> str:
    """How long the decode hold lasts: the calls' first token, their 16th token and their 48th, the whole reply (p50
    and p95, the median over the replicates), and the agents' computed tokens a second."""
    rows = [
        "| Node | Arm | Runs | First token p50, ms | p95 | 16th token p50 | p95 | 48th token p50 | p95 | Agents' tokens/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = points()
    for node in NODES:
        for arm in TABLES["hold"]:
            runs = groups.get((node, arm))
            if not runs:
                continue
            seen = [client(run) for run in runs]
            reply = [REPLY.search((run / "tail.txt").read_text().split("\nreplies: ", 1)[1]) for run in runs]
            agents = [json.loads((run / "stages.json").read_text())["agent_tokens_per_s"] for run in runs]
            rows.append(
                f"| {node} | {ARMS[arm]} | {len(runs)} | {spread([s[0] for s in seen])} | "
                f"{spread([s[1] for s in seen])} | {spread([s[2] for s in seen])} | {spread([s[3] for s in seen])} | "
                f"{spread([float(r[1]) for r in reply])} | {spread([float(r[3]) for r in reply])} | "
                f"{statistics.median(agents):,.0f} |"
            )
    return "\n".join(rows)


def stages() -> str:
    rows = [
        "| Node | Arm | First token in the engine, mean ms | "
        + " | ".join(STAGES)
        + " | Turns loading from the CPU tier |",
        "|---|---|---:|" + "---:|" * len(STAGES) + "---:|",
    ]
    groups = points()
    for node in NODES:
        for arm in TABLES["stages"]:
            runs = groups.get((node, arm))
            if not runs:
                continue
            stage = [json.loads((run / "stages.json").read_text()) for run in runs]
            turns = sum(s["turns"] for s in stage)

            def mean(key: str, stage: list[dict] = stage) -> float:
                return sum(s["first_token_ms"][key]["mean"] * s["turns"] for s in stage) / turns

            loaded = sum(s["turns_loaded_from_the_cpu_tier"] for s in stage)
            rows.append(
                f"| {node} | {ARMS[arm]} | {mean('ttft'):.0f} | "
                + " | ".join(f"{mean(key):.0f}" for key in STAGES)
                + f" | {loaded} of {turns} |"
            )
    return "\n".join(rows)


def speculation() -> str:
    rows = [
        "| Drafts | Tokens a step | ms a token, p50 | First token p50, ms | 16th token p50, ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in sorted((HERE / "gray").glob("g-spec-*"), key=lambda p: (p.name != "g-spec-base", p.name)):
        data = json.loads((run / "spec.json").read_text())
        turns = data["turns"]
        spec = data["spec"]
        per_token = [
            (t["last"] - t["first"]) / (t["tokens"] - 1) for t in turns if t["tokens"] > 1 and t["last"] is not None
        ]
        sixteenth = [t["16th"] for t in turns if t["16th"] is not None]
        drafts = run.name.removeprefix("g-spec-")
        accepted = (
            f"{1 + spec['num_accepted_tokens'] / spec['num_drafts']:.2f}" if spec.get("num_drafts") else "1.00"
        )
        rows.append(
            f"| {'none' if drafts == 'base' else drafts.removeprefix('k')} | {accepted} | "
            f"{statistics.median(per_token):.1f} | {statistics.median(t['first'] for t in turns):.0f} | "
            f"{statistics.median(sixteenth):.0f} |"
        )
    return "\n".join(rows)


def restore() -> str:
    rows = [
        "| Node | Voice conversations answered | Long documents | Restored from the CPU tier, identical to computed |",
        "|---|---:|---:|---:|",
    ]
    for node in NODES:
        for run in sorted((HERE / node).glob("?-agree-*")):
            data = json.loads((run / "agree.json").read_text())
            same = sum(a == b for a, b in zip(data["documents"], data["restored"], strict=True))
            rows.append(
                f"| {node} | {len(data['conversations'])} | {len(data['documents'])} | "
                f"{same} of {len(data['restored'])} |"
            )
    return "\n".join(rows)


def table(name: str) -> str:
    if name == "stages":
        return stages()
    if name == "speculation":
        return speculation()
    if name == "restore":
        return restore()
    if name == "hold":
        return hold()
    return outcomes(name)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TABLES:
        sys.exit(f"usage: python3 table.py {'|'.join(TABLES)}")
    print(table(sys.argv[1]))
