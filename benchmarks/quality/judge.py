"""The quality gate's judge (GATE.md): Claude Opus through the `claude` CLI on the subscription, never an API key.

Each item goes with only what a rater needs (the question, the annotators' answers and the answer; or the cited
sentences and the fact or summary sentence), never the model's name, in batches of about 20 of one kind, and comes
back in a fixed JSON form (--json-schema; the answer is the output's structured_output). The CLI runs with a
minimal environment (HOME, PATH and USER, with CLAUDE.md files and auto memory off), no tools, no settings, MCP
servers or session, in an empty directory. Every run must name an Opus model with output tokens in its modelUsage,
and no other model may produce any; the account (claude auth status) is pinned for the whole run.

Rulings are cached by item hash and rater: a rerun asks only what has no ruling, here or in a --cache file (another
record's rulings). Then a tenth of the record's rulings, drawn with seed 2026, are asked again in fresh batches in
another order (pass "reask"); gate.py reads their agreement.

  python judge.py --items ITEMS.jsonl --rulings OUT/rulings.jsonl [--cache OTHER/rulings.jsonl] --workdir EMPTY_DIR
  python judge.py --probe --workdir EMPTY_DIR   # one tiny call, to check the transport
Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 2026
BATCH = 20
REASK = 0.10

QA_RATER = """You rate answers to questions about NLP research papers. Each item gives a question about a paper, the \
answers that annotators wrote after reading the whole paper, and a candidate answer to rate. Compare the candidate \
with the annotators' answers only; do not use your own knowledge of the paper or the field.

Rule each item:
- correct: the candidate gives the same answer as at least one annotator. Wording, length and detail may differ, as \
long as the substance of that annotator's answer is there and nothing in the candidate contradicts it. A candidate \
that answers yes or no is correct only when an annotator answered the same yes or no.
- partial: the candidate gives only part of the answer of the annotator it comes closest to, or gives it mixed with \
something wrong.
- incorrect: anything else, including a different answer, a refusal, or an answer that no annotator's answer contains.

Give every item one ruling and a reason of one short sentence."""

FACT_RATER = """You check facts extracted from Wikipedia articles. Each item gives the article's sentences that the \
extractor cited for the fact, numbered as in the article, and the fact as a head, a relation and a tail. The \
relation is a Wikidata property: the head is the item and the tail the value, so head Paris, relation country, \
tail France says that Paris is in France, and head Rihanna, relation country of citizenship, tail Barbados says \
that Rihanna is a citizen of Barbados.

Rule each item:
- yes: the cited sentences state the fact, in so many words, by a plain paraphrase, or by a reading of them that \
any careful reader would make with certainty (for example "Lyon, France" states that Lyon's country is France).
- no: the fact is wrong or reversed, the relation does not fit what the sentences say, or the fact needs other \
sentences, outside knowledge or a guess.

Give every item one ruling and a reason of one short sentence."""

SUMMARY_RATER = """You check the sentences of document summaries against their sources. Each item gives the \
sentences of a Wikipedia article that a summary sentence cites, numbered as in the article, and the summary \
sentence.

Rule each item:
- yes: everything the summary sentence says is stated by the cited sentences, in so many words or by a plain \
paraphrase.
- no: some part of it is not in the cited sentences, contradicts them, or needs other sentences or outside knowledge.

Give every item one ruling and a reason of one short sentence."""

# kind: (the rater's instructions, its rulings, the ruling that counts for the gate)
RATERS = {
    "qa": (QA_RATER, ("correct", "partial", "incorrect"), "correct"),
    "fact": (FACT_RATER, ("yes", "no"), "yes"),
    "summary": (SUMMARY_RATER, ("yes", "no"), "yes"),
}


def schema(kind: str) -> dict:
    labels = list(RATERS[kind][1])
    ruling = {
        "type": "object",
        "properties": {
            "item": {"type": "integer"},
            "reason": {"type": "string"},
            "ruling": {"type": "string", "enum": labels},
        },
        "required": ["item", "reason", "ruling"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"rulings": {"type": "array", "items": ruling}},
        "required": ["rulings"],
        "additionalProperties": False,
    }


def rater_id(kind: str) -> str:
    """The rater's version: its instructions and form."""
    blob = json.dumps({"instructions": RATERS[kind][0], "schema": schema(kind)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _sentences(pairs: list) -> str:
    return "\n".join(f"[{n}] {text}" for n, text in pairs)


def render(kind: str, k: int, content: dict) -> str:
    if kind == "qa":
        refs = "\n".join(f"- {r}" for r in content["references"])
        return (
            f"Item {k}\nQuestion: {content['question']}\nAnnotators' answers:\n{refs}\n"
            f"Candidate answer: {content['answer']}"
        )
    if kind == "fact":
        head, relation, tail = content["fact"]
        return (
            f"Item {k}\nCited sentences:\n{_sentences(content['sentences'])}\n"
            f"Fact:\n  head: {head}\n  relation: {relation}\n  tail: {tail}"
        )
    return (
        f"Item {k}\nCited sentences:\n{_sentences(content['sentences'])}\n"
        f"Summary sentence: {content['summary']}"
    )


def prompt(kind: str, contents: list[dict]) -> str:
    body = "\n\n".join(render(kind, k, c) for k, c in enumerate(contents, 1))
    return f"Rule each of these {len(contents)} items.\n\n{body}"


# ---- the transport -----------------------------------------------------------------------------------------------


def clean_env() -> dict:
    """HOME, PATH and USER only, so that no key and no parent session reaches the child; CLAUDE.md files and auto
    memory off."""
    return {
        "HOME": os.environ["HOME"],
        "PATH": os.environ["PATH"],
        "USER": os.environ.get("USER") or os.environ.get("LOGNAME", ""),
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    }


def command(system: str, form: dict, text: str) -> list[str]:
    return [
        "claude",
        "--print",
        "--model",
        "opus",
        "--output-format",
        "json",
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--exclude-dynamic-system-prompt-sections",
        "--system-prompt",
        system,
        "--json-schema",
        json.dumps(form),
        text,
    ]


def account(workdir: Path) -> dict:
    p = subprocess.run(
        ["claude", "auth", "status"],
        env=clean_env(),
        cwd=workdir,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    status = json.loads(p.stdout)
    if not status.get("loggedIn") or status.get("authMethod") != "claude.ai":
        raise SystemExit(f"claude is not on a claude.ai subscription login: {status}")
    return {"email": status.get("email"), "subscription": status.get("subscriptionType")}


def opus_check(out: dict) -> tuple[str, int]:
    """The Opus model that answered and its output tokens; raises unless an Opus model produced output and no other
    model produced any."""
    usage = out.get("modelUsage") or {}
    producing = {
        m: u.get("outputTokens", 0) for m, u in usage.items() if u.get("outputTokens", 0) > 0
    }
    opus = [m for m in producing if "opus" in m]
    if (
        not opus
        or set(producing) != set(opus)
        or (out.get("usage") or {}).get("output_tokens", 0) <= 0
    ):
        raise ValueError(f"modelUsage {usage} does not show Opus alone producing the answer")
    return opus[0], producing[opus[0]]


def call(system: str, form: dict, text: str, workdir: Path) -> dict:
    p = subprocess.run(
        command(system, form, text),
        env=clean_env(),
        cwd=workdir,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=1200,
    )
    if p.returncode != 0:
        raise RuntimeError(f"claude exited {p.returncode}: {p.stderr[-500:]} {p.stdout[-500:]}")
    out = json.loads(p.stdout)
    if out.get("is_error") or out.get("subtype") != "success":
        raise RuntimeError(
            f"claude answered an error: {out.get('subtype')} {out.get('result', '')[:300]}"
        )
    return out


# ---- batches -----------------------------------------------------------------------------------------------------


class Judge:
    def __init__(self, workdir: Path, rulings: Path, jobs: int) -> None:
        if any(workdir.iterdir()):
            raise SystemExit(f"{workdir} is not empty")
        self.workdir, self.rulings, self.jobs = workdir, rulings, jobs
        self.lock = threading.Lock()
        self.account = account(workdir)
        print(f"judge: account {self.account}", file=sys.stderr)

    def batch(self, kind: str, items: list[dict], pass_: str) -> list[dict]:
        system, labels, _ = RATERS[kind]
        form = schema(kind)
        text = prompt(kind, [it["content"] for it in items])
        batch_id = hashlib.sha256(
            json.dumps([pass_, [it["hash"] for it in items]]).encode()
        ).hexdigest()[:12]
        last = None
        for attempt in range(1, 4):
            try:
                before = account(self.workdir)
                t0 = time.monotonic()
                out = call(system, form, text, self.workdir)
                after = account(self.workdir)
                if before != self.account or after != self.account:
                    raise SystemExit(
                        f"the login moved during the run: {self.account} -> {before} -> {after}"
                    )
                model, tokens = opus_check(out)
                got = out.get("structured_output") or {}
                rows = got.get("rulings")
                if not isinstance(rows, list) or sorted(r.get("item") for r in rows) != list(
                    range(1, len(items) + 1)
                ):
                    raise ValueError(f"rulings do not name items 1..{len(items)} once each")
                if any(r.get("ruling") not in labels for r in rows):
                    raise ValueError("a ruling outside the form")
                at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                by_item = {r["item"]: r for r in rows}
                return [
                    {
                        "item": it["hash"],
                        "kind": kind,
                        "rater": rater_id(kind),
                        "pass": pass_,
                        "ruling": by_item[k]["ruling"],
                        "reason": by_item[k]["reason"],
                        "batch": batch_id,
                        "position": k,
                        "size": len(items),
                        "attempt": attempt,
                        "model": model,
                        "output_tokens": tokens,
                        "seconds": round(time.monotonic() - t0, 1),
                        "account": self.account["email"],
                        "at": at,
                    }
                    for k, it in enumerate(items, 1)
                ]
            except (RuntimeError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
                last = e
                print(f"judge: batch {batch_id} attempt {attempt} failed: {e}", file=sys.stderr)
                time.sleep(10 * attempt)
        raise RuntimeError(f"batch {batch_id} failed three times: {last}")

    def run(self, batches: list[tuple[str, list[dict]]], pass_: str) -> list[dict]:
        out: list[dict] = []

        def one(b: tuple[str, list[dict]]) -> None:
            rows = self.batch(b[0], b[1], pass_)
            with self.lock:
                with self.rulings.open("a") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.extend(rows)
                print(
                    f"judge: {pass_} {b[0]} batch of {len(rows)} done, {len(out)} rulings",
                    file=sys.stderr,
                )

        with ThreadPoolExecutor(self.jobs) as pool:
            list(pool.map(one, batches))
        return out


def batches_of(items: list[dict], seed: int) -> list[tuple[str, list[dict]]]:
    """Items grouped by kind, shuffled with the seed and cut into batches of about BATCH."""
    out = []
    for kind in RATERS:
        group = [it for it in items if it["kind"] == kind]
        random.Random(seed).shuffle(group)
        if not group:
            continue
        n = math.ceil(len(group) / BATCH)
        size = math.ceil(len(group) / n)
        out += [(kind, group[i : i + size]) for i in range(0, len(group), size)]
    return out


def load(paths: list[Path]) -> dict[tuple[str, str, str], dict]:
    have = {}
    for path in paths:
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                have.setdefault((r["item"], r["rater"], r["pass"]), r)
    return have


def reask_sample(hashes: list[str], seed: int = SEED, share: float = REASK) -> list[str]:
    """The tenth of the rulings asked again: ceil(share x n) of the sorted hashes, drawn with the seed."""
    pool = sorted(hashes)
    return random.Random(seed).sample(pool, math.ceil(share * len(pool)))


def agreement(rows: list[dict]) -> dict:
    """How often a re-asked item's ruling agrees with its first, on what the gate counts (and on the label)."""
    first = {r["item"]: r for r in rows if r["pass"] == "first"}
    again = [r for r in rows if r["pass"] == "reask"]
    same = sum(
        (r["ruling"] == RATERS[r["kind"]][2])
        == (first[r["item"]]["ruling"] == RATERS[r["kind"]][2])
        for r in again
    )
    label = sum(r["ruling"] == first[r["item"]]["ruling"] for r in again)
    return {"reasked": len(again), "agree": same, "agree_label": label}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=Path, help="score.py's judge items")
    p.add_argument("--rulings", type=Path, help="the record's rulings.jsonl (appended to)")
    p.add_argument("--cache", type=Path, action="append", default=[], help="other rulings to reuse")
    p.add_argument("--workdir", type=Path, required=True, help="an empty directory for the CLI")
    p.add_argument("--jobs", type=int, default=4, help="calls at once")
    p.add_argument("--probe", action="store_true", help="one tiny call, then stop")
    a = p.parse_args()
    a.workdir.mkdir(parents=True, exist_ok=True)
    if a.probe:
        judge = Judge(a.workdir, Path(os.devnull), 1)
        item = {
            "hash": "probe",
            "kind": "summary",
            "content": {
                "sentences": [[1, "Paris is the capital of France."]],
                "summary": "Paris is in France.",
            },
        }
        rows = judge.batch("summary", [item], "probe")
        print(json.dumps(rows[0], indent=1))
        return
    items = [json.loads(line) for line in a.items.read_text().splitlines()]
    judge = Judge(a.workdir, a.rulings, a.jobs)
    have = load([a.rulings])
    cached = load(a.cache)
    reused = []
    for it in items:
        key = (it["hash"], rater_id(it["kind"]), "first")
        if key not in have and key in cached:
            reused.append(cached[key])
            have[key] = cached[key]
    pending = [it for it in items if (it["hash"], rater_id(it["kind"]), "first") not in have]
    print(
        f"judge: {len(items)} items, {len(items) - len(pending) - len(reused)} ruled here, "
        f"{len(reused)} from the cache, {len(pending)} to ask",
        file=sys.stderr,
    )
    if reused:
        with a.rulings.open("a") as f:
            for r in reused:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    judge.run(batches_of(pending, SEED), "first")

    by_hash = {it["hash"]: it for it in items}
    again = [by_hash[h] for h in reask_sample(list(by_hash))]
    have = load([a.rulings])
    cached = load(a.cache)
    reused, pending = [], []
    for it in again:
        key = (it["hash"], rater_id(it["kind"]), "reask")
        if key in have:
            continue
        if key in cached:
            reused.append(cached[key])
        else:
            pending.append(it)
    if reused:
        with a.rulings.open("a") as f:
            for r in reused:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"judge: {len(again)} re-asked, {len(reused)} from the cache, {len(pending)} to ask",
        file=sys.stderr,
    )
    judge.run(batches_of(pending, SEED + 1), "reask")
    rows = [r for r in load([a.rulings]).values() if r["item"] in by_hash]
    print(json.dumps(agreement(rows)))


if __name__ == "__main__":
    main()
