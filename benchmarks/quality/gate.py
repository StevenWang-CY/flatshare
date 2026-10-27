"""The quality gate's verdict (GATE.md) from a record's automatic scores and the judge's rulings.

Every criterion with its value, its count, its Wilson 95% interval and whether that interval straddles the
threshold; PASS when both tasks meet every criterion, FAIL otherwise, and INCONCLUSIVE, rather than either, when
fewer than 90% of the re-asked rulings agree with the first.

  python gate.py RECORD_DIR      # reads scores.json and rulings.jsonl, writes verdict.json and prints it
Stdlib only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054  # the normal's 97.5th percentile
AGREEMENT = 0.90
# the ruling of each kind that counts for the gate (judge.RATERS)
POSITIVE = {"qa": "correct", "fact": "yes", "summary": "yes"}


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float] | None:
    """The Wilson score interval of k successes in n trials, or None for no trials."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # At 0 of n and n of n an end is exactly 0 or 1, which rounding would otherwise leave a hair inside.
    return (0.0 if k == 0 else max(0.0, centre - half)), (
        1.0 if k == n else min(1.0, centre + half)
    )


def criterion(task: str, name: str, k: int, n: int, threshold: float, note: str = "") -> dict:
    ci = wilson(k, n)
    value = k / n if n else None
    return {
        "task": task,
        "name": name,
        "k": k,
        "n": n,
        "value": None if value is None else round(value, 4),
        "interval": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
        "threshold": threshold,
        "pass": value is not None and value >= threshold,
        "straddles": ci is not None and ci[0] < threshold <= ci[1],
        "note": note,
    }


def first_rulings(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        if r["pass"] != "first":
            continue
        if r["item"] in out and out[r["item"]]["ruling"] != r["ruling"]:
            raise SystemExit(f"two first rulings for {r['item']}")
        out[r["item"]] = r
    return out


def verdict(scores: dict, rows: list[dict]) -> dict:
    first = first_rulings(rows)

    def good(h: str) -> bool:
        if h not in first:
            raise SystemExit(f"no ruling for item {h}")
        r = first[h]
        return r["ruling"] == POSITIVE[r["kind"]]

    qa = scores["qa"]
    answerable = [s for s in qa if s["stratum"] == "answerable"]
    unanswerable = [s for s in qa if s["stratum"] == "unanswerable"]
    answered = [s for s in qa if not s["abstained"]]
    correct = {s["id"] for s in answered if s["stratum"] == "answerable" and good(s["judge"])}
    coverage = [s for s in answerable if s["retrieved_gold"]]
    covered_correct = [s for s in coverage if s["id"] in correct]
    citations = sum(len(s["citations"]) for s in qa)
    invalid = sum(s["invalid_citations"] for s in qa)

    en = scores["enrich"]
    accepted = [s for s in en if s["accepted"]]
    facts = [f for s in accepted for f in s["facts"]]
    summary = [x for s in accepted for x in s["summary"]]

    criteria = [
        criterion("qa", "precision", len(correct), len(answered), 0.80, "answered questions that are correct"),
        criterion("qa", "coverage", len(covered_correct), len(coverage), 0.60,
                  "answerable questions with evidence retrieved, answered correctly"),
        criterion("qa", "abstention", sum(s["abstained"] for s in unanswerable), len(unanswerable), 0.70,
                  "unanswerable questions abstained on"),
        criterion("qa", "citations", sum(s["supporting_citation"] for s in covered_correct), len(covered_correct),
                  0.80, "correct answers in the coverage set citing a supporting paragraph"),
        criterion("qa", "citations valid", citations - invalid, citations, 1.0,
                  "every citation names a paragraph that was given"),
        criterion("qa", "speakable", sum(s["speakable"] for s in answered), len(answered), 0.95,
                  "answers of at most two sentences with no list or markdown"),
        criterion("enrich", "accepted records", len(accepted), len(en), 0.95, "documents with an accepted record"),
        criterion("enrich", "fact precision", sum(f["gold"] or good(f["judge"]) for f in facts), len(facts), 0.80,
                  "facts in accepted records that match gold or that their cited sentences state"),
        criterion("enrich", "fact recall", sum(len(s["recovered"]) for s in en), sum(s["gold_facts"] for s in en),
                  0.20, "gold facts of all the documents recovered by accepted records"),
        criterion("enrich", "entity recall", sum(len(s["matched_entities"]) for s in en),
                  sum(s["gold_entities"] for s in en), 0.70,
                  "gold PER, ORG, LOC and TIME entities of all the documents matched by accepted records"),
        criterion("enrich", "summaries", sum(good(x["judge"]) for x in summary), len(summary), 0.90,
                  "summary sentences in accepted records that their cited sentences state"),
    ]  # fmt: skip

    items = {s["judge"] for s in qa if s["judge"]}
    items |= {f["judge"] for f in facts if f["judge"]} | {x["judge"] for x in summary}
    again = [r for r in rows if r["pass"] == "reask" and r["item"] in items]
    agree = sum(good(r["item"]) == (r["ruling"] == POSITIVE[r["kind"]]) for r in again)
    same_label = sum(first[r["item"]]["ruling"] == r["ruling"] for r in again)
    consistency = criterion("judge", "consistency", agree, len(again), AGREEMENT,
                            "re-asked rulings that agree with the first on what the gate counts")  # fmt: skip
    consistency["same_label"] = same_label
    consistency["ruled_items"] = len(items)
    if again and len(again) < math.ceil(0.1 * len(items)):
        raise SystemExit(
            f"only {len(again)} re-asked rulings of {len(items)}: the judge has not finished"
        )

    if not consistency["pass"]:
        result = "INCONCLUSIVE"
    elif all(c["pass"] for c in criteria):
        result = "PASS"
    else:
        result = "FAIL"
    retrieval = criterion("qa", "retrieval recall", len(coverage), len(answerable), 0.0,
                          "answerable questions with a gold evidence paragraph among the six (reported, no threshold)")  # fmt: skip
    return {
        "verdict": result,
        "criteria": criteria,
        "judge": consistency,
        "reported": [retrieval],
        "failed": [f"{c['task']} {c['name']}" for c in criteria if not c["pass"]],
    }


def table(v: dict) -> str:
    lines = [
        f"{'task':7} {'criterion':17} {'value':>7} {'k/n':>11} {'95% interval':>17} {'threshold':>9}  result"
    ]
    for c in v["criteria"] + [v["judge"]] + v["reported"]:
        iv = "-" if c["interval"] is None else f"[{c['interval'][0]:.3f}, {c['interval'][1]:.3f}]"
        val = "-" if c["value"] is None else f"{c['value']:.3f}"
        res = "reported" if c in v["reported"] else ("pass" if c["pass"] else "FAIL")
        if c["straddles"] and c not in v["reported"]:
            res += " (interval straddles)"
        lines.append(
            f"{c['task']:7} {c['name']:17} {val:>7} {c['k']:>5}/{c['n']:<5} {iv:>17} {c['threshold']:>9.2f}  {res}"
        )
    lines.append(f"verdict: {v['verdict']}")
    return "\n".join(lines)


def main() -> None:
    record = Path(sys.argv[1])
    scores = json.loads((record / "scores.json").read_text())
    rows = [json.loads(line) for line in (record / "rulings.jsonl").read_text().splitlines()]
    v = verdict(scores, rows)
    (record / "verdict.json").write_text(json.dumps(v, indent=1) + "\n")
    print(table(v))


if __name__ == "__main__":
    main()
