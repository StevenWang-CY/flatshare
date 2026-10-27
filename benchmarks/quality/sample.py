"""The quality gate's samples (GATE.md), drawn with seed 2026 and written as ids only.

QASPER: from the test split's questions, 150 that at least one annotator answered and 50 that every annotator marked
unanswerable, at most two per paper. The scarcer stratum is drawn first, each from its questions sorted by id and
shuffled with its own random.Random(2026), a question taken while its stratum needs more and its paper has fewer than
two. Re-DocRED: 100 of the test split's 500 documents, random.Random(2026).sample over their indices, kept with their
titles (DocRED's document ids).

  python sample.py --data DIR                      # writes sample.json beside this script
  python sample.py --data DIR --dev --out FILE     # the development sample: 20 QASPER validation questions (15 + 5)
                                                   # and 10 Re-DocRED dev documents, for format and parsing only
DIR is fetch.sh's directory. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 2026
PER_PAPER = 2
TEST = {
    "qasper": (
        "qasper-test-v0.3.json",
        "6e29ad410e6e39aa1936017fb965b30a20eb2e7751997f55b97c9d281aa884e5",
    ),
    "redocred": (
        "redocred-test.json",
        "ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c",
    ),
    "answerable": 150,
    "unanswerable": 50,
    "documents": 100,
}
DEV = {
    "qasper": (
        "qasper-dev-v0.3.json",
        "2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93",
    ),
    "redocred": (
        "redocred-dev.json",
        "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f",
    ),
    "answerable": 15,
    "unanswerable": 5,
    "documents": 10,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(data: Path, name: str, expected: str):
    path = data / name
    got = sha256(path)
    if got != expected:
        raise SystemExit(f"{path} has sha256 {got}, expected {expected} (fetch.sh)")
    return json.loads(path.read_text())


def strata(papers: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(paper, question id) of the questions at least one annotator answered, and of those every annotator marked
    unanswerable, each sorted by question id."""
    answerable, unanswerable = [], []
    for paper, p in papers.items():
        for q in p["qas"]:
            marks = [a["answer"]["unanswerable"] for a in q["answers"]]
            if not marks:
                continue
            (unanswerable if all(marks) else answerable).append((paper, q["question_id"]))
    key = lambda pq: pq[1]  # noqa: E731
    return sorted(answerable, key=key), sorted(unanswerable, key=key)


def draw_questions(
    papers: dict,
    n_answerable: int,
    n_unanswerable: int,
    seed: int = SEED,
    per_paper: int = PER_PAPER,
) -> list[dict]:
    answerable, unanswerable = strata(papers)
    taken: Counter[str] = Counter()
    out = []
    for name, pool, n in (
        ("unanswerable", unanswerable, n_unanswerable),
        ("answerable", answerable, n_answerable),
    ):
        order = list(pool)
        random.Random(seed).shuffle(order)
        got = 0
        for paper, qid in order:
            if got == n:
                break
            if taken[paper] < per_paper:
                taken[paper] += 1
                got += 1
                out.append({"paper": paper, "question_id": qid, "stratum": name})
        if got < n:
            raise ValueError(f"only {got} of {n} {name} questions within {per_paper} per paper")
    return out


def draw_documents(docs: list, n: int, seed: int = SEED) -> list[dict]:
    return [
        {"index": i, "title": docs[i]["title"]}
        for i in sorted(random.Random(seed).sample(range(len(docs)), n))
    ]


def build(data: Path, spec: dict) -> dict:
    qname, qsum = spec["qasper"]
    rname, rsum = spec["redocred"]
    papers = load(data, qname, qsum)
    docs = load(data, rname, rsum)
    return {
        "gate": "benchmarks/quality/GATE.md",
        "seed": SEED,
        "qasper": {
            "file": qname,
            "sha256": qsum,
            "per_paper": PER_PAPER,
            "questions": draw_questions(papers, spec["answerable"], spec["unanswerable"]),
        },
        "redocred": {
            "file": rname,
            "sha256": rsum,
            "documents": draw_documents(docs, spec["documents"]),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="fetch.sh's directory")
    p.add_argument(
        "--dev", action="store_true", help="the development sample (validation and dev splits)"
    )
    p.add_argument("--out", type=Path, default=HERE / "sample.json")
    a = p.parse_args()
    sample = build(a.data, DEV if a.dev else TEST)
    a.out.write_text(json.dumps(sample, indent=1) + "\n")
    qs = sample["qasper"]["questions"]
    print(
        f"{a.out}: {sum(q['stratum'] == 'answerable' for q in qs)} answerable and "
        f"{sum(q['stratum'] == 'unanswerable' for q in qs)} unanswerable questions from "
        f"{len({q['paper'] for q in qs})} papers; {len(sample['redocred']['documents'])} documents"
    )


if __name__ == "__main__":
    main()
