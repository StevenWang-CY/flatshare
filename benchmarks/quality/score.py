"""The quality gate's automatic scoring (GATE.md), and the items that need the judge.

QASPER: abstention (the fixed sentence, nothing else), the citations' validity (each names a paragraph that was
given) and support (it names a paragraph some annotator marked as evidence), retrieval recall (a gold evidence
paragraph among the six), and whether an answer is speakable (at most two sentences, no list or markdown). Every
answer to an answerable question goes to the judge; an answer to an unanswerable one is wrong without it.
Re-DocRED: a record is accepted when it parses, matches the schema (with unique entity ids), every entity name
occurs verbatim in the document, every fact's head and tail are entities of the record (ids it lists) and every
cited sentence number exists. In accepted
records, entities match gold ones by name after normalization (case, whitespace, punctuation) with any mention, a
fact matches a gold one when both entities match and the relation is the same, and a fact that matches none, and
every summary sentence, go to the judge with its cited sentences.

  python score.py --data DIR --sample sample.json --out OUT --items ITEMS.jsonl
Reads OUT/outputs.jsonl; writes OUT/scores.json (ids, flags and judge items' hashes; no dataset text) and
ITEMS.jsonl, the judge's items with what a rater needs, which hold the sets' text and so stay out of the repository.
Stdlib only; the prompts and parsers are generate.py's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate

# ---- QASPER ------------------------------------------------------------------------------------------------------


def _letters(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'").replace("'", "")
    return re.findall(r"[^\W_]+", t)


ABSTENTION_WORDS = _letters(generate.ABSTENTION)


def is_abstention(text: str) -> bool:
    """The reply is the fixed abstention sentence, up to case, punctuation, whitespace and citation markers."""
    return _letters(generate.strip_citations(text)) == ABSTENTION_WORDS


ABBREVIATIONS = {
    "al", "approx", "cf", "dr", "e.g", "eq", "eqs", "etc", "fig", "figs", "i.e", "inc", "jr", "ltd", "mr", "mrs",
    "ms", "no", "nos", "prof", "resp", "sec", "sr", "st", "tab", "u.k", "u.s", "viz", "vs",
}  # fmt: skip
SENTENCE_END = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s|$)")


def count_sentences(text: str) -> int:
    """Sentences in a spoken answer: ends are . ! or ? before whitespace or the end, not after a known abbreviation
    or a single capital's initial, and not before a lowercase word. Citation markers are dropped first."""
    t = " ".join(generate.strip_citations(text).split())
    n, start = 0, 0
    for m in SENTENCE_END.finditer(t):
        if t[m.start()] == ".":
            word = re.search(r"([^\W\d_][\w.]*)$", t[start : m.start()])
            w = word.group(1).lower() if word else ""
            if w in ABBREVIATIONS or (word and len(word.group(1)) == 1 and word.group(1).isupper()):
                continue
        rest = t[m.end() :].lstrip()
        if rest and rest[0].islower():
            continue
        if t[start : m.end()].strip():
            n += 1
        start = m.end()
    return n + (1 if t[start:].strip() else 0)


MARKDOWN = re.compile(
    r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+"  # a list item
    r"|^\s*#{1,6}\s"  # a heading
    r"|^\s*>\s"  # a quote block
    r"|^\s*\|.*\|\s*$"  # a table row
    r"|\*\*|__|`"  # bold, code
    r"|(?<![\w*])\*(?=\S)[^*\n]+(?<=\S)\*(?![\w*])"  # italic
)


def has_markdown(text: str) -> bool:
    return MARKDOWN.search(text) is not None


def squash(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def gold_paragraphs(paper: dict, question: dict) -> set[int]:
    """The paragraphs (indices into generate.paragraphs) that some annotator marked as evidence; evidence that is a
    figure, a table or a section heading names no paragraph."""
    evidence = {squash(e) for a in question["answers"] for e in a["answer"]["evidence"]}
    return {
        i for i, p in enumerate(generate.paragraphs(paper)) if p.strip() and squash(p) in evidence
    }


def references(question: dict) -> list[str]:
    """The annotators' answers, as the official evaluator reads them (spans, else free form, else yes or no); the
    annotators who marked the question unanswerable give none."""
    out = []
    for a in question["answers"]:
        ans = a["answer"]
        if ans["unanswerable"]:
            continue
        if ans["extractive_spans"]:
            ref = "Spans quoted from the paper: " + "; ".join(
                f'"{s.strip()}"' for s in ans["extractive_spans"]
            )
        elif ans["free_form_answer"].strip():
            ref = ans["free_form_answer"].strip()
        elif ans["yes_no"] is not None:
            ref = "Yes" if ans["yes_no"] else "No"
        else:
            continue
        if ref not in out:
            out.append(ref)
    return out


def item_hash(kind: str, content: dict) -> str:
    return hashlib.sha256(
        json.dumps({"kind": kind, "content": content}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def score_answer(row: dict, paper: dict, question: dict) -> tuple[dict, dict | None]:
    """A QASPER output's automatic scores and, for an answer to an answerable question, its judge item."""
    text = row["text"]
    cites = generate.parse_citations(text)
    given = len(row["retrieved"])
    gold = gold_paragraphs(paper, question)
    abstained = is_abstention(text)
    s = {
        "id": row["id"],
        "paper": row["paper"],
        "stratum": row["stratum"],
        "abstained": abstained,
        "citations": cites,
        "invalid_citations": sum(not 1 <= n <= given for n in cites),
        "supporting_citation": any(
            1 <= n <= given and row["retrieved"][n - 1] in gold for n in cites
        ),
        "retrieved_gold": any(i in gold for i in row["retrieved"]),
        "gold_paragraphs": len(gold),
        "sentences": count_sentences(text),
        "markdown": has_markdown(text),
        "finish_reason": row["finish_reason"],
        "judge": None,
    }
    s["speakable"] = s["sentences"] <= 2 and not s["markdown"]
    item = None
    if row["stratum"] == "answerable" and not abstained:
        content = {
            "question": question["question"].strip(),
            "references": references(question),
            "answer": generate.strip_citations(text),
        }
        s["judge"] = item_hash("qa", content)
        item = {"hash": s["judge"], "kind": "qa", "content": content}
    return s, item


# ---- Re-DocRED ---------------------------------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """A name for matching: NFKC, case folded, letters and digits only (no whitespace or punctuation)."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", name).casefold() if ch.isalnum())


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ints(value) -> bool:
    return isinstance(value, list) and bool(value) and all(_int(x) for x in value)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def schema_problems(record, relations: set[str]) -> list[str]:
    """Where a parsed record departs from the schema: an object of exactly entities (id, name, type; the ids
    unique), facts (head and tail ids, relation, sentences) and a summary of one or two sentences (text, sentences);
    sentences are non-empty lists of integers."""
    if not isinstance(record, dict):
        return ["the record is not an object"]
    problems = []
    if set(record) != {"entities", "facts", "summary"}:
        problems.append(f"keys {sorted(record)}")
    entities, facts, summary = record.get("entities"), record.get("facts"), record.get("summary")
    if not isinstance(entities, list):
        problems.append("entities is not a list")
    else:
        for k, e in enumerate(entities):
            if not isinstance(e, dict) or set(e) != {"id", "name", "type"}:
                problems.append(f"entity {k} is not an id, a name and a type")
            elif not _int(e["id"]):
                problems.append(f"entity {k}'s id is not a number")
            elif not _text(e["name"]):
                problems.append(f"entity {k} has no name")
            elif e["type"] not in generate.ENTITY_TYPES:
                problems.append(f"entity {k} has type {e['type']!r}")
        ids = [e["id"] for e in entities if isinstance(e, dict) and "id" in e]
        if len(ids) != len(set(map(repr, ids))):
            problems.append("entity ids repeat")
    if not isinstance(facts, list):
        problems.append("facts is not a list")
    else:
        for k, f in enumerate(facts):
            if not isinstance(f, dict) or set(f) != {"head", "relation", "tail", "sentences"}:
                problems.append(f"fact {k} is not a head, relation, tail and sentences")
            elif not (_int(f["head"]) and _int(f["tail"])):
                problems.append(f"fact {k}'s head or tail is not an id")
            elif f["relation"] not in relations:
                problems.append(f"fact {k} has relation {f['relation']!r}")
            elif not _ints(f["sentences"]):
                problems.append(f"fact {k}'s sentences are not a list of numbers")
    if not isinstance(summary, list) or not 1 <= len(summary) <= 2:
        problems.append("the summary is not one or two sentences")
    else:
        for k, s in enumerate(summary):
            if not isinstance(s, dict) or set(s) != {"text", "sentences"}:
                problems.append(f"summary sentence {k} is not a text and sentences")
            elif not _text(s["text"]):
                problems.append(f"summary sentence {k} has no text")
            elif not _ints(s["sentences"]):
                problems.append(f"summary sentence {k}'s sentences are not a list of numbers")
    return problems


def acceptance_problems(
    record, error: str | None, sentences: list[str], relations: set[str]
) -> list[str]:
    """Why a record is rejected (GATE.md's acceptance), or nothing when it is accepted."""
    if error is not None:
        return [error]
    problems = schema_problems(record, relations)
    if problems:
        return problems
    text = squash(" ".join(sentences))
    ids = {e["id"] for e in record["entities"]}
    for e in record["entities"]:
        if squash(e["name"]) not in text:
            problems.append(f"entity {e['name']!r} is not in the document")
    for f in record["facts"]:
        for end in ("head", "tail"):
            if f[end] not in ids:
                problems.append(f"fact {end} {f[end]!r} is not an entity of the record")
    cited = [n for f in record["facts"] for n in f["sentences"]] + [
        n for s in record["summary"] for n in s["sentences"]
    ]
    problems += [
        f"sentence {n} does not exist" for n in sorted(set(cited)) if not 1 <= n <= len(sentences)
    ]
    return problems


def gold_keys(doc: dict) -> list[set[str]]:
    """Each gold entity's mention names, normalized."""
    return [{normalize_name(m["name"]) for m in v} - {""} for v in doc["vertexSet"]]


def gold_facts(doc: dict) -> list[tuple[int, str, int]]:
    """The document's gold facts as (head, relation id, tail), each once."""
    return list(dict.fromkeys((lab["h"], lab["r"], lab["t"]) for lab in doc["labels"]))


RECALL_TYPES = ("PER", "ORG", "LOC", "TIME")


def cited(sentences: list[str], numbers: list[int]) -> list[list]:
    return [[n, sentences[n - 1]] for n in sorted(set(numbers))]


def score_record(row: dict, doc: dict, rel_ids: dict[str, str]) -> tuple[dict, list[dict]]:
    """A Re-DocRED output's acceptance, its matches with gold and, for an accepted record, its judge items."""
    sentences = generate.document(doc)
    record, error = generate.parse_record(row["text"])
    problems = acceptance_problems(record, error, sentences, set(rel_ids))
    keys = gold_keys(doc)
    facts = gold_facts(doc)
    entity_type = [v[0]["type"] for v in doc["vertexSet"]]
    counted = [g for g, t in enumerate(entity_type) if t in RECALL_TYPES]
    s = {
        "id": row["id"],
        "title": row["title"],
        "accepted": not problems,
        "problems": problems[:10],
        "finish_reason": row["finish_reason"],
        "gold_facts": len(facts),
        "gold_entities": len(counted),
        "recovered": [],
        "matched_entities": [],
        "entities": 0,
        "facts": [],
        "summary": [],
    }
    if problems:
        return s, []
    items = []
    name = {e["id"]: e["name"] for e in record["entities"]}
    names = {normalize_name(n) for n in name.values()} - {""}
    s["entities"] = len(record["entities"])
    s["matched_entities"] = [g for g in counted if keys[g] & names]
    recovered: set[int] = set()
    for f in record["facts"]:
        head, tail = name[f["head"]], name[f["tail"]]
        h, t, r = normalize_name(head), normalize_name(tail), rel_ids[f["relation"]]
        matches = [
            j for j, (gh, gr, gt) in enumerate(facts) if gr == r and h in keys[gh] and t in keys[gt]
        ]
        recovered.update(matches)
        fs = {"gold": bool(matches), "judge": None}
        if not matches:
            content = {
                "sentences": cited(sentences, f["sentences"]),
                "fact": [head, f["relation"], tail],
            }
            fs["judge"] = item_hash("fact", content)
            items.append({"hash": fs["judge"], "kind": "fact", "content": content})
        s["facts"].append(fs)
    s["recovered"] = sorted(recovered)
    for sent in record["summary"]:
        content = {
            "sentences": cited(sentences, sent["sentences"]),
            "summary": sent["text"].strip(),
        }
        h = item_hash("summary", content)
        s["summary"].append({"judge": h})
        items.append({"hash": h, "kind": "summary", "content": content})
    return s, items


# ---- the run -----------------------------------------------------------------------------------------------------


def score(data: Path, sample: dict, outputs: list[dict]) -> tuple[dict, list[dict]]:
    rows = {(r["task"], r["id"]): r for r in outputs}
    papers = json.loads((data / sample["qasper"]["file"]).read_text())
    docs = json.loads((data / sample["redocred"]["file"]).read_text())
    rel_ids = {name: pid for pid, name in json.loads((data / "rel_info.json").read_text()).items()}
    qa, enrich, items = [], [], {}
    for q in sample["qasper"]["questions"]:
        paper = papers[q["paper"]]
        (question,) = [x for x in paper["qas"] if x["question_id"] == q["question_id"]]
        s, item = score_answer(rows[("qa", q["question_id"])], paper, question)
        qa.append(s)
        if item:
            items[item["hash"]] = item
    for d in sample["redocred"]["documents"]:
        s, its = score_record(rows[("enrich", str(d["index"]))], docs[d["index"]], rel_ids)
        enrich.append(s)
        items.update((it["hash"], it) for it in its)
    return {"qa": qa, "enrich": enrich}, list(items.values())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="fetch.sh's directory")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument(
        "--out", type=Path, required=True, help="the record's directory, with outputs.jsonl"
    )
    p.add_argument(
        "--items",
        type=Path,
        required=True,
        help="where the judge's items go (outside the repository)",
    )
    a = p.parse_args()
    outputs = [json.loads(line) for line in (a.out / "outputs.jsonl").read_text().splitlines()]
    scores, items = score(a.data, json.loads(a.sample.read_text()), outputs)
    (a.out / "scores.json").write_text(json.dumps(scores, indent=1, ensure_ascii=False) + "\n")
    a.items.parent.mkdir(parents=True, exist_ok=True)
    with a.items.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    qa, en = scores["qa"], scores["enrich"]
    kinds = {k: sum(it["kind"] == k for it in items) for k in ("qa", "fact", "summary")}
    print(
        f"qa: {len(qa)} questions, {sum(s['abstained'] for s in qa)} abstentions, "
        f"{sum(s['retrieved_gold'] for s in qa if s['stratum'] == 'answerable')} answerable with evidence retrieved; "
        f"enrich: {sum(s['accepted'] for s in en)} of {len(en)} records accepted, "
        f"{sum(len(s['facts']) for s in en)} facts ({sum(f['gold'] for s in en for f in s['facts'])} match gold); "
        f"judge items {kinds}"
    )


if __name__ == "__main__":
    main()
