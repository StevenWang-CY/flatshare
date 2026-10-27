"""The quality gate's screen (GATE.md): asks the served model for the sample's answers and records.

Foreground, QASPER: BM25 over the paper's paragraphs with the question as the query, the top 6 numbered after the
title and abstract; the model answers in one or two short sentences citing them as [n], or says the fixed abstention
sentence. The engine's default sampling (the model's generation config), as the voice agent asks.
Background, Re-DocRED: the document's numbered sentences in, the JSON record of GATE.md out (entities with ids,
facts naming their head and tail by those ids with DocRED's relations and their sentence numbers, a summary of at
most two sentences citing its), at temperature 0, with the record's schema enforced by the engine's structured
outputs.
One request at a time, each item once, with no retry and no best-of; a rerun continues after the items already in
OUT/outputs.jsonl.

  python generate.py --engine unix:/run/user/UID/flatshare/engine.sock --data DIR --sample sample.json --out OUT \
    [--task qa|enrich|both]
DIR is fetch.sh's directory. Self-contained: the standard library and httpx (the engine's venv has it), which only
asking needs, so score.py imports the prompts and parsers from here without it. Writes OUT/outputs.jsonl (one line
an item: its ids, what was retrieved, the raw text, the parsed result and the token counts; no dataset text) and
OUT/generate.json (the run's settings and the engine's model).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
import sys
import time
from collections import Counter
from pathlib import Path

TOP_K = 6
QA_MAX_TOKENS = 256
ENRICH_MAX_TOKENS = 8192
ABSTENTION = "I cannot answer that from these documents."

QA_SYSTEM = (
    "You answer a caller's questions about a document during a phone call. You are given the document's title and "
    "abstract and numbered paragraphs from it. Answer from these documents only, in one or two short, natural "
    "sentences, and cite the numbered paragraphs you used as [n], for example [2] or [1][4], at the end of the "
    "sentence that uses them. You are speaking, so never use lists, markdown or emoji. If these documents do not "
    f"answer the question, say exactly: {ABSTENTION}"
)

ENTITY_TYPES = ("PER", "ORG", "LOC", "TIME", "NUM", "MISC")
ENRICH_EXAMPLE_SENTENCES = (
    "Maria Lopez (born 3 March 1970 in Seville) is an architect.",
    "She studied at the University of Madrid and won the Gold Medal in 2005.",
)
ENRICH_EXAMPLE_RECORD = {
    "entities": [
        {"id": 1, "name": "Maria Lopez", "type": "PER"},
        {"id": 2, "name": "3 March 1970", "type": "TIME"},
        {"id": 3, "name": "Seville", "type": "LOC"},
        {"id": 4, "name": "University of Madrid", "type": "ORG"},
        {"id": 5, "name": "Gold Medal", "type": "MISC"},
        {"id": 6, "name": "2005", "type": "TIME"},
    ],
    "facts": [
        {"head": 1, "relation": "date of birth", "tail": 2, "sentences": [1]},
        {"head": 1, "relation": "place of birth", "tail": 3, "sentences": [1]},
        {"head": 1, "relation": "educated at", "tail": 4, "sentences": [2]},
        {"head": 1, "relation": "award received", "tail": 5, "sentences": [2]},
    ],
    "summary": [
        {"text": "Maria Lopez is an architect born in Seville in 1970.", "sentences": [1]},
        {
            "text": "She studied at the University of Madrid and won the Gold Medal in 2005.",
            "sentences": [2],
        },
    ],
}


def numbered(lines: list[str] | tuple[str, ...]) -> str:
    return "\n".join(f"[{i}] {line}" for i, line in enumerate(lines, 1))


def enrich_system(relations: list[str]) -> str:
    return (
        "You turn documents into records for a search index. You are given a document's numbered sentences. Answer "
        "with one JSON object and nothing else, of this form:\n"
        '{"entities": [{"id": ..., "name": ..., "type": ...}, ...], "facts": [{"head": ..., "relation": ..., "tail": ..., '
        '"sentences": [...]}, ...], "summary": [{"text": ..., "sentences": [...]}, ...]}\n\n'
        "entities: the people, organizations, places, dates, numbers and other named things the document mentions, "
        "each once.\n"
        "- id: the entity's number, 1 for the first, 2 for the next, and so on.\n"
        "- name: exactly as it is written in the sentences, copied character for character.\n"
        "- type: exactly one of PER (a person), ORG (an organization), LOC (a location), TIME (a date or time), NUM "
        "(a number) and MISC (anything else); there is no other type.\n\n"
        "facts: the relations the document states between two of the entities.\n"
        "- head and tail: the ids of two of the entities above; a fact links only entities.\n"
        "- relation: exactly one of the relations listed below, copied as written. There are no other relations: "
        "when what the document says between two entities is not one of them, leave that fact out.\n"
        "- A fact reads as in Wikidata: the head's relation is the tail.\n"
        "- sentences: the numbers of the sentences that state the fact.\n\n"
        "summary: one or two sentences that summarize the document, never more than two, each with the numbers of "
        "the sentences it draws on.\n\n"
        "Relations:\n" + "\n".join(relations) + "\n\n"
        "Example. Sentences:\n"
        f"{numbered(ENRICH_EXAMPLE_SENTENCES)}\n"
        f"Record:\n{json.dumps(ENRICH_EXAMPLE_RECORD)}\n\n"
        "Before you answer, check that every relation is one of the listed relations, every type one of the six, "
        "every head and tail the id of an entity, every name copied exactly from the sentences, and that the summary "
        "has at most two sentences."
    )


def record_schema(relations: list[str]) -> dict:
    """The record's JSON schema, which the engine's structured outputs enforce (vLLM's response_format): what
    GATE.md's acceptance calls matching the schema. A fact names its head and tail by the ids of the record's
    entities. Unique ids, names occurring verbatim, facts' heads and tails being entities of the record and cited
    sentences existing are left to the model, and score.py checks them."""
    numbers = {"type": "array", "items": {"type": "integer"}, "minItems": 1}
    text = {"type": "string", "minLength": 1}

    def obj(props: dict) -> dict:
        return {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        }

    number = {"type": "integer"}
    entity = obj(
        {"id": number, "name": text, "type": {"type": "string", "enum": list(ENTITY_TYPES)}}
    )
    fact = obj({"head": number, "relation": {"type": "string", "enum": list(relations)}, "tail": number,
                "sentences": numbers})  # fmt: skip
    sentence = obj({"text": text, "sentences": numbers})
    return obj(
        {
            "entities": {"type": "array", "items": entity},
            "facts": {"type": "array", "items": fact},
            "summary": {"type": "array", "items": sentence, "minItems": 1, "maxItems": 2},
        }
    )


# ---- QASPER: paragraphs and BM25 --------------------------------------------------------------------------------

WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def paragraphs(paper: dict) -> list[str]:
    """The paper's paragraphs in order, every section's, empty ones included so that an index names the same
    paragraph in the data; retrieval skips the empty ones."""
    return [para for section in paper["full_text"] for para in section["paragraphs"]]


class BM25:
    """Okapi BM25 (k1 1.2, b 0.75; Lucene's idf, log(1 + (N - df + 0.5) / (df + 0.5))) over a few documents, each a
    paragraph, with lowercased word tokens and no stemming or stop words. A query's distinct terms count once."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(words(d)) for d in docs]
        self.length = [sum(c.values()) for c in self.tf]
        self.avgdl = sum(self.length) / len(docs) if docs else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in dict.fromkeys(words(query)) if t in self.idf]
        out = []
        for tf, dl in zip(self.tf, self.length, strict=True):
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            out.append(
                sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            )
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """The k best documents as (index, score), best first, ties to the earlier document."""
        s = self.scores(query)
        return [(i, s[i]) for i in sorted(range(len(s)), key=lambda i: (-s[i], i))[:k]]


def retrieve(paper: dict, question: str, k: int = TOP_K) -> list[tuple[int, float]]:
    """The k paragraphs BM25 ranks best for the question, as (index into paragraphs(paper), score), best first."""
    paras = paragraphs(paper)
    kept = [i for i, p in enumerate(paras) if p.strip()]
    bm25 = BM25([paras[i] for i in kept])
    return [(kept[j], score) for j, score in bm25.top(question, k)]


def qa_messages(paper: dict, question: str, retrieved: list[int]) -> list[dict]:
    paras = paragraphs(paper)
    given = "\n\n".join(f"[{n}] {paras[i]}" for n, i in enumerate(retrieved, 1))
    user = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}\n\n{given}\n\nQuestion: {question}"
    return [{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": user}]


CITATION = re.compile(r"\[\s*(\d+(?:\s*(?:[,;]|[-\u2013]|and)\s*\d+)*)\s*\]")


def parse_citations(text: str) -> list[int]:
    """The paragraph numbers an answer cites, in order, each once: [2], [1][4], [1, 4], [1-3]."""
    out: list[int] = []
    for m in CITATION.finditer(text):
        for part in re.split(r"\s*(?:[,;]|and)\s*", m.group(1)):
            ends = [int(x) for x in re.split(r"\s*[-\u2013]\s*", part)]
            span = (
                range(ends[0], ends[-1] + 1)
                if len(ends) == 2 and 0 < ends[1] - ends[0] < 10
                else ends
            )
            out.extend(n for n in span if n not in out)
    return out


def strip_citations(text: str) -> str:
    """The answer without its citation markers, as a listener hears it."""
    t = CITATION.sub("", text)
    t = re.sub(r"[ \t]+([.,;:!?])", r"\1", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


# ---- Re-DocRED: the document and the record ---------------------------------------------------------------------

NO_SPACE_BEFORE = {",", ".", ";", ":", "!", "?", "%", ")", "]", "}", "\u201d", "\u2019", "...", "…"}
NO_SPACE_BEFORE |= {"'s", "'re", "'ll", "'m", "'ve", "'d", "n't", "'em", "'"}
NO_SPACE_AFTER = {"(", "[", "{", "$", "£", "\u201c", "\u2018", "#", "(;"}


def detokenize(tokens: list[str]) -> str:
    """A Re-DocRED sentence's tokens as text: no space before closing punctuation and clitics or after opening
    brackets and currency signs, hyphens joined to their neighbours, straight double quotes opening and closing in
    turn, whitespace tokens dropped. The gold mentions' names are matched after normalization, so spacing never
    decides a match."""
    out = ""
    glue = True  # no space before the next token
    quotes = 0
    for tok in tokens:
        if not tok.strip():
            continue
        if tok == '"':
            opening = quotes % 2 == 0
            quotes += 1
            out += tok if glue or not opening else " " + tok
            glue = opening
            continue
        if tok == "-":
            out += tok
            glue = True
            continue
        out += tok if glue or tok in NO_SPACE_BEFORE else " " + tok
        glue = tok in NO_SPACE_AFTER or tok.endswith("$")  # $, US$, A$
    return out


def document(doc: dict) -> list[str]:
    """A Re-DocRED document's sentences as text, numbered from 1 in the prompt."""
    return [detokenize(s) for s in doc["sents"]]


def enrich_messages(sentences: list[str], relations: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": enrich_system(relations)},
        {"role": "user", "content": f"Sentences:\n{numbered(sentences)}\nRecord:"},
    ]


FENCE = re.compile(r"^```[A-Za-z]*\s*\n?(.*?)\n?```$", re.S)


def parse_record(text: str) -> tuple[object, str | None]:
    """The JSON value of a reply, optionally inside one markdown code fence, and None; or None and why not."""
    t = text.strip()
    m = FENCE.match(t)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t), None
    except json.JSONDecodeError as e:
        return None, f"not JSON: {e.msg} at {e.pos} of {len(t)}"


# ---- the run -----------------------------------------------------------------------------------------------------


def relation_names(rel_info: dict) -> list[str]:
    """DocRED's 96 relation names in rel_info.json's order."""
    return list(rel_info.values())


def digest(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def ask(client, model: str, messages: list[dict], extra: dict) -> dict:
    t0 = time.monotonic()
    r = client.post(
        "/v1/chat/completions", json={"model": model, "messages": messages, **extra}, timeout=900
    )
    r.raise_for_status()
    body = r.json()
    choice = body["choices"][0]
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "completion_tokens": body["usage"]["completion_tokens"],
        "seconds": round(time.monotonic() - t0, 3),
    }


def qa_request() -> dict:
    """Q&A's sampling: the engine's default (the model's generation config), as the voice agent asks; only a cap."""
    return {"max_tokens": QA_MAX_TOKENS}


def enrich_request(relations: list[str]) -> dict:
    """Enrichment's sampling and form: temperature 0, the record's schema enforced by the engine."""
    return {
        "max_tokens": ENRICH_MAX_TOKENS,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "record", "schema": record_schema(relations), "strict": True},
        },
    }


def qa_items(data: Path, sample: dict):
    papers = json.loads((data / sample["qasper"]["file"]).read_text())
    for q in sample["qasper"]["questions"]:
        paper = papers[q["paper"]]
        (question,) = [x["question"] for x in paper["qas"] if x["question_id"] == q["question_id"]]
        yield ("qa", q["question_id"]), q, paper, question


def enrich_items(data: Path, sample: dict):
    docs = json.loads((data / sample["redocred"]["file"]).read_text())
    for d in sample["redocred"]["documents"]:
        doc = docs[d["index"]]
        if doc["title"] != d["title"]:
            raise SystemExit(
                f"document {d['index']} is {doc['title']!r}, the sample says {d['title']!r}"
            )
        yield ("enrich", str(d["index"])), d, doc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--engine", required=True, help="unix:PATH of the engine's socket, or http://HOST:PORT"
    )
    p.add_argument("--data", type=Path, required=True, help="fetch.sh's directory")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", choices=("qa", "enrich", "both"), default="both")
    a = p.parse_args()

    import httpx  # only asking needs it

    if a.engine.startswith("unix:"):
        client = httpx.Client(
            transport=httpx.HTTPTransport(uds=a.engine[5:]), base_url="http://engine"
        )
    else:
        client = httpx.Client(base_url=a.engine)
    models = client.get("/v1/models").json()["data"]
    model = models[0]["id"]
    sample = json.loads(a.sample.read_text())
    relations = relation_names(json.loads((a.data / "rel_info.json").read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / "outputs.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            done.add((row["task"], row["id"]))
    meta_path = a.out / "generate.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": []}
    meta.update(
        {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sample_sha256": hashlib.sha256(a.sample.read_bytes()).hexdigest(),
            "model": model,
            "max_model_len": models[0].get("max_model_len"),
            "qa": {"top_k": TOP_K, "max_tokens": QA_MAX_TOKENS, "sampling": "the engine's default"},
            "enrich": {"max_tokens": ENRICH_MAX_TOKENS, "temperature": 0},
        }
    )
    try:
        meta["engine_version"] = client.get("/version").json().get("version")
    except Exception:  # an engine without /version
        meta["engine_version"] = None
    run = {
        "task": a.task,
        "host": socket.gethostname(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta["runs"].append(run)
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")

    n = 0
    with out.open("a") as f:
        if a.task in ("qa", "both"):
            for key, q, paper, question in qa_items(a.data, sample):
                if key in done:
                    continue
                ranked = retrieve(paper, question)
                retrieved = [i for i, _ in ranked]
                messages = qa_messages(paper, question, retrieved)
                reply = ask(client, model, messages, qa_request())
                row = {
                    "task": "qa",
                    "id": q["question_id"],
                    "paper": q["paper"],
                    "stratum": q["stratum"],
                    "retrieved": retrieved,
                    "bm25": [round(s, 4) for _, s in ranked],
                    "prompt_sha256": digest(messages),
                    **reply,
                    "parsed": {"citations": parse_citations(reply["text"])},
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
                print(
                    f"qa {n}: {q['question_id'][:8]} {reply['completion_tokens']} tokens",
                    file=sys.stderr,
                )
        if a.task in ("enrich", "both"):
            for key, d, doc in enrich_items(a.data, sample):
                if key in done:
                    continue
                sentences = document(doc)
                messages = enrich_messages(sentences, relations)
                reply = ask(client, model, messages, enrich_request(relations))
                record, error = parse_record(reply["text"])
                row = {
                    "task": "enrich",
                    "id": str(d["index"]),
                    "title": d["title"],
                    "sentences": len(sentences),
                    "prompt_sha256": digest(messages),
                    **reply,
                    "parsed": {"record": record, "error": error},
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
                print(
                    f"enrich {n}: {d['index']} {reply['completion_tokens']} tokens", file=sys.stderr
                )
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run["items"] = n
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{n} items written to {out}")


if __name__ == "__main__":
    main()
