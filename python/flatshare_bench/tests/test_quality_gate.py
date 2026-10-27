"""The quality gate's screen (benchmarks/quality, GATE.md) on small hand-made fixtures: BM25, the abstention rule,
citation scoring, the speakable check, Re-DocRED's acceptance and matching, the judge's transport and batches,
Wilson intervals and the verdict's logic, and the sample's determinism."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
QUALITY = REPO / "benchmarks" / "quality"


def _load(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, QUALITY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate = _load("generate")
score = _load("score")
judge = _load("judge")
gate = _load("gate")
sample = _load("sample")

# ---- QASPER ------------------------------------------------------------------------------------------------------


def test_bm25_ranks_by_okapi_scores() -> None:
    bm25 = generate.BM25(["the cat sat", "the dog sat on the mat", "cats and dogs"])
    scores = bm25.scores("dog mat dog")
    # N 3, df 1 for both terms: idf log(1 + 2.5 / 1.5); the second document has 6 of the average 4 words.
    idf = 0.9808292530117263
    norm = 1.2 * (1 - 0.75 + 0.75 * 6 / 4)
    assert scores[1] == pytest.approx(2 * idf * 2.2 / (1 + norm))
    assert scores[0] == scores[2] == 0
    # A repeated query term counts once; ties go to the earlier document.
    assert bm25.top("dog mat dog", 2) == [(1, pytest.approx(1.6285466842458853)), (0, 0.0)]
    assert bm25.top("Cat", 3)[0][0] == 0  # lowercased words, no stemming: "cats" is not "cat"


def test_retrieval_skips_empty_paragraphs_and_names_the_datas() -> None:
    paper = {
        "title": "T",
        "abstract": "A",
        "full_text": [
            {"section_name": "Intro", "paragraphs": ["We train a model.", ""]},
            {
                "section_name": "Data",
                "paragraphs": ["The dataset has 3,044 sentences.", "Results follow."],
            },
        ],
    }
    ranked = generate.retrieve(paper, "How many sentences does the dataset have?", k=6)
    assert [i for i, _ in ranked] == [
        2,
        0,
        3,
    ]  # paragraph 1 is empty; indices are into all paragraphs
    msgs = generate.qa_messages(paper, "Q?", [2, 0])
    assert msgs[0]["content"] == generate.QA_SYSTEM
    assert msgs[1]["content"] == (
        "Title: T\n\nAbstract: A\n\n[1] The dataset has 3,044 sentences.\n\n[2] We train a model.\n\nQuestion: Q?"
    )
    assert generate.ABSTENTION in generate.QA_SYSTEM


def test_abstention_is_the_fixed_sentence_alone() -> None:
    for text in (
        "I cannot answer that from these documents.",
        "i cannot answer that from these documents",
        "  I cannot answer that from these documents [2].",
        "I cannot answer that, from these documents!",
    ):
        assert score.is_abstention(text), text
    for text in (
        "I can't answer that from these documents.",
        "I cannot answer that from these documents, but they use BERT [1].",
        "The documents do not say. I cannot answer that from these documents.",
        "They use BERT [1].",
    ):
        assert not score.is_abstention(text), text


def test_citations_are_parsed_and_stripped() -> None:
    assert generate.parse_citations("BERT [2] and ELMo [1][4].") == [2, 1, 4]
    assert generate.parse_citations("Both [1, 3], again [3] and [2-4]; [5 and 6].") == [
        1,
        3,
        2,
        4,
        5,
        6,
    ]
    assert generate.parse_citations("In 2019 they had [ten] runs.") == []
    assert (
        generate.strip_citations("They use BERT [2], then ELMo [1][4].")
        == "They use BERT, then ELMo."
    )


def _paper_and_question() -> tuple[dict, dict]:
    paper = {
        "title": "T",
        "abstract": "A",
        "full_text": [
            {"section_name": "S", "paragraphs": ["p0", "p1 evidence", "", "p3", "p4 evidence"]}
        ],
        "qas": [],
    }
    answers = [
        {"answer": {"unanswerable": False, "extractive_spans": ["3,044 sentences "], "yes_no": None,
                    "free_form_answer": "", "evidence": ["p1  evidence", "FLOAT SELECTED: Table 1"]}},
        {"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": None,
                    "free_form_answer": "About 3k sentences", "evidence": ["p4 evidence", "S"]}},
        {"answer": {"unanswerable": True, "extractive_spans": [], "yes_no": None,
                    "free_form_answer": "", "evidence": []}},
        {"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": False,
                    "free_form_answer": "", "evidence": []}},
    ]  # fmt: skip
    return paper, {"question_id": "q1", "question": "How big is it? ", "answers": answers}


def test_citation_scoring_against_evidence() -> None:
    paper, question = _paper_and_question()
    # Evidence names a paragraph after whitespace is squashed; tables and section names name none.
    assert score.gold_paragraphs(paper, question) == {1, 4}
    assert score.references(question) == [
        'Spans quoted from the paper: "3,044 sentences"',
        "About 3k sentences",
        "No",
    ]
    row = {
        "id": "q1",
        "paper": "p",
        "stratum": "answerable",
        "retrieved": [3, 4, 0],
        "finish_reason": "stop",
    }
    s, item = score.score_answer({**row, "text": "It has 3,044 sentences [2]."}, paper, question)
    assert (
        s["supporting_citation"],
        s["invalid_citations"],
        s["retrieved_gold"],
        s["abstained"],
    ) == (True, 0, True, False)
    assert s["speakable"] and item["content"]["answer"] == "It has 3,044 sentences."
    assert item["hash"] == s["judge"] == score.item_hash("qa", item["content"])
    # A paragraph that was not given is an invalid citation; one that no annotator marked does not support.
    s, _ = score.score_answer({**row, "text": "It has 3,044 sentences [1][7]."}, paper, question)
    assert (s["supporting_citation"], s["invalid_citations"], s["citations"]) == (False, 1, [1, 7])
    s, item = score.score_answer({**row, "text": generate.ABSTENTION}, paper, question)
    assert s["abstained"] and item is None and s["judge"] is None
    s, item = score.score_answer(
        {**row, "stratum": "unanswerable", "text": "Yes [1]."}, paper, question
    )
    assert item is None and not s["abstained"]  # wrong without the judge
    s, _ = score.score_answer({**row, "retrieved": [0, 3], "text": "Yes [1]."}, paper, question)
    assert not s["retrieved_gold"]


def test_speakable_counts_sentences_and_finds_markdown() -> None:
    assert score.count_sentences("They use BERT [2]. It reaches a 0.874 kappa [1].") == 2
    assert (
        score.count_sentences("Smith et al. report it, e.g. on U.S. data, in Fig. 3 by J. Smith.")
        == 1
    )
    assert score.count_sentences("It works. It is fast. It is small.") == 3
    assert score.count_sentences("It is 3.5 points better... and smaller") == 1
    assert score.count_sentences('He said "yes." Then he left') == 2
    assert score.count_sentences("") == 0
    for text in (
        "- one\n- two",
        "1. one\n2. two",
        "It is **big**.",
        "Use `x`.",
        "# Answer",
        "It is *big*.",
    ):
        assert score.has_markdown(text), text
    for text in ("It costs 2 * 3 dollars.", "They use 3-4 layers [1].", "Yes, it does."):
        assert not score.has_markdown(text), text


# ---- Re-DocRED ---------------------------------------------------------------------------------------------------

DOC = {
    "title": "Maria Lopez",
    "sents": [
        [
            "Maria",
            "Lopez",
            "(",
            "born",
            "3",
            "March",
            "1970",
            ")",
            "was",
            "born",
            "in",
            "Seville",
            ".",
        ],
        [
            "Lopez",
            "won",
            "the",
            '"',
            "Gold",
            "Medal",
            '"',
            "in",
            "2005",
            ",",
            "Seville",
            "'s",
            "first",
            ".",
        ],
    ],
    "vertexSet": [
        [{"name": "Maria Lopez", "type": "PER"}, {"name": "Lopez", "type": "PER"}],
        [{"name": "3 March 1970", "type": "TIME"}],
        [{"name": "Seville", "type": "LOC"}, {"name": "Seville", "type": "LOC"}],
        [{"name": "Gold Medal", "type": "MISC"}],
        [{"name": "2005", "type": "TIME"}],
    ],
    "labels": [
        {"h": 0, "r": "P19", "t": 2, "evidence": [0]},
        {"h": 0, "r": "P569", "t": 1, "evidence": [0]},
        {"h": 0, "r": "P166", "t": 3, "evidence": [1]},
        {"h": 0, "r": "P166", "t": 3, "evidence": []},
    ],
}
REL_IDS = {
    "place of birth": "P19",
    "date of birth": "P569",
    "award received": "P166",
    "country": "P17",
}


ENTITIES = [
    {"id": 1, "name": "Lopez", "type": "PER"},
    {"id": 2, "name": "Seville", "type": "LOC"},
    {"id": 3, "name": "Gold Medal", "type": "MISC"},
    {"id": 4, "name": "2005", "type": "TIME"},
]


def _fact(
    head: int = 1, relation: str = "country", tail: int = 2, sentences: object = (1,)
) -> dict:
    return {"head": head, "relation": relation, "tail": tail, "sentences": list(sentences)}


def _record(**change) -> dict:
    record = {
        "entities": ENTITIES,
        "facts": [
            _fact(1, "place of birth", 2, [1]),
            _fact(1, "award received", 3, [2, 2]),
            _fact(3, "country", 2, [2]),
        ],
        "summary": [
            {"text": "Maria Lopez, born in Seville, won the Gold Medal.", "sentences": [1, 2]}
        ],
    }
    record.update(change)
    return record


def _scored(text: str) -> tuple[dict, list[dict]]:
    return score.score_record(
        {"id": "0", "title": DOC["title"], "text": text, "finish_reason": "stop"}, DOC, REL_IDS
    )


def test_the_document_is_detokenized() -> None:
    assert generate.document(DOC) == [
        "Maria Lopez (born 3 March 1970) was born in Seville.",
        'Lopez won the "Gold Medal" in 2005, Seville\'s first.',
    ]
    assert generate.detokenize(
        ["a", "two", "-", "year", "deal", "(;", ";", ")", "\xa0", "US$", "5", "%"]
    ) == ("a two-year deal (;;) US$5%")
    msgs = generate.enrich_messages(generate.document(DOC), list(REL_IDS))
    assert msgs[1]["content"].startswith("Sentences:\n[1] Maria Lopez (born") and msgs[1][
        "content"
    ].endswith("Record:")
    assert "\nplace of birth\ndate of birth\naward received\ncountry\n" in msgs[0]["content"]


def test_the_requests_settings() -> None:
    # Q&A: the engine's default sampling (GATE.md), only a cap; enrichment: temperature 0 and the enforced schema.
    assert generate.qa_request() == {"max_tokens": 256}
    req = generate.enrich_request(list(REL_IDS))
    assert (req["temperature"], req["max_tokens"]) == (0, 8192)
    form = req["response_format"]
    assert form["type"] == "json_schema" and form["json_schema"]["strict"] is True
    schema = form["json_schema"]["schema"]
    assert schema == generate.record_schema(list(REL_IDS))
    assert (
        schema["required"] == ["entities", "facts", "summary"]
        and schema["additionalProperties"] is False
    )
    entity = schema["properties"]["entities"]["items"]
    fact = schema["properties"]["facts"]["items"]
    summary = schema["properties"]["summary"]
    assert entity["properties"]["type"]["enum"] == ["PER", "ORG", "LOC", "TIME", "NUM", "MISC"]
    assert fact["properties"]["relation"]["enum"] == list(REL_IDS)
    assert fact["properties"]["head"] == fact["properties"]["tail"] == {"type": "integer"}
    assert fact["properties"]["sentences"]["minItems"] == 1
    assert (summary["minItems"], summary["maxItems"]) == (1, 2)
    # The prompt's example is a record that this schema and the acceptance checks take.
    example = generate.ENRICH_EXAMPLE_RECORD
    relations = {f["relation"] for f in example["facts"]}
    assert (
        score.acceptance_problems(example, None, list(generate.ENRICH_EXAMPLE_SENTENCES), relations)
        == []
    )


def test_an_accepted_record_and_its_matches() -> None:
    s, items = _scored(json.dumps(_record()))
    assert s["accepted"] and s["problems"] == []
    # Two facts match gold (the duplicate gold label counts once); the third goes to the judge, as does the summary.
    assert [f["gold"] for f in s["facts"]] == [True, True, False]
    assert s["gold_facts"] == 3 and s["recovered"] == [0, 2]
    assert [it["kind"] for it in items] == ["fact", "summary"]
    assert items[0]["content"] == {
        "sentences": [[2, DOC_SENTENCES[1]]],
        "fact": ["Gold Medal", "country", "Seville"],
    }
    assert items[1]["content"]["sentences"] == [[1, DOC_SENTENCES[0]], [2, DOC_SENTENCES[1]]]
    # Entity recall counts PER, ORG, LOC and TIME: "Lopez" matches entity 0 by its second mention; 1 is missed.
    assert s["gold_entities"] == 4 and s["matched_entities"] == [0, 2, 4]
    # Ids need not run 1..n; facts name entities by them. Fenced JSON parses; names match after case, whitespace
    # and punctuation are dropped.
    renumbered = [{**e, "id": e["id"] * 10} for e in ENTITIES]
    facts = [{**f, "head": f["head"] * 10, "tail": f["tail"] * 10} for f in _record()["facts"]]
    s, items = _scored(json.dumps(_record(entities=renumbered, facts=facts)))
    assert s["accepted"] and [f["gold"] for f in s["facts"]] == [True, True, False]
    assert items[0]["content"]["fact"] == ["Gold Medal", "country", "Seville"]
    s, _ = _scored("```json\n" + json.dumps(_record()) + "\n```")
    assert s["accepted"]
    assert score.normalize_name(" U.S.  Army ") == score.normalize_name("US army") == "usarmy"
    assert score.normalize_name("November 29, 2007") == score.normalize_name("November 29 , 2007")


DOC_SENTENCES = generate.document(DOC)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("{not json", "not JSON"),
        ("Here it is: {}", "not JSON"),
        (json.dumps([]), "the record is not an object"),
        (json.dumps({**_record(), "notes": "x"}), "keys"),
        (json.dumps(_record(entities=[{"id": 1, "name": "Lopez", "type": "PERSON"}])), "entity 0 has type 'PERSON'"),
        (json.dumps(_record(entities=[{"id": 1, "name": "Lopez"}])), "entity 0 is not an id, a name and a type"),
        (json.dumps(_record(entities=[{"id": "1", "name": "Lopez", "type": "PER"}])), "entity 0's id is not a number"),
        (json.dumps(_record(entities=[*ENTITIES, {"id": 4, "name": "Seville", "type": "LOC"}])), "entity ids repeat"),
        (json.dumps(_record(facts=[_fact(relation="born in")])), "fact 0 has relation 'born in'"),
        (json.dumps(_record(facts=[_fact(tail="Seville")])), "fact 0's head or tail is not an id"),
        (json.dumps(_record(facts=[_fact(sentences=[])])), "fact 0's sentences are not a list of numbers"),
        (json.dumps(_record(facts=[_fact(sentences=[True])])), "fact 0's sentences are not a list of numbers"),
        (json.dumps(_record(summary=[])), "the summary is not one or two sentences"),
        (json.dumps(_record(summary=[{"text": "a", "sentences": [1]}] * 3)), "the summary is not one or two"),
        (json.dumps(_record(entities=[*ENTITIES, {"id": 5, "name": "María Lopez", "type": "PER"}])),
         "entity 'María Lopez' is not in the document"),
        (json.dumps(_record(entities=[*ENTITIES, {"id": 5, "name": "lopez", "type": "PER"}])),
         "entity 'lopez' is not in the document"),
        (json.dumps(_record(facts=[_fact(head=0)])), "fact head 0 is not an entity of the record"),
        (json.dumps(_record(facts=[_fact(tail=9)])), "fact tail 9 is not an entity of the record"),
        (json.dumps(_record(summary=[{"text": "Lopez won.", "sentences": [3]}])), "sentence 3 does not exist"),
        (json.dumps(_record(facts=[_fact(sentences=[0])])), "sentence 0 does not exist"),
    ],
)  # fmt: skip
def test_rejected_records(text: str, problem: str) -> None:
    s, items = _scored(text)
    assert not s["accepted"] and items == []
    assert any(p.startswith(problem) for p in s["problems"]), s["problems"]
    # A rejected record counts as none of its facts and none of its entities.
    assert (
        s["recovered"] == []
        and s["matched_entities"] == []
        and s["gold_facts"] == 3
        and s["gold_entities"] == 4
    )


def test_a_name_with_other_whitespace_is_verbatim() -> None:
    doc = {
        **DOC,
        "sents": [["It", "ends", "at", "11:59\xa0p.m.", "."]],
        "vertexSet": [],
        "labels": [],
    }
    record = {"entities": [{"id": 1, "name": "11:59 p.m.", "type": "TIME"}], "facts": [],
              "summary": [{"text": "It ends at midnight.", "sentences": [1]}]}  # fmt: skip
    s, _ = score.score_record(
        {"id": "0", "title": "x", "text": json.dumps(record), "finish_reason": "stop"}, doc, {}
    )
    assert s["accepted"], s["problems"]


# ---- the judge ---------------------------------------------------------------------------------------------------


def test_the_judges_transport() -> None:
    cmd = judge.command("SYSTEM", {"type": "object"}, "PROMPT")
    assert cmd[:-5] == [
        "claude", "--print", "--model", "opus", "--output-format", "json", "--tools", "", "--setting-sources", "",
        "--strict-mcp-config", "--no-session-persistence", "--exclude-dynamic-system-prompt-sections",
    ]  # fmt: skip
    assert cmd[-5:] == [
        "--system-prompt",
        "SYSTEM",
        "--json-schema",
        '{"type": "object"}',
        "PROMPT",
    ]
    env = judge.clean_env()
    assert sorted(env) == [
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY", "CLAUDE_CODE_DISABLE_CLAUDE_MDS", "HOME", "PATH", "USER",
    ]  # fmt: skip
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    # The record keeps the login as its subscription and a hash, never the email.
    login = {"email": "someone@example.com", "subscription": "max"}
    assert (
        judge.fingerprint(login) == "max:" + hashlib.sha256(b"someone@example.com").hexdigest()[:12]
    )
    assert "@" not in judge.fingerprint(login)
    opus = {"modelUsage": {"claude-opus-5-5": {"outputTokens": 12}}, "usage": {"output_tokens": 12}}
    assert judge.opus_check(opus) == ("claude-opus-5-5", 12)
    for bad in (
        {"modelUsage": {"claude-sonnet-5": {"outputTokens": 12}}, "usage": {"output_tokens": 12}},
        {"modelUsage": {"claude-opus-5-5": {"outputTokens": 0}}, "usage": {"output_tokens": 0}},
        {"modelUsage": {"claude-opus-5-5": {"outputTokens": 9}, "claude-haiku-4-5": {"outputTokens": 3}},
         "usage": {"output_tokens": 12}},
    ):  # fmt: skip
        with pytest.raises(ValueError):
            judge.opus_check(bad)


def test_the_judge_sees_only_what_a_rater_needs() -> None:
    qa = {"question": "How big?", "references": ["3k", "No"], "answer": "It is 3k."}
    text = judge.prompt("qa", [qa])
    assert text == (
        "Rule each of these 1 items.\n\nItem 1\nQuestion: How big?\nAnnotators' answers:\n- 3k\n- No\n"
        "Candidate answer: It is 3k."
    )
    fact = {"sentences": [[2, "Lopez won."]], "fact": ["Lopez", "award received", "Gold Medal"]}
    assert judge.render("fact", 3, fact) == (
        "Item 3\nCited sentences:\n[2] Lopez won.\nFact:\n  head: Lopez\n  relation: award received\n  tail: Gold Medal"
    )
    for kind, (rater, labels, positive) in judge.RATERS.items():
        assert positive in labels and positive == gate.POSITIVE[kind]
        assert "Qwen" not in rater and judge.schema(kind)["properties"]["rulings"]["items"][
            "properties"
        ]["ruling"]["enum"] == list(labels)


def test_batches_and_the_reask_sample() -> None:
    items = [
        {"hash": f"{i:03d}", "kind": "fact" if i % 3 else "qa", "content": {}} for i in range(95)
    ]
    batches = judge.batches_of(items, 2026)
    assert batches == judge.batches_of(items, 2026)
    assert Counter(k for k, _ in batches) == {"qa": 2, "fact": 4}
    assert all(15 <= len(b) <= 20 for _, b in batches)
    assert sorted(it["hash"] for _, b in batches for it in b) == sorted(it["hash"] for it in items)
    again = judge.reask_sample([it["hash"] for it in items])
    assert len(again) == 10 and again == judge.reask_sample([it["hash"] for it in reversed(items)])
    rows = [
        {"item": "a", "kind": "qa", "pass": "first", "ruling": "partial"},
        {"item": "a", "kind": "qa", "pass": "reask", "ruling": "incorrect"},
        {"item": "b", "kind": "fact", "pass": "first", "ruling": "yes"},
        {"item": "b", "kind": "fact", "pass": "reask", "ruling": "no"},
    ]
    assert judge.agreement(rows) == {"reasked": 2, "agree": 1, "agree_label": 0}


# ---- the verdict -------------------------------------------------------------------------------------------------


def test_wilson_intervals() -> None:
    assert gate.wilson(8, 10) == pytest.approx((0.4901624715366418, 0.9433178485456248))
    assert gate.wilson(0, 10) == (0.0, pytest.approx(0.2775327998628892))
    assert gate.wilson(10, 10) == (pytest.approx(0.7224672001371107), 1.0)
    assert gate.wilson(160, 200) == pytest.approx((0.7391448134346212, 0.8495479907390189))
    assert gate.wilson(0, 0) is None
    c = gate.criterion("qa", "x", 80, 100, 0.80)
    assert c["pass"] and c["straddles"] and c["value"] == 0.8
    assert not gate.criterion("qa", "x", 0, 0, 0.5)["pass"]


def _toy() -> tuple[dict, list[dict]]:
    qa = []
    for i in range(20):
        qa.append({"id": f"a{i}", "stratum": "answerable", "abstained": False, "citations": [1], "invalid_citations": 0,
                   "supporting_citation": True, "retrieved_gold": True, "speakable": True, "judge": f"q{i}"})  # fmt: skip
    for i in range(10):
        qa.append({"id": f"u{i}", "stratum": "unanswerable", "abstained": True, "citations": [], "invalid_citations": 0,
                   "supporting_citation": False, "retrieved_gold": False, "speakable": True, "judge": None})  # fmt: skip
    enrich = [
        {"accepted": True, "facts": [{"gold": True, "judge": None}] * 4 + [{"gold": False, "judge": "f"}],
         "summary": [{"judge": "s"}], "recovered": [0, 1, 2, 3], "gold_facts": 10, "matched_entities": [0, 1, 2],
         "gold_entities": 3}
    ] * 20  # fmt: skip
    rows = [
        {"item": f"q{i}", "kind": "qa", "pass": "first", "ruling": "correct"} for i in range(20)
    ]
    rows += [{"item": "f", "kind": "fact", "pass": "first", "ruling": "yes"},
             {"item": "s", "kind": "summary", "pass": "first", "ruling": "yes"}]  # fmt: skip
    rows += [{"item": "q0", "kind": "qa", "pass": "reask", "ruling": "correct"},
             {"item": "q1", "kind": "qa", "pass": "reask", "ruling": "partial"},
             {"item": "f", "kind": "fact", "pass": "reask", "ruling": "yes"}]  # fmt: skip
    return {"qa": qa, "enrich": enrich}, rows


def test_the_verdicts_logic() -> None:
    scores, rows = _toy()
    # 2 of 3 re-asked rulings agree: inconclusive, whatever the criteria say.
    v = gate.verdict(scores, rows)
    assert v["verdict"] == "INCONCLUSIVE" and v["failed"] == [] and v["judge"]["k"] == 2
    rows[-2]["ruling"] = "correct"
    v = gate.verdict(scores, rows)
    assert (
        v["verdict"] == "PASS"
        and {c["name"]: c["value"] for c in v["criteria"]}["fact precision"] == 1.0
    )
    rows[2]["ruling"] = (
        "partial"  # one wrong answer of twenty: precision 0.95, coverage 0.95, citations 1.0
    )
    rejected = {
        "accepted": False,
        "facts": [],
        "summary": [],
        "recovered": [],
        "matched_entities": [],
    }
    scores["enrich"] = [{**d, **rejected} for d in scores["enrich"][:2]] + scores["enrich"][2:]
    v = gate.verdict(scores, rows)
    assert v["verdict"] == "FAIL" and v["failed"] == ["enrich accepted records"]
    got = {c["name"]: (c["k"], c["n"]) for c in v["criteria"]}
    assert (
        got["precision"] == (19, 20)
        and got["citations"] == (19, 19)
        and got["accepted records"] == (18, 20)
    )
    assert got["fact recall"] == (72, 200) and got["entity recall"] == (54, 60)
    with pytest.raises(SystemExit):
        gate.verdict(scores, [r for r in rows if r["item"] != "f"])


# ---- the sample --------------------------------------------------------------------------------------------------


def _papers(n: int = 40) -> dict:
    papers = {}
    for p in range(n):
        qas = []
        for q in range(3):
            unanswerable = (p + q) % 4 == 0
            answers = [
                {"answer": {"unanswerable": unanswerable}},
                {"answer": {"unanswerable": unanswerable or q == 1}},
            ]
            qas.append({"question_id": f"{p:02d}{q}", "answers": answers})
        papers[f"paper{p}"] = {"qas": qas}
    return papers


def test_the_sample_is_deterministic_and_capped() -> None:
    papers = _papers()
    got = sample.draw_questions(papers, 20, 10, seed=2026)
    assert got == sample.draw_questions(dict(reversed(papers.items())), 20, 10, seed=2026)
    assert got != sample.draw_questions(papers, 20, 10, seed=2027)
    assert Counter(q["stratum"] for q in got) == {"answerable": 20, "unanswerable": 10}
    assert max(Counter(q["paper"] for q in got).values()) <= 2
    # Answered by one annotator of two is answerable; marked unanswerable by both is not.
    assert ("paper1", "011") in sample.strata(papers)[0]
    with pytest.raises(ValueError):
        sample.draw_questions(papers, 20, 40)
    docs = [{"title": f"d{i}"} for i in range(50)]
    assert sample.draw_documents(docs, 10) == sample.draw_documents(docs, 10)
    assert [d["index"] for d in sample.draw_documents(docs, 10)] == sorted(
        d["index"] for d in sample.draw_documents(docs, 10)
    )


def test_the_committed_sample() -> None:
    s = json.loads((QUALITY / "sample.json").read_text())
    qs = s["qasper"]["questions"]
    assert s["seed"] == sample.SEED == 2026
    assert Counter(q["stratum"] for q in qs) == {"answerable": 150, "unanswerable": 50}
    assert len({q["question_id"] for q in qs}) == 200
    assert max(Counter(q["paper"] for q in qs).values()) == 2
    docs = s["redocred"]["documents"]
    assert len({d["index"] for d in docs}) == 100 and all(0 <= d["index"] < 500 for d in docs)
    assert (s["qasper"]["file"], s["qasper"]["sha256"]) == sample.TEST["qasper"]
    assert (s["redocred"]["file"], s["redocred"]["sha256"]) == sample.TEST["redocred"]


@pytest.mark.skipif(not os.environ.get("FLATSHARE_QUALITY_DATA"), reason="needs fetch.sh's data")
def test_the_committed_sample_is_what_sample_py_draws() -> None:
    drawn = sample.build(Path(os.environ["FLATSHARE_QUALITY_DATA"]), sample.TEST)
    assert drawn == json.loads((QUALITY / "sample.json").read_text())


# ---- the record --------------------------------------------------------------------------------------------------

RECORD = REPO / "benchmarks" / "results" / "2026-10-27-quality"
RUNS = [""]


@pytest.mark.parametrize("run", RUNS)
def test_the_records_verdict_is_what_gate_py_computes(run: str) -> None:
    d = RECORD / run
    scores = json.loads((d / "scores.json").read_text())
    rows = [json.loads(line) for line in (d / "rulings.jsonl").read_text().splitlines()]
    verdict = json.loads((d / "verdict.json").read_text())
    assert verdict == gate.verdict(scores, rows)
    # Every item of the sample ran once, and every ruling is Opus's, on one login, by the committed raters.
    s = json.loads((QUALITY / "sample.json").read_text())
    outputs = [json.loads(line) for line in (d / "outputs.jsonl").read_text().splitlines()]
    assert sorted((r["task"], r["id"]) for r in outputs) == sorted(
        [("qa", q["question_id"]) for q in s["qasper"]["questions"]]
        + [("enrich", str(x["index"])) for x in s["redocred"]["documents"]]
    )
    assert [x["id"] for x in scores["qa"]] == [q["question_id"] for q in s["qasper"]["questions"]]
    assert all("opus" in r["model"] and r["output_tokens"] > 0 for r in rows)
    assert len({r["account"] for r in rows}) == 1 and "@" not in rows[0]["account"]
    assert {(r["kind"], r["rater"]) for r in rows} <= {(k, judge.rater_id(k)) for k in judge.RATERS}
    hashes = {x["judge"] for x in scores["qa"] if x["judge"]}
    hashes |= {f["judge"] for x in scores["enrich"] for f in x["facts"] if f["judge"]}
    hashes |= {y["judge"] for x in scores["enrich"] for y in x["summary"]}
    assert {r["item"] for r in rows if r["pass"] == "first"} == hashes
    assert sorted(r["item"] for r in rows if r["pass"] == "reask") == sorted(
        judge.reask_sample(list(hashes))
    )
    meta = json.loads((d / "generate.json").read_text())
    script = hashlib.sha256((QUALITY / "generate.py").read_bytes()).hexdigest()
    assert meta["script_sha256"] == script and meta["model"] == "Qwen/Qwen3-4B-Instruct-2507-FP8"
    assert (
        meta["sample_sha256"] == hashlib.sha256((QUALITY / "sample.json").read_bytes()).hexdigest()
    )
    # The README's table is the verdict's.
    readme = (RECORD / "README.md").read_text()
    for c in verdict["criteria"]:
        assert f"{c['k']:,} of {c['n']:,}" in readme, c["name"]
