# The quality gate: can the served model do both jobs of a useful pair

Fixed before the screen runs (committed with the sample and the prompts, before any test-set output exists). The
question is the one an operator asks first: does one resident model, the one the box serves, do the foreground task
and the background task of a useful pair well enough to deploy, before any contention is measured. The pair is
private-document question answering in a call, beside background enrichment of new documents into searchable
records. Both screens use public sets labeled by people independent of this project, and a judge independent of the
model under test.

## The model and its settings

Qwen3-4B-Instruct-2507-FP8 at revision 8591804019c8b22094c3b5b4454e0edc05dffc98 on vLLM 0.32.0, the engine the box
serves (deploy/phastform/engine.yaml, BF16 KV cache), on an RTX 5060 Ti. One request at a time; scheduling does not
change outputs. Answers use the engine's default sampling (the model's generation config), as the voice agent does;
enrichment uses temperature 0, as a batch job would. Each item is run once, with no retry and no best-of.

## Foreground: grounded answers with citations, or abstention (QASPER)

- **Set.** QASPER v0.3's test split (AllenAI, CC BY 4.0; questions about NLP papers asked by readers of their title
  and abstract, answered by other annotators from the full text, with evidence paragraphs). The sample: 150 questions
  that at least one annotator answered and 50 that every annotator marked unanswerable, drawn with seed 2026 from the
  test split's questions (1,372 and 79 of 1,451), at most two per paper.
- **Pipeline.** CPU retrieval: BM25 over the paper's paragraphs, the question as the query, the top 6 paragraphs, each
  numbered, after the title and abstract. The model answers in one or two short sentences, citing the numbered
  paragraphs it used as [n], or says it cannot answer from these documents.
- **Scoring.** An answer is *correct* when the judge rules it equivalent to at least one annotator's answer (a yes/no
  answer must agree with a yes/no reference); a partial answer is not correct. An *abstention* is the fixed refusal.
  A citation is *supporting* when it names a paragraph that some annotator marked as evidence. Retrieval recall (a gold
  evidence paragraph among the six) is reported beside, and the answerable questions whose evidence was retrieved are
  the coverage set.
- **Criteria.** All must hold:
  1. *Precision*: of the questions the model answers (not abstains on), answerable or not, at least 80% are correct;
     an answer to an unanswerable question is wrong.
  2. *Coverage*: at least 60% of the coverage set is answered correctly.
  3. *Abstention*: at least 70% of the unanswerable questions are abstained on.
  4. *Citations*: at least 80% of the correct answers in the coverage set cite a supporting paragraph, and every
     citation names a paragraph that was given.
  5. *Speakable*: at least 95% of the answers are at most two sentences, with no list or markdown.

## Background: enrichment with evidence (Re-DocRED)

- **Set.** Re-DocRED's test split (MIT; DocRED's Wikipedia documents with its missing facts restored, 500 documents,
  about 35 facts each, typed entities among them dates and numbers, evidence sentences for 43% of the facts). The
  sample: 100 documents drawn with seed 2026.
- **Task.** From the numbered sentences of a document, a JSON record: its entities (name as written, type of PER,
  ORG, LOC, TIME, NUM, MISC), its facts (head entity, one of DocRED's 96 relations, tail entity, the sentence numbers
  that state it) and a summary of at most two sentences, each citing its sentence numbers.
- **Acceptance**, automatic, before any quality is scored: the record parses and matches the schema, every entity name
  occurs verbatim in the document, every fact's head and tail are entities of the record, and every cited sentence
  number exists. Only an accepted record counts as completed work; a rejected one counts as none of its facts.
- **Scoring.** An entity matches a gold one when their names agree after normalization (case, whitespace, punctuation)
  with any mention of it. A fact is *correct* when it matches a gold fact (both entities matched, the same relation)
  or, when no gold fact matches, the judge rules that its cited sentences state it. A summary sentence is *supported*
  when the judge rules its cited sentences state it.
- **Criteria.** All must hold:
  1. *Accepted records*: at least 95% of the documents.
  2. *Fact precision*: at least 80% of the facts in accepted records are correct.
  3. *Fact recall*: the accepted records recover at least 20% of the gold facts of all 100 documents.
  4. *Entity recall*: at least 70% of the gold entities of the PER, ORG, LOC and TIME types are matched.
  5. *Summaries*: at least 90% of the summary sentences are supported.

## The judge

Claude Opus through the `claude` CLI, given only what a rater needs: the question, the annotators' answers and the
model's answer; or the cited sentences and the fact or summary sentence. It answers in a fixed JSON form, in batches,
and never sees the model's name. A tenth of its rulings, drawn at random, are asked again in a fresh batch in another
order: if fewer than 90% agree, the gate is inconclusive rather than passed or failed.

## Development and the verdict

The prompts and the output parser are developed on QASPER's validation split and Re-DocRED's dev split, for format
and parsing only, and frozen in this directory before the test sample runs. The test sample runs once. The verdict is
PASS when both tasks meet every criterion, FAIL otherwise, with each criterion's value and its 95% interval (Wilson)
reported; a criterion whose interval straddles its threshold is reported as such. The useful-workload pilot (the
contention measurement on these two tasks) runs only on a PASS.
