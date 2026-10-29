# Can the served model do both jobs of a useful pair: the quality gate's screen (2026-10-27)

[GATE.md](../../quality/GATE.md) asks the question an operator asks before any contention is measured: does the one
model the box serves do the foreground job and the background job of a useful pair well enough to deploy? The pair is
answering questions about private documents in a call (QASPER) beside enriching new documents into searchable records
(Re-DocRED). Its criteria, sample sizes, seed, settings and judge were committed in 0853e45, and the screen's scripts,
prompts and sample in c765696, before any output on the test split existed. This is the screen's one run of the test
sample, on Gray's RTX 5060 Ti on 2026-10-27 between 22:17 and 22:54 UTC, with the box's engine as it serves.

**The verdict is FAIL.** Nine of the eleven criteria fail, among them both tasks' central ones: of the questions the
model answers, 50.6% are correct (42.8–58.4%, against 80%), and 23.9% of the facts in its records are (21.6–26.4%,
against 80%). The judge agreed with itself on 134 of the 136 rulings it was asked again (98.5%), so the verdict is not
inconclusive. The useful-workload pilot, which GATE.md runs only on a PASS, does not run. The same screen fails
too with an FP8 KV cache ([below](#with-an-fp8-kv-cache)) and with Qwen3-8B-FP8 on a card of its own
([below](#qwen3-8b-fp8-on-a-card-of-its-own)).

## How it ran

- **The sample** ([`sample.json`](../../quality/sample.json), ids only, drawn by [`sample.py`](../../quality/sample.py)
  with seed 2026): 150 QASPER test questions that at least one annotator answered and 50 that every annotator marked
  unanswerable, from 165 papers, at most two from each; 100 of Re-DocRED's 500 test documents.
  [`fetch.sh`](../../quality/fetch.sh) downloads the sets from pinned URLs and checks their sha256s; the repository
  keeps only the sample's ids and what the model and the judge wrote.
- **Questions** ([`generate.py`](../../quality/generate.py)): BM25 over the paper's paragraphs with the question as the
  query; the top 6, numbered, after the title and abstract. The model answers in one or two short sentences citing the
  paragraphs as [n], or says "I cannot answer that from these documents." It samples as the voice agent does, with the
  engine's default, the model's generation config (temperature 0.7, top-k 20, top-p 0.8), up to 256 tokens. An
  abstention is that sentence alone: the two answers that said it after an explanation count as answers.
- **Records**: the document's sentences, numbered, in; out, a JSON record of entities (an id, the name as written, a
  type of PER, ORG, LOC, TIME, NUM or MISC), facts (head and tail by entity id, one of DocRED's 96 relation names, the
  numbers of the sentences that state it) and a summary of one or two sentences, each citing its sentence numbers. At
  temperature 0, up to 8,192 tokens, with the record's JSON schema enforced by the engine's structured outputs
  (vLLM's `response_format`, backend auto): that settles GATE.md's "matches the schema" and nothing else, so names
  copied verbatim, facts' ends that are entities of the record and cited sentences that exist stay with the model.
- **Development**, for format only, on 20 QASPER validation questions and 10 Re-DocRED dev documents. The answers
  needed no change. Asked for the record by prompt alone, the model gave 145 of 195 facts a relation outside DocRED's
  list and 48 of their ends a phrase rather than an entity, so no dev record would have been accepted; with the schema
  enforced 3 of 10 were, and with facts naming their ends by entity id 7 of 10. The commit c765696 gives the steps.
- **Scoring** ([`score.py`](../../quality/score.py)) does what needs no judge: abstentions, citations against the
  annotators' evidence paragraphs, retrieval, the sentence count, the acceptance checks and the matches with gold. The
  [judge](#the-judge) rules on every answer to an answerable question, on every fact in an accepted record that
  matches no gold fact, and on every summary sentence. [`gate.py`](../../quality/gate.py) gives the verdict, each
  criterion with its Wilson 95% interval.

## The criteria

| Criterion | Value | Count | Wilson 95% | Threshold | Result |
|---|---:|---:|---:|---:|---|
| Precision: answered questions that are correct | 50.6% | 78 of 154 | 42.8%–58.4% | ≥ 80% | **fail** |
| Coverage: answerable questions with evidence retrieved, answered correctly | 68.0% | 68 of 100 | 58.3%–76.3% | ≥ 60% | pass, interval straddles |
| Abstention: unanswerable questions abstained on | 60.0% | 30 of 50 | 46.2%–72.4% | ≥ 70% | **fail**, interval straddles |
| Citations: correct answers in the coverage set citing a supporting paragraph | 94.1% | 64 of 68 | 85.8%–97.7% | ≥ 80% | pass |
| Citations: citations that name a paragraph that was given | 99.6% | 272 of 273 | 98.0%–99.9% | all | **fail** |
| Speakable: answers of at most two sentences, no list or markdown | 91.6% | 141 of 154 | 86.1%–95.0% | ≥ 95% | **fail**, interval straddles |
| Accepted records | 76.0% | 76 of 100 | 66.8%–83.3% | ≥ 95% | **fail** |
| Fact precision: facts in accepted records that are correct | 23.9% | 292 of 1,221 | 21.6%–26.4% | ≥ 80% | **fail** |
| Fact recall: gold facts of all 100 documents recovered | 3.8% | 129 of 3,404 | 3.2%–4.5% | ≥ 20% | **fail** |
| Entity recall: gold PER, ORG, LOC and TIME entities matched | 46.9% | 692 of 1,475 | 44.4%–49.5% | ≥ 70% | **fail** |
| Summaries: summary sentences that their cited sentences state | 82.2% | 125 of 152 | 75.4%–87.5% | ≥ 90% | **fail** |
| Judge: re-asked rulings that agree with the first | 98.5% | 134 of 136 | 94.8%–99.6% | ≥ 90% | pass |
| Retrieval recall: answerable questions with gold evidence among the six (reported, no threshold) | 66.7% | 100 of 150 | 58.8%–73.7% | – | – |

- **The answers are wrong about as often as right.** Of the 154 questions the model answered, 78 are correct; the judge
  ruled 29 partial and 27 incorrect, and 20 answer a question no annotator could. Precision's whole interval lies
  under 59%. What holds is the part retrieval sets up: 68 of the 100 answerable questions whose evidence was retrieved
  are answered correctly, and 64 of those 68 cite a paragraph an annotator marked.
- **Abstention and speakability miss by less.** 30 of the 50 unanswerable questions got the abstention (interval
  46–72%, straddling 70%), and 141 of the 154 answers are speakable (its interval reaches 95%): 12 ran to three or four
  sentences, and one quotes the paper's LaTeX-style `(Laughter)', whose backtick the markdown check counts as code.
  One citation names a paragraph that was not given, an [8] of six, in an answer to an unanswerable question.
- **The records fail every criterion.** 76 of the 100 documents got an accepted record: 18 were rejected for an entity
  name not copied verbatim, 4 ran to the 8,192-token cap and are not JSON, and 2 have a fact whose end is not one of
  the record's entities. In the accepted records 129 of the 1,221 facts match a gold fact and the judge found 163 more
  stated by their cited sentences: 292, or 23.9%. Held to DocRED's relations, the model puts them on wrong pairs (a
  naval commander's "sister city", a year as an "award received"). It recovers 129 of the 3,404 gold facts (3.8%) and
  matches 692 of the 1,475 gold people, organizations, places and times (46.9%): a name with more or less around it
  ("cruiser Bogatyr" for "Bogatyr") does not match under GATE.md's rule. 125 of its 152 summary sentences are stated by
  the sentences they cite.

## With an FP8 KV cache

The same screen ran once more with the engine's KV cache in FP8, recorded in `fp8-kv/` with the same files,
on 2026-10-27 between 22:54 and 23:31 UTC: the same sample, prompts, settings, judge and criteria, with a host-local
drop-in on Gray adding `--kv-cache-dtype fp8` to the engine for this run only, and removed afterwards (its first
start, with the flag's value unquoted in the drop-in, stopped at vLLM's argument parsing, before the GPU). vLLM 0.32.0
then ran attention on FlashInfer, chosen from FlashInfer and Triton since FlashAttention 2 takes no FP8 cache on this
card, with FP8 (e4m3) keys and values, BF16 queries and the XQA decode kernel. The checkpoint carries no KV scales and
the engine log says so: "Using KV cache scaling factor 1.0 for fp8_e4m3". The same 5 GiB pool held 72,816 tokens
where BF16 held 36,400. The judge reused its rulings on the items identical to the BF16 run's (432 of 1,432), so judge
noise does not separate the runs on them, asked the other 1,000 anew, and agreed with itself on 139 of the 144 rulings
asked again.

| Criterion | Threshold | BF16 KV | FP8 KV | Difference, points |
|---|---:|---:|---:|---:|
| Precision | ≥ 80% | 50.6% (78 of 154) | 55.6% (84 of 151) | +5.0 |
| Coverage | ≥ 60% | 68.0% (68 of 100) | 72.0% (72 of 100) | +4.0 |
| Abstention | ≥ 70% | 60.0% (30 of 50) | 62.0% (31 of 50) | +2.0 |
| Citations: supporting | ≥ 80% | 94.1% (64 of 68) | 93.1% (67 of 72) | −1.1 |
| Citations: naming a given paragraph | all | 99.6% (272 of 273), **fail** | 100% (261 of 261) | +0.4 |
| Speakable | ≥ 95% | 91.6% (141 of 154) | 94.0% (142 of 151) | +2.5 |
| Accepted records | ≥ 95% | 76.0% (76 of 100) | 79.0% (79 of 100) | +3.0 |
| Fact precision | ≥ 80% | 23.9% (292 of 1,221) | 23.9% (341 of 1,427) | 0.0 |
| Fact recall | ≥ 20% | 3.8% (129 of 3,404) | 4.4% (151 of 3,404) | +0.6 |
| Entity recall | ≥ 70% | 46.9% (692 of 1,475) | 52.3% (771 of 1,475) | +5.4 |
| Summaries | ≥ 90% | 82.2% (125 of 152) | 80.4% (127 of 158) | −1.9 |
| Judge consistency | ≥ 90% | 98.5% (134 of 136) | 96.5% (139 of 144) | −2.0 |
| Retrieval recall (reported) | – | 66.7% (100 of 150) | 66.7% (100 of 150) | 0.0 |

The FP8 intervals, in `fp8-kv/verdict.json`: precision 47.7–63.3%, coverage 62.5–79.9%, abstention 48.1–74.1%,
supporting citations 84.8–97.0%, speakable 89.1–96.8%, accepted records 70.0–85.8%, fact precision 21.8–26.2%, fact
recall 3.8–5.2%, entity recall 49.7–54.8%, summaries 73.5–85.8%, judge consistency 92.1–98.5%.

- **FP8 KV fails the gate too**, on the same criteria but one: none of its 261 citations names a paragraph that was
  not given, where the BF16 run had one such [8].
- **It costs no quality this screen can see.** The two runs' intervals overlap on every criterion but entity recall,
  and that difference is which records were accepted, not what they say: on the 69 documents both runs accepted, the
  two match 641 and 640 of the same 980 gold entities. The answers are sampled (temperature 0.7), so even one engine
  gives different ones from run to run; 69 of the 200 are the same text in both. No record is the same text, though
  both are greedy: the cache's precision changes their tokens.

## Qwen3-8B-FP8 on a card of its own

In the two-card comparison ([RESOURCE-GATE.md](../../quality/RESOURCE-GATE.md)) a dedicated background card runs
without the voice servers and has their memory to spend. This run asks whether it could spend it on a more capable
model that would pass; GATE.md's verdict is about the served model and stays FAIL. The same screen ran with
Qwen/Qwen3-8B-FP8 at revision 220b46e3b2180893580a4454f21f22d3ebb187d3 alone on Gray's card, recorded in `qwen3-8b/`
with the same files: the same sample, prompts, settings, judge and criteria, and the box's engine settings otherwise
(max-model-len 32768), through a host-local drop-in for the run only, removed afterwards. Two starts:

1. **BF16 KV cache in the box's 5 GiB pool: it does not fit.** At 23:39 UTC the engine loaded the weights (8.85 GiB)
   and reserved the pool, which holds 36,400 tokens as for the 4B (the same 36 layers, 8 KV heads and head size 128),
   out of 15.34 GiB free; then its warmup ran out of memory, asking for 20 MiB with 13 MiB left, PyTorch holding
   14.04 GiB and keeping 1.14 GiB more reserved but unallocated.
2. **FP8 KV cache in a 3 GiB pool: it runs**, at 23:48 UTC, with 43,680 tokens in the pool (more than the box's 36,400
   BF16 ones) and 14,132 MiB on the card, on FlashInfer with the scale 1.0 as for the 4B's FP8 run. No smaller pool
   was needed; below 2.25 GiB an FP8 pool could not hold one 32,768-token request.

Qwen3-8B thinks before it answers unless its chat template is told not to, which the served Instruct-2507 model never
does; the drop-in set the template's default (`--default-chat-template-kwargs '{"enable_thinking":false}'`), so the
requests are those of the other runs, and no output holds a thinking block. Its default sampling is its own generation
config: temperature 0.6, top-k 20, top-p 0.95. Its answers came at 39.5 tokens a second and its records at 46.7, where
the 4B's came at 66.0 and 75.1, so its records took from 23:57 to 01:15 UTC; the engine ran without a restart.

| Criterion | Threshold | 4B, BF16 KV (served) | 4B, FP8 KV | 8B, FP8 KV, own card |
|---|---:|---:|---:|---:|
| Precision | ≥ 80% | 50.6% (78 of 154), **fail** | 55.6% (84 of 151), **fail** | 38.4% (71 of 185), **fail** |
| Coverage | ≥ 60% | 68.0% (68 of 100) | 72.0% (72 of 100) | 63.0% (63 of 100) |
| Abstention | ≥ 70% | 60.0% (30 of 50), **fail** | 62.0% (31 of 50), **fail** | 24.0% (12 of 50), **fail** |
| Citations: supporting | ≥ 80% | 94.1% (64 of 68) | 93.1% (67 of 72) | 88.9% (56 of 63) |
| Citations: naming a given paragraph | all | 99.6% (272 of 273), **fail** | 100% (261 of 261) | 100% (211 of 211) |
| Speakable | ≥ 95% | 91.6% (141 of 154), **fail** | 94.0% (142 of 151), **fail** | 97.3% (180 of 185) |
| Accepted records | ≥ 95% | 76.0% (76 of 100), **fail** | 79.0% (79 of 100), **fail** | 73.0% (73 of 100), **fail** |
| Fact precision | ≥ 80% | 23.9% (292 of 1,221), **fail** | 23.9% (341 of 1,427), **fail** | 28.4% (306 of 1,078), **fail** |
| Fact recall | ≥ 20% | 3.8% (129 of 3,404), **fail** | 4.4% (151 of 3,404), **fail** | 4.0% (137 of 3,404), **fail** |
| Entity recall | ≥ 70% | 46.9% (692 of 1,475), **fail** | 52.3% (771 of 1,475), **fail** | 46.6% (687 of 1,475), **fail** |
| Summaries | ≥ 90% | 82.2% (125 of 152), **fail** | 80.4% (127 of 158), **fail** | 93.8% (137 of 146) |
| Judge consistency | ≥ 90% | 98.5% (134 of 136) | 96.5% (139 of 144) | 98.4% (121 of 123) |
| Retrieval recall (reported) | – | 66.7% (100 of 150) | 66.7% (100 of 150) | 66.7% (100 of 150) |

The 8B's intervals, in `qwen3-8b/verdict.json`: precision 31.7–45.6%, coverage 53.2–71.8%, abstention 14.3–37.4%,
supporting citations 78.8–94.5%, citations naming a given paragraph 98.2–100%, speakable 93.8–98.8%, accepted records
63.6–80.7%, fact precision 25.8–31.1%, fact recall 3.4–4.7%, entity recall 44.0–49.1%, summaries 88.7–96.7%, judge
consistency 94.3–99.6%.

- **The larger model does not rescue the background job.** Its records fail four of the five criteria, much as the
  4B's do: 73 of 100 accepted (17 looped to the 8,192-token cap and are not JSON, 16 repeating one kind of fact and
  one its 17 entities; 9 have a name not copied verbatim, 1 a fact whose end is not an entity), and of their facts
  28.4% are correct (25.8–31.1%, against 80%).
  Only its summaries pass, 137 of 146 sentences stated by what they cite.
- **On the answers it trades abstention for brevity.** It answers in fewer sentences (180 of 185 speakable, a pass)
  but abstains on only 12 of the 50 unanswerable questions, and says the abstention sentence after an explanation 16
  times, which counts as an answer; so it answers more and gets fewer right, 71 of 185.
- **The pool is not what limits it.** 43,680 FP8 tokens hold more than the box's BF16 pool, and no request came near
  them: the longest, prompt and reply, took 9,966.

## Files

- `outputs.jsonl`: the 300 outputs, one a line: the task and its ids (QASPER's question and paper, Re-DocRED's document
  index and title), for a question the paragraphs retrieved (indices into the paper's paragraphs, best first) and their
  BM25 scores, the prompt's sha256, the raw text, the parsed result, the finish reason and the token counts.
- `generate.json`: the run's settings, the engine's model and version, the script's and the sample's sha256s.
- `scores.json`: [`score.py`](../../quality/score.py)'s automatic scores, an entry a question and a document, with the
  hash of each item that went to the judge.
- `rulings.jsonl`: the judge's 1,492 rulings, one a line: the item's hash, its kind, the rater's version, the pass
  (first, or the re-ask), the ruling and its reason, the batch and the place in it, the model and its output tokens,
  the login's subscription plan and the time.
- `verdict.json`: [`gate.py`](../../quality/gate.py)'s verdict from `scores.json` and `rulings.jsonl`;
  `python/flatshare_bench/tests/test_quality_gate.py` checks that it is what they give, and this table against it.
- `fp8-kv/`: the same five files for the run with an FP8 KV cache.
- `qwen3-8b/`: the same five files for Qwen3-8B-FP8 on a card of its own.

## The judge

Claude Opus through the `claude` CLI (2.1.311) on the subscription, never an API key, spawned by
[`judge.py`](../../quality/judge.py) with only `HOME`, `PATH` and `USER` in its environment, plus
`CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` and `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`, in an empty directory:

```sh
claude --print --model opus --output-format json --tools "" --setting-sources "" --strict-mcp-config \
  --no-session-persistence --exclude-dynamic-system-prompt-sections \
  --system-prompt "$RATER" --json-schema "$FORM" "$BATCH"
```

`$RATER` is the kind's instructions (`QA_RATER`, `FACT_RATER`, `SUMMARY_RATER`), `$FORM` its fixed JSON form (an item
number, a reason and a ruling of correct, partial or incorrect, or of yes or no), and `$BATCH` up to 20 items of one
kind with only what a rater needs: the question, the annotators' answers and the answer with its citation markers
dropped; or the cited sentences and the fact (head, relation, tail) or the summary sentence. The model's name never
appears. The answer is the output's `structured_output`, which must rule on every item of the batch once. Every call's
`modelUsage` named `claude-opus-5-5` with output tokens and no other model with any, and `claude auth status` before and
after every batch showed the same subscription login (every ruling records its plan, `max`). The
1,356 first rulings came in 69 batches and the re-ask in 8, all on the first attempt, between 22:27 and 23:06 UTC.
Rulings are cached by the item's hash and the rater's version, so a second run asks nothing again. The re-ask drew 136
of the 1,356 items, a tenth rounded up, with seed 2026 and asked them again in fresh batches in another order; 134
agree with the first on what the gate counts, and on the label too.

## Environment

Gray's RTX 5060 Ti 16 GB (driver 580.178.04), vLLM 0.32.0 and Qwen3-4B-Instruct-2507-FP8 at revision
8591804019c8b22094c3b5b4454e0edc05dffc98: the frozen box's engine at v1.5
([`deploy/phastform/engine.yaml`](../../../deploy/phastform/engine.yaml)), with its BF16 KV cache in a 5 GiB pool of
36,400 tokens (the engine log's GPU KV cache size), FlashAttention as its attention backend, prefix caching and the
20 GiB CPU tier, and Gray's host-local drop-ins (its NUMA node and the host network, as for
[the first sentences' record](../2026-10-27-first-sentence/README.md)). The engine ran only while no other process
used the GPU, in each run: none before it started, only its own between the two tasks, and the card back to 4 MiB
once it stopped. One request at a time.

## To run again

```sh
sh benchmarks/quality/fetch.sh DATA
python benchmarks/quality/generate.py --engine unix:$XDG_RUNTIME_DIR/flatshare/engine.sock --data DATA \
  --sample benchmarks/quality/sample.json --out OUT          # the engine's venv has httpx
python benchmarks/quality/score.py --data DATA --sample benchmarks/quality/sample.json --out OUT --items ITEMS.jsonl
python benchmarks/quality/judge.py --items ITEMS.jsonl --rulings OUT/rulings.jsonl --workdir EMPTY_DIR
python benchmarks/quality/gate.py OUT
```

`ITEMS.jsonl` holds the sets' text the judge sees, so it stays out of the repository; `score.py` rebuilds it, with the
same hashes, from the data and `outputs.jsonl`.
