# Changelog

What each release changed, newest first, summarized from its release notes; each release's page has its assets (raw
records, the demo video). The project was called Leaseholder through v1.1, and those releases' tools, assets and links
keep that name. Numbers are each release's own, with the campaign that measured them: v1.0 re-measured v0.3's
headline in rounds, and [BENCHMARKS.md](docs/BENCHMARKS.md) has every run.

## [v1.26](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.26) (2026-11-10): the documents cut to what each reader needs, and every figure in one grammar

A documentation release: no record, no measurement and nothing the boxes serve changes. The hand-written documents
shrink from about 211,000 words to about 125,000. Each names its reader in its first paragraph, answers before it shows
the evidence, and gives each number one home, in BENCHMARKS or a record, with one statement of it in the report's
argument. Statements that a fresh read found stale or unscoped were checked against the code and the records.

- **The report** makes one argument in about 20,000 words, from 31,000: each results section opens on its claim, then
  its figure or table, then the record. Campaign chronology, restated limits and the tables that repeated a figure are
  gone, the tables after the dropped Table 7 are renumbered, and the revision history is a compact appendix.
- **BENCHMARKS** answers one question a section, in about 18,000 words from 34,000, with the configurations named as
  the glossary names them, and EXPERIMENTS keeps each experiment's design and the campaign map.
- **The README** shows its results as the report's headline figure, whose caption carries the voice tail and the
  agents' token rate. PRODUCTION, ARCHITECTURE, the runbook, PRIOR-ART and the component READMEs keep their reader's
  job: the contract and every key, how the system is built, the operator's steps in the order they run, the related
  work, and each component's commands and settings.
- **Every figure** follows one grammar: no title or note drawn in, axes that name their quantity and unit, and one
  colour and marker per configuration in every chart, legible in grayscale. Figure 7 loses its second axis,
  BENCHMARKS gains two charts, the serving engine's additions and arm 9's ablations, and the chart of both GPUs, which
  no document showed, is retired.
- **Links** reach the new headings. The anchors that records and the alert rules name stay where they were.

## [v1.25](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.25) (2026-11-09): §6.11 drawn, the pull requests rebased, and the queue and the copy proposed to vLLM

Version 1.24's envelope is drawn and stated where operators read, the pull requests are rebased on vLLM's main at
821851eab, and version 1.24's two parts are proposed to vLLM as drafts. Nothing the boxes serve changes schedule: they
stay on v1.24 until a release is deployed.

- **The report's §6.11** gains two figures from the record: Figure 17, where the agents' prompt tokens came from under
  first come first served and under the cache-aware queue, and a step's time beside a 1.46 GB CPU-tier load by copy
  path; and Figure 18, the card's envelope in three rounds at 4, 12 and 24 sessions, which the README shows too.
- **The copy probes** are stated as what they sampled: beside a load submitted from another thread, the slowest of two
  to four steps took 192–209 ms in one batch and 136–152 ms in chained jobs, and the 12-program run's 19.2 ms, the
  kernel's first call at the load's job count, is not counted as that program count's cost (the record, §6.11, the
  hardware guide, flatshare_copy's docstring and serving.json's pin of it).
- **The documents** state the envelope of 24 where operators read it: the README's mechanism and the agents' tail beside
  24 sessions, the runbook's envelope check, ADR-0008's status, the glossary, and the hardware guide's copy path on
  PCIe 3.0 x8.
- **The pull requests** are rebased on vLLM's main at 821851eab (#65282 f1a1003f8, #65283 fe4720517), where #65285
  keeps a parked async KV load's reservation apart from the in-flight prefills: the room and the streaming check read
  both, and the first-tokens hold records a parked load's first step where it is parked, so finding the holders takes
  0.6 µs a step beside 4,096 parked loads instead of 0.70 ms. The options are in `SchedulerConfig`'s `ignored_factors`
  (#65652). make-room's six patches, compat-upstream's adapted lines and the module carry the new heads; the module
  schedules as v1.23's on all 2,036 traced configurations.
- **Upstream proposals**, drafts until they run on a GPU: [#66558](https://github.com/vllm-project/vllm/pull/66558) lets vLLM's cache-aware admission (#54625),
  under first come first served, count what the CPU tier holds, and [#66527](https://github.com/vllm-project/vllm/pull/66527) loads the tier through vLLM's
  Triton copy kernel. PRIOR-ART and the report's §8 place them beside SGLang's cache-aware scheduling, #54625, issue
  #53277, issue #57555 and #66511.

## [v1.24](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.24) (2026-11-08): the card's envelope from 4 agent sessions to 24

Beside 24 agent sessions, first come first served over the LRU CPU tier recomputed 89.5% of the prompt tokens, nearly all of them the agents':
each request waited longer than the tier kept its session's prefix. The serve module now orders the agents' waiting
queue by what each has left to compute, and loads the tier through vLLM's SM copy kernel. In three rounds on new samples
the card qualifies beside both calls with 24 sessions, where the profile through v1.23 qualified at four
([the record](benchmarks/results/2026-11-08-envelope/README.md); [REPORT §6.11](docs/REPORT.md#611-ordering-the-agents-by-what-they-have-left-raises-the-cards-envelope-from-4-sessions-to-24)).

- **The cache-aware queue** (`queue_max_wait_s = 120`, `serve/flatshare_queue.py`): within a priority, the waiting
  request with the least prefill left, net of what the GPU pool and the CPU tier hold, goes first. One that has waited
  120 s goes ahead of the rest, and the calls keep their arrival order. Each step refreshes at most two counts, and none
  beside a call. The tier's part is counted as vLLM's own lookup counts it.
- **The SM copy** (`sm_load_programs = 4`, `serve/flatshare_copy.py`): on the box's PCIe 3.0 x8 links, a CPU-tier load on
  the copy engine holds a concurrent step's inputs until the load ends, about 200 ms per 1.46 GB. The tier's loads now go
  through vLLM's own Triton kernel, its size threshold lifted, which moves 6.1 GB/s and leaves the step alone. Both of
  the kernel's specializations are compiled and checked before the first load.
- **The envelope** (`max_agents = 24`): 3 of 177 replies over the SLO beside 24 sessions and 4 of 177 beside 12, at 7,253
  and 6,906 agents' tokens a second against 3,278 at four. The agents' TTFT p99 is 133 s at 24 sessions against 24 s at
  12, so a node may hold fewer.
- **serving.json** pins the modules the serve module installs from (flatshare_jit, flatshare_queue, flatshare_copy)
  beside the serve module, and the effective configuration records what was installed.
- **flatshare_jit's early finish** never fired on Model Runner V2, the box's runner, which names its event otherwise.
  Made to fire, it measured no gain, so its docstring says so and the behaviour stays as measured.

## [v1.23](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.23) (2026-11-05): the pull requests' revision of 5 November

A review of #65283 and then a fresh one of both found stalls on the real scheduler, where a request that waits for what
it holds back stops the engine: a request whose KV load is done, behind a request that fails its allocation, and a running
prefill scheduled nothing. The serving profile loads a call's turn from the CPU tier, so the engine could reach the first.
Both pull requests are revised, each with regression tests that fail before the fix, and make-room's patches carry them.

- **The budget cap** bounds every request whose first tokens are due by the steps it computes nothing: at most T in a
  row, which its progress resets. A load in flight keeps its T steps from its start. A request a higher priority's hold
  keeps out does not count those steps, and a running request with one token left is no longer held.
- **The KV preemption's gate** waits only on a higher-priority prefill that gets a chunk in the step, so a KV load in flight
  or a prefill scheduled nothing keeps no request out, and room is made for it again if it needs more.
- **A search for stalls of this kind**
  ([the record](benchmarks/results/2026-11-05-liveness/README.md)) runs 400,000 random workloads on the pull
  requests, and 200,000 on the module with the serving profile's switches, each with the options off and on.
  None stalls because of the options, where the same search finds 241 and 153 on the
  code before. A state that vLLM's main cannot leave either remains, with nothing running.
- **The pull requests** are rebased on vLLM's main at 0b3a608dd, where #54625's fields landed beside make-room's, and make-room's
  six patches, the module's pin and compat-upstream carry the new heads.
- **make-room's trace** against the module the evaluation ran: with the budget cap off, the same but for 98 of the
  320 configurations with a KV connector, where the gate's change shows; with it on, different in 1,080 of
  1,264. Against v1.22's module, the revision
  changes 426 of 2,036 configurations, 329 of them with a KV connector.

## [v1.22](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.22) (2026-11-04): the pull requests rebased on vLLM's main

#65282 and #65283 are rebased on vLLM's main after two changes landed there: #65319's queue of requests waiting to
start an async KV load, and #65300's weak reference from a KV cache block to its pool. Neither is in v0.32.0 or
v0.33.0, so the backport's code and the module stay as v1.21's.

- **The pull requests** dereference a block's pool in the count of the blocks preempting frees, and take the requests
  waiting to start an async KV load into make-room's target scan, its in-flight gate, the budget cap's scan of lower
  priorities and its waiting hold, each with a test that fails without it.
- **The backport's commits** are remade with the same trees on the fork's `v0.32.0-` and `v0.33.0-` branches: their
  Backport paragraphs say how the releases differ, and make-room's six patches change in their headers alone.
- **compat-upstream** names the rebased heads' new lines as adapted under those commits.

## [v1.21](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.21) (2026-11-04): the pull requests' revision of 3 November

A fresh review of #65282 and #65283 found cases where the KV preemption could make room a request would not get, and
where the budget cap could hold prefill back without end or for a request it could not help. Both pull requests are
revised, each with regression tests that fail before the fix, and make-room's patches carry the revision.

- **The KV preemption** bounds the waiting request's prefix-cache hit as the waiting loop does (prompt scoring's limit
  and the aux-output connector's), and in the step it makes room, a lower-priority request that holds KV blocks but no
  reservation for the rest, such as a streaming session's next chunk, waits instead of taking the room.
- **The budget cap** holds for a waiting request that the waiting loop does not admit for at most T steps, never holds
  a request with one token left, lets a held request's KV load start, holds a lower-priority request that already holds
  KV blocks only while a request of its priority or a lower one runs, counts a step in flight as one token, and keeps
  its encoder exemption to a prefill midway through a cached input while a request it is held for needs encoder input.
- **The variant** counts the admissions the budget cap holds at both of its hold sites, adds an announced turn's
  priority 0 to the hold beside the new holders, and counts a step in flight as one token in its decode hold. The
  fork's `v0.32.0-` and `v0.33.0-` branches carry the revision, and make-room's patches are written from them.
- **The module** built on 0.32.0 schedules as v1.20's wherever the budget cap is off (`make test-trace` against the
  module Phastform's engine built on 2026-10-27: none of 772 configurations differ) and differs with it on (1,106 of
  1,264); the serving profile pins its new hash.
- **The report, version 1.21,** gives the revised rules in §3.1, §3.2, Algorithms 1 and 2 and §3.4, and §4 discloses
  that the evaluation predates them.
- **The probes** the pull requests cite, rerun on the revision in one session with stock vLLM as the control
  (`benchmarks/results/2026-11-04-r3-probes`): the KV case's first token after 284 ms against 4,976 ms on stock, and the
  prefill case's 16th token after 560 ms against 1,374 ms, as with the earlier code.
- **The packages** are versioned 1.21.0, and are not published to a package index.
- **Upgrading from v1.20.** A deploy drains and restarts the engine, whose patches changed; it builds the new module
  at its start, and `serving-check` expects the new hash.

## [v1.20](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.20) (2026-11-03): no hold on a prefill whose encoder outputs are cached

A review of #65283 found that its budget cap could stall a multimodal model. A lower-priority prefill midway through an
image, whose encoder output filled the encoder cache, was held back while the higher-priority request it was held for
waited for that space, which only the held prefill's progress frees. #65283 now holds back no prefill whose encoder
outputs are cached, and make-room's patches carry it. The voice agent's model takes no encoder inputs, so the engine
schedules as v1.19's.

- **The budget cap** holds back no prefill whose encoder outputs are cached: #65283's commit 8c753658a, with a
  regression test. The fork's `v0.32.0-priority-prefill-hold` and `v0.32.0-flatshare-make-room` branches and their
  v0.33.0 counterparts carry it, and make-room's patches are written from them.
- **The module** built on 0.32.0 differs from v1.19's by that clause alone, and `make test-trace` against the module
  Phastform's engine built on 2026-10-27 finds every step the same; the serving profile pins its new hash.
- **The report, version 1.20,** gives the bound in §3.2 and Algorithm 2.
- **The packages** are versioned 1.20.0, and are not published to a package index.
- **Upgrading from v1.19.** A deploy drains and restarts the engine, whose patches changed; it builds the new module
  at its start, and `serving-check` expects the new hash.

## [v1.19](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.19) (2026-11-03): the change's pull requests as #65282 and #65283

The change is proposed to vLLM as #65282 and #65283, which succeed #63350 and #63351: the same change, rebased onto
vLLM's main as one commit each. make-room's patches now come from the new pull requests and build v1.18's code. No
measured number changed.

- **The pull requests.** #65282 carries the KV preemption and #65283, stacked on it, the budget cap; #63350 and #63351
  are closed, each with a pointer to its successor. Every link and every statement of their state names the new ones,
  open with no maintainer review as of 2026-11-03, while the records, tables and figures keep the first numbers, as
  the glossary says. ARCHITECTURE's §2 is "The change", without the numbers.
- **make-room's patches** come from the fork's new backport branches, `v0.32.0-priority-kv-preemption`,
  `v0.32.0-priority-prefill-hold` and `v0.32.0-flatshare-make-room`, and their v0.33.0 counterparts, with each pull
  request's part one commit; they keep their names, `63350.patch` and `63351.patch`. The module they build on 0.32.0
  and on 0.33.0 has v1.18's code, its syntax tree equal but for docstrings, and `make test-trace` against the module
  Phastform's engine built on 2026-10-27 finds every step the same. Its comments, docstrings and header differ, so the
  serving profile pins its new hash.
- **`make compat-upstream`** checks #65282 and #65283 against vLLM's main, each with its patch (`PATCH`), and its
  check of the backport against the pull requests' heads names the new backport commits.
- **The report, version 1.19,** names the new pull requests in its note, §4 and references [9] and [10], and corrects
  §2.4, which said a voice reply needs a few dozen steps before it can be spoken: its first sentence takes a step for
  each of its tokens, 15 at the median and 39 at most (§6.3).
- **The packages** are versioned 1.19.0, and are not published to a package index.
- **Upgrading from v1.18.** A deploy drains and restarts the engine, whose patches changed; it builds the new module
  at its start, and `serving-check` expects the new hash. The engine schedules as v1.18's.

## [v1.18](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.18) (2026-11-02): the documents made current and consistent

The documents, as GitHub renders them, follow one form and say where v1.17 stands: the README, the report, the method,
operations and decision documents, and the packages' and the records' READMEs. No measured number changed, and the code
that serves is v1.17's.

- **The README** gives each further result one bullet with its section of the report, sets the demo figure after the
  opening, and puts its Results table's units in the header, so that no cell wraps. Its limits give v1.17's decode hold
  (beside 24 sessions, 0.8–5.1% of the replies over the SLO in two rounds), and its code blocks fit the page.
- **The report, version 1.18,** corrects Appendix A, which said BENCHMARKS lists every measurement: it lists the
  campaigns' measurements, and the records' READMEs the probes and checks around them. Its tables' empty cells are em
  dashes, as every other document's are.
- **BENCHMARKS and the method documents** give each question one layout: provenance, how it ran, figure, caption, table,
  findings. The tables BENCHMARKS shares with the engine points' and the rounds' records set each median over its range,
  as those records now print them too. PRIOR-ART says the configuration to deploy ends 0.20 s over the SLO, where it
  said within 0.2 s, and that vLLM 0.33.0 includes #61976 and #54627.
- **Every table and code block fits the page** at github.com's width, as Chrome measures GitHub's rendering of each
  file: BENCHMARKS' costs to the agents by mechanism and the follow-up rounds' are two tables each, the soak's reads
  down, and the FP8 crash's runs are two tables, which `summarize.py` prints and a test holds to the runs' files; long
  commands are cut into continuation lines and their comments wrapped, the commands themselves unchanged. The tools' own
  summaries stay as their tools write them.
- **The records' READMEs** share one skeleton, and what they say in the present tense is dated or brought up to date:
  vllm-project/agentic-api#478 merged on 2026-10-26, vLLM 0.33.0 is out and make-room ported to it, and vLLM #64165 is
  open as of 2026-11-02.
- **The packages', operations' and decision records' documents** share one skeleton and name make-room's two vLLM
  releases and the profile's decode hold. flatshare-sim's two tables that ran off the page are two each, and ADR-0006
  and ADR-0009 gain dated notes under Later.
- **A test holds every relative link** of the repository's Markdown, its images and anchors too, to its target as GitHub
  resolves it.
- **The packages** are versioned 1.18.0, and are not published to a package index.
- **Upgrading from v1.17** needs no change on a host: the code that serves and the serving profile are v1.17's.

## [v1.17](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.17) (2026-11-02): the decode hold in the serving profile

The serving profile holds the agents' decodes out of a voice reply's first 48 tokens, the budget cap's window, for the
voice tail. In two follow-up rounds beside 24 agent sessions the hold put 16 of 720 replies over the SLO against 55
without it, fewer in five of the six rounds and caller counts, for 3.8% of the agents' token rate at the median: past
the 3% a rule fixed before the rounds allowed, so that v1.15 left it off. v1.17 takes it for the voice tail.

- **The profile.** `decode_hold_tokens` goes from 0 to 48 in `rtx-5060-ti-serve.toml`, and with it `serving.json`, the
  `make-room-serve` variant, the README's quick start and PRODUCTION's row; `serving-check` holds the engine to it.
- **Its records.** The quick start on Phastform with the hold: make-room answered the voice turn in 173–179 ms against
  1,543–1,556 ms on stock vLLM, where without the hold it took 272–277 ms ([the
  record](benchmarks/results/2026-11-02-quick-start/README.md)). On vLLM 0.33.0, run again: 167–207 ms against
  1,778–1,788 ms, where without the hold it took 264–267 ms, and the serving variant's own engine 170–175 ms against
  208–211 ([the port's record](benchmarks/results/2026-11-02-port-v0.33.0/README.md)). Each record's test now holds the
  switches its engine logged to the commands it gives.
- **The scheduler's cost a step** comes from the baseline measured for this configuration, at a tolerance of 4%: the
  step the hold keeps clear costs 63 µs less, and a step with only lower-priority work running 48–182 µs more, against
  33–151 µs without the hold (REPORT §4, Figure 7). On peacock the check passes in all 17 states, and its negative
  controls fail by 17–88%.
- **The docs.** REPORT §6.10 and §7, BENCHMARKS, GLOSSARY, ARCHITECTURE and the engine points' and follow-up rounds'
  records say the profile holds the decodes, and the follow-up rounds' verdicts stand as measured. Beside 12 agents the
  five additions take a call's first token from 354 to 88 ms at p95 and its whole reply from 1,226 to 722 ms, for 4% of
  the agents' token rate (Table 9).
- **The packages** are versioned 1.17.0, and are not published to a package index.
- **Upgrading from v1.16.** A release renders the engine's new flag, `--additional-config.make_room.decode_hold_tokens
  48`, and `serving-check` expects it. An operator who prefers the agents' rate sets `decode_hold_tokens = 0` in the
  profile and renders it again (`make profiles UPDATE=1`).

## [v1.16](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.16) (2026-11-02): make-room on vLLM 0.33.0

make-room ported to vLLM 0.33.0, released the same day: the change builds and passes its suites there, and the quick
start's probe and the serving variant's engine ran on the card. The serving profile stays on 0.32.0, which the rounds
measured.

- **make-room on vLLM 0.33.0.** `make-room/v0.33.0` holds the three patches: the same 26 commits cherry-picked onto
  v0.33.0, on the fork's `v0.33.0-*` branches. Two needed resolving, against 0.33.0's local prefill cadence and its cap
  of the external hit for auxiliary outputs, and by patch ID every other commit's change to the scheduler is its
  source's. `build.py` takes the installed vLLM's release from its metadata, and a source checkout's from the one
  release whose patches apply to it (`--release` names one).
- **Its checks.** The build's tests (`make test-vllm VLLM_REF=v0.33.0`), the behavior suite, the serve module and the
  trace against the pull requests' diff on 0.33.0's real scheduler (`make test-make-room-ci VLLM_REF=v0.33.0`, 570 and
  1,081 passed in `requirements-cpu-v0.33.0.txt`'s environment), and `make compat-upstream`, which now holds a ported
  release to its own patches. On Phastform the quick start's probe gave the voice turn's first token in 264–267 ms with
  make-room against 1,779–1,792 ms on stock 0.33.0, and the serving variant's engine ran there, serving-check finding
  only the release and the module changed ([the record](benchmarks/results/2026-11-02-port-v0.33.0/README.md)).
- **The packages** are versioned 1.16.0, and are not published to a package index.
- **Upgrading from v1.15.** No configuration changes: the serving profile still runs vLLM 0.32.0. An operator who moves
  to 0.33.0 installs it and builds make-room as before, and `build.py` takes 0.33.0's patches.

## [v1.15](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.15) (2026-11-02): the serving engine optimized

What the engine adds to v1.14's serving profile, each piece chosen on the box from its scheduler's traces and measured
against the configuration before it. Beside 12 agents a call's first token goes from 175 to 64 ms at the median and from
354 to 92 ms at p95, and its whole 48-token reply's p95 from 1,226 to 998 ms. In five rounds of the whole pipeline the
four additions cut the replies over the SLO beside 24 sessions from 18.6%, 18.5% and 18.0% to 5.4%, 6.4% and 5.8% with
two, four and six callers, fewer in every round, and beside 16 sessions from 7.5% to 3.4%, for 1.6–4.9% of the agents'
token rate. Two more rounds turned down a fifth, the agents' decodes held, and speculative decoding, by rules fixed
before they ran.

- **The calls' prefixes kept on the GPU.** A finished call turn's blocks stay referenced until the conversation's next
  turn takes them (`make_room.retain_tokens`, `retain_s`), counted in distinct blocks, as LiveKit's preemptive
  generation retains a cancelled request beside its replacement.
- **The next batch just in time.** The serve module schedules the next batch when the step in flight is due to end, by
  the card's step-time model, so a call arriving during an agent's long step joins the next batch.
- **Each turn announced.** The voice worker sends a realtime request for one token when the user stops speaking, and the
  budget cap holds the agents' prefill back for 400 ms from it.
- **A call never waits behind an agent's reservations.** A call's turn could wait 1.4–4.5 s while agents' prefix
  loads held the pool's free blocks in reservation: the load admission counted every in-flight reservation, and the
  waiting loop stopped at the first completed load it could not admit. With `make_room.take_reservations` a request
  counts only the reservations of its own priority or higher, and make-room preempts for the rest.
- **What was measured and not adopted.** The agents' decodes held through the budget cap's window
  (`make_room.decode_hold_tokens` 48): in the engine points it took the 48th token's p95 from 998–1,016 to 722–735 ms,
  and in two follow-up rounds it put 16 of 720 replies over the SLO against 55, but for 3.8% of the agents' token rate
  at the median, past the 3% its rule allowed. Speculative decoding on top, EAGLE-3 with two drafts and the calls' short
  loads recomputed (`make_room.recompute_tokens`), put 12 over against the hold's 16, fewer in three of the six rounds
  and caller counts where its rule asked for five ([the follow-up
  rounds](benchmarks/results/2026-11-02-opt-rounds/README.md#the-decode-hold-and-speculative-decoding)). Both stay
  switches, off. Nor were synchronous scheduling, Triton with full CUDA graphs, FP8 KV and two API servers ([the engine
  points](benchmarks/results/2026-11-01-engine-points/README.md#what-was-not)).
- **The records.** [The engine points](benchmarks/results/2026-11-01-engine-points/README.md), 82 runs on both nodes
  with each addition against the point before it; [the rounds](benchmarks/results/2026-11-02-opt-rounds/README.md), five
  of the four additions against v1.14's configuration and two follow-up rounds; [the quick
  start](benchmarks/results/2026-11-02-quick-start/README.md) again, 272–277 ms against 1,544–1,547 ms on stock vLLM,
  and 173–179 ms while the profile held the agents' decodes; the scheduler's cost a step in the profile's configuration,
  measured again, a baseline whose 7% tolerance its negative controls exceed by 17–84%.
- **The packages** are versioned 1.15.0, and are not published to a package index.
- **Upgrading from v1.14.**
  - The serving profile's engine adds make-room's `retain_tokens` 3584, `retain_s` 30, `hint_hold_ms` 400 and
    `take_reservations` true, with `decode_hold_tokens` 0, and runs the serve module's scheduler,
    `flatshare_make_room_serve.AsyncScheduler`, with `flatshare_jit` at a 30 ms margin and the card's step-time model: a
    release renders them, and `serving-check` holds the engine to every one.
  - In serve mode the voice worker announces each turn (the profile's voice `turn_hints`, `FLATSHARE_TURN_HINTS` over
    it). The engine's log line counts the turns announced: a count that stays at 0 while calls run means the hints do
    not reach the engine.

## [v1.14](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.14) (2026-11-01): the code reviewed, and the FP8 crash found

A review of the code's architecture, plane by plane, with what it found fixed, and the FP8 KV crash of 10-27 traced to
its cause in vLLM and FlashInfer. E1's simulated cells at higher pressure change; no measured result does, and the
soak's record now says what a harness restart added to it.

- **The system at a glance.** [ARCHITECTURE.md](docs/ARCHITECTURE.md#at-a-glance) opens with a figure of the change and
  the three planes around it, serving, measurement and operations, each part named with its home in the repository.
- **The FP8 crash, found.** [The FP8 record](benchmarks/results/2026-10-31-fp8-crash/README.md) reproduces the
  10-27 crash on stock vLLM 0.32.0 and 0.33.0rc3 with none of the change's code, and finds it in FlashInfer's
  `plan()`, which refills one page-locked buffer per wrapper while its last copy to the GPU may still be queued
  (vLLM #61239): it needs the CPU tier, asynchronous scheduling and FlashInfer, and the wait of vLLM #64165 closes it.
  On this card the CPU tier's DMA holds the plan's copy back on the one host-to-GPU copy engine: without the CPU tier
  no plan in 10,000 found its copy pending. A GPU core dump names the faulting kernel, FlashInfer's
  `PersistentVariableLengthMergeStatesKernel`; XQA decode does not avoid the race, and vLLM main stalls too.
  [`benchmarks/fp8`](benchmarks/fp8/README.md) is the harness. The serving profile keeps its BF16 cache.
- **The gateway's drain.** When `drain_timeout_s` passed, the process exited under the relays still running, and their
  records said `completed`, the only success; they now say `drained`. A stop also left a request that a covering node
  holds back unanswered for the whole drain, logged `cancelled`; it now gets 503 `draining` with `Retry-After: 1` as the
  drain begins, as at a handover. Two tests run the binary.
- **Releases.** A dropped session no longer stops a release halfway: the tool's output drops the terminal when a write
  fails and keeps every line in `releases/release.log`, and a switch records itself in `releases/switching.json`, so
  that the next switch finishes one a killed run left, restarting every active unit. The voice drain trusts only a
  status the worker wrote after it began and removes its drain file whenever it ends early. A release refuses while an
  E2 point's drop-ins are linked into cyan's units, as the soak does.
- **The simulator.** A prefetch paid for every block the CPU tier held, also those the pool had no room for, which it
  never placed; it now pays for what fits when it begins. E1's prefetching arms change at 16 sessions and more (the
  oracle at 24 sessions: +4.9% / −3.9% at p50 / p90, from −2.4% / −16.9%), and its conclusions hold: under 8% of the
  median resume, 28–29% of the tail at low pressure, nothing from session history.
- **The harness.** A soak's window starts on the millisecond its first scrape is stamped with, so the counters' rise
  before it never falls in its first hour; neither committed soak changes, the day's having started on a millisecond
  and voice alone's having no agents. A run of arriving calls lasts its whole time when its last call ends first:
  voice alone's soak started its callers again 87 s before its end, a second draw that placed one call where the first
  had placed none, and without it the day's replies over the SLO number 77, not 74, 1.5% either way (the soak's
  record says so). Each soak's `summary.json` is held to what its record gives, and the record of E2 names the
  two points whose summaries leave their failed turns out (the report recomputes every point's V2V from its turns).
  A record that names an account's home directory fails the tests, since the records are copied from shared hosts.
- **What the docs say.** A round's sample of the agents is a draw of where each session enters the same traces, since
  AIPerf's sampler gives every round the same ones, and REPORT §9 states the limitation; serve mode's endpointing,
  0.5 s at least and 3 s at most, is a pin, not LiveKit's default for its end-of-turn model; the kill drills' streams
  ended 6–11, 10–14 and 53–57 ms after the kill command, as their records give them, where the docs began each range
  at a median; #45558 preempts for KV as #40087 does (PRIOR-ART, REPORT §8 and #63350's description); the restore
  stall's upper quartile is 0.94, as the simulator's fit computes it; and the bench suite takes four to five minutes.
  The report is at version 1.14.
- **The packages** are versioned 1.14.0, and are not published to a package index.
- **Upgrading from v1.13.**
  - No configuration changes. A request cut at the drain's timeout is counted `drained` in the request log and the
    outcome series, where it was `completed`.
  - A release writes `releases/release.log` and, while it switches, `releases/switching.json`, and it refuses while an
    E2 point's drop-ins are linked on cyan: `flatshare-bench e2 reset` removes them, and `--force` releases anyway.

## [v1.13](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.13) (2026-10-30): the serving engine in a container

The serving engine on vLLM's own image, verified on the box's GPU. No measured result changes.

- **The container recipe.** [`deploy/container`](deploy/container/README.md) builds an image on
  `vllm/vllm-openai:v0.32.0` with make-room's scheduler at the serving profile's settings, for a stack that runs
  vLLM's image rather than the box's systemd units. Its entrypoint gives `vllm serve` the make-room-serve drop-in's
  flags from `serving.json` and holds the engine to that file at each start, as the unit does: an engine that runs
  anything else is stopped, and the container exits 1. Every compile and JIT cache the engine's start reads is under
  one directory, so that a volume there keeps a restart warm. A test holds the entrypoint's flags to the drop-in's,
  the image's tag to the profile's vLLM version, its caches to the A100 profile's and the CUDA driver's, and the
  build context to the files the image uses.
- **Verified on the box's GPU** with rootless podman, the GPU through CDI: the check passed, the engine answered a
  completion and a chat completion at priority 0, and it stopped in 8.1 to 8.3 s. With the cache volume filled, a
  start reached the check in 90 s, as the box's unit starts its engine in 88.8 s; empty, in 708 s, and with every
  cache warm but the CUDA driver's, in 431 s. Started with a step cap of 512, the check named the step cap, and the
  container exited 1. The KV cache's CPU tier needs `--shm-size 24g`.
- **The packages** are versioned 1.13.0, and are not published to a package index.
- **Upgrading from v1.12.** Nothing changes for the box, the gateway or a configuration.

## [v1.12](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.12) (2026-10-30): drain one node, and v1.11's open items

What v1.11 left open, done: the relay's held upstream at a passed deadline, a drain of one node for maintenance, and
the pins' advisories. No measured result changes.

- **A passed deadline ends the engine's request.** When one of a stream's deadlines passed while its client's channel
  was full, the relay waited up to `stall_timeout_s` (30 s) to send the error event, and kept the engine's request
  open meanwhile for a client that would get a timeout
  ([#1](https://github.com/StevenWang-CY/flatshare/issues/1)). It now lets the upstream go first, as the gateway's
  README said it did, and the event still reaches a client that reads again within the stall timeout. A broken
  upstream is let go the same way.
- **Drain one node.** With `[[nodes]]`, a node set `draining = true`, once a reload takes the key, takes no new
  session, keeps its calls until they end, and lets its agents' sessions move by their next request. `/ready` does
  not count it, and `/nodes` shows `draining`, and `drained` once it holds no session and no request.
  `flatshare_gateway_node_draining`, which the dashboard's node panel draws, keeps `FlatshareNodeUnhealthy` quiet for
  it, so that it can be stopped and upgraded without a page. The runbook gives the steps; a drill on the two GPU nodes
  waits for a window in which both serve.
- **Pillow 13.0.0** in the bench's lock fixes 39 advisories against 12.3.0. It overrides AIPerf 0.13.0's
  `pillow~=12.3.0`, and AIPerf's image paths give the same bytes under it. The demo script takes it too.
- **The advisories the pins carry.** SECURITY.md lists the two left: setuptools 80.10.2, held by vLLM 0.32.0's
  `setuptools<81`, whose two advisories' paths do not run on the Linux hosts, and diskcache 5.6.3, which has no fixed
  release and serves only the SGLang comparator. A test holds each entry to the files it names.
- **The packages** are versioned 1.12.0, and are not published to a package index.
- **Upgrading from v1.11.**
  - `draining` is a new key of `[[nodes]]`, off by default, so no configuration needs a change. A v1.11 gateway
    refuses it as an unknown key.
  - A rules file copied from `ops/prometheus/rules.yml` needs `FlatshareNodeUnhealthy`'s expression taken from it
    again, or a drained node that is stopped pages.
  - `/nodes` answers two more fields for each node, `draining` and `drained`.

## [v1.11](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.11) (2026-10-30): a sweep of v1.10, fixed

What a sweep of v1.10's code, configuration, dependencies and documents found, fixed. No measured result changes.

- **A held request at a reload.** With `[[nodes]]`, a covering node holds back its agents' interactive and background
  requests. A reload's predecessor now refuses one it still holds with 503 `draining` and `Retry-After: 1`, as the
  gateway's README and the runbook said it would. Before, it kept the request past `drain_timeout_s` and the process
  stopped with it unanswered, logged as 499 `cancelled`.
- **The outcome alerts' first event.** `FlatshareRequestsRefusedForCapacity` and `FlatshareDeadlinesMissed` read
  `flatshare_gateway_request_outcomes_total`, whose series exist at zero from the gateway's start. On
  `requests_terminal_total`, whose series appear with their first request, `increase()` missed a class's first
  refusal or missed deadline after each start or reload. The rules' tests now hold that first event.
- **A duration past the clock.** A key that ends in `_s` took any value up to about 1.8e19 s, but the gateway adds
  these durations to the clock, which panics past about 9.2e18 s from now: with `total_timeout_s = 1e19` every
  request of its class closed unanswered, and with `placement.idle_s = 1e19` no session idled out. The gateway now
  refuses a duration over 1e9 s, about 32 years, at start.
- **Dependencies.** ctutils 0.4.4 fixes RUSTSEC-2026-0385 under the gateway's and the simulator's digests (the gateway
  compares key digests with `subtle`, not through it), and brotli 1.2.1 fixes PYSEC-2026-4322 in the bench's lock,
  where only AIPerf's compression middleware uses it. `cargo audit` reports no vulnerability. pillow stays at 12.3.0,
  which AIPerf 0.13.0 pins.
- **Documents.** CITATION's preferred citation and the report's hosting line name the current release, and a test
  holds the citation to the report's header. Four facts now match their records: PRODUCTION's two-card claim, the
  engine's start times and their builds, the killed engine's 0.13 s, and 9 of the 50 measured turns with a barge-in.
  Stale statements are brought up to date (the refused `leaseholder` policy, the gateway as the serving control plane,
  ARCHITECTURE with the Messages API, the agents' 150 s `idle_s`), each concept goes by the glossary's name, and
  `not_ready` is documented for placement refusals.
- **Headers, alerts and notes.** PROTOCOL lists the canary's `X-Flatshare-Canary`, which reaches the engine, and its
  rule that `X-Flatshare-` headers end at the gateway now names callers' headers. `FlatshareCallsRefused`'s summary
  and the runbook's row name the 502 a call gets when it is sent again after its node failed, besides the 503
  `no_call_slot`. bounds.rs's notes count the reply-caps fixture's 380 cases, not 285.
- **Tests.** `test_prose` skips make's `build/`, where `make reproduce-*` writes its reports.
- **The packages** are versioned 1.11.0, and are not published to a package index.
- **Upgrading from v1.10.**
  - During a reload with `[[nodes]]`, a request a covering node holds back gets 503 `draining` with `Retry-After: 1`
    and outcome `drained`, instead of a closed connection after `drain_timeout_s`.
  - A key that ends in `_s` over 1e9 s (about 32 years) now fails at start, and the message names the key. v1.10
    loaded one up to about 1.8e19 s, and from about 9.2e18 s it overflowed the clock. No example or deploy
    configuration comes near the bound: the longest default is background's `total_timeout_s`, 3,600 s.
  - A rules file copied from `ops/prometheus/rules.yml` needs the two outcome alerts' expressions, and
    `FlatshareCallsRefused`'s summary, taken from it again.

## [v1.10](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.10) (2026-10-30): no wall of text left

The paragraphs v1.9 left over 150 words, in the component READMEs, the runbook and the measurement documents, split
where their ideas change, and a test that holds the limit. No result, number or behaviour changes.

- **No wall of text.** No paragraph or list item of the documents outside the records holds more than 150 words: the
  67 that did, in 13 files and up to 305 words (the gateway README's), are split where their ideas change or turned
  into lists and tables, and long table cells move to numbered notes under their tables. Every number, link target
  and anchor stays, checked file by file on every commit. `test_prose.py` now holds the limit over every Markdown file
  but the records, the CHANGELOG and the simulator's written forecasts, so a new wall fails CI.
- **Names and pointers.** The README's and PRODUCTION's headline rows name the configuration to deploy, as GLOSSARY
  does; ADR-0006 keeps its old title's anchor; and the A100 kit's comments point at BENCHMARKS, where the run's layout,
  arms, analysis and Marlin dry run moved in v1.9.
- **The packages** are versioned 1.10.0, and are not published to a package index.

## [v1.9](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.9) (2026-10-30): the documents made clear and visual, and the Messages API

What a review of v1.8 for clarity, visuals and production parity found, fixed: the documents read at a glance, every
chart has a phone layout, the operator documents gain figures, the gateway's request path reads as its steps, and the
gateway routes the Anthropic Messages API that Claude Code speaks. The measured results of v1.6 to v1.8 are unchanged.

- **The Messages API.** `/v1/messages` and `/v1/messages/count_tokens` go through the gateway under the chat routes'
  contract: tenants, classes, sessions, sanitization and placement, with a class's reply cap on `max_tokens`. Claude
  Code reaches the deployment with `ANTHROPIC_BASE_URL` and the tenant's key in `ANTHROPIC_AUTH_TOKEN`; a client's
  `x-api-key` is never forwarded to the engine. The request line counts a Messages prompt with its cache reads and
  writes, as for chat. On the deployment, both ways in answered a plain, a streamed and a count_tokens request.
- **The quick start, run as written.** It installs vLLM 0.32.0 as the deployment pins it (Python 3.12, CUDA 13.0's
  compiler, FlashInfer's link to pip's CUDA runtime), probes stock vLLM before make-room, and quotes what the probe
  measured on the box: the voice turn's first token in 258–292 ms with make-room against 1,963–1,967 ms on stock vLLM
  ([`2026-10-29-quick-start`](benchmarks/results/2026-10-29-quick-start)). Tests hold the commands to the serving
  profile and the result to its logs.
- **The gateway's cost on the deployed build.** On v1.8 it adds +0.12 ms to a voice request's first token at the median,
  paired against the engine's own socket over 80 requests
  ([`2026-10-29-gateway-window`](benchmarks/results/2026-10-29-gateway-window)).
- **Figures.** ARCHITECTURE shows the serving path in four figures (the hosts and their sockets, a request's steps and
  refusals, the two cards, a release and its exits), and the runbook a triage figure from a failing call to its rows.
  REPORT charts the step cap the deployment runs, the 24-hour soak and the engine's starts. Every chart in REPORT,
  BENCHMARKS and the README has a phone layout, one label table names each configuration in GLOSSARY's terms, the
  README's animation opens on its result, and a campaign map shows which campaign measured what, on which code.
- **Documents.** The README opens on its headline in two sentences. REPORT splits its long paragraphs at their ideas and
  gives its results at a glance in a table. PRODUCTION opens on its service contract and the deployed results, with the
  envelope at a looser margin and notes for an existing LiveKit or Pipecat agent. The runbook's Failures open with an
  alert table, held to the rules by a test, and a triage figure. The gateway's bounds, SLO.md's alerts and PRIOR-ART are
  tables of short facts. Each concept goes by the one name GLOSSARY's table gives it, in the prose, the tables and the
  figures, and each fact has one home: BENCHMARKS asks one question a section, each chart above the table it draws;
  EXPERIMENTS keeps to the design; METHODS and HARDWARE's A100 part moved to the report and BENCHMARKS, and every old
  link still lands.
- **Defects.** `first_byte_timeout_s` bounded the upstream's headers, never the first byte: it is `headers_timeout_s`.
  The capacity and deadline alerts read the counter family their description and the dashboard read. The gateway
  registers its metrics once. SECURITY.md named v1.7 as the supported release at v1.8; a test now holds it to the
  CHANGELOG.
- **Code.** The gateway's request path reads as its steps, its handlers live in a routes module, imports point down its
  layers, and its log lines carry their values as fields. The Python packages import one way, with no cycles. No
  behaviour changes.
- **Policy.** CONTRIBUTING says what a minor release may change and that each such change is listed under "Upgrading",
  and that a record is frozen once a release carries it, but for its `report/` and its `figure.toml`.
- **Upgrading from v1.8.**
  - `gateway.toml`: a configuration that sets `first_byte_timeout_s` is refused at start, naming `headers_timeout_s`.
    `grep -n first_byte_timeout_s ~/.config/flatshare/gateway.toml` on cyan says whether a change is needed first.
  - `FlatshareRequestsRefusedForCapacity` and `FlatshareDeadlinesMissed` compute from `requests_terminal_total`, with
    the same thresholds and labels. A copy of `rules.yml` kept elsewhere needs the same change; `request_outcomes_total`
    is still exported.
  - New routes: `POST /v1/messages` and `POST /v1/messages/count_tokens` (404 before). Their requests count in the
    offered, admitted and terminal families, and add series to the overhead and body histograms.
  - Log lines: 18 of the gateway's messages carry their values as fields (`error`, `path`, `wait_s`, `status`,
    `timeout_s`, `max_bytes`, `failures`, `cause`) beside a constant message, and six reword where a value stood. A
    query for an old line with its value inside matches its message and reads the field; the request line and the
    "reload failed:" prefix are unchanged.
  - `make e2-figures` and `make e2-w5-figure` are gone: `make figures` draws `report-deployable` and
    `report-deployable-stages` in place of `e2-frontier.svg` and `e2-waterfall.svg`.
- **The packages** are versioned 1.9.0, and are not published to a package index.

## [v1.8](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.8) (2026-10-29): a review of v1.7 fixed, and a free line offered at once

What a review of v1.7's code, operations and documents found, fixed, and the one defect the 24-hour soak showed: LiveKit
turned calls away while the voice worker had a line free. The measured results of v1.6 and v1.7 are unchanged.

- **A free line is offered at once.** LiveKit kept a full voice worker marked full until the agents SDK's next status,
  every 2.5 s, so a call arriving then got no agent although a line was free: the 24-hour soak turned 2 calls away so.
  The worker now sends its status as soon as a call's process closes. With calls arriving faster than two lines take
  them, a free line reached LiveKit 44 ms after a call's process exited, against 1.5 s before (medians), and the calls
  turned away with a line free fell from 10 of 233 to 0
  ([`2026-10-29-status-send`](benchmarks/results/2026-10-29-status-send)).
- **Each node's agents, bounded and checked.** A node with both roles must say how many agents' sessions it takes
  beside its calls (`max_agents`, a count or `"unbounded"`), and the gateway refuses to start without it: a call node
  with no bound beside 24 agents' session trees put 37 of 177 replies over the SLO in the two-card rounds. An agent's
  request that names no session takes no slot, and runs only on a node that carries no call and is below its bound, or
  on an unbounded one, or gets 503 `no_agent_slot`. `GET /nodes` shows the placement (`calls` and
  `failover_background_max_in_flight`), and a release on cyan holds the running gateway's nodes to the serving
  profile's envelope, which the profile and `serving.json` now carry (`serving-check --nodes`): a release that finds a
  node outside it fails and switches back. cyan's `install.sh` enables at boot the tunnels `gateway.toml` reaches, so a
  reboot of cyan brings back the spare's.
- **Placements end through the placement API.** With `[[nodes]]` the gateway serves `DELETE /v1/sessions/{id}`, which
  ends a session's placement and frees its slot at once. The voice worker sends it at every bound call's end, whatever
  `FLATSHARE_EVENTS` says, and in serve mode for a call it placed whose job does not run (a refusal after asking, or an
  accept that fails), which held a call slot for `idle_s` and sent the next call to the spare. Serve mode posts
  `session_closed` alone; the turns' events go out in bench mode, for E2's lifecycle policy.
- **The voice worker reads the gateway's error codes.** It read every error's `type`, which names the kind only in a
  stream's error event: a plain 502 `upstream_unavailable` or 503 `no_healthy_node` answer did not fence the call's
  node, and a refused placement gave no reason. It reads a plain answer's `code` now, and the gateway's own code writes
  the errors its tests and the worker's fakes share (`tests/fixtures/errors.json`, `make gateway-fixtures`).
- **Make-room's preempting step in one pass.** The step in which make-room preempts counted the blocks its victims
  release by mapping every block to its holders, three times over the victim's KV; with two agents filling the serving
  profile's pool it took 27.8 times stock vLLM's step on peacock. The count now takes a block held once as released
  where it sees it, which schedules identically (the trace against the old module, and both counts compared at 677,559
  calls); the step falls to 3.5 times stock's. `make perf-make-room` times four states of the serving profile beside the
  PRs' 13, that step among them, and Figure 7 sets the costs against the box's CPU, 1.68 times as long, and the box's
  own steps. #63350 and #63351 carry the same commit, with rows for that step in their cost tables, and the patches are
  written again from the fork's backport branches, rebuilt on it; the fork's tag `flatshare-v1.7-backport` keeps the
  commits that v1.6's and v1.7's patch headers name. Both engines restart for the new module.
- **The records and the engine's upkeep.** Every committed `engine.json` is held to the reduction that wrote it, and a
  record's reduction to the settings `serving.json` pins, each exception named with why. make-room's patches are
  generated from the fork's branches, and `make compat-upstream` gives each hunk's fit to a newer vLLM at three lines of
  context and at one, and holds the backport to the PRs' heads but for the adaptations its patches' headers name.
- **Releases.** A change to Markdown alone restarts no unit (the variants' README was an input of both engines). E2's
  hint for an engine without `/server_info` says that the serving variant keeps vLLM's development routes off, and that
  a measured point runs another variant.
- **Operations.** Every alert links its row in the runbook's Failures table (`runbook_url`), and a test holds each alert
  to a row that names it. promtool cases test the refusal and deadline tickets, the dashboard's engine panels are split
  by node, and the overhead alert says which span it times. SLO.md names the voice SLO and the first-byte objective
  apart, and says which alerts burn on each.
- **Documentation.** ADRs 0007-0009 record the production program's decisions: the gateway as the serving control
  plane, two cards with a warm spare, and immutable releases with the exact serving profile. ARCHITECTURE.md describes
  the serving path. THIRD-PARTY.md lists what the serving path runs, with each licence; SECURITY.md says how a fix in a
  pinned dependency ships. PRODUCTION.md says what the voice path keeps and where its numbers ran, and the runbook each
  component's upgrade and rollback and each secret's rotation. The README opens on the results with their
  denominators, among them the 256-token step cap's 44 of 295 replies over the SLO against 94 at 512, its table adds
  the agents' TTFT p99 (the router's 193 s against the change's 111 s), and its Quick start comes before word 1,200;
  the report's abstract carries the step cap's confirmation and says that most of the two cards' gain is the
  arrangement's. The documents say the agents' token rate, not goodput, date the code of 2026-10-25 rather than call
  it final, and say once, in the report's method, that each gate was fixed before its runs.
- **Removed.** `policy = "leaseholder"`, the lifecycle policy's old name, and `realtime_max_tokens`, which
  `[classes.realtime] max_reply_tokens` replaces; a configuration that sets either is refused at start. The benchmark of
  a request's metric bookkeeping now times every series a request updates: 135-136 ns a request on one thread and
  1,684-1,850 ns with four at once on an M5 Max, against 44-46 and 881-999 ns for the old set. A test holds the idle
  deadline taking over at a stream's first frame.
- **Upgrading from v1.7.**
  - `gateway.toml`: a node with both roles needs `max_agents`, which the serving arrangement's has;
    `realtime_max_tokens` and `policy = "leaseholder"` are refused at start. A gateway that refuses its configuration
    fails the release, which switches back with the old gateway serving:
    `grep -n 'realtime_max_tokens\|leaseholder\|\[\[nodes\]\]' ~/.config/flatshare/gateway.toml` on cyan says whether a
    change is needed first.
  - `serving.json` changes (the envelope, and make-room's module and patches), so both engines and the voice worker
    restart at the first deploy.
  - With `[[nodes]]`, a release enables cyan's tunnel to each node `gateway.toml` names; nothing disables one, so when a
    window on Gray ends, `systemctl --user disable --now flatshare-tunnel@gray` (the runbook's Gray's windows).
  - A client that ends its sessions with `session_closed` needs no change.
- **The packages** are versioned 1.8.0, and are not published to a package index.

## [v1.7](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.7) (2026-10-29): two cards, measured against a gate fixed in advance

The rest of the external audit of v1.5: the two-card comparison its resource gate asked for, the serving profile's step
cap, the gateway and the deployment on two GPU nodes, a failover drill and a soak of the arrangement, and the
operator's service contract. The single-card headline of v1.6 is unchanged.

- **Two cards.** Three rounds, each on a new sample of the agents and on the two hosts in turn, measured each card's
  part of every arrangement that [the resource gate](benchmarks/quality/RESOURCE-GATE.md) compares
  ([`2026-10-28-rounds`](benchmarks/results/2026-10-28-rounds), whose `gate.py` gives the verdict). Consolidation,
  both calls and four agent sessions on one card with the change beside a card of agents alone, did 34.2% more
  background work than the best dedicated arrangement, with 5 of 177 replies over the SLO: past the gate's 20%
  hurdle. Stock vLLM's shared card qualifies at two sessions, and that arrangement did 27.3% more, so most of the gain
  is the arrangement's; the change's own value is 5.4%. One call and 12 sessions on each card qualified on neither
  engine. The dedicated card's agents did the most at 14 sessions, not 24, where their pool thrashes.
- **The serving profile's step cap.** The profile now runs the change at 256 tokens a step without a per-request
  threshold, the cap the sweep's screens chose ([`2026-10-27-sweep`](benchmarks/results/2026-10-27-sweep)).
  Five rounds on new samples confirmed it beside 24 sessions: 44 of 295 replies over each round's SLO, against 94
  at the box's 512-token cap, fewer in every round, at 7% less of the agents' token rate. A profile whose step cap
  is below the engine's request slots is refused, since vLLM 0.32.0 refuses to start with it.
- **The gateway on two nodes.** A node bounds the agents' sessions it holds (`max_agents`, 503 `no_agent_slot` with
  `Retry-After: 1` beyond it), and `[placement] calls = "fill"` places the calls on the first node with a free call
  slot, so that the second node is a warm spare. A node covers for a failed one, holding its agents, until the calls
  moved to it have ended, and a warm spare while it carries any call. The configuration is in
  [`deploy/cyan/gateway.example.toml`](deploy/cyan/gateway.example.toml) and the runbook's
  [Two GPU nodes](ops/RUNBOOK.md#two-gpu-nodes).
- **The service contract.** [docs/PRODUCTION.md](docs/PRODUCTION.md#two-cards) says what the two cards accept (two
  calls, 16 agent sessions), what each side gets, what is refused under overload, and how a failure is recovered.
- **The failover drills** ([`2026-10-28-node-failover`](benchmarks/results/2026-10-28-node-failover)). With the
  first card's engine killed and then stopped, both calls' next turns went to the spare within 0.7 and 1.9 s, and none
  of the 300 turns after failed; the drill's own client, which sent a wrong model id, is fixed, and the drill now waits
  for whole turns before it breaks a node. Two smokes of the arrangement
  ([`2026-10-28-two-node-smoke`](benchmarks/results/2026-10-28-two-node-smoke)) set the placement's idle time to 150 s:
  a client that never ends its sessions holds an ended session's agent slot until it idles out.
- **The soak.** 24 hours of the serving arrangement ([`2026-10-28-soak`](benchmarks/results/2026-10-28-soak)), after two
  hours of voice alone with the same calls set its SLO at 1,976 ms: 74 of 5,058 replies over it (1.5%), 5.5% in the
  worst hour; 476 calls, 72 refused with both lines busy, 2 turned away with a line free, LiveKit still holding the
  worker full from its last status ([docs/PRODUCTION.md](docs/PRODUCTION.md#calls-what-the-box-admits)), and none
  failed; nothing restarted or paused, and the spare carried no call. The agents did 9,300 tokens a second in the first
  hour, the replay's start, where the rounds' windows sit, and 582–5,402 in each hour after.
- **The endpointing sweep** ([`2026-10-28-endpointing`](benchmarks/results/2026-10-28-endpointing)). The fixed
  endpointing every measurement runs waits out the VAD's silence and about 0.2 s more, LiveKit's delay never binding:
  0.3 s of silence answers 0.27 s sooner at the median, and cut one line in 40 short at its first sentence. The
  measured default stays. Answering the audit's last question, no adaptive scheduler is built: the two-card envelope
  is set by the speech servers' share of the GPU, which no scheduling in the engine removes.
- **The report** is at version 1.7: the two cards (§6.8) and the step cap, with the rule that carries its 256-token
  steps (§6.9), operating two cards (§7), adaptive prefill control and what is new (§8), among its limitations the
  quality gate, the two ways to a larger pool and the next vLLM release (§9), which v1.6 measured and left out of it,
  and its header, which v1.6 left at 1.5.
- **Corrections.** The prior art said vLLM's open fixes preempt only at the request cap, but #40087 also preempts for a
  waiting request's KV, one victim at a time; [docs/PRIOR-ART.md](docs/PRIOR-ART.md) and the report's §8 now say how
  #63350 differs. They also misdated `prefill_schedule_interval`'s single-engine change and called SGLang's #45686 a
  draft that keeps KV in host memory. [SECURITY.md](SECURITY.md) named v1.2 and a gateway without body bounds or
  deadlines, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) a two-node design that never ran. The example
  `gateway.toml` now holds the serving arrangement, two engine alerts hold for each node with `[[nodes]]`, and the
  quality gate's records no longer carry a hash of the judge's login. [README.md](README.md)'s limits now include the
  quality gate the served model failed, which only the report gave.
- **Upgrading from v1.6.** The serving variant's engine flags change, so a deploy restarts the engine. `gateway.toml`
  takes `max_agents` on a node and `[placement] calls`; without them nothing changes. The voice worker takes
  `FLATSHARE_VAD_MIN_SILENCE` and `FLATSHARE_ENDPOINTING_MIN_DELAY`, bounded and refused by name, and
  `flatshare-bench soak` and `flatshare-voice arrivals` are new. With `[[nodes]]`, Prometheus scrapes each node's
  engine as its own job, labeled with its node ([`prometheus.example.yml`](ops/prometheus/prometheus.example.yml)),
  which the engine's alerts now group by.
- **The packages** are versioned 1.7.0, and are not published to a package index.

## [v1.6](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.6) (2026-10-28): what an external audit found, fixed

No headline measurement changed. An external audit of v1.5 found defects in the gateway and the deployment and asked
for measures the documents lacked; this release fixes the first and adds the second.

- **Reply caps on every route.** The realtime cap was skipped when a body carried any reply-limit field, so a chat
  request with only `max_output_tokens`, which vLLM's chat model ignores, ran uncapped. A capped body now gets its
  route's own limit fields, at the lower of the client's limit and the cap, and loses the other routes'; a limit or a
  choice count given twice, or as anything but a whole number, is refused; `n` and `best_of` are bounded by
  `max_choices`, 1 by default; and each class can have a cap of its own. A CPU test reads every body of the
  gateway's fixture with vLLM 0.32.0's own request models.
- **Bounded intake.** Global, per-class and per-tenant limits on requests in flight are taken before a body is read,
  a budget of buffered body bytes is taken as the body arrives, and the admission queues hold at most `max_queued`
  requests; refusals come at once, with `Retry-After`. Through a reload the successor counts what its predecessor
  still holds, so the limits and an admission policy hold over both processes.
- **Deadlines.** Each class has deadlines for the upload, the queue, the response headers, the first content, the gap
  between frames and the whole request. A timeout cancels the engine's work, gives back what the request held and
  ends it once; a retry runs within the request's first deadline, and only before the client has seen any output.
- **Accounting that counts failures.** The gateway counts each request offered, admitted and ended, by outcome, and
  `completed` means a 2xx response that ended. The service level is per class, good over valid requests, so
  failures, refusals and hangs count against it, and the burn-rate alerts read it; admission coverage is exported. An
  inference canary, sent only after 30 s without a successful request, drives the engine's states (warming, serving,
  degraded, drained) and `/ready`. `make check-rules` runs promtool over the rules and their tests.
- **The serving profile.** [`deploy/profiles/rtx-5060-ti-serve.toml`](deploy/profiles/rtx-5060-ti-serve.toml) names
  the configuration to deploy, the change with two preemptions a step, a 48-token window and no reserve, with every
  setting and revision it depends on. The `make-room-serve` variant runs it, and `serving-check` fails an engine
  start whose settings differ from it.
- **The voice worker's modes.** Serve mode admits calls by capacity, at most the profile's number of calls, and takes
  none while the gateway's `/ready` says the engine does not serve. Bench mode, for the measurement callers, stays the
  default. Real microphones get settings of their own, apart from the synthetic callers', and unmeasured.
- **Releases.** [`deploy/deploy.sh`](deploy/deploy.sh) releases each host as an immutable release with a manifest (the
  commit, the hashes of every artifact and environment, the lockfiles, the model and patch revisions, the effective
  configuration), checks it before switching, switches an atomic link, restarts only what changed, switches back on
  any failure, and keeps earlier releases for `--rollback`.
- **Measurement.** The README, PRODUCTION, BENCHMARKS, METHODS and the report give the share of replies over the SLO
  beside the median of the runs' p95: 286 of the change's 885 replies (32%) are over it, where the SLO allows 5% and
  tuned vLLM leaves 90%. The report says what the agents' 97% measures, the token rate of a replay that runs no task,
  and that the change's fifteen runs replay five samples of the agents. The callers now end a reply when the agent's
  state says it has finished, not after a second of silence, and record each reply's pauses and concealed audio.
- **Records.** How long the voice agent's first sentences are
  ([`2026-10-27-first-sentence`](benchmarks/results/2026-10-27-first-sentence): 15 tokens at the median, 39 at most,
  all within the 48-token window), and speech synthesis and recognition on the CPU
  ([`2026-10-27-cpu-audio`](benchmarks/results/2026-10-27-cpu-audio): too slow for the voice SLO on these hosts).
- **Gates fixed in advance.** The quality gate's criteria and the two-card comparison's resource gate
  ([`benchmarks/quality`](benchmarks/quality)) were committed before their runs.
- **Upgrading from v1.5.** A request with `n` or `best_of` above 1 now gets 400 unless `max_choices` allows it, as
  does a limit field given twice or as a string; `/ready` answers 503 until the first inference succeeds after a
  start; realtime streams end after 30 s without a frame or 300 s in all; the outcome `error` is now
  `transport_error`. A host moves to releases once, with `deploy/deploy.sh HOST --adopt`.
- **A test** read a non-dumpable process's environment through `access()`, which answers yes to root; it now reads it.
- **The packages** are versioned 1.6.0, and are not published to a package index.

## [v1.5](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.5) (2026-10-27): the technical report, rewritten

No measurement changed. The technical report is rewritten in full, every number in it is recomputed from the committed
records, and its figures are drawn from them.

- **The report.** [docs/REPORT.md](docs/REPORT.md) is now a technical report of ten sections and six appendices: the
  problem and the evidence for each claim, the background a vLLM reader needs, the two rules with their pseudocode,
  the implementation and what the rules cost a scheduler step, the evaluation setup, the results under claims, how to
  operate the change, related work with numbered references, the limitations, and appendices on reproducing the
  results, every run by round, the configurations, the failure drills, the A100 forecast and the report's versions.
  It calls #63350's rule make-room, #63351's the prefill hold and the evaluated module the generated scheduler, and
  uses no campaign codes.
- **Its figures.** Twelve figures are drawn from the records by `flatshare_bench.report` (`make report-figures`), with
  phone layouts of the first and the headline. The tests check every committed SVG against a fresh render and pin the
  values the report quotes. In the method figure, the waiting turn now waits until agent blocks are freed, which a
  preemption for another request's growth does as well as a request's end.
- **Corrections.** v1.4 said each rule is necessary. The hold is, but in the campaign before the main evaluation a
  class-aware KV reserve made the room in place of make-room, and the report now says so. With every running slot
  taken, the hold still covers a running request's first tokens; only a waiting request or a KV load holds nothing
  then, where v1.4 and ARCHITECTURE.md said nothing is held. On vLLM 0.32 the prefill case's reply tokens come
  60–66 ms apart, where v1.4 gave make-room alone's 60–65 ms. Distributions over replies now pool every reply, 59 a
  run, as each run's p95 does, and two, four and six callers are compared on the same three samples of the agents.
- **A record.** The main evaluation's voice replies with their output tokens
  ([`replies.tsv`](benchmarks/results/2026-10-24-e2-w5/replies.tsv), 5,571 replies), which the compact records did not
  keep, extracted by `replies.py` from the v1.0 release's raw records.
- **Docs.** The glossary gives the report's names for the rules and the module, and names the A100's tuned vLLM by
  its step cap. The links into the report follow its new sections, and `CITATION.cff` gives the report as the
  preferred citation.
- **The packages** are versioned 1.5.0, and are not published to a package index.

## [v1.4](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.4) (2026-10-27): a rewritten history, and the code reorganized

No measurement changed, and every report and figure regenerates from the same records.

- **The history.** The 740 commits up to v1.3 are now 104, each titled "[Area] What it does" with a body that says why.
  Every file of every commit was cleaned of a host login, a lab host's address and the name of a LiveKit key; none of
  them was a secret, and none had been in the tree since v1.3. Each tag, and each revision a record names, keeps
  exactly its tree under a new hash, and [docs/REVISIONS.md](docs/REVISIONS.md) gives each record's revision its
  commit in this history. A clone made before 2026-10-27 has to be made again.
- **The figures.** Where a figure shows the change in two blues, a note says the darker one is the configuration to
  deploy, and the voice figures state their SLO. On the A100 both of the PRs' rows are drawn alike, and tuned vLLM's
  512-token row says it runs the RTX 5060 Ti's tuning. A row's name breaks to keep its run count on its last line, and
  the method figure's leader to (c) runs straight down. The README's links say where they lead.
- **The simulator.** The replay is split into its loop, its clock, the sessions' trees, the E1 arms' retention and the
  working-set accounting, and the regime map is split out of the hardware profiles. One function prices a lone prefill
  chunk, where the regime map and the forecast each had a copy. Every committed output is byte for byte as before:
  E1's tables, the three forecasts, the calibration report and the A100 analysis, 158 files. `flatshare-sim` refuses a
  count, duration, ratio or capacity that describes no run, where `--concurrency 0` wrote an empty table and exited 0.
- **The gateway.** The engine client and the response relay are modules of their own. The metric series a request
  updates under fixed labels are resolved at start, which takes its bookkeeping from 170 to 49 ns, and from 2,440 to
  1,481 ns with four threads at once. A session's lifecycle events reach the admission point in the order they were
  accepted, where two concurrent posts could arrive reversed. A lifecycle throttle without a step budget is a
  configuration error that names the key, where the core panicked.
- **The harness.** `e2/run.py` keeps the point's steps and its records (647 of its 1,472 lines); the engine check,
  cyan's settings and the tools' processes are modules of their own. A reader of a few metrics parses their lines
  alone, so the KV sample an agent point takes twice a second went from 1.07–1.09 ms to 72–74 µs. The voice worker
  checks its settings before it registers with LiveKit, so a misspelled value stops it instead of failing every call.
  The A100 scripts read a campaign's points from `campaign.sh`, so a campaign that saves audio reads as complete. Each
  CLI action is a function with one error path. A point gives AIPerf a temp directory of its own, which goes once
  AIPerf has stopped: AIPerf leaves its IPC directory and the run's dataset map in the temp directory, and cyan's
  /tmp held 234 maps (264 MB). The tests' stand-in servers stop within 10 ms and leave nothing behind, which took
  the bench suite from 222–232 s to 181–199 s on the same machine.
- **Fixes.** `gpu-guard` refuses to start a GPU unit when `nvidia-smi` cannot list the GPU's processes, where it
  passed with none listed. The gateway's tests remove the sockets that ended test runs left, which had built up to
  322 files, and `rehearsal-ssh` prints its message with its apostrophes.
- **Packages** versioned 1.4.0, and not published to a package index.

## [v1.3](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.3) (2026-10-26): the box's running share, and a public release

- **The box's running share, measured (RS).** Five rounds on the RTX 5060 Ti of the frozen engine and the PRs'
  48-token configuration, on W5's arm files and samples, with the engine's free KV sampled every 0.5 s: two calls found
  KV short 8.1% of the time beside the frozen engine and 6.1% beside the PRs', where the simulator gives the box 17%,
  and the A100 measured 5.7% and 6.3% on the same samples. The documents that gave the box the simulator's share now
  give the measurement, and the report no longer calls the A100 less saturated at the box's pool.
- **A logo and a social preview**, drawn by `flatshare_bench.brand` (`make brand`) in light and dark, with a small mark
  and a 1280×640 preview; the tests check each file against the generator.
- **Community files.** A security policy, a contributing guide, the Contributor Covenant 2.1, `CITATION.cff`, issue
  forms, a pull request template, this changelog and a map of the documentation.
- **The documents.** The README leads with the logo and the result and states that Flatshare is an independent
  project. #63350 and #63351 are described as pull requests proposed to vLLM and under review, and each document opens
  with the reader it serves.
- **Hardening.** The gateway binds its own Unix sockets only in a directory closed to other users, the tunnel and the
  forwarder create theirs private, `flatshare-bench e2 run` writes the gateway's configuration private from its
  creation, and `build.py` replaces the scheduler module whole. The tunnel's unit no longer restarts a configuration it
  refuses, and cyan's install says when the gateway's binary is not the source's version.
- **Efficiency.** The gateway finds the session a new one replaces through an ordered index (6.1–6.6 µs to
  0.22–0.24 µs with a full table) and caps a realtime reply within its one body rewrite. A point's counters read each
  metric snapshot once (the gateway's from 15 ms to 1.4 ms a point).
- **Fixes.** flatshare-core's `estimate_error` example describes an export without long requests instead of
  panicking; a trace line that does not parse is named; the exposition parser reads timestamps, exemplars and escaped
  label values; `long_traces.py` keeps its corpus to the traces it names; `flatshare-forward` takes bracketed IPv6
  addresses; the callers refuse settings no call can run with; every command-line argument has help; the Makefile
  parses under GNU Make 3.81; the scripts default to the hosts' SSH aliases.
- **Packages** versioned 1.3.0, each with its license in its sdist and wheel (flatshare-bench's with the fonts' OFL),
  and not published to a package index. Asset: `flatshare-raw-records-rs.tar.gz`.

## [v1.2](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.2) (2026-10-26): Flatshare, and the demo in its original form

No measurement changed: every record keeps the names it was recorded under, and the reports and figures regenerate
from them.

- **The name.** The repository, crates, Python packages, command-line tools, systemd units, metrics, environment
  variables and host paths are renamed from Leaseholder to Flatshare (`flatshare-gateway`, `flatshare_bench`,
  `flatshare.target`, `FLATSHARE_*`, `~/flatshare`); the old repository URL redirects. The code still reads the
  records' former names, which [the glossary](docs/GLOSSARY.md#former-names) lists, and
  [the runbook](ops/RUNBOOK.md#deploy) says how a host deployed before the rename moves.
- **The demo.** The README's animation and the demo video draw the v1.0 demo's dashboard, a panel per engine side by
  side, in the documents' figure style, in light and dark themes and with a phone layout.
- **The documents.** A pass for names, numbers and tense after the A100 run. The box's 17% running share is the
  simulator's, on the box's measured costs (W5's points did not sample it); the A100 agents' TTFT p50 moved by −1.2 to
  +1.8 s; the Colab host had 167 GiB of RAM.
- **The limit of overlapping replies, tested.** The behavior suite on vLLM's real scheduler runs the model behind the
  documents' limit: with the 48-token window, six agents keep 82% of stock's prompt tokens beside a reply every 200
  steps, 9% beside one every 50 and none beside one every 40.
- **The tunnel's former keys.** `deploy/authorize-tunnel.sh` also removes the entries of cyan's tunnel keys under their
  former names.
- **Packages** versioned 1.2.0. Asset: `flatshare-demo.mp4`.

## [v1.1](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.1) (2026-10-26): the A100 run

- **The A100 run.** The release's kit ran on an A100 SXM4 80 GB on Google Colab: E2's core like for like with the RTX
  5060 Ti (the box's model, KV pool, step cap, CPU tier, agents and callers, W5's five samples of the agents' traces),
  the probes, E0, a calibration point against the forecast written before the run, and five arms after the core.
- **What it measured.** With #63350 and #63351 the voice p95 went from 4,986 to 1,634 ms, under the card's SLO of
  1.67 s in each of five rounds, and the agents' tokens a second were 94–111% of the stock engine's round by round. At
  vLLM's default 2,048-token steps, which give the card's agents 1.4 times the throughput, the PRs left the p95
  0.14–0.17 s over the SLO. At a batch of 256 requests, the PRs' code path with its switches off made 1.018 of vLLM's
  output tokens a second. The KV case's waits came out about 30% longer than forecast, since the card's decode
  attention reads KV at 42% of its bandwidth.
- **The kit.** GPU units that name the driver's libraries where the linker does not search; the batch arm's tokenizer
  from the engine's own snapshot; `bench_host.py` and a forecast placed on the measured host; arms for tuned vLLM, the
  PRs' values at its step cap and FlashInfer's attention.
- **The analysis and figures.** `make reproduce-a100` from the raw records; AIPerf's dataset cache for a round's later
  points; the README's chart and the demo's closing card with both GPUs.
- Assets: the demo video with the A100's runs, and the A100 host's raw records.

## [v1.0](https://github.com/StevenWang-CY/flatshare/releases/tag/v1.0) (2026-10-26): the scheduler change, re-measured and hardened

- **W5's re-measure.** E2's headline on the fixed make-room, in rounds over five samples of the agents' traces: two
  callers beside 24 long-context agent sessions on one RTX 5060 Ti 16 GB. With a 48-token window the PRs gave a voice
  p95 of 2,236 ms against an SLO of 2,034 ms, against tuned vLLM's 10,477 ms, with the agents at 97% of their
  throughput; #63350 alone gave 5,085 ms and a 16-token window 3,094 ms. In a pool with room it met the SLO in one round
  of five, and with 4 and 6 callers it was 0.27 s and 0.26 s over each one's SLO. Five more rounds on the final code
  (FB) gave 2.27 s.
- **The change's three gaps, fixed**, and the first tokens in flight counted under asynchronous scheduling, as #63351's
  revision of 2026-10-25 counts them.
- **make-room is the PRs' diff**, backported to vLLM 0.32.0 with a patch of its own; a trace test holds it to the PRs'
  patches step for step, and CI runs the behavior suite and the trace on vLLM's real scheduler.
- **Production hardening (W6).** The GPU servers on Unix sockets in a private directory, forwarded over SSH socket to
  socket; the gateway under a systemd socket unit, with reloads that refuse no connection, typed stream errors and 503s
  while the engine restarts, and `/metrics` behind a key; deploys that sync the commit and nothing else; Prometheus
  rules and a Grafana dashboard; failure drills on the final build.
- **Upkeep checks.** `make perf-make-room` and `make compat-upstream`.
- **Other GPUs.** Hardware profiles, a support matrix, single-host mode, and a kit for an A100 run, rehearsed on the
  box, with its forecast written first.
- **The documents.** A glossary, one home per number, and the technical report.
- Assets: W5's and FB's raw records, the demo video and its points' run directories.

## [v0.3](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.3) (2026-10-24): E2 complete, and the scheduler change

- **E2**, 129 points over 15 campaigns on one RTX 5060 Ti 16 GB (W2 to W4). In the discriminating scenario, with
  #63350 and #63351 and a 48-token window, the voice p95 was 2.19 s over three runs against an SLO of 2.02 s, where
  tuned vLLM gave 10.1 s. The prefill cap (#63351) is necessary; the gateway's lifecycle-timed mechanisms could not act
  while the agents' KV filled the pool.
- **E1, in the simulator.** Knowing when an agent returns from its tool buys under 9% of resume TTFT at p50, and
  session history nothing. This release corrected the simulated oracle's prefetch timing, and the corrections were
  posted on the vLLM and router threads that had the first numbers.
- **In the tree.** make-room generated from the installed vLLM's source; arm 9's gateway half; placement across GPU
  nodes (built, not measured); E2's harness with repeats, the stage waterfall, censored failed turns, compact records
  and `make reproduce-e2`; session ids scoped to their tenant, and SGLang's control fields stripped.
- **Dropped:** leases, prefetch, the per-node sidecar, cache contracts and the recompute ledger
  ([ADR-0006](docs/adr/0006-the-engine-change-carries-the-realtime-tier.md)).
- **Known issues then:** three gaps in the scheduler change, fixed in v1.0; the tunnels' TCP ports on cyan reachable by
  the shared hosts' other users, with the engine in vLLM's development mode, which v1.0 addressed by moving the servers
  to Unix sockets in a private directory and development mode into a drop-in for measured boxes.
- Assets: the E2 demo video, every E2 point's raw run directory, and, added after the release, the simulator's
  calibration exports.

## [v0.1](https://github.com/StevenWang-CY/flatshare/releases/tag/v0.1) (2026-10-23): the E2 baselines

- **E2's arms 1 to 8**, 45 points. In the standard scenario every vLLM arm held the voice SLO at 8 agent sessions,
  the best were 100–220 ms over it at 16, and none held it at 24. In the discriminating scenario every stock, engine
  and router arm left the voice p95 at 5.7–12.1 s, and preempt-resubmit at 2.6 s.
- **The simulator**, calibrated within 8% of resume TTFT at 8 and 16 sessions, and E1's first, negative result (its
  numbers corrected in v0.3).
- **In the tree.** The gateway with E2's admission baselines; E2's harness; the engine variants for vLLM #54530 and
  SGLang; the lifecycle policy, the voice worker's lifecycle events and make-room, merged but not yet measured.
