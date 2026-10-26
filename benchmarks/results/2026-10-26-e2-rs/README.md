# E2 records of RS, the box's running share (2026-10-26)

The compact records of RS's five rounds ([E2](../../../docs/EXPERIMENTS.md#e2-voice-under-agent-load)), kept as
[W5's records](../2026-10-24-e2-w5) keep each point: `run.json`, `arm.toml`, `engine.json`, `summary.json` and the
callers' `bots/summary.json` and `bots/turns.jsonl`. RS measures on the box what W5's and FB's points did not sample:
the running share, the share of the engine's free-KV samples, taken every 0.5 s while the agents ran, in which free KV
was under two calls' 2,048 tokens ([the glossary](../../../docs/GLOSSARY.md#measures)). Until RS, the documents gave
the box the simulator's 17%, on the box's measured costs.

10 points, all on vLLM 0.32.0 and Qwen3-4B-Instruct-2507-FP8 on Phastform's RTX 5060 Ti, with both hosts at v1.2, in
the campaigns of [`benchmarks/e2`](../../e2/README.md):

| Campaign | Points | What |
|---|---:|---|
| [`rs-r1`](rs-r1) to [`rs-r5`](rs-r5) | 2 each | The frozen box and the PRs' 48-token configuration on the `make-room` variant, round N on the agents' sample of seed 10 + N, as W5's, FB's and the A100 core's round N; odd rounds run the frozen box first |

Each round ran W5's arm files, `w5-disc-frozen-s24.toml` and `w5-disc-pr-ab48-s24.toml`, with the campaign's
`agents-seed` and `kv-sample-s 0.5`. The campaign logs name the revision the arm files were read at: v1.2-2-g8004f26a
for round 1 and v1.2-31-g22f70eab for rounds 2 to 5, between which only the documents, the logo and the issue forms
changed, nothing a campaign runs; round 1's log opens with the dry run that checked the five campaigns at
v1.2-1-gd4b776b1, which measured nothing. cyan's gateway names itself 1.1.0 in `run.json`'s `versions`: it was built
at the rename, and nothing it is built from has changed since but the crates' version number. No turn of the 500
failed.

`running_share.py` gives BENCHMARKS' RS table from these records and the A100 core's
([`../2026-10-26-a100/e2`](../2026-10-26-a100/e2)): each configuration's running share on both cards, and its V2V p95
and the agents' tokens a second over the calls' first 240 s on the box, then one row a round and configuration.
`flatshare-bench e2 report` reads these records as it reads W5's. The raw records (the gateway's request log, AIPerf's
exports, the metric snapshots and the engine's free-KV samples, the voice worker's journal, the GPU's samples and the
tools' logs) are not committed: they are the v1.3 release's asset `flatshare-raw-records-rs.tar.gz`.
