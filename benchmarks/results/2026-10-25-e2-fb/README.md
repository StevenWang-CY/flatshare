# E2 records of FB, the final code against W5 (2026-10-25)

The compact records of FB's five rounds ([E2](../../../docs/EXPERIMENTS.md#e2-voice-under-agent-load)), kept as [W5's records](../2026-10-24-e2-w5) keep each point: `run.json`, `arm.toml`, `engine.json`, `summary.json` and the callers' `bots/summary.json` and `bots/turns.jsonl`. FB re-measures, on the final code, the three configurations of W5's own campaign that the README and the PRs cite: voice alone, the frozen box, and the PRs' 48-token configuration in the discriminating scenario (24 agent sessions of `weka-x32-long`). Round N replays W5's sample of the agents for round N (`agents-seed` 10 + N) with W5's arm files and rotates the three's order by one from round to round, and every call opens with an unmeasured warm-up turn, as in W5's points, so that each round compares with W5's round of the same number.

15 points, all on vLLM 0.32.0 and Qwen3-4B-Instruct-2507-FP8 on Phastform's RTX 5060 Ti, with both hosts at v0.3-191-g987fd09, in the campaigns of [`benchmarks/e2`](../../e2/README.md):

| Campaign | Points | What |
|---|---:|---|
| [`fb-r1`](fb-r1) to [`fb-r5`](fb-r5) | 3 each | Voice alone, the frozen box, and the PRs' 48-token configuration on the `make-room` variant |

What differs from W5's points, all of it the final code's:
- **The transport** (W6): the GPU servers on Unix sockets in a directory only the deploying user may enter, cyan's tunnel one SSH session per connection to each socket, and the gateway on its socket unit, where W5's points went through TCP forwards of one SSH session.
- **The budget cap's count of a reply's first tokens.** It counts the tokens in flight too ([after W5](../../../docs/BENCHMARKS.md#after-w5-the-first-tokens-in-flight)), as #63351 does since its revision 9013ed433, so the hold ends at its window instead of a step past it; with the 48-token window that step touched 1 of W5's 2,469 replies.
- **The harness**: the callers join their rooms a quarter second apart, and the window a point measures ends at the callers' own summary.

`compare.py` gives BENCHMARKS' FB table from these records and W5's: each configuration's median and range in both, and the differences round by round, then one row a round and configuration. The question FB answers is whether W5's numbers hold on the final code, which that table answers, so these records keep no report of their own; `flatshare-bench e2 report` reads them as it reads W5's. The raw records (the gateway's request log, AIPerf's exports, the metric snapshots, the voice worker's journal, the GPU's samples and the tools' logs) are not committed: they are kept for the v1.0 release's assets.
