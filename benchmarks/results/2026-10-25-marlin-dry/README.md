# The Marlin dry run on the box (2026-10-25)

The records behind [HARDWARE.md](../../../docs/HARDWARE.md#the-marlin-dry-run-on-the-box)'s dry run: the A100 run's code
path (the FP8 checkpoint through vLLM 0.32.0's `MarlinFP8ScaledMMLinearKernel`, W8A16, FlashAttention 2 on the LBHNC
layout, the CPU tier, make-room) on Phastform's RTX 5060 Ti, before the A100. Session 1 ran HARDWARE.md's steps on the
box's deployment of W5 (the servers on TCP, 10:28-10:56Z); session 2's GEMM step ran `bench_marlin.py` as corrected on
the same day (each call's GPU time inside a CUDA graph, 11:38Z); and session 2 ran in full, `marlin-session2.sh`, on the
box's deployment of W6 at v0.3-191-g987fd09 (the servers on sockets, 16:31-16:51Z), in `session2/`.

| Path | What |
|---|---|
| `session.log` | session 1, step by step: the engine switches, the engine's kernel lines, `/server_info`'s `linear_backend`, `smoke.sh`, E0's last lines, the probes' lines |
| `e0-deepgemm/`, `e0-marlin/` | `lh-bench tiers --reps 3` from cyan on the frozen engine with DeepGEMM (the box's own path) and with `--linear-backend marlin`: `summary.json` holds the cold, GPU-hit and restore fits |
| `engine-lines/` | each engine's own account of its kernels (its journal, from its start): the frozen engine on DeepGEMM and on Marlin, make-room on Marlin |
| `probes/` | make-room at the PRs' values on Marlin: the KV case twice, the prefill case twice, the calls' tail for 240 s with 12 agents |
| `gemm.csv`, `gemm.log` | `bench_marlin.py`: Marlin, BF16 and DeepGEMM at Qwen3-4B's four linear shapes for M of 1 to 1,024, each call's GPU time in µs; the log's last lines are each M's Marlin over BF16, summed over the shapes |
| `session2/session.log` | session 2, step by step: the GEMM step's sums, the engine at a 3 GiB pool, the probes' lines, the E2 point's campaign |
| `session2/gemm.csv` | session 2's GEMM step: each M's sum over the shapes within 0.005 of the first run's, its cells a median 0.5% from them |
| `session2/pool3g/` | the probes at a 3 GiB pool (1,364 blocks) and a 16,384-token context, which they size themselves from: the KV case, the prefill case, the calls' tail for 120 s beside 13 agents |
| `session2/marlin-dry/` | the E2 point on Marlin ([`marlin-dry.campaign`](../../a100/marlin-dry.campaign)): the PRs' 48-token configuration in the discriminating scenario on W5's round-1 sample of the agents (seed 11), its compact record as `lh-bench e2 export` writes it, which `lh-bench e2 report` reads |
