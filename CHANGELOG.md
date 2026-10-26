# Changelog

What each release changed, newest first, summarized from its release notes; each release's page has its assets (raw
records, the demo video). The project was called Leaseholder through v1.1, and those releases' tools, assets and links
keep that name. Numbers are each release's own, with the campaign that measured them: v1.0 re-measured v0.3's
headline in rounds, and [BENCHMARKS.md](docs/BENCHMARKS.md) has every run.

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
