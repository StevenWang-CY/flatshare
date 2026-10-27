# Changelog

What each release changed, newest first, summarized from its release notes; each release's page has its assets (raw
records, the demo video). The project was called Leaseholder through v1.1, and those releases' tools, assets and links
keep that name. Numbers are each release's own, with the campaign that measured them: v1.0 re-measured v0.3's
headline in rounds, and [BENCHMARKS.md](docs/BENCHMARKS.md) has every run.

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
