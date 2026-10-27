# Development tasks. Each Python package is tested in its own uv environment, because the engine variants'
# tests, the benchmark tools and the voice worker have disjoint dependencies (a vLLM checkout, AIPerf, LiveKit).

RUFF := uvx ruff@0.17.1
# A literal '#' for the recipes and functions below: make before 4.3, macOS's 3.81 among them, reads one inside a
# function call as the start of a comment.
HASH := \#
VLLM_REF := v0.32.0
VLLM_SRC ?= $(HOME)/.cache/flatshare/vllm-$(VLLM_REF)
# Every Python package except the engine variants' tests (test-vllm); each one has a `test` extra.
PY_PACKAGES := $(filter-out python/flatshare_vllm,$(patsubst %/pyproject.toml,%,$(wildcard python/*/pyproject.toml)))
# The Python that make lint checks: the packages, and the scripts of flatshare-sim, E2, the A100 run, the replies'
# measure, the deployment (the hosts' scripts without a suffix among them, found by their shebang) and the engine
# variants (not the recorded runs' own scripts under benchmarks/results).
PY_SOURCES := python crates/flatshare-sim/scripts $(sort $(wildcard benchmarks/e2/*.py benchmarks/a100/*.py \
  benchmarks/replies/*.py deploy/*.py deploy/bin/*.py deploy/a100/*.py deploy/phastform/variants/*/*.py) \
  $(shell grep -ls '^$(HASH)!.*python' deploy/bin/*))
# E2's committed records (flatshare-bench e2 export), one directory per point under its campaign, and where
# reproduce-e2 writes their report.
E2_RESULTS := benchmarks/results/2026-10-23-e2
E2_REPORT ?= build/reproduce-e2
# W5's re-measure, whose report reads its own voice-alone runs for the SLO.
E2_W5_RESULTS := benchmarks/results/2026-10-24-e2-w5
E2_W5_REPORT ?= build/reproduce-e2-w5
# The A100 run's E2 points (benchmarks/a100/run.sh on Google Colab, 2026-10-26), under e2/ beside its other records:
# the core's five rounds, and tuned vLLM's two and the PRs' values' at its step cap, which pair with the core's first two:
# the paired design reproduce-e2-a100 reports. A' (32 sessions) is a workload of its own, which make reproduce-a100
# reports from the raw records.
E2_A100_RESULTS := benchmarks/results/2026-10-26-a100
E2_A100_POINTS = $(wildcard $(foreach c,a100-r a100-tuned-r a100-prtuned-r,$(E2_A100_RESULTS)/e2/$(c)[0-9]/*/run.json))
E2_A100_REPORT ?= build/reproduce-e2-a100

.PHONY: check lint lint-rust lint-python fmt test test-rust test-vllm test-make-room test-trace test-make-room-ci \
  test-python schema metrics \
  reproduce-e1 reproduce-calibration reproduce-e2 reproduce-e2-w5 reproduce-e2-a100 figures e2-figures e2-w5-figure \
  e2-gpus-figure method-figure report-figures brand \
  demo-video forecast-a100 forecast-fitted profiles reproduce-a100 check-a100-calibration

check: lint test

lint: lint-rust lint-python

lint-rust:
	cargo fmt --all -- --check
	cargo clippy --all-targets --locked -- -D warnings

lint-python:
	$(RUFF) check $(PY_SOURCES)
	$(RUFF) format --check $(PY_SOURCES)

fmt:
	cargo fmt --all
	$(RUFF) check --fix $(PY_SOURCES)
	$(RUFF) format $(PY_SOURCES)

test: test-rust test-vllm test-python

test-rust:
	cargo test --locked

# The engine variants' build tests read vLLM's scheduler from a source checkout: no GPU or torch.
$(VLLM_SRC):
	git -c advice.detachedHead=false clone -q --depth 1 --branch $(VLLM_REF) --filter=blob:none --sparse https://github.com/vllm-project/vllm $@
	git -C $@ sparse-checkout set vllm

test-vllm: $(VLLM_SRC)
	cd python/flatshare_vllm && VLLM_SRC=$(VLLM_SRC) uv run -q --extra test pytest -q

# The CPU suites' imports, checked before pytest runs: without one of them pytest would skip a suite, not fail it.
CPU_SUITE_IMPORTS = cd python/flatshare_vllm && VLLM_TARGET_DEVICE=cpu PYTHONPATH=$(VLLM_SRC) $(TORCH_PYTHON) -c \
  'import pytest, torch, vllm.v1.core.sched.scheduler' || { echo "$(TORCH_PYTHON) cannot import pytest, torch or \
  vLLM's scheduler from $(VLLM_SRC)" >&2; exit 1; }

# The make-room variant's behavior suite on vLLM's real v1 scheduler, on CPU, outside make check: it needs an
# interpreter with CPU torch 2.13 and vLLM 0.32.0's Python dependencies, and runs on the source checkout.
test-make-room: $(VLLM_SRC)
	@test -n "$(TORCH_PYTHON)" || { echo "usage: make test-make-room TORCH_PYTHON=<a python with CPU torch 2.13 and vLLM 0.32.0's dependencies>" >&2; exit 2; }
	@$(CPU_SUITE_IMPORTS)
	cd python/flatshare_vllm && VLLM_TARGET_DEVICE=cpu PYTHONPATH=$(VLLM_SRC) $(TORCH_PYTHON) -m pytest -q -p no:cacheprovider \
	  tests/test_make_room.py tests/test_perf_schedule.py

# make-room against the PRs' own patches, step for step, on the same CPU environment (python/flatshare_vllm/README.md):
# the trace that keeps the variant equal to #63350 and #63351, with the PRs' one-line mutants as its negative control.
# TRACE_SEEDS sets the random workloads' seeds; TRACE_REFERENCE, a generated module, replaces the PRs' as the other side.
TRACE_SEEDS ?= 20
test-trace: $(VLLM_SRC)
	@test -n "$(TORCH_PYTHON)" || { echo "usage: make test-trace TORCH_PYTHON=<a python with CPU torch 2.13 and vLLM 0.32.0's dependencies> [TRACE_SEEDS=100]" >&2; exit 2; }
	@$(CPU_SUITE_IMPORTS)
	cd python/flatshare_vllm && VLLM_TARGET_DEVICE=cpu PYTHONPATH=$(VLLM_SRC) TRACE_SEEDS=$(TRACE_SEEDS) $(TORCH_PYTHON) -m pytest -q -p no:cacheprovider \
	  tests/test_trace.py

# Both CPU suites as CI's make-room job runs them: in the environment python/flatshare_vllm/requirements-cpu.txt pins
# (Python 3.12 on Linux x86_64, CPU torch), which uv builds in CPU_ENV, and with every test required to run
# (FLATSHARE_NO_SKIPS: a skip fails the session, tests/conftest.py).
CPU_ENV ?= build/cpu-env
$(CPU_ENV)/synced: python/flatshare_vllm/requirements-cpu.txt
	uv venv -q --allow-existing --python 3.12 $(CPU_ENV)
	uv pip sync -q --python $(CPU_ENV)/bin/python --torch-backend cpu $<
	@touch $@

test-make-room-ci: $(CPU_ENV)/synced
	FLATSHARE_NO_SKIPS=1 $(MAKE) --no-print-directory test-make-room test-trace \
	  TORCH_PYTHON=$(abspath $(CPU_ENV))/bin/python

# Integration tests need hosts (a LiveKit server, a running engine); CI runs everything else.
test-python:
	@for pkg in $(PY_PACKAGES); do \
	  echo "== $$pkg"; (cd $$pkg && uv run -q --extra test pytest -q -m "not integration") || exit 1; \
	done

# The engine change's upkeep checks, outside make check (python/flatshare_vllm/README.md says how to read them).
.PHONY: perf-make-room compat-upstream

# The change's cost per step: Scheduler.schedule() on CPU for stock vLLM and for make-room with its switches off and in
# the configuration to deploy, in the 13 steady states of the PRs' cost tables (tests/perf_schedule.py; on a 2.1 GHz
# Xeon 14 minutes, and 22 for a baseline, the negative control included). It prints the tables, writes PERF_OUT, fails
# a state whose off/stock or on/off grew past PERF_BASELINE's tolerance (measured again first), and then runs the
# check's negative control against the same baseline. UPDATE=1 measures a new baseline into PERF_BASELINE instead and
# needs PERF_BESIDE, what else ran on the machine. The tracked baseline is peacock's; another machine's goes under
# build/, as build/perf-baseline.json.
PERF_BASELINE ?= benchmarks/perf/make-room-schedule.json
PERF_OUT ?= build/perf-make-room.json
perf-make-room: $(VLLM_SRC)
	@test -n "$(TORCH_PYTHON)" || { echo "usage: make perf-make-room TORCH_PYTHON=<a python with CPU torch 2.13 and vLLM 0.32.0's dependencies> [PERF_BASELINE=FILE] [UPDATE=1 PERF_BESIDE=TEXT]" >&2; exit 2; }
	@if [ "$(filter 1,$(UPDATE))" = 1 ] && [ -z "$(strip $(PERF_BESIDE))" ]; then \
	  echo "make perf-make-room UPDATE=1 needs PERF_BESIDE=<what else ran on the machine>" >&2; exit 2; fi
	cd python/flatshare_vllm && VLLM_TARGET_DEVICE=cpu PYTHONPATH=$(VLLM_SRC) $(TORCH_PYTHON) tests/perf_schedule.py \
	  $(if $(filter 1,$(UPDATE)),baseline $(abspath $(PERF_BASELINE)) --beside "$(PERF_BESIDE)",check $(abspath $(PERF_BASELINE)) --out $(abspath $(PERF_OUT)))
	cd python/flatshare_vllm && FLATSHARE_PERF=1 FLATSHARE_PERF_BASELINE=$(abspath $(PERF_BASELINE)) VLLM_TARGET_DEVICE=cpu PYTHONPATH=$(VLLM_SRC) \
	  $(TORCH_PYTHON) -m pytest -q -s -p no:cacheprovider tests/test_perf_control.py

# Whether #63350 and #63351 merge into vLLM's main and, with TORCH_PYTHON (an interpreter for vLLM main's scheduler and
# tests), pass their tests there on CPU; and whether make-room's patches apply to vLLM's newest release. It keeps a
# blobless vLLM clone in VLLM_UPSTREAM, and needs the network and git 2.38 or newer (GIT names one).
VLLM_UPSTREAM ?= $(HOME)/.cache/flatshare/vllm-upstream
GIT ?= git
compat-upstream:
	uv run -q --no-project --python 3.12 python python/flatshare_vllm/scripts/compat_upstream.py --clone $(VLLM_UPSTREAM) \
	  --git $(GIT) $(if $(TORCH_PYTHON),--python $(TORCH_PYTHON))

schema:
	cargo run -q -p flatshare-proto --bin flatshare-proto-schema

# crates/flatshare-gateway/metrics.txt, the gateway's exposition before any traffic, after a change to its metrics.
metrics:
	FLATSHARE_GATEWAY_WRITE_METRICS=1 cargo test -q -p flatshare-gateway --lib \
	  metrics::tests::the_checked_in_exposition_is_current

# E1's tables (docs/BENCHMARKS.md's E1 section, crates/flatshare-sim/README.md's other hardware) from the 32x corpus,
# which stays outside the repository (`flatshare-bench scale-traces` writes it; python/flatshare_bench/README.md): the
# regime map, the realistic regime over 20 seeds, the other GPUs at the 32x replay and at their own scalings, and the
# miss ratios with the idle service. Usage: make reproduce-e1 CORPUS=<its directory>.
E1_REPORT ?= build/reproduce-e1
E1_SIM = target/release/flatshare-sim
reproduce-e1:
	@test -n "$(CORPUS)" || { echo "usage: make reproduce-e1 CORPUS=<the 32x corpus directory>" >&2; exit 2; }
	cargo build --release --locked -p flatshare-sim
	@mkdir -p $(E1_REPORT)
	$(E1_SIM) regime $(CORPUS) --csv $(E1_REPORT)/regime.csv --markdown $(E1_REPORT)/regime.md >/dev/null
	$(E1_SIM) regime $(CORPUS) --sessions 8,12,16,24 --cpu-ratio 4 --pause-sources trace \
	  --seeds $$(seq 11 30 | paste -sd, -) --csv $(E1_REPORT)/realistic.csv --markdown $(E1_REPORT)/realistic.md \
	  >/dev/null
	$(E1_SIM) hardware $(CORPUS) --csv $(E1_REPORT)/hardware.csv --markdown $(E1_REPORT)/hardware.md >/dev/null
	$(E1_SIM) hardware $(CORPUS) --fit --csv $(E1_REPORT)/hardware-fit.csv --markdown $(E1_REPORT)/hardware-fit.md \
	  >/dev/null
	$(E1_SIM) mrc $(CORPUS) --service idle --csv $(E1_REPORT)/mrc.csv --markdown $(E1_REPORT)/mrc.md >/dev/null
	@ls $(E1_REPORT)/*.md

# The A100 run's forecast, benchmarks/a100/forecast.md, from the 32x corpus (outside the repository) and its 48
# traces with the largest contexts, E2's discriminating corpus, which long_traces.py links into FORECAST_LONG. GPU names
# the card (flatshare-sim's profile): GPU=a100-80gb writes benchmarks/a100/forecast-80gb.md, for the A100 SXM4 80 GB,
# which analysis.py reads for records whose card is one. HOST_JSON, the host.json benchmarks/a100/bench_host.py writes
# on the card's host, places the card there (flatshare-sim forecast --host) and writes the forecast beside the one made
# before the run, with -on-host in its name, so that one stays as it was. Usage: make forecast-a100 CORPUS=<the 32x
# corpus directory> [GPU=a100-80gb] [HOST_JSON=<host.json>].
FORECAST_LONG ?= build/weka-x32-long
GPU ?= a100-40gb
FORECAST_MD = benchmarks/a100/forecast$(if $(filter-out a100-40gb,$(GPU)),-$(GPU:a100-%=%))$(if $(HOST_JSON),-on-host).md
forecast-a100:
	@test -n "$(CORPUS)" || { echo "usage: make forecast-a100 CORPUS=<the 32x corpus directory> [GPU=a100-80gb] [HOST_JSON=<host.json>]" >&2; exit 2; }
	cargo build --release --locked -p flatshare-sim
	uv run -q --no-project python benchmarks/e2/long_traces.py $(CORPUS) $(FORECAST_LONG) 48 >/dev/null
	@mkdir -p benchmarks/a100
	$(E1_SIM) forecast $(CORPUS) $(FORECAST_LONG) --gpu $(GPU) $(if $(HOST_JSON),--host $(HOST_JSON)) --markdown $(FORECAST_MD) \
	  >/dev/null
	@ls $(FORECAST_MD)

# The same forecast with a section on the card's fitted costs, into OUT: the card's E0 summary.json (COSTS; the
# records' runs/a100/block1/frozen-e0/summary.json) and the step-time model fit_step_cost.py fits on its calibration
# point (STEPS; make reproduce-a100 writes it to build/reproduce-a100/steps.json) place the calibration point and the
# core at the box's pool at 24 to 40 sessions, and the section names the count whose running share is closest to the
# box's in the core. GPU is the A100 SXM4 80 GB here unless it names another card, and HOST_JSON places it on its
# host as for forecast-a100. Usage: make forecast-fitted CORPUS=<the 32x corpus directory> COSTS=<E0's summary.json>
# STEPS=<steps.json> [GPU=a100-80gb] [HOST_JSON=<host.json>] [OUT=build/forecast-fitted.md].
forecast-fitted: GPU = a100-80gb
forecast-fitted: OUT = build/forecast-fitted.md
forecast-fitted:
	@test -n "$(CORPUS)" -a -n "$(COSTS)" -a -n "$(STEPS)" || \
	  { echo "usage: make forecast-fitted CORPUS=<the 32x corpus directory> COSTS=<E0's summary.json> STEPS=<steps.json> [GPU=a100-80gb] [HOST_JSON=<host.json>] [OUT=build/forecast-fitted.md]" >&2; exit 2; }
	cargo build --release --locked -p flatshare-sim
	uv run -q --no-project python benchmarks/e2/long_traces.py $(CORPUS) $(FORECAST_LONG) 48 >/dev/null
	@mkdir -p $(dir $(OUT))
	$(E1_SIM) forecast $(CORPUS) $(FORECAST_LONG) --gpu $(GPU) --costs $(COSTS) --step-costs $(STEPS) \
	  $(if $(HOST_JSON),--host $(HOST_JSON)) --markdown $(OUT) >/dev/null
	@ls $(OUT)

# The A100 run's analysis (benchmarks/a100/analysis.py; docs/HARDWARE.md, the analysis) from a single host's records
# as deploy/a100/pull.sh copies them, and the 32x corpus, into build/reproduce-a100/a100.md, with the bench venv for
# E2's report. check-a100-calibration runs its calibration code on the box's W2 points (the calibration exports, as
# reproduce-calibration takes them) and checks that it gives the box's rows. Usage: make reproduce-a100 RECORDS=<dir>
# CORPUS=<dir>; make check-a100-calibration EXPORTS=<dir> CORPUS=<dir>.
A100_REPORT ?= build/reproduce-a100
reproduce-a100:
	@test -n "$(RECORDS)" -a -n "$(CORPUS)" || \
	  { echo "usage: make reproduce-a100 RECORDS=<the host's records> CORPUS=<the 32x corpus directory>" >&2; exit 2; }
	cargo build --release --locked -p flatshare-sim
	cd python/flatshare_bench && uv run -q python ../../benchmarks/a100/analysis.py report $(abspath $(RECORDS)) \
	  --corpus $(abspath $(CORPUS)) --flatshare-sim $(abspath $(E1_SIM)) --out $(abspath $(A100_REPORT)) \
	  --flatshare-bench flatshare-bench

check-a100-calibration:
	@test -n "$(EXPORTS)" -a -n "$(CORPUS)" || \
	  { echo "usage: make check-a100-calibration EXPORTS=<the runs' exports> CORPUS=<the 32x corpus directory>" >&2; exit 2; }
	cargo build --release --locked -p flatshare-sim
	uv run -q --no-project python benchmarks/a100/analysis.py calibration-check $(EXPORTS) --corpus $(CORPUS) \
	  --flatshare-sim $(E1_SIM) --out build/check-a100-calibration

# The simulator's calibration and its validation against W0 (crates/flatshare-sim/README.md, and BENCHMARKS' table of W2's points)
# from six measured runs' AIPerf exports and /metrics snapshots, one directory per run as `scripts/calibration.py export`
# writes them, and the 32x corpus. Usage: make reproduce-calibration EXPORTS=<dir> CORPUS=<dir>.
CALIBRATION_REPORT ?= build/reproduce-calibration
reproduce-calibration:
	@test -n "$(EXPORTS)" -a -n "$(CORPUS)" || \
	  { echo "usage: make reproduce-calibration EXPORTS=<the runs' exports> CORPUS=<the 32x corpus directory>" >&2; exit 2; }
	cargo build --release --locked -p flatshare-sim
	uv run -q --no-project python crates/flatshare-sim/scripts/calibration.py report $(EXPORTS) $(CORPUS) $(E1_SIM) \
	  $(CALIBRATION_REPORT) >/dev/null
	@ls $(CALIBRATION_REPORT)/calibration.md

# E2's report from the committed records, with the bench venv: frontier.md, frontier.csv, a frontier figure per workload
# (W4's four, W5's six) and waterfall.svg, the files each set of records keeps in its report/, which the bench's tests
# check and E2_REPORT=<that report/> rewrites. It prints the files it writes.
reproduce-e2:
	@cd python/flatshare_bench && uv run -q flatshare-bench e2 report \
	  $(abspath $(dir $(wildcard $(E2_RESULTS)/*/*/run.json))) --out $(abspath $(E2_REPORT))

# The same of W5's re-measure, from its own records.
reproduce-e2-w5:
	@$(MAKE) --no-print-directory reproduce-e2 E2_RESULTS=$(E2_W5_RESULTS) E2_REPORT=$(E2_W5_REPORT)

# The same of the A100's core, tuned vLLM and the PRs' values at its step cap, from the A100 run's records
# (benchmarks/results/2026-10-26-a100).
reproduce-e2-a100:
	@cd python/flatshare_bench && uv run -q flatshare-bench e2 report \
	  $(abspath $(dir $(E2_A100_POINTS))) --out $(abspath $(E2_A100_REPORT))

# E2's figures in the docs, docs/figures/e2-frontier.svg and e2-waterfall.svg, from the records of the configurations an
# operator could deploy (the ablations left out), each with all its runs. The README's animation is demo.py's
# (demo-video).
E2_DEPLOYABLE := a1-voice disc-frozen-s24 disc-a4-b1024-t0-s24 disc-a6-reserve2048-s24 disc-a7b-sglang-hicache-s24 \
  disc-a8c-resubmit-s24 disc-a9-s24 disc-pr-ab48-s24
E2_STAGES := a1-voice disc-frozen-s24 disc-a6-reserve2048-s24 disc-a8c-resubmit-s24 disc-a9-s24 disc-pr-ab48-s24
e2-figures:
	@cd python/flatshare_bench && uv run -q flatshare-bench e2 report \
	  $(abspath $(dir $(foreach p,$(E2_DEPLOYABLE),$(wildcard $(E2_RESULTS)/*/$(p)/run.json)))) \
	  $(foreach p,$(E2_STAGES),--waterfall $(p)) --out $(abspath build/e2-figures) >/dev/null
	@mkdir -p docs/figures
	cp build/e2-figures/frontier-weka-x32-long.svg docs/figures/e2-frontier.svg
	cp build/e2-figures/waterfall.svg docs/figures/e2-waterfall.svg

# REPORT.md's chart, docs/figures/e2-voice-p95.svg and its phone layout e2-voice-p95-narrow.svg, from W5's records:
# the configurations figure.toml names, each run in the same rounds over the same five samples of the agents' traces.
e2-w5-figure:
	@cd python/flatshare_bench && uv run -q flatshare-bench e2 report \
	  $(abspath $(dir $(wildcard $(E2_W5_RESULTS)/*/*/run.json))) \
	  --dots $(abspath $(E2_W5_RESULTS)/figure.toml) --out $(abspath build/e2-w5-figure) >/dev/null
	@mkdir -p docs/figures
	cp build/e2-w5-figure/voice-p95.svg docs/figures/e2-voice-p95.svg
	cp build/e2-w5-figure/voice-p95-narrow.svg docs/figures/e2-voice-p95-narrow.svg

# The README's chart of both GPUs, docs/figures/e2-voice-p95-gpus.svg and its phone layout e2-voice-p95-gpus-narrow.svg:
# the rows the A100 run's figure.toml names, W5's from the box's records and the A100's from its own, each group
# against its own voice alone's SLO (the A100 ran the box's arm files, so each group names the GPU its points ran on).
e2-gpus-figure:
	@cd python/flatshare_bench && uv run -q flatshare-bench e2 report \
	  $(abspath $(dir $(wildcard $(E2_W5_RESULTS)/*/*/run.json) $(E2_A100_POINTS))) \
	  --dots $(abspath $(E2_A100_RESULTS)/figure.toml) --out $(abspath build/e2-gpus-figure) >/dev/null
	@mkdir -p docs/figures
	cp build/e2-gpus-figure/voice-p95.svg docs/figures/e2-voice-p95-gpus.svg
	cp build/e2-gpus-figure/voice-p95-narrow.svg docs/figures/e2-voice-p95-gpus-narrow.svg

# The README's method figure, docs/figures/method.svg and its phone layout method-narrow.svg: the voice agent and the
# coding agents on one GPU, and the two scheduler rules of #63350 and #63351, drawn by flatshare_bench.diagrams.
method-figure:
	cd python/flatshare_bench && uv run -q python -m flatshare_bench.diagrams $(abspath docs/figures)

# The technical report's figures, docs/figures/report-*.svg with the phone layouts of its first figure and its headline
# (report-stages-narrow.svg, report-headline-narrow.svg), drawn by flatshare_bench.report from the committed records
# each cites: W4's, W5's and the A100's E2 points, the probes, the window sweep, W5's voice replies and the scheduler's
# cost per step (its data.py names them).
report-figures:
	cd python/flatshare_bench && uv run -q python -m flatshare_bench.report.figures $(abspath docs/figures)

# Every figure under docs/figures that its records or its code draw; the bench's tests check each is current.
figures: e2-figures e2-w5-figure e2-gpus-figure method-figure report-figures

# The logo, the mark and the social preview under docs/brand, drawn by flatshare_bench.brand (docs/brand/README.md);
# the bench's tests check each is current.
brand:
	cd python/flatshare_bench && uv run -q --extra brand python -m flatshare_bench.brand $(abspath docs/brand)

# The demo video, DEMO_OUT, and the README's animation, docs/figures/e2-demo.webp with its dark theme and its phone
# layout (e2-demo-dark.webp, e2-demo-narrow.webp, e2-demo-narrow-dark.webp), from the calls that demo.campaign's two
# points saved (the v1.0 release's leaseholder-raw-records-demo.tar.gz has them): make demo-video DEMO=<the directory
# of DEMO_POINTS>. demo.py draws them (benchmarks/e2/README.md, The demo video) with the names the README's chart gives
# the two configurations, W5's report for the SLO and the closing card, and the A100's (reproduce-e2-a100's, committed
# as its report/) for the card's second group: the same two configurations on the A100.
DEMO_POINTS := w5-disc-frozen-s24 w5-disc-pr-ab48-s24
DEMO_OUT ?= build/flatshare-demo.mp4
demo-video:
	@test -n "$(DEMO)" || { echo "make demo-video DEMO=<the directory of $(DEMO_POINTS)>" >&2; exit 2; }
	@mkdir -p $(dir $(DEMO_OUT)) docs/figures
	uv run -q benchmarks/e2/demo.py $(addprefix $(abspath $(DEMO))/,$(DEMO_POINTS)) \
	  --labels "Tuned vLLM 0.32" "#63350 and #63351" --summary $(E2_W5_RESULTS)/report/frontier.csv \
	  --also $(E2_A100_RESULTS)/report/frontier.csv \
	  --heading "A voice agent and coding agents on one GPU" --turns 10 --out $(DEMO_OUT) \
	  --excerpt docs/figures/e2-demo

# The frozen box's hardware profile against deploy/phastform (deploy/profile.py): `make profiles` checks that it
# renders today's files byte for byte, and `make profiles UPDATE=1` writes them from deploy/templates after a
# template changes.
profiles:
	uv run -q --no-project --python 3.12 python deploy/profile.py $(if $(UPDATE),update,check) rtx-5060-ti
