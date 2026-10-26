# Contributing to Flatshare

How to build and test the repository, what a change needs before review, and where questions go. Contributions are
under the project's [Apache-2.0](LICENSE) license, and everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## Build and test

You need Rust 1.98 (`rust-toolchain.toml` pins it), [uv](https://docs.astral.sh/uv/), Python 3.12, git and GNU Make
3.81 or newer: CI runs Ubuntu 24.04's 4.3, and the 3.81 that macOS ships reads the Makefile too. Nothing CI runs needs a
GPU.

| Command | What it runs | What it needs |
|---|---|---|
| `make check` | `make lint` (rustfmt, clippy with warnings as errors, ruff's check and format) and `make test` (every Rust test, the engine variants' build tests against vLLM v0.32.0's source, each Python package's unit tests): CI's `rust` and `python` jobs | The network on first use, to make a sparse vLLM checkout in `~/.cache/flatshare` |
| `make fmt` | rustfmt and ruff's fixes | |
| `make test-make-room-ci` | The make-room variant's behavior suite and its trace against the PRs' own patches, on vLLM's real scheduler with CPU torch: CI's `make-room` job | Linux x86-64, for the environment `python/flatshare_vllm/requirements-cpu.txt` pins (uv builds it in `build/cpu-env`) |
| `make test-make-room TORCH_PYTHON=PY`, `make test-trace TORCH_PYTHON=PY` | The same two suites with an interpreter of your own | An absolute path to a Python with CPU torch 2.13 and vLLM 0.32.0's Python dependencies |
| `make perf-make-room TORCH_PYTHON=PY` | The change's cost per scheduler step against the tracked baseline, measured on peacock | The same interpreter; on another machine, measure its own baseline first ([how](python/flatshare_vllm/README.md#upkeep-the-changes-cost-and-its-fit-upstream)) |
| `make compat-upstream` | Whether #63350 and #63351 still merge into vLLM's main, and whether make-room's patches apply to vLLM's newest release | The network and git 2.38 or newer |
| `make schema`, `make metrics` | Regenerate the wire types' JSON Schemas and the gateway's metrics exposition; a test fails until a changed type or metric is regenerated | |
| `make reproduce-e2`, `make reproduce-e2-w5`, `make reproduce-e2-a100` | Rebuild E2's reports from the committed records | |
| `make figures` | Regenerate the documents' figures from the records | |

What CI does not run:

- **A GPU node with the deployment** for the probes (`probe.py`, `tail.py`, `knee.sh`), E0, E2 points and the failure
  drills. [ops/RUNBOOK.md](ops/RUNBOOK.md) deploys the box's hosts, and [docs/HARDWARE.md](docs/HARDWARE.md) another
  GPU or a single host.
- **The integration tests** (`pytest -m integration`), which need a LiveKit server on Linux with unprivileged user
  namespaces, or fetch their own dependencies.
- **The corpus**: `make reproduce-e1 CORPUS=DIR` and `make forecast-a100 CORPUS=DIR` need the 32× agent traces, which
  stay outside the repository ([how to make them](python/flatshare_bench/README.md#quickstart-cyan)).

## Changes

- **The scheduler change.** Its source is the two pull requests proposed to vLLM, #63350 and #63351; make-room's
  patches are rewritten from their branches ([one source](python/flatshare_vllm/README.md#one-source-for-the-change)).
  A change to the rules belongs on those pull requests, or in an issue here first.
- **Records.** Never edit a file under `benchmarks/results`. A new measurement gets a dated directory of its own,
  with a README that says what ran, on which revision, and the command that recomputes its tables.
- **Documents.** A number comes from a record, a test or a script, and says which; a term is the one
  [the glossary](docs/GLOSSARY.md) gives; [docs/README.md](docs/README.md) says which document a fact belongs in.

## Commits and pull requests

- One logical change per commit, with an imperative subject ("Bound the budget cap's held tokens by the step's
  budget") and a body that says why, wrapped at about 100 columns.
- Sign off every commit (`git commit -s`). The `Signed-off-by` line certifies the
  [Developer Certificate of Origin 1.1](https://developercertificate.org/) for the change.
- Open pull requests against `main`. CI's three jobs must pass, and the pull request says what changed, why, and
  what you ran.

## Questions

Ask in an [issue](https://github.com/StevenWang-CY/flatshare/issues/new/choose) with the question form. Report a
vulnerability privately instead, as [SECURITY.md](SECURITY.md) describes.
