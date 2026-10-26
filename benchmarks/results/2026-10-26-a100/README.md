# The A100 run on Google Colab (2026-10-26)

The records of [the A100 run](../../../docs/HARDWARE.md#the-a100-run): the release's kit
([`benchmarks/a100`](../../a100), `deploy/a100`) on one A100 SXM4 80 GB, a Colab runtime with high RAM, from its
creation at 04:07Z to its stop at 09:12Z. The core ran the v1.0 release's tree (`src/DEPLOYED`: v1.0); the arms after
it ran this repository's `run.sh`, `batch.sh` and `probes.sh`, with the arm files
[`a100-tuned-s24.toml`](../../a100/arms/a100-tuned-s24.toml),
[`a100-pr-ab48-2048-s24.toml`](../../a100/arms/a100-pr-ab48-2048-s24.toml) (written during the run),
[`a100-aprime-frozen-s32.toml`](../../a100/arms/a100-aprime-frozen-s32.toml) and
[`a100-aprime-pr-ab48-s32.toml`](../../a100/arms/a100-aprime-pr-ab48-s32.toml). The host: 12 vCPUs of a 2.20 GHz Xeon
(Cascade Lake, PCIe 3.0 root ports), 167 GB of RAM, driver 580.82.07; the node rendered from the A100 profile
(`a100-40gb`, which keeps the box's pool on either A100) under flatshare-svc. The runtime used 34.35 of Colab's compute
units (37.38 at its creation, 3.03 at its stop), at 6.77 an hour.

`make reproduce-a100 RECORDS=<these records, raw> CORPUS=<weka-x32>` writes [`a100.md`](a100.md) from the release's
raw-records asset (`leaseholder-raw-records-a100.tar.gz`, every file the host wrote but the units' journal), and
`make reproduce-e2-a100` the E2 report of the core and of the two arms paired with its first rounds from the compact
records here ([`report/`](report/frontier.md)).

| Path | What |
|---|---|
| `node/` | the node's facts and resolved values (`host.json`, `node.json`, `staged.json`, `engine.yaml`), `bootstrap.sh`'s log, the preflight (`preflight.json`), the warm starts (`warm.json`) and FlashInfer's JIT check |
| `blocks.log`, `run.log` | `run.sh`'s blocks, the core's and the arms', with each step |
| `block0/` | `smoke.sh` with the listener audit, and a 2-turn call |
| `block1/` | E0 on the frozen engine (`frozen-e0`, 3 repetitions), the KV case and the prefill case three times each, the calls' tail twice |
| `block2/` | stock vLLM at its default 2,048/0 (KV x3, tail x2) and at 1,536/768 (KV x1, tail x2) |
| `block3/` | the PRs' values (make-room, no reserve, a 48-token window): KV x3, prefill x3, tail x2 |
| `calibration-s16/`, `calibration.log` | the calibration point, AIPerf's replay of the whole corpus at 16 sessions on a fresh frozen engine: AIPerf's summary and log (its export and the engine's /metrics are the raw records'), from which [`steps.json`](steps.json) is the step model fitted |
| `host/` | `bench_host.py`'s `host.json`: the host's CPU at a fixed Python workload and its link at the CPU tier's page sizes, each way |
| `batch/` | the batch arm: `vllm bench serve`'s results for each engine's two clients and the engine's running requests a second |
| `backend/` | the backend arm: the KV case three times on the frozen engine with FlashInfer's attention, and its log, the first attempt's refusal in it |
| `e2/` | every complete E2 point as `flatshare-bench e2 export` keeps it, by campaign (`a100-r1` to `-r5`: the core; `a100-tuned-r1`, `-r2`; `a100-prtuned-r1`, `-r2`; `a100-aprime-r1`), with each campaign's log and frontier, and `durations.tsv`, each point's and engine switch's measured seconds |
| `report/` | `make reproduce-e2-a100`: `flatshare-bench e2 report` over the core, tuned vLLM and the PRs' values at its cap |
| `a100.md`, `steps.json` | `make reproduce-a100` over the raw records |
| `figure.toml` | the rows of the README's chart of both GPUs (`make e2-gpus-figure`) |

Home paths are cut to `~` (Colab's `~` among them) and the workstation's records directory to `$RECORDS`; no
account name is left, and LiveKit's key is redacted as its secret is.

## The run

| Step | UTC | Minutes | |
|---|---|---:|---|
| `push`, `bootstrap.sh stage` | 04:21–04:28 | 7 | the source and the data; uv, Python 3.12 and the venvs, Rust and the gateway, LiveKit, FlashInfer's sampler (103 s), the control plane up |
| `bootstrap.sh gpu` | 04:28–04:31 | 3 | refused at first: the driver's libraries were outside the linker's search (`/usr/lib64-nvidia`), registered by hand, then passed; the kit names that directory since |
| `bootstrap.sh warm` | 04:31–04:40 | 9 | the four compile configurations from empty caches |
| block 0 | 04:40–04:43 | 3 | the smoke test, the listener audit, a 2-turn call |
| blocks 1–3 | 04:43–05:25 | 42 | E0 and the probes on the frozen engine, stock vLLM and the PRs' values |
| block 4 | 05:25–05:51 | 26 | E2 round 1 (seed 11) |
| block 5 | 05:51–06:08 | 17 | the calibration point |
| blocks 6–9 | 06:08–07:53 | 105 | E2 rounds 2 to 5 (seeds 12 to 15), 26 minutes each |
| `bench_host.py` | 07:53 | 0.2 | the host's CPU and link, the GPU units stopped |
| batch | 07:53–08:03 | 9 | three engines, two clients each |
| tuned vLLM | 08:03–08:23 | 20 | vLLM's default 2,048/0, rounds 1 and 2 (seeds 11, 12) |
| the PRs' values at 2,048/0 | 08:23–08:43 | 20 | arm 8, added at 08:16 once tuned vLLM's first round was in: rounds 1 and 2 |
| A' round 1 | 08:43–09:05 | 22 | 32 sessions, frozen then the PRs' values (seed 11); round 2 would have ended past the deadline |
| FlashInfer's attention | 09:05–09:12 | 7 | the frozen engine with `--attention-backend FLASHINFER`, the KV case x3; refused by its own check at first (vLLM 0.32 names FlashInfer in another log line), fixed and run again from 09:09 |
| the last pull, `colab stop` | 09:12 | 1 | 21 E2 points complete; the GPU units stopped at 09:12:01, the runtime at 09:12:42 |

## The numbers

The analysis is [`a100.md`](a100.md); [docs/HARDWARE.md](../../../docs/HARDWARE.md#what-it-measured-2026-10-26) gives
each against the forecast and the box, and [the report](../../../docs/REPORT.md#57-another-gpu) what they say.
