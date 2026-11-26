# The A100 run and the single-host kit

What ran the main comparison on a rented A100 on 2026-10-26, for a reader who checks that run or repeats it on another
GPU. One host is both GPU node and control host ([single-host mode](../../docs/HARDWARE.md#single-host-mode)), and
[HARDWARE.md](../../docs/HARDWARE.md#running-on-another-gpu-step-by-step) gives the steps. The run's records are
[`2026-10-26-a100`](../results/2026-10-26-a100/README.md), and [`deploy/a100`](../../deploy/a100) prepares the host.

| Path | What it is |
|---|---|
| `run.sh` | The run, block by block, so that a stop anywhere leaves whole comparisons: a smoke test, the probes, E2's five rounds and a calibration point, then the optional arms by name |
| `probes.sh`, `probe_results.py` | The probe blocks on one engine configuration at a time, and their table |
| `calibration.sh` | One of the simulator's calibration points on this host |
| `batch.sh` | The scheduler's cost with 256 requests in flight |
| `bench_host.py` | The host's link and CPU, which a forecast for that host reads |
| [`campaigns/`](campaigns), [`arms/`](arms) | The run's campaign files and the arm files written for it. The core replays [E2's own arm files](../e2/README.md) |
| `analysis.py` | The run's report from a host's records (`make reproduce-a100`) |
| `forecast.md`, `forecast-80gb.md`, `forecast-80gb-on-host.md` | The simulator's forecasts for the A100 40 GB, for the 80 GB card the run got, and for that card on its measured host (`make forecast-a100`) |
| `marlin-session2.sh`, `bench_marlin.py` | The A100's code path dry-run on the box ([its record](../results/2026-10-25-marlin-dry/README.md)) |
| [`rehearsal/`](rehearsal/README.md) | The whole kit rehearsed on the box before the rented hours |
