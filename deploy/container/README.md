# The serving engine in a container

The box's engine as an image, for a stack that runs vLLM's image rather than the box's systemd units. It is vLLM
0.32.0's own image, `docker.io/vllm/vllm-openai:v0.32.0`, with make-room's scheduler at the serving profile's settings.
make-room runs the pull requests proposed to vLLM,
[vllm-project/vllm#65282](https://github.com/vllm-project/vllm/pull/65282) and
[#65283](https://github.com/vllm-project/vllm/pull/65283), as [`python/flatshare_vllm`](../../python/flatshare_vllm)
builds them. The box itself keeps its units ([`deploy/phastform`](../phastform)), and both run the same engine.

## Build

From the repository's root:

```sh
docker build -f deploy/container/Dockerfile -t flatshare-engine .
```

- **What goes in.** The root's `.dockerignore` lets in the five paths the image and its entrypoint use, each at its
  path in the repository, as a release lays them out on the box. The image copies them to `/opt/flatshare`.
- **make-room's module.** `build.py` writes it once, at build time, from the patches against the image's vLLM. Each
  start holds the engine to `serving.json`'s vLLM 0.32.0 and module hash, so `--build-arg VLLM_IMAGE=...` can name only
  another build of that release.

## Run

```sh
docker run --gpus all --shm-size 24g --stop-timeout 90 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v flatshare-cache:/opt/flatshare/cache \
  -p 127.0.0.1:8000:8000 flatshare-engine --host 0.0.0.0 --port 8000
```

- **Shared memory.** `--shm-size 24g` holds the KV cache's CPU tier, a 21.47 GB file in `/dev/shm`, which a
  container's default 64 MB cannot ([HARDWARE.md's rule R3](../../docs/HARDWARE.md#the-rules)).
- **The engine's flags** come from the serving profile. Flags after the image's name follow them, such as a port, as
  above, or `--uds /run/flatshare/sock/engine.sock` with a directory mounted there, as the box serves.
- **Stopping.** The engine stops on SIGINT, as the unit's `KillSignal` says, and `serve.py` passes `docker stop`'s
  SIGTERM on as SIGINT. `--stop-timeout 90` gives it the unit's 90 s, which an engine still starting may need.
- **The caches.** The model comes from the Hugging Face cache, offline with `-e HF_HUB_OFFLINE=1` once it is there.
  Every compile and JIT cache the engine's start reads is under `/opt/flatshare/cache`, the CUDA driver's among them,
  so with a volume there only the first start compiles.
- **Podman** takes the same image, with the GPU through CDI, `--device nvidia.com/gpu=all` in place of `--gpus all`.

## What a start checks

`serve.py`, the entrypoint, gives `vllm serve` the flags that `deploy/phastform/serving.json` sets, the effective
configuration that `deploy/profile.py` renders from [the serving profile](../profiles/rtx-5060-ti-serve.toml). Once the
engine has written the configuration it runs with, [`deploy/bin/serving-check`](../bin/serving-check) holds it to
`serving.json`: the scheduler module's hash, the switches, the KV pool, the CPU tier, and the patches. An engine that
runs anything else is stopped, and the container exits 1, as the unit's start fails.
[`test_container.py`](../../python/flatshare_vllm/tests/test_container.py) holds the entrypoint's flags to the
make-room-serve drop-in's, and the image's caches to the A100 profile's and the CUDA driver's.

## On the box

The recipe ran on 2026-10-30, at v1.13, on the box's RTX 5060 Ti 16 GB (driver 580.173.02), with rootless podman 5.8.8
and the GPU through the NVIDIA Container Toolkit 1.20.1's CDI specification. Each run had no network, the engine on a
Unix socket and the Hugging Face cache read-only. The profile has since gained the engine's additions, the cache-aware
queue and the SM copy, whose flags `serve.py` takes from `serving.json` like the others. Docker did not run, since the
box has none, and the `docker run` above is the same run with Docker's flags.

- **It served.** The check passed with the serving profile's flags, `/health` answered 200, and a completion and a
  chat completion at priority 0 answered.
- **The start**, from the container's start to the check, took 708 s with the cache volume empty, 90 s with it filled,
  and 431 s with every cache warm but the CUDA driver's. The box's unit starts its engine in 531.8 s with empty caches
  and 88.8 s with warm ones ([HARDWARE.md's preflight, note 5](../../docs/HARDWARE.md#the-preflight)).
- **It stopped** in 8.1 to 8.3 s on `podman stop -t 90` in each of four runs, and the GPU's memory went back to 2 MiB.
- **A changed engine was refused.** Started with `--max-num-batched-tokens 512` after the profile's flags, the engine
  came up, the check named `engine.scheduler.max_num_batched_tokens  256  DIFFERS: runs 512` as its one difference,
  `serve.py` stopped the engine, and the container exited 1. Stopped as its server starts, vLLM prints a traceback of
  its own (`'NoSignalServer' object has no attribute 'servers'`), and the line from `serve.py` before it gives the
  reason.
