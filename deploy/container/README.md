# The serving engine in a container

The box's engine as an image: vLLM 0.32.0's own image, `docker.io/vllm/vllm-openai:v0.32.0`, with make-room's
scheduler ([vllm-project/vllm#63350](https://github.com/vllm-project/vllm/pull/63350) and
[#63351](https://github.com/vllm-project/vllm/pull/63351), as [`python/flatshare_vllm`](../../python/flatshare_vllm)
builds them) at the serving profile's settings. It is for a stack that runs vLLM's image rather than the box's
systemd units. The box itself keeps its units ([`deploy/phastform`](../phastform)), and both run the same engine.

## Build

From the repository's root:

```sh
docker build -f deploy/container/Dockerfile -t flatshare-engine .
```

- **What goes in.** The root's `.dockerignore` lets in the five paths the image and its entrypoint use, each at its
  path in the repository, as a release lays them out on the box. The image copies them to `/opt/flatshare`.
- **make-room's module.** `build.py` writes it once, at build time, from the patches against the image's vLLM. It
  refuses any vLLM but 0.32.0, so `--build-arg VLLM_IMAGE=...` can name only another build of that release.

## Run

```sh
docker run --gpus all --shm-size 24g --stop-timeout 90 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v flatshare-cache:/opt/flatshare/cache \
  -p 127.0.0.1:8000:8000 flatshare-engine --host 0.0.0.0 --port 8000
```

- **Shared memory.** `--shm-size 24g` holds the KV cache's CPU tier, a 21.47 GB file in `/dev/shm`
  (`cpu_bytes_to_use`), which a container's default 64 MB cannot ([HARDWARE.md, R3](../../docs/HARDWARE.md)).
- **The engine's flags** come from the serving profile. Flags after the image's name follow them: a port, as above, or
  `--uds /run/flatshare/sock/engine.sock` with a directory mounted there, as the box serves.
- **Stopping.** The engine stops on SIGINT, as the unit's `KillSignal` says, and `serve.py` passes `docker stop`'s
  SIGTERM on as SIGINT. `--stop-timeout 90` gives it the unit's 90 s before `docker stop` kills it; an engine still
  starting may need all of them.
- **The caches.** The model comes from the Hugging Face cache, offline with `-e HF_HUB_OFFLINE=1` once it is there.
  Every compile and JIT cache the engine's start reads is under `/opt/flatshare/cache`: vLLM's, Inductor's, Triton's
  and FlashInfer's, as the A100 profile keeps them, and the CUDA driver's, which a host keeps in `~/.nv`. With a volume
  there, only the first start compiles.
- **Podman** takes the same image, with the GPU through CDI: `--device nvidia.com/gpu=all` in place of `--gpus all`.

## What a start checks

`serve.py`, the entrypoint, gives `vllm serve` the flags that `deploy/phastform/serving.json` sets: the step cap of
256 tokens with no long-prefill threshold, make-room's scheduler class, and its switches. That file is the effective
configuration that `deploy/profile.py` renders from [the serving profile](../profiles/rtx-5060-ti-serve.toml).

Once the engine has written the configuration it runs with, [`deploy/bin/serving-check`](../bin/serving-check) holds
it to `serving.json`: the scheduler module's hash, the switches, the KV pool, the CPU tier, and the patches. An engine
that runs anything else is stopped, and the container exits 1, as the unit's start fails.
[`test_container.py`](../../python/flatshare_vllm/tests/test_container.py) holds the entrypoint's flags to the
make-room-serve drop-in's, and the image's caches to the A100 profile's and the CUDA driver's.

## Verified

On 2026-10-30, on the box (an RTX 5060 Ti 16 GB, driver 580.173.02), with rootless podman 5.8.8 and the GPU through
the NVIDIA Container Toolkit 1.20.1's CDI specification. Each run had no network, the engine on a Unix socket and the
Hugging Face cache read-only:

- **The check passed** with the serving profile's flags: `serving-check: ok: engine and patches as rtx-5060-ti-serve
  names them`, for the scheduler's class and module hash, the step cap and threshold, the box's pool of 2,275 GPU
  blocks, the 21,474,836,480-byte CPU tier, make-room's switches and the three patches' hashes.
- **It served.** `/health` answered 200, and a completion and a chat completion at priority 0 answered.
- **The start**, from the container's start to the check, took 708 s with the cache volume empty and 90 s with it
  filled; with every cache warm but the CUDA driver's, 431 s. The box's unit starts its engine in 531.8 s with its
  caches empty and 88.8 s with them warm, as W8's start drills measured ([HARDWARE.md](../../docs/HARDWARE.md), note 5).
- **It stopped** in 8.1 to 8.3 s on `podman stop -t 90` in each of four runs, with "Application shutdown complete",
  and the GPU's memory went back to 2 MiB.
- **A changed engine was refused.** Started with `--max-num-batched-tokens 512` after the profile's flags, the engine
  came up, the check named `engine.scheduler.max_num_batched_tokens  256  DIFFERS: runs 512` as its one difference,
  `serve.py` stopped the engine, and the container exited 1. vLLM, stopped as its server starts, prints a traceback of
  its own (`'NoSignalServer' object has no attribute 'servers'`); the line from `serve.py` before it gives the reason.
- **Docker** did not run there: the box has none, and nothing is installed on it system-wide. The `docker run` above
  is the same run with Docker's flags.
