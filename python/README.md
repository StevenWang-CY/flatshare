# Python packages

Four packages, for a developer who builds or changes them. Each has its own `pyproject.toml` and `uv.lock`, because
each needs a stack of its own (a vLLM checkout, AIPerf, LiveKit), and `make test-vllm` and `make test-python` test each
in its own environment.

| Package | What it is | Depends on |
|---|---|---|
| [`flatshare_vllm`](flatshare_vllm/README.md) | The change inside vLLM: the two pull requests as patches, the build of the `make-room` scheduler, the serving engine's additions, and their probes and tests | Nothing to install: the build and the probes use the standard library, and the engine's own vLLM |
| [`flatshare_voice`](flatshare_voice/README.md) | The voice worker on LiveKit Agents, and the callers that measure voice-to-voice latency from waveforms | LiveKit Agents 1.8.8 with its SDK, the OpenAI client, `flatshare_stats` |
| [`flatshare_bench`](flatshare_bench/README.md) | The measurement tools: the trace replay, E2's harness and report, the soak, the drills' client, and the code that draws every figure | `flatshare_stats`, aiohttp, httpx, orjson and fonttools; AIPerf 0.13.0 as the `aiperf` extra |
| [`flatshare_stats`](flatshare_stats/README.md) | The statistics the records publish, each defined once | The standard library only |

`flatshare_bench`'s tests also hold the rest of the repository in step: the deployment's scripts and units, the alert
rules, the committed records' tables, and the documents' links, numbers and prose.
[ARCHITECTURE.md](../docs/ARCHITECTURE.md#at-a-glance) places each package in the system.
