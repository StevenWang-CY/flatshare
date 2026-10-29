# Third-party software and models

This file lists what Flatshare's serving path runs but does not write, with each item's licence. Flatshare's own code
is under the Apache License 2.0 ([LICENSE](LICENSE)).

**Who carries the obligations.** Flatshare redistributes none of these items. Each host installs them from their
publishers when it is deployed: PyPI and PyTorch's index for the Python environments, Hugging Face for the models,
LiveKit's GitHub releases for its server, and a pinned commit of Kokoro-FastAPI ([`deploy/`](deploy)). Whoever ships
them in an image or another distribution takes on their licences' obligations. For the GPL packages below, those
obligations are their licence texts and their corresponding source.

**How the licences were found** (2026-10-29):
- for the Python environments, each locked package's metadata on PyPI;
- for the models, the model cards on Hugging Face;
- for the rest, the projects' own repositories.

The environments are the engine's and TTS server's requirements on the GPU nodes
([`deploy/phastform/requirements`](deploy/phastform/requirements)) and the voice worker's lock
([`python/flatshare_voice/uv.lock`](python/flatshare_voice/uv.lock)). The gateway's crates are in
[`Cargo.lock`](Cargo.lock).

## Models

| Model | Where it runs | Licence |
|---|---|---|
| Qwen3-4B-Instruct-2507-FP8, at the revision [`engine.yaml`](deploy/phastform/engine.yaml) pins | The engine (vLLM) | Apache-2.0 |
| whisper-large-v3-turbo, at the revision [`stt.yaml`](deploy/phastform/stt.yaml) pins | The STT server (vLLM) | MIT |
| Kokoro-82M and its v1.0 voices, as the pinned Kokoro-FastAPI fetches them | The TTS server | Apache-2.0 |
| LiveKit's end-of-turn model, turn-detector v1-mini, inside `livekit-local-inference` 0.2.7 | The voice worker, serve mode | LiveKit Model License (below) |
| Silero VAD, inside `livekit-plugins-silero` 1.8.8 | The voice worker | MIT |

**The LiveKit Model License.** The end-of-turn model's wheel declares `Apache-2.0 AND LicenseRef-LiveKit-Model`, and
its `MODEL_LICENSE` restricts the model to LiveKit Agents:
- it may not be used on its own;
- its outputs may not be used to develop other models;
- either party may terminate the licence on immediate written notice.

Flatshare runs it only inside LiveKit Agents. Serve mode with real callers' microphones needs it, since the worker
refuses the experiments' fixed endpointing (`FLATSHARE_TURN_DETECTION=vad`) with the microphone configuration
([`worker.py`](python/flatshare_voice/src/flatshare_voice/worker.py)).

## Software

| Component | Version | Licence |
|---|---|---|
| vLLM | 0.32.0 | Apache-2.0 |
| PyTorch | 2.13.0 (engine), 2.8.0 (TTS), from PyTorch's CUDA index | BSD-3-Clause, and in 2.13.0 also Apache-2.0, BSD-2-Clause, BSL-1.0 and MIT parts |
| Kokoro-FastAPI | 0.9.1-rc1, at commit `ebd4ec9` | Apache-2.0 |
| `kokoro` and `misaki` | 0.9.4 | Apache-2.0 |
| LiveKit server | 1.13.10 | Apache-2.0 |
| LiveKit Agents and its plugins | 1.8.8 | Apache-2.0 |
| `flatshare-gateway`'s 130 third-party crates | [`Cargo.lock`](Cargo.lock) | MIT or Apache-2.0, with BSD-3-Clause, Unicode-3.0 (`unicode-ident`) and Unlicense-or-MIT among them. `r-efi`, a UEFI crate that a Linux build does not compile, also offers LGPL-2.1-or-later |

## Packages whose licences are not permissive

Every other package in the three environments declares a permissive licence (MIT, BSD, Apache-2.0, ISC, PSF and
similar), except those below and those that declare none (next section).

Each environment is one kind of process's:
- **engine:** the engine's and the STT server's vLLM processes, on each GPU node;
- **TTS:** the TTS server's process, Kokoro-FastAPI, on each GPU node;
- **voice:** the voice worker's process, on cyan.

No process loads another environment's packages. The gateway and the voice worker reach the GPU nodes' servers only
over sockets, through cyan's tunnels.

The licences are SPDX ids, from each package's own metadata: its licence expression where it has one, or else its
`License` field and classifiers.

| Package | Version | Loaded by | Licence |
|---|---|---|---|
| `phonemizer-fork` | 3.3.2 | The TTS server (TTS) | GPL-3.0-or-later (its classifier; its `License` field is the GPL 3 text) |
| `espeakng-loader` | 0.2.4 | The TTS server (TTS) | None declared. Its wheel bundles `libespeak-ng` and `espeak-ng-data`, which espeak-ng releases under GPL-3.0-or-later |
| `praat-parselmouth` | 0.4.7 | The TTS server (TTS) | GPL-3.0-or-later (its classifier; its `License` field says GPLv3) |
| `mutagen` | 1.48.1 | The TTS server (TTS) | GPL-2.0-or-later |
| `pycountry` | 26.2.16 | The engine and STT servers (engine) | LGPL-2.1-only |
| `soxr` | 1.1.0 | The engine and STT servers (engine) | LGPL-2.1-or-later |
| `num2words` | 0.5.14 | The TTS server (TTS) | LGPL, with no version given (its `License` field and classifier) |
| `certifi` | 2026.10.19 | The engine and STT servers (engine), the TTS server (TTS), the voice worker (voice) | MPL-2.0 |
| `tqdm` | 4.70.1 (engine), 4.67.1 (TTS) | The engine and STT servers (engine), the TTS server (TTS) | MPL-2.0 AND MIT |
| `livekit-local-inference` | 0.2.7 | The voice worker (voice) | Apache-2.0 AND LicenseRef-LiveKit-Model (above) |
| NVIDIA's CUDA libraries (`nvidia-*`, `cuda-toolkit`) | as locked | The engine and STT servers (engine), the TTS server (TTS) | LicenseRef-NVIDIA-Proprietary, or "NVIDIA Proprietary Software", where their metadata names one |

## Packages that declare no licence

None of these packages' metadata names a licence. Where a primary source gives one, it is listed:
- `nvtx` 0.2.15 (engine): Apache-2.0 with LLVM exceptions, per NVIDIA/NVTX's LICENSE.
- `quack-kernels` 0.6.5 (engine): Apache-2.0, per Dao-AILab/quack, whose `pyproject.toml` names this package.
- `unicode-segmentation-rs` 0.3.3 (TTS): Apache-2.0, per WeblateOrg/unicode-segmentation-rs.
- `oss-harmony` 0.0.11 (engine): none found. Its summary calls it a fork of `openai-harmony`, which is Apache-2.0.
- The CUDA 13 wheels in the engine (`cuda-toolkit`, `nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cudnn-cu13`,
  `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`, `nvidia-nvvm`): NVIDIA's.
- PyTorch's own builds from its CUDA index (`torch`, `torchaudio`, `torchcodec` and `torchvision`, `+cu130` and
  `+cu128`), which are not on PyPI: PyTorch's licences, as in the table above.

A lock change that brings in a package outside the permissive licences, or changes one of the versions above, updates
this file. A test holds the tables' versions to the locks.
