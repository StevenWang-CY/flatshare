#!/usr/bin/env bash
# bootstrap.sh: prepare a GPU node for the frozen box under ~/flatshare (Phastform; Gray reuses it), once a release is
# deployed there: deploy/deploy.sh's release builds the Python environments from the lock files (venvs/engine, vLLM
# 0.32.0 with the audio extra, serving the LLM and Whisper; venvs/tts, Kokoro-FastAPI 0.9.1-rc1 on torch 2.8.0+cu128;
# deploy/bin/flatshare-release), which this reads through ~/flatshare/venvs. Safe to re-run.
#   vendor/Kokoro-FastAPI   source checkout at the measured commit, and its model weights
#   cache/                  FlashInfer JIT workspace and the curand header links
#   ~/.cache/huggingface    the LLM and Whisper weights, at pinned revisions
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$HOME/flatshare
for env in engine tts; do
  [[ -x $root/venvs/$env/bin/python ]] || {
    echo "bootstrap: no $root/venvs/$env: deploy a release first (deploy/deploy.sh), which builds the environments" >&2
    exit 2
  }
done
mkdir -p "$root"/{vendor,cache/flashinfer,cache/curand_include,runs}

# Kokoro-FastAPI publishes no tag for 0.9.1-rc1, so the measured commit is the version pin; the release installs the
# checkout into the TTS environment and refuses a checkout at another commit (deploy/bin/flatshare-release's TTS pin,
# which this must equal). This is where the pin moves.
kokoro=$root/vendor/Kokoro-FastAPI
[[ -d $kokoro/.git ]] || git clone -q https://github.com/remsky/Kokoro-FastAPI.git "$kokoro"
git -C "$kokoro" checkout -q ebd4ec9cc97b258b25afc39444a5e1701055db50
if [[ ! -f $kokoro/api/src/models/v1_0/kokoro-v1_0.pth ]]; then
  (cd "$kokoro" && "$root/venvs/tts/bin/python" docker/scripts/download_model.py --output api/src/models/v1_0)
fi

# The units run offline (HF_HUB_OFFLINE=1), so the models are fetched here, at the revisions their configs pin.
for config in "$here/engine.yaml" "$here/stt.yaml"; do
  read -r model revision < <(awk '$1 == "model:" {m = $2} $1 == "revision:" {r = $2} END {print m, r}' "$config")
  [[ -n $revision ]] || { echo "bootstrap: $config pins no revision" >&2; exit 2; }
  "$root/venvs/engine/bin/hf" download -q "$model" --revision "$revision" >/dev/null
done

# FlashInfer's JIT needs curand headers that /usr/local/cuda-13.0 lacks on this host; expose the venv's copies.
ln -sf "$root"/venvs/engine/lib/python3.12/site-packages/nvidia/cu13/include/curand*.h "$root/cache/curand_include/"

echo "bootstrap: engine $("$root/venvs/engine/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)')"
echo "bootstrap: tts $("$root/venvs/tts/bin/python" -c 'import torch; print("torch", torch.__version__)')"
