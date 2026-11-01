#!/usr/bin/env bash
# install-main.sh: vLLM main as its nightly wheel (the build of 564ac0de1, 0.33.0rc1.dev444, whose flashinfer.py is
# main's and rc3's: no wait around plan()) into venv-main, as README.md installs rc3; it pins the file, so the run names
# the commit it tested. Writes install-main.log beside itself.
set -euo pipefail
cd "$(dirname "$0")"
wheel=https://wheels.vllm.ai/564ac0de1b228568cafcbfbe8ccaccc82d6023eb/vllm-0.33.0rc1.dev444%2Bg564ac0de1-cp38-abi3-manylinux_2_28_x86_64.whl
{
  "$HOME/.local/bin/uv" venv -q -p 3.12 venv-main
  "$HOME/.local/bin/uv" pip install --python venv-main/bin/python "$wheel" --torch-backend=cu130
  venv-main/bin/python -c 'import vllm, torch, flashinfer; print(vllm.__version__, torch.__version__, flashinfer.__version__)'
} > install-main.log 2>&1
echo "install-main: done" >> install-main.log
