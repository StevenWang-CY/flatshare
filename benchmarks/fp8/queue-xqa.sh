#!/usr/bin/env bash
# queue-xqa.sh: does the crash need FlashInfer's native decode? On sm_120 vLLM 0.32.0 decodes an FP8 cache with XQA
# only when it believes NVIDIA's artifactory reachable (vllm/utils/flashinfer.py:620-667); the box's engine runs in a
# network namespace with no way out, so the probe fails and decode falls back to the native wrapper, which plans (and
# refills its page-locked buffer) every step. XQA is JIT-compiled from FlashInfer's own sources, so
# VLLM_HAS_FLASHINFER_CUBIN=1 enables it offline. Starts only after queue-rc3.sh has logged "queue done". The kernels
# compile into a copy of the box's FlashInfer workspace, so the box's own stays as it is. Progress: runs/queue-xqa.log.
set -uo pipefail
cd "$(dirname "$0")"
log=runs/queue-xqa.log
until grep -q 'queue done' runs/queue-rc3.log 2>/dev/null; do sleep 30; done
gpu_free() {
  local used procs
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  (( used < 100 && procs == 0 ))
}
run() {
  local name=$1
  shift
  until gpu_free; do sleep 10; done
  echo "$(date -u +%FT%TZ) start $name" >> "$log"
  "$@" > /dev/null 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name exit $rc $(grep -E '^verdict' "runs/$name/result.txt" 2>/dev/null)" >> "$log"
}
export FLASHINFER_WORKSPACE_BASE=$PWD/cache/flashinfer-xqa
[[ -d $FLASHINFER_WORKSPACE_BASE ]] || cp -a "$HOME/flatshare/cache/flashinfer" "$FLASHINFER_WORKSPACE_BASE"
fix=$PWD/venv-fix
# r1 with XQA decode: stock 0.32.0, FP8 and the offload, 15 min.
run r16-fp8-offload-xqa env VLLM_HAS_FLASHINFER_CUBIN=1 ./run.sh r16-fp8-offload-xqa offload.yaml 900 --kv-cache-dtype fp8
# The same counting only: with decode on XQA only the prefill wrapper plans; do its plans still find a copy pending?
run r17-fp8-offload-xqa-count env FP8_VENV="$fix" FP8INV_PLAN_WAIT=count VLLM_HAS_FLASHINFER_CUBIN=1 \
  ./run.sh r17-fp8-offload-xqa-count offload.yaml 900 --kv-cache-dtype fp8
echo "$(date -u +%FT%TZ) queue done" >> "$log"
