#!/usr/bin/env bash
# queue-rc3.sh: first the copy-contention microbenchmark (a minute; from queue-core.sh, moved ahead so that its answer
# comes early), then: does vLLM 0.33.0rc3 (FlashInfer 0.7.2.post1, torch 2.14.0) still crash under r1's load, and does
# #64165's wait hold there too? Starts only after queue-fix.sh has logged "queue done", so two queues never share the
# GPU. venv-rc3 is stock rc3; venv-rc3-fix is the same install with patch_plan_wait.py applied. FlashInfer JIT-compiles
# 0.7.2.post1's kernels into a scratch workspace, not the box's. Progress goes to runs/queue-rc3.log.
set -uo pipefail
cd "$(dirname "$0")"
log=runs/queue-rc3.log
until grep -q 'queue done' runs/queue-fix.log 2>/dev/null; do sleep 30; done
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
export FLASHINFER_WORKSPACE_BASE=$PWD/cache/flashinfer-rc3
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
stock=$PWD/venv-rc3 fix=$PWD/venv-rc3-fix
until gpu_free; do sleep 10; done
echo "$(date -u +%FT%TZ) start copy-contention" >> "$log"
"$PWD/venv-fix/bin/python" copy_contention.py runs/copy-contention.json > runs/copy-contention.out 2>&1
echo "$(date -u +%FT%TZ) end copy-contention exit $?" >> "$log"
# A short run first, so FlashInfer's kernels are compiled before any measured run: a kernel compiled lazily under load
# would hold requests long enough to look like a stall. The patch changes vLLM, not FlashInfer, so both builds share it.
run r13w-rc3-warmup env FP8_VENV="$stock" ./run.sh r13w-rc3-warmup offload.yaml 180 --kv-cache-dtype fp8
# r1 on rc3: the stock build under the same load, for 15 min (over seven times r1's 124 s to the crash).
run r13-rc3-fp8-offload env FP8_VENV="$stock" ./run.sh r13-rc3-fp8-offload offload.yaml 900 --kv-cache-dtype fp8
# r9 on rc3: #64165's wait.
run r15-rc3-fp8-offload-fix env FP8_VENV="$fix" FP8INV_PLAN_WAIT=block ./run.sh r15-rc3-fp8-offload-fix offload.yaml 900 --kv-cache-dtype fp8
# r10 on rc3: counting only, to see the window open on 0.33 as well.
run r14-rc3-fp8-offload-count env FP8_VENV="$fix" FP8INV_PLAN_WAIT=count ./run.sh r14-rc3-fp8-offload-count offload.yaml 900 --kv-cache-dtype fp8
# The plan counter without the CPU tier (from queue-core.sh): do plans still find the previous copy pending?
run r19-fp8-nooffload-count env FP8_VENV="$PWD/venv-fix" FP8INV_PLAN_WAIT=count \
  FLASHINFER_WORKSPACE_BASE="$HOME/flatshare/cache/flashinfer" ./run.sh r19-fp8-nooffload-count nooffload.yaml 900 --kv-cache-dtype fp8
echo "$(date -u +%FT%TZ) queue done" >> "$log"
