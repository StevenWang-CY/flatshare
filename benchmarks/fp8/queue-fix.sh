#!/usr/bin/env bash
# queue-fix.sh: the runs that test the FlashInfer plan-buffer race (vLLM #61239, fixed by #64165) against the FP8
# crash, the decisive ones first, then the rest of the bisection. Each waits until the GPU is free. Progress goes to
# runs/queue-fix.log; each run's evidence to its own directory. venv-fix is the engine env's packages with
# patch_plan_wait.py applied (FP8INV_PLAN_WAIT=block waits as #64165 does; count only counts).
set -uo pipefail
cd "$(dirname "$0")"
log=runs/queue-fix.log
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
fix=$PWD/venv-fix
# The fix: r1's configuration with the wait, for 30 min (over ten times r1's 124 s to the crash).
run r9-fp8-offload-fix env FP8_VENV="$fix" FP8INV_PLAN_WAIT=block ./run.sh r9-fp8-offload-fix offload.yaml 1800 --kv-cache-dtype fp8
# Its negative control: the same build counting only.
run r10-fp8-offload-count env FP8_VENV="$fix" FP8INV_PLAN_WAIT=count ./run.sh r10-fp8-offload-count offload.yaml 900 --kv-cache-dtype fp8
# BF16 on FlashInfer with the offload stalled (r3); with the wait?
run r12-bf16-flashinfer-offload-fix env FP8_VENV="$fix" FP8INV_PLAN_WAIT=block ./run.sh r12-bf16-flashinfer-offload-fix offload.yaml 900 --attention-backend FLASHINFER
# Is asynchronous scheduling needed? (#61239 reports none without it.)
run r7-fp8-offload-noasync env ./run.sh r7-fp8-offload-noasync offload.yaml 900 --kv-cache-dtype fp8 --no-async-scheduling
# Is FlashInfer needed? FP8 on Triton attention, which keeps no plan buffer, beside the offload.
run r8-fp8-triton-offload env ./run.sh r8-fp8-triton-offload offload.yaml 900 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN
# The control: the box's own BF16 cache on FlashAttention, beside the offload, under the same load.
run r4-bf16-flashattn-offload env ./run.sh r4-bf16-flashattn-offload offload.yaml 900
# r1 again, for a second time to the crash on stock vLLM.
run r1b-fp8-offload env ./run.sh r1b-fp8-offload offload.yaml 600 --kv-cache-dtype fp8
echo "$(date -u +%FT%TZ) queue done" >> "$log"
