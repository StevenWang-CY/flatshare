#!/usr/bin/env bash
# queue-bisect.sh: the FP8 crash's bisection, one run.sh run after another on the free GPU, each changing one thing
# from r1 (stock vLLM 0.32.0, FP8 KV cache, FlashInfer, CPU offload of 20 GiB, async scheduling, CUDA graphs), which
# crashed 124 s into the load. Progress goes to runs/queue-bisect.log; each run's evidence to its own directory.
set -uo pipefail
cd "$(dirname "$0")"
log=runs/queue-bisect.log
run() {
  local name=$1
  shift
  echo "$(date -u +%FT%TZ) start $name" >> "$log"
  "$@" > /dev/null 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name exit $rc $(grep -E '^verdict' "runs/$name/result.txt" 2>/dev/null)" >> "$log"
}
# Is the CPU offload needed? FP8 on FlashInfer without the connector.
run r2-fp8-nooffload env ./run.sh r2-fp8-nooffload nooffload.yaml 900 --kv-cache-dtype fp8
# Is FP8 needed? BF16 on FlashInfer, beside the offload.
run r3-bf16-flashinfer-offload env ./run.sh r3-bf16-flashinfer-offload offload.yaml 900 --attention-backend FLASHINFER
# Is FlashInfer needed? FP8 on Triton attention, beside the offload.
run r8-fp8-triton-offload env ./run.sh r8-fp8-triton-offload offload.yaml 900 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN
# The control: the box's own BF16 cache on FlashAttention, beside the offload, under the same load.
run r4-bf16-flashattn-offload env ./run.sh r4-bf16-flashattn-offload offload.yaml 900
# r1 once more, with every launch synchronous, so the fault is raised at the operation that launched its kernel.
run r5-fp8-offload-blocking env CUDA_LAUNCH_BLOCKING=1 ./run.sh r5-fp8-offload-blocking offload.yaml 600 --kv-cache-dtype fp8
# Is asynchronous scheduling needed?
run r7-fp8-offload-noasync env ./run.sh r7-fp8-offload-noasync offload.yaml 900 --kv-cache-dtype fp8 --no-async-scheduling
# Without CUDA graphs, with synchronous launches: the faulting operation outside any graph.
run r6-fp8-offload-eager-blocking env CUDA_LAUNCH_BLOCKING=1 ./run.sh r6-fp8-offload-eager-blocking offload.yaml 600 --kv-cache-dtype fp8 --enforce-eager
echo "$(date -u +%FT%TZ) queue done" >> "$log"
