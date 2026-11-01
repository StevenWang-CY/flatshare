#!/usr/bin/env bash
# queue-core.sh: the mechanism's remaining tests, after queue-xqa.sh has logged "queue done".
#   (copy_contention.py and r19 moved into queue-rc3.sh, to run earlier.)
#   r18: r1 with a CUDA core dump on the exception, which names the faulting kernel (read with cuda-gdb); skip_abort
#        keeps the driver from aborting the process after it (no CPU core dump of a process pinning a 20 GiB CPU tier,
#        and vLLM still logs the CUDA error), and should the race end in a stall instead, run.sh asks for a dump
#        through the engine core's pipe (user-triggered), which names the kernel that never finishes.
#   r5:  r1 with blocking launches, under which every copy completes before the host goes on: H1 predicts no crash.
# Progress: runs/queue-core.log.
set -uo pipefail
cd "$(dirname "$0")"
log=runs/queue-core.log
until grep -q 'queue done' runs/queue-xqa.log 2>/dev/null; do sleep 30; done
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
mkdir -p dumps
cuda_gdb=$PWD/tools/cuda-gdb/cuda_gdb-linux-x86_64-13.0.85-archive/bin/cuda-gdb
# The positive control for the stall's dump: a user-triggered dump of a kernel still running, named by cuda-gdb.
until gpu_free; do sleep 10; done
echo "$(date -u +%FT%TZ) start dump-selftest" >> "$log"
"$PWD/venv-fix/bin/python" dump_selftest.py dumps "$cuda_gdb" runs/dump-selftest.json > runs/dump-selftest.out 2>&1
echo "$(date -u +%FT%TZ) end dump-selftest exit $?" >> "$log"
run r18-fp8-offload-coredump env CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION=0 \
  CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 CUDA_COREDUMP_PIPE="$PWD/dumps/r18-pipe.%p" \
  CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory,skip_abort \
  CUDA_COREDUMP_FILE="$PWD/dumps/r18.%p.%t" ./run.sh r18-fp8-offload-coredump offload.yaml 900 --kv-cache-dtype fp8
# Read each of r18's dumps into its run: the exception, the kernels, the backtrace and the instructions at the error.
for dump in dumps/r18.*; do
  [[ -f $dump ]] || continue
  "$cuda_gdb" -batch -nx -ex "target cudacore $PWD/$dump" -ex 'info cuda kernels' -ex 'bt' -ex 'x/6i $errorpc' \
    -ex 'x/6i $pc' -ex 'info cuda warps' > "runs/r18-fp8-offload-coredump/cuda-gdb.${dump#dumps/}.txt" 2>&1
  echo "$(date -u +%FT%TZ) read $dump exit $?" >> "$log"
done
run r5-fp8-offload-blocking env CUDA_LAUNCH_BLOCKING=1 ./run.sh r5-fp8-offload-blocking offload.yaml 600 --kv-cache-dtype fp8
echo "$(date -u +%FT%TZ) queue done" >> "$log"
