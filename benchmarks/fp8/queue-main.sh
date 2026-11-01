#!/usr/bin/env bash
# queue-main.sh: r1's load on vLLM main (venv-main, install-main.sh: the nightly build of 564ac0de1, FlashInfer
# 0.7.2.post1 and torch 2.14.0 as rc3), after queue-core.sh has logged "queue done": main still has no wait around
# FlashInfer's plan(), so H1 predicts the crash there too. It shares rc3's FlashInfer workspace (the same FlashInfer
# and torch); a short run first compiles whatever main needs beyond rc3's kernels, so that no kernel compiled under
# load passes for a stall. Progress: runs/queue-main.log.
set -uo pipefail
cd "$(dirname "$0")"
log=runs/queue-main.log
until grep -q 'queue done' runs/queue-core.log 2>/dev/null; do sleep 30; done
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
if ! grep -q 'install-main: done' install-main.log 2>/dev/null; then
  echo "$(date -u +%FT%TZ) venv-main is not installed (install-main.log); queue done" >> "$log"
  exit 1
fi
export FLASHINFER_WORKSPACE_BASE=$PWD/cache/flashinfer-rc3
main=$PWD/venv-main
run r20w-main-warmup env FP8_VENV="$main" ./run.sh r20w-main-warmup offload.yaml 180 --kv-cache-dtype fp8
run r20-main-fp8-offload env FP8_VENV="$main" ./run.sh r20-main-fp8-offload offload.yaml 900 --kv-cache-dtype fp8
echo "$(date -u +%FT%TZ) queue done" >> "$log"
