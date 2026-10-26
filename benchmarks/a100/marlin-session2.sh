#!/usr/bin/env bash
# marlin-session2.sh: session 2 of the Marlin dry run on the box (docs/HARDWARE.md, the Marlin dry run), from the
# workstation, after session 1, with the GPU free and the box at a revision that has benchmarks/a100. About 30 minutes:
#   1. the GEMMs: the engine stopped, the STT and TTS idle, bench_marlin.py at Qwen3-4B's four linear shapes for
#      M of 1 to 1,024 (Marlin W8A16, torch BF16, DeepGEMM W8A8), each call's GPU time in a CUDA graph, in the engine
#      unit's environment (DeepGEMM builds its kernels with the unit's nvcc): Marlin's factor over BF16 on the box
#   2. the probes at a 3 GiB pool (P-1's prep 3): make-room with the PRs' values, the pool 3 GiB and the context
#      16,384 tokens (vLLM refuses a pool that cannot hold one request at max_model_len), the KV case, the prefill
#      case and a 120 s tail once each, which size themselves from the smaller pool and context; the context is in
#      vLLM's compile hash, so this start compiles
#   3. one E2 point on Marlin, benchmarks/a100/marlin-dry.campaign through benchmarks/e2/campaign.sh
#   4. the frozen engine back
# The outputs stay on Phastform under ~/flatshare/runs/marlin-dry (the point's on cyan, as a campaign's) and are
# copied to FLATSHARE_OUT (default ~/flatshare/hosts/marlin-dry); a step whose outputs are there is skipped.
# FLATSHARE_SSH, FLATSHARE_GPU and FLATSHARE_CYAN as for benchmarks/e2/campaign.sh.
set -euo pipefail
(($# == 0)) || { echo "usage: marlin-session2.sh" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
ssh=${FLATSHARE_SSH:-ssh}
gpu=${FLATSHARE_GPU:-phastform}
out=${FLATSHARE_OUT:-$HOME/flatshare/hosts/marlin-dry}
runs=flatshare/runs/marlin-dry
variant=flatshare/src/deploy/phastform/variant.sh
probes=flatshare/src/python/flatshare_vllm/src/flatshare_vllm/make-room
engine='/run/user/$(id -u)/flatshare/engine.sock' # the box's engine socket, as the remote shell expands it
python=flatshare/venvs/engine/bin/python
mkdir -p "$out"
log() { echo "[$(date -u +%FT%TZ)] session 2: $*" | tee -a "$out/session.log"; }
on_gpu() { $ssh "$gpu" "$@"; }
probe() { # probe NAME PROGRAM ARG...
  local name=$1
  shift
  if on_gpu "test -f $runs/pool3g/$name.txt"; then log "$name: done before"; return; fi
  log "$name: $*"
  on_gpu "mkdir -p $runs/pool3g && $python $probes/$1 unix:$engine ${*:2} >$runs/pool3g/$name.partial 2>&1 &&
    mv $runs/pool3g/$name.partial $runs/pool3g/$name.txt && cat $runs/pool3g/$name.txt" | tee -a "$out/session.log"
}

# 1. The GEMMs, with the engine stopped (the audio servers stay, idle), in the engine unit's environment
#    (host.py with-engine-env: its CUDA_HOME and PATH, whose nvcc DeepGEMM's JIT needs). A gemm.csv without the timing
#    column is the first bench's, which timed one call between two CUDA events and so its dispatch from Python: it is
#    kept as gemm-events.csv and the GEMMs run again.
if on_gpu "head -1 $runs/gemm.csv 2>/dev/null | grep -q ',timing,'"; then
  log "gemm.csv: done before"
else
  log "bench_marlin.py, in the engine unit's environment (a gemm.csv without timing kept as gemm-events.csv)"
  on_gpu "systemctl --user stop flatshare-engine && mkdir -p $runs &&
    { test ! -f $runs/gemm.csv || mv $runs/gemm.csv $runs/gemm-events.csv; } &&
    $python flatshare/src/deploy/a100/host.py with-engine-env flatshare/src/deploy/phastform \
      $python flatshare/src/benchmarks/a100/bench_marlin.py --m 1,16,64,128,256,512,1024 --out $runs/gemm.csv" 2>&1 |
    tee -a "$out/session.log"
fi

# 2. The probes at a 3 GiB pool and a 16,384-token context.
log "engine: make-room, a 3 GiB pool, a 16,384-token context"
on_gpu "bash $variant use make-room --kv-cache-memory 3221225472 --max-model-len 16384 \
  --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48" 2>&1 |
  tee -a "$out/session.log"
probe kv probe.py kv
probe prefill probe.py prefill
probe tail tail.py 120 pool fresh --reply 40

# 3. One E2 point on Marlin, as campaign.sh measures one.
log "E2: marlin-dry.campaign"
bash "$here/../e2/campaign.sh" "$here/marlin-dry.campaign" 2>&1 | tee -a "$out/session.log"

# 4. Back to the frozen box.
log "engine: frozen"
on_gpu "bash $variant use frozen" 2>&1 | tee -a "$out/session.log"
rsync -a -e "$ssh" "$gpu:$runs/" "$out/gpu/"
log "done: $out (the E2 point: campaign.sh's results, ~/flatshare/runs/e2/marlin-dry)"
