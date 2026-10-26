#!/usr/bin/env bash
# marlin-session1.sh [--fallback]: session 1 of the Marlin dry run on the box (docs/HARDWARE.md, the Marlin dry run),
# from the workstation, with the GPU free (no campaign running) and the box at a revision that has benchmarks/a100.
# About 25 minutes of GPU time:
#   1. the same day's reference: the frozen engine (DeepGEMM W8A8) with the measurement drop-in on (variant.sh
#      measure on: /server_info, which step 2 reads), E0 with 3 reps from cyan (flatshare-bench tiers, through the
#      tunnel)
#   2. the frozen engine with the linear layers on Marlin (--linear-backend marlin): the engine log must show
#      MarlinFP8ScaledMMLinearKernel, FlashAttention 2 and the LBHNC layout, /server_info's linear_backend marlin;
#      smoke.sh; E0 again
#   3. make-room with the PRs' values on Marlin: the KV case x2, the prefill case x2, the calls' tail x1
#   4. the frozen engine back
# With --fallback, step 2 and 3 select Marlin by disabling the four block-FP8 kernels ahead of it in vLLM's list
# (a drop-in, VLLM_DISABLED_KERNELS) in place of the flag, and remove the drop-in after. Each step's outputs stay on
# their host under ~/flatshare/runs/marlin-dry (cyan's E0, Phastform's log lines and probes) and are copied to
# FLATSHARE_OUT (default ~/flatshare/hosts/marlin-dry); a step whose outputs are there is skipped, so a rerun resumes.
# FLATSHARE_SSH, FLATSHARE_GPU (default phastform) and FLATSHARE_CYAN (default cyan) as for benchmarks/e2/campaign.sh:
# the hosts by SSH host names or aliases.
set -euo pipefail
fallback=false
case ${1:-} in
  "") ;;
  --fallback) fallback=true ;;
  *) echo "usage: marlin-session1.sh [--fallback]" >&2; exit 2 ;;
esac
ssh=${FLATSHARE_SSH:-ssh}
gpu=${FLATSHARE_GPU:-phastform}
cyan=${FLATSHARE_CYAN:-cyan}
out=${FLATSHARE_OUT:-$HOME/flatshare/hosts/marlin-dry}
runs=flatshare/runs/marlin-dry
variant=flatshare/src/deploy/phastform/variant.sh
probes=flatshare/src/python/flatshare_vllm/src/flatshare_vllm/make-room
engine='/run/user/$(id -u)/flatshare/engine.sock' # the box's engine socket, as the remote shell expands it
python=flatshare/venvs/engine/bin/python
mkdir -p "$out"
log() { echo "[$(date -u +%FT%TZ)] session 1: $*" | tee -a "$out/session.log"; }
on_gpu() { $ssh "$gpu" "$@"; }
on_cyan() { $ssh "$cyan" "$@"; }
disabled=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel
dropin=.config/systemd/user/flatshare-engine.service.d/marlin.conf
marlin=(--linear-backend marlin)
if $fallback; then marlin=(); fi
use() { # use VARIANT [FLAG...] on the GPU node, with Marlin's drop-in in fallback mode for a Marlin engine
  local name=$1
  shift
  log "engine: $name $*"
  on_gpu "bash $variant use $name $*" 2>&1 | tee -a "$out/session.log"
}
marlin_dropin() { # marlin_dropin on|off
  if [[ $1 == on ]]; then
    on_gpu "mkdir -p \$(dirname $dropin) && printf '[Service]\nEnvironment=VLLM_DISABLED_KERNELS=$disabled\n' >$dropin"
  else
    on_gpu "rm -f $dropin"
  fi
  on_gpu "systemctl --user daemon-reload"
  if [[ $1 == on ]]; then
    # variant.sh sees no change of its own drop-ins in this one: the engine restarts here.
    log "engine: frozen, VLLM_DISABLED_KERNELS=$disabled"
    on_gpu "systemctl --user reset-failed flatshare-engine; systemctl --user restart flatshare-engine" 2>&1 | tee -a "$out/session.log"
  fi
}
tiers() { # tiers NAME: E0 from cyan through the tunnel (flatshare-bench's default, its phastform-engine.sock)
  if on_cyan "test -f $runs/$1/summary.json"; then log "$1: done before"; return; fi
  log "$1: flatshare-bench tiers --reps 3"
  on_cyan "flatshare/venvs/bench/bin/flatshare-bench tiers --reps 3 --out $runs/$1" 2>&1 |
    tail -12 | tee -a "$out/session.log"
}
probe() { # probe NAME PROGRAM ARG...: one probe run on the GPU node into NAME.txt
  local name=$1
  shift
  if on_gpu "test -f $runs/probes/$name.txt"; then log "$name: done before"; return; fi
  log "$name: $*"
  on_gpu "mkdir -p $runs/probes && $python $probes/$1 unix:$engine ${*:2} >$runs/probes/$name.partial 2>&1 &&
    mv $runs/probes/$name.partial $runs/probes/$name.txt && tail -1 $runs/probes/$name.txt" | tee -a "$out/session.log"
}

# 1. The same day's reference.
log "the measurement drop-in on"
on_gpu "bash $variant measure on" 2>&1 | tee -a "$out/session.log"
use frozen
tiers tiers-deepgemm

# 2. The frozen engine on Marlin, and the engine's own account of its kernels.
since=$(on_gpu date +%s)
if $fallback; then marlin_dropin on; fi
use frozen "${marlin[@]}"
on_gpu "mkdir -p $runs && journalctl --user -u flatshare-engine --since @$since -o cat |
  grep -E 'Selected .* for Fp8LinearMethod|native support for FP8|attention backend|FlashAttention version|KV cache layout' \
  >$runs/marlin-engine-lines.txt; cat $runs/marlin-engine-lines.txt" | tee -a "$out/session.log"
on_gpu "grep -q 'Selected MarlinFP8ScaledMMLinearKernel for Fp8LinearMethod' $runs/marlin-engine-lines.txt" ||
  { log "the engine did not select Marlin (above); run again with --fallback"; exit 1; }
backend=$(on_gpu "curl -fsS --unix-socket $engine 'http://localhost/server_info?config_format=json' | $python -c \
  'import json, sys; print(json.load(sys.stdin)[\"vllm_config\"][\"kernel_config\"][\"linear_backend\"])'")
log "linear_backend: $backend"
$fallback || [[ $backend == marlin ]] || { log "/server_info names $backend, not marlin"; exit 1; }
on_gpu "bash flatshare/src/deploy/phastform/smoke.sh" 2>&1 | tee -a "$out/session.log"
tiers tiers-marlin

# 3. make-room with the PRs' values on Marlin.
use make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48 \
  "${marlin[@]}"
probe marlin-kv-1 probe.py kv
probe marlin-kv-2 probe.py kv
probe marlin-prefill-1 probe.py prefill
probe marlin-prefill-2 probe.py prefill
probe marlin-tail-1 tail.py 240 12 fresh

# 4. Back to the frozen box.
if $fallback; then marlin_dropin off; fi
use frozen
rsync -a -e "$ssh" "$cyan:$runs/" "$out/cyan/"
rsync -a -e "$ssh" "$gpu:$runs/" "$out/gpu/"
log "done: $out"
