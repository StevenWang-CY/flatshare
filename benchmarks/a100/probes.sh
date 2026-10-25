#!/usr/bin/env bash
# probes.sh BLOCK [--cap TOKENS]: one of the A100 run's probe blocks (docs/HARDWARE.md, the core block by block) on this
# host, the GPU node rendered at ~/leaseholder/node (LH_NODE_DIR another): engine-level runs on one engine configuration
# at a time, E0's cells (lh-bench tiers), make-room's KV and prefill cases (probe.py) and the calls' tail beside agents
# (tail.py with whole 40-token replies, at the box's KV pressure for the pool). Each run writes
# ~/leaseholder/runs/a100/blockN/NAME: a .txt of the probe's output, renamed from .partial once the probe has exited 0,
# or E0's directory with its summary.json; a run already there is skipped, so a block run again resumes where it
# stopped.
#   1  frozen, the box's flags: E0 (--reps 3), kv x3, prefill x3, tail x2
#   2  stock at vLLM's default here, 2,048/0: kv x3, tail x2; then stock at the time-equivalent step cap (R4; --cap,
#      1,536 by default, the threshold at half): kv x1, tail x2, its runs named by cap and threshold (stock1536-768-*),
#      so that a cap of 2,048, the forecast's low value, is not taken for vLLM's default (stock2048-*)
#   3  the PRs' values (make-room with no reserve and a 48-token window): kv x3, prefill x3, tail x2
set -euo pipefail
usage() { echo "usage: probes.sh 1|2|3 [--cap TOKENS]" >&2; exit 2; }
block=${1:-}
[[ $block == 1 || $block == 2 || $block == 3 ]] || usage
shift
cap=1536
while (($#)); do
  case $1 in
    --cap)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] && ((${2} % 256 == 0)) || { echo "probes: --cap is a multiple of 256" >&2; exit 2; }
      cap=$2
      shift
      ;;
    *) usage ;;
  esac
  shift
done
root=$HOME/leaseholder
src=$(cd "$(dirname "$0")/../.." && pwd)
node=${LH_NODE_DIR:-$root/node}
out=$root/runs/a100/block$block
probes=$src/python/leaseholder_vllm/src/leaseholder_vllm/make-room
python=$root/venvs/engine/bin/python
mkdir -p "$out"
# shellcheck source=/dev/null
[[ ! -f $node/env.sh ]] || source "$node/env.sh"
sock=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/lh/engine.sock # the engine's own socket (lh-engine.service)
engine=unix:$sock
log() { echo "[$(date -u +%FT%TZ)] probes $block: $*" | tee -a "$out/probes.log"; }

use() { # use VARIANT [FLAG...]: the engine for the runs that follow
  log "engine: $*"
  bash "$node/variant.sh" use "$@" 2>&1 | tee -a "$out/probes.log"
}
probe() { # probe NAME PROGRAM ARG...: one run of probe.py or tail.py into NAME.txt
  local name=$1 program=$2
  shift 2
  if [[ -f $out/$name.txt ]]; then
    log "$name: done before"
    return
  fi
  log "$name: $program $*"
  "$python" "$probes/$program" "$engine" "$@" >"$out/$name.partial" 2>&1 || {
    log "$name: failed (its output stays in $name.partial)"
    exit 1
  }
  mv "$out/$name.partial" "$out/$name.txt"
  tail -1 "$out/$name.txt" | sed "s/^/  /" | tee -a "$out/probes.log"
}
tiers() { # tiers NAME: E0's cells on the running engine
  if [[ -f $out/$1/summary.json ]]; then
    log "$1: done before"
    return
  fi
  rm -rf "$out/$1"
  local model
  model=$(curl -fsS -m 10 --unix-socket "$sock" http://localhost/v1/models |
    "$python" -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
  log "$1: lh-bench tiers --reps 3 ($model)"
  "$root/venvs/bench/bin/lh-bench" tiers --url "$engine" --model "$model" --reps 3 --out "$out/$1" 2>&1 |
    tail -3 | tee -a "$out/probes.log"
  [[ -f $out/$1/summary.json ]] || { log "$1: no summary.json"; exit 1; }
}
runs() { # runs PREFIX CASE COUNT [ARG...]
  local prefix=$1 case=$2 count=$3 i
  shift 3
  for ((i = 1; i <= count; i++)); do
    case $case in
      tail) probe "$prefix-tail-$i" tail.py 240 pool fresh --reply 40 "$@" ;;
      *) probe "$prefix-$case-$i" probe.py "$case" "$@" ;;
    esac
  done
}

case $block in
  1)
    use frozen
    tiers frozen-e0
    runs frozen kv 3
    runs frozen prefill 3
    runs frozen tail 2
    ;;
  2)
    use frozen --max-num-batched-tokens 2048 --long-prefill-token-threshold 0
    runs stock2048 kv 3
    runs stock2048 tail 2
    use frozen --max-num-batched-tokens "$cap" --long-prefill-token-threshold $((cap / 2))
    runs "stock$cap-$((cap / 2))" kv 1
    runs "stock$cap-$((cap / 2))" tail 2
    ;;
  3)
    use make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48
    runs pr kv 3
    runs pr prefill 3
    runs pr tail 2
    ;;
esac
log "block $block complete"
