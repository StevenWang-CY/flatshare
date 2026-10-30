#!/usr/bin/env bash
# calibration.sh [--sessions N] [--seed S]: one of flatshare-sim's calibration points on this host (block 5 of the A100
# run, docs/BENCHMARKS.md, the core block by block; the box's are W2's, crates/flatshare-sim/README.md): on a freshly
# started frozen engine, AIPerf's agentic_replay of the whole 32x corpus (~/flatshare/data/weka-x32) straight to the
# engine at N sessions (16), 600 s plus 60 s of grace, a 120 s think cap, seed S (11), burst starts, with the engine's
# /metrics before and after. The engine serves on its socket only and AIPerf speaks TCP only, so AIPerf reaches it
# through deploy/bin/flatshare-forward, from a loopback port (FLATSHARE_FORWARD_PORT, 18000) to the socket, on AIPerf's
# cores, as the box's calibration points reached their engine through the tunnel's TCP forward. It writes
# ~/flatshare/runs/a100/calibration-sN as crates/flatshare-sim/scripts/calibration.py reads a run: before.prom,
# aiperf/profile_export.jsonl and after.prom, and a point already there is not measured again.
set -euo pipefail
sessions=16 seed=11
while (($#)); do
  case $1 in
    --sessions | --seed)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || { echo "calibration: $1 takes a positive number" >&2; exit 2; }
      if [[ $1 == --sessions ]]; then sessions=$2; else seed=$2; fi
      shift
      ;;
    *) echo "usage: calibration.sh [--sessions N] [--seed S]" >&2; exit 2 ;;
  esac
  shift
done
root=$HOME/flatshare
src=$(cd "$(dirname "$0")/../.." && pwd)
node=${FLATSHARE_NODE_DIR:-$root/node}
out=$root/runs/a100/calibration-s$sessions
bench=$root/venvs/bench/bin
configs=$src/python/flatshare_bench/configs
port=${FLATSHARE_FORWARD_PORT:-18000}
# shellcheck source=/dev/null
[[ ! -f $node/env.sh ]] || source "$node/env.sh"
sock=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/flatshare/engine.sock # the engine's own socket (flatshare-engine.service)
get() { curl -fsS -m 30 --unix-socket "$sock" "http://localhost$1"; } # get PATH: from the engine
log() { echo "[$(date -u +%FT%TZ)] calibration: $*" | tee -a "$root/runs/a100/calibration.log"; }
mkdir -p "$root/runs/a100"
if [[ -s $out/after.prom ]]; then
  log "$out is complete"
  exit 0
fi
[[ -d $root/data/weka-x32 ]] || { log "no corpus at $root/data/weka-x32 (deploy/a100/push.sh copies it)"; exit 2; }
# The load generators' cores, as flatshare-bench e2 run takes them (host.py config writes e2-host.toml from the core
# plan).
cores=$(sed -n 's/^aiperf_cores = "\(.*\)"$/\1/p' "$HOME/.config/flatshare/e2-host.toml")
[[ -n $cores ]] || { log "no aiperf_cores in ~/.config/flatshare/e2-host.toml (deploy/a100/host.py config)"; exit 2; }
rm -rf "$out"
mkdir -p "$out"
log "the frozen engine, freshly started"
bash "$node/variant.sh" use frozen 2>&1 | tee -a "$root/runs/a100/calibration.log"
# The units' crash-loop guard counts every start (StartLimitBurst), as variant.sh's own switches do.
systemctl --user reset-failed flatshare-engine.service 2>/dev/null || true
systemctl --user restart flatshare-engine.service # returns once the engine answers /health
model=$(get /v1/models | "$bench/python" -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
# AIPerf's way to the engine's socket, until this script ends.
taskset -c "$cores" "$bench/python" "$src/deploy/bin/flatshare-forward" "127.0.0.1:$port=unix:$sock" \
  >"$out/forward.log" 2>&1 &
forward=$!
trap 'kill "$forward" 2>/dev/null || true' EXIT
engine=http://127.0.0.1:$port
for _ in $(seq 100); do
  curl -fsS -m 2 -o /dev/null "$engine/health" 2>/dev/null && break
  sleep 0.1
done
curl -fsS -m 2 -o /dev/null "$engine/health" ||
  { log "flatshare-forward does not relay $engine to $sock ($out/forward.log)"; exit 1; }
get /metrics >"$out/before.prom"
log "$sessions sessions of weka-x32, seed $seed, $model, on cores $cores, through $engine"
(
  # shellcheck source=/dev/null
  . "$configs/aiperf.env"
  export FLATSHARE_BENCH_URL=$engine/v1/chat/completions FLATSHARE_BENCH_DATA=$root/data/weka-x32 \
    FLATSHARE_BENCH_SESSIONS=$sessions FLATSHARE_BENCH_DURATION=600 FLATSHARE_BENCH_GRACE=60 \
    FLATSHARE_BENCH_THINK_CAP=120 FLATSHARE_BENCH_SEED=$seed FLATSHARE_BENCH_BURST_STARTS=true \
    FLATSHARE_BENCH_ARTIFACTS=$out/aiperf FLATSHARE_BENCH_MODEL=$model FLATSHARE_BENCH_TOKENIZER=$model
  taskset -c "$cores" "$bench/aiperf" profile --config "$configs/agentic_replay.yaml"
) >"$out/aiperf.log" 2>&1 || { log "AIPerf failed: $out/aiperf.log"; exit 1; }
[[ -s $out/aiperf/profile_export.jsonl ]] || { log "AIPerf wrote no profile_export.jsonl: $out/aiperf.log"; exit 1; }
get /metrics >"$out/after.prom.partial"
mv "$out/after.prom.partial" "$out/after.prom" # last: a point is complete once it has after.prom
log "complete: $(wc -l <"$out/aiperf/profile_export.jsonl") requests in $out"
