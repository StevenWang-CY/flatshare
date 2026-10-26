#!/usr/bin/env bash
# batch.sh [--rounds N]: the A100 run's optional arm 4 (docs/HARDWARE.md, the optional arms, P-1's slot S-7), on this
# host: the scheduler's cost at a datacenter batch. vllm bench serve keeps 256 requests in flight on the core's engine,
# 26 of them (10%) at priority 0 and 230 at priority 2 (two clients side by side, each its requests' priority in
# --extra-body), N rounds (20) of each client's concurrency, on three engines in turn: frozen (stock vLLM 0.32.0),
# make-room with its switches off (its code path, nothing it does), and make-room with the PRs' values. Each request
# holds its KV beside the 255 others' in the pool: random 48-token prompts and 64 output tokens, 112 tokens, 7 blocks
# of 16 (8 where a prompt re-tokenizes a few tokens longer), so 256 take 2,048 of the core pool's 2,275 blocks (5 GiB)
# and the whole batch runs at once: the arm measures the scheduler's per-step cost at a batch of 256, not a queue
# behind the KV, which 1,024-token prompts would have made of it (36,400 tokens hold about 28 of them). Before each
# engine's clients it reads the engine's KV geometry (vllm:cache_config_info) and refuses a pool that would not hold
# the batch; while they run it samples the engine's running and waiting requests once a second (ENGINE-running.txt),
# and ENGINE-running.json keeps their median and maximum, so that the record shows the batch formed. Each client's
# result (vllm bench serve's JSON: request and output throughput, TTFT, TPOT and ITL percentiles) goes to
# ~/leaseholder/runs/a100/batch/ENGINE-pPRIORITY.json; a result already there is kept.
# vllm bench serve speaks TCP and the engine serves on its socket only: the clients reach it through
# deploy/bin/lh-forward, from a loopback port (LH_FORWARD_PORT, 18000) to the socket, on their own cores, the same
# hop for each engine.
set -euo pipefail
rounds=20
while (($#)); do
  case $1 in
    --rounds)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || { echo "batch: --rounds takes a positive number" >&2; exit 2; }
      rounds=$2
      shift
      ;;
    *) echo "usage: batch.sh [--rounds N]" >&2; exit 2 ;;
  esac
  shift
done
root=$HOME/leaseholder
node=${LH_NODE_DIR:-$root/node}
src=$(cd "$(dirname "$0")/../.." && pwd)
port=${LH_FORWARD_PORT:-18000}
engine=http://127.0.0.1:$port # lh-forward's end of the engine's socket
out=$root/runs/a100/batch
mkdir -p "$out"
# shellcheck source=/dev/null
[[ ! -f $node/env.sh ]] || source "$node/env.sh"
sock=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/lh/engine.sock # the engine's own socket (lh-engine.service)
log() { echo "[$(date -u +%FT%TZ)] batch: $*" | tee -a "$out/batch.log"; }
trap 'jobs -p | xargs -r kill 2>/dev/null || true' EXIT # a client left when the other failed
cores=$(sed -n 's/^aiperf_cores = "\(.*\)"$/\1/p' "$HOME/.config/leaseholder/e2-host.toml")
[[ -n $cores ]] || { log "no aiperf_cores in ~/.config/leaseholder/e2-host.toml (deploy/a100/host.py config)"; exit 2; }
taskset -c "$cores" "$root/venvs/bench/bin/python" "$src/deploy/bin/lh-forward" "127.0.0.1:$port=unix:$sock" \
  >"$out/forward.log" 2>&1 &

batch=256 input=48 output=64 slack=16 # the requests in flight, and each one's tokens (slack: a longer re-tokenization)

# vllm bench serve loads its tokenizer by name at the Hub's main, which the node's download by commit leaves unresolved
# offline (no refs/main): the clients load the engine's own snapshot, at the revision engine.yaml pins, from the hub
# cache as huggingface_hub resolves it (deploy/profile.py's hub_cache).
hub=${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}
pinned=$(sed -n 's/^model: *//p' "$node/engine.yaml")
revision=$(sed -n 's/^revision: *//p' "$node/engine.yaml")
tokenizer=$hub/models--${pinned//\//--}/snapshots/$revision
[[ -f $tokenizer/tokenizer_config.json ]] ||
  { log "no tokenizer for $pinned at ${revision:-no revision} in $hub (bootstrap.sh stage downloads it)"; exit 2; }

geometry() { # the engine's KV blocks and their size, from its /metrics (vllm:cache_config_info's labels)
  curl -fsS -m 10 --unix-socket "$sock" http://localhost/metrics | "$root/venvs/bench/bin/python" -c '
import re, sys
info = re.search(r"^vllm:cache_config_info[{]([^}]*)[}]", sys.stdin.read(), re.M)
labels = dict(re.findall(r"(\w+)=\"([^\"]*)\"", info.group(1))) if info else {}
print(labels.get("num_gpu_blocks", "?"), labels.get("block_size", "?"))'
}

sample_running() { # sample_running FILE: the engine's running and waiting requests once a second, until it is killed
  while :; do
    curl -fsS -m 2 --unix-socket "$sock" http://localhost/metrics 2>/dev/null | awk -v t="$(date -u +%s)" '
      /^vllm:num_requests_running[{ ]/ { r = $NF } /^vllm:num_requests_waiting[{ ]/ { w = $NF }
      END { if (r != "") print t, r, w }' >>"$1" || true
    sleep 1
  done
}

client() { # client NAME PRIORITY CONCURRENCY SEED: one vllm bench serve, in the background
  local name=$1 priority=$2 concurrency=$3 seed=$4
  HF_HUB_OFFLINE=1 taskset -c "$cores" "$root/venvs/engine/bin/vllm" bench serve --backend vllm \
    --base-url "$engine" --endpoint /v1/completions --model "$model" --tokenizer "$tokenizer" \
    --dataset-name random --random-input-len "$input" --random-output-len "$output" --ignore-eos --seed "$seed" \
    --num-prompts $((rounds * concurrency)) --max-concurrency "$concurrency" \
    --extra-body "{\"priority\": $priority}" --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
    --save-result --result-dir "$out" --result-filename "$name-p$priority.partial.json" \
    >"$out/$name-p$priority.log" 2>&1
}

measure() { # measure NAME VARIANT [FLAG...]
  local name=$1 a b sampler blocks size need
  shift
  if [[ -f $out/$name-p0.json && -f $out/$name-p2.json ]]; then
    log "$name: done before"
    return
  fi
  log "$name: $*"
  bash "$node/variant.sh" use "$@" 2>&1 | tee -a "$out/batch.log"
  model=$(curl -fsS -m 10 --unix-socket "$sock" http://localhost/v1/models | "$root/venvs/engine/bin/python" -c \
    'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
  read -r blocks size < <(geometry)
  [[ $blocks =~ ^[0-9]+$ && $size =~ ^[1-9][0-9]*$ ]] ||
    { log "$name: no KV geometry in the engine's /metrics (vllm:cache_config_info)"; exit 2; }
  need=$((batch * ((input + output + slack + size - 1) / size)))
  if ((need > blocks)); then
    log "$name: the pool holds $blocks blocks of $size; $batch requests of $((input + output)) tokens need $need, so" \
      "the batch would not form"
    exit 2
  fi
  rm -f "$out/$name-running.txt"
  sample_running "$out/$name-running.txt" &
  sampler=$!
  client "$name" 2 230 2 &
  a=$!
  client "$name" 0 26 0 &
  b=$!
  wait "$a" || { log "$name: the priority-2 client failed ($out/$name-p2.log)"; exit 1; }
  wait "$b" || { log "$name: the priority-0 client failed ($out/$name-p0.log)"; exit 1; }
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  log "$name: $("$root/venvs/bench/bin/python" - "$out/$name-running.txt" "$out/$name-running.json" "$blocks" \
    "$need" "$batch" <<'PY'
import json, statistics, sys
rows = [line.split() for line in open(sys.argv[1]) if len(line.split()) >= 2]
running = [float(row[1]) for row in rows]
waiting = [float(row[2]) for row in rows if len(row) > 2]
record = {
    "samples": len(rows),
    "running_p50": statistics.median(running) if running else None,
    "running_max": max(running) if running else None,
    "waiting_max": max(waiting) if waiting else None,
    "blocks": int(sys.argv[3]),
    "blocks_needed": int(sys.argv[4]),
    "batch": int(sys.argv[5]),
}
with open(sys.argv[2], "w") as f:
    json.dump(record, f, indent=2)
if running:
    print(f"running p50 {record['running_p50']:g}, max {record['running_max']:g} (the batch: {record['batch']})")
else:
    print("no running-requests samples")
PY
)"
  for priority in 0 2; do
    mv "$out/$name-p$priority.partial.json" "$out/$name-p$priority.json"
  done
  log "$name: $(grep -h -E '^Output token throughput' "$out/$name-p2.log" "$out/$name-p0.log" | tr '\n' ' ')"
}

measure frozen frozen
measure off make-room --additional-config.make_room.max_preemptions_per_step 0 \
  --additional-config.make_room.budget_cap false --additional-config.make_room.reserve_tokens 0
measure on make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48
bash "$node/variant.sh" use frozen 2>&1 | tee -a "$out/batch.log"
log "complete"
