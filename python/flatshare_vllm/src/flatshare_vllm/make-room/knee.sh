#!/usr/bin/env bash
# knee.sh [--out DIR] [--seconds N] [--runs N] [--reply TOKENS] [--mode fresh|grow] [WINDOW...]: the budget cap's
# window against whole call replies, on the GPU node, from its deployed source. With the measurement drop-in on
# (variant.sh measure on: /server_info, where the check reads the window), for each WINDOW (16, 24, 32, 48 and 64
# tokens by default; `off` runs the KV preemption without the budget cap, for reference) the box runs make-room with the
# PRs' switches but that window (variant.sh use make-room --additional-config.make_room.reserve_tokens 0
# --additional-config.make_room.budget_cap_tokens WINDOW), the engine's additional_config must read it back, and
# tail.py runs RUNS (2) times for SECONDS (240) with the box's agents and fresh prompts and call replies of TOKENS (40)
# tokens, each timed to its first, 16th and last token, into DIR/wWINDOW-RUN.out (DIR: ~/flatshare/runs/window-knee).
# A run whose output ends in tail.py's summary line is skipped, so running the sweep again resumes it. DIR also gets the
# deployed revision (DEPLOYED), each window's additional_config as /server_info reports it (wWINDOW-engine.json), the
# engine's make-room lines over its runs (wWINDOW-make-room.log) and the sweep's log (knee.log). At the end, or at a
# failure, the box runs the frozen engine again. With the defaults it takes about 45 minutes: five engine switches of
# about 100 s and ten runs of 240 s.
set -euo pipefail
usage() {
  echo "usage: knee.sh [--out DIR] [--seconds N] [--runs N] [--reply TOKENS] [--mode fresh|grow] [WINDOW...]" >&2
  exit 2
}
out=$HOME/flatshare/runs/window-knee seconds=240 runs=2 reply=40 mode=fresh
while [[ ${1:-} == --* ]]; do
  (($# >= 2)) || usage
  case $1 in
    --out) out=$2 ;;
    --seconds) seconds=$2 ;;
    --runs) runs=$2 ;;
    --reply) reply=$2 ;;
    --mode) mode=$2 ;;
    *) usage ;;
  esac
  shift 2
done
windows=("$@")
((${#windows[@]})) || windows=(16 24 32 48 64)
for w in "${windows[@]}"; do [[ $w == off || $w =~ ^[1-9][0-9]*$ ]] || usage; done
[[ $seconds =~ ^[1-9][0-9]*$ && $runs =~ ^[1-9][0-9]*$ && $reply =~ ^[1-9][0-9]*$ ]] || usage
[[ $mode == fresh || $mode == grow ]] || usage
here=$(cd "$(dirname "$0")" && pwd)  # python/flatshare_vllm/src/flatshare_vllm/make-room
src=$(cd "$here/../../../../.." && pwd) # the deployed source tree
node=$src/deploy/phastform
py=$HOME/flatshare/venvs/engine/bin/python
engine=unix:${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/flatshare/engine.sock # the GPU node's engine socket
mkdir -p "$out"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$out/knee.log"; }
complete() { [[ -s $1 ]] && tail -n 1 "$1" | grep -q '^calls: .* stopped early$'; }
back() {
  log "knee: the box back on the frozen engine"
  bash "$node/variant.sh" use frozen >>"$out/knee.log" 2>&1 || log "knee: variant.sh use frozen failed"
}

deployed=$(cat "$src/DEPLOYED" 2>/dev/null || echo unknown)
echo "$deployed" >"$out/DEPLOYED"
log "knee: windows ${windows[*]}, $runs runs of $seconds s each, $mode prompts, $reply-token replies; deployed $deployed"
trap back EXIT
bash "$node/variant.sh" measure on >>"$out/knee.log" 2>&1 ||
  { log "knee: variant.sh measure on failed (knee.log has its output)"; exit 1; }
for w in "${windows[@]}"; do
  pending=()
  for ((i = 1; i <= runs; i++)); do complete "$out/w$w-$i.out" || pending+=("$i"); done
  if ((${#pending[@]} == 0)); then
    log "window $w: complete"
    continue
  fi
  if [[ $w == off ]]; then
    flags=(--additional-config.make_room.budget_cap false)
  else
    flags=(--additional-config.make_room.budget_cap_tokens "$w")
  fi
  log "window $w: switching the engine (${flags[*]})"
  bash "$node/variant.sh" use make-room --additional-config.make_room.reserve_tokens 0 "${flags[@]}" \
    >>"$out/knee.log" 2>&1 || { log "window $w: variant.sh failed (knee.log has its output)"; exit 1; }
  curl -fsS --unix-socket "${engine#unix:}" "http://localhost/server_info?config_format=json" | "$py" -c '
import json, sys
config = json.load(sys.stdin)["vllm_config"].get("additional_config") or {}
print(json.dumps(config, indent=2, sort_keys=True))' >"$out/w$w-engine.json" ||
    { log "window $w: no additional_config from the engine's /server_info"; exit 1; }
  # The PRs' switches with this window: two preemptions a step, no reserve, the budget cap at the window (or off).
  problem=$("$py" - "$out/w$w-engine.json" "$w" 2>&1 <<'EOF'
import json, sys
room = json.load(open(sys.argv[1])).get("make_room", {})
window = sys.argv[2]
want = {"max_preemptions_per_step": "2", "reserve_tokens": "0", "budget_cap": "false" if window == "off" else "true"}
if window != "off":
    want["budget_cap_tokens"] = window
def same(reported, text):
    if isinstance(reported, bool):
        return str(reported).lower() == text
    try:
        return float(reported) == float(text)
    except (TypeError, ValueError):
        return str(reported).lower() == text
bad = {key: room.get(key) for key, text in want.items() if not same(room.get(key), text)}
if bad:
    sys.exit(f"the engine runs make_room {bad}, the window needs {want}")
EOF
) || { log "window $w: $problem"; exit 1; }
  since=$(date '+%Y-%m-%d %H:%M:%S')
  for i in "${pending[@]}"; do
    log "window $w: run $i"
    if ! "$py" "$here/tail.py" "$engine" "$seconds" pool "$mode" --reply "$reply" >"$out/w$w-$i.part" 2>&1; then
      log "window $w: run $i failed: $(tail -n 1 "$out/w$w-$i.part")"
      exit 1
    fi
    mv "$out/w$w-$i.part" "$out/w$w-$i.out"
    log "window $w: run $i: $(tail -n 1 "$out/w$w-$i.out")"
  done
  journalctl --user -u flatshare-engine --since "$since" | grep -E 'make-room:' >"$out/w$w-make-room.log" || true
done
log "knee: done"
