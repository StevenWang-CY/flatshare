#!/usr/bin/env bash
# cprime.sh [--sessions N] [--rounds N] [--cap TOKENS]: the A100 run's optional arm 1, C' (docs/HARDWARE.md, the
# optional arms), on this host after the core: Qwen3-14B-FP8 at the card's own pool
# (deploy/profiles/a100-40gb-qwen3-14b.toml, rendered into ~/flatshare/node-14b), at N sessions (40, the forecast's
# largest running share for it), in N rounds (3) of tuned stock against the PRs' values, the first with the 14B's own
# voice alone for its own SLO. Tuned stock is P-1's rule: of the stock step-cap settings blocks 1 and 2 measured on the
# 4B (512/256, 2,048/0, the time-equivalent cap --cap), the one with the lowest calls' first-token p95 (probe_results.py
# pick-stock). In order:
#   1. the 14B's node rendered with this host's facts, its weights at their pin (bootstrap.sh stage fetches them
#      with FLATSHARE_MODELS; here if it did not);
#   2. the host switched to it: the GPU units stopped, the 14B node's units in place of the core's, the agent's
#      requests naming the 14B (the qwen3-14b worker variant); on any exit the core's node and worker come back;
#   3. its first starts (host.py warm: the frozen engine and the stock setting, each compiling its own graphs), the
#      pool checked against the profile's 7,334 blocks, smoke.sh (a reply with no <think>), E0 with one rep;
#   4. the arm files and campaigns written into ~/flatshare/runs/a100/cprime, and the campaigns run.
# A campaign done before is skipped by campaign.sh's own resume. --until-epoch SECONDS hands each round's campaign
# that deadline (benchmarks/a100/run.sh's); a round it stopped with points left ends C' with exit status 3.
set -euo pipefail
sessions=40 rounds=3 cap=1536 until_epoch=""
usage() { echo "usage: cprime.sh [--sessions N] [--rounds N] [--cap TOKENS] [--until-epoch SECONDS]" >&2; exit 2; }
while (($#)); do
  case $1 in
    --sessions | --rounds | --cap)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || { echo "cprime: $1 takes a positive number" >&2; exit 2; }
      case $1 in --sessions) sessions=$2 ;; --rounds) rounds=$2 ;; *) cap=$2 ;; esac
      shift
      ;;
    --until-epoch)
      [[ ${2:-} =~ ^[0-9]+$ ]] || usage
      until_epoch=$2
      shift
      ;;
    *) usage ;;
  esac
  shift
done
root=$HOME/flatshare
src=$(cd "$(dirname "$0")/../.." && pwd)
here=$src/benchmarks/a100
core=$root/node
node=$root/node-14b
profile=a100-40gb-qwen3-14b
out=$root/runs/a100/cprime
units=$HOME/.config/systemd/user
py=$root/venvs/bench/bin/python
mkdir -p "$out"
# shellcheck source=/dev/null
[[ ! -f $core/env.sh ]] || source "$core/env.sh"
log() { echo "[$(date -u +%FT%TZ)] cprime: $*" | tee -a "$out/cprime.log"; }

# 1. The node and its weights.
"$py" "$src/deploy/profile.py" render "$profile" --host "$core/host.json" --out "$node" | tee -a "$out/cprime.log"
cp "$core/host.json" "$node/host.json"
[[ ! -f $core/env.sh ]] || cp "$core/env.sh" "$node/env.sh"
read -r model revision < <(awk '$1 == "model:" {m = $2} $1 == "revision:" {r = $2} END {print m, r}' "$node/engine.yaml")
"$root/venvs/engine/bin/hf" download -q "$model" --revision "$revision" >/dev/null
read -r stock stock_flags < <("$py" "$here/probe_results.py" pick-stock "$root/runs/a100" --cap "$cap" 2>>"$out/cprime.log")
log "$model@${revision:0:7}; tuned stock: $stock (${stock_flags:-the frozen flags})"

# 2. The switch, and the switch back on any exit.
switch() { # switch NODE on|off
  systemctl --user stop flatshare.target flatshare-engine flatshare-stt flatshare-tts
  "$py" "$src/deploy/a100/host.py" units "$1" >/dev/null
  local variant=$units/flatshare-voice-worker.service.d/variant-qwen3-14b.conf
  if [[ $2 == on ]]; then
    mkdir -p "$(dirname "$variant")"
    ln -sf "$src/deploy/cyan/variants/qwen3-14b/flatshare-voice-worker.conf" "$variant"
  else
    rm -f "$variant"
  fi
  systemctl --user daemon-reload
  systemctl --user restart flatshare-voice-worker.service
  log "the host runs $(basename "$1")'s units, the worker variant qwen3-14b $2"
}
trap 'status=$?; switch "$core" off || log "the core node did not come back: host.py units $core"; exit $status' EXIT
switch "$node" on

# 3. The first starts, the pool, the smoke test, E0.
configs=$out/warm.conf
echo "14b-frozen frozen" >"$configs"
[[ $stock == frozen ]] || echo "14b-$stock frozen $stock_flags" >>"$configs"
"$py" "$src/deploy/a100/host.py" warm "$node" --configs "$configs" | tee -a "$out/cprime.log"
blocks=$("$py" -c 'import json, sys
entry = json.load(open(sys.argv[1]))["configurations"]["14b-frozen"]
print(entry.get("num_gpu_blocks") or "")' "$root/cache/warm.json")
want=$(($(sed -n 's/^kv-cache-memory: //p' "$node/engine.yaml") / (16 * 163840)))
[[ $blocks == "$want" ]] || { log "the 14B's pool has ${blocks:-no} blocks, not the profile's $want"; exit 1; }
bash "$node/variant.sh" use frozen | tee -a "$out/cprime.log"
bash "$src/deploy/phastform/smoke.sh" | tee "$out/smoke.txt"
! grep -q '<think>' "$out/smoke.txt" || { log "the 14B's reply thinks aloud: enable_thinking is not off"; exit 1; }
if [[ ! -f $out/e0/summary.json ]]; then
  "$root/venvs/bench/bin/flatshare-bench" tiers \
    --url "unix:${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/flatshare/engine.sock" \
    --model "$model" --reps 1 --out "$out/e0" |
    tail -3 | tee -a "$out/cprime.log"
fi

# 4. The arms and the campaigns.
points_done() { # points_done CAMPAIGN: whether every point of the campaign file, as campaign.sh reads it, is complete
  local name points point
  name=$(basename "$1" .campaign)
  points=$(bash "$src/benchmarks/e2/campaign.sh" --points "$1") || return 1
  while IFS= read -r point; do
    grep -qs '"status": "complete"' "$root/runs/e2/$name/$point/run.json" || return 1
  done <<<"$points"
}
arm() { # arm FILE LABEL VARIANT [FLAG...]: an agents arm at C''s sessions on the long corpus, on VARIANT (or frozen)
  local file=$1 label=$2 variant=$3 flag list=""
  shift 3
  for flag; do list+="${list:+, }\"$flag\""; done
  {
    echo "# Generated by benchmarks/a100/cprime.sh: C', Qwen3-14B-FP8 at the card's own pool, $sessions sessions."
    echo "arm = \"cprime\""
    echo "label = \"$label\""
    printf '\n[voice]\nwarmup_turns = 1\n\n[agents]\nsessions = %s\ndata = "weka-x32-long"\n' "$sessions"
    if [[ $variant != frozen || -n $list ]]; then
      printf '\n[engine]\n'
      [[ $variant == frozen ]] || printf 'variant = "%s"\n' "$variant"
      [[ -z $list ]] || printf 'flags = [%s]\n' "$list"
    fi
  } >"$out/$file"
}
cp "$src/benchmarks/e2/arms/w5-a1-voice.toml" "$out/cprime-voice.toml"
# shellcheck disable=SC2086 # the stock flags are words
arm "cprime-stock$stock-s$sessions.toml" "Qwen3-14B-FP8, tuned stock ($stock), a warm-up turn opening each call" \
  frozen $stock_flags
arm "cprime-pr-ab48-s$sessions.toml" "Qwen3-14B-FP8, the PRs' values (48-token window, no reserve), a warm-up turn" \
  make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48
for ((n = 1; n <= rounds; n++)); do
  {
    echo "# Generated by benchmarks/a100/cprime.sh: C' round $n of $rounds, the agents' sample of seed $((10 + n))."
    echo "agents-seed $((10 + n))"
    if ((n == 1)); then echo cprime-voice.toml; fi
    if ((n % 2)); then
      printf '%s\n' "cprime-stock$stock-s$sessions.toml" "cprime-pr-ab48-s$sessions.toml"
    else
      printf '%s\n' "cprime-pr-ab48-s$sessions.toml" "cprime-stock$stock-s$sessions.toml"
    fi
  } >"$out/a100-cprime-r$n.campaign"
  log "round $n"
  (cd "$HOME" && FLATSHARE_NODE_DIR=flatshare/node-14b FLATSHARE_PROFILE=$profile \
    bash "$src/benchmarks/e2/campaign.sh" --local ${until_epoch:+--until-epoch "$until_epoch"} \
    "$out/a100-cprime-r$n.campaign")
  # campaign.sh exits 0 with points left only where the deadline stopped it.
  if ! points_done "$out/a100-cprime-r$n.campaign"; then
    log "round $n stopped before its deadline with points left: C' stops here"
    exit 3
  fi
done
log "complete"
