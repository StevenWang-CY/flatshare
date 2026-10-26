#!/usr/bin/env bash
# run.sh [--until HH:MM] [--cap TOKENS] [--estimate BLOCK=MINUTES]... [BLOCK...]: the A100 run on this host
# (docs/HARDWARE.md, the core block by block), block by block. A block is complete when it ends, a block done before is
# skipped, so that a rerun resumes and a stop anywhere leaves whole comparisons; without BLOCK it runs the core, 0 to
# 10, in order. Started from an SSH session, it outlives the session:
#   nohup setsid bash ~/flatshare/src/benchmarks/a100/run.sh >>~/flatshare/runs/a100/run.log 2>&1 </dev/null &
# The core:
#   0     bootstrap.sh gpu and warm (the preflight with the card, each compile configuration started once), the
#         box's smoke.sh (from the source tree, its listener audit beside it), and one 2-turn call through LiveKit
#   1-3   the probe blocks (probes.sh; --cap is block 2's time-equivalent step cap)
#   4     E2 round 1 (a100-r1.campaign)
#   5     the calibration point (calibration.sh, 16 sessions)
#   6-9   E2 rounds 2 to 5
#   10    the GPU units stopped; the records stay for deploy/a100/pull.sh
# The optional arms (docs/HARDWARE.md, the optional arms), by name, in its order: cprime (cprime.sh), branches
# (branches.sh run, once branches.sh stage has built the two envs), resubmit (the stock engine behind the gateway's
# preempt-resubmit, seeds 13 to 15), batch (batch.sh), aprime (the box's pool at 32 sessions, two rounds), tuned (the
# stock engine at the A100's best step cap in the probes, 2,048/0, on the samples of the core's rounds 1 and 2),
# prtuned (the PRs' values at that cap, on the same samples, paired with tuned) and backend (probes.sh backend: the KV
# case on FlashInfer's attention against block 1's on FlashAttention 2).
# --until HH:MM (UTC; tomorrow's when today's has passed) starts no block that the page's estimate says cannot end
# before then, and hands the deadline, once computed, as an epoch, to each campaign and to cprime.sh, which then start
# no point that cannot. --estimate BLOCK=MINUTES puts another estimate in the page's place for BLOCK (a block or an
# arm's name): a host measured faster or slower, or an E2 block run for its first point alone, which campaign.sh's own
# budget then stops at the deadline (the single-host rehearsal's). Each block's start and end go to
# ~/flatshare/runs/a100/blocks.log; a block done leaves blocks/BLOCK.done.
set -euo pipefail
usage() { echo "usage: run.sh [--until HH:MM] [--cap TOKENS] [--estimate BLOCK=MINUTES]... [BLOCK...]" >&2; exit 2; }
until="" cap=1536 estimates=""
while [[ ${1:-} == --* ]]; do
  case $1 in
    --until)
      [[ ${2:-} =~ ^([01][0-9]|2[0-3]):([0-5][0-9])$ ]] || usage
      until=$2
      ;;
    --cap)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || usage
      cap=$2
      ;;
    --estimate)
      [[ ${2:-} =~ ^(10|[0-9]|cprime|branches|resubmit|batch|aprime|tuned|prtuned|backend)=[1-9][0-9]*$ ]] || usage
      estimates+=" $2"
      ;;
    *) usage ;;
  esac
  shift 2
done
blocks=("$@")
((${#blocks[@]})) || blocks=(0 1 2 3 4 5 6 7 8 9 10)
root=$HOME/flatshare
src=$(cd "$(dirname "$0")/../.." && pwd)
here=$src/benchmarks/a100
node=$root/node
runs=$root/runs/a100
mkdir -p "$runs/blocks"
log() { echo "[$(date -u +%FT%TZ)] run: $*" | tee -a "$runs/blocks.log"; }
deadline=""
if [[ -n $until ]]; then
  now=$(date -u +%s)
  deadline=$((now - now % 86400 + 10#${until%:*} * 3600 + 10#${until#*:} * 60))
  ((deadline > now)) || deadline=$((deadline + 86400))
fi

# docs/HARDWARE.md's estimates, in minutes, for --until (block 0 assumes staging did not warm the caches), or the last
# --estimate given for the block.
minutes() {
  local pair given=""
  for pair in $estimates; do
    [[ ${pair%%=*} != "$1" ]] || given=${pair#*=}
  done
  if [[ -n $given ]]; then
    echo "$given"
    return
  fi
  case $1 in
    0) echo 44 ;; 1) echo 25 ;; 2) echo 26 ;; 3) echo 17 ;; 4) echo 29 ;; 5) echo 17 ;; 6 | 7 | 8 | 9) echo 28 ;;
    10) echo 10 ;; cprime) echo 95 ;; branches) echo 45 ;; resubmit) echo 36 ;; batch) echo 15 ;; aprime) echo 56 ;;
    tuned) echo 28 ;; prtuned) echo 24 ;; backend) echo 12 ;;
    *) echo 0 ;;
  esac
}

points_done() { # points_done CAMPAIGN: whether every point of the campaign file is complete
  local file=$1 name line
  name=$(basename "$file" .campaign)
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%%#*}
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}
    [[ -z $line || $line == agents-seed* || $line == kv-sample-s* ]] && continue
    grep -qs '"status": "complete"' "$root/runs/e2/$name/$(basename "$line" .toml)/run.json" || return 1
  done <"$file"
}

campaign() { # campaign FILE: its points on this host, as campaign.sh --local runs them
  local status=0
  # The run's one deadline, as an epoch: a campaign that ends after HH:MM hands the next none 24 hours away.
  (cd "$HOME" && FLATSHARE_NODE_DIR=flatshare/node bash "$src/benchmarks/e2/campaign.sh" --local \
    ${deadline:+--until-epoch "$deadline"} "$1") || status=$?
  points_done "$1" && return 0
  # campaign.sh exits 0 with points left only where --until stopped it; any other status is a failure.
  if [[ -n $deadline && $status == 0 ]]; then
    log "$(basename "$1" .campaign) stopped before its deadline ($until UTC) with points left"
    exit 0
  fi
  log "$(basename "$1" .campaign) did not complete (campaign.sh exit status $status)"
  exit 1
}

call() { # one 2-turn call through LiveKit, as an E2 point's callers make theirs
  local cores
  cores=$(sed -n 's/^caller_cores = "\(.*\)"$/\1/p' "$HOME/.config/flatshare/e2-host.toml")
  rm -rf "$runs/block0/call"
  (
    set -a
    # shellcheck source=/dev/null
    . "$HOME/.config/flatshare/livekit.env"
    set +a
    taskset -c "$cores" "$src/deploy/bin/in-voicenet" "$root/venvs/voice/bin/flatshare-voice" bots \
      --corpus "$root/data/callers" --callers 1 --turns 2 --out "$runs/block0/call"
  )
}

block() { # block NAME: run one block unless it is done
  local name=$1 need
  if [[ -f $runs/blocks/$name.done ]]; then
    log "block $name: done ($(cat "$runs/blocks/$name.done"))"
    return
  fi
  need=$(minutes "$name")
  if [[ -n $deadline ]] && (($(date -u +%s) + need * 60 > deadline)); then
    log "block $name: about $need minutes would end after $until UTC; stopping before it"
    exit 0
  fi
  log "block $name: start"
  # shellcheck source=/dev/null
  [[ ! -f $node/env.sh ]] || source "$node/env.sh"
  case $name in
    0)
      bash "$src/deploy/a100/bootstrap.sh" gpu
      # Staging may have warmed every compile configuration already (bootstrap.sh warm before the billed clock).
      if "$root/venvs/bench/bin/python" "$src/deploy/a100/host.py" warmed "$node" >/dev/null 2>&1; then
        log "block 0: every compile configuration warm already (cache/warm.json)"
      else
        bash "$src/deploy/a100/bootstrap.sh" warm
      fi
      mkdir -p "$runs/block0"
      bash "$node/variant.sh" use frozen
      bash "$src/deploy/phastform/smoke.sh" | tee "$runs/block0/smoke.txt"
      call | tee "$runs/block0/call.txt"
      ;;
    1 | 3) bash "$here/probes.sh" "$name" ;;
    2) bash "$here/probes.sh" 2 --cap "$cap" ;;
    4) campaign "$here/a100-r1.campaign" ;;
    5) bash "$here/calibration.sh" --sessions 16 ;;
    6 | 7 | 8 | 9) campaign "$here/a100-r$((name - 4)).campaign" ;;
    10) systemctl --user stop flatshare.target flatshare-engine flatshare-stt flatshare-tts ;;
    cprime) # its rounds take the run's deadline; it exits 3 where that stopped it with points left
      local status=0
      bash "$here/cprime.sh" --cap "$cap" ${deadline:+--until-epoch "$deadline"} || status=$?
      if ((status == 3)) && [[ -n $deadline ]]; then
        log "cprime stopped before its deadline ($until UTC) with points left"
        exit 0
      fi
      ((status == 0)) || { log "cprime did not complete (cprime.sh exit status $status)"; exit 1; }
      ;;
    branches) bash "$here/branches.sh" run ;;
    batch) bash "$here/batch.sh" ;;
    resubmit) for n in 3 4 5; do campaign "$here/a100-resubmit-r$n.campaign"; done ;;
    aprime) for n in 1 2; do campaign "$here/a100-aprime-r$n.campaign"; done ;;
    tuned) for n in 1 2; do campaign "$here/a100-tuned-r$n.campaign"; done ;;
    prtuned) for n in 1 2; do campaign "$here/a100-prtuned-r$n.campaign"; done ;;
    backend) bash "$here/probes.sh" backend ;;
    *) log "no block $name"; exit 2 ;;
  esac
  date -u +%FT%TZ >"$runs/blocks/$name.done"
  log "block $name: done"
}

log "blocks ${blocks[*]}${until:+, until $until UTC}${estimates:+, estimates$estimates}"
# The control plane (the voice worker, LiveKit in the voice plane, the gateway), which a reboot leaves down (nothing
# starts it at boot: flatshare-svc is no boot-time service, and the units are not enabled): block 0's call and every E2
# point need it, and starting it where it runs is nothing.
# shellcheck source=/dev/null
[[ ! -f $node/env.sh ]] || source "$node/env.sh"
systemctl --user start flatshare-voice-worker.service ||
  { log "the control plane did not start: journalctl --user -u flatshare-voice-worker (and flatshare-livekit, flatshare-gateway)"; exit 1; }
for name in "${blocks[@]}"; do
  block "$name"
done
log "complete: ${blocks[*]}"
