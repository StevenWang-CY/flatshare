#!/usr/bin/env bash
# sweep.sh NAME SILENCE/DELAY...: the endpointing sweep, from a workstation with SSH to the hosts, like
# benchmarks/e2/campaign.sh. The end of a turn waits for Silero VAD to hear SILENCE seconds of silence (0.55 by
# default) and then for LiveKit's endpointing DELAY (0.5 by default): most of the voice turn's floor on this box. For
# each SILENCE/DELAY, on cyan, it sets the voice worker's FLATSHARE_VAD_MIN_SILENCE and FLATSHARE_ENDPOINTING_MIN_DELAY
# in a drop-in (variant-endpointing-sweep.conf), restarts the worker, and runs one measurement caller that plays every
# line of the caller corpus once, in order (flatshare-voice bots --in-order, no interruptions, a warm-up turn first),
# so that each setting meets the same pauses inside the same lines. Each turn's V2V, and whether the agent began to
# answer before the line had ended (spoke_during_turn: a pause inside the line taken for its end), come from the
# caller; each turn's endpointing delay from the worker's turn lines. The worker goes back to its own settings at the
# end, whatever happened. The sweep holds E2's lock on cyan throughout, so that no point, drill or soak runs beside
# it, and needs the worker in bench mode with fixed endpointing (the vad-turns variant, as E2 runs it) and the GPU
# node serving. Into $FLATSHARE_SWEEPS/NAME (default ~/flatshare/runs/endpointing) it fetches each setting's record:
# setting.json (the two values, the worker's start, the release), bots/ (turns.jsonl, summary.json) and
# voice-worker.log (the worker's journal while the caller ran); benchmarks/endpointing/summary.py tables them.
# FLATSHARE_SSH and FLATSHARE_CYAN (default cyan) as for campaign.sh.
set -euo pipefail
usage() { echo "usage: sweep.sh NAME SILENCE/DELAY..." >&2; exit 2; }
(($# >= 2)) || usage
name=$1
shift
for setting in "$@"; do
  [[ $setting =~ ^[0-9]+(\.[0-9]+)?/[0-9]+(\.[0-9]+)?$ ]] || { echo "sweep: not SILENCE/DELAY: $setting" >&2; exit 2; }
done
ssh=${FLATSHARE_SSH:-ssh}
cyan=${FLATSHARE_CYAN:-cyan}
out=${FLATSHARE_SWEEPS:-$HOME/flatshare/runs/endpointing}/$name
[[ ! -e $out ]] || { echo "sweep: $out exists" >&2; exit 2; }

# The remote script is quoted: cyan expands it, with NAME and the settings as its arguments.
$ssh "$cyan" bash -s -- "$name" "$@" <<'REMOTE'
set -euo pipefail
name=$1
shift
root=$HOME/flatshare
lock=$root/runs/.flatshare-bench-e2.lock
dropin=$HOME/.config/systemd/user/flatshare-voice-worker.service.d/variant-endpointing-sweep.conf
dir=$root/runs/endpointing/$name
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
mkdir -p "$(dirname "$lock")"
exec 9>>"$lock"
flock -n 9 || { echo "sweep: another e2 run, reset, drill or soak holds $lock" >&2; exit 3; }
echo $$ >"$lock"
[[ ! -e $dir ]] || { echo "sweep: $dir exists on cyan" >&2; exit 2; }
env=$(systemctl --user show flatshare-voice-worker -p Environment --value)
[[ $env == *FLATSHARE_TURN_DETECTION=vad* ]] ||
  { echo "sweep: the worker does not run fixed endpointing: link deploy/cyan/variants/vad-turns" >&2; exit 2; }
[[ $env != *FLATSHARE_VOICE_MODE=serve* ]] || { echo "sweep: the worker runs serve mode: measure bench mode" >&2; exit 2; }
code=$(curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:18080/ready || true)
[[ $code == 200 ]] || { echo "sweep: the gateway's /ready answers $code: the GPU node must serve" >&2; exit 2; }
lines=$(python3 -c "import json, sys; print(sum(e.get('kind') == 'turn' for e in json.load(open(sys.argv[1])).values()))" \
  "$root/data/callers/manifest.json")

restart() {
  local since
  since=$(date +%s)
  systemctl --user daemon-reload
  systemctl --user restart flatshare-voice-worker
  for _ in $(seq 120); do
    if journalctl --user -u flatshare-voice-worker --since "@$since" -o cat 2>/dev/null | grep -q 'registered worker'; then
      echo "$since"
      return
    fi
    sleep 1
  done
  echo "sweep: the worker did not register within 120 s" >&2
  return 1
}
restore() {
  rm -f "$dropin"
  restart >/dev/null && log "the worker runs its own settings again" || log "the worker did not come back: see its journal"
}
trap restore EXIT

mkdir -p "$dir" "$(dirname "$dropin")"
set -a
. "$HOME/.config/flatshare/livekit.env"
set +a
for setting in "$@"; do
  silence=${setting%/*} delay=${setting#*/}
  here=$dir/silence-$silence-delay-$delay
  mkdir -p "$here"
  printf '[Service]\nEnvironment=FLATSHARE_VAD_MIN_SILENCE=%s FLATSHARE_ENDPOINTING_MIN_DELAY=%s\n' "$silence" "$delay" >"$dropin"
  started=$(restart)
  log "silence $silence s, delay $delay s: the worker restarted; $lines lines in order"
  python3 -c "import json, sys; json.dump(dict(vad_min_silence=float(sys.argv[1]), endpointing_min_delay=float(sys.argv[2]),
worker_started=int(sys.argv[3]), release=sys.argv[4]), open(sys.argv[5], 'w'), indent=1)" \
    "$silence" "$delay" "$started" "$(readlink "$root/current")" "$here/setting.json"
  "$root/src/deploy/bin/in-voicenet" "$root/venvs/voice/bin/flatshare-voice" bots --corpus "$root/data/callers" \
    --callers 1 --turns "$lines" --in-order --barge-in 0 --warmup-turns 1 --out "$here/bots" >"$here/bots.log" 2>&1 ||
    log "the caller exited $? (bots.log)"
  journalctl --user -u flatshare-voice-worker --since "@$started" -o cat >"$here/voice-worker.log"
  log "silence $silence s, delay $delay s: $(grep -c . "$here/bots/turns.jsonl" || true) turns"
done
REMOTE
mkdir -p "$out"
$ssh "$cyan" "tar -C flatshare/runs/endpointing/$name -cf - ." | tar -x -C "$out"
echo "sweep: $out"
