#!/usr/bin/env bash
# cold-start.sh [--cold] engine|box [NAME]: time a start of the GPU node's servers, from a workstation with SSH to the
# hosts, like deploy/deploy.sh. engine restarts the engine alone, with the audio servers running, as variant.sh does;
# box restarts STT, TTS and the engine (flatshare.target), as after a reboot. --cold starts the engine with empty
# compile and JIT caches (vLLM's, FlashInfer's, TorchInductor's and Triton's, in a fresh directory a drop-in names), as
# on a new host or after an upgrade, then removes the drop-in and its caches and restarts the engine warm. The model's
# files stay in the page cache: a start after a reboot also reads them from disk. Into $FLATSHARE_DRILLS/NAME (default
# ~/flatshare/runs/drills) it writes each unit's journal from the restart on, and prints each start's phases
# (`flatshare-bench drill start`, on cyan). FLATSHARE_SSH, FLATSHARE_CYAN (default cyan) and FLATSHARE_GPU (default
# phastform) as for benchmarks/e2/campaign.sh, with one GPU node: the hosts by SSH host names or aliases.
set -euo pipefail
usage() { echo "usage: cold-start.sh [--cold] engine|box [NAME]" >&2; exit 2; }
cold=false caches_note=""
if [[ ${1:-} == --cold ]]; then
  cold=true caches_note=", with empty caches"
  shift
fi
(($# >= 1 && $# <= 2)) || usage
what=$1
[[ $what == engine || $what == box ]] || usage
name=${2:-start-$what${caches_note:+-cold}-$(date -u +%Y%m%dT%H%M%SZ)}
ssh=${FLATSHARE_SSH:-ssh}
cyan=${FLATSHARE_CYAN:-cyan}
gpu=${FLATSHARE_GPU:-phastform}
out=${FLATSHARE_DRILLS:-$HOME/flatshare/runs/drills}/$name
remote=flatshare/runs/drills/$name # on cyan
bench=flatshare/venvs/bench/bin/flatshare-bench
dropin=.config/systemd/user/flatshare-engine.service.d/zz-cold-start.conf
caches=flatshare/cache/cold-start # on the GPU node
mkdir -p "$out"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$out/drill.log"; }

$ssh "$gpu" 'systemctl --user is-active --quiet flatshare-engine' || { log "the engine is not running"; exit 1; }
# The restart would break an E2 point or a drill: they hold this lock on cyan while they run. The check is advisory:
# the lock is not held through the restart, so start nothing else meanwhile.
$ssh "$cyan" 'flock -n ~/flatshare/runs/.flatshare-bench-e2.lock true' || { log "an E2 point or a drill is running"; exit 1; }
restore() { # with --cold: remove the drop-in and its caches, and start the engine warm, whatever happened before
  $ssh "$gpu" "rm -f $dropin && rm -rf $caches && systemctl --user daemon-reload && systemctl --user restart flatshare-engine" &&
    log "the drop-in and its caches removed; the engine restarted warm" ||
    log "removing the drop-in or the warm restart failed: check $dropin on ${gpu%%.*}"
}
if $cold; then
  trap restore EXIT
  # Every cache directory the engine's JIT and compile steps read, emptied by pointing them at a new directory.
  $ssh "$gpu" "rm -rf $caches && mkdir -p ${dropin%/*} $caches && cat >$dropin" <<EOF
[Service]
Environment=VLLM_CACHE_ROOT=%h/$caches/vllm
Environment=FLASHINFER_WORKSPACE_BASE=%h/$caches/flashinfer
Environment=TORCHINDUCTOR_CACHE_DIR=%h/$caches/inductor
Environment=TRITON_CACHE_DIR=%h/$caches/triton
EOF
  $ssh "$gpu" 'systemctl --user daemon-reload'
fi
units=flatshare-engine target=flatshare-engine stops=flatshare-engine
if [[ $what == box ]]; then
  # Stopping flatshare.target alone returns once the target is down, before its units: a start right after it
  # cancels their pending stops. Naming every unit makes systemctl wait for each one's stop.
  units="flatshare-stt flatshare-tts flatshare-engine" target=flatshare.target
  stops="flatshare.target flatshare-stt flatshare-tts flatshare-engine"
fi
# The journals are fetched whatever the start did: for a start that fails they are the record that matters.
before=$(date +%s)
status=0
event=$($ssh "$gpu" "date +%s.%N; systemctl --user reset-failed $units 2>/dev/null || true
  systemctl --user stop $stops; systemctl --user start $target" | sed -n 1p) || status=$?
event=${event:-$before}
# flatshare.target only Wants= its units, so its start succeeds without them: each must be active, and must have
# become active after the restart, not stayed up through it.
for unit in $units; do
  since=$($ssh "$gpu" "systemctl --user is-active --quiet $unit &&
    date -d \"\$(systemctl --user show $unit -p ActiveEnterTimestamp --value)\" +%s") ||
    { log "$unit is not active after the start"; status=1; continue; }
  ((since >= ${event%.*})) || { log "$unit did not restart: active since $since, before the restart at $event"; status=1; }
done
if ((status)); then
  log "$what restarted at $event$caches_note, and its start FAILED (exit status $status)"
else
  log "$what restarted at $event$caches_note"
fi
for unit in $units; do
  $ssh "$gpu" "journalctl --user -u $unit -o short-unix --no-pager --since @${event%.*}" >"$out/$unit-journal.txt" || true
done
$ssh "$cyan" "mkdir -p $remote"
rsync -a -e "$ssh" "$out/" "$cyan:$remote/"
$ssh "$cyan" "cd $remote && ~/$bench drill start $(for u in $units; do printf '%s ' "$u-journal.txt"; done)" | tee "$out/report.txt"
exit "$status"
