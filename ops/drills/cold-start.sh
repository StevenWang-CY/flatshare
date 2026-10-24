#!/usr/bin/env bash
# cold-start.sh [--cold] engine|box [NAME]: time a start of the GPU node's servers, from a workstation with SSH to the
# hosts, like deploy/deploy.sh. engine restarts the engine alone, with the audio servers running, as variant.sh does;
# box restarts STT, TTS and the engine (leaseholder.target), as after a reboot. --cold starts the engine with empty
# compile and JIT caches (vLLM's, FlashInfer's, TorchInductor's and Triton's, in a fresh directory a drop-in names), as
# on a new host or after an upgrade, then removes the drop-in and its caches and restarts the engine warm. The model's
# files stay in the page cache: a start after a reboot also reads them from disk. Into $LH_DRILLS/NAME (default
# ~/leaseholder/runs/drills) it writes each unit's journal from the restart on, and prints each start's phases
# (`lh-bench drill start`, on cyan). LH_SSH, LH_CYAN and LH_GPU as for benchmarks/e2/campaign.sh, with one GPU node.
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
ssh=${LH_SSH:-ssh}
cyan=${LH_CYAN:-cyan}
gpu=${LH_GPU:-phastform}
out=${LH_DRILLS:-$HOME/leaseholder/runs/drills}/$name
remote=leaseholder/runs/drills/$name # on cyan
bench=leaseholder/venvs/bench/bin/lh-bench
dropin=.config/systemd/user/lh-engine.service.d/zz-cold-start.conf
caches=leaseholder/cache/cold-start # on the GPU node
mkdir -p "$out"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$out/drill.log"; }

$ssh "$gpu" 'systemctl --user is-active --quiet lh-engine' || { log "the engine is not running"; exit 1; }
# The restart would break an E2 point or a drill: they hold this lock on cyan while they run. The check is advisory:
# the lock is not held through the restart, so start nothing else meanwhile.
$ssh "$cyan" 'flock -n ~/leaseholder/runs/.lh-bench-e2.lock true' || { log "an E2 point or a drill is running"; exit 1; }
restore() { # with --cold: remove the drop-in and its caches, and start the engine warm, whatever happened before
  $ssh "$gpu" "rm -f $dropin && rm -rf $caches && systemctl --user daemon-reload && systemctl --user restart lh-engine" &&
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
units=lh-engine target=lh-engine
if [[ $what == box ]]; then
  units="lh-stt lh-tts lh-engine" target=leaseholder.target
fi
# The journals are fetched whatever the start did: for a start that fails they are the record that matters.
before=$(date +%s)
status=0
event=$($ssh "$gpu" "date +%s.%N; systemctl --user reset-failed $units 2>/dev/null || true
  systemctl --user stop $target; systemctl --user start $target" | sed -n 1p) || status=$?
event=${event:-$before}
# leaseholder.target only Wants= its units, so its start succeeds without them: each must be active.
for unit in $units; do
  $ssh "$gpu" "systemctl --user is-active --quiet $unit" || { log "$unit is not active after the start"; status=1; }
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
