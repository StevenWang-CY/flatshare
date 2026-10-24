#!/usr/bin/env bash
# drill.sh engine-kill|gateway-restart [NAME]: a failure drill on the deployed box, from a workstation with SSH to the
# hosts, like deploy/deploy.sh. On cyan, `lh-bench drill streams` opens realtime and background streams through the
# gateway; once every one streams, engine-kill sends SIGKILL to every process of the GPU node's engine unit, as an OOM
# kill or a crash would end it, and gateway-restart restarts cyan's gateway. systemd starts a killed engine again
# (Restart=on-failure, after 15 s). The client then probes until a request succeeds. Into $LH_DRILLS/NAME (default
# ~/leaseholder/runs/drills) it fetches the client's record (drill.json, client.log) and writes the event's time
# (event, the host's wall clock), the engine's running and waiting requests before the event, 2 s after it and at the
# end, the GPU node's /dev/shm before and after (the CPU tier lives there), the engine's scheduler class, and for
# engine-kill the engine's journal from the kill on; then it prints `lh-bench drill report`. LH_SSH, LH_CYAN and LH_GPU
# as for benchmarks/e2/campaign.sh, with one GPU node. The drill runs on whatever variant the node runs: switch it
# first with deploy/phastform/variant.sh use.
set -euo pipefail
usage() { echo "usage: drill.sh engine-kill|gateway-restart [NAME]" >&2; exit 2; }
(($# >= 1 && $# <= 2)) || usage
kind=$1
[[ $kind == engine-kill || $kind == gateway-restart ]] || usage
name=${2:-$kind-$(date -u +%Y%m%dT%H%M%SZ)}
ssh=${LH_SSH:-ssh}
cyan=${LH_CYAN:-cyan}
gpu=${LH_GPU:-phastform}
out=${LH_DRILLS:-$HOME/leaseholder/runs/drills}/$name
remote=leaseholder/runs/drills/$name # on cyan, under the home directory
bench=leaseholder/venvs/bench/bin/lh-bench
mkdir -p "$out"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$out/drill.log"; }
die() { log "drill: $*"; exit 1; }
requests() { $ssh "$cyan" 'curl -fs -m 5 127.0.0.1:18000/metrics' | grep -E '^vllm:num_requests_(running|waiting)\{' || true; }
shm() { $ssh "$gpu" 'df -B1 --output=used,size /dev/shm | tail -n 1; ls -l /dev/shm'; }

$ssh "$cyan" 'curl -fs -m 5 -o /dev/null 127.0.0.1:18000/health && curl -fs -m 5 -o /dev/null 127.0.0.1:18080/health' ||
  die "the box is not up: the engine (through the tunnel) or the gateway does not answer on cyan"
$ssh "$cyan" 'curl -fs -m 5 "127.0.0.1:18000/server_info?config_format=json"' | grep -o '"scheduler_cls": *"[^"]*"\|"scheduler_cls": *null' \
  >"$out/scheduler" || true
# A deliberate kill must not run into the units' crash-loop guard (StartLimitBurst=3 in 30 min).
$ssh "$gpu" 'systemctl --user reset-failed lh-engine 2>/dev/null || true'
shm >"$out/shm-before.txt"

log "$kind: starting the client on cyan"
$ssh "$cyan" "mkdir -p $remote && cd $remote && (nohup ~/$bench drill streams --out drill.json >client.log 2>&1 </dev/null &)"
state=""
for _ in $(seq 180); do
  state=$($ssh "$cyan" "grep -Eo '^(streaming|streams ended)' $remote/client.log | head -n 1" || true)
  [[ -n $state ]] && break
  sleep 1
done
fetch() { rsync -a -e "$ssh" "$cyan:$remote/" "$out/"; }
[[ $state == streaming ]] || { fetch; die "the client did not stream (${state:-no line in 180 s}): see $out/client.log"; }
sleep 5 # every stream decodes for a few seconds first
requests >"$out/requests-before.txt"
if [[ $kind == engine-kill ]]; then
  event=$($ssh "$gpu" 'date +%s.%N; systemctl --user kill --signal=SIGKILL lh-engine' | sed -n 1p)
else
  event=$($ssh "$cyan" 'date +%s.%N; systemctl --user restart lh-gateway' | sed -n 1p)
fi
echo "$event" >"$out/event"
log "$kind at $event"
sleep 2
requests >"$out/requests-after-2s.txt"
for _ in $(seq 900); do
  $ssh "$cyan" "test -s $remote/drill.json" && break
  sleep 1
done
requests >"$out/requests-end.txt"
shm >"$out/shm-after.txt"
fetch
[[ -s $out/drill.json ]] || die "the client wrote no record in 900 s: see $out/client.log"
if [[ $kind == engine-kill ]]; then
  $ssh "$gpu" "journalctl --user -u lh-engine -o short-unix --no-pager --since @${event%.*}" >"$out/engine-journal.txt"
fi
$ssh "$cyan" "~/$bench drill report $remote/drill.json --event $event --label 'the ${kind/-/ }'" | tee "$out/report.txt"
log "$kind: done, $out"
