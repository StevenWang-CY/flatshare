#!/usr/bin/env bash
# drill.sh engine-kill|gateway-reload|gateway-restart|gateway-kill [NAME]: a failure drill on the deployed box, from a
# workstation with SSH to the hosts, like deploy/deploy.sh. On cyan, `flatshare-bench drill streams` opens realtime and
# background streams through the gateway. Once every one streams, the event:
#   engine-kill      SIGKILL to every process of the GPU node's engine unit, as an OOM kill or a crash would end it;
#                    systemd starts it again (Restart=on-failure, 5 s after its last process exits);
#   gateway-reload   systemctl --user reload flatshare-gateway on cyan, as a deploy does: the gateway hands its sockets
#                    to a successor and drains;
#   gateway-restart  systemctl --user restart flatshare-gateway: it drains, then a new one starts, while
#                    flatshare-gateway.socket holds the connections made meanwhile;
#   gateway-kill     SIGKILL to the gateway; systemd starts it again a second later, and its socket unit holds the
#                    connections made meanwhile.
# The event's time goes to the event file on cyan in the same command as the event (for engine-kill, at once after,
# from the GPU node's clock), and from the moment the client sees it, it probes both planes every second, each probe
# on a new connection, beside the streams, until they have ended and requests succeed again; from the first byte of
# every stream on it samples the engine's running and waiting requests, each sample stamped. Into $FLATSHARE_DRILLS/NAME
# (default ~/flatshare/runs/drills) it fetches the client's record (drill.json, client.log) and writes the event's
# time (event), the GPU node's /dev/shm before and after (the CPU tier lives there), the engine's scheduler class,
# and the journal from the event on (the engine's or the gateway's); then it prints `flatshare-bench drill report`.
# FLATSHARE_SSH, FLATSHARE_CYAN (default cyan) and FLATSHARE_GPU (default phastform) as for benchmarks/e2/campaign.sh,
# with one GPU node: the hosts by SSH host names or aliases. The drill runs on whatever variant the node runs: switch it
# first with deploy/phastform/variant.sh use. The scheduler class comes from /server_info, which only the measurement
# drop-in serves (variant.sh measure on), and never the serving variant; without it the scheduler file is empty.
set -euo pipefail
usage() { echo "usage: drill.sh engine-kill|gateway-reload|gateway-restart|gateway-kill [NAME]" >&2; exit 2; }
(($# >= 1 && $# <= 2)) || usage
kind=$1
case $kind in
  engine-kill) ;;
  gateway-reload) action="reload flatshare-gateway" ;;
  gateway-restart) action="restart flatshare-gateway" ;;
  gateway-kill) action="kill --signal=SIGKILL flatshare-gateway" ;;
  *) usage ;;
esac
name=${2:-$kind-$(date -u +%Y%m%dT%H%M%SZ)}
ssh=${FLATSHARE_SSH:-ssh}
cyan=${FLATSHARE_CYAN:-cyan}
gpu=${FLATSHARE_GPU:-phastform}
node=${gpu%%.*}
out=${FLATSHARE_DRILLS:-$HOME/flatshare/runs/drills}/$name
remote=flatshare/runs/drills/$name # on cyan, under the home directory
bench=flatshare/venvs/bench/bin/flatshare-bench
mkdir -p "$out"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$out/drill.log"; }
die() { log "drill: $*"; exit 1; }
# engine PATH: GET PATH from the node's engine on cyan, through its tunnel's socket (flatshare-tunnel@NODE).
engine() {
  $ssh "$cyan" "curl -fs -m 5 --unix-socket /run/user/\$(id -u)/flatshare/$node-engine.sock 'http://localhost$1'"
}
shm() { $ssh "$gpu" 'df -B1 --output=used,size /dev/shm | tail -n 1; ls -l /dev/shm'; }

{ engine /health >/dev/null && $ssh "$cyan" 'curl -fs -m 5 -o /dev/null 127.0.0.1:18080/health'; } ||
  die "the box is not up: the engine (through the tunnel) or the gateway does not answer on cyan"
engine '/server_info?config_format=json' | grep -o '"scheduler_cls": *"[^"]*"\|"scheduler_cls": *null' \
  >"$out/scheduler" || true
# A deliberate kill must not run into the units' crash-loop guards (StartLimitBurst).
$ssh "$gpu" 'systemctl --user reset-failed flatshare-engine 2>/dev/null || true'
$ssh "$cyan" 'systemctl --user reset-failed flatshare-gateway 2>/dev/null || true'
shm >"$out/shm-before.txt"

log "$kind: starting the client on cyan"
$ssh "$cyan" "mkdir -p $remote && cd $remote && rm -f event && (nohup ~/$bench drill streams --event-file event \
  --engine unix:/run/user/\$(id -u)/flatshare/$node-engine.sock --out drill.json >client.log 2>&1 </dev/null &)"
state=""
for _ in $(seq 180); do
  state=$($ssh "$cyan" "grep -Eo '^(streaming|streams ended)' $remote/client.log | head -n 1" || true)
  [[ -n $state ]] && break
  sleep 1
done
fetch() { rsync -a -e "$ssh" "$cyan:$remote/" "$out/"; }
[[ $state == streaming ]] || { fetch; die "the client did not stream (${state:-no line in 180 s}): see $out/client.log"; }
sleep 5 # every stream decodes for a few seconds first
# The event file appears by a rename, so that the client never reads half of it.
if [[ $kind == engine-kill ]]; then
  event=$($ssh "$gpu" 'date +%s.%N; systemctl --user kill --signal=SIGKILL flatshare-engine' | sed -n 1p)
  $ssh "$cyan" "echo $event >$remote/event.new && mv $remote/event.new $remote/event"
else
  # The command returns once systemctl does: at once for a reload or a kill, after the drain and the start for a
  # restart. Its output, a failed reload's among it, goes to event.log.
  $ssh "$cyan" "cd $remote && date +%s.%N >event.new && mv event.new event && systemctl --user $action >event.log 2>&1" ||
    log "$kind: systemctl --user $action failed: see $out/event.log"
  event=$($ssh "$cyan" "cat $remote/event")
fi
echo "$event" >"$out/event"
log "$kind at $event"
for _ in $(seq 900); do
  $ssh "$cyan" "test -s $remote/drill.json" && break
  sleep 1
done
shm >"$out/shm-after.txt"
fetch
[[ -s $out/drill.json ]] || die "the client wrote no record in 900 s: see $out/client.log"
if [[ $kind == engine-kill ]]; then
  $ssh "$gpu" "journalctl --user -u flatshare-engine -o short-unix --no-pager --since @${event%.*}" \
    >"$out/engine-journal.txt"
else
  $ssh "$cyan" \
    "journalctl --user -u flatshare-gateway -u flatshare-gateway.socket -o short-unix --no-pager --since @${event%.*}" \
    >"$out/gateway-journal.txt"
fi
$ssh "$cyan" "~/$bench drill report $remote/drill.json --label 'the ${kind/-/ }'" | tee "$out/report.txt"
log "$kind: done, $out"
