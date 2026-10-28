#!/usr/bin/env bash
# node-failover.sh kill|stop [NAME]: the two-node failover drill, from a workstation with SSH to the hosts, like
# drill.sh. The gateway on cyan must run with [[nodes]] (crates/flatshare-gateway/README.md, Nodes; ops/RUNBOOK.md, Two
# GPU nodes), every node available. On cyan a client places FLATSHARE_CALLS calls (default 2) with POST /v1/sessions, as
# the voice worker does, and runs their turns, a streaming realtime chat completion of at most 32 tokens every
# FLATSHARE_TURN_GAP seconds (default 3) for each call, beside FLATSHARE_AGENTS agents' sessions (default 2) that stream
# background completions back to back, and it polls GET /nodes every second. After FLATSHARE_WARM turns of each call
# (default 5), the event, on the node FLATSHARE_FAIL names (default: the first call's node), whose host is
# FLATSHARE_FAIL_HOST (default: the node's name):
#   kill  SIGKILL to every process of the node's engine unit, as a crash would end it; systemd starts it again
#         (Restart=on-failure, 5 s after its last process exits), so the drill also sees the node come back;
#   stop  systemctl --user stop flatshare-engine: the engine stays down until the drill has ended, and then starts.
# The client goes on for FLATSHARE_AFTER seconds (default 240) after the event. Into $FLATSHARE_DRILLS/NAME (default
# ~/flatshare/runs/drills) it fetches the client's records: turns.tsv (one line a turn: when it began, the call, the node
# the session was on before and after it, its status, first byte and total seconds, and how its stream ended: done,
# error or cut), agents.tsv (the same for the agents' requests), nodes.jsonl (each /nodes answer, stamped), the event's
# time (event), the gateway's journal and the failed node's engine journal from the event on; then it prints a report:
# the turns before and after the event, when the gateway passed the node over and when it was healthy again, the calls'
# moves, the turns that failed or were cut, and how long the agents were held back. A failover does what the gateway's
# README says when the calls' turns move to the survivor within a turn or two, the turns that fail are those in flight
# at the event (cut, never retried after their first content), no turn after the first moved one fails, and the agents
# wait while the survivor covers for the failed node. FLATSHARE_SSH and FLATSHARE_CYAN (default cyan) as for drill.sh.
set -euo pipefail
usage() { echo "usage: node-failover.sh kill|stop [NAME]" >&2; exit 2; }
(($# >= 1 && $# <= 2)) || usage
kind=$1
case $kind in kill | stop) ;; *) usage ;; esac
name=${2:-node-$kind-$(date -u +%Y%m%dT%H%M%SZ)}
ssh=${FLATSHARE_SSH:-ssh}
cyan=${FLATSHARE_CYAN:-cyan}
calls=${FLATSHARE_CALLS:-2}
agents=${FLATSHARE_AGENTS:-2}
gap=${FLATSHARE_TURN_GAP:-3}
warm=${FLATSHARE_WARM:-5}
after=${FLATSHARE_AFTER:-240}
out=${FLATSHARE_DRILLS:-$HOME/flatshare/runs/drills}/$name
remote=flatshare/runs/drills/$name # on cyan, under the home directory
mkdir -p "$out"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$out/drill.log"; }
die() { log "node-failover: $*"; exit 1; }
# gw PATH [CURL ARGS]: the gateway's PATH on cyan, through its private socket (no key).
gw() {
  local path=$1
  shift
  $ssh "$cyan" "curl -fsS -m 5 --unix-socket /run/user/\$(id -u)/flatshare/gateway.sock $* 'http://localhost$path'"
}

nodes=$(gw /nodes) || die "GET /nodes failed on cyan: does the gateway run with [[nodes]]?"
python3 - "$nodes" <<'EOF' || die "the gateway's nodes are not all available (GET /nodes above)"
import json, sys
nodes = json.loads(sys.argv[1])["nodes"]
print("nodes:", ", ".join(f"{n['name']} ({n['verdict']}, {'available' if n['available'] else 'passed over'})" for n in nodes))
sys.exit(0 if len(nodes) >= 2 and all(n["available"] for n in nodes) else 1)
EOF
echo "$nodes" >"$out/nodes-before.json"

# The client, written to cyan and run there detached: it needs curl, python3 and the gateway.
$ssh "$cyan" "mkdir -p $remote && cat >$remote/client.sh" <<'CLIENT'
#!/usr/bin/env bash
# The drill's client on cyan: CALLS calls' turns every GAP seconds, AGENTS agents' requests back to back, /nodes every
# second, until the file `stop` appears. Each call is placed with POST /v1/sessions first.
set -uo pipefail
cd "$(dirname "$0")"
sock=/run/user/$(id -u)/flatshare/gateway.sock
# The agents tenant's key, as e2 run reads it.
set -a
. "$HOME/.config/flatshare/gateway-keys.env"
set +a
# The model the engine serves: its list's first entry (each entry's permissions carry ids of their own).
model=$(curl -fsS -m 5 --unix-socket "$sock" http://localhost/v1/models |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
node_of() { # node_of SESSION: the node POST /v1/sessions answers, or the status it got.
  local reply
  reply=$(curl -sS -m 5 --unix-socket "$sock" -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"$1\",\"class\":\"realtime\"}" -w '\n%{http_code}' http://localhost/v1/sessions)
  case ${reply##*$'\n'} in
    200) sed -n 's/.*"node": *"\([^"]*\)".*/\1/p' <<<"$reply" | head -n 1 ;;
    *) echo "refused-${reply##*$'\n'}" ;;
  esac
}
ended() { # ended FILE: how a stream ended.
  if grep -q 'data: \[DONE\]' "$1"; then echo done
  elif grep -q '"error"' "$1"; then echo error
  else echo cut
  fi
}
call() { # call N: a call's turns.
  local session=drill-call-$1 n=0 whole=0 before after line began
  before=$(node_of "$session")
  while [[ ! -e stop ]]; do
    began=$(date +%s.%N)
    line=$(curl -sS -N -m 60 --unix-socket "$sock" -H "X-Session-ID: $session" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"stream\":true,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Turn $n: say one short sentence.\"}]}" \
      -o "turn-$1.body" -w '%{http_code}\t%{time_starttransfer}\t%{time_total}' http://localhost/v1/chat/completions) ||
      line="000\t0\t0"
    after=$(node_of "$session")
    printf '%s\t%s\t%s\t%s\t%b\t%s\n' "$began" "$session" "$before" "$after" "$line" "$(ended "turn-$1.body")" >>turns.tsv
    before=$after
    n=$((n + 1))
    # The event waits for WARM whole turns of each call: a drill whose calls do not work measures nothing.
    [[ ${line%%$'\t'*} == 200 && $(ended "turn-$1.body") == done ]] && whole=$((whole + 1))
    [[ $whole == "$WARM" ]] && touch "warm-$1"
    sleep "$GAP"
  done
}
agent() { # agent N: an agent's requests back to back.
  local session=drill-agent-$1 line began
  while [[ ! -e stop ]]; do
    began=$(date +%s.%N)
    line=$(curl -sS -N -m 300 http://127.0.0.1:18080/v1/chat/completions \
      -H "Authorization: Bearer $FLATSHARE_GATEWAY_AGENTS_KEY" -H "X-Session-ID: $session" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"stream\":true,\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about tunnels.\"}]}" \
      -o "agent-$1.body" -w '%{http_code}\t%{time_starttransfer}\t%{time_total}') || line="000\t0\t0"
    printf '%s\t%s\t%b\t%s\n' "$began" "$session" "$line" "$(ended "agent-$1.body")" >>agents.tsv
    [[ ${line%%$'\t'*} == 200 ]] || sleep 1
  done
}
poll() {
  while [[ ! -e stop ]]; do
    printf '{"t":%s,"view":%s}\n' "$(date +%s.%N)" \
      "$(curl -fsS -m 2 --unix-socket "$sock" http://localhost/nodes || echo null)" >>nodes.jsonl
    sleep 1
  done
}
poll &
for i in $(seq "$CALLS"); do call "$i" & done
for i in $(seq "$AGENTS"); do agent "$i" & done
wait
CLIENT

log "$kind: starting the client on cyan ($calls calls, $agents agents)"
$ssh "$cyan" "cd $remote && rm -f stop warm-* && chmod +x client.sh && \
  (CALLS=$calls AGENTS=$agents GAP=$gap WARM=$warm nohup ./client.sh >client.log 2>&1 </dev/null &)"
for _ in $(seq 300); do
  [[ $($ssh "$cyan" "ls $remote/warm-* 2>/dev/null | wc -l") -ge $calls ]] && break
  sleep 1
done
[[ $($ssh "$cyan" "ls $remote/warm-* 2>/dev/null | wc -l") -ge $calls ]] ||
  { $ssh "$cyan" "touch $remote/stop"; die "the calls did not reach $warm whole turns in 300 s: see $remote/turns.tsv and the turns' bodies on cyan"; }

fail=${FLATSHARE_FAIL:-$($ssh "$cyan" "awk -F'\t' '\$2 == \"drill-call-1\" { node = \$4 } END { print node }' $remote/turns.tsv")}
host=${FLATSHARE_FAIL_HOST:-$fail}
[[ -n $fail ]] || die "no node to fail: the first call has no node"
$ssh "$host" 'systemctl --user reset-failed flatshare-engine 2>/dev/null || true'
log "$kind: the engine of $fail ($host)"
if [[ $kind == kill ]]; then
  event=$($ssh "$host" 'date +%s.%N; systemctl --user kill --signal=SIGKILL flatshare-engine' | sed -n 1p)
else
  event=$($ssh "$host" 'date +%s.%N; systemctl --user stop flatshare-engine' | sed -n 1p)
fi
echo "$event" >"$out/event"
echo "$fail" >"$out/failed-node"
log "$kind of $fail at $event; the calls go on for $after s"
sleep "$after"
$ssh "$cyan" "touch $remote/stop"
sleep 5
if [[ $kind == stop ]]; then
  log "starting $fail's engine again"
  $ssh "$host" 'systemctl --user start flatshare-engine' || log "the engine of $fail did not start: see its journal"
fi
rsync -a -e "$ssh" "$cyan:$remote/" "$out/"
$ssh "$cyan" "journalctl --user -u flatshare-gateway -o short-unix --no-pager --since @${event%.*}" >"$out/gateway-journal.txt"
$ssh "$host" "journalctl --user -u flatshare-engine -o short-unix --no-pager --since @${event%.*}" >"$out/engine-journal.txt"

python3 - "$out" <<'EOF' | tee "$out/report.txt"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
event = float((out / "event").read_text())
failed = (out / "failed-node").read_text().strip()
def rows(name):
    path = out / name
    return [line.split("\t") for line in path.read_text().splitlines()] if path.exists() else []
turns = rows("turns.tsv")
print(f"the {failed} engine at {event:.3f}")
for label, picked in (("before", [t for t in turns if float(t[0]) < event]), ("after", [t for t in turns if float(t[0]) >= event])):
    ok = [t for t in picked if t[4] == "200" and t[7] == "done"]
    print(f"turns {label} the event: {len(picked)}, {len(ok)} whole")
for t in turns:
    if float(t[0]) >= event - 10 and (t[4] != "200" or t[7] != "done"):
        print(f"  {float(t[0]) - event:+7.1f} s {t[1]} on {t[2]} -> {t[3]}: status {t[4]}, ended {t[7]}")
moves = [t for t in turns if t[2] != t[3]]
for t in moves:
    print(f"  {float(t[0]) - event:+7.1f} s {t[1]} moved {t[2]} -> {t[3]}")
first_moved = min((float(t[0]) for t in moves if float(t[0]) >= event), default=None)
late = [t for t in turns if first_moved is not None and float(t[0]) > first_moved and (t[4] != "200" or t[7] != "done")]
print(f"turns that failed after the first move: {len(late)}")
views = [json.loads(line) for line in (out / "nodes.jsonl").read_text().splitlines()] if (out / "nodes.jsonl").exists() else []
def first(predicate):
    for view in views:
        if view["t"] >= event and view["view"] and predicate(view["view"]):
            return view["t"] - event
    return None
node = lambda view: next((n for n in view["nodes"] if n["name"] == failed), None)
passed = first(lambda v: node(v) and not node(v)["available"])
unhealthy = first(lambda v: node(v) and node(v)["verdict"] == "unhealthy")
back = None
if passed is not None:
    for view in views:
        current = node(view["view"]) if view["view"] else None
        if view["t"] >= event + passed and current and current["available"]:
            back = view["t"] - event
            break
covering = [v["t"] for v in views if v["view"] and any(n["covering"] for n in v["view"]["nodes"])]
print(f"{failed} passed over after {passed} s, unhealthy after {unhealthy} s, available again after {back} s")
print(f"a survivor covered for {failed} for {len(covering)} s of polls")
agents = rows("agents.tsv")
after = [a for a in agents if float(a[0]) >= event]
print(f"agents' requests after the event: {len(after)}, {sum(a[2] == '200' and a[5] == 'done' for a in after)} whole")
EOF
log "$kind: done, $out"
