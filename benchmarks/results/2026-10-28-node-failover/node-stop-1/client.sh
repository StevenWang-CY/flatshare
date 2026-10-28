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
