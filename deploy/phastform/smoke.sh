#!/usr/bin/env bash
# smoke.sh [ENGINE] [STT] [TTS]: end-to-end check of the frozen box, on the GPU node or through its tunnel. Each server
# is an address, http://host:port or unix: and its socket's path; the defaults are the GPU node's sockets,
# unix:$XDG_RUNTIME_DIR/lh/engine.sock, stt.sock and tts.sock (through the tunnel on cyan, phastform-engine.sock and so
# on). Speaks a question with the TTS, transcribes it with the STT, asks the LLM, and speaks the reply; prints each
# stage's latency. On the GPU node it also prints GPU memory and audits the listening sockets (deploy/bin/
# audit-listeners): the servers on the sockets given, and no TCP listener of the box's units.
set -euo pipefail
lh=unix:${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/lh
engine=${1:-$lh/engine.sock}
stt=${2:-$lh/stt.sock}
tts=${3:-$lh/tts.sock}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

field() { python3 -c "import json, sys; print(json.load(sys.stdin)$1)"; }
stage() { printf '%-26s %s\n' "$1" "$2"; }
# at ADDRESS PATH CURL_ARG...: curl PATH of the server at ADDRESS.
at() {
  local address=$1 path=$2
  shift 2
  if [[ $address == unix:* ]]; then
    curl --unix-socket "${address#unix:}" "$@" "http://localhost$path"
  else
    curl "$@" "$address$path"
  fi
}

for server in "$engine" "$stt" "$tts"; do at "$server" /health -fsS -m 5 -o /dev/null; done
stage health ok
model=$(at "$engine" /v1/models -fsS -m 5 | field '["data"][0]["id"]') # the engine's own model

t=$(at "$tts" /v1/audio/speech -fsS -o "$work/question.wav" -w '%{time_total}' -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"af_heart","response_format":"wav","input":"What is the capital of France?"}')
stage "tts question (wav)" "${t}s, $(wc -c <"$work/question.wav") bytes"

at "$stt" /v1/audio/transcriptions -fsS -o "$work/stt.json" -w '%{time_total}\n' -F file=@"$work/question.wav" \
  -F model=openai/whisper-large-v3-turbo >"$work/stt.time"
text=$(field '["text"]' <"$work/stt.json")
stage "stt" "$(cat "$work/stt.time")s: ${text}"
grep -qi france <<<"$text" || { echo "smoke: transcript does not mention France" >&2; exit 1; }

python3 -c 'import json, sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 32, "temperature": 0,
  "messages": [{"role": "user", "content": sys.argv[2] + " Answer in one short sentence."}]}))' \
  "$model" "$text" >"$work/chat.json"
t=$(at "$engine" /v1/chat/completions -fsS -o "$work/reply.json" -w '%{time_total}' \
  -H 'Content-Type: application/json' -d @"$work/chat.json")
reply=$(field '["choices"][0]["message"]["content"]' <"$work/reply.json")
stage "llm" "${t}s: ${reply}"
grep -qi paris <<<"$reply" || { echo "smoke: reply does not mention Paris" >&2; exit 1; }

python3 -c 'import json, sys; print(json.dumps({"model": "kokoro", "voice": "af_heart",
  "response_format": "pcm", "input": sys.argv[1]}))' "$reply" >"$work/speak.json"
t=$(at "$tts" /v1/audio/speech -fsS -o "$work/reply.pcm" -w '%{time_starttransfer}/%{time_total}' \
  -H 'Content-Type: application/json' -d @"$work/speak.json")
stage "tts reply (first/total)" "${t}s, $(wc -c <"$work/reply.pcm") bytes"

# Only on the GPU node: the audit reads the host it runs on, and through a tunnel the servers are another host's.
if command -v nvidia-smi >/dev/null; then
  stage "gpu memory" "$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
  expect=()
  for server in "$engine" "$stt" "$tts"; do
    if [[ $server == unix:* ]]; then expect+=(--expect "${server#unix:}"); fi
  done
  audit=$("$(dirname "$0")/../bin/audit-listeners" ${expect[@]+"${expect[@]}"}) ||
    { echo "smoke: the listener audit failed (above)" >&2; exit 1; }
  stage "listener audit" "${audit#audit-listeners: }"
fi
echo "smoke: ok"
