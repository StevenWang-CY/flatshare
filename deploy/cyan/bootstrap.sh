#!/usr/bin/env bash
# bootstrap.sh: prepare cyan, the control host, once a release is deployed there: deploy/deploy.sh's release builds the
# Python environments from the lock files (venvs/voice, flatshare-voice-worker and flatshare-voice, from
# python/flatshare_voice/uv.lock; venvs/bench, flatshare-bench and the AIPerf plugins, from python/flatshare_bench/uv.lock;
# venvs/smg, SMG 1.13.0 for E2's arm 8b, from requirements/smg.txt; deploy/bin/flatshare-release), which this reads
# through ~/flatshare/venvs. Safe to re-run.
#   bin/livekit-server                  LiveKit server 1.13.10
#   ~/.config/flatshare/livekit.env     API key pair for the local LiveKit server and the voice worker
#   ~/.ssh/flatshare_tunnel_{engine,stt,tts}_ed25519 the tunnel keys, one per socket of a GPU node; authorize them
#                                       on each node with deploy/authorize-tunnel.sh
set -euo pipefail
root=$HOME/flatshare
for env in voice bench smg; do
  [[ -x $root/venvs/$env/bin/python ]] || {
    echo "bootstrap: no $root/venvs/$env: deploy a release first (deploy/deploy.sh), which builds the environments" >&2
    exit 2
  }
done
mkdir -p "$root"/{bin,data,runs} "$HOME/.config/flatshare" "$HOME/.ssh"

livekit=1.13.10
if ! "$root/bin/livekit-server" --version 2>/dev/null | grep -q "$livekit"; then
  curl -fsSL "https://github.com/livekit/livekit/releases/download/v$livekit/livekit_${livekit}_linux_amd64.tar.gz" |
    tar -xz -C "$root/bin" livekit-server
fi

# The worker's models ship in its wheels (the v1-mini end-of-turn model in livekit-local-inference, Silero's ONNX
# file in its plugin), so there is nothing to download. Check that they load offline, as flatshare-voice-worker.service
# runs them (HF_HUB_OFFLINE=1, inside a namespace with no route out).
HF_HUB_OFFLINE=1 "$root/venvs/voice/bin/python" -c \
  'from livekit.local_inference import init_eot; init_eot(); from livekit.plugins import silero; silero.VAD.load()'

env_file=$HOME/.config/flatshare/livekit.env
if [[ ! -f $env_file ]]; then
  (
    umask 077
    key=flatshare$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')
    secret=$(head -c 32 /dev/urandom | base64 | tr -d '/+=\n')
    printf 'LIVEKIT_KEYS="%s: %s"\nLIVEKIT_API_KEY=%s\nLIVEKIT_API_SECRET=%s\nLIVEKIT_URL=ws://127.0.0.1:17880\n' \
      "$key" "$secret" "$key" "$secret" >"$env_file"
  )
fi

for name in engine stt tts; do
  [[ -f $HOME/.ssh/flatshare_tunnel_${name}_ed25519 ]] ||
    ssh-keygen -q -t ed25519 -N '' -C "flatshare-tunnel-$name@$(hostname -s)" \
      -f "$HOME/.ssh/flatshare_tunnel_${name}_ed25519"
done
[[ -f $HOME/.config/flatshare/tunnel-phastform.env ]] ||
  echo "bootstrap: write FLATSHARE_TUNNEL_TARGET=<user>@<gpu node> to ~/.config/flatshare/tunnel-phastform.env" >&2
version() { "$root/venvs/$1/bin/python" -c 'import importlib.metadata as m, sys; print(sys.argv[1], m.version(sys.argv[1]))' "$2"; }
echo "bootstrap: $("$root/bin/livekit-server" --version)"
echo "bootstrap: voice $(version voice livekit-agents); its models load offline"
echo "bootstrap: bench $(version bench aiperf)"
echo "bootstrap: smg $(version smg smg)"
