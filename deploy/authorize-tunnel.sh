#!/usr/bin/env bash
# authorize-tunnel.sh CLIENT GPU_NODE: let CLIENT's tunnel keys reach GPU_NODE's servers and nothing else. Each of the
# keys deploy/cyan/bootstrap.sh makes on CLIENT, ~/.ssh/flatshare_tunnel_engine_ed25519, _stt_ and _tts_, gets one
# authorized_keys entry on GPU_NODE: `restrict`, which allows no forwarding of any kind, pty, agent, X11 or ~/.ssh/rc,
# and a forced command that every session of the key runs whatever its client asks for, `exec nc -q0 -U SOCKET`,
# joining the session to GPU_NODE's socket of the key's name in its $XDG_RUNTIME_DIR/flatshare.
# deploy/bin/flatshare-tunnel opens them. Each entry replaces any earlier one for its key, and the entry of CLIENT's
# former tunnel key (~/.ssh/flatshare_tunnel_ed25519, which forwarded the node's ports) goes. GPU_NODE's nc must be
# OpenBSD's, which takes -U and -q. Uses FLATSHARE_SSH like deploy.sh.
set -euo pipefail
client=${1:?usage: authorize-tunnel.sh CLIENT GPU_NODE}
node=${2:?usage: authorize-tunnel.sh CLIENT GPU_NODE}
ssh=${FLATSHARE_SSH:-ssh}
# One line per key, its socket's name first: "engine ssh-ed25519 AAAA... comment". A key's type and blob identify its
# entry; the comment after them is optional.
keys=""
for name in engine stt tts; do
  key=$($ssh "$client" cat ".ssh/flatshare_tunnel_${name}_ed25519.pub" 2>/dev/null || true)
  read -r _ blob _ <<<"$key"
  [[ -n ${blob:-} ]] || {
    echo "authorize-tunnel: no public key in $client's .ssh/flatshare_tunnel_${name}_ed25519.pub (deploy/cyan/bootstrap.sh)" >&2
    exit 2
  }
  keys+="$name $key"$'\n'
done
former=$($ssh "$client" cat .ssh/flatshare_tunnel_ed25519.pub 2>/dev/null || true)
if [[ -n $former ]]; then keys+="former $former"$'\n'; fi
printf '%s' "$keys" | $ssh "$node" 'nc -h 2>&1 | grep -q "^OpenBSD netcat" || {
    echo "authorize-tunnel: $(hostname)'"'"'s nc is not OpenBSD netcat, which takes -U and -q" >&2; exit 2; }
  umask 077; mkdir -p .ssh; touch .ssh/authorized_keys
  socket_dir=/run/user/$(id -u)/flatshare
  cp .ssh/authorized_keys .ssh/authorized_keys.new
  : >.ssh/authorized_keys.add
  while read -r name type blob comment; do
    { grep -vF "$type $blob" .ssh/authorized_keys.new || true; } >.ssh/authorized_keys.kept
    mv .ssh/authorized_keys.kept .ssh/authorized_keys.new
    [ "$name" = former ] && continue
    printf "restrict,command=\"exec nc -q0 -U %s/%s.sock\" %s %s%s\n" "$socket_dir" "$name" "$type" "$blob" \
      "${comment:+ $comment}" >>.ssh/authorized_keys.add
  done
  cat .ssh/authorized_keys.add >>.ssh/authorized_keys.new
  rm .ssh/authorized_keys.add
  mv .ssh/authorized_keys.new .ssh/authorized_keys
  echo "authorized: engine, stt and tts on $socket_dir ($(grep -c "^restrict,command=\"exec nc -q0 -U $socket_dir/" .ssh/authorized_keys) entries)"'
