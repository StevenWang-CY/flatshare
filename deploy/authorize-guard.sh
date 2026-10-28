#!/usr/bin/env bash
# authorize-guard.sh CLIENT GPU_NODE: let CLIENT read what a soak reads of GPU_NODE at its checks, and nothing else
# (flatshare-bench soak's guard, which stops the soak's load while another's work holds a node's GPU and keeps the
# node's units' restarts and failures). CLIENT's guard key, ~/.ssh/flatshare_guard_ed25519, made there if it has none,
# gets one authorized_keys entry on GPU_NODE: `restrict`, which allows no forwarding of any kind, pty, agent, X11 or
# ~/.ssh/rc, and a forced command that every session of the key runs whatever its client asks for,
# `exec flatshare/src/deploy/bin/node-report`: the GPU's compute processes, each the box's own or another's, and the
# user manager's journal lines about the node's flatshare units since a time the session names (deploy/bin/node-report).
# The entry replaces any earlier one for the key, and no other entry changes. The node must have the deployment's
# node-report. The soak reaches the node at the target its tunnel's environment names
# (~/.config/flatshare/tunnel-GPU_NODE.env on CLIENT). Uses FLATSHARE_SSH like deploy.sh.
set -euo pipefail
client=${1:?usage: authorize-guard.sh CLIENT GPU_NODE}
node=${2:?usage: authorize-guard.sh CLIENT GPU_NODE}
ssh=${FLATSHARE_SSH:-ssh}
key=$($ssh "$client" 'key=.ssh/flatshare_guard_ed25519
  if [ ! -f "$key" ]; then
    mkdir -p .ssh && chmod 700 .ssh
    ssh-keygen -q -t ed25519 -N "" -C "flatshare-guard@$(hostname -s)" -f "$key" </dev/null >&2
  fi
  cat "$key.pub"') || true
read -r type blob comment <<<"$key"
[[ -n ${blob:-} ]] || {
  echo "authorize-guard: no public key in $client's .ssh/flatshare_guard_ed25519.pub" >&2
  exit 2
}
printf '%s %s %s\n' "$type" "$blob" "${comment:-}" | $ssh "$node" 'read -r type blob comment
  report=flatshare/src/deploy/bin/node-report
  [ -x "$report" ] || {
    echo "authorize-guard: $(hostname -s) has no ~/$report: deploy the node first" >&2; exit 2; }
  umask 077; mkdir -p .ssh; touch .ssh/authorized_keys
  { grep -vF "$type $blob" .ssh/authorized_keys || true; } >.ssh/authorized_keys.new
  printf "restrict,command=\"exec %s\" %s %s%s\n" "$report" "$type" "$blob" "${comment:+ $comment}" \
    >>.ssh/authorized_keys.new
  mv .ssh/authorized_keys.new .ssh/authorized_keys
  echo "authorized: the guard key reads what a soak reads of $(hostname -s) and nothing else"'
