#!/usr/bin/env bash
# install.sh: link cyan's units into the user manager and enable them at boot, and make the gateway's cache-salt
# secret and metrics key once. Lingering must be on (`loginctl enable-linger`), or the user manager stops with the
# last SSH session.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
units=$HOME/.config/systemd/user
mkdir -p "$units"
ln -sf "$here"/systemd/* "$units/"
# private FILE: write 32 random bytes in hex to FILE, mode 600, unless it exists. Never replaced.
private() {
  [[ -f $1 ]] && return
  mkdir -p "$(dirname "$1")"
  (umask 077 && od -An -N32 -tx1 /dev/urandom | tr -d ' \n' >"$1.new" && echo >>"$1.new")
  mv "$1.new" "$1"
}
# The key of every tenant's cache_salt (cache_salt_secret_file in gateway.toml): a new secret would change every
# salt, which orphans every tenant's prefix cache.
private "$HOME/.config/leaseholder/cache-salt-secret"
# The key GET /metrics asks for on the gateway's TCP listener (metrics_key_file), which Prometheus sends.
private "$HOME/.config/leaseholder/metrics-key"
# Links to units the repository no longer has, such as the tunnels' before lh-tunnel@.service, go; a unit that still
# runs from one stops at the next reboot (stop it by hand, ops/RUNBOOK.md).
for link in "$units"/*; do
  [[ -L $link && ! -e $link && $(readlink "$link") == "$here"/systemd/* ]] || continue
  systemctl --user disable -q "$(basename "$link")" 2>/dev/null || true
  rm "$link"
done
systemctl --user daemon-reload
systemctl --user enable -q lh-voicenet.service lh-livekit.service lh-tunnel@phastform.service lh-voice-worker.service
loginctl show-user "$(id -un)" --property=Linger
