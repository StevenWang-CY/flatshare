#!/usr/bin/env bash
# install.sh: link cyan's units into the user manager and enable them at boot, and make the gateway's cache-salt
# secret and metrics key once. Lingering must be on (`loginctl enable-linger`), or the user manager stops with the
# last SSH session.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# The units' links go through ~/flatshare/src, current's release, never into one release's own directory, which a later
# release prunes (deploy/bin/flatshare-release): whichever path ran this, its links name the one through src.
case $here in "$HOME"/flatshare/releases/*/deploy/*) here=$HOME/flatshare/src/deploy/${here#"$HOME"/flatshare/releases/*/deploy/} ;; esac
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
private "$HOME/.config/flatshare/cache-salt-secret"
# The key GET /metrics asks for on the gateway's TCP listener (metrics_key_file), which Prometheus sends.
private "$HOME/.config/flatshare/metrics-key"
# Links to units the repository no longer has, such as the tunnels' before flatshare-tunnel@.service, go; a unit that
# still runs from one stops at the next reboot (stop it by hand, ops/RUNBOOK.md). Disabling a linked unit removes its
# link already, so the link may be gone when rm comes to it.
for link in "$units"/*; do
  [[ -L $link && ! -e $link && $(readlink "$link") == "$here"/systemd/* ]] || continue
  systemctl --user disable -q "$(basename "$link")" 2>/dev/null || true
  rm -f "$link"
done
systemctl --user daemon-reload
systemctl --user enable -q flatshare-voicenet.service flatshare-livekit.service flatshare-tunnel@phastform.service \
  flatshare-voice-worker.service
# The gateway's binary is the release's (bin/flatshare-gateway, a link into current, which deploy.sh's release checked
# against the source's version before it switched). Installed outside a release, from a tree deployed otherwise, it
# can lag the source: a version that is not the source's is said, not refused.
gateway=$HOME/flatshare/bin/flatshare-gateway
wanted=$(sed -n '/^\[workspace\.package\]/,/^\[/ s/^version *= *"\([^"]*\)".*/\1/p' "$here/../../Cargo.toml")
built=missing
if [[ -e $gateway ]]; then
  version=$("$gateway" --version 2>/dev/null | awk '{print $2}') || true
  built=${version:+version $version}
  built=${built:-not runnable}
fi
if [[ $built != "version $wanted" ]]; then
  echo "install: ~/flatshare/bin/flatshare-gateway is $built, the source version $wanted: release this commit with" \
    "its gateway as ops/RUNBOOK.md's Deploy says" >&2
fi
loginctl show-user "$(id -un)" --property=Linger
