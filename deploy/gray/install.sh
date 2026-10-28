#!/usr/bin/env bash
# install.sh: link the frozen-box units (deploy/phastform/systemd) into Gray's user manager, with Gray's drop-ins. Gray is
# a second GPU node in booked windows, shared with other work, so flatshare.target is not enabled at boot: start it
# for a window and stop it after. Lingering must be on (`loginctl enable-linger`).
#
# Gray's kernel refuses unprivileged user namespaces (Ubuntu 24.04's kernel.apparmor_restrict_unprivileged_userns=1, and
# the deploying user has no root to lift it), so deploy/bin/in-private-net cannot give the GPU servers network
# namespaces of their own there, and fails a unit that asks for one. install.sh finds out with deploy/bin/private-net-check
# at every install: where the kernel refuses them, it links deploy/gray/private-net-host.conf into each GPU server's
# drop-ins (the engine's, the STT's and the TTS's), whose FLATSHARE_PRIVATE_NET=host has in-private-net run the server in
# the host's network namespace, where it serves only its Unix socket and deploy/bin/audit-listeners accepts nothing else
# of its but its engine core's loopback listeners and CUDA's abstract sockets; where they are allowed, it removes that
# drop-in, and each server gets a namespace of its own. A check that cannot run (no unshare or no ip) changes nothing,
# and says so. A server takes the change at its next start.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# The units' links go through ~/flatshare/src, current's release, never into one release's own directory, which a later
# release prunes (deploy/bin/flatshare-release): whichever path ran this, its links name the one through src.
case $here in "$HOME"/flatshare/releases/*/deploy/*) here=$HOME/flatshare/src/deploy/${here#"$HOME"/flatshare/releases/*/deploy/} ;; esac
units=$HOME/.config/systemd/user
mkdir -p "$units"
ln -sf "$here"/../phastform/systemd/* "$units/"
for conf in "$here"/systemd/*.conf; do
  unit=$(basename "$conf" .conf)
  mkdir -p "$units/$unit.service.d"
  ln -sf "$conf" "$units/$unit.service.d/gray.conf"
done
# The GPU servers' network namespaces: their own where the kernel allows unprivileged user namespaces, else the host's.
servers=(flatshare-engine flatshare-stt flatshare-tts)
check=$here/../bin/private-net-check
status=2
if [[ -f $check ]]; then
  said=$(bash "$check") && status=0 || status=$?
else
  said="unknown: no $check"
fi
case $status in
  0)
    for unit in "${servers[@]}"; do rm -f "$units/$unit.service.d/private-net-host.conf"; done
    echo "install: user namespaces are allowed: each GPU server runs in a network namespace of its own"
    ;;
  1)
    for unit in "${servers[@]}"; do
      mkdir -p "$units/$unit.service.d"
      ln -sfn "$here/private-net-host.conf" "$units/$unit.service.d/private-net-host.conf"
    done
    echo "install: user namespaces are ${said}: the GPU servers run in the host's network namespace" \
      "(FLATSHARE_PRIVATE_NET=host, deploy/gray/private-net-host.conf), serving only their Unix sockets"
    ;;
  *)
    echo "install: the GPU servers' network namespaces are left as they were: private-net-check said '$said'" >&2
    ;;
esac
systemctl --user daemon-reload
loginctl show-user "$(id -un)" --property=Linger
