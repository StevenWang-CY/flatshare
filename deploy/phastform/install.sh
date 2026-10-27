#!/usr/bin/env bash
# install.sh: link the frozen-box units into the user manager and enable flatshare.target at boot.
# Lingering must be on (`loginctl enable-linger`), or the user manager stops with the last SSH session.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# The units' links go through ~/flatshare/src, current's release, never into one release's own directory, which a later
# release prunes (deploy/bin/flatshare-release): whichever path ran this, its links name the one through src.
case $here in "$HOME"/flatshare/releases/*/deploy/*) here=$HOME/flatshare/src/deploy/${here#"$HOME"/flatshare/releases/*/deploy/} ;; esac
units=$HOME/.config/systemd/user
mkdir -p "$units"
ln -sf "$here"/systemd/* "$units/"
systemctl --user daemon-reload
systemctl --user enable -q flatshare.target
loginctl show-user "$(id -un)" --property=Linger
