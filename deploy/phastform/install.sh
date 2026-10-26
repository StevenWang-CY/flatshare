#!/usr/bin/env bash
# install.sh: link the frozen-box units into the user manager and enable flatshare.target at boot.
# Lingering must be on (`loginctl enable-linger`), or the user manager stops with the last SSH session.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
units=$HOME/.config/systemd/user
mkdir -p "$units"
ln -sf "$here"/systemd/* "$units/"
systemctl --user daemon-reload
systemctl --user enable -q flatshare.target
loginctl show-user "$(id -un)" --property=Linger
