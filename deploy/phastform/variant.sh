#!/usr/bin/env bash
# variant.sh on|off NAME: apply or remove a variant of the frozen box, then restart it.
# variant.sh use NAME|frozen [FLAG...]: make NAME the box's only variant (frozen: none) and FLAG... the engine's extra
# flags, restarting only what that changes; benchmarks/e2/campaign.sh switches the box this way before each point.
# variant.sh measure on|off: link or remove the measurement drop-in (measure/flatshare-engine.conf, vLLM's development
# routes, which E2 and the drills read) as flatshare-engine.service.d/measure.conf, restarting a running engine when
# that changes it. use, on and off keep it as it is.
# A variant is a directory variants/NAME/ of systemd drop-ins, one UNIT.conf per unit it changes. Each is linked into
# ~/.config/systemd/user/UNIT.service.d/ as variant-NAME.conf. Engine flags go into a generated drop-in,
# flatshare-engine.service.d/variant-flags.conf, that sets FLATSHARE_ENGINE_FLAGS: they beat engine.yaml and the
# variant's own flags. A restart takes ~2 min with warm caches (variants/README.md lists the variants).
# The box's variant drop-ins are the GPU units' only (gpu below, every unit a variant here changes): on a single host
# the control plane's units share the directory, and their drop-ins (cyan's variants, linked by deploy/a100/host.py
# units and flatshare-bench e2 run) are not the box's to switch.
set -euo pipefail
action=${1:?on|off|use|measure} name=${2:?variant name}
shift 2
[[ $action == on || $action == off || $action == use || $action == measure ]] ||
  { echo "variant: action must be on, off, use or measure" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
# Drop-ins link through ~/flatshare/src, current's release, never into one release's own directory, which a later
# release prunes (deploy/bin/flatshare-release): whichever path ran this, its links name the one through src.
case $here in "$HOME"/flatshare/releases/*/deploy/*) here=$HOME/flatshare/src/deploy/${here#"$HOME"/flatshare/releases/*/deploy/} ;; esac
units=$HOME/.config/systemd/user
gpu=(flatshare-engine flatshare-stt flatshare-tts)
box=(flatshare.target "${gpu[@]}")

if [[ $action == measure ]]; then
  [[ $name == on || $name == off ]] || { echo "variant: measure on or measure off" >&2; exit 2; }
  drop_in=$units/flatshare-engine.service.d/measure.conf
  if [[ $name == on && $(readlink "$drop_in" || true) != "$here/measure/flatshare-engine.conf" ]]; then
    mkdir -p "$units/flatshare-engine.service.d"
    ln -sfn "$here/measure/flatshare-engine.conf" "$drop_in"
  elif [[ $name == off && ( -e $drop_in || -L $drop_in ) ]]; then
    rm -f "$drop_in"
  else
    echo "variant: the measurement drop-in is already $name"
    exit
  fi
  systemctl --user daemon-reload
  # A stopped engine takes it at its next start; a running one restarts, keeping the audio servers.
  if systemctl --user is-active -q flatshare-engine; then
    systemctl --user reset-failed flatshare-engine 2>/dev/null || true
    systemctl --user restart flatshare-engine
  fi
  echo "variant: the measurement drop-in is $name"
  exit
fi
[[ $action == use && $name == frozen ]] || [[ -d $here/variants/$name ]] ||
  { echo "variant: no such variant: $here/variants/$name" >&2; exit 2; }

link() { # link NAME: link NAME's drop-ins into their units
  local conf unit
  for conf in "$here/variants/$1"/*.conf; do
    unit=$(basename "$conf" .conf)
    mkdir -p "$units/$unit.service.d"
    ln -sf "$conf" "$units/$unit.service.d/variant-$1.conf"
  done
}
# The units' crash-loop guard (StartLimitBurst=3 in 30 min) counts every start; a switch is no crash, so it clears the
# counters before starting (a unit not loaded keeps none).
clear_start_limits() { systemctl --user reset-failed flatshare-engine flatshare-stt flatshare-tts 2>/dev/null || true; }
# Whether every named unit is active: `systemctl is-active` given several units succeeds when any one of them is, so a
# box half stopped by a pause, or with a failed engine beside running audio servers, would pass for running.
all_active() { local u; for u; do systemctl --user is-active -q "$u" || return 1; done; }

if [[ $action != use ]]; then
  systemctl --user stop "${box[@]}"
  if [[ $action == on ]]; then
    link "$name"
  else
    for u in "${gpu[@]}"; do rm -f "$units/$u.service.d/variant-$name.conf"; done
  fi
  systemctl --user daemon-reload
  clear_start_limits
  systemctl --user start flatshare.target
  systemctl --user is-active flatshare-stt flatshare-tts flatshare-engine
  all_active flatshare-stt flatshare-tts flatshare-engine
  exit
fi

for flag in "$@"; do
  [[ $flag =~ ^[A-Za-z0-9._:/=,+-]+$ ]] || { echo "variant: not a plain engine flag: $flag" >&2; exit 2; }
done
flags_line="Environment=\"FLATSHARE_ENGINE_FLAGS=$*\""
# The box's variant drop-ins, one line each (a link with its target, the generated flags with their line), as they
# are and as NAME and FLAG... want them.
current() {
  local u f
  for u in "${gpu[@]}"; do
    for f in "$units/$u.service.d"/variant-*.conf; do
      if [[ -L $f ]]; then
        echo "${f#"$units/"} -> $(readlink "$f")"
      elif [[ -e $f ]]; then
        echo "${f#"$units/"}: $(grep '^Environment=' "$f")"
      fi
    done
  done
}
wanted() {
  local conf
  if [[ $name != frozen ]]; then
    for conf in "$here/variants/$name"/*.conf; do
      echo "$(basename "$conf" .conf).service.d/variant-$name.conf -> $conf"
    done
  fi
  if (($#)); then echo "flatshare-engine.service.d/variant-flags.conf: $flags_line"; fi
}
changed=$(comm -3 <(current | sort) <(wanted "$@" | sort) | sed 's/^[[:space:]]*//; s/\.service\.d\/.*//' | sort -u)
if [[ -z $changed ]] && all_active flatshare-stt flatshare-tts flatshare-engine; then
  echo "variant: the box already runs $name $*"
  exit
fi
# Only the engine restarts when only its drop-ins change and the audio servers run; anything else (re)starts the box.
restart=("${box[@]}")
if [[ $changed == flatshare-engine ]] && all_active flatshare-stt flatshare-tts; then restart=(flatshare-engine); fi
systemctl --user stop "${restart[@]}"
for u in "${gpu[@]}"; do rm -f "$units/$u.service.d"/variant-*.conf; done
[[ $name == frozen ]] || link "$name"
if (($#)); then
  mkdir -p "$units/flatshare-engine.service.d"
  printf '# Generated by variant.sh use: engine flags over engine.yaml and the variant flags.\n[Service]\n%s\n' \
    "$flags_line" >"$units/flatshare-engine.service.d/variant-flags.conf"
fi
systemctl --user daemon-reload
clear_start_limits
systemctl --user start "${restart[0]}" # flatshare.target or flatshare-engine; returns once /health answers
systemctl --user is-active flatshare-stt flatshare-tts flatshare-engine
all_active flatshare-stt flatshare-tts flatshare-engine
