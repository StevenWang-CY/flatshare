#!/usr/bin/env bash
# rehearse.sh [--redo STEP]... [--purge] STEP...|all|status|abort: the A100 run's single-host rehearsal on the box
# (Phastform), from this workstation: docs/HARDWARE.md's "Running on another GPU, step by step" on real hardware,
# beside the box's deployment and leaving it as it was found (README.md beside this file: the commands, what each step
# prints and takes, how to abort and restore). The steps, in order:
#   check      read-only: the box's units stopped, the GPU idle, the ports free, the disk, the models, Kokoro, logind
#   init       the rehearsal's home on the host (~/a100-rehearsal): its environment, Kokoro, the box's state recorded
#   data       LH_DATA on this workstation: weka-x32, weka-x32-long (long_traces.py), callers (cyan's)
#   push       deploy/a100/push.sh, into the rehearsal's home
#   stage, gpu, warm     bootstrap.sh's three stages, on the host
#   block0, block1       run.sh 0 (its listener audit with each GPU server in a network namespace of its own) and
#                        run.sh 1
#   block4     run.sh --until <17 minutes on> --estimate 4=15 4: round 1's first point, then the deadline's stop
#   pull       deploy/a100/pull.sh --once, into hosts/HOST
#   reproduce  make reproduce-a100 over the pulled records, on peacock; the report into reproduce/
#   cleanup    the rehearsal's units and processes ended, the box's state compared with init's, the GPU idle
# `all` runs each step not done yet, in order; `status` shows each step's state; `abort` stops a step that runs on
# the host and cleans up. A step done is skipped (state/STEP.done); a step on the host runs detached from the SSH
# session, so that a dropped connection loses nothing, and rehearse.sh run again waits for it.
#   --redo STEP          run STEP again although it is done (repeatable): a step on the host is run again, and a
#                        block's blocks/N.done on the host removed first, so that run.sh runs the block again
#   --purge              cleanup, run again although it is done, also removes the rehearsal's home on the host
# The environment: REHEARSAL_HOST (phastform), REHEARSAL_SSH (the ssh command: LH_SSH, else ssh),
# REHEARSAL_HOME (a100-rehearsal, a directory under the host's login home), REHEARSAL_REPO (the checkout pushed: the one
# this script is in), REHEARSAL_CYAN and REHEARSAL_PEACOCK (cyan, peacock),
# REHEARSAL_CORPUS (this workstation's weka-x32, ~/leaseholder/data/weka-x32; cyan's when it has none),
# REHEARSAL_PEACOCK_CORPUS (peacock's, under its home: leaseholder-data/weka-x32; this workstation's is copied there when
# it has none), REHEARSAL_GATEWAY_BIN (an lh-gateway for push.sh's LH_GATEWAY_BIN; by default bootstrap.sh builds one on
# the host, as the A100's must where a copy does not run), REHEARSAL_MIN_FREE_GIB (40), REHEARSAL_WORK (where state/,
# data/ (LH_DATA), hosts/ (the pulled records) and reproduce/ go: ~/leaseholder/rehearsal).
# Bash 3.2 (macOS's) or newer.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
host=${REHEARSAL_HOST:-phastform}
export REHEARSAL_SSH=${REHEARSAL_SSH:-${LH_SSH:-ssh}}
export REHEARSAL_HOME=${REHEARSAL_HOME:-a100-rehearsal}
rhome=$REHEARSAL_HOME
repo=${REHEARSAL_REPO:-$(cd "$here/../../.." && pwd)}
work=${REHEARSAL_WORK:-$HOME/leaseholder/rehearsal}
data=$work/data
state=$work/state
cyan=${REHEARSAL_CYAN:-cyan}
peacock=${REHEARSAL_PEACOCK:-peacock}
corpus=${REHEARSAL_CORPUS:-$HOME/leaseholder/data/weka-x32}
peacock_corpus=${REHEARSAL_PEACOCK_CORPUS:-leaseholder-data/weka-x32}
gateway=${REHEARSAL_GATEWAY_BIN:-}
min_gib=${REHEARSAL_MIN_FREE_GIB:-40}
read -r -a ssh_cmd <<<"$REHEARSAL_SSH"
STEPS="check init data push stage gpu warm block0 block1 block4 pull reproduce cleanup"
HOST_STEPS="stage gpu warm block0 block1 block4"

usage() {
  echo "usage: rehearse.sh [--redo STEP]... [--purge] STEP...|all|status|abort" >&2
  echo "  steps: $STEPS" >&2
  exit 2
}
purge=false redo=" "
requested=()
while (($#)); do
  case $1 in
    --purge) purge=true redo+="cleanup " ;;
    --redo)
      [[ " $STEPS " == *" ${2:-} "* ]] || usage
      redo+="$2 "
      shift
      ;;
    -*) usage ;;
    *) requested+=("$1") ;;
  esac
  shift
done
((${#requested[@]})) || usage
for word in "${requested[@]}"; do
  [[ " $STEPS all status abort " == *" $word "* ]] || usage
done
[[ $rhome =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
  { echo "rehearse: REHEARSAL_HOME is one directory name under the host's login home: '$rhome'" >&2; exit 2; }

mkdir -p "$state/logs"
log() { echo "[$(date -u +%FT%TZ)] rehearse: $*" | tee -a "$state/rehearse.log"; }
die() {
  log "$*"
  exit 1
}
q() { printf '%q ' "$@"; }
# In the host's login environment, where the box's deployment is (read-only but for init's home, cleanup and purge).
on_host() { "${ssh_cmd[@]}" "$host" "$@"; }
host_sh() { "${ssh_cmd[@]}" "$host" "bash -s -- $(q "$@")" <"$here/host.sh"; }
# In the rehearsal's environment, from its home (bin/rehearsal-ssh).
in_rehearsal() { "$here/bin/rehearsal-ssh" "$host" "$@"; }
rehearsal_sh() { "$here/bin/rehearsal-ssh" "$host" "bash -s -- $(q "$@")" <"$here/host.sh"; }

done_before() { [[ -f $state/$1.done && $redo != *" $1 "* ]]; }
mark() {
  echo "$(date -u +%FT%TZ) $2" >"$state/$1.done"
  log "$1: done: $2"
}
revision() { git -C "$repo" describe --always --dirty; }

# host_step NAME COMMAND: COMMAND on the host, detached from this SSH session, then waited for (again after a dropped
# connection); its exit status. A step that --redo names is run again although it is done.
host_step() {
  local name=$1 status tries=0 again=()
  [[ $redo != *" $name "* ]] || again=(--again)
  rehearsal_sh start ${again[@]+"${again[@]}"} "$name" "$2" || return $?
  while :; do
    status=0
    rehearsal_sh wait "$name" || status=$?
    if ((status == 255)); then # ssh's own: the connection, not the step
      tries=$((tries + 1))
      ((tries < 30)) || return 255
      log "$name: the connection to $host dropped; waiting for the step again in 20 s"
      sleep 20
      continue
    fi
    return "$status"
  done
}

model_of() { awk '$1 == "model:" {m = $2} $1 == "revision:" {r = $2} END {print m "@" r}' "$repo/deploy/phastform/$1"; }

step_check() {
  local file rev status=0
  git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || die "check: REHEARSAL_REPO is no git checkout: $repo"
  for file in deploy/a100/push.sh deploy/a100/pull.sh deploy/a100/bootstrap.sh benchmarks/a100/run.sh \
    deploy/phastform/engine.yaml deploy/phastform/stt.yaml; do
    [[ -f $repo/$file ]] || die "check: $repo has no $file"
  done
  [[ -f $repo/deploy/bin/in-private-net ]] ||
    die "check: $repo has no deploy/bin/in-private-net: block 0's listener audit would refuse the GPU servers' sockets"
  grep -q -- '--estimate' "$repo/benchmarks/a100/run.sh" ||
    die "check: $repo's run.sh takes no --estimate, which block4 runs"
  rev=$(revision)
  [[ $rev != *-dirty ]] || log "check: $repo is dirty ($rev): push.sh records it so"
  log "check: $host, its ~/$rhome, the source at $rev ($repo)"
  host_sh check-box "$rhome" "18080 17880" "17882" "$min_gib" "$(model_of engine.yaml)" "$(model_of stt.yaml)" |
    tee "$state/check.txt" || status=${PIPESTATUS[0]}
  ((status == 0)) || die "check: $host is not ready for the rehearsal (refused above)"
  mark check "$rev on $host"
}

step_init() {
  local hub answer
  # Without check.txt no hub cache is named, as the next line says.
  hub=$(sed -n 's/^check: hub cache: ok: //p' "$state/check.txt" 2>/dev/null || true)
  [[ -n $hub ]] || die "init: state/check.txt names no hub cache (rehearse.sh --redo check check)"
  host_sh init "$rhome" "$hub" || die "init: the rehearsal's home was not made (above)"
  in_rehearsal 'echo "init: the rehearsal environment: HOME $HOME, XDG_RUNTIME_DIR $XDG_RUNTIME_DIR, LH_SUPERVISOR $LH_SUPERVISOR, HF_HUB_CACHE $HF_HUB_CACHE (offline $HF_HUB_OFFLINE), session bus ${DBUS_SESSION_BUS_ADDRESS:-none}"'
  # The box's user manager must be out of the rehearsal's reach: bootstrap.sh, under LH_SUPERVISOR=lh-svc where a user
  # manager runs, would stop the units of that name it runs (the box's). Only systemctl's answer that it reaches no
  # manager passes: "Failed to connect to bus" ("to user scope bus" from systemd 257), or offline where systemd did
  # not boot the host. Anything else fails: a manager's state, and no answer, as from a dropped connection or
  # rehearsal-ssh's refusal.
  answer=$(in_rehearsal 'systemctl --user is-system-running 2>&1 || true') ||
    die "init: the rehearsal's environment did not answer whether it reaches the box's user manager (above)"
  answer=${answer%%$'\n'*}
  [[ $answer == "Failed to connect to "* || $answer == offline* ]] ||
    die "init: the rehearsal's environment may reach the box's user manager (systemctl --user there: ${answer:-nothing})"
  log "init: the box's user manager is out of the rehearsal's reach (systemctl --user there: $answer)"
  mark init "~/$rhome on $host"
}

step_data() {
  local traces long clips
  mkdir -p "$data"
  if [[ ! -e $data/weka-x32 ]]; then
    if [[ -d $corpus ]]; then
      ln -s "$corpus" "$data/weka-x32"
    else
      log "data: weka-x32 from $cyan"
      rsync -a -e "$REHEARSAL_SSH" "$cyan:leaseholder/data/weka-x32/" "$data/weka-x32/"
    fi
  fi
  if [[ ! -e $data/weka-x32-long ]]; then
    python3 "$repo/benchmarks/e2/long_traces.py" "$data/weka-x32" "$data/weka-x32-long" 48
  fi
  if [[ ! -e $data/callers ]]; then
    log "data: callers from $cyan"
    rsync -a -e "$REHEARSAL_SSH" "$cyan:leaseholder/data/callers/" "$data/callers/"
  fi
  traces=$(find -L "$data/weka-x32/" -maxdepth 1 -name '*.json' -type f | wc -l | tr -d ' ')
  long=$(find -L "$data/weka-x32-long/" -maxdepth 1 -name '*.json' -type f | wc -l | tr -d ' ')
  clips=$(find "$data/callers/" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')
  [[ -f $data/callers/manifest.json ]] || die "data: $data/callers has no manifest.json"
  ((traces > 0 && long == 48 && clips > 0)) ||
    die "data: weka-x32 $traces traces, weka-x32-long $long (48 wanted), callers $clips clips"
  mark data "LH_DATA $data: weka-x32 $traces traces, weka-x32-long $long, callers $clips clips"
}

step_push() {
  local rev deployed counts want="" d
  rev=$(revision)
  env -u LH_GATEWAY_BIN LH_SSH="$here/bin/rehearsal-ssh" LH_DATA="$data" ${gateway:+"LH_GATEWAY_BIN=$gateway"} \
    bash "$repo/deploy/a100/push.sh" "$host" || die "push: push.sh failed (above)"
  deployed=$(in_rehearsal cat leaseholder/src/DEPLOYED)
  [[ $deployed == "$rev" ]] || die "push: the rehearsal's DEPLOYED on $host is '$deployed', not $rev"
  # The host's data as push.sh copies LH_DATA's (-L: the links as files): as many files in each directory, no link.
  for d in weka-x32 weka-x32-long callers; do
    want+="$d $(find -L "$data/$d" -type f | wc -l | tr -d ' ') files 0 links; "
  done
  counts=$(in_rehearsal 'for d in weka-x32 weka-x32-long callers; do
    printf "%s %s files %s links; " $d $(find leaseholder/data/$d -type f | wc -l) $(find leaseholder/data/$d -type l | wc -l)
    done') || die "push: the rehearsal's data on $host was not counted (above)"
  log "push: ~/$rhome/leaseholder on $host: the source at $deployed; $counts"
  [[ $counts == "$want" ]] || die "push: the host's data is not LH_DATA's, copied as files: $want wanted"
  mark push "$deployed; $counts"
}

step_stage() {
  host_step stage "bash leaseholder/src/deploy/a100/bootstrap.sh stage" ||
    die "stage: bootstrap.sh stage failed (above; ~/$rhome/leaseholder/node/bootstrap.log on $host)"
  rehearsal_sh verify stage || die "stage: its checks failed (above)"
  mark stage "bootstrap.sh stage, the control plane up under lh-svc"
}

step_gpu() {
  host_step gpu "bash leaseholder/src/deploy/a100/bootstrap.sh gpu" ||
    die "gpu: bootstrap.sh gpu failed (above; ~/$rhome/leaseholder/node/bootstrap.log on $host)"
  rehearsal_sh verify gpu || die "gpu: its checks failed (above)"
  mark gpu "bootstrap.sh gpu"
}

step_warm() {
  host_step warm "bash leaseholder/src/deploy/a100/bootstrap.sh warm" ||
    die "warm: bootstrap.sh warm failed (above; ~/$rhome/leaseholder/cache/warm.json on $host)"
  rehearsal_sh verify warm || die "warm: its checks failed (above)"
  mark warm "bootstrap.sh warm"
}

# run.sh as docs/HARDWARE.md starts it, its output appended to runs/a100/run.log too.
run_sh() { echo "mkdir -p leaseholder/runs/a100 && set -o pipefail && bash leaseholder/src/benchmarks/a100/run.sh $* 2>&1 | tee -a leaseholder/runs/a100/run.log"; }

# again BLOCK: a block that --redo names starts afresh on the host, so that run.sh, which skips a block done, runs it.
again() {
  [[ $redo != *" block$1 "* ]] || in_rehearsal "rm -f leaseholder/runs/a100/blocks/$1.done"
}

step_block0() {
  again 0
  host_step block0 "$(run_sh 0)" ||
    die "block0: run.sh 0 failed (above, and ~/$rhome/leaseholder/runs/a100/run.log on $host)"
  rehearsal_sh verify block0 || die "block0: its checks failed (above)"
  mark block0 "run.sh 0, the listener audit ok"
}

step_block1() {
  again 1
  host_step block1 "$(run_sh 1)" || die "block1: run.sh 1 failed (above)"
  rehearsal_sh verify block1 || die "block1: its checks failed (above)"
  mark block1 "run.sh 1"
}

step_block4() {
  # The deadline 17 minutes on (in whole minutes, 16 to 17 away): block 4 at an estimate of 15 starts, its first
  # point (W5's 13 minutes and an engine switch's 2) starts, and the second (voice alone, 7 and a switch) cannot.
  local until='$(python3 -c "import time; print(time.strftime(\"%H:%M\", time.gmtime(time.time() + 1020)))")'
  again 4
  host_step block4 "until=$until && echo \"rehearsal: block 4 until \$until UTC\" && $(run_sh '--until "$until" --estimate 4=15 4')" ||
    die "block4: run.sh --until --estimate 4=15 4 failed (above)"
  rehearsal_sh verify block4 || die "block4: its checks failed (above)"
  mark block4 "run.sh --until --estimate 4=15 4: the first point, then the deadline's stop"
}

step_pull() {
  local dir=$work/hosts/${host%%.*} what missing=""
  LH_SSH="$here/bin/rehearsal-ssh" bash "$repo/deploy/a100/pull.sh" --once "$host" "$dir" || die "pull: pull.sh failed (above)"
  for what in runs/a100/blocks.log runs/a100/block1/frozen-e0/summary.json node/preflight.json cache/warm.json \
    results/e2/a100-r1/w5-disc-pr-ab48-s24/run.json runs/journal/units.log; do
    [[ -s $dir/$what ]] || missing+=" $what"
  done
  [[ -z $missing ]] || die "pull: $dir lacks$missing"
  log "pull: $dir: $(du -sh "$dir" | cut -f1), the journal's $(wc -l <"$dir/runs/journal/units.log" | tr -d ' ') lines"
  mark pull "$dir"
}

step_reproduce() {
  local dir=$work/hosts/${host%%.*} out=$work/reproduce there=leaseholder-rehearsal-reproduce corpus_dir section
  "${ssh_cmd[@]}" "$peacock" "mkdir -p $there/src $there/records && cd $there &&
    find src -mindepth 1 -maxdepth 1 ! -name target -exec rm -rf {} + && rm -rf records && mkdir records" ||
    die "reproduce: ~/$there on $peacock was not emptied for the checkout and the records (above)"
  (cd "$repo" && git ls-files -z | COPYFILE_DISABLE=1 tar --no-xattrs --null -T - -cf -) |
    "${ssh_cmd[@]}" "$peacock" "tar -xf - -C $there/src" ||
    die "reproduce: the checkout was not copied to $peacock (above)"
  rsync -a -e "$REHEARSAL_SSH" "$dir/" "$peacock:$there/records/" ||
    die "reproduce: the pulled records were not copied to $peacock (above)"
  corpus_dir=$peacock_corpus
  if ! "${ssh_cmd[@]}" "$peacock" "test -d $peacock_corpus"; then
    log "reproduce: no $peacock_corpus on $peacock: copying this workstation's weka-x32 there"
    corpus_dir=$there/corpus/weka-x32
    # Its parent first: GNU rsync makes a destination's last directory alone (macOS's openrsync makes them all).
    "${ssh_cmd[@]}" "$peacock" "mkdir -p $there/corpus" &&
      rsync -aL -e "$REHEARSAL_SSH" "$data/weka-x32/" "$peacock:$corpus_dir/" ||
      die "reproduce: this workstation's weka-x32 was not copied to $peacock (above)"
  fi
  "${ssh_cmd[@]}" "$peacock" "cd $there/src && export PATH=\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH &&
    make reproduce-a100 RECORDS=../records CORPUS=\$HOME/$corpus_dir" || die "reproduce: make reproduce-a100 failed (above)"
  mkdir -p "$out"
  rsync -a -e "$REHEARSAL_SSH" "$peacock:$there/src/build/reproduce-a100/" "$out/"
  [[ -s $out/a100.md ]] || die "reproduce: no $out/a100.md"
  for section in "## 3. The probes" "## 4. E2 within the A100"; do
    awk -v s="$section" '$0 == s {on = 1; next} on && /^## / {exit} on && NF {print; exit}' "$out/a100.md" |
      grep -qv '^Not measured' || die "reproduce: a100.md's '$section' is not measured"
  done
  log "reproduce: $out/a100.md, its sections: $(grep '^## ' "$out/a100.md" | tr '\n' ' ')"
  mark reproduce "$out/a100.md"
}

step_cleanup() {
  local status=0 name home=0
  # Whether the host has the rehearsal's home: test's 1 is no. Any other failure, as ssh's 255 for a dropped
  # connection, is no answer, on which cleanup fails rather than passes.
  on_host "test -f $rhome/rehearsal.env" || home=$?
  if ((home == 1)); then
    log "cleanup: $host has no ~/$rhome: nothing of the rehearsal to clean"
    mark cleanup "nothing to clean"
    return 0
  fi
  ((home == 0)) || die "cleanup: $host did not answer whether it has ~/$rhome (exit status $home)"
  # Each stop's line, but those of a step that was not running (none, or done): grep selecting no line is not a
  # failure, and a stop that fails is.
  for name in $HOST_STEPS; do
    rehearsal_sh stop "$name" | { grep -Ev ': (none|done [0-9]+)$' || true; } || status=1
  done
  rehearsal_sh kill-tools || status=1
  rehearsal_sh stop-units || status=1
  host_sh leftovers "$rhome" --kill || status=1
  host_sh verify-box "$rhome" || status=1
  # The box's units still as the main session left them, the GPU idle, the ports free again.
  host_sh check-box "$rhome" "18080 17880" "17882" 0 "$(model_of engine.yaml)" "$(model_of stt.yaml)" || status=1
  ((status == 0)) || die "cleanup: something of the rehearsal remains, or the box is not as it was found (above)"
  if $purge; then
    host_sh purge "$rhome" || die "cleanup: ~/$rhome on $host was not removed (above)"
    mark cleanup "the box's deployment as found; ~/$rhome removed"
  else
    mark cleanup "the box's deployment as found"
  fi
}

abort() {
  local name
  log "abort: stopping what runs of the rehearsal on $host"
  if on_host "test -f $rhome/rehearsal.env"; then
    for name in $HOST_STEPS; do
      rehearsal_sh stop "$name" | grep -Ev ': (none|done [0-9]+)$' || true
    done
  fi
  redo+="cleanup "
  run_step cleanup
}

status() {
  local step
  for step in $STEPS; do
    if [[ -f $state/$step.done ]]; then
      printf '%-10s done %s\n' "$step" "$(cat "$state/$step.done")"
    else
      printf '%-10s not done\n' "$step"
    fi
  done
  if [[ -f $state/init.done ]] && on_host "test -f $rhome/rehearsal.env"; then
    for step in $HOST_STEPS; do
      printf '%-10s on %s: %s\n' "$step" "$host" "$(rehearsal_sh state "$step")"
    done
  fi
}

run_step() {
  local step=$1 previous="" word status
  if done_before "$step"; then
    log "$step: done before ($(cat "$state/$step.done"))"
    return 0
  fi
  for word in $STEPS; do
    [[ $word != "$step" ]] || break
    previous=$word
  done
  if [[ $step != cleanup && -n $previous && ! -f $state/$previous.done ]]; then
    die "$step: $previous is not done (rehearse.sh status)"
  fi
  log "$step: start"
  # The step in a subshell with set -e, so that a command that fails ends it with its status. Not as `(...) | tee ...
  # || status=...`: bash ignores set -e in everything on the left of || or &&, a subshell that sets it again included.
  # So set -e is off only around the pipeline here, whose first status is the step's.
  set +e
  (
    set -e
    "step_$step"
  ) 2>&1 | tee -a "$state/logs/$step.log"
  status=${PIPESTATUS[0]}
  set -e
  ((status == 0)) || { log "$step: failed, exit status $status (state/logs/$step.log)"; exit "$status"; }
}

# One rehearse.sh at a time, but for status.
if [[ " ${requested[*]} " != " status " ]]; then
  if ! mkdir "$state/lock" 2>/dev/null; then
    owner=$(cat "$state/lock/pid" 2>/dev/null || true)
    if [[ -n $owner ]] && kill -0 "$owner" 2>/dev/null; then
      echo "rehearse: another rehearse.sh runs (pid $owner)" >&2
      exit 2
    fi
  fi
  echo $$ >"$state/lock/pid"
  trap 'rm -rf "$state/lock"' EXIT
fi

for word in "${requested[@]}"; do
  case $word in
    all) for step in $STEPS; do run_step "$step"; done ;;
    status) status ;;
    abort) abort ;;
    *) run_step "$word" ;;
  esac
done
