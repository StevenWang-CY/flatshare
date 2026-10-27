#!/usr/bin/env bash
# campaign.sh [--dry-run] [--pause] [--local] [--until HH:MM] CAMPAIGN: measure the E2 points a campaign file lists,
# from a workstation with SSH to the hosts, like deploy/deploy.sh. CAMPAIGN names arm files, one per line, relative to
# itself; a line `agents-seed N` sets AIPerf's seed for all of its points (flatshare-bench e2 run --agents-seed), in
# place of each arm file's [agents] seed, a line `kv-sample-s SECONDS` the KV sampler's interval (flatshare-bench e2 run
# --kv-sample-s; 0 samples nothing, as W5's points ran), and a line `save-audio` has the callers save each call's audio
# (flatshare-bench e2 run --save-audio), for benchmarks/e2/demo.py's video. For each point whose run directory on cyan
# is not complete, in order, it
#   1. switches the GPU node to the arm's engine (deploy/phastform/variant.sh use, with what flatshare-bench e2 engine
#      prints), with the measurement drop-in on (variant.sh measure on: vLLM's development routes, which flatshare-bench
#      e2 run reads),
#   2. measures the point on cyan (flatshare-bench e2 run) into ~/flatshare/runs/e2/CAMPAIGN/POINT, with the GPU node's
#      deployed revision (its ~/flatshare/src/DEPLOYED as the point starts) for run.json, sampling its GPU's
#      clocks and temperature once a second meanwhile into that directory's gpu.csv,
#   3. fetches that run directory into $FLATSHARE_E2_RESULTS/CAMPAIGN/POINT (default ~/flatshare/runs/e2).
# A complete point is skipped, so running a stopped campaign again resumes it; the first failure stops it. At the end
# it writes the campaign's frontier (flatshare-bench e2 report, fetched into CAMPAIGN/report) and puts cyan's units back
# (flatshare-bench e2 reset). --pause then stops the GPU node's units. A campaign that fails once it has switched the
# node puts cyan's units back and pauses too, unless a point may still be running on cyan (the lock is held, or the
# connection to cyan was lost): then it leaves both as they are, and says so. --dry-run copies and checks every arm file
# and lists the points with their engines, without switching or measuring anything. FLATSHARE_SSH is your ssh command
# for the hosts, as for deploy.sh. FLATSHARE_CYAN names cyan (default cyan), and FLATSHARE_GPU the GPU node every point
# of the campaign runs on (default phastform), one host, which run.json names after its first label (phastform for
# phastform.example.org); each is an SSH host name or alias.
# FLATSHARE_PROFILE names the GPU node's hardware profile (deploy/profiles) for flatshare-bench e2 run's engine check,
# when it is not the one cyan's host settings name, and FLATSHARE_NODE_DIR the GPU node's directory under its home,
# whose variant.sh switches the engine (default flatshare/src/deploy/phastform; a node rendered from another profile has
# its own, flatshare/node). --local runs the campaign on a single host that is cyan and the GPU node at once
# (docs/HARDWARE.md): every command runs here (deploy/bin/flatshare-local in place of ssh), the node is localhost, and
# the results go to ~/flatshare/results/e2 unless FLATSHARE_E2_RESULTS says otherwise. --until HH:MM (UTC; tomorrow's
# when today's has passed) starts no point that cannot end before then, by W5's durations: 13 minutes for a point with
# agents, 7 without, and 2 more for an engine switch, until this host has measured its own: each point and engine switch
# campaign.sh runs goes to durations.tsv beside the campaigns' results (KIND SECONDS a line: switch, voice, or
# agents:SESSIONS), and a kind measured here is budgeted by its longest. --until-epoch SECONDS does the same at an
# absolute time, which has no tomorrow: benchmarks/a100/run.sh hands its one deadline to every campaign of a run so.
# --points prints the points CAMPAIGN lists, one a line, as this script reads the file, and reaches no host:
# benchmarks/a100/run.sh and cprime.sh take a campaign's points from it.
set -euo pipefail
usage() {
  echo "usage: campaign.sh [--dry-run] [--pause] [--local] [--until HH:MM | --until-epoch SECONDS] CAMPAIGN" >&2
  echo "       campaign.sh --points CAMPAIGN" >&2
  exit 2
}
dry_run=false pause=false local=false until="" until_epoch="" list=false
while [[ ${1:-} == --* ]]; do
  case $1 in
    --dry-run) dry_run=true ;;
    --points) list=true ;;
    --pause) pause=true ;;
    --local) local=true ;;
    --until)
      [[ ${2:-} =~ ^([01][0-9]|2[0-3]):([0-5][0-9])$ ]] || usage
      until=$2
      shift
      ;;
    --until-epoch)
      [[ ${2:-} =~ ^[0-9]+$ ]] || usage
      until_epoch=$2
      shift
      ;;
    *) usage ;;
  esac
  shift
done
(($# == 1)) || usage
[[ -f $1 ]] || { echo "campaign: no such file: $1" >&2; exit 2; }
if $local; then
  ssh=$(cd "$(dirname "$0")/../../deploy/bin" && pwd)/flatshare-local
  cyan=localhost
  gpu=localhost
else
  ssh=${FLATSHARE_SSH:-ssh}
  cyan=${FLATSHARE_CYAN:-cyan}
  gpu=${FLATSHARE_GPU:-phastform}
fi
node_dir=${FLATSHARE_NODE_DIR:-flatshare/src/deploy/phastform}
[[ $node_dir =~ ^[A-Za-z0-9._/-]+$ && $node_dir != /* ]] ||
  { echo "campaign: FLATSHARE_NODE_DIR is a directory under the GPU node's home: '$node_dir'" >&2; exit 2; }
[[ -z $until || -z $until_epoch ]] || { echo "campaign: --until or --until-epoch, not both" >&2; exit 2; }
deadline=""
if [[ -n $until ]]; then
  now=$(date -u +%s)
  deadline=$((now - now % 86400 + 10#${until%:*} * 3600 + 10#${until#*:} * 60))
  ((deadline > now)) || deadline=$((deadline + 86400))
elif [[ -n $until_epoch ]]; then
  deadline=$until_epoch
  until=$(date -u -d "@$deadline" +%H:%M 2>/dev/null || date -u -r "$deadline" +%H:%M) # GNU date, else BSD's
fi
[[ -n $gpu && $gpu != *[,[:space:]]* ]] || { echo "campaign: FLATSHARE_GPU names the GPU node's host, one: '${FLATSHARE_GPU-}'" >&2; exit 2; }
node=${gpu%%.*}
profile=""
if [[ -n ${FLATSHARE_PROFILE:-} ]]; then
  [[ $FLATSHARE_PROFILE =~ ^[a-z0-9][a-z0-9.-]*$ ]] || { echo "campaign: FLATSHARE_PROFILE is a profile's name: '$FLATSHARE_PROFILE'" >&2; exit 2; }
  profile=" --profile $FLATSHARE_PROFILE"
fi
name=$(basename "$1" .campaign)
here=$(cd "$(dirname "$1")" && pwd)
if $local; then
  results=${FLATSHARE_E2_RESULTS:-$HOME/flatshare/results/e2}/$name # beside the runs, not over them
else
  results=${FLATSHARE_E2_RESULTS:-$HOME/flatshare/runs/e2}/$name
fi
runs=flatshare/runs/e2/$name # on cyan, under the home directory, as every path given to ssh here
bench=flatshare/venvs/bench/bin/flatshare-bench
complete() { echo "grep -qs '\"status\": \"complete\"' $runs/$1/run.json"; } # run.json as e2 run writes it
if $list; then
  log() { echo "$*" >&2; } # --points writes no log
else
  mkdir -p "$results"
  log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$results/campaign.log"; }
fi
die() { log "campaign: $*"; exit 1; }

# The points in order, each named after its arm file, and the agents' seed and the KV sampler's interval the
# campaign sets, if it sets them.
arms=() points=() seed="" sampling="" audio=""
while IFS= read -r line || [[ -n $line ]]; do
  line=${line%%#*}
  line=${line#"${line%%[![:space:]]*}"}
  line=${line%"${line##*[![:space:]]}"}
  [[ -n $line ]] || continue
  if [[ $line == agents-seed* ]]; then
    [[ -z $seed && $line =~ ^agents-seed[[:space:]]+([0-9]+)$ ]] ||
      die "$1: '$line': a campaign sets the agents' seed once, as agents-seed N"
    seed=${BASH_REMATCH[1]}
    continue
  fi
  if [[ $line == kv-sample-s* ]]; then
    [[ -z $sampling && $line =~ ^kv-sample-s[[:space:]]+([0-9]+(\.[0-9]+)?)$ ]] ||
      die "$1: '$line': a campaign sets the KV sampler's interval once, as kv-sample-s SECONDS"
    sampling=${BASH_REMATCH[1]}
    continue
  fi
  if [[ $line == save-audio* ]]; then
    [[ -z $audio && $line == save-audio ]] || die "$1: '$line': a campaign saves its calls' audio once, as save-audio"
    audio=" --save-audio"
    continue
  fi
  [[ -f $here/$line ]] || die "$1 names $line, which is not a file"
  point=$(basename "$line" .toml)
  for seen in ${points[@]+"${points[@]}"}; do
    [[ $seen != "$point" ]] || die "$1 names two points $point"
  done
  arms+=("$here/$line") points+=("$point")
done <"$1"
((${#points[@]})) || die "$1 names no point"
if $list; then
  printf '%s\n' "${points[@]}"
  exit 0
fi
seeded=""
if [[ -n $seed ]]; then seeded=", the agents' seed $seed"; fi
if [[ -n $sampling ]]; then
  if [[ $sampling =~ ^0+(\.0+)?$ ]]; then seeded+=", no KV sampler"; else seeded+=", the KV sampler every $sampling s"; fi
fi
if [[ -n $audio ]]; then seeded+=", the calls' audio saved"; fi
count="${#points[@]} point"; ((${#points[@]} == 1)) || count+=s
# The arm files' commit: their checkout's, or on a tree that deploy.sh or push.sh copied without .git, the one they
# wrote into its DEPLOYED.
at=$(git -C "$here" describe --always --dirty 2>/dev/null || head -n 1 "$(dirname "$0")/../../DEPLOYED" 2>/dev/null || true)
log "campaign $name: $count, arm files at ${at:-?}$seeded${deadline:+, until $until UTC (epoch $deadline)}"

# Copy the arm files to cyan, where flatshare-bench reads them, and check each one there.
$ssh "$cyan" mkdir -p "$runs/arms"
rsync -a -e "$ssh" "${arms[@]}" "$cyan:$runs/arms/"
engines=()
for point in "${points[@]}"; do
  state="{ $(complete "$point") && echo complete || echo pending; }"
  status=0
  answer=$($ssh "$cyan" "$bench e2 engine $runs/arms/$point.toml && $state") || status=$?
  ((status != 255)) || die "$point: lost the connection to cyan (ssh exit status 255)"
  ((status == 0)) || die "$point: its arm file does not load (above)"
  engines+=("${answer%%$'\n'*}")
  log "$point: ${answer#*$'\n'}, engine ${answer%%$'\n'*}"
done
if $dry_run; then exit 0; fi

fetch() { # fetch POINT: mirror its run directory, without AIPerf's copy of the dataset
  rsync -a --delete -e "$ssh" --exclude aiperf/inputs.json "$cyan:$runs/$1/" "$results/$1/"
}
# A consumer card's boost clock follows its temperature, so each point keeps the clocks it ran at: nvidia-smi on the
# GPU node, once a second while the point runs (line-buffered, timestamps in UTC); it stops when its ssh does. The
# GPU's name and UUID come last, so that a record says which card it ran on. A host that lends one of several GPUs
# names it in its environment (FLATSHARE_GPU_ID, from its env.sh): only that GPU is sampled, and the file's first line,
# a comment, says which (flatshare-bench's record reads that GPU's rows alone).
gpu_query=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active,name,uuid
sampler=""
restore() { # put cyan's units back and, with --pause, stop the GPU node's units
  $ssh "$cyan" "$bench e2 reset" 2>&1 | tee -a "$results/campaign.log" ||
    { log "campaign: flatshare-bench e2 reset failed"; return 1; }
  if $pause; then
    # Naming the units makes systemctl wait for their stops too, not only the target's, which returns at once: a
    # campaign started right after would otherwise meet an engine still shutting down.
    $ssh "$gpu" systemctl --user stop flatshare.target flatshare-engine flatshare-stt flatshare-tts ||
      { log "campaign: the units of $node did not stop"; return 1; }
    log "$node is paused; start it again with systemctl --user start flatshare.target"
  fi
}
# changed: the GPU node was switched; busy: a point may still be running on cyan, which a reset or a pause would spoil;
# ended: the campaign's end ran restore itself.
changed=false busy=false ended=false
finish() {
  local status=$?
  if [[ -n $sampler ]]; then kill "$sampler" 2>/dev/null || true; fi
  if $changed && ! $ended; then
    if $busy; then
      log "campaign: cyan's units and the GPU node stay as they are: a point may still be running on cyan"
    else
      restore || true
    fi
  fi
  exit "$status"
}
trap finish EXIT
sample_gpu() { # sample_gpu POINT
  $ssh "$gpu" "if [ -n \"\${FLATSHARE_GPU_ID:-}\" ]; then echo \"# gpu \$FLATSHARE_GPU_ID\"; fi
    exec env TZ=UTC stdbuf -oL nvidia-smi \${FLATSHARE_GPU_ID:+-i \"\$FLATSHARE_GPU_ID\"} --query-gpu=$gpu_query --format=csv,noheader -l 1" \
    >"$results/$1.gpu.csv" 2>>"$results/campaign.log" &
  sampler=$!
}
stop_gpu() { # stop_gpu POINT: end the sampling and put the samples into the point's run directory on cyan
  if [[ -n $sampler ]]; then
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
  fi
  sampler=""
  if $ssh "$cyan" "cat >$runs/$1/gpu.csv" <"$results/$1.gpu.csv"; then
    rm "$results/$1.gpu.csv"
  else
    log "$1: the samples of $node's GPU stay in $results/$1.gpu.csv"
  fi
}
# A point's budget for --until: the longest of its kind measured on this host (durations.tsv), and of an engine
# switch, in whole minutes rounded up; before one is, W5's on cyan: its longest agent point (11 min 34 s) and
# voice-alone point (5 min 54 s), rounded up, and an engine switch (1 min 41 s at most).
durations=$(dirname "$results")/durations.tsv
kind() { # kind ARM: agents:SESSIONS for a point with agents, else voice
  local sessions
  sessions=$(awk '/^\[/ { table = $0 } table == "[agents]" && /^sessions *= *[0-9]+/ {
    sub(/^sessions *= */, ""); sub(/[^0-9].*/, ""); print; exit }' "$1")
  if [[ -n $sessions && $sessions != 0 ]]; then echo "agents:$sessions"; else echo voice; fi
}
measured() { # measured KIND: the longest duration of KIND measured here, in whole minutes rounded up; none before one
  [[ -f $durations ]] || return 0
  awk -v kind="$1" '$1 == kind && $2 + 0 > longest { longest = $2 + 0 }
    END { if (longest) printf "%d\n", int((longest + 59) / 60) }' "$durations"
}
minutes() { # minutes ARM ENGINE PREVIOUS
  local m switch
  m=$(measured "$(kind "$1")")
  if [[ -z $m ]]; then
    m=7
    [[ $(kind "$1") == voice ]] || m=13
  fi
  if [[ $2 != "$3" ]]; then
    switch=$(measured switch)
    m=$((m + ${switch:-2}))
  fi
  echo "$m"
}
previous=""
for i in "${!points[@]}"; do
  point=${points[$i]}
  status=0
  $ssh "$cyan" "$(complete "$point")" || status=$?
  if ((status == 255)); then
    busy=true
    die "$point: lost the connection to cyan (ssh exit status 255)"
  elif ((status == 0)); then
    fetch "$point"
    previous=${engines[$i]}
    continue
  fi
  if [[ -n $deadline ]]; then
    need=$(minutes "${arms[$i]}" "${engines[$i]}" "$previous")
    if (($(date -u +%s) + need * 60 > deadline)); then
      log "$point: $need minutes would end after $until UTC; stopping before it (--until)"
      break
    fi
  fi
  switching=false
  [[ ${engines[$i]} == "$previous" ]] || switching=true
  previous=${engines[$i]}
  # A point whose ssh session dropped may still be running on cyan; switching the engine would spoil it.
  status=0
  $ssh "$cyan" flock -n flatshare/runs/.flatshare-bench-e2.lock true || status=$?
  if ((status)); then
    busy=true
    ((status != 255)) || die "$point: lost the connection to cyan (ssh exit status 255)"
    die "a point is still running on cyan (its pid is in flatshare/runs/.flatshare-bench-e2.lock); run again once it ends"
  fi
  changed=true
  switched=$(date -u +%s)
  log "$point: $node runs ${engines[$i]}"
  variant="bash $node_dir/variant.sh"
  $ssh "$gpu" "$variant measure on && $variant use ${engines[$i]}" 2>&1 |
    tee -a "$results/campaign.log" || die "$point: $node did not start ${engines[$i]} (journalctl there)"
  # The node's deployed revision as the point starts (deploy/deploy.sh writes it), for run.json beside cyan's; empty
  # when the node has none.
  deployed=$($ssh "$gpu" cat flatshare/src/DEPLOYED 2>/dev/null) || deployed=""
  [[ $deployed =~ ^[A-Za-z0-9._+-]+$ ]] || deployed=""
  log "$point: $node has ${deployed:-no DEPLOYED}"
  if $switching; then echo "switch $(($(date -u +%s) - switched))" >>"$durations"; fi
  log "$point: measuring"
  measuring=$(date -u +%s)
  sample_gpu "$point"
  status=0
  $ssh "$cyan" "$bench e2 run $runs/arms/$point.toml --out $runs/$point${seed:+ --agents-seed $seed}${sampling:+ --kv-sample-s $sampling}$audio --gpu-source $node=$deployed$profile" 2>&1 |
    tee -a "$results/campaign.log" || status=$?
  stop_gpu "$point"
  if ((status == 255)); then
    busy=true
    die "$point: lost the connection to cyan; the point may still be running there"
  elif ((status)); then
    fetch "$point" || true
    die "$point failed (exit status $status); its run directory is $results/$point"
  fi
  fetch "$point"
  echo "$(kind "${arms[$i]}") $(($(date -u +%s) - measuring))" >>"$durations"
done

dirs=""
for point in "${points[@]}"; do dirs+=" $runs/$point"; done
log "the frontier of $name"
$ssh "$cyan" "$bench e2 report$dirs --out $runs/report" 2>&1 | tee -a "$results/campaign.log" ||
  die "flatshare-bench e2 report failed"
rsync -a --delete -e "$ssh" "$cyan:$runs/report/" "$results/report/"
ended=true
restore || exit 1
log "done: $results/report/frontier.md"
