#!/usr/bin/env bash
# host.sh SUBCOMMAND...: the single-host rehearsal's commands on the host. rehearse.sh, beside it on the workstation,
# pipes this file to bash over ssh (bash -s), so nothing of it is installed on the host. In the login environment,
# where the box's deployment is, these read it and write only the rehearsal's own home, HOME (a directory under the
# login home):
#   check-box HOME TCP_PORTS UDP_PORTS MIN_FREE_GIB MODEL@REVISION...   read-only: the box's units stopped, the GPU
#             idle, the ports free, the disk, the box's models in its hub cache, Kokoro, the tools, logind
#   init HOME HUB_CACHE   the rehearsal's home: rehearsal.env, its runtime and temporary directories, the sudo shim,
#             Kokoro-FastAPI from the box's, and the box's state as found (.rehearsal/box-before.state)
#   verify-box HOME       the box's state against the one init recorded, the GPU idle, the ports free again
#   leftovers HOME [--kill]   the processes whose environment names HOME (the rehearsal's), and those in their
#             process groups and sessions whose environment a title has blanked; with --kill their end
#   purge HOME            the rehearsal's home removed
# In the rehearsal's environment (bin/rehearsal-ssh sources rehearsal.env first), from its home:
#   start [--again] NAME COMMAND   COMMAND detached from the SSH session (its own session, stdin /dev/null), its
#             output in .rehearsal/steps/NAME.log, its exit status in NAME.status, once its pid file is written
#             (it fails, COMMAND not run, if the step's bash cannot write it); nothing if it runs already, or if
#             it is done and --again is not given
#   wait NAME             follow the step's log until it ends; exit with its status (254 for a status of 255)
#   state NAME            done STATUS, running PID, died PID (gone without a status) or none
#   stop NAME             end a running step: TERM to its process group, then KILL to what is left of it after
#             180 s (flatshare-bench's own stop of its tools); status 143
#   verify STEP           the checks of rehearse.sh's STEP: env, stage, gpu, warm, block0, block1, block4
#   kill-tools            the process groups of flatshare-bench's tools that a killed step left, ended by
#             flatshare-bench's kill_left (the bench venv)
#   stop-units            the rehearsal's units stopped, under flatshare-svc (its node's env.sh): fails on one not
#             inactive or failed after its stop
# Bash 3.2 (the CPU test's macOS) or newer, and python3 for what the shell has no portable way to do. REHEARSAL_POLL_S
# is wait's interval (5 s; the test's shorter).
set -euo pipefail

say() { echo "check: $1: $2"; }

# The box's units, as the user manager runs them: every flatshare* unit, its state, and its enablement.
box_units() {
  systemctl --user list-units --all --no-legend --plain 'flatshare*' | LC_ALL=C sort
  systemctl --user list-unit-files --no-legend 'flatshare*' | LC_ALL=C sort
}

# The box's deployment as files: each entry's kind, size, mtime, mode and link target under the trees the box runs
# from or reads (its source, venvs, records and caches; its units; its configuration; the hub cache; vLLM's caches).
box_files() {
  python3 - "$@" <<'EOF'
import os
import stat
import sys

for top in sys.argv[1:]:
    if not os.path.lexists(top):
        print(f"absent {top}")
        continue
    stack = [top]
    while stack:
        path = stack.pop()
        st = os.lstat(path)
        mode = stat.S_IMODE(st.st_mode)
        if stat.S_ISLNK(st.st_mode):
            print(f"l {path} {oct(mode)} -> {os.readlink(path)}")
        elif stat.S_ISDIR(st.st_mode):
            print(f"d {path} {oct(mode)} {st.st_mtime_ns}")
            stack.extend(sorted((os.path.join(path, n) for n in os.listdir(path)), reverse=True))
        else:
            print(f"f {path} {oct(mode)} {st.st_size} {st.st_mtime_ns}")
EOF
}

BOX_TREES=(flatshare .config/systemd/user .config/flatshare .cache/vllm .cache/flashinfer)

box_state() { # box_state HUB_CACHE: what cleanup compares, from the login home (the hub cache wherever it is)
  echo "# units"
  box_units
  echo "# files"
  box_files "${BOX_TREES[@]}" "$1"
}

# Where huggingface_hub would put the hub cache in this (the login) environment: the box's.
hub_cache() {
  local cache=${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-}}
  if [[ -z $cache ]]; then cache=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub; fi
  (cd "$cache" 2>/dev/null && pwd) || echo "$cache"
}

check_box() {
  local home=$1 tcp=$2 udp=$3 min_gib=$4
  shift 4
  local refused=0 line active apps listening kib hub model revision dir t version
  no() {
    say "$1" "refused: $2"
    refused=1
  }
  if [[ -e $home && ! -f $home/rehearsal.env ]]; then
    no "rehearsal home" "~/$home exists and is no rehearsal's (it has no rehearsal.env)"
  elif [[ -e $home ]]; then
    say "rehearsal home" "ok: ~/$home, a rehearsal's (resumed)"
  else
    say "rehearsal home" "ok: ~/$home, new"
  fi
  # The box's units, stopped: under its user manager, none of them active, activating or deactivating.
  if ! active=$(systemctl --user list-units --all --no-legend --plain 'flatshare*' 2>&1); then
    no "the box's units" "the user manager does not answer: ${active%%$'\n'*}"
  else
    active=$(awk '$3 ~ /^(active|activating|deactivating|reloading)$/ {printf "%s%s", sep, $1; sep = ", "}' <<<"$active")
    if [[ -n $active ]]; then
      no "the box's units" "running: $active (stop them first: README.md, before)"
    else
      say "the box's units" "ok: every flatshare* unit is inactive"
    fi
  fi
  # The GPU, idle: no compute process (the box's engine, FB's, anyone's).
  if ! apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1); then
    no "GPU" "nvidia-smi failed: ${apps%%$'\n'*}"
  elif [[ -n $apps ]]; then
    no "GPU" "in use: $(tr '\n' ';' <<<"$apps")"
  else
    say "GPU" "ok: idle, $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -1)"
  fi
  # The rehearsal's listeners in the host's network namespace: nobody else's on their ports, on any address.
  listening=$( (ss -Hltn | awk '{print "tcp " $4}'; ss -Hlun | awk '{print "udp " $4}') 2>&1) ||
    { no "ports" "ss failed: ${listening%%$'\n'*}"; listening=""; }
  local port taken=""
  for port in $tcp; do
    taken+=$(awk -v p=":$port" '$1 == "tcp" && substr($2, length($2) - length(p) + 1) == p {printf " %s", $0}' <<<"$listening")
  done
  for port in $udp; do
    taken+=$(awk -v p=":$port" '$1 == "udp" && substr($2, length($2) - length(p) + 1) == p {printf " %s", $0}' <<<"$listening")
  done
  if [[ -n $taken ]]; then
    no "ports" "held:$taken (TCP 18080 is the gateway's; TCP 17880 and UDP 17882 LiveKit's, where the voice plane runs in the host's namespace)"
  else
    say "ports" "ok: TCP ${tcp// /, } and UDP ${udp// /, } free"
  fi
  # The disk the rehearsal's home is on: its venvs, caches and records.
  kib=$(df -Pk . | awk 'NR == 2 {print $4}')
  kib=${kib:-0}
  if ((kib < min_gib * 1048576)); then
    no "disk" "$((kib / 1048576)) GiB free under the login home, below the $min_gib GiB the rehearsal takes [estimate]"
  else
    say "disk" "ok: $((kib / 1048576)) GiB free"
  fi
  # The box's models, at the profile's revisions, in the box's hub cache, which the rehearsal reads offline.
  hub=$(hub_cache)
  say "hub cache" "ok: $hub"
  for model in "$@"; do
    revision=${model#*@}
    model=${model%@*}
    dir=$hub/models--${model//\//--}/snapshots/$revision
    if [[ -d $dir ]]; then
      say "model $model" "ok: $revision"
    else
      no "model $model" "no snapshot $revision in $hub (the rehearsal downloads nothing)"
    fi
  done
  # Kokoro-FastAPI and its model, which init copies from the box's (bootstrap.sh then neither clones nor downloads).
  if [[ -d flatshare/vendor/Kokoro-FastAPI/.git && -f flatshare/vendor/Kokoro-FastAPI/api/src/models/v1_0/kokoro-v1_0.pth ]]; then
    say "Kokoro" "ok: the box's ~/flatshare/vendor/Kokoro-FastAPI, with its model"
  else
    no "Kokoro" "no ~/flatshare/vendor/Kokoro-FastAPI with api/src/models/v1_0/kokoro-v1_0.pth"
  fi
  # What bootstrap.sh would install with sudo, which the rehearsal refuses it: present already.
  local missing=""
  for t in cc c++ ss git curl rsync python3 tar; do
    command -v "$t" >/dev/null || missing+=" $t"
  done
  if [[ -n $missing ]]; then
    no "tools" "missing:$missing"
  else
    version=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    if python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
      say "tools" "ok: cc, c++, ss, git, curl, rsync, tar; python3 $version"
    else
      no "tools" "python3 $version, below the 3.10 deploy/bin/flatshare-svc runs on"
    fi
  fi
  if command -v unshare >/dev/null && unshare --user --map-root-user --net true 2>/dev/null; then
    say "user namespaces" "ok: the voice plane and each GPU server in a network namespace of its own"
  else
    say "user namespaces" "note: refused, so the voice plane runs in the host's network namespace (LiveKit's TCP 17880 and UDP 17882 there), and so do the GPU servers (FLATSHARE_PRIVATE_NET=host)"
  fi
  # A logind that kills a session's processes at logout would end every detached step with its SSH session.
  line=$(grep -hE '^[[:space:]]*KillUserProcesses[[:space:]]*=' /etc/systemd/logind.conf /etc/systemd/logind.conf.d/*.conf 2>/dev/null | tail -1 || true)
  if [[ ${line##*=} =~ ^[[:space:]]*(yes|true|1)[[:space:]]*$ ]]; then
    no "logind" "KillUserProcesses=yes: the steps would end with their SSH session"
  else
    say "logind" "ok: KillUserProcesses=${line:+$(tr -d ' ' <<<"${line##*=}")}${line:-no (the default)}"
  fi
  say "CPUs" "note: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?') logical, $(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l | tr -d ' ') NUMA nodes (below 17, R7 halves the plan's wants and warns)"
  say "/dev/shm" "note: $(df -Pk /dev/shm 2>/dev/null | awk 'NR == 2 {printf "%d GiB free", $4 / 1048576}') (the 20 GiB CPU tier needs it)"
  return "$refused"
}

init() {
  local home=$1 hub=$2 root qroot qhub
  [[ ! -e $home || -f $home/rehearsal.env ]] || { echo "init: ~/$home exists and is no rehearsal's" >&2; exit 2; }
  mkdir -p "$home"
  chmod 700 "$home"
  root=$(cd "$home" && pwd)
  mkdir -p "$root/run" "$root/tmp" "$root/.rehearsal/steps" "$root/.rehearsal/bin"
  chmod 700 "$root/run"
  # The paths quoted for the shell that sources the file, so that a space in the login home or the hub cache's
  # path leaves them whole.
  qroot=$(printf %q "$root") qhub=$(printf %q "$hub")
  cat >"$root/rehearsal.env" <<EOF
# The single-host rehearsal's environment on this host (rehearse.sh init wrote it; bin/rehearsal-ssh sources it before
# each of the rehearsal's commands): its own home, runtime directory, caches and supervisor; the box's hub cache, read
# offline; no way to the box's user manager (no session bus, and a runtime directory that is not /run/user/UID), so
# that nothing of the rehearsal stops or starts a unit of the box's; and sudo refused.
export FLATSHARE_REHEARSAL=1
export HOME=$qroot
export XDG_RUNTIME_DIR=$qroot/run XDG_CONFIG_HOME=$qroot/.config XDG_CACHE_HOME=$qroot/.cache
export XDG_DATA_HOME=$qroot/.local/share XDG_STATE_HOME=$qroot/.local/state TMPDIR=$qroot/tmp
export UV_CACHE_DIR=$qroot/.cache/uv UV_PYTHON_INSTALL_DIR=$qroot/.local/share/uv/python
export CARGO_HOME=$qroot/.cargo RUSTUP_HOME=$qroot/.rustup
unset DBUS_SESSION_BUS_ADDRESS UV VIRTUAL_ENV PYTHONPATH HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
unset VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCH_HOME
unset FLATSHARE_GPU_ID FLATSHARE_PROFILE FLATSHARE_NODE_DIR FLATSHARE_E2_RESULTS
export FLATSHARE_SUPERVISOR=flatshare-svc
export HF_HUB_CACHE=$qhub HF_HUB_OFFLINE=1
export PATH=$qroot/.rehearsal/bin:$qroot/.local/bin:\$PATH
cd "\$HOME"
EOF
  cat >"$root/.rehearsal/bin/sudo" <<'EOF'
#!/bin/sh
# sudo in the rehearsal: refused, so that bootstrap.sh changes nothing of the host's system; rehearse.sh check found
# the tools it would install with it present already.
echo "rehearsal: sudo refused: the rehearsal changes nothing of the host's system" >&2
exit 1
EOF
  chmod 755 "$root/.rehearsal/bin/sudo"
  if [[ ! -d $root/flatshare/vendor/Kokoro-FastAPI ]]; then
    mkdir -p "$root/flatshare/vendor"
    cp -a flatshare/vendor/Kokoro-FastAPI "$root/flatshare/vendor/"
    echo "init: Kokoro-FastAPI copied from the box's ~/flatshare/vendor ($(du -sh "$root/flatshare/vendor/Kokoro-FastAPI" | cut -f1))"
  fi
  if [[ ! -f $root/.rehearsal/box-before.state ]]; then
    box_state "$hub" >"$root/.rehearsal/box-before.state.new"
    mv "$root/.rehearsal/box-before.state.new" "$root/.rehearsal/box-before.state"
    echo "init: the box's state recorded: $(grep -c . "$root/.rehearsal/box-before.state") lines in ~/$home/.rehearsal/box-before.state"
  fi
  echo "init: ~/$home ready (HOME $root, XDG_RUNTIME_DIR $root/run, the box's hub cache $hub)"
}

verify_box() {
  local home=$1 root before after hub apps refused=0
  root=$(cd "$home" && pwd)
  before=$root/.rehearsal/box-before.state
  after=$root/.rehearsal/box-after.state
  [[ -f $before ]] || { echo "verify: no $before: init recorded no state" >&2; return 2; }
  # The hub cache that init recorded, as rehearsal.env sets it (in a subshell: it sets HOME and the directory).
  hub=$(. "$root/rehearsal.env" && printf '%s\n' "$HF_HUB_CACHE")
  box_state "$hub" >"$after"
  if cmp -s "$before" "$after"; then
    echo "verify: the box's deployment as the rehearsal found it: $(grep -c . "$before") lines (units, files) alike"
  else
    echo "verify: the box's deployment changed (< as found, > now):"
    diff "$before" "$after" | head -60 || true
    refused=1
  fi
  if apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) && [[ -z $apps ]]; then
    echo "verify: the GPU is idle"
  else
    echo "verify: the GPU is not idle: $(tr '\n' ';' <<<"$apps")"
    refused=1
  fi
  return "$refused"
}

leftovers() {
  local home=$1 root
  root=$(cd "$home" && pwd)
  python3 - "$root" "${2:-}" <<'EOF'
import os
import signal
import sys
import time

root, kill = sys.argv[1], sys.argv[2] == "--kill"
mark = f"HOME={root}".encode()
ancestors = set()
pid = os.getpid()
while pid > 1:
    ancestors.add(pid)
    try:
        pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        break
# The process groups and sessions of the processes found, kept from one look to the next. A process that set its
# title (setproctitle, as vLLM's engine core and AIPerf's processes do) has blanked its environment and the mark in
# it, and is the rehearsal's by its group or session: one whose environment is blank there is ours. One whose
# environment can be read and lacks the mark is never ours, whatever its group.
groups, sessions = set(), set()


def ours():
    found, blank = {}, {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in ancestors:
            continue
        try:
            environ = open(f"/proc/{entry}/environ", "rb").read()
            state, _, group, session = open(f"/proc/{entry}/stat").read().rsplit(")", 1)[1].split()[:4]
            cmdline = open(f"/proc/{entry}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError, IndexError):
            continue
        if state == "Z":  # exited already
            continue
        process = (int(group), int(session), cmdline.strip()[:160])
        if mark in environ.split(b"\0"):
            found[int(entry)] = process
        elif b"=" not in environ:  # no variable left: blank, or a title over it
            blank[int(entry)] = process
    groups.update(group for group, _, _ in found.values() if group > 1)
    sessions.update(session for _, session, _ in found.values() if session > 1)
    found.update((pid, p) for pid, p in blank.items() if p[0] in groups or p[1] in sessions)
    return {pid: cmd for pid, (_, _, cmd) in found.items()}


if not os.path.isdir("/proc"):
    print("leftovers: not checked (no /proc)")
    sys.exit(0)
found = ours()
for pid, cmd in sorted(found.items()):
    print(f"leftovers: {pid} {cmd}")
if found and kill:
    for pid in found:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(30):
        if not ours():
            break
        time.sleep(1)
    for pid in ours():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)
    found = ours()
    print(f"leftovers: {'none left' if not found else f'{len(found)} still running'} after TERM and KILL")
elif not found:
    print("leftovers: none")
sys.exit(1 if found else 0)
EOF
}

purge() {
  local home=$1
  [[ -f $home/rehearsal.env ]] || { echo "purge: ~/$home is no rehearsal's home" >&2; exit 2; }
  rm -rf "$home"
  echo "purge: ~/$home removed"
}

# Steps, detached.
steps=$HOME/.rehearsal/steps

state() {
  local base=$steps/$1 pid
  # A status, once written, is final, whatever process its pid names by now.
  if [[ -f $base.status ]]; then
    echo "done $(cat "$base.status")"
    return 0
  fi
  pid=$(cat "$base.pid" 2>/dev/null || true)
  # Then the process, then the status again: a step writes its status before it exits, so one seen gone has written
  # any status it was to write (without the second look a step that ends between the two reads as died). On Linux
  # the process must be the step's own (FLATSHARE_REHEARSAL_STEP in its environment), not a pid reused since: one whose
  # environment cannot be read (another user's, or a non-dumpable one of this user's, as ssh-agent is) is not. grep
  # reads the file itself, as behind a pipe under pipefail its exit at the first match can kill the writer (SIGPIPE).
  if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null &&
    { [[ ! -d /proc/self ]] || grep -qzx "FLATSHARE_REHEARSAL_STEP=$1" "/proc/$pid/environ" 2>/dev/null; }; then
    echo "running $pid"
  elif [[ -f $base.status ]]; then
    echo "done $(cat "$base.status")"
  elif [[ -n $pid ]]; then
    echo "died $pid"
  else
    echo none
  fi
}

start() {
  local again=false name command base
  if [[ $1 == --again ]]; then
    again=true
    shift
  fi
  name=$1 command=$2 base=$steps/$1
  mkdir -p "$steps"
  case $(state "$name") in
    running*)
      echo "rehearsal: $name runs already ($(state "$name")): waiting for it"
      return 0
      ;;
    "done 0")
      if ! $again; then
        echo "rehearsal: $name is done"
        return 0
      fi
      ;;
  esac
  rm -f "$base.status" "$base.pid"
  # Where this run's output begins in the step's log, which keeps the runs before it.
  if [[ -f $base.log ]]; then wc -c <"$base.log" | tr -d ' ' >"$base.offset"; else echo 0 >"$base.offset"; fi
  echo "[$(date -u +%FT%TZ)] rehearsal: $name: $command" >>"$base.log"
  FLATSHARE_REHEARSAL_STEP=$name python3 -c '
import os
import sys
import time

base, command = sys.argv[1], sys.argv[2]
child = os.fork()
if child:
    # The parent returns once the bash of the step has written its pid file, so that start reads the step as
    # running, or fails once that bash has ended without writing it, which no fixed wait could tell from a slow start.
    while not os.path.exists(base + ".pid"):
        if os.waitpid(child, os.WNOHANG)[0]:
            os._exit(0 if os.path.exists(base + ".pid") else 1)
        time.sleep(0.01)
    os._exit(0)
os.setsid()
log = os.open(base + ".log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
null = os.open(os.devnull, os.O_RDONLY)
os.dup2(null, 0)
os.dup2(log, 1)
os.dup2(log, 2)
# The bash of the step writes the pid file once the exec is done: until then /proc/PID/environ reads empty, and the
# step would read as died. A bash that cannot write it ends there, its command not run. The command runs in a
# subshell, so that an exit in it ends the subshell alone, and the status is still written.
script = (
    "{ echo $$ >\"$0.pid.new\" && mv \"$0.pid.new\" \"$0.pid\"; } || exit 1\n(\n" + command + "\n)\n"
    "status=$?\necho $status >\"$0.status.new\" && mv \"$0.status.new\" \"$0.status\"\nexit $status\n"
)
os.execvp("bash", ["bash", "-c", script, base])
' "$base" "$command" ||
    { echo "rehearsal: $name did not start: its bash ended with no pid file (.rehearsal/steps/$name.log)"; return 1; }
  echo "rehearsal: $name started ($(state "$name"))"
}

wait_step() {
  local name=$1 base shown size s status
  base=$steps/$name
  shown=$(cat "$base.offset" 2>/dev/null || echo 0)
  while :; do
    s=$(state "$name")
    size=$(wc -c <"$base.log" 2>/dev/null | tr -d ' ' || echo 0)
    if ((size > shown)); then
      # The log's first SIZE bytes, then those not shown yet: no reader of the pipe stops early, where head behind
      # tail, with the log grown past SIZE, would leave tail to die of SIGPIPE, and wait with it (pipefail).
      head -c "$size" "$base.log" | tail -c +$((shown + 1))
      shown=$size
    fi
    case $s in
      "done "*)
        status=${s#done }
        echo "rehearsal: $name ended, exit status $status"
        ((status != 255)) || status=254
        return "$status"
        ;;
      died* | none)
        echo "rehearsal: $name ended without an exit status ($s)"
        return 1
        ;;
    esac
    sleep "${REHEARSAL_POLL_S:-5}"
  done
}

stop_step() {
  local name=$1 s pid i
  s=$(state "$name")
  if [[ $s != running* ]]; then
    echo "rehearsal: $name: $s"
    return 0
  fi
  pid=${s#running }
  # TERM to the step's process group, then up to 180 s for all of the group to end, not its leader alone: the step's
  # bash ends at once, while flatshare-bench, on TERM, stops its callers (30 s of grace) and AIPerf (120 s; e2/run.py)
  # before it exits. AIPerf runs in a session of its own, out of this group's reach: a KILL sooner leaves it running.
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for i in $(seq 180); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  # A status once written is final: the step's own, where it ended before the TERM.
  if [[ ! -f $steps/$name.status ]]; then
    echo 143 >"$steps/$name.status.new" && mv "$steps/$name.status.new" "$steps/$name.status"
  fi
  echo "rehearsal: $name stopped"
}

# The steps' checks, from the rehearsal's home.
verify_preflight() { # the node's preflight.json: no refused check; the counts and every check that is not ok
  local file=flatshare/node/preflight.json
  [[ -f $file ]] || { echo "verify: no $file"; return 1; }
  python3 - "$file" <<'EOF'
import json
import sys

d = json.load(open(sys.argv[1]))
print("verify: preflight: " + ", ".join(f"{n} {s}" for s, n in d["counts"].items()))
for c in d["checks"]:
    if c["status"] != "ok":
        print(f"verify:   {c['status']:<8} {c['check']}: {c['value']}")
sys.exit(1 if d["counts"].get("refused") else 0)
EOF
}

verify() {
  local runs=flatshare/runs/a100 missing="" f
  case $1 in
    env)
      grep -qx 'export FLATSHARE_SUPERVISOR=flatshare-svc' flatshare/node/env.sh ||
        { echo "verify: node/env.sh does not keep flatshare-svc"; return 1; }
      grep -qx "export HF_HUB_CACHE=$HF_HUB_CACHE" flatshare/node/env.sh ||
        { echo "verify: node/env.sh does not name the box's hub cache $HF_HUB_CACHE"; return 1; }
      echo "verify: node/env.sh keeps flatshare-svc and the box's hub cache"
      ;;
    stage)
      verify env
      verify_preflight
      local states
      # shellcheck source=/dev/null
      states=$(source flatshare/node/env.sh &&
        systemctl --user is-active flatshare-voicenet.service flatshare-livekit.service flatshare-gateway.service \
          flatshare-voice-worker.service |
        tr '\n' ' ' || true)
      [[ $states == "active active active active " ]] ||
        { echo "verify: the control plane is not up under flatshare-svc (voicenet, livekit, gateway, worker: $states)"; return 1; }
      echo "verify: the control plane is up under flatshare-svc"
      ;;
    gpu) verify_preflight ;;
    warm)
      [[ -f flatshare/cache/warm.json ]] || { echo "verify: no cache/warm.json"; return 1; }
      python3 - flatshare/cache/warm.json <<'EOF'
import json
import sys

configurations = json.load(open(sys.argv[1]))["configurations"]
print("verify: warm: " + ", ".join(f"{name} {'ok' if c.get('ok') else 'FAILED'}" for name, c in configurations.items()))
sys.exit(0 if configurations and all(c.get("ok") for c in configurations.values()) else 1)
EOF
      verify_preflight
      ;;
    block0)
      local audit want
      [[ -f $runs/blocks/0.done ]] || { echo "verify: no blocks/0.done"; return 1; }
      [[ $(tail -n 1 "$runs/block0/smoke.txt") == 'smoke: ok' ]] || { echo "verify: smoke.txt does not end in smoke: ok"; return 1; }
      [[ -s $runs/block0/call.txt ]] || { echo "verify: no block0/call.txt"; return 1; }
      # The listener audit's line: the GPU servers each in a network namespace of its own (deploy/bin/in-private-net)
      # where the kernel allows unprivileged user namespaces, else in the host's, marked FLATSHARE_PRIVATE_NET=host.
      audit=$(sed -n 's/^listener audit  *//p' "$runs/block0/smoke.txt")
      if unshare --user --map-root-user --net true 2>/dev/null; then
        want="flatshare-engine, flatshare-stt, flatshare-tts in network namespaces of their own"
      else
        want="the GPU servers run in the host's network namespace (FLATSHARE_PRIVATE_NET=host)"
      fi
      [[ $audit == ok:* && $audit == *"$want"* ]] ||
        { echo "verify: the listener audit's line is not ok with $want: ${audit:-none}"; return 1; }
      echo "verify: block 0: smoke ok; the listener audit: $audit"
      echo "verify:   the call: $(tail -1 "$runs/block0/call.txt")"
      ;;
    block1)
      [[ -f $runs/blocks/1.done ]] || missing+=" blocks/1.done"
      [[ -f $runs/block1/frozen-e0/summary.json ]] || missing+=" frozen-e0/summary.json"
      for f in kv-1 kv-2 kv-3 prefill-1 prefill-2 prefill-3 tail-1 tail-2; do
        [[ -f $runs/block1/frozen-$f.txt ]] || missing+=" frozen-$f.txt"
      done
      [[ -z $missing ]] || { echo "verify: block 1 lacks$missing"; return 1; }
      echo "verify: block 1: E0 and the eight probe runs, each with its result line:"
      for f in kv-1 prefill-1 tail-1; do echo "verify:   frozen-$f: $(tail -1 "$runs/block1/frozen-$f.txt")"; done
      ;;
    block4)
      local points=flatshare/results/e2/a100-r1
      grep -q 'a100-r1 stopped before its deadline' "$runs/blocks.log" ||
        { echo "verify: blocks.log has no deadline stop of a100-r1"; return 1; }
      grep -qs '"status": "complete"' "$points/w5-disc-pr-ab48-s24/run.json" ||
        { echo "verify: the first point (w5-disc-pr-ab48-s24) is not complete"; return 1; }
      ! grep -qs '"status": "complete"' "$points/w5-a1-voice/run.json" ||
        { echo "verify: a second point ran (w5-a1-voice): the deadline did not stop the block after its first"; return 1; }
      [[ ! -f $runs/blocks/4.done ]] || { echo "verify: blocks/4.done exists: the block was not left undone"; return 1; }
      echo "verify: block 4: the first point complete, then: $(grep 'stopping before it (--until)' "$points/campaign.log" | tail -1 | sed 's/^\[[^]]*\] //')"
      echo "verify:   $(grep 'a100-r1 stopped before its deadline' "$runs/blocks.log" | tail -1 | sed 's/^\[[^]]*\] //')"
      ;;
    *)
      echo "verify: no step $1" >&2
      return 2
      ;;
  esac
}

# What flatshare-bench's tools left when a step was killed before flatshare-bench could stop them. AIPerf runs in a
# session of its own, out of the step's group, and its processes set their titles, which blanks the environment
# leftovers reads; but flatshare-bench names each tool's process group in flatshare/.flatshare-bench-e2-tools until the
# tool is stopped. flatshare-bench's own kill_left ends those groups, from the bench venv; no such directory is no tool
# started.
kill_tools() {
  local python=flatshare/venvs/bench/bin/python
  if [[ ! -d flatshare/.flatshare-bench-e2-tools ]]; then
    echo "kill-tools: flatshare-bench started no tool here"
    return 0
  fi
  "$python" - <<'EOF'
from flatshare_bench.e2.cyan import Host
from flatshare_bench.e2.tools import kill_left, tool_scope

killed = sorted(kill_left(tool_scope(Host.load())))
print(
    f"kill-tools: flatshare-bench's kill_left "
    f"{'ended ' + ' '.join(map(str, killed)) if killed else 'found nothing left'}"
)
EOF
}

stop_units() {
  if [[ ! -f flatshare/node/env.sh ]]; then
    echo "stop-units: no node/env.sh: no unit was started"
    return 0
  fi
  # shellcheck source=/dev/null
  source flatshare/node/env.sh
  local units=(flatshare.target flatshare-engine.service flatshare-stt.service flatshare-tts.service
    flatshare-voice-worker.service flatshare-gateway.service flatshare-gateway.socket flatshare-livekit.service
    flatshare-voicenet.service) states state i=0 left=""
  systemctl --user stop "${units[@]}" || echo "stop-units: systemctl --user stop failed (above)"
  # A unit is stopped once inactive, or failed; one deactivating (flatshare-svc's state while it ends what is left of a
  # unit's group) is not yet. is-active fails unless a unit is active, so its answer is read, not its status.
  states=$(systemctl --user is-active "${units[@]}" || true)
  echo "stop-units: $(tr '\n' ' ' <<<"$states")"
  while read -r state; do
    case $state in inactive | failed) ;; *) left+=" ${units[i]:-?} (${state:-no state})" ;; esac
    i=$((i + 1))
  done <<<"$states"
  ((i == ${#units[@]})) || left+=" (is-active gave $i states for ${#units[@]} units)"
  [[ -z $left ]] || { echo "stop-units: not stopped:$left"; return 1; }
}

dispatch() {
  local command=${1:-}
  shift || true
  case $command in
    check-box) check_box "$@" ;;
    init) init "$@" ;;
    verify-box) verify_box "$@" ;;
    leftovers) leftovers "$@" ;;
    purge) purge "$@" ;;
    start) start "$@" ;;
    wait) wait_step "$@" ;;
    state) state "$@" ;;
    stop) stop_step "$@" ;;
    verify) verify "$@" ;;
    kill-tools) kill_tools ;;
    stop-units) stop_units ;;
    *)
      echo "usage: host.sh check-box|init|verify-box|leftovers|purge|start|wait|state|stop|verify|kill-tools|stop-units ..." >&2
      exit 2
      ;;
  esac
}

# bash -s reads this file from stdin as it runs it: the commands run once all of it is read, with stdin /dev/null, so
# that none of them reads the rest of the script as its input.
dispatch "$@" </dev/null
