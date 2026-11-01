#!/usr/bin/env bash
# run.sh NAME CONFIG DURATION_S [VLLM_FLAG...]: serve vLLM (stock, from $FP8_VENV) with CONFIG and the flags on a
# private Unix socket, drive it with load.py for DURATION_S seconds or until its engine dies, then stop it. Everything
# lands in runs/NAME: server.log, requests.jsonl, summary.json and result.txt (the verdict, the times, the first CUDA
# error). Variables in the environment (CUDA_LAUNCH_BLOCKING, CUDA core dump settings) pass through to the server.
# Every Python process of the server loads diag/sitecustomize.py, so a stalled engine dumps its threads' stacks to
# server.log on SIGUSR1, without ptrace (the host's ptrace_scope is 1).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
name=$1 config=$2 duration=$3
shift 3
venv=${FP8_VENV:-$HOME/flatshare/venvs/engine}
out=$here/runs/$name
if [[ -e $out ]]; then echo "run $name exists" >&2; exit 2; fi
mkdir -p "$out"
rundir=$XDG_RUNTIME_DIR/fp8inv
install -d -m 0700 "$rundir"
sock=$rundir/$name.sock
rm -f "$sock"
stamp() { date -u +%FT%TZ; }

# The GPU must be ours alone: no compute process, under 100 MiB used.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
if (( used > 100 || procs > 0 )); then echo "GPU busy: ${used} MiB, ${procs} processes" >&2; exit 2; fi

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin
export VIRTUAL_ENV=$venv
export FLASHINFER_EXTRA_CUDAFLAGS=-I$HOME/flatshare/cache/curand_include
export FLASHINFER_EXTRA_CFLAGS=-I$HOME/flatshare/cache/curand_include
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$HOME/flatshare/cache/flashinfer}
export HF_HUB_OFFLINE=1
export VLLM_RPC_BASE_PATH=$rundir/$name.zmq
export PYTHONPATH=$here/diag${PYTHONPATH:+:$PYTHONPATH}
install -d -m 0700 "$VLLM_RPC_BASE_PATH"

{
  echo "name: $name"
  echo "config: $config"
  echo "flags: $*"
  echo "venv: $venv"
  echo "env: $(env | grep -E '^(CUDA_LAUNCH_BLOCKING|CUDA_ENABLE|CUDA_COREDUMP|VLLM_|TORCH_|FP8INV_|FLASHINFER_WORKSPACE_BASE)' | grep -v VLLM_RPC_BASE_PATH | tr '\n' ' ')"
  echo "versions: $("$venv/bin/python" -c 'import vllm,torch,flashinfer;print(vllm.__version__,torch.__version__,flashinfer.__version__)' 2>&1)"
  echo "started: $(stamp)"
} > "$out/result.txt"

"$HOME/flatshare/src/deploy/bin/in-private-net" "$venv/bin/vllm" serve --config "$here/$config" --uds "$sock" "$@" \
  > "$out/server.log" 2>&1 &
pid=$!

healthy=0
for _ in $(seq 1 600); do
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  if curl -s -o /dev/null -w '%{http_code}' --unix-socket "$sock" http://localhost/health 2>/dev/null | grep -q 200; then
    healthy=1
    break
  fi
  sleep 2
done
echo "healthy: $healthy $(stamp)" >> "$out/result.txt"

load_rc=-1
if (( healthy )); then
  "$venv/bin/python" "$here/load.py" --uds "$sock" --duration "$duration" --out "$out" > "$out/load.out" 2>&1
  load_rc=$?
fi
echo "load: exit $load_rc $(stamp)" >> "$out/result.txt"

alive=0
kill -0 "$pid" 2>/dev/null && alive=1
fault=$(grep -m1 -E 'CUDA error|AcceleratorError|illegal memory|address space|EngineCore encountered a fatal error' \
  "$out/server.log" | cut -c1-400)
if (( load_rc == 4 )); then
  # Stalled: ask every engine core for its threads' stacks, and keep the GPU's state, before stopping it.
  for core in $(pgrep -f '[V]LLM::EngineCore'); do kill -USR1 "$core"; done
  sleep 5
  nvidia-smi > "$out/stall-nvidia-smi.txt" 2>&1
  # With user-triggered core dumps on (CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1, and CUDA_COREDUMP_PIPE naming a pipe per
  # process with %p), a write to an engine core's pipe has the driver dump the GPU's state, the kernel that never
  # finishes included. The process must not be signalled while the driver writes, so wait for the dump to stop growing.
  if [[ ${CUDA_ENABLE_USER_TRIGGERED_COREDUMP:-0} == 1 && -n ${CUDA_COREDUMP_PIPE:-} && -n ${CUDA_COREDUMP_FILE:-} ]]; then
    for core in $(pgrep -f '[V]LLM::EngineCore'); do
      pipe=${CUDA_COREDUMP_PIPE//%p/$core}
      if [[ -p $pipe ]] && timeout 10 bash -c 'echo > "$1"' _ "$pipe"; then
        echo "user core dump: requested from $core $(stamp)" >> "$out/result.txt"
      else
        echo "user core dump: no pipe answered for $core" >> "$out/result.txt"
      fi
    done
    dumps=$(dirname "$CUDA_COREDUMP_FILE") prefix=$(basename "${CUDA_COREDUMP_FILE%%\%*}")
    last=-1 same=0 size=0
    for _ in $(seq 1 150); do
      size=$(find "$dumps" -maxdepth 1 -type f -name "$prefix*" -printf '%s\n' | awk '{s += $1} END {print s + 0}')
      if (( size > 0 && size == last )); then (( ++same >= 5 )) && break; else same=0; fi
      last=$size
      sleep 2
    done
    echo "user core dump: $size bytes in $(find "$dumps" -maxdepth 1 -type f -name "$prefix*" -printf '%f ')$(stamp)" \
      >> "$out/result.txt"
  fi
fi
if [[ -n $fault ]]; then verdict=CRASHED
elif (( load_rc == 4 )); then verdict=STALLED
# The server stopped answering without a fault line the pattern knows: still the engine's death, not a clean run.
elif (( load_rc == 3 )); then verdict=DIED
elif (( healthy )); then verdict=SURVIVED
else verdict=NO-START; fi
{
  echo "verdict: $verdict"
  echo "server alive at the end: $alive"
  echo "first fault line: ${fault:-none}"
} >> "$out/result.txt"

kill -INT "$pid" 2>/dev/null
for _ in $(seq 1 45); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid"; echo "stop: SIGKILL" >> "$out/result.txt"; fi
wait "$pid" 2>/dev/null
echo "server exit: $? $(stamp)" >> "$out/result.txt"
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( used < 100 )) && break
  sleep 2
done
echo "gpu after: ${used} MiB" >> "$out/result.txt"
rm -rf "$sock" "$VLLM_RPC_BASE_PATH"
cat "$out/result.txt"
