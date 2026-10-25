#!/bin/bash
# w5-final-probes.sh: the GPU check of Lane A's final make-room module on Phastform (deployed src, e277e85), against
# v0.3's module. First the make-room suite on CPU with the engine venv (real torch, the installed vLLM 0.32.0); then,
# in the PRs' configuration (two preemptions a step, the budget cap's 16-token window, no reserve), each module: the KV
# case 3 times, the prefill case 4 times, tail.py fresh twice and growing twice, and the engine's make-room log lines.
# Restores the final build.py and stops the box at the end (restart: systemctl --user start leaseholder.target).
set -u
src=~/leaseholder/src
build=$src/deploy/phastform/variants/make-room/build.py
py=~/leaseholder/venvs/engine/bin/python
dir=$src/deploy/phastform/variants/make-room
out=~/leaseholder/scratch/w5-final
mkdir -p "$out"
cp "$build" "$out/build-final.py"
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$out/run.log"; }
ready() { local i; for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && return 0; sleep 5; done; return 1; }
others() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }
log "start; DEPLOYED $(cat $src/DEPLOYED); final build.py $(sha256sum "$build" | cut -c1-16), v0.3 build.py $(sha256sum "$out/build-v03.py" | cut -c1-16)"
rm -rf $src/python/leaseholder_vllm/src/leaseholder_vllm/__pycache__ $src/python/leaseholder_vllm/tests/__pycache__
(cd $src/python/leaseholder_vllm && VLLM_TARGET_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/leaseholder/cache/testtools \
  $py -m pytest -q -p no:cacheprovider tests/test_make_room.py tests/test_tail.py tests/test_make_room_build.py > "$out/suite.log" 2>&1)
log "CPU suite on the engine venv: exit $?: $(tail -1 "$out/suite.log")"
block() { # block NAME BUILD
  local name=$1 t0
  cp "$2" "$build"
  log "block $name: build.py sha256 $(sha256sum "$build" | cut -c1-16)"
  bash $src/deploy/phastform/variant.sh use make-room --additional-config.make_room.reserve_tokens 0 >>"$out/run.log" 2>&1
  t0=$(date '+%Y-%m-%d %H:%M:%S')
  systemctl --user restart lh-engine   # rebuild the module from this build.py
  ready || { log "$name: engine not ready"; journalctl --user -u lh-engine --since "$t0" | tail -n 40 >"$out/$name-engine-fail.log"; return 1; }
  log "$name: engine ready; additional_config $(curl -s 'http://127.0.0.1:8000/server_info?config_format=json' | $py -c 'import json,sys; print(json.load(sys.stdin)["vllm_config"].get("additional_config"))'); GPU processes: $(others | tr '\n' ';')"
  local i mode
  for i in 1 2 3; do log "$name kv $i"; $py $dir/probe.py http://127.0.0.1:8000 kv >>"$out/$name-kv.log" 2>&1; done
  for i in 1 2 3 4; do log "$name prefill $i"; $py $dir/probe.py http://127.0.0.1:8000 prefill >>"$out/$name-prefill.log" 2>&1; done
  for mode in fresh grow; do
    for i in 1 2; do log "$name tail $mode $i"; $py $dir/tail.py http://127.0.0.1:8000 240 12 $mode >"$out/$name-tail-$mode-$i.log" 2>&1; done
  done
  journalctl --user -u lh-engine --since "$t0" | grep -E "make-room:|Traceback|ERROR" >"$out/$name-engine-make-room.log"
  log "$name: done; engine make-room lines: $(grep -c 'make-room:' "$out/$name-engine-make-room.log"), errors: $(grep -c -E 'Traceback|ERROR' "$out/$name-engine-make-room.log")"
}
block final "$out/build-final.py"
block v03 "$out/build-v03.py"
cp "$out/build-final.py" "$build"
log "restored the final build.py: $(sha256sum "$build" | cut -c1-16)"
systemctl --user stop leaseholder.target
log "box stopped (restart: systemctl --user start leaseholder.target); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
log "done"
