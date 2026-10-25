#!/bin/bash
# w5-pr-blocks.sh: the probes the vLLM PRs cite, on Phastform in one session, on the deployed src (make-room as W5's
# fixes leave it): stock vLLM 0.32.0 (the frozen box), the KV preemption alone (#63350), and both PRs at the budget
# cap's default 16-token window (#63350 and #63351). Each block: the KV case 4 times, the prefill case 4 times, tail.py
# fresh and growing twice each, the engine's scheduler class and additional_config as /server_info reads them back,
# and its make-room counter lines. Stops the box at the end (restart: systemctl --user start leaseholder.target).
set -u
src=~/leaseholder/src
py=~/leaseholder/venvs/engine/bin/python
dir=$src/deploy/phastform/variants/make-room
out=~/leaseholder/scratch/w5-pr
mkdir -p "$out"
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$out/run.log"; }
ready() { local i; for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && return 0; sleep 5; done; return 1; }
others() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }
info() {
  curl -s 'http://127.0.0.1:8000/server_info?config_format=json' | $py -c 'import json, sys
c = json.load(sys.stdin)["vllm_config"]
print("scheduler_cls", c["scheduler_config"].get("scheduler_cls"), "additional_config", c.get("additional_config"))'
}
log "start; DEPLOYED $(cat $src/DEPLOYED); make-room build.py sha256 $(sha256sum $dir/build.py | cut -c1-16); GPU $(nvidia-smi --query-gpu=name,memory.used,temperature.gpu --format=csv,noheader); GPU processes: $(others | tr '\n' ';')"
block() { # block NAME VARIANT [FLAG...]
  local name=$1 t0 t1 i mode
  shift
  t0=$(date '+%Y-%m-%d %H:%M:%S')
  log "block $name: variant.sh use $*"
  bash $src/deploy/phastform/variant.sh use "$@" >>"$out/run.log" 2>&1 || { log "$name: variant.sh use failed"; return 1; }
  ready || { log "$name: engine not ready"; journalctl --user -u lh-engine --since "$t0" | tail -n 40 >"$out/$name-engine-fail.log"; return 1; }
  t1=$(date '+%Y-%m-%d %H:%M:%S') # the engine measured: its log from its ready line, not the instance it replaced
  log "$name: engine ready; $(info); GPU $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader); GPU processes: $(others | tr '\n' ';')"
  for i in 1 2 3 4; do log "$name kv $i"; $py $dir/probe.py http://127.0.0.1:8000 kv >>"$out/$name-kv.log" 2>&1; done
  for i in 1 2 3 4; do log "$name prefill $i"; $py $dir/probe.py http://127.0.0.1:8000 prefill >>"$out/$name-prefill.log" 2>&1; done
  for mode in fresh grow; do
    for i in 1 2; do log "$name tail $mode $i"; $py $dir/tail.py http://127.0.0.1:8000 240 12 $mode >"$out/$name-tail-$mode-$i.log" 2>&1; done
  done
  journalctl --user -u lh-engine --since "$t1" | grep -E "make-room:|Traceback|ERROR" >"$out/$name-engine-make-room.log"
  log "$name: done; engine make-room lines: $(grep -c 'make-room:' "$out/$name-engine-make-room.log"), errors: $(grep -c -E 'Traceback|ERROR' "$out/$name-engine-make-room.log")"
}
status=0
block stock frozen || status=1
((status)) || block a make-room --additional-config.make_room.budget_cap false --additional-config.make_room.reserve_tokens 0 || status=1
((status)) || block ab make-room --additional-config.make_room.reserve_tokens 0 || status=1
systemctl --user stop leaseholder.target
log "box stopped (restart: systemctl --user start leaseholder.target); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
((status)) && log "failed" || log "done"
