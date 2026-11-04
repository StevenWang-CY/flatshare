#!/bin/bash
# r3-pr-blocks.sh: the probes vLLM #65282 and #65283 cite, on Phastform, on the deployed src (v1.21: the PRs' revision
# of 3 November 2026), as w5-pr-blocks.sh ran them on 2026-10-25: the KV preemption alone (block a: budget_cap false,
# reserve 0) and both PRs at the budget cap's default 16-token window (block ab: reserve 0), two preemptions a step in
# both, through the engine's socket, as the box serves it since W5. Each block: the KV case 4 times and the prefill
# case 4 times, the engine's scheduler class and additional_config as /server_info reads them back, and its make-room
# counter lines. Stock vLLM 0.32.0 (block stock, the frozen box) runs first as the control: the same session's
# baseline, as on 2026-10-25. Stops the box at the end, as it was.
set -u
src=~/flatshare/src
py=~/flatshare/venvs/engine/bin/python
dir=$src/python/flatshare_vllm/src/flatshare_vllm/make-room
out=~/flatshare/scratch/r3-pr
sock=/run/user/$(id -u)/flatshare/engine.sock  # the engine listens on its socket only
mkdir -p "$out"
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$out/run.log"; }
ready() { local i; for i in $(seq 1 120); do curl -sf --unix-socket "$sock" http://localhost/health >/dev/null && return 0; sleep 5; done; return 1; }
others() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }
info() {
  curl -s --unix-socket "$sock" 'http://localhost/server_info?config_format=json' | $py -c 'import json, sys
c = json.load(sys.stdin)["vllm_config"]
print("scheduler_cls", c["scheduler_config"].get("scheduler_cls"), "additional_config", c.get("additional_config"))'
}
log "start; DEPLOYED $(cat $src/DEPLOYED); module sha256 of the make-room patches: $(cat $dir/63350.patch $dir/63351.patch $dir/variant.patch | sha256sum | cut -c1-16); GPU $(nvidia-smi --query-gpu=name,memory.used,temperature.gpu --format=csv,noheader); GPU processes: $(others | tr '\n' ';')"
block() { # block NAME VARIANT [FLAG...]
  local name=$1 t0 t1 i
  shift
  t0=$(date '+%Y-%m-%d %H:%M:%S')
  log "block $name: variant.sh use $*"
  bash $src/deploy/phastform/variant.sh use "$@" >>"$out/run.log" 2>&1 || { log "$name: variant.sh use failed"; return 1; }
  ready || { log "$name: engine not ready"; journalctl --user -u flatshare-engine --since "$t0" | tail -n 40 >"$out/$name-engine-fail.log"; return 1; }
  t1=$(date '+%Y-%m-%d %H:%M:%S')
  log "$name: engine ready; $(info); GPU $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader); GPU processes: $(others | tr '\n' ';')"
  for i in 1 2 3 4; do log "$name kv $i"; $py $dir/probe.py "unix:$sock" kv >>"$out/$name-kv.log" 2>&1; done
  for i in 1 2 3 4; do log "$name prefill $i"; $py $dir/probe.py "unix:$sock" prefill >>"$out/$name-prefill.log" 2>&1; done
  journalctl --user -u flatshare-engine --since "$t1" | grep -E "make-room:|Traceback|ERROR" >"$out/$name-engine-make-room.log"
  log "$name: done; engine make-room lines: $(grep -c 'make-room:' "$out/$name-engine-make-room.log"), errors: $(grep -c -E 'Traceback|ERROR' "$out/$name-engine-make-room.log")"
}
status=0
block stock frozen || status=1
((status)) || block a make-room --additional-config.make_room.budget_cap false --additional-config.make_room.reserve_tokens 0 || status=1
((status)) || block ab make-room --additional-config.make_room.reserve_tokens 0 || status=1
systemctl --user stop flatshare.target
log "box stopped (restart: systemctl --user start flatshare.target); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
((status)) && log "failed" || log "done"
