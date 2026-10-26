#!/bin/bash
# w5-loadhold.sh: the third block of the GPU check, after w5-final-probes.sh: e277e85's build.py with a bounded load
# hold (a KV load in flight holds lower-priority prefill for at most budget_cap_tokens steps) and the scheduler logger
# kept, in the PRs' configuration: the KV case 3 times, the prefill case 4 times, tail.py fresh and growing twice
# each, and the engine's make-room counter lines. Restores the final build.py and stops the box at the end
# (restart: systemctl --user start flatshare.target).
set -u
src=~/flatshare/src
build=$src/deploy/phastform/variants/make-room/build.py
py=~/flatshare/venvs/engine/bin/python
dir=$src/deploy/phastform/variants/make-room
out=~/flatshare/scratch/w5-final
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$out/run.log"; }
ready() { local i; for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && return 0; sleep 5; done; return 1; }
while pgrep -f w5-final-probes.sh >/dev/null; do sleep 10; done
name=loadhold
cp "$out/build-loadhold.py" "$build"
log "block $name: build.py sha256 $(sha256sum "$build" | cut -c1-16)"
bash $src/deploy/phastform/variant.sh use make-room --additional-config.make_room.reserve_tokens 0 >>"$out/run.log" 2>&1
t0=$(date '+%Y-%m-%d %H:%M:%S')
systemctl --user restart flatshare-engine
ready || { log "$name: engine not ready"; journalctl --user -u flatshare-engine --since "$t0" | tail -n 40 >"$out/$name-engine-fail.log"; cp "$out/build-final.py" "$build"; systemctl --user stop flatshare.target; exit 1; }
log "$name: engine ready; additional_config $(curl -s 'http://127.0.0.1:8000/server_info?config_format=json' | $py -c 'import json,sys; print(json.load(sys.stdin)["vllm_config"].get("additional_config"))')"
for i in 1 2 3; do log "$name kv $i"; $py $dir/probe.py http://127.0.0.1:8000 kv >>"$out/$name-kv.log" 2>&1; done
for i in 1 2 3 4; do log "$name prefill $i"; $py $dir/probe.py http://127.0.0.1:8000 prefill >>"$out/$name-prefill.log" 2>&1; done
for mode in fresh grow; do
  for i in 1 2; do log "$name tail $mode $i"; $py $dir/tail.py http://127.0.0.1:8000 240 12 $mode >"$out/$name-tail-$mode-$i.log" 2>&1; done
done
journalctl --user -u flatshare-engine --since "$t0" | grep -E "make-room:|Traceback|ERROR" \
  >"$out/$name-engine-make-room.log"
log "$name: done; engine make-room lines: $(grep -c 'make-room:' "$out/$name-engine-make-room.log"), errors: $(grep -c -E 'Traceback|ERROR' "$out/$name-engine-make-room.log")"
cp "$out/build-final.py" "$build"
log "restored the final build.py: $(sha256sum "$build" | cut -c1-16)"
systemctl --user stop flatshare.target
log "box stopped (restart: systemctl --user start flatshare.target); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
log "loadhold done"
