# The README's quick start on vLLM 0.33.0 as run on Phastform on 2026-11-02, with the changes 0.33.0 needs: vllm==0.33.0
# and the CUDA toolkit its torch was built with (13.0), and the module in ~/make-room-033; and with --host 127.0.0.1 on
# each serve, so that the shared host's network could not reach the engine (the probe connects to 127.0.0.1 either
# way). The tree was the one environment.txt names, the files a clone at that commit has.
uv venv --python 3.12 --seed ~/vllm-0.33 && . ~/vllm-0.33/bin/activate
pip install vllm==0.33.0 && pip install 'cuda-toolkit[nvcc]==13.0.*'
export CUDA_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13
export FLASHINFER_EXTRA_LDFLAGS=-L$CUDA_HOME/lib && ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
python python/flatshare_vllm/src/flatshare_vllm/build.py make-room ~/make-room-033

vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 256 \
  --scheduling-policy priority --max-num-batched-tokens 256 --long-prefill-token-threshold 0 --host 127.0.0.1
for run in 1 2 3 4; do
  [ "$run" -gt 1 ] && sleep 31   # retain_s and one, as for make-room below
  python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv >> stock-kv.log
done

# Once the stock engine had stopped:
PYTHONPATH=~/make-room-033 vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 256 \
  --scheduling-policy priority --max-num-batched-tokens 256 --long-prefill-token-threshold 0 \
  --scheduler-cls flatshare_make_room_scheduler.AsyncScheduler \
  --additional-config.make_room.max_preemptions_per_step 2 \
  --additional-config.make_room.budget_cap true --additional-config.make_room.budget_cap_tokens 48 \
  --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.take_reservations true \
  --additional-config.make_room.decode_hold_tokens 48 \
  --additional-config.make_room.retain_tokens 3584 --additional-config.make_room.retain_s 30 \
  --additional-config.make_room.hint_hold_ms 400 --host 127.0.0.1
for run in 1 2 3 4; do
  [ "$run" -gt 1 ] && sleep 31   # the last voice turn's prefix stays on the GPU for retain_s (30 s)
  python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv >> make-room-kv.log
done

# Then the serving variant's engine (make-room-serve's own arguments, the box's engine.yaml with its CPU KV tier,
# the serve module and just in time) on the same environment, on 127.0.0.1:8000 in place of the unit's socket:
PYTHONPATH=~/make-room-033:python/flatshare_vllm/src/flatshare_vllm/serve FLATSHARE_ENGINE_EFFECTIVE=engine-effective.json \
  vllm serve --config deploy/phastform/engine.yaml --max-num-batched-tokens 256 --long-prefill-token-threshold 0 --scheduler-cls flatshare_make_room_serve.AsyncScheduler --additional-config.make_room.max_preemptions_per_step 2 --additional-config.make_room.budget_cap true --additional-config.make_room.budget_cap_tokens 48 --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.take_reservations true --additional-config.make_room.decode_hold_tokens 48 --additional-config.make_room.retain_tokens 3584 --additional-config.make_room.retain_s 30 --additional-config.make_room.hint_hold_ms 400 --additional-config.flatshare_jit.margin_ms 30 --additional-config.flatshare_jit.step_cost.step_ms 11.56 --additional-config.flatshare_jit.step_cost.us_per_token 47.99 --additional-config.flatshare_jit.step_cost.ns_per_pair 14.17 --additional-config.flatshare_jit.step_cost.ns_per_context 429.3 --additional-config.flatshare_jit.step_cost.mixed_ms 7.6 --host 127.0.0.1
python3 deploy/bin/serving-check deploy/phastform/serving.json --engine engine-effective.json --pid <the engine's>
for run in 1 2 3 4; do
  [ "$run" -gt 1 ] && sleep 31
  python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv >> serve-kv.log
done
