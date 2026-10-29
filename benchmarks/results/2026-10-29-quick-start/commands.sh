# The README's quick start as run on Phastform on 2026-10-29, with one change: --host 127.0.0.1 on each serve, so that
# the shared host's network could not reach the engine (the probe connects to 127.0.0.1 either way). The tree was the
# git archive of the commit in environment.txt, the files a clone at that commit has.
uv venv --python 3.12 --seed ~/vllm-0.32 && . ~/vllm-0.32/bin/activate
pip install vllm==0.32.0 'cuda-toolkit[nvcc]==13.0.3.0'
export CUDA_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13
export FLASHINFER_EXTRA_LDFLAGS=-L$CUDA_HOME/lib && ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
python python/flatshare_vllm/src/flatshare_vllm/build.py make-room ~/make-room

vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 256 \
  --scheduling-policy priority --max-num-batched-tokens 256 --long-prefill-token-threshold 0 --host 127.0.0.1
for run in 1 2 3 4; do
  python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv >> stock-kv.log
done

# Once the stock engine had stopped:
PYTHONPATH=~/make-room vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 256 \
  --scheduling-policy priority --max-num-batched-tokens 256 --long-prefill-token-threshold 0 \
  --scheduler-cls flatshare_make_room_scheduler.AsyncScheduler \
  --additional-config.make_room.max_preemptions_per_step 2 \
  --additional-config.make_room.budget_cap true --additional-config.make_room.budget_cap_tokens 48 \
  --additional-config.make_room.reserve_tokens 0 --host 127.0.0.1
for run in 1 2 3 4; do
  python python/flatshare_vllm/src/flatshare_vllm/make-room/probe.py http://127.0.0.1:8000 kv >> make-room-kv.log
done
