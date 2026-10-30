# The serving engine on vLLM's own image: vLLM 0.32.0 with make-room's scheduler (vllm-project/vllm#63350 and #63351,
# as python/flatshare_vllm builds them) at the serving profile's settings, held to that profile once it starts, as the
# box's unit holds it. deploy/container/README.md says how to build, run and check it. From the repository's root:
#
#     docker build -f deploy/container/Dockerfile -t flatshare-engine .
ARG VLLM_IMAGE=docker.io/vllm/vllm-openai:v0.32.0
FROM ${VLLM_IMAGE}

# The repository's files the engine needs, at their own paths, as a release lays them out on the box, so that
# serving-check finds make-room's patches where it finds them there. .dockerignore lets in these and nothing else.
COPY . /opt/flatshare/
# make-room's module, written once from the patches against the image's vLLM: build.py refuses any other version.
RUN python3 /opt/flatshare/python/flatshare_vllm/src/flatshare_vllm/build.py make-room /opt/flatshare/make-room

ENV PYTHONPATH=/opt/flatshare/make-room:/opt/flatshare/python/flatshare_vllm/src/flatshare_vllm/serve \
    FLATSHARE_ENGINE_EFFECTIVE=/run/flatshare/engine-effective.json \
    VLLM_SERVER_DEV_MODE=0 \
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE=60
# Every compile and JIT cache the engine's start reads, under one directory for a volume: the four the A100 profile
# keeps under cache/ (deploy/profiles/a100-40gb.toml), and the CUDA driver's, which a host keeps in ~/.nv. With the
# volume, only a compile configuration's first start compiles; at their defaults they would go with each container.
ENV VLLM_CACHE_ROOT=/opt/flatshare/cache/vllm \
    TORCHINDUCTOR_CACHE_DIR=/opt/flatshare/cache/inductor \
    TRITON_CACHE_DIR=/opt/flatshare/cache/triton \
    FLASHINFER_WORKSPACE_BASE=/opt/flatshare/cache/flashinfer \
    CUDA_CACHE_PATH=/opt/flatshare/cache/nv
ENTRYPOINT ["python3", "/opt/flatshare/deploy/container/serve.py"]
