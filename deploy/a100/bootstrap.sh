#!/usr/bin/env bash
# bootstrap.sh stage|gpu|warm: make one host the GPU node and the control host of a campaign (single-host mode,
# docs/HARDWARE.md), from the source tree at ~/leaseholder/src, which deploy/a100/push.sh copies there with the
# corpora and the callers' audio. Each stage is safe to re-run, skips what is done and ends with the preflight
# (deploy/a100/host.py preflight), whose refused checks fail the stage. Its log: ~/leaseholder/node/bootstrap.log.
#   stage  needs no GPU, so it can run before the billed clock: uv and Python 3.12; the host's facts and the node
#          rendered from deploy/profiles/$LH_PROFILE (a100-40gb) into ~/leaseholder/node; the venvs, the engine's
#          from the lock the driver needs (profile.py's R8); Kokoro and the models at their pinned revisions;
#          LiveKit and lh-gateway; the files the JIT compilers need; the units and the control plane's
#          configuration, and the control plane started; FlashInfer's sampler compiled for the card
#   gpu    with the card: the facts again, now with the driver (the engine venv re-synced if the lock changes),
#          vLLM's own parse of the node's configs, and the preflight with the GPU's checks
#   warm   one engine start for each compile configuration (cache/warm.json), so that the campaign starts warm
# LH_GPU_ID names the GPU the host lends, by nvidia-smi's index or its UUID, on a host with several (the preflight
# refuses several and none named): detect resolves it to its UUID, which the GPU units run on alone and env.sh keeps;
# LH_ALLOW_MIG=1 measures on a MIG slice, which the preflight otherwise refuses, and env.sh keeps it too.
# LH_PROFILE names another profile; LH_MODELS adds MODEL@REVISION words to download (the 14B, for C'); LH_WARM is a
# file of the configurations to warm (host.py warm --configs). The first stage chooses what runs the units, the user
# manager where it outlives the session (lingering, which it enables where a user may) or else deploy/bin/lh-svc, and
# env.sh keeps the choice for the later stages and every script that sources it. LH_SUPERVISOR=lh-svc runs them under
# lh-svc even where systemd's user manager could (docs/HARDWARE.md, single-host mode), and LH_SUPERVISOR=auto
# chooses again from the facts; either moves the units, stopping them under the supervisor they leave first.
# Sourced (as a test does), it defines its functions and runs nothing.
set -euo pipefail
src=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=$HOME/leaseholder
node=$root/node
profile=${LH_PROFILE:-a100-40gb}
stage=""
log() { echo "[$(date -u +%FT%TZ)] bootstrap $stage: $*"; }

choose_supervisor() { # LH_SUPERVISOR, checked before anything is written; previous: what ran an earlier stage's units
  previous=""
  [[ ! -f $node/env.sh ]] || previous=$(sed -n 's/^export LH_SUPERVISOR=//p' "$node/env.sh")
  case ${LH_SUPERVISOR:-} in
    "") LH_SUPERVISOR=${previous:-auto} ;; # an earlier stage's choice (lh-svc or systemd), kept
    auto | lh-svc) ;;
    *) echo "bootstrap: LH_SUPERVISOR is auto or lh-svc, not '$LH_SUPERVISOR'" >&2; exit 2 ;;
  esac
  export LH_SUPERVISOR
}

choose_gpu() { # LH_GPU_ID, checked before anything is written: unset, an earlier stage's (env.sh, its UUID)
  if [[ -z ${LH_GPU_ID:-} && -f $node/env.sh ]]; then
    LH_GPU_ID=$(sed -n 's/^export LH_GPU_ID=//p' "$node/env.sh")
  fi
  [[ -z ${LH_GPU_ID:-} || $LH_GPU_ID =~ ^([0-9]+|GPU-[0-9A-Za-z-]+)$ ]] ||
    { echo "bootstrap: LH_GPU_ID is a GPU's index or UUID (nvidia-smi -L), not '$LH_GPU_ID'" >&2; exit 2; }
  [[ -z ${LH_GPU_ID:-} ]] || export LH_GPU_ID
  # LH_ALLOW_MIG=1, measuring on a MIG slice (the preflight refuses one without it), is kept so too.
  if [[ -z ${LH_ALLOW_MIG:-} && -f $node/env.sh ]]; then
    LH_ALLOW_MIG=$(sed -n 's/^export LH_ALLOW_MIG=//p' "$node/env.sh")
  fi
  [[ -z ${LH_ALLOW_MIG:-} ]] || export LH_ALLOW_MIG
}

choose_hub_cache() { # the Hugging Face hub cache, resolved once: an earlier stage's (env.sh), else as huggingface_hub
  # resolves it from the operator's environment (HF_HOME on a data volume is often where the room is)
  if [[ -z ${HF_HUB_CACHE:-} && -f $node/env.sh ]]; then # (sourced: env.sh quotes a path as the shell reads it)
    HF_HUB_CACHE=$(bash -c 'unset HF_HUB_CACHE; source "$1" && printf %s "${HF_HUB_CACHE:-}"' _ "$node/env.sh")
  fi
  : "${HF_HUB_CACHE:=${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
  export HF_HUB_CACHE
}

choose() { # what runs the units, on which GPU, and where the models are
  choose_supervisor
  choose_gpu
  choose_hub_cache
}

# uv at the version the locks were resolved with or newer (uv 0.10.8 could not resolve the engine lock's torchcodec
# through --torch-backend), and its own Python 3.12 for the stdlib tools below and every venv (--managed-python: uv
# prefers a system Python, and Ubuntu 24.04's 3.12 has no Python.h without libpython3.12-dev, which Triton compiles
# its launcher against).
uv=${UV:-$HOME/.local/bin/uv}
uv_new_enough() {
  [[ -x $1 ]] && [[ $(printf '%s\n' 0.11.27 "$("$1" --version | awk '{print $2}')" | sort -V | head -1) == 0.11.27 ]]
}
tools() { # uv, its Python 3.12 (py), the compilers and ss
  if ! uv_new_enough "$uv"; then
    log "uv 0.11.27 into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/0.11.27/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh -s -- -q
    uv=$HOME/.local/bin/uv
  fi
  "$uv" python install -q 3.12
  py=$("$uv" python find --managed-python 3.12)
  # Triton builds its launchers with a C compiler and nvcc needs a host C++ compiler.
  if ! command -v cc >/dev/null || ! command -v c++ >/dev/null; then
    log "no C or C++ compiler: build-essential"
    apt_install build-essential || log "build-essential not installed (neither root nor sudo, or no apt): the preflight refuses"
  fi
  # smoke.sh's listener audit (deploy/bin/audit-listeners) reads the host's sockets with ss.
  if ! command -v ss >/dev/null; then
    log "no ss: iproute2"
    apt_install iproute2 || log "iproute2 not installed (neither root nor sudo, or no apt): the preflight refuses"
  fi
}
host_py() { "$py" "$src/deploy/a100/host.py" "$@"; }
field() { # field FILE KEY...: a value of a JSON file
  "$py" -c 'import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    v = v.get(k) if isinstance(v, dict) else None
print("" if v is None else v)' "$@"
}
as_root() { # as_root COMMAND...: as it is where this user is root (a container's), else through sudo -n
  if (($(id -u) == 0)); then "$@"; else sudo -n "$@"; fi
}
with_sudo() { { (($(id -u) == 0)) || sudo -n true 2>/dev/null; } && command -v apt-get >/dev/null; }
apt_install() { # apt_install PACKAGE...: where root or sudo, and apt, work; fails otherwise
  with_sudo || return 1
  as_root apt-get update -q >/dev/null && as_root apt-get install -y -q "$@" >/dev/null
}

detect() { # the host's facts, as the host's own systemctl answers (not the shims, should env.sh be sourced)
  PATH=${PATH//"$src"\/deploy\/bin\/shims:/} "$py" "$src/deploy/profile.py" detect --out "$node/host.json"
}

render() {
  detect
  # Lingering keeps the user manager, and so the units, past the last SSH session; without it, lh-svc runs them. A
  # user may enable its own (polkit's set-self-linger): the facts are read again at once, so that this stage
  # chooses the supervisor the later ones keep.
  if [[ $LH_SUPERVISOR != lh-svc && $(field "$node/host.json" systemd_user) == True &&
    $(field "$node/host.json" linger) != True ]]; then
    if loginctl enable-linger "$(id -un)" 2>/dev/null; then
      detect
    elif [[ $LH_SUPERVISOR == auto ]]; then
      log "lingering stays off (loginctl enable-linger was refused): lh-svc runs the units"
    else
      log "lingering stays off (loginctl enable-linger was refused): env.sh's systemd would stop with the session"
    fi
  fi
  "$py" "$src/deploy/profile.py" render "$profile" --host "$node/host.json" --out "$node"
  # R535 or R570 on a data-center card runs the box's lock once cuda-compat-13-0 is installed.
  if [[ $(field "$node/node.json" install path) == cu129 && $(field "$node/node.json" install compat_eligible) == True ]]; then
    log "the driver can run CUDA 13.0 through cuda-compat-13-0: installing it"
    if apt_install cuda-compat-13-0; then
      detect
      "$py" "$src/deploy/profile.py" render "$profile" --host "$node/host.json" --out "$node"
    else
      log "cuda-compat-13-0 not installed (neither root nor sudo, no apt, or no NVIDIA repository): the cu129 lock"
    fi
  fi
  log "install path: $(field "$node/node.json" install path): $(field "$node/node.json" install reason)"
}

gpu_units=(leaseholder.target lh-engine lh-stt lh-tts)
units=("${gpu_units[@]}" lh-voice-worker lh-gateway.service lh-gateway.socket lh-livekit lh-voicenet)
stop_units() { # stop_units lh-svc|systemd [UNIT...]: stop this host's units (all of them by default) under it
  local supervisor=$1
  shift
  (($#)) || set -- "${units[@]}"
  if [[ $supervisor == lh-svc ]]; then
    # With the env.sh that started them: lh-svc's shims and runtime directory.
    # shellcheck source=/dev/null
    (source "$node/env.sh" && systemctl --user stop "$@") 2>/dev/null || true
  else
    local manager
    manager=$(PATH=${PATH//"$src"\/deploy\/bin\/shims:/} command -v systemctl || true)
    [[ -z $manager ]] || "$manager" --user stop "$@" 2>/dev/null || true
  fi
}

shell_env() { # env.sh, which every command on this host sources: lh-svc's shims where it runs the units
  local chosen
  chosen=$(host_py supervisor "$node")
  if [[ -n $previous && $previous != "$chosen" ]]; then
    # LH_SUPERVISOR moves the units: they stop where they ran, so that no unit runs under both.
    log "the units move from $previous to $chosen (LH_SUPERVISOR=$LH_SUPERVISOR): stopping them under $previous"
    stop_units "$previous"
  elif [[ $chosen == lh-svc && $(field "$node/host.json" systemd_user) == True ]]; then
    # The user manager may run units that no env.sh recorded: they stop, so that lh-svc's take their sockets.
    stop_units systemd
  fi
  host_py env "$node"
  # shellcheck source=/dev/null
  source "$node/env.sh"
  previous=$chosen
}

stage_facts() { # the facts and the node, kept as staged.json, whose CPU tier the later stages' preflight holds to
  render
  shell_env
  cp "$node/node.json" "$node/staged.json"
}

gpu_facts() { # the facts with the card, the GPU units stopped first: the engine that warm or a block left running
  # holds its /dev/shm region and RAM, which R3 bounds the CPU tier by (block 0 starts the units again)
  [[ -z $previous ]] || stop_units "$previous" "${gpu_units[@]}"
  render
  shell_env
}

venv() { # venv DIR: a venv of uv's own Python 3.12; one an earlier stage made from a Python without Python.h goes
  if [[ -x $1/bin/python ]] && ! host_py headers "$1/bin/python" >/dev/null; then
    log "$1 has no Python.h (a system Python's): making it anew from uv's own Python 3.12"
    rm -rf "$1"
  fi
  [[ -x $1/bin/python ]] || "$uv" venv -q --managed-python --python 3.12 "$1"
}

venvs() {
  local path lock backend
  path=$(field "$node/node.json" install path)
  case $path in
    cu130 | cu130-compat) lock=$src/deploy/phastform/requirements/engine.txt backend=cu130 ;;
    cu129) lock=$src/deploy/a100/requirements/engine-cu129.txt backend=cu129 ;;
    *) log "no install path in node.json"; exit 2 ;;
  esac
  venv "$root/venvs/engine"
  log "engine venv: $path, ${lock#"$src/"}"
  "$uv" pip sync -q --python "$root/venvs/engine/bin/python" --torch-backend "$backend" "$lock"
  echo "$path" >"$root/venvs/engine/lh-install-path"
}

compilers() { # the files FlashInfer's JIT reads beside the compiler the units name (units.cuda_home)
  local site=$root/venvs/engine/lib/python3.12/site-packages/nvidia cuda_home
  mkdir -p "$root"/cache/{vllm,inductor,triton,flashinfer,curand_include,cuda_lib}
  find "$root/cache/curand_include" "$root/cache/cuda_lib" -type l -delete
  if [[ -d $site/cu13 ]]; then
    ln -sf "$site"/cu13/include/curand*.h "$root/cache/curand_include/"
    # The venv's CUDA 13 has lib/libcudart.so.13 and no stubs, where FlashInfer links -lcudart and -lcuda.
    ln -sf "$site/cu13/lib/libcudart.so.13" "$root/cache/cuda_lib/libcudart.so"
  else
    ln -sf "$site"/curand/include/curand*.h "$root/cache/curand_include/"
  fi
  # The driver's library is libcuda.so.1: a stub of that soname links before the GPU is visible, as the toolkit's does.
  if [[ ! -f $root/cache/cuda_lib/libcuda.so ]] && command -v cc >/dev/null; then
    cc -shared -Wl,-soname,libcuda.so.1 -o "$root/cache/cuda_lib/libcuda.so" -x c /dev/null
  fi
  cuda_home=$(field "$node/node.json" values units.cuda_home)
  if [[ $cuda_home == "%h/leaseholder/cuda-12.9" && ! -x $root/cuda-12.9/bin/nvcc ]]; then
    # CUDA 12's wheels carry no nvcc; NVIDIA's runfile installs the toolkit alone into a user directory, no root.
    log "CUDA 12.9.1's toolkit into ~/leaseholder/cuda-12.9 (a 5.9 GB download, about 10 GB installed)"
    local run=$root/vendor/cuda_12.9.1_575.57.08_linux.run
    curl -fsSL -o "$run" https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
    mkdir -p "$root/vendor/cuda-tmp"
    sh "$run" --silent --toolkit --toolkitpath="$root/cuda-12.9" --no-man-page --override --tmpdir="$root/vendor/cuda-tmp"
    rm -rf "$run" "$root/vendor/cuda-tmp"
  fi
}

audio_and_models() {
  venv "$root/venvs/tts"
  "$uv" pip sync -q --python "$root/venvs/tts/bin/python" --torch-backend cu128 "$src/deploy/phastform/requirements/tts.txt"
  # Kokoro-FastAPI publishes no tag for 0.9.1-rc1, so the measured commit is the version pin (as on the box).
  local kokoro=$root/vendor/Kokoro-FastAPI
  command -v git >/dev/null || apt_install git || { log "no git for Kokoro-FastAPI's checkout"; exit 2; }
  [[ -d $kokoro/.git ]] || git clone -q https://github.com/remsky/Kokoro-FastAPI.git "$kokoro"
  git -C "$kokoro" checkout -q ebd4ec9
  "$uv" pip install -q --python "$root/venvs/tts/bin/python" --no-deps -e "$kokoro"
  if [[ ! -f $kokoro/api/src/models/v1_0/kokoro-v1_0.pth ]]; then
    (cd "$kokoro" && "$root/venvs/tts/bin/python" docker/scripts/download_model.py --output api/src/models/v1_0)
  fi
  # The units run offline (HF_HUB_OFFLINE=1), so the models come here, at the revisions the node's configs pin.
  local config model revision pin
  for config in "$node/engine.yaml" "$node/stt.yaml"; do
    read -r model revision < <(awk '$1 == "model:" {m = $2} $1 == "revision:" {r = $2} END {print m, r}' "$config")
    [[ -n $revision ]] || { log "$config pins no revision"; exit 2; }
    "$root/venvs/engine/bin/hf" download -q "$model" --revision "$revision" >/dev/null
  done
  for pin in ${LH_MODELS:-}; do
    "$root/venvs/engine/bin/hf" download -q "${pin%@*}" --revision "${pin#*@}" >/dev/null
  done
}

control_plane() {
  local venv
  for venv in voice:leaseholder_voice bench:leaseholder_bench; do
    UV_PROJECT_ENVIRONMENT=$root/venvs/${venv%%:*} "$uv" sync -q --locked --managed-python --python 3.12 \
      --extra test --project "$src/python/${venv#*:}"
  done
  HF_HUB_OFFLINE=1 "$root/venvs/voice/bin/python" -c \
    'from livekit.local_inference import init_eot; init_eot(); from livekit.plugins import silero; silero.VAD.load()'
  local livekit=1.13.10 arch
  case $(uname -m) in x86_64) arch=amd64 ;; aarch64) arch=arm64 ;; *) log "no LiveKit build for $(uname -m)"; exit 2 ;; esac
  if ! "$root/bin/livekit-server" --version 2>/dev/null | grep -q "$livekit"; then
    curl -fsSL "https://github.com/livekit/livekit/releases/download/v$livekit/livekit_${livekit}_linux_$arch.tar.gz" |
      tar -xz -C "$root/bin" livekit-server
  fi
  # A gateway built elsewhere may not run here (another glibc); then it is built here, with the repository's toolchain.
  if ! "$root/bin/lh-gateway" --version >/dev/null 2>&1; then
    log "building lh-gateway"
    [[ -x $HOME/.cargo/bin/rustup ]] ||
      curl -fsSL https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain none --no-modify-path
    "$HOME/.cargo/bin/rustup" toolchain install "$(sed -n 's/^channel = "\(.*\)"/\1/p' "$src/rust-toolchain.toml")" \
      --profile minimal >/dev/null 2>&1
    (cd "$src" && "$HOME/.cargo/bin/cargo" build -q --release --locked -p lh-gateway)
    install -m 755 "$src/target/release/lh-gateway" "$root/bin/lh-gateway"
  fi
  host_py units "$node" >/dev/null
  host_py config "$node"
  # The voice worker and what it wants: LiveKit in the voice plane and the gateway, on its socket unit.
  systemctl --user start lh-voice-worker.service ||
    log "the control plane did not start: journalctl --user -u lh-voice-worker (and lh-livekit, lh-gateway)"
}

preflight() { # preflight [--gpu]: the stage's status is the preflight's
  local status=0
  host_py preflight "$node" "$@" || status=$?
  log "done ($([[ $status == 0 ]] && echo 'no check refused' || echo 'the preflight refuses: see above'))"
  exit "$status"
}

main() {
  stage=${1:-}
  [[ $stage == stage || $stage == gpu || $stage == warm ]] || { echo "usage: bootstrap.sh stage|gpu|warm" >&2; exit 2; }
  choose
  mkdir -p "$root"/{bin,venvs,vendor,data,runs,results,cache} "$node"
  exec > >(tee -a "$node/bootstrap.log") 2>&1
  log "from $src ($(cat "$src/DEPLOYED" 2>/dev/null || echo 'no DEPLOYED'))"
  tools
  case $stage in
    stage)
      stage_facts
      venvs
      compilers
      audio_and_models
      control_plane
      host_py jit-check "$node" || log "FlashInfer's sampler did not build: the preflight refuses"
      preflight
      ;;
    gpu)
      nvidia-smi -L >/dev/null 2>&1 || { log "nvidia-smi shows no GPU"; exit 2; }
      local built
      built=$(cat "$root/venvs/engine/lh-install-path" 2>/dev/null || true)
      gpu_facts
      if [[ $(field "$node/node.json" install path) != "$built" ]]; then
        venvs
      fi
      compilers
      host_py units "$node" >/dev/null
      host_py config "$node"
      # Again where the node now names another compiler or card than the staged build's (a cu129 lock, the card's own
      # compute capability), or where it failed.
      host_py jit-check "$node" --if-stale || true
      log "vLLM parses the node's configs"
      host_py with-engine-env "$node" "$root/venvs/engine/bin/python" "$src/deploy/profile.py" check-vllm "$node"
      preflight --gpu
      ;;
    warm)
      shell_env
      host_py warm "$node" ${LH_WARM:+--configs "$LH_WARM"} || log "a configuration did not start: see warm.json"
      preflight --gpu
      ;;
  esac
}

[[ ${BASH_SOURCE[0]} != "$0" ]] || main "$@"
