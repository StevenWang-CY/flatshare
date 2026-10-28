#!/usr/bin/env python3
"""profile.py: a GPU node's deployment from a hardware profile, deploy/profiles/NAME.toml.

    profile.py detect [--out host.json] [--peer HOST]
    profile.py render PROFILE [--host host.json] --out DIR
    profile.py check PROFILE [--against DIR]
    profile.py update PROFILE
    profile.py effective PROFILE
    profile.py check-vllm DIR [FLAG...]

`detect` writes the host's facts as JSON: the GPU as `nvidia-smi` shows it (name, memory, compute capability, driver,
PCIe link, MIG mode, power limit, NUMA node), the CPUs by NUMA node, RAM, /dev/shm, shmem huge pages, the kernel, a
container or VM, root and sudo, systemd's user manager and lingering, user namespaces, a CUDA toolkit, ninja and uv,
the directory of the driver's libcuda.so.1, and whether the linker searches it (the GPU units name it where it does
not), and with --peer the TCP round trip to a control-plane host. `render` writes a node's directory: deploy/templates
filled from the profile (engine.yaml, stt.yaml, the units, the make-room drop-in), the box's scripts and the mps
variant as they are, and node.json, the profile's values resolved with the core plan and its warnings. A profile's
"auto" values come from the host's facts by the rules below; a profile without them, as the frozen box's, needs no
host. `check` renders a profile into a scratch directory and compares it byte for byte with its repository copy
(the box's: deploy/phastform), printing each difference, and `update` writes the rendered files back into it after
a template changes (`make profiles`). When the box's own files change, a change to its units for one, `check` shows
the lines to carry into deploy/templates, and a file added to the box's directory has to be rendered, copied or named
in BOX_ONLY before the tests pass. `check-vllm` parses a rendered engine.yaml and stt.yaml with vLLM's own argument
parser and builds their engine configs, as `vllm serve` would, without starting an engine: run it with the engine
venv's Python on a GPU host.

A serving profile is a profile with a [serving] table (the box's: rtx-5060-ti-serve, on rtx-5060-ti): the configuration
the box serves, as against the frozen box that experiments change. It renders, beside the node's files, its engine
variant's drop-in (variants/make-room-serve) and serving.json, the effective configuration the box must run: every
setting of the engine's (model and tokenizer revisions, policy, step cap, pool, CPU tier, make-room's switches), the
vLLM version, the SHA-256 of the scheduler module and of the patches it is built from, and the calls the voice worker
admits. `effective` prints it. deploy/bin/serving-check holds the running engine, and the voice worker in serve mode,
to it: the engine writes what it runs as it starts (python/flatshare_vllm's serve module), since a production box
serves no /server_info. The step cap is the serving profile's own ([serving]'s max_num_batched_tokens and
long_prefill_token_threshold), which its variant passes over engine.yaml's, so that the node's files stay its base's:
on the box, the frozen box's, which every experiment changes.

The rules (docs/HARDWARE.md):
- R2, memory budgets: vLLM refuses a start whose free memory is below gpu_memory_utilization x total, with
  kv-cache-memory set too, so the box's budgets in GiB carry over: utilization = budget / the card's memory, rounded
  up to 0.01. Both vLLM servers fix their KV pools, so neither sizes its pool from the memory free at its start and
  the three servers start at once: each one's check passes in any order wherever starting the audio servers first
  passed, since the engine's worst case, the audio servers resident, is that order's, and the STT's, the engine
  resident, holds while the two budgets and the TTS's share (budgets.tts_gib) fit the card together, the engine
  within its budget; a server's start-up peak (profiling pass, graph capture) meets the margin they leave. A profile
  where they take more than 0.95 of it, the ceiling of one budget alone, is refused: vLLM's total is the card's memory
  less the driver's reserve. A node whose engine outgrows its budget (C''s 14B) starts it after the audio servers
  instead (text.engine_order).
- R3, the CPU tier: the box's ratio to the pool (20 GiB over 5 GiB), bounded by /dev/shm (less 2 GiB) and by the RAM
  available less what a single host keeps for its other processes, in whole GiB; below the pool's own size it is
  refused, since a tier no larger than the pool only mirrors it.
- R7, the core plan: the engine, the audio servers, the gateway, the voice plane, AIPerf and the callers take their
  shares of the host's CPUs in proportion 8:4:2:8:8:4 (34 CPUs give each its want), the engine on the GPU's NUMA node
  and the audio servers off it where there is another. 17 CPUs, half the wants, is the floor for measuring; below it
  the plan still holds every component, and warns.
- R8, the install path: the box's cu130 lock where the driver runs CUDA 13.0, natively (R580 and newer) or through
  the forward-compatibility libraries a data-center GPU on R535 or R570 can load (cuda-compat-13-0, installed);
  the cu129 lock (deploy/a100/requirements) on any other driver from R525; none below. JIT compiles with the host's
  toolkit of the runtime's CUDA version where it has one, else with the venv's CUDA 13.0 compiler (cu130) or with CUDA
  12.9's toolkit, which bootstrap.sh installs into ~/flatshare/cuda-12.9 (cu129: CUDA 12's wheels carry no nvcc).

A profile whose [profile] table names a base takes the base's values under its own, table by table: the A100's
second model (a100-40gb-qwen3-14b) is the core's profile with its own model, pool and directory.

Standard library only (Python 3.11 or newer), so that a bare host runs it before any venv exists.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEMPLATES = HERE / "templates"
PROFILES = HERE / "profiles"
BOX = HERE / "phastform"

RENDERED = (
    "engine.yaml",
    "stt.yaml",
    "systemd/flatshare.target",
    "systemd/flatshare-engine.service",
    "systemd/flatshare-stt.service",
    "systemd/flatshare-tts.service",
    "variants/make-room/flatshare-engine.conf",
)
"""The files a node's directory renders from deploy/templates."""
COPIED = (
    "variant.sh",
    "install.sh",
    "measure/flatshare-engine.conf",
    "systemd/flatshare-mps.service",
    "variants/mps/flatshare-engine.conf",
    "variants/mps/flatshare-stt.conf",
    "variants/mps/flatshare-tts.conf",
)
"""The files it takes from the box's directory as they are: they name nothing of a node's own."""
SERVED = (
    "variants/make-room-serve/flatshare-engine.conf",
    "serving.json",
)
"""The files a serving profile renders beside RENDERED: its variant's drop-in, from deploy/templates, and serving.json,
written by serving_config."""
BOX_ONLY = {
    "bootstrap.sh": "the box's own setup; another node's is its own",
    "requirements/*": "the engine's and TTS's locks, which a node's bootstrap installs from the source tree",
    "variants/README.md": "the variants' documentation",
    "smoke.sh": "the end-to-end check, which runs from the source tree beside deploy/bin's listener audit and asks "
    "the engine for its model",
    "systemd/flatshare-engine-proxy.*": (
        "the sglang variant's socket in front of its TCP port, measured on the box only"
    ),
    "variants/agents-only/*": (
        "the dedicated background card's engine, its pool grown into the speech servers' memory on the box's card, "
        "for the resource gate's two RTX 5060 Ti cards only"
    ),
    "variants/make-room-v03/*": "W5's control, measured on the box only",
    "variants/pr54530/*": "E2 arm 7a's variant, measured on the box only",
    "variants/sglang/*": "E2 arm 7b's variant, measured on the box only",
}
"""The box's files that a node's directory does not take, and why. Every file of the box's directory is
rendered, copied or here (a test holds it), so that a file another change adds to the box, a unit or a
drop-in, is not silently left out of other nodes: it fails the test until it is placed."""

VLLM_PACKAGE = REPO / "python" / "flatshare_vllm" / "src" / "flatshare_vllm"
SERVE_MODULE = VLLM_PACKAGE / "serve" / "flatshare_make_room_serve.py"
"""The scheduler class the serving variant loads: make-room's, which writes the configuration the engine runs."""
SCHEDULER_MODULE = "flatshare_make_room_scheduler"
"""The module build.py writes for make-room, which the serve module extends."""

PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+\.[a-z0-9_]+)\}\}")
ALONE = re.compile(r"\{\{([a-z0-9_]+\.[a-z0-9_]+)\}\}\n?")

CORES = (("engine", 8), ("audio", 4), ("gateway", 2), ("voice", 8), ("aiperf", 8), ("callers", 4))
"""R7's components and the CPUs each wants. Half of every want is its minimum."""
CORE_FLOOR = sum(want for _, want in CORES) // 2

GIB = 1 << 30
VENV_CUDA = "%h/flatshare/venvs/engine/lib/python3.12/site-packages/nvidia/cu13"
"""The CUDA 13 toolkit the engine venv carries (nvidia-cuda-nvcc, pinned by the engine lock), Gray's CUDA_HOME."""
HOME_CUDA = "%h/flatshare/cuda-12.9"
"""Where bootstrap.sh installs CUDA 12.9.1's toolkit for the cu129 lock on a host without one."""
COMPAT = "/usr/local/cuda-13.0/compat"
"""Where cuda-compat-13-0 puts CUDA 13.0's forward-compatibility libraries."""
COMPAT_BRANCHES = (535, 570)
"""The driver branches cuda-compat-13-0 runs CUDA 13.0 on, on data-center GPUs only (NOTES §3)."""
LIBCUDA = "libcuda.so.1"
"""The driver's CUDA library, by the soname the CUDA runtime loads; where it cannot, torch finds "no NVIDIA driver"."""
DRIVER_DIRS = ("/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu")
"""Where a host keeps the driver's libraries, for facts that find them neither in the linker's cache nor on the
operator's LD_LIBRARY_PATH: Colab's container's directory, and Debian's multiarch ones, where the driver's packages
put them."""
LINKER_DIRS = (
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/lib/aarch64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
)
"""Debian's multiarch library directories, which its dynamic linker searches whether or not its cache lists them
(`ld.so --help`'s system search path)."""
LIBC6 = {"x86_64": "x86-64", "aarch64": "AArch64"}
"""ldconfig -p's mark of a machine's own libraries ("libc6,x86-64"), by its architecture, beside a 32-bit one's."""


class ProfileError(Exception):
    pass


# The host's facts.


def _run(cmd: list[str], timeout: float = 10) -> subprocess.CompletedProcess[str] | None:
    """`cmd`, or None when it is not installed or does not finish."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def parse_cpulist(text: str) -> list[int]:
    """The CPUs of a Linux cpulist, such as "0-7,16-23"."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def cpulist(cpus: list[int]) -> str:
    """A cpulist of `cpus`, as systemd's CPUAffinity takes it: "0-7,16-23"."""
    parts, run = [], []
    for c in sorted(set(cpus)):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
        run = [c]
    if run:
        parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
    return ",".join(parts)


GPU_QUERY = (
    "index,uuid,name,memory.total,compute_cap,driver_version,pcie.link.gen.current,pcie.link.width.current,"
    "mig.mode.current,power.limit,pci.bus_id,ecc.mode.current,compute_mode"
)


def _mig_devices(uuid: str) -> list[dict]:
    """The MIG devices of the GPU ``uuid``, from nvidia-smi -L: each one's profile and UUID."""
    done = _run(["nvidia-smi", "-L"])
    devices: list[dict] = []
    ours = False
    for line in done.stdout.splitlines() if done is not None and done.returncode == 0 else []:
        if line.startswith("GPU "):
            ours = f"(UUID: {uuid})" in line
        elif ours and (m := re.match(r"\s+MIG (\S+)\s+Device\s+\d+: \(UUID: (\S+)\)", line)):
            devices.append({"profile": m.group(1), "uuid": m.group(2)})
    return devices


def _gpu(sysfs: Path) -> dict | None:
    """The GPU's facts: on a host with several, the one FLATSHARE_GPU_ID names (nvidia-smi's index or the GPU's UUID);
    with one, it. ``chosen`` says whether one is: several and none named leave GPU 0's facts and chosen false, which the
    preflight refuses."""
    done = _run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"])
    if done is None or done.returncode != 0 or not done.stdout.strip():
        return None
    rows = [[x.strip() for x in line.split(",")] for line in done.stdout.strip().splitlines()]
    wanted = os.environ.get("FLATSHARE_GPU_ID", "").strip() or None
    if wanted:
        matches = [row for row in rows if wanted in (row[0], row[1])]
    else:
        matches = rows if len(rows) == 1 else []
    index, uuid, name, memory, cap, driver, gen, width, mig, power, bus, ecc, mode = (
        matches[0] if matches else rows[0]
    )
    number = lambda x: float(x) if re.fullmatch(r"[0-9.]+", x) else None  # noqa: E731
    # nvidia-smi gives the bus id with an 8-digit domain ("00000000:3B:00.0"), sysfs with a 4-digit one.
    domain, _, rest = bus.partition(":")
    node = _read(sysfs / "bus" / "pci" / "devices" / f"{domain[-4:]}:{rest}".lower() / "numa_node")
    return {
        "name": name,
        "count": len(rows),
        "index": int(index) if index.isdigit() else None,
        "uuid": uuid,
        "chosen": bool(matches),
        "wanted": wanted,
        "all": [f"{row[0]}: {row[2]} ({row[1]})" for row in rows],
        "memory_mib": int(number(memory) or 0),
        "compute_cap": cap,
        "driver": driver,
        "pcie_gen": int(number(gen) or 0),
        "pcie_width": int(number(width) or 0),
        "mig": mig,
        "mig_devices": _mig_devices(uuid) if mig.lower() == "enabled" else [],
        "power_limit_w": number(power),
        "ecc": ecc,
        # Default lets the STT, the TTS and the engine each hold a CUDA context; Exclusive_Process one at a time.
        "compute_mode": mode,
        "bus_id": bus,
        "numa_node": int(node) if node is not None and node.strip().lstrip("-").isdigit() else None,
    }


def gpu_select(gpu: dict) -> str:
    """The GPU units' lines that run them on the lent GPU alone, by UUID (on its MIG slice where MIG is on and
    allowed), and tell gpu-guard which GPU's processes to read; none until one is chosen."""
    if not gpu.get("chosen") or not gpu.get("uuid"):
        return ""
    slices = gpu.get("mig_devices") or []
    visible = slices[0]["uuid"] if len(slices) == 1 else gpu["uuid"]
    return (
        f"# GPU {gpu.get('index')} of {gpu.get('count', 1)}, the one this host lends "
        "(bootstrap.sh's FLATSHARE_GPU_ID, else "
        "its only one): CUDA sees it alone, by UUID,\n# and gpu-guard reads its processes alone.\n"
        "Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID\n"
        f"Environment=CUDA_VISIBLE_DEVICES={visible}\n"
        f"Environment=FLATSHARE_GPU_ID={gpu['uuid']}"
    )


def _cpus(sysfs: Path) -> dict[str, list[int]]:
    """The CPUs this process may use, by NUMA node."""
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    nodes: dict[str, list[int]] = {}
    for path in sorted((sysfs / "devices" / "system" / "node").glob("node[0-9]*")):
        cpus = parse_cpulist(_read(path / "cpulist") or "")
        if allowed is not None and not os.environ.get("FLATSHARE_SYSFS"):
            cpus = [c for c in cpus if c in allowed]
        if cpus:
            nodes[path.name.removeprefix("node")] = cpus
    if not nodes:
        nodes["0"] = sorted(allowed) if allowed else list(range(os.cpu_count() or 1))
    return nodes


def _meminfo(text: str | None) -> dict[str, int]:
    """meminfo's fields in bytes."""
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"(?:Node \d+ )?(\w+):\s+(\d+)(?: kB)?", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * 1024
    return out


def _memory(sysfs: Path, procfs: Path) -> dict:
    total = _meminfo(_read(procfs / "meminfo"))
    nodes = {}
    for path in sorted((sysfs / "devices" / "system" / "node").glob("node[0-9]*")):
        info = _meminfo(_read(path / "meminfo"))
        if info:
            nodes[path.name.removeprefix("node")] = {
                "total_bytes": info.get("MemTotal", 0),
                "free_bytes": info.get("MemFree", 0),
            }
    return {
        "total_bytes": total.get("MemTotal", 0),
        "available_bytes": total.get("MemAvailable", 0),
        "nodes": nodes,
    }


def _shm(path: Path) -> dict | None:
    try:
        s = os.statvfs(path)
    except OSError:
        return None
    return {
        "path": str(path),
        "total_bytes": s.f_blocks * s.f_frsize,
        "free_bytes": s.f_bavail * s.f_frsize,
    }


def _bracketed(text: str | None) -> str | None:
    m = re.search(r"\[(\w+)\]", text or "")
    return m.group(1) if m else None


def _cuda_toolkits(local: Path) -> list[dict]:
    """Each CUDA toolkit whose nvcc is on the PATH or under `local` (/usr/local), with its version, newest first."""
    candidates = [shutil.which("nvcc"), *sorted(str(p) for p in local.glob("cuda*/bin/nvcc"))]
    found: dict[str, tuple[int, int]] = {}
    for nvcc in filter(None, candidates):
        done = _run([nvcc, "--version"])
        m = re.search(r"release (\d+)\.(\d+)", done.stdout if done else "")
        if m:
            found[str(Path(nvcc).resolve().parent.parent)] = (int(m.group(1)), int(m.group(2)))
    return [
        {"version": f"{v[0]}.{v[1]}", "path": path}
        for path, v in sorted(found.items(), key=lambda item: item[1], reverse=True)
    ]


def _driver_libs(root: Path) -> dict | None:
    """Where the driver's libcuda.so.1 is, and whether the dynamic linker finds it there by itself, as the GPU units
    need it to: their environment has no LD_LIBRARY_PATH of the operator's. Listed in the linker's cache (ldconfig
    -p), it does; else on the operator's LD_LIBRARY_PATH (a container's shell, as Colab's, reaching a directory that
    the cache does not list), cuda-compat's directories aside, whose libcuda.so.1 is not the driver's; else in
    DRIVER_DIRS. Found on either, the linker finds it only in a directory it searches anyway (LINKER_DIRS). None where
    none of them holds it: no driver, or a container started without the NVIDIA runtime."""
    ldconfig = shutil.which("ldconfig") or next(
        (str(p) for p in (root / "sbin" / "ldconfig", root / "usr/sbin/ldconfig") if p.is_file()),
        None,
    )
    done = _run([ldconfig, "-p"]) if ldconfig else None
    mark = LIBC6.get(os.uname().machine)
    for line in done.stdout.splitlines() if done is not None and done.returncode == 0 else []:
        m = re.fullmatch(r"\s*libcuda\.so\.1 \(([^)]*)\) => (.+)", line)
        if m and (mark is None or mark in m.group(1)) and Path(m.group(2)).exists():
            return {
                "path": str(Path(m.group(2)).parent),
                "linker": True,
                "via": "the linker's cache",
            }
    # glibc splits LD_LIBRARY_PATH at colons and semicolons; a relative entry is the working directory's.
    path = [
        os.path.normpath(p)
        for p in re.split(r"[:;]", os.environ.get("LD_LIBRARY_PATH", ""))
        if p.startswith("/")
    ]
    places = [(Path(p), p, "LD_LIBRARY_PATH") for p in path if Path(p).name != "compat"]
    places += [(root / d.lstrip("/"), d, "a known location") for d in DRIVER_DIRS]
    for directory, name, via in places:
        if (directory / LIBCUDA).exists():
            return {"path": str(directory), "linker": name in LINKER_DIRS, "via": via}
    return None


def _userns() -> bool | None:
    """Whether an unprivileged user and network namespace can make a link, as voicenet-holder does."""
    if shutil.which("unshare") is None or shutil.which("ip") is None:
        return None
    done = _run(
        [
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--",
            "sh",
            "-c",
            "ip link add flatshare0 type dummy && ip link del flatshare0",
        ]
    )
    return done is not None and done.returncode == 0


def _rtt_ms(host: str, port: int = 22, tries: int = 5) -> list[float] | None:
    """TCP connect times to host:port, in ms."""
    times = []
    for _ in range(tries):
        start = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=3):
                times.append((time.perf_counter() - start) * 1e3)
        except OSError:
            return None
    return times


def hub_cache() -> str:
    """The Hugging Face hub cache, as huggingface_hub resolves it: HF_HUB_CACHE (or the older HUGGINGFACE_HUB_CACHE),
    else HF_HOME/hub, else XDG_CACHE_HOME/huggingface/hub, else ~/.cache/huggingface/hub."""
    if cache := os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return str(Path(cache).expanduser())
    cache_home = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return str(
        Path(os.environ.get("HF_HOME") or Path(cache_home) / "huggingface").expanduser() / "hub"
    )


def detect(peer: str | None = None) -> dict:
    """The host's facts. FLATSHARE_ROOTFS, FLATSHARE_SYSFS, FLATSHARE_PROCFS and FLATSHARE_SHM put /, /sys, /proc and
    /dev/shm elsewhere (for tests)."""
    root = Path(os.environ.get("FLATSHARE_ROOTFS", "/"))
    sysfs = Path(os.environ.get("FLATSHARE_SYSFS", "/sys"))
    procfs = Path(os.environ.get("FLATSHARE_PROCFS", "/proc"))
    systemd = _run(["systemctl", "--user", "is-system-running"])
    linger = _run(
        ["loginctl", "show-user", os.environ.get("USER", ""), "--property=Linger", "--value"]
    )
    cgroup = _read(procfs / "1" / "cgroup") or ""
    virt = _run(["systemd-detect-virt"])
    sudo = _run(["sudo", "-n", "true"])
    uname = os.uname()
    facts = {
        "hostname": socket.gethostname().split(".")[0],
        "kernel": uname.release,
        "arch": uname.machine,
        "gpu": _gpu(sysfs),
        "cpus": _cpus(sysfs),
        "memory": _memory(sysfs, procfs),
        "shm": _shm(Path(os.environ.get("FLATSHARE_SHM", "/dev/shm"))),
        "thp_shmem": _bracketed(
            _read(sysfs / "kernel" / "mm" / "transparent_hugepage" / "shmem_enabled")
        ),
        "container": (root / ".dockerenv").exists()
        or (root / "run" / ".containerenv").exists()
        or bool(re.search(r"docker|containerd|kubepods|lxc", cgroup)),
        "virtualization": virt.stdout.strip() if virt is not None else None,
        "root": os.geteuid() == 0,
        "sudo": sudo is not None and sudo.returncode == 0,
        "systemd_user": systemd is not None and systemd.stdout.strip() in ("running", "degraded"),
        "linger": None if linger is None else linger.stdout.strip() == "yes",
        "userns": _userns(),
        "cuda_toolkits": (toolkits := _cuda_toolkits(root / "usr" / "local")),
        "cuda_toolkit": toolkits[0] if toolkits else None,
        "cuda_compat": str(compat)
        if (compat := root / COMPAT.lstrip("/")).joinpath(LIBCUDA).exists()
        else None,
        # The units' environment has no LD_LIBRARY_PATH of the operator's: they name this directory where the linker
        # does not search it (Colab's container's /usr/lib64-nvidia).
        "driver_libs": _driver_libs(root),
        # Triton builds its launchers with a C compiler, and nvcc needs a host C++ compiler.
        "cc": shutil.which("cc") or shutil.which("gcc"),
        "cxx": shutil.which("c++") or shutil.which("g++"),
        "ninja": shutil.which("ninja") is not None,
        "uv": shutil.which("uv")
        or (
            str(Path.home() / ".local/bin/uv") if (Path.home() / ".local/bin/uv").exists() else None
        ),
        "python": sys.version.split()[0],
        # Where bootstrap.sh downloads the models and the units, offline, read them (bootstrap.sh resolves it once).
        "hf_hub_cache": hub_cache(),
    }
    if peer:
        facts["peer"] = {"host": peer, "rtt_ms": _rtt_ms(peer)}
    return facts


# The rules.


CARD_SHARE = 0.95
"""R2's ceiling: the share of a card's memory, as nvidia-smi shows it, that one budget, or the two vLLM servers'
budgets and the TTS's share together, may take. vLLM's total is the card's memory less the driver's reserve: on the
box 15,852 of nvidia-smi's 16,311 MiB."""


def utilization(budget_gib: float, memory_mib: int) -> str:
    """R2: `budget_gib` of a card of `memory_mib`, rounded up to 0.01."""
    u = math.ceil(budget_gib / (memory_mib / 1024) * 100 - 1e-9) / 100
    if u > CARD_SHARE:
        raise ProfileError(
            f"{budget_gib} GiB is {u:.2f} of a {memory_mib} MiB card: too small a card"
        )
    return f"{u:.2f}"


def card_share(profile: dict, values: dict[str, str], memory_mib: int | str | None) -> str:
    """R2: the share of a card of `memory_mib` that the engine's and the STT's budgets (their resolved
    gpu_memory_utilization) and the TTS's share take together. The three servers start at once, so the last to
    check its free memory may find the other two resident: above CARD_SHARE, the profile is refused."""
    name = values["profile.name"]
    tts_gib = profile.get("budgets", {}).get("tts_gib")
    missing = [
        key
        for key, value in (("budgets.tts_gib", tts_gib), ("profile.memory_mib", memory_mib))
        if value is None
    ]
    if missing:
        raise ProfileError(
            f"{name} names no {' and no '.join(missing)}: R2 checks the budgets and the TTS's share against the card"
        )
    engine = float(values["engine.gpu_memory_utilization"])
    stt = float(values["stt.gpu_memory_utilization"])
    share = engine + stt + float(tts_gib) / (int(memory_mib) / 1024)
    if share > CARD_SHARE + 1e-9:
        raise ProfileError(
            f"refused: the engine's {engine:.2f} and the STT's {stt:.2f} of the card and the TTS's {tts_gib:g} GiB are "
            f"{share:.2f} of the {memory_mib} MiB card, above {CARD_SHARE}: the servers start at once, so the last to "
            "start may find the others resident and too little free for vLLM's check"
        )
    return f"{share:.2f}"


def install_path(host: dict) -> tuple[str, str]:
    """R8: the engine venv's CUDA build (cu130, cu130-compat or cu129) by the host's driver, and why."""
    gpu = host.get("gpu") or {}
    driver = gpu.get("driver") or ""
    m = re.match(r"(\d+)\.", driver)
    if not m:
        return "cu130", "no driver seen yet: the box's lock until bootstrap.sh gpu reads the driver"
    major = int(m.group(1))
    if major >= 580:
        return "cu130", f"driver {driver} runs CUDA 13.0: the box's lock"
    if major < 525:
        raise ProfileError(
            f"refused: driver {driver} is older than R525, the oldest that runs CUDA 12; vLLM 0.32.0's builds are "
            "for CUDA 12.9 and 13.0"
        )
    if compat_eligible(host) and host.get("cuda_compat"):
        return (
            "cu130-compat",
            f"driver {driver} on a data-center GPU runs CUDA 13.0 through the forward-compatibility libraries "
            f"in {host['cuda_compat']}: the box's lock",
        )
    return (
        "cu129",
        f"driver {driver} runs CUDA 12 but not 13.0: the cu129 lock (vLLM 0.32.0+cu129, torch 2.13.0+cu129)",
    )


def compat_eligible(host: dict) -> bool:
    """Whether the host's driver and card can run CUDA 13.0 through cuda-compat-13-0: R535 or R570 on a data-center
    GPU (NVIDIA's table; GeForce and workstation cards get error 803)."""
    gpu = host.get("gpu") or {}
    m = re.match(r"(\d+)\.", gpu.get("driver") or "")
    consumer = re.search(r"GeForce|TITAN|Quadro|RTX", gpu.get("name") or "")
    return bool(m) and int(m.group(1)) in COMPAT_BRANCHES and not consumer


def compiler(host: dict, install: str) -> tuple[str, str]:
    """R8: the CUDA_HOME FlashInfer's JIT compiles with, and the middle of the comment sentence that says so."""
    want = "12.9" if install == "cu129" else "13.0"
    toolkits = host.get("cuda_toolkits")
    if toolkits is None:
        toolkits = [host["cuda_toolkit"]] if host.get("cuda_toolkit") else []
    for toolkit in toolkits:
        if toolkit["version"] == want:
            return toolkit["path"], f"the host's CUDA {want} toolkit at {toolkit['path']} compiles"
    if install == "cu129":
        return HOME_CUDA, (
            "the host has no CUDA 12.9 toolkit and CUDA 12's wheels carry no nvcc, so the engine\n# JIT-compiles "
            "with CUDA 12.9.1's toolkit, which bootstrap.sh installs into ~/flatshare/cuda-12.9"
        )
    return VENV_CUDA, (
        "the host has no CUDA 13 toolkit, so the engine JIT-compiles with the CUDA 13 compiler its\n# venv "
        "carries (nvidia-cuda-nvcc, pinned by the engine lock), as on Gray, and links with the venv's runtime and a "
        "driver\n# stub, which bootstrap.sh puts in cache/cuda_lib"
    )


def library_path(driver: str | None, compat: str | None, libs: str | None) -> str:
    """A GPU unit's LD_LIBRARY_PATH line and the comment that says why, or nothing: cuda-compat-13-0's libraries where
    the install path runs on them (R8), first, so that their libcuda.so.1 is the one loaded, then the directory of the
    driver's own libraries where the dynamic linker does not search it (``libs``), which the unit's environment would
    not otherwise reach, having no LD_LIBRARY_PATH of the operator's (Colab's container: /usr/lib64-nvidia)."""
    if compat:
        why = (
            f"# Driver {driver}: CUDA 13.0 through the forward-compatibility libraries of cuda-compat-13-0,\n"
            "# which a data-center GPU can load"
            + (
                ", ahead of the driver's own (libcuda.so.1 and the rest), which are in a\n"
                "# directory the dynamic linker does not search (not in its cache, as in some containers)"
                if libs
                else ""
            )
            + "."
        )
    elif libs:
        why = (
            "# The driver's own libraries (libcuda.so.1 and the rest) are in a directory the dynamic linker does not "
            "search\n# (not in its cache, as in some containers): the unit names it."
        )
    else:
        return ""
    return why + "\nEnvironment=LD_LIBRARY_PATH=" + ":".join(p for p in (compat, libs) if p)


def cpu_tier(pool_bytes: int, ratio: float, host: dict, reserve_gib: float) -> tuple[int, str]:
    """R3: the CPU tier's bytes, in whole GiB, and what bounds it."""
    want = int(ratio * pool_bytes)
    bounds = [(want, f"{ratio:g} times the GPU pool, as on the box")]
    shm = host.get("shm")
    if shm:
        bounds.append((shm["free_bytes"] - 2 * GIB, "what /dev/shm holds, less 2 GiB"))
    available = host.get("memory", {}).get("available_bytes")
    if available:
        bounds.append(
            (
                available - int(reserve_gib * GIB),
                f"the RAM available less {reserve_gib:g} GiB for the host",
            )
        )
    got, why = min(bounds)
    got = got // GIB * GIB
    if got < pool_bytes:
        raise ProfileError(
            f"refused: a CPU tier of {got / GIB:.0f} GiB ({why}) is below the pool's {pool_bytes / GIB:g} GiB, "
            "so it would only mirror the pool; give /dev/shm and RAM room (docker: --shm-size or --ipc=host)"
        )
    if got < want:
        why = f"{got / pool_bytes:.1f} times the GPU pool, bounded by {why}"
    return got, why


def core_plan(
    nodes: dict[str, list[int]], gpu_node: int | None
) -> tuple[dict[str, str], list[str]]:
    """R7: each component's CPUs and the engine's and the audio servers' NUMA nodes, and the warnings."""
    all_cpus = [c for n in sorted(nodes, key=int) for c in nodes[n]]
    total = len(all_cpus)
    wants = dict(CORES)
    full = sum(wants.values())
    if total >= full:
        shares = dict(wants)
    else:
        # Hamilton's method: each its whole quota, the rest by the largest remainders, and at least one each.
        quotas = {k: total * w / full for k, w in CORES}
        shares = {k: max(1, math.floor(q)) for k, q in quotas.items()}
        for k in sorted(quotas, key=lambda k: quotas[k] - math.floor(quotas[k]), reverse=True):
            if sum(shares.values()) >= total:
                break
            if shares[k] < wants[k]:
                shares[k] += 1
    warnings = []
    if total < CORE_FLOOR:
        warnings.append(
            f"{total} logical CPUs, below the single-host floor of {CORE_FLOOR}: the components share fewer CPUs "
            "than their minimums, so this host checks the mechanics, not the numbers"
        )
    gpu = (
        str(gpu_node)
        if gpu_node is not None and str(gpu_node) in nodes
        else sorted(nodes, key=int)[0]
    )
    others = [n for n in sorted(nodes, key=int) if n != gpu]
    free = {n: list(nodes[n]) for n in nodes}

    def take(count: int, prefer: list[str]) -> list[int]:
        got: list[int] = []
        for n in prefer + [m for m in sorted(nodes, key=int) if m not in prefer]:
            while free[n] and len(got) < count:
                got.append(free[n].pop(0))
        if len(got) < count:  # fewer CPUs than shares: wrap around and share
            got += all_cpus[: count - len(got)]
        return got

    plan, placed = {}, {}
    placed["engine"] = take(shares["engine"], [gpu])
    placed["audio"] = take(shares["audio"], others or [gpu])
    for k in ("gateway", "voice", "aiperf", "callers"):
        placed[k] = take(shares[k], [])
    node_of = {c: n for n, cpus in nodes.items() for c in cpus}
    for k, cpus in placed.items():
        plan[k] = cpulist(cpus)
    plan["engine_numa"] = gpu
    plan["audio_numa"] = node_of[placed["audio"][0]]
    spare = [c for c in all_cpus if not any(c in v for v in placed.values())]
    if spare:
        plan["spare"] = cpulist(spare)
    return plan, warnings


# Profiles.


def load(profile: str) -> dict:
    """A profile by name (deploy/profiles/NAME.toml) or path. One whose [profile] table names a base is that
    profile with its own values over the base's, table by table: another model on the same card, say."""
    path = Path(profile)
    if not path.suffix:
        path = PROFILES / f"{profile}.toml"
    try:
        data = tomllib.loads(path.read_text())
    except OSError as e:
        raise ProfileError(f"no profile {profile}: {e}") from None
    base = data.get("profile", {}).pop("base", None)
    if base is None:
        return data
    merged = load(base)
    for section, table in data.items():
        merged[section] = merged.get(section, {}) | table if isinstance(table, dict) else table
    return merged


def flatten(profile: dict) -> dict[str, str]:
    """The profile's values by section.key, as text: a boolean as TOML writes it (true, false)."""
    return {
        f"{section}.{key}": str(value).lower() if isinstance(value, bool) else str(value)
        for section, table in profile.items()
        if isinstance(table, dict)
        for key, value in table.items()
        if not isinstance(value, dict)
    }


def resolve(profile: dict, host: dict | None) -> tuple[dict[str, str], dict | None, list[str]]:
    """The profile's values with every "auto" one derived from the host's facts, the core plan, and the warnings."""
    values = flatten(profile)
    autos = sorted(k for k, v in values.items() if v == "auto")
    if not autos:
        values["derived.card_share"] = card_share(profile, values, values.get("profile.memory_mib"))
        return values, None, []
    name = values["profile.name"]
    if host is None:
        raise ProfileError(
            f"{name} derives {', '.join(autos)} from the host: pass --host (profile.py detect)"
        )
    gpu = host.get("gpu")
    early = []
    if not gpu:
        # Staging can come before the card is visible: the profile's card stands in until bootstrap.sh gpu.
        if not values.get("profile.memory_mib"):
            raise ProfileError(
                f"{name}: the host shows no GPU (nvidia-smi) and the profile names no card"
            )
        gpu = {
            "name": values["profile.gpu"],
            "memory_mib": int(values["profile.memory_mib"]),
            "compute_cap": values.get("profile.compute_cap"),
            "numa_node": None,
        }
        early.append(
            "no GPU visible: rendered for the profile's card (its memory_mib and compute_cap) and the box's lock; "
            "bootstrap.sh gpu renders again from the card's own facts"
        )
    budgets = profile.get("budgets", {})
    derived: dict[str, str] = {}
    if values.get("profile.node") == "auto":
        values["profile.node"] = host.get("hostname") or "the GPU host"
    if values.get("engine.gpu_memory_utilization") == "auto":
        values["engine.gpu_memory_utilization"] = utilization(
            budgets["engine_gib"], gpu["memory_mib"]
        )
    if values.get("stt.gpu_memory_utilization") == "auto":
        values["stt.gpu_memory_utilization"] = utilization(budgets["stt_gib"], gpu["memory_mib"])
    derived["card_share"] = card_share(profile, values, gpu["memory_mib"])
    derived["total_gib"] = f"{gpu['memory_mib'] / 1024:.1f}"
    derived["engine_budget"] = f"{budgets.get('engine_gib', 0):.2f} GiB"
    pool = int(values["engine.kv_cache_memory"])
    if values.get("engine.cpu_bytes_to_use") == "auto":
        tier, why = cpu_tier(pool, budgets["cpu_tier_ratio"], host, budgets["host_reserve_gib"])
        values["engine.cpu_bytes_to_use"] = str(tier)
        derived["cpu_tier_gib"] = f"{tier // GIB}"
        derived["cpu_tier_note"] = why
    install, derived["install_note"] = install_path(host)
    derived["install"] = install
    cuda, derived["cuda_note"] = compiler(
        host, install
    )  # the middle of a sentence: text.cuda ends it
    system = not cuda.startswith("%h/")
    compat = host["cuda_compat"] if install == "cu130-compat" else None
    libs = host.get("driver_libs") or {}
    outside = libs.get("path") if not libs.get("linker") else None
    unit_values = {
        "units.cuda_home": cuda,
        "units.engine_path": f"%h/flatshare/venvs/engine/bin:{cuda}/bin:/usr/local/bin:/usr/bin:/bin:%h/.local/bin",
        "units.stt_path": f"%h/flatshare/venvs/engine/bin:{cuda}/bin:/usr/local/bin:/usr/bin:/bin",
        "units.stt_cuda_home": "" if system else f"Environment=CUDA_HOME={cuda}",
        # FlashInfer's JIT links -lcudart and -lcuda from $CUDA_HOME/lib64 and lib64/stubs, which the venv's CUDA 13
        # lacks (lib/ holds libcudart.so.13 only): bootstrap.sh links them into cache/cuda_lib.
        "units.cuda_ldflags": "Environment=FLASHINFER_EXTRA_LDFLAGS=-L%h/flatshare/cache/cuda_lib"
        if cuda == VENV_CUDA
        else "",
        "units.gpu_select": gpu_select(gpu),
        "units.hf_hub_cache": f'Environment="HF_HUB_CACHE={host["hf_hub_cache"]}"'
        if host.get("hf_hub_cache")
        else "",
        # The vLLM units' library path, cuda-compat's libraries first, and the TTS's, Kokoro's own CUDA 12.8 build,
        # which takes the driver's alone.
        "units.library_path": library_path(gpu.get("driver"), compat, outside),
        "units.tts_library_path": library_path(gpu.get("driver"), None, outside),
    }
    plan, warnings = core_plan({k: v for k, v in host["cpus"].items()}, gpu.get("numa_node"))
    warnings = early + warnings
    unit_values |= {
        "units.engine_cpus": plan["engine"],
        "units.engine_numa": plan["engine_numa"],
        "units.audio_cpus": plan["audio"],
        "units.audio_numa": plan["audio_numa"],
    }
    for k, v in unit_values.items():
        if values.get(k) == "auto":
            values[k] = v
    several = len(host["cpus"]) > 1
    derived["engine_numa_note"] = (
        f'NUMA node {plan["engine_numa"]} is local to the GPU (detected). "preferred" spills to another node when '
        "it is short."
        if several
        else f"One NUMA node; the engine's CPUs are {plan['engine']} (the core plan in node.json)."
    )
    derived["audio_numa_note"] = (
        f"NUMA node {plan['audio_numa']} (CPUs {plan['audio']}) for the audio servers"
        + (
            "; the GPU-local node is the engine's."
            if plan["audio_numa"] != plan["engine_numa"]
            else "."
        )
    )
    values |= {f"derived.{k}": v for k, v in derived.items()}
    left = sorted(k for k, v in values.items() if v == "auto")
    if left:
        raise ProfileError(f"{name}: no rule derives {', '.join(left)}")
    return values, plan, warnings


def fill(text: str, values: dict[str, str]) -> str:
    """`text` with every {{section.key}} replaced; a line that holds only a placeholder of an empty value goes."""
    try:
        lines = [
            line
            for line in text.splitlines(keepends=True)
            if not ((m := ALONE.fullmatch(line)) and values[m.group(1)] == "")
        ]
        text = "".join(lines)
        for _ in range(4):  # a value may hold placeholders of its own
            text, n = PLACEHOLDER.subn(lambda m: values[m.group(1)], text)
            if n == 0:
                break
    except KeyError as e:
        raise ProfileError(f"no value for the placeholder {e.args[0]}") from None
    if PLACEHOLDER.search(text):
        raise ProfileError(f"placeholders left unfilled: {sorted(set(PLACEHOLDER.findall(text)))}")
    return text


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_room_patches() -> tuple[str, ...]:
    """make-room's patches in the order build.py applies them, from build.py's own table."""
    spec = importlib.util.spec_from_file_location("flatshare_vllm_build", VLLM_PACKAGE / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build.VARIANTS["make-room"][0]


SERVING_KEYS = {
    "variant": str,
    "vllm_version": str,
    "max_preemptions_per_step": int,
    "budget_cap": bool,
    "budget_cap_tokens": int,
    "reserve_tokens": int,
    "max_num_batched_tokens": int,
    "long_prefill_token_threshold": int,
    "block_size": int,
    "pool_tokens": int,
    "max_num_seqs": int,
    "async_scheduling": bool,
    "scheduler_module_sha256": str,
    "max_calls": int,
    "audio": str,
}
"""A [serving] table's keys and their types: each is required, and no other is taken."""
AUDIO = ("microphone", "synthetic")
"""The voice worker's audio configurations (python/flatshare_voice's worker.py)."""


def serving_config(profile: dict, values: dict[str, str]) -> dict:
    """serving.json: the configuration a serving profile's engine must run, each setting as the engine's own
    configuration names it, as deploy/bin/serving-check compares it with what the engine writes at start; the SHA-256
    of make-room's patches, which build.py makes its module from; and the voice worker's admission. The step cap is
    the [serving] table's, which the variant passes over engine.yaml's, not the base profile's [engine]."""
    serve = profile["serving"]
    name = values["profile.name"]
    for key, kind in SERVING_KEYS.items():
        if not isinstance(serve.get(key), kind) or (kind is int and isinstance(serve[key], bool)):
            raise ProfileError(f"{name}: [serving] needs {key}, a {kind.__name__}")
    if extra := sorted(set(serve) - set(SERVING_KEYS)):
        raise ProfileError(f"{name}: [serving] has no key {', '.join(extra)}")
    if serve["variant"] != "make-room-serve":
        raise ProfileError(
            f"{name}: the serving variant is make-room-serve, not {serve['variant']}"
        )
    step_cap, threshold = serve["max_num_batched_tokens"], serve["long_prefill_token_threshold"]
    if step_cap < 1:
        raise ProfileError(f"{name}: max_num_batched_tokens must be 1 or more")
    if step_cap < serve["max_num_seqs"]:
        # vLLM 0.32.0's SchedulerConfig refuses the start below it.
        raise ProfileError(
            f"{name}: max_num_batched_tokens ({step_cap}) must be at least max_num_seqs ({serve['max_num_seqs']})"
        )
    if not 0 <= threshold <= step_cap:
        raise ProfileError(
            f"{name}: long_prefill_token_threshold must be from 0 to max_num_batched_tokens ({step_cap}), "
            f"not {threshold}"
        )
    if serve["pool_tokens"] % serve["block_size"]:
        raise ProfileError(
            f"{name}: pool_tokens is no whole number of {serve['block_size']}-token blocks"
        )
    if serve["max_calls"] < 1:
        raise ProfileError(f"{name}: max_calls must be 1 or more")
    if serve["audio"] not in AUDIO:
        raise ProfileError(f"{name}: audio must be {' or '.join(AUDIO)}, not {serve['audio']!r}")
    return {
        "profile": name,
        "node": values["profile.node"],
        "variant": serve["variant"],
        "engine": {
            "vllm_version": serve["vllm_version"],
            "scheduler_cls": f"{SERVE_MODULE.stem}.AsyncScheduler",
            "scheduler_module": {
                "name": SCHEDULER_MODULE,
                "sha256": serve["scheduler_module_sha256"],
            },
            "serve_module_sha256": sha256(SERVE_MODULE),
            "model": {
                "model": values["model.name"],
                "revision": values["model.revision"],
                # vLLM's default: the model's own repository and revision.
                "tokenizer": values["model.name"],
                "tokenizer_revision": values["model.revision"],
                "max_model_len": int(values["model.max_model_len"]),
            },
            "scheduler": {
                "policy": "priority",  # engine.yaml's for every profile (deploy/templates)
                "max_num_batched_tokens": step_cap,
                "long_prefill_token_threshold": threshold,
                "max_num_seqs": serve["max_num_seqs"],
                "async_scheduling": serve["async_scheduling"],
            },
            "cache": {
                "block_size": serve["block_size"],
                "num_gpu_blocks": serve["pool_tokens"] // serve["block_size"],
                "pool_tokens": serve["pool_tokens"],
                "kv_cache_memory_bytes": int(values["engine.kv_cache_memory"]),
                "gpu_memory_utilization": float(values["engine.gpu_memory_utilization"]),
                "enable_prefix_caching": True,  # engine.yaml's
            },
            "kv_transfer": {
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": int(values["engine.cpu_bytes_to_use"]),
                    "offload_prompt_only": False,
                },
            },
            "make_room": {
                key: serve[key]
                for key in (
                    "max_preemptions_per_step",
                    "budget_cap",
                    "budget_cap_tokens",
                    "reserve_tokens",
                )
            },
        },
        "patches": {
            patch: sha256(VLLM_PACKAGE / "make-room" / patch) for patch in make_room_patches()
        },
        "voice": {"mode": "serve", "max_calls": serve["max_calls"], "audio": serve["audio"]},
    }


def outputs(profile: dict) -> tuple[str, ...]:
    """The files a profile renders: a node's, and a serving profile's own."""
    return RENDERED + (SERVED if "serving" in profile else ())


def render(profile: dict, host: dict | None, out: Path) -> dict:
    """Writes the node's directory to `out`; returns node.json's content."""
    values, plan, warnings = resolve(profile, host)
    # A serving profile's table is checked before any file is written, its variant's drop-in among them, which takes
    # some of its keys: a key it lacks or a value out of bounds is refused by name.
    serving = serving_config(profile, values) if "serving" in profile else None
    for rel in outputs(profile):
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if rel == "serving.json":
            dest.write_text(json.dumps(serving, indent=2) + "\n")
        else:
            dest.write_text(fill((TEMPLATES / rel).read_text(), values))
    for rel in COPIED:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BOX / rel, dest)
    node = {
        "profile": values["profile.name"],
        "values": {k: v for k, v in sorted(values.items()) if not k.startswith("text.")},
        "install": {
            "path": values["derived.install"],
            "reason": values["derived.install_note"],
            # On cu129 here, the box's lock would run once cuda-compat-13-0 is installed (bootstrap.sh tries).
            "compat_eligible": compat_eligible(host or {}),
        }
        if "derived.install" in values
        else None,
        "plan": plan,
        "warnings": warnings,
    }
    (out / "node.json").write_text(json.dumps(node, indent=2) + "\n")
    return node


def check(profile: dict, against: Path) -> list[str]:
    """The files that differ between the profile's rendering and `against`."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        render(profile, None, out)
        return [
            rel
            for rel in outputs(profile) + COPIED
            if not (against / rel).is_file()
            or (out / rel).read_bytes() != (against / rel).read_bytes()
        ]


def differences(profile: dict, against: Path) -> dict[str, str]:
    """Each file that differs between the profile's rendering and `against`, with its unified diff
    (rendered, then `against`): where the box's files changed, the template line to change."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        render(profile, None, out)
        diffs = {}
        for rel in check(profile, against):
            theirs = (
                (against / rel).read_text().splitlines(keepends=True)
                if (against / rel).is_file()
                else []
            )
            diffs[rel] = "".join(
                difflib.unified_diff(
                    (out / rel).read_text().splitlines(keepends=True),
                    theirs,
                    f"rendered/{rel}",
                    str(against / rel),
                )
            )
        return diffs


def unplaced(box: Path = BOX) -> list[str]:
    """The files of the box's directory that a node's directory neither renders nor copies and that
    BOX_ONLY does not name: each needs a place before another node runs without it."""
    files = sorted(
        str(path.relative_to(box))
        for path in box.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    return [
        rel
        for rel in files
        if rel not in RENDERED + SERVED + COPIED
        and not any(fnmatch.fnmatch(rel, pattern) for pattern in BOX_ONLY)
    ]


def update(profile: dict, into: Path) -> list[str]:
    """Writes the profile's rendered files into its repository copy `into`; returns those that changed."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        render(profile, None, out)
        changed = []
        for rel in outputs(profile):
            new = (out / rel).read_bytes()
            if not (into / rel).is_file() or (into / rel).read_bytes() != new:
                (into / rel).parent.mkdir(parents=True, exist_ok=True)
                (into / rel).write_bytes(new)
                changed.append(rel)
        return changed


def check_vllm(node_dir: Path, flags: list[str]) -> int:
    """Parses the node's engine.yaml and stt.yaml with vLLM's argument parser and builds their engine configs."""
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.launchers.cli_args import make_arg_parser
        from vllm.usage.usage_lib import UsageContext
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError as e:
        print(f"profile: check-vllm runs with the engine venv's Python ({e})", file=sys.stderr)
        return 2
    for name, extra in (("engine.yaml", flags), ("stt.yaml", [])):
        parser = make_arg_parser(FlexibleArgumentParser(prog="vllm serve"))
        args = parser.parse_args(["--config", str(node_dir / name), *extra])
        config = AsyncEngineArgs.from_cli_args(args).create_engine_config(
            UsageContext.OPENAI_API_SERVER
        )
        cache = config.cache_config
        print(
            f"{name}: {config.model_config.model}, kv_cache_memory_bytes {cache.kv_cache_memory_bytes}, "
            f"gpu_memory_utilization {cache.gpu_memory_utilization}, "
            f"max_num_batched_tokens {config.scheduler_config.max_num_batched_tokens}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="profile.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    profile_help = "a profile: its name, NAME for deploy/profiles/NAME.toml, or a TOML file's path"
    p = commands.add_parser("detect", help="write the host's facts as JSON")
    p.add_argument("--out", type=Path, help="write them to this file instead of stdout")
    p.add_argument(
        "--peer", help="also time TCP connects to this host's SSH port (split mode's RTT check)"
    )
    p = commands.add_parser("render", help="write a node's directory from a profile")
    p.add_argument("profile", help=profile_help)
    p.add_argument(
        "--host",
        type=Path,
        help="the host's facts (profile.py detect), for a profile's auto values",
    )
    p.add_argument("--out", type=Path, required=True, help="the node's directory to write")
    p = commands.add_parser(
        "check", help="compare a profile's rendering with its repository copy, byte for byte"
    )
    p.add_argument("profile", help=profile_help)
    p.add_argument(
        "--against", type=Path, help="the directory to compare with (default: the profile's dir)"
    )
    p = commands.add_parser(
        "update", help="write a profile's rendered files into its repository copy"
    )
    p.add_argument("profile", help=profile_help)
    p = commands.add_parser(
        "effective", help="print a serving profile's effective configuration (its serving.json)"
    )
    p.add_argument("profile", help=profile_help)
    p = commands.add_parser(
        "check-vllm", help="parse a rendered node's configs with vLLM's own parser"
    )
    p.add_argument("dir", type=Path, help="a node's directory, as render writes it")
    p.add_argument(
        "flags",
        nargs=argparse.REMAINDER,
        help="engine flags after the config, as a variant adds them",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "detect":
            text = json.dumps(detect(args.peer), indent=2) + "\n"
            if args.out:
                args.out.write_text(text)
            else:
                sys.stdout.write(text)
        elif args.command == "render":
            host = json.loads(args.host.read_text()) if args.host else None
            node = render(load(args.profile), host, args.out)
            for warning in node["warnings"]:
                print(f"profile: warning: {warning}", file=sys.stderr)
            print(f"profile: {node['profile']} rendered into {args.out}")
        elif args.command in ("check", "update"):
            profile = load(args.profile)
            copy = REPO / profile["profile"]["dir"] if profile["profile"].get("dir") else None
            against = getattr(args, "against", None) or copy
            if against is None:
                raise ProfileError(f"{args.profile} has no repository copy: pass --against")
            if args.command == "update":
                for rel in update(profile, against):
                    print(f"profile: wrote {against / rel}")
                return 0
            differ = differences(profile, against)
            for rel, diff in differ.items():
                print(f"profile: {rel} differs from {against / rel}:\n{diff}", file=sys.stderr)
            if differ:
                return 1
            print(f"profile: {args.profile} renders {against} byte for byte")
        elif args.command == "effective":
            profile = load(args.profile)
            if "serving" not in profile:
                raise ProfileError(
                    f"{args.profile} is no serving profile: it has no [serving] table"
                )
            values, _, _ = resolve(profile, None)
            sys.stdout.write(json.dumps(serving_config(profile, values), indent=2) + "\n")
        else:
            return check_vllm(args.dir, args.flags)
    except ProfileError as e:
        print(f"profile: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    # This file has the name of the standard library's profile module, which cProfile imports, as torch does under
    # vLLM's import (check-vllm). A script's directory comes first on sys.path, where this file would be found in
    # the library's place, so the directory leaves the path.
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != HERE]
    sys.exit(main())
