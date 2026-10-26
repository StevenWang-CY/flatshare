"""deploy/profile.py: the frozen box's profile renders deploy/phastform byte for byte, the A100's derives its
values from a host's facts by the rules, the core plan shares a host's CPUs, and detect reads a host through
stand-ins for nvidia-smi, systemd and the kernel's files."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from leaseholder_bench.e2 import run as e2run

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
GIB = 1 << 30
CACHES = {
    "VLLM_CACHE_ROOT": "vllm",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
}
"""The caches a vLLM start reads (the W8 cold-start drill empties these four), as a non-box node's units name them."""

spec = importlib.util.spec_from_file_location("lh_profile", DEPLOY / "profile.py")
profile = importlib.util.module_from_spec(spec)
sys.modules["lh_profile"] = profile
spec.loader.exec_module(profile)


def host(cpus: dict[str, list[int]] | None = None, **facts) -> dict:
    """An A100 host's facts: one NUMA node of 32 CPUs, 200 GiB of RAM, 100 GiB of /dev/shm, no CUDA toolkit."""
    base = {
        "hostname": "a100-vm",
        "gpu": {
            "name": "NVIDIA A100-SXM4-40GB",
            "memory_mib": 40960,
            "compute_cap": "8.0",
            "numa_node": None,
        },
        "cpus": cpus or {"0": list(range(32))},
        "memory": {"total_bytes": 200 * GIB, "available_bytes": 190 * GIB, "nodes": {}},
        "shm": {"path": "/dev/shm", "total_bytes": 100 * GIB, "free_bytes": 100 * GIB},
        "cuda_toolkit": None,
    }
    return base | facts


def render(tmp_path: Path, name: str, facts: dict | None) -> tuple[Path, dict]:
    out = tmp_path / name
    node = profile.render(profile.load(name), facts, out)
    return out, node


def test_the_box_profile_renders_todays_files(tmp_path: Path) -> None:
    assert profile.check(profile.load("rtx-5060-ti"), DEPLOY / "phastform") == []
    done = subprocess.run(
        [sys.executable, str(DEPLOY / "profile.py"), "check", "rtx-5060-ti"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "byte for byte" in done.stdout
    # Every rendered file is the box's, and the box's own scripts come along as they are.
    out, node = render(tmp_path, "rtx-5060-ti", None)
    for rel in profile.RENDERED + profile.COPIED:
        assert (out / rel).read_bytes() == (DEPLOY / "phastform" / rel).read_bytes(), rel
    assert node["plan"] is None and node["warnings"] == []


def test_every_file_of_the_box_has_its_place_in_a_node(tmp_path: Path) -> None:
    # Rendered, copied or the box's own: a file another change adds to deploy/phastform, a unit or a drop-in,
    # fails here until it has a place, rather than missing from other nodes.
    assert profile.unplaced() == []
    box = tmp_path / "box"
    shutil.copytree(DEPLOY / "phastform", box, ignore=shutil.ignore_patterns("__pycache__"))
    (box / "systemd" / "lh-engine.service.d").mkdir()
    (box / "systemd" / "lh-engine.service.d" / "measure.conf").write_text("[Service]\n")
    assert profile.unplaced(box) == ["systemd/lh-engine.service.d/measure.conf"]


def test_a_value_that_differs_from_the_box_shows(tmp_path: Path) -> None:
    box = profile.load("rtx-5060-ti")
    box["engine"]["kv_cache_memory"] = 6 << 30
    box["text"]["audio_numa"] = "# NUMA node 0 for the audio servers."
    assert profile.check(box, DEPLOY / "phastform") == [
        "engine.yaml",
        "systemd/lh-stt.service",
        "systemd/lh-tts.service",
    ]
    diff = profile.differences(box, DEPLOY / "phastform")["engine.yaml"]
    assert "\n-kv-cache-memory: 6442450944\n+kv-cache-memory: 5368709120\n" in diff
    # A box file that changed shows the line to carry into its template.
    copy = tmp_path / "box"
    shutil.copytree(DEPLOY / "phastform", copy, ignore=shutil.ignore_patterns("__pycache__"))
    unit = copy / "systemd" / "lh-engine.service"
    text = unit.read_text()
    assert text.count("RestartSec=5\n") == 1
    unit.write_text(text.replace("RestartSec=5\n", "RestartSec=7\n"))
    done = subprocess.run(
        [
            sys.executable,
            str(DEPLOY / "profile.py"),
            "check",
            "rtx-5060-ti",
            "--against",
            str(copy),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1
    assert "systemd/lh-engine.service differs" in done.stderr
    assert "\n-RestartSec=5\n+RestartSec=7\n" in done.stderr


def test_the_a100_profile_derives_its_values_from_the_host(tmp_path: Path) -> None:
    out, node = render(tmp_path, "a100-40gb", host())
    values, plan = node["values"], node["plan"]
    # R2: the box's 10.83 and 2.32 GiB budgets of the card's 40 GiB.
    assert (values["engine.gpu_memory_utilization"], values["stt.gpu_memory_utilization"]) == (
        "0.28",
        "0.06",
    )
    # R3: four times the box's 5 GiB pool, which /dev/shm and the RAM hold.
    assert values["engine.cpu_bytes_to_use"] == str(20 * GIB)
    engine = (out / "engine.yaml").read_text()
    assert "kv-cache-memory: 5368709120\n" in engine and "gpu-memory-utilization: 0.28\n" in engine
    assert "cpu_bytes_to_use: 21474836480\n" in engine
    assert (
        "max-num-batched-tokens: 512\n" in engine
        and "long-prefill-token-threshold: 256\n" in engine
    )
    assert "model: Qwen/Qwen3-4B-Instruct-2507-FP8\n" in engine
    assert "{{" not in engine and ": auto\n" not in engine
    # No CUDA toolkit on the host: the engine and STT use the venv's, as Gray does.
    unit = (out / "systemd" / "lh-engine.service").read_text()
    assert f"Environment=CUDA_HOME={profile.VENV_CUDA}\n" in unit
    assert (
        "# cache/curand_include; the host has no CUDA 13 toolkit, so the engine JIT-compiles with the CUDA 13 "
        "compiler its\n# venv carries (nvidia-cuda-nvcc, pinned by the engine lock), as on Gray, and links with the "
        "venv's runtime and a driver\n# stub, which bootstrap.sh puts in cache/cuda_lib.\n" in unit
    )
    assert all(len(line) <= 120 for line in unit.splitlines() if line.startswith("#")), unit
    assert (
        "--config %h/leaseholder/node/engine.yaml --uds ${LH_ENGINE_SOCKET} $LH_ENGINE_FLAGS"
        in unit
    )
    assert (
        f"Environment=CUDA_HOME={profile.VENV_CUDA}\n"
        in (out / "systemd" / "lh-stt.service").read_text()
    )
    # Every compile and JIT cache of a start under cache/, so that the engine switches keep them warm.
    stt = (out / "systemd" / "lh-stt.service").read_text()
    for variable, directory in CACHES.items():
        line = f"Environment={variable}=%h/leaseholder/cache/{directory}\n"
        assert unit.count(line) == 1 and stt.count(line) == 1, variable
    make_room = (out / "variants" / "make-room" / "lh-engine.conf").read_text()
    assert "--config %h/leaseholder/node/engine.yaml --scheduler-cls" in make_room
    # R7 on 32 CPUs, under the 34 that give each component its want.
    assert plan == {
        "engine": "0-7",
        "audio": "8-11",
        "gateway": "12-13",
        "voice": "14-20",
        "aiperf": "21-27",
        "callers": "28-31",
        "engine_numa": "0",
        "audio_numa": "0",
    }
    assert f"CPUAffinity={plan['engine']}\n" in unit
    assert node["warnings"] == []
    assert json.loads((out / "node.json").read_text()) == node


STT_KV = 458_752_000
"""The STT's fixed KV pool: 1,400 blocks of 16 tokens at 327,680 bytes a block (Whisper large-v3-turbo's 4 decoder
layers, 20 heads of 64, FP16 keys and values), the 4,020 tokens the box's 0.15 gave it (vLLM's "GPU KV cache size:
4,020 tokens" at the 2026-10-25 start, benchmarks/results/2026-10-25-drills/w6-start-box)."""


def test_both_vllm_servers_fix_their_pools_and_the_stt_keeps_the_boxs_on_every_card(
    tmp_path: Path,
) -> None:
    # A fixed pool is what lets the three servers start at once: neither vLLM server profiles memory, which would
    # read what the others allocate meanwhile. vLLM still checks the budget's free memory at start.
    assert STT_KV == 1400 * 16 * 4 * 20 * 64 * 2 * 2
    box, _ = render(tmp_path, "rtx-5060-ti", None)
    a100, _ = render(tmp_path, "a100-40gb", host())
    for out, budget in ((box, "0.15"), (a100, "0.06")):
        stt = (out / "stt.yaml").read_text()
        assert f"\nkv-cache-memory: {STT_KV}\n" in stt, out
        assert f"\ngpu-memory-utilization: {budget}\n" in stt, out
        assert "\nkv-cache-memory: 5368709120\n" in (out / "engine.yaml").read_text(), out


def test_the_budgets_and_the_tts_must_fit_the_card_together(tmp_path: Path) -> None:
    # R2: the servers start at once, so the last to check its free memory may find the others resident. The
    # engine's and the STT's budgets and the TTS's share fit the card together, at most 0.95 of it, as one budget
    # alone: vLLM's total is the card less the driver's reserve (the box: 15,852 of nvidia-smi's 16,311 MiB).
    values, _, _ = profile.resolve(profile.load("rtx-5060-ti"), None)
    # The box: 0.70 + 0.15 + the TTS's 1.08 GiB of 16,311 MiB.
    assert values["derived.card_share"] == "0.92"
    box = profile.load("rtx-5060-ti")
    box["engine"]["gpu_memory_utilization"] = "0.75"
    with pytest.raises(profile.ProfileError, match=r"refused: .* 0\.97 of the 16311 MiB card"):
        profile.render(box, None, tmp_path / "box")
    # The A100: the box's budgets in GiB, 0.28 + 0.06 + 1.08 GiB of 40 GiB.
    _, node = render(tmp_path, "a100-40gb", host())
    assert node["values"]["derived.card_share"] == "0.37"
    # 36 GiB for the engine passes as one budget (0.90), not beside the STT and the TTS.
    big = profile.load("a100-40gb")
    big["budgets"]["engine_gib"] = 36
    assert profile.utilization(36, 40960) == "0.90"
    with pytest.raises(profile.ProfileError, match=r"refused: .* 0\.99 of the 40960 MiB card"):
        profile.render(big, host(), tmp_path / "big")
    # A profile that does not say what the TTS takes, or what its card holds, is refused, not passed.
    for table, key in (("budgets", "tts_gib"), ("profile", "memory_mib")):
        partial = profile.load("rtx-5060-ti")
        del partial[table][key]
        with pytest.raises(profile.ProfileError, match=f"{table}.{key}"):
            profile.resolve(partial, None)


def test_a_host_with_its_own_cuda_13_toolkit_compiles_with_it(tmp_path: Path) -> None:
    facts = host(cuda_toolkit={"version": "13.0", "path": "/usr/local/cuda-13.0"})
    out, _ = render(tmp_path, "a100-40gb", facts)
    unit = (out / "systemd" / "lh-engine.service").read_text()
    assert "Environment=CUDA_HOME=/usr/local/cuda-13.0\n" in unit
    assert "Environment=PATH=%h/leaseholder/venvs/engine/bin:/usr/local/cuda-13.0/bin:" in unit
    assert (
        "# cache/curand_include; the host's CUDA 13.0 toolkit at /usr/local/cuda-13.0 compiles.\n"
        in unit
    )
    # The STT unit then needs no CUDA_HOME of its own, as on the box; a 12.x toolkit is not used.
    assert "CUDA_HOME" not in (out / "systemd" / "lh-stt.service").read_text()
    out, _ = render(
        tmp_path / "old",
        "a100-40gb",
        host(cuda_toolkit={"version": "12.4", "path": "/usr/local/cuda"}),
    )
    assert f"CUDA_HOME={profile.VENV_CUDA}" in (out / "systemd" / "lh-engine.service").read_text()


A100 = {
    "name": "NVIDIA A100-SXM4-40GB",
    "memory_mib": 40960,
    "compute_cap": "8.0",
    "numa_node": None,
}
COMPAT = "/usr/local/cuda-13.0/compat"


@pytest.mark.parametrize(
    ("driver", "name", "compat", "path"),
    [
        # Before the card is visible (staging): the box's lock until bootstrap.sh gpu reads the driver.
        (None, A100["name"], None, "cu130"),
        ("580.65.06", A100["name"], None, "cu130"),
        ("590.48.01", A100["name"], None, "cu130"),
        ("570.86.15", A100["name"], COMPAT, "cu130-compat"),
        ("535.230.02", A100["name"], COMPAT, "cu130-compat"),
        # The forward-compatibility libraries not installed.
        ("570.86.15", A100["name"], None, "cu129"),
        # No R550 in cuda-compat-13-0's table.
        ("550.54.15", A100["name"], COMPAT, "cu129"),
        # A GeForce card gets error 803 from the libraries.
        ("570.86.15", "NVIDIA GeForce RTX 4090", COMPAT, "cu129"),
        ("525.60.13", A100["name"], None, "cu129"),
    ],
)
def test_the_install_path_follows_the_driver(driver, name, compat, path) -> None:
    facts = {"gpu": {"name": name, "driver": driver} if driver else None, "cuda_compat": compat}
    assert profile.install_path(facts)[0] == path


def test_a_driver_older_than_r525_is_refused(tmp_path: Path) -> None:
    with pytest.raises(profile.ProfileError, match="older than R525"):
        profile.install_path({"gpu": {"name": A100["name"], "driver": "520.61.05"}})
    with pytest.raises(profile.ProfileError, match="older than R525"):
        render(tmp_path, "a100-40gb", host(gpu=A100 | {"driver": "470.256.02"}))


def test_each_install_path_compiles_at_its_runtimes_cuda_version(tmp_path: Path) -> None:
    def units(name: str, **facts) -> tuple[str, str, dict]:
        out, node = render(tmp_path / name, "a100-40gb", host(**facts))
        read = lambda unit: (out / "systemd" / unit).read_text()  # noqa: E731
        assert "LD_LIBRARY_PATH" not in read("lh-tts.service")  # Kokoro's own CUDA 12.8 build
        return read("lh-engine.service"), read("lh-stt.service"), node

    # R580 without a toolkit: the venv's CUDA 13.0, which FlashInfer links through cache/cuda_lib (the venv's
    # runtime and a driver stub): its lib/ has no libcudart.so and no stubs.
    engine, stt, node = units("venv13", gpu=A100 | {"driver": "580.65.06"})
    assert node["install"]["path"] == "cu130"
    assert f"Environment=CUDA_HOME={profile.VENV_CUDA}\n" in engine
    assert "Environment=FLASHINFER_EXTRA_LDFLAGS=-L%h/leaseholder/cache/cuda_lib\n" in engine
    assert "LD_LIBRARY_PATH" not in engine + stt
    # The host's own 13.0 toolkit has both, so no link flags; a 13.2 toolkit is not the runtime's version.
    toolkits = [
        {"version": "13.2", "path": "/usr/local/cuda-13.2"},
        {"version": "13.0", "path": "/usr/local/cuda-13.0"},
    ]
    engine, stt, _ = units("system13", gpu=A100 | {"driver": "580.65.06"}, cuda_toolkits=toolkits)
    assert "Environment=CUDA_HOME=/usr/local/cuda-13.0\n" in engine
    assert "FLASHINFER_EXTRA_LDFLAGS" not in engine and "CUDA_HOME" not in stt
    # R550: the cu129 lock, compiled by a CUDA 12.9 toolkit, the host's where it has one, ...
    toolkits.append({"version": "12.9", "path": "/usr/local/cuda-12.9"})
    engine, _, node = units("system129", gpu=A100 | {"driver": "550.54.15"}, cuda_toolkits=toolkits)
    assert node["install"]["path"] == "cu129"
    assert "Environment=CUDA_HOME=/usr/local/cuda-12.9\n" in engine
    # ... else the one bootstrap.sh installs, since CUDA 12's wheels carry no nvcc.
    engine, stt, node = units("home129", gpu=A100 | {"driver": "550.54.15"})
    assert f"Environment=CUDA_HOME={profile.HOME_CUDA}\n" in engine
    assert f"Environment=CUDA_HOME={profile.HOME_CUDA}\n" in stt
    assert (
        "FLASHINFER_EXTRA_LDFLAGS" not in engine
        and "installs into ~/leaseholder/cuda-12.9." in engine
    )
    assert node["install"]["compat_eligible"] is False
    for unit in (engine, stt):
        assert all(len(line) <= 120 for line in unit.splitlines() if line.startswith("#")), unit
    # R570 with the forward-compatibility libraries: the box's lock, with them on both vLLM units' library path.
    engine, stt, node = units("compat", gpu=A100 | {"driver": "570.86.15"}, cuda_compat=COMPAT)
    assert node["install"]["path"] == "cu130-compat"
    assert f"Environment=LD_LIBRARY_PATH={COMPAT}\n" in engine
    assert f"Environment=LD_LIBRARY_PATH={COMPAT}\n" in stt
    # R570 without them: cu129, and bootstrap.sh tries to install them.
    _, _, node = units("nocompat", gpu=A100 | {"driver": "570.86.15"})
    assert node["install"]["path"] == "cu129" and node["install"]["compat_eligible"] is True


NVIDIA_LIBS = "/usr/lib64-nvidia"
"""Where Colab's container keeps the driver's libraries: a directory the linker's cache does not list."""


def test_the_gpu_units_name_the_drivers_libraries_where_the_linker_does_not_find_them(
    tmp_path: Path,
) -> None:
    # Colab's container: the driver's libcuda.so.1 is in /usr/lib64-nvidia, which only its shells' LD_LIBRARY_PATH
    # reaches. The units' environment has no such path, so every GPU unit names the directory itself, the TTS's
    # Kokoro too, whose torch loads the same libcuda.so.1.
    outside = {"path": NVIDIA_LIBS, "linker": False, "via": "LD_LIBRARY_PATH"}

    def units(name: str, **facts) -> dict[str, list[str]]:
        out, _ = render(tmp_path / name, "a100-40gb", host(**facts))
        return {
            unit: (out / "systemd" / f"{unit}.service").read_text().splitlines()
            for unit in ("lh-engine", "lh-stt", "lh-tts")
        }

    def library_path(lines: list[str]) -> list[str]:
        return [line for line in lines if line.startswith("Environment=LD_LIBRARY_PATH=")]

    for unit, lines in units(
        "colab", gpu=A100 | {"driver": "580.82.07"}, driver_libs=outside
    ).items():
        assert library_path(lines) == [f"Environment=LD_LIBRARY_PATH={NVIDIA_LIBS}"], unit
        assert all(len(line) <= 120 for line in lines if line.startswith("#")), unit
    # With cuda-compat-13-0 (R570): its libraries first, so that its libcuda.so.1 is the one loaded, then the driver's
    # own; the TTS, Kokoro's own CUDA 12.8 build, takes the driver's alone.
    got = units(
        "compat", gpu=A100 | {"driver": "570.86.15"}, cuda_compat=COMPAT, driver_libs=outside
    )
    for unit in ("lh-engine", "lh-stt"):
        assert library_path(got[unit]) == [f"Environment=LD_LIBRARY_PATH={COMPAT}:{NVIDIA_LIBS}"]
        assert all(len(line) <= 120 for line in got[unit] if line.startswith("#")), unit
    assert library_path(got["lh-tts"]) == [f"Environment=LD_LIBRARY_PATH={NVIDIA_LIBS}"]
    # Where the linker finds them on its own (its cache, or the directory it searches anyway), nothing changes:
    # compat's libraries alone where they are used, else no library path; as without the fact (older facts).
    inside = {"path": "/usr/lib/x86_64-linux-gnu", "linker": True, "via": "the linker's cache"}
    for name, libs in (("cache", inside), ("unknown", None)):
        got = units(
            f"{name}-compat",
            gpu=A100 | {"driver": "570.86.15"},
            cuda_compat=COMPAT,
            driver_libs=libs,
        )
        assert library_path(got["lh-engine"]) == [f"Environment=LD_LIBRARY_PATH={COMPAT}"], name
        assert library_path(got["lh-stt"]) == [f"Environment=LD_LIBRARY_PATH={COMPAT}"], name
        assert library_path(got["lh-tts"]) == [], name
        for unit, lines in units(
            name, gpu=A100 | {"driver": "580.82.07"}, driver_libs=libs
        ).items():
            assert library_path(lines) == [], (name, unit)


def test_the_second_models_profile_renders_the_cores_node_with_its_own_model_and_pool(
    tmp_path: Path,
) -> None:
    host = {
        "hostname": "a100-vm",
        "gpu": {
            "name": "NVIDIA A100-SXM4-40GB",
            "memory_mib": 40960,
            "compute_cap": "8.0",
            "driver": "580.65.06",
        },
        "cpus": {"0": list(range(32))},
        "memory": {"total_bytes": 200 << 30, "available_bytes": 190 << 30, "nodes": {}},
        "shm": {"path": "/dev/shm", "total_bytes": 100 << 30, "free_bytes": 100 << 30},
    }
    core = profile.render(profile.load("a100-40gb"), host, tmp_path / "node")
    second = profile.render(profile.load("a100-40gb-qwen3-14b"), host, tmp_path / "node-14b")
    engine = (tmp_path / "node-14b" / "engine.yaml").read_text()
    assert "\nmodel: Qwen/Qwen3-14B-FP8\n" in engine
    assert "\nrevision: 9a283b4a5efbc09ce247e0ae5b02b744739e525a\n" in engine
    assert '\ndefault-chat-template-kwargs: {"enable_thinking": false}\n' in engine
    # 7,334 blocks of 16 tokens of the 14B's 160 KiB a token; the CPU tier four times it, as /dev/shm and RAM allow.
    assert "kv-cache-memory: 19225640960\n" in engine and 19225640960 == 7334 * 16 * 160 * 1024
    assert second["values"]["engine.cpu_bytes_to_use"] == str(71 << 30)
    # Everything else is the core's: its STT, its plan and its units, in their own directory.
    assert (tmp_path / "node-14b" / "stt.yaml").read_text() == (
        tmp_path / "node" / "stt.yaml"
    ).read_text()
    assert second["plan"] == core["plan"]
    # The 14B's engine outgrows its budget, so it starts after the audio servers (R2); the core's starts beside them.
    order = "\nAfter=lh-stt.service lh-tts.service\n"
    its_order = re.compile(
        r"# On this node the engine starts after the audio servers.*?" + order, re.S
    )
    assert order in (tmp_path / "node-14b" / "systemd" / "lh-engine.service").read_text()
    assert "After=lh-stt" not in (tmp_path / "node" / "systemd" / "lh-engine.service").read_text()
    for unit in ("lh-engine.service", "lh-stt.service", "lh-tts.service"):
        ours = (tmp_path / "node-14b" / "systemd" / unit).read_text()
        theirs = (tmp_path / "node" / "systemd" / unit).read_text()
        ours = ours.replace("leaseholder/node-14b/", "leaseholder/node/")
        ours = its_order.sub("", ours)
        assert ours.replace("Qwen3-14B-FP8", "Qwen3-4B-Instruct-2507-FP8") == theirs, unit
    # The harness's engine check reads the same profile, its base's values under its own.
    flags = dict(zip(*[iter(e2run.profile_flags("a100-40gb-qwen3-14b"))] * 2, strict=True))
    assert flags["--model"] == "Qwen/Qwen3-14B-FP8" and flags["--max-num-batched-tokens"] == "512"
    assert flags["--revision"] == second["values"]["model.revision"]


def test_the_cpu_tier_shrinks_to_what_the_host_holds_and_is_refused_below_the_pool(
    tmp_path: Path,
) -> None:
    pool = 5 * GIB
    small = host(shm={"path": "/dev/shm", "total_bytes": 14 * GIB, "free_bytes": 14 * GIB})
    tier, why = profile.cpu_tier(pool, 4, small, 25)
    assert tier == 12 * GIB and "2.4 times the GPU pool" in why and "/dev/shm" in why
    tight = host(memory={"total_bytes": 64 * GIB, "available_bytes": 40 * GIB, "nodes": {}})
    assert profile.cpu_tier(pool, 4, tight, 25)[0] == 15 * GIB
    # Docker's default 64 MB /dev/shm holds no tier at all.
    docker = host(shm={"path": "/dev/shm", "total_bytes": 64 << 20, "free_bytes": 64 << 20})
    with pytest.raises(profile.ProfileError, match=r"refused.*--shm-size"):
        profile.cpu_tier(pool, 4, docker, 25)
    with pytest.raises(profile.ProfileError, match="refused"):
        render(tmp_path, "a100-40gb", docker)


def test_the_core_plan_shares_the_cpus_and_warns_below_the_floor() -> None:
    def sizes(total: int) -> dict[str, int]:
        plan, _ = profile.core_plan({"0": list(range(total))}, 0)
        return {k: len(profile.parse_cpulist(plan[k])) for k, _ in profile.CORES}

    # 34 CPUs give each component its want; more leave the rest spare; 17 give each half of it.
    assert sizes(34) == dict(profile.CORES)
    plan, warnings = profile.core_plan({"0": list(range(64))}, 0)
    assert plan["spare"] == "34-63" and warnings == []
    assert sizes(17) == {k: w // 2 for k, w in profile.CORES}
    for total in (8, 12, 16, 17, 20, 30, 33):
        got = sizes(total)
        assert sum(got.values()) == total and min(got.values()) >= 1, (total, got)
        assert all(got[k] <= w for k, w in profile.CORES), (total, got)
    # Phastform's 16 CPUs on two nodes, the GPU on node 1: below the floor, with a warning; the engine takes the
    # GPU's node and the audio servers the other, as on the box.
    plan, warnings = profile.core_plan({"0": list(range(8)), "1": list(range(8, 16))}, 1)
    assert len(warnings) == 1 and "below the single-host floor of 17" in warnings[0]
    assert plan["engine_numa"] == "1" and set(profile.parse_cpulist(plan["engine"])) <= set(
        range(8, 16)
    )
    assert plan["audio_numa"] == "0" and set(profile.parse_cpulist(plan["audio"])) <= set(range(8))
    # Fewer CPUs than components: they share.
    plan, warnings = profile.core_plan({"0": [0, 1, 2, 3]}, None)
    assert warnings and all(plan[k] for k, _ in profile.CORES)


def test_a_phastform_sized_host_renders_with_its_warning(tmp_path: Path) -> None:
    facts = host(cpus={"0": list(range(8)), "1": list(range(8, 16))})
    facts["gpu"] = facts["gpu"] | {"numa_node": 1}
    out, node = render(tmp_path, "a100-40gb", facts)
    assert node["warnings"] and "mechanics" in node["warnings"][0]
    unit = (out / "systemd" / "lh-engine.service").read_text()
    assert "NUMAMask=1\n" in unit and "# NUMA node 1 is local to the GPU (detected)." in unit
    assert "NUMAMask=0\n" in (out / "systemd" / "lh-tts.service").read_text()


def test_cpulists_round_trip() -> None:
    assert profile.parse_cpulist("0-7,16-23,30") == [*range(8), *range(16, 24), 30]
    assert profile.cpulist([3, 0, 1, 2, 8, 10, 9, 12]) == "0-3,8-10,12"
    assert profile.cpulist(profile.parse_cpulist("0-3,5")) == "0-3,5"


def test_placeholders_fill_and_an_empty_one_takes_its_line() -> None:
    values = {"a.b": "x", "a.empty": "", "a.nested": "{{a.b}}y"}
    assert profile.fill("1 {{a.b}}\n{{a.empty}}\n{{a.nested}}\n", values) == "1 x\nxy\n"
    with pytest.raises(profile.ProfileError, match=r"a\.missing"):
        profile.fill("{{a.missing}}\n", values)
    with pytest.raises(profile.ProfileError, match="--host"):
        profile.resolve(profile.load("a100-40gb"), None)


A100_ROW = "NVIDIA A100-SXM4-40GB, 40960, 8.0, 580.65.06, 4, 16, Disabled, 400.00, 00000000:3B:00.0, Enabled, Default"
FAKES = {
    # nvidia-smi --query-gpu=index,uuid,name,...: the card's row; -L its line.
    "nvidia-smi": f"""case "$*" in
  -L) echo "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-6f1c2d3e-aaaa-bbbb-cccc-000000000001)" ;;
  *) echo "0, GPU-6f1c2d3e-aaaa-bbbb-cccc-000000000001, {A100_ROW}" ;;
esac""",
    "systemctl": "echo running",
    "loginctl": "echo yes",
    "unshare": "exit 1",
    "ip": "exit 0",
    "nvcc": 'echo "Cuda compilation tools, release 13.0, V13.0.88"',
    "sudo": "exit 1",
}


def test_detect_reads_the_host_through_its_tools_and_files(tmp_path: Path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in FAKES.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    sysfs, procfs, rootfs, shm = (tmp_path / d for d in ("sys", "proc", "root", "shm"))
    for node, cpus in (("0", "0-7"), ("1", "8-15")):
        d = sysfs / "devices" / "system" / "node" / f"node{node}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(f"{cpus}\n")
        (d / "meminfo").write_text(
            f"Node {node} MemTotal: 98304000 kB\nNode {node} MemFree: 90000000 kB\n"
        )
    pci = sysfs / "bus" / "pci" / "devices" / "0000:3b:00.0"
    pci.mkdir(parents=True)
    (pci / "numa_node").write_text("1\n")
    thp = sysfs / "kernel" / "mm" / "transparent_hugepage"
    thp.mkdir(parents=True)
    (thp / "shmem_enabled").write_text("always within_size advise [never] deny force\n")
    (procfs / "1").mkdir(parents=True)
    (procfs / "meminfo").write_text("MemTotal: 196608000 kB\nMemAvailable: 180000000 kB\n")
    (procfs / "1" / "cgroup").write_text("0::/init.scope\n")
    (rootfs / "usr" / "local").mkdir(parents=True)
    shm.mkdir()
    for key, value in {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "LH_SYSFS": str(sysfs),
        "LH_PROCFS": str(procfs),
        "LH_ROOTFS": str(rootfs),
        "LH_SHM": str(shm),
    }.items():
        monkeypatch.setenv(key, value)
    facts = profile.detect()
    gpu = facts["gpu"]
    assert (gpu["name"], gpu["memory_mib"], gpu["compute_cap"], gpu["driver"]) == (
        "NVIDIA A100-SXM4-40GB",
        40960,
        "8.0",
        "580.65.06",
    )
    assert (
        gpu["pcie_gen"],
        gpu["pcie_width"],
        gpu["mig"],
        gpu["power_limit_w"],
        gpu["numa_node"],
    ) == (
        4,
        16,
        "Disabled",
        400.0,
        1,
    )
    assert facts["cpus"] == {"0": list(range(8)), "1": list(range(8, 16))}
    assert facts["memory"]["available_bytes"] == 180000000 * 1024
    assert facts["memory"]["nodes"]["1"]["total_bytes"] == 98304000 * 1024
    assert facts["thp_shmem"] == "never" and facts["shm"]["path"] == str(shm)
    assert (
        facts["systemd_user"],
        facts["linger"],
        facts["userns"],
        facts["sudo"],
        facts["container"],
    ) == (
        True,
        True,
        False,
        False,
        False,
    )
    assert facts["cuda_toolkit"] == {"version": "13.0", "path": str(tmp_path.resolve())}
    # The facts render the A100's node: the GPU on node 1 takes the engine (with a /dev/shm that holds the tier,
    # where the scratch directory's file system may not).
    shm_facts = {"path": "/dev/shm", "total_bytes": 100 * GIB, "free_bytes": 100 * GIB}
    _, node = render(tmp_path, "a100-40gb", facts | {"shm": shm_facts})
    assert node["plan"]["engine_numa"] == "1"
    assert node["values"]["units.cuda_home"] == str(tmp_path.resolve())


TWO_GPUS = f"""case "$*" in
  -L)
    echo "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-aaaa)"
    echo "GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-bbbb)"
    echo "  MIG 3g.20gb     Device  0: (UUID: MIG-bbbb-0)" ;;
  *)
    echo "0, GPU-aaaa, {A100_ROW}"
    echo "1, GPU-bbbb, {A100_ROW.replace("Disabled", "Enabled").replace("3B:00.0", "AF:00.0").replace("Default", "Exclusive_Process")}" ;;
esac"""
"""nvidia-smi on a host with two A100s, the second in MIG mode with one slice."""


def test_detect_takes_the_lent_gpu_of_several_by_index_or_uuid(tmp_path: Path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in FAKES.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{TWO_GPUS if name == 'nvidia-smi' else body}\n")
        (bin_dir / name).chmod(0o755)
    sysfs, procfs, rootfs, shm = (tmp_path / d for d in ("sys", "proc", "root", "shm"))
    for bus, node in (("0000:3b:00.0", "0"), ("0000:af:00.0", "1")):
        (sysfs / "bus" / "pci" / "devices" / bus).mkdir(parents=True)
        (sysfs / "bus" / "pci" / "devices" / bus / "numa_node").write_text(f"{node}\n")
    (procfs / "1").mkdir(parents=True)
    (procfs / "1" / "cgroup").write_text("0::/init.scope\n")
    rootfs.mkdir()
    shm.mkdir()
    for key, value in {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "LH_SYSFS": str(sysfs),
        "LH_PROCFS": str(procfs),
        "LH_ROOTFS": str(rootfs),
        "LH_SHM": str(shm),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("LH_GPU_ID", raising=False)
    # Two cards and none chosen: the facts say so (the preflight refuses), and the units name no GPU.
    gpu = profile.detect()["gpu"]
    assert (gpu["count"], gpu["chosen"], gpu["wanted"]) == (2, False, None)
    assert gpu["all"] == [
        "0: NVIDIA A100-SXM4-40GB (GPU-aaaa)",
        "1: NVIDIA A100-SXM4-40GB (GPU-bbbb)",
    ]
    out, _ = render(tmp_path, "a100-40gb", host(gpu=gpu))
    assert "CUDA_VISIBLE_DEVICES" not in (out / "systemd" / "lh-engine.service").read_text()
    for wanted in ("1", "GPU-bbbb"):
        monkeypatch.setenv("LH_GPU_ID", wanted)
        gpu = profile.detect()["gpu"]
        assert (gpu["index"], gpu["uuid"], gpu["chosen"], gpu["numa_node"]) == (
            1,
            "GPU-bbbb",
            True,
            1,
        )
        assert gpu["bus_id"] == "00000000:AF:00.0" and gpu["mig"] == "Enabled"
        assert gpu["compute_mode"] == "Exclusive_Process"
        assert gpu["mig_devices"] == [{"profile": "3g.20gb", "uuid": "MIG-bbbb-0"}]
    monkeypatch.setenv("LH_GPU_ID", "GPU-aaaa")
    gpu = profile.detect()["gpu"]
    assert (gpu["index"], gpu["chosen"], gpu["mig_devices"]) == (0, True, [])
    # The three GPU units run on it alone, by its UUID (in nvidia-smi's PCI order for CUDA), and gpu-guard asks
    # nvidia-smi about it alone.
    out, _ = render(tmp_path / "chosen", "a100-40gb", host(gpu=gpu))
    for unit in ("lh-engine", "lh-stt", "lh-tts"):
        text = (out / "systemd" / f"{unit}.service").read_text()
        for line in (
            "Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID",
            "Environment=CUDA_VISIBLE_DEVICES=GPU-aaaa",
            "Environment=LH_GPU_ID=GPU-aaaa",
        ):
            assert line in text.splitlines(), (unit, line)
        # The unit's environment is its commands', gpu-guard's (an ExecStartPre) among them.
        assert "ExecStartPre=%h/leaseholder/src/deploy/bin/gpu-guard" in text.splitlines()
    monkeypatch.setenv("LH_GPU_ID", "7")
    gpu = profile.detect()["gpu"]
    assert (gpu["chosen"], gpu["wanted"]) == (False, "7")


def libcuda(directory: Path, version: str = "580.82.07") -> Path:
    """A driver's libcuda.so.1 in ``directory``: the soname's link to the versioned library, as the driver installs it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"libcuda.so.{version}").write_text("")
    (directory / "libcuda.so.1").symlink_to(f"libcuda.so.{version}")
    return directory


LIBC6 = {"x86_64": "libc6,x86-64", "aarch64": "libc6,AArch64"}.get(os.uname().machine, "libc6")
"""ldconfig -p's tag of this machine's 64-bit libraries."""


def test_detect_finds_the_drivers_libraries_where_the_linker_does_not_look(
    tmp_path: Path, monkeypatch
) -> None:
    # Colab's container: the driver's libraries (580.82.07) are in /usr/lib64-nvidia, which the linker's cache does
    # not list; its shells reach them through LD_LIBRARY_PATH, where nvidia-smi works, but the units' environment has
    # no such path, so torch found no driver there. detect finds the directory, and that the linker does not.
    bin_dir, rootfs = tmp_path / "bin", tmp_path / "root"
    bin_dir.mkdir()
    rootfs.mkdir()
    cache = tmp_path / "cache.txt"
    libc = f"\tlibc.so.6 ({LIBC6}) => /lib/x86_64-linux-gnu/libc.so.6\n"
    cache.write_text("1 libs found in cache `/etc/ld.so.cache'\n" + libc)
    # ldconfig -p: the linker's cache, as the host's ldconfig prints it.
    (bin_dir / "ldconfig").write_text(f'#!/bin/sh\n[ "$1" = -p ] && cat {cache}\n')
    (bin_dir / "ldconfig").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    monkeypatch.setenv("LH_ROOTFS", str(rootfs))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    # No libcuda.so.1 anywhere the facts look (no driver installed, or a container without the NVIDIA runtime).
    assert profile.detect()["driver_libs"] is None
    # On the operator's LD_LIBRARY_PATH, after cuda-compat's directory, whose libcuda.so.1 is not the driver's.
    nvidia = libcuda(tmp_path / "lib64-nvidia")
    compat = libcuda(tmp_path / "cuda-13.0" / "compat", "580.65.06")
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{compat}:{nvidia}")
    assert profile.detect()["driver_libs"] == {
        "path": str(nvidia),
        "linker": False,
        "via": "LD_LIBRARY_PATH",
    }
    # Listed in the linker's cache (another architecture's entry aside): the linker finds it on its own.
    i386 = libcuda(tmp_path / "i386-linux-gnu")
    cache.write_text(
        "3 libs found in cache `/etc/ld.so.cache'\n"
        f"\tlibcuda.so.1 (libc6) => {i386}/libcuda.so.1\n"
        f"\tlibcuda.so.1 ({LIBC6}) => {nvidia}/libcuda.so.1\n" + libc
    )
    assert profile.detect()["driver_libs"] == {
        "path": str(nvidia),
        "linker": True,
        "via": "the linker's cache",
    }
    # Not on LD_LIBRARY_PATH (a shell without Colab's) nor in the cache: where Colab keeps them, which the linker
    # does not search; and Debian's multiarch directory, which it searches whether or not its cache lists it.
    cache.write_text("1 libs found in cache `/etc/ld.so.cache'\n" + libc)
    monkeypatch.delenv("LD_LIBRARY_PATH")
    multiarch = libcuda(rootfs / "usr" / "lib" / "x86_64-linux-gnu")
    assert profile.detect()["driver_libs"] == {
        "path": str(multiarch),
        "linker": True,
        "via": "a known location",
    }
    colab = libcuda(rootfs / NVIDIA_LIBS.lstrip("/"))
    assert profile.detect()["driver_libs"] == {
        "path": str(colab),
        "linker": False,
        "via": "a known location",
    }
    # The facts render a node whose GPU units name it.
    out, _ = render(tmp_path, "a100-40gb", host(driver_libs=profile.detect()["driver_libs"]))
    for unit in ("lh-engine", "lh-stt", "lh-tts"):
        lines = (out / "systemd" / f"{unit}.service").read_text().splitlines()
        assert f"Environment=LD_LIBRARY_PATH={colab}" in lines, unit


def test_the_hub_cache_is_resolved_as_huggingface_hub_does_and_rendered_into_the_vllm_units(
    tmp_path: Path, monkeypatch
) -> None:
    # The operator's HF_HOME (a data volume, often where the room is) is where the models land; the units run
    # offline from the same cache, so detect records it and the engine's and the STT's units name it.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    assert profile.detect()["hf_hub_cache"] == str(
        tmp_path / "home" / ".cache" / "huggingface" / "hub"
    )
    monkeypatch.setenv("XDG_CACHE_HOME", "/scratch/cache")
    assert profile.detect()["hf_hub_cache"] == "/scratch/cache/huggingface/hub"
    monkeypatch.setenv("HF_HOME", "/data/hf")
    assert profile.detect()["hf_hub_cache"] == "/data/hf/hub"
    monkeypatch.setenv("HF_HUB_CACHE", "/data/hub")
    assert profile.detect()["hf_hub_cache"] == "/data/hub"
    out, _ = render(tmp_path, "a100-40gb", host(hf_hub_cache="/data/hf/hub"))
    for unit in ("lh-engine", "lh-stt"):
        lines = (out / "systemd" / f"{unit}.service").read_text().splitlines()
        assert 'Environment="HF_HUB_CACHE=/data/hf/hub"' in lines, unit
        assert (
            lines.index('Environment="HF_HUB_CACHE=/data/hf/hub"')
            == lines.index("Environment=HF_HUB_OFFLINE=1") + 1
        )
    out, _ = render(
        tmp_path / "unknown", "a100-40gb", host()
    )  # facts from before the cache was recorded
    assert "HF_HUB_CACHE" not in (out / "systemd" / "lh-engine.service").read_text()


def test_check_vllm_imports_vllm_although_torch_imports_cprofile(tmp_path: Path) -> None:
    # profile.py has the name of the standard library's profile module, which cProfile imports, and vLLM's import
    # imports cProfile (torch's dynamo, under vLLM's env_override). Run as a script, profile.py's directory led
    # sys.path, so cProfile got profile.py in the library's place: the rehearsal on the box failed in bootstrap.sh
    # gpu's check-vllm with "module 'profile' has no attribute 'run'". A vllm that imports cProfile, then exits 7,
    # stands in for vLLM.
    fake = tmp_path / "site" / "vllm"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("import cProfile\n\nraise SystemExit(7)\n")
    done = subprocess.run(
        [sys.executable, str(DEPLOY / "profile.py"), "check-vllm", str(tmp_path)],
        env=os.environ | {"PYTHONPATH": str(tmp_path / "site")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 7, done.stderr
