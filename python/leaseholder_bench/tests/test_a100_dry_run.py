"""The A100 run's core on a CPU, end to end (docs/HARDWARE.md, the core block by block): benchmarks/a100/run.sh runs
blocks 1 to 10 on a single host whose node is rendered from the A100's profile, with the real probes.sh,
calibration.sh (and its lh-forward from a loopback port to the engine's socket), campaign.sh --local and the core's
campaign files, against stand-ins for the engine (its endpoints here, on its Unix socket; its probes' result lines
from fakes/a100/python), lh-bench, AIPerf, nvidia-smi, systemctl and the node's variant.sh. A probe that dies and
an E2 point that fails each stop the run, and each rerun resumes where it stopped, measuring nothing twice; then
pull.sh copies the host's records, and analysis.py reads them back."""

from __future__ import annotations

import http.server
import importlib.util
import json
import os
import pwd
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FAKES = Path(__file__).parent / "fakes"
RUN = REPO / "benchmarks" / "a100" / "run.sh"
DASH = "\u2013"  # the range dash, as analysis.py writes it
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
GIB = 1 << 30
FACTS = {
    "hostname": "a100-dry",
    "gpu": {
        "name": "NVIDIA A100-SXM4-40GB",
        "memory_mib": 40960,
        "compute_cap": "8.0",
        "numa_node": None,
        "driver": "580.65.06",
    },
    "cpus": {"0": list(range(32))},
    "memory": {"total_bytes": 200 * GIB, "available_bytes": 190 * GIB, "nodes": {}},
    "shm": {"path": "/dev/shm", "total_bytes": 100 * GIB, "free_bytes": 100 * GIB},
    "cuda_toolkit": None,
}
"""An A100 host's facts, as host.py's detect gives them (test_profile.py's host())."""
PR = "make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48"
S2048 = "frozen --max-num-batched-tokens 2048 --long-prefill-token-threshold 0"
S1536 = "frozen --max-num-batched-tokens 1536 --long-prefill-token-threshold 768"
pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("curl") is None,
    reason="the run's scripts need bash and curl",
)


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profile = _module("lh_profile", REPO / "deploy" / "profile.py")
analysis = _module("lh_a100_analysis", REPO / "benchmarks" / "a100" / "analysis.py")
host_py = _module("lh_a100_host", REPO / "deploy" / "a100" / "host.py")


class Engine(http.server.BaseHTTPRequestHandler):
    """The engine's /v1/models, /metrics and /health, as the run's scripts read them."""

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": MODEL, "max_model_len": 32768}]})
        elif self.path == "/metrics":
            body = "vllm:num_requests_running 0.0\nvllm:num_requests_waiting 0.0\n"
        elif self.path == "/health":
            body = ""
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args) -> None:
        pass


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


@pytest.fixture
def runtime() -> Iterator[Path]:
    """XDG_RUNTIME_DIR, with the engine on lh/engine.sock as lh-engine.service serves it."""
    # A short path: a socket's must fit sun_path, which a test's tmp_path may not.
    run = Path(tempfile.mkdtemp(prefix="lh-dry-"))
    (run / "lh").mkdir(mode=0o700)
    server = UnixServer(str(run / "lh" / "engine.sock"), Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield run
    server.shutdown()
    shutil.rmtree(run, ignore_errors=True)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def dry_env(home: Path, hosts: Path, runtime: Path) -> dict[str, str]:
    """The environment of the dry run's stand-ins for run.sh on HOME."""
    env = os.environ | {
        "HOME": str(home),
        "PATH": os.pathsep.join([str(FAKES / "a100"), str(FAKES / "campaign"), os.environ["PATH"]]),
        "FAKE_HOSTS": str(hosts),
        "FAKE_LH_BENCH": str(Path(sys.executable).with_name("lh-bench")),
        "FAKE_PYTHON": sys.executable,
        "XDG_RUNTIME_DIR": str(runtime),
        "LH_FORWARD_PORT": str(free_port()),
    }
    for name in (
        "FAKE_PROBE_FAIL_AT",
        "FAKE_E2_FAIL",
        "LH_NODE_DIR",
        "LH_PROFILE",
        "LH_E2_RESULTS",
        "LH_GPU_ID",
    ):
        env.pop(name, None)
    return env


def a100_host(hosts: Path) -> Path:
    """The single host's home: the node rendered from a100-40gb with the stand-in variant.sh, the venvs' stand-ins,
    the corpus and the host settings host.py config writes."""
    home = hosts / "localhost"
    node = home / "leaseholder" / "node"
    profile.render(profile.load("a100-40gb"), FACTS, node)
    shutil.copy(FAKES / "campaign" / "variant.sh", node / "variant.sh")
    (home / "leaseholder" / "src").mkdir(parents=True)
    (home / "leaseholder" / "src" / "DEPLOYED").write_text("v0.3-dry\n")
    engine_bin = home / "leaseholder" / "venvs" / "engine" / "bin"
    engine_bin.mkdir(parents=True)
    shutil.copy(FAKES / "a100" / "python", engine_bin / "python")
    bench_bin = home / "leaseholder" / "venvs" / "bench" / "bin"
    bench_bin.mkdir(parents=True)
    shutil.copy(FAKES / "campaign" / "lh-bench", bench_bin / "lh-bench")
    shutil.copy(FAKES / "a100" / "aiperf", bench_bin / "aiperf")
    (bench_bin / "python").symlink_to(sys.executable)
    (home / "leaseholder" / "data" / "weka-x32").mkdir(parents=True)
    config = home / ".config" / "leaseholder"
    config.mkdir(parents=True)
    (config / "e2-host.toml").write_text('aiperf_cores = "21-27"\ncaller_cores = "28-31"\n')
    return home


def lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def ttft(path: Path) -> int:
    """The voice turn's first token in a probe.py run's output."""
    return analysis.results.parse_probe(path.read_text()).ttft_ms


def test_the_core_runs_in_order_resumes_after_each_stop_and_reads_back(
    tmp_path: Path, runtime: Path
) -> None:
    hosts = tmp_path / "hosts"
    home = a100_host(hosts)
    env = os.environ | {
        "HOME": str(home),
        "PATH": os.pathsep.join([str(FAKES / "a100"), str(FAKES / "campaign"), os.environ["PATH"]]),
        "FAKE_HOSTS": str(hosts),
        "FAKE_LH_BENCH": str(Path(sys.executable).with_name("lh-bench")),
        "FAKE_PYTHON": sys.executable,
        "XDG_RUNTIME_DIR": str(runtime),
        "LH_FORWARD_PORT": str(port := free_port()),
    }
    for name in (
        "FAKE_PROBE_FAIL_AT",
        "FAKE_E2_FAIL",
        "LH_NODE_DIR",
        "LH_PROFILE",
        "LH_E2_RESULTS",
    ):
        env.pop(name, None)

    def run(**extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(RUN), *map(str, range(1, 11))],
            env=env | extra,
            capture_output=True,
            text=True,
            timeout=600,
        )

    runs = home / "leaseholder" / "runs"
    blocks = runs / "a100"

    def done() -> list[int]:
        return sorted(int(p.stem) for p in (blocks / "blocks").glob("*.done"))

    # Block 1 runs 8 probes; the 11th, block 2's third, dies as with its engine, and the run stops after block 1.
    first = run(FAKE_PROBE_FAIL_AT="11")
    assert first.returncode == 1, first.stdout + first.stderr
    assert done() == [1]
    assert sorted(p.name for p in (blocks / "block2").glob("stock*")) == [
        "stock2048-kv-1.txt",
        "stock2048-kv-2.txt",
        "stock2048-kv-3.partial",
    ]
    # The rerun resumes at that probe; round 1's frozen point fails, and the run stops in block 4.
    second = run(FAKE_E2_FAIL="w5-disc-frozen-s24")
    assert second.returncode == 1, second.stdout + second.stderr
    assert done() == [1, 2, 3]
    third = run()
    assert third.returncode == 0, third.stdout + third.stderr
    assert done() == list(range(1, 11))
    assert "run: complete: 1 2 3 4 5 6 7 8 9 10" in (blocks / "blocks.log").read_text()

    # Every probe ran once, and the one that died once more: three blocks of 8, and the attempt. The stand-in's
    # first-token times count the probes in order, so each output is the one its probe first wrote.
    assert len(lines(home / "probe-calls")) == 25
    for block, names in (
        ("block1", ["frozen-kv-1", "frozen-kv-3", "frozen-prefill-3", "frozen-tail-2"]),
        (
            "block2",
            ["stock2048-kv-3", "stock2048-tail-2", "stock1536-768-kv-1", "stock1536-768-tail-2"],
        ),
        ("block3", ["pr-kv-3", "pr-prefill-3", "pr-tail-2"]),
    ):
        for name in names:
            assert (blocks / block / f"{name}.txt").exists(), name
    assert not list(blocks.glob("block*/*.partial"))
    assert ttft(blocks / "block1" / "frozen-kv-1.txt") == 1650 + 1  # run 1's first probe
    assert (
        ttft(blocks / "block2" / "stock2048-kv-2.txt") == 1650 + 10
    )  # written before the probe that died
    assert (
        ttft(blocks / "block2" / "stock2048-kv-3.txt") == 1650 + 12
    )  # run 2's first probe: the one that died
    assert (
        ttft(blocks / "block3" / "pr-kv-1.txt") == 150 + 18
    )  # make-room's, behind the same agents
    assert json.loads((blocks / "block1" / "frozen-e0" / "summary.json").read_text())["fits"]

    # The engines, in the order the blocks and rounds start them: the probe blocks' four settings; round 1 on block
    # 3's engine; the calibration's frozen engine; then each round in its order, starting on the engine the last
    # round ended on. Each point's switch turns the measurement drop-in on first, as campaign.sh does.
    switches = lines(home / "variant.log")
    engines = [
        s.removeprefix("variant.sh use ") for s in switches if s.startswith("variant.sh use ")
    ]
    assert switches.count("variant.sh measure on") == 16 and len(switches) == len(engines) + 16
    assert engines == [
        "frozen",  # block 1
        S2048,  # block 2, in the run that stopped at its third probe
        S2048,  # block 2 again
        S1536,
        PR,  # block 3
        PR,  # round 1: the PRs' values, voice alone, frozen (the point that failed)
        "frozen",
        "frozen",
        "frozen",  # round 1 again: only its frozen point
        "frozen",  # block 5, the calibration
        *("frozen", "frozen", PR),  # round 2
        *(PR, "frozen", "frozen"),  # round 3
        *("frozen", "frozen", PR),  # round 4
        *(PR, "frozen", "frozen"),  # round 5
    ]
    points = [line for line in lines(home / "lh-bench.log") if line.startswith("lh-bench e2 run ")]
    assert len(points) == 16  # the 15 points, and round 1's frozen point twice
    for n in range(1, 6):
        assert len(list(runs.glob(f"e2/a100-r{n}/*/run.json"))) == 3
        mine = [p for p in points if f"/a100-r{n}/" in p]
        assert mine and all(f"--agents-seed {10 + n}" in p for p in mine)
    complete = '"status": "complete"'
    assert complete in (runs / "e2" / "a100-r1" / "w5-disc-frozen-s24" / "run.json").read_text()

    # The calibration point, as calibration.py reads a run: AIPerf's replay straight to the freshly started engine.
    point = blocks / "calibration-s16"
    assert (point / "before.prom").read_text() and (point / "after.prom").read_text()
    assert len(lines(point / "aiperf" / "profile_export.jsonl")) == 3
    settings = lines(point / "aiperf" / "env.txt")
    for setting in (
        f"LH_BENCH_URL=http://127.0.0.1:{port}/v1/chat/completions",  # lh-forward's end of the socket
        "LH_BENCH_SESSIONS=16",
        "LH_BENCH_DURATION=600",
        "LH_BENCH_GRACE=60",
        "LH_BENCH_THINK_CAP=120",
        "LH_BENCH_SEED=11",
        "LH_BENCH_BURST_STARTS=true",
        f"LH_BENCH_MODEL={MODEL}",
        f"LH_BENCH_DATA={home}/leaseholder/data/weka-x32",
    ):
        assert setting in settings, setting
    assert (
        lines(home / "taskset.log") == ["taskset -c 21-27"] * 2
    )  # the relay and AIPerf, on AIPerf's cores
    assert "127.0.0.1:" + str(port) + " -> unix:" in (point / "forward.log").read_text()
    # Each run first brings the control plane up, which a reboot leaves down (starting it running is nothing).
    up = "systemctl --user start lh-voice-worker.service"
    assert lines(home / "systemctl.log") == [
        up,
        up,
        up,
        "systemctl --user reset-failed lh-engine.service",
        "systemctl --user restart lh-engine.service",
        "systemctl --user stop leaseholder.target lh-engine lh-stt lh-tts",  # block 10
    ]

    # The workstation's copy, and the analysis of it.
    pulled = tmp_path / "pulled"
    pull = subprocess.run(
        ["bash", str(REPO / "deploy" / "a100" / "pull.sh"), "--once", "localhost", str(pulled)],
        env=env | {"LH_SSH": str(FAKES / "campaign" / "ssh")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert pull.returncode == 0, pull.stdout + pull.stderr
    assert json.loads((pulled / "node" / "node.json").read_text())["profile"] == "a100-40gb"
    assert len(list(pulled.glob("results/e2/a100-r*/*/run.json"))) == 15
    assert (pulled / "runs" / "a100" / "calibration-s16" / "after.prom").exists()
    lh_sim = tmp_path / "no-lh-sim"
    text = analysis.report(
        pulled, tmp_path / "corpus", lh_sim, tmp_path / "report", None
    ).read_text()
    # The calibration point and E0 are there; lh-sim is not.
    assert f"## 1. The step model\n\nNot computed: no lh-sim at {lh_sim}" in text
    assert "## 2. The calibration table\n\nNot computed." in text
    # The probes' cases, each on the engine its block ran: about 1.65 s on stock, about 0.15 s on the PRs' values.
    for row in (
        f"| KV case, stock: the voice turn's wait | 1,652 ms (1,651{DASH}1,653, 3 runs) |",
        f"| KV case, stock at 2,048/0 | 1,660 ms (1,659{DASH}1,662, 3 runs) |",
        f"| KV case, the PRs' values: first token | 169 ms (168{DASH}170, 3 runs) |",
        "| block2/stock1536-768-kv-1 | kv | 1665 | 1765 |",
    ):
        assert row in text, row
    assert (
        "The calls' first-token p95 beside the agents (tail.py): "
        f"frozen, 512/256 1,858 ms (1,857{DASH}1,858, 2 runs); stock, 2,048/0 1,864 ms (1,863{DASH}1,864, 2 runs); "
        f"stock, 1,536/768 1,866 ms (1,866{DASH}1,867, 2 runs); the PRs' values 344 ms (344{DASH}345, 2 runs)."
    ) in text
    assert "## 4. E2 within the A100\n\nThe measured running share" in text
    assert "- Not measured: no point's summary.json has a running share" in text


@pytest.mark.skipif(shutil.which("rsync") is None, reason="push.sh copies with rsync")
def test_push_copies_the_corpora_as_files_that_outlive_the_workstations_paths(
    tmp_path: Path,
) -> None:
    # deploy/a100/push.sh into a "host", a local directory (fakes/campaign/ssh runs its commands there; rsync is
    # the real one): the source tree and the data, whose weka-x32-long is long_traces.py's links into the
    # workstation's weka-x32 by absolute path. With the workstation's data moved away, every trace on the host is
    # a file it can read, which the preflight's corpus row counts, an earlier push's links included.
    repo = tmp_path / "repo"
    (repo / "deploy" / "a100").mkdir(parents=True)
    shutil.copy(REPO / "deploy" / "a100" / "push.sh", repo / "deploy" / "a100" / "push.sh")
    (repo / "README.md").write_text("the source tree\n")
    data = tmp_path / "workstation" / "data"
    traces = data / "weka-x32"
    traces.mkdir(parents=True)
    for n in range(1, 5):
        (traces / f"trace-{n}.json").write_text(json.dumps({"requests": [{"in": 1000 * n}]}))
    long_traces = REPO / "benchmarks" / "e2" / "long_traces.py"
    made = subprocess.run(
        [sys.executable, str(long_traces), str(traces), str(data / "weka-x32-long"), "2"],
        capture_output=True,
        text=True,
    )
    assert made.returncode == 0, made.stderr
    assert all(p.is_symlink() for p in (data / "weka-x32-long").iterdir())
    (data / "callers").mkdir()
    (data / "callers" / "manifest.json").write_text(json.dumps({"hello": {"kind": "turn"}}))
    (data / "callers" / "hello.wav").write_bytes(b"RIFF")
    hosts = tmp_path / "hosts"
    root = hosts / "a100" / "leaseholder"
    # An earlier copy, as rsync -a makes one: the links as links.
    (root / "data").mkdir(parents=True)
    old = subprocess.run(
        ["rsync", "-a", str(data / "weka-x32-long"), f"{root / 'data'}/"],
        capture_output=True,
        text=True,
    )
    assert old.returncode == 0, old.stderr
    assert all(p.is_symlink() for p in (root / "data" / "weka-x32-long").iterdir())
    env = {k: v for k, v in os.environ.items() if k != "LH_GATEWAY_BIN"} | {
        "FAKE_HOSTS": str(hosts),
        "LH_SSH": str(FAKES / "campaign" / "ssh"),
        "LH_DATA": str(data),
    }
    pushed = subprocess.run(
        ["bash", str(repo / "deploy" / "a100" / "push.sh"), "a100"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert pushed.returncode == 0, pushed.stdout + pushed.stderr
    assert (root / "src" / "README.md").read_text() == "the source tree\n"
    assert (root / "src" / "DEPLOYED").exists()
    (tmp_path / "workstation").rename(tmp_path / "elsewhere")
    long = sorted((root / "data" / "weka-x32-long").iterdir())
    assert [p.name for p in long] == ["trace-3.json", "trace-4.json"]  # the two largest
    for path in long:
        assert not path.is_symlink() and json.loads(path.read_text())["requests"], path
    check = host_py.data_check(root)
    assert check.status == "ok", check
    assert check.value == "weka-x32-long 2 traces, weka-x32 4 traces, callers 1 clips"


BOOTSTRAP = REPO / "deploy" / "a100" / "bootstrap.sh"
HOST_TOOLS = {
    "loginctl": """\
#!/bin/sh
# loginctl: show-user reads the user's linger state; enable-linger sets it, as polkit's set-self-linger lets a user
# without sudo, unless FAKE_NO_LINGER refuses it.
state="$FAKE_HOST/linger"
case "$1" in
  show-user) cat "$state" 2>/dev/null || echo no ;;
  enable-linger)
    echo "loginctl $*" >>"$FAKE_HOST/calls"
    [ -z "$FAKE_NO_LINGER" ] || exit 1
    echo yes >"$state" ;;
esac
""",
    "systemctl": """\
#!/bin/sh
# The host's own systemctl: a user manager runs (the SSH session's); each call but is-system-running is logged.
case "$*" in
  *is-system-running*) echo running ;;
  *) echo "systemctl $*" >>"$FAKE_HOST/calls" ;;
esac
""",
    "nvidia-smi": "#!/bin/sh\nexit 1\n",  # staging: no card yet
    "sudo": "#!/bin/sh\nexit 1\n",
    "systemd-detect-virt": "#!/bin/sh\nexit 1\n",
    "unshare": "#!/bin/sh\nexit 1\n",
}
"""A stock Ubuntu host's tools as bootstrap.sh's flow meets them: a user manager without lingering, no sudo."""


@pytest.fixture
def ubuntu(tmp_path: Path) -> dict:
    """A host for bootstrap.sh's own steps: its home, the host's tools (HOST_TOOLS), and /proc, /sys and /dev/shm
    that its facts read (32 CPUs, 200 GiB of RAM; /dev/shm is the scratch file system)."""
    if shutil.disk_usage(tmp_path).free < 8 * GIB:
        pytest.skip("the node's render needs 8 GiB free where /dev/shm is faked (R3)")
    tools, state = tmp_path / "tools", tmp_path / "host"
    tools.mkdir()
    state.mkdir()
    for name, body in HOST_TOOLS.items():
        (tools / name).write_text(body)
        (tools / name).chmod(0o755)
    sysfs, procfs, rootfs = (tmp_path / d for d in ("sys", "proc", "root"))
    cpus = sysfs / "devices" / "system" / "node" / "node0"
    cpus.mkdir(parents=True)
    (cpus / "cpulist").write_text("0-31\n")
    (procfs / "1").mkdir(parents=True)
    (procfs / "1" / "cgroup").write_text("0::/init.scope\n")
    (procfs / "meminfo").write_text("MemTotal: 209715200 kB\nMemAvailable: 199229440 kB\n")
    rootfs.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LH_", "XDG_"))} | {
        "HOME": str(home),
        "USER": "ops",
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "FAKE_HOST": str(state),
        "LH_SYSFS": str(sysfs),
        "LH_PROCFS": str(procfs),
        "LH_ROOTFS": str(rootfs),
        "LH_SHM": str(tmp_path),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    return {"home": home, "env": env, "state": state, "node": home / "leaseholder" / "node"}


def bootstrap(host: dict, *steps: str, **env: str) -> subprocess.CompletedProcess[str]:
    """One invocation of bootstrap.sh's own steps, sourced: its choices (the supervisor, the GPU), then STEPS, each a
    command under its set -e, as main runs a stage's (its tools aside: py is this Python)."""
    script = "\n".join(
        [f"source {BOOTSTRAP}", f"py={sys.executable}", "choose", 'mkdir -p "$node"', *steps]
    )
    return subprocess.run(
        ["bash", "-c", script],
        env=host["env"] | env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def calls(host: dict) -> list[str]:
    return lines(host["state"] / "calls")


def test_bootstrap_lingers_and_chooses_the_supervisor_once_in_the_first_stage(ubuntu: dict) -> None:
    # A stock Ubuntu host: the SSH user's manager runs, lingering is off, and a user may enable its own. The first
    # stage enables it and reads the facts again at once, so systemd runs the units from the start; the gpu stage
    # finds the same, with nothing to enable or to move.
    node = ubuntu["node"]
    for invocation in ("stage", "gpu"):
        done = bootstrap(ubuntu, "render", "shell_env")
        assert done.returncode == 0, (invocation, done.stdout + done.stderr)
        facts = json.loads((node / "host.json").read_text())
        assert facts["systemd_user"] and facts["linger"], invocation
        env_sh = (node / "env.sh").read_text()
        assert "systemd runs the units" in env_sh and "shims" not in env_sh, (invocation, env_sh)
        assert "export LH_SUPERVISOR=systemd\n" in env_sh, invocation  # kept for the next stages
    user = pwd.getpwuid(os.getuid()).pw_name  # id -un
    assert [c for c in calls(ubuntu) if c.startswith("loginctl")] == [
        f"loginctl enable-linger {user}"
    ]
    assert not [c for c in calls(ubuntu) if c.startswith("systemctl")]  # nothing stopped


def test_bootstrap_keeps_lh_svc_once_chosen_until_auto_moves_the_units(ubuntu: dict) -> None:
    # Lingering refused at the first stage: lh-svc runs the units. Enabled later (an administrator), the facts would
    # choose systemd, but the next stages keep lh-svc, whose units run the control plane; LH_SUPERVISOR=auto moves
    # them, stopping lh-svc's first, so that no unit runs under both.
    node, home = ubuntu["node"], ubuntu["home"]
    done = bootstrap(ubuntu, "render", "shell_env", FAKE_NO_LINGER="1")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "deploy/bin/shims" in (node / "env.sh").read_text()
    assert "export LH_SUPERVISOR=lh-svc\n" in (node / "env.sh").read_text()
    # The units, as host.py units links them (stand-ins), and the control plane started under lh-svc.
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    for unit in host_py.UNITS:
        (units / unit).write_text(
            f"[Service]\nExecStart={sys.executable} -c 'import time; time.sleep(60)'\n"
            if unit.endswith(".service")
            else ""
        )
    shim = REPO / "deploy" / "bin" / "shims" / "systemctl"
    lh_svc = lambda *args: subprocess.run(  # noqa: E731
        ["bash", "-c", f"source {node / 'env.sh'} && {shim} --user {' '.join(args)}"],
        env=ubuntu["env"],
        capture_output=True,
        text=True,
    )
    assert lh_svc("start", "lh-gateway.service").returncode == 0
    try:
        (ubuntu["state"] / "linger").write_text("yes\n")
        done = bootstrap(ubuntu, "render", "shell_env")
        assert done.returncode == 0, done.stdout + done.stderr
        assert json.loads((node / "host.json").read_text())["linger"] is True
        env_sh = (node / "env.sh").read_text()
        assert "deploy/bin/shims" in env_sh and "export LH_SUPERVISOR=lh-svc\n" in env_sh
        assert lh_svc("is-active", "lh-gateway.service").stdout.strip() == "active"
        moved = bootstrap(ubuntu, "render", "shell_env", LH_SUPERVISOR="auto")
        assert moved.returncode == 0, moved.stdout + moved.stderr
        assert "from lh-svc to systemd" in moved.stdout
        assert lh_svc("is-active", "lh-gateway.service").stdout.strip() == "inactive"
        env_sh = (node / "env.sh").read_text()
        assert "systemd runs the units" in env_sh and "export LH_SUPERVISOR=systemd\n" in env_sh
    finally:
        lh_svc("stop", "lh-gateway.service")


FAKE_UV = """\
#!/bin/sh
# uv for bootstrap.sh's venv steps: logs each call; venv DIR makes DIR/bin/python, a stand-in whose headers' probe
# answers yes when the call asked for uv's own Python (--managed-python), no otherwise (a system Python 3.12's).
echo "uv $*" >>"$FAKE_HOST/uv.log"
if [ "$1" = venv ]; then
  for dir; do :; done
  mkdir -p "$dir/bin"
  case " $* " in *" --managed-python "*) headers=True ;; *) headers=False ;; esac
  printf '#!/bin/sh\necho %s/include\necho %s\n' "$dir" "$headers" >"$dir/bin/python"
  chmod +x "$dir/bin/python"
fi
"""


def test_bootstrap_makes_the_venvs_from_uvs_own_python(ubuntu: dict) -> None:
    # On Ubuntu 24.04 uv prefers the system's Python 3.12, which has no headers there (libpython3.12-dev is not
    # installed), so Triton could not build its launcher at the engine's first start. The venvs come from uv's own
    # Python, and an engine venv an earlier stage made from the system's is made anew.
    node, root = ubuntu["node"], ubuntu["home"] / "leaseholder"
    (ubuntu["state"] / "uv").write_text(FAKE_UV)
    (ubuntu["state"] / "uv").chmod(0o755)
    node.mkdir(parents=True)
    (node / "node.json").write_text(json.dumps({"install": {"path": "cu130"}}))
    old = root / "venvs" / "engine" / "bin"
    old.mkdir(parents=True)
    (old / "python").write_text("#!/bin/sh\necho /usr/include/python3.12\necho False\n")
    (old / "python").chmod(0o755)
    done = bootstrap(ubuntu, f"uv={ubuntu['state'] / 'uv'}", "venvs")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "has no Python.h" in done.stdout
    log = lines(ubuntu["state"] / "uv.log")
    assert log[0] == f"uv venv -q --managed-python --python 3.12 {root}/venvs/engine", log
    assert log[1].startswith(
        f"uv pip sync -q --python {root}/venvs/engine/bin/python --torch-backend cu130 "
    )
    assert (
        subprocess.run([str(old / "python")], capture_output=True, text=True).stdout.split()[1]
        == "True"
    )
    # Every uv command of bootstrap.sh's that chooses an interpreter asks for uv's own.
    for line in BOOTSTRAP.read_text().splitlines():
        if re.search(r'"\$uv" (venv|sync|python find)\b', line):
            assert "--managed-python" in line, line


TWO_GPUS = """\
#!/bin/sh
# nvidia-smi on a host with two A100s.
case "$*" in
  -L) printf 'GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-aaaa)\\nGPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-bbbb)\\n' ;;
  --query-gpu=*)
    for row in "0, GPU-aaaa" "1, GPU-bbbb"; do
      echo "$row, NVIDIA A100-SXM4-40GB, 40960, 8.0, 580.65.06, 4, 16, Disabled, 400.00, 00000000:3B:00.0, Enabled, Default"
    done ;;
  *) exit 1 ;;
esac
"""


def test_bootstrap_keeps_the_lent_gpu_that_the_first_stage_named(ubuntu: dict) -> None:
    # LH_GPU_ID=1 at the first stage: the facts resolve it to its UUID, env.sh keeps that for the later stages,
    # which then need not name it again; a bad value is refused before anything is written.
    tools = Path(ubuntu["env"]["PATH"].split(os.pathsep)[0])
    (tools / "nvidia-smi").write_text(TWO_GPUS)
    node = ubuntu["node"]
    done = bootstrap(ubuntu, "render", "shell_env", LH_GPU_ID="1")
    assert done.returncode == 0, done.stdout + done.stderr
    for invocation in ("stage", "gpu"):
        facts = json.loads((node / "host.json").read_text())["gpu"]
        assert (facts["uuid"], facts["chosen"]) == ("GPU-bbbb", True), invocation
        assert "export LH_GPU_ID=GPU-bbbb\n" in (node / "env.sh").read_text(), invocation
        units = (node / "systemd" / "lh-engine.service").read_text()
        assert "Environment=CUDA_VISIBLE_DEVICES=GPU-bbbb" in units.splitlines(), invocation
        done = bootstrap(ubuntu, "render", "shell_env")
        assert done.returncode == 0, done.stdout + done.stderr
    bad = bootstrap(ubuntu, "true", LH_GPU_ID="all")
    assert bad.returncode == 2 and "LH_GPU_ID is a GPU's index or UUID" in bad.stderr
    # LH_ALLOW_MIG=1, a choice like the GPU's, is kept for the later stages and the run's block 0 too.
    for env in ({"LH_ALLOW_MIG": "1"}, {}):
        done = bootstrap(ubuntu, "render", "shell_env", **env)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "export LH_ALLOW_MIG=1\n" in (node / "env.sh").read_text(), env


def test_the_gpu_stage_stops_the_gpu_units_before_it_reads_the_facts(ubuntu: dict) -> None:
    # In block 0 the engine that warm left running holds its /dev/shm region and RAM, which R3 bounds the CPU tier
    # by: the gpu stage stops the GPU units (block 0 starts them again) before it detects, so that the facts are the
    # host's own. The staged node is the one the tier is compared with.
    node = ubuntu["node"]
    done = bootstrap(ubuntu, "stage_facts")
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads((node / "staged.json").read_text()) == json.loads(
        (node / "node.json").read_text()
    )
    calls = ubuntu["state"] / "calls"
    calls.unlink(missing_ok=True)
    tools = Path(ubuntu["env"]["PATH"].split(os.pathsep)[0])
    (tools / "systemctl").write_text(
        '#!/bin/sh\necho "systemctl $*" >>"$FAKE_HOST/calls"\necho running\n'
    )
    done = bootstrap(ubuntu, "gpu_facts")
    assert done.returncode == 0, done.stdout + done.stderr
    asked = lines(calls)
    stop = "systemctl --user stop leaseholder.target lh-engine lh-stt lh-tts"
    assert stop in asked, asked
    assert asked.index(stop) < asked.index("systemctl --user is-system-running"), asked


def test_a_failed_point_under_a_deadline_stops_the_run_as_a_failure(
    tmp_path: Path, runtime: Path
) -> None:
    # The documented invocation passes --until: a point that fails far from the deadline is a failure (exit 1, with
    # campaign.sh's status), not a stop before the deadline.
    hosts = tmp_path / "hosts"
    home = a100_host(hosts)
    env = dry_env(home, hosts, runtime) | {
        "FAKE_E2_FAIL": "w5-disc-pr-ab48-s24"
    }  # round 1's first point
    until = (datetime.now(UTC) - timedelta(minutes=30)).strftime("%H:%M")  # about 23.5 hours away
    done = subprocess.run(
        ["bash", str(RUN), "--until", until, "4"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    blocks = (home / "leaseholder" / "runs" / "a100" / "blocks.log").read_text()
    assert done.returncode == 1, done.stdout + done.stderr
    assert "a100-r1 did not complete (campaign.sh exit status 1)" in blocks
    assert "stopped before its deadline" not in blocks


def test_bootstrap_keeps_the_hub_cache_the_first_stage_resolved(ubuntu: dict) -> None:
    # The first stage runs with the operator's HF_HOME on a data volume: the hub cache is resolved there once, the
    # units and every script that sources env.sh read it there, and a later stage without HF_HOME keeps it.
    node = ubuntu["node"]
    for invocation, env in (("stage", {"HF_HOME": "/data/hf"}), ("gpu", {})):
        done = bootstrap(ubuntu, "render", "shell_env", **env)
        assert done.returncode == 0, (invocation, done.stdout + done.stderr)
        assert json.loads((node / "host.json").read_text())["hf_hub_cache"] == "/data/hf/hub", (
            invocation
        )
        assert "export HF_HUB_CACHE=/data/hf/hub\n" in (node / "env.sh").read_text(), invocation
        engine = (node / "systemd" / "lh-engine.service").read_text().splitlines()
        assert 'Environment="HF_HUB_CACHE=/data/hf/hub"' in engine, invocation


FAKE_VLLM = """\
#!/usr/bin/env bash
# The engine venv's vllm in the batch arm's dry run: `vllm bench serve` logs how it was asked, holds its clients a
# moment, and writes its result where --result-dir and --result-filename say.
echo "vllm $*" >>"$HOME/vllm.log"
dir="" file=""
while (($#)); do
  case $1 in
    --result-dir) dir=$2; shift ;;
    --result-filename) file=$2; shift ;;
  esac
  shift
done
sleep 1.5
echo '{"output_throughput": 9000.0, "completed": 20}' >"$dir/$file"
echo "Output token throughput (tok/s):  9000.0"
"""


class BatchEngine(http.server.BaseHTTPRequestHandler):
    """The engine as the batch arm reads it: its model, its KV geometry (vllm:cache_config_info) and its
    running and waiting requests, as the batch formed."""

    blocks = 2275

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": MODEL}]})
        elif self.path == "/metrics":
            body = (
                f'vllm:cache_config_info{{block_size="16",engine="0",num_gpu_blocks="{self.blocks}"}} 1.0\n'
                'vllm:num_requests_running{engine="0",model_name="m"} 256.0\n'
                'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0\n'
            )
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args) -> None:
        pass


def test_the_batch_arm_holds_its_whole_batch_in_the_pool_and_records_that_it_formed(
    tmp_path: Path,
) -> None:
    # vllm bench serve's 256 clients on the core's engine: each request must hold its KV beside the others' in the
    # 5 GiB pool (2,275 blocks of 16), so that 256 run at once and the arm measures the scheduler at that batch, not
    # a queue; the running gauge, sampled while the clients run, shows it did. A pool too small for the batch is
    # refused before a client starts.
    run = Path(tempfile.mkdtemp(prefix="lh-batch-"))
    (run / "lh").mkdir(mode=0o700)
    server = UnixServer(str(run / "lh" / "engine.sock"), BatchEngine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        hosts = tmp_path / "hosts"
        home = a100_host(hosts)
        vllm = home / "leaseholder" / "venvs" / "engine" / "bin" / "vllm"
        vllm.write_text(FAKE_VLLM)
        vllm.chmod(0o755)
        env = dry_env(home, hosts, run)
        batch = REPO / "benchmarks" / "a100" / "batch.sh"
        done = subprocess.run(
            ["bash", str(batch), "--rounds", "1"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        calls = lines(home / "vllm.log")
        assert len(calls) == 6  # two clients on each of three engines
        for call in calls:
            assert "--random-input-len 48 --random-output-len 64 " in call, call
        out = home / "leaseholder" / "runs" / "a100" / "batch"
        for name in ("frozen", "off", "on"):
            running = json.loads((out / f"{name}-running.json").read_text())
            assert running["samples"] >= 1 and running["running_max"] == 256, running
            assert running["blocks"] == 2275 and running["blocks_needed"] == 2048, running
        assert "running p50 256, max 256 (the batch: 256)" in (out / "batch.log").read_text()
        BatchEngine.blocks = 1000
        for name in ("frozen", "off", "on"):
            for kept in out.glob(f"{name}-*.json"):
                kept.unlink()
        small = subprocess.run(
            ["bash", str(batch), "--rounds", "1"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert small.returncode == 2, small.stdout + small.stderr
        assert (
            "the pool holds 1000 blocks of 16; 256 requests of 112 tokens need 2048" in small.stdout
        )
        assert len(lines(home / "vllm.log")) == 6  # no client started
    finally:
        BatchEngine.blocks = 2275
        server.shutdown()
        shutil.rmtree(run, ignore_errors=True)


def deadline_of(until: str) -> int:
    """run.sh's deadline for --until HH:MM: today's, or tomorrow's once today's has passed."""
    now = int(datetime.now(UTC).timestamp())
    hours, minutes = map(int, until.split(":"))
    deadline = now - now % 86400 + hours * 3600 + minutes * 60
    return deadline if deadline > now else deadline + 86400


def test_the_run_hands_its_one_deadline_to_every_campaign_and_to_cprime(
    tmp_path: Path, runtime: Path
) -> None:
    # One deadline, as an epoch, from run.sh: a campaign takes it as it is (one that ends after HH:MM hands the next
    # no deadline 24 hours away), and so does cprime.sh, whose stop before the deadline leaves its block undone.
    hosts = tmp_path / "hosts"
    home = a100_host(hosts)
    env = dry_env(home, hosts, runtime)
    until = (datetime.now(UTC) + timedelta(hours=3)).strftime("%H:%M")
    done = subprocess.run(
        ["bash", str(RUN), "--until", until, "4"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    log = (home / "leaseholder" / "results" / "e2" / "a100-r1" / "campaign.log").read_text()
    assert f"until {until} UTC (epoch {deadline_of(until)})" in log
    # cprime.sh, beside a copy of run.sh: what it is handed, and what its exit status makes of the block.
    tree = tmp_path / "tree" / "benchmarks" / "a100"
    tree.mkdir(parents=True)
    shutil.copy(RUN, tree / "run.sh")
    (tree / "cprime.sh").write_text(
        'echo "cprime.sh $*" >>"$HOME/cprime.log"\nexit "${FAKE_CPRIME_STATUS:-0}"\n'
    )
    blocks = home / "leaseholder" / "runs" / "a100" / "blocks"
    for status, code in (("3", 0), ("1", 1), ("0", 0)):
        done = subprocess.run(
            ["bash", str(tree / "run.sh"), "--until", until, "cprime"],
            env=env | {"FAKE_CPRIME_STATUS": status},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert done.returncode == code, (status, done.stdout + done.stderr)
        assert (blocks / "cprime.done").exists() == (status == "0"), status
    assert (
        lines(home / "cprime.log")
        == [f"cprime.sh --cap 1536 --until-epoch {deadline_of(until)}"] * 3
    )
    assert (
        "cprime stopped before its deadline"
        in (home / "leaseholder" / "runs" / "a100" / "blocks.log").read_text()
    )


def test_a_cap_of_2048_is_measured_beside_vllms_default(tmp_path: Path, runtime: Path) -> None:
    # --cap 2048, the forecast's low value: block 2's second setting is 2,048/1,024, whose runs are its own, not
    # 2,048/0's under the same names (skipped as done, the engine restarted for nothing).
    hosts = tmp_path / "hosts"
    home = a100_host(hosts)
    done = subprocess.run(
        ["bash", str(RUN), "--cap", "2048", "2"],
        env=dry_env(home, hosts, runtime),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    block2 = home / "leaseholder" / "runs" / "a100" / "block2"
    assert sorted(p.name for p in block2.glob("stock2048-1024-*.txt")) == [
        "stock2048-1024-kv-1.txt",
        "stock2048-1024-tail-1.txt",
        "stock2048-1024-tail-2.txt",
    ]
    assert len(list(block2.glob("stock2048-kv-*.txt"))) == 3
    switches = [s for s in lines(home / "variant.log") if s.startswith("variant.sh use ")]
    assert switches == [
        "variant.sh use frozen --max-num-batched-tokens 2048 --long-prefill-token-threshold 0",
        "variant.sh use frozen --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024",
    ]


def test_pull_keeps_the_units_logs_as_the_run_goes(tmp_path: Path, runtime: Path) -> None:
    # The engine's and the control plane's logs are in the host's journal (lh-svc's log files under it), which
    # leaves with the host: each pull exports what the units logged since the last one (journalctl, the shims' under
    # lh-svc), beside the runs it copies.
    hosts = tmp_path / "hosts"
    home = a100_host(hosts)
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "journalctl").write_text('#!/bin/sh\necho "journal $*"\n')
    (tools / "journalctl").chmod(0o755)
    env = dry_env(home, hosts, runtime) | {
        "LH_SSH": str(FAKES / "campaign" / "ssh"),
        "PATH": f"{tools}{os.pathsep}{dry_env(home, hosts, runtime)['PATH']}",
    }
    pulled = tmp_path / "pulled"
    for _ in range(2):
        done = subprocess.run(
            ["bash", str(REPO / "deploy" / "a100" / "pull.sh"), "--once", "localhost", str(pulled)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stdout + done.stderr
    exported = lines(pulled / "runs" / "journal" / "units.log")
    assert len(exported) == 2, exported
    units = "-u lh-engine -u lh-stt -u lh-tts -u lh-gateway -u lh-voice-worker -u lh-livekit -u lh-voicenet"
    first, second = exported
    assert first.startswith(f"journal --user {units} --since @0 --until @") and first.endswith(
        " -o short-unix"
    )
    until = first.split("--until ")[1].split()[0]
    assert second.startswith(f"journal --user {units} --since {until} --until @")


def test_bootstrap_installs_as_root_without_sudo(ubuntu: dict) -> None:
    # A container whose user is root and that has no sudo: apt-get runs as it is (build-essential, iproute2, git),
    # where it once needed `sudo -n`; a user who is not root still needs sudo.
    tools = Path(ubuntu["env"]["PATH"].split(os.pathsep)[0])
    log = ubuntu["state"] / "apt.log"
    (tools / "apt-get").write_text(f'#!/bin/sh\necho "apt-get $*" >>{log}\n')
    (tools / "apt-get").chmod(0o755)
    (tools / "id").write_text(
        '#!/bin/sh\nif [ "$1" = -u ]; then echo "${FAKE_UID:-0}"; else echo root; fi\n'
    )
    (tools / "id").chmod(0o755)
    done = bootstrap(ubuntu, "apt_install iproute2")
    assert done.returncode == 0, done.stdout + done.stderr
    assert lines(log) == ["apt-get update -q", "apt-get install -y -q iproute2"]
    refused = bootstrap(ubuntu, "apt_install iproute2 || echo refused", FAKE_UID="1000")
    assert refused.returncode == 0 and refused.stdout.strip().endswith("refused")
    assert len(lines(log)) == 2  # sudo -n fails: nothing installed
