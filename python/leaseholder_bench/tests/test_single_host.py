"""Single-host mode's machinery (docs/HARDWARE.md): deploy/bin/lh-svc with its systemctl and journalctl shims, which
run the repository's user units where there is no systemd user manager, against stand-in services and the box's own
variant.sh, and deploy/bin/lh-forward, which relays a loopback port to the engine's socket for AIPerf, which speaks
TCP only."""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

from leaseholder_bench.e2 import run as e2
from leaseholder_bench.e2.arm import load_arm

REPO = Path(__file__).resolve().parents[3]
BIN = REPO / "deploy" / "bin"


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


host_py = _module("lh_a100_host", REPO / "deploy" / "a100" / "host.py")
GIB = 1 << 30
FACTS = {
    "hostname": "a100-single",
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
"""An A100 host's facts, as host.py's detect gives them (test_a100_dry_run.py's)."""
SERVICE = """\
import os, signal, sys, time
from pathlib import Path
# A stand-in service: prints its arguments and environment, says it is ready, and runs until it is stopped; with
# STUBBORN it ignores the stop signals.
print("argv", sys.argv[1:], flush=True)
print("env", os.environ.get("GREETING"), os.environ.get("FROM_FILE"), os.environ.get("LH_SVC_UNIT"), flush=True)
if os.environ.get("STUBBORN"):
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, signal.SIG_IGN)
else:
    signal.signal(signal.SIGINT, lambda *_: (print("stopped by SIGINT", flush=True), sys.exit(0)))
Path(os.environ["READY"]).write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
"""
WAIT_READY = "import os, sys, time\nwhile not os.path.exists(sys.argv[1]): time.sleep(0.02)"
NOTIFY = """\
import os, socket, sys, time
from pathlib import Path
# A Type=notify stand-in, as the gateway: READY=1 to NOTIFY_SOCKET a moment after it starts, then it serves.
time.sleep(0.3)
Path(os.environ["MARK"]).write_text("ready")
with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify:
    notify.sendto(b"READY=1", os.environ["NOTIFY_SOCKET"])
while True:
    time.sleep(0.05)
"""


SMOKE_SERVER = """\
import json, os, socketserver, sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
# A GPU server for smoke.sh on its Unix socket: the engine's model and reply, the STT's transcript, the TTS's audio.
class Handler(BaseHTTPRequestHandler):
    def reply(self, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        self.reply({"data": [{"id": "qwen"}]} if self.path == "/v1/models" else b"", "text/plain")
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/v1/audio/speech":
            self.reply(b"RIFF" + bytes(64), "audio/wav")
        elif self.path == "/v1/audio/transcriptions":
            self.reply({"text": "What is the capital of France?"})
        else:
            self.reply({"choices": [{"message": {"content": "Paris."}}]})
    def log_message(self, *args):
        pass
class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
server = Server(sys.argv[1], Handler)
Path(os.environ["READY"]).write_text(str(os.getpid()))
server.serve_forever()
"""


@pytest.fixture
def host(tmp_path: Path) -> dict:
    """A home with units in ~/.config/systemd/user, the shims first on PATH, and the services' scripts."""
    home = tmp_path / "home"
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    (home / "svc.py").write_text(SERVICE)
    (home / "wait.py").write_text(WAIT_READY)
    (home / "env-a").write_text('# for svc-a\nFROM_FILE="from the file"\n')
    py = sys.executable
    unit = {
        "svc-a.service": f"""
            [Unit]
            PartOf=box.target
            [Service]
            Type=exec
            ExecStartPre=/bin/mkdir -p %t/lh
            Environment=GREETING=hello READY=%h/a.ready
            EnvironmentFile=%h/env-a
            EnvironmentFile=-%h/missing
            ExecStart={py} %h/svc.py first
            ExecStartPost={py} %h/wait.py %h/a.ready
            KillSignal=SIGINT
            TimeoutStopSec=5
            Restart=on-failure
            RestartSec=0.1
        """,
        "svc-b.service": f"""
            [Unit]
            PartOf=box.target
            After=svc-a.service
            [Service]
            Environment=STUBBORN=1 READY=%h/b.ready
            ExecStart={py} %h/svc.py b
            ExecStartPost={py} %h/wait.py %h/b.ready
            TimeoutStopSec=1
        """,
        "box.target": """
            [Unit]
            Wants=svc-a.service svc-b.service
        """,
        "broken.service": f"""
            [Service]
            ExecStart={py} -c "import sys; print('broken'); sys.exit(3)"
            Restart=on-failure
            RestartSec=0.05
            StartLimitBurst=2
        """,
        "unhealthy.service": f"""
            [Service]
            ExecStart={py} -c "import time; time.sleep(30)"
            ExecStartPost={py} -c "import sys; sys.exit(1)"
        """,
    }
    for name, text in unit.items():
        (units / name).write_text(textwrap.dedent(text).lstrip())
    # A drop-in that replaces svc-a's command and adds flags, which $FLAGS splits into words.
    (units / "svc-a.service.d").mkdir()
    (units / "svc-a.service.d" / "flags.conf").write_text(
        f'[Service]\nEnvironment="FLAGS=--x 1 --y"\nExecStart=\nExecStart={py} %h/svc.py $FLAGS ${{GREETING}}-word\n'
    )
    env = {
        "HOME": str(home),
        "PATH": f"{BIN / 'shims'}{os.pathsep}{os.environ['PATH']}",
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    (tmp_path / "run").mkdir(mode=0o700)
    yield {"home": home, "units": units, "env": env, "tmp": tmp_path}
    subprocess.run(
        ["systemctl", "--user", "stop", "box.target", "broken.service", "unhealthy.service"],
        env=os.environ | env,
        capture_output=True,
    )


def ctl(host: dict, *args: str, tool: str = "systemctl") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [tool, *args], env=os.environ | host["env"], capture_output=True, text=True, timeout=60
    )


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_target_starts_its_units_in_order_and_stops_every_process(host: dict) -> None:
    before = time.time()
    done = ctl(host, "--user", "start", "box.target")
    assert done.returncode == 0, done.stderr
    assert ctl(host, "--user", "is-active", "svc-a", "svc-b").stdout.split() == ["active"] * 2
    a, b = (int((host["home"] / f"{n}.ready").read_text()) for n in ("a", "b"))
    journal = ctl(
        host, "--user", "-u", "svc-a", "--since", f"@{before}", "-o", "cat", tool="journalctl"
    )
    # The drop-in's command, with $FLAGS split into words and ${GREETING} in place, and the environment's files.
    assert "argv ['--x', '1', '--y', 'hello-word']" in journal.stdout
    # LH_SVC_UNIT names the unit, where systemd's cgroup would (deploy/bin/gpu-guard reads it).
    assert "env hello from the file svc-a.service" in journal.stdout
    assert (Path(host["env"]["XDG_RUNTIME_DIR"]) / "lh").is_dir()  # ExecStartPre, %t
    order = ctl(host, "-u", "svc-a", "-u", "svc-b", "-o", "short-unix", tool="journalctl").stdout
    assert order.index("Started svc-a.service") < order.index("Starting svc-b.service")
    shown = ctl(host, "--user", "show", "svc-a", "--property=Environment", "--value").stdout
    # As lh-bench reads it (Systemd.environment).
    assert shlex.split(shown) == [
        "GREETING=hello",
        f"READY={host['home']}/a.ready",
        "FLAGS=--x 1 --y",
    ]
    # Stopping the target stops both, and returns only once both have exited: svc-b ignores the signal and
    # is killed after its 1 s TimeoutStopSec; svc-a stops on its KillSignal.
    done = ctl(host, "--user", "stop", "box.target")
    assert done.returncode == 0, done.stderr
    assert gone(a) and gone(b)
    assert "stopped by SIGINT" in ctl(host, "-u", "svc-a", "-o", "cat", tool="journalctl").stdout
    inactive = ctl(host, "--user", "is-active", "svc-a", "svc-b")
    assert inactive.returncode == 3 and inactive.stdout.split() == ["inactive"] * 2


def test_a_unit_that_fails_restarts_and_then_fails_its_start(host: dict) -> None:
    assert ctl(host, "--user", "start", "svc-a").returncode == 0
    pid = int((host["home"] / "a.ready").read_text())
    (host["home"] / "a.ready").unlink()
    os.kill(pid, signal.SIGKILL)  # the main process dies: Restart=on-failure starts it again
    deadline = time.time() + 10
    while not (host["home"] / "a.ready").exists() and time.time() < deadline:
        time.sleep(0.05)
    deadline = time.time() + 10
    while ctl(host, "is-active", "svc-a").stdout.strip() != "active" and time.time() < deadline:
        time.sleep(0.05)
    assert int((host["home"] / "a.ready").read_text()) != pid
    assert ctl(host, "is-active", "svc-a").stdout.strip() == "active"
    # A start whose ExecStartPost fails fails, as systemd's does, and leaves no process behind.
    failed = ctl(host, "--user", "start", "unhealthy")
    assert failed.returncode == 1 and "Job for unhealthy.service failed" in failed.stderr
    assert ctl(host, "is-active", "unhealthy").stdout.strip() == "failed"
    # A unit that exits at once has started, as a Type=exec unit has once it runs, then fails after its restarts
    # (StartLimitBurst=2).
    assert ctl(host, "--user", "start", "broken").returncode == 0
    deadline = time.time() + 10
    while ctl(host, "is-active", "broken").stdout.strip() != "failed" and time.time() < deadline:
        time.sleep(0.05)
    assert ctl(host, "is-failed", "broken").returncode == 0
    journal = ctl(host, "-u", "broken", "-o", "cat", tool="journalctl").stdout
    assert journal.count("broken\n") == 2 and "start-limit-hit" in journal  # the burst's two starts
    assert ctl(host, "--user", "reset-failed", "broken").returncode == 0
    assert ctl(host, "is-active", "broken").stdout.strip() == "inactive"


def system_pythons() -> list[str]:
    """The host's own interpreters from 3.10 on, which the shims run before any venv exists (the tests' own
    Python is the venv's)."""
    found = []
    for path in sorted(
        {*Path("/usr/bin").glob("python3"), *Path("/usr/bin").glob("python3.[0-9]*")}
    ):
        if path.name.endswith("-config") or not os.access(path, os.X_OK):
            continue
        done = subprocess.run(
            [str(path), "-c", "import sys; print(sys.version_info >= (3, 10))"],
            capture_output=True,
            text=True,
        )
        if done.stdout.strip() == "True":
            found.append(str(path))
    return found


@pytest.mark.skipif(not system_pythons(), reason="no system python3 of 3.10 or newer in /usr/bin")
def test_the_host_python_runs_the_supervisor_and_the_forwarder(host: dict) -> None:
    # A bare host's python3 runs them, as the shims and the scripts do: Ubuntu 22.04's is 3.10.
    for python in system_pythons():
        done = subprocess.run(
            [python, str(BIN / "lh-svc"), "systemctl", "is-system-running"],
            env=os.environ | host["env"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stdout.strip() == "lh-svc", (python, done.stderr)
        usage = subprocess.run([python, str(BIN / "lh-forward")], capture_output=True, text=True)
        assert usage.returncode == 2 and "lh-forward LISTEN=TARGET" in usage.stderr, (
            python,
            usage.stderr,
        )


def test_a_start_whose_server_dies_fails_at_once_not_at_its_timeout(host: dict) -> None:
    # As the GPU units: ExecStartPost waits for the server with wait-healthy --pid=${MAINPID}, which lh-svc sets for
    # its control commands as systemd does; a server that dies in its start fails the start in seconds.
    py, units = sys.executable, host["units"]
    (units / "dies.service").write_text(
        "[Service]\n"
        f"ExecStart={py} -c 'import time; time.sleep(0.5)'\n"
        f"ExecStartPost={BIN / 'wait-healthy'} --pid=${{MAINPID}} http://127.0.0.1:9/health 600\n"
        "TimeoutStartSec=700\n"
    )
    start = time.monotonic()
    done = ctl(host, "--user", "start", "dies.service")
    assert done.returncode == 1 and "Job for dies.service failed" in done.stderr
    assert time.monotonic() - start < 30
    journal = ctl(host, "-u", "dies", "-o", "cat", tool="journalctl").stdout
    assert "exited before http://127.0.0.1:9/health was healthy" in journal


@pytest.mark.skipif(
    not Path("/proc/sys/kernel/random/boot_id").exists(), reason="reads /proc (Linux)"
)
def test_a_state_file_from_another_boot_or_a_reused_pid_is_not_a_running_unit(host: dict) -> None:
    # After a reboot, or once a pid is reused, a state file's runner names some other process: the unit is not
    # running, a start starts it, and a stop leaves that process alone.
    state = Path(host["env"]["XDG_STATE_HOME"]) / "lh-svc"
    state.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    (host["units"] / "sleeper.service").write_text(
        f"[Service]\nExecStart={py} -c 'import time; time.sleep(60)'\n"
    )
    other = subprocess.Popen(["sleep", "300"])
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    started = Path(f"/proc/{other.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    try:
        for recorded in (
            {"boot": boot, "runner_start": str(int(started) - 100)},  # the pid reused since
            {
                "boot": "00000000-0000-0000-0000-000000000000",
                "runner_start": started,
            },  # another boot
        ):
            (state / "sleeper.service.json").write_text(
                json.dumps({"state": "active", "runner": other.pid, "main": other.pid} | recorded)
            )
            assert ctl(host, "--user", "is-active", "sleeper").stdout.strip() != "active", recorded
            assert ctl(host, "--user", "stop", "sleeper").returncode == 0
            assert other.poll() is None, recorded  # the unrelated process lives on
        (state / "sleeper.service.json").write_text(
            json.dumps(
                {
                    "state": "active",
                    "runner": other.pid,
                    "boot": boot,
                    "runner_start": str(int(started) - 1),
                }
            )
        )
        assert ctl(host, "--user", "start", "sleeper").returncode == 0
        runner = json.loads((state / "sleeper.service.json").read_text())
        assert runner["runner"] != other.pid and runner["boot"] == boot and runner["runner_start"]
        assert ctl(host, "--user", "is-active", "sleeper").stdout.strip() == "active"
    finally:
        ctl(host, "--user", "stop", "sleeper")
        other.kill()
        other.wait()
    # A target's or a socket's state is a boot's too.
    (state / "box.target.json").write_text(
        json.dumps({"state": "active", "boot": "00000000-0000-0000-0000-000000000000"})
    )
    assert ctl(host, "--user", "is-active", "box.target").stdout.strip() == "inactive"


def test_a_broken_unit_link_or_a_missing_unit_stops_only_itself(host: dict) -> None:
    # A stop reads every unit for what is PartOf or BindsTo the units it stops: one link that dangles there (an earlier
    # tenant's, or into a node directory since deleted) fails no stop of the others, and a unit asked for that does
    # not exist fails the command, as systemd's, after the others have stopped.
    assert ctl(host, "--user", "start", "svc-a").returncode == 0
    pid = int((host["home"] / "a.ready").read_text())
    (host["units"] / "lh-tunnel@.service").symlink_to(
        host["home"] / "nowhere" / "lh-tunnel@.service"
    )
    done = ctl(host, "--user", "stop", "svc-a")
    assert done.returncode == 0, done.stderr
    assert gone(pid) and ctl(host, "--user", "is-active", "svc-a").stdout.strip() == "inactive"
    assert ctl(host, "--user", "start", "svc-a").returncode == 0
    pid = int((host["home"] / "a.ready").read_text())
    done = ctl(host, "--user", "stop", "svc-a", "nothing.service")
    assert done.returncode == 5 and "Unit nothing.service not loaded" in done.stderr
    assert gone(pid) and ctl(host, "--user", "is-active", "svc-a").stdout.strip() == "inactive"


def test_a_stop_cancels_a_start_it_meets_rather_than_wait_for_it(host: dict) -> None:
    # A start whose server never becomes ready waits up to its TimeoutStartSec; a stop meanwhile ends it at once, as
    # systemd's cancels the start job, rather than waiting the start out.
    py, units = sys.executable, host["units"]
    (units / "slow.service").write_text(
        f"[Service]\nExecStart={py} -c 'import time; time.sleep(120)'\n"
        f"ExecStartPost={py} %h/wait.py %h/never.ready\nTimeoutStartSec=60\n"
    )
    starting = subprocess.Popen(
        ["systemctl", "--user", "start", "slow.service"],
        env=os.environ | host["env"],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 10
        while (
            ctl(host, "--user", "is-active", "slow").stdout.strip() != "activating"
            and time.time() < deadline
        ):
            time.sleep(0.05)
        begun = time.monotonic()
        done = ctl(host, "--user", "stop", "slow.service")
        assert done.returncode == 0, done.stderr
        assert time.monotonic() - begun < 15
        assert (
            starting.wait(timeout=15) == 1
            and "Job for slow.service failed" in starting.stderr.read()
        )
        assert ctl(host, "--user", "is-active", "slow").stdout.strip() == "inactive"
    finally:
        starting.kill()
        starting.wait()
        ctl(host, "--user", "stop", "slow.service")


def test_a_stop_returns_once_the_main_process_group_is_gone() -> None:
    # A stop returns once the unit's processes have exited, as systemd's waits for its cgroup to empty: an engine
    # core still exiting holds its GPU memory and /dev/shm region from the next start. After the SIGKILL the group's
    # last process is gone only once it is reaped (its parent, here this test, reaps it 1.5 s on).
    lh_svc = _module("lh_svc", BIN / "lh-svc")
    stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    child = subprocess.Popen([sys.executable, "-c", stubborn], start_new_session=True)
    time.sleep(0.3)
    reaper = threading.Timer(1.5, child.wait)
    reaper.start()
    try:
        begun = time.monotonic()
        lh_svc.group_gone(child.pid, 0.2)
        assert time.monotonic() - begun >= 1.4
        with pytest.raises(ProcessLookupError):
            os.killpg(child.pid, 0)
    finally:
        reaper.join()
        child.kill()
        child.wait()


def test_the_shims_answer_what_the_scripts_ask(host: dict) -> None:
    assert ctl(host, "--user", "daemon-reload").returncode == 0
    assert ctl(host, "--user", "enable", "-q", "svc-a.service").returncode == 0
    cat = ctl(host, "--user", "cat", "svc-a").stdout
    assert f"# {host['units'] / 'svc-a.service'}" in cat and "flags.conf" in cat
    assert ctl(host, "--user", "start", "nothing").returncode == 5  # no such unit
    assert ctl(host, "--user", "frobnicate").returncode == 2
    assert ctl(host, "-u", "svc-a", "--since", "not a time", tool="journalctl").returncode == 1


def test_a_notify_unit_starts_at_ready_with_its_socket_unit_and_without_a_missing_wanted_one(
    host: dict,
) -> None:
    # As cyan's gateway: Type=notify, Requires= its socket unit, Wants= the tunnel a single host has none of.
    units, py = host["units"], sys.executable
    (host["home"] / "notify.py").write_text(NOTIFY)
    (units / "gw.socket").write_text("[Socket]\nListenStream=%t/lh/gw.sock\n")
    check = f'{py} -c "import pathlib, sys; sys.exit(not pathlib.Path(sys.argv[1]).exists())" %h/gw.ready'
    (units / "gw.service").write_text(
        "[Unit]\nRequires=gw.socket\nWants=lh-tunnel@node.service\nAfter=gw.socket lh-tunnel@node.service\n"
        f"[Service]\nType=notify\nEnvironment=MARK=%h/gw.ready\nExecStart={py} %h/notify.py\n"
        f"ExecStartPost={check}\n"
    )
    sleeper = f"{py} -c 'import time; time.sleep(30)'"
    (units / "needs.service").write_text(
        f"[Unit]\nRequires=absent.service\n[Service]\nExecStart={sleeper}\n"
    )
    (units / "silent.service").write_text(
        f"[Service]\nType=notify\nTimeoutStartSec=1\nExecStart={sleeper}\n"
    )
    try:
        done = ctl(host, "--user", "start", "gw.service")
        assert done.returncode == 0, done.stderr
        # The start returned once READY=1 came, and ExecStartPost ran after it: it found the mark.
        assert ctl(host, "is-active", "gw.socket", "gw.service").stdout.split() == ["active"] * 2
        # A unit that Requires= one that does not exist fails its start, as under systemd.
        missing = ctl(host, "--user", "start", "needs.service")
        assert missing.returncode == 5 and "absent.service" in missing.stderr
        # A notify unit whose main process never says READY=1 fails at its TimeoutStartSec.
        silent = ctl(host, "--user", "start", "silent.service")
        assert (
            silent.returncode == 1 and ctl(host, "is-active", "silent").stdout.strip() == "failed"
        )
        assert "no READY=1" in ctl(host, "-u", "silent", "-o", "cat", tool="journalctl").stdout
    finally:
        ctl(host, "--user", "stop", "gw.service", "gw.socket", "silent.service")
    assert ctl(host, "is-active", "gw.service", "gw.socket").stdout.split() == ["inactive"] * 2


def test_variant_sh_runs_on_the_supervisor(host: dict, tmp_path: Path) -> None:
    # The box's variant.sh with stand-in units: lh-engine runs a make-room drop-in's command, and the flags'
    # drop-in it writes, with the audio servers left running.
    node = tmp_path / "node"
    (node / "variants" / "make-room").mkdir(parents=True)
    shutil.copy(REPO / "deploy" / "phastform" / "variant.sh", node / "variant.sh")
    py, units = sys.executable, host["units"]
    for name in ("lh-stt", "lh-tts", "lh-engine"):
        ready = f"%h/{name}.ready"
        (units / f"{name}.service").write_text(
            f"[Unit]\nPartOf=leaseholder.target\n[Service]\nEnvironment=READY={ready}\nEnvironment=LH_ENGINE_FLAGS=\n"
            f"ExecStart={py} %h/svc.py {name} $LH_ENGINE_FLAGS\nExecStartPost={py} %h/wait.py {ready}\n"
        )
    (units / "leaseholder.target").write_text(
        "[Unit]\nWants=lh-stt.service lh-tts.service lh-engine.service\n"
    )
    (node / "variants" / "make-room" / "lh-engine.conf").write_text(
        f"[Service]\nExecStart=\nExecStart={py} %h/svc.py make-room $LH_ENGINE_FLAGS\n"
    )
    env = os.environ | host["env"]
    try:
        done = subprocess.run(
            ["bash", str(node / "variant.sh"), "use", "make-room", "--budget", "48"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        engine = ctl(host, "-u", "lh-engine", "-o", "cat", tool="journalctl").stdout
        assert "argv ['make-room', '--budget', '48']" in engine
        stt = int((host["home"] / "lh-stt.ready").read_text())
        again = subprocess.run(
            ["bash", str(node / "variant.sh"), "use", "frozen"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert again.returncode == 0, again.stdout + again.stderr
        assert (
            "argv ['lh-engine']"
            in ctl(host, "-u", "lh-engine", "-o", "cat", tool="journalctl").stdout
        )
        # Only the engine's drop-ins changed, so the audio servers kept running.
        assert int((host["home"] / "lh-stt.ready").read_text()) == stt and not gone(stt)
    finally:
        ctl(host, "--user", "stop", "leaseholder.target")


def single_host_node(host: dict, tmp_path: Path) -> Path:
    """A node rendered from a100-40gb and linked as deploy/a100/host.py units links it: its GPU units beside cyan's
    control plane, with this host's drop-ins and the worker's vad-turns variant, in one ~/.config/systemd/user.
    Stand-ins replace the GPU servers' commands (and make-room's) so that lh-svc can run them; the control plane
    is only linked. Returns the node's directory, whose variant.sh is the box's, as the node copies it."""
    node = tmp_path / "node"
    host_py.profile.render(host_py.profile.load("a100-40gb"), FACTS, node)
    (node / "host.json").write_text(json.dumps(FACTS))
    py = sys.executable
    for name in ("lh-stt", "lh-tts", "lh-engine"):
        ready = f"%h/{name}.ready"
        (node / "systemd" / f"{name}.service").write_text(
            f"[Unit]\nPartOf=leaseholder.target\n[Service]\nEnvironment=READY={ready}\nEnvironment=LH_ENGINE_FLAGS=\n"
            f"ExecStart={py} %h/svc.py {name} $LH_ENGINE_FLAGS\nExecStartPost={py} %h/wait.py {ready}\n"
        )
    (node / "variants" / "make-room" / "lh-engine.conf").write_text(
        f"[Service]\nExecStart=\nExecStart={py} %h/svc.py make-room $LH_ENGINE_FLAGS\n"
    )
    host_py.install_units(host_py.Node.load(node), host["units"], systemctl="true")
    return node


class ShimSystemd(e2.Systemd):
    """lh-bench e2 run's view of the units through lh-svc's shims (what it reads: show); the restarts it asks for
    are recorded, not run, since the control plane is only linked here."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> str:
        if args[0] == "show":
            return super().__call__(*args)
        self.calls.append(args)
        return ""

    def active(self, unit: str) -> bool:
        return False

    def journal(self, unit: str, since: float, until: float | None = None) -> str:
        return "registered worker"


def test_variant_sh_on_a_single_host_switches_only_the_gpu_units(
    host: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On one host the box's variant.sh shares ~/.config/systemd/user with the control plane's drop-ins: the
    # worker's vad-turns, which every E2 point's check needs, and e2 run's own for an arm (the gateway's overlay,
    # the worker's no-events). A switch, as campaign.sh makes one before every point, must leave them alone.
    node = single_host_node(host, tmp_path)
    units, home = host["units"], host["home"]
    for name, value in host["env"].items():
        monkeypatch.setenv(name, value)
    (home / "leaseholder").mkdir()
    (home / "leaseholder" / "src").symlink_to(REPO)
    config = home / ".config" / "leaseholder"
    config.mkdir(parents=True)
    (config / "gateway.toml").write_text(
        (REPO / "deploy" / "cyan" / "gateway.example.toml").read_text()
    )
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "overlay.toml").write_text(
        'arm = 3\nlabel = "an arm with a gateway table, events off"\n[voice]\nevents = false\n'
        '[gateway.admission]\npolicy = "static-cap"\nbackground_max_in_flight = 2\n'
    )
    (arms / "voice.toml").write_text('arm = 1\nlabel = "voice alone"\n')
    overlay, plain = load_arm(arms / "overlay.toml"), load_arm(arms / "voice.toml")
    e2_host = e2.Host(home=home)
    systemd = ShimSystemd()
    first = e2.configure_cyan(overlay, e2_host, systemd)  # links e2-gateway and no-events
    assert first["gateway"]["restarted"] and first["worker"]["restarted"]
    control = sorted(
        str(p.relative_to(units))
        for p in units.glob("*.d/*")
        if not p.parent.name.startswith(("lh-engine.", "lh-stt.", "lh-tts."))
    )
    assert "lh-voice-worker.service.d/variant-vad-turns.conf" in control
    assert "lh-gateway.service.d/variant-e2-gateway.conf" in control
    assert "lh-voice-worker.service.d/variant-no-events.conf" in control
    env = os.environ.copy()
    try:
        for switch in (
            ("measure", "on"),
            ("use", "make-room", "--additional-config.make_room.reserve_tokens", "0"),
            ("use", "frozen", "--max-num-batched-tokens", "2048"),
            ("use", "frozen"),
        ):
            done = subprocess.run(
                ["bash", str(node / "variant.sh"), *switch],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert done.returncode == 0, (switch, done.stdout + done.stderr)
            # The control plane's drop-ins came through the switch, and e2 run's check passes the worker: the
            # point's arm finds its drop-ins as it left them, so nothing of cyan's restarts.
            assert control == sorted(
                str(p.relative_to(units))
                for p in units.glob("*.d/*")
                if not p.parent.name.startswith(("lh-engine.", "lh-stt.", "lh-tts."))
            ), switch
            again = e2.configure_cyan(overlay, e2_host, systemd)
            assert not again["gateway"]["restarted"] and not again["worker"]["restarted"], switch
        engine = ctl(host, "-u", "lh-engine", "-o", "cat", tool="journalctl").stdout
        assert "argv ['make-room', '--additional-config.make_room.reserve_tokens', '0']" in engine
        assert "argv ['lh-engine', '--max-num-batched-tokens', '2048']" in engine
        # The GPU units' variant drop-ins are the box's: none once it is frozen again; the measurement drop-in
        # stays.
        assert not [
            p
            for u in ("lh-engine", "lh-stt", "lh-tts")
            for p in units.glob(f"{u}.service.d/variant-*")
        ]
        assert (units / "lh-engine.service.d" / "measure.conf").is_symlink()
    finally:
        ctl(host, "--user", "stop", "leaseholder.target")
    # An arm without a gateway table or events off takes its drop-ins away, as on cyan.
    last = e2.configure_cyan(plain, e2_host, systemd)
    assert last["gateway"]["restarted"] and last["worker"]["restarted"]
    assert not (units / "lh-gateway.service.d" / "variant-e2-gateway.conf").exists()
    assert (units / "lh-voice-worker.service.d" / "variant-vad-turns.conf").is_symlink()


def test_the_box_variants_change_only_the_units_variant_sh_switches() -> None:
    # variant.sh's switches touch only the GPU units' variant drop-ins, so that on a single host the control
    # plane's (cyan's variants, linked into the same directory) are not the box's to remove. That holds while every
    # variant a node can switch to changes only those units: the box's, as the node takes them, and the node's
    # rendered make-room. cyan's variants change none of them.
    text = (REPO / "deploy" / "phastform" / "variant.sh").read_text()
    gpu = re.search(r"^gpu=\(([^)]*)\)", text, re.MULTILINE)
    assert gpu, "variant.sh names its GPU units in gpu=(...)"
    switched = set(gpu.group(1).split())
    box = {p.stem for p in (REPO / "deploy" / "phastform" / "variants").glob("*/*.conf")}
    templates = {p.stem for p in (REPO / "deploy" / "templates" / "variants").glob("*/*.conf")}
    cyan = {p.stem for p in (REPO / "deploy" / "cyan" / "variants").glob("*/*.conf")}
    assert box and templates and cyan
    assert box | templates <= switched == {"lh-engine", "lh-stt", "lh-tts"}
    assert not cyan & switched


def echo_server() -> tuple[socket.socket, int]:
    server = socket.create_server(("127.0.0.1", 0))

    def serve() -> None:
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            with connection:
                while data := connection.recv(65536):
                    connection.sendall(data.upper())

    threading.Thread(target=serve, daemon=True).start()
    return server, server.getsockname()[1]


def test_lh_forward_relays_a_socket_to_a_port() -> None:
    server, port = echo_server()
    # A short directory: a socket's path has room for 104 bytes on macOS.
    path = Path(tempfile.mkdtemp(prefix="lhf")) / "lh" / "phastform-stt.sock"
    path.parent.mkdir()
    stale = socket.socket(socket.AF_UNIX)  # a socket left by an earlier run is replaced
    stale.bind(str(path))
    stale.close()
    forward = subprocess.Popen(
        [sys.executable, str(BIN / "lh-forward"), f"{path}=127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert "lh-forward:" in forward.stdout.readline()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(path))
            client.sendall(b"what is the capital of france")
            client.shutdown(socket.SHUT_WR)
            reply = b""
            while data := client.recv(65536):
                reply += data
        assert reply == b"WHAT IS THE CAPITAL OF FRANCE"
    finally:
        forward.send_signal(signal.SIGTERM)
        assert forward.wait(timeout=10) == 0
        server.close()
    assert not path.exists()  # removed when stopped
    path.write_text("not a socket")
    refused = subprocess.run(
        [sys.executable, str(BIN / "lh-forward"), f"{path}=127.0.0.1:{port}"],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0 and "is not a socket" in refused.stderr
    bad = subprocess.run([sys.executable, str(BIN / "lh-forward"), "nowhere"], capture_output=True)
    assert bad.returncode == 2


def test_lh_forward_relays_a_port_to_a_socket() -> None:
    # As AIPerf reaches the engine for the calibration point: a loopback port to the engine's Unix socket.
    path = Path(tempfile.mkdtemp(prefix="lhf")) / "engine.sock"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(path))
    server.listen()

    def serve() -> None:
        with contextlib.suppress(OSError):
            connection, _ = server.accept()
            with connection:
                while data := connection.recv(65536):
                    connection.sendall(data[::-1])

    threading.Thread(target=serve, daemon=True).start()
    with socket.socket() as probe:  # a free loopback port
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    forward = subprocess.Popen(
        [sys.executable, str(BIN / "lh-forward"), f"127.0.0.1:{port}=unix:{path}"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert f"127.0.0.1:{port} -> unix:{path}" in forward.stdout.readline()
        with socket.create_connection(("127.0.0.1", port)) as client:
            client.sendall(b"engine")
            client.shutdown(socket.SHUT_WR)
            reply = b""
            while data := client.recv(65536):
                reply += data
        assert reply == b"enigne"
    finally:
        forward.send_signal(signal.SIGTERM)
        assert forward.wait(timeout=10) == 0
        server.close()
    relative = subprocess.run(
        [sys.executable, str(BIN / "lh-forward"), f"127.0.0.1:{port}=unix:engine.sock"],
        capture_output=True,
        text=True,
    )
    assert relative.returncode == 2 and "absolute socket path" in relative.stderr


def test_the_voice_plane_runs_in_the_hosts_namespace_where_there_is_none_of_its_own() -> None:
    run = Path(tempfile.mkdtemp(prefix="lhv"))
    (run / "lh").mkdir()
    pidfile = run / "lh" / "voicenet.pid"
    holder = subprocess.Popen(
        ["bash", str(BIN / "voicenet-holder"), str(pidfile)],
        env=os.environ | {"LH_VOICENET": "host"},
    )
    try:
        deadline = time.time() + 10
        while not (pidfile.exists() and pidfile.read_text().strip()) and time.time() < deadline:
            time.sleep(0.05)
        assert pidfile.read_text().strip() == "host"
        done = subprocess.run(
            ["bash", str(BIN / "in-voicenet"), sys.executable, "-c", "print('the host network')"],
            env=os.environ | {"XDG_RUNTIME_DIR": str(run)},
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stdout.strip() == "the host network", done.stderr
    finally:
        holder.terminate()
        holder.wait()


def test_gpu_guard_reads_the_lent_gpu_alone(tmp_path: Path) -> None:
    # On a host that lends one of its GPUs (LH_GPU_ID, which the GPU units carry), another tenant's work on the
    # others is none of the guard's business; without it (the box, one GPU), it reads every GPU, as before.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    asked = tmp_path / "asked"
    (bin_dir / "nvidia-smi").write_text(f'#!/bin/sh\necho "$*" >>{asked}\n')
    (bin_dir / "nvidia-smi").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "LH_GPU_ID"}
    env["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    for extra in ({"LH_GPU_ID": "GPU-bbbb"}, {}):
        done = subprocess.run(
            ["bash", str(BIN / "gpu-guard")], env=env | extra, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
    assert asked.read_text().splitlines() == [
        "-i GPU-bbbb --query-compute-apps=pid --format=csv,noheader",
        "--query-compute-apps=pid --format=csv,noheader",
    ]


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="reads /proc (Linux)")
def test_gpu_guard_tells_the_supervisors_units_from_foreign_processes(tmp_path: Path) -> None:
    # Two processes on the GPU (a stand-in nvidia-smi lists them): one of lh-svc's units, which carries its
    # LH_SVC_UNIT, and a foreign one. The guard passes the unit and refuses the stranger.
    sleep = [sys.executable, "-c", "import time; time.sleep(30)"]
    ours = subprocess.Popen(sleep, env=os.environ | {"LH_SVC_UNIT": "lh-stt.service"})
    theirs = subprocess.Popen(
        sleep, env={k: v for k, v in os.environ.items() if k != "LH_SVC_UNIT"}
    )
    try:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        smi = bin_dir / "nvidia-smi"

        def guard(*pids: int) -> subprocess.CompletedProcess[str]:
            smi.write_text("#!/bin/sh\n" + "".join(f"echo {pid}\n" for pid in pids))
            smi.chmod(0o755)
            return subprocess.run(
                ["bash", str(BIN / "gpu-guard")],
                env=os.environ | {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
                capture_output=True,
                text=True,
            )

        assert guard(ours.pid).returncode == 0
        refused = guard(ours.pid, theirs.pid)
        assert refused.returncode == 1 and f"pid {theirs.pid}" in refused.stderr
    finally:
        for process in (ours, theirs):
            process.kill()
            process.wait()


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="the audit reads /proc (Linux)")
def test_block_0s_smoke_test_passes_its_listener_audit_under_lh_svc(host: dict) -> None:
    # Block 0 runs the box's smoke.sh, whose listener audit reads the host's listeners (a stand-in ss lists them,
    # held by the units' real processes) and attributes each to its unit by /proc. Under lh-svc every process is in
    # the session's cgroup and the gateway binds 127.0.0.1:18080 itself; the voice plane runs in the host's network
    # namespace (the kernel refuses user namespaces: voicenet-holder with LH_VOICENET=host, in-voicenet).
    units, home, py = host["units"], host["home"], sys.executable
    run = Path(host["env"]["XDG_RUNTIME_DIR"])
    (run / "lh").mkdir(mode=0o700)
    (home / "smoke_server.py").write_text(SMOKE_SERVER)
    for name in ("engine", "stt", "tts"):
        (units / f"lh-{name}.service").write_text(
            f"[Unit]\nPartOf=leaseholder.target\n[Service]\nEnvironment=READY=%h/lh-{name}.ready\n"
            f"ExecStart={py} %h/smoke_server.py %t/lh/{name}.sock\n"
            f"ExecStartPost={py} %h/wait.py %h/lh-{name}.ready\n"
        )
    (units / "leaseholder.target").write_text(
        "[Unit]\nWants=lh-stt.service lh-tts.service lh-engine.service\n"
    )
    (units / "lh-voicenet.service").write_text(
        f"[Service]\nEnvironment=LH_VOICENET=host\nExecStart=/bin/bash {BIN}/voicenet-holder %t/lh/voicenet.pid\n"
        f"ExecStartPost={py} %h/wait.py %t/lh/voicenet.pid\n"
    )
    for name in ("gateway", "livekit", "voice-worker"):
        door = "" if name == "gateway" else f"{BIN}/in-voicenet "
        (units / f"lh-{name}.service").write_text(
            f"[Unit]\nAfter=lh-voicenet.service\n[Service]\nEnvironment=READY=%h/lh-{name}.ready\n"
            f"ExecStart={door}{py} %h/svc.py {name}\nExecStartPost={py} %h/wait.py %h/lh-{name}.ready\n"
        )
    started = ("leaseholder.target", "lh-voicenet", "lh-gateway", "lh-livekit", "lh-voice-worker")
    try:
        for unit in started:
            done = ctl(host, "--user", "start", unit)
            assert done.returncode == 0, (unit, done.stderr)
        pid = {name: (home / f"lh-{name}.ready").read_text() for name in ("engine", "stt", "tts")}
        pid |= {
            name: (home / f"lh-{name}.ready").read_text()
            for name in ("gateway", "livekit", "voice-worker")
        }
        held = lambda name: f'users:(("python3",pid={pid[name]},fd=3))'  # noqa: E731
        listed = home / "listed"
        listed.mkdir()
        (listed / "inet").write_text(
            f"tcp LISTEN 0 1024 127.0.0.1:18080 0.0.0.0:* {held('gateway')}\n"
            f"tcp LISTEN 0 4096 127.0.0.1:17880 0.0.0.0:* {held('livekit')}\n"
            f"udp UNCONN 0 0 127.0.0.1:17882 0.0.0.0:* {held('livekit')}\n"
            f"udp UNCONN 0 0 10.0.0.5:51234 0.0.0.0:* {held('voice-worker')}\n"
        )
        (listed / "unix").write_text(
            "".join(
                f"u_str LISTEN 0 4096 {run}/lh/{name}.sock 3624 * 0 {held(name)}\n"
                for name in ("engine", "stt", "tts")
            )
        )
        tools = home / "tools"
        tools.mkdir()
        (tools / "ss").write_text(
            f"#!/bin/sh\ncase $2 in -lntup) cat {listed}/inet ;; -lxp) cat {listed}/unix ;; esac\n"
        )
        (tools / "nvidia-smi").write_text("#!/bin/sh\necho '4321 MiB, 40960 MiB'\n")
        for tool in tools.iterdir():
            tool.chmod(0o755)
        env = os.environ | host["env"] | {"PATH": f"{tools}{os.pathsep}{host['env']['PATH']}"}
        smoke = subprocess.run(
            ["bash", str(REPO / "deploy" / "phastform" / "smoke.sh")],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert smoke.returncode == 0, smoke.stdout + smoke.stderr
        audit = next(
            line for line in smoke.stdout.splitlines() if line.startswith("listener audit")
        )
        assert (
            f"ok: the box's units hold 3 Unix sockets, all in {run}, and 1 TCP or UDP listeners, all allowed; "
            "the voice plane runs in the host's network namespace (LH_VOICENET=host) with 3 listeners"
        ) in audit
        assert smoke.stdout.rstrip().endswith("smoke: ok")
    finally:
        ctl(host, "--user", "stop", *reversed(started))
