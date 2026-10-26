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
DRAINING = """\
import os, signal, socket, sys, time
from pathlib import Path
# The gateway as its unit runs it (Type=notify): READY=1 once it serves, its pid in sys.argv[1]; at a SIGTERM it
# drains for 1.5 s, a second SIGTERM swallowed as lh-gateway's handler swallows it, marks in sys.argv[2] that it
# drained and exits 0.
def drain(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(1.5)
    Path(sys.argv[2]).write_text("drained")
    sys.exit(0)
signal.signal(signal.SIGTERM, drain)
Path(sys.argv[1]).write_text(str(os.getpid()))
with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify:
    notify.sendto(b"READY=1", os.environ["NOTIFY_SOCKET"])
while True:
    time.sleep(0.05)
"""
LEAVES = """\
import os, signal, subprocess, sys, time
from pathlib import Path
# A main process that leaves a child in its group: at the unit's first start it starts the child, says it is ready
# (ready.PID) once the child is, and exits with the status sys.argv[1] when the file exit appears; at a later start
# it says it is ready and serves. The child records each signal it gets in signals, ignores it, and ends itself a
# minute on.
home = Path(os.environ["HOME"])
if sys.argv[1:] == ["child"]:
    def record(signum, _):
        with open(home / "signals", "a") as signals:
            signals.write(signal.Signals(signum).name + "\\n")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, record)
    (home / "child.pid").write_text(str(os.getpid()))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        time.sleep(0.05)
    sys.exit()
first = not (home / "child.pid").exists()
if first:
    subprocess.Popen([sys.executable, __file__, "child"])
    while not (home / "child.pid").exists():
        time.sleep(0.02)
(home / f"ready.{os.getpid()}").touch()
while not (first and (home / "exit").exists()):
    time.sleep(0.05)
sys.exit(int(sys.argv[1]))
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


@pytest.mark.parametrize("kind", ["simple", "exec", "notify"])
def test_a_unit_has_started_though_its_main_process_exits_before_the_runner_looks(
    host: dict, kind: str
) -> None:
    # As under systemd, a Type=simple or exec unit has started once its main process has been executed, though that
    # exits before the runner first looks at it, as on a busy host (LH_SVC_LOOK_DELAY makes it so every time): the start
    # succeeds, and the exit takes the unit's Restart= and start limit. lh-svc failed such a start, so that
    # broken.service above, whose main process exits at once, failed its start now and then on a loaded host. A notify
    # unit starts at its READY=1, which this one never sends.
    (host["units"] / "quick.service").write_text(
        f"[Service]\nType={kind}\nExecStart=/bin/sh -c 'exit 3'\n"
        "Restart=on-failure\nRestartSec=0.05\nStartLimitBurst=2\n"
    )
    started = kind != "notify"
    try:
        done = subprocess.run(
            ["systemctl", "--user", "start", "quick.service"],
            env=os.environ | host["env"] | {"LH_SVC_LOOK_DELAY": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert done.returncode == (0 if started else 1), done.stderr
        deadline = time.monotonic() + 30
        while ctl(host, "--user", "is-failed", "quick.service").stdout.strip() != "failed":
            assert time.monotonic() < deadline, "the unit did not fail after its restarts"
            time.sleep(0.1)
        # The start limit's two starts: each a started unit's exit, which Restart= took, or a failed start.
        journal = ctl(host, "-u", "quick", "-o", "cat", tool="journalctl").stdout
        assert journal.count("lh-svc: Started quick.service") == (2 if started else 0), journal
        assert journal.count("Main process exited, status=3") == (2 if started else 0), journal
        assert journal.count("lh-svc: start failed") == (0 if started else 2), journal
        assert "start-limit-hit" in journal, journal
    finally:
        ctl(host, "--user", "stop", "quick.service")


def test_a_start_keeps_what_its_runner_wrote_while_the_client_was_held_up(host: dict) -> None:
    # A start wrote the unit's state after spawning its runner: activating, the runner's pid and no main process. A
    # runner that had started its main process and reached active by then (the client descheduled on a busy host;
    # LH_SVC_SPAWN_DELAY makes it so) lost those writes: the unit read activating for as long as it ran, and a second
    # start waited out its TimeoutStartSec for it and failed. The client now writes the state before the spawn, and the
    # runner's pid alone after it.
    py = sys.executable
    (host["units"] / "held.service").write_text(
        f"[Service]\nExecStart={py} -c 'import time; time.sleep(60)'\nTimeoutStartSec=5\n"
    )
    try:
        done = subprocess.run(
            ["systemctl", "--user", "start", "held.service"],
            env=os.environ | host["env"] | {"LH_SVC_SPAWN_DELAY": "3"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert done.returncode == 0, done.stderr
        assert ctl(host, "--user", "is-active", "held.service").stdout.strip() == "active"
        # The main process its runner recorded, still there.
        recorded = Path(host["env"]["XDG_STATE_HOME"]) / "lh-svc" / "held.service.json"
        state = json.loads(recorded.read_text())
        assert state["main"] and not gone(state["main"]), state
        begun = time.monotonic()
        again = ctl(host, "--user", "start", "held.service")
        took = time.monotonic() - begun
        assert again.returncode == 0 and took < 3, (again.stderr, took)
    finally:
        ctl(host, "--user", "stop", "held.service")


def test_a_unit_reads_activating_while_its_start_spawns_its_runner(
    host: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A start writes its unit activating before it spawns the runner: until the runner is named, the client stands for
    # it, or is-active reads the unit failed meanwhile, as it reads an activating unit whose runner is gone. Here this
    # process is the start's client, and reads the unit at the spawn.
    for key, value in host["env"].items():
        monkeypatch.setenv(key, value)
    lh_svc = _module("lh_svc", BIN / "lh-svc")
    (host["units"] / "spawned.service").write_text(
        f"[Service]\nExecStart={sys.executable} -c 'import time; time.sleep(60)'\n"
    )
    read: list[str] = []
    runners: list[subprocess.Popen] = []
    popen = subprocess.Popen

    def spawn(argv: list[str], **kwargs: object) -> subprocess.Popen:
        if argv[2:] != ["run", "spawned.service"]:
            return popen(argv, **kwargs)
        read.append(lh_svc.current("spawned.service"))
        runners.append(popen(argv, **kwargs))
        return runners[-1]

    monkeypatch.setattr(subprocess, "Popen", spawn)
    try:
        assert lh_svc.start(["spawned.service"]) == 0
    finally:
        monkeypatch.setattr(subprocess, "Popen", popen)
        # This process is the runner's parent, and reaps it as it exits, as init reaps one whose client has exited:
        # the stop waits for the runner to be gone, and where there is no /proc a zombie reads as alive.
        for runner in runners:
            threading.Thread(target=runner.wait, daemon=True).start()
        ctl(host, "--user", "stop", "spawned.service")
    assert read == ["activating"]


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
def test_the_host_python_runs_the_hosts_scripts(host: dict) -> None:
    # A bare host's python3 runs them, as the shims and the units do: Ubuntu 22.04's is 3.10. ruff's 3.10 target
    # (ruff.toml) checks their syntax, not what they import, which only a run on 3.10 does.
    for python in system_pythons():
        done = subprocess.run(
            [python, str(BIN / "lh-svc"), "systemctl", "is-system-running"],
            env=os.environ | host["env"],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0 and done.stdout.strip() == "lh-svc", (python, done.stderr)
        for script, status, usage in (
            ("lh-forward", 2, "lh-forward LISTEN=TARGET"),
            ("lh-tunnel", 2, "usage: lh-tunnel NODE"),
            ("lh-gateway-stop", 2, "usage: lh-gateway-stop"),
            ("audit-listeners", 0, "usage: audit-listeners"),
        ):
            args = ["--help"] if script == "audit-listeners" else []
            done = subprocess.run(
                [python, str(BIN / script), *args], capture_output=True, text=True
            )
            assert done.returncode == status and usage in done.stdout + done.stderr, (
                python,
                script,
                done.stderr,
            )


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc (Linux)")
def test_a_unit_whose_main_process_exits_leaves_no_process_of_its_group(host: dict) -> None:
    # systemd ends what is left of a unit's cgroup once its main process has exited, before ExecStopPost and a restart:
    # an engine core that outlives its server (killed by the OOM killer, say) holds the GPU, and every restart's
    # gpu-guard would refuse to start beside it. Each start of this unit leaves a child in its group as its main
    # process fails; lh-svc ends each before the next start.
    children = host["tmp"] / "children"
    (host["units"] / "leaves.service").write_text(
        "[Service]\n"
        f"ExecStart=/bin/sh -c 'sleep 300 & echo $! >> {children}; sleep 0.3; exit 1'\n"
        "Restart=on-failure\nRestartSec=0.1\nStartLimitBurst=2\nTimeoutStopSec=5\n"
    )

    def ended(pid: int) -> bool:
        """Exited: gone, or a zombie its new parent has yet to reap."""
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].startswith("Z")
        except FileNotFoundError:
            return True

    # The children not yet seen ended, which alone the cleanup kills: a pid seen gone may name another process since.
    left: list[int] = []
    try:
        assert ctl(host, "--user", "start", "leaves.service").returncode == 0
        deadline = time.monotonic() + 30
        while ctl(host, "--user", "is-failed", "leaves.service").stdout.strip() != "failed":
            assert time.monotonic() < deadline, "the unit failed after its restarts"
            time.sleep(0.1)
        left = pids = [int(pid) for pid in children.read_text().split()]
        assert len(pids) == 2, pids  # StartLimitBurst
        deadline = time.monotonic() + 10
        while left := [pid for pid in left if not ended(pid)]:
            assert time.monotonic() < deadline, left
            time.sleep(0.05)
    finally:
        for pid in left:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="a subreaper (Linux)")
def test_a_unit_restarts_at_once_where_nothing_reaps_its_orphans(host: dict) -> None:
    # A unit's process whose parent exits before it goes to the nearest subreaper: lh-svc's runner, as systemd's
    # manager for its units. It went to init, which in a container may reap none (`sleep infinity`): a zombie, it
    # stayed a member of its group, which a restart waits to be gone, so that every exit of the main process waited
    # out TimeoutStopSec and 30 s more. Here the runners are the children of a subreaper that reaps none of them.
    children = host["tmp"] / "children"
    (host["units"] / "orphans.service").write_text(
        "[Service]\n"
        f"ExecStart=/bin/sh -c 'sleep 30 & echo $! >> {children}; sleep 0.3; exit 1'\n"
        "Restart=on-failure\nRestartSec=0.1\nStartLimitBurst=2\nTimeoutStopSec=20\n"
    )
    init = (
        "import ctypes, subprocess, sys, time\n"
        "assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER\n"
        "print(subprocess.run(sys.argv[1:]).returncode, flush=True)\n"
        "time.sleep(120)  # reaping none of the orphans it adopts\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", init, "systemctl", "--user", "start", "orphans.service"],
        env=os.environ | host["env"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert parent.stdout is not None and parent.stdout.readline().strip() == "0"
        deadline = time.monotonic() + 15
        while ctl(host, "--user", "is-failed", "orphans.service").stdout.strip() != "failed":
            assert time.monotonic() < deadline, "a restart waited out TimeoutStopSec for a zombie"
            time.sleep(0.1)
        pids = [int(pid) for pid in children.read_text().split()]
        assert len(pids) == 2, pids  # StartLimitBurst
        assert not [pid for pid in pids if Path(f"/proc/{pid}").exists()]  # reaped, by the runner
        # Each main process's status, which its own wait reads, untouched by the reaping (a status taken reads 0).
        journal = ctl(host, "-u", "orphans", "-o", "cat", tool="journalctl").stdout
        assert journal.count("Main process exited, status=1") == 2, journal
    finally:
        ctl(host, "--user", "stop", "orphans.service")
        parent.kill()
        parent.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="a subreaper (Linux)")
def test_a_runner_reaps_the_orphans_it_adopts_and_no_process_it_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The runner reaps what it adopts, by pid, and leaves what it started to its Popen, which reads its status: a
    # status the reaping took would read 0, and a failed main process would not restart (Restart=on-failure). This
    # test's own children stand in for both, the orphan the older, both exited when the runner reaps.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    lh_svc = _module("lh_svc", BIN / "lh-svc")
    runner = lh_svc.Runner("reaps.service")
    runner.subreaper = True  # as run() makes it
    orphan = os.posix_spawn(
        sys.executable, [sys.executable, "-c", "raise SystemExit(4)"], os.environ
    )
    started, _ = runner.spawn(
        lh_svc.Unit("reaps.service", []), f"{sys.executable} -c 'raise SystemExit(3)'", {}
    )
    for pid in (orphan, started.pid):
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)  # it has exited, and is left unreaped
    runner.reap()
    with pytest.raises(ChildProcessError):
        os.waitpid(orphan, os.WNOHANG)  # reaped
    assert started.wait(timeout=5) == 3


def test_a_runner_stops_whichever_line_of_its_wait_its_sigterm_is_handled_at(
    tmp_path: Path,
) -> None:
    # A Python signal handler runs in the main thread between two of its bytecodes, wherever that is: inside the
    # runner's wait for a stop too, which threading.Event.wait does holding the Event's lock, and there the SIGTERM
    # handler's Event.set() waited for that lock for good, the runner frozen and its unit running on past the stop
    # (once in nine suite runs on a loaded host). Here a trace makes the handler's call at each of the wait's first lines
    # in turn, as a signal handled there would, in a process of its own, which a deadlock leaves hanging.
    program = textwrap.dedent(
        """\
        import importlib.machinery, importlib.util, sys
        loader = importlib.machinery.SourceFileLoader("lh_svc", sys.argv[1])
        spec = importlib.util.spec_from_loader("lh_svc", loader)
        lh_svc = sys.modules["lh_svc"] = importlib.util.module_from_spec(spec)
        loader.exec_module(lh_svc)
        for at in range(1, 13):
            runner, lines = lh_svc.Runner("waits.service"), []
            def handled_at(frame, event, arg):
                if event == "line":
                    lines.append(frame.f_lineno)
                    if len(lines) == at:
                        runner.stop()  # what the runner's SIGTERM handler calls
                return handled_at
            sys.settrace(handled_at)
            stopped = runner.stopping.wait(5)
            sys.settrace(None)
            assert stopped, (at, lines)
        print("stopped")
        """
    )
    try:
        done = subprocess.run(
            [sys.executable, "-c", program, str(BIN / "lh-svc")],
            env=os.environ | {"HOME": str(tmp_path), "XDG_STATE_HOME": str(tmp_path / "state")},
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the runner froze: its SIGTERM handler waited for good on its wait's lock")
    assert done.returncode == 0 and done.stdout.strip() == "stopped", done.stderr


def test_two_writes_of_a_units_state_at_once_keep_each_others_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # write_state reads the unit's state, merges its fields in and writes it back through a temporary file: two writers
    # at once, a client and its runner, lost each other's fields, or the second found the first's temporary file
    # renamed. Each now holds the state's own lock from its read to its rename. Here each read is held up long enough
    # for both writers to have read before either writes.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    lh_svc = _module("lh_svc", BIN / "lh-svc")
    read = lh_svc.read_state

    def slow_read(name: str) -> dict:
        values = read(name)
        time.sleep(0.3)
        return values

    monkeypatch.setattr(lh_svc, "read_state", slow_read)
    raised: list[Exception] = []

    def write(**values: object) -> None:
        try:
            lh_svc.write_state("both.service", **values)
        except Exception as exc:
            raised.append(exc)

    writers = [threading.Thread(target=write, kwargs={key: 1}) for key in ("first", "second")]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join()
    assert not raised and read("both.service").keys() >= {"first", "second"}, raised


def journal_times(host: dict, unit: str, message: str) -> list[float]:
    """When the unit's journal logged each line holding ``message``."""
    out = ctl(host, "-u", unit, "-o", "short-unix", tool="journalctl").stdout
    return [float(line.split(" ", 1)[0]) for line in out.splitlines() if message in line]


@pytest.mark.parametrize("status", [1, 0], ids=["restarted", "started-afresh"])
def test_a_unit_is_deactivating_while_what_is_left_of_its_group_is_ended(
    host: dict, status: int
) -> None:
    # As under systemd: from the main process's exit until what is left of its group has been ended, the unit is
    # deactivating, not active, and a start meanwhile waits for that to be over: it then joins the restart that
    # follows a failure, or, after an exit with status 0 (Restart=on-failure), starts the unit afresh. lh-svc said
    # active, and such a start returned at once, the unit not running. Here the child ignores its SIGINT for 3 s.
    home, py = host["home"], sys.executable
    (home / "leaves.py").write_text(LEAVES)
    (host["units"] / "leaves.service").write_text(
        f"[Service]\nExecStart={py} %h/leaves.py {status}\n"
        f"ExecStartPost={py} %h/wait.py %h/ready.${{MAINPID}}\n"
        "KillSignal=SIGINT\nTimeoutStopSec=3\nRestart=on-failure\nRestartSec=0.1\n"
    )
    try:
        assert ctl(host, "--user", "start", "leaves.service").returncode == 0
        (home / "exit").touch()  # the main process exits, leaving the child
        deadline = time.monotonic() + 10
        while not (exited := journal_times(host, "leaves", "Main process exited")):
            assert time.monotonic() < deadline, "the main process did not exit"
            time.sleep(0.05)
        # Not active: is-active exits 3, as systemd's does for a unit deactivating.
        state = ctl(host, "--user", "is-active", "leaves.service")
        assert (state.returncode, state.stdout.strip()) == (3, "deactivating")
        done = ctl(host, "--user", "start", "leaves.service")
        took = time.time() - exited[0]
        assert done.returncode == 0, done.stderr
        # It returned once that was over, TimeoutStopSec on, with the next main process ready: not at once, nor by
        # starting a runner of its own beside the one deactivating.
        assert took >= 3 and len(list(home.glob("ready.*"))) == 2, took
        assert ctl(host, "--user", "is-active", "leaves.service").stdout.strip() == "active"
    finally:
        ctl(host, "--user", "stop", "leaves.service")


def test_what_is_left_of_a_units_group_gets_its_kill_signal_then_sigkill(host: dict) -> None:
    # As systemd ends what is left of a unit's cgroup once its main process has exited: the unit's KillSignal first,
    # SIGKILL once TimeoutStopSec has passed, then ExecStopPost, which finds it gone, and the restart. Here the child
    # records each signal it gets and ignores it.
    home, py = host["home"], sys.executable
    (home / "leaves.py").write_text(LEAVES)
    alive = "kill -0 $(cat %h/child.pid) 2>/dev/null && echo alive || echo gone"
    (host["units"] / "leaves.service").write_text(
        f"[Service]\nExecStart={py} %h/leaves.py 1\nExecStartPost={py} %h/wait.py %h/ready.${{MAINPID}}\n"
        f"ExecStopPost=/bin/sh -c '({alive}) >> %h/post'\n"
        "KillSignal=SIGINT\nTimeoutStopSec=1\nRestart=on-failure\nRestartSec=0.1\n"
    )
    try:
        assert ctl(host, "--user", "start", "leaves.service").returncode == 0
        (home / "exit").touch()  # the main process exits 1, leaving the child
        deadline = time.monotonic() + 15
        while len(list(home.glob("ready.*"))) < 2:  # the restart's main process
            assert time.monotonic() < deadline, "the unit did not restart"
            time.sleep(0.05)
        signals = home / "signals"
        assert (signals.read_text().split() if signals.exists() else []) == ["SIGINT"]
        assert (home / "post").read_text().split()[0] == "gone"
        [exited] = journal_times(host, "leaves", "Main process exited, status=1")
        restarted = journal_times(host, "leaves", "lh-svc: Starting leaves.service")[1]
        assert restarted - exited >= 1.0, restarted - exited  # TimeoutStopSec
        # The journal says why the restart came so late, in systemd's words.
        [signaled] = journal_times(host, "leaves", "with signal SIGINT.")
        [killed] = journal_times(host, "leaves", "State 'stop-sigterm' timed out. Killing.")
        assert exited <= signaled and signaled + 1.0 <= killed <= restarted
    finally:
        ctl(host, "--user", "stop", "leaves.service")


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


@pytest.mark.parametrize(
    ("wrapper", "lines"),
    [("taskset", "CPUAffinity=0\n"), ("numactl", "NUMAPolicy=preferred\nNUMAMask=0\n")],
    ids=["taskset", "numactl"],
)
def test_a_start_fails_whose_command_is_not_there_though_a_wrapper_runs_it(
    host: dict, wrapper: str, lines: str
) -> None:
    # CPUAffinity= runs a unit's commands through taskset, NUMAPolicy= through numactl, which exec a command only once
    # they run themselves: one that is not there was then the wrapper's exit, a started unit's (a Type=simple unit has
    # started once its main process runs), and the start succeeded, the unit failing after it. The command is now found
    # first, and its absence fails the start, as the exec of the command alone fails it.
    if not shutil.which(wrapper):
        pytest.skip(f"the host has no {wrapper}")
    (host["units"] / "wrapped.service").write_text(
        f"[Service]\nExecStart=%h/no-such-server --port 1\n{lines}"
    )
    try:
        done = ctl(host, "--user", "start", "wrapped.service")
        assert done.returncode == 1 and "Job for wrapped.service failed" in done.stderr, done
        assert ctl(host, "--user", "is-active", "wrapped.service").stdout.strip() == "failed"
        journal = ctl(host, "-u", "wrapped", "-o", "cat", tool="journalctl").stdout
        missing = f"{host['home']}/no-such-server: [Errno 2] No such file or directory"
        assert missing in journal, journal
    finally:
        ctl(host, "--user", "stop", "wrapped.service")


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


@pytest.mark.skipif(
    not Path("/proc/sys/kernel/random/boot_id").exists(), reason="reads /proc (Linux)"
)
def test_lh_svc_reads_a_units_process_whose_name_the_kernel_cut_inside_a_character(
    host: dict,
) -> None:
    # The kernel keeps 15 bytes of a process's name (PR_SET_NAME's, a title's), which can end inside a character:
    # /proc/PID/stat is then not UTF-8, and lh-svc raised reading whether a unit's process lives and when it started.
    # A name may hold ") " too, which lh-svc took for the name's end, and so a live process for a zombie.
    state = Path(host["env"]["XDG_STATE_HOME"]) / "lh-svc"
    state.mkdir(parents=True, exist_ok=True)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    program = (
        "import ctypes, sys, time\n"
        "ctypes.CDLL(None).prctl(15, bytes.fromhex(sys.argv[1]), 0, 0, 0)  # PR_SET_NAME\n"
        "time.sleep(60)"
    )
    for name in (("abcdefghijklmn" + chr(0xE9)).encode(), b"x) Z y"):  # 16 bytes: é is two
        named = subprocess.Popen([sys.executable, "-c", program, name.hex()])
        try:
            deadline = time.monotonic() + 10
            while Path(f"/proc/{named.pid}/comm").read_bytes() != name[:15] + b"\n":
                assert time.monotonic() < deadline, "the process never took its name"
                time.sleep(0.02)
            stat = Path(f"/proc/{named.pid}/stat").read_bytes()
            started = stat.rsplit(b") ", 1)[1].split()[19].decode()
            # The runner of a unit, as its state names it: alive, of this boot, started when it did.
            (state / "named.service.json").write_text(
                json.dumps(
                    {"state": "active", "runner": named.pid, "runner_start": started, "boot": boot}
                )
            )
            done = ctl(host, "--user", "is-active", "named.service")
            assert (done.returncode, done.stdout.strip()) == (0, "active"), (name, done.stderr)
        finally:
            named.kill()
            named.wait()


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


@pytest.mark.skipif(
    not hasattr(os, "pidfd_open"),
    reason="the unit's ExecStop, lh-gateway-stop, holds the gateway by a pidfd (Linux)",
)
def test_cyans_gateway_unit_drains_at_a_stop_under_lh_svc(host: dict) -> None:
    # cyan's lh-gateway.service as the single host links it, only its ExecStart replaced by a stand-in that drains:
    # the stop runs the unit's own ExecStop (lh-gateway-stop with %n and $MAINPID), which sends the SIGTERM and
    # waits for the drain, and the unit ends inactive. Without %n, lh-svc's runner died at the ExecStop line: the
    # gateway, never signaled, was killed 5 s into the stop and the unit left failed.
    units, home, py = host["units"], host["home"], sys.executable
    for name in ("lh-gateway.service", "lh-gateway.socket"):
        (units / name).symlink_to(REPO / "deploy" / "cyan" / "systemd" / name)
    (units / "lh-gateway.service.d").mkdir()
    (units / "lh-gateway.service.d" / "stand-in.conf").write_text(
        f"[Service]\nExecStart=\nExecStart={py} %h/gateway.py %h/gateway.pid %h/gateway.drained\n"
    )
    (home / "gateway.py").write_text(DRAINING)
    (home / "leaseholder").mkdir()
    (home / "leaseholder" / "src").symlink_to(REPO)  # the unit's ExecStop and ExecReload
    try:
        done = ctl(host, "--user", "start", "lh-gateway.service")
        assert done.returncode == 0, done.stderr
        pid = int((home / "gateway.pid").read_text())
        begun = time.monotonic()
        done = ctl(host, "--user", "stop", "lh-gateway.service")
        took = time.monotonic() - begun
        assert done.returncode == 0, done.stderr
        drained = (home / "gateway.drained").exists()
        assert drained and gone(pid), (drained, took)
        assert 1.5 <= took < 4, took  # the drain's 1.5 s, not lh-svc's SIGKILL 5 s on
        state = ctl(host, "--user", "is-active", "lh-gateway.service").stdout.strip()
        journal = ctl(host, "-u", "lh-gateway", "-o", "cat", tool="journalctl").stdout
        assert state == "inactive", journal
        assert "ExecStop failed" not in journal and "lh-svc: Stopped lh-gateway.service" in journal
    finally:
        ctl(host, "--user", "stop", "lh-gateway.service", "lh-gateway.socket")


def test_a_stop_signals_the_main_process_when_exec_stop_cannot_run(host: dict) -> None:
    # As under systemd, a stop whose ExecStop fails, or cannot run at all (a command that is not there), still
    # sends the main process its KillSignal, and it drains. lh-svc's runner used to die at the command it could not
    # run, and the main process was killed 5 s into the stop, undrained.
    units, home, py = host["units"], host["home"], sys.executable
    (home / "gateway.py").write_text(DRAINING)
    (units / "drains.service").write_text(
        f"[Service]\nType=notify\nExecStart={py} %h/gateway.py %h/gateway.pid %h/gateway.drained\n"
        "ExecStop=%h/no-such-command $MAINPID\nExecStopPost=/usr/bin/touch %h/post.ran\n"
    )
    try:
        assert ctl(host, "--user", "start", "drains.service").returncode == 0
        pid = int((home / "gateway.pid").read_text())
        begun = time.monotonic()
        done = ctl(host, "--user", "stop", "drains.service")
        took = time.monotonic() - begun
        assert done.returncode == 0, done.stderr
        drained = (home / "gateway.drained").exists()
        assert drained and gone(pid), (drained, took)
        assert took < 4, took
        # ExecStopPost runs all the same, as systemd runs it.
        assert (home / "post.ran").exists()
        # The unit ends failed, as systemd's does when its ExecStop cannot run, with the cause in its journal.
        assert ctl(host, "--user", "is-active", "drains.service").stdout.strip() == "failed"
        journal = ctl(host, "-u", "drains", "-o", "cat", tool="journalctl").stdout
        assert f"lh-svc: {home}/no-such-command: " in journal, journal
    finally:
        ctl(host, "--user", "stop", "drains.service")


def test_a_failed_exec_stop_leaves_the_unit_failed(host: dict) -> None:
    # An ExecStop that runs and fails, as lh-gateway-stop does when it kills a gateway still running at the end of
    # its wait, leaves the unit failed, as systemd's does, so that is-failed shows the stop that did not drain; the
    # main process still gets its KillSignal and ExecStopPost runs.
    units, home, py = host["units"], host["home"], sys.executable
    (home / "gateway.py").write_text(DRAINING)
    (units / "drains.service").write_text(
        f"[Service]\nType=notify\nExecStart={py} %h/gateway.py %h/gateway.pid %h/gateway.drained\n"
        f"ExecStop={py} -c 'raise SystemExit(1)'\nExecStopPost=/usr/bin/touch %h/post.ran\n"
    )
    try:
        assert ctl(host, "--user", "start", "drains.service").returncode == 0
        pid = int((home / "gateway.pid").read_text())
        done = ctl(host, "--user", "stop", "drains.service")
        assert done.returncode == 0, done.stderr
        assert (home / "gateway.drained").exists() and gone(pid)
        assert (home / "post.ran").exists()
        assert ctl(host, "--user", "is-active", "drains.service").stdout.strip() == "failed"
        assert ctl(host, "--user", "is-failed", "drains.service").returncode == 0
        # A start clears it.
        assert ctl(host, "--user", "start", "drains.service").returncode == 0
        assert ctl(host, "--user", "is-active", "drains.service").stdout.strip() == "active"
    finally:
        ctl(host, "--user", "stop", "drains.service")


def test_lh_svc_expands_the_units_name(host: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    # %n and %N, the unit's name with and without its suffix, as the gateway's ExecReload and ExecStop pass it.
    units, py = host["units"], sys.executable
    (units / "named.service").write_text(
        f"[Service]\nEnvironment=READY=%h/named.ready\nExecStart={py} %h/svc.py %n %N\n"
        f"ExecStartPost={py} %h/wait.py %h/named.ready\n"
    )
    try:
        assert ctl(host, "--user", "start", "named.service").returncode == 0
        journal = ctl(host, "-u", "named", "-o", "cat", tool="journalctl").stdout
        assert "argv ['named.service', 'named']" in journal, journal
    finally:
        ctl(host, "--user", "stop", "named.service")
    for name, value in host["env"].items():
        monkeypatch.setenv(name, value)
    lh_svc = _module("lh_svc", BIN / "lh-svc")
    assert lh_svc.specifiers("%n %N %%n", "lh-tunnel@phastform.service") == (
        "lh-tunnel@phastform.service lh-tunnel@phastform %n"
    )


@pytest.mark.skipif(
    not Path("/usr/bin/rm").exists(),
    reason="runs the units' commands, Linux's /usr/bin/rm among them",
)
def test_a_vllm_servers_start_empties_its_own_directory_of_zmq_sockets(
    host: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # vLLM names the ZMQ sockets between its API server and its engine core afresh at each start
    # (VLLM_RPC_BASE_PATH/<uuid4>) and removes none at a stop: the box's runtime directory held 111 after five hours of
    # engine switches. Each vLLM unit, the box's and the template's, keeps them in a directory of its own, which its
    # start lines empty before the server starts (all of them but gpu-guard's, as lh-svc expands them).
    for name, value in host["env"].items():
        monkeypatch.setenv(name, value)
    lh_svc = _module("lh_svc", BIN / "lh-svc")
    lh = Path(host["env"]["XDG_RUNTIME_DIR"]) / "lh"
    for source in ("phastform", "templates"):
        dirs = set()
        for name in ("lh-engine.service", "lh-stt.service"):
            text = (REPO / "deploy" / source / "systemd" / name).read_text()
            (host["units"] / name).write_text(text)
            unit = lh_svc.load(name)
            env = lh_svc.environment(unit)
            zmq = Path(env["VLLM_RPC_BASE_PATH"])
            assert zmq.parent == lh, (source, name, zmq)
            zmq.mkdir(parents=True, exist_ok=True)
            (zmq / "0073c204-9efa-4d7b-abc0-0e7fd817fa54").touch()  # an earlier start's
            for line in unit.values["ExecStartPre"]:
                argv, _ = lh_svc.command(line, env, unit.name)
                if not argv[0].endswith("/gpu-guard"):
                    subprocess.run(argv, env=env, check=True)
            assert zmq.is_dir() and not any(zmq.iterdir()), (source, name)
            assert stat.S_IMODE(zmq.stat().st_mode) == 0o700, (source, name)
            dirs.add(zmq)
        assert len(dirs) == 2, source  # the engine's and the STT's own


def test_a_start_refuses_a_unit_whose_lines_hold_a_specifier_lh_svc_does_not_expand(
    host: dict,
) -> None:
    # systemd will not start a unit whose lines it cannot expand, and neither does lh-svc, whichever command line
    # holds the specifier (%S, the state directory, which none of the repository's units uses): a line that only a
    # stop or a reload runs fails the start, where starts are tested, rather than the stop, with the unit running.
    units, home, py = host["units"], host["home"], sys.executable
    good = f"Environment=READY=%h/refused.ready\nExecStart={py} %h/svc.py refused\n"
    keys = ("ExecStartPre", "ExecStart", "ExecStartPost", "ExecReload", "ExecStop", "ExecStopPost")
    for key in keys:
        bad = f"{key}=/bin/echo %S\n"
        (units / "refused.service").write_text(
            "[Service]\n" + (bad if key == "ExecStart" else good + bad)
        )
        try:
            done = ctl(host, "--user", "start", "refused.service")
            assert done.returncode == 5, (key, done.returncode, done.stderr)
            assert (
                f"refused.service: {key}=/bin/echo %S: specifier %S is not supported" in done.stderr
            ), (key, done.stderr)
            assert not (home / "refused.ready").exists(), key  # nothing ran
            assert ctl(host, "--user", "is-active", "refused").stdout.strip() == "inactive", key
        finally:
            ctl(host, "--user", "stop", "refused.service")
    # Also when the start only pulls it in, as lh-voice-worker's Wants= pulls the gateway: nothing of it starts.
    (units / "refused.service").write_text("[Service]\n" + good + "ExecStop=/bin/echo %S\n")
    (units / "pulls.target").write_text("[Unit]\nWants=svc-b.service refused.service\n")
    try:
        done = ctl(host, "--user", "start", "pulls.target")
        assert done.returncode == 5 and "specifier %S is not supported" in done.stderr, done.stderr
        assert not (home / "b.ready").exists() and not (home / "refused.ready").exists()
    finally:
        ctl(host, "--user", "stop", "pulls.target", "svc-b.service", "refused.service")


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
    # Processes on the GPU (a stand-in nvidia-smi lists them): one of lh-svc's units, which carries its LH_SVC_UNIT;
    # a child of a unit's process whose environment no longer shows the mark, as vLLM's engine core's does once
    # setproctitle has blanked /proc/PID/environ (the rehearsal on the box: the STT's engine core stopped the TTS's
    # and the engine's starts); and a foreign one. The guard passes the unit and its child and refuses the stranger,
    # which is detached (its starter gone, PID 1 or a subreaper adopts it), so that the guard's walk up its parents
    # does not reach this test's own process, which may run in a unit.
    sleep = [sys.executable, "-c", "import time; time.sleep(30)"]
    unmarked = {k: v for k, v in os.environ.items() if k != "LH_SVC_UNIT"}
    ours = subprocess.Popen(sleep, env=os.environ | {"LH_SVC_UNIT": "lh-stt.service"})
    detach = (
        "import os, sys\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"
        "    null = os.open(os.devnull, os.O_RDWR)\n"
        "    for fd in (0, 1, 2):\n"
        "        os.dup2(null, fd)  # the starter's pipes end with the starter\n"
        "    os.execv(sys.executable, [sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print(pid, flush=True)\n"
    )
    starter = subprocess.run(
        [sys.executable, "-c", detach], env=unmarked, capture_output=True, text=True, check=True
    )
    theirs = int(starter.stdout)
    spawn = (
        "import os, subprocess, sys, time\n"
        "env = {k: v for k, v in os.environ.items() if k != 'LH_SVC_UNIT'}\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], env=env)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", spawn],
        env=os.environ | {"LH_SVC_UNIT": "lh-stt.service"},
        stdout=subprocess.PIPE,
        text=True,
    )
    child = int(parent.stdout.readline())
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
        passed = guard(ours.pid, child)
        assert passed.returncode == 0, passed.stderr
        refused = guard(ours.pid, child, theirs)
        assert refused.returncode == 1 and f"pid {theirs}" in refused.stderr
    finally:
        for pid in (child, theirs):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        for process in (ours, parent):
            process.kill()
            process.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="gpu-guard runs on Linux (GNU stat)")
def test_gpu_guard_takes_a_units_cgroup_from_the_process_itself(tmp_path: Path) -> None:
    # Under systemd a unit's processes run in its cgroup, its engine core too. A process in another cgroup is not the
    # unit's, although its parent is (one started with systemd-run --scope, say): systemd neither counts nor stops it
    # with the unit, so the guard refuses it, as the listener audit does. A stand-in /proc (LH_PROCFS) holds them.
    proc = tmp_path / "proc"
    units = "/user.slice/user-1000.slice/user@1000.service/app.slice"

    def process(pid: int, cgroup: str, parent: int = 1) -> None:
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cgroup").write_text(f"0::{cgroup}\n")
        (proc / str(pid) / "status").write_text(f"Name:\tpython3\nPPid:\t{parent}\n")
        (proc / str(pid) / "environ").write_bytes(b"\0" * 64)  # setproctitle's blank
        (proc / str(pid) / "comm").write_text("python3\n")

    process(101, f"{units}/lh-engine.service")
    process(102, f"{units}/lh-engine.service", parent=101)  # its engine core
    process(103, f"{units}/run-r0123.scope", parent=101)
    process(104, "/user.slice/user-1000.slice/session-4.scope")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"

    def guard(*pids: int) -> subprocess.CompletedProcess[str]:
        smi.write_text("#!/bin/sh\n" + "".join(f"echo {pid}\n" for pid in pids))
        smi.chmod(0o755)
        return subprocess.run(
            ["bash", str(BIN / "gpu-guard")],
            env=os.environ
            | {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "LH_PROCFS": str(proc)},
            capture_output=True,
            text=True,
        )

    assert guard(101, 102).returncode == 0
    for stranger in (103, 104):
        refused = guard(101, 102, stranger)
        assert refused.returncode == 1, refused.stderr
        assert refused.stderr.startswith(f"gpu-guard: the GPU is in use by pid {stranger} (")
        assert "python3), outside the box's units" in refused.stderr


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="the audit reads /proc (Linux)")
def test_block_0s_smoke_test_passes_its_listener_audit_under_lh_svc(host: dict) -> None:
    # Block 0 runs the box's smoke.sh, whose listener audit reads the host's listeners (a stand-in ss lists them,
    # held by the units' real processes) and attributes each to its unit by /proc. Under lh-svc every process is in
    # the session's cgroup and the gateway binds 127.0.0.1:18080 itself; the voice plane runs in the host's network
    # namespace (the kernel refuses user namespaces: voicenet-holder with LH_VOICENET=host, in-voicenet), and so do
    # the GPU servers, started through in-private-net as the rendered units start them, with the mark host.py's
    # drop-ins give them (LH_PRIVATE_NET=host): the engine core's gloo listener and CUDA's socket are accepted there.
    units, home, py = host["units"], host["home"], sys.executable
    run = Path(host["env"]["XDG_RUNTIME_DIR"])
    (run / "lh").mkdir(mode=0o700)
    (home / "smoke_server.py").write_text(SMOKE_SERVER)
    for name in ("engine", "stt", "tts"):
        (units / f"lh-{name}.service").write_text(
            f"[Unit]\nPartOf=leaseholder.target\n[Service]\nEnvironment=READY=%h/lh-{name}.ready\n"
            "Environment=LH_PRIVATE_NET=host\n"
            f"ExecStart={BIN}/in-private-net {py} %h/smoke_server.py %t/lh/{name}.sock\n"
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
            f"tcp LISTEN 0 128 127.0.1.1:36087 0.0.0.0:* {held('engine')}\n"
        )
        (listed / "unix").write_text(
            "".join(
                f"u_str LISTEN 0 4096 {run}/lh/{name}.sock 3624 * 0 {held(name)}\n"
                for name in ("engine", "stt", "tts")
            )
            + f"u_str LISTEN 0 4096 @cuda-uvmfd-4026531836-1@ 3625 * 0 {held('stt')}\n"
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
            "the GPU servers run in the host's network namespace (LH_PRIVATE_NET=host) with 1 loopback TCP "
            "listeners and 1 abstract sockets; the voice plane runs in the host's network namespace "
            "(LH_VOICENET=host) with 3 listeners"
        ) in audit
        assert smoke.stdout.rstrip().endswith("smoke: ok")
    finally:
        ctl(host, "--user", "stop", *reversed(started))
