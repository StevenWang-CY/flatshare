"""deploy/bin/flatshare-tunnel against a real sshd on loopback, as a GPU node authorizes cyan's tunnel keys: each key's
every session is `exec nc -q0 -U SOCKET` under `restrict`. The tunnel runs on the host's /usr/bin/python3 where there is
one, as its unit runs it (3.10 on Ubuntu 22.04). Needs Linux, OpenSSH's sshd and OpenBSD's netcat (as the GPU nodes
have); skipped without them."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import importlib.machinery
import importlib.util
import os
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TUNNEL = REPO / "deploy" / "bin" / "flatshare-tunnel"
PYTHON = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
SSHD = Path("/usr/sbin/sshd")
SOCKETS = ("engine", "stt", "tts")


def _openbsd_nc() -> bool:
    nc = shutil.which("nc")
    if nc is None:
        return False
    usage = subprocess.run([nc, "-h"], capture_output=True, text=True)
    return "-U" in usage.stdout + usage.stderr and "-q" in usage.stdout + usage.stderr


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not SSHD.exists() or not _openbsd_nc(),
    reason="needs Linux, sshd and OpenBSD's netcat",
)


class _Named(BaseHTTPRequestHandler):
    """A server on a node's socket: answers with its socket's name; keeps connections open (HTTP/1.1)."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        body = self.server.name.encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def get(path: Path, connection: socket.socket | None = None) -> tuple[bytes, socket.socket]:
    """GET /health on the socket at ``path`` (or on ``connection``, kept open); the body and the connection."""
    if connection is None:
        connection = socket.socket(socket.AF_UNIX)
        connection.settimeout(10)
        connection.connect(str(path))
    connection.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
    answer = b""
    while True:
        head, end, body = answer.partition(b"\r\n\r\n")
        if end:
            fields = dict(line.split(b": ", 1) for line in head.split(b"\r\n")[1:])
            length = int(fields[b"Content-Length"])
            if len(body) >= length:
                return body[:length], connection
        chunk = connection.recv(4096)
        if not chunk:
            raise ConnectionError(f"the tunnel closed the connection: {answer!r}")
        answer += chunk


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for(condition, what: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"{what} within {timeout} s"
        time.sleep(0.05)


@pytest.fixture
def node() -> Iterator[dict]:
    """A GPU node on loopback (its sshd and its three servers) and cyan's side (keys, ssh's configuration)."""
    root = Path(tempfile.mkdtemp(prefix="flatshare-t-"))  # short: sockets' paths must fit sun_path
    socket_dir = root / "node"
    socket_dir.mkdir(mode=0o700)
    servers = {}
    for name in SOCKETS:
        server = _Server(str(socket_dir / f"{name}.sock"), _Named)
        server.name = name  # type: ignore[attr-defined]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers[name] = server
    home = root / "home"
    (home / ".ssh").mkdir(parents=True, mode=0o700)
    entries = []
    for name in SOCKETS:
        key = home / ".ssh" / f"flatshare_tunnel_{name}_ed25519"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key], check=True)
        public = key.with_suffix(".pub").read_text().strip()
        # The entry deploy/authorize-tunnel.sh writes, with this node's socket.
        entries.append(f'restrict,command="exec nc -q0 -U {socket_dir}/{name}.sock" {public}\n')
    (root / "authorized_keys").write_text("".join(entries))
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", root / "host"], check=True)
    port = free_port()
    (root / "sshd_config").write_text(
        f"Port {port}\nListenAddress 127.0.0.1\nHostKey {root}/host\nPidFile {root}/sshd.pid\n"
        f"AuthorizedKeysFile {root}/authorized_keys\nUsePAM no\nPasswordAuthentication no\n"
        "KbdInteractiveAuthentication no\nStrictModes no\nMaxSessions 3\nMaxStartups 6:100:6\n"
        "LogLevel VERBOSE\n"
    )
    (root / "ssh_config").write_text(
        f"Host gpu-node\n  HostName 127.0.0.1\n  Port {port}\n  UserKnownHostsFile {root}/known_hosts\n"
        "  StrictHostKeyChecking no\n"
    )
    # ssh as cyan runs it, with this node's configuration.
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text(f'#!/bin/sh\nexec /usr/bin/ssh -F {root}/ssh_config "$@"\n')
    (bin_dir / "ssh").chmod(0o755)
    runtime = root / "run"
    runtime.mkdir(mode=0o700)
    env = os.environ | {
        "HOME": str(home),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "XDG_RUNTIME_DIR": str(runtime),
        "FLATSHARE_TUNNEL_TARGET": "gpu-node",
        "FLATSHARE_TUNNEL_SESSIONS": "3",
    }
    node = {
        "root": root,
        "socket_dir": socket_dir,
        "servers": servers,
        "runtime": runtime,
        "env": env,
        "sshd": [],
    }
    start_sshd(node)
    yield node
    stop_sshd(node)
    for server in servers.values():
        server.shutdown()
    shutil.rmtree(root, ignore_errors=True)


def start_sshd(node: dict) -> None:
    root = node["root"]
    with contextlib.suppress(FileNotFoundError):
        (root / "sshd.pid").unlink()  # a killed sshd's
    node["sshd"].append(
        subprocess.Popen([SSHD, "-D", "-f", root / "sshd_config", "-E", root / "sshd.log"])
    )
    wait_for(lambda: (root / "sshd.pid").exists(), "sshd started")


def stop_sshd(node: dict, signum: int = signal.SIGTERM) -> None:
    """Ends the node's sshd: each connection's first, which runs in a session of its own (setsid) and outlives the
    listener, then the listener."""
    for sshd in node["sshd"]:
        for child in children(sshd.pid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child, signum)
        sshd.send_signal(signum)
        sshd.wait(5)
    node["sshd"].clear()


def children(pid: int) -> list[int]:
    """A process's children (Linux)."""
    path = Path(f"/proc/{pid}/task/{pid}/children")
    return [int(child) for child in path.read_text().split()] if path.exists() else []


def probe(path: Path, timeout: float) -> bool:
    """The gateway's probe of a recovering upstream: GET /health within ``timeout``, the connection dropped after."""
    connection = socket.socket(socket.AF_UNIX)
    connection.settimeout(timeout)
    try:
        connection.connect(str(path))
        return get(path, connection)[0] == b"engine"
    except (OSError, ConnectionError):
        return False
    finally:
        connection.close()


def masters(node: dict) -> list[Path]:
    """The tunnel's masters' control sockets: each master removes its own as it ends."""
    return list((node["runtime"] / "flatshare" / "ssh-phastform").glob("*-*"))


def start(node: dict, runtime: Path | None = None) -> subprocess.Popen:
    runtime = runtime or node["runtime"]
    tunnel = subprocess.Popen(
        [PYTHON, TUNNEL, "phastform"],
        env=node["env"] | {"XDG_RUNTIME_DIR": str(runtime)},
        stderr=subprocess.PIPE,
        text=True,
    )
    for name in SOCKETS:
        path = runtime / "flatshare" / f"phastform-{name}.sock"
        wait_for(path.exists, f"{path} listening")
    return tunnel


def test_each_socket_reaches_the_nodes_socket_of_its_name_through_its_own_key(node: dict) -> None:
    tunnel = start(node)
    try:
        socket_dir = node["runtime"] / "flatshare"
        assert oct(socket_dir.stat().st_mode & 0o777) == "0o700"
        for name in SOCKETS:
            path = socket_dir / f"phastform-{name}.sock"
            assert oct(path.stat().st_mode & 0o777) == "0o600"
            body, connection = get(path)
            assert body == name.encode()
            # The connection stays open for the next request, as the gateway's pooled ones do.
            assert get(path, connection)[0] == name.encode()
            connection.close()
    finally:
        tunnel.send_signal(signal.SIGTERM)
        assert tunnel.wait(10) == 0
    # It removes its sockets as it stops, and ends its masters.
    assert not any((node["runtime"] / "flatshare").glob("phastform-*.sock"))
    assert masters(node) == []


def test_connections_beyond_a_masters_sessions_go_through_more_masters(node: dict) -> None:
    tunnel = start(node)
    try:
        engine = node["runtime"] / "flatshare" / "phastform-engine.sock"
        get(engine)[1].close()  # the first master is up
        held = [get(engine)[1] for _ in range(8)]  # sshd allows 3 sessions a connection here
        assert all(get(engine, connection)[0] == b"engine" for connection in held)
        controls = list((node["runtime"] / "flatshare" / "ssh-phastform").glob("engine-*"))
        assert len(controls) >= 3, controls
        for connection in held:
            connection.close()
    finally:
        tunnel.send_signal(signal.SIGTERM)
        tunnel.wait(10)
    assert masters(node) == [], "every master ends with the tunnel"


def test_a_burst_of_connections_waits_for_masters_rather_than_connect_on_its_own(
    node: dict,
) -> None:
    # A gateway restarted with a backlog of requests opens its upstream connections at once: about 40 through the
    # engine's socket in the restart drill, where the node's sshd refused the sessions past a master's and dropped the
    # connections they then made on their own past its MaxStartups. sshd here takes 3 sessions a connection and fewer
    # than 6 handshakes at once (MaxStartups 6:100:6): 21 connections held at once need 7 masters, which the tunnel
    # opens 3 at a time (OPENING); were they opened all at once, or each connection to connect on its own, sshd would
    # drop some.
    tunnel = start(node)
    burst = 21
    go, done = threading.Barrier(burst), threading.Event()
    answers: list[bytes] = []
    errors: list[Exception] = []

    def one() -> None:
        try:
            go.wait(10)
            body, connection = get(engine)
            answers.append(body)
            done.wait(30)  # held, as the gateway's pooled connections are
            connection.close()
        except Exception as exc:
            errors.append(exc)

    try:
        engine = node["runtime"] / "flatshare" / "phastform-engine.sock"
        get(engine)[1].close()  # the first master is up
        threads = [threading.Thread(target=one) for _ in range(burst)]
        for thread in threads:
            thread.start()
        wait_for(
            lambda: len(answers) + len(errors) == burst, "every connection answered or failed", 30
        )
        done.set()
        for thread in threads:
            thread.join(10)
        assert not errors and answers == [b"engine"] * burst, errors
        log = (node["root"] / "sshd.log").read_text()
        assert "past MaxStartups" not in log
        # No master was asked for a session past its sshd's MaxSessions (it refuses it, and its client connects on
        # its own): OpenSSH logs each.
        assert "no more sessions" not in log
    finally:
        done.set()
        tunnel.send_signal(signal.SIGTERM)
        tunnel.wait(10)


def test_after_an_outage_a_waiting_connection_is_served_as_soon_as_the_node_answers(
    node: dict,
) -> None:
    # The gateway's probe of a recovering engine asks /health with a 1 s deadline, every second, and the gateway
    # answers 503 until one succeeds. While the node is down the masters fail, and the backoff before the next grows
    # to 10 s and more; a connection that waits for a master cuts it to 1 s, since nothing else serves it.
    tunnel = start(node)
    lines: list[str] = []
    threading.Thread(target=lambda: lines.extend(tunnel.stderr), daemon=True).start()
    try:
        engine = node["runtime"] / "flatshare" / "phastform-engine.sock"
        get(engine)[1].close()
        # The node is gone: its connections end, and new ones are refused.
        stop_sshd(node, signal.SIGKILL)
        ended = "phastform-engine.sock: master engine-"
        failed = lambda: sum(ended in line and "ended after" in line for line in lines)  # noqa: E731
        wait_for(lambda: failed() >= 4, "four masters failed in a row (a 10 s backoff)", 30)
        start_sshd(node)
        back = time.monotonic()
        while not probe(engine, 1.0):
            assert time.monotonic() - back < 4, "a probe answered within 4 s of the node's return"
    finally:
        tunnel.send_signal(signal.SIGTERM)
        tunnel.wait(10)


def test_a_connection_that_no_master_serves_connects_on_its_own_after_waiting(
    node: dict,
) -> None:
    # Where no master can open but the node takes a connection's own, a connection waits WAIT_S (10 s), then connects
    # on its own. Here the runtime directory is long enough that a master's control socket does not fit a socket's
    # path (ssh -M binds it as PATH.XXXXXXXXXXXXXXXX first: 75 + 26 + 17 characters, past 107), while the tunnel's
    # own sockets do (75 + 25). Python 3.10's asyncio raises its own TimeoutError at the wait's end, not the builtin.
    runtime = node["root"] / ("r" * (75 - len(str(node["root"])) - 1))
    runtime.mkdir(mode=0o700)
    tunnel = start(node, runtime)
    try:
        engine = runtime / "flatshare" / "phastform-engine.sock"
        connection = socket.socket(socket.AF_UNIX)
        connection.settimeout(30)
        started = time.monotonic()
        connection.connect(str(engine))
        assert get(engine, connection)[0] == b"engine"
        assert 9.5 < time.monotonic() - started < 15
        connection.close()
    finally:
        tunnel.send_signal(signal.SIGTERM)
        assert tunnel.wait(10) == 0
    assert "Task exception was never retrieved" not in tunnel.stderr.read()


def test_a_master_whose_ssh_cannot_start_holds_no_room(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Out of file descriptors or memory, a master's ssh may not start: the master goes as one that failed to connect
    # does. Were it kept, it would count as opening for good, and the socket would open no other.
    loader = importlib.machinery.SourceFileLoader("flatshare_tunnel", str(TUNNEL))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)

    async def no_ssh(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EMFILE, "Too many open files")

    async def scenario() -> None:
        tunnel = module.Tunnel("phastform", "engine", "gpu-node", tmp_path, 3)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", no_ssh)
        assert tunnel.open_master()
        await asyncio.sleep(0.2)
        assert tunnel.masters == [] and tunnel.failures == 1
        tunnel.closing = True
        tunnel.listener.close()

    asyncio.run(scenario())


def test_a_dead_server_on_the_node_closes_the_connection_without_an_answer(node: dict) -> None:
    tunnel = start(node)
    try:
        engine = node["runtime"] / "flatshare" / "phastform-engine.sock"
        get(engine)[1].close()
        # The engine dies: its socket is gone, as a killed engine's is until its unit removes the file.
        node["servers"]["engine"].shutdown()
        node["servers"]["engine"].server_close()
        (node["socket_dir"] / "engine.sock").unlink()
        started = time.monotonic()
        with pytest.raises(ConnectionError, match="closed the connection"):
            get(engine)
        assert time.monotonic() - started < 2
        # The other sockets still answer.
        assert get(node["runtime"] / "flatshare" / "phastform-stt.sock")[0] == b"stt"
    finally:
        tunnel.send_signal(signal.SIGTERM)
        tunnel.wait(10)


def test_a_tunnel_key_opens_no_forward_and_reaches_no_other_socket(node: dict) -> None:
    root = node["root"]
    key = root / "home" / ".ssh" / "flatshare_tunnel_engine_ed25519"
    ssh = [
        str(root / "bin" / "ssh"),
        "-i",
        str(key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
    ]
    # A forward to the node's STT socket: ssh listens here, and the node refuses each connection's channel.
    stolen = root / "stolen.sock"
    forward = subprocess.Popen(
        [*ssh, "-N", "-L", f"{stolen}:{root}/node/stt.sock", "gpu-node"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for(stolen.exists, "the forward's local socket")
        with pytest.raises(ConnectionError):  # closed or reset
            get(stolen)
    finally:
        forward.terminate()
        _, error = forward.communicate(timeout=10)
    assert b"channel 1: open failed" in error  # sshd: "refused streamlocal port forward"
    # Whatever the client asks to run, the session is the bridge to the engine's socket.
    asked = subprocess.run(
        [*ssh, "-T", "gpu-node", "cat /etc/passwd"],
        input=b"GET /health HTTP/1.0\r\n\r\n",
        capture_output=True,
        timeout=20,
    )
    assert asked.stdout.endswith(b"engine")
