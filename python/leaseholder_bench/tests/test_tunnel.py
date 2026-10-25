"""deploy/bin/lh-tunnel against a real sshd on loopback, as a GPU node authorizes cyan's tunnel keys: each key's every
session is `exec nc -q0 -U SOCKET` under `restrict`. Needs Linux, OpenSSH's sshd and OpenBSD's netcat (as the GPU
nodes have); skipped without them."""

from __future__ import annotations

import contextlib
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
TUNNEL = REPO / "deploy" / "bin" / "lh-tunnel"
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
    root = Path(tempfile.mkdtemp(prefix="lh-t-"))  # short: sockets' paths must fit sun_path
    lh = root / "node"
    lh.mkdir(mode=0o700)
    servers = {}
    for name in SOCKETS:
        server = _Server(str(lh / f"{name}.sock"), _Named)
        server.name = name  # type: ignore[attr-defined]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers[name] = server
    home = root / "home"
    (home / ".ssh").mkdir(parents=True, mode=0o700)
    entries = []
    for name in SOCKETS:
        key = home / ".ssh" / f"lh_tunnel_{name}_ed25519"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key], check=True)
        public = key.with_suffix(".pub").read_text().strip()
        # The entry deploy/authorize-tunnel.sh writes, with this node's socket.
        entries.append(f'restrict,command="exec nc -q0 -U {lh}/{name}.sock" {public}\n')
    (root / "authorized_keys").write_text("".join(entries))
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", root / "host"], check=True)
    port = free_port()
    (root / "sshd_config").write_text(
        f"Port {port}\nListenAddress 127.0.0.1\nHostKey {root}/host\nPidFile {root}/sshd.pid\n"
        f"AuthorizedKeysFile {root}/authorized_keys\nUsePAM no\nPasswordAuthentication no\n"
        "KbdInteractiveAuthentication no\nStrictModes no\nMaxSessions 3\nLogLevel VERBOSE\n"
    )
    sshd = subprocess.Popen([SSHD, "-D", "-f", root / "sshd_config", "-E", root / "sshd.log"])
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
        "LH_TUNNEL_TARGET": "gpu-node",
        "LH_TUNNEL_SESSIONS": "3",
    }
    wait_for(lambda: (root / "sshd.pid").exists(), "sshd started")
    yield {"root": root, "lh": lh, "servers": servers, "runtime": runtime, "env": env}
    # Each connection's sshd runs in a session of its own (setsid), which outlives the listener: end them first.
    for child in children(sshd.pid):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child, signal.SIGTERM)
    sshd.terminate()
    sshd.wait(5)
    for server in servers.values():
        server.shutdown()
    shutil.rmtree(root, ignore_errors=True)


def children(pid: int) -> list[int]:
    """A process's children (Linux)."""
    path = Path(f"/proc/{pid}/task/{pid}/children")
    return [int(child) for child in path.read_text().split()] if path.exists() else []


def masters(node: dict) -> list[Path]:
    """The tunnel's masters' control sockets: each master removes its own as it ends."""
    return list((node["runtime"] / "lh" / "ssh-phastform").glob("*-*"))


def start(node: dict) -> subprocess.Popen:
    tunnel = subprocess.Popen(
        [sys.executable, TUNNEL, "phastform"], env=node["env"], stderr=subprocess.PIPE, text=True
    )
    for name in SOCKETS:
        path = node["runtime"] / "lh" / f"phastform-{name}.sock"
        wait_for(path.exists, f"{path} listening")
    return tunnel


def test_each_socket_reaches_the_nodes_socket_of_its_name_through_its_own_key(node: dict) -> None:
    tunnel = start(node)
    try:
        lh = node["runtime"] / "lh"
        assert oct(lh.stat().st_mode & 0o777) == "0o700"
        for name in SOCKETS:
            path = lh / f"phastform-{name}.sock"
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
    assert not any((node["runtime"] / "lh").glob("phastform-*.sock"))
    assert masters(node) == []


def test_connections_beyond_a_masters_sessions_go_through_more_masters(node: dict) -> None:
    tunnel = start(node)
    try:
        engine = node["runtime"] / "lh" / "phastform-engine.sock"
        get(engine)[1].close()  # the first master is up
        held = [get(engine)[1] for _ in range(8)]  # sshd allows 3 sessions a connection here
        assert all(get(engine, connection)[0] == b"engine" for connection in held)
        controls = list((node["runtime"] / "lh" / "ssh-phastform").glob("engine-*"))
        assert len(controls) >= 3, controls
        for connection in held:
            connection.close()
    finally:
        tunnel.send_signal(signal.SIGTERM)
        tunnel.wait(10)
    assert masters(node) == [], "every master ends with the tunnel"


def test_a_dead_server_on_the_node_closes_the_connection_without_an_answer(node: dict) -> None:
    tunnel = start(node)
    try:
        engine = node["runtime"] / "lh" / "phastform-engine.sock"
        get(engine)[1].close()
        # The engine dies: its socket is gone, as a killed engine's is until its unit removes the file.
        node["servers"]["engine"].shutdown()
        node["servers"]["engine"].server_close()
        (node["lh"] / "engine.sock").unlink()
        started = time.monotonic()
        with pytest.raises(ConnectionError, match="closed the connection"):
            get(engine)
        assert time.monotonic() - started < 2
        # The other sockets still answer.
        assert get(node["runtime"] / "lh" / "phastform-stt.sock")[0] == b"stt"
    finally:
        tunnel.send_signal(signal.SIGTERM)
        tunnel.wait(10)


def test_a_tunnel_key_opens_no_forward_and_reaches_no_other_socket(node: dict) -> None:
    root = node["root"]
    key = root / "home" / ".ssh" / "lh_tunnel_engine_ed25519"
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
