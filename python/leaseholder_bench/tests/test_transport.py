"""Addresses: a server over TCP, or on its Unix socket, as the gateway's ``upstream`` names one."""

from __future__ import annotations

import socketserver
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from leaseholder_bench.kv_metrics import fetch
from leaseholder_bench.transport import client, runtime_socket, socket_path


class _Metrics(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = f"seen {self.path}\n".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def test_an_address_is_http_or_a_unix_socket() -> None:
    assert socket_path("unix:/run/user/1007/lh/phastform-engine.sock") == (
        "/run/user/1007/lh/phastform-engine.sock"
    )
    assert socket_path("http://127.0.0.1:18080") is None
    for wrong in ("unix:lh/engine.sock", "unix://cyan/engine.sock", "127.0.0.1:18000", "https://e"):
        with pytest.raises(ValueError, match=r"expected http://host:port|absolute socket path"):
            socket_path(wrong)


def test_a_unix_address_reaches_the_server_on_its_socket() -> None:
    path = Path(tempfile.mkdtemp(prefix="lh-transport-")) / "s.sock"
    server = _UnixServer(str(path), _Metrics)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with client(f"unix:{path}") as http:
            assert http.get("/server_info", params={"config_format": "json"}).text == (
                "seen /server_info?config_format=json\n"
            )
        assert fetch(f"unix:{path}") == "seen /metrics\n"
    finally:
        server.shutdown()


def test_the_deployments_sockets_live_in_the_users_runtime_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1007")
    assert runtime_socket("gateway.sock") == "unix:/run/user/1007/lh/gateway.sock"
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert runtime_socket("engine.sock").endswith("/lh/engine.sock")
