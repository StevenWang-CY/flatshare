"""Addresses: a server over TCP, or on its Unix socket, as the gateway's ``upstream`` names one."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler

import pytest
from servers import UnixServer, serve, socket_file, stop

from flatshare_bench.cli import main
from flatshare_bench.kv_metrics import fetch
from flatshare_bench.transport import client, runtime_socket, socket_path


class _Metrics(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = f"seen {self.path}\n".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def test_an_address_is_http_or_a_unix_socket() -> None:
    assert socket_path("unix:/run/user/1007/flatshare/phastform-engine.sock") == (
        "/run/user/1007/flatshare/phastform-engine.sock"
    )
    assert socket_path("http://127.0.0.1:18080") is None
    for wrong in (
        "unix:flatshare/engine.sock",
        "unix://cyan/engine.sock",
        "127.0.0.1:18000",
        "https://e",
    ):
        with pytest.raises(ValueError, match=r"expected http://host:port|absolute socket path"):
            socket_path(wrong)


@pytest.mark.parametrize(
    "command",
    [
        ["kv-metrics", "snapshot", "{}", "out.prom"],
        ["tiers", "--url", "{}", "--out", "run"],
        ["voice-turn", "--audio", "turn.wav", "--llm", "{}", "--out", "run"],
        ["drill", "cpu-tier", "--engine", "{}", "--out", "record.json"],
    ],
)
def test_a_command_refuses_an_address_that_is_none_before_it_runs(
    command: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    # Taken as it was given, such an address failed once the command made its client, with a traceback.
    with pytest.raises(SystemExit) as refused:
        main([word.format("unix:flatshare/engine.sock") for word in command])
    assert refused.value.code == 2
    assert "'unix:flatshare/engine.sock': a unix: address names an absolute socket path" in (
        capsys.readouterr().err
    )


def test_a_unix_address_reaches_the_server_on_its_socket() -> None:
    path = socket_file("flatshare-transport-")
    server = serve(UnixServer(str(path), _Metrics))
    try:
        with client(f"unix:{path}") as http:
            assert http.get("/server_info", params={"config_format": "json"}).text == (
                "seen /server_info?config_format=json\n"
            )
        assert fetch(f"unix:{path}") == "seen /metrics\n"
    finally:
        stop(server)


def test_the_deployments_sockets_live_in_the_users_runtime_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1007")
    assert runtime_socket("gateway.sock") == "unix:/run/user/1007/flatshare/gateway.sock"
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert runtime_socket("engine.sock").endswith("/flatshare/engine.sock")
