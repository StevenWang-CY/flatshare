"""How lh-bench reaches a server: by an address, ``http://host:port`` or ``unix:`` and the absolute path of its
socket, as the gateway's ``upstream`` is written. On cyan the engine, STT and TTS are the tunnel's sockets in
``$XDG_RUNTIME_DIR/lh/`` and the gateway is its TCP port and its own socket (ops/RUNBOOK.md); on a GPU node they are
the servers' own sockets there. A client's requests take paths from the server's root (``/metrics``,
``/v1/completions``)."""

from __future__ import annotations

import os
from typing import Any

import httpx

UNIX = "unix:"


def socket_path(address: str) -> str | None:
    """The socket of a ``unix:`` address, or None for an ``http://`` one."""
    if address.startswith(UNIX):
        path = address.removeprefix(UNIX)
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError(f"{address!r}: a unix: address names an absolute socket path")
        return path
    if not address.startswith("http://"):
        raise ValueError(f"{address!r}: expected http://host:port, or unix: and a socket path")
    return None


def client(address: str, **options: Any) -> httpx.Client:
    """An HTTP client of the server at ``address``; ``options`` go to ``httpx.Client``."""
    path = socket_path(address)
    if path is None:
        return httpx.Client(base_url=address, **options)
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=path), base_url="http://localhost", **options
    )


def async_client(address: str, **options: Any) -> httpx.AsyncClient:
    """An asynchronous HTTP client of the server at ``address``; ``options`` go to ``httpx.AsyncClient``."""
    path = socket_path(address)
    if path is None:
        return httpx.AsyncClient(base_url=address, **options)
    transport = httpx.AsyncHTTPTransport(uds=path)
    return httpx.AsyncClient(transport=transport, base_url="http://localhost", **options)


def runtime_socket(name: str) -> str:
    """The address of a socket in this user's ``$XDG_RUNTIME_DIR/lh/``, where the deployment keeps them
    (``phastform-engine.sock`` on cyan, ``engine.sock`` on a GPU node)."""
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return f"{UNIX}{runtime}/lh/{name}"
