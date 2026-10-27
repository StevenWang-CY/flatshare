"""The servers that tests stand up in process, as an engine, a gateway or a node's sockets, on loopback or on a
Unix socket. :func:`serve` runs one from a daemon thread, which looks for its shutdown every POLL_S: socketserver's
default, half a second, is what each shutdown would wait, three times over at the end of every E2 point's test.
:func:`stop` shuts it down, closes its listener and removes its socket's file and, once empty, the socket's
directory, which :func:`socket_file` makes."""

from __future__ import annotations

import contextlib
import socketserver
import tempfile
import threading
from pathlib import Path

POLL_S = 0.01
"""How often a serving thread looks for its shutdown."""


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """A server on a Unix socket that handles each connection in a thread of its own, which does not keep the
    tests' process alive."""

    daemon_threads = True


def socket_file(prefix: str, name: str = "s.sock") -> Path:
    """A socket's path, ``name`` in a new directory of its own: short, since a socket's path must fit sun_path
    (104 bytes on macOS), which a test's tmp_path may not."""
    return Path(tempfile.mkdtemp(prefix=prefix)) / name


def serve[Server: socketserver.BaseServer](server: Server) -> Server:
    """``server``, serving from a daemon thread until :func:`stop`."""
    threading.Thread(target=server.serve_forever, args=(POLL_S,), daemon=True).start()
    return server


def stop(server: socketserver.BaseServer) -> None:
    """Stop ``server`` and close its listener; on a Unix socket, its file goes, and its directory once empty."""
    server.shutdown()
    server.server_close()
    if isinstance(server, socketserver.UnixStreamServer):
        path = Path(server.server_address)
        path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):  # not empty: another socket, or what the test put there
            path.parent.rmdir()
