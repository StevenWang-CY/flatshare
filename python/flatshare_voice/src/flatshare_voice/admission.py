"""Call-capacity admission for the voice worker's serve mode.

A box admits as many calls as its measured envelope does (the serving profile's ``voice.max_calls``), and no new call
while its engine does not serve or while a release drains the worker. ``Admission`` decides both of LiveKit's ways in:

- the worker's **load**, which LiveKit reads every 0.5 s and before it answers a job request: the calls in progress or
  being accepted over ``max_calls``, and 1 while the worker takes no call. With LiveKit's load threshold at 1 the
  worker is marked full at ``max_calls`` and the server sends it no job;
- each **job request**, refused once the calls in progress and being accepted reach ``max_calls``. LiveKit's own check
  of the load counts a request it has not launched yet at a guessed share, so two requests that arrive together could
  both pass it; this count is exact, since the event loop runs one request's decision at a time.

``EngineProbe`` asks the LLM's socket whether the engine serves: the gateway's ``/ready``, which answers 200 only while
it does (warming, degraded or drained otherwise, and 503 while the engine's ``/health`` fails), or, when the LLM
bypasses the gateway, the engine's own ``/health``. A gateway without ``/ready`` (404 or 405) leaves admission to
capacity alone, said once in the log; one that does not answer within a second takes no call. ``Monitor`` asks it
every second and writes the worker's state to a file, which a release reads to drain the worker
(deploy/bin/flatshare-release) and deploy/bin/serving-check holds to the profile.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

PROBE_INTERVAL_S = 1.0
"""How often the probe asks the LLM's socket whether the engine serves."""
PROBE_TIMEOUT_S = 1.0
"""How long an answer may take: the gateway answers /ready within its own one-second window."""
SCHEMA = 1
"""The status file's format (deploy/bin/serving-check's)."""


class _UnixConnection(http.client.HTTPConnection):
    """HTTP/1.1 to a server on a Unix socket."""

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self._path)
        except OSError:
            sock.close()
            raise
        self.sock = sock


def get(uds: str, path: str, timeout: float = PROBE_TIMEOUT_S) -> tuple[int, str]:
    """GET ``path`` on the server at ``uds``: its status and the first line of its body."""
    connection = _UnixConnection(uds, timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read(512).decode("utf-8", "replace")
        return response.status, body.strip().splitlines()[0][:120] if body.strip() else ""
    finally:
        connection.close()


class EngineProbe:
    """Whether the engine serves, by the LLM's socket ``uds``: ``path`` is ``/ready`` on the gateway's and ``/health``
    on an engine's. ``ask`` asks once, and ``state`` is the latest answer (none yet: not serving)."""

    def __init__(
        self, uds: str, path: str, fetch: Callable[[str, str], tuple[int, str]] = get
    ) -> None:
        self.uds, self.path = uds, path
        self._fetch = fetch
        self._state = (False, "not asked yet")
        self._said_no_route = False

    def ask(self) -> tuple[bool, str]:
        try:
            status, line = self._fetch(self.uds, self.path)
        except (OSError, http.client.HTTPException) as error:
            state = (False, f"{self.path} on {self.uds} did not answer: {error}")
        else:
            if status == 200:
                state = (True, f"{self.path} answered 200")
            elif status in (404, 405):
                # A server without the route says nothing of the engine: capacity alone admits.
                if not self._said_no_route:
                    logger.warning(
                        "the LLM's socket serves no %s (%d): admitting calls by capacity alone",
                        self.path,
                        status,
                    )
                    self._said_no_route = True
                state = (True, f"{self.path} is not served ({status}): capacity alone admits")
            else:
                state = (False, f"{self.path} answered {status}" + (f": {line}" if line else ""))
        if state[0] != self._state[0]:
            log = logger.info if state[0] else logger.warning
            log("the engine %s: %s", "serves" if state[0] else "does not serve", state[1])
        self._state = state
        return state

    def state(self) -> tuple[bool, str]:
        return self._state


class JobRequest(Protocol):
    """What admission needs of LiveKit's JobRequest (its room, for the call's placement)."""

    @property
    def room(self) -> Any: ...

    async def accept(self) -> None: ...

    async def reject(self, *, terminate: bool = ...) -> None: ...


class Admission:
    """Serve mode's admission: at most ``max_calls`` calls, none while ``probe`` says the engine does not serve or
    while ``drain`` exists. ``active`` counts the calls in progress (LiveKit's running jobs)."""

    def __init__(
        self,
        max_calls: int,
        active: Callable[[], int],
        probe: EngineProbe | None = None,
        drain: str | os.PathLike[str] | None = None,
        place: Callable[[JobRequest], Awaitable[tuple[bool, str]]] | None = None,
    ) -> None:
        if max_calls < 1:
            raise ValueError(f"max_calls must be 1 or more, got {max_calls}")
        self.max_calls = max_calls
        self.active = active
        self.probe = probe
        self.drain = Path(drain) if drain else None
        self.place = place
        """With nodes, whether one takes the job's call (:func:`flatshare_voice.nodes.admit`), asked before it is
        accepted, so that a call no node can serve is refused rather than accepted and left without speech."""
        self.accepting = 0
        self.refused = 0

    def calls(self) -> int:
        """The calls in progress and those accepted but not yet running."""
        return self.active() + self.accepting

    def gate(self) -> tuple[bool, str]:
        """Whether new calls may come at all, and why."""
        if self.drain is not None and self.drain.exists():
            return False, f"draining: {self.drain} exists"
        if self.probe is not None:
            serving, why = self.probe.state()
            if not serving:
                return False, f"the engine does not serve: {why}"
        return True, "the engine serves"

    def load(self) -> float:
        """The worker's load for LiveKit: its calls over max_calls, and 1 while it takes none."""
        if not self.gate()[0]:
            return 1.0
        return min(1.0, self.calls() / self.max_calls)

    async def request(self, job: JobRequest) -> None:
        """Accept a job request within capacity, refuse it otherwise. A refused request may go to another worker
        (``terminate=False``); with none, the room gets no agent."""
        admitting, why = self.gate()
        calls = self.calls()
        if not admitting or calls >= self.max_calls:
            self.refused += 1
            reason = why if not admitting else f"{calls} of {self.max_calls} calls in progress"
            logger.warning("refusing a call: %s", reason, extra={"refused": self.refused})
            await job.reject(terminate=False)
            return
        # No await between the count and this: the event loop decides one request at a time, and the call counts as
        # being accepted while its placement is asked.
        self.accepting += 1
        try:
            if self.place is not None:
                placed, why = await self.place(job)
                if not placed:
                    self.refused += 1
                    logger.warning("refusing a call: %s", why, extra={"refused": self.refused})
                    await job.reject(terminate=False)
                    return
            await job.accept()
        finally:
            self.accepting -= 1


class Monitor:
    """Serve mode's thread: every ``interval`` it asks the admission's probe whether the engine serves and, with a
    ``path``, writes the worker's state there whole: its pid, mode and admission (the serving profile's settings it
    runs, ``describe``), the calls in progress and whether it takes new ones."""

    def __init__(
        self,
        admission: Admission,
        path: str | os.PathLike[str] | None,
        describe: Mapping[str, Any],
        interval: float = PROBE_INTERVAL_S,
    ) -> None:
        self.admission = admission
        self.path = Path(path) if path else None
        self.describe = dict(describe)
        self.interval = interval
        self._stop = threading.Event()

    def snapshot(self) -> dict[str, Any]:
        admitting, why = self.admission.gate()
        calls = self.admission.calls()
        return {
            "schema": SCHEMA,
            "pid": os.getpid(),
            "updated": time.time(),
            **self.describe,
            "active_calls": calls,
            "admitting": admitting and calls < self.admission.max_calls,
            "reason": why if not admitting else f"{calls} of {self.admission.max_calls} calls",
            "refused": self.admission.refused,
        }

    def write(self) -> None:
        if self.path is None:
            return
        staged = self.path.with_name(f".{self.path.name}.{os.getpid()}")
        try:
            staged.write_text(json.dumps(self.snapshot(), indent=1) + "\n")
            staged.replace(self.path)
        except OSError as error:
            with contextlib.suppress(OSError):
                staged.unlink()
            logger.warning("cannot write the worker's status to %s: %s", self.path, error)

    def tick(self) -> None:
        if self.admission.probe is not None:
            self.admission.probe.ask()
        self.write()

    def start(self) -> None:
        """A first round now, so that the worker registers knowing whether the engine serves, then one every
        interval."""
        self.tick()
        threading.Thread(target=self._run, name="voice-admission", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.tick()
