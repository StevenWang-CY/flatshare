"""Serve mode's call-capacity admission (flatshare_voice.admission): the worker's load and its limit, LiveKit's own
availability check over that load, the exact count of job requests that arrive together, the engine's state from the
gateway's /ready (and the fallback where it has none), a release's drain, and the status file that
deploy/bin/serving-check holds to the serving profile."""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from livekit.agents import AgentServer

from flatshare_voice.admission import Admission, EngineProbe, Monitor

REPO = Path(__file__).resolve().parents[3]
SERVING = REPO / "deploy" / "phastform" / "serving.json"


class Calls:
    """The calls in progress, as LiveKit's running jobs count them."""

    def __init__(self, n: int = 0) -> None:
        self.n = n

    def __call__(self) -> int:
        return self.n


class Serving:
    """A probe's state, set by the test."""

    def __init__(self, serving: bool = True, why: str = "/ready answered 200") -> None:
        self.serving, self.why = serving, why

    def state(self) -> tuple[bool, str]:
        return self.serving, self.why


class Job:
    """A job request: ``accept`` waits for ``assigned`` (the server's assignment), then the job runs."""

    def __init__(self, calls: Calls, assigned: asyncio.Event | None = None) -> None:
        self.calls, self.assigned = calls, assigned
        self.answer: str | None = None

    async def accept(self) -> None:
        self.answer = "accepted"
        if self.assigned is not None:
            await self.assigned.wait()
        self.calls.n += 1

    async def reject(self, *, terminate: bool = True) -> None:
        assert terminate is False  # another worker may take it
        self.answer = "refused"


def test_the_load_is_the_calls_over_max_calls_and_one_while_no_call_may_come(
    tmp_path: Path,
) -> None:
    calls, serving = Calls(), Serving()
    admission = Admission(2, calls, probe=serving, drain=tmp_path / "voice-worker.drain")
    assert [admission.load() for calls.n in (0, 1, 2, 3)] == [0.0, 0.5, 1.0, 1.0]
    calls.n = 0
    serving.serving, serving.why = False, "/ready answered 503: warming"
    assert admission.load() == 1.0
    assert admission.gate() == (False, "the engine does not serve: /ready answered 503: warming")
    serving.serving = True
    (tmp_path / "voice-worker.drain").touch()
    assert admission.load() == 1.0 and admission.gate()[1].startswith("draining:")
    with pytest.raises(ValueError, match="max_calls must be 1 or more"):
        Admission(0, calls)


def test_requests_that_arrive_together_are_admitted_exactly_to_max_calls() -> None:
    # Three requests at once for a worker that admits two: LiveKit's own check would count the ones not yet launched
    # at a guessed share of the load; the admission counts each accepted one at once.
    async def go() -> list[str | None]:
        calls, assigned = Calls(), asyncio.Event()
        admission = Admission(2, calls)
        jobs = [Job(calls, assigned) for _ in range(3)]
        tasks = [asyncio.create_task(admission.request(job)) for job in jobs]
        await asyncio.sleep(0.01)
        assert (admission.accepting, admission.calls(), admission.load()) == (2, 2, 1.0)
        assigned.set()
        await asyncio.gather(*tasks)
        assert (admission.accepting, calls.n, admission.refused) == (0, 2, 1)
        # A call ends: the next request is taken.
        calls.n -= 1
        late = Job(calls)
        await admission.request(late)
        return [job.answer for job in [*jobs, late]]

    assert asyncio.run(go()) == ["accepted", "accepted", "refused", "accepted"]


def test_an_accept_that_fails_gives_its_place_back() -> None:
    class Lost(Job):
        async def accept(self) -> None:
            raise TimeoutError("assignment timed out")

    async def go() -> int:
        admission = Admission(1, Calls())
        with pytest.raises(TimeoutError):
            await admission.request(Lost(Calls()))
        return admission.accepting

    assert asyncio.run(go()) == 0


def test_no_request_is_taken_while_the_engine_does_not_serve_or_a_drain_holds(
    tmp_path: Path,
) -> None:
    async def answer(admission: Admission) -> str | None:
        job = Job(Calls())
        await admission.request(job)
        return job.answer

    serving, drain = Serving(False, "/ready answered 503"), tmp_path / "drain"
    admission = Admission(2, Calls(), probe=serving, drain=drain)
    assert asyncio.run(answer(admission)) == "refused"
    serving.serving = True
    assert asyncio.run(answer(admission)) == "accepted"
    drain.touch()
    assert asyncio.run(answer(admission)) == "refused"
    drain.unlink()
    assert asyncio.run(answer(admission)) == "accepted"


def test_livekits_availability_check_marks_the_worker_full_at_max_calls() -> None:
    # LiveKit 1.8.8's own decision (AgentServer._is_available, which answers each job request, and the status it
    # sends the server), with the worker's load function and serve mode's threshold of 1: available below max_calls,
    # full at it. The server is not run: its process pool is a stand-in holding the running jobs.
    calls = Calls()
    admission = Admission(2, calls)
    server = AgentServer(load_fnc=lambda worker: admission.load(), load_threshold=1.0)
    server._devmode, server._reserved_slots, server._worker_load = False, 0, 0.0

    def available(n: int) -> bool:
        calls.n = n
        server._proc_pool = SimpleNamespace(
            processes=[SimpleNamespace(running_job=object()) for _ in range(n)]
        )
        asyncio.run(server._refresh_worker_load())
        return server._is_available()

    assert [available(n) for n in (0, 1, 2, 3)] == [True, True, False, False]
    # Bench mode's load, always 0, never fills the worker, whatever LiveKit's threshold.
    server = AgentServer(load_fnc=lambda: 0.0)
    server._devmode, server._reserved_slots, server._worker_load = False, 0, 0.0
    server._proc_pool = SimpleNamespace(processes=[SimpleNamespace(running_job=object())] * 6)
    asyncio.run(server._refresh_worker_load())
    assert server._is_available()


class _Answers(http.server.BaseHTTPRequestHandler):
    answers: ClassVar[dict[str, tuple[int, str]]] = {}

    def do_GET(self) -> None:
        status, body = self.answers.get(self.path, (404, "not found\n"))
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def address_string(self) -> str:
        return "unix"

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def gateway() -> Iterator[tuple[str, dict[str, tuple[int, str]]]]:
    """An HTTP server on a Unix socket in a short directory, answering each path as the dict it yields says."""
    with tempfile.TemporaryDirectory(prefix="flatshare-voice-") as directory:
        path = str(Path(directory) / "gateway.sock")
        answers: dict[str, tuple[int, str]] = {}
        handler = type("Handler", (_Answers,), {"answers": answers})
        server = socketserver.ThreadingUnixStreamServer(path, handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield path, answers
        finally:
            server.shutdown()
            server.server_close()


def test_the_probe_reads_the_gateways_ready_and_falls_back_without_it(
    gateway: tuple[str, dict], caplog: pytest.LogCaptureFixture
) -> None:
    path, answers = gateway
    probe = EngineProbe(path, "/ready")
    assert probe.state() == (False, "not asked yet")
    answers["/ready"] = (200, "ready\n")
    assert probe.ask() == (True, "/ready answered 200")
    # The gateway says why the engine does not serve: today's 503, and the engine's states.
    answers["/ready"] = (503, "upstream not ready\n")
    assert probe.ask() == (False, "/ready answered 503: upstream not ready")
    answers["/ready"] = (503, "warming\nthe engine starts\n")
    assert probe.ask() == (False, "/ready answered 503: warming")
    # A server without the route says nothing of the engine: capacity alone admits, said once.
    del answers["/ready"]
    with caplog.at_level(logging.WARNING, logger="flatshare_voice.admission"):
        assert probe.ask() == (True, "/ready is not served (404): capacity alone admits")
        assert probe.ask()[0]
    said = [r.getMessage() for r in caplog.records if "capacity alone" in r.getMessage()]
    assert said == ["the LLM's socket serves no /ready (404): admitting calls by capacity alone"]
    # Nobody on the socket: no call.
    serving, why = EngineProbe(path + ".none", "/ready").ask()
    assert not serving and why.startswith(f"/ready on {path}.none did not answer:")


def test_the_status_file_says_what_the_worker_runs_and_passes_the_profiles_check(
    gateway: tuple[str, dict], tmp_path: Path
) -> None:
    path, answers = gateway
    answers["/ready"] = (200, "ready\n")
    calls = Calls(1)
    admission = Admission(2, calls, probe=EngineProbe(path, "/ready"), drain=tmp_path / "drain")
    status = tmp_path / "voice-worker.json"
    describe = {"mode": "serve", "max_calls": 2, "audio": "microphone", "turn_detection": "model"}
    monitor = Monitor(admission, status, describe)
    monitor.tick()
    written = json.loads(status.read_text())
    assert written == written | describe | {"schema": 1, "pid": os.getpid()}
    assert (written["active_calls"], written["admitting"], written["reason"]) == (
        1,
        True,
        "1 of 2 calls",
    )
    check = [sys.executable, str(REPO / "deploy" / "bin" / "serving-check"), str(SERVING)]
    done = subprocess.run(
        [*check, "--voice", str(status), "--voice-pid", str(os.getpid())],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    # Full, then the engine down: the file says so, and why.
    calls.n = 2
    monitor.tick()
    assert json.loads(status.read_text())["admitting"] is False
    calls.n = 0
    answers["/ready"] = (503, "drained\n")
    monitor.tick()
    written = json.loads(status.read_text())
    assert (written["admitting"], written["reason"]) == (
        False,
        "the engine does not serve: /ready answered 503: drained",
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "voice-worker.json"
    ]  # no staged file left


def test_a_call_no_node_takes_is_refused_before_its_job_is_accepted() -> None:
    # With nodes, each job's call is placed before it is accepted: a placement the gateway refuses (every realtime
    # node at its max_calls, say) refuses the job, and a call being placed counts against max_calls meanwhile.
    async def go() -> tuple[list[str | None], int]:
        calls, gate = Calls(), asyncio.Event()
        answers = iter([(True, "placed (200)"), (False, "no node takes the call: no_call_slot")])

        async def place(job: Job) -> tuple[bool, str]:
            await gate.wait()
            return next(answers)

        admission = Admission(1, calls, place=place)
        first, second, third = Job(calls), Job(calls), Job(calls)
        pending = asyncio.create_task(admission.request(first))
        await asyncio.sleep(0.01)
        # The first call is being placed: it holds the one slot, so a request now is refused at once.
        await admission.request(second)
        gate.set()
        await pending
        calls.n -= 1  # the first call ends
        await admission.request(third)  # its placement is refused
        return [job.answer for job in (first, second, third)], admission.refused

    assert asyncio.run(go()) == (["accepted", "refused", "refused"], 2)


def test_the_placement_at_admission_reads_the_gateways_answers() -> None:
    # nodes.admit over a gateway socket: 200 places the call, 404 (no nodes) admits it by capacity, a 503 naming no
    # node for it refuses it, and a gateway that does not answer refuses it too.
    from aiohttp import web

    from flatshare_voice import nodes

    answers: list[tuple[int, dict]] = []

    async def sessions(request: web.Request) -> web.Response:
        await request.json()
        status, body = answers.pop(0)
        if status == 0:
            await asyncio.sleep(5)
        return web.json_response(body, status=status or 200)

    async def go() -> list[tuple[bool, str]]:
        app = web.Application()
        app.router.add_post("/v1/sessions", sessions)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
        await runner.setup()
        # a short directory: socket paths are limited to 104 bytes on macOS
        directory = tempfile.TemporaryDirectory(prefix="fs-adm-")
        sock = Path(directory.name) / "gateway.sock"
        await web.UnixSite(runner, str(sock)).start()
        found = []
        try:
            for answer in (
                (200, {"session_id": "voice-RM_a", "node": "gray"}),
                (404, {"error": {"type": "not_found"}}),
                (503, {"error": {"type": "no_call_slot"}}),
                (503, {"error": {"type": "placement_full"}}),
                (503, {"error": {"type": "in_flight_limit"}}),
                (0, {}),
            ):
                answers.append(answer)
                found.append(await nodes.admit(str(sock), "voice-RM_a", timeout=0.3))
        finally:
            await runner.cleanup()
            directory.cleanup()
        return found

    results = asyncio.run(go())
    assert [ok for ok, _ in results] == [True, True, False, False, False, False]
    assert "no_call_slot" in results[2][1] and "placement_full" in results[3][1]
    assert "did not place the call" in results[5][1]
