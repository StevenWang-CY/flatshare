"""A call's GPU node, when the gateway fronts several of them.

The gateway places each session on one GPU node and keeps it there: the session's KV is in that node's engine, and the
call's STT and TTS belong on the same node. With node binding on (serve mode's default, ``FLATSHARE_NODE_BINDING=on``
in bench mode), a call starts by asking the gateway for its node, ``POST /v1/sessions`` on the gateway's socket with
the call's ``X-Session-ID`` (the contract is in the package's README, "Nodes"), and binds the call's STT and TTS to
that node's sockets, which the answer names and which sit beside the configured ones (the tunnels' ``NODE-stt.sock``
and ``NODE-tts.sock`` on cyan), and its LLM to the gateway with the session id, which routes it to the session's
node. A gateway without nodes answers 404: the call keeps the configured sockets and is served as it always was. The
call's end, whatever ends it, ends its placement (:func:`end`, ``DELETE /v1/sessions/ID``), so that its call slot is
free for the next call at once, events or none.

A request of the call that fails because its node did (the node's STT or TTS socket breaks, refuses or times out; the
gateway answers the call's LLM request, or ends its stream, with an upstream failure) fences the call
(:meth:`CallBinding.node_failed`): the reply under way ends where the caller's audio stands (the session's speech is
interrupted), and no request of the call reaches any node until the caller speaks again, so that a reply cut by a node
failure is never continued, retried or followed by a new generation on top of the audio already played. At the
caller's next speech (:meth:`CallBinding.turn_started`) the call asks for its node again, which the gateway places
anew, and its STT, TTS and LLM go there from that turn on. The conversation's transcript is the session's chat context,
which the worker keeps: the next reply runs on the new node with all of it, the KV being only a cache.

The plugins keep their clients for the whole call; only where their requests go changes, through the transports here
(:class:`SpeechTransport`, :class:`GatewayTransport`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2 as httpx

from .protocol import CLASS_HEADER, REALTIME_CLASS, SESSION_ID_HEADER

SESSIONS_URL = "http://flatshare.invalid/v1/sessions"
"""The gateway's placement route; requests go over its Unix socket, so the host never resolves."""
PLACEMENT_TIMEOUT_S = 1.0
"""The longest a placement may take: the gateway answers from its own table, in a millisecond."""
NODE_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
"""A GPU node's name, as the gateway and the tunnels name it."""
SOCKET_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,122}\.sock")
"""A speech socket's file name. A placement names the socket as the gateway's `[[nodes]]` configure it
(``unix:/run/user/UID/flatshare/gray-stt.sock``), or by its file name alone; either way the worker takes it only beside
its own configured sockets, in the same directory (the tunnels' on cyan)."""
NO_NODES = frozenset({404, 405})
"""A gateway that serves no placement (a single upstream): the call keeps its configured sockets."""
NODE_FAILURES = frozenset({"upstream_unavailable", "upstream_timeout", "no_healthy_node"})
"""The gateway's errors that say the session's node failed: an upstream that broke or stalled, or no node of the
session's role available. An answer names its error in its ``code`` (its ``type`` is the status's OpenAI class), and
a stream's error event in its ``type`` (crates/flatshare-gateway/tests/fixtures/errors.json has both). Its other
errors (a client's mistake, the gateway's own capacity, a drain) say nothing of the node."""
PLACEMENT_REFUSALS = frozenset({"no_call_slot", "no_healthy_node", "placement_full"})
"""The ``code`` of the gateway's 503 answers to a placement that say no node takes the call now: every available
realtime node at its ``max_calls``, no available realtime node, or a full placement table. Serve mode's admission
refuses the call."""
STREAM_FAILURES = tuple(f'"type":"{kind}"'.encode() for kind in sorted(NODE_FAILURES))
"""How those errors appear in an event stream's error event, which the gateway sends before it breaks a stream off
(``data: {"error": {..., "type": "upstream_unavailable", ...}}``, compact JSON)."""

logger = logging.getLogger(__name__)


def beside(named: str, configured: str) -> str | None:
    """The path of the speech socket a placement names, beside the ``configured`` one: ``named`` is its file name, or
    its ``unix:`` address or absolute path in the same directory. None for anything else, which places the call
    nowhere."""
    path = named.removeprefix("unix:")
    if "/" in path:
        if Path(path).parent != Path(configured).parent:
            return None
        path = Path(path).name
    if not SOCKET_NAME.fullmatch(path):
        return None
    return str(Path(configured).with_name(path))


async def admit(
    gateway: str, session_id: str, timeout: float = PLACEMENT_TIMEOUT_S
) -> tuple[bool, str]:
    """Whether a node takes the call of ``session_id`` before its job is accepted (serve mode's admission): the
    gateway places the session (``POST /v1/sessions``, reason ``start``), and the call's own placement at its start
    finds the same node. A gateway without nodes (404, 405) admits it by capacity alone; a 503 that names no node for
    it (PLACEMENT_REFUSALS) refuses it, as does a gateway that does not answer, which can place nothing."""
    body = {"session_id": session_id, "class": REALTIME_CLASS, "reason": "start"}
    headers = {SESSION_ID_HEADER: session_id, CLASS_HEADER: REALTIME_CLASS}
    try:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=gateway), timeout=timeout
        ) as client:
            response = await client.post(SESSIONS_URL, json=body, headers=headers)
    except httpx.HTTPError as exc:
        return False, f"the gateway did not place the call: {type(exc).__name__}: {exc}"
    if response.status_code in NO_NODES or response.status_code == 200:
        return True, f"placed ({response.status_code})"
    try:
        code = response.json()["error"]["code"]
    except (ValueError, KeyError, TypeError):
        code = None
    if response.status_code == 503 and code in PLACEMENT_REFUSALS:
        return False, f"no node takes the call: {code}"
    return (
        False,
        f"the gateway placed the call nowhere: HTTP {response.status_code} {code or ''}".rstrip(),
    )


async def end(gateway: str, session_id: str, timeout: float = PLACEMENT_TIMEOUT_S) -> None:
    """End the placement of ``session_id``'s call (``DELETE /v1/sessions/ID``), which frees its call slot at once:
    at the call's end, whatever ends it and whether or not its lifecycle events are on, and for a job placed but not
    run (serve mode's admission). A session without a placement ends as well, and a gateway without nodes (404, 405)
    holds none. A failure is logged, never raised: the placement then idles out, after the gateway's ``idle_s``."""
    context = {"session_id": session_id}
    try:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=gateway), timeout=timeout
        ) as client:
            response = await client.delete(f"{SESSIONS_URL}/{session_id}")
    except httpx.HTTPError as exc:
        cause = f"{type(exc).__name__}: {exc}"  # a timeout's own message can be empty
        logger.warning(
            "the call's placement did not end: the gateway did not answer (%s)",
            cause,
            extra=context,
        )
        return
    if response.status_code != 200 and response.status_code not in NO_NODES:
        logger.warning(
            "the call's placement did not end: HTTP %d: %s", response.status_code, response.text[:200],
            extra=context,
        )  # fmt: skip


class NodeFailed(httpx.TransportError):
    """A request of a call whose node failed, or which no node takes: refused before it leaves the worker."""


@dataclass(frozen=True)
class Placement:
    """Where a call's speech goes: its node, and the paths of that node's STT and TTS sockets here."""

    node: str
    stt: str
    tts: str


class CallBinding:
    """One call's node: the gateway's placement of the session ``session_id``, asked on the gateway's socket
    ``gateway``, and the call's configured STT and TTS sockets (``stt``, ``tts``), beside which a node's are.

    ``nodes`` is None until the gateway has answered, False for a gateway without nodes (the configured sockets for the
    whole call), and True once it placed the call. ``failed`` names the node that failed while the call is fenced.
    ``on_failure`` is called once per failure, to cut the reply under way."""

    def __init__(
        self,
        session_id: str,
        gateway: str,
        stt: str,
        tts: str,
        *,
        timeout: float = PLACEMENT_TIMEOUT_S,
        on_failure: Callable[[], None] | None = None,
    ) -> None:
        self.session_id = session_id
        self.gateway = gateway
        self.configured = {"stt": stt, "tts": tts}
        self.timeout = timeout
        self.on_failure = on_failure
        self.nodes: bool | None = None
        self.placement: Placement | None = None
        self.failed: str | None = None
        self.history: list[dict[str, Any]] = []
        """Each placement and failure, in order: what the call ran on, for its log and the tests."""
        self._placing: asyncio.Future[Placement | None] | None = None
        self._client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=gateway), timeout=timeout
        )

    @property
    def fenced(self) -> bool:
        """Whether the call's node failed and the caller has not spoken since: no request of the call may leave."""
        return self.failed is not None

    async def place(self, reason: str = "start") -> Placement | None:
        """Ask the gateway for the call's node (``reason``: ``start``, ``node_failed`` or ``unplaced``); one request
        at a time, which concurrent callers share. Returns the placement, or None: none (no nodes, or the gateway
        placed it nowhere for now)."""
        if self._placing is None or self._placing.done():
            self._placing = asyncio.ensure_future(self._place(reason))
        return await asyncio.shield(self._placing)

    async def _place(self, reason: str) -> Placement | None:
        body: dict[str, Any] = {
            "session_id": self.session_id,
            "class": REALTIME_CLASS,
            "reason": reason,
        }
        if self.failed is not None and self.placement is not None:
            body["failed_node"] = self.placement.node  # the node the call ran on, which failed
        headers = {SESSION_ID_HEADER: self.session_id, CLASS_HEADER: REALTIME_CLASS}
        context = {"session_id": self.session_id, "reason": reason}
        try:
            response = await self._client.post(SESSIONS_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            cause = f"{type(exc).__name__}: {exc}"  # a timeout's own message can be empty
            logger.warning(
                "the call has no node: the gateway did not answer (%s)", cause, extra=context
            )
            return None
        if response.status_code in NO_NODES:
            self.nodes, self.failed, self.placement = False, None, None
            logger.info(
                "the gateway places no call (HTTP %d): its configured sockets", response.status_code,
                extra=context,
            )  # fmt: skip
            return None
        placement = self._answer(response)
        if placement is None:
            logger.warning(
                "the call has no node: HTTP %d: %s", response.status_code, response.text[:200],
                extra=context,
            )  # fmt: skip
            return None
        moved = self.placement is not None and self.placement.node != placement.node
        self.nodes, self.placement, self.failed = True, placement, None
        self.history.append({"placed": placement.node, "reason": reason, "moved": moved})
        logger.info(
            "call placed", extra=context | {"node": placement.node, "moved": moved}
        )  # fmt: skip
        return placement

    def _answer(self, response: httpx.Response) -> Placement | None:
        """The placement a 200 answer names, its sockets beside the configured ones; None for any other answer."""
        if response.status_code != 200:
            return None
        try:
            answer = response.json()
        except ValueError:
            return None
        if (
            not isinstance(answer, dict)
            or answer.get("session_id", self.session_id) != self.session_id
        ):
            return None
        node, stt, tts = (answer.get(key) for key in ("node", "stt", "tts"))
        if not all(isinstance(value, str) for value in (node, stt, tts)):
            return None
        stt, tts = beside(stt, self.configured["stt"]), beside(tts, self.configured["tts"])
        if not NODE_NAME.fullmatch(node) or stt is None or tts is None:
            return None
        return Placement(node, stt, tts)

    async def socket(self, server: str) -> str:
        """The socket a request to the call's ``server`` (``stt`` or ``tts``) goes to: its node's, the configured one
        on a gateway without nodes. A fenced call's request is refused (NodeFailed), and an unplaced one asks for its
        node first, refused if it still has none. A placement in flight is waited for."""
        if self.nodes is False:
            return self.configured[server]
        if self._placing is not None and not self._placing.done():
            await asyncio.shield(self._placing)
        if self.failed is not None:
            raise NodeFailed(
                f"the call's node, {self.failed}, failed: no request of the call reaches a node until the caller "
                "speaks again"
            )
        if self.placement is None:
            await self.place("unplaced")
            if self.nodes is False:
                return self.configured[server]
            if self.placement is None:
                raise NodeFailed("the gateway placed the call on no node")
        return self.placement.stt if server == "stt" else self.placement.tts

    def node_failed(self, origin: str, error: BaseException | str) -> None:
        """A request of the call (``origin``: stt, tts or llm) failed because its node did: fence the call and cut
        its reply (``on_failure``), once a failure. Nothing on a gateway without nodes, which never placed it."""
        if self.nodes is False or self.failed is not None:
            return
        self.failed = self.placement.node if self.placement is not None else "unplaced"
        cause = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        self.history.append({"failed": self.failed, "origin": origin, "error": cause})
        logger.warning(
            "the call's node failed: its reply ends, and the call asks for a node when the caller speaks again",
            extra={
                "session_id": self.session_id,
                "node": self.failed,
                "origin": origin,
                "error": cause,
            },
        )
        if self.on_failure is not None:
            self.on_failure()

    def turn_started(self) -> None:
        """The caller speaks: a fenced call, or one without a node, asks for its node again, in the background, so
        that the turn's transcription waits for the placement and goes where it says."""
        if self.nodes is False or (self.failed is None and self.placement is not None):
            return
        if self._placing is None or self._placing.done():
            self._placing = asyncio.ensure_future(
                self._place("node_failed" if self.failed is not None else "unplaced")
            )

    def transport(self, server: str, limits: httpx.Limits) -> httpx.AsyncBaseTransport:
        """The transport of the call's ``server`` client: ``stt`` and ``tts`` through the call's node, ``llm``
        through the gateway."""
        if server == "llm":
            return GatewayTransport(self, limits)
        return SpeechTransport(self, server, limits)

    async def aclose(self) -> None:
        if self._placing is not None and not self._placing.done():
            self._placing.cancel()
        await self._client.aclose()


class _Watched(httpx.AsyncByteStream):
    """A response's body, read as it comes: a transport error while it is read is the node's failure, and so is an
    event stream's error event that names one (``scan``)."""

    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        failed: Callable[[BaseException | str], None],
        scan: tuple[bytes, ...] = (),
    ) -> None:
        self._stream = stream
        self._failed = failed
        self._scan = scan
        self._tail = b""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        keep = max((len(pattern) for pattern in self._scan), default=1) - 1
        try:
            async for chunk in self._stream:
                if self._scan:
                    seen = self._tail + chunk
                    if any(pattern in seen for pattern in self._scan):
                        self._failed("the gateway's stream ended with an upstream failure")
                    self._tail = seen[-keep:] if keep else b""
                yield chunk
        except httpx.TransportError as exc:
            self._failed(exc)
            raise

    async def aclose(self) -> None:
        await self._stream.aclose()


def _forward(response: httpx.Response, stream: httpx.AsyncByteStream) -> httpx.Response:
    return httpx.Response(
        status_code=response.status_code,
        headers=response.headers,
        stream=stream,
        extensions=response.extensions,
    )


class SpeechTransport(httpx.AsyncBaseTransport):
    """A call's STT or TTS requests through the socket of the call's node (:meth:`CallBinding.socket`), a
    connection pool per socket; a transport error or a timeout there, before the answer or while its audio comes, is
    the node's failure."""

    def __init__(self, binding: CallBinding, server: str, limits: httpx.Limits) -> None:
        self.binding = binding
        self.server = server
        self.limits = limits
        self._pools: dict[str, httpx.AsyncHTTPTransport] = {}

    def _pool(self, path: str) -> httpx.AsyncHTTPTransport:
        if path not in self._pools:
            self._pools[path] = httpx.AsyncHTTPTransport(uds=path, limits=self.limits)
        return self._pools[path]

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = await self.binding.socket(self.server)
        try:
            response = await self._pool(path).handle_async_request(request)
        except httpx.TransportError as exc:
            self.binding.node_failed(self.server, exc)
            raise
        return _forward(
            response,
            _Watched(response.stream, lambda error: self.binding.node_failed(self.server, error)),
        )

    async def aclose(self) -> None:
        for pool in self._pools.values():
            await pool.aclose()


class GatewayTransport(httpx.AsyncBaseTransport):
    """A call's LLM requests through the gateway's socket, which routes them by the session's placement: refused
    while the call is fenced; an answer or a stream whose error says the session's node failed (NODE_FAILURES), or
    the gateway's socket breaking, fences it."""

    def __init__(self, binding: CallBinding, limits: httpx.Limits) -> None:
        self.binding = binding
        self._pool = httpx.AsyncHTTPTransport(uds=binding.gateway, limits=limits)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.binding.fenced:
            raise NodeFailed(
                f"the call's node, {self.binding.failed}, failed: no reply is generated until the caller speaks "
                "again"
            )
        try:
            response = await self._pool.handle_async_request(request)
        except httpx.TransportError as exc:
            self.binding.node_failed("llm", exc)
            raise
        if response.status_code >= 500:
            body = b"".join([chunk async for chunk in response.stream])
            with contextlib.suppress(httpx.HTTPError):
                await response.stream.aclose()
            if _names_failure(body):
                self.binding.node_failed("llm", f"HTTP {response.status_code}: {body[:200]!r}")
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=body,
                extensions=response.extensions,
            )
        return _forward(
            response,
            _Watched(
                response.stream,
                lambda error: self.binding.node_failed("llm", error),
                STREAM_FAILURES,
            ),
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


def _names_failure(body: bytes) -> bool:
    """Whether an error answer's body (``{"error": {"code": ...}}``) names a failure of the session's node."""
    try:
        error = json.loads(body).get("error", {})
    except (ValueError, AttributeError):
        return False
    return isinstance(error, dict) and error.get("code") in NODE_FAILURES


def watch(session: Any, binding: CallBinding) -> None:
    """Tie ``binding`` to the call's ``AgentSession``: a node failure interrupts the session's speech (the reply
    under way ends; nothing of it is generated or played again), and the caller's next speech asks for the call's
    node again; the session's close closes the binding's client."""
    loop = asyncio.get_running_loop()

    def interrupt() -> None:
        with contextlib.suppress(RuntimeError):  # the session is not running, or nothing speaks
            session.interrupt(force=True)

    binding.on_failure = lambda: loop.call_soon(interrupt)

    def user_state(event: Any) -> None:
        if getattr(event, "new_state", None) == "speaking":
            binding.turn_started()

    session.on("user_state_changed", user_state)
    session.on("close", lambda _event: asyncio.ensure_future(binding.aclose()))
