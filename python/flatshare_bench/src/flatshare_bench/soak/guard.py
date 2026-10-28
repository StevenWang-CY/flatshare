"""The soak's guard (:class:`Guard`): its load stops while another's process holds the GPU of a node it loads.

Each GPU node a soak loads is checked every minute by ``deploy/bin/node-report`` (:class:`NodeGuard`): the GPU's
compute processes, each the box's own (a unit's cgroup, or flatshare-svc's mark) or another's (``gpu-guard --list``),
and the user manager's journal lines about the node's flatshare units since the check before, which the soak keeps
with cyan's. By the node's method:

- ``ssh``, a GPU node of its own as cyan's are: over SSH with the guard key, ``~/.ssh/flatshare_guard_ed25519``, which
  ``deploy/authorize-guard.sh`` lets run node-report and nothing else, at the target the node's tunnel names
  (``FLATSHARE_TUNNEL_TARGET`` in ``~/.config/flatshare/tunnel-NODE.env``), or at ``ssh:TARGET``;
- ``local``: this host's own GPU (a single host), with the deployed node-report;
- ``off``: not checked, which the record says.

A check is ``clear``, ``taken`` (another's process holds the GPU: its pid, whose it is, its memory and name) or
``blind`` (the check did not run to its end, or nvidia-smi could not list the processes). A soak starts only with
every node clear. It pauses its load when a node is taken, or blind for ``blind_after`` checks in a row, as gpu-guard
refuses to start a unit when it cannot tell, and resumes once every node has been clear for ``clear_s``. The gateway
places the calls and the agents' sessions, so a soak cannot keep its load off one node: a pause stops all of it.

A process's owner is recorded as ``self`` (the soak's own user) or ``another``, never by name, and an SSH error names
the node in place of its target, so that a record says whose work it met and nothing of who or where.
"""

from __future__ import annotations

import getpass
import json
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flatshare_bench.e2.arm import E2Error
from flatshare_bench.e2.cyan import Host, read_env

METHODS = ("ssh", "local", "off")
CHECK_TIMEOUT_S = 30.0
"""The longest a check may take: an SSH connection (10 s at most to connect), nvidia-smi and the journal."""
OVERLAP_S = 5.0
"""How far back of the last check a check asks the node's journal from, against the hosts' clocks differing: a line
read twice is kept once (:class:`~flatshare_bench.soak.run.Soak`)."""
GUARD_KEY = Path(".ssh") / "flatshare_guard_ed25519"
"""The guard key under the home directory, which deploy/authorize-guard.sh makes and authorizes on each node."""
SSH_OPTIONS = (
    "-T",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "LogLevel=ERROR",
)  # fmt: skip
"""As the tunnel's (deploy/bin/flatshare-tunnel): no prompt, the guard key alone, a bounded connect."""
CLEAR_S = 300.0
"""How long every node must stay clear before a paused soak loads them again."""
BLIND_AFTER = 3
"""Checks in a row that cannot tell before the soak pauses, as gpu-guard refuses when it cannot tell."""


@dataclass(frozen=True)
class Check:
    """One check of a node's GPU."""

    node: str
    verdict: str
    """``clear``, ``taken`` or ``blind``."""
    at: float
    processes: tuple[Mapping[str, Any], ...] = ()
    """Each compute process: ``whose`` (ours or other), ``pid``, ``owner`` (self or another), ``memory_mib``, ``name``."""
    error: str | None = None
    journal: tuple[Mapping[str, Any], ...] = ()
    """The user manager's lines about the node's flatshare units since the check before: ``at``, ``unit``,
    ``message``, and a failure's ``own_lines``."""

    def others(self) -> list[Mapping[str, Any]]:
        return [p for p in self.processes if p.get("whose") == "other"]

    def record(self) -> dict[str, Any]:
        found: dict[str, Any] = {
            "node": self.node,
            "verdict": self.verdict,
            "at": round(self.at, 3),
        }
        if self.processes:
            found["processes"] = [dict(p) for p in self.processes]
        if self.error is not None:
            found["error"] = self.error
        return found

    def describe(self) -> str:
        if self.verdict == "taken":
            return ", ".join(
                f"pid {p['pid']} ({p['owner']}'s {p['name']}, {p['memory_mib']} MiB)"
                for p in self.others()
            )
        return self.error or self.verdict


def parse_list(text: str, me: str) -> tuple[dict[str, Any], ...]:
    """The processes of ``gpu-guard --list``'s output, a tab-separated line each, the owner as ``self`` when it is
    ``me`` and ``another`` otherwise; lines of any other shape are left out."""
    found = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 5 or fields[0] not in ("ours", "other"):
            continue
        whose, pid, owner, memory, name = fields
        found.append(
            {
                "whose": whose,
                "pid": int(pid) if pid.isdigit() else pid,
                "owner": "self" if owner == me else "another",
                "memory_mib": int(memory) if memory.isdigit() else memory,
                "name": name,
            }
        )
    return tuple(found)


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass
class NodeGuard:
    """How one node's GPU is checked: ``command`` runs node-report there (None: ``off``). ``hidden`` are strings
    a check's error must not carry (the SSH target), each written as the node's name."""

    node: str
    method: str
    command: Sequence[str] | None
    hidden: tuple[str, ...] = ()
    run: Runner = field(default=subprocess.run, repr=False)
    me: str = field(default_factory=getpass.getuser, repr=False)
    since: float | None = None
    """Where the next check asks the node's journal from; None asks for none."""

    def check(self, now: float | None = None) -> Check | None:
        """Check the node's GPU now, and read its units' journal from ``since``; None for a node whose guard is
        off."""
        if self.command is None:
            return None
        at = time.time() if now is None else now
        command = [*self.command, *([f"since={self.since:.3f}"] if self.since is not None else [])]
        try:
            done = self.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=CHECK_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Check(self.node, "blind", at, error=self._hide(f"{type(exc).__name__}: {exc}"))
        try:
            report = json.loads(done.stdout)
            guard = report["gpu_guard"]
            listed, said = str(guard["stdout"]), str(guard["stderr"])
        except (ValueError, KeyError, TypeError):
            # No report: SSH failed (255), or the node runs no node-report.
            said = (done.stderr.strip().splitlines() or [""])[-1]
            error = f"exit status {done.returncode}" + (f": {said}" if said else "")
            return Check(self.node, "blind", at, error=self._hide(error))
        journal = tuple(self._journal(report.get("journal")))
        if "journal" in report:
            self.since = at - OVERLAP_S
        processes = parse_list(listed, self.me)
        if done.returncode == 0:
            return Check(self.node, "clear", at, processes, journal=journal)
        if done.returncode == 1 and any(p["whose"] == "other" for p in processes):
            return Check(self.node, "taken", at, processes, journal=journal)
        last = (said.strip().splitlines() or [""])[-1]
        error = f"exit status {done.returncode}" + (f": {last}" if last else "")
        return Check(self.node, "blind", at, processes, self._hide(error), journal)

    def _journal(self, lines: Any) -> Iterable[dict[str, Any]]:
        """The report's journal lines, each of the shape node-report writes, with the hidden strings hidden."""
        for line in lines if isinstance(lines, list) else []:
            if isinstance(line, dict) and isinstance(line.get("message"), str) and "at" in line:
                kept = {
                    "at": line["at"],
                    "unit": str(line.get("unit")),
                    "message": self._hide(line["message"]),
                }
                if isinstance(line.get("own_lines"), list):
                    kept["own_lines"] = [self._hide(str(text)) for text in line["own_lines"]]
                yield kept

    def _hide(self, text: str) -> str:
        for secret in self.hidden:
            if secret:
                text = text.replace(secret, f"{self.node}'s target")
        return text

    def record(self) -> dict[str, Any]:
        """What the soak's record keeps of the guard: the node and its method, never its target."""
        return {"node": self.node, "method": self.method}


def parse_guards(values: Iterable[str]) -> dict[str, str]:
    """``--guard NODE=METHOD`` values by node: ``ssh``, ``ssh:TARGET``, ``local`` or ``off``."""
    found: dict[str, str] = {}
    for value in values:
        node, sep, method = value.partition("=")
        if not sep or not node or not (method in METHODS or method.startswith("ssh:")):
            raise E2Error(
                f"--guard {value!r} is not NODE=ssh, NODE=ssh:TARGET, NODE=local or NODE=off"
            )
        if node in found:
            raise E2Error(f"--guard names {node} twice")
        found[node] = method
    return found


def node_guard(node: str, method: str, host: Host, run: Runner = subprocess.run) -> NodeGuard:
    """The guard of ``node`` by ``method`` (:func:`parse_guards`), as this host reaches it."""
    if method == "off":
        return NodeGuard(node, "off", None, run=run)
    if method == "local":
        report = host.root / "src" / "deploy" / "bin" / "node-report"
        return NodeGuard(node, "local", ["python3", str(report)], run=run)
    target = method.removeprefix("ssh:") if method.startswith("ssh:") else None
    if target is None:
        env_file = host.config / f"tunnel-{node}.env"
        target = read_env(env_file).get("FLATSHARE_TUNNEL_TARGET") if env_file.exists() else None
        if not target:
            raise E2Error(
                f"the guard of {node} goes over SSH to its tunnel's target, and {env_file} names none "
                f"(FLATSHARE_TUNNEL_TARGET): pass --guard {node}=ssh:TARGET, or {node}=off"
            )
    key = host.home / GUARD_KEY
    if not key.exists():
        raise E2Error(
            f"no guard key {key}: deploy/authorize-guard.sh cyan {node}, from a workstation, makes it and lets it "
            f"run {node}'s node-report alone (or pass --guard {node}=off)"
        )
    host_part = target.rpartition("@")[2]
    hidden = tuple(sorted({target, host_part}, key=len, reverse=True))
    command = ["ssh", "-i", str(key), *SSH_OPTIONS, target]
    return NodeGuard(node, "ssh", command, hidden=hidden, run=run)


class Guard:
    """Whether the soak's load may run, over the nodes' checks: paused while a node is taken, or blind for
    ``blind_after`` checks in a row, until every node has been clear for ``clear_s``."""

    def __init__(
        self,
        guards: Mapping[str, NodeGuard],
        clear_s: float = CLEAR_S,
        blind_after: int = BLIND_AFTER,
    ) -> None:
        self.guards = dict(guards)
        self.clear_s = clear_s
        self.blind_after = blind_after
        self.tripped: dict[str, str] = {}
        """Each node that holds the load back, with why."""
        self._blind = dict.fromkeys(self.guards, 0)
        self._clear_since: dict[str, float | None] = dict.fromkeys(self.guards)

    @property
    def paused(self) -> bool:
        return bool(self.tripped)

    def why(self) -> str:
        return "; ".join(f"{node}: {why}" for node, why in sorted(self.tripped.items()))

    def check(self, now: float | None = None) -> list[Check]:
        """Check every node whose guard is on."""
        return [c for g in self.guards.values() if (c := g.check(now)) is not None]

    def observe(self, checks: Iterable[Check]) -> list[dict[str, Any]]:
        """Take the checks in, and return the events they make: a node ``guard_taken`` or ``guard_blind`` as it holds
        the load back, ``guard_failed`` for each first check in a row that could not tell, and ``guard_clear`` as a
        node that held it back has been clear for ``clear_s``."""
        events: list[dict[str, Any]] = []
        for check in checks:
            node = check.node
            if check.verdict == "taken":
                self._blind[node], self._clear_since[node] = 0, None
                if node not in self.tripped:
                    events.append({"event": "guard_taken"} | check.record())
                self.tripped[node] = f"its GPU is in use by {check.describe()}"
            elif check.verdict == "blind":
                self._blind[node] = self._blind.get(node, 0) + 1
                self._clear_since[node] = None
                if self._blind[node] == 1:
                    events.append({"event": "guard_failed"} | check.record())
                if self._blind[node] >= self.blind_after and node not in self.tripped:
                    events.append(
                        {"event": "guard_blind", "checks": self._blind[node]} | check.record()
                    )
                    self.tripped[node] = (
                        f"its GPU could not be read {self._blind[node]} times in a row"
                    )
            else:
                self._blind[node] = 0
                if node in self.tripped:
                    since = self._clear_since.get(node)
                    if since is None:
                        since = self._clear_since[node] = check.at
                    if check.at - since >= self.clear_s:
                        del self.tripped[node]
                        self._clear_since[node] = None
                        events.append(
                            {"event": "guard_clear", "clear_s": round(check.at - since, 1)}
                            | check.record()
                        )
        return events
