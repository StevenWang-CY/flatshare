"""The tools a point runs, AIPerf and the callers, and how a run waits on them and on the services
(:mod:`.run`).

Each tool runs in a process group of its own, marked in its environment and named in the run's scope
until it is stopped (:class:`Process`), so that a run's start ends whatever a run killed before it
could stop its tools left (:func:`kill_left`). Every wait of a run looks again every POLL_S
(:func:`until`, :func:`pause`).
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from flatshare_bench.e2.arm import E2Error

if TYPE_CHECKING:
    from flatshare_bench.e2.cyan import Host

POLL_S = 1.0
"""How often a run checks on the tools and services it waits for."""


def until(done: Callable[[], bool], timeout: float, message: str) -> None:
    """Return once ``done()``, asked every POLL_S; past ``timeout`` seconds, raise an E2Error that
    says ``message``."""
    deadline = time.monotonic() + timeout
    while not done():
        if time.monotonic() > deadline:
            raise E2Error(message)
        time.sleep(POLL_S)


def pause(at_most: float = math.inf) -> None:
    """Wait before a run looks again at what it waits for: POLL_S, or ``at_most`` seconds if that is
    less."""
    time.sleep(min(POLL_S, at_most))


TOOL_MARK = "FLATSHARE_E2_TOOL"
"""The environment variable that marks every process a tool of e2 run started, its children too:
``SCOPE/PID-NS``, the run's scope (:func:`tool_scope`), then its pid and the tool's start."""
TOOLS = ".flatshare-bench-e2-tools"
"""The directory in the run's scope (the root whose ``runs/`` holds the run lock) that names each running tool's
process group by its leader's pid, the leader's start and cgroup in the file: AIPerf's services set their titles
(setproctitle), which blanks /proc/PID/environ and the mark in it, but they stay in AIPerf's process group, so a run's
start ends the groups a run that was killed before it could stop them left (:func:`kill_left`)."""
NEEDS_WAITID = (
    "flatshare-bench e2 run needs os.waitid, to watch its tools without reaping them: Linux, or macOS "
    "from Python 3.13"
)


class Process:
    """A tool in its own process group, so that stopping it reaches every process it started:
    AIPerf renames its processes, so they cannot be found by name, and its group is named in the
    run's scope (``TOOLS``) until it is stopped. Each is also marked in its environment, which its
    children inherit: AIPerf starts some services in sessions of their own, out of the group's reach.
    Output goes to ``log``."""

    def __init__(
        self, name: str, command: Sequence[str], log: Path, env: Mapping[str, str], scope: str
    ) -> None:
        self.name = name
        self.log = log
        self.scope = scope
        self.mark = f"{scope}/{os.getpid()}-{time.monotonic_ns()}"
        # Made before the tool starts: a scope where its group cannot be named starts none.
        tools = Path(scope) / TOOLS
        tools.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as out:
            self.popen = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=dict(env) | {TOOL_MARK: self.mark},
                start_new_session=True,
            )
        self.status: int | None = None
        pid = self.popen.pid
        self.record = tools / str(pid)
        try:
            self.record.write_text(f"{_started(pid) or ''}\n{_cgroup(pid) or ''}")
        except BaseException:
            # Nothing would stop a tool that neither a Process nor a record names.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)
            self.popen.wait()
            raise

    def exited(self) -> bool:
        """Whether the tool has exited, without reaping it: its group id stays taken until stop()."""
        if self.status is not None:
            return True
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        return os.waitid(os.P_PID, self.popen.pid, flags) is not None

    def stop(self, grace: float = 0.0) -> int:
        """Interrupt the tool if it still runs, give it ``grace`` seconds to finish, kill what is left
        of its process group and return its exit status."""
        if self.status is None:
            pid = self.popen.pid
            if not self.exited():
                os.kill(pid, signal.SIGINT)
                deadline = time.monotonic() + grace
                while not self.exited() and time.monotonic() < deadline:
                    time.sleep(POLL_S / 5)
            # A group with nothing left but its exited leader answers ESRCH (Linux) or EPERM (macOS).
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)
            kill_marked(self.scope, self.mark)
            self.status = self.popen.wait()
            self.record.unlink(missing_ok=True)
        return self.status

    def failure(self, what: str) -> E2Error:
        """An error naming the tool's exit status and the end of its log."""
        tail = "\n".join(self.log.read_text(errors="replace").splitlines()[-15:])
        return E2Error(f"{self.name} {what} (exit status {self.stop()}); {self.log}:\n{tail}")


def tool_scope(host: Host) -> str:
    """The scope of a run's tools: the root its run lock is in (``host.root``). A run's start kills
    what a run under the same lock left, and no tool of a run beside it under another root, as a
    test session's runs are beside another's on one host."""
    return str(host.root)


def kill_marked(scope: str, mark: str | None = None) -> list[int]:
    """Kill the processes whose environment carries a mark of ``scope`` under ``TOOL_MARK`` (``mark``
    alone when given), and return their pids. It reads ``/proc``, so it finds none off Linux."""
    killed = []
    for environ in Path("/proc").glob("[0-9]*/environ"):
        try:
            variables = environ.read_bytes().split(b"\0")
        except OSError:  # another user's process, or one that just exited
            continue
        marks = [v.split(b"=", 1)[1] for v in variables if v.startswith(f"{TOOL_MARK}=".encode())]
        ours = marks and marks[0].rsplit(b"/", 1)[0] == scope.encode()
        if ours and (mark is None or marks[0] == mark.encode()):
            pid = int(environ.parent.name)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
    return killed


def _boot() -> str | None:
    """The id the kernel draws for this boot, which no other boot shares; None off Linux."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def _started(pid: int) -> str | None:
    """Process ``pid``'s start: the boot's id and its start in clock ticks since boot (/proc/PID/stat's
    22nd field), which a pid used again does not share; None off Linux, or once it is gone."""
    try:
        # Bytes: the kernel keeps 15 bytes of a name, which may end inside a character.
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return None
    boot = _boot()
    return f"{boot}/{int(stat.rsplit(b') ', 1)[1].split()[19])}" if boot else None


def _cgroup(pid: int) -> str | None:
    """Process ``pid``'s cgroups (/proc/PID/cgroup), which its children inherit; None off Linux, or
    once it is gone."""
    try:
        return Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return None


def _members(pgid: int) -> list[int]:
    """The processes of process group ``pgid`` (/proc/PID/stat's 5th field)."""
    members = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        with contextlib.suppress(OSError, IndexError, ValueError):
            if int(stat.read_bytes().rsplit(b") ", 1)[1].split()[2]) == pgid:
                members.append(int(stat.parent.name))
    return members


def _titled(pid: int, cgroup: str) -> bool:
    """Whether process ``pid`` reads as a titled process of a tool that started in ``cgroup``: its
    environment blank, as a title (setproctitle) leaves it, and in that cgroup. AIPerf's services
    read blank, but so does vLLM's engine core, which on a single host runs as flatshare-bench's user, in
    its unit's cgroup under systemd."""
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:  # another user's process, one that is not dumpable, or a zombie
        return False
    return not environ.strip(b"\0") and _cgroup(pid) == cgroup


def kill_left(scope: str) -> list[int]:
    """Kill what tools of ``scope`` left running, a run's that was killed before it could stop them, and
    return the pids: each process group named in ``TOOLS`` in this boot whose leader is the one named
    there, the titled members (:func:`_titled`) of one whose leader is gone, then each process marked
    in ``scope`` (:func:`kill_marked`). It reads ``/proc``, so it finds none off Linux."""
    killed = []
    tools, boot = Path(scope) / TOOLS, _boot()
    for record in sorted(tools.glob("[0-9]*")) if tools.is_dir() else []:
        if not re.fullmatch("[1-9][0-9]*", record.name):
            continue  # not a record: Process names each by its tool's pid
        pgid = int(record.name)
        started, _, cgroup = record.read_text().partition("\n")
        # A record of another boot names a number that means nothing in this one, and an empty one
        # (off Linux, or a write cut short) names nothing.
        if boot and started.startswith(f"{boot}/"):
            leader = _started(pgid)
            if leader == started:
                members = _members(pgid)
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pgid, signal.SIGKILL)
                    killed += members
            elif leader is None:
                # Gone, its members keeping the number taken; or the tool's group ended, and the
                # number was given again, to a group whose own leader has exited since. A tool's
                # processes carry its mark, which kill_marked finds below, or read blank, titled, in
                # the cgroup it started in: any other member is another's.
                for pid in _members(pgid):
                    if _titled(pid, cgroup):
                        with contextlib.suppress(ProcessLookupError, PermissionError):
                            os.kill(pid, signal.SIGKILL)
                            killed.append(pid)
        record.unlink(missing_ok=True)
    return killed + [pid for pid in kill_marked(scope) if pid not in killed]
