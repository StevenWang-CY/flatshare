"""cyan, the control host, as ``flatshare-bench e2 run`` (:mod:`.run`) uses it: its settings
(:class:`Host`: the defaults, or those its ``~/.config/flatshare/e2-host.toml`` sets), the GPU node a
point measures on (:class:`Node`: the host's own, Phastform on cyan, or another one through its
tunnel), its user units (:class:`Systemd`), and what a point changes in them (:func:`configure_cyan`):
the router, the gateway's configuration and the voice worker's drop-ins, each unit restarted only when
its configuration changes, and, on another node, the gateway's upstream and the worker's speech sockets
pointed at that node's tunnel, all of it undone by :func:`reset_cyan` (``flatshare-bench e2 reset``).
One run, reset or drill holds cyan at a time (:func:`exclusive`).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from flatshare_bench.e2.arm import Arm, E2Error, Voice
from flatshare_bench.e2.wait import until
from flatshare_bench.transport import UNIX, runtime_socket

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
"""The AIPerf configuration, from the source tree (``uv sync`` installs the package editable)."""
BOX_PROFILE = "rtx-5060-ti"
"""The frozen box's profile: Phastform's, and Gray's."""
HOST_FILE = Path(".config") / "flatshare" / "e2-host.toml"
"""Where a control host other than cyan sets :class:`Host`'s fields, under its home directory: a
single host its cores from its profile's core plan, and its profile. cyan has none."""

SMG = "flatshare-smg"
GATEWAY = "flatshare-gateway"
WORKER = "flatshare-voice-worker"
E2_GATEWAY = "e2-gateway"
NO_PREEMPTIVE = "no-preemptive-generation"
NO_EVENTS = "no-events"
"""Drop-ins under deploy/cyan/variants/ that e2 run links into cyan's units as ``variant-NAME.conf``."""
E2_NODE = "e2-node"
"""The voice worker's drop-in that e2 run writes, as ``variant-e2-node.conf``, for a point on a GPU node other than
the host's own: the node's speech sockets through its tunnel, and its engine's where the worker's LLM bypasses the
gateway (:func:`node_drop_in`). e2 reset removes it."""
DEFAULT_NODE = "phastform"
"""cyan's own GPU node: the one whose tunnel's sockets cyan's settings and units name (phastform-engine.sock and so
on)."""
NODE_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
"""A GPU node's name (its host's first label, as campaign.sh gives it), which unit, file and socket names carry."""
SINGLE_HOST = "localhost"
"""A single host's own node, as campaign.sh --local names it (deploy/a100/host.py's e2-host.toml)."""
E2_GATEWAY_TABLE: Mapping[str, Any] = {"canary": {"enabled": False}}
"""What every point's gateway configuration sets over gateway.toml, before the arm's ``[gateway]`` table: the
gateway's inference canary off. An idle gateway's canary sends the engine one-token completions, which would add to
the engine's counters a point reads and to what it serves beside the point's requests, and a gateway that has just
restarted with it answers /ready only once one has succeeded; the recorded campaigns ran without it. An arm's own
``[gateway.canary]`` table goes over this one."""


@dataclass(frozen=True)
class Host:
    """cyan (ops/RUNBOOK.md): its servers' addresses (``http://host:port``, or ``unix:`` and a socket's
    path), its paths and how to run the tools. The tests swap the tools for fakes."""

    home: Path = field(default_factory=Path.home)
    gateway_url: str = "http://127.0.0.1:18080"
    """The gateway's TCP listener, AIPerf's way in, with the agents tenant's key."""
    gateway_socket: str = field(default_factory=lambda: runtime_socket("gateway.sock"))
    """The gateway's socket, private to this user, from which a point reads its metrics with no key."""
    engine_url: str = field(default_factory=lambda: runtime_socket("phastform-engine.sock"))
    """The engine through the tunnel: its socket on cyan."""
    aiperf_cores: str = "16-23"
    caller_cores: str = "24-31"
    profile: str = BOX_PROFILE
    """The GPU node's hardware profile (deploy/profiles/NAME.toml), whose engine settings the engine
    check expects."""
    preflight: str = ""
    """A single host's preflight.json (deploy/a100/host.py preflight), whose degraded and refused checks
    each point's run.json keeps; cyan has none."""
    node: str = DEFAULT_NODE
    """The GPU node this host's settings reach, its own: on cyan Phastform, through its tunnel, whose sockets the
    defaults and cyan's units name; a single host names itself (``localhost``, as campaign.sh --local does). A point
    on another node reaches that node's tunnel instead (:meth:`node_for`)."""
    pinned: tuple[str, ...] = ()
    """The fields the host's settings file sets, in its order (:meth:`load`); none of the defaults'."""

    @classmethod
    def load(cls, home: Path | None = None) -> Host:
        """The host's settings: the defaults, and the fields its HOST_FILE sets, if it has one."""
        home = home or Path.home()
        path = home / HOST_FILE
        if not path.exists():
            return cls(home=home)
        try:
            table = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as exc:
            raise E2Error(f"{path}: {exc}") from None
        names = [f.name for f in fields(cls) if f.name not in ("home", "pinned")]
        if unknown := sorted(set(table) - set(names)):
            raise E2Error(f"{path} sets {', '.join(unknown)}; a host sets {', '.join(names)}")
        if wrong := sorted(key for key, value in table.items() if not isinstance(value, str)):
            raise E2Error(f"{path}: {', '.join(wrong)} must be strings")
        if "node" in table and not NODE_NAME.fullmatch(table["node"]):
            raise E2Error(f"{path}: node {table['node']!r} is not a GPU node's name")
        # A file that names the engine and no node is a single host's from before it named itself (deploy/a100/host.py
        # config writes node = "localhost" now): its own node is the one campaign.sh --local names, localhost.
        node = {"node": SINGLE_HOST} if "engine_url" in table and "node" not in table else {}
        return cls(home=home, pinned=tuple(table), **table, **node)

    def node_for(self, name: str | None = None) -> Node:
        """The GPU node ``name`` as a point reaches it from here: the host's own (``name`` None or its ``node``),
        through the engine its settings name, or another node through that node's tunnel on this host, its
        ``NAME-engine.sock`` (flatshare-tunnel@NAME's). Only a host that reaches its nodes by tunnel has another: one
        whose settings name the engine themselves (a single host) measures its own node alone."""
        name = name or self.node
        if not NODE_NAME.fullmatch(name):
            raise E2Error(f"{name!r} is not a GPU node's name")
        if name == self.node:
            return Node(name, self.engine_url, own=True)
        if "engine_url" in self.pinned:
            raise E2Error(
                f"{self.home / HOST_FILE} names this host's engine, {self.engine_url}, {self.node}'s: a point "
                f"here measures {self.node}, not {name}"
            )
        return Node(name, runtime_socket(f"{name}-engine.sock"), own=False)

    @property
    def root(self) -> Path:
        return self.home / "flatshare"

    @property
    def config(self) -> Path:
        return self.home / ".config" / "flatshare"

    @property
    def units(self) -> Path:
        return self.home / ".config" / "systemd" / "user"

    def aiperf(self) -> list[str]:
        """AIPerf from this environment (the bench venv) on the load generators' cores."""
        aiperf = Path(sys.executable).with_name("aiperf")
        config = CONFIGS / "agentic_replay.yaml"
        return ["taskset", "-c", self.aiperf_cores, str(aiperf), "profile", "--config", str(config)]

    def bots(self) -> list[str]:
        """``flatshare-voice bots`` from the voice venv, inside the voice-plane namespace, on the callers'
        cores."""
        voicenet = self.root / "src" / "deploy" / "bin" / "in-voicenet"
        flatshare_voice = self.root / "venvs" / "voice" / "bin" / "flatshare-voice"
        return ["taskset", "-c", self.caller_cores, str(voicenet), str(flatshare_voice), "bots"]


@dataclass(frozen=True)
class Node:
    """The GPU node a point measures on, as the control host reaches it (:meth:`Host.node_for`)."""

    name: str
    engine_url: str
    """The engine: the address the host's settings name on its own node, or the node's tunnel socket."""
    own: bool
    """Whether it is the host's own node, which cyan's units already reach: nothing is pointed anywhere else."""

    def socket(self, server: str) -> str:
        """The path of the node's tunnel socket for ``server`` (engine, stt or tts) on this host."""
        return runtime_socket(f"{self.name}-{server}.sock").removeprefix(UNIX)

    @property
    def tunnel(self) -> str:
        """The tunnel unit that serves the node's sockets here."""
        return f"flatshare-tunnel@{self.name}.service"


class Systemd:
    """cyan's user units, through ``systemctl --user`` and ``journalctl --user``; the tests use a
    fake."""

    def __call__(self, *args: str) -> str:
        """``systemctl --user ARGS``'s output; a failure is an error with systemctl's message."""
        done = subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)
        if done.returncode:
            reason = done.stderr.strip() or f"exit status {done.returncode}"
            raise E2Error(f"systemctl --user {' '.join(args)}: {reason}")
        return done.stdout

    def active(self, unit: str) -> bool:
        return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0

    def environment(self, unit: str) -> dict[str, str]:
        """The environment systemd gives ``unit`` (its ``Environment=`` settings)."""
        env = {}
        for assignment in shlex.split(self("show", unit, "--property=Environment", "--value")):
            name, _, value = assignment.partition("=")
            env[name] = value
        return env

    def journal(self, unit: str, since: float, until: float | None = None) -> str:
        command = ["journalctl", "--user", "--unit", unit, "--since", f"@{int(since)}", "-o", "cat"]
        if until is not None:
            command += ["--until", f"@{math.ceil(until)}"]
        return subprocess.run(command, capture_output=True, text=True).stdout


def configure_cyan(
    arm: Arm, host: Host, systemd: Systemd, node: Node | None = None
) -> dict[str, Any]:
    """Make cyan's units serve ``arm`` on GPU node ``node`` (the host's own when None), restarting only
    those whose configuration changes; returns what they run, for run.json. On another node than the
    host's own, the gateway's upstream is that node's engine socket and the worker's speech sockets its
    tunnel's (:func:`node_drop_in`), which run.json's ``node`` says."""
    node = node or host.node_for()
    env = systemd.environment(WORKER)
    if env.get("FLATSHARE_TURN_DETECTION") != "vad":
        raise E2Error(
            f"{WORKER} ends turns with LiveKit's end-of-turn model; E2 points need fixed endpointing: "
            "link deploy/cyan/variants/vad-turns (ops/RUNBOOK.md)"
        )
    if env.get("FLATSHARE_VOICE_MODE", "bench") != "bench":
        raise E2Error(
            f"{WORKER} runs serve mode, whose admission refuses calls past the serving profile's; E2 points "
            "measure bench mode: unlink deploy/cyan/variants/serve (ops/RUNBOOK.md)"
        )
    overlay = _merge(E2_GATEWAY_TABLE, arm.gateway)
    if not node.own:
        overlay = _on_node(arm, host, node, overlay)
    applied = _apply(host, systemd, arm.router, overlay, arm.voice, node, env)
    if applied["worker"]["restarted"]:
        env = systemd.environment(WORKER)  # the LLM's path as the node's drop-in leaves it
    return applied | {"llm_path": env.get("FLATSHARE_LLM_UDS")}


def reach_node(node: Node, systemd: Systemd) -> dict[str, Any] | None:
    """Make sure a GPU node other than the host's own is reachable from here before a point reads its engine:
    its tunnel unit running (started when it is not; Type=notify, so it returns once the node's sockets
    listen here). None for the host's own node, which is left as it is. Returns what run.json keeps."""
    if node.own:
        return None
    started = not systemd.active(node.tunnel)
    if started:
        try:
            systemd("start", node.tunnel)
        except E2Error as exc:
            raise E2Error(
                f"{exc}\n(the tunnel to {node.name} needs ~/.config/flatshare/tunnel-{node.name}.env, its SSH "
                f"target, and its keys authorized on the node: deploy/authorize-tunnel.sh, ops/RUNBOOK.md)"
            ) from None
    sockets = {server: node.socket(server) for server in ("engine", "stt", "tts")}
    return {"name": node.name, "tunnel": node.tunnel, "tunnel_started": started, "sockets": sockets}


def reset_cyan(host: Host, systemd: Systemd) -> dict[str, Any]:
    """Undo what e2 run changes on cyan: SMG stopped, the gateway back on gateway.toml, preemptive
    generation and lifecycle events on, and the worker back on the host's own GPU node. A tunnel to
    another node that a point started stays up: nothing reaches the node through it once the gateway
    and the worker are back."""
    with exclusive(host):
        return _apply(host, systemd, None, {}, Voice(), host.node_for(), {})


def check_node(arm: Arm, host: Host, node: Node) -> None:
    """Refuse, before anything starts, a point that cannot run on ``node`` as the host reaches it: on another node
    than the host's own, an arm with a router or an upstream of its own, which reach the host's own engine, and a
    gateway.toml with [[nodes]], whose placement would route the point's requests itself."""
    if node.own:
        return
    if arm.router or "upstream" in arm.gateway:
        raise E2Error(
            f"{arm.name} sets its own router or [gateway] upstream, which reach {host.node}'s engine: it "
            f"runs on {host.node}, not {node.name}"
        )
    if "nodes" in _read_toml(host.config / "gateway.toml"):
        raise E2Error(
            f"{host.config / 'gateway.toml'} configures [[nodes]], whose placement would route the point's "
            f"requests itself: a point on {node.name} needs a gateway.toml with one upstream"
        )


def _on_node(arm: Arm, host: Host, node: Node, overlay: Mapping[str, Any]) -> dict[str, Any]:
    """The point's gateway configuration on ``node``, another than the host's own: ``overlay`` with the
    gateway's upstream at the node's engine socket (:func:`check_node` says which points may)."""
    check_node(arm, host, node)
    return _merge(overlay, {"upstream": f"{UNIX}{node.socket('engine')}"})


def node_drop_in(node: Node, env: Mapping[str, str]) -> str | None:
    """The voice worker's drop-in for a point on ``node``: the node's STT and TTS sockets through its tunnel
    (its NAME-stt.sock and NAME-tts.sock, where the worker's unit names Phastform's), and its engine's where
    the worker's LLM bypasses the gateway (``env``'s FLATSHARE_LLM_UDS, the unit's, names an engine's tunnel
    socket), with the tunnel unit wanted and ordered before it. None on the host's own node: the unit's own
    sockets are that node's."""
    if node.own:
        return None
    bypass = env.get("FLATSHARE_LLM_UDS", "").endswith("-engine.sock")
    llm = (
        f"its LLM, which bypasses the gateway, through {node.name}'s engine socket"
        if bypass
        else "its LLM through the gateway, whose upstream the point points there too"
    )
    lines = [
        f"# Generated by flatshare-bench e2 run for a point on GPU node {node.name}: the worker's STT and TTS through",
        f"# {node.name}'s tunnel, and {llm}. flatshare-bench e2 reset removes it.",
        "[Unit]",
        f"Wants={node.tunnel}",
        f"After={node.tunnel}",
        "[Service]",
        f"Environment=FLATSHARE_STT_UDS=%t/flatshare/{node.name}-stt.sock",
        f"Environment=FLATSHARE_TTS_UDS=%t/flatshare/{node.name}-tts.sock",
    ]
    if bypass:
        lines.append(f"Environment=FLATSHARE_LLM_UDS=%t/flatshare/{node.name}-engine.sock")
    return "\n".join(lines) + "\n"


def _apply(
    host: Host,
    systemd: Systemd,
    router: str | None,
    overlay: Mapping[str, Any],
    voice: Voice,
    node: Node,
    env: Mapping[str, str],
) -> dict[str, Any]:
    if router == "smg":
        systemd("start", SMG)  # returns once SMG answers /health
    elif systemd.active(SMG):
        systemd("stop", SMG)
    config = _merge(_read_toml(host.config / "gateway.toml"), overlay)
    gateway = _link(host, GATEWAY, E2_GATEWAY, bool(overlay))
    if overlay:
        gateway |= _write_private(
            host.config / "gateway-e2.toml", GATEWAY_HEADER + toml_text(config)
        )
    links = [
        _link(host, WORKER, NO_PREEMPTIVE, not voice.preemptive_generation),
        _link(host, WORKER, NO_EVENTS, not voice.events),
        _write_drop_in(host, WORKER, E2_NODE, node_drop_in(node, env)),
    ]
    worker = any(links)
    if gateway or worker:
        systemd("daemon-reload")
    if gateway:
        since = time.time()
        try:
            systemd("restart", GATEWAY)  # returns once the gateway answers /health
        except E2Error as exc:
            # A configuration the gateway refuses (a key its build does not know) is in its log.
            logged = "\n".join(systemd.journal(GATEWAY, since).strip().splitlines()[-5:])
            raise E2Error(
                f"{exc}\n{GATEWAY} logged:\n{logged}\n(flatshare-bench e2 reset puts it back on gateway.toml)"
            ) from None
    if worker:
        since = time.time()
        systemd("restart", WORKER)
        until(
            lambda: "registered worker" in systemd.journal(WORKER, since),
            90.0,
            f"{WORKER} did not register with LiveKit within 90 s (journalctl --user -u {WORKER})",
        )
    return {
        "router": router,
        "gateway": {"restarted": gateway, "config": redacted(config)},
        "worker": {
            "restarted": worker,
            "preemptive_generation": voice.preemptive_generation,
            "events": voice.events,
        },
    }


GATEWAY_HEADER = (
    "# Generated by flatshare-bench e2 run: gateway.toml with the inference canary off and an arm file's [gateway]\n"
    "# table merged in.\n"
    "# flatshare-gateway reads it through the e2-gateway drop-in; edit gateway.toml or the arm instead.\n"
)


def _link(host: Host, unit: str, variant: str, wanted: bool) -> bool:
    """Link (or remove) one of deploy/cyan/variants/ into ``unit``; returns whether that changed."""
    link = host.units / f"{unit}.service.d" / f"variant-{variant}.conf"
    target = host.root / "src" / "deploy" / "cyan" / "variants" / variant / f"{unit}.conf"
    if wanted and link.is_symlink() and link.readlink() == target:
        return False
    if not wanted and not (link.is_symlink() or link.exists()):
        return False
    link.unlink(missing_ok=True)
    if wanted:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    return True


def _write_drop_in(host: Host, unit: str, name: str, text: str | None) -> bool:
    """Write ``unit``'s generated drop-in ``variant-NAME.conf`` (or remove it, ``text`` None); returns whether that
    changed it."""
    path = host.units / f"{unit}.service.d" / f"variant-{name}.conf"
    if text is None:
        if not (path.exists() or path.is_symlink()):
            return False
        path.unlink()
        return True
    if path.is_file() and not path.is_symlink() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.write_text(text)
    return True


def _write_private(path: Path, text: str) -> bool:
    """Write ``text`` to a file only its owner can read (it holds key digests), from its creation
    on: a new file of mode 600 replaces it. Returns whether it changed."""
    if path.exists() and path.read_text() == text:
        return False
    staged = path.with_name(path.name + ".new")
    # One an interrupted write left goes: whoever opened it would read what is written over it.
    staged.unlink(missing_ok=True)
    with open(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as file:
        file.write(text)
    staged.replace(path)
    return True


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise E2Error(f"{path}: {exc}") from None


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``overlay`` merged in: tables merge key by key, any other value replaces."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


SECRETS = frozenset({"keys_sha256", "api_key", "admin_api_key", "hf_token"})
"""Settings that hold a credential or its digest: the gateway's key digests, the engines' API keys and
Hugging Face token. A run directory keeps only whether they are set."""


def redacted(config: Any) -> Any:
    """A configuration (the gateway's, an engine's) without its credentials."""
    if isinstance(config, Mapping):
        return {
            k: (["<redacted>"] * len(v) if isinstance(v, list) else "<redacted>")
            if k in SECRETS and v
            else redacted(v)
            for k, v in config.items()
        }
    if isinstance(config, list):
        return [redacted(v) for v in config]
    return config


_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def toml_text(table: Mapping[str, Any]) -> str:
    """TOML for a parsed TOML document of strings, numbers, booleans, arrays and tables (all the
    gateway's configuration holds)."""
    return "\n".join(_toml_lines(table, ())) + "\n"


def _toml_lines(table: Mapping[str, Any], path: tuple[str, ...]) -> list[str]:
    def is_tables(value: Any) -> bool:
        return (
            isinstance(value, list) and bool(value) and all(isinstance(v, Mapping) for v in value)
        )

    lines = [
        f"{_toml_key(k)} = {_toml_value(v)}"
        for k, v in table.items()
        if not isinstance(v, Mapping) and not is_tables(v)
    ]
    for key, value in table.items():
        header = ".".join(_toml_key(k) for k in (*path, key))
        if isinstance(value, Mapping):
            lines += ["", f"[{header}]", *_toml_lines(value, (*path, key))]
        elif is_tables(value):
            for item in value:
                lines += ["", f"[[{header}]]", *_toml_lines(item, (*path, key))]
    return lines


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else json.dumps(key, ensure_ascii=False)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, str):
        # A JSON string is a TOML basic string once DEL, which JSON leaves alone, is escaped.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items()) + "}"
    raise E2Error(f"cannot write {value!r} as TOML")


def read_env(path: Path) -> dict[str, str]:
    """The ``NAME=value`` lines of a shell environment file (``export`` and quotes allowed)."""
    env = {}
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if not sep or name.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[name.strip()] = value
    return env


@contextlib.contextmanager
def exclusive(host: Host) -> Iterator[None]:
    """One e2 run, reset or drill at a time: they share cyan's units and cores. The lock file names the
    holder's pid, to stop it by."""
    path = host.root / "runs" / ".flatshare-bench-e2.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Opened to append: "w" would empty the holder's pid before the lock is even tried.
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise E2Error(f"another flatshare-bench e2 run or reset holds {path}") from None
        lock.truncate(0)
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        yield


def agents_key(host: Host) -> str:
    """The agents tenant's key, from cyan's gateway-keys.env: AIPerf's replay and the drills'
    background streams go through the gateway as that tenant."""
    keys = host.config / "gateway-keys.env"
    key = read_env(keys).get("FLATSHARE_GATEWAY_AGENTS_KEY") if keys.exists() else None
    if not key:
        raise E2Error(f"{keys} has no FLATSHARE_GATEWAY_AGENTS_KEY, the agents tenant's key")
    return key
